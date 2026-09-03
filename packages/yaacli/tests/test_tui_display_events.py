from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from pydantic_ai import EnqueuedMessagesEvent
from pydantic_ai.usage import RunUsage
from ya_agent_sdk.context import StreamEvent
from ya_agent_sdk.events import ModelRequestCompleteEvent, SubagentCompleteEvent, SubagentStartEvent, UsageSnapshotEvent
from ya_agent_sdk.usage import UsageAgentTotal, UsageSnapshot
from ya_agent_stream_protocol.sdk import AguiEventAdapter
from yaacli.app import TUIApp
from yaacli.app.tui import YAACLI_AGUI_ADAPTER_CONFIG, PendingAttachment
from yaacli.config import CommandDefinition
from yaacli.durable.executor import _merge_durable_display_projection, _safe_recovery_display_projection
from yaacli.durable.models import InputPriority, InputRecord, InputState, RevisionRecord
from yaacli.events import GoalCompleteEvent, GoalCompleteReason
from yaacli.session import TUIContext


@dataclass
class MockConfig:
    general: object = field(default_factory=lambda: MagicMock(max_requests=10))
    display: object = field(default_factory=lambda: MagicMock(max_lines=500, mouse=True))
    commands: dict[str, CommandDefinition] = field(default_factory=dict)

    def get_commands(self) -> dict[str, CommandDefinition]:
        return self.commands


@dataclass
class MockConfigManager:
    def get_sessions_dir(self) -> object:
        return MagicMock(exists=lambda: False)


def test_tui_display_tool_call_chunks_render_calling_once() -> None:
    config = MockConfig()
    config_manager = MockConfigManager()
    app = TUIApp(config=config, config_manager=config_manager)  # type: ignore[arg-type]
    app._append_block = MagicMock(wraps=app._append_block)

    app._handle_display_events([
        {"type": "TOOL_CALL_CHUNK", "toolCallId": "tool-1", "toolCallName": "shell", "delta": '{"command":'},
        {"type": "TOOL_CALL_CHUNK", "toolCallId": "tool-1", "toolCallName": "shell", "delta": '"pytest"}'},
    ])

    calling_blocks = [line for line in app._output_lines if "Calling:" in line and "shell" in line]
    assert len(calling_blocks) == 1
    assert app._append_block.call_count == 1


def test_tui_display_tool_call_arguments_are_bounded_without_mutating_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())  # type: ignore[arg-type]
    monkeypatch.setattr("yaacli.app.tui._MAX_RETAINED_TOOL_ARG_CHARS", 64)
    event = {
        "type": "TOOL_CALL_CHUNK",
        "toolCallId": "tool-large",
        "toolCallName": "shell",
        "delta": "x" * 1_000,
    }
    original = dict(event)

    app._handle_display_events([event])

    tool_args = app._tool_messages["tool-large"].args
    tracker_args = app._event_renderer.tracker.tool_calls["tool-large"].args
    assert isinstance(tool_args, str)
    assert len(tool_args) <= 64
    assert "tool arguments truncated for display" in tool_args
    assert tracker_args == tool_args
    assert event == original


def test_tui_display_tool_result_renders_once() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())  # type: ignore[arg-type]
    app._append_block = MagicMock(wraps=app._append_block)

    app._handle_display_events([
        {"type": "TOOL_CALL_RESULT", "toolCallId": "tool-1", "toolCallName": "shell", "content": "done"},
        {"type": "TOOL_CALL_RESULT", "toolCallId": "tool-1", "toolCallName": "shell", "content": "done"},
    ])

    assert app._append_block.call_count == 1


def test_tui_display_tool_result_uses_agui_timestamps_for_duration() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())  # type: ignore[arg-type]

    app._handle_display_events([
        {
            "type": "TOOL_CALL_CHUNK",
            "toolCallId": "tool-1",
            "toolCallName": "shell",
            "delta": '{"command":"sleep"}',
            "timestamp": 1_000,
        },
        {
            "type": "TOOL_CALL_RESULT",
            "toolCallId": "tool-1",
            "content": "done",
            "timestamp": 2_500,
        },
    ])

    assert any("(1.5s)" in line for line in app._output_lines)
    assert abs(app._event_renderer.tracker.tool_calls["tool-1"].duration() - 1.5) < 0.01


def test_terminal_recovery_projection_keeps_observed_incomplete_tool_calls() -> None:
    projected = _safe_recovery_display_projection([
        {"type": "RUN_STARTED", "runId": "run-1"},
        {"type": "TEXT_MESSAGE_CHUNK", "messageId": "text-1", "delta": "partial text"},
        {
            "type": "TOOL_CALL_CHUNK",
            "toolCallId": "complete-tool",
            "toolCallName": "shell",
            "delta": '{"command":"true"}',
        },
        {"type": "TOOL_CALL_RESULT", "toolCallId": "complete-tool", "content": "done"},
        {
            "type": "TOOL_CALL_CHUNK",
            "toolCallId": "incomplete-tool",
            "toolCallName": "shell",
            "delta": '{"command":',
        },
        {
            "type": "TOOL_CALL_START",
            "toolCallId": "start-only-tool",
            "toolCallName": "shell",
        },
        {"type": "RUN_STARTED", "runId": "run-2"},
        {
            "type": "TOOL_CALL_CHUNK",
            "toolCallId": "complete-tool",
            "toolCallName": "shell",
            "delta": '{"command":"must not match the prior run"}',
        },
    ])

    assert any(event.get("type") == "TEXT_MESSAGE_CHUNK" for event in projected)
    complete_tool_events = [event for event in projected if event.get("toolCallId") == "complete-tool"]
    assert [event["type"] for event in complete_tool_events] == [
        "TOOL_CALL_CHUNK",
        "TOOL_CALL_RESULT",
        "TOOL_CALL_CHUNK",
    ]
    assert any(event.get("toolCallId") == "incomplete-tool" for event in projected)
    assert any(event.get("toolCallId") == "start-only-tool" for event in projected)


def test_non_success_reconciliation_rebuilds_canonical_tools_after_live_blocks_are_lost() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())  # type: ignore[arg-type]
    app._active_display_run_id = "run-1"
    canonical_events = [
        {"type": "RUN_STARTED", "runId": "run-1"},
        {"type": "TEXT_MESSAGE_START", "messageId": "text-1"},
        {"type": "TEXT_MESSAGE_CHUNK", "messageId": "text-1", "delta": "unstable partial text"},
        {
            "type": "TOOL_CALL_CHUNK",
            "toolCallId": "tool-1",
            "toolCallName": "shell",
            "delta": '{"command":"printf kept"}',
        },
        {
            "type": "TOOL_CALL_RESULT",
            "toolCallId": "tool-1",
            "toolCallName": "shell",
            "content": "kept tool result",
        },
    ]
    app._handle_and_record_display_events(canonical_events)
    app._append_system_output("background worker is ready")

    # Transcript retention or an explicit UI reset may discard live blocks
    # while the tool render caches still claim that those calls were printed.
    app._transcript.remove(set(app._active_run_block_ids))
    app._sync_transcript_state()

    app._reconcile_terminal_transcript(
        RevisionRecord(
            revision_id="revision-1",
            session_id="session-1",
            logical_run_id="logical-run-1",
            commit_kind="failed",
            display_projection=canonical_events,
            terminal={"status": "failed"},
            created_at=datetime.now(UTC),
        )
    )

    output = "\n".join(app._output_lines)
    assert "unstable partial text" in output
    assert "Calling:" in output
    assert "Complete:" in output
    assert "kept tool result" in output
    assert "background worker is ready" in output
    assert app._tool_messages["tool-1"].content == "kept tool result"


def test_non_success_reconciliation_removes_live_foreground_subagent_progress() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())  # type: ignore[arg-type]
    app._active_display_run_id = "run-1"
    app._handle_stream_event(
        StreamEvent(
            agent_id="main",
            agent_name="main",
            event=SubagentStartEvent(
                event_id="subagent-start-1",
                execution_id="worker-1",
                agent_id="worker-1",
                agent_name="worker",
                mode="foreground",
            ),
        )
    )
    app._append_system_output("background worker is ready")
    assert any("Running..." in line for line in app._output_lines)

    app._reconcile_terminal_transcript(
        RevisionRecord(
            revision_id="revision-1",
            session_id="session-1",
            logical_run_id="logical-run-1",
            commit_kind="cancelled",
            display_projection=[{"type": "RUN_STARTED", "runId": "run-1"}],
            terminal={"status": "cancelled"},
            created_at=datetime.now(UTC),
        )
    )

    output = "\n".join(app._output_lines)
    assert "Running..." not in output
    assert "background worker is ready" in output
    assert app._subagent_states == {}


def test_tui_terminal_replay_reconstructs_tools_after_live_render() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())  # type: ignore[arg-type]
    events = [
        {"type": "RUN_STARTED", "runId": "run-1"},
        {
            "type": "TOOL_CALL_CHUNK",
            "toolCallId": "tool-1",
            "toolCallName": "shell",
            "delta": '{"command":"printf done"}',
            "timestamp": 1_000,
        },
        {
            "type": "TOOL_CALL_RESULT",
            "toolCallId": "tool-1",
            "content": "done",
            "timestamp": 2_500,
        },
        {"type": "RUN_STARTED", "runId": "run-2"},
        {
            "type": "TOOL_CALL_CHUNK",
            "toolCallId": "tool-1",
            "toolCallName": "shell",
            "delta": '{"command":"printf second"}',
            "timestamp": 3_000,
        },
        {
            "type": "TOOL_CALL_RESULT",
            "toolCallId": "tool-1",
            "content": "second",
            "timestamp": 4_000,
        },
    ]
    app._handle_and_record_display_events(events)
    live_output = list(app._output_lines)
    replay = app._display_replay.snapshot()

    app._restore_output_from_display_events(replay)

    assert app._output_lines == live_output
    assert sum("Calling:" in line and "shell" in line for line in app._output_lines) == 2
    assert sum("Complete:" in line and "shell" in line for line in app._output_lines) == 2
    assert app._tool_messages["tool-1"].content == "second"
    assert "tool-1" in app._printed_tool_calls


def test_tui_display_empty_reasoning_start_does_not_render_blank_block() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())  # type: ignore[arg-type]

    app._handle_display_events([{"type": "REASONING_MESSAGE_START", "messageId": "thinking-1"}])

    assert app._streaming_thinking == ""
    assert app._streaming_thinking_line_index == 0
    assert app._output_lines == []
    assert app._block_line_counts == []
    assert app._total_line_count == 0

    app._handle_display_events([{"type": "REASONING_MESSAGE_CHUNK", "messageId": "thinking-1", "delta": "reasoning"}])

    assert len(app._output_lines) == 1
    assert app._block_line_counts == [1]
    assert app._total_line_count == 1
    assert "reasoning" in app._output_lines[0]


def test_tui_display_skips_subagent_detail_events() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())  # type: ignore[arg-type]
    app._append_block = MagicMock(wraps=app._append_block)

    app._handle_display_events([
        {"type": "TEXT_MESSAGE_START", "messageId": "text-1", "yaacliAgentId": "subagent-1"},
        {"type": "TEXT_MESSAGE_CHUNK", "messageId": "text-1", "delta": "hidden", "yaacliAgentId": "subagent-1"},
        {"type": "TEXT_MESSAGE_END", "messageId": "text-1", "yaacliAgentId": "subagent-1"},
        {"type": "REASONING_MESSAGE_START", "messageId": "thinking-1", "yaacliAgentId": "subagent-1"},
        {
            "type": "REASONING_MESSAGE_CHUNK",
            "messageId": "thinking-1",
            "delta": "hidden thought",
            "yaacliAgentId": "subagent-1",
        },
        {"type": "REASONING_MESSAGE_END", "messageId": "thinking-1", "yaacliAgentId": "subagent-1"},
        {
            "type": "TOOL_CALL_CHUNK",
            "toolCallId": "tool-1",
            "toolCallName": "shell",
            "delta": "{}",
            "yaacliAgentId": "subagent-1",
        },
        {
            "type": "TOOL_CALL_RESULT",
            "toolCallId": "tool-1",
            "content": "done",
            "yaacliAgentId": "subagent-1",
        },
    ])

    assert app._append_block.call_count == 0
    assert app._output_lines == []


def test_tui_display_subagent_tool_chunk_updates_progress_line() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())  # type: ignore[arg-type]
    app._handle_subagent_start(SubagentStartEvent(event_id="subagent-1", agent_id="subagent-1", agent_name="worker"))

    app._handle_display_events([
        {
            "type": "TOOL_CALL_CHUNK",
            "toolCallId": "tool-1",
            "toolCallName": "shell",
            "delta": "{}",
            "yaacliAgentId": "subagent-1",
        }
    ])

    assert len(app._output_lines) == 1
    assert "worker-subagent-1" in app._output_lines[0]
    assert "shell" in app._output_lines[0]
    assert app._subagent_states["subagent-1"]["tool_names"] == ["shell"]


def test_tui_usage_snapshots_are_scoped_to_durable_logical_runs() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())  # type: ignore[arg-type]

    for run_id, tokens in (("logical-1", 100), ("logical-2", 250)):
        app._active_logical_run_id = run_id
        usage = RunUsage(input_tokens=tokens)
        snapshot = UsageSnapshot(
            run_id="provider-session",
            total_usage=usage,
            agent_usages={
                "main": UsageAgentTotal(agent_name="main", model_id="model-a", usage=usage),
            },
            model_usages={"model-a": usage},
        )
        app._handle_stream_event(
            StreamEvent(
                agent_id="main",
                agent_name="main",
                event=UsageSnapshotEvent(event_id=f"usage-{run_id}", snapshot=snapshot, source="test"),
            )
        )
        app._session_usage.commit_run_snapshot(run_id)
        app._session_usage.finalize_run_snapshots(run_id)

    assert app._session_usage.total_tokens == 350
    assert app._session_usage._run_snapshots == {}


def test_tui_suppresses_background_subagent_inline_progress_by_explicit_mode() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())  # type: ignore[arg-type]
    app._display_adapter = AguiEventAdapter(
        session_id="session",
        run_id="run",
        config=YAACLI_AGUI_ADAPTER_CONFIG,
    )
    start = SubagentStartEvent(
        event_id="worker-bg-a7b9",
        execution_id="worker-bg-a7b9",
        mode="background",
        agent_id="worker-bg-a7b9",
        agent_name="worker",
    )
    app._handle_execution_stream_event(StreamEvent(agent_id="worker-bg-a7b9", agent_name="worker", event=start))

    assert app._output_lines == []
    assert app._display_replay.snapshot() == []
    assert app._background_subagent_ids == {"worker-bg-a7b9"}

    complete = SubagentCompleteEvent(
        event_id="worker-bg-a7b9",
        execution_id="worker-bg-a7b9",
        mode="background",
        agent_id="worker-bg-a7b9",
        agent_name="worker",
    )
    app._handle_execution_stream_event(StreamEvent(agent_id="worker-bg-a7b9", agent_name="worker", event=complete))

    assert app._output_lines == []
    assert app._display_replay.snapshot() == []
    assert app._background_subagent_ids == set()


def test_tui_projects_native_applied_steering_before_product_state_reconciliation() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())  # type: ignore[arg-type]
    app._active_logical_run_id = "run-1"
    app._display_adapter = AguiEventAdapter(
        session_id="session",
        run_id="run-1",
        config=YAACLI_AGUI_ADAPTER_CONFIG,
    )
    now = datetime.now(UTC)
    store = MagicMock()
    store.list_inputs.return_value = (
        InputRecord(
            input_id="input-2",
            logical_run_id="run-1",
            order_index=1,
            idempotency_key="steer-1",
            origin="user",
            priority=InputPriority.asap,
            content=["follow up\n" + "x" * 200],
            state=InputState.enqueued,
            native_enqueue_id="enqueue-1",
            created_at=now,
            updated_at=now,
        ),
    )
    app._durable_store = store
    app._pending_steering_count = 1
    render = MagicMock(return_value="steering rendered")
    app._event_renderer.render_steering_injected = render  # type: ignore[method-assign]

    app._handle_execution_stream_event(
        StreamEvent(
            agent_id="main",
            agent_name="main",
            event=EnqueuedMessagesEvent(enqueue_id="enqueue-1", messages=()),
        )
    )

    render.assert_called_once()
    preview = render.call_args.args[0][0]
    assert "\n" not in preview
    assert len(preview) == 100
    assert preview.endswith("...")
    custom_events = [
        event for event in app._display_replay.snapshot() if event.get("name") == "yaacli.steering_applied"
    ]
    assert len(custom_events) == 1
    assert custom_events[0]["type"] == "CUSTOM"
    value = custom_events[0]["value"]
    assert value["messages"] == [preview]
    assert value["projection_key"].startswith("steering-")
    serialized_replay = json.dumps(app._display_replay.snapshot())
    assert "input-2" not in serialized_replay
    assert "enqueue-1" not in serialized_replay
    store.list_inputs.assert_called_once_with(
        "run-1",
        states=(InputState.enqueued, InputState.applied),
    )
    assert app._pending_steering_count == 0


def test_tui_replays_applied_steering_custom_event() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())  # type: ignore[arg-type]
    render = MagicMock(return_value="steering rendered")
    app._event_renderer.render_steering_injected = render  # type: ignore[method-assign]
    event = {
        "type": "CUSTOM",
        "name": "yaacli.steering_applied",
        "value": {"projection_key": "steering-projection", "messages": ["follow up"]},
    }

    app._restore_output_from_display_events([event])

    render.assert_called_once_with(["follow up"])
    assert app._output_lines == ["steering rendered"]
    assert app._projected_steering_keys == {"steering-projection"}


def test_tui_merges_adjacent_applied_steering_events_into_one_block() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())  # type: ignore[arg-type]
    app._event_renderer.render_steering_injected = MagicMock(
        side_effect=lambda previews: f"steering: {', '.join(previews)}"
    )  # type: ignore[method-assign]
    events = [
        {
            "type": "CUSTOM",
            "name": "yaacli.steering_applied",
            "value": {"projection_key": "steering-one", "messages": ["first"]},
        },
        {
            "type": "CUSTOM",
            "name": "yaacli.steering_applied",
            "value": {"projection_key": "steering-two", "messages": ["second"]},
        },
    ]

    app._restore_output_from_display_events(events)

    assert app._output_lines == ["steering: first, second"]
    assert [call.args[0] for call in app._event_renderer.render_steering_injected.call_args_list] == [
        ["first"],
        ["first", "second"],
    ]


def test_durable_applied_steering_is_restored_beside_acceptance_not_at_bottom() -> None:
    accepted = {
        "type": "CUSTOM",
        "name": "yaacli.steering_accepted",
        "value": {"projection_key": "steering-one"},
    }
    applied = {
        "type": "CUSTOM",
        "name": "yaacli.steering_applied",
        "value": {"projection_key": "steering-one", "messages": ["focus"]},
    }
    trailing_output = {"type": "TEXT_MESSAGE_CHUNK", "messageId": "message", "delta": "done"}

    merged = _merge_durable_display_projection([accepted, trailing_output], [accepted, applied])

    assert merged == [accepted, applied, trailing_output]


def test_tui_steering_projection_deduplicates_after_replay_restore() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())  # type: ignore[arg-type]
    app._active_logical_run_id = "run-1"
    now = datetime.now(UTC)
    store = MagicMock()
    store.list_inputs.return_value = (
        InputRecord(
            input_id="input-2",
            logical_run_id="run-1",
            order_index=1,
            idempotency_key="steer-1",
            origin="user",
            priority=InputPriority.asap,
            content=["follow up"],
            state=InputState.applied,
            native_enqueue_id="enqueue-1",
            created_at=now,
            updated_at=now,
        ),
    )
    app._durable_store = store
    event = StreamEvent(
        agent_id="main",
        agent_name="main",
        event=EnqueuedMessagesEvent(enqueue_id="enqueue-1", messages=()),
    )

    app._handle_execution_stream_event(event)
    replay = app._display_replay.snapshot()
    app._restore_output_from_display_events(replay)
    app._handle_execution_stream_event(event)

    custom_events = [
        event for event in app._display_replay.snapshot() if event.get("name") == "yaacli.steering_applied"
    ]
    assert len(custom_events) == 1
    assert sum("Guidance injected" in line for line in app._output_lines) == 1


def test_tui_applied_steering_replay_bounds_untrusted_messages() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())  # type: ignore[arg-type]
    render = MagicMock(return_value="steering rendered")
    app._event_renderer.render_steering_injected = render  # type: ignore[method-assign]

    app._handle_display_events([
        {
            "type": "CUSTOM",
            "name": "yaacli.steering_applied",
            "value": {
                "projection_key": "steering-untrusted",
                "messages": ["x" * 1_000, 42, "bad\ud800value", *[f"extra-{index}" for index in range(20)]],
            },
        }
    ])

    previews = render.call_args.args[0]
    assert len(previews) == 7
    assert all(len(preview) <= 100 for preview in previews)
    assert all("\ud800" not in preview for preview in previews)


def test_tui_does_not_project_applied_feature_input_as_steering() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())  # type: ignore[arg-type]
    app._active_logical_run_id = "run-1"
    now = datetime.now(UTC)
    store = MagicMock()
    store.list_inputs.return_value = (
        InputRecord(
            input_id="input-2",
            logical_run_id="run-1",
            order_index=1,
            idempotency_key="feature-1",
            origin="feature",
            priority=InputPriority.asap,
            content=["internal completion"],
            state=InputState.applied,
            native_enqueue_id="enqueue-1",
            created_at=now,
            updated_at=now,
        ),
    )
    app._durable_store = store

    app._handle_execution_stream_event(
        StreamEvent(
            agent_id="main",
            agent_name="main",
            event=EnqueuedMessagesEvent(enqueue_id="enqueue-1", messages=()),
        )
    )

    assert app._display_replay.snapshot() == []
    assert app._output_lines == []


def test_tui_deduplicates_repeated_applied_steering_event() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())  # type: ignore[arg-type]
    app._active_logical_run_id = "run-1"
    now = datetime.now(UTC)
    store = MagicMock()
    store.list_inputs.return_value = (
        InputRecord(
            input_id="input-2",
            logical_run_id="run-1",
            order_index=1,
            idempotency_key="steer-1",
            origin="user",
            priority=InputPriority.asap,
            content=["follow up"],
            state=InputState.applied,
            native_enqueue_id="enqueue-1",
            created_at=now,
            updated_at=now,
        ),
    )
    app._durable_store = store
    event = StreamEvent(
        agent_id="main",
        agent_name="main",
        event=EnqueuedMessagesEvent(enqueue_id="enqueue-1", messages=()),
    )

    app._handle_execution_stream_event(event)
    app._handle_execution_stream_event(event)

    custom_events = [
        event for event in app._display_replay.snapshot() if event.get("name") == "yaacli.steering_applied"
    ]
    assert len(custom_events) == 1
    assert sum("Guidance injected" in line for line in app._output_lines) == 1


def test_tui_append_user_input_renders_once_and_records_replay_event() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())  # type: ignore[arg-type]

    app._append_user_input("hello")

    assert sum(1 for line in app._output_lines if "hello" in line) == 1
    replay = app._display_replay.snapshot()
    assert len(replay) == 1
    assert replay[0]["type"] == "CUSTOM"
    assert replay[0]["name"] == "yaacli.user_input"
    assert replay[0]["value"] == {"text": "hello", "attachments": []}


def test_tui_append_user_input_records_attachment_replay_event() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())  # type: ignore[arg-type]
    attachment = PendingAttachment(data=b"img", media_type="image/png", size_bytes=3)

    app._append_user_input("", [attachment])

    replay = app._display_replay.snapshot()
    assert replay[0]["value"] == {"text": "", "attachments": [{"media_type": "image/png", "size_bytes": 3}]}
    assert any("[Attached 1 image]" in line for line in app._output_lines)
    assert any("image/png 3B" in line for line in app._output_lines)


def test_tui_display_user_input_attachment_fallback() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())  # type: ignore[arg-type]

    app._handle_display_events([
        {
            "type": "CUSTOM",
            "name": "yaacli.user_input",
            "value": {"text": "", "attachments": [{"media_type": "image/png", "size_bytes": 1}]},
        }
    ])

    assert any("[Attached 1 image]" in line for line in app._output_lines)


def test_tui_updates_live_context_usage_from_model_request_completion() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())  # type: ignore[arg-type]
    app._display_adapter = AguiEventAdapter(
        session_id="session",
        run_id="run-1",
        config=YAACLI_AGUI_ADAPTER_CONFIG,
    )

    app._handle_execution_stream_event(
        StreamEvent(
            agent_id="main",
            agent_name="main",
            event=ModelRequestCompleteEvent(
                event_id="native-run-id",
                loop_index=1,
                context_tokens=75_000,
                context_window_size=200_000,
            ),
        )
    )

    assert app._current_context_tokens == 75_000
    assert app._context_window_size == 200_000
    replay = app._display_replay.snapshot()
    assert replay[0]["name"] == "ya_agent.model_request_complete"
    assert replay[0]["value"]["payload"]["context_tokens"] == 75_000  # type: ignore[index]
    assert "native-run-id" not in json.dumps(replay)


def test_tui_goal_usage_report_shows_delta_with_commas() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())  # type: ignore[arg-type]
    app._session_usage.add("main", "openai-chat:gpt-4o", RunUsage(input_tokens=10_000, output_tokens=500))
    app._goal_usage_start_breakdown = app._session_usage.token_breakdown
    app._goal_usage_report_pending = True

    app._session_usage.add(
        "main",
        "openai-chat:gpt-4o",
        RunUsage(input_tokens=1_000, output_tokens=234, cache_read_tokens=800, cache_write_tokens=20),
    )
    app._append_goal_usage_report_if_pending()

    assert app._goal_usage_start_breakdown is None
    assert app._goal_usage_report_pending is False
    output_text = " ".join(" ".join(line.split()) for line in app._output_lines)
    assert (
        "Total tokens used this goal: 1,234 tokens "
        "(input: 1,000, cache read: 800, cache write: 20, output: 234)" in output_text
    )


def test_tui_goal_usage_report_wraps_to_terminal_width() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())  # type: ignore[arg-type]
    app._get_terminal_width = MagicMock(return_value=60)  # type: ignore[method-assign]
    app._goal_usage_start_breakdown = app._session_usage.token_breakdown
    app._goal_usage_report_pending = True

    app._session_usage.add(
        "main",
        "oauth@codex:gpt-5.5",
        RunUsage(
            input_tokens=3_650_580,
            output_tokens=13_400,
            cache_read_tokens=3_000_000,
            cache_write_tokens=42_000,
        ),
    )
    app._append_goal_usage_report_if_pending()

    assert len(app._output_lines) == 1
    assert app._output_lines[0].count("\n") >= 1
    output_text = " ".join(app._output_lines[0].split())
    assert "Total tokens used this goal: 3,663,980 tokens" in output_text
    assert "input: 3,650,580" in output_text
    assert "output: 13,400" in output_text


def test_tui_goal_complete_event_renders_unverified_stop() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())  # type: ignore[arg-type]

    app._handle_stream_event(
        StreamEvent(
            agent_id="main",
            agent_name="main",
            event=GoalCompleteEvent(
                event_id="goal-1",
                iteration=2,
                reason=GoalCompleteReason.unverified_stop,
                task="fix tests",
            ),
        )
    )

    assert any("Stopped without verified completion at iteration 2" in line for line in app._output_lines)


def test_tui_finish_active_goal_emits_reason_and_resets_goal() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())  # type: ignore[arg-type]
    ctx = TUIContext.model_construct()
    ctx.goal_task = "fix tests"
    ctx.goal_iteration = 4
    ctx.goal_max_iterations = 10
    ctx.goal_needs_post_restore_audit = True
    ctx.goal_last_context_handoff_source = "compact"
    app._runtime = MagicMock(ctx=ctx)

    app._finish_active_goal(GoalCompleteReason.cancelled)

    assert ctx.goal_task is None
    assert ctx.goal_iteration == 0
    assert ctx.goal_needs_post_restore_audit is False
    assert ctx.goal_last_context_handoff_source is None
    assert any("Cancelled at iteration 4" in line for line in app._output_lines)
