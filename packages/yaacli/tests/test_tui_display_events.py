from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock

from pydantic_ai.usage import RunUsage
from ya_agent_sdk.context import StreamEvent
from ya_agent_sdk.events import SubagentStartEvent
from yaacli.app import TUIApp
from yaacli.app.tui import PendingAttachment
from yaacli.config import CommandDefinition
from yaacli.events import GoalCompleteEvent, GoalCompleteReason
from yaacli.session import TUIContext


@dataclass
class MockConfig:
    general: object = field(default_factory=lambda: MagicMock(max_requests=10, mode="act"))
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
    assert "shell" in app._output_lines[0]
    assert app._subagent_states["subagent-1"]["tool_names"] == ["shell"]


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
    assert any(
        "Total tokens used this goal: 1,234 tokens "
        "(input: 1,000, cache read: 800, cache write: 20, output: 234)" in line
        for line in app._output_lines
    )


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
