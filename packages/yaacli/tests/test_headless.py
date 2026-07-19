from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest
from pydantic import ValidationError
from pydantic_ai import (
    AgentRunResult,
    DeferredToolRequests,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPartDelta,
    ToolDenied,
)
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from ya_agent_sdk.agents.main import AgentInterrupted
from ya_agent_sdk.context import ResumableState
from ya_agent_sdk.context.agent import StreamEvent
from ya_agent_sdk.events import ModelRequestStartEvent
from yaacli.config import ConfigManager, GeneralConfig, ModelProfileConfig, YaacliConfig
from yaacli.errors import safe_exception_str
from yaacli.headless import HeadlessEventSink, _load_session_artifacts, run_headless_prompt
from yaacli.model_profiles import save_selected_model_profile_id


def _headless_config(*, max_requests: int = 10) -> YaacliConfig:
    return YaacliConfig(
        general=GeneralConfig(
            max_requests=max_requests,
            agent_stream_resume_on_error=False,
            agent_stream_resume_max_attempts=0,
        )
    )


def test_headless_event_sink_flushes_after_every_event() -> None:
    output_stream = MagicMock()
    sink = HeadlessEventSink(output_stream=output_stream)

    sink.emit({"type": "FIRST", "value": 1})
    sink.emit({"type": "SECOND", "value": 2})

    assert output_stream.method_calls == [
        call.write('{"type":"FIRST","value":1}\n'),
        call.flush(),
        call.write('{"type":"SECOND","value":2}\n'),
        call.flush(),
    ]


def test_headless_explicit_restore_rejects_incomplete_schema_v2_session(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    session_dir = sessions_dir / "missing-history"
    turn_dir = session_dir / "turns" / "turn-1"
    turn_dir.mkdir(parents=True)
    (session_dir / "metadata.json").write_text(
        json.dumps({
            "schema_version": 2,
            "session_id": "missing-history",
            "head_turn_id": "turn-1",
            "updated_at": "2026-01-01T00:00:00+00:00",
        })
    )
    config_manager = MagicMock(spec=ConfigManager)
    config_manager.get_sessions_dir.return_value = sessions_dir

    with pytest.raises(FileNotFoundError, match="Session not found: missing-history"):
        _load_session_artifacts(config_manager, "missing-history")


def test_headless_session_restore_requests_bounded_atomic_display_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = SimpleNamespace(
        session_id="session-1",
        message_history_json=b"[]",
        context_state_json=None,
        display_messages_json=None,
    )
    read_head = MagicMock(return_value=artifacts)
    monkeypatch.setattr("yaacli.headless.read_head_artifacts", read_head)
    monkeypatch.setattr("yaacli.headless.MAX_DISPLAY_REPLAY_LOAD_BYTES", 1)
    config_manager = MagicMock()

    session_id, history, state, display_messages = _load_session_artifacts(config_manager, "session")

    read_head.assert_called_once_with(config_manager, "session", max_display_messages_bytes=1)
    assert session_id == "session-1"
    assert history == []
    assert state is None
    assert display_messages == []


def test_headless_session_restore_skips_malformed_display_and_keeps_valid_history(tmp_path: Path) -> None:
    history = [ModelRequest(parts=[UserPromptPart(content="previous")])]
    sessions_dir = tmp_path / "sessions"
    session_dir = sessions_dir / "session-1"
    turn_dir = session_dir / "turns" / "turn-1"
    turn_dir.mkdir(parents=True)
    (session_dir / "metadata.json").write_text(
        json.dumps({
            "schema_version": 2,
            "session_id": session_dir.name,
            "head_turn_id": turn_dir.name,
        })
    )
    (turn_dir / "metadata.json").write_text("{}")
    (turn_dir / "message_history.json").write_bytes(ModelMessagesTypeAdapter.dump_json(history))
    (turn_dir / "display_messages.json").write_bytes(b"not-json")
    config_manager = MagicMock(spec=ConfigManager)
    config_manager.get_sessions_dir.return_value = sessions_dir

    session_id, restored_history, state, display_messages = _load_session_artifacts(config_manager, "session-1")

    assert session_id == "session-1"
    assert restored_history == history
    assert state is None
    assert display_messages == []


def test_headless_session_restore_keeps_malformed_history_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    artifacts = SimpleNamespace(
        session_id="session-1",
        message_history_json=b"not-json",
        context_state_json=None,
        display_messages_json=None,
    )
    monkeypatch.setattr("yaacli.headless.read_head_artifacts", MagicMock(return_value=artifacts))

    with pytest.raises(ValidationError):
        _load_session_artifacts(MagicMock(), "session-1")


class NonStringError(RuntimeError):
    def __str__(self) -> str:
        return 42  # type: ignore[return-value]

    def __repr__(self) -> str:
        return "non-string __str__ fallback"


class UnprintableError(RuntimeError):
    def __str__(self) -> str:
        raise ValueError("broken __str__")

    def __repr__(self) -> str:
        raise ValueError("broken __repr__")


def test_safe_exception_str_survives_broken_exception_formatters() -> None:
    assert safe_exception_str(NonStringError()) == "non-string __str__ fallback"
    assert safe_exception_str(UnprintableError()) == "<UnprintableError: exception text unavailable>"


class FakeRuntime:
    def __init__(self) -> None:
        self.ctx = MagicMock()
        self.ctx.injected_context_tags = ()
        self.ctx.usage_snapshot_entries = []
        self.ctx.export_state.return_value.model_dump_json.return_value = "{}"

    async def __aenter__(self) -> FakeRuntime:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None


class FakeStreamer:
    def __init__(self, output: object = "hello world") -> None:
        self._history = [
            ModelRequest(parts=[UserPromptPart(content="hello")]),
            ModelResponse(parts=[TextPart(content="hello world")]),
        ]
        self.run = SimpleNamespace(
            result=AgentRunResult(output=output),
            usage=SimpleNamespace(requests=1),
        )

    def __aiter__(self):  # type: ignore[no-untyped-def]
        async def _events():
            yield StreamEvent(
                agent_id="main",
                agent_name="main",
                event=ModelRequestStartEvent(event_id="run-1", loop_index=0, message_count=0),
            )
            yield StreamEvent(
                agent_id="main", agent_name="main", event=PartStartEvent(index=0, part=TextPart(content=""))
            )
            yield StreamEvent(
                agent_id="main",
                agent_name="main",
                event=PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="hello world")),
            )
            yield StreamEvent(
                agent_id="main", agent_name="main", event=PartEndEvent(index=0, part=TextPart(content="hello world"))
            )

        return _events()

    def raise_if_exception(self) -> None:
        return None

    def recoverable_messages(self):  # type: ignore[no-untyped-def]
        return self._history


class FailingFakeStreamer(FakeStreamer):
    def raise_if_exception(self) -> None:
        print("stream diagnostic")
        raise RuntimeError("stream failed")


class UnprintableFailingFakeStreamer(FakeStreamer):
    def raise_if_exception(self) -> None:
        raise UnprintableError


class FakeStreamContext:
    def __init__(self, streamer: FakeStreamer) -> None:
        self.streamer = streamer

    async def __aenter__(self) -> FakeStreamer:
        return self.streamer

    async def __aexit__(self, *_args: object) -> None:
        return None


@pytest.mark.asyncio
async def test_headless_success_saves_when_interactive_auto_save_is_disabled(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Headless success is durable regardless of the interactive auto-save policy."""
    config = _headless_config()
    config.session.auto_save_history = False

    config_manager = MagicMock(spec=ConfigManager)
    config_manager.config_dir = tmp_path / "config"
    config_manager.get_sessions_dir.return_value = tmp_path / "sessions"
    config_manager.load_mcp_config.return_value = None

    runtime = FakeRuntime()
    monkeypatch.setattr("yaacli.headless.create_tui_runtime", MagicMock(return_value=runtime))
    monkeypatch.setattr("yaacli.headless.stream_agent", MagicMock(return_value=FakeStreamContext(FakeStreamer())))
    monkeypatch.setattr("yaacli.headless.get_latest_request_usage", MagicMock(return_value=None))

    result = await run_headless_prompt(
        config=config,
        config_manager=config_manager,
        prompt="hello",
        working_dir=tmp_path,
    )

    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert lines[0]["type"] == "RUN_STARTED"
    assert any(event["type"] == "TEXT_MESSAGE_CHUNK" and event["delta"] == "hello world" for event in lines)
    assert lines[-1]["type"] == "RUN_FINISHED"
    assert lines[-1]["result"] == {"output_text": "hello world"}

    metadata = json.loads(
        (config_manager.get_sessions_dir.return_value / result.session_id / "metadata.json").read_text()
    )
    assert metadata["input_text"] == "hello"
    assert metadata["output_text"] == "hello world"
    display_file = (
        next((config_manager.get_sessions_dir.return_value / result.session_id / "turns").iterdir())
        / "display_messages.json"
    )
    saved_events = json.loads(display_file.read_text())
    assert saved_events[0]["type"] == "RUN_STARTED"
    assert any(event["type"] == "TEXT_MESSAGE_CHUNK" and event["delta"] == "hello world" for event in saved_events)
    assert saved_events[-1]["type"] == "RUN_FINISHED"


@pytest.mark.asyncio
async def test_headless_success_persists_terminal_event_even_with_empty_recoverable_history(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RUN_FINISHED must never be emitted before the successful turn is durable."""
    config = _headless_config()
    config_manager = MagicMock(spec=ConfigManager)
    config_manager.config_dir = tmp_path / "config"
    config_manager.get_sessions_dir.return_value = tmp_path / "sessions"
    config_manager.load_mcp_config.return_value = None
    streamer = FakeStreamer(output="done")
    streamer._history = []

    monkeypatch.setattr("yaacli.headless.create_tui_runtime", MagicMock(return_value=FakeRuntime()))
    monkeypatch.setattr("yaacli.headless.stream_agent", MagicMock(return_value=FakeStreamContext(streamer)))
    monkeypatch.setattr("yaacli.headless.get_latest_request_usage", MagicMock(return_value=None))

    result = await run_headless_prompt(
        config=config,
        config_manager=config_manager,
        prompt="hello",
        working_dir=tmp_path,
    )

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[-1]["type"] == "RUN_FINISHED"
    turn_dir = next((config_manager.get_sessions_dir.return_value / result.session_id / "turns").iterdir())
    assert json.loads((turn_dir / "message_history.json").read_text()) == []
    assert json.loads((turn_dir / "display_messages.json").read_text())[-1]["type"] == "RUN_FINISHED"


@pytest.mark.asyncio
async def test_headless_restore_preserves_current_approval_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Headless restore must keep approval requirements from the current runtime."""
    config = _headless_config()
    config_manager = MagicMock(spec=ConfigManager)
    config_manager.config_dir = tmp_path / "config"
    config_manager.get_sessions_dir.return_value = tmp_path / "sessions"
    config_manager.load_mcp_config.return_value = None

    runtime = FakeRuntime()
    runtime.ctx.shell_env = {}
    runtime.ctx.need_user_approve_tools = ["shell"]
    runtime.ctx.need_user_approve_mcps = ["filesystem"]
    saved = ResumableState(need_user_approve_tools=[], need_user_approve_mcps=[])
    monkeypatch.setattr("yaacli.headless.create_tui_runtime", MagicMock(return_value=runtime))
    monkeypatch.setattr(
        "yaacli.headless._load_session_artifacts",
        MagicMock(return_value=("session-1", [], saved, [])),
    )
    monkeypatch.setattr("yaacli.headless.stream_agent", MagicMock(return_value=FakeStreamContext(FakeStreamer())))
    monkeypatch.setattr("yaacli.headless.get_latest_request_usage", MagicMock(return_value=None))

    await run_headless_prompt(
        config=config,
        config_manager=config_manager,
        prompt="continue",
        working_dir=tmp_path,
        session_id="session-1",
    )

    assert runtime.ctx.need_user_approve_tools == ["shell"]
    assert runtime.ctx.need_user_approve_mcps == ["filesystem"]


@pytest.mark.asyncio
async def test_headless_prompt_failure_keeps_stdout_valid_ndjson(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _headless_config()

    config_manager = MagicMock(spec=ConfigManager)
    config_manager.config_dir = tmp_path / "config"
    config_manager.load_mcp_config.return_value = None

    runtime = FakeRuntime()
    save_session_artifacts = MagicMock()
    monkeypatch.setattr("yaacli.headless.create_tui_runtime", MagicMock(return_value=runtime))
    monkeypatch.setattr(
        "yaacli.headless.stream_agent",
        MagicMock(return_value=FakeStreamContext(FailingFakeStreamer())),
    )
    monkeypatch.setattr("yaacli.headless.get_latest_request_usage", MagicMock(return_value=None))
    monkeypatch.setattr("yaacli.headless._save_session_artifacts", save_session_artifacts)

    with pytest.raises(RuntimeError, match="stream failed"):
        await run_headless_prompt(
            config=config,
            config_manager=config_manager,
            prompt="hello",
            working_dir=tmp_path,
        )

    captured = capsys.readouterr()
    events = [json.loads(line) for line in captured.out.splitlines()]
    assert events[0]["type"] == "RUN_STARTED"
    assert events[-1]["type"] == "RUN_ERROR"
    assert events[-1]["message"] == "stream failed"
    assert "stream diagnostic" in captured.err
    assert "stream diagnostic" not in captured.out
    save_session_artifacts.assert_not_called()


@pytest.mark.asyncio
async def test_headless_bad_exception_text_still_emits_terminal_run_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _headless_config()
    config_manager = MagicMock(spec=ConfigManager)
    config_manager.config_dir = tmp_path / "config"
    config_manager.load_mcp_config.return_value = None

    monkeypatch.setattr("yaacli.headless.create_tui_runtime", MagicMock(return_value=FakeRuntime()))
    monkeypatch.setattr(
        "yaacli.headless.stream_agent",
        MagicMock(return_value=FakeStreamContext(UnprintableFailingFakeStreamer())),
    )

    with pytest.raises(UnprintableError):
        await run_headless_prompt(
            config=config,
            config_manager=config_manager,
            prompt="hello",
            working_dir=tmp_path,
        )

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[-1] == {
        **events[-1],
        "type": "RUN_ERROR",
        "code": "UnprintableError",
        "message": "<UnprintableError: exception text unavailable>",
    }
    assert [event["type"] for event in events if event["type"] in {"RUN_FINISHED", "RUN_ERROR"}] == ["RUN_ERROR"]


@pytest.mark.asyncio
async def test_headless_persistence_failure_emits_run_error_without_finished(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _headless_config()
    config_manager = MagicMock(spec=ConfigManager)
    config_manager.config_dir = tmp_path / "config"
    config_manager.load_mcp_config.return_value = None
    runtime = FakeRuntime()
    monkeypatch.setattr("yaacli.headless.create_tui_runtime", MagicMock(return_value=runtime))
    monkeypatch.setattr("yaacli.headless.stream_agent", MagicMock(return_value=FakeStreamContext(FakeStreamer())))
    monkeypatch.setattr("yaacli.headless.get_latest_request_usage", MagicMock(return_value=None))
    monkeypatch.setattr(
        "yaacli.headless._save_session_artifacts",
        MagicMock(side_effect=OSError("disk full")),
    )

    with pytest.raises(OSError, match="disk full"):
        await run_headless_prompt(
            config=config,
            config_manager=config_manager,
            prompt="hello",
            working_dir=tmp_path,
        )

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[-1]["type"] == "RUN_ERROR"
    assert events[-1]["code"] == "OSError"
    assert not any(event["type"] == "RUN_FINISHED" for event in events)


@pytest.mark.asyncio
async def test_headless_runtime_teardown_failure_emits_one_error_without_finished(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    teardown_error = OSError("runtime teardown failed")

    class TeardownFailingRuntime(FakeRuntime):
        async def __aexit__(self, *_args: object) -> None:
            raise teardown_error

    config = _headless_config()
    config_manager = MagicMock(spec=ConfigManager)
    config_manager.config_dir = tmp_path / "config"
    config_manager.load_mcp_config.return_value = None
    save_session_artifacts = MagicMock()
    monkeypatch.setattr(
        "yaacli.headless.create_tui_runtime",
        MagicMock(return_value=TeardownFailingRuntime()),
    )
    monkeypatch.setattr("yaacli.headless.stream_agent", MagicMock(return_value=FakeStreamContext(FakeStreamer())))
    monkeypatch.setattr("yaacli.headless.get_latest_request_usage", MagicMock(return_value=None))
    monkeypatch.setattr("yaacli.headless._save_session_artifacts", save_session_artifacts)

    with pytest.raises(OSError, match="runtime teardown failed") as exc_info:
        await run_headless_prompt(
            config=config,
            config_manager=config_manager,
            prompt="hello",
            working_dir=tmp_path,
        )

    assert exc_info.value is teardown_error
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    terminal_events = [event for event in events if event["type"] in {"RUN_FINISHED", "RUN_ERROR"}]
    assert len(terminal_events) == 1
    assert terminal_events[0]["type"] == "RUN_ERROR"
    assert terminal_events[0]["code"] == "OSError"
    assert terminal_events[0]["message"] == "runtime teardown failed"
    save_session_artifacts.assert_not_called()


@pytest.mark.parametrize(
    "interruption",
    [
        pytest.param(AgentInterrupted("agent interrupted"), id="agent-interrupted"),
        pytest.param(KeyboardInterrupt(), id="keyboard-interrupt"),
    ],
)
@pytest.mark.asyncio
async def test_headless_interruption_emits_run_cancelled_and_reraises(
    interruption: BaseException,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InterruptedStreamContext:
        async def __aenter__(self) -> FakeStreamer:
            raise interruption

        async def __aexit__(self, *_args: object) -> None:
            return None

    config = _headless_config()
    config_manager = MagicMock(spec=ConfigManager)
    config_manager.config_dir = tmp_path / "config"
    config_manager.load_mcp_config.return_value = None
    save_session_artifacts = MagicMock()
    monkeypatch.setattr("yaacli.headless.create_tui_runtime", MagicMock(return_value=FakeRuntime()))
    monkeypatch.setattr("yaacli.headless.stream_agent", MagicMock(return_value=InterruptedStreamContext()))
    monkeypatch.setattr("yaacli.headless._save_session_artifacts", save_session_artifacts)

    with pytest.raises(type(interruption)) as exc_info:
        await run_headless_prompt(
            config=config,
            config_manager=config_manager,
            prompt="hello",
            working_dir=tmp_path,
        )

    assert exc_info.value is interruption
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[-1]["type"] == "CUSTOM"
    assert events[-1]["name"].endswith("run_cancelled")
    assert events[-1]["value"] == {"reason": "interrupted"}
    assert not any(event["type"] in {"RUN_FINISHED", "RUN_ERROR"} for event in events)
    save_session_artifacts.assert_not_called()


@pytest.mark.asyncio
async def test_headless_cancellation_emits_run_cancelled(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CancelledStreamContext:
        async def __aenter__(self) -> FakeStreamer:
            raise asyncio.CancelledError

        async def __aexit__(self, *_args: object) -> None:
            return None

    config = _headless_config()
    config_manager = MagicMock(spec=ConfigManager)
    config_manager.config_dir = tmp_path / "config"
    config_manager.load_mcp_config.return_value = None
    runtime = FakeRuntime()
    save_session_artifacts = MagicMock()
    monkeypatch.setattr("yaacli.headless.create_tui_runtime", MagicMock(return_value=runtime))
    monkeypatch.setattr("yaacli.headless.stream_agent", MagicMock(return_value=CancelledStreamContext()))
    monkeypatch.setattr("yaacli.headless._save_session_artifacts", save_session_artifacts)

    with pytest.raises(asyncio.CancelledError):
        await run_headless_prompt(
            config=config,
            config_manager=config_manager,
            prompt="hello",
            working_dir=tmp_path,
        )

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[-1]["type"] == "CUSTOM"
    assert events[-1]["name"].endswith("run_cancelled")
    assert events[-1]["value"] == {"reason": "cancelled"}
    assert not any(event["type"] in {"RUN_FINISHED", "RUN_ERROR"} for event in events)
    save_session_artifacts.assert_not_called()


@pytest.mark.asyncio
async def test_headless_prompt_uses_persisted_startup_profile_for_runtime_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _headless_config()
    config.general.model = "openai:test-default"
    config.model_profiles = {
        "persisted": ModelProfileConfig(
            label="Persisted",
            model="openai:test-persisted",
        )
    }
    config_manager = MagicMock(spec=ConfigManager)
    config_manager.config_dir = tmp_path / "config"
    config_manager.get_sessions_dir.return_value = tmp_path / "sessions"
    config_manager.load_mcp_config.return_value = None
    save_selected_model_profile_id(config_manager.config_dir, "persisted")

    runtime_factory = MagicMock(return_value=FakeRuntime())
    monkeypatch.setattr("yaacli.headless.create_tui_runtime", runtime_factory)
    monkeypatch.setattr("yaacli.headless.stream_agent", MagicMock(return_value=FakeStreamContext(FakeStreamer())))
    monkeypatch.setattr("yaacli.headless.get_latest_request_usage", MagicMock(return_value=None))

    result = await run_headless_prompt(
        config=config,
        config_manager=config_manager,
        prompt="hello",
        working_dir=tmp_path,
    )

    runtime_profile = runtime_factory.call_args.kwargs["model_profile"]
    metadata = json.loads(
        (config_manager.get_sessions_dir.return_value / result.session_id / "metadata.json").read_text()
    )
    assert runtime_profile is not None
    assert (
        (
            metadata["model_profile_id"],
            metadata["model_label"],
            metadata["model"],
        )
        == (
            runtime_profile.id,
            runtime_profile.label,
            runtime_profile.model,
        )
        == (
            "persisted",
            "Persisted",
            "openai:test-persisted",
        )
    )


@pytest.mark.asyncio
async def test_headless_prompt_uses_model_profile_and_auto_denies_hitl(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _headless_config()
    config.general.model = "openai:test-default"
    config.model_profiles = {
        "fast": ModelProfileConfig(
            label="Fast",
            model="openai:test-fast",
        )
    }

    config_manager = MagicMock(spec=ConfigManager)
    config_manager.config_dir = tmp_path / "config"
    config_manager.get_sessions_dir.return_value = tmp_path / "sessions"
    config_manager.load_mcp_config.return_value = None
    save_selected_model_profile_id(config_manager.config_dir, "default")

    runtime = FakeRuntime()
    runtime_factory = MagicMock(return_value=runtime)
    monkeypatch.setattr("yaacli.headless.create_tui_runtime", runtime_factory)

    deferred = DeferredToolRequests(
        approvals=[ToolCallPart(tool_name="edit", args={}, tool_call_id="approval-1")],
        calls=[ToolCallPart(tool_name="fetch_secret", args={}, tool_call_id="call-1")],
    )
    stream_agent_mock = MagicMock(
        side_effect=[
            FakeStreamContext(FakeStreamer(output=deferred)),
            FakeStreamContext(FakeStreamer(output="denied done")),
        ]
    )
    monkeypatch.setattr("yaacli.headless.stream_agent", stream_agent_mock)
    monkeypatch.setattr("yaacli.headless.get_latest_request_usage", MagicMock(return_value=None))

    result = await run_headless_prompt(
        config=config,
        config_manager=config_manager,
        prompt="hello",
        working_dir=tmp_path,
        model_profile_id="fast",
    )

    assert result.output_text == "denied done"
    assert runtime_factory.call_args.kwargs["model_profile"].id == "fast"
    assert runtime_factory.call_args.kwargs["enable_async_subagents"] is False
    assert runtime_factory.call_args.kwargs["enable_delegate_subagents"] is True
    second_call = stream_agent_mock.call_args_list[1]
    deferred_results = second_call.kwargs["deferred_tool_results"]
    approval_result = deferred_results.approvals["approval-1"]
    assert isinstance(approval_result, ToolDenied)
    assert approval_result.message == "Headless mode denies HITL requests by default."
    call_result = deferred_results.calls["call-1"]
    assert isinstance(call_result, RetryPromptPart)
    assert call_result.content == "Headless mode denies HITL requests by default."
    assert call_result.tool_name == "fetch_secret"
    assert call_result.tool_call_id == "call-1"

    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    hitl_event = next(event for event in lines if event.get("name") == "yaacli.hitl_auto_denied")
    assert hitl_event["value"] == {
        "approval_count": 1,
        "approvals": ["approval-1"],
        "call_count": 1,
        "calls": ["call-1"],
        "reason": "Headless mode denies HITL requests by default.",
    }
    assert lines[-1]["type"] == "RUN_FINISHED"
    assert lines[-1]["result"] == {"output_text": "denied done"}

    metadata = json.loads(
        (config_manager.get_sessions_dir.return_value / result.session_id / "metadata.json").read_text()
    )
    assert metadata["model_profile_id"] == "fast"
    assert metadata["model_label"] == "Fast"
    assert metadata["model"] == "openai:test-fast"
    persisted_state = json.loads((config_manager.config_dir / "state.json").read_text())
    assert persisted_state["model_profile"]["selected_profile_id"] == "default"


@pytest.mark.asyncio
async def test_headless_rejects_empty_deferred_requests_immediately(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _headless_config()
    config_manager = MagicMock(spec=ConfigManager)
    config_manager.config_dir = tmp_path / "config"
    config_manager.load_mcp_config.return_value = None
    stream_agent_mock = MagicMock(return_value=FakeStreamContext(FakeStreamer(output=DeferredToolRequests())))
    monkeypatch.setattr("yaacli.headless.create_tui_runtime", MagicMock(return_value=FakeRuntime()))
    monkeypatch.setattr("yaacli.headless.stream_agent", stream_agent_mock)
    monkeypatch.setattr("yaacli.headless.get_latest_request_usage", MagicMock(return_value=None))

    with pytest.raises(RuntimeError, match="empty DeferredToolRequests"):
        await run_headless_prompt(
            config=config,
            config_manager=config_manager,
            prompt="hello",
            working_dir=tmp_path,
        )

    assert stream_agent_mock.call_count == 1
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[-1]["type"] == "RUN_ERROR"
    assert events[-1]["message"] == "Agent returned an empty DeferredToolRequests payload."
    assert not any(event.get("name") == "yaacli.hitl_auto_denied" for event in events)


@pytest.mark.asyncio
async def test_headless_deferred_continuations_share_cumulative_request_limit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _headless_config(max_requests=2)
    config_manager = MagicMock(spec=ConfigManager)
    config_manager.config_dir = tmp_path / "config"
    config_manager.load_mcp_config.return_value = None
    deferred = DeferredToolRequests(approvals=[ToolCallPart(tool_name="edit", args={}, tool_call_id="approval-1")])
    stream_agent_mock = MagicMock(
        side_effect=[
            FakeStreamContext(FakeStreamer(output=deferred)),
            FakeStreamContext(FakeStreamer(output=deferred)),
        ]
    )
    monkeypatch.setattr("yaacli.headless.create_tui_runtime", MagicMock(return_value=FakeRuntime()))
    monkeypatch.setattr("yaacli.headless.stream_agent", stream_agent_mock)
    monkeypatch.setattr("yaacli.headless.get_latest_request_usage", MagicMock(return_value=None))

    with pytest.raises(RuntimeError, match="cumulative model request limit of 2"):
        await run_headless_prompt(
            config=config,
            config_manager=config_manager,
            prompt="hello",
            working_dir=tmp_path,
        )

    assert stream_agent_mock.call_count == 2
    assert [call.kwargs["usage_limits"].request_limit for call in stream_agent_mock.call_args_list] == [2, 1]
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert sum(event.get("name") == "yaacli.hitl_auto_denied" for event in events) == 2
    assert events[-1]["type"] == "RUN_ERROR"
    assert not any(event["type"] == "RUN_FINISHED" for event in events)


@pytest.mark.asyncio
async def test_headless_worker_disables_delegate_subagents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _headless_config()

    config_manager = MagicMock(spec=ConfigManager)
    config_manager.config_dir = tmp_path / "config"
    config_manager.get_sessions_dir.return_value = tmp_path / "sessions"
    config_manager.load_mcp_config.return_value = None

    runtime = FakeRuntime()
    runtime_factory = MagicMock(return_value=runtime)
    monkeypatch.setattr("yaacli.headless.create_tui_runtime", runtime_factory)
    monkeypatch.setattr("yaacli.headless.stream_agent", MagicMock(return_value=FakeStreamContext(FakeStreamer())))
    monkeypatch.setattr("yaacli.headless.get_latest_request_usage", MagicMock(return_value=None))

    await run_headless_prompt(
        config=config,
        config_manager=config_manager,
        prompt="hello",
        working_dir=tmp_path,
        worker=True,
    )

    assert runtime_factory.call_args.kwargs["enable_async_subagents"] is False
    assert runtime_factory.call_args.kwargs["enable_delegate_subagents"] is False


@pytest.mark.asyncio
async def test_headless_prompt_restores_session_by_prefix(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _headless_config()

    sessions_dir = tmp_path / "sessions"
    session_dir = sessions_dir / "abcdef123456"
    session_dir.mkdir(parents=True)
    (session_dir / "metadata.json").write_text(json.dumps({"session_id": session_dir.name}))
    (session_dir / "message_history.json").write_bytes(b"[]")
    (session_dir / "display_messages.json").write_text(
        json.dumps([{"type": "TEXT_MESSAGE_CHUNK", "messageId": "old", "delta": "previous"}])
    )

    config_manager = MagicMock(spec=ConfigManager)
    config_manager.config_dir = tmp_path / "config"
    config_manager.get_sessions_dir.return_value = sessions_dir
    config_manager.load_mcp_config.return_value = None

    runtime = FakeRuntime()
    monkeypatch.setattr("yaacli.headless.create_tui_runtime", MagicMock(return_value=runtime))
    stream_agent_mock = MagicMock(return_value=FakeStreamContext(FakeStreamer()))
    monkeypatch.setattr("yaacli.headless.stream_agent", stream_agent_mock)
    monkeypatch.setattr("yaacli.headless.get_latest_request_usage", MagicMock(return_value=None))

    result = await run_headless_prompt(
        config=config,
        config_manager=config_manager,
        prompt="hello",
        working_dir=tmp_path,
        session_id="abc",
    )

    assert result.session_id == "abcdef123456"
    assert stream_agent_mock.call_args.kwargs["message_history"] == []
    saved_events = json.loads(next((session_dir / "turns").iterdir()).joinpath("display_messages.json").read_text())
    assert any(event.get("delta") == "previous" for event in saved_events)
    assert json.loads(capsys.readouterr().out.splitlines()[0])["type"] == "RUN_STARTED"
