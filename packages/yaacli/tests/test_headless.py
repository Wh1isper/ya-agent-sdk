from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic_ai import (
    AgentRunResult,
    DeferredToolRequests,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPartDelta,
    ToolDenied,
)
from pydantic_ai.messages import ModelRequest, ModelResponse, RetryPromptPart, TextPart, ToolCallPart, UserPromptPart
from ya_agent_sdk.context.agent import StreamEvent
from ya_agent_sdk.events import ModelRequestStartEvent
from yaacli.config import ConfigManager
from yaacli.headless import run_headless_prompt


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
        self.run = SimpleNamespace(result=AgentRunResult(output=output))

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


class FakeStreamContext:
    def __init__(self, streamer: FakeStreamer) -> None:
        self.streamer = streamer

    async def __aenter__(self) -> FakeStreamer:
        return self.streamer

    async def __aexit__(self, *_args: object) -> None:
        return None


@pytest.mark.asyncio
async def test_headless_prompt_streams_ndjson_and_saves_display_messages(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = MagicMock()
    config.general.max_requests = 10
    config.general.agent_stream_resume_on_error = False
    config.general.agent_stream_resume_max_attempts = 0
    config.general.agent_stream_resume_prompt = None

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

    display_file = (
        next((config_manager.get_sessions_dir.return_value / result.session_id / "turns").iterdir())
        / "display_messages.json"
    )
    saved_events = json.loads(display_file.read_text())
    assert saved_events[0]["type"] == "RUN_STARTED"
    assert any(event["type"] == "TEXT_MESSAGE_CHUNK" and event["delta"] == "hello world" for event in saved_events)
    assert saved_events[-1]["type"] == "RUN_FINISHED"


@pytest.mark.asyncio
async def test_headless_prompt_uses_model_profile_and_auto_denies_hitl(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = MagicMock()
    config.general.max_requests = 10
    config.general.agent_stream_resume_on_error = False
    config.general.agent_stream_resume_max_attempts = 0
    config.general.agent_stream_resume_prompt = None
    config.general.model = "openai:test-default"
    config.general.model_settings = None
    config.general.model_cfg = None
    profile = MagicMock()
    profile.label = "Fast"
    profile.model = "openai:test-fast"
    profile.model_settings = None
    profile.model_cfg = None
    config.model_profiles = {"fast": profile}

    config_manager = MagicMock(spec=ConfigManager)
    config_manager.config_dir = tmp_path / "config"
    config_manager.get_sessions_dir.return_value = tmp_path / "sessions"
    config_manager.load_mcp_config.return_value = None

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


@pytest.mark.asyncio
async def test_headless_worker_disables_delegate_subagents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = MagicMock()
    config.general.max_requests = 10
    config.general.agent_stream_resume_on_error = False
    config.general.agent_stream_resume_max_attempts = 0
    config.general.agent_stream_resume_prompt = None

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
    config = MagicMock()
    config.general.max_requests = 10
    config.general.agent_stream_resume_on_error = False
    config.general.agent_stream_resume_max_attempts = 0
    config.general.agent_stream_resume_prompt = None

    sessions_dir = tmp_path / "sessions"
    session_dir = sessions_dir / "abcdef123456"
    session_dir.mkdir(parents=True)
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
