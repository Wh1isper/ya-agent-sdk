from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models.test import TestModel
from ya_agent_sdk.agents.lifecycle import (
    BaseLifecycleExtension,
    CompactCompleteContext,
    ContextHandoffCompleteContext,
    ContextHandoffSource,
)
from ya_agent_sdk.agents.main import create_agent, stream_agent
from ya_agent_sdk.agents.trim import TrimHistoryOptions, trim_history_for_summary
from ya_agent_sdk.context import AgentContext, BusMessage
from ya_agent_sdk.environment.local import LocalEnvironment


class RecordingExtension(BaseLifecycleExtension[AgentContext, LocalEnvironment]):
    name = "recording"

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def on_runtime_ready(self, ctx) -> None:
        self.calls.append("runtime_ready")
        if isinstance(ctx.user_prompt, str):
            ctx.user_prompt = f"{ctx.user_prompt} with extension"

    async def on_agent_start(self, ctx) -> None:
        self.calls.append("agent_start")

    async def on_agent_complete(self, ctx) -> None:
        self.calls.append("agent_complete")


class RecordingHandoffExtension(BaseLifecycleExtension[AgentContext, LocalEnvironment]):
    name = "handoff_recording"

    def __init__(self) -> None:
        self.seen: list[ContextHandoffCompleteContext[AgentContext]] = []

    async def on_context_handoff_complete(self, ctx: ContextHandoffCompleteContext[AgentContext]) -> None:
        self.seen.append(ctx)


class FailingHandoffExtension(BaseLifecycleExtension[AgentContext, LocalEnvironment]):
    async def on_context_handoff_complete(self, ctx: ContextHandoffCompleteContext[AgentContext]) -> None:
        raise RuntimeError("hook failed")


class CapturingTestModel(TestModel):
    last_messages: list[ModelMessage] | None = None

    async def request(self, messages, model_settings, model_request_parameters):
        self.last_messages = messages
        return await super().request(messages, model_settings, model_request_parameters)

    @asynccontextmanager
    async def request_stream(self, messages, model_settings, model_request_parameters, run_context=None):
        self.last_messages = messages
        async with super().request_stream(messages, model_settings, model_request_parameters, run_context) as response:
            yield response


@pytest.mark.asyncio
async def test_stream_agent_runs_lifecycle_extension(tmp_path):
    env = LocalEnvironment(tmp_base_dir=tmp_path)
    extension = RecordingExtension()
    runtime = create_agent(
        TestModel(custom_output_text="ok"),
        env=env,
        lifecycle_extensions=[extension],
    )

    async with stream_agent(runtime, "hello") as streamer:
        async for _event in streamer:
            pass

    assert extension.calls == ["runtime_ready", "agent_start", "agent_complete"]
    messages = streamer.run.all_messages() if streamer.run is not None else []
    assert "hello with extension" in str(messages[-2])


@pytest.mark.asyncio
async def test_stream_agent_captures_resolved_user_prompt_from_factory(tmp_path):
    env = LocalEnvironment(tmp_base_dir=tmp_path)
    runtime = create_agent(TestModel(custom_output_text="ok"), env=env)

    async def user_prompt_factory(_runtime):
        return "factory prompt"

    async with stream_agent(runtime, user_prompt_factory=user_prompt_factory) as streamer:
        async for _event in streamer:
            pass

    assert runtime.ctx.user_prompts == "factory prompt"


@pytest.mark.asyncio
async def test_stream_agent_enters_fresh_context_when_runtime_is_already_entered(tmp_path):
    env = LocalEnvironment(tmp_base_dir=tmp_path)
    model = CapturingTestModel(custom_output_text="ok")
    runtime = create_agent(model, env=env)

    async with runtime:
        runtime.ctx.send_message(BusMessage(content="steer now", source="user", target="main"))
        async with stream_agent(runtime, "hello") as streamer:
            async for _event in streamer:
                pass

    assert model.last_messages is not None
    assert any("steer now" in str(message) for message in model.last_messages)


def test_trim_history_for_summary_returns_metrics():
    from pydantic_ai.messages import ModelRequest, ToolReturnPart, UserPromptPart

    history = [
        ModelRequest(
            parts=[
                UserPromptPart(content="<runtime-context>old</runtime-context>keep"),
                ToolReturnPart(tool_name="tool", tool_call_id="call", content="x" * 1000),
            ]
        )
    ]

    result = trim_history_for_summary(
        history,
        TrimHistoryOptions(max_tool_return_chars=20, tool_return_keep_head=5, tool_return_keep_tail=5),
    )

    assert result.original_message_count == 1
    assert result.trimmed_message_count == 1
    assert result.truncated_tool_return_count == 1
    assert result.stripped_injected_context_count == 1


@pytest.mark.asyncio
async def test_compact_callback_receives_trimmed_messages(agent_context: AgentContext, monkeypatch):
    from pydantic_ai.messages import ModelRequest, UserPromptPart
    from pydantic_ai.usage import RunUsage
    from ya_agent_sdk.agents import compact as compact_module
    from ya_agent_sdk.agents.compact import create_cache_friendly_compact_filter

    message_history = [ModelRequest(parts=[UserPromptPart(content="<runtime-context>old</runtime-context>hello")])]
    agent_context.model_cfg.context_window = 10
    object.__setattr__(agent_context, "_stream_queue_enabled", True)

    mock_run_ctx = MagicMock()
    mock_run_ctx.deps = agent_context
    mock_run_ctx.model = MagicMock(model_name="main-model")
    mock_run_ctx.agent = MagicMock()
    mock_run_ctx.agent._output_validators = []

    class _StreamResult:
        def usage(self):
            return RunUsage(input_tokens=1, output_tokens=1, requests=1)

        async def get_output(self):
            return "summary"

    class _RunStream:
        async def __aenter__(self):
            return _StreamResult()

        async def __aexit__(self, exc_type, exc, tb):
            return None

    compact_agent = MagicMock()
    compact_agent._output_validators = []
    compact_agent.run_stream.return_value = _RunStream()
    monkeypatch.setattr(compact_module.copy, "copy", lambda _agent: compact_agent)
    monkeypatch.setattr(compact_module, "_need_compact", lambda *_args, **_kwargs: True)

    seen: list[CompactCompleteContext[AgentContext]] = []

    async def callback(ctx: CompactCompleteContext[AgentContext]) -> None:
        seen.append(ctx)

    compact_filter = create_cache_friendly_compact_filter(model_cfg=agent_context.model_cfg, callbacks=[callback])
    await compact_filter(mock_run_ctx, message_history)

    assert len(seen) == 1
    assert seen[0].source == ContextHandoffSource.COMPACT
    assert seen[0].summary_markdown == "summary"
    assert seen[0].handoff_messages == seen[0].compacted_messages
    request = seen[0].trimmed_messages[0]
    assert "runtime-context" not in request.parts[0].content


@pytest.mark.asyncio
async def test_compact_hook_failure_propagates(agent_context: AgentContext, monkeypatch):
    from pydantic_ai.usage import RunUsage
    from ya_agent_sdk.agents import compact as compact_module
    from ya_agent_sdk.agents.compact import create_cache_friendly_compact_filter

    message_history = [ModelRequest(parts=[UserPromptPart(content="hello")])]
    agent_context.model_cfg.context_window = 10
    agent_context.lifecycle_extensions = [FailingHandoffExtension()]
    object.__setattr__(agent_context, "_stream_queue_enabled", True)

    mock_run_ctx = MagicMock()
    mock_run_ctx.deps = agent_context
    mock_run_ctx.model = MagicMock(model_name="main-model")
    mock_run_ctx.agent = MagicMock()
    mock_run_ctx.agent._output_validators = []

    class _StreamResult:
        def usage(self):
            return RunUsage(input_tokens=1, output_tokens=1, requests=1)

        async def get_output(self):
            return "summary"

    class _RunStream:
        async def __aenter__(self):
            return _StreamResult()

        async def __aexit__(self, exc_type, exc, tb):
            return None

    compact_agent = MagicMock()
    compact_agent._output_validators = []
    compact_agent.run_stream.return_value = _RunStream()
    monkeypatch.setattr(compact_module.copy, "copy", lambda _agent: compact_agent)
    monkeypatch.setattr(compact_module, "_need_compact", lambda *_args, **_kwargs: True)

    compact_filter = create_cache_friendly_compact_filter(model_cfg=agent_context.model_cfg)

    with pytest.raises(RuntimeError, match="hook failed"):
        await compact_filter(mock_run_ctx, message_history)

    events = []
    queue = agent_context.agent_stream_queues[agent_context.agent_id]
    while not queue.empty():
        events.append(queue.get_nowait())
    assert [event.__class__.__name__ for event in events] == ["CompactStartEvent", "UsageSnapshotEvent"]


@pytest.mark.asyncio
async def test_handoff_filter_emits_context_handoff_complete(agent_context: AgentContext):
    from ya_agent_sdk.filters.handoff import process_handoff_message

    extension = RecordingHandoffExtension()
    agent_context.lifecycle_extensions = [extension]
    agent_context.handoff_message = "summary from summarize tool"
    message_history = [ModelRequest(parts=[UserPromptPart(content="<runtime-context>old</runtime-context>hello")])]

    mock_run_ctx = MagicMock()
    mock_run_ctx.deps = agent_context

    result = await process_handoff_message(mock_run_ctx, message_history)

    assert len(extension.seen) == 1
    seen = extension.seen[0]
    assert seen.source == ContextHandoffSource.SUMMARIZE_TOOL
    assert seen.summary_markdown == "summary from summarize tool"
    assert seen.original_messages == message_history
    assert seen.handoff_messages == result
    assert "runtime-context" not in seen.trimmed_messages[0].parts[0].content


@pytest.mark.asyncio
async def test_handoff_hook_failure_propagates(agent_context: AgentContext):
    from ya_agent_sdk.filters.handoff import process_handoff_message

    agent_context.lifecycle_extensions = [FailingHandoffExtension()]
    agent_context.handoff_message = "summary from summarize tool"
    message_history = [ModelRequest(parts=[UserPromptPart(content="hello")])]

    mock_run_ctx = MagicMock()
    mock_run_ctx.deps = agent_context

    with pytest.raises(RuntimeError, match="hook failed"):
        await process_handoff_message(mock_run_ctx, message_history)

    assert agent_context.handoff_message == "summary from summarize tool"
