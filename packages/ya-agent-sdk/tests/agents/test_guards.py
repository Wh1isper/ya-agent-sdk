"""Tests for message-bus completion guards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelRequest, RetryPromptPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_graph import End
from ya_agent_sdk.agents.guards import MessageBusGuardCapability
from ya_agent_sdk.agents.main import create_agent, stream_agent
from ya_agent_sdk.context import AgentContext, BusMessage, MessageBus
from ya_agent_sdk.environment.local import LocalEnvironment


def create_mock_ctx(agent_id: str = "main", message_bus: MessageBus | None = None) -> RunContext[AgentContext]:
    """Create a mock RunContext with AgentContext."""
    deps = AgentContext()
    deps._agent_id = agent_id
    if message_bus is not None:
        deps.message_bus = message_bus

    ctx = MagicMock(spec=RunContext)
    ctx.deps = deps
    ctx.enqueue.return_value = "enqueue-1"
    return ctx


async def test_message_bus_guard_only_wakes_terminal_pending_messages() -> None:
    bus = MessageBus()
    bus.subscribe("main")
    ctx = create_mock_ctx(message_bus=bus)
    guard = MessageBusGuardCapability()
    terminal = End("done")

    assert await guard.after_node_run(ctx, node=object(), result=terminal) is terminal
    ctx.enqueue.assert_not_called()

    bus.send(BusMessage(content="Please focus", source="user", target="main"))
    non_terminal = object()
    assert await guard.after_node_run(ctx, node=object(), result=non_terminal) is non_terminal
    ctx.enqueue.assert_not_called()

    assert await guard.after_node_run(ctx, node=object(), result=terminal) is terminal
    ctx.enqueue.assert_called_once()
    assert "pending messages" in ctx.enqueue.call_args.args[0].lower()
    assert ctx.enqueue.call_args.kwargs == {"priority": "asap"}


async def test_message_bus_guard_respects_target_and_subscription() -> None:
    bus = MessageBus()
    bus.subscribe("main")
    bus.subscribe("other-agent")
    bus.send(BusMessage(content="For other", source="user", target="other-agent"))
    ctx = create_mock_ctx(agent_id="main", message_bus=bus)
    guard = MessageBusGuardCapability()

    await guard.after_node_run(ctx, node=object(), result=End("done"))
    ctx.enqueue.assert_not_called()

    unsubscribed_bus = MessageBus()
    unsubscribed_bus.send(BusMessage(content="For main", source="user", target="main"))
    unsubscribed_ctx = create_mock_ctx(message_bus=unsubscribed_bus)
    await guard.after_node_run(unsubscribed_ctx, node=object(), result=End("done"))
    unsubscribed_ctx.enqueue.assert_not_called()


@dataclass
class _ObserveStreamedNodeBoundaryCapability(AbstractCapability[AgentContext]):
    """Record whether wrap_node_run executes while node streaming is active."""

    streaming: bool = False
    observed_streaming_states: list[bool] | None = None

    async def wrap_node_run(self, ctx, *, node, handler):  # type: ignore[no-untyped-def]
        assert self.observed_streaming_states is not None
        self.observed_streaming_states.append(self.streaming)
        return await handler(node)


@dataclass
class _SendAtEndCapability(AbstractCapability[AgentContext]):
    """Inject a bus message after output validation but before the SDK guard."""

    sent: bool = False

    async def after_node_run(self, ctx: RunContext[AgentContext], *, node: object, result: object) -> object:
        if isinstance(result, End) and not self.sent:
            self.sent = True
            ctx.deps.send_message(BusMessage(content="Please focus", source="user", target="main"))
        return result


async def test_streamed_driver_keeps_node_stream_outside_wrap_node_run(tmp_path: Path) -> None:
    observed_states: list[bool] = []
    capability = _ObserveStreamedNodeBoundaryCapability(observed_streaming_states=observed_states)

    async def stream_function(_messages: list[ModelMessage], _info: AgentInfo):
        capability.streaming = True
        try:
            yield "answer"
        finally:
            capability.streaming = False

    env = LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
    runtime = create_agent(
        FunctionModel(stream_function=stream_function),
        env=env,
        capabilities=[capability],
    )

    async with stream_agent(runtime, "start") as streamer:
        async for _event in streamer:
            pass
        streamer.raise_if_exception()

    assert observed_states
    assert all(state is False for state in observed_states)


async def test_end_boundary_bus_message_redirects_streamed_run_without_model_retry(tmp_path: Path) -> None:
    """A message arriving after output validation still redirects the ending run."""
    model_requests: list[list[ModelMessage]] = []

    async def stream_function(messages: list[ModelMessage], _info: AgentInfo):
        model_requests.append(list(messages))
        yield "first answer" if len(model_requests) == 1 else "revised answer"

    env = LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path)
    runtime = create_agent(
        FunctionModel(stream_function=stream_function),
        env=env,
        output_retries=0,
        capabilities=[_SendAtEndCapability()],
    )

    async with stream_agent(runtime, "start") as streamer:
        async for _event in streamer:
            pass
        streamer.raise_if_exception()

    assert len(model_requests) == 2
    assert streamer.run is not None
    assert streamer.run.result is not None
    assert streamer.run.result.output == "revised answer"
    assert any(
        isinstance(part, UserPromptPart) and "Please focus" in str(part.content)
        for message in model_requests[1]
        if isinstance(message, ModelRequest)
        for part in message.parts
    )
    assert not any(
        isinstance(part, RetryPromptPart)
        for message in streamer.run.result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
    )
