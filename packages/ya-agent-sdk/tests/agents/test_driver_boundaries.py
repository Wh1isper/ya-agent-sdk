"""Tests for streamed graph-driver boundary ordering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from ya_agent_sdk.agents.main import create_agent, stream_agent
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.environment.local import LocalEnvironment


@dataclass
class _ObserveStreamedNodeBoundaryCapability(AbstractCapability[AgentContext]):
    streaming: bool = False
    observed_streaming_states: list[bool] | None = None

    async def wrap_node_run(self, ctx, *, node, handler):  # type: ignore[no-untyped-def]
        assert self.observed_streaming_states is not None
        self.observed_streaming_states.append(self.streaming)
        return await handler(node)


async def test_streamed_driver_keeps_node_stream_outside_wrap_node_run(
    tmp_path: Path,
) -> None:
    observed_states: list[bool] = []
    capability = _ObserveStreamedNodeBoundaryCapability(observed_streaming_states=observed_states)

    async def stream_function(_messages: list[ModelMessage], _info: AgentInfo):
        capability.streaming = True
        try:
            yield "answer"
        finally:
            capability.streaming = False

    env = LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    )
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
