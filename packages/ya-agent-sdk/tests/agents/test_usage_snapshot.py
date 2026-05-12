from __future__ import annotations

import pytest
from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from ya_agent_sdk.agents.main import create_agent, stream_agent
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.environment.local import LocalEnvironment
from ya_agent_sdk.events import UsageSnapshotEvent
from ya_agent_sdk.toolsets.core.base import BaseTool


class UsageBoundaryTool(BaseTool):
    name = "tool"
    description = "A tool that forces a second model request."

    async def call(self, ctx: RunContext[AgentContext]) -> str:
        return "tool result"


@pytest.mark.asyncio
async def test_stream_agent_emits_usage_snapshot_after_node_complete(tmp_path):
    env = LocalEnvironment(tmp_base_dir=tmp_path)
    runtime = create_agent(TestModel(custom_output_text="ok"), env=env)

    usage_events: list[UsageSnapshotEvent] = []
    async with stream_agent(runtime, "hello") as streamer:
        async for stream_event in streamer:
            if isinstance(stream_event.event, UsageSnapshotEvent):
                usage_events.append(stream_event.event)

    assert [event.source for event in usage_events] == ["model_request_complete", "session_end"]
    snapshot = usage_events[-1].snapshot
    assert snapshot is not None
    assert snapshot.run_id == runtime.ctx.run_id
    assert snapshot.total_usage.requests >= 1
    assert snapshot.entries
    assert snapshot.entries[0].agent_id == "main"
    assert snapshot.entries[0].agent_name == "main"
    assert snapshot.agent_usages["main"].usage == snapshot.entries[0].usage
    assert snapshot.model_usages


@pytest.mark.asyncio
async def test_stream_agent_emits_usage_snapshot_after_changed_nodes(tmp_path):
    env = LocalEnvironment(tmp_base_dir=tmp_path)
    runtime = create_agent(TestModel(call_tools=["tool"]), env=env, tools=[UsageBoundaryTool])

    events: list[object] = []
    async with stream_agent(runtime, "hello") as streamer:
        async for stream_event in streamer:
            events.append(stream_event.event)

    usage_events = [event for event in events if isinstance(event, UsageSnapshotEvent)]
    assert [event.source for event in usage_events] == [
        "model_request_complete",
        "model_request_complete",
        "session_end",
    ]
    assert usage_events[0].snapshot is not None
    assert usage_events[0].snapshot.total_usage.requests >= 1
    assert usage_events[1].snapshot is not None
    assert usage_events[1].snapshot.total_usage.requests > usage_events[0].snapshot.total_usage.requests
    assert usage_events[2].snapshot == usage_events[1].snapshot


@pytest.mark.asyncio
async def test_usage_ledger_update_can_be_emitted_later(agent_context):
    agent_context.update_usage_snapshot_entry(
        agent_id="shell_review",
        agent_name="shell_review",
        model_id="review-model",
        usage=RunUsage(requests=1, input_tokens=10, output_tokens=2),
        source="shell_review",
        usage_id="review-1",
        ledger_key="review-1",
    )

    assert agent_context.agent_stream_queues[agent_context.agent_id].empty()

    object.__setattr__(agent_context, "_stream_queue_enabled", True)
    await agent_context.emit_usage_snapshot_event(source="session_end")

    event = await agent_context.agent_stream_queues[agent_context.agent_id].get()
    assert isinstance(event, UsageSnapshotEvent)
    assert event.source == "session_end"
    assert event.snapshot is not None
    assert event.snapshot.entries[0].agent_id == "shell_review"
    assert event.snapshot.total_usage.requests == 1
