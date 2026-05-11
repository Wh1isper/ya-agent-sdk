from __future__ import annotations

import pytest
from pydantic_ai.models.test import TestModel
from ya_agent_sdk.agents.main import create_agent, stream_agent
from ya_agent_sdk.environment.local import LocalEnvironment
from ya_agent_sdk.events import UsageSnapshotEvent


@pytest.mark.asyncio
async def test_stream_agent_emits_usage_snapshot(tmp_path):
    env = LocalEnvironment(tmp_base_dir=tmp_path)
    runtime = create_agent(TestModel(custom_output_text="ok"), env=env)

    usage_events: list[UsageSnapshotEvent] = []
    async with stream_agent(runtime, "hello") as streamer:
        async for stream_event in streamer:
            if isinstance(stream_event.event, UsageSnapshotEvent):
                usage_events.append(stream_event.event)

    assert usage_events
    snapshot = usage_events[-1].snapshot
    assert snapshot is not None
    assert snapshot.run_id == runtime.ctx.run_id
    assert snapshot.total_usage.requests >= 1
    assert snapshot.entries
    assert snapshot.entries[0].agent_id == "main"
    assert snapshot.entries[0].agent_name == "main"
    assert snapshot.agent_usages["main"].usage == snapshot.entries[0].usage
    assert snapshot.model_usages
