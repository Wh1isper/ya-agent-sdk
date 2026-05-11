from pathlib import Path

import pytest
from ya_agent_sdk.context import AgentContext, BusMessage
from ya_agent_sdk.environment.local import LocalEnvironment


@pytest.fixture
async def env(tmp_path: Path):
    async with LocalEnvironment(
        default_path=tmp_path,
        allowed_paths=[tmp_path],
        tmp_base_dir=tmp_path,
    ) as environment:
        yield environment


async def test_main_agent_keeps_message_bus_subscription_between_runs(env: LocalEnvironment) -> None:
    ctx = AgentContext(agent_id="main", env=env)

    async with ctx:
        assert ctx.message_bus.subscriber_count == 1

    ctx.send_message(BusMessage(content="background result", source="executor-bg-123", target="main"))

    assert ctx.message_bus.has_pending("main") is True
    assert [message.content for message in ctx.consume_messages()] == ["background result"]


async def test_subagent_unsubscribes_from_message_bus_on_exit(env: LocalEnvironment) -> None:
    parent = AgentContext(agent_id="main", env=env)

    async with parent:
        async with parent.create_subagent_context("executor", agent_id="executor-bg-123") as child:
            assert child.message_bus.subscriber_count == 2

        assert parent.message_bus.subscriber_count == 1
