"""Price-update scheduling belongs to the application, not individual SDK runtimes."""

from unittest.mock import MagicMock

from pydantic_ai.models.test import TestModel
from ya_agent_sdk.agents.main import create_agent
from ya_agent_sdk.context import AgentContext


async def test_sdk_runtime_does_not_start_price_updates(
    agent_context: AgentContext, mock_price_updates: MagicMock
) -> None:
    runtime = create_agent(TestModel(), env=agent_context.env)
    for _ in range(2):
        async with runtime:
            async with runtime:
                result = await runtime.agent.run("Hello", deps=runtime.ctx)
                assert result.output
    mock_price_updates.assert_not_called()
