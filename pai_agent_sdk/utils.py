from pydantic_ai import Agent, RunContext
from pydantic_ai.output import OutputDataT

from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.toolsets.base import BaseToolset


def add_toolset_instructions(
    agent: Agent[AgentContext, OutputDataT], toolsets: list[BaseToolset]
) -> Agent[AgentContext, OutputDataT]:
    @agent.instructions
    def _(ctx: RunContext[AgentContext]) -> str:
        return "\n\n".join(
            instructions for instructions in (toolset.get_instructions(ctx) for toolset in toolsets) if instructions
        )

    return agent
