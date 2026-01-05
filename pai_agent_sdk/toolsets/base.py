from abc import ABC

from pydantic_ai import AbstractToolset, RunContext
from typing_extensions import TypeVar

from pai_agent_sdk.context import AgentContext

AgentDepsT = TypeVar("AgentDepsT", bound=AgentContext, default=AgentContext, contravariant=True)


class BaseToolset(AbstractToolset[AgentDepsT], ABC):
    def get_instructions(self, ctx: RunContext[AgentDepsT]) -> str | None:
        return None
