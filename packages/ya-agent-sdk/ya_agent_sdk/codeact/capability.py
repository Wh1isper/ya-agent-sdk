"""Pydantic AI capability entrypoint for CodeAct."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import AbstractToolset

from ya_agent_sdk.codeact.config import CodeActConfig
from ya_agent_sdk.codeact.toolset import CodeActToolset
from ya_agent_sdk.context import AgentContext


@dataclass(kw_only=True)
class CodeActCapability(AbstractCapability[AgentContext]):
    """Install a run-local CodeAct wrapper around an agent's final toolset."""

    config: CodeActConfig = field(default_factory=CodeActConfig)
    id: str | None = "codeact"

    async def for_run(self, ctx: RunContext[AgentContext]) -> AbstractCapability[AgentContext]:
        return replace(self)

    def get_wrapper_toolset(
        self,
        toolset: AbstractToolset[AgentContext],
    ) -> AbstractToolset[AgentContext]:
        return CodeActToolset(wrapped=toolset, config=self.config)
