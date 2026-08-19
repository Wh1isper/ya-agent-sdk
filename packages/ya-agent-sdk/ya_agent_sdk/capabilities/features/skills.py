"""Skill catalog capability."""

from __future__ import annotations

from dataclasses import dataclass, replace

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import AbstractToolset

from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.toolsets.skills import SkillToolset


@dataclass(kw_only=True)
class SkillsCapability(AbstractCapability[AgentContext]):
    """Own one run-isolated skill catalog, tools, and routing guidance."""

    skills_dir_name: str = "skills"
    extra_dir_names: tuple[str, ...] = ()
    id: str | None = "skills"

    async def for_run(self, ctx: RunContext[AgentContext]) -> AbstractCapability[AgentContext]:
        return replace(self)

    def get_toolset(self) -> AbstractToolset[AgentContext]:
        return SkillToolset(
            self.skills_dir_name,
            extra_dir_names=list(self.extra_dir_names),
            toolset_id=self.id,
        )
