"""Capability-owned fixed-surface tool proxy."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import ClassVar

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import AbstractToolset

from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.toolsets.tool_proxy import ToolProxyToolset
from ya_agent_sdk.toolsets.tool_search.strategies import SearchStrategy


@dataclass(kw_only=True)
class ToolProxyCapability(AbstractCapability[AgentContext]):
    """Own a fixed discovery/call surface over explicitly granted toolsets."""

    toolsets: tuple[AbstractToolset[AgentContext], ...]
    namespace_descriptions: dict[str, str] = field(default_factory=dict)
    search_strategy: SearchStrategy | None = None
    max_results: int = 5
    optional_namespaces: frozenset[str] = frozenset()
    prefix: str | None = None
    max_retries: int | None = None
    id: str | None = "tool_proxy"

    _safe_at_runtime: ClassVar[bool] = False

    @classmethod
    def get_serialization_name(cls) -> None:
        """Runtime toolset objects cannot be represented in portable specs."""
        return None

    async def for_run(
        self,
        ctx: RunContext[AgentContext],
    ) -> AbstractCapability[AgentContext]:
        del ctx
        return replace(self)

    def get_toolset(self) -> AbstractToolset[AgentContext]:
        return ToolProxyToolset(
            toolsets=self.toolsets,
            namespace_descriptions=dict(self.namespace_descriptions),
            search_strategy=self.search_strategy,
            max_results=self.max_results,
            optional_namespaces=set(self.optional_namespaces),
            prefix=self.prefix,
            max_retries=self.max_retries,
        )
