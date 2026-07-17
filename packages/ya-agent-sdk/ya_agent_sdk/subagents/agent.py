"""Pydantic AI Agent variant with an enforced subagent tool boundary."""

from __future__ import annotations

import inspect
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.toolsets._dynamic import DynamicToolset

from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.toolsets.core.base import Toolset


def _sanitize_dynamic_toolset(
    toolset: DynamicToolset[AgentContext],
) -> DynamicToolset[AgentContext]:
    """Apply the subagent policy whenever a dynamic factory materializes."""
    toolset_func = toolset.toolset_func
    last_resolved: AbstractToolset[AgentContext] | None = None
    last_sanitized: AbstractToolset[AgentContext] | None = None

    async def sanitized_toolset_func(
        ctx: RunContext[AgentContext],
    ) -> AbstractToolset[AgentContext] | None:
        nonlocal last_resolved, last_sanitized

        resolved = toolset_func(ctx)
        if inspect.isawaitable(resolved):
            resolved = await resolved
        if resolved is None:
            return None

        if resolved is last_resolved:
            return last_sanitized

        sanitized = _filter_subagent_toolset_tree(resolved)
        last_resolved = resolved
        last_sanitized = sanitized
        return sanitized

    return DynamicToolset(
        sanitized_toolset_func,
        per_run_step=toolset.per_run_step,
        id=toolset.id,
    )


def _filter_subagent_toolset_tree(toolset: AbstractToolset[AgentContext]) -> AbstractToolset[AgentContext]:
    """Remove main-agent-only tools, including from future dynamic results."""

    def replace(candidate: AbstractToolset[AgentContext]) -> AbstractToolset[AgentContext]:
        if isinstance(candidate, Toolset):
            return candidate.for_subagent()
        if isinstance(candidate, DynamicToolset):
            return _sanitize_dynamic_toolset(candidate)
        return candidate

    if isinstance(toolset, DynamicToolset):
        return _sanitize_dynamic_toolset(toolset)
    return toolset.visit_and_replace(replace)


class SubagentAgent(Agent[AgentContext, str]):
    """Enforce tool policy after all capability toolset wrappers are applied."""

    def _get_toolset(self, *args: Any, **kwargs: Any) -> AbstractToolset[AgentContext]:
        """Sanitize the final assembled toolset at the subagent execution boundary."""
        return _filter_subagent_toolset_tree(super()._get_toolset(*args, **kwargs))
