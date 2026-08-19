"""Shell execution capability."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.toolsets import AbstractToolset

from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.filters.background_shell import consume_background_results
from ya_agent_sdk.inputs import InputOrigin
from ya_agent_sdk.toolsets.core.base import Toolset
from ya_agent_sdk.toolsets.core.shell import tools as shell_tools


@dataclass(kw_only=True)
class ShellCapability(AbstractCapability[AgentContext]):
    """Own shell tools and route process completion through native enqueue."""

    id: str | None = "shell"

    def get_ordering(self) -> CapabilityOrdering:
        from ..foundation.request import (
            EnvironmentContextCapability,
            RuntimeContextCapability,
        )

        return CapabilityOrdering(wraps=(EnvironmentContextCapability, RuntimeContextCapability))

    def get_toolset(self) -> AbstractToolset[AgentContext]:
        return Toolset(tools=shell_tools, toolset_id=self.id or "shell")

    async def before_model_request(
        self,
        ctx: RunContext[AgentContext],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        router = ctx.deps.input_router
        if router is None:
            return request_context
        completion = await consume_background_results(ctx.deps)
        if completion is not None:
            await router.enqueue(completion, origin=InputOrigin.feature)
        return request_context
