"""Independent native tool execution policy capabilities."""

from __future__ import annotations

import asyncio
import math
import os
import time
from dataclasses import dataclass, field, replace
from typing import Any, Self

from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.capabilities import (
    AbstractCapability,
    CapabilityOrdering,
    ValidatedToolArgs,
    WrapToolExecuteHandler,
)
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, WrapperToolset

from ya_agent_sdk._logger import get_logger
from ya_agent_sdk.capabilities.foundation.deferred import SupportsDeferredOutput
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.toolsets.base import (
    ACTIVE_TOOL_SUPERSESSION_TAGS,
    TOOL_SUPERSEDED_BY_TAGS_METADATA_KEY,
    TOOL_TAGS_METADATA_KEY,
)

logger = get_logger(__name__)

_DEFAULT_TOOL_TIMEOUT_SECONDS = 600.0
_TOOL_TIMEOUT_ENV_VAR = "YA_AGENT_TOOL_TIMEOUT_SECONDS"


def _default_tool_timeout_seconds() -> float:
    value = os.getenv(_TOOL_TIMEOUT_ENV_VAR)
    if value is None or not value.strip():
        return _DEFAULT_TOOL_TIMEOUT_SECONDS
    try:
        timeout = float(value)
    except ValueError as exc:
        raise ValueError(f"{_TOOL_TIMEOUT_ENV_VAR} must be a positive finite number") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(f"{_TOOL_TIMEOUT_ENV_VAR} must be a positive finite number")
    return timeout


def _tool_metadata_tags(tool_def: ToolDefinition, key: str) -> frozenset[str]:
    """Read one SDK-owned string tag collection from a tool definition."""
    value = (tool_def.metadata or {}).get(key)
    if not isinstance(value, (list, tuple, set, frozenset)):
        return frozenset()
    return frozenset(tag for tag in value if isinstance(tag, str))


@dataclass
class _ToolSupersessionToolset(WrapperToolset[AgentContext]):
    """Resolve SDK tool supersession after all capability toolsets are assembled."""

    _active_tags: frozenset[str] | None = field(init=False, default=None, repr=False)

    async def for_run(self, ctx: RunContext[AgentContext]) -> Self:
        return type(self)(await self.wrapped.for_run(ctx))

    async def for_run_step(self, ctx: RunContext[AgentContext]) -> Self:
        return type(self)(await self.wrapped.for_run_step(ctx))

    async def get_tools(self, ctx: RunContext[AgentContext]):
        tools = await super().get_tools(ctx)
        active_tags: set[str] = set()
        for tool in tools.values():
            active_tags.update(_tool_metadata_tags(tool.tool_def, TOOL_TAGS_METADATA_KEY))
        self._active_tags = frozenset(active_tags)
        ctx.deps.tool_tags = active_tags
        return {
            name: tool
            for name, tool in tools.items()
            if not (_tool_metadata_tags(tool.tool_def, TOOL_SUPERSEDED_BY_TAGS_METADATA_KEY) & active_tags)
        }

    async def get_instructions(self, ctx: RunContext[AgentContext]):
        # Pydantic AI resolves tools before instructions for each run step. Reuse that
        # exact availability snapshot instead of resolving dynamic toolsets twice.
        token = ACTIVE_TOOL_SUPERSESSION_TAGS.set(self._active_tags or frozenset())
        try:
            return await super().get_instructions(ctx)
        finally:
            ACTIVE_TOOL_SUPERSESSION_TAGS.reset(token)


@dataclass(kw_only=True)
class ToolSupersessionCapability(AbstractCapability[AgentContext]):
    """Hide SDK tools superseded by tags from any assembled capability toolset."""

    id: str | None = "tool_supersession"

    def get_wrapper_toolset(
        self,
        toolset: AbstractToolset[AgentContext],
    ) -> AbstractToolset[AgentContext]:
        return _ToolSupersessionToolset(toolset)


@dataclass(kw_only=True)
class ToolVisibilityCapability(AbstractCapability[AgentContext]):
    """Filter model-visible tools and fail closed at the execution boundary."""

    allow: frozenset[str] | None = None
    deny: frozenset[str] = frozenset()
    id: str | None = "tool_visibility"

    def _allowed(self, ctx: RunContext[AgentContext], tool_def: ToolDefinition) -> bool:
        if self.allow is not None and tool_def.name not in self.allow:
            return False
        if tool_def.name in self.deny:
            return False
        metadata = tool_def.metadata or {}
        if metadata.get("main_agent_only") is True:
            return ctx.deps.agent_id == "main" and ctx.deps.parent_run_id is None
        return True

    async def prepare_tools(
        self,
        ctx: RunContext[AgentContext],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        return [tool_def for tool_def in tool_defs if self._allowed(ctx, tool_def)]

    async def before_tool_execute(
        self,
        ctx: RunContext[AgentContext],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
    ) -> ValidatedToolArgs:
        if not self._allowed(ctx, tool_def):
            raise ModelRetry(f"Tool {tool_def.name!r} is not available in this agent context")
        return args


@dataclass(kw_only=True)
class ToolApprovalCapability(SupportsDeferredOutput, AbstractCapability[AgentContext]):
    """Project host approval policy onto native tool definitions."""

    tools: frozenset[str] = frozenset()
    toolset_ids: frozenset[str] = frozenset()
    id: str | None = "tool_approval"

    async def prepare_tools(
        self,
        ctx: RunContext[AgentContext],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        configured_tools = self.tools or frozenset(ctx.deps.need_user_approve_tools)
        configured_toolsets = self.toolset_ids or frozenset(ctx.deps.need_user_approve_mcps)
        return [
            replace(tool_def, kind="unapproved")
            if tool_def.kind == "function"
            and (
                tool_def.name in configured_tools
                or (tool_def.toolset_id is not None and tool_def.toolset_id in configured_toolsets)
            )
            else tool_def
            for tool_def in tool_defs
        ]


@dataclass(kw_only=True)
class ToolTimeoutCapability(AbstractCapability[AgentContext]):
    """Bound one validated function-tool execution attempt."""

    timeout: float = field(default_factory=_default_tool_timeout_seconds)
    id: str | None = "tool_timeout"

    def __post_init__(self) -> None:
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be a positive finite number")

    async def wrap_tool_execute(
        self,
        ctx: RunContext[AgentContext],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        handler: WrapToolExecuteHandler,
    ) -> Any:
        timeout = min(tool_def.timeout, self.timeout) if tool_def.timeout is not None else self.timeout
        timeout_scope = asyncio.timeout(timeout)
        try:
            async with timeout_scope:
                return await handler(args)
        except TimeoutError as exc:
            if not timeout_scope.expired():
                raise
            raise ModelRetry(
                f"Tool {tool_def.name!r} exceeded the execution timeout of {timeout:g} seconds. "
                "The call was cancelled and may have produced partial side effects. Inspect current state "
                "before retrying, or continue with another approach."
            ) from exc


@dataclass(kw_only=True)
class ToolObservationCapability(AbstractCapability[AgentContext]):
    """Observe one logical validated call around all execution attempts."""

    id: str | None = "tool_observation"

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(wraps=(ToolTimeoutCapability,))

    async def wrap_tool_execute(
        self,
        ctx: RunContext[AgentContext],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        handler: WrapToolExecuteHandler,
    ) -> Any:
        started = time.perf_counter()
        try:
            return await handler(args)
        finally:
            logger.debug(
                "Tool execution completed tool=%s duration_seconds=%.6f",
                tool_def.name,
                time.perf_counter() - started,
            )
