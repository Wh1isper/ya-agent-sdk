"""Independent native tool execution policy capabilities."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, replace
from typing import Any

from pydantic_ai import ApprovalRequired, CallDeferred, ModelRetry, RunContext
from pydantic_ai.capabilities import (
    AbstractCapability,
    CapabilityOrdering,
    ValidatedToolArgs,
    WrapToolExecuteHandler,
)
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import ToolDefinition

from ya_agent_sdk._logger import get_logger
from ya_agent_sdk.context import AgentContext

logger = get_logger(__name__)


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
class ToolApprovalCapability(AbstractCapability[AgentContext]):
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

    timeout: float = 120.0
    id: str | None = "tool_timeout"

    def __post_init__(self) -> None:
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")

    async def wrap_tool_execute(
        self,
        ctx: RunContext[AgentContext],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        handler: WrapToolExecuteHandler,
    ) -> Any:
        timeout = tool_def.timeout or self.timeout
        async with asyncio.timeout(timeout):
            return await handler(args)


@dataclass(kw_only=True)
class ToolRetryCapability(AbstractCapability[AgentContext]):
    """Retry transient non-native execution failures without swallowing control flow."""

    max_attempts: int = 1
    id: str | None = "tool_retry"

    def __post_init__(self) -> None:
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")

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
        for attempt in range(1, self.max_attempts + 1):
            try:
                return await handler(args)
            except (ApprovalRequired, CallDeferred, ModelRetry):
                raise
            except (OSError, TimeoutError):
                if attempt >= self.max_attempts:
                    raise
                logger.warning(
                    "Retrying transient tool execution tool=%s attempt=%d/%d",
                    tool_def.name,
                    attempt + 1,
                    self.max_attempts,
                )
        raise AssertionError("unreachable")


@dataclass(kw_only=True)
class ToolObservationCapability(AbstractCapability[AgentContext]):
    """Observe one logical validated call around all execution attempts."""

    id: str | None = "tool_observation"

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(wraps=(ToolRetryCapability, ToolTimeoutCapability))

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
