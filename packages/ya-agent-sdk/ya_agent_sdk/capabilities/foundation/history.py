"""Canonical history capabilities for the YA Agent runtime."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, ReinjectSystemPrompt
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.toolsets import AbstractToolset

from ya_agent_sdk.agents.compact import create_cache_friendly_compact_filter
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.filters.cold_start import cold_start_trim
from ya_agent_sdk.filters.handoff import process_handoff_message
from ya_agent_sdk.filters.reasoning_normalize import normalize_reasoning_for_model
from ya_agent_sdk.filters.tool_args import fix_truncated_tool_args

from ._request import HistoryProcessor, apply_history_processor


@dataclass(kw_only=True)
class ReasoningCompatibilityCapability(AbstractCapability[AgentContext]):
    """Normalize provider reasoning parts without mutating caller-owned history."""

    id: str | None = "reasoning_compatibility"

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(
            wraps=(
                HandoffCapability,
                ContextCompactionCapability,
                ColdStartCapability,
                ReinjectSystemPrompt,
            )
        )

    async def before_model_request(
        self,
        ctx: RunContext[AgentContext],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        return await apply_history_processor(normalize_reasoning_for_model, ctx, request_context)


@dataclass(kw_only=True)
class ToolArgumentRepairCapability(AbstractCapability[AgentContext]):
    """Repair recoverable truncated function-call arguments before reduction."""

    id: str | None = "tool_argument_repair"

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(
            wraps=(
                HandoffCapability,
                ContextCompactionCapability,
                ColdStartCapability,
                ReinjectSystemPrompt,
            )
        )

    async def before_model_request(
        self,
        ctx: RunContext[AgentContext],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        return await apply_history_processor(fix_truncated_tool_args, ctx, request_context)


@dataclass(kw_only=True)
class ToolIdCompatibilityCapability(AbstractCapability[AgentContext]):
    """Normalize tool call/result identities on immutable message replacements."""

    id: str | None = "tool_id_compatibility"

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(
            wraps=(
                HandoffCapability,
                ContextCompactionCapability,
                ColdStartCapability,
                ReinjectSystemPrompt,
            )
        )

    async def before_model_request(
        self,
        ctx: RunContext[AgentContext],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        return await apply_history_processor(ctx.deps.tool_id_wrapper.wrap_messages, ctx, request_context)


@dataclass(kw_only=True)
class HandoffCapability(AbstractCapability[AgentContext]):
    """Apply an explicit summarized-context handoff transition."""

    id: str | None = "handoff"

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(wraps=(ContextCompactionCapability, ColdStartCapability, ReinjectSystemPrompt))

    def get_toolset(self) -> AbstractToolset[AgentContext]:
        from ya_agent_sdk.toolsets.core.base import Toolset
        from ya_agent_sdk.toolsets.core.context import tools

        return Toolset(tools=tools, toolset_id=self.id or "handoff")

    async def before_model_request(
        self,
        ctx: RunContext[AgentContext],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        return await apply_history_processor(process_handoff_message, ctx, request_context)


@dataclass(kw_only=True)
class ContextCompactionCapability(AbstractCapability[AgentContext]):
    """Compact oversized canonical history with the active agent and model."""

    id: str | None = "context_compaction"
    _processor: HistoryProcessor[AgentContext] = field(
        default_factory=create_cache_friendly_compact_filter,
        init=False,
        repr=False,
        compare=False,
    )

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(wraps=(ColdStartCapability, ReinjectSystemPrompt))

    async def before_model_request(
        self,
        ctx: RunContext[AgentContext],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        return await apply_history_processor(self._processor, ctx, request_context)


@dataclass(kw_only=True)
class ColdStartCapability(AbstractCapability[AgentContext]):
    """Trim stale large tool returns when the provider cache has expired."""

    id: str | None = "cold_start"

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(wraps=(ReinjectSystemPrompt,))

    async def before_model_request(
        self,
        ctx: RunContext[AgentContext],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        return await apply_history_processor(cold_start_trim, ctx, request_context)
