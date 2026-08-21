"""Run-wide model-correction retry capability."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from pydantic_ai import RunContext, UnexpectedModelBehavior
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, ReinjectSystemPrompt
from pydantic_ai.messages import ModelRequest, RetryPromptPart
from pydantic_ai.models import ModelRequestContext

from .history import ColdStartCapability, ContextCompactionCapability, HandoffCapability


@dataclass(kw_only=True)
class OverallRetryBudget(AbstractCapability[Any]):
    """Limit model-correction retries cumulatively within one native run."""

    max_retries: int = 5
    retries_used: int = 0
    id: str | None = "overall_retry_budget"

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(
            wraps=(
                HandoffCapability,
                ContextCompactionCapability,
                ColdStartCapability,
                ReinjectSystemPrompt,
            )
        )

    async def for_run(self, ctx: RunContext[Any]) -> AbstractCapability[Any]:
        return replace(self, retries_used=0)

    async def before_model_request(
        self,
        ctx: RunContext[Any],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        if not request_context.messages:
            return request_context
        request = request_context.messages[-1]
        if not isinstance(request, ModelRequest) or request.run_id != ctx.run_id:
            return request_context
        retry_count = sum(isinstance(part, RetryPromptPart) for part in request.parts)
        if retry_count == 0:
            return request_context
        self.retries_used += retry_count
        if self.retries_used > self.max_retries:
            raise UnexpectedModelBehavior(f"Exceeded the run-wide model correction retry limit of {self.max_retries}")
        return request_context
