"""Run-wide model correction retry budget."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from pydantic_ai import RunContext, UnexpectedModelBehavior
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelRequest, RetryPromptPart
from pydantic_ai.models import ModelRequestContext


@dataclass(kw_only=True)
class OverallRetryBudget(AbstractCapability[Any]):
    """Limit model correction retries cumulatively within one agent run.

    Pydantic AI tracks tool retries per tool name and clears a tool's counter
    after that tool succeeds. This capability adds a run-wide ceiling that does
    not reset after successful tool calls. It counts retry prompts at the model
    request boundary, which covers tool validation and execution retries,
    structured output retries, and retries raised by capability hooks without
    including transport-level model request retries or enqueued steering.
    """

    max_retries: int
    retries_used: int = 0
    id: str | None = "overall_retry_budget"

    async def for_run(self, ctx: RunContext[Any]) -> AbstractCapability[Any]:
        """Return a fresh counter for each agent run."""
        return replace(self, retries_used=0)

    async def before_model_request(
        self,
        ctx: RunContext[Any],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Count correction prompts in the request that is about to reach the model."""
        if not request_context.messages:
            return request_context

        request = request_context.messages[-1]
        if not isinstance(request, ModelRequest):
            return request_context

        if request.run_id != ctx.run_id:
            return request_context

        retry_count = sum(isinstance(part, RetryPromptPart) for part in request.parts)
        if retry_count == 0:
            return request_context

        self.retries_used += retry_count
        if self.retries_used > self.max_retries:
            raise UnexpectedModelBehavior(f"Exceeded the run-wide model correction retry limit of {self.max_retries}")
        return request_context
