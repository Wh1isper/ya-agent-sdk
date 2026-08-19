"""Typed host resolution for Pydantic AI deferred tool interactions."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import DeferredToolRequests, DeferredToolResults
from pydantic_ai.tools import ToolApproved, ToolDenied


class DeferredCallResolution(BaseModel):
    """Externally completed result for a ``CallDeferred`` request."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["call"] = "call"
    tool_call_id: str
    result: Any
    metadata: dict[str, Any] = Field(default_factory=dict)


class DeferredApprovalResolution(BaseModel):
    """Typed approval decision for an ``ApprovalRequired`` request."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["approval"] = "approval"
    tool_call_id: str
    approved: bool
    override_args: dict[str, Any] | None = None
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


DeferredInteractionResolution = Annotated[
    DeferredCallResolution | DeferredApprovalResolution,
    Field(discriminator="kind"),
]


class DeferredInteractionResolver:
    """Validate host decisions against one exact deferred request set."""

    def resolve(
        self,
        requests: DeferredToolRequests,
        resolutions: list[DeferredInteractionResolution],
        *,
        require_complete: bool = True,
    ) -> DeferredToolResults:
        call_ids = {part.tool_call_id for part in requests.calls}
        approval_ids = {part.tool_call_id for part in requests.approvals}
        expected_ids = call_ids | approval_ids
        seen: set[str] = set()
        results = DeferredToolResults()

        for resolution in resolutions:
            tool_call_id = resolution.tool_call_id
            if tool_call_id in seen:
                raise ValueError(f"Deferred tool call {tool_call_id!r} was resolved more than once")
            seen.add(tool_call_id)
            _apply_resolution(
                results,
                resolution,
                call_ids=call_ids,
                approval_ids=approval_ids,
            )

        if require_complete and seen != expected_ids:
            missing = sorted(expected_ids - seen)
            unexpected = sorted(seen - expected_ids)
            details: list[str] = []
            if missing:
                details.append(f"missing={missing!r}")
            if unexpected:
                details.append(f"unexpected={unexpected!r}")
            raise ValueError("Deferred interaction resolutions do not match requests: " + ", ".join(details))
        return results


def _apply_resolution(
    results: DeferredToolResults,
    resolution: DeferredCallResolution | DeferredApprovalResolution,
    *,
    call_ids: set[str],
    approval_ids: set[str],
) -> None:
    tool_call_id = resolution.tool_call_id
    if isinstance(resolution, DeferredCallResolution):
        if tool_call_id not in call_ids:
            raise ValueError(f"Deferred tool call {tool_call_id!r} is not an external call")
        results.calls[tool_call_id] = resolution.result
    else:
        if tool_call_id not in approval_ids:
            raise ValueError(f"Deferred tool call {tool_call_id!r} is not an approval request")
        if resolution.approved:
            results.approvals[tool_call_id] = ToolApproved(override_args=resolution.override_args)
        else:
            results.approvals[tool_call_id] = ToolDenied(message=resolution.reason or "User rejected the tool call.")
    if resolution.metadata:
        results.metadata[tool_call_id] = dict(resolution.metadata)
