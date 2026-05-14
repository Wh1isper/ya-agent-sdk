"""Approval review helpers for toolset execution."""

from __future__ import annotations

from typing import Any

from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.events import ApprovalReviewCompletedEvent, ApprovalReviewDeniedEvent, ApprovalReviewRequestedEvent
from ya_agent_sdk.security.approval import (
    ApprovalReviewRequest,
    ApprovalReviewResult,
    permission_summary,
)


def _event_metadata(request: ApprovalReviewRequest) -> dict[str, Any]:
    metadata = dict(request.metadata)
    summary = permission_summary(request.permission)
    if summary.get("metadata"):
        metadata.setdefault("permission", summary.get("metadata"))
    if request.mcp_server:
        metadata["mcp_server"] = request.mcp_server
    if request.mcp_tool:
        metadata["mcp_tool"] = request.mcp_tool
    return metadata


async def emit_approval_requested(ctx: AgentContext, request: ApprovalReviewRequest, decision: str) -> None:
    await ctx.emit_event(
        ApprovalReviewRequestedEvent(
            event_id=f"approval-review-requested-{request.request_id}",
            request_id=request.request_id,
            tool_call_id=request.tool_call_id,
            tool_name=request.tool_name,
            source=request.source.value,
            categories=sorted(category.value for category in request.permission.categories),
            scopes=sorted(scope.value for scope in request.permission.scopes),
            decision=decision,
            metadata=_event_metadata(request),
        )
    )


async def emit_approval_completed(
    ctx: AgentContext,
    request: ApprovalReviewRequest,
    result: ApprovalReviewResult,
    decision: str,
) -> None:
    await ctx.emit_event(
        ApprovalReviewCompletedEvent(
            event_id=f"approval-review-completed-{request.request_id}",
            request_id=request.request_id,
            tool_call_id=request.tool_call_id,
            tool_name=request.tool_name,
            source=request.source.value,
            categories=sorted(category.value for category in request.permission.categories),
            scopes=sorted(scope.value for scope in request.permission.scopes),
            decision=decision,
            outcome=result.outcome.value,
            risk_level=result.risk_level.value,
            authorization=result.authorization.value,
            rationale=result.rationale,
            metadata=_event_metadata(request),
        )
    )


async def emit_approval_denied(
    ctx: AgentContext,
    request: ApprovalReviewRequest,
    result: ApprovalReviewResult,
    decision: str,
) -> None:
    await ctx.emit_event(
        ApprovalReviewDeniedEvent(
            event_id=f"approval-review-denied-{request.request_id}",
            request_id=request.request_id,
            tool_call_id=request.tool_call_id,
            tool_name=request.tool_name,
            source=request.source.value,
            categories=sorted(category.value for category in request.permission.categories),
            scopes=sorted(scope.value for scope in request.permission.scopes),
            decision=decision,
            outcome=result.outcome.value,
            risk_level=result.risk_level.value,
            authorization=result.authorization.value,
            rationale=result.rationale,
            metadata=_event_metadata(request),
        )
    )
