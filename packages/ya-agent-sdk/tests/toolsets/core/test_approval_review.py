from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic_ai import RunContext
from ya_agent_sdk.context import AgentContext, SecurityConfig
from ya_agent_sdk.security.approval import (
    ApprovalReviewConfig,
    ApprovalReviewOutcome,
    ApprovalReviewRequest,
    ApprovalReviewResult,
    ApprovalRiskLevel,
    PermissionDecision,
    ToolPermissionProfile,
    ToolResultTruncationConfig,
    UserAuthorizationLevel,
)
from ya_agent_sdk.toolsets.base import BaseTool
from ya_agent_sdk.toolsets.core.base import Toolset


class AllowTool(BaseTool):
    name = "allow_tool"
    description = "Allow tool"
    permission = ToolPermissionProfile.read_workspace()

    async def call(self, ctx: RunContext[AgentContext], value: str = "ok") -> str:
        return value


class DenyTool(BaseTool):
    name = "deny_tool"
    description = "Deny tool"
    permission = ToolPermissionProfile.write_workspace().with_decision(
        PermissionDecision.DENY,
        rationale="Configured deny.",
    )

    async def call(self, ctx: RunContext[AgentContext], value: str = "blocked") -> str:
        return value


class AutoReviewTool(BaseTool):
    name = "auto_review_tool"
    description = "Auto review tool"
    permission = ToolPermissionProfile.write_workspace()

    async def call(self, ctx: RunContext[AgentContext], value: str = "ok") -> str:
        return value


class MutatingHookTool(BaseTool):
    name = "mutating_hook_tool"
    description = "Mutating hook tool"
    permission = ToolPermissionProfile.read_workspace()

    async def call(self, ctx: RunContext[AgentContext], path: str) -> str:
        return path


class LargeOutputTool(BaseTool):
    name = "large_output_tool"
    description = "Large output tool"
    permission = ToolPermissionProfile.read_workspace()

    async def call(self, ctx: RunContext[AgentContext]) -> str:
        return "a" * 100


def _run_context(ctx: AgentContext) -> MagicMock:
    run_ctx = MagicMock(spec=RunContext)
    run_ctx.deps = ctx
    run_ctx.tool_call_approved = False
    run_ctx.tool_call_id = "tool-call-1"
    return run_ctx


async def _call_tool(tool_cls: type[BaseTool], ctx: AgentContext, args: dict[str, Any]) -> Any:
    run_ctx = _run_context(ctx)
    toolset = Toolset(tools=[tool_cls])
    tools = await toolset.get_tools(run_ctx)
    return await toolset.call_tool(tool_cls.name, args, run_ctx, tools[tool_cls.name])


async def test_approval_review_allow_decision_executes_without_reviewer(agent_context: AgentContext) -> None:
    result = await _call_tool(AllowTool, agent_context, {"value": "allowed"})

    assert result == "allowed"
    assert list(agent_context.approval_review_records) == []


async def test_approval_review_deny_decision_returns_denial(agent_context: AgentContext) -> None:
    result = await _call_tool(DenyTool, agent_context, {"value": "blocked"})

    assert "Tool call denied by approval review" in result
    assert "Configured deny." in result
    assert list(agent_context.approval_review_records) == []


async def test_approval_review_auto_review_denies_with_closed_fallback(agent_context: AgentContext) -> None:
    agent_context.security = SecurityConfig(
        approval_review=ApprovalReviewConfig(
            enabled=True,
            model="test:model",
            timeout_seconds=0.001,
        )
    )

    result = await _call_tool(AutoReviewTool, agent_context, {"value": "reviewed"})

    assert "Tool call denied by approval review" in result
    assert "closed-deny fallback" in result
    assert len(agent_context.approval_review_records) == 1
    record = agent_context.approval_review_records[0]
    assert record.request.tool_name == "auto_review_tool"
    assert record.result.outcome == ApprovalReviewOutcome.DENY


async def test_approval_review_uses_post_hook_arguments(
    monkeypatch: pytest.MonkeyPatch, agent_context: AgentContext
) -> None:
    captured_request: ApprovalReviewRequest | None = None

    async def fake_review(ctx: AgentContext, request: ApprovalReviewRequest, **_: Any) -> ApprovalReviewResult:
        nonlocal captured_request
        captured_request = request
        return ApprovalReviewResult(
            request_id=request.request_id,
            outcome=ApprovalReviewOutcome.ALLOW,
            risk_level=ApprovalRiskLevel.LOW,
            authorization=UserAuthorizationLevel.IMPLIED,
            rationale="Allowed.",
        )

    monkeypatch.setattr("ya_agent_sdk.toolsets.core.base.evaluate_approval_review", fake_review)
    agent_context.security = SecurityConfig(approval_review=ApprovalReviewConfig(enabled=True, model="test:model"))

    async def mutate_path(
        ctx: RunContext[AgentContext], args: dict[str, Any], metadata: dict[str, Any]
    ) -> dict[str, Any]:
        return {"path": ".env"}

    run_ctx = _run_context(agent_context)
    toolset = Toolset(tools=[MutatingHookTool], pre_hooks={"mutating_hook_tool": mutate_path})
    tools = await toolset.get_tools(run_ctx)

    result = await toolset.call_tool("mutating_hook_tool", {"path": "README.md"}, run_ctx, tools["mutating_hook_tool"])

    assert result == ".env"
    assert captured_request is not None
    assert captured_request.tool_args == {"path": ".env"}
    assert "credential" in {category.value for category in captured_request.permission.categories}


async def test_approval_review_truncates_tool_result(agent_context: AgentContext) -> None:
    agent_context.security = SecurityConfig(
        approval_review=ApprovalReviewConfig(
            enabled=False,
            truncation=ToolResultTruncationConfig(
                enabled=True,
                max_text_chars=20,
                head_chars=8,
                tail_chars=4,
            ),
        )
    )

    result = await _call_tool(LargeOutputTool, agent_context, {})

    assert result.startswith("a" * 8)
    assert result.endswith("a" * 4)
    assert "Tool output truncated" in result
