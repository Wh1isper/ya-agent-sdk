from __future__ import annotations

from collections import deque
from contextlib import AsyncExitStack
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic_ai import ApprovalRequired, RunContext
from ya_agent_sdk.context import AgentContext, ShellReviewConfig
from ya_agent_sdk.environment.local import LocalEnvironment
from ya_agent_sdk.toolsets.core.shell import ShellTool
from ya_agent_sdk.toolsets.core.shell.review import (
    ShellReviewContextSnapshot,
    ShellReviewDecision,
    ShellReviewPreviousDecision,
    ShellReviewRecord,
    ShellReviewRequest,
    get_previous_shell_reviews,
    review_shell_command,
)


async def _ctx(tmp_path: Path) -> AgentContext:
    stack = AsyncExitStack()
    env = await stack.enter_async_context(LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path))
    ctx = await stack.enter_async_context(AgentContext(env=env))
    ctx._test_stack = stack  # type: ignore[attr-defined]
    return ctx


async def test_shell_review_config_requires_model_when_enabled() -> None:
    with pytest.raises(ValueError, match="model is required"):
        ShellReviewConfig(enabled=True)


async def test_shell_review_records_usage(tmp_path: Path) -> None:
    ctx = await _ctx(tmp_path)
    ctx.security.shell_review = ShellReviewConfig(enabled=True, model="test")

    decision = await review_shell_command(ctx, request=ShellReviewRequest(command="echo hi"), usage_uuid="review-1")

    assert decision.risk_level == "low"
    assert len(ctx.extra_usages) == 1
    assert ctx.extra_usages[0].uuid == "review-1"
    assert ctx.extra_usages[0].agent == "shell_review"
    assert ctx.extra_usages[0].model_id == "test"


async def test_shell_review_request_builds_prompt_and_metadata() -> None:
    request = ShellReviewRequest(
        command="rm -rf build",
        cwd="/workspace",
        background=True,
        environment_keys=["PATH"],
    )
    decision = ShellReviewDecision(risk_level="high", reason="risky")

    prompt = request.to_prompt()
    assert "<command>\nrm -rf build\n</command>" in prompt
    assert "cwd: /workspace" in prompt
    assert "background: True" in prompt
    assert "environment_keys: ['PATH']" in prompt
    assert request.to_approval_metadata(decision) == {
        "reviewer": "shell_command_reviewer",
        "reason": "risky",
        "risk_level": "high",
        "command": "rm -rf build",
        "cwd": "/workspace",
        "background": True,
    }


async def test_shell_review_request_includes_context_and_previous_reviews() -> None:
    request = ShellReviewRequest(
        command="rm -rf build",
        cwd="/workspace",
        background=False,
        environment_keys=["PATH"],
        context_snapshot=ShellReviewContextSnapshot(
            timeout_seconds=30,
            tool_call_id="tool-1",
            default_cwd="/workspace",
            allowed_paths=["/workspace"],
            shell_platform="linux",
            shell_executable="/bin/bash",
        ),
        previous_reviews=[
            ShellReviewPreviousDecision(
                approved=True,
                risk_level="medium",
                reason="Deletes generated artifacts.",
                command="rm -rf build",
                cwd="/workspace",
            )
        ],
    )
    decision = ShellReviewDecision(risk_level="medium", reason="still risky")

    prompt = request.to_prompt()
    assert "<workspace_context>" in prompt
    assert "timeout_seconds: 30" in prompt
    assert "tool_call_id: tool-1" not in prompt
    assert "tool_call_approved: False" in prompt
    assert "allowed_paths: ['/workspace']" in prompt
    assert "<previous_shell_reviews>" in prompt
    assert "approved: True" in prompt
    assert "command: rm -rf build" in prompt
    assert "reason: Deletes generated artifacts." in prompt

    metadata = request.to_approval_metadata(decision)
    assert metadata["context"] == {
        "timeout_seconds": 30,
        "tool_call_id": "tool-1",
        "tool_call_approved": False,
        "default_cwd": "/workspace",
        "allowed_paths": ["/workspace"],
        "shell_platform": "linux",
        "shell_executable": "/bin/bash",
    }
    assert metadata["previous_shell_reviews"] == [request.previous_reviews[0].model_dump()]


async def test_shell_review_keeps_last_n_previous_reviews(tmp_path: Path) -> None:
    ctx = await _ctx(tmp_path)
    ctx.shell_review_records = deque(maxlen=5)
    for index in range(6):
        request = ShellReviewRequest(command=f"echo {index}")
        ctx.shell_review_records.append(
            ShellReviewRecord(
                request=request,
                decision=ShellReviewDecision(risk_level="low", reason=f"safe {index}"),
                tool_call_id=f"tool-{index}",
                approved=True,
            )
        )

    previous = get_previous_shell_reviews(ctx, ShellReviewRequest(command="echo current"))

    assert len(previous) == 5
    assert [item.command for item in previous] == ["echo 5", "echo 4", "echo 3", "echo 2", "echo 1"]
    assert all(item.approved for item in previous)


async def test_shell_review_prioritizes_same_tool_call_and_same_command(tmp_path: Path) -> None:
    ctx = await _ctx(tmp_path)
    request = ShellReviewRequest(command="rm -rf build", cwd="/workspace")
    ctx.shell_review_records.append(
        ShellReviewRecord(
            request=ShellReviewRequest(command="echo old"),
            decision=ShellReviewDecision(risk_level="low", reason="safe"),
            tool_call_id="old-tool",
            approved=True,
        )
    )
    ctx.shell_review_records.append(
        ShellReviewRecord(
            request=request,
            decision=ShellReviewDecision(risk_level="medium", reason="same command"),
            tool_call_id="same-command-tool",
        )
    )
    ctx.shell_review_records.append(
        ShellReviewRecord(
            request=ShellReviewRequest(command="echo replay"),
            decision=ShellReviewDecision(risk_level="low", reason="same tool"),
            tool_call_id="target-tool",
            approved=True,
        )
    )

    previous = get_previous_shell_reviews(ctx, request, tool_call_id="target-tool")

    assert previous[0].command == "echo replay"
    assert previous[0].approved is True
    assert previous[1].command == "rm -rf build"
    assert previous[1].approved is False


async def test_shell_tool_passes_previous_reviews_to_reviewer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = await _ctx(tmp_path)
    ctx.security.shell_review = ShellReviewConfig(enabled=True, model="test:model", on_needs_approval="deny")
    seen_requests: list[ShellReviewRequest] = []

    async def fake_review(*args, request: ShellReviewRequest, **kwargs):
        seen_requests.append(request)
        return ShellReviewDecision(risk_level="low", reason="safe")

    import ya_agent_sdk.toolsets.core.shell.shell as shell_module

    monkeypatch.setattr(shell_module, "review_shell_command", fake_review)
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = ctx
    mock_run_ctx.tool_call_id = "tool-1"
    mock_run_ctx.tool_call_approved = False

    await ShellTool().call(mock_run_ctx, "echo first")
    await ShellTool().call(mock_run_ctx, "echo second")

    assert seen_requests[0].previous_reviews == []
    assert len(seen_requests[1].previous_reviews) == 1
    assert seen_requests[1].previous_reviews[0].command == "echo first"
    assert seen_requests[1].previous_reviews[0].approved is True


async def test_shell_review_denies_at_configured_threshold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = await _ctx(tmp_path)
    ctx.security.shell_review = ShellReviewConfig(
        enabled=True,
        model="test:model",
        on_needs_approval="deny",
        risk_threshold="extra_high",
    )

    async def fake_review(*args, **kwargs):
        return ShellReviewDecision(risk_level="extra_high", reason="risky")

    import ya_agent_sdk.toolsets.core.shell.shell as shell_module

    monkeypatch.setattr(shell_module, "review_shell_command", fake_review)
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = ctx
    mock_run_ctx.tool_call_approved = False
    result = await ShellTool().call(mock_run_ctx, "rm -rf build")
    assert result["return_code"] == 1
    assert "blocked" in result["error"]
    assert result["shell_review"]["reason"] == "risky"


async def test_shell_review_allows_below_threshold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = await _ctx(tmp_path)
    ctx.security.shell_review = ShellReviewConfig(
        enabled=True,
        model="test:model",
        on_needs_approval="deny",
        risk_threshold="high",
    )

    async def fake_review(*args, **kwargs):
        return ShellReviewDecision(risk_level="medium", reason="review only")

    import ya_agent_sdk.toolsets.core.shell.shell as shell_module

    monkeypatch.setattr(shell_module, "review_shell_command", fake_review)
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = ctx
    result = await ShellTool().call(mock_run_ctx, "echo safe")
    assert result["return_code"] == 0
    assert "safe" in result["stdout"]
    assert ctx.shell_review_records[-1].approved is True


async def test_shell_review_defers_when_configured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = await _ctx(tmp_path)
    ctx.security.shell_review = ShellReviewConfig(
        enabled=True,
        model="test:model",
        on_needs_approval="defer",
        risk_threshold="extra_high",
    )

    async def fake_review(*args, **kwargs):
        return ShellReviewDecision(risk_level="extra_high", reason="risky")

    import ya_agent_sdk.toolsets.core.shell.shell as shell_module

    monkeypatch.setattr(shell_module, "review_shell_command", fake_review)
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = ctx
    mock_run_ctx.tool_call_id = "tool-1"
    mock_run_ctx.tool_call_approved = False
    with pytest.raises(ApprovalRequired):
        await ShellTool().call(mock_run_ctx, "rm -rf build")
    assert ctx.shell_review_records[-1].approved is False


async def test_shell_review_bypasses_reviewer_after_deferred_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = await _ctx(tmp_path)
    ctx.security.shell_review = ShellReviewConfig(enabled=True, model="test:model", on_needs_approval="defer")
    call_count = 0

    async def fake_review(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return ShellReviewDecision(risk_level="high", reason="approved risky command")

    import ya_agent_sdk.toolsets.core.shell.shell as shell_module

    monkeypatch.setattr(shell_module, "review_shell_command", fake_review)
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = ctx
    mock_run_ctx.tool_call_id = "tool-1"
    mock_run_ctx.tool_call_approved = False

    with pytest.raises(ApprovalRequired):
        await ShellTool().call(mock_run_ctx, "echo approved")

    mock_run_ctx.tool_call_approved = True
    result = await ShellTool().call(mock_run_ctx, "echo approved")

    assert result["return_code"] == 0
    assert "approved" in result["stdout"]
    assert call_count == 1
    assert ctx.shell_review_records[-1].approved is True
