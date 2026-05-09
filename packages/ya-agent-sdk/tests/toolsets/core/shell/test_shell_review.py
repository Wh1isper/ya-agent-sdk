from __future__ import annotations

from contextlib import AsyncExitStack
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic_ai import ApprovalRequired, RunContext
from ya_agent_sdk.context import AgentContext, ShellReviewConfig
from ya_agent_sdk.environment.local import LocalEnvironment
from ya_agent_sdk.toolsets.core.shell import ShellTool
from ya_agent_sdk.toolsets.core.shell import review as review_module
from ya_agent_sdk.toolsets.core.shell.review import ShellReviewDecision


async def _ctx(tmp_path: Path) -> AgentContext:
    stack = AsyncExitStack()
    env = await stack.enter_async_context(LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path))
    ctx = await stack.enter_async_context(AgentContext(env=env))
    ctx._test_stack = stack  # type: ignore[attr-defined]
    return ctx


async def test_shell_review_config_requires_model_when_enabled() -> None:
    with pytest.raises(ValueError, match="model is required"):
        ShellReviewConfig(enabled=True)


async def test_shell_review_denies_at_configured_threshold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = await _ctx(tmp_path)
    ctx.security.shell_review = ShellReviewConfig(enabled=True, model="test:model", on_needs_approval="deny")

    async def fake_review(*args, **kwargs):
        return ShellReviewDecision(action="needs_approval", risk_level="extra_high", reason="risky")

    monkeypatch.setattr(review_module, "review_shell_command", fake_review)
    import ya_agent_sdk.toolsets.core.shell.shell as shell_module

    monkeypatch.setattr(shell_module, "review_shell_command", fake_review)
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = ctx
    mock_run_ctx.tool_call_approved = False
    result = await ShellTool().call(mock_run_ctx, "rm -rf build")
    assert result["return_code"] == 1
    assert "blocked" in result["error"]
    assert result["shell_review"]["reason"] == "risky"


async def test_shell_review_allows_below_deny_threshold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = await _ctx(tmp_path)
    ctx.security.shell_review = ShellReviewConfig(enabled=True, model="test:model", on_needs_approval="deny")

    async def fake_review(*args, **kwargs):
        return ShellReviewDecision(action="needs_approval", risk_level="medium", reason="review only")

    import ya_agent_sdk.toolsets.core.shell.shell as shell_module

    monkeypatch.setattr(shell_module, "review_shell_command", fake_review)
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = ctx
    result = await ShellTool().call(mock_run_ctx, "echo safe")
    assert result["return_code"] == 0
    assert "safe" in result["stdout"]


async def test_shell_review_defers_when_configured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = await _ctx(tmp_path)
    ctx.security.shell_review = ShellReviewConfig(enabled=True, model="test:model", on_needs_approval="defer")

    async def fake_review(*args, **kwargs):
        return ShellReviewDecision(action="needs_approval", risk_level="high", reason="risky")

    import ya_agent_sdk.toolsets.core.shell.shell as shell_module

    monkeypatch.setattr(shell_module, "review_shell_command", fake_review)
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = ctx
    mock_run_ctx.tool_call_approved = False
    with pytest.raises(ApprovalRequired):
        await ShellTool().call(mock_run_ctx, "rm -rf build")


async def test_shell_review_executes_after_deferred_approval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = await _ctx(tmp_path)
    ctx.security.shell_review = ShellReviewConfig(enabled=True, model="test:model", on_needs_approval="defer")

    async def fake_review(*args, **kwargs):
        return ShellReviewDecision(action="needs_approval", risk_level="high", reason="approved risky command")

    import ya_agent_sdk.toolsets.core.shell.shell as shell_module

    monkeypatch.setattr(shell_module, "review_shell_command", fake_review)
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = ctx
    mock_run_ctx.tool_call_approved = True

    result = await ShellTool().call(mock_run_ctx, "echo approved")

    assert result["return_code"] == 0
    assert "approved" in result["stdout"]
