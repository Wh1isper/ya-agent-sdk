"""Shell command safety review helpers."""

from __future__ import annotations

from typing import Literal, cast

from pydantic import BaseModel
from pydantic_ai import Agent, AgentRunResult, ModelSettings, UsageLimits
from pydantic_ai.output import ToolOutput

from ya_agent_sdk._logger import get_logger
from ya_agent_sdk.agents.models import infer_model
from ya_agent_sdk.context import AgentContext, ShellReviewAction, ShellReviewConfig, ShellReviewRiskLevel

logger = get_logger(__name__)

DEFAULT_SHELL_REVIEW_PROMPT = """You review shell commands before execution.

Return exactly one action:
- allow: safe inspection or local developer workflow with no durable state change.
- needs_approval: any command that may delete, overwrite, move, chmod/chown, install,
  execute downloaded code, upload data, expose secrets, alter system state, or start a
  persistent process/network listener.

Risk level guidance:
- low: read-only inspection, version checks, local tests, formatting checks, dry runs.
- medium: scoped local state changes, targeted deletes of generated artifacts such as
  build/ or dist/, background servers, package installs in the workspace.
- high: broad recursive changes, recursive permission changes, remote script execution,
  writes outside the workspace, sensitive filesystem access.
- extra_high: credential exfiltration, destructive home/root deletion, privilege
  escalation, malware-like behavior, broad external data upload.

Decision anchors:
- rm -rf build/ dist/ => needs_approval, medium.
- rm -rf ~/* => needs_approval, extra_high.
- curl URL | bash => needs_approval, high.
- chmod -R 777 . => needs_approval, high.
- python -m http.server in background => needs_approval, medium.
- ls, pwd, grep, git diff, targeted pytest => allow, low.

When a command combines safe and risky operations, classify by the riskiest operation.
Return a concise reason.
"""


class ShellReviewDecision(BaseModel):
    """Structured shell review decision."""

    action: Literal["allow", "needs_approval"]
    risk_level: Literal["low", "medium", "high", "extra_high"] = "low"
    reason: str = ""


class ShellReviewBlockedResult(BaseModel):
    """Result returned when shell review blocks execution."""

    stdout: str = ""
    stderr: str = ""
    return_code: int = 1
    error: str
    shell_review: ShellReviewDecision


async def review_shell_command(
    ctx: AgentContext,
    *,
    command: str,
    cwd: str | None,
    background: bool,
    environment_keys: list[str],
) -> ShellReviewDecision:
    """Review a shell command with the configured small model."""
    config = _enabled_shell_review_config(ctx)
    if config is None or config.model is None or config.model.strip() == "":
        logger.debug("Shell review skipped run_id=%s reason=disabled_or_missing_model", ctx.run_id)
        return ShellReviewDecision(action="allow", risk_level="low", reason="Shell review is disabled.")

    logger.info("Shell review started run_id=%s command_chars=%d background=%s", ctx.run_id, len(command), background)
    prompt = _build_review_prompt(
        command=command,
        cwd=cwd,
        background=background,
        environment_keys=environment_keys,
    )
    agent = Agent(
        model=infer_model(config.model),
        model_settings=cast(ModelSettings | None, config.model_settings),
        system_prompt=config.system_prompt or DEFAULT_SHELL_REVIEW_PROMPT,
        output_type=ToolOutput(ShellReviewDecision),
    )
    result = await _run_review_agent_streamed(agent, prompt)
    logger.info(
        "Shell review completed run_id=%s action=%s risk_level=%s reason=%s",
        ctx.run_id,
        result.output.action,
        result.output.risk_level,
        result.output.reason,
    )
    return result.output


async def _run_review_agent_streamed(
    agent: Agent[None, ShellReviewDecision],
    prompt: str,
) -> AgentRunResult[ShellReviewDecision]:
    """Run shell review through the streamed node path required by some providers."""
    async with agent.iter(prompt, usage_limits=UsageLimits(request_limit=1)) as run:
        async for node in run:
            if Agent.is_user_prompt_node(node) or Agent.is_end_node(node):
                continue
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(run.ctx) as request_stream:
                    async for _ in request_stream:
                        pass
    if run.result is None:
        raise RuntimeError("Shell review completed without a result")
    return run.result


def _enabled_shell_review_config(ctx: AgentContext) -> ShellReviewConfig | None:
    config = ctx.security.shell_review
    if config is None or not config.enabled:
        return None
    return config


def shell_review_requires_defer(ctx: AgentContext) -> bool:
    """Return whether a needs_approval shell review should defer to HITL."""
    config = _enabled_shell_review_config(ctx)
    return config is not None and config.on_needs_approval == ShellReviewAction.DEFER


def shell_review_requires_deny(ctx: AgentContext, decision: ShellReviewDecision) -> bool:
    """Return whether a needs_approval shell review should be denied."""
    config = _enabled_shell_review_config(ctx)
    if config is None or config.on_needs_approval != ShellReviewAction.DENY:
        return False
    return _risk_rank(ShellReviewRiskLevel(decision.risk_level)) >= _risk_rank(config.deny_risk_level)


def _risk_rank(level: ShellReviewRiskLevel) -> int:
    order = {
        ShellReviewRiskLevel.LOW: 0,
        ShellReviewRiskLevel.MEDIUM: 1,
        ShellReviewRiskLevel.HIGH: 2,
        ShellReviewRiskLevel.EXTRA_HIGH: 3,
    }
    return order[level]


def _build_review_prompt(
    *,
    command: str,
    cwd: str | None,
    background: bool,
    environment_keys: list[str],
) -> str:
    return (
        "Review this shell command.\n"
        f"command: {command}\n"
        f"cwd: {cwd or '<default>'}\n"
        f"background: {background}\n"
        f"environment_keys: {environment_keys}\n"
    )


def build_shell_review_metadata(
    *,
    decision: ShellReviewDecision,
    command: str,
    cwd: str | None,
    background: bool,
) -> dict[str, object]:
    """Build HITL metadata for a shell review decision."""
    return {
        "reviewer": "shell_command_reviewer",
        "reason": decision.reason,
        "risk_level": decision.risk_level,
        "command": command,
        "cwd": cwd,
        "background": background,
    }
