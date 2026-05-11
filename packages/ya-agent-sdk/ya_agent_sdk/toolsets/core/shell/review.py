"""Shell command safety review helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal, cast
from uuid import uuid4

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelSettings, UsageLimits
from pydantic_ai.models import Model
from pydantic_ai.output import ToolOutput

from ya_agent_sdk._logger import get_logger
from ya_agent_sdk.agents.models import infer_model
from ya_agent_sdk.context import AgentContext, ShellReviewAction, ShellReviewRiskLevel

logger = get_logger(__name__)

DEFAULT_SHELL_REVIEW_PROMPT = """You review shell commands before execution.

Return a risk_level and concise reason. Commands below the configured risk threshold
execute directly. Commands at or above the threshold enter the configured approval or deny policy.

Risk heuristics:
- low: read-only inspection or local developer verification such as tests, lint, type checks,
  imports, and printing.
- medium: bounded workspace-local state changes such as file writes/deletes, generated
  artifact or cache cleanup/generation, chmod/chown, package changes, and local servers.
- high: untrusted remote code execution, broad destructive workspace changes, writes outside
  the workspace, sensitive file reads, sudo usage, or system-level package/service changes.
- extra_high: confirmed credential exfiltration, destructive home/root deletion, explicit
  privilege escalation, malware-like behavior, or broad external data upload.

Reserve extra_high for visible catastrophic or hostile intent. Remote script execution is high by default.
Classify Python and uv commands by the script's visible effect; `uv run python` alone is a runner. Python compileall writes __pycache__/bytecode and is medium risk. Explicit cache/bytecode generation and outbound network access need approval.
Use workspace context for path scope. Source/package-tree wildcard deletion is broad destructive workspace change.
Use previous shell reviews as consistency hints. When an identical or equivalent command was previously approved and the current visible effect has not expanded, lower the risk by at least one level. Repeated approved commands that remain bounded and workspace-local can be low risk.
When a command combines safe and risky operations, classify by the riskiest operation after applying relevant previous approval context.
Return a concise reason.
"""

RISK_RANK: dict[ShellReviewRiskLevel, int] = {
    ShellReviewRiskLevel.LOW: 0,
    ShellReviewRiskLevel.MEDIUM: 1,
    ShellReviewRiskLevel.HIGH: 2,
    ShellReviewRiskLevel.EXTRA_HIGH: 3,
}


class ShellReviewDecision(BaseModel):
    """Structured shell review decision."""

    risk_level: Literal["low", "medium", "high", "extra_high"] = "low"
    reason: str = ""

    def requires_approval(self, ctx: AgentContext) -> bool:
        """Return whether this decision should enter the configured approval policy."""
        config = ctx.security.shell_review
        return (
            config is not None
            and config.enabled
            and RISK_RANK[ShellReviewRiskLevel(self.risk_level)] >= RISK_RANK[config.risk_threshold]
        )

    def requires_defer(self, ctx: AgentContext) -> bool:
        """Return whether this decision should defer to HITL."""
        config = ctx.security.shell_review
        return (
            self.requires_approval(ctx) and config is not None and config.on_needs_approval == ShellReviewAction.DEFER
        )

    def requires_deny(self, ctx: AgentContext) -> bool:
        """Return whether this decision should block execution."""
        config = ctx.security.shell_review
        if not self.requires_approval(ctx) or config is None:
            return False
        return config.on_needs_approval == ShellReviewAction.DENY


class ShellReviewPreviousDecision(BaseModel):
    """Previous shell review decision used as reviewer context."""

    approved: bool = False
    risk_level: Literal["low", "medium", "high", "extra_high"] = "low"
    reason: str = ""
    command: str | None = None
    cwd: str | None = None


class ShellReviewContextSnapshot(BaseModel):
    """Execution context submitted to the safety reviewer."""

    timeout_seconds: int | None = None
    tool_call_id: str | None = None
    tool_call_approved: bool = False
    default_cwd: str | None = None
    allowed_paths: list[str] = Field(default_factory=list)
    shell_platform: str | None = None
    shell_executable: str | None = None

    def to_metadata(self) -> dict[str, object | None]:
        """Build approval-safe context metadata."""
        return {
            "timeout_seconds": self.timeout_seconds,
            "tool_call_id": self.tool_call_id,
            "tool_call_approved": self.tool_call_approved,
            "default_cwd": self.default_cwd,
            "allowed_paths": self.allowed_paths,
            "shell_platform": self.shell_platform,
            "shell_executable": self.shell_executable,
        }


class ShellReviewRequest(BaseModel):
    """Shell command context submitted to the safety reviewer."""

    command: str
    cwd: str | None = None
    background: bool = False
    environment_keys: list[str] = Field(default_factory=list)
    context_snapshot: ShellReviewContextSnapshot | None = None
    previous_reviews: list[ShellReviewPreviousDecision] = Field(default_factory=list)

    def command_fingerprint(self) -> tuple[str, str | None, bool, tuple[str, ...]]:
        """Return a stable fingerprint for matching repeated shell reviews."""
        return (self.command, self.cwd, self.background, tuple(sorted(self.environment_keys)))

    def to_prompt(self) -> str:
        """Render the model prompt for this review request."""
        lines = [
            "Review this shell command.",
            "",
            "<command>",
            self.command,
            "</command>",
            "",
            "<execution_context>",
            f"cwd: {self.cwd or '<default>'}",
            f"background: {self.background}",
            f"environment_keys: {self.environment_keys}",
        ]
        if self.context_snapshot is not None:
            lines.extend([
                f"timeout_seconds: {self.context_snapshot.timeout_seconds}",
                f"tool_call_approved: {self.context_snapshot.tool_call_approved}",
            ])
        lines.extend(["</execution_context>", ""])

        if self.context_snapshot is not None:
            lines.extend([
                "<workspace_context>",
                f"default_cwd: {self.context_snapshot.default_cwd or '<unknown>'}",
                f"allowed_paths: {self.context_snapshot.allowed_paths}",
                f"shell_platform: {self.context_snapshot.shell_platform or '<unknown>'}",
                f"shell_executable: {self.context_snapshot.shell_executable or '<unknown>'}",
                "</workspace_context>",
                "",
            ])

        if self.previous_reviews:
            lines.append("<previous_shell_reviews>")
            for index, review in enumerate(self.previous_reviews, start=1):
                lines.extend([
                    f"review_{index}:",
                    f"  approved: {review.approved}",
                    f"  risk_level: {review.risk_level}",
                    f"  reason: {review.reason}",
                    f"  command: {review.command or '<unknown>'}",
                    f"  cwd: {review.cwd or '<default>'}",
                ])
            lines.extend(["</previous_shell_reviews>", ""])

        return "\n".join(lines)

    def to_approval_metadata(self, decision: ShellReviewDecision) -> dict[str, object]:
        """Build HITL metadata for this review request and decision."""
        metadata: dict[str, object] = {
            "reviewer": "shell_command_reviewer",
            "reason": decision.reason,
            "risk_level": decision.risk_level,
            "command": self.command,
            "cwd": self.cwd,
            "background": self.background,
        }
        if self.context_snapshot is not None:
            metadata["context"] = self.context_snapshot.to_metadata()
        if self.previous_reviews:
            metadata["previous_shell_reviews"] = [review.model_dump() for review in self.previous_reviews]
        return metadata


class ShellReviewRecord(BaseModel):
    """Stored shell review result for short-term reviewer context."""

    request: ShellReviewRequest
    decision: ShellReviewDecision
    tool_call_id: str | None = None
    approved: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def get_previous_shell_reviews(
    ctx: AgentContext,
    request: ShellReviewRequest,
    *,
    tool_call_id: str | None = None,
) -> list[ShellReviewPreviousDecision]:
    """Return last N previous shell reviews, prioritizing exact matches."""
    previous: list[ShellReviewPreviousDecision] = []
    seen: set[int] = set()
    records = [record for record in ctx.shell_review_records if isinstance(record, ShellReviewRecord)]
    fingerprint = request.command_fingerprint()

    for predicate in (
        lambda record: tool_call_id is not None and record.tool_call_id == tool_call_id,
        lambda record: record.request.command_fingerprint() == fingerprint,
        lambda record: True,
    ):
        for record in reversed(records):
            record_id = id(record)
            if record_id in seen or not predicate(record):
                continue
            previous.append(
                ShellReviewPreviousDecision(
                    approved=record.approved,
                    risk_level=record.decision.risk_level,
                    reason=record.decision.reason,
                    command=record.request.command,
                    cwd=record.request.cwd,
                )
            )
            seen.add(record_id)
    return previous


async def _run_shell_review_agent(agent: Agent[None, ShellReviewDecision], prompt: str):
    """Run the shell review agent to completion."""
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


async def review_shell_command(
    ctx: AgentContext,
    *,
    request: ShellReviewRequest,
    usage_uuid: str | None = None,
) -> ShellReviewDecision:
    """Review a shell command with the configured small model and record usage."""
    config = ctx.security.shell_review
    if config is None or not config.enabled or config.model is None or config.model.strip() == "":
        logger.debug("Shell review skipped run_id=%s reason=disabled_or_missing_model", ctx.run_id)
        return ShellReviewDecision(risk_level="low", reason="Shell review is disabled.")

    logger.info(
        "Shell review started run_id=%s command_chars=%d background=%s",
        ctx.run_id,
        len(request.command),
        request.background,
    )
    agent = Agent(
        model=infer_model(config.model),
        model_settings=cast(ModelSettings | None, config.model_settings),
        system_prompt=config.system_prompt or DEFAULT_SHELL_REVIEW_PROMPT,
        output_type=ToolOutput(ShellReviewDecision),
    )
    result = await _run_shell_review_agent(agent, request.to_prompt())

    model_id = cast(Model, agent.model).model_name
    usage_id = usage_uuid or uuid4().hex
    await ctx.emit_usage_snapshot(
        agent_id="shell_review",
        agent_name="shell_review",
        model_id=model_id,
        usage=result.usage(),
        source="shell_review",
        usage_id=usage_id,
        ledger_key=usage_id,
    )
    logger.info(
        "Shell review completed run_id=%s risk_level=%s reason=%s",
        ctx.run_id,
        result.output.risk_level,
        result.output.reason,
    )
    return result.output


class ShellReviewBlockedResult(BaseModel):
    """Result returned when shell review blocks execution."""

    stdout: str = ""
    stderr: str = ""
    return_code: int = 1
    error: str
    shell_review: ShellReviewDecision
