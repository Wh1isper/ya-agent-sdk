"""Approval review security models and helpers."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, cast
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.output import ToolOutput

from ya_agent_sdk._logger import get_logger
from ya_agent_sdk.agents.models import infer_model
from ya_agent_sdk.usage import coerce_run_usage

if TYPE_CHECKING:
    from pydantic_ai import ModelSettings

    from ya_agent_sdk.context import AgentContext

logger = get_logger(__name__)


class ToolSource(StrEnum):
    """Origin of a tool call."""

    BUILTIN = "builtin"
    MCP = "mcp"
    SUBAGENT = "subagent"
    SKILL = "skill"
    USER = "user"


class ToolCategory(StrEnum):
    """Security-relevant action categories."""

    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    NETWORK = "network"
    DESTRUCTIVE = "destructive"
    CREDENTIAL = "credential"
    EXTERNAL_INTEGRATION = "external_integration"
    CONTEXT_MANAGEMENT = "context_management"
    DELEGATION = "delegation"


class ToolScope(StrEnum):
    """Security-relevant affected scopes."""

    WORKSPACE = "workspace"
    SESSION = "session"
    LOCAL_SYSTEM = "local_system"
    NETWORK = "network"
    EXTERNAL_SERVICE = "external_service"


class PermissionDecision(StrEnum):
    """Default policy action for a permission profile."""

    ALLOW = "allow"
    AUTO_REVIEW = "auto_review"
    DENY = "deny"


class ApprovalReviewOutcome(StrEnum):
    """Reviewer outcome."""

    ALLOW = "allow"
    DENY = "deny"


class ApprovalRiskLevel(StrEnum):
    """Reviewer risk levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTRA_HIGH = "extra_high"


class UserAuthorizationLevel(StrEnum):
    """How clearly the user authorized the pending action."""

    EXPLICIT = "explicit"
    IMPLIED = "implied"
    MISSING = "missing"
    CONFLICTING = "conflicting"


class ToolResultTruncationConfig(BaseModel):
    """Central tool result truncation settings."""

    enabled: bool = True
    max_text_chars: int = 60000
    head_chars: int = 30000
    tail_chars: int = 20000
    max_json_chars: int = 60000
    marker: str = "\n\n[Tool output truncated: {omitted_chars} characters omitted]\n\n"

    @model_validator(mode="after")
    def validate_limits(self) -> ToolResultTruncationConfig:
        if self.max_text_chars < 0:
            raise ValueError("max_text_chars must be >= 0")
        if self.max_json_chars < 0:
            raise ValueError("max_json_chars must be >= 0")
        if self.head_chars < 0 or self.tail_chars < 0:
            raise ValueError("head_chars and tail_chars must be >= 0")
        return self


class ToolPermissionProfile(BaseModel):
    """Security metadata for a tool or tool call."""

    source: ToolSource = ToolSource.BUILTIN
    categories: frozenset[ToolCategory] = Field(default_factory=frozenset)
    scopes: frozenset[ToolScope] = Field(default_factory=frozenset)
    default_decision: PermissionDecision = PermissionDecision.ALLOW
    rationale: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("categories", mode="before")
    @classmethod
    def _coerce_categories(cls, value: Any) -> frozenset[ToolCategory]:
        return frozenset(ToolCategory(item) for item in _iter_values(value))

    @field_validator("scopes", mode="before")
    @classmethod
    def _coerce_scopes(cls, value: Any) -> frozenset[ToolScope]:
        return frozenset(ToolScope(item) for item in _iter_values(value))

    def with_categories(self, *categories: ToolCategory | str) -> ToolPermissionProfile:
        values = set(self.categories)
        values.update(ToolCategory(item) for item in categories)
        return self.model_copy(update={"categories": frozenset(values)})

    def with_scopes(self, *scopes: ToolScope | str) -> ToolPermissionProfile:
        values = set(self.scopes)
        values.update(ToolScope(item) for item in scopes)
        return self.model_copy(update={"scopes": frozenset(values)})

    def with_decision(
        self, decision: PermissionDecision | str, *, rationale: str | None = None
    ) -> ToolPermissionProfile:
        update: dict[str, Any] = {"default_decision": PermissionDecision(decision)}
        if rationale is not None:
            update["rationale"] = rationale
        return self.model_copy(update=update)

    def with_metadata(self, **metadata: Any) -> ToolPermissionProfile:
        merged = dict(self.metadata)
        merged.update(metadata)
        return self.model_copy(update={"metadata": merged})

    @classmethod
    def read_workspace(cls) -> ToolPermissionProfile:
        return cls(categories=frozenset({ToolCategory.READ}), scopes=frozenset({ToolScope.WORKSPACE}))

    @classmethod
    def write_workspace(cls) -> ToolPermissionProfile:
        return cls(
            categories=frozenset({ToolCategory.WRITE}),
            scopes=frozenset({ToolScope.WORKSPACE}),
            default_decision=PermissionDecision.AUTO_REVIEW,
            rationale="Writes alter workspace state.",
        )

    @classmethod
    def destructive_workspace(cls) -> ToolPermissionProfile:
        return cls(
            categories=frozenset({ToolCategory.WRITE, ToolCategory.DESTRUCTIVE}),
            scopes=frozenset({ToolScope.WORKSPACE}),
            default_decision=PermissionDecision.AUTO_REVIEW,
            rationale="Destructive workspace changes need approval review.",
        )

    @classmethod
    def execute_local_system(cls) -> ToolPermissionProfile:
        return cls(
            categories=frozenset({ToolCategory.EXECUTE}),
            scopes=frozenset({ToolScope.WORKSPACE, ToolScope.LOCAL_SYSTEM}),
            default_decision=PermissionDecision.AUTO_REVIEW,
            rationale="Local command execution can affect workspace and host state.",
        )

    @classmethod
    def network_read(cls) -> ToolPermissionProfile:
        return cls(
            categories=frozenset({ToolCategory.READ, ToolCategory.NETWORK}), scopes=frozenset({ToolScope.NETWORK})
        )

    @classmethod
    def network_download(cls) -> ToolPermissionProfile:
        return cls(
            categories=frozenset({ToolCategory.READ, ToolCategory.WRITE, ToolCategory.NETWORK}),
            scopes=frozenset({ToolScope.WORKSPACE, ToolScope.NETWORK}),
            default_decision=PermissionDecision.AUTO_REVIEW,
            rationale="Downloads persist external content into the workspace.",
        )

    @classmethod
    def external_integration_read(cls) -> ToolPermissionProfile:
        return cls(
            categories=frozenset({ToolCategory.READ, ToolCategory.EXTERNAL_INTEGRATION, ToolCategory.NETWORK}),
            scopes=frozenset({ToolScope.EXTERNAL_SERVICE}),
        )

    @classmethod
    def external_integration_write(cls) -> ToolPermissionProfile:
        return cls(
            categories=frozenset({ToolCategory.WRITE, ToolCategory.EXTERNAL_INTEGRATION, ToolCategory.NETWORK}),
            scopes=frozenset({ToolScope.EXTERNAL_SERVICE}),
            default_decision=PermissionDecision.AUTO_REVIEW,
            rationale="External service mutation needs approval review.",
        )

    @classmethod
    def context_management(cls) -> ToolPermissionProfile:
        return cls(
            categories=frozenset({ToolCategory.CONTEXT_MANAGEMENT}),
            scopes=frozenset({ToolScope.SESSION}),
        )

    @classmethod
    def session_state(cls) -> ToolPermissionProfile:
        return cls(categories=frozenset({ToolCategory.READ, ToolCategory.WRITE}), scopes=frozenset({ToolScope.SESSION}))

    @classmethod
    def delegation(cls) -> ToolPermissionProfile:
        return cls(categories=frozenset({ToolCategory.DELEGATION}), scopes=frozenset({ToolScope.SESSION}))


class McpPermissionProfile(BaseModel):
    """MCP server and tool permission overrides."""

    server_name: str = ""
    transport: str = "stdio"
    default_decision: PermissionDecision = PermissionDecision.AUTO_REVIEW
    categories: frozenset[ToolCategory] = Field(default_factory=frozenset)
    scopes: frozenset[ToolScope] = Field(default_factory=frozenset)
    tool_overrides: dict[str, ToolPermissionProfile] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("categories", mode="before")
    @classmethod
    def _coerce_categories(cls, value: Any) -> frozenset[ToolCategory]:
        return frozenset(ToolCategory(item) for item in _iter_values(value))

    @field_validator("scopes", mode="before")
    @classmethod
    def _coerce_scopes(cls, value: Any) -> frozenset[ToolScope]:
        return frozenset(ToolScope(item) for item in _iter_values(value))


class ApprovalReviewRequest(BaseModel):
    """Pending action sent to an approval reviewer."""

    request_id: str = Field(default_factory=lambda: f"apr_{uuid4().hex}")
    run_id: str | None = None
    agent_id: str | None = None
    tool_call_id: str | None = None
    source: ToolSource
    tool_name: str
    tool_args: dict[str, Any]
    permission: ToolPermissionProfile
    mcp_server: str | None = None
    mcp_tool: str | None = None
    user_goal: str | None = None
    recent_context: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ApprovalReviewResult(BaseModel):
    """Structured approval review result."""

    request_id: str
    outcome: ApprovalReviewOutcome
    risk_level: ApprovalRiskLevel
    authorization: UserAuthorizationLevel
    rationale: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ApprovalReviewResultRecord(BaseModel):
    """Current-run approval review record used for auditing and reviewer context."""

    request: ApprovalReviewRequest
    result: ApprovalReviewResult
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ApprovalReviewConfig(BaseModel):
    """Generic approval review runtime configuration."""

    model_config = {"arbitrary_types_allowed": True}

    enabled: bool = False
    model: str | Model | None = None
    model_settings: str | dict[str, Any] | None = None
    prompt: str | None = None
    timeout_seconds: float = 30.0
    max_denials: int = 3
    include_recent_messages: int = 12
    truncation: ToolResultTruncationConfig = Field(default_factory=ToolResultTruncationConfig)
    mcp_permissions: dict[str, McpPermissionProfile] = Field(default_factory=dict)

    @field_validator("model_settings", mode="before")
    @classmethod
    def resolve_model_settings_preset(cls, value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        from ya_agent_sdk.presets import resolve_model_settings

        return resolve_model_settings(value)

    @model_validator(mode="after")
    def validate_model_when_enabled(self) -> ApprovalReviewConfig:
        if self.enabled and self.model is None:
            raise ValueError("approval_review.model is required when approval review is enabled.")
        if self.timeout_seconds <= 0:
            raise ValueError("approval_review.timeout_seconds must be > 0")
        if self.max_denials < 1:
            raise ValueError("approval_review.max_denials must be >= 1")
        return self


class ApprovalReviewer(Protocol):
    """Approval reviewer protocol."""

    async def review(self, ctx: AgentContext, request: ApprovalReviewRequest) -> ApprovalReviewResult: ...


DEFAULT_APPROVAL_REVIEW_PROMPT = """You review pending agent tool calls before execution.

Return strict JSON matching the requested schema. Decide whether the exact pending action is allowed.

Risk guidance:
- low: bounded read or harmless session state action
- medium: bounded workspace write or routine local command
- high: destructive, credential, broad local-system, network execution, or external mutation behavior
- extra_high: catastrophic, hostile, credential exfiltration, broad deletion, or clearly unauthorized behavior

Authorization guidance:
- explicit: the user directly requested this action
- implied: the action is a normal bounded step toward the user goal
- missing: the user goal does not authorize this boundary
- conflicting: the action conflicts with user instructions or safety policy

Deny actions that cross a protected boundary with missing or conflicting authorization. Allow bounded actions that are necessary for the user goal. Return concise rationale.
"""


class ModelApprovalReviewer:
    """Model-backed approval reviewer."""

    def __init__(self, config: ApprovalReviewConfig) -> None:
        self._config = config

    async def review(self, ctx: AgentContext, request: ApprovalReviewRequest) -> ApprovalReviewResult:
        if self._config.model is None:
            return closed_deny_result(request, rationale="Approval reviewer model is missing.")
        try:
            result = await asyncio.wait_for(self._review(ctx, request), timeout=self._config.timeout_seconds)
        except Exception as exc:
            logger.warning(
                "Approval reviewer failed run_id=%s request_id=%s error=%r", ctx.run_id, request.request_id, exc
            )
            return closed_deny_result(request, rationale="Approval reviewer used closed-deny fallback.")
        return result

    async def _review(self, ctx: AgentContext, request: ApprovalReviewRequest) -> ApprovalReviewResult:
        model = self._config.model if isinstance(self._config.model, Model) else infer_model(str(self._config.model))
        agent = Agent(
            model=model,
            model_settings=cast("ModelSettings | None", self._config.model_settings),
            system_prompt=self._config.prompt or DEFAULT_APPROVAL_REVIEW_PROMPT,
            output_type=ToolOutput(ApprovalReviewResult),
        )
        result = await agent.run(_render_review_prompt(ctx, request))
        output = result.output
        if output.request_id != request.request_id:
            output = output.model_copy(update={"request_id": request.request_id})
        model_id = cast(Model, agent.model).model_name
        usage_id = f"approval_review:{request.request_id}"
        ctx.update_usage_snapshot_entry(
            agent_id="approval_review",
            agent_name="approval_review",
            model_id=model_id,
            usage=coerce_run_usage(result.usage),
            source="approval_review",
            usage_id=usage_id,
            ledger_key=usage_id,
        )
        return output


def closed_deny_result(request: ApprovalReviewRequest, *, rationale: str) -> ApprovalReviewResult:
    """Create a deny result for reviewer failure or circuit breaker fallback."""

    return ApprovalReviewResult(
        request_id=request.request_id,
        outcome=ApprovalReviewOutcome.DENY,
        risk_level=ApprovalRiskLevel.EXTRA_HIGH,
        authorization=UserAuthorizationLevel.MISSING,
        rationale=rationale,
    )


def approval_review_enabled(ctx: AgentContext) -> bool:
    config = ctx.security.approval_review
    return config is not None and config.enabled


async def evaluate_approval_review(
    ctx: AgentContext,
    request: ApprovalReviewRequest,
    *,
    reviewer: ApprovalReviewer | None = None,
) -> ApprovalReviewResult:
    """Evaluate a pending action against approval review policy."""

    config = ctx.security.approval_review
    if config is None or not config.enabled:
        return ApprovalReviewResult(
            request_id=request.request_id,
            outcome=ApprovalReviewOutcome.ALLOW,
            risk_level=ApprovalRiskLevel.LOW,
            authorization=UserAuthorizationLevel.IMPLIED,
            rationale="Approval review is disabled.",
        )

    denied_count = sum(
        1 for record in ctx.approval_review_records if record.result.outcome == ApprovalReviewOutcome.DENY
    )
    if denied_count >= config.max_denials:
        result = closed_deny_result(request, rationale="Approval review denial circuit breaker activated.")
        record_approval_review(ctx, request, result)
        return result

    reviewer = reviewer or ModelApprovalReviewer(config)
    result = await reviewer.review(ctx, request)
    record_approval_review(ctx, request, result)
    return result


def record_approval_review(ctx: AgentContext, request: ApprovalReviewRequest, result: ApprovalReviewResult) -> None:
    ctx.approval_review_records.append(ApprovalReviewResultRecord(request=request, result=result))


def resolve_policy_decision(permission: ToolPermissionProfile) -> PermissionDecision:
    return permission.default_decision


def build_approval_request(
    ctx: AgentContext,
    *,
    tool_name: str,
    tool_args: Mapping[str, Any],
    permission: ToolPermissionProfile,
    source: ToolSource | None = None,
    mcp_server: str | None = None,
    mcp_tool: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ApprovalReviewRequest:
    return ApprovalReviewRequest(
        run_id=ctx.run_id,
        agent_id=ctx.agent_id,
        tool_call_id=getattr(ctx, "tool_call_id", None),
        source=source or permission.source,
        tool_name=tool_name,
        tool_args=dict(tool_args),
        permission=permission,
        mcp_server=mcp_server,
        mcp_tool=mcp_tool,
        user_goal=_user_goal_text(ctx),
        recent_context=_recent_review_context(ctx),
        metadata=metadata or {},
    )


def denial_tool_result(result: ApprovalReviewResult) -> str:
    return (
        "Tool call denied by approval review. "
        "Continue with a safe alternative plan that respects the denied boundary. "
        f"Reason: {result.rationale}"
    )


def infer_mcp_permission(  # noqa: C901
    *,
    server_name: str,
    transport: str,
    tool_name: str,
    tool_args: Mapping[str, Any] | None = None,
    override: McpPermissionProfile | None = None,
) -> ToolPermissionProfile:
    """Infer permission metadata for an MCP tool."""

    if override is not None:
        tool_override = override.tool_overrides.get(tool_name)
        if tool_override is not None:
            return refine_permission_from_args(
                tool_override.with_metadata(mcp_server=server_name, mcp_tool=tool_name, mcp_transport=transport),
                tool_args or {},
            )
        categories = set(override.categories)
        scopes = set(override.scopes)
        decision = override.default_decision
    else:
        categories = set[ToolCategory]()
        scopes = set[ToolScope]()
        decision = PermissionDecision.AUTO_REVIEW

    if not categories and not scopes:
        categories.add(ToolCategory.EXTERNAL_INTEGRATION)
        if transport == "streamable_http":
            categories.add(ToolCategory.NETWORK)
            scopes.add(ToolScope.EXTERNAL_SERVICE)
        else:
            scopes.add(ToolScope.LOCAL_SYSTEM)

    lowered = tool_name.lower()
    if _contains_any(lowered, ("read", "get", "list", "search", "find", "query", "fetch", "inspect")):
        categories.add(ToolCategory.READ)
    if _contains_any(lowered, ("write", "create", "update", "patch", "set", "send", "post", "upload")):
        categories.add(ToolCategory.WRITE)
        decision = PermissionDecision.AUTO_REVIEW
    if _contains_any(lowered, ("delete", "remove", "drop", "destroy", "truncate", "reset")):
        categories.update({ToolCategory.WRITE, ToolCategory.DESTRUCTIVE})
        decision = PermissionDecision.AUTO_REVIEW
    if _contains_any(lowered, ("run", "exec", "execute", "command", "shell", "script")):
        categories.add(ToolCategory.EXECUTE)
        decision = PermissionDecision.AUTO_REVIEW
    if _contains_any(lowered, ("token", "secret", "credential", "auth", "key")):
        categories.add(ToolCategory.CREDENTIAL)
        decision = PermissionDecision.AUTO_REVIEW
    if _contains_any(lowered, ("issue", "pull", "repo", "branch", "commit", "deploy", "release")):
        categories.add(ToolCategory.EXTERNAL_INTEGRATION)

    profile = ToolPermissionProfile(
        source=ToolSource.MCP,
        categories=frozenset(categories),
        scopes=frozenset(scopes),
        default_decision=decision,
        rationale="MCP tool permissions inferred from server transport and tool name.",
        metadata={"mcp_server": server_name, "mcp_tool": tool_name, "mcp_transport": transport},
    )
    return refine_permission_from_args(profile, tool_args or {})


def refine_permission_from_args(  # noqa: C901
    permission: ToolPermissionProfile,
    args: Mapping[str, Any],
) -> ToolPermissionProfile:
    """Apply generic path, URL, environment, and command refinements."""

    categories = set(permission.categories)
    scopes = set(permission.scopes)
    decision = permission.default_decision
    metadata = dict(permission.metadata)
    path_values: list[str] = []
    url_values: list[str] = []
    command_values: list[str] = []
    secret_like = False
    broad = False

    for key, value in _flatten_args(args):
        key_lower = key.lower()
        text_values = _string_values(value)
        if key_lower in {
            "path",
            "file_path",
            "directory",
            "save_dir",
            "source",
            "destination",
            "src",
            "dst",
            "cwd",
            "working_dir",
        }:
            path_values.extend(text_values)
        if key_lower in {"url", "urls", "endpoint", "host", "base_url"}:
            url_values.extend(text_values)
        if key_lower in {"command", "cmd", "script", "code"}:
            command_values.extend(text_values)
        if (
            key_lower in {"env", "environment", "headers", "token", "api_key", "authorization", "secret"}
            and text_values
        ):
            categories.add(ToolCategory.CREDENTIAL)
            decision = PermissionDecision.AUTO_REVIEW
            secret_like = True

    for path in path_values:
        if is_secret_path(path):
            categories.add(ToolCategory.CREDENTIAL)
            decision = PermissionDecision.AUTO_REVIEW
            secret_like = True
        if is_broad_path(path):
            broad = True

    for url in url_values:
        categories.add(ToolCategory.NETWORK)
        scopes.add(ToolScope.NETWORK)
        if _looks_secret_like(url):
            categories.add(ToolCategory.CREDENTIAL)
            decision = PermissionDecision.AUTO_REVIEW
            secret_like = True
        if url.startswith("file:"):
            scopes.add(ToolScope.LOCAL_SYSTEM)
            decision = PermissionDecision.AUTO_REVIEW

    for command in command_values:
        categories.add(ToolCategory.EXECUTE)
        scopes.add(ToolScope.LOCAL_SYSTEM)
        command_lower = command.lower()
        if _contains_any(
            command_lower, ("curl", "wget", "git clone", "pip install", "npm install", "uv add", "docker pull")
        ):
            categories.add(ToolCategory.NETWORK)
            scopes.add(ToolScope.NETWORK)
        if _contains_any(command_lower, ("rm -rf", "rm -r", "truncate", "git reset --hard", "clean -fd", ">", "tee ")):
            categories.add(ToolCategory.DESTRUCTIVE)
        if _contains_any(command_lower, (".env", "token", "secret", "credential", "api_key", "ssh/", "id_rsa")):
            categories.add(ToolCategory.CREDENTIAL)
            secret_like = True
        decision = PermissionDecision.AUTO_REVIEW

    metadata.update({
        "path_summary": {
            "paths": path_values[:20],
            "secret_like": secret_like,
            "broad": broad,
        },
        "network_summary": {"values": url_values[:20]},
    })
    return permission.model_copy(
        update={
            "categories": frozenset(categories),
            "scopes": frozenset(scopes),
            "default_decision": decision,
            "metadata": metadata,
        }
    )


def truncate_tool_output(value: Any, config: ToolResultTruncationConfig | None) -> Any:
    """Truncate a tool result while preserving its broad shape."""

    if config is None or not config.enabled:
        return value
    if isinstance(value, str):
        return truncate_text(value, config)
    if isinstance(value, bytes | bytearray):
        return f"[Binary tool output: {len(value)} bytes]"
    if isinstance(value, BaseModel):
        return _truncate_json_like(value.model_dump(mode="json"), config)
    if isinstance(value, Mapping | list | tuple):
        return _truncate_json_like(value, config)
    return value


def truncate_text(value: str, config: ToolResultTruncationConfig) -> str:
    if len(value) <= config.max_text_chars:
        return value
    head = value[: config.head_chars]
    tail = value[-config.tail_chars :] if config.tail_chars > 0 else ""
    omitted = max(len(value) - len(head) - len(tail), 0)
    return head + config.marker.format(omitted_chars=omitted) + tail


def permission_summary(permission: ToolPermissionProfile) -> dict[str, Any]:
    return {
        "source": permission.source.value,
        "categories": sorted(category.value for category in permission.categories),
        "scopes": sorted(scope.value for scope in permission.scopes),
        "default_decision": permission.default_decision.value,
        "rationale": permission.rationale,
        "metadata": permission.metadata,
    }


def is_secret_path(path: str) -> bool:
    lowered = path.lower()
    return _looks_secret_like(lowered) or any(
        token in lowered
        for token in (
            ".ssh/",
            ".aws/",
            ".config/gh/",
            "credentials",
            "id_rsa",
            "id_ed25519",
            "known_hosts",
            "auth.json",
        )
    )


def is_broad_path(path: str) -> bool:
    stripped = path.strip()
    return stripped in {"", ".", "./", "/", "..", "../", "*", "**", "**/*"}


def is_generated_path(path: str) -> bool:
    lowered = path.lower()
    return any(part in lowered.split("/") for part in ("build", "dist", ".cache", "__pycache__", "node_modules"))


def is_source_path(path: str) -> bool:
    lowered = path.lower()
    return lowered.endswith((
        ".py",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".rs",
        ".go",
        ".java",
        ".md",
        ".toml",
        ".yaml",
        ".yml",
        ".json",
    ))


def _truncate_json_like(value: Any, config: ToolResultTruncationConfig) -> Any:
    try:
        serialized = json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        return truncate_text(str(value), config)
    if len(serialized) <= config.max_json_chars:
        return value
    return truncate_text(serialized, config)


def _render_review_prompt(ctx: AgentContext, request: ApprovalReviewRequest) -> str:
    payload = {
        "user_goal": request.user_goal,
        "run_id": request.run_id,
        "agent_id": request.agent_id,
        "tool_call_id": request.tool_call_id,
        "tool_name": request.tool_name,
        "source": request.source.value,
        "permission": permission_summary(request.permission),
        "tool_args": request.tool_args,
        "mcp_server": request.mcp_server,
        "mcp_tool": request.mcp_tool,
        "metadata": request.metadata,
        "recent_reviews": [
            {
                "tool_name": record.request.tool_name,
                "outcome": record.result.outcome.value,
                "risk_level": record.result.risk_level.value,
                "authorization": record.result.authorization.value,
                "rationale": record.result.rationale,
            }
            for record in ctx.approval_review_records
        ],
    }
    return "Review this pending tool call.\n\n" + json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _user_goal_text(ctx: AgentContext) -> str | None:
    prompts = ctx.user_prompts
    if isinstance(prompts, str):
        return prompts[-4000:]
    if prompts is None:
        return None
    return str(prompts)[-4000:]


def _recent_review_context(ctx: AgentContext) -> list[dict[str, Any]]:
    return [
        {
            "tool_name": record.request.tool_name,
            "outcome": record.result.outcome.value,
            "risk_level": record.result.risk_level.value,
            "rationale": record.result.rationale,
        }
        for record in ctx.approval_review_records
    ]


def _flatten_args(args: Mapping[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = []
    for key, value in args.items():
        key_text = f"{prefix}.{key}" if prefix else str(key)
        leaf_key = str(key)
        items.append((leaf_key, value))
        if isinstance(value, Mapping):
            items.extend(_flatten_args(value, key_text))
    return items


def _string_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray | str):
        return [item for item in value if isinstance(item, str)]
    return []


def _looks_secret_like(value: str) -> bool:
    lowered = value.lower()
    return bool(
        re.search(
            r"(^|[._/-])(env|secret|token|credential|credentials|api[_-]?key|private[_-]?key|password)([._/-]|$)",
            lowered,
        )
    )


def _contains_any(value: str, tokens: Sequence[str]) -> bool:
    return any(token in value for token in tokens)


def _iter_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, frozenset | set | list | tuple):
        return list(value)
    return [value]
