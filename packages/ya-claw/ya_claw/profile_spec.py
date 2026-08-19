"""Native YA Claw profile documents and host-only policy."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_ai import AgentSpec
from ya_agent_sdk.subagents import SubagentSpec

CLAW_PROFILE_SCHEMA_VERSION = 2
CLAW_HOST_TOOL_GROUPS = frozenset({"agency", "schedule", "session", "workflow"})


class ClawProfileHostConfig(BaseModel):
    """Claw-owned execution policy kept outside the portable agent definition."""

    model_config = ConfigDict(extra="forbid")

    model_config_preset: str | None = None
    model_config_override: dict[str, Any] | None = None
    tool_groups: tuple[str, ...] = ()
    need_user_approve_tools: tuple[str, ...] = ()
    need_user_approve_mcps: tuple[str, ...] = ()
    enabled_mcps: tuple[str, ...] = ()
    disabled_mcps: tuple[str, ...] = ()
    mcp_servers: dict[str, Any] = Field(default_factory=dict)
    workspace_backend_hint: str | None = None

    @field_validator(
        "tool_groups",
        "need_user_approve_tools",
        "need_user_approve_mcps",
        "enabled_mcps",
        "disabled_mcps",
    )
    @classmethod
    def _normalize_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if any(not item for item in normalized):
            raise ValueError("Profile policy names cannot be empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("Profile policy names must be unique")
        return normalized

    @field_validator("tool_groups")
    @classmethod
    def _validate_tool_groups(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        unknown = sorted(set(value) - CLAW_HOST_TOOL_GROUPS)
        if unknown:
            raise ValueError("Unsupported Claw host tool groups: " + ", ".join(unknown))
        return value

    @model_validator(mode="after")
    def _validate_mcp_selection(self) -> ClawProfileHostConfig:
        overlap = set(self.enabled_mcps) & set(self.disabled_mcps)
        if overlap:
            raise ValueError("MCP namespaces cannot be both enabled and disabled: " + ", ".join(sorted(overlap)))
        return self


class ClawProfileSeedDefinition(BaseModel):
    """Versioned seed document; API path identity remains outside the body."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    schema_version: Literal[2] = CLAW_PROFILE_SCHEMA_VERSION
    name: str
    agent: AgentSpec
    host: ClawProfileHostConfig = Field(default_factory=ClawProfileHostConfig)
    subagents: tuple[SubagentSpec, ...] = ()
    enabled: bool = True
    source_type: str | None = "seed"
    source_version: str | None = None
    source_checksum: str | None = None

    @field_validator("name")
    @classmethod
    def _normalize_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Profile name cannot be empty")
        return normalized

    @model_validator(mode="after")
    def _validate_agent(self) -> ClawProfileSeedDefinition:
        if not isinstance(self.agent.model, str) or not self.agent.model.strip():
            raise ValueError("Profile AgentSpec must define a model")
        if self.agent.name is not None and self.agent.name != self.name:
            raise ValueError(f"Profile AgentSpec name {self.agent.name!r} must match profile name {self.name!r}")
        routes = [spec.route for spec in self.subagents]
        if len(routes) != len(set(routes)):
            raise ValueError("Profile subagent routes must be unique")
        return self
