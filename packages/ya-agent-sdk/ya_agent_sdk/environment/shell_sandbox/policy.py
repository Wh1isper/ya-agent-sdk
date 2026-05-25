from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# Policy labels carried into metadata and shell context. Current filesystem
# enforcement comes from ShellSandboxMountPolicy.mode; backend, network,
# env_allowlist, and raw_shell_allowed drive command execution behavior.
ShellSandboxProfile = Literal[
    "read_only",
    "workspace_write",
    "relay_workspace_write",
    "network_proxy",
    "danger_full_access",
]
# Execution backend selector. Implemented local backends are
# linux_bwrap_seccomp, macos_seatbelt, windows_restricted_token gate behavior,
# and raw_host with explicit allowance; docker/podman/nsjail are reserved policy values.
ShellSandboxBackend = Literal[
    "auto",
    "linux_bwrap_seccomp",
    "macos_seatbelt",
    "windows_restricted_token",
    "docker",
    "podman",
    "nsjail",
    "raw_host",
]
# Enforced by backend builders where supported: Linux bwrap unshares network for
# blocked/restricted; macOS seatbelt allows network only for proxy/full.
ShellSandboxNetwork = Literal["blocked", "restricted", "proxy", "full"]
# Raw host escalation policy input. Runtime enforcement uses the derived
# ShellSandboxRuntimePolicy.raw_shell_allowed boolean.
ShellSandboxRawApproval = Literal["forbidden", "requires_human", "allowed_for_profile"]

# Environment variables copied into sandboxed subprocesses. "*" means pass
# the effective environment through. Profiles can set a narrower explicit list.
DEFAULT_SHELL_SANDBOX_ENV_ALLOWLIST = ("*",)


class ShellSandboxConfig(BaseModel):
    """User/profile shell sandbox configuration.

    Directly enforced today: enabled, backend_preference, network,
    env_allowlist, raw_shell_approval. Metadata/prompt fields today: profile,
    audit_enabled. Profile-specific filesystem meaning is represented by
    workspace mount modes after Claw resolves the workspace binding.
    """

    enabled: bool = True
    profile: ShellSandboxProfile = "workspace_write"
    backend_preference: ShellSandboxBackend = "auto"
    network: ShellSandboxNetwork = "full"
    env_allowlist: list[str] = Field(default_factory=lambda: list(DEFAULT_SHELL_SANDBOX_ENV_ALLOWLIST))
    raw_shell_approval: ShellSandboxRawApproval = "requires_human"
    audit_enabled: bool = True

    @field_validator("env_allowlist", mode="before")
    @classmethod
    def _normalize_env_allowlist(cls, value: object) -> list[str]:
        if value is None:
            return list(DEFAULT_SHELL_SANDBOX_ENV_ALLOWLIST)
        if isinstance(value, str):
            items = [item.strip() for item in value.split(",")]
        elif isinstance(value, list):
            items = [str(item).strip() for item in value]
        else:
            return list(DEFAULT_SHELL_SANDBOX_ENV_ALLOWLIST)
        normalized: list[str] = []
        seen: set[str] = set()
        for item in items:
            if item == "" or item in seen:
                continue
            seen.add(item)
            normalized.append(item)
        return normalized or list(DEFAULT_SHELL_SANDBOX_ENV_ALLOWLIST)


@dataclass(frozen=True, slots=True)
class ShellSandboxMountPolicy:
    """Concrete filesystem mount policy consumed by sandbox backends."""

    id: str
    host_path: Path
    mode: Literal["ro", "rw"] = "rw"


@dataclass(frozen=True, slots=True)
class ShellSandboxRuntimePolicy:
    """Resolved policy used by SandboxedLocalShell.

    The backend, network, mounts, env_allowlist, and raw_shell_allowed fields
    affect subprocess creation. Profile and audit fields are exposed as
    metadata/context for agents, logs, and future audit storage.
    """

    enabled: bool
    profile: ShellSandboxProfile
    backend: ShellSandboxBackend
    requested_backend: ShellSandboxBackend
    network: ShellSandboxNetwork
    mounts: list[ShellSandboxMountPolicy]
    env_allowlist: tuple[str, ...] = field(default_factory=lambda: tuple(DEFAULT_SHELL_SANDBOX_ENV_ALLOWLIST))
    raw_shell_allowed: bool = False
    audit_enabled: bool = True

    @property
    def writable_paths(self) -> list[Path]:
        return [mount.host_path for mount in self.mounts if mount.mode == "rw"]

    @property
    def read_only_paths(self) -> list[Path]:
        return [mount.host_path for mount in self.mounts if mount.mode == "ro"]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "profile": self.profile,
            "backend": self.backend,
            "requested_backend": self.requested_backend,
            "network": self.network,
            "env_allowlist": list(self.env_allowlist),
            "raw_shell_allowed": self.raw_shell_allowed,
            "audit_enabled": self.audit_enabled,
            "mounts": [
                {
                    "id": mount.id,
                    "host_path": str(mount.host_path),
                    "mode": mount.mode,
                }
                for mount in self.mounts
            ],
        }


def default_backend_for_platform() -> ShellSandboxBackend:
    if sys.platform.startswith("linux"):
        return "linux_bwrap_seccomp"
    if sys.platform == "darwin":
        return "macos_seatbelt"
    if sys.platform.startswith("win"):
        return "windows_restricted_token"
    return "raw_host"


def resolve_shell_sandbox_runtime_policy(
    *,
    enabled: bool,
    backend: ShellSandboxBackend,
    network: ShellSandboxNetwork,
    allow_raw_host: bool,
    mounts: list[ShellSandboxMountPolicy],
    profile_config: ShellSandboxConfig | None = None,
) -> ShellSandboxRuntimePolicy:
    config = profile_config or ShellSandboxConfig(enabled=enabled, backend_preference=backend, network=network)
    requested_backend = config.backend_preference if config.backend_preference != "auto" else backend
    resolved_backend = default_backend_for_platform() if requested_backend == "auto" else requested_backend
    raw_shell_allowed = allow_raw_host or config.raw_shell_approval == "allowed_for_profile"
    return ShellSandboxRuntimePolicy(
        enabled=enabled and config.enabled,
        profile=config.profile,
        backend=resolved_backend,
        requested_backend=requested_backend,
        network=config.network or network,
        mounts=mounts,
        env_allowlist=tuple(config.env_allowlist),
        raw_shell_allowed=raw_shell_allowed,
        audit_enabled=config.audit_enabled,
    )
