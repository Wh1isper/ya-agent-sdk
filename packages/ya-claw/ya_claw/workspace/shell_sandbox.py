from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ya_agent_sdk.environment import (
    ShellSandboxBackend,
    ShellSandboxConfig,
    ShellSandboxMountPolicy,
    ShellSandboxNetwork,
    ShellSandboxRuntimePolicy,
    resolve_shell_sandbox_runtime_policy,
)

from ya_claw.workspace.models import WorkspaceBinding, WorkspaceMountBinding

if TYPE_CHECKING:
    from ya_claw.execution.profile import ResolvedProfile


@dataclass(frozen=True, slots=True)
class WorkspaceShellSandboxDefaults:
    enabled: bool = True
    backend: ShellSandboxBackend = "auto"
    network: ShellSandboxNetwork = "full"
    allow_raw_host: bool = False


def resolve_workspace_shell_sandbox_policy(
    *,
    binding: WorkspaceBinding,
    defaults: WorkspaceShellSandboxDefaults,
    profile: ResolvedProfile | None = None,
) -> ShellSandboxRuntimePolicy:
    return resolve_shell_sandbox_runtime_policy(
        enabled=defaults.enabled,
        backend=defaults.backend,
        network=defaults.network,
        allow_raw_host=defaults.allow_raw_host,
        mounts=shell_sandbox_mounts_from_binding(binding),
        profile_config=shell_sandbox_config_from_profile(profile),
    )


def shell_sandbox_mounts_from_binding(binding: WorkspaceBinding) -> list[ShellSandboxMountPolicy]:
    return [shell_sandbox_mount_from_workspace_mount(mount) for mount in binding.mounts]


def shell_sandbox_mount_from_workspace_mount(mount: WorkspaceMountBinding) -> ShellSandboxMountPolicy:
    return ShellSandboxMountPolicy(
        id=mount.id or "workspace",
        host_path=Path(mount.host_path),
        mode=mount.mode,
    )


def shell_sandbox_config_from_profile(profile: ResolvedProfile | None) -> ShellSandboxConfig | None:
    if profile is None:
        return None
    return profile.shell_sandbox
