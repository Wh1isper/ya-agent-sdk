from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ya_agent_sdk.environment import (
    ShellSandboxBackend,
    ShellSandboxMountPolicy,
    ShellSandboxNetwork,
    ShellSandboxRuntimePolicy,
    resolve_shell_sandbox_runtime_policy,
)

from ya_claw.workspace.models import WorkspaceBinding

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
        mounts=[
            ShellSandboxMountPolicy(
                id=mount.id or "workspace",
                host_path=mount.host_path,
                mode=mount.mode,
            )
            for mount in binding.mounts
        ],
        profile_config=None if profile is None else profile.shell_sandbox,
    )
