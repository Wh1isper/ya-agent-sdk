"""Environment abstractions for file operations and shell execution.

This module provides Protocol-based interfaces and implementations for
environment operations, allowing different backends (local, remote, S3, SSH, etc.)
to be used interchangeably.
"""

from ya_agent_sdk.environment.composite import CompositeFileOperator, LocalMountBackend, Mount, MountBackend
from ya_agent_sdk.environment.local import (
    LocalEnvironment,
    LocalFileOperator,
    LocalShell,
    VirtualLocalFileOperator,
    VirtualMount,
)
from ya_agent_sdk.environment.shell_sandbox import (
    SHELL_SANDBOX_MASKED_PATH_ALIASES,
    SandboxedLocalShell,
    ShellSandboxBackend,
    ShellSandboxConfig,
    ShellSandboxMaskedPathAlias,
    ShellSandboxMountPolicy,
    ShellSandboxNetwork,
    ShellSandboxProfile,
    ShellSandboxRawApproval,
    ShellSandboxRuntimePolicy,
    default_backend_for_platform,
    resolve_masked_paths,
    resolve_shell_sandbox_runtime_policy,
    shell_sandbox_diagnostics,
)

# Sandbox environment is optional (requires docker package)
try:
    from ya_agent_sdk.environment.sandbox import (  # noqa: F401
        DeferredDockerShell,
        DockerShell,
        SandboxEnvironment,
    )

    _DOCKER_AVAILABLE = True
except ModuleNotFoundError:
    _DOCKER_AVAILABLE = False

__all__ = [
    "SHELL_SANDBOX_MASKED_PATH_ALIASES",
    "CompositeFileOperator",
    "LocalEnvironment",
    "LocalFileOperator",
    "LocalMountBackend",
    "LocalShell",
    "Mount",
    "MountBackend",
    "SandboxedLocalShell",
    "ShellSandboxBackend",
    "ShellSandboxConfig",
    "ShellSandboxMaskedPathAlias",
    "ShellSandboxMountPolicy",
    "ShellSandboxNetwork",
    "ShellSandboxProfile",
    "ShellSandboxRawApproval",
    "ShellSandboxRuntimePolicy",
    "VirtualLocalFileOperator",
    "VirtualMount",
    "default_backend_for_platform",
    "resolve_masked_paths",
    "resolve_shell_sandbox_runtime_policy",
    "shell_sandbox_diagnostics",
]

# Add Sandbox exports if available
if _DOCKER_AVAILABLE:
    __all__.extend(["DeferredDockerShell", "DockerShell", "SandboxEnvironment"])
