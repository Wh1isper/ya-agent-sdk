from __future__ import annotations

from pathlib import Path

from ya_agent_sdk.environment.local import LocalShell
from ya_agent_sdk.environment.shell_sandbox.policy import ShellSandboxRuntimePolicy


class SandboxedLocalShell(LocalShell):
    """Alias subclass for LocalShell configured with a sandbox policy."""

    def __init__(
        self,
        *,
        policy: ShellSandboxRuntimePolicy,
        environment_overrides: dict[str, str] | None = None,
        default_cwd: Path | None = None,
        allowed_paths: list[Path] | None = None,
        default_timeout: float = 30.0,
        include_os_env: bool = True,
        shell_executable: str | None = None,
    ) -> None:
        super().__init__(
            default_cwd=default_cwd,
            allowed_paths=allowed_paths,
            default_timeout=default_timeout,
            include_os_env=include_os_env,
            shell_executable=shell_executable,
            environment_overrides=environment_overrides,
            sandbox_policy=policy,
        )


__all__ = ["SandboxedLocalShell"]
