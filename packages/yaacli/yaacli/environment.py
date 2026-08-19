"""TUI environment with host-side background shell readiness monitoring.

Shell process ownership remains in the shared environment abstraction. The
additional monitor only reports readiness to the durable TUI application.

Example:
    async with TUIEnvironment(default_path=Path.cwd()) as env:
        # Background shell processes via Shell ABC
        process_id = await env.shell.start("npm run dev")
        for pid, proc in env.shell.active_background_processes.items():
            print(f"{pid}: {proc.command}")
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from ya_agent_environment import ResourceFactory, ResourceRegistryState
from ya_agent_sdk.environment.local import LocalEnvironment

from yaacli.shell_monitor import SHELL_MONITOR_KEY, ShellMonitor


class TUIEnvironment(LocalEnvironment):
    """Local environment with a lifecycle-managed shell monitor resource."""

    def __init__(
        self,
        allowed_paths: list[Path] | None = None,
        default_path: Path | None = None,
        instructions_paths: list[Path] | None = None,
        shell_timeout: float = 30.0,
        enable_tmp_dir: bool = True,
        resource_state: ResourceRegistryState | None = None,
        resource_factories: dict[str, ResourceFactory] | None = None,
        include_os_env: bool = True,
    ) -> None:
        super().__init__(
            allowed_paths=allowed_paths,
            default_path=default_path,
            instructions_paths=instructions_paths,
            shell_timeout=shell_timeout,
            tmp_base_dir=Path(tempfile.gettempdir()).resolve(),
            enable_tmp_dir=enable_tmp_dir,
            resource_state=resource_state,
            resource_factories=resource_factories,
            include_os_env=include_os_env,
        )
        self._shell_monitor: ShellMonitor | None = None

    async def _setup(self) -> None:
        await super()._setup()
        self._shell_monitor = ShellMonitor()
        self.resources.set(SHELL_MONITOR_KEY, self._shell_monitor)

    async def _teardown(self) -> None:
        """Clean up environment-owned state after registered resources close."""
        try:
            await super()._teardown()
        finally:
            self._shell_monitor = None

    @property
    def shell_monitor(self) -> ShellMonitor:
        """Return the entered shell monitor resource."""
        if self._shell_monitor is None:
            raise RuntimeError("TUIEnvironment not entered. Use 'async with TUIEnvironment() as env:'")
        return self._shell_monitor
