"""TUI Environment for yaacli.

TUIEnvironment extends LocalEnvironment with BackgroundMonitor
for managing background subagent tasks and shell process monitoring.
Shell background process management is handled by the Shell ABC from
ya-agent-environment directly.

Example:
    async with TUIEnvironment(default_path=Path.cwd()) as env:
        # Background shell processes via Shell ABC
        process_id = await env.shell.start("npm run dev")
        for pid, proc in env.shell.active_background_processes.items():
            print(f"{pid}: {proc.command}")
"""

from __future__ import annotations

from pathlib import Path

from ya_agent_environment import ResourceFactory, ResourceRegistryState
from ya_agent_sdk.environment.local import LocalEnvironment

from yaacli.background import BACKGROUND_MONITOR_KEY, BackgroundMonitor


class TUIEnvironment(LocalEnvironment):
    """Extended environment for TUI with background monitoring.

    Background process management is provided by Shell ABC directly
    (start/drain_output/wait_process/kill_process). BackgroundMonitor
    handles subagent task tracking and shell process completion
    monitoring, and is registered as a resource.
    """

    def __init__(
        self,
        allowed_paths: list[Path] | None = None,
        default_path: Path | None = None,
        instructions_paths: list[Path] | None = None,
        shell_timeout: float = 30.0,
        tmp_base_dir: Path | None = None,
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
            tmp_base_dir=tmp_base_dir,
            enable_tmp_dir=enable_tmp_dir,
            resource_state=resource_state,
            resource_factories=resource_factories,
            include_os_env=include_os_env,
        )
        self._background_monitor: BackgroundMonitor | None = None

    async def _setup(self) -> None:
        await super()._setup()
        self._background_monitor = BackgroundMonitor()
        self.resources.set(BACKGROUND_MONITOR_KEY, self._background_monitor)

    async def _teardown(self) -> None:
        """Clean up environment resources.

        Keep _background_monitor set until Environment.__aexit__ closes the
        resource registry, so BackgroundMonitor.close() can stop its polling
        task and cancel tracked background tasks.
        """
        await super()._teardown()

    @property
    def background_monitor(self) -> BackgroundMonitor:
        """Get the BackgroundMonitor resource."""
        if self._background_monitor is None:
            raise RuntimeError("TUIEnvironment not entered. Use 'async with TUIEnvironment() as env:'")
        return self._background_monitor
