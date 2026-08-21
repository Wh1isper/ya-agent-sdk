"""Model tool for starting a background shell process with host monitoring."""

from __future__ import annotations

from typing import Annotated, cast

from pydantic import Field
from pydantic_ai import RunContext
from ya_agent_environment import Shell
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.events import BackgroundShellStartEvent
from ya_agent_sdk.toolsets.core.base import BaseTool

from yaacli.shell_monitor import SHELL_MONITOR_KEY, ShellMonitor


def _get_shell_monitor(ctx: RunContext[AgentContext]) -> ShellMonitor | None:
    resources = ctx.deps.resources
    if resources is None:
        return None
    resource = resources.get(SHELL_MONITOR_KEY)
    return resource if isinstance(resource, ShellMonitor) else None


class MonitoredShellTool(BaseTool):
    """Start a background shell process whose readiness wakes the host."""

    name = "shell_monitor"
    description = (
        "Start a background shell process with output monitoring. "
        "The host notifies the agent when output is ready or the process completes."
    )
    tags = frozenset({"shell"})

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        monitor = _get_shell_monitor(ctx)
        return ctx.deps.shell is not None and monitor is not None and monitor.is_running

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        del ctx
        return (
            "Use monitored shell for long-running processes where new output should wake the agent. "
            "Prefer it over ordinary background shell when waiting would otherwise require polling; "
            "keep foreground shell for short commands."
        )

    async def call(
        self,
        ctx: RunContext[AgentContext],
        command: Annotated[str, Field(description="The shell command to execute.")],
        environment: Annotated[
            dict[str, str] | None,
            Field(description="Environment variables to set for the command."),
        ] = None,
        cwd: Annotated[
            str | None,
            Field(description="Working directory (relative or absolute path)."),
        ] = None,
    ) -> dict[str, str]:
        if not command.strip():
            return {"error": "Command cannot be empty."}
        monitor = _get_shell_monitor(ctx)
        if monitor is None:
            return {"error": "ShellMonitor not available"}
        shell = cast(Shell, ctx.deps.shell)
        merged_environment = (
            {**ctx.deps.shell_env, **(environment or {})} if ctx.deps.shell_env or environment else None
        )
        try:
            process_id = await shell.start(command, env=merged_environment, cwd=cwd)
            monitor.register(process_id)
        except Exception as exc:
            return {"error": f"Failed to start background command: {exc}"}

        await ctx.deps.emit_event(
            BackgroundShellStartEvent(
                event_id=f"bg-{process_id}",
                process_id=process_id,
                command=command,
            )
        )
        return {
            "process_id": process_id,
            "hint": (
                "The process is monitored. Use shell_wait to read output, shell_input to send stdin, "
                "or shell_kill to terminate it."
            ),
        }
