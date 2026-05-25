"""Background delegate tool for TUI environment.

This tool launches subagent tasks in the background without blocking
the main agent. Results are delivered via message bus when complete.

SteerSubagentTool allows the main agent to send steering guidance to
running background subagents via the shared message bus.

Example:
    from ya_agent_sdk.toolsets.core.base import Toolset
    from yaacli.toolsets.background import SpawnDelegateTool, SteerSubagentTool

    toolset = Toolset(tools=[..., SpawnDelegateTool, SteerSubagentTool])
"""

from __future__ import annotations

import asyncio
from typing import Annotated, cast

from pydantic import Field
from pydantic_ai import RunContext
from ya_agent_environment import Shell
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.context.bus import BusMessage
from ya_agent_sdk.events import BackgroundShellStartEvent
from ya_agent_sdk.toolsets.core.base import BaseTool
from ya_agent_sdk.toolsets.core.subagent.factory import generate_unique_id

from yaacli.background import BACKGROUND_MONITOR_KEY, BackgroundMonitor
from yaacli.logging import get_logger

logger = get_logger(__name__)


def _get_background_monitor(ctx: RunContext[AgentContext]) -> BackgroundMonitor | None:
    """Get BackgroundMonitor from resources."""
    if ctx.deps.resources is None:
        return None
    resource = ctx.deps.resources.get(BACKGROUND_MONITOR_KEY)
    if isinstance(resource, BackgroundMonitor):
        return resource
    return None


class SpawnDelegateTool(BaseTool):
    """Launch a subagent in the background without blocking.

    This tool wraps the SDK's `delegate` tool and runs it as an asyncio task.
    The main agent continues immediately while the subagent works.
    Results are delivered via message bus when the subagent completes.

    Supports resume via agent_id parameter: pass the agent_id of a previously
    completed background subagent to continue its conversation in the background.
    """

    name = "spawn_delegate"
    description = "Spawn a subagent in the background (non-blocking). Result delivered via message bus."
    tags = frozenset({"delegation"})

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        """Available only for main agent with BackgroundMonitor and delegate tool.

        Restricted to main agent because:
        - Results are sent to target=deps.agent_id via message bus
        - Subagents unsubscribe from bus when they exit
        - Messages sent to a subagent's agent_id would become unreachable
        """
        # Only available for main agent to avoid unreachable messages
        if ctx.deps.agent_id != "main":
            return False
        monitor = _get_background_monitor(ctx)
        return monitor is not None and monitor.has_delegate_tool

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        """Generate instruction for spawn delegate."""
        monitor = _get_background_monitor(ctx)
        if monitor is None:
            return None

        # Get active background tasks info
        task_info = monitor.get_context_instruction()

        lines = [
            "Same subagent names as the `delegate` tool.",
            "Use this for long-running tasks where you don't need immediate results.",
            "The subagent runs in the background and its result is delivered via message bus.",
            "You can continue working on other tasks while the background subagent runs.",
        ]
        if task_info:
            lines.append("")
            lines.append(task_info)
        return "\n".join(lines)

    async def call(
        self,
        ctx: RunContext[AgentContext],
        subagent_name: Annotated[str, Field(description="Name of the subagent to delegate to")],
        prompt: Annotated[str, Field(description="The prompt to send to the subagent")],
        agent_id: Annotated[
            str | None, Field(description="Optional agent ID to resume a previous background subagent")
        ] = None,
    ) -> str:
        """Launch a subagent in the background."""
        monitor = _get_background_monitor(ctx)
        if monitor is None:
            return "Error: BackgroundMonitor not available"

        delegate = monitor.get_delegate_tool()
        if delegate is None:
            return "Error: delegate tool not available"

        deps = ctx.deps

        # Use provided agent_id for resume, or generate a new one
        is_resume = agent_id is not None and agent_id in deps.subagent_history
        if not agent_id:
            short_id = generate_unique_id(deps.subagent_history)
            agent_id = f"{subagent_name}-bg-{short_id}"

        async def _run_background() -> None:
            """Background coroutine that runs the subagent and posts result to bus."""
            try:
                result = await delegate.call(
                    ctx,
                    subagent_name=subagent_name,
                    prompt=prompt,
                    agent_id=agent_id,
                )
                monitor.enqueue_usage_snapshot(deps.build_usage_snapshot())
                # Queue result for TUI-managed redelivery. The main SDK context
                # clears bus state on exit, so direct send can be lost when a
                # background task completes while the main agent is still running.
                monitor.enqueue_message(
                    BusMessage(
                        content=result,
                        source=agent_id,
                        target=deps.agent_id,
                    )
                )
                logger.info("Spawned delegate '%s' (%s) completed", subagent_name, agent_id)
            except Exception as e:
                logger.warning("Spawned delegate '%s' (%s) failed: %s", subagent_name, agent_id, e)
                monitor.enqueue_message(
                    BusMessage(
                        content=f"Spawned delegate '{subagent_name}' (id: {agent_id}) failed: {e}",
                        source=agent_id,
                        target=deps.agent_id,
                    )
                )
            finally:
                # Notify completion so TUI can trigger a new agent turn if idle
                monitor.notify_completion(agent_id)

        task = asyncio.create_task(_run_background())
        monitor.register_task(agent_id, task, subagent_name=subagent_name, prompt=prompt, is_resume=is_resume)

        action = "Resumed" if is_resume else "Spawned"
        return (
            f"{action} delegate: {subagent_name} (id: {agent_id}). "
            "Result will be delivered via message bus when complete. "
            "You can continue with other work."
        )

    @staticmethod
    def _get_monitor(ctx: RunContext[AgentContext]) -> BackgroundMonitor | None:
        """Get BackgroundMonitor from resources."""
        return _get_background_monitor(ctx)


class SteerSubagentTool(BaseTool):
    """Send steering guidance to a running background subagent.

    Injects a message into the subagent's execution via the shared message bus.
    The subagent will receive it on its next LLM call via inject_bus_messages filter.

    If the target subagent has already completed, suggests using spawn_delegate
    with agent_id to resume the conversation instead.
    """

    name = "steer_subagent"
    description = "Send additional guidance to a running background subagent."
    tags = frozenset({"delegation"})

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        """Available only for main agent with active background tasks."""
        if ctx.deps.agent_id != "main":
            return False
        monitor = _get_background_monitor(ctx)
        return monitor is not None and monitor.has_active_tasks

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        """Only show instruction when there are active background tasks."""
        monitor = _get_background_monitor(ctx)
        if monitor is None or not monitor.has_active_tasks:
            return None
        return (
            "Send additional guidance to a running background subagent.\n"
            "The message is injected into the subagent's context on its next LLM call.\n"
            "Use this to redirect, refine, or add constraints to an in-progress task."
        )

    async def call(
        self,
        ctx: RunContext[AgentContext],
        agent_id: Annotated[str, Field(description="ID of the background subagent (e.g., 'searcher-bg-a7b9')")],
        message: Annotated[str, Field(description="Steering guidance to send")],
    ) -> str:
        """Send steering message to a running background subagent."""
        monitor = _get_background_monitor(ctx)
        if monitor is None:
            return "Error: BackgroundMonitor not available"

        # Verify the target agent is actually running (single snapshot to avoid race)
        tasks = monitor.active_tasks
        if agent_id not in tasks or tasks[agent_id].done():
            return self._suggest_resume(ctx, agent_id, message, monitor)

        # Send targeted message via shared bus
        ctx.deps.send_message(
            BusMessage(
                content=message,
                source=ctx.deps.agent_id,  # "main"
                target=agent_id,
            )
        )

        return f"Steering message sent to {agent_id}. It will be injected on the subagent's next LLM call."

    def _suggest_resume(
        self,
        ctx: RunContext[AgentContext],
        agent_id: str,
        message: str,
        monitor: BackgroundMonitor,
    ) -> str:
        """Build error message suggesting resume for finished agents."""
        # Look up agent_name from registry for the delegate call
        agent_info = ctx.deps.agent_registry.get(agent_id)
        agent_name = agent_info.agent_name if agent_info else agent_id.rsplit("-bg-", 1)[0]

        # Check if there are other active tasks to mention
        active = [aid for aid, t in monitor.active_tasks.items() if not t.done()]
        active_hint = f" Active tasks: {', '.join(active)}" if active else ""

        # Truncate message for the suggestion if too long, and escape quotes
        prompt_preview = message[:80] + "..." if len(message) > 80 else message
        prompt_preview = prompt_preview.replace('"', '\\"')

        return (
            f"'{agent_id}' has already completed and cannot be steered.{active_hint}\n"
            f"To continue its conversation in the background, use spawn_delegate with agent_id to resume:\n"
            f'  spawn_delegate(subagent_name="{agent_name}", prompt="{prompt_preview}", agent_id="{agent_id}")\n'
            f"Or use blocking delegate for synchronous resume:\n"
            f'  delegate(subagent_name="{agent_name}", prompt="{prompt_preview}", agent_id="{agent_id}")'
        )

    @staticmethod
    def _get_monitor(ctx: RunContext[AgentContext]) -> BackgroundMonitor | None:
        """Get BackgroundMonitor from resources."""
        return _get_background_monitor(ctx)


class MonitoredShellTool(BaseTool):
    """Start a background shell process with output monitoring.

    This tool wraps shell.start() and registers the process with
    BackgroundMonitor for output monitoring. When the process produces
    new stdout/stderr output, the agent is automatically notified via
    message bus.

    The returned process_id works with all standard shell tools:
    shell_wait, shell_kill, shell_status, shell_input, shell_signal.
    """

    name = "shell_monitor"
    description = (
        "Start a background shell process with output monitoring. Automatically notifies when new output is available."
    )
    tags = frozenset({"shell"})

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        """Available when shell and BackgroundMonitor are both present."""
        if ctx.deps.shell is None:
            return False
        monitor = _get_background_monitor(ctx)
        return monitor is not None and monitor.is_shell_monitor_running

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        """Instruction for the monitored shell tool."""
        return (
            "Start a background shell process that automatically notifies you when new output is available.\n"
            "The process works with all existing shell tools (shell_wait, shell_kill, shell_input, etc.).\n"
            "Use this instead of shell_exec(background=True) when you want to be notified of output "
            "without having to poll manually."
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
        """Start a monitored background shell process."""
        if not command or not command.strip():
            return {"error": "Command cannot be empty."}

        monitor = _get_background_monitor(ctx)
        if monitor is None:
            return {"error": "BackgroundMonitor not available"}

        shell = cast(Shell, ctx.deps.shell)

        # Merge environment: ctx.shell_env (base) + per-call env (overrides)
        shell_env = ctx.deps.shell_env
        if shell_env or environment:
            merged_env = {**shell_env, **(environment or {})}
            environment = merged_env

        try:
            process_id = await shell.start(command, env=environment, cwd=cwd)
        except Exception as e:
            return {"error": f"Failed to start background command: {e}"}

        # Register for output monitoring
        monitor.register_monitored_process(process_id)

        # Emit event for consistency with shell_exec background mode
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
                f"Monitored background process started (id={process_id}). "
                "You will be notified automatically when new output is available. "
                "Use shell_wait to read output, shell_input to send stdin, "
                "shell_kill to terminate."
            ),
        }


background_tools: list[type[BaseTool]] = [SpawnDelegateTool, SteerSubagentTool, MonitoredShellTool]
