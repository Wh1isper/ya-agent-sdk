"""Background delegate tool for TUI environment.

This tool launches subagent tasks in the background without blocking
the main agent. Results are delivered via message bus when complete.

SteerSubagentTool allows the main agent to send steering guidance to
running background subagents via the shared message bus.

Example:
    from ya_agent_sdk.toolsets.core.base import Toolset
    from yaacli.toolsets.background import SpawnDelegateTool, SteerSubagentTool, WaitSubagentTool

    toolset = Toolset(tools=[..., SpawnDelegateTool, SteerSubagentTool, WaitSubagentTool])
"""

from __future__ import annotations

import asyncio
import math
from typing import Annotated, Any, cast

from pydantic import Field
from pydantic_ai import RunContext
from ya_agent_environment import Shell
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.context.bus import BusMessage, MessageBus
from ya_agent_sdk.events import BackgroundShellStartEvent
from ya_agent_sdk.toolsets.core.base import BaseTool
from ya_agent_sdk.toolsets.core.subagent.factory import generate_unique_id

from yaacli.background import (
    BACKGROUND_MONITOR_KEY,
    DELEGATE_BACKEND_TOOL_NAME,
    BackgroundMonitor,
    BackgroundTaskResult,
)
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
    description = "Spawn a subagent in the background. Result delivered via message bus."
    tags = frozenset({"delegation"})
    delegate_backend_tool_name: str | None = None

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
        return monitor is not None and monitor.can_delegate(ctx, self.delegate_backend_tool_name)

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        """Generate instruction for background delegation."""
        monitor = _get_background_monitor(ctx)
        if monitor is None:
            return None

        # Get active background tasks info
        task_info = monitor.get_context_instruction()

        lines = [
            "Use asynchronous delegation only for bounded work with clear scope, independent value, or useful parallelism.",
            "Give the subagent enough context, expected output, and constraints to work independently.",
            "Do not poll or loop for results; wait once with a bounded timeout only when integration is blocked.",
            "If no immediate integration work remains, finish the current response and let the CLI notify you on completion.",
            "Steer running subagents only when their scope or constraints need correction.",
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

        delegate = monitor.get_delegate_tool(self.delegate_backend_tool_name)
        if delegate is None:
            return "Error: delegate backend tool not available"

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
                monitor.record_task_result(
                    BackgroundTaskResult(
                        agent_id=agent_id,
                        subagent_name=subagent_name,
                        status="completed",
                        content=result,
                    )
                )
                monitor.enqueue_usage_snapshot(deps.build_usage_snapshot())
                message = BusMessage(
                    id=monitor.get_task_result_message_id(agent_id),
                    content=result,
                    source=agent_id,
                    target=deps.agent_id,
                )
                # Deliver directly to the active SDK message bus so the main
                # run's message_bus_guard can see the result before accepting
                # final output. If the target is no longer subscribed, queue a
                # TUI-managed fallback instead; direct-sending to an unsubscribed
                # bus would make later fallback redelivery look like a duplicate.
                # If wait_subagent is already waiting on this task, let the tool
                # return the result and avoid a duplicate bus delivery.
                if monitor.should_deliver_task_result_message(agent_id):
                    if deps.message_bus.is_subscribed(deps.agent_id):
                        deps.send_message(message)
                    else:
                        monitor.enqueue_message(message)
                logger.info("Spawned delegate '%s' (%s) completed", subagent_name, agent_id)
            except asyncio.CancelledError:
                monitor.record_task_result(
                    BackgroundTaskResult(
                        agent_id=agent_id,
                        subagent_name=subagent_name,
                        status="cancelled",
                        error="Background delegate task was cancelled.",
                    )
                )
                raise
            except Exception as e:
                logger.warning("Spawned delegate '%s' (%s) failed: %s", subagent_name, agent_id, e)
                error_message = f"Spawned delegate '{subagent_name}' (id: {agent_id}) failed: {e}"
                monitor.record_task_result(
                    BackgroundTaskResult(
                        agent_id=agent_id,
                        subagent_name=subagent_name,
                        status="failed",
                        error=str(e),
                    )
                )
                message = BusMessage(
                    id=monitor.get_task_result_message_id(agent_id),
                    content=error_message,
                    source=agent_id,
                    target=deps.agent_id,
                )
                if monitor.should_deliver_task_result_message(agent_id):
                    if deps.message_bus.is_subscribed(deps.agent_id):
                        deps.send_message(message)
                    else:
                        monitor.enqueue_message(message)
            finally:
                # Notify completion so TUI can trigger a new agent turn if idle
                monitor.notify_completion(agent_id)

        task = asyncio.create_task(_run_background())
        monitor.register_task(agent_id, task, subagent_name=subagent_name, prompt=prompt, is_resume=is_resume)

        action = "Resumed" if is_resume else "Spawned"
        return (
            f"{action} delegate: {subagent_name} (id: {agent_id}). "
            "Do not manually poll or loop for the result. "
            "If you need the result before continuing, call wait_subagent once with a bounded timeout. "
            "If you have no other immediate work, finish your current response now; "
            "the CLI will automatically notify you when the result arrives via message bus. "
            f'To adjust the running subagent, call steer_subagent(agent_id="{agent_id}", message=...).'
        )

    @staticmethod
    def _get_monitor(ctx: RunContext[AgentContext]) -> BackgroundMonitor | None:
        """Get BackgroundMonitor from resources."""
        return _get_background_monitor(ctx)


class AsyncDelegateTool(SpawnDelegateTool):
    """Expose background delegation under the standard delegate name."""

    name = "delegate"
    description = "Delegate task to a subagent asynchronously. Result delivered via message bus."
    delegate_backend_tool_name = DELEGATE_BACKEND_TOOL_NAME

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        """Generate concise async delegate instructions with available subagents."""
        monitor = _get_background_monitor(ctx)
        if monitor is None:
            return None

        roster = monitor.get_delegate_roster_instruction(ctx, self.delegate_backend_tool_name)
        if roster is None:
            return None

        lines = [
            "In this TUI, delegate is asynchronous: it returns an agent ID immediately; the final result arrives via message bus.",
            "Delegate only bounded subtasks with clear scope, independent value, or useful parallelism; do not delegate tiny one-step actions or simple lookups.",
            "Give delegates enough context, expected output, and constraints to work independently.",
            "The parent agent owns planning, integration, user-facing synthesis, and final decisions.",
            "After delegating, do not manually poll or loop; call wait_subagent once with a bounded timeout only when integration is blocked.",
            "Otherwise finish the current response and let the CLI notify you.",
            "Use steer_subagent only when a running subagent's scope or constraints need correction.",
            "If using task tracking, pass the relevant task ID; otherwise do not ask delegates to create or claim tasks.",
            "",
            roster,
        ]
        task_info = monitor.get_context_instruction()
        if task_info:
            lines.append("")
            lines.append(task_info)
        return "\n".join(lines)


class WaitSubagentTool(BaseTool):
    """Wait for background subagents to finish and return cached results."""

    name = "wait_subagent"
    description = "Wait for one or more background subagents to finish and return their results."
    tags = frozenset({"delegation"})
    max_timeout_seconds = 300.0

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        """Available only for main agent with active or cached background subagent work."""
        if ctx.deps.agent_id != "main":
            return False
        monitor = _get_background_monitor(ctx)
        if monitor is None:
            return False
        return monitor.has_active_tasks or bool(monitor.task_results)

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        """Generate instruction for bounded fan-in waits."""
        monitor = _get_background_monitor(ctx)
        if monitor is None or not (monitor.has_active_tasks or monitor.task_results):
            return None
        return (
            "Use wait_subagent only as a bounded fan-in point when you cannot continue without a "
            "background subagent result. Do not repeatedly call it in a polling loop. "
            "Prefer finishing the current response when no immediate integration work remains."
        )

    async def call(
        self,
        ctx: RunContext[AgentContext],
        agent_id: Annotated[
            str | None,
            Field(description="Optional background subagent ID to wait for. Omit to wait for all active subagents."),
        ] = None,
        timeout_seconds: Annotated[
            float,
            Field(description="Maximum seconds to wait before returning without cancelling the subagent."),
        ] = 30.0,
    ) -> dict[str, Any]:
        """Wait for one or all background subagents with a bounded timeout."""
        monitor = _get_background_monitor(ctx)
        if monitor is None:
            return {"status": "error", "error": "BackgroundMonitor not available"}

        timeout = self._normalize_timeout(timeout_seconds)
        if agent_id is not None:
            return await self._wait_for_one(
                monitor,
                agent_id,
                timeout,
                bus=ctx.deps.message_bus,
                target=ctx.deps.agent_id,
            )
        return await self._wait_for_all(monitor, timeout, bus=ctx.deps.message_bus, target=ctx.deps.agent_id)

    async def _wait_for_one(
        self,
        monitor: BackgroundMonitor,
        agent_id: str,
        timeout: float,
        *,
        bus: MessageBus,
        target: str,
    ) -> dict[str, Any]:
        """Wait for a single background subagent."""
        known_ids = set(monitor.known_task_ids())
        if agent_id not in known_ids:
            return {
                "status": "not_found",
                "agent_id": agent_id,
                "timed_out": False,
                "known_agent_ids": sorted(known_ids),
            }

        monitor.begin_task_result_wait(agent_id)
        try:
            result = await monitor.wait_for_agent(agent_id, timeout=timeout)
        finally:
            monitor.end_task_result_wait(agent_id)
        if result is None:
            return {
                "status": "running",
                "agent_id": agent_id,
                "timed_out": True,
                "message": "Subagent is still running.",
            }
        monitor.mark_task_result_delivered(agent_id, bus=bus, target=target)
        return self._format_result(result)

    async def _wait_for_all(
        self,
        monitor: BackgroundMonitor,
        timeout: float,
        *,
        bus: MessageBus,
        target: str,
    ) -> dict[str, Any]:
        """Wait for all active background subagents and include cached completed results."""
        agent_ids = monitor.known_task_ids()
        if not agent_ids:
            return {"status": "empty", "timed_out": False, "results": []}

        for known_agent_id in agent_ids:
            monitor.begin_task_result_wait(known_agent_id)
        try:
            results_by_id = await monitor.wait_for_agents(agent_ids, timeout=timeout)
        finally:
            for known_agent_id in agent_ids:
                monitor.end_task_result_wait(known_agent_id)
        formatted_results = []
        timed_out = False
        for agent_id in agent_ids:
            result = results_by_id.get(agent_id)
            if result is None:
                timed_out = True
                formatted_results.append({
                    "status": "running",
                    "agent_id": agent_id,
                    "timed_out": True,
                    "message": "Subagent is still running.",
                })
            else:
                monitor.mark_task_result_delivered(agent_id, bus=bus, target=target)
                formatted_results.append(self._format_result(result))

        if timed_out and any(item.get("status") != "running" for item in formatted_results):
            status = "partial"
        elif timed_out:
            status = "running"
        else:
            status = "completed"

        return {
            "status": status,
            "timed_out": timed_out,
            "results": formatted_results,
        }

    @staticmethod
    def _format_result(result: BackgroundTaskResult) -> dict[str, Any]:
        """Format a cached terminal result for tool output."""
        payload: dict[str, Any] = {
            "status": result.status,
            "agent_id": result.agent_id,
            "subagent_name": result.subagent_name,
            "timed_out": False,
            "completed_at": result.completed_at.isoformat(),
        }
        if result.content is not None:
            payload["result"] = result.content
        if result.error is not None:
            payload["error"] = result.error
        return payload

    @classmethod
    def _normalize_timeout(cls, timeout_seconds: float) -> float:
        """Clamp timeout to a safe finite range."""
        if not math.isfinite(timeout_seconds):
            return cls.max_timeout_seconds
        if timeout_seconds < 0:
            return 0.0
        return min(timeout_seconds, cls.max_timeout_seconds)

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
            "Steer a running background subagent only to redirect, refine, or add constraints. "
            "Do not poll after steering; the CLI will notify you when the subagent completes."
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

        delegate_tool = monitor.get_delegate_tool("delegate")
        resume_tool = (
            "spawn_delegate"
            if delegate_tool is not None and not isinstance(delegate_tool, AsyncDelegateTool)
            else "delegate"
        )
        return (
            f"'{agent_id}' has already completed and cannot be steered.{active_hint}\n"
            f"To continue its conversation, resume it with {resume_tool} and agent_id:\n"
            f'  {resume_tool}(subagent_name="{agent_name}", prompt="{prompt_preview}", agent_id="{agent_id}")'
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
            "Use monitored shell for long-running processes where new output should wake the agent. "
            "Prefer it over ordinary background shell when waiting would otherwise require polling; keep foreground shell for short commands."
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


background_tools: list[type[BaseTool]] = [
    SpawnDelegateTool,
    SteerSubagentTool,
    WaitSubagentTool,
    MonitoredShellTool,
]
