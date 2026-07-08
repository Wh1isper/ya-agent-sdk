"""Background subagent tools for YA Claw runtime."""

from __future__ import annotations

import asyncio
import time as time_module
from typing import Annotated

from loguru import logger
from pydantic import Field
from pydantic_ai import RunContext
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.context.bus import BusMessage
from ya_agent_sdk.toolsets.core.base import BaseTool
from ya_agent_sdk.toolsets.core.subagent.factory import generate_unique_id

from ya_claw.execution.background import (
    BACKGROUND_MONITOR_KEY,
    BackgroundMonitor,
    BackgroundTaskAlreadyActiveError,
)


def _get_background_monitor(ctx: RunContext[AgentContext]) -> BackgroundMonitor | None:
    if ctx.deps.resources is None:
        return None
    resource = ctx.deps.resources.get(BACKGROUND_MONITOR_KEY)
    if isinstance(resource, BackgroundMonitor):
        return resource
    return None


class SpawnDelegateTool(BaseTool):
    """Launch a subagent in the background during the current run."""

    name = "spawn_delegate"
    description = "Spawn a subagent in the background for the current run. Result delivered via message bus."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        if ctx.deps.agent_id != "main":
            return False
        monitor = _get_background_monitor(ctx)
        return monitor is not None and monitor.has_delegate_tool and not monitor.closed

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        monitor = _get_background_monitor(ctx)
        if monitor is None:
            return None

        task_info = monitor.get_context_instruction()
        lines = [
            "Use current-run background subagents only for bounded work with clear scope, independent value, or useful parallelism.",
            "Use session-backed async subagents instead for work that needs durable execution beyond the current run.",
            "Do not spawn tiny steps or work whose result the parent cannot integrate before commit.",
            "The current run waits before committing; do not poll manually.",
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
        monitor = _get_background_monitor(ctx)
        if monitor is None:
            return "Error: BackgroundMonitor not available"
        if monitor.closed:
            return "Error: BackgroundMonitor is closed"

        delegate = monitor.get_delegate_tool()
        if delegate is None:
            return "Error: delegate tool not available"

        deps = ctx.deps
        is_resume = agent_id is not None and agent_id in deps.subagent_history
        if not agent_id:
            short_id = generate_unique_id(deps.subagent_history)
            agent_id = f"{subagent_name}-bg-{short_id}"

        async def _run_background() -> None:
            start_time = time_module.monotonic()
            try:
                result = await delegate.call(
                    ctx,
                    subagent_name=subagent_name,
                    prompt=prompt,
                    agent_id=agent_id,
                )
                deps.send_message(
                    BusMessage(
                        content=result,
                        source=agent_id,
                        target=deps.agent_id,
                    )
                )
                duration = time_module.monotonic() - start_time
                preview = result[:80] + "..." if len(result) > 80 else result
                await monitor.emit_subagent_completed(
                    agent_id=agent_id,
                    subagent_name=subagent_name,
                    duration_seconds=duration,
                    result_preview=preview,
                )
                logger.info("Spawned delegate completed subagent_name={} agent_id={}", subagent_name, agent_id)
            except asyncio.CancelledError:
                logger.info("Spawned delegate cancelled subagent_name={} agent_id={}", subagent_name, agent_id)
                raise
            except Exception as exc:
                logger.warning(
                    "Spawned delegate failed subagent_name={} agent_id={} error={}", subagent_name, agent_id, exc
                )
                error_msg = str(exc)[:200]
                await monitor.emit_subagent_failed(
                    agent_id=agent_id,
                    subagent_name=subagent_name,
                    error=error_msg,
                )
                deps.send_message(
                    BusMessage(
                        content=f"Spawned delegate '{subagent_name}' (id: {agent_id}) failed: {exc}",
                        source=agent_id,
                        target=deps.agent_id,
                    )
                )

        task = asyncio.create_task(_run_background(), name=f"ya-claw-bg-subagent-{agent_id}")
        try:
            monitor.register_task(agent_id, task, subagent_name=subagent_name, prompt=prompt, is_resume=is_resume)
        except BackgroundTaskAlreadyActiveError as exc:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            return f"Error: {exc}"

        await monitor.emit_subagent_spawned(
            agent_id=agent_id,
            subagent_name=subagent_name,
            prompt=prompt,
        )

        action = "Resumed" if is_resume else "Spawned"
        return (
            f"{action} delegate: {subagent_name} (id: {agent_id}). "
            "Result will be delivered via message bus when complete. "
            "The current run will wait for background tasks before committing."
        )


class SteerSubagentTool(BaseTool):
    """Send steering guidance to a running background subagent."""

    name = "steer_subagent"
    description = "Send additional guidance to a running background subagent."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        if ctx.deps.agent_id != "main":
            return False
        monitor = _get_background_monitor(ctx)
        return monitor is not None and monitor.has_active_tasks and not monitor.closed

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        monitor = _get_background_monitor(ctx)
        if monitor is None or not monitor.has_active_tasks:
            return None
        return "Steer a running background subagent only to redirect, refine, or add constraints."

    async def call(
        self,
        ctx: RunContext[AgentContext],
        agent_id: Annotated[str, Field(description="ID of the background subagent, e.g. 'searcher-bg-a7b9'")],
        message: Annotated[str, Field(description="Steering guidance to send")],
    ) -> str:
        monitor = _get_background_monitor(ctx)
        if monitor is None:
            return "Error: BackgroundMonitor not available"
        if monitor.closed:
            return "Error: BackgroundMonitor is closed"

        tasks = monitor.active_tasks
        if agent_id not in tasks or tasks[agent_id].done():
            return _suggest_resume(ctx, agent_id, message, monitor)

        ctx.deps.send_message(
            BusMessage(
                content=message,
                source=ctx.deps.agent_id,
                target=agent_id,
            )
        )

        await monitor.emit_subagent_steered(agent_id=agent_id, message=message)
        return f"Steering message sent to {agent_id}. It will be injected on the subagent's next LLM call."


def _suggest_resume(
    ctx: RunContext[AgentContext],
    agent_id: str,
    message: str,
    monitor: BackgroundMonitor,
) -> str:
    agent_info = ctx.deps.agent_registry.get(agent_id)
    agent_name = agent_info.agent_name if agent_info else agent_id.rsplit("-bg-", 1)[0]

    active = [aid for aid, task in monitor.active_tasks.items() if not task.done()]
    active_hint = f" Active tasks: {', '.join(active)}" if active else ""

    prompt_preview = message[:80] + "..." if len(message) > 80 else message
    prompt_preview = prompt_preview.replace('"', '\\"')

    return (
        f"'{agent_id}' has already completed and cannot be steered.{active_hint}\n"
        f"To continue its conversation in the background, use spawn_delegate with agent_id to resume:\n"
        f'  spawn_delegate(subagent_name="{agent_name}", prompt="{prompt_preview}", agent_id="{agent_id}")\n'
        f"Or use blocking delegate for synchronous resume:\n"
        f'  delegate(subagent_name="{agent_name}", prompt="{prompt_preview}", agent_id="{agent_id}")'
    )
