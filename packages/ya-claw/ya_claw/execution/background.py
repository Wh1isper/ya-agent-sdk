"""Run-scoped background subagent monitor for YA Claw runtime."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from loguru import logger
from ya_agent_environment import BaseResource

if TYPE_CHECKING:
    from ya_agent_sdk.toolsets.core.base import BaseTool, Toolset

    from ya_claw.runtime_state import InMemoryRuntimeState

BACKGROUND_MONITOR_KEY = "background_monitor"


@dataclass(slots=True)
class BackgroundTaskInfo:
    agent_id: str
    subagent_name: str
    prompt: str
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    is_resume: bool = False


class BackgroundTaskAlreadyActiveError(ValueError):
    """Raised when a background task is already active for an agent id."""


class BackgroundMonitor(BaseResource):
    """Tracks background subagent tasks for one YA Claw run."""

    def __init__(self, *, run_id: str, runtime_state: InMemoryRuntimeState) -> None:
        self._run_id = run_id
        self._runtime_state = runtime_state
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._task_info: dict[str, BackgroundTaskInfo] = {}
        self._core_toolset: Toolset[Any] | None = None
        self._closed = False

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def closed(self) -> bool:
        return self._closed

    def set_core_toolset(self, toolset: Toolset[Any] | None) -> None:
        self._core_toolset = toolset

    def get_delegate_tool(self) -> BaseTool | None:
        if self._core_toolset is None:
            return None
        try:
            return self._core_toolset._get_tool_instance("delegate")
        except Exception:
            return None

    @property
    def has_delegate_tool(self) -> bool:
        return self.get_delegate_tool() is not None

    @property
    def has_active_tasks(self) -> bool:
        return any(not task.done() for task in self._tasks.values())

    @property
    def active_tasks(self) -> dict[str, asyncio.Task[Any]]:
        return dict(self._tasks)

    @property
    def task_infos(self) -> dict[str, BackgroundTaskInfo]:
        return dict(self._task_info)

    def register_task(
        self,
        agent_id: str,
        task: asyncio.Task[Any],
        *,
        subagent_name: str = "",
        prompt: str = "",
        is_resume: bool = False,
    ) -> None:
        existing = self._tasks.get(agent_id)
        if existing is not None and not existing.done():
            raise BackgroundTaskAlreadyActiveError(f"Background task already active: {agent_id}")

        self._tasks[agent_id] = task
        self._task_info[agent_id] = BackgroundTaskInfo(
            agent_id=agent_id,
            subagent_name=subagent_name,
            prompt=prompt,
            is_resume=is_resume,
        )
        task.add_done_callback(lambda _task: self._on_task_done(agent_id))
        logger.debug("Registered background task agent_id={} subagent_name={}", agent_id, subagent_name)

    def _on_task_done(self, agent_id: str) -> None:
        task = self._tasks.get(agent_id)
        if task is not None and task.done():
            self._tasks.pop(agent_id, None)
            self._task_info.pop(agent_id, None)
        logger.debug("Background task completed agent_id={}", agent_id)

    async def emit_subagent_spawned(
        self,
        agent_id: str,
        subagent_name: str,
        prompt: str,
    ) -> None:
        await self._emit_event({
            "type": "ya_claw.subagent_spawned",
            "agent_id": agent_id,
            "subagent_name": subagent_name,
            "prompt_preview": prompt[:100] + "..." if len(prompt) > 100 else prompt,
        })

    async def emit_subagent_completed(
        self,
        agent_id: str,
        subagent_name: str,
        duration_seconds: float,
        result_preview: str | None = None,
    ) -> None:
        await self._emit_event({
            "type": "ya_claw.subagent_completed",
            "agent_id": agent_id,
            "subagent_name": subagent_name,
            "duration_seconds": duration_seconds,
            "result_preview": result_preview,
        })

    async def emit_subagent_failed(
        self,
        agent_id: str,
        subagent_name: str,
        error: str,
    ) -> None:
        await self._emit_event({
            "type": "ya_claw.subagent_failed",
            "agent_id": agent_id,
            "subagent_name": subagent_name,
            "error": error[:200],
        })

    async def emit_subagent_cancelled(
        self,
        agent_id: str,
        subagent_name: str,
    ) -> None:
        await self._emit_event({
            "type": "ya_claw.subagent_cancelled",
            "agent_id": agent_id,
            "subagent_name": subagent_name,
        })

    async def emit_subagent_steered(
        self,
        agent_id: str,
        message: str,
    ) -> None:
        await self._emit_event({
            "type": "ya_claw.subagent_steered",
            "agent_id": agent_id,
            "message_preview": message[:100] + "..." if len(message) > 100 else message,
        })

    async def _emit_event(self, payload: dict[str, Any]) -> None:
        await self._runtime_state.append_run_event(self._run_id, payload)

    def get_context_instruction(self) -> str | None:
        running = [
            (agent_id, info)
            for agent_id, info in self._task_info.items()
            if agent_id in self._tasks and not self._tasks[agent_id].done()
        ]
        if not running:
            return None
        lines = ["<background-tasks>"]
        for agent_id, info in running:
            lines.append(f'  <task agent-id="{agent_id}" name="{info.subagent_name}" status="running"/>')
        lines.append("</background-tasks>")
        return "\n".join(lines)

    async def wait_for_all(self, timeout: float | None = None) -> bool:
        tasks = [task for task in self._tasks.values() if not task.done()]
        if not tasks:
            return True
        done, pending = await asyncio.wait(tasks, timeout=timeout)
        for task in done:
            with contextlib.suppress(Exception):
                task.result()
        return not pending

    async def cancel_all(self) -> None:
        task_items = [
            (agent_id, task, self._task_info.get(agent_id)) for agent_id, task in self._tasks.items() if not task.done()
        ]
        for _agent_id, task, _info in task_items:
            task.cancel()
        if task_items:
            await asyncio.gather(*(task for _agent_id, task, _info in task_items), return_exceptions=True)
        for agent_id, task, info in task_items:
            if task.cancelled():
                await self.emit_subagent_cancelled(
                    agent_id=agent_id,
                    subagent_name=info.subagent_name if info is not None else "",
                )
        self._tasks.clear()
        self._task_info.clear()

    async def drain_or_cancel(self, *, timeout: float | None = None) -> bool:
        completed = await self.wait_for_all(timeout=timeout)
        if completed:
            return True
        await self.cancel_all()
        return False

    async def close(self) -> None:
        self._closed = True
        await self.cancel_all()
        self._core_toolset = None
