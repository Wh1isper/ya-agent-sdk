from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ya_claw.agui_adapter import AguiReplayBuffer
from ya_claw.controller.models import ActiveInteraction, UserInteraction
from ya_claw.json_types import JsonValue


@dataclass(slots=True)
class BufferedEvent:
    id: str
    payload: dict[str, Any]
    terminal: bool = False


@dataclass(slots=True)
class ActiveRunHandle:
    run_id: str
    session_id: str
    dispatch_mode: str = "async"
    steering_inputs: list[list[dict[str, Any]]] = field(default_factory=list)
    events: list[BufferedEvent] = field(default_factory=list)
    replay: AguiReplayBuffer = field(default_factory=AguiReplayBuffer)
    next_event_id: int = 1
    closed: bool = False
    terminal_event_seen: bool = False
    terminal_event_consumed: bool = False
    termination_requested: str | None = None
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)


@dataclass(slots=True)
class HitlRunState:
    run_id: str
    session_id: str
    interactions: list[ActiveInteraction]
    resolved: dict[str, UserInteraction] = field(default_factory=dict)
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)

    @property
    def current_interaction(self) -> ActiveInteraction | None:
        for interaction in self.interactions:
            if interaction.interaction_id not in self.resolved:
                return interaction
        return None

    @property
    def remaining_count(self) -> int:
        return sum(1 for interaction in self.interactions if interaction.interaction_id not in self.resolved)


@dataclass(slots=True)
class InMemoryRuntimeState:
    run_handles: dict[str, ActiveRunHandle] = field(default_factory=dict)
    session_latest_run_ids: dict[str, str] = field(default_factory=dict)
    background_tasks: dict[str, asyncio.Task[None]] = field(default_factory=dict)
    hitl_states: dict[str, HitlRunState] = field(default_factory=dict)
    cleanup_tasks: dict[str, asyncio.Task[None]] = field(default_factory=dict)
    session_locks: dict[str, asyncio.Lock] = field(default_factory=dict)
    subscribers: int = 0

    def session_lock(self, session_id: str) -> asyncio.Lock:
        lock = self.session_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self.session_locks[session_id] = lock
        return lock

    def register_run(self, session_id: str, run_id: str, *, dispatch_mode: str = "async") -> ActiveRunHandle:
        self.prune_closed_runs()
        handle = ActiveRunHandle(run_id=run_id, session_id=session_id, dispatch_mode=dispatch_mode)
        self.run_handles[run_id] = handle
        self.session_latest_run_ids[session_id] = run_id
        return handle

    def get_run_handle(self, run_id: str) -> ActiveRunHandle | None:
        return self.run_handles.get(run_id)

    def get_session_run_handle(self, session_id: str) -> ActiveRunHandle | None:
        run_id = self.session_latest_run_ids.get(session_id)
        if run_id is None:
            return None
        return self.get_run_handle(run_id)

    def clear_run(self, run_id: str) -> None:
        cleanup_task = self.cleanup_tasks.pop(run_id, None)
        try:
            current_task = asyncio.current_task()
        except RuntimeError:
            current_task = None
        if cleanup_task is not None and cleanup_task is not current_task and not cleanup_task.done():
            cleanup_task.cancel()
        handle = self.run_handles.pop(run_id, None)
        if handle is None:
            return
        if self.session_latest_run_ids.get(handle.session_id) == run_id:
            self.session_latest_run_ids.pop(handle.session_id, None)
        self.background_tasks.pop(run_id, None)
        self.hitl_states.pop(run_id, None)

    def schedule_run_cleanup(self, run_id: str, *, delay_seconds: float = 30.0) -> None:
        handle = self.get_run_handle(run_id)
        if not isinstance(handle, ActiveRunHandle) or not handle.closed:
            return
        existing_task = self.cleanup_tasks.pop(run_id, None)
        if existing_task is not None and not existing_task.done():
            existing_task.cancel()
        task = asyncio.create_task(
            self._cleanup_closed_run_after_delay(run_id, max(delay_seconds, 0.0)),
            name=f"ya-claw-run-cleanup-{run_id}",
        )
        self.cleanup_tasks[run_id] = task
        task.add_done_callback(lambda _: self.cleanup_tasks.pop(run_id, None))

    async def _cleanup_closed_run_after_delay(self, run_id: str, delay_seconds: float) -> None:
        await asyncio.sleep(delay_seconds)
        handle = self.get_run_handle(run_id)
        if isinstance(handle, ActiveRunHandle) and handle.closed:
            self.clear_run(run_id)

    def prune_closed_runs(self) -> None:
        clearable_run_ids = [
            run_id for run_id, handle in self.run_handles.items() if handle.closed and handle.terminal_event_consumed
        ]
        for run_id in clearable_run_ids:
            self.clear_run(run_id)

    def get_replay_events(self, run_id: str) -> list[dict[str, Any]]:
        handle = self.get_run_handle(run_id)
        if not isinstance(handle, ActiveRunHandle):
            raise KeyError(run_id)
        return handle.replay.snapshot()

    def register_background_task(self, run_id: str, task: asyncio.Task[None]) -> None:
        self.background_tasks[run_id] = task

    def get_background_task(self, run_id: str) -> asyncio.Task[None] | None:
        return self.background_tasks.get(run_id)

    def clear_background_task(self, run_id: str) -> None:
        self.background_tasks.pop(run_id, None)

    def set_hitl_pending(self, run_id: str, session_id: str, interactions: list[ActiveInteraction]) -> HitlRunState:
        state = HitlRunState(run_id=run_id, session_id=session_id, interactions=list(interactions))
        self.hitl_states[run_id] = state
        return state

    def get_hitl_state(self, run_id: str) -> HitlRunState | None:
        return self.hitl_states.get(run_id)

    def get_hitl_state_by_session(self, session_id: str) -> HitlRunState | None:
        run_id = self.session_latest_run_ids.get(session_id)
        if run_id is None:
            return None
        return self.hitl_states.get(run_id)

    async def resolve_hitl_interaction(
        self,
        run_id: str,
        interaction_id: str,
        *,
        approved: bool,
        reason: str | None = None,
        user_input: JsonValue = None,
    ) -> tuple[ActiveInteraction, ActiveInteraction | None, int]:
        state = self.hitl_states.get(run_id)
        if state is None:
            raise KeyError(run_id)
        target = next((item for item in state.interactions if item.interaction_id == interaction_id), None)
        if target is None:
            raise KeyError(interaction_id)
        async with state.condition:
            if interaction_id not in state.resolved:
                state.resolved[interaction_id] = UserInteraction(
                    tool_call_id=target.tool_call_id,
                    approved=approved,
                    reason=reason,
                    user_input=user_input,
                )
                target.status = "approved" if approved else "denied"
                target.resolved_at = datetime.now(UTC)
                state.condition.notify_all()
            return target, state.current_interaction, state.remaining_count

    async def wait_hitl_batch(self, run_id: str) -> list[UserInteraction]:
        state = self.hitl_states.get(run_id)
        if state is None:
            raise KeyError(run_id)
        async with state.condition:
            while state.remaining_count > 0:
                await state.condition.wait()
            return [state.resolved[item.interaction_id] for item in state.interactions]

    def clear_hitl(self, run_id: str) -> None:
        self.hitl_states.pop(run_id, None)

    async def append_run_event(
        self,
        run_id: str,
        payload: dict[str, Any],
        *,
        terminal: bool = False,
        replay: bool = True,
    ) -> BufferedEvent:
        handle = self.get_run_handle(run_id)
        if not isinstance(handle, ActiveRunHandle):
            raise KeyError(run_id)

        if replay:
            handle.replay.append(payload)

        event = BufferedEvent(id=str(handle.next_event_id), payload=payload, terminal=terminal)
        handle.next_event_id += 1
        handle.events.append(event)
        if terminal:
            handle.closed = True
            handle.terminal_event_seen = True

        async with handle.condition:
            handle.condition.notify_all()

        return event

    async def record_steering(self, run_id: str, input_parts: list[dict[str, Any]]) -> None:
        handle = self.get_run_handle(run_id)
        if not isinstance(handle, ActiveRunHandle):
            raise KeyError(run_id)
        handle.steering_inputs.append(list(input_parts))

    def consume_steering_inputs(self, run_id: str) -> list[list[dict[str, Any]]]:
        handle = self.get_run_handle(run_id)
        if not isinstance(handle, ActiveRunHandle):
            return []
        pending = list(handle.steering_inputs)
        handle.steering_inputs.clear()
        return pending

    def get_termination_requested(self, run_id: str) -> str | None:
        handle = self.get_run_handle(run_id)
        if not isinstance(handle, ActiveRunHandle):
            return None
        return handle.termination_requested

    async def request_stop(self, run_id: str, termination_reason: str) -> None:
        handle = self.get_run_handle(run_id)
        if not isinstance(handle, ActiveRunHandle):
            raise KeyError(run_id)
        handle.termination_requested = termination_reason
        async with handle.condition:
            handle.condition.notify_all()
        hitl_state = self.hitl_states.get(run_id)
        if hitl_state is not None:
            async with hitl_state.condition:
                hitl_state.condition.notify_all()

    async def close_run(self, run_id: str) -> None:
        handle = self.get_run_handle(run_id)
        if not isinstance(handle, ActiveRunHandle):
            return
        handle.closed = True
        async with handle.condition:
            handle.condition.notify_all()

    async def stream_run_events(self, run_id: str, last_event_id: str | None = None) -> AsyncIterator[dict[str, str]]:
        handle = self.get_run_handle(run_id)
        if not isinstance(handle, ActiveRunHandle):
            raise KeyError(run_id)

        cursor = _resolve_cursor(last_event_id)
        self.subscribers += 1
        try:
            while True:
                async with handle.condition:
                    while cursor >= len(handle.events) and not handle.closed:
                        await handle.condition.wait()
                    pending_events = list(handle.events[cursor:])
                    handle_closed = handle.closed

                for event in pending_events:
                    cursor += 1
                    yield {
                        "id": event.id,
                        "event": str(event.payload.get("type", "message")),
                        "data": json.dumps(event.payload, ensure_ascii=False),
                    }
                    if event.terminal:
                        handle.terminal_event_consumed = True
                        return

                if handle_closed:
                    if handle.terminal_event_seen and cursor >= len(handle.events):
                        handle.terminal_event_consumed = True
                    return
        finally:
            self.subscribers = max(self.subscribers - 1, 0)
            self.prune_closed_runs()

    async def stream_session_events(
        self,
        session_id: str,
        last_event_id: str | None = None,
    ) -> AsyncIterator[dict[str, str]]:
        handle = self.get_session_run_handle(session_id)
        if not isinstance(handle, ActiveRunHandle):
            raise KeyError(session_id)
        async for event in self.stream_run_events(handle.run_id, last_event_id=last_event_id):
            yield event

    async def aclose(self) -> None:
        background_tasks = list(self.background_tasks.values())
        cleanup_tasks = list(self.cleanup_tasks.values())
        for task in [*background_tasks, *cleanup_tasks]:
            if not task.done():
                task.cancel()
        if background_tasks or cleanup_tasks:
            await asyncio.gather(*background_tasks, *cleanup_tasks, return_exceptions=True)

        for hitl_state in self.hitl_states.values():
            async with hitl_state.condition:
                hitl_state.condition.notify_all()
        for handle in self.run_handles.values():
            handle.closed = True
            async with handle.condition:
                handle.condition.notify_all()
        self.run_handles.clear()
        self.session_latest_run_ids.clear()
        self.background_tasks.clear()
        self.hitl_states.clear()
        self.cleanup_tasks.clear()
        self.subscribers = 0


def _resolve_cursor(last_event_id: str | None) -> int:
    if last_event_id is None:
        return 0
    try:
        return max(int(last_event_id), 0)
    except ValueError:
        return 0


def create_runtime_state() -> InMemoryRuntimeState:
    return InMemoryRuntimeState()
