from __future__ import annotations

import asyncio

from ya_claw.runtime_state import ActiveRunHandle, InMemoryRuntimeState, create_runtime_state


async def _collect_events(state: InMemoryRuntimeState, run_id: str) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    async for event in state.stream_run_events(run_id):
        events.append(event)
    return events


async def test_stream_run_events_closes_after_terminal_event() -> None:
    state = create_runtime_state()
    state.register_run("session-1", "run-1")

    consumer = asyncio.create_task(_collect_events(state, "run-1"))
    await asyncio.sleep(0)
    await state.append_run_event("run-1", {"type": "run.created"})
    await state.append_run_event("run-1", {"type": "run.cancelled"}, terminal=True)

    events = await asyncio.wait_for(consumer, timeout=1)

    assert [event["event"] for event in events] == ["run.created", "run.cancelled"]
    handle = state.get_run_handle("run-1")
    assert isinstance(handle, ActiveRunHandle)
    assert handle.closed is True


async def test_stream_run_events_replays_from_last_event_id() -> None:
    state = create_runtime_state()
    state.register_run("session-1", "run-1")
    await state.append_run_event("run-1", {"type": "run.created"})
    await state.append_run_event("run-1", {"type": "run.started"})
    await state.append_run_event("run-1", {"type": "run.completed"}, terminal=True)

    events: list[dict[str, str]] = []
    async for event in state.stream_run_events("run-1", last_event_id="1"):
        events.append(event)

    assert [event["event"] for event in events] == ["run.started", "run.completed"]


async def test_runtime_state_aclose_releases_waiting_subscribers() -> None:
    state = create_runtime_state()
    state.register_run("session-1", "run-1")

    consumer = asyncio.create_task(_collect_events(state, "run-1"))
    await asyncio.sleep(0.05)
    assert state.subscribers == 1

    await state.aclose()
    events = await asyncio.wait_for(consumer, timeout=1)

    assert events == []
    assert state.subscribers == 0


async def test_runtime_state_hitl_waits_until_all_interactions_resolve() -> None:
    from datetime import UTC, datetime

    from ya_claw.controller.models import ActiveInteraction

    state = create_runtime_state()
    state.register_run("session-1", "run-1")
    interactions = [
        ActiveInteraction(
            interaction_id="hitl-1",
            run_id="run-1",
            session_id="session-1",
            tool_call_id="tool-1",
            tool_name="shell_exec",
            title="Approve shell",
            sequence_no=1,
            total_count=2,
            created_at=datetime.now(UTC),
        ),
        ActiveInteraction(
            interaction_id="hitl-2",
            run_id="run-1",
            session_id="session-1",
            tool_call_id="tool-2",
            tool_name="file_write",
            title="Approve write",
            sequence_no=2,
            total_count=2,
            created_at=datetime.now(UTC),
        ),
    ]
    state.set_hitl_pending("run-1", "session-1", interactions)

    waiter = asyncio.create_task(state.wait_hitl_batch("run-1"))
    await asyncio.sleep(0)
    assert not waiter.done()

    resolved, current, remaining = await state.resolve_hitl_interaction("run-1", "hitl-1", approved=True)
    assert resolved.status == "approved"
    assert current is not None
    assert current.interaction_id == "hitl-2"
    assert remaining == 1
    assert not waiter.done()

    await state.resolve_hitl_interaction("run-1", "hitl-2", approved=False, reason="no")
    results = await asyncio.wait_for(waiter, timeout=1)

    assert [item.tool_call_id for item in results] == ["tool-1", "tool-2"]
    assert results[0].approved is True
    assert results[1].approved is False
    assert results[1].reason == "no"
