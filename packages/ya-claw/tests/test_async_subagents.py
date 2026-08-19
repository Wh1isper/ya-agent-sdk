from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from pydantic_ai import AgentSpec
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from ya_agent_sdk.inputs import EnqueueReceipt, InputDisposition
from ya_agent_sdk.subagents import SubagentDurability, SubagentExecutionMode, SubagentSpec
from ya_claw.config import ClawSettings
from ya_claw.controller.async_task import AsyncTaskController
from ya_claw.controller.models import (
    AsyncTaskSpawnRequest,
    AsyncTaskSteerRequest,
    CommandPart,
    RunCreateRequest,
)
from ya_claw.controller.run import RunController
from ya_claw.db.engine import create_engine, create_session_factory
from ya_claw.execution.state_machine import complete_run, fail_run, mark_run_running
from ya_claw.execution.subagents import resolve_claw_subagent_plan
from ya_claw.orm.tables import (
    ProfileRecord,
    RunInputInboxRecord,
    RunRecord,
    SessionAsyncTaskRecord,
    SessionRecord,
)
from ya_claw.runtime_state import create_runtime_state


class StubProfile:
    name = "general"
    subagent_specs = (
        SubagentSpec(
            route="explorer",
            agent=AgentSpec(
                model="test",
                name="explorer",
                description="Explore code",
                instructions="You are an explorer.",
                capabilities=["FilesystemCapability"],
            ),
            execution_modes=(SubagentExecutionMode.foreground, SubagentExecutionMode.background),
            durability=SubagentDurability.restart,
        ),
    )


class StubProfileResolver:
    async def resolve(self, profile_name: str | None) -> StubProfile:
        return StubProfile()


@pytest.fixture
async def db_engine(tmp_path: Path, initialize_sqlite_database: Callable[[str], None]) -> AsyncEngine:
    database_url = f"sqlite+aiosqlite:///{(tmp_path / 'async-subagents.sqlite3').resolve()}"
    initialize_sqlite_database(database_url)
    engine = create_engine(database_url)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture
async def db_session(db_engine: AsyncEngine) -> AsyncSession:
    session_factory = create_session_factory(db_engine)
    async with session_factory() as session:
        yield session


@pytest.fixture
def settings(tmp_path: Path) -> ClawSettings:
    data_dir = tmp_path / "runtime-data"
    workspace_dir = tmp_path / "workspace"
    data_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    return ClawSettings(
        api_token="test-token",  # noqa: S106
        data_dir=data_dir,
        workspace_dir=workspace_dir,
    )


async def _create_parent_session(db_session: AsyncSession) -> SessionRecord:
    profile = ProfileRecord(
        name="general",
        agent_spec={"model": "test", "name": "general"},
        host_config={},
        subagent_specs=[],
    )
    parent = SessionRecord(
        id="parent-session", profile_name="general", session_metadata={}, session_type="conversation"
    )
    db_session.add(profile)
    db_session.add(parent)
    await db_session.commit()
    return parent


async def test_async_task_spawn_creates_child_session_and_run(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    await _create_parent_session(db_session)
    runtime_state = create_runtime_state()
    controller = AsyncTaskController()

    response = await controller.spawn_delegate(
        db_session,
        settings,
        runtime_state,
        parent_session_id="parent-session",
        parent_run_id="parent-run",
        request=AsyncTaskSpawnRequest(subagent_name="explorer", name="repo-map", prompt="map repo"),
        profile_resolver=StubProfileResolver(),
    )

    task = response.task
    task_record = await db_session.get(SessionAsyncTaskRecord, task.task_id)
    child_session = await db_session.get(SessionRecord, task.task_session_id)
    child_run = await db_session.get(RunRecord, task.task_run_id)

    assert task.delivery == "submitted"
    assert task.name == "repo-map"
    assert task.status == "queued"
    assert isinstance(task_record, SessionAsyncTaskRecord)
    assert isinstance(child_session, SessionRecord)
    assert isinstance(child_run, RunRecord)
    assert child_session.session_type == "async_task"
    assert child_session.parent_session_id == "parent-session"
    assert child_session.session_metadata["async_task"] == {"task_id": task.task_id}
    assert child_run.trigger_type == "async_task"
    assert child_run.run_metadata["async_task"] == {"task_id": task.task_id}
    assert task_record.plan_descriptor_ref
    assert task_record.plan_descriptor is not None
    assert "subagent_spec" not in task_record.task_metadata
    assert runtime_state.get_run_handle(child_run.id) is not None


async def test_async_task_spawn_rejects_stale_parent_plan_fingerprint(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    await _create_parent_session(db_session)

    with pytest.raises(HTTPException, match="plan is stale") as exc_info:
        await AsyncTaskController().spawn_delegate(
            db_session,
            settings,
            create_runtime_state(),
            parent_session_id="parent-session",
            parent_run_id="parent-run",
            request=AsyncTaskSpawnRequest(
                subagent_name="explorer",
                name="repo-map",
                prompt="map repo",
                context={"plan_fingerprint": "stale-plan"},
            ),
            profile_resolver=StubProfileResolver(),
        )

    assert exc_info.value.status_code == 409
    task_count = (await db_session.execute(select(func.count(SessionAsyncTaskRecord.id)))).scalar_one()
    assert task_count == 0


async def test_async_task_existing_running_returns_instruction(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    await _create_parent_session(db_session)
    runtime_state = create_runtime_state()
    controller = AsyncTaskController()
    first = await controller.spawn_delegate(
        db_session,
        settings,
        runtime_state,
        parent_session_id="parent-session",
        parent_run_id="parent-run",
        request=AsyncTaskSpawnRequest(subagent_name="explorer", name="repo-map", prompt="map repo"),
        profile_resolver=StubProfileResolver(),
    )
    run = await db_session.get(RunRecord, first.task.task_run_id)
    session = await db_session.get(SessionRecord, first.task.task_session_id)
    assert isinstance(run, RunRecord)
    assert isinstance(session, SessionRecord)
    mark_run_running(session, run, claimed_by="test")
    await db_session.commit()

    second = await controller.spawn_delegate(
        db_session,
        settings,
        runtime_state,
        parent_session_id="parent-session",
        parent_run_id="parent-run-2",
        request=AsyncTaskSpawnRequest(subagent_name="explorer", name="repo-map", prompt="continue"),
        profile_resolver=StubProfileResolver(),
    )

    assert second.task.delivery == "existing_active"
    assert second.task.task_id == first.task.task_id
    assert "steer_subagent" in (second.task.instruction or "")


async def test_async_task_steer_running_child_records_input(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    await _create_parent_session(db_session)
    runtime_state = create_runtime_state()
    controller = AsyncTaskController()
    spawned = await controller.spawn_delegate(
        db_session,
        settings,
        runtime_state,
        parent_session_id="parent-session",
        parent_run_id="parent-run",
        request=AsyncTaskSpawnRequest(subagent_name="explorer", name="repo-map", prompt="map repo"),
        profile_resolver=StubProfileResolver(),
    )
    run = await db_session.get(RunRecord, spawned.task.task_run_id)
    session = await db_session.get(SessionRecord, spawned.task.task_session_id)
    assert isinstance(run, RunRecord)
    assert isinstance(session, SessionRecord)
    mark_run_running(session, run, claimed_by="test")
    await db_session.commit()
    received: list[list[dict[str, Any]]] = []

    async def ingress(input_id: str, input_parts: list[dict[str, Any]]) -> EnqueueReceipt:
        received.append(input_parts)
        return EnqueueReceipt(
            logical_run_id="logical-child",
            input_id="input-1",
            disposition=InputDisposition.enqueued,
            enqueue_id="enqueue-1",
        )

    runtime_state.bind_input_ingress(run.id, ingress)

    steered = await controller.steer_task(
        db_session,
        settings,
        runtime_state,
        parent_session_id="parent-session",
        task_id_or_name="repo-map",
        request=AsyncTaskSteerRequest(prompt="focus on tests"),
    )

    assert steered.task.delivery == "steered"
    assert steered.task.input_disposition == "enqueued"
    assert steered.task.input_sdk_id == "input-1"
    assert steered.task.input_enqueue_id == "enqueue-1"
    assert isinstance(steered.task.input_id, str)
    assert steered.task.input_delivery_key == steered.task.input_id
    assert received[-1][0]["text"] == "focus on tests"


async def test_async_task_steer_rejects_queued_and_terminal_children_without_persistence(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    await _create_parent_session(db_session)
    runtime_state = create_runtime_state()
    controller = AsyncTaskController()
    spawned = await controller.spawn_delegate(
        db_session,
        settings,
        runtime_state,
        parent_session_id="parent-session",
        parent_run_id="parent-run",
        request=AsyncTaskSpawnRequest(subagent_name="explorer", name="reject-steer", prompt="work"),
        profile_resolver=StubProfileResolver(),
    )

    with pytest.raises(HTTPException, match="queued and is not accepting") as queued_error:
        await controller.steer_task(
            db_session,
            settings,
            runtime_state,
            parent_session_id="parent-session",
            task_id_or_name=spawned.task.task_id,
            request=AsyncTaskSteerRequest(prompt="too early", idempotency_key="queued-input"),
        )
    assert queued_error.value.status_code == 409

    run = await db_session.get(RunRecord, spawned.task.task_run_id)
    child_session = await db_session.get(SessionRecord, spawned.task.task_session_id)
    assert isinstance(run, RunRecord)
    assert isinstance(child_session, SessionRecord)
    fail_run(child_session, run, finished_at=datetime.now(UTC))
    await db_session.commit()

    with pytest.raises(HTTPException, match="failed and is not accepting") as terminal_error:
        await controller.steer_task(
            db_session,
            settings,
            runtime_state,
            parent_session_id="parent-session",
            task_id_or_name=spawned.task.task_id,
            request=AsyncTaskSteerRequest(prompt="too late", idempotency_key="terminal-input"),
        )
    assert terminal_error.value.status_code == 409
    count = (await db_session.execute(select(func.count(RunInputInboxRecord.id)))).scalar_one()
    assert count == 0


async def test_async_task_terminal_wakes_idle_parent_from_last_run(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    parent = await _create_parent_session(db_session)
    runtime_state = create_runtime_state()
    run_controller = RunController()
    parent_run = await run_controller.create(
        db_session,
        settings,
        runtime_state,
        RunCreateRequest(session_id=parent.id, profile_name="general", input_parts=[{"type": "text", "text": "base"}]),
    )
    parent_record = await db_session.get(SessionRecord, parent.id)
    parent_run_record = await db_session.get(RunRecord, parent_run.id)
    assert isinstance(parent_record, SessionRecord)
    assert isinstance(parent_run_record, RunRecord)
    fail_run(parent_record, parent_run_record, finished_at=datetime.now(UTC))
    runtime_state.clear_run(parent_run.id)
    await db_session.commit()

    controller = AsyncTaskController()
    spawned = await controller.spawn_delegate(
        db_session,
        settings,
        runtime_state,
        parent_session_id=parent.id,
        parent_run_id=parent_run.id,
        request=AsyncTaskSpawnRequest(subagent_name="explorer", name="repo-map", prompt="map repo"),
        profile_resolver=StubProfileResolver(),
    )
    child_session = await db_session.get(SessionRecord, spawned.task.task_session_id)
    child_run = await db_session.get(RunRecord, spawned.task.task_run_id)
    assert isinstance(child_session, SessionRecord)
    assert isinstance(child_run, RunRecord)
    child_run.output_text = "child done"
    complete_run(child_session, child_run, committed_at=datetime.now(UTC))
    submitted: list[str] = []

    await controller.on_run_terminal(
        db_session,
        settings,
        runtime_state,
        run_record=child_run,
        submit_run=submitted.append,
    )

    await db_session.refresh(parent_record)
    task_record = await db_session.get(SessionAsyncTaskRecord, spawned.task.task_id)
    assert isinstance(task_record, SessionAsyncTaskRecord)
    assert task_record.status == "completed"
    assert submitted
    wake_run = await db_session.get(RunRecord, submitted[0])
    assert isinstance(wake_run, RunRecord)
    assert wake_run.session_id == parent.id
    assert wake_run.restore_from_run_id is None
    assert wake_run.input_parts[0]["name"] == "async_task_completed"


async def test_async_task_completed_spawn_resumes_child_session(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    await _create_parent_session(db_session)
    runtime_state = create_runtime_state()
    controller = AsyncTaskController()
    first = await controller.spawn_delegate(
        db_session,
        settings,
        runtime_state,
        parent_session_id="parent-session",
        parent_run_id="parent-run",
        request=AsyncTaskSpawnRequest(subagent_name="explorer", name="repo-map", prompt="map repo"),
        profile_resolver=StubProfileResolver(),
    )
    child_session = await db_session.get(SessionRecord, first.task.task_session_id)
    child_run = await db_session.get(RunRecord, first.task.task_run_id)
    assert isinstance(child_session, SessionRecord)
    assert isinstance(child_run, RunRecord)
    fail_run(child_session, child_run, finished_at=datetime.now(UTC))
    await controller.on_run_terminal(db_session, settings, runtime_state, run_record=child_run, submit_run=None)

    resumed = await controller.spawn_delegate(
        db_session,
        settings,
        runtime_state,
        parent_session_id="parent-session",
        parent_run_id="parent-run-2",
        request=AsyncTaskSpawnRequest(subagent_name="explorer", name="repo-map", prompt="continue"),
        profile_resolver=StubProfileResolver(),
    )

    resumed_run = await db_session.get(RunRecord, resumed.task.task_run_id)
    assert isinstance(resumed_run, RunRecord)
    assert resumed.task.delivery == "resumed"
    assert resumed.task.task_id == first.task.task_id
    assert resumed.task.task_session_id == first.task.task_session_id
    assert resumed_run.restore_from_run_id is None


async def test_async_completion_recovery_reroutes_terminal_parent_once(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    parent = await _create_parent_session(db_session)
    runtime_state = create_runtime_state()
    run_controller = RunController()
    parent_run = await run_controller.create(
        db_session,
        settings,
        runtime_state,
        RunCreateRequest(
            session_id=parent.id,
            profile_name="general",
            input_parts=[{"type": "text", "text": "parent work"}],
        ),
    )
    parent_record = await db_session.get(SessionRecord, parent.id)
    parent_run_record = await db_session.get(RunRecord, parent_run.id)
    assert isinstance(parent_record, SessionRecord)
    assert isinstance(parent_run_record, RunRecord)
    mark_run_running(parent_record, parent_run_record, claimed_by="test")
    await db_session.commit()

    controller = AsyncTaskController()
    spawned = await controller.spawn_delegate(
        db_session,
        settings,
        runtime_state,
        parent_session_id=parent.id,
        parent_run_id=parent_run.id,
        request=AsyncTaskSpawnRequest(
            subagent_name="explorer",
            name="recoverable-result",
            prompt="map repo",
        ),
        profile_resolver=StubProfileResolver(),
    )
    child_session = await db_session.get(SessionRecord, spawned.task.task_session_id)
    child_run = await db_session.get(RunRecord, spawned.task.task_run_id)
    assert isinstance(child_session, SessionRecord)
    assert isinstance(child_run, RunRecord)
    complete_run(child_session, child_run, committed_at=datetime.now(UTC))

    await controller.on_run_terminal(
        db_session,
        settings,
        runtime_state,
        run_record=child_run,
    )
    task_record = await db_session.get(SessionAsyncTaskRecord, spawned.task.task_id)
    assert isinstance(task_record, SessionAsyncTaskRecord)
    assert task_record.delivery_status == "accepted"
    assert task_record.delivery_run_id == parent_run.id

    await run_controller.cancel(
        db_session,
        settings,
        runtime_state,
        parent_run.id,
    )
    await db_session.refresh(task_record)
    assert task_record.delivery_status == "accepted"
    assert task_record.delivery_run_id is None

    submitted: list[str] = []
    await controller.recover_pending_deliveries(
        db_session,
        settings,
        runtime_state,
        submit_run=lambda run_id: submitted.append(run_id) or True,
    )
    await controller.recover_pending_deliveries(
        db_session,
        settings,
        runtime_state,
        submit_run=lambda run_id: submitted.append(run_id) or True,
    )

    delivery_runs = (
        (await db_session.execute(select(RunRecord).where(RunRecord.source_delivery_id == task_record.delivery_id)))
        .scalars()
        .all()
    )
    assert len(delivery_runs) == 1
    assert delivery_runs[0].input_parts[0]["name"] == "async_task_completed"
    assert submitted


async def test_cancelled_async_task_uses_terminal_completion_delivery(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    await _create_parent_session(db_session)
    runtime_state = create_runtime_state()
    controller = AsyncTaskController()
    spawned = await controller.spawn_delegate(
        db_session,
        settings,
        runtime_state,
        parent_session_id="parent-session",
        parent_run_id=None,
        request=AsyncTaskSpawnRequest(
            subagent_name="explorer",
            name="cancel-me",
            prompt="wait",
        ),
        profile_resolver=StubProfileResolver(),
    )
    submitted: list[str] = []

    response = await controller.cancel_task(
        db_session,
        settings,
        runtime_state,
        parent_session_id="parent-session",
        task_id_or_name=spawned.task.task_id,
        submit_run=lambda run_id: submitted.append(run_id) or True,
    )

    task_record = await db_session.get(SessionAsyncTaskRecord, spawned.task.task_id)
    assert isinstance(task_record, SessionAsyncTaskRecord)
    assert response.task.status == "cancelled"
    assert task_record.delivery_status == "enqueued"
    assert isinstance(task_record.delivery_run_id, str)
    assert submitted == [task_record.delivery_run_id]


@pytest.mark.parametrize("termination", ["interrupt_before_claim", "cancel_after_claim"])
async def test_unapplied_completion_run_is_retargeted(
    db_session: AsyncSession,
    settings: ClawSettings,
    termination: str,
) -> None:
    await _create_parent_session(db_session)
    runtime_state = create_runtime_state()
    controller = AsyncTaskController()
    spawned = await controller.spawn_delegate(
        db_session,
        settings,
        runtime_state,
        parent_session_id="parent-session",
        parent_run_id=None,
        request=AsyncTaskSpawnRequest(
            subagent_name="explorer",
            name="retarget-cancelled-wake",
            prompt="wait",
        ),
        profile_resolver=StubProfileResolver(),
    )
    await controller.cancel_task(
        db_session,
        settings,
        runtime_state,
        parent_session_id="parent-session",
        task_id_or_name=spawned.task.task_id,
    )
    task_record = await db_session.get(SessionAsyncTaskRecord, spawned.task.task_id)
    assert isinstance(task_record, SessionAsyncTaskRecord)
    first_delivery_run_id = task_record.delivery_run_id
    assert isinstance(first_delivery_run_id, str)
    first_delivery_run = await db_session.get(RunRecord, first_delivery_run_id)
    assert isinstance(first_delivery_run, RunRecord)
    assert first_delivery_run.status == "queued"
    assert first_delivery_run.started_at is None

    run_controller = RunController()
    if termination == "interrupt_before_claim":
        await run_controller.interrupt(
            db_session,
            settings,
            runtime_state,
            first_delivery_run.id,
        )
    else:
        parent_session = await db_session.get(SessionRecord, first_delivery_run.session_id)
        assert isinstance(parent_session, SessionRecord)
        mark_run_running(parent_session, first_delivery_run, claimed_by="test")
        await db_session.commit()
        await run_controller.cancel(
            db_session,
            settings,
            runtime_state,
            first_delivery_run.id,
        )
    submitted: list[str] = []
    await controller.recover_pending_deliveries(
        db_session,
        settings,
        runtime_state,
        submit_run=lambda run_id: submitted.append(run_id) or True,
    )

    await db_session.refresh(task_record)
    await db_session.refresh(first_delivery_run)
    assert task_record.delivery_status == "enqueued"
    assert isinstance(task_record.delivery_run_id, str)
    assert task_record.delivery_run_id != first_delivery_run.id
    assert first_delivery_run.source_delivery_id is None
    replacement = await db_session.get(RunRecord, task_record.delivery_run_id)
    assert isinstance(replacement, RunRecord)
    assert replacement.source_delivery_id == task_record.delivery_id
    assert replacement.status == "queued"
    assert submitted == [replacement.id]


async def test_running_source_delivery_is_enqueued_not_applied(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = AsyncTaskController()
    parent_session = await _create_parent_session(db_session)
    running = RunRecord(
        id="running-source-delivery",
        session_id=parent_session.id,
        sequence_no=1,
        restore_from_run_id=None,
        status="running",
        trigger_type="async_task",
        profile_name="general",
        input_parts=[],
        run_metadata={},
        source_delivery_id="delivery-running",
    )
    create_record = AsyncMock(return_value=running)
    monkeypatch.setattr(controller._run_controller, "create_record", create_record)

    resolution = await controller._create_wake_run(
        db_session,
        parent_session=parent_session,
        delivery_id="delivery-running",
        wake_part=CommandPart(
            type="command",
            name="async_task_completed",
            params={"task_id": "child"},
        ),
    )

    assert resolution.delivery_status == "enqueued"
    assert resolution.action is None
    create_record.assert_awaited_once()


async def test_model_visible_source_delivery_is_not_retargeted(
    db_session: AsyncSession,
) -> None:
    source_run = RunRecord(
        id="applied-source-delivery",
        session_id="parent-session",
        sequence_no=1,
        restore_from_run_id=None,
        status="cancelled",
        trigger_type="async_task",
        profile_name="general",
        input_parts=[],
        run_metadata={},
        source_delivery_id="delivery-applied",
        source_delivery_applied_at=datetime.now(UTC),
    )
    resolution = await AsyncTaskController()._resolve_source_wake(
        db_session,
        source_run,
    )

    assert resolution is not None
    assert resolution.delivery_status == "applied"
    assert source_run.source_delivery_id == "delivery-applied"


async def test_async_steer_idempotency_key_is_forwarded(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    await _create_parent_session(db_session)
    runtime_state = create_runtime_state()
    controller = AsyncTaskController()
    spawned = await controller.spawn_delegate(
        db_session,
        settings,
        runtime_state,
        parent_session_id="parent-session",
        parent_run_id=None,
        request=AsyncTaskSpawnRequest(
            subagent_name="explorer",
            name="idempotent-steer",
            prompt="map",
        ),
        profile_resolver=StubProfileResolver(),
    )
    run = await db_session.get(RunRecord, spawned.task.task_run_id)
    session = await db_session.get(SessionRecord, spawned.task.task_session_id)
    assert isinstance(run, RunRecord)
    assert isinstance(session, SessionRecord)
    mark_run_running(session, run, claimed_by="test")
    await db_session.commit()
    request = AsyncTaskSteerRequest(
        prompt="once",
        idempotency_key="tool-call-1",
    )

    first = await controller.steer_task(
        db_session,
        settings,
        runtime_state,
        parent_session_id="parent-session",
        task_id_or_name=spawned.task.task_id,
        request=request,
    )
    second = await controller.steer_task(
        db_session,
        settings,
        runtime_state,
        parent_session_id="parent-session",
        task_id_or_name=spawned.task.task_id,
        request=request,
    )

    count = (await db_session.execute(select(func.count()).where(RunInputInboxRecord.run_id == run.id))).scalar_one()
    assert count == 1
    assert first.task.input_id == second.task.input_id
    assert first.task.input_delivery_key == second.task.input_delivery_key == "tool-call-1"
    assert first.task.input_disposition == second.task.input_disposition
    assert first.task.input_sdk_id == second.task.input_sdk_id
    assert first.task.input_enqueue_id == second.task.input_enqueue_id


async def test_named_resume_rejects_missing_plan_descriptor_before_new_run(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    await _create_parent_session(db_session)
    runtime_state = create_runtime_state()
    controller = AsyncTaskController()
    spawned = await controller.spawn_delegate(
        db_session,
        settings,
        runtime_state,
        parent_session_id="parent-session",
        parent_run_id=None,
        request=AsyncTaskSpawnRequest(
            subagent_name="explorer",
            name="legacy-task",
            prompt="map",
        ),
        profile_resolver=StubProfileResolver(),
    )
    task_record = await db_session.get(SessionAsyncTaskRecord, spawned.task.task_id)
    child_session = await db_session.get(SessionRecord, spawned.task.task_session_id)
    child_run = await db_session.get(RunRecord, spawned.task.task_run_id)
    assert isinstance(task_record, SessionAsyncTaskRecord)
    assert isinstance(child_session, SessionRecord)
    assert isinstance(child_run, RunRecord)
    fail_run(child_session, child_run, finished_at=datetime.now(UTC))
    task_record.status = "failed"
    task_record.plan_descriptor = None
    await db_session.commit()
    before_count = (
        await db_session.execute(select(func.count()).where(RunRecord.session_id == child_session.id))
    ).scalar_one()

    with pytest.raises(Exception, match="plan descriptor"):
        await controller.spawn_delegate(
            db_session,
            settings,
            runtime_state,
            parent_session_id="parent-session",
            parent_run_id=None,
            request=AsyncTaskSpawnRequest(
                subagent_name="explorer",
                name="legacy-task",
                prompt="continue",
            ),
            profile_resolver=StubProfileResolver(),
        )

    after_count = (
        await db_session.execute(select(func.count()).where(RunRecord.session_id == child_session.id))
    ).scalar_one()
    assert after_count == before_count


def test_subagent_resolution_rejects_native_host_policy_collision() -> None:
    spec = SubagentSpec(
        route="worker",
        agent=AgentSpec(
            model="test",
            name="worker",
            capabilities=[{"ToolTimeoutCapability": {"timeout": 5}}],
        ),
        execution_modes=(SubagentExecutionMode.background,),
        durability=SubagentDurability.restart,
    )

    with pytest.raises(ValueError, match=r"Capability id 'tool_timeout'.*multiple"):
        resolve_claw_subagent_plan(spec)
