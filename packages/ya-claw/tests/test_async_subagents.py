from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from ya_agent_sdk.subagents.config import SubagentConfig
from ya_claw.config import ClawSettings
from ya_claw.controller.async_task import AsyncTaskController
from ya_claw.controller.models import AsyncTaskSpawnRequest, AsyncTaskSteerRequest, RunCreateRequest
from ya_claw.controller.run import RunController
from ya_claw.db.engine import create_engine, create_session_factory
from ya_claw.execution.state_machine import complete_run, fail_run, mark_run_running
from ya_claw.orm.base import Base
from ya_claw.orm.tables import ProfileRecord, RunRecord, SessionAsyncTaskRecord, SessionRecord
from ya_claw.runtime_state import create_runtime_state


class StubProfile:
    name = "general"
    include_builtin_subagents = False
    subagent_configs: ClassVar[list[SubagentConfig]] = [
        SubagentConfig(
            name="explorer",
            description="Explore code",
            system_prompt="You are an explorer.",
            tools=["view", "grep"],
        )
    ]


class StubProfileResolver:
    async def resolve(self, profile_name: str | None) -> StubProfile:
        return StubProfile()


@pytest.fixture
async def db_engine(tmp_path: Path) -> AsyncEngine:
    engine = create_engine(f"sqlite+aiosqlite:///{(tmp_path / 'async-subagents.sqlite3').resolve()}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
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
        model="test",
        subagents=[
            {
                "name": "explorer",
                "description": "Explore code",
                "system_prompt": "You are an explorer.",
                "tools": ["view", "grep"],
            }
        ],
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
    assert child_session.session_metadata["async_task"]["subagent_name"] == "explorer"
    assert child_run.trigger_type == "async_task"
    assert child_run.run_metadata["async_task"]["name"] == "repo-map"
    assert runtime_state.get_run_handle(child_run.id) is not None


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
    assert "steer_async_subagent" in (second.task.instruction or "")


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

    steered = await controller.steer_task(
        db_session,
        settings,
        runtime_state,
        parent_session_id="parent-session",
        task_id_or_name="repo-map",
        request=AsyncTaskSteerRequest(prompt="focus on tests"),
    )

    handle = runtime_state.get_run_handle(run.id)
    assert steered.task.delivery == "steered"
    assert handle is not None
    assert handle.steering_inputs[-1][0]["text"] == "focus on tests"


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
    child_run.output_summary = "child done"
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
    assert task_record.result_summary == "child done"
    assert submitted
    wake_run = await db_session.get(RunRecord, submitted[0])
    assert isinstance(wake_run, RunRecord)
    assert wake_run.session_id == parent.id
    assert wake_run.restore_from_run_id == parent_run.id
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
    assert resumed_run.restore_from_run_id == child_run.id
