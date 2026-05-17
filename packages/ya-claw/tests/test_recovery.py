from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from ya_agent_environment import Environment
from ya_claw.config import ClawSettings
from ya_claw.db.engine import create_engine, create_session_factory
from ya_claw.execution.coordinator import ExecutionSupervisor
from ya_claw.execution.profile import ResolvedProfile
from ya_claw.execution.state_machine import mark_run_running
from ya_claw.orm.base import Base
from ya_claw.orm.tables import RunRecord, SessionRecord
from ya_claw.runtime_state import InMemoryRuntimeState, create_runtime_state
from ya_claw.workspace import LocalWorkspaceProvider, WorkspaceBinding


class StubProfileResolver:
    async def resolve(self, profile_name: str | None) -> ResolvedProfile:
        return ResolvedProfile(
            name=profile_name or "default",
            model="test",
            model_settings=None,
            model_config=None,
        )


class StubEnvironment(Environment):
    async def _setup(self) -> None:
        return None

    async def _teardown(self) -> None:
        return None


class StubEnvironmentFactory:
    def build(self, binding: WorkspaceBinding) -> Environment:
        return StubEnvironment()


class StubRuntimeBuilder:
    def build(self, **_: object) -> object:
        return object()


@pytest.fixture
async def db_engine(tmp_path: Path) -> AsyncEngine:
    engine = create_engine(f"sqlite+aiosqlite:///{(tmp_path / 'recovery.sqlite3').resolve()}")
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
    return ClawSettings(
        api_token="test-token",  # noqa: S106
        data_dir=tmp_path / "runtime-data",
        workspace_dir=tmp_path / "workspace",
        instance_id="instance-test",
    )


@pytest.fixture
def runtime_state() -> InMemoryRuntimeState:
    return create_runtime_state()


def _build_supervisor(
    *,
    settings: ClawSettings,
    db_engine: AsyncEngine,
    runtime_state: InMemoryRuntimeState,
) -> ExecutionSupervisor:
    return ExecutionSupervisor(
        settings=settings,
        session_factory=create_session_factory(db_engine),
        runtime_state=runtime_state,
        workspace_provider=LocalWorkspaceProvider(settings.resolved_workspace_dir),
        environment_factory=StubEnvironmentFactory(),
        profile_resolver=StubProfileResolver(),
        runtime_builder=StubRuntimeBuilder(),
    )


async def test_supervisor_claim_records_instance_owner(
    db_session: AsyncSession,
    db_engine: AsyncEngine,
    settings: ClawSettings,
    runtime_state: InMemoryRuntimeState,
) -> None:
    session_record = SessionRecord(id="session-1", profile_name="default", session_metadata={})
    run_record = RunRecord(
        id="run-1",
        session_id="session-1",
        sequence_no=1,
        status="queued",
        trigger_type="api",
        profile_name="default",
        input_parts=[],
        run_metadata={},
    )
    db_session.add(session_record)
    db_session.add(run_record)
    await db_session.commit()
    runtime_state.register_run("session-1", "run-1")
    supervisor = _build_supervisor(settings=settings, db_engine=db_engine, runtime_state=runtime_state)

    claimed = await supervisor._claim_run("run-1")

    await db_session.refresh(run_record)
    assert claimed is True
    assert run_record.status == "running"
    assert run_record.claimed_by == "instance-test"
    assert run_record.claimed_at is not None


async def test_supervisor_startup_recovery_cancels_orphan_running_and_submits_queued(
    db_session: AsyncSession,
    db_engine: AsyncEngine,
    settings: ClawSettings,
    runtime_state: InMemoryRuntimeState,
) -> None:
    session_a = SessionRecord(id="session-a", profile_name="default", session_metadata={})
    session_b = SessionRecord(id="session-b", profile_name="default", session_metadata={})
    running_run = RunRecord(
        id="run-running",
        session_id="session-a",
        sequence_no=1,
        status="queued",
        trigger_type="api",
        profile_name="default",
        input_parts=[],
        run_metadata={},
    )
    queued_run = RunRecord(
        id="run-queued",
        session_id="session-b",
        sequence_no=1,
        status="queued",
        trigger_type="api",
        profile_name="default",
        input_parts=[],
        run_metadata={},
    )
    db_session.add_all([session_a, session_b, running_run, queued_run])
    mark_run_running(session_a, running_run, claimed_by="dead-instance")
    await db_session.commit()
    supervisor = _build_supervisor(settings=settings, db_engine=db_engine, runtime_state=runtime_state)

    result = await supervisor.startup_recover()

    await db_session.refresh(running_run)
    await db_session.refresh(session_a)
    assert result["cancelled_running"] == ["run-running"]
    assert result["submitted_queued"] == ["run-queued"]
    assert running_run.status == "cancelled"
    assert running_run.termination_reason == "interrupt"
    assert running_run.error_message is not None
    assert session_a.active_run_id is None
    assert runtime_state.get_background_task("run-queued") is not None

    task = runtime_state.get_background_task("run-queued")
    if task is not None:
        task.cancel()
        await runtime_state.aclose()
