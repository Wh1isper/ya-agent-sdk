from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from ya_claw.agency.lifecycle import AgencyLifecycle
from ya_claw.config import ClawSettings
from ya_claw.controller.models import AgencySignalReason, AgencySignalRequest, TriggerType
from ya_claw.db.engine import create_engine, create_session_factory
from ya_claw.orm.base import Base
from ya_claw.orm.tables import AgencySignalRecord, RunRecord, SessionAgencyStateRecord, SessionRecord
from ya_claw.runtime_state import create_runtime_state


async def test_agency_manual_signal_creates_paired_session_and_run(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    source_session = SessionRecord(id="session-1", profile_name="general", session_metadata={})
    db_session.add(source_session)
    await db_session.commit()
    submitted: list[str] = []
    runtime_state = create_runtime_state()
    lifecycle = AgencyLifecycle(
        settings=settings,
        runtime_state=runtime_state,
        submit_run=lambda run_id: not submitted.append(run_id),
    )

    client_marker = "manual-1"
    delivery = await lifecycle.create_signal(
        db_session,
        "session-1",
        AgencySignalRequest(reason=AgencySignalReason.MANUAL, client_token=client_marker),
    )

    state = await db_session.get(SessionAgencyStateRecord, "session-1")
    assert isinstance(state, SessionAgencyStateRecord)
    assert state.enabled is True
    assert isinstance(state.agency_session_id, str)
    agency_session = await db_session.get(SessionRecord, state.agency_session_id)
    assert isinstance(agency_session, SessionRecord)
    assert agency_session.session_type == "agency"
    assert agency_session.source_session_id == "session-1"
    assert delivery.delivery == "submitted"
    assert delivery.run_id is not None
    assert submitted == [delivery.run_id]
    run = await db_session.get(RunRecord, delivery.run_id)
    assert isinstance(run, RunRecord)
    assert run.trigger_type == TriggerType.AGENCY.value
    assert run.run_metadata["agency"]["source_session_id"] == "session-1"
    assert run.input_parts[0]["name"] == "agency_signal"


async def test_agency_signal_steers_active_agency_run(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    source_session = SessionRecord(id="session-1", profile_name="general", session_metadata={})
    agency_session = SessionRecord(
        id="agency-1",
        parent_session_id="session-1",
        profile_name="general",
        session_type="agency",
        source_session_id="session-1",
        session_metadata={"agency": {"source_session_id": "session-1"}},
        active_run_id="agency-run-1",
    )
    active_run = RunRecord(
        id="agency-run-1",
        session_id="agency-1",
        sequence_no=1,
        status="running",
        trigger_type=TriggerType.AGENCY.value,
        input_parts=[],
        run_metadata={"agency": {"source_session_id": "session-1"}},
    )
    state = SessionAgencyStateRecord(
        source_session_id="session-1",
        agency_session_id="agency-1",
        enabled=True,
        agency_metadata={},
    )
    db_session.add_all([source_session, agency_session, active_run, state])
    await db_session.commit()
    runtime_state = create_runtime_state()
    runtime_state.register_run("agency-1", "agency-run-1", dispatch_mode="async")
    lifecycle = AgencyLifecycle(settings=settings, runtime_state=runtime_state)

    client_marker = "manual-2"
    delivery = await lifecycle.create_signal(
        db_session,
        "session-1",
        AgencySignalRequest(reason=AgencySignalReason.MANUAL, client_token=client_marker),
    )

    assert delivery.delivery == "steered"
    assert delivery.active_run_id == "agency-run-1"
    batches = runtime_state.consume_steering_inputs("agency-run-1")
    assert len(batches) == 1
    assert batches[0][0]["name"] == "agency_signal"
    signal = await db_session.get(AgencySignalRecord, delivery.signal.id)
    assert isinstance(signal, AgencySignalRecord)
    assert signal.status == "steered"


async def test_agency_tick_creates_scheduled_signal_for_idle_session(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    old_time = datetime.now(UTC) - timedelta(seconds=settings.agency_idle_after_seconds + 60)
    source_session = SessionRecord(
        id="session-1",
        profile_name="general",
        session_metadata={},
        updated_at=old_time,
    )
    source_run = RunRecord(
        id="run-1",
        session_id="session-1",
        sequence_no=1,
        status="completed",
        trigger_type=TriggerType.API.value,
        input_parts=[],
        run_metadata={},
        committed_at=old_time,
    )
    db_session.add_all([source_session, source_run])
    await db_session.commit()
    submitted: list[str] = []
    lifecycle = AgencyLifecycle(
        settings=settings,
        runtime_state=create_runtime_state(),
        submit_run=lambda run_id: not submitted.append(run_id),
    )

    result = await lifecycle.tick(db_session)

    assert len(result.created_signal_ids) == 1
    assert len(result.submitted_run_ids) == 1
    assert submitted == result.submitted_run_ids
    signal = await db_session.get(AgencySignalRecord, result.created_signal_ids[0])
    assert isinstance(signal, AgencySignalRecord)
    assert signal.reason == AgencySignalReason.SCHEDULE.value


@pytest.fixture
async def db_engine(tmp_path: Path) -> AsyncEngine:
    engine = create_engine(f"sqlite+aiosqlite:///{(tmp_path / 'agency.sqlite3').resolve()}")
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
        agency_enabled=True,
        agency_idle_after_seconds=1,
        agency_cooldown_seconds=0,
        _env_file=None,
    )
