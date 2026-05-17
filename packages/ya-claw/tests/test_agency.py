from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from ya_claw.agency.lifecycle import AGENCY_SINGLETON_SOURCE_SESSION_ID, AgencyLifecycle
from ya_claw.config import ClawSettings
from ya_claw.controller.models import AgencyFireKind, TriggerType
from ya_claw.db.engine import create_engine, create_session_factory
from ya_claw.orm.base import Base
from ya_claw.orm.tables import AgencyFireRecord, RunRecord, SessionRecord
from ya_claw.runtime_state import create_runtime_state


def make_client_token(value: str) -> str:
    return value


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
        api_token=make_client_token("test-token"),
        data_dir=data_dir,
        workspace_dir=workspace_dir,
        agency_enabled=True,
        agency_tick_seconds=30,
        _env_file=None,
    )


async def test_manual_fire_creates_singleton_agency_session_and_run(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    source_session = SessionRecord(id="session-1", profile_name="general", session_metadata={})
    db_session.add(source_session)
    await db_session.commit()
    submitted: list[str] = []
    lifecycle = AgencyLifecycle(
        settings=settings,
        runtime_state=create_runtime_state(),
        submit_run=lambda run_id: not submitted.append(run_id),
    )

    delivery = await lifecycle.create_fire(
        db_session,
        kind=AgencyFireKind.MANUAL,
        source_session_id="session-1",
        client_token=make_client_token("manual-1"),
        prompt="Focus on the follow-up plan.",
    )

    assert delivery.delivery == "submitted"
    assert delivery.run_id is not None
    assert submitted == [delivery.run_id]
    agency_session = delivery.agency_session
    assert agency_session.session_type == "agency"
    assert agency_session.source_session_id == AGENCY_SINGLETON_SOURCE_SESSION_ID
    assert agency_session.parent_session_id is None
    assert agency_session.session_metadata["agency"]["kind"] == "claw_agency_session"
    assert agency_session.session_metadata["agency"]["scope"] == "global"

    fire = await db_session.get(AgencyFireRecord, delivery.fire.id)
    assert isinstance(fire, AgencyFireRecord)
    assert fire.kind == "manual"
    assert fire.status == "submitted"
    assert fire.source_session_id == "session-1"
    assert fire.payload["prompt"] == "Focus on the follow-up plan."

    run = await db_session.get(RunRecord, delivery.run_id)
    assert isinstance(run, RunRecord)
    assert run.session_id == agency_session.id
    assert run.trigger_type == TriggerType.AGENCY.value
    assert run.run_metadata["agency"]["kind"] == "episode"
    assert run.run_metadata["agency"]["fire_ids"] == [fire.id]
    assert run.run_metadata["agency"]["trigger_kinds"] == ["manual"]
    assert run.run_metadata["agency"]["source_session_ids"] == ["session-1"]
    assert run.run_metadata["agency"]["budget"]["external_actions"] == "deny"
    assert run.input_parts[0]["name"] == "agency_fire"


async def test_singleton_agency_session_reused_for_all_sources(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    db_session.add_all([
        SessionRecord(id="session-1", profile_name="general", session_metadata={}),
        SessionRecord(id="session-2", profile_name="other", session_metadata={}),
    ])
    await db_session.commit()
    lifecycle = AgencyLifecycle(settings=settings, runtime_state=create_runtime_state())

    first = await lifecycle.create_fire(
        db_session,
        kind=AgencyFireKind.MANUAL,
        source_session_id="session-1",
        client_token=make_client_token("manual-1"),
        dispatch=False,
    )
    second = await lifecycle.create_fire(
        db_session,
        kind=AgencyFireKind.MANUAL,
        source_session_id="session-2",
        client_token=make_client_token("manual-2"),
        dispatch=False,
    )

    assert first.agency_session.id == second.agency_session.id
    result = await db_session.execute(select(SessionRecord).where(SessionRecord.session_type == "agency"))
    assert len(list(result.scalars().all())) == 1


async def test_pending_fires_batch_into_new_agency_run(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    db_session.add_all([
        SessionRecord(id="session-1", profile_name="general", session_metadata={}),
        SessionRecord(id="session-2", profile_name="general", session_metadata={}),
    ])
    await db_session.commit()
    submitted: list[str] = []
    lifecycle = AgencyLifecycle(
        settings=settings,
        runtime_state=create_runtime_state(),
        submit_run=lambda run_id: not submitted.append(run_id),
    )

    first = await lifecycle.create_fire(
        db_session,
        kind=AgencyFireKind.MANUAL,
        source_session_id="session-1",
        client_token=make_client_token("manual-1"),
        dispatch=False,
    )
    second = await lifecycle.create_fire(
        db_session,
        kind=AgencyFireKind.MEMORY_COMMITTED,
        source_session_id="session-2",
        source_run_id="memory-run-1",
        client_token=make_client_token("memory-run-1"),
        payload={"memory_kind": "extract"},
        dispatch=False,
    )
    delivery = await lifecycle.dispatch_pending(db_session)

    assert delivery.delivery == "submitted"
    assert submitted == [delivery.run_id]
    run = await db_session.get(RunRecord, delivery.run_id)
    assert isinstance(run, RunRecord)
    assert set(run.run_metadata["agency"]["fire_ids"]) == {first.fire.id, second.fire.id}
    assert set(run.run_metadata["agency"]["trigger_kinds"]) == {"manual", "memory_committed"}
    assert len(run.input_parts) == 2


async def test_fire_steers_active_agency_run(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    source_session = SessionRecord(id="session-1", profile_name="general", session_metadata={})
    agency_session = SessionRecord(
        id="agency-1",
        profile_name="general",
        session_type="agency",
        source_session_id=AGENCY_SINGLETON_SOURCE_SESSION_ID,
        session_metadata={"agency": {"kind": "claw_agency_session", "enabled": True}},
        active_run_id="agency-run-1",
        head_run_id="agency-run-1",
    )
    active_run = RunRecord(
        id="agency-run-1",
        session_id="agency-1",
        sequence_no=1,
        status="running",
        trigger_type=TriggerType.AGENCY.value,
        input_parts=[],
        run_metadata={"agency": {"agency_session_id": "agency-1", "fire_ids": []}},
    )
    db_session.add_all([source_session, agency_session, active_run])
    await db_session.commit()
    runtime_state = create_runtime_state()
    runtime_state.register_run("agency-1", "agency-run-1", dispatch_mode="async")
    lifecycle = AgencyLifecycle(settings=settings, runtime_state=runtime_state)

    delivery = await lifecycle.create_fire(
        db_session,
        kind=AgencyFireKind.MANUAL,
        source_session_id="session-1",
        client_token=make_client_token("manual-2"),
    )

    assert delivery.delivery == "steered"
    assert delivery.active_run_id == "agency-run-1"
    batches = runtime_state.consume_steering_inputs("agency-run-1")
    assert len(batches) == 1
    assert batches[0][0]["name"] == "agency_fire"
    fire = await db_session.get(AgencyFireRecord, delivery.fire.id)
    assert isinstance(fire, AgencyFireRecord)
    assert fire.status == "steered"
    await db_session.refresh(active_run)
    assert fire.id in active_run.run_metadata["agency"]["fire_ids"]
    assert fire.id in active_run.run_metadata["agency"]["steered_fire_ids"]


async def test_fire_merges_into_queued_agency_run(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    db_session.add_all([
        SessionRecord(id="session-1", profile_name="general", session_metadata={}),
        SessionRecord(id="session-2", profile_name="general", session_metadata={}),
    ])
    await db_session.commit()
    submitted: list[str] = []
    lifecycle = AgencyLifecycle(
        settings=settings,
        runtime_state=create_runtime_state(),
        submit_run=lambda run_id: not submitted.append(run_id),
    )

    first = await lifecycle.create_fire(
        db_session,
        kind=AgencyFireKind.MANUAL,
        source_session_id="session-1",
        client_token=make_client_token("manual-1"),
    )
    second = await lifecycle.create_fire(
        db_session,
        kind=AgencyFireKind.MANUAL,
        source_session_id="session-2",
        client_token=make_client_token("manual-2"),
    )

    assert first.delivery == "submitted"
    assert second.delivery == "merged"
    assert second.run_id == first.run_id
    assert submitted == [first.run_id]
    run = await db_session.get(RunRecord, first.run_id)
    assert isinstance(run, RunRecord)
    assert len(run.input_parts) == 2
    assert set(run.run_metadata["agency"]["source_session_ids"]) == {"session-1", "session-2"}


async def test_duplicate_fire_reports_duplicate(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    db_session.add(SessionRecord(id="session-1", profile_name="general", session_metadata={}))
    await db_session.commit()
    lifecycle = AgencyLifecycle(settings=settings, runtime_state=create_runtime_state())

    first = await lifecycle.create_fire(
        db_session,
        kind=AgencyFireKind.MANUAL,
        source_session_id="session-1",
        client_token=make_client_token("manual-1"),
        dispatch=False,
    )
    second = await lifecycle.create_fire(
        db_session,
        kind=AgencyFireKind.MANUAL,
        source_session_id="session-1",
        client_token=make_client_token("manual-1"),
        dispatch=False,
    )

    assert first.delivery == "pending"
    assert second.delivery == "duplicate"
    assert second.fire.id == first.fire.id
    result = await db_session.execute(select(AgencyFireRecord))
    assert len(list(result.scalars().all())) == 1


async def test_timer_tick_creates_timer_fire(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    submitted: list[str] = []
    lifecycle = AgencyLifecycle(
        settings=settings,
        runtime_state=create_runtime_state(),
        submit_run=lambda run_id: not submitted.append(run_id),
    )

    result = await lifecycle.tick(db_session)

    assert len(result.created_fire_ids) == 1
    assert len(result.submitted_run_ids) == 1
    assert submitted == result.submitted_run_ids
    fire = await db_session.get(AgencyFireRecord, result.created_fire_ids[0])
    assert isinstance(fire, AgencyFireRecord)
    assert fire.kind == AgencyFireKind.TIMER.value


async def test_agency_run_commit_consumes_fires(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    agency_session = SessionRecord(
        id="agency-1",
        profile_name="general",
        session_type="agency",
        source_session_id=AGENCY_SINGLETON_SOURCE_SESSION_ID,
        session_metadata={"agency": {"kind": "claw_agency_session", "enabled": True}},
    )
    run = RunRecord(
        id="agency-run-1",
        session_id="agency-1",
        sequence_no=1,
        status="completed",
        trigger_type=TriggerType.AGENCY.value,
        input_parts=[],
        run_metadata={"agency": {"agency_session_id": "agency-1", "fire_ids": ["fire-1", "fire-2"]}},
        committed_at=datetime.now(UTC),
    )
    fires = [
        AgencyFireRecord(
            id="fire-1",
            kind="manual",
            status="submitted",
            scheduled_at=datetime.now(UTC),
            dedupe_key="fire-1",
            agency_session_id="agency-1",
            run_id="agency-run-1",
            payload={},
        ),
        AgencyFireRecord(
            id="fire-2",
            kind="memory_committed",
            status="steered",
            scheduled_at=datetime.now(UTC),
            dedupe_key="fire-2",
            agency_session_id="agency-1",
            run_id="agency-run-1",
            payload={},
        ),
    ]
    db_session.add_all([agency_session, run, *fires])
    await db_session.commit()

    lifecycle = AgencyLifecycle(settings=settings, runtime_state=create_runtime_state())
    await lifecycle.on_agency_run_committed(db_session, run)
    await db_session.commit()

    for fire in fires:
        await db_session.refresh(fire)
        assert fire.status == "consumed"
        assert fire.consumed_at is not None


async def test_next_timer_handles_sqlite_naive_datetimes(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    old_time = datetime.now(UTC) - timedelta(seconds=settings.agency_tick_seconds + 60)
    fire = AgencyFireRecord(
        id="fire-1",
        kind="timer",
        status="consumed",
        scheduled_at=old_time.replace(tzinfo=None),
        dedupe_key="timer-old",
        payload={},
    )
    db_session.add(fire)
    await db_session.commit()
    lifecycle = AgencyLifecycle(settings=settings, runtime_state=create_runtime_state())

    next_fire_at = await lifecycle.next_timer_fire_at(db_session)

    assert next_fire_at is not None
    assert next_fire_at.tzinfo is not None
