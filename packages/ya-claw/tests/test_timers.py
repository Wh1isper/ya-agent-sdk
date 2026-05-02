from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from ya_claw.app import create_app
from ya_claw.config import ClawSettings, get_settings
from ya_claw.controller.heartbeat import HeartbeatController
from ya_claw.controller.schedule import ScheduleController, ScheduleCreateRequest, compute_next_fire_at
from ya_claw.db.engine import create_engine, create_session_factory
from ya_claw.execution.dispatcher import RunDispatcher
from ya_claw.execution.heartbeat import HeartbeatDispatcher
from ya_claw.execution.schedule import ScheduleDispatcher
from ya_claw.orm.base import Base
from ya_claw.orm.tables import HeartbeatFireRecord, RunRecord, ScheduleFireRecord, ScheduleRecord, SessionRecord
from ya_claw.runtime_state import InMemoryRuntimeState


class RecordingSupervisor:
    def __init__(self, *, accepting_submissions: bool = True) -> None:
        self.submitted_run_ids: list[str] = []
        self.accepting_submissions = accepting_submissions

    def get_background_task(self, run_id: str) -> None:
        return None

    def submit_run(self, run_id: str) -> bool:
        if not self.accepting_submissions:
            return False
        self.submitted_run_ids.append(run_id)
        return True


@pytest.fixture(autouse=True)
def clear_claw_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    for env_name in (
        "YA_CLAW_API_TOKEN",
        "YA_CLAW_DATABASE_URL",
        "YA_CLAW_DATA_DIR",
        "YA_CLAW_WEB_DIST_DIR",
        "YA_CLAW_WORKSPACE_DIR",
        "YA_CLAW_PROFILE_SEED_FILE",
        "YA_CLAW_AUTO_SEED_PROFILES",
        "YA_CLAW_SCHEDULE_DISPATCH_ENABLED",
        "YA_CLAW_HEARTBEAT_ENABLED",
        "YA_CLAW_HEARTBEAT_INTERVAL_SECONDS",
        "YA_CLAW_HEARTBEAT_PROFILE",
    ):
        monkeypatch.delenv(env_name, raising=False)

    monkeypatch.setenv("YA_CLAW_API_TOKEN", "test-token")
    monkeypatch.setenv("YA_CLAW_DATA_DIR", str(tmp_path / "runtime-data"))
    monkeypatch.setenv("YA_CLAW_WORKSPACE_DIR", str(tmp_path / "workspace"))
    monkeypatch.setenv("YA_CLAW_AUTO_SEED_PROFILES", "false")
    monkeypatch.setenv("YA_CLAW_WORKSPACE_PROVIDER_BACKEND", "local")
    monkeypatch.setenv("YA_CLAW_BRIDGE_DISPATCH_MODE", "manual")

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
async def db_engine(tmp_path: Path) -> AsyncEngine:
    engine = create_engine(f"sqlite+aiosqlite:///{(tmp_path / 'timers.sqlite3').resolve()}")
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
        workspace_provider_backend="local",
        bridge_dispatch_mode="manual",
        _env_file=None,
    )


def _auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer test-token"}


def _create_schema() -> None:
    import asyncio

    async def _run() -> None:
        settings = get_settings()
        engine = create_engine(settings.resolved_database_url)
        try:
            async with engine.begin() as connection:
                await connection.run_sync(Base.metadata.create_all)
        finally:
            await engine.dispose()

    asyncio.run(_run())


async def test_cron_next_fire_supports_steps_ranges_and_timezone() -> None:
    now = datetime(2026, 4, 26, 5, 7, tzinfo=UTC)

    assert compute_next_fire_at("*/15 5-6 * * *", "UTC", now=now) == datetime(2026, 4, 26, 5, 15, tzinfo=UTC)
    assert compute_next_fire_at("0 9 * * *", "Asia/Shanghai", now=now) == datetime(2026, 4, 27, 1, 0, tzinfo=UTC)


async def test_schedule_controller_dispatch_due_scans_due_records_and_submits_run(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    controller = ScheduleController()
    runtime_state = InMemoryRuntimeState()
    supervisor = RecordingSupervisor()
    dispatcher = RunDispatcher(supervisor)  # type: ignore[arg-type]
    now = datetime(2026, 4, 26, 5, 30, tzinfo=UTC)

    schedule = await controller.create(
        db_session,
        ScheduleCreateRequest(
            name="Every minute smoke test",
            prompt="Report timer status.",
            cron="* * * * *",
            timezone="UTC",
            profile_name="default",
            metadata={"purpose": "test"},
        ),
    )
    record = await db_session.get(ScheduleRecord, schedule.id)
    assert isinstance(record, ScheduleRecord)
    record.next_fire_at = now - timedelta(minutes=1)
    await db_session.commit()

    fired = await controller.dispatch_due(db_session, settings, runtime_state, dispatcher, now=now)

    assert len(fired) == 1
    fire = fired[0]
    assert fire.status == "submitted"
    assert fire.run_id in supervisor.submitted_run_ids
    assert fire.created_session_id is not None

    run = await db_session.get(RunRecord, fire.run_id)
    assert isinstance(run, RunRecord)
    assert run.status == "queued"
    assert run.trigger_type == "schedule"
    assert run.profile_name == "default"
    assert run.run_metadata["source"] == "schedule"
    assert run.run_metadata["schedule_id"] == schedule.id
    assert run.run_metadata["schedule_fire_id"] == fire.id
    assert run.run_metadata["restore_state"] is False
    assert run.restore_from_run_id is None

    await db_session.refresh(record)
    assert record.fire_count == 1
    assert record.last_fire_id == fire.id
    assert record.last_run_id == fire.run_id
    assert record.next_fire_at is not None
    assert record.next_fire_at.replace(tzinfo=UTC) > now


async def test_schedule_fork_session_creates_isolated_run_without_state_restore(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    controller = ScheduleController()
    runtime_state = InMemoryRuntimeState()
    supervisor = RecordingSupervisor()
    dispatcher = RunDispatcher(supervisor)  # type: ignore[arg-type]
    source_session = SessionRecord(
        id="source-session-1",
        profile_name="default",
        session_metadata={},
        head_run_id="source-run-1",
        head_success_run_id="source-run-1",
    )
    source_run = RunRecord(
        id="source-run-1",
        session_id="source-session-1",
        sequence_no=1,
        restore_from_run_id=None,
        status="completed",
        trigger_type="api",
        profile_name="default",
        input_parts=[{"type": "text", "text": "base"}],
        run_metadata={},
    )
    db_session.add_all([source_session, source_run])
    await db_session.commit()

    schedule = await controller.create(
        db_session,
        ScheduleCreateRequest(
            name="Fork schedule",
            prompt="Report forked timer status.",
            cron="* * * * *",
            timezone="UTC",
            owner_session_id="source-session-1",
            start_from_current_session=True,
            profile_name="default",
        ),
    )

    fire = await controller.trigger(db_session, settings, runtime_state, dispatcher, schedule.id)

    assert fire.status == "submitted"
    assert fire.run_id in supervisor.submitted_run_ids
    assert fire.created_session_id is not None
    fork_session = await db_session.get(SessionRecord, fire.created_session_id)
    run = await db_session.get(RunRecord, fire.run_id)
    assert isinstance(fork_session, SessionRecord)
    assert isinstance(run, RunRecord)
    assert fork_session.parent_session_id == "source-session-1"
    assert fork_session.head_run_id == run.id
    assert fork_session.head_success_run_id is None
    assert fork_session.session_metadata["source"] == "schedule"
    assert fork_session.head_run_id != "source-run-1"
    assert run.trigger_type == "schedule"
    assert run.restore_from_run_id is None
    assert run.run_metadata["restore_state"] is False
    assert run.run_metadata["execution_mode"] == "fork_session"


async def test_schedule_fire_stays_pending_when_dispatch_skips(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    controller = ScheduleController()
    runtime_state = InMemoryRuntimeState()
    supervisor = RecordingSupervisor(accepting_submissions=False)
    dispatcher = RunDispatcher(supervisor)  # type: ignore[arg-type]
    now = datetime(2026, 4, 26, 5, 30, tzinfo=UTC)

    schedule = await controller.create(
        db_session,
        ScheduleCreateRequest(
            name="Shutdown skip schedule",
            prompt="Report timer status.",
            cron="* * * * *",
            timezone="UTC",
            profile_name="default",
        ),
    )
    record = await db_session.get(ScheduleRecord, schedule.id)
    assert isinstance(record, ScheduleRecord)
    record.next_fire_at = now - timedelta(minutes=1)
    await db_session.commit()

    fired = await controller.dispatch_due(db_session, settings, runtime_state, dispatcher, now=now)

    assert len(fired) == 1
    assert fired[0].status == "pending"
    assert fired[0].run_id is not None
    assert fired[0].error_message == "Dispatch skipped: supervisor_shutting_down"
    assert supervisor.submitted_run_ids == []


async def test_once_schedule_create_sets_next_fire_at_to_run_at(
    db_session: AsyncSession,
) -> None:
    controller = ScheduleController()
    run_at = datetime(2026, 4, 26, 6, 30, tzinfo=UTC)

    schedule = await controller.create(
        db_session,
        ScheduleCreateRequest(
            name="One-time reminder",
            prompt="Run once.",
            trigger_kind="once",
            run_at=run_at,
            timezone="UTC",
            profile_name="default",
        ),
    )

    assert schedule.status == "active"
    assert schedule.trigger["kind"] == "once"
    assert schedule.trigger["run_at"] == run_at.replace(tzinfo=None)
    assert schedule.trigger["next_fire_at"] == run_at.replace(tzinfo=None)
    assert schedule.cron["expr"] is None
    record = await db_session.get(ScheduleRecord, schedule.id)
    assert isinstance(record, ScheduleRecord)
    assert record.trigger_kind == "once"
    assert record.run_at == run_at.replace(tzinfo=None)
    assert record.cron_expr is None
    assert record.next_fire_at == run_at.replace(tzinfo=None)


async def test_once_schedule_dispatch_submits_run_and_completes_schedule(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    controller = ScheduleController()
    runtime_state = InMemoryRuntimeState()
    supervisor = RecordingSupervisor()
    dispatcher = RunDispatcher(supervisor)  # type: ignore[arg-type]
    run_at = datetime(2026, 4, 26, 6, 30, tzinfo=UTC)

    schedule = await controller.create(
        db_session,
        ScheduleCreateRequest(
            name="One-time report",
            prompt="Run one-time report.",
            trigger_kind="once",
            run_at=run_at,
            timezone="UTC",
            profile_name="default",
        ),
    )

    fired = await controller.dispatch_due(
        db_session,
        settings,
        runtime_state,
        dispatcher,
        now=run_at + timedelta(seconds=1),
    )

    assert len(fired) == 1
    fire = fired[0]
    assert fire.status == "submitted"
    assert fire.run_id in supervisor.submitted_run_ids
    record = await db_session.get(ScheduleRecord, schedule.id)
    assert isinstance(record, ScheduleRecord)
    assert record.status == "completed"
    assert record.next_fire_at is None
    assert record.fire_count == 1
    run = await db_session.get(RunRecord, fire.run_id)
    assert isinstance(run, RunRecord)
    assert run.trigger_type == "schedule"
    assert run.run_metadata["execution_mode"] == "isolate_session"


async def test_once_schedule_pending_queue_stays_active_until_delivered(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    controller = ScheduleController()
    runtime_state = InMemoryRuntimeState()
    supervisor = RecordingSupervisor(accepting_submissions=False)
    dispatcher = RunDispatcher(supervisor)  # type: ignore[arg-type]
    run_at = datetime(2026, 4, 26, 6, 30, tzinfo=UTC)

    schedule = await controller.create(
        db_session,
        ScheduleCreateRequest(
            name="One-time queued report",
            prompt="Run one-time report.",
            trigger_kind="once",
            run_at=run_at,
            timezone="UTC",
            profile_name="default",
        ),
    )

    fired = await controller.dispatch_due(
        db_session,
        settings,
        runtime_state,
        dispatcher,
        now=run_at + timedelta(seconds=1),
    )

    assert len(fired) == 1
    assert fired[0].status == "pending"
    record = await db_session.get(ScheduleRecord, schedule.id)
    assert isinstance(record, ScheduleRecord)
    assert record.status == "active"
    assert record.next_fire_at == run_at.replace(tzinfo=None)

    supervisor.accepting_submissions = True
    fired = await controller.dispatch_pending(
        db_session,
        settings,
        runtime_state,
        dispatcher,
        now=run_at + timedelta(seconds=2),
    )

    assert len(fired) == 1
    assert fired[0].status == "submitted"
    record = await db_session.get(ScheduleRecord, schedule.id)
    assert isinstance(record, ScheduleRecord)
    assert record.status == "completed"
    assert record.next_fire_at is None


def test_once_schedule_requires_run_at() -> None:
    with pytest.raises(ValueError, match="run_at is required"):
        ScheduleCreateRequest(
            name="One-time missing run_at",
            prompt="Run once.",
            trigger_kind="once",
            timezone="UTC",
        )


async def test_heartbeat_dispatch_due_handles_sqlite_naive_datetimes(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    settings.heartbeat_enabled = True
    settings.heartbeat_interval_seconds = 1
    controller = HeartbeatController()
    runtime_state = InMemoryRuntimeState()
    supervisor = RecordingSupervisor()
    dispatcher = RunDispatcher(supervisor)  # type: ignore[arg-type]
    naive_scheduled_at = datetime(2026, 4, 26, 5, 30)
    db_session.add(
        HeartbeatFireRecord(
            id="heartbeat-fire-naive",
            scheduled_at=naive_scheduled_at,
            fired_at=naive_scheduled_at,
            status="submitted",
            dedupe_key="heartbeat-fire-naive",
            fire_metadata={"manual": False},
        )
    )
    await db_session.commit()

    fire = await controller.dispatch_due(
        db_session,
        settings,
        runtime_state,
        dispatcher,
    )

    assert fire is not None
    assert fire.status == "submitted"
    assert fire.run_id in supervisor.submitted_run_ids


async def test_heartbeat_fire_stays_pending_when_dispatch_skips(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    settings.heartbeat_enabled = True
    settings.heartbeat_interval_seconds = 1
    controller = HeartbeatController()
    runtime_state = InMemoryRuntimeState()
    supervisor = RecordingSupervisor(accepting_submissions=False)
    dispatcher = RunDispatcher(supervisor)  # type: ignore[arg-type]

    fire = await controller.trigger(db_session, settings, runtime_state, dispatcher)

    assert fire.status == "pending"
    assert fire.run_id is not None
    assert fire.error_message == "Dispatch skipped: supervisor_shutting_down"
    assert supervisor.submitted_run_ids == []


async def test_schedule_dispatch_due_handles_sqlite_naive_datetimes(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    controller = ScheduleController()
    runtime_state = InMemoryRuntimeState()
    supervisor = RecordingSupervisor()
    dispatcher = RunDispatcher(supervisor)  # type: ignore[arg-type]
    now = datetime(2026, 4, 26, 5, 30, tzinfo=UTC)

    schedule = await controller.create(
        db_session,
        ScheduleCreateRequest(
            name="Naive datetime schedule",
            prompt="Report timer status.",
            cron="* * * * *",
            timezone="UTC",
            profile_name="default",
        ),
    )
    record = await db_session.get(ScheduleRecord, schedule.id)
    assert isinstance(record, ScheduleRecord)
    record.next_fire_at = datetime(2026, 4, 26, 5, 29)
    await db_session.commit()

    fired = await controller.dispatch_due(db_session, settings, runtime_state, dispatcher, now=now)

    assert len(fired) == 1
    assert fired[0].status == "submitted"
    assert fired[0].run_id in supervisor.submitted_run_ids


async def test_heartbeat_controller_defaults_and_manual_trigger_create_isolated_run(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    settings.heartbeat_guidance_path.write_text("# Heartbeat\nCheck runtime health.\n", encoding="utf-8")
    controller = HeartbeatController()
    runtime_state = InMemoryRuntimeState()
    supervisor = RecordingSupervisor()
    dispatcher = RunDispatcher(supervisor)  # type: ignore[arg-type]

    config = await controller.config(db_session, settings)
    assert config.enabled is False
    assert config.interval_seconds == 300
    assert config.profile_name == "default"
    assert config.profile_source == "default"
    assert config.prompt_source == "heartbeat_setting"
    assert config.guidance_file["exists"] is True
    assert config.next_fire_at is None

    settings.heartbeat_enabled = True
    fire = await controller.trigger(db_session, settings, runtime_state, dispatcher)

    assert fire.status == "submitted"
    assert fire.metadata["manual"] is True
    assert fire.run_id in supervisor.submitted_run_ids
    assert fire.session_id is not None

    run = await db_session.get(RunRecord, fire.run_id)
    assert isinstance(run, RunRecord)
    assert run.trigger_type == "heartbeat"
    assert run.profile_name == "default"
    assert run.run_metadata["source"] == "heartbeat"
    assert run.run_metadata["heartbeat_fire_id"] == fire.id
    assert run.restore_from_run_id is None
    run.status = "completed"
    await db_session.commit()

    fires = await controller.list_fires(db_session)
    assert fires.fires[0].run_status == "completed"


async def test_schedule_fire_history_includes_run_status(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    controller = ScheduleController()
    runtime_state = InMemoryRuntimeState()
    supervisor = RecordingSupervisor()
    dispatcher = RunDispatcher(supervisor)  # type: ignore[arg-type]
    schedule = await controller.create(
        db_session,
        ScheduleCreateRequest(
            name="Run status schedule",
            prompt="Report timer status.",
            cron="* * * * *",
            timezone="UTC",
            profile_name="default",
        ),
    )

    fire = await controller.trigger(db_session, settings, runtime_state, dispatcher, schedule.id)
    run = await db_session.get(RunRecord, fire.run_id)
    assert isinstance(run, RunRecord)
    run.status = "completed"
    await db_session.commit()

    fires = await controller.list_fires(db_session, schedule.id)
    assert fires.fires[0].run_status == "completed"


async def test_heartbeat_dispatcher_scan_triggers_due_fire(
    db_engine: AsyncEngine,
    settings: ClawSettings,
) -> None:
    settings.heartbeat_enabled = True
    settings.heartbeat_interval_seconds = 1
    session_factory = create_session_factory(db_engine)
    runtime_state = InMemoryRuntimeState()
    supervisor = RecordingSupervisor()
    dispatcher = HeartbeatDispatcher(
        settings=settings,
        session_factory=session_factory,
        runtime_state=runtime_state,
        run_dispatcher=RunDispatcher(supervisor),  # type: ignore[arg-type]
    )

    count = await dispatcher.dispatch_once()

    assert count == 1
    assert len(supervisor.submitted_run_ids) == 1
    async with session_factory() as session:
        fire = (await session.execute(select(HeartbeatFireRecord))).scalar_one()
        run = await session.get(RunRecord, fire.run_id)
    assert fire.status == "submitted"
    assert fire.fire_metadata["manual"] is False
    assert isinstance(run, RunRecord)
    assert run.trigger_type == "heartbeat"


async def test_schedule_dispatcher_scan_processes_pending_and_due_fires(
    db_engine: AsyncEngine,
    settings: ClawSettings,
) -> None:
    session_factory = create_session_factory(db_engine)
    runtime_state = InMemoryRuntimeState()
    supervisor = RecordingSupervisor()
    controller = ScheduleController()
    now = datetime(2026, 4, 26, 5, 30, tzinfo=UTC)

    async with session_factory() as session:
        schedule = await controller.create(
            session,
            ScheduleCreateRequest(
                name="Due schedule",
                prompt="Report schedule status.",
                cron="* * * * *",
                timezone="UTC",
                profile_name="default",
            ),
        )
        record = await session.get(ScheduleRecord, schedule.id)
        assert isinstance(record, ScheduleRecord)
        record.next_fire_at = now - timedelta(minutes=1)
        await session.commit()

    dispatcher = ScheduleDispatcher(
        settings=settings,
        session_factory=session_factory,
        runtime_state=runtime_state,
        run_dispatcher=RunDispatcher(supervisor),  # type: ignore[arg-type]
    )

    count = await dispatcher.dispatch_once()

    assert count == 1
    assert len(supervisor.submitted_run_ids) == 1
    async with session_factory() as session:
        fire = (await session.execute(select(ScheduleFireRecord))).scalar_one()
        run = await session.get(RunRecord, fire.run_id)
        record = await session.get(ScheduleRecord, schedule.id)
    assert fire.status == "submitted"
    assert fire.fire_metadata["manual"] is False
    assert isinstance(run, RunRecord)
    assert run.trigger_type == "schedule"
    assert isinstance(record, ScheduleRecord)
    assert record.fire_count == 1


def test_timer_api_routes_expose_config_create_trigger_and_fire_history() -> None:
    _create_schema()

    with TestClient(create_app()) as client:
        client.app.state.execution_supervisor = None
        heartbeat_config = client.get("/api/v1/heartbeat/config", headers=_auth_headers())
        assert heartbeat_config.status_code == 200
        assert heartbeat_config.json()["enabled"] is False
        assert heartbeat_config.json()["interval_seconds"] == 300
        assert heartbeat_config.json()["profile_source"] == "default"
        assert heartbeat_config.json()["prompt_source"] == "heartbeat_setting"

        create_schedule = client.post(
            "/api/v1/schedules",
            headers=_auth_headers(),
            json={
                "name": "API cron smoke test",
                "prompt": "Report API timer status.",
                "cron": "* * * * *",
                "timezone": "UTC",
                "enabled": True,
                "owner_kind": "user",
            },
        )
        assert create_schedule.status_code == 201
        schedule_id = create_schedule.json()["id"]
        assert create_schedule.json()["trigger"]["kind"] == "cron"
        assert create_schedule.json()["cron"]["next_fire_at"] is not None

        create_once_schedule = client.post(
            "/api/v1/schedules",
            headers=_auth_headers(),
            json={
                "name": "API one-time smoke test",
                "prompt": "Report API one-time status.",
                "trigger_kind": "once",
                "run_at": "2026-04-26T06:30:00Z",
                "timezone": "UTC",
                "enabled": True,
                "owner_kind": "user",
            },
        )
        assert create_once_schedule.status_code == 201
        assert create_once_schedule.json()["trigger"]["kind"] == "once"
        assert create_once_schedule.json()["trigger"]["run_at"] == "2026-04-26T06:30:00"
        assert create_once_schedule.json()["cron"]["expr"] is None

        manual_fire = client.post(
            f"/api/v1/schedules/{schedule_id}:trigger",
            headers=_auth_headers(),
        )
        assert manual_fire.status_code == 201
        assert manual_fire.json()["status"] == "pending"
        assert manual_fire.json()["run_id"] is not None
        assert manual_fire.json()["error_message"] == "Dispatch skipped: supervisor_unavailable"

        schedule_fires = client.get(f"/api/v1/schedules/{schedule_id}/fires", headers=_auth_headers())
        assert schedule_fires.status_code == 200
        assert len(schedule_fires.json()["fires"]) == 1
        assert schedule_fires.json()["fires"][0]["run_status"] == "queued"

        heartbeat_fire = client.post("/api/v1/heartbeat:trigger", headers=_auth_headers())
        assert heartbeat_fire.status_code == 201
        assert heartbeat_fire.json()["status"] == "pending"
        assert heartbeat_fire.json()["run_id"] is not None
        assert heartbeat_fire.json()["error_message"] == "Dispatch skipped: supervisor_unavailable"

        heartbeat_fires = client.get("/api/v1/heartbeat/fires", headers=_auth_headers())
        assert heartbeat_fires.status_code == 200
        assert len(heartbeat_fires.json()["fires"]) == 1
        assert heartbeat_fires.json()["fires"][0]["run_status"] == "queued"
