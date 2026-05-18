from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from ya_claw.config import ClawSettings
from ya_claw.controller.session_prune import SessionPruneController
from ya_claw.db.engine import create_engine, create_session_factory
from ya_claw.orm.base import Base
from ya_claw.orm.tables import HeartbeatFireRecord, RunRecord, ScheduleFireRecord, ScheduleRecord, SessionRecord


@pytest.fixture
async def db_engine(tmp_path: Path) -> AsyncEngine:
    engine = create_engine(f"sqlite+aiosqlite:///{(tmp_path / 'prune.sqlite3').resolve()}")
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
        session_prune_enabled=True,
        session_prune_run_keep_recent=3,
        session_prune_run_older_than_days=0,
        session_prune_batch_size=1000,
        session_prune_generated_sessions_enabled=False,
        session_prune_fire_records_older_than_days=0,
        _env_file=None,
    )


async def test_prune_keeps_recent_run_rows_heads_active_and_restore_sources(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    controller = SessionPruneController()
    session = SessionRecord(
        id="session-1",
        profile_name="default",
        head_run_id="run-6",
        head_success_run_id="run-2",
        active_run_id="run-6",
    )
    db_session.add(session)
    now = datetime(2026, 4, 26, 5, 30, tzinfo=UTC)
    for sequence_no in range(1, 7):
        run_id = f"run-{sequence_no}"
        db_session.add(
            RunRecord(
                id=run_id,
                session_id="session-1",
                sequence_no=sequence_no,
                restore_from_run_id="run-2" if sequence_no == 6 else None,
                status="running" if sequence_no == 6 else "completed",
                trigger_type="api",
                input_parts=[],
                run_metadata={},
                created_at=now + timedelta(minutes=sequence_no),
            )
        )
        run_dir = settings.run_store_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "message.json").write_text(f"[{sequence_no}]", encoding="utf-8")
    await db_session.commit()

    result = await controller.prune_once(db_session, settings)

    remaining_ids = set((await db_session.execute(select(RunRecord.id))).scalars().all())
    assert remaining_ids == {"run-1", "run-2", "run-3", "run-4", "run-5", "run-6"}
    assert result.pruned_run_store_dirs == 2
    assert result.deleted_runs == 0
    assert not (settings.run_store_dir / "run-1").exists()
    assert not (settings.run_store_dir / "run-3").exists()
    assert (settings.run_store_dir / "run-2").exists()
    assert (settings.run_store_dir / "run-6").exists()


async def test_prune_generated_heartbeat_sessions_keeps_recent_group(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    settings.session_prune_generated_sessions_enabled = True
    settings.session_prune_heartbeat_keep_recent = 2
    settings.session_prune_heartbeat_older_than_days = 0
    controller = SessionPruneController()
    now = datetime(2026, 4, 26, 5, 30, tzinfo=UTC)
    for index in range(5):
        session_id = f"heartbeat-session-{index}"
        run_id = f"heartbeat-run-{index}"
        fire_id = f"heartbeat-fire-{index}"
        db_session.add(SessionRecord(id=session_id, profile_name="default"))
        db_session.add(
            RunRecord(
                id=run_id,
                session_id=session_id,
                sequence_no=1,
                status="completed",
                trigger_type="heartbeat",
                input_parts=[],
                run_metadata={"source": "heartbeat", "heartbeat_fire_id": fire_id},
                created_at=now + timedelta(minutes=index),
            )
        )
        db_session.add(
            HeartbeatFireRecord(
                id=fire_id,
                scheduled_at=now + timedelta(minutes=index),
                fired_at=now + timedelta(minutes=index),
                status="submitted",
                dedupe_key=fire_id,
                session_id=session_id,
                run_id=run_id,
                fire_metadata={},
                created_at=now + timedelta(minutes=index),
            )
        )
        run_dir = settings.run_store_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "message.json").write_text(f"[{index}]", encoding="utf-8")
    await db_session.commit()

    result = await controller.prune_once(db_session, settings)

    remaining_sessions = set((await db_session.execute(select(SessionRecord.id))).scalars().all())
    assert remaining_sessions == {"heartbeat-session-3", "heartbeat-session-4"}
    assert result.deleted_sessions == 3
    assert result.deleted_runs == 3
    assert not (settings.run_store_dir / "heartbeat-run-0").exists()
    assert (settings.run_store_dir / "heartbeat-run-4").exists()


async def test_prune_generated_schedule_sessions_groups_by_schedule_id(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    settings.session_prune_generated_sessions_enabled = True
    settings.session_prune_schedule_keep_recent = 1
    settings.session_prune_schedule_older_than_days = 0
    controller = SessionPruneController()
    now = datetime(2026, 4, 26, 5, 30, tzinfo=UTC)
    db_session.add(
        ScheduleRecord(
            id="schedule-1",
            name="schedule",
            cron_expr="* * * * *",
            timezone="UTC",
            execution_mode="isolate_session",
            input_parts_template=[],
            status="active",
        )
    )
    for index in range(3):
        session_id = f"schedule-session-{index}"
        run_id = f"schedule-run-{index}"
        fire_id = f"schedule-fire-{index}"
        db_session.add(SessionRecord(id=session_id, profile_name="default", session_metadata={"source": "schedule"}))
        db_session.add(
            RunRecord(
                id=run_id,
                session_id=session_id,
                sequence_no=1,
                status="completed",
                trigger_type="schedule",
                input_parts=[],
                run_metadata={"source": "schedule", "schedule_id": "schedule-1", "schedule_fire_id": fire_id},
                created_at=now + timedelta(minutes=index),
            )
        )
        db_session.add(
            ScheduleFireRecord(
                id=fire_id,
                schedule_id="schedule-1",
                scheduled_at=now + timedelta(minutes=index),
                fired_at=now + timedelta(minutes=index),
                status="submitted",
                dedupe_key=fire_id,
                created_session_id=session_id,
                run_id=run_id,
                input_parts=[],
                fire_metadata={},
                created_at=now + timedelta(minutes=index),
            )
        )
        (settings.run_store_dir / run_id).mkdir(parents=True, exist_ok=True)
    await db_session.commit()

    result = await controller.prune_once(db_session, settings)

    remaining_sessions = set((await db_session.execute(select(SessionRecord.id))).scalars().all())
    assert remaining_sessions == {"schedule-session-2"}
    assert result.deleted_sessions == 2
    assert result.deleted_runs == 2


async def test_prune_orphan_run_store_dirs(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    settings.session_prune_generated_sessions_enabled = False
    settings.session_prune_orphans_enabled = True
    controller = SessionPruneController()
    db_session.add(SessionRecord(id="session-1", profile_name="default"))
    db_session.add(
        RunRecord(
            id="run-existing",
            session_id="session-1",
            sequence_no=1,
            status="completed",
            trigger_type="api",
            input_parts=[],
            run_metadata={},
        )
    )
    existing_dir = settings.run_store_dir / "run-existing"
    orphan_dir = settings.run_store_dir / "run-orphan"
    existing_dir.mkdir(parents=True, exist_ok=True)
    orphan_dir.mkdir(parents=True, exist_ok=True)
    (orphan_dir / "message.json").write_text("[]", encoding="utf-8")
    await db_session.commit()

    result = await controller.prune_once(db_session, settings)

    assert result.deleted_orphan_run_dirs == 1
    assert existing_dir.exists()
    assert not orphan_dir.exists()


async def test_prune_fire_records_keeps_pending_and_latest(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    settings.session_prune_fire_records_older_than_days = 1
    settings.session_prune_orphans_enabled = False
    controller = SessionPruneController()
    old = datetime(2026, 4, 20, 5, 30, tzinfo=UTC)
    recent = datetime(2026, 4, 26, 5, 30, tzinfo=UTC)
    db_session.add(
        ScheduleRecord(
            id="schedule-1",
            name="schedule",
            cron_expr="* * * * *",
            timezone="UTC",
            execution_mode="isolate_session",
            input_parts_template=[],
            status="active",
        )
    )
    db_session.add_all([
        ScheduleFireRecord(
            id="schedule-fire-old",
            schedule_id="schedule-1",
            scheduled_at=old,
            status="submitted",
            dedupe_key="schedule-fire-old",
            input_parts=[],
            fire_metadata={},
            created_at=old,
        ),
        ScheduleFireRecord(
            id="schedule-fire-pending",
            schedule_id="schedule-1",
            scheduled_at=old + timedelta(minutes=1),
            status="pending",
            dedupe_key="schedule-fire-pending",
            input_parts=[],
            fire_metadata={},
            created_at=old + timedelta(minutes=1),
        ),
        ScheduleFireRecord(
            id="schedule-fire-latest",
            schedule_id="schedule-1",
            scheduled_at=recent,
            status="submitted",
            dedupe_key="schedule-fire-latest",
            input_parts=[],
            fire_metadata={},
            created_at=recent,
        ),
        HeartbeatFireRecord(
            id="heartbeat-fire-old",
            scheduled_at=old,
            status="submitted",
            dedupe_key="heartbeat-fire-old",
            fire_metadata={},
            created_at=old,
        ),
        HeartbeatFireRecord(
            id="heartbeat-fire-latest",
            scheduled_at=recent,
            status="submitted",
            dedupe_key="heartbeat-fire-latest",
            fire_metadata={},
            created_at=recent,
        ),
    ])
    await db_session.commit()

    result = await controller.prune_once(db_session, settings)

    schedule_fire_ids = set((await db_session.execute(select(ScheduleFireRecord.id))).scalars().all())
    heartbeat_fire_ids = set((await db_session.execute(select(HeartbeatFireRecord.id))).scalars().all())
    assert result.deleted_schedule_fires == 1
    assert result.deleted_heartbeat_fires == 1
    assert schedule_fire_ids == {"schedule-fire-pending", "schedule-fire-latest"}
    assert heartbeat_fire_ids == {"heartbeat-fire-latest"}


async def test_prune_hides_expired_once_schedules(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    settings.session_prune_once_schedules_hide_after_days = 1
    settings.session_prune_orphans_enabled = False
    controller = SessionPruneController()
    old = datetime(2026, 4, 20, 5, 30, tzinfo=UTC)
    db_session.add(
        ScheduleRecord(
            id="once-old",
            name="old once",
            trigger_kind="once",
            run_at=old,
            timezone="UTC",
            execution_mode="isolate_session",
            input_parts_template=[],
            status="completed",
            last_fire_at=old,
            schedule_metadata={"source": "test"},
            updated_at=old,
        )
    )
    db_session.add(
        ScheduleFireRecord(
            id="once-old-fire",
            schedule_id="once-old",
            scheduled_at=old,
            fired_at=old,
            status="submitted",
            dedupe_key="once-old-fire",
            input_parts=[],
            fire_metadata={},
            created_at=old,
        )
    )
    await db_session.commit()

    result = await controller.prune_once(db_session, settings)

    record = await db_session.get(ScheduleRecord, "once-old")
    assert isinstance(record, ScheduleRecord)
    assert result.hidden_once_schedules == 1
    assert record.status == "deleted"
    assert record.next_fire_at is None
    assert record.schedule_metadata["source"] == "test"
    assert record.schedule_metadata["auto_hidden"] is True
    assert record.schedule_metadata["auto_hidden_reason"] == "expired_once_schedule"
    assert isinstance(record.schedule_metadata["auto_hidden_at"], str)


async def test_prune_hides_expired_once_schedules_preserves_recent_pending_and_active_runs(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    settings.session_prune_once_schedules_hide_after_days = 1
    settings.session_prune_orphans_enabled = False
    settings.session_prune_batch_size = 10
    controller = SessionPruneController()
    old = datetime(2026, 4, 20, 5, 30, tzinfo=UTC)
    recent = datetime.now(UTC)
    db_session.add(SessionRecord(id="session-active", profile_name="default"))
    db_session.add(
        RunRecord(
            id="run-active",
            session_id="session-active",
            sequence_no=1,
            status="running",
            trigger_type="schedule",
            input_parts=[],
            run_metadata={},
            created_at=old,
        )
    )
    schedules = [
        ScheduleRecord(
            id="once-recent",
            name="recent once",
            trigger_kind="once",
            run_at=recent,
            timezone="UTC",
            execution_mode="isolate_session",
            input_parts_template=[],
            status="completed",
            last_fire_at=recent,
            schedule_metadata={},
        ),
        ScheduleRecord(
            id="once-pending",
            name="pending once",
            trigger_kind="once",
            run_at=old,
            timezone="UTC",
            execution_mode="isolate_session",
            input_parts_template=[],
            status="completed",
            last_fire_at=old,
            schedule_metadata={},
        ),
        ScheduleRecord(
            id="once-active-run",
            name="active run once",
            trigger_kind="once",
            run_at=old,
            timezone="UTC",
            execution_mode="isolate_session",
            input_parts_template=[],
            status="completed",
            last_fire_at=old,
            schedule_metadata={},
        ),
        ScheduleRecord(
            id="cron-old",
            name="cron old",
            trigger_kind="cron",
            cron_expr="* * * * *",
            timezone="UTC",
            execution_mode="isolate_session",
            input_parts_template=[],
            status="completed",
            last_fire_at=old,
            schedule_metadata={},
        ),
    ]
    db_session.add_all(schedules)
    db_session.add_all([
        ScheduleFireRecord(
            id="once-pending-fire",
            schedule_id="once-pending",
            scheduled_at=old,
            status="pending",
            dedupe_key="once-pending-fire",
            input_parts=[],
            fire_metadata={},
            created_at=old,
        ),
        ScheduleFireRecord(
            id="once-active-run-fire",
            schedule_id="once-active-run",
            scheduled_at=old,
            status="submitted",
            dedupe_key="once-active-run-fire",
            run_id="run-active",
            input_parts=[],
            fire_metadata={},
            created_at=old,
        ),
    ])
    await db_session.commit()

    result = await controller.prune_once(db_session, settings)

    statuses = dict((await db_session.execute(select(ScheduleRecord.id, ScheduleRecord.status))).all())
    assert result.hidden_once_schedules == 0
    assert statuses == {
        "once-recent": "completed",
        "once-pending": "completed",
        "once-active-run": "completed",
        "cron-old": "completed",
    }


async def test_prune_once_schedule_hiding_can_be_disabled(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    settings.session_prune_once_schedules_hide_after_days = 0
    settings.session_prune_orphans_enabled = False
    controller = SessionPruneController()
    old = datetime(2026, 4, 20, 5, 30, tzinfo=UTC)
    db_session.add(
        ScheduleRecord(
            id="once-old-disabled",
            name="old once disabled",
            trigger_kind="once",
            run_at=old,
            timezone="UTC",
            execution_mode="isolate_session",
            input_parts_template=[],
            status="completed",
            last_fire_at=old,
            schedule_metadata={},
        )
    )
    await db_session.commit()

    result = await controller.prune_once(db_session, settings)

    record = await db_session.get(ScheduleRecord, "once-old-disabled")
    assert isinstance(record, ScheduleRecord)
    assert result.hidden_once_schedules == 0
    assert record.status == "completed"


async def test_prune_preserves_head_success_run_for_future_continuation(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    settings.session_prune_run_keep_recent = 1
    controller = SessionPruneController()
    session = SessionRecord(
        id="session-continue",
        profile_name="default",
        head_run_id="run-5",
        head_success_run_id="run-2",
    )
    db_session.add(session)
    now = datetime(2026, 4, 26, 5, 30, tzinfo=UTC)
    for sequence_no in range(1, 6):
        run_id = f"run-{sequence_no}"
        db_session.add(
            RunRecord(
                id=run_id,
                session_id="session-continue",
                sequence_no=sequence_no,
                status="completed" if sequence_no == 2 else "failed",
                trigger_type="api",
                input_parts=[],
                run_metadata={},
                created_at=now + timedelta(minutes=sequence_no),
            )
        )
        run_dir = settings.run_store_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "state.json").write_text(f'{{"sequence_no": {sequence_no}}}', encoding="utf-8")
    await db_session.commit()

    result = await controller.prune_once(db_session, settings)

    session_record = await db_session.get(SessionRecord, "session-continue")
    head_success_run = await db_session.get(RunRecord, "run-2")
    latest_run = await db_session.get(RunRecord, "run-5")
    assert isinstance(session_record, SessionRecord)
    assert isinstance(head_success_run, RunRecord)
    assert isinstance(latest_run, RunRecord)
    assert session_record.head_success_run_id == "run-2"
    assert result.pruned_run_store_dirs == 3
    assert result.deleted_runs == 0
    assert (settings.run_store_dir / "run-2" / "state.json").exists()
    assert (settings.run_store_dir / "run-5" / "state.json").exists()


async def test_prune_generated_sessions_preserves_active_schedule_source_and_target(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    settings.session_prune_generated_sessions_enabled = True
    settings.session_prune_schedule_keep_recent = 1
    settings.session_prune_schedule_older_than_days = 0
    controller = SessionPruneController()
    now = datetime(2026, 4, 26, 5, 30, tzinfo=UTC)
    for session_id in ("target-session", "source-session"):
        run_id = f"{session_id}-run"
        db_session.add(
            SessionRecord(
                id=session_id,
                profile_name="default",
                head_run_id=run_id,
                head_success_run_id=run_id,
            )
        )
        db_session.add(
            RunRecord(
                id=run_id,
                session_id=session_id,
                sequence_no=1,
                status="completed",
                trigger_type="schedule",
                input_parts=[],
                run_metadata={},
                created_at=now,
            )
        )
        (settings.run_store_dir / run_id).mkdir(parents=True, exist_ok=True)
    db_session.add(
        ScheduleRecord(
            id="schedule-continue",
            name="continue schedule",
            cron_expr="* * * * *",
            timezone="UTC",
            execution_mode="continue_session",
            target_session_id="target-session",
            input_parts_template=[],
            status="active",
        )
    )
    db_session.add(
        ScheduleRecord(
            id="schedule-fork",
            name="fork schedule",
            cron_expr="* * * * *",
            timezone="UTC",
            execution_mode="fork_session",
            source_session_id="source-session",
            input_parts_template=[],
            status="active",
        )
    )
    db_session.add_all([
        ScheduleFireRecord(
            id="target-fire",
            schedule_id="schedule-continue",
            scheduled_at=now - timedelta(days=10),
            status="submitted",
            dedupe_key="target-fire",
            target_session_id="target-session",
            created_session_id="target-session",
            run_id="target-session-run",
            input_parts=[],
            fire_metadata={},
            created_at=now - timedelta(days=10),
        ),
        ScheduleFireRecord(
            id="source-fire",
            schedule_id="schedule-fork",
            scheduled_at=now - timedelta(days=10),
            status="submitted",
            dedupe_key="source-fire",
            source_session_id="source-session",
            created_session_id="source-session",
            run_id="source-session-run",
            input_parts=[],
            fire_metadata={},
            created_at=now - timedelta(days=10),
        ),
    ])
    await db_session.commit()

    result = await controller.prune_once(db_session, settings)

    remaining_sessions = set((await db_session.execute(select(SessionRecord.id))).scalars().all())
    remaining_runs = set((await db_session.execute(select(RunRecord.id))).scalars().all())
    assert "target-session" in remaining_sessions
    assert "source-session" in remaining_sessions
    assert "target-session-run" in remaining_runs
    assert "source-session-run" in remaining_runs
    assert result.deleted_sessions == 0
    assert result.deleted_runs == 0
    assert result.pruned_run_store_dirs == 0


async def test_prune_generated_sessions_preserves_external_restore_references(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    settings.session_prune_generated_sessions_enabled = True
    settings.session_prune_heartbeat_keep_recent = 1
    settings.session_prune_heartbeat_older_than_days = 0
    controller = SessionPruneController()
    now = datetime(2026, 4, 26, 5, 30, tzinfo=UTC)
    db_session.add(SessionRecord(id="old-heartbeat-session", profile_name="default"))
    db_session.add(
        RunRecord(
            id="old-heartbeat-run",
            session_id="old-heartbeat-session",
            sequence_no=1,
            status="completed",
            trigger_type="heartbeat",
            input_parts=[],
            run_metadata={"source": "heartbeat"},
            created_at=now - timedelta(days=10),
        )
    )
    db_session.add(
        HeartbeatFireRecord(
            id="old-heartbeat-fire",
            scheduled_at=now - timedelta(days=10),
            status="submitted",
            dedupe_key="old-heartbeat-fire",
            session_id="old-heartbeat-session",
            run_id="old-heartbeat-run",
            fire_metadata={},
            created_at=now - timedelta(days=10),
        )
    )
    db_session.add(SessionRecord(id="new-heartbeat-session", profile_name="default"))
    db_session.add(
        RunRecord(
            id="new-heartbeat-run",
            session_id="new-heartbeat-session",
            sequence_no=1,
            status="completed",
            trigger_type="heartbeat",
            input_parts=[],
            run_metadata={"source": "heartbeat"},
            created_at=now,
        )
    )
    db_session.add(
        HeartbeatFireRecord(
            id="new-heartbeat-fire",
            scheduled_at=now,
            status="submitted",
            dedupe_key="new-heartbeat-fire",
            session_id="new-heartbeat-session",
            run_id="new-heartbeat-run",
            fire_metadata={},
            created_at=now,
        )
    )
    db_session.add(SessionRecord(id="consumer-session", profile_name="default"))
    db_session.add(
        RunRecord(
            id="consumer-run",
            session_id="consumer-session",
            sequence_no=1,
            restore_from_run_id="old-heartbeat-run",
            status="completed",
            trigger_type="api",
            input_parts=[],
            run_metadata={},
            created_at=now + timedelta(minutes=1),
        )
    )
    await db_session.commit()

    result = await controller.prune_once(db_session, settings)

    old_session = await db_session.get(SessionRecord, "old-heartbeat-session")
    old_run = await db_session.get(RunRecord, "old-heartbeat-run")
    assert isinstance(old_session, SessionRecord)
    assert isinstance(old_run, RunRecord)
    assert result.deleted_sessions == 0
    assert result.deleted_runs == 0
    assert result.pruned_run_store_dirs == 0
