from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from ya_claw.config import ClawSettings
from ya_claw.controller.models import RunCreateRequest, SessionCreateRequest, SessionRunCreateRequest, SteerRequest
from ya_claw.controller.run import RunController
from ya_claw.controller.session import SessionController
from ya_claw.db.engine import create_engine, create_session_factory
from ya_claw.orm.base import Base
from ya_claw.orm.tables import RunRecord, SessionRecord
from ya_claw.runtime_state import create_runtime_state


@pytest.fixture
async def db_engine(tmp_path: Path) -> AsyncEngine:
    engine = create_engine(f"sqlite+aiosqlite:///{(tmp_path / 'controller.sqlite3').resolve()}")
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


async def test_session_controller_creates_session_and_initial_run(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    controller = SessionController()
    runtime_state = create_runtime_state()

    response = await controller.create(
        db_session,
        settings,
        runtime_state,
        SessionCreateRequest(
            profile_name="general",
            metadata={"source": "api"},
            input_parts=[{"type": "text", "text": "hello from api"}],
        ),
    )

    sessions = await controller.list(db_session)

    assert response.run is not None
    assert response.session.id == response.run.session_id
    assert response.session.status == "queued"
    assert response.session.status_reason == "run_queued"
    assert response.session.status_detail["run_id"] == response.run.id
    assert response.session.head_run_id == response.run.id
    assert response.session.head_success_run_id is None
    assert response.run.input_preview == "hello from api"
    assert len(sessions) == 1
    assert sessions[0].latest_run is not None
    assert sessions[0].latest_run.id == response.run.id


async def test_session_controller_lists_lightweight_keyset_pages(
    db_session: AsyncSession,
    settings: ClawSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = SessionController()
    runtime_state = create_runtime_state()
    created_sessions = []
    for index in range(3):
        created = await controller.create(
            db_session,
            settings,
            runtime_state,
            SessionCreateRequest(
                input_parts=[{"type": "text", "text": f"prompt-{index}"}],
            ),
        )
        assert created.run is not None
        run_record = await db_session.get(RunRecord, created.run.id)
        session_record = await db_session.get(SessionRecord, created.session.id)
        assert isinstance(run_record, RunRecord)
        assert isinstance(session_record, SessionRecord)
        run_record.output_text = "large output" * 100
        session_record.updated_at = datetime(2026, 1, 1, tzinfo=UTC)
        created_sessions.append(created.session.id)
    await db_session.commit()

    async def unexpected_workspace_reconcile(*args: object, **kwargs: object) -> None:
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(controller, "_reconcile_workspace_states", unexpected_workspace_reconcile)
    first_page = await controller.list_page(db_session, settings=settings, limit=2)
    second_page = await controller.list_page(
        db_session,
        settings=settings,
        limit=2,
        before_updated_at=first_page.next_before_updated_at,
        before_id=first_page.next_before_id,
    )

    assert first_page.total == 3
    assert first_page.has_more is True
    assert first_page.next_before_updated_at is not None
    assert first_page.next_before_id is not None
    assert len(first_page.sessions) == 2
    assert all(session.latest_run is not None for session in first_page.sessions)
    assert all(session.latest_run.output_text is None for session in first_page.sessions if session.latest_run)
    assert second_page.has_more is False
    assert len(second_page.sessions) == 1
    assert {session.id for session in [*first_page.sessions, *second_page.sessions]} == set(created_sessions)


async def test_session_controller_get_embeds_paginated_runs_with_optional_message(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    session_controller = SessionController()
    run_controller = RunController()
    runtime_state = create_runtime_state()

    created = await session_controller.create(db_session, settings, runtime_state, SessionCreateRequest())
    first_run = await run_controller.create(
        db_session,
        settings,
        runtime_state,
        RunCreateRequest(
            session_id=created.session.id,
            input_parts=[{"type": "text", "text": "hello-1"}],
            metadata={"source": "tool"},
        ),
    )
    second_run = await run_controller.create(
        db_session,
        settings,
        runtime_state,
        RunCreateRequest(
            session_id=created.session.id,
            input_parts=[{"type": "text", "text": "hello-2"}],
            metadata={"source": "tool"},
        ),
    )

    now = datetime.now(UTC)
    for index, run_id in enumerate((first_run.id, second_run.id), start=1):
        run_record = await db_session.get(RunRecord, run_id)
        assert isinstance(run_record, RunRecord)
        run_record.status = "completed"
        run_record.started_at = now - timedelta(seconds=3 - index)
        run_record.finished_at = now - timedelta(seconds=2 - index)
        run_record.committed_at = now - timedelta(seconds=2 - index)

    session_record = await db_session.get(SessionRecord, created.session.id)
    assert isinstance(session_record, SessionRecord)
    session_record.active_run_id = None
    session_record.head_success_run_id = second_run.id
    await db_session.commit()

    first_dir = settings.run_store_dir / first_run.id
    second_dir = settings.run_store_dir / second_run.id
    first_dir.mkdir(parents=True, exist_ok=True)
    second_dir.mkdir(parents=True, exist_ok=True)
    (first_dir / "message.json").write_text(
        '[{"type": "message", "content": "first"}]',
        encoding="utf-8",
    )
    (second_dir / "message.json").write_text(
        '[{"type": "message", "content": "second"}]',
        encoding="utf-8",
    )

    payload = await session_controller.get(
        db_session,
        settings,
        created.session.id,
        runs_limit=1,
        include_message=True,
    )
    next_page = await session_controller.get(
        db_session,
        settings,
        created.session.id,
        runs_limit=1,
        before_sequence_no=2,
        include_message=True,
    )

    assert payload.session.head_success_run_id == second_run.id
    assert payload.session.runs[0].id == second_run.id
    assert payload.session.runs[0].message == [{"type": "message", "content": "second"}]
    assert payload.session.runs_has_more is True
    assert payload.session.runs_next_before_sequence_no == 2

    assert next_page.session.runs[0].id == first_run.id
    assert next_page.session.runs[0].message == [{"type": "message", "content": "first"}]
    assert next_page.session.runs_has_more is False


async def test_session_controller_create_run_supports_reset_state(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    controller = SessionController()
    runtime_state = create_runtime_state()

    session_record = SessionRecord(
        id="session-1",
        profile_name="general",
        session_metadata={"sandbox": {"container_ref": "ya-claw-workspace-ref", "container_id": "cid-1"}},
        head_run_id="run-1",
        head_success_run_id="run-1",
    )
    base_run = RunRecord(
        id="run-1",
        session_id="session-1",
        sequence_no=1,
        restore_from_run_id=None,
        status="completed",
        trigger_type="api",
        profile_name="general",
        input_parts=[{"type": "text", "text": "base"}],
        run_metadata={},
    )
    db_session.add(session_record)
    db_session.add(base_run)
    await db_session.commit()

    rerun = await controller.create_run(
        db_session,
        settings,
        runtime_state,
        "session-1",
        SessionRunCreateRequest(
            reset_state=True,
            input_parts=[{"type": "text", "text": "fresh start"}],
        ),
    )

    refreshed_session = await db_session.get(SessionRecord, "session-1")
    refreshed_run = await db_session.get(RunRecord, rerun.id)
    assert isinstance(refreshed_session, SessionRecord)
    assert isinstance(refreshed_run, RunRecord)
    await db_session.refresh(refreshed_session)
    await db_session.refresh(refreshed_run)

    assert rerun.restore_from_run_id is None
    assert refreshed_run.restore_from_run_id is None
    assert rerun.metadata["reset_state"] is True
    assert rerun.metadata["restore_state"] is False
    assert refreshed_session.session_metadata == {
        "sandbox": {"container_ref": "ya-claw-workspace-ref", "container_id": "cid-1"}
    }


async def test_run_controller_create_auto_creates_session_and_supports_steer_cancel(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    controller = RunController()
    runtime_state = create_runtime_state()

    run = await controller.create(
        db_session,
        settings,
        runtime_state,
        RunCreateRequest(
            session_id=None,
            profile_name="general",
            input_parts=[{"type": "text", "text": "hello"}],
            metadata={"source": "tool"},
        ),
    )

    steer_response = await controller.steer(
        db_session,
        runtime_state,
        run.id,
        SteerRequest(input_parts=[{"type": "text", "text": "focus on tests"}]),
    )
    cancelled = await controller.cancel(db_session, settings, runtime_state, run.id)

    handle = runtime_state.get_run_handle(run.id)
    session_record = await db_session.get(SessionRecord, run.session_id)
    assert isinstance(session_record, SessionRecord)
    assert run.status == "queued"
    assert run.session_id
    assert run.profile_name == "general"
    assert run.metadata == {"source": "tool"}
    assert run.input_preview == "hello"
    assert (settings.run_store_dir / run.id).exists()
    assert steer_response.run_id == run.id
    assert steer_response.accepted is True
    assert handle is not None
    assert handle.steering_inputs[0][0]["type"] == "text"
    assert session_record.active_run_id is None
    assert cancelled.status == "cancelled"
    assert cancelled.termination_reason == "cancel"
