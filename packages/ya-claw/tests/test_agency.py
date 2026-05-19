from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from ya_claw.agency.lifecycle import AGENCY_SINGLETON_SOURCE_SESSION_ID, AgencyLifecycle
from ya_claw.config import ClawSettings
from ya_claw.controller.models import AgencyFireKind, RunStatus, SessionSubmitRequest, TextPart, TriggerType
from ya_claw.controller.session import SessionController
from ya_claw.db.engine import create_engine, create_session_factory
from ya_claw.orm.base import Base
from ya_claw.orm.tables import AgencyFireRecord, RunRecord, SessionRecord
from ya_claw.runtime_state import create_runtime_state


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
        _env_file=None,
    )


def _text(value: str) -> TextPart:
    return TextPart(type="text", text=value)


async def test_message_observed_creates_singleton_agency_session_and_run(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    db_session.add(SessionRecord(id="session-1", profile_name="general", session_metadata={}))
    await db_session.commit()
    submitted: list[str] = []
    lifecycle = AgencyLifecycle(
        settings=settings,
        runtime_state=create_runtime_state(),
        submit_run=lambda run_id: not submitted.append(run_id),
    )

    delivery = await lifecycle.observe_message(
        db_session,
        source_session_id="session-1",
        source_run_id="run-1",
        input_parts=[_text("hello agency")],
        source_kind=TriggerType.API.value,
        client_token="message-1",  # noqa: S106
        metadata={"client": "test"},
    )

    assert delivery is not None
    assert delivery.delivery == "submitted"
    assert delivery.run_id is not None
    assert submitted == [delivery.run_id]
    assert delivery.agency_session.session_type == "agency"
    assert delivery.agency_session.source_session_id == AGENCY_SINGLETON_SOURCE_SESSION_ID

    fire = await db_session.get(AgencyFireRecord, delivery.fire.id)
    assert isinstance(fire, AgencyFireRecord)
    assert fire.kind == AgencyFireKind.MESSAGE_OBSERVED.value
    assert fire.status == "submitted"
    assert fire.source_session_id == "session-1"
    assert fire.source_run_id == "run-1"
    assert fire.payload["input_parts"][0]["text"] == "hello agency"
    assert fire.payload["metadata"] == {"client": "test"}

    run = await db_session.get(RunRecord, delivery.run_id)
    assert isinstance(run, RunRecord)
    assert run.session_id == delivery.agency_session.id
    assert run.trigger_type == TriggerType.AGENCY.value
    assert run.run_metadata["agency"]["fire_ids"] == [fire.id]
    assert run.run_metadata["agency"]["trigger_kinds"] == [AgencyFireKind.MESSAGE_OBSERVED.value]
    assert run.input_parts[0]["name"] == "agency_fire"


async def test_memory_session_completed_fire_carries_output(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    db_session.add(SessionRecord(id="session-1", profile_name="general", session_metadata={}))
    await db_session.commit()
    lifecycle = AgencyLifecycle(settings=settings, runtime_state=create_runtime_state())

    delivery = await lifecycle.on_memory_session_completed(
        db_session,
        source_session_id="session-1",
        memory_session_id="memory-session-1",
        memory_run_id="memory-run-1",
        memory_job_kind="extract",
        output_text="memory output text",
        output_summary="memory summary",
        payload={"source_run_ids": ["run-1"]},
        dispatch=False,
    )

    assert delivery is not None
    assert delivery.delivery == "pending"
    fire = await db_session.get(AgencyFireRecord, delivery.fire.id)
    assert isinstance(fire, AgencyFireRecord)
    assert fire.kind == AgencyFireKind.MEMORY_SESSION_COMPLETED.value
    assert fire.source_session_id == "session-1"
    assert fire.source_run_id == "memory-run-1"
    assert fire.payload["memory_session_id"] == "memory-session-1"
    assert fire.payload["memory_run_id"] == "memory-run-1"
    assert fire.payload["output_text"] == "memory output text"
    assert fire.payload["output_summary"] == "memory summary"


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

    first = await lifecycle.observe_message(
        db_session,
        source_session_id="session-1",
        source_run_id="run-1",
        input_parts=[_text("hello")],
        source_kind=TriggerType.API.value,
        client_token="message-1",  # noqa: S106
        dispatch=False,
    )
    second = await lifecycle.on_memory_session_completed(
        db_session,
        source_session_id="session-2",
        memory_session_id="memory-session-1",
        memory_run_id="memory-run-1",
        memory_job_kind="summary",
        output_text="summary output",
        output_summary="summary",
        dispatch=False,
    )
    delivery = await lifecycle.dispatch_pending(db_session)

    assert first is not None
    assert second is not None
    assert delivery.delivery == "submitted"
    assert submitted == [delivery.run_id]
    run = await db_session.get(RunRecord, delivery.run_id)
    assert isinstance(run, RunRecord)
    assert set(run.run_metadata["agency"]["fire_ids"]) == {first.fire.id, second.fire.id}
    assert set(run.run_metadata["agency"]["trigger_kinds"]) == {
        AgencyFireKind.MESSAGE_OBSERVED.value,
        AgencyFireKind.MEMORY_SESSION_COMPLETED.value,
    }
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
        session_metadata={"agency": {"kind": "claw_agency_session"}},
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

    delivery = await lifecycle.observe_message(
        db_session,
        source_session_id="session-1",
        source_run_id="run-1",
        input_parts=[_text("follow up")],
        source_kind=TriggerType.API.value,
        client_token="message-1",  # noqa: S106
    )

    assert delivery is not None
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

    first = await lifecycle.observe_message(
        db_session,
        source_session_id="session-1",
        source_run_id="run-1",
        input_parts=[_text("first")],
        source_kind=TriggerType.API.value,
        client_token="message-1",  # noqa: S106
    )
    second = await lifecycle.observe_message(
        db_session,
        source_session_id="session-2",
        source_run_id="run-2",
        input_parts=[_text("second")],
        source_kind=TriggerType.API.value,
        client_token="message-2",  # noqa: S106
    )

    assert first is not None
    assert second is not None
    assert first.delivery == "submitted"
    assert second.delivery == "merged"
    assert second.run_id == first.run_id
    assert submitted == [first.run_id]
    run = await db_session.get(RunRecord, first.run_id)
    assert isinstance(run, RunRecord)
    assert len(run.input_parts) == 2
    assert set(run.run_metadata["agency"]["source_session_ids"]) == {"session-1", "session-2"}


async def test_duplicate_message_fire_reports_duplicate(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    db_session.add(SessionRecord(id="session-1", profile_name="general", session_metadata={}))
    await db_session.commit()
    lifecycle = AgencyLifecycle(settings=settings, runtime_state=create_runtime_state())

    first = await lifecycle.observe_message(
        db_session,
        source_session_id="session-1",
        source_run_id="run-1",
        input_parts=[_text("hello")],
        source_kind=TriggerType.API.value,
        client_token="message-1",  # noqa: S106
        dispatch=False,
    )
    second = await lifecycle.observe_message(
        db_session,
        source_session_id="session-1",
        source_run_id="run-1",
        input_parts=[_text("hello")],
        source_kind=TriggerType.API.value,
        client_token="message-1",  # noqa: S106
        dispatch=False,
    )

    assert first is not None
    assert second is not None
    assert first.delivery == "pending"
    assert second.delivery == "duplicate"
    assert second.fire.id == first.fire.id
    result = await db_session.execute(select(AgencyFireRecord))
    assert len(list(result.scalars().all())) == 1


async def test_tick_dispatches_pending_fires_without_timer(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    db_session.add(SessionRecord(id="session-1", profile_name="general", session_metadata={}))
    await db_session.commit()
    submitted: list[str] = []
    lifecycle = AgencyLifecycle(
        settings=settings,
        runtime_state=create_runtime_state(),
        submit_run=lambda run_id: not submitted.append(run_id),
    )
    pending = await lifecycle.observe_message(
        db_session,
        source_session_id="session-1",
        source_run_id="run-1",
        input_parts=[_text("hello")],
        source_kind=TriggerType.API.value,
        client_token="message-1",  # noqa: S106
        dispatch=False,
    )

    result = await lifecycle.tick(db_session)
    empty = await lifecycle.tick(db_session)

    assert pending is not None
    assert result.created_fire_ids == [pending.fire.id]
    assert len(result.submitted_run_ids) == 1
    assert submitted == result.submitted_run_ids
    assert empty.created_fire_ids == []
    assert await lifecycle.next_timer_fire_at(db_session) is None


async def test_agency_run_commit_consumes_fires(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    agency_session = SessionRecord(
        id="agency-1",
        profile_name="general",
        session_type="agency",
        source_session_id=AGENCY_SINGLETON_SOURCE_SESSION_ID,
        session_metadata={"agency": {"kind": "claw_agency_session"}},
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
            kind=AgencyFireKind.MESSAGE_OBSERVED.value,
            status="submitted",
            scheduled_at=datetime.now(UTC),
            dedupe_key="fire-1",
            agency_session_id="agency-1",
            run_id="agency-run-1",
            payload={},
        ),
        AgencyFireRecord(
            id="fire-2",
            kind=AgencyFireKind.MEMORY_SESSION_COMPLETED.value,
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


async def test_session_submit_creates_merges_and_steers(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    session = SessionRecord(id="session-1", profile_name="general", session_metadata={})
    db_session.add(session)
    await db_session.commit()
    runtime_state = create_runtime_state()
    controller = SessionController()

    created = await controller.submit_input(
        db_session,
        settings,
        runtime_state,
        "session-1",
        SessionSubmitRequest(input_parts=[_text("first")]),
    )
    merged = await controller.submit_input(
        db_session,
        settings,
        runtime_state,
        "session-1",
        SessionSubmitRequest(input_parts=[_text("second")], metadata={"agency": {"fire_ids": ["fire-2"]}}),
    )
    run = await db_session.get(RunRecord, created.run_id)
    assert isinstance(run, RunRecord)
    run.status = "running"
    session.active_run_id = run.id
    await db_session.commit()
    steered = await controller.submit_input(
        db_session,
        settings,
        runtime_state,
        "session-1",
        SessionSubmitRequest(input_parts=[_text("third")]),
    )

    assert created.delivery == "submitted"
    assert merged.delivery == "merged"
    assert steered.delivery == "steered"
    assert merged.run_id == created.run_id
    await db_session.refresh(run)
    assert [part["text"] for part in run.input_parts] == ["first", "second", "third"]
    assert runtime_state.consume_steering_inputs(run.id)[0][0]["text"] == "third"


async def test_agency_submit_to_source_session_creates_handoff_run(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    from ya_claw.controller.agency import AgencyController
    from ya_claw.controller.models import AgencySourceSessionSubmitRequest

    source_session = SessionRecord(id="source-session", profile_name="general", session_metadata={})
    agency_session = SessionRecord(
        id="agency-session",
        profile_name="general",
        session_type="agency",
        source_session_id=AGENCY_SINGLETON_SOURCE_SESSION_ID,
        session_metadata={"agency": {"kind": "claw_agency_session"}},
        active_run_id="agency-run",
        head_run_id="agency-run",
    )
    agency_run = RunRecord(
        id="agency-run",
        session_id="agency-session",
        sequence_no=1,
        status=RunStatus.RUNNING.value,
        trigger_type=TriggerType.AGENCY.value,
        input_parts=[],
        run_metadata={"agency": {"fire_ids": ["fire-1"]}},
    )
    db_session.add_all([source_session, agency_session, agency_run])
    await db_session.commit()
    runtime_state = create_runtime_state()
    runtime_state.register_run("agency-session", "agency-run", dispatch_mode="async")

    response = await AgencyController().submit_to_source_session(
        db_session,
        settings,
        runtime_state,
        AgencySourceSessionSubmitRequest(
            source_session_id="source-session",
            prompt="Review the async findings and update the user.",
            metadata={"fire_ids": ["fire-1"]},
            agency_session_id="agency-session",
            agency_run_id="agency-run",
        ),
    )

    assert response.delivery == "submitted"
    assert response.source_session_id == "source-session"
    run = await db_session.get(RunRecord, response.run_id)
    assert isinstance(run, RunRecord)
    assert run.session_id == "source-session"
    assert run.trigger_type == TriggerType.AGENCY_HANDOFF.value
    assert run.input_parts[0]["text"] == "Review the async findings and update the user."
    assert run.input_parts[0]["metadata"]["source"] == "agency_handoff"
    handoff = run.run_metadata["agency_handoff"]
    assert handoff["latest"]["agency_session_id"] == "agency-session"
    assert handoff["latest"]["agency_run_id"] == "agency-run"
    assert handoff["latest"]["source_session_id"] == "source-session"
    assert handoff["latest"]["metadata"] == {"fire_ids": ["fire-1"]}
    assert handoff["handoffs"] == [handoff["latest"]]


async def test_agency_submit_to_source_session_merges_queued_source_run(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    from ya_claw.controller.agency import AgencyController
    from ya_claw.controller.models import AgencySourceSessionSubmitRequest

    source_session = SessionRecord(
        id="source-session",
        profile_name="general",
        session_metadata={},
        active_run_id="source-run",
        head_run_id="source-run",
    )
    source_run = RunRecord(
        id="source-run",
        session_id="source-session",
        sequence_no=1,
        status=RunStatus.QUEUED.value,
        trigger_type=TriggerType.API.value,
        input_parts=[{"type": "text", "text": "original"}],
        run_metadata={},
    )
    agency_session = SessionRecord(
        id="agency-session",
        profile_name="general",
        session_type="agency",
        source_session_id=AGENCY_SINGLETON_SOURCE_SESSION_ID,
        session_metadata={"agency": {"kind": "claw_agency_session"}},
    )
    agency_run = RunRecord(
        id="agency-run",
        session_id="agency-session",
        sequence_no=1,
        status=RunStatus.RUNNING.value,
        trigger_type=TriggerType.AGENCY.value,
        input_parts=[],
        run_metadata={},
    )
    db_session.add_all([source_session, source_run, agency_session, agency_run])
    await db_session.commit()
    agency_session.active_run_id = "agency-run"
    await db_session.commit()
    runtime_state = create_runtime_state()
    runtime_state.register_run("agency-session", "agency-run", dispatch_mode="async")

    response = await AgencyController().submit_to_source_session(
        db_session,
        settings,
        runtime_state,
        AgencySourceSessionSubmitRequest(
            source_session_id="source-session",
            prompt="Add agency context.",
            metadata={"source_run_ids": ["run-1"]},
            agency_session_id="agency-session",
            agency_run_id="agency-run",
        ),
    )

    assert response.delivery == "merged"
    assert response.run_id == "source-run"
    await db_session.refresh(source_run)
    assert [part["text"] for part in source_run.input_parts] == ["original", "Add agency context."]
    assert source_run.run_metadata["agency_handoff"]["latest"]["metadata"] == {"source_run_ids": ["run-1"]}
    assert len(source_run.run_metadata["agency_handoff"]["handoffs"]) == 1


async def test_agency_submit_to_source_session_steers_running_source_run(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    from ya_claw.controller.agency import AgencyController
    from ya_claw.controller.models import AgencySourceSessionSubmitRequest

    source_session = SessionRecord(
        id="source-session",
        profile_name="general",
        session_metadata={},
        active_run_id="source-run",
        head_run_id="source-run",
    )
    source_run = RunRecord(
        id="source-run",
        session_id="source-session",
        sequence_no=1,
        status=RunStatus.RUNNING.value,
        trigger_type=TriggerType.API.value,
        input_parts=[{"type": "text", "text": "original"}],
        run_metadata={},
    )
    agency_session = SessionRecord(
        id="agency-session",
        profile_name="general",
        session_type="agency",
        source_session_id=AGENCY_SINGLETON_SOURCE_SESSION_ID,
        session_metadata={"agency": {"kind": "claw_agency_session"}},
    )
    agency_run = RunRecord(
        id="agency-run",
        session_id="agency-session",
        sequence_no=1,
        status=RunStatus.RUNNING.value,
        trigger_type=TriggerType.AGENCY.value,
        input_parts=[],
        run_metadata={},
    )
    db_session.add_all([source_session, source_run, agency_session, agency_run])
    await db_session.commit()
    agency_session.active_run_id = "agency-run"
    await db_session.commit()
    runtime_state = create_runtime_state()
    runtime_state.register_run("source-session", "source-run", dispatch_mode="async")
    runtime_state.register_run("agency-session", "agency-run", dispatch_mode="async")

    response = await AgencyController().submit_to_source_session(
        db_session,
        settings,
        runtime_state,
        AgencySourceSessionSubmitRequest(
            source_session_id="source-session",
            prompt="Steer with agency context.",
            metadata={"async_task_ids": ["task-1"]},
            agency_session_id="agency-session",
            agency_run_id="agency-run",
        ),
    )

    assert response.delivery == "steered"
    assert response.run_id == "source-run"
    steering = runtime_state.consume_steering_inputs("source-run")
    assert steering[0][0]["text"] == "Steer with agency context."
    await db_session.refresh(source_run)
    assert [part["text"] for part in source_run.input_parts] == ["original", "Steer with agency context."]
    assert source_run.run_metadata["agency_handoff"]["latest"]["metadata"] == {"async_task_ids": ["task-1"]}


async def test_agency_submit_to_source_session_rejects_invalid_callers_and_targets(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    from fastapi import HTTPException
    from ya_claw.controller.agency import AgencyController
    from ya_claw.controller.models import AgencySourceSessionSubmitRequest

    controller = AgencyController()
    source_session = SessionRecord(id="source-session", profile_name="general", session_metadata={})
    memory_session = SessionRecord(
        id="memory-session",
        profile_name="general",
        session_type="memory",
        source_session_id="source-session",
        session_metadata={},
    )
    agency_session = SessionRecord(
        id="agency-session",
        profile_name="general",
        session_type="agency",
        source_session_id=AGENCY_SINGLETON_SOURCE_SESSION_ID,
        session_metadata={"agency": {"kind": "claw_agency_session"}},
    )
    queued_agency_run = RunRecord(
        id="agency-run",
        session_id="agency-session",
        sequence_no=1,
        status=RunStatus.QUEUED.value,
        trigger_type=TriggerType.AGENCY.value,
        input_parts=[],
        run_metadata={},
    )
    db_session.add_all([source_session, memory_session, agency_session, queued_agency_run])
    await db_session.commit()

    with pytest.raises(HTTPException) as queued_exc:
        await controller.submit_to_source_session(
            db_session,
            settings,
            create_runtime_state(),
            AgencySourceSessionSubmitRequest(
                source_session_id="source-session",
                prompt="hello",
                agency_session_id="agency-session",
                agency_run_id="agency-run",
            ),
        )
    assert queued_exc.value.status_code == 403

    queued_agency_run.status = RunStatus.RUNNING.value
    agency_session.active_run_id = "agency-run"
    await db_session.commit()
    runtime_state = create_runtime_state()
    runtime_state.register_run("agency-session", "agency-run", dispatch_mode="async")
    with pytest.raises(HTTPException) as target_exc:
        await controller.submit_to_source_session(
            db_session,
            settings,
            runtime_state,
            AgencySourceSessionSubmitRequest(
                source_session_id="memory-session",
                prompt="hello",
                agency_session_id="agency-session",
                agency_run_id="agency-run",
            ),
        )
    assert target_exc.value.status_code == 422
