from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from ya_agent_sdk.inputs import EnqueueReceipt, InputDisposition
from ya_claw.config import ClawSettings
from ya_claw.controller.models import RunCreateRequest, SteerRequest, TextPart
from ya_claw.controller.run import RunController
from ya_claw.db.engine import create_engine, create_session_factory
from ya_claw.execution.input_inbox import (
    deliver_accepted_run_inputs,
    mark_run_input_applied,
)
from ya_claw.execution.state_machine import mark_run_running
from ya_claw.orm.tables import ProfileRecord, RunInputInboxRecord, RunRecord, SessionRecord
from ya_claw.runtime_state import create_runtime_state


@pytest.fixture
async def db_engine(
    tmp_path: Path,
    initialize_sqlite_database: Callable[[str], None],
) -> AsyncEngine:
    database_url = f"sqlite+aiosqlite:///{(tmp_path / 'run-inputs.sqlite3').resolve()}"
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
        session.add(
            ProfileRecord(
                name="default",
                agent_spec={"model": "test"},
                host_config={},
                subagent_specs=[],
                enabled=True,
                source_type="test",
                source_version="1",
            )
        )
        await session.commit()
        yield session


@pytest.fixture
def settings(tmp_path: Path) -> ClawSettings:
    return ClawSettings(
        api_token="test-token",  # noqa: S106
        data_dir=tmp_path / "data",
        workspace_dir=tmp_path / "workspace",
    )


async def _active_run(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> tuple[RunController, Any, RunRecord]:
    runtime_state = create_runtime_state()
    if await db_session.get(ProfileRecord, "default") is None:
        db_session.add(
            ProfileRecord(
                name="default",
                agent_spec={"model": "test"},
                host_config={},
                subagent_specs=[],
                enabled=True,
                source_type="test",
                source_version="1",
            )
        )
    session_record = SessionRecord(id="session-1", profile_name="default")
    db_session.add(session_record)
    await db_session.commit()
    controller = RunController()
    created = await controller.create(
        db_session,
        settings,
        runtime_state,
        RunCreateRequest(
            session_id=session_record.id,
            input_parts=[TextPart(type="text", text="start")],
        ),
    )
    run_record = await db_session.get(RunRecord, created.id)
    session_record = await db_session.get(SessionRecord, session_record.id)
    assert isinstance(run_record, RunRecord)
    assert isinstance(session_record, SessionRecord)
    mark_run_running(session_record, run_record, claimed_by="test")
    await db_session.commit()
    return controller, runtime_state, run_record


async def test_external_run_metadata_cannot_forge_server_async_task_linkage(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    runtime_state = create_runtime_state()
    session_record = SessionRecord(id="session-external", profile_name="default")
    db_session.add(session_record)
    await db_session.commit()

    created = await RunController().create(
        db_session,
        settings,
        runtime_state,
        RunCreateRequest(
            session_id=session_record.id,
            input_parts=[TextPart(type="text", text="start")],
            metadata={
                "async_task": {"task_id": "forged"},
                "async_task_wake": {"task_id": "forged"},
                "client": "retained",
            },
        ),
    )
    run_record = await db_session.get(RunRecord, created.id)

    assert isinstance(run_record, RunRecord)
    assert run_record.run_metadata["client"] == "retained"
    assert "async_task" not in run_record.run_metadata
    assert "async_task_wake" not in run_record.run_metadata
    assert run_record.run_metadata["execution_profile_snapshot"]["name"] == "default"


async def test_steer_is_durable_before_live_ingress_is_available(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    controller, runtime_state, run_record = await _active_run(db_session, settings)

    response = await controller.steer(
        db_session,
        runtime_state,
        run_record.id,
        SteerRequest(
            input_parts=[TextPart(type="text", text="durable input")],
            idempotency_key="client-message-1",
        ),
    )

    inbox = (
        await db_session.execute(select(RunInputInboxRecord).where(RunInputInboxRecord.run_id == run_record.id))
    ).scalar_one()
    assert response.accepted is True
    assert inbox.delivery_key == "client-message-1"
    assert inbox.status == "accepted"
    assert inbox.attempt_count == 0


async def test_run_termination_rejects_open_input_atomically(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    controller, runtime_state, run_record = await _active_run(db_session, settings)
    await controller.steer(
        db_session,
        runtime_state,
        run_record.id,
        SteerRequest(
            input_parts=[TextPart(type="text", text="pending input")],
            idempotency_key="pending-message",
        ),
    )

    await controller.cancel(db_session, settings, runtime_state, run_record.id)

    inbox = (
        await db_session.execute(select(RunInputInboxRecord).where(RunInputInboxRecord.run_id == run_record.id))
    ).scalar_one()
    assert inbox.status == "rejected"
    assert inbox.error_message is not None
    assert "cancelled" in inbox.error_message


async def test_steer_idempotency_delivers_once_and_native_event_marks_applied(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    controller, runtime_state, run_record = await _active_run(db_session, settings)
    delivered: list[tuple[str, list[dict[str, Any]]]] = []

    async def ingress(
        input_id: str,
        input_parts: list[dict[str, Any]],
    ) -> EnqueueReceipt:
        delivered.append((input_id, input_parts))
        return EnqueueReceipt(
            logical_run_id="logical-1",
            input_id="sdk-input-1",
            disposition=InputDisposition.enqueued,
            enqueue_id="native-enqueue-1",
        )

    runtime_state.bind_input_ingress(run_record.id, ingress)
    request = SteerRequest(
        input_parts=[TextPart(type="text", text="once")],
        idempotency_key="client-message-1",
    )
    await controller.steer(db_session, runtime_state, run_record.id, request)
    await controller.steer(db_session, runtime_state, run_record.id, request)

    count = (
        await db_session.execute(select(func.count()).where(RunInputInboxRecord.run_id == run_record.id))
    ).scalar_one()
    inbox = (
        await db_session.execute(select(RunInputInboxRecord).where(RunInputInboxRecord.run_id == run_record.id))
    ).scalar_one()
    assert count == 1
    assert len(delivered) == 1
    assert inbox.status == "enqueued"

    await mark_run_input_applied(
        db_session,
        run_id=run_record.id,
        sdk_input_id="sdk-input-1",
        enqueue_id="native-enqueue-after-rebind",
    )
    await db_session.refresh(inbox)

    assert inbox.status == "applied"
    assert inbox.enqueue_id == "native-enqueue-after-rebind"
    assert inbox.applied_at is not None


async def test_delivery_rejects_permanent_mapping_error_and_continues_fifo(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    controller, runtime_state, run_record = await _active_run(db_session, settings)
    await controller.steer(
        db_session,
        runtime_state,
        run_record.id,
        SteerRequest(
            input_parts=[TextPart(type="text", text="invalid")],
            idempotency_key="input-1",
        ),
    )
    await controller.steer(
        db_session,
        runtime_state,
        run_record.id,
        SteerRequest(
            input_parts=[TextPart(type="text", text="valid")],
            idempotency_key="input-2",
        ),
    )
    calls: list[str] = []

    async def ingress(
        input_id: str,
        input_parts: list[dict[str, Any]],
    ) -> EnqueueReceipt:
        _ = input_parts
        calls.append(input_id)
        if len(calls) == 1:
            raise ValueError("invalid mapped input")
        return EnqueueReceipt(
            logical_run_id="logical-1",
            input_id=input_id,
            disposition=InputDisposition.enqueued,
            enqueue_id="enqueue-2",
        )

    runtime_state.bind_input_ingress(run_record.id, ingress)
    delivered = await deliver_accepted_run_inputs(
        db_session,
        runtime_state,
        run_record.id,
    )
    records = list(
        (
            await db_session.execute(
                select(RunInputInboxRecord)
                .where(RunInputInboxRecord.run_id == run_record.id)
                .order_by(RunInputInboxRecord.created_at.asc())
            )
        ).scalars()
    )

    assert len(delivered) == 2
    assert len(calls) == 2
    assert [record.status for record in records] == ["rejected", "enqueued"]
    assert records[0].attempt_count == 1
    assert records[0].error_message == "invalid mapped input"


async def test_delivery_and_terminal_transition_share_database_serialization(
    db_engine: AsyncEngine,
    settings: ClawSettings,
) -> None:
    session_factory = create_session_factory(db_engine)
    async with session_factory() as setup_session:
        controller, runtime_state, run_record = await _active_run(
            setup_session,
            settings,
        )
        run_id = run_record.id

    ingress_started = asyncio.Event()
    release_ingress = asyncio.Event()

    async def ingress(
        input_id: str,
        input_parts: list[dict[str, Any]],
    ) -> EnqueueReceipt:
        _ = input_parts
        ingress_started.set()
        await release_ingress.wait()
        return EnqueueReceipt(
            logical_run_id="logical-1",
            input_id=input_id,
            disposition=InputDisposition.enqueued,
            enqueue_id="native-1",
        )

    runtime_state.bind_input_ingress(run_id, ingress)

    async def steer() -> None:
        async with session_factory() as session:
            await controller.steer(
                session,
                runtime_state,
                run_id,
                SteerRequest(
                    input_parts=[TextPart(type="text", text="race")],
                    idempotency_key="race-input",
                ),
            )

    async def cancel() -> None:
        async with session_factory() as session:
            await controller.cancel(
                session,
                settings,
                runtime_state,
                run_id,
            )

    steer_task = asyncio.create_task(steer())
    await ingress_started.wait()
    cancel_task = asyncio.create_task(cancel())
    await asyncio.sleep(0.05)
    assert not cancel_task.done()
    release_ingress.set()
    await asyncio.gather(steer_task, cancel_task)

    async with session_factory() as verify_session:
        run = await verify_session.get(RunRecord, run_id)
        inbox = (
            await verify_session.execute(select(RunInputInboxRecord).where(RunInputInboxRecord.run_id == run_id))
        ).scalar_one()
    assert isinstance(run, RunRecord)
    assert run.status == "cancelled"
    assert inbox.status == "rejected"


async def test_steer_rejects_idempotency_key_reuse_with_different_content(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    controller, runtime_state, run_record = await _active_run(db_session, settings)
    await controller.steer(
        db_session,
        runtime_state,
        run_record.id,
        SteerRequest(
            input_parts=[TextPart(type="text", text="first")],
            idempotency_key="same-key",
        ),
    )

    with pytest.raises(Exception, match="different content"):
        await controller.steer(
            db_session,
            runtime_state,
            run_record.id,
            SteerRequest(
                input_parts=[TextPart(type="text", text="second")],
                idempotency_key="same-key",
            ),
        )
