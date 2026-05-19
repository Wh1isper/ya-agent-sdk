from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from ya_claw.bridge.models import BridgeAdapterType, BridgeEventStatus, BridgeInboundMessage
from ya_claw.config import ClawSettings
from ya_claw.controller.hitl import HitlController
from ya_claw.controller.models import ActiveInteraction, InteractionRespondRequest, TextPart
from ya_claw.db.engine import create_engine, create_session_factory
from ya_claw.execution.dispatcher import RunDispatcher
from ya_claw.orm.base import Base
from ya_claw.orm.tables import (
    HitlBatchRecord,
    HitlDeferredInputRecord,
    HitlInteractionRecord,
    RunRecord,
    SessionRecord,
)
from ya_claw.runtime_state import create_runtime_state


@pytest.fixture
async def db_engine(tmp_path: Path) -> AsyncEngine:
    engine = create_engine(f"sqlite+aiosqlite:///{(tmp_path / 'hitl.sqlite3').resolve()}")
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


async def _seed_running_run(db_session: AsyncSession) -> None:
    session = SessionRecord(id="session-1", profile_name="default", active_run_id="run-1")
    run = RunRecord(
        id="run-1",
        session_id="session-1",
        sequence_no=1,
        status="running",
        trigger_type="bridge",
        input_parts=[],
        run_metadata={},
    )
    db_session.add_all([session, run])
    await db_session.commit()


async def test_hitl_controller_persists_batch_and_advances_interactions(db_session: AsyncSession) -> None:
    await _seed_running_run(db_session)
    runtime_state = create_runtime_state()
    runtime_state.register_run("session-1", "run-1")
    interactions = [
        ActiveInteraction(
            interaction_id="hitl-1",
            run_id="run-1",
            session_id="session-1",
            tool_call_id="tool-1",
            tool_name="shell_exec",
            title="Approve shell",
            sequence_no=1,
            total_count=2,
            created_at=datetime.now(UTC),
        ),
        ActiveInteraction(
            interaction_id="hitl-2",
            run_id="run-1",
            session_id="session-1",
            tool_call_id="tool-2",
            tool_name="file_write",
            title="Approve write",
            sequence_no=2,
            total_count=2,
            created_at=datetime.now(UTC),
        ),
    ]
    runtime_state.set_hitl_pending("run-1", "session-1", interactions)
    controller = HitlController()

    batch = await controller.create_batch(
        db_session,
        session_id="session-1",
        run_id="run-1",
        interactions=interactions,
    )
    run = await db_session.get(RunRecord, "run-1")
    assert isinstance(run, RunRecord)
    run.run_metadata = {
        "active_hitl_batch_id": batch.batch_id,
        "active_interactions": [interaction.model_dump(mode="json") for interaction in interactions],
    }
    await db_session.commit()

    first = await controller.respond_interaction(
        db_session,
        runtime_state,
        "run-1",
        "hitl-1",
        InteractionRespondRequest(approved=True, reason="ok"),
    )
    await db_session.commit()

    assert first.remaining_interaction_count == 1
    assert first.current_interaction is not None
    assert first.current_interaction.interaction_id == "hitl-2"
    interaction_rows = (await db_session.execute(select(HitlInteractionRecord))).scalars().all()
    assert [row.status for row in interaction_rows] == ["approved", "pending"]

    second = await controller.respond_interaction(
        db_session,
        runtime_state,
        "run-1",
        "hitl-2",
        InteractionRespondRequest(approved=False, reason="deny"),
    )
    await db_session.commit()

    assert second.remaining_interaction_count == 0
    assert second.current_interaction is None
    stored_batch = await db_session.get(HitlBatchRecord, batch.batch_id)
    assert isinstance(stored_batch, HitlBatchRecord)
    assert stored_batch.status == "completed"
    refreshed_run = await db_session.get(RunRecord, "run-1")
    assert isinstance(refreshed_run, RunRecord)
    assert "active_interactions" not in refreshed_run.run_metadata


async def test_bridge_message_during_hitl_is_deferred_and_consumed(db_session: AsyncSession) -> None:
    from ya_claw.bridge.controller import BridgeController
    from ya_claw.orm.tables import AgencyFireRecord, BridgeConversationRecord

    await _seed_running_run(db_session)
    db_session.add(
        BridgeConversationRecord(
            id="conversation-1",
            adapter="lark",
            tenant_key="tenant-1",
            external_chat_id="oc_1",
            session_id="session-1",
        )
    )
    controller = HitlController()
    await controller.create_batch(
        db_session,
        session_id="session-1",
        run_id="run-1",
        interactions=[
            ActiveInteraction(
                interaction_id="hitl-1",
                run_id="run-1",
                session_id="session-1",
                tool_call_id="tool-1",
                tool_name="shell_exec",
                title="Approve shell",
            )
        ],
    )
    await db_session.commit()

    result = await BridgeController().handle_inbound_message(
        db_session,
        ClawSettings(api_token="test-token", agency_enabled=True, _env_file=None),  # noqa: S106
        create_runtime_state(),
        RunDispatcher(None),
        BridgeInboundMessage(
            adapter=BridgeAdapterType.LARK,
            tenant_key="tenant-1",
            event_id="event-1",
            message_id="om_1",
            chat_id="oc_1",
            content_text="continue after approval",
        ),
    )

    assert result.status == BridgeEventStatus.DEFERRED
    assert result.queued_count == 1
    deferred_rows = (await db_session.execute(select(HitlDeferredInputRecord))).scalars().all()
    assert len(deferred_rows) == 1
    assert deferred_rows[0].status == "pending"
    assert deferred_rows[0].input_parts[0]["type"] == TextPart(type="text", text="x").type
    assert "continue after approval" in deferred_rows[0].input_parts[0]["text"]
    agency_fires = (await db_session.execute(select(AgencyFireRecord))).scalars().all()
    assert len(agency_fires) == 1
    assert agency_fires[0].kind == "message_observed"
    assert agency_fires[0].source_session_id == "session-1"
    assert agency_fires[0].source_run_id == "run-1"
    assert agency_fires[0].payload["metadata"]["bridge"]["event_id"] == "event-1"

    payloads = await controller.consume_deferred_inputs(db_session, run_id="run-1", batch_id=deferred_rows[0].batch_id)
    await db_session.commit()

    assert len(payloads) == 1
    assert payloads[0].sequence_no == 1
    refreshed = await db_session.get(HitlDeferredInputRecord, deferred_rows[0].id)
    assert isinstance(refreshed, HitlDeferredInputRecord)
    assert refreshed.status == "consumed"
