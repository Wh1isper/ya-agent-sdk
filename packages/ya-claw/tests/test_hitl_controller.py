from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi import HTTPException
from pydantic_ai import DeferredToolRequests
from pydantic_ai.messages import ToolCallPart
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from ya_claw.bridge.models import BridgeAdapterType, BridgeEventStatus, BridgeInboundMessage
from ya_claw.config import ClawSettings
from ya_claw.controller.hitl import HitlController
from ya_claw.controller.models import ActiveInteraction, InteractionRespondRequest, TextPart
from ya_claw.db.engine import create_engine, create_session_factory
from ya_claw.execution.dispatcher import RunDispatcher
from ya_claw.execution.input_inbox import deliver_accepted_run_inputs
from ya_claw.hitl import build_active_interactions
from ya_claw.orm.tables import (
    HitlBatchRecord,
    HitlDeferredInputRecord,
    HitlInteractionRecord,
    ProfileRecord,
    RunInputInboxRecord,
    RunRecord,
    SessionRecord,
)
from ya_claw.runtime_state import create_runtime_state


@pytest.fixture
async def db_engine(tmp_path: Path, initialize_sqlite_database: Callable[[str], None]) -> AsyncEngine:
    database_url = f"sqlite+aiosqlite:///{(tmp_path / 'hitl.sqlite3').resolve()}"
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


def test_build_active_interactions_includes_external_calls() -> None:
    interactions = build_active_interactions(
        DeferredToolRequests(
            approvals=[
                ToolCallPart(
                    tool_name="shell_exec",
                    args={"command": "pwd"},
                    tool_call_id="approval-call",
                )
            ],
            calls=[
                ToolCallPart(
                    tool_name="ask_user_question",
                    args={"question": "Choose"},
                    tool_call_id="external-call",
                )
            ],
        ),
        run_id="run-1",
        session_id="session-1",
    )

    assert [interaction.tool_call_id for interaction in interactions] == [
        "approval-call",
        "external-call",
    ]
    assert interactions[0].kind == "tool_approval"
    assert interactions[1].kind == "external_result"
    assert interactions[1].total_count == 2


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

    first_result = await controller.respond_interaction(
        db_session,
        "run-1",
        "hitl-1",
        InteractionRespondRequest(approved=True, reason="ok"),
    )
    await db_session.commit()
    first = first_result.response

    assert first_result.resolution.approved is True
    assert first.remaining_interaction_count == 1
    assert first.current_interaction is not None
    assert first.current_interaction.interaction_id == "hitl-2"
    interaction_rows = (await db_session.execute(select(HitlInteractionRecord))).scalars().all()
    assert [row.status for row in interaction_rows] == ["approved", "pending"]

    second_result = await controller.respond_interaction(
        db_session,
        "run-1",
        "hitl-2",
        InteractionRespondRequest(approved=False, reason="deny"),
    )
    await db_session.commit()
    second = second_result.response

    assert second_result.resolution.approved is False
    assert second_result.resolution.reason == "deny"
    assert second.remaining_interaction_count == 0
    assert second.current_interaction is None

    conflicting_retry = await controller.respond_interaction(
        db_session,
        "run-1",
        "hitl-2",
        InteractionRespondRequest(approved=True, reason="conflict"),
    )
    await db_session.commit()
    assert conflicting_retry.response.status == "denied"
    assert conflicting_retry.resolution.approved is False
    assert conflicting_retry.resolution.reason == "deny"

    stored_batch = await db_session.get(HitlBatchRecord, batch.batch_id)
    assert isinstance(stored_batch, HitlBatchRecord)
    assert stored_batch.status == "pending"
    refreshed_run = await db_session.get(RunRecord, "run-1")
    assert isinstance(refreshed_run, RunRecord)
    assert refreshed_run.run_metadata["active_hitl_batch_id"] == batch.batch_id
    assert [item["status"] for item in refreshed_run.run_metadata["active_interactions"]] == [
        "approved",
        "denied",
    ]


async def test_stale_interaction_id_cannot_resolve_a_later_batch(
    db_session: AsyncSession,
) -> None:
    await _seed_running_run(db_session)
    controller = HitlController()
    first_interaction = build_active_interactions(
        DeferredToolRequests(
            approvals=[
                ToolCallPart(
                    tool_name="shell_exec",
                    args={"command": "first"},
                    tool_call_id="tool-first",
                )
            ]
        ),
        run_id="run-1",
        session_id="session-1",
    )[0]
    first_batch = await controller.create_batch(
        db_session,
        session_id="session-1",
        run_id="run-1",
        interactions=[first_interaction],
    )
    run_record = await db_session.get(RunRecord, "run-1")
    assert isinstance(run_record, RunRecord)
    run_record.run_metadata = {"active_hitl_batch_id": first_batch.batch_id}
    await db_session.commit()
    await controller.respond_interaction(
        db_session,
        "run-1",
        first_interaction.interaction_id,
        InteractionRespondRequest(approved=True, reason="first"),
    )
    await controller.mark_batch_completed(db_session, run_id="run-1")
    run_record.run_metadata = {}
    await db_session.commit()

    second_interaction = build_active_interactions(
        DeferredToolRequests(
            approvals=[
                ToolCallPart(
                    tool_name="shell_exec",
                    args={"command": "second"},
                    tool_call_id="tool-second",
                )
            ]
        ),
        run_id="run-1",
        session_id="session-1",
    )[0]
    second_batch = await controller.create_batch(
        db_session,
        session_id="session-1",
        run_id="run-1",
        interactions=[second_interaction],
    )
    run_record.run_metadata = {"active_hitl_batch_id": second_batch.batch_id}
    await db_session.commit()

    assert first_interaction.interaction_id != second_interaction.interaction_id
    with pytest.raises(HTTPException) as stale_response:
        await controller.respond_interaction(
            db_session,
            "run-1",
            first_interaction.interaction_id,
            InteractionRespondRequest(approved=True, reason="stale"),
        )
    assert stale_response.value.status_code == 404

    result = await controller.respond_interaction(
        db_session,
        "run-1",
        second_interaction.interaction_id,
        InteractionRespondRequest(approved=False, reason="second"),
    )
    await db_session.commit()

    rows = (
        (await db_session.execute(select(HitlInteractionRecord).order_by(HitlInteractionRecord.created_at.asc())))
        .scalars()
        .all()
    )
    assert first_batch.batch_id != second_batch.batch_id
    assert [row.status for row in rows] == ["approved", "denied"]
    assert result.resolution.tool_call_id == "tool-second"
    assert result.resolution.reason == "second"


async def test_deferred_input_admission_revalidates_stale_batch_after_staging(
    db_engine: AsyncEngine,
) -> None:
    controller = HitlController()
    session_factory = create_session_factory(db_engine)
    async with session_factory() as setup_session:
        await _seed_running_run(setup_session)
        payload = await controller.create_batch(
            setup_session,
            session_id="session-1",
            run_id="run-1",
            interactions=[
                ActiveInteraction(
                    interaction_id="hitl-stale",
                    run_id="run-1",
                    session_id="session-1",
                    tool_call_id="tool-stale",
                    title="Stale batch",
                )
            ],
        )
        await setup_session.commit()
        stale_batch = await setup_session.get(HitlBatchRecord, payload.batch_id)
        assert isinstance(stale_batch, HitlBatchRecord)
        setup_session.expunge(stale_batch)

    async with session_factory() as staging_session:
        assert (
            await controller.stage_deferred_inputs(
                staging_session,
                run_id="run-1",
                batch_id=payload.batch_id,
            )
            == []
        )
        await controller.mark_batch_completed(staging_session, run_id="run-1")
        await staging_session.commit()

    async with session_factory() as ingress_session:
        queued_count = await controller.enqueue_deferred_input(
            ingress_session,
            batch=stale_batch,
            message=BridgeInboundMessage(
                adapter=BridgeAdapterType.LARK,
                tenant_key="tenant-1",
                event_id="event-late",
                message_id="message-late",
                chat_id="chat-late",
                content_text="late input",
            ),
            conversation_id=None,
            input_parts=[TextPart(type="text", text="late input").model_dump(mode="json")],
        )
        assert queued_count is None
        deferred_count = len((await ingress_session.execute(select(HitlDeferredInputRecord))).scalars().all())
        assert deferred_count == 0


async def test_run_controller_commits_hitl_decision_before_runtime_signal(
    db_session: AsyncSession,
    db_engine: AsyncEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ya_claw.controller.run import RunController

    await _seed_running_run(db_session)
    interaction = ActiveInteraction(
        interaction_id="hitl-commit",
        run_id="run-1",
        session_id="session-1",
        tool_call_id="tool-commit",
        tool_name="shell_exec",
        title="Approve shell",
    )
    await HitlController().create_batch(
        db_session,
        session_id="session-1",
        run_id="run-1",
        interactions=[interaction],
    )
    await db_session.commit()

    runtime_state = create_runtime_state()
    observed_statuses: list[str] = []
    session_factory = create_session_factory(db_engine)

    async def observe_committed_decision(
        _runtime_state: object,
        run_id: str,
        interaction_id: str,
        **_resolution: object,
    ) -> None:
        async with session_factory() as verify_session:
            record = (
                await verify_session.execute(
                    select(HitlInteractionRecord).where(
                        HitlInteractionRecord.run_id == run_id,
                        HitlInteractionRecord.interaction_id == interaction_id,
                    )
                )
            ).scalar_one()
            observed_statuses.append(record.status)

    monkeypatch.setattr(type(runtime_state), "resolve_hitl_interaction", observe_committed_decision)
    response = await RunController().respond_interaction(
        db_session,
        runtime_state,
        "run-1",
        "hitl-commit",
        InteractionRespondRequest(approved=True, reason="committed"),
    )

    assert response.status == "approved"
    assert observed_statuses == ["approved"]


async def test_cancel_discards_pending_inputs_from_completed_hitl_batch(
    db_session: AsyncSession,
) -> None:
    await _seed_running_run(db_session)
    controller = HitlController()
    payload = await controller.create_batch(
        db_session,
        session_id="session-1",
        run_id="run-1",
        interactions=[
            ActiveInteraction(
                interaction_id="hitl-completed",
                run_id="run-1",
                session_id="session-1",
                tool_call_id="tool-completed",
                title="Completed approval",
            )
        ],
    )
    batch = await db_session.get(HitlBatchRecord, payload.batch_id)
    assert isinstance(batch, HitlBatchRecord)
    batch.status = "completed"
    batch.current_interaction_id = None
    batch.completed_at = datetime.now(UTC)
    interaction = (
        await db_session.execute(select(HitlInteractionRecord).where(HitlInteractionRecord.batch_id == batch.id))
    ).scalar_one()
    interaction.status = "approved"
    interaction.response = {"approved": True, "reason": "done", "user_input": None}
    interaction.resolved_at = datetime.now(UTC)
    deferred_input = HitlDeferredInputRecord(
        id="deferred-completed",
        batch_id=batch.id,
        session_id="session-1",
        run_id="run-1",
        adapter="lark",
        tenant_key="tenant-1",
        external_event_id="event-completed",
        sequence_no=1,
        input_parts=[TextPart(type="text", text="must not be stranded").model_dump(mode="json")],
        source_metadata={},
        status="pending",
    )
    db_session.add(deferred_input)
    await db_session.commit()

    closed = await controller.cancel_pending_batch(
        db_session,
        run_id="run-1",
        reason="startup recovery",
    )
    await db_session.commit()

    assert closed is True
    await db_session.refresh(batch)
    await db_session.refresh(deferred_input)
    assert batch.status == "completed"
    assert deferred_input.status == "discarded"


async def test_bridge_message_during_hitl_survives_staging_delivery_crash_window(
    db_session: AsyncSession,
    bind_recording_input_ingress: Callable[..., list[list[dict[str, object]]]],
) -> None:
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

    payloads = await controller.stage_deferred_inputs(
        db_session,
        run_id="run-1",
        batch_id=deferred_rows[0].batch_id,
    )
    await db_session.commit()

    assert len(payloads) == 1
    assert payloads[0].sequence_no == 1
    refreshed = await db_session.get(HitlDeferredInputRecord, deferred_rows[0].id)
    assert isinstance(refreshed, HitlDeferredInputRecord)
    assert refreshed.status == "consumed"
    inbox_rows = (await db_session.execute(select(RunInputInboxRecord))).scalars().all()
    assert len(inbox_rows) == 1
    assert inbox_rows[0].status == "accepted"
    assert inbox_rows[0].delivery_key == f"hitl-deferred:{deferred_rows[0].id}"
    assert inbox_rows[0].input_parts == deferred_rows[0].input_parts

    restarted_runtime = create_runtime_state()
    restarted_runtime.register_run("session-1", "run-1")
    delivered_batches = bind_recording_input_ingress(restarted_runtime, "run-1")
    delivered = await deliver_accepted_run_inputs(db_session, restarted_runtime, "run-1")

    assert len(delivered) == 1
    assert delivered_batches == [deferred_rows[0].input_parts]
    await db_session.refresh(inbox_rows[0])
    assert inbox_rows[0].status == "enqueued"
