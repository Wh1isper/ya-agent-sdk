from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from ya_claw.bridge.context_snapshot import BridgePreviousMessageSnapshotItem, BridgePreviousMessagesSnapshot
from ya_claw.bridge.controller import BridgeController
from ya_claw.bridge.models import BridgeAdapterType, BridgeEventStatus, BridgeInboundAction, BridgeInboundMessage
from ya_claw.config import ClawSettings
from ya_claw.db.engine import create_engine, create_session_factory
from ya_claw.execution.dispatcher import RunDispatcher
from ya_claw.orm.base import Base
from ya_claw.orm.tables import BridgeConversationRecord, BridgeEventRecord, RunRecord, SessionRecord
from ya_claw.runtime_state import create_runtime_state


@pytest.fixture
async def db_engine(tmp_path: Path) -> AsyncEngine:
    engine = create_engine(f"sqlite+aiosqlite:///{(tmp_path / 'bridge.sqlite3').resolve()}")
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


async def test_bridge_controller_maps_chat_to_session_and_dedupes(db_session: AsyncSession) -> None:
    runtime_state = create_runtime_state()
    controller = BridgeController()
    settings = ClawSettings(
        api_token="test-token",  # noqa: S106
        bridge_lark_default_profile="default",
        _env_file=None,
    )
    message = BridgeInboundMessage(
        adapter=BridgeAdapterType.LARK,
        tenant_key="tenant-1",
        event_id="event-1",
        message_id="om_1",
        root_id="om_root",
        parent_id="om_parent",
        thread_id="omt_1",
        chat_id="oc_1",
        sender_id="ou_1",
        content_text="hello",
    )

    result = await controller.handle_inbound_message(
        db_session,
        settings,
        runtime_state,
        RunDispatcher(None),
        message,
    )

    assert result.status == BridgeEventStatus.QUEUED
    assert result.session_id is not None
    assert result.run_id is not None
    conversation_result = await db_session.execute(
        select(BridgeConversationRecord).where(
            BridgeConversationRecord.adapter == BridgeAdapterType.LARK,
            BridgeConversationRecord.tenant_key == "tenant-1",
            BridgeConversationRecord.external_chat_id == "oc_1",
        )
    )
    conversation = conversation_result.scalar_one()
    assert conversation.session_id == result.session_id

    event_result = await db_session.execute(
        select(BridgeEventRecord).where(
            BridgeEventRecord.adapter == BridgeAdapterType.LARK,
            BridgeEventRecord.tenant_key == "tenant-1",
            BridgeEventRecord.event_id == "event-1",
        )
    )
    event_record = event_result.scalar_one()
    assert event_record.status == BridgeEventStatus.QUEUED
    assert event_record.conversation_id == conversation.id
    assert event_record.session_id == result.session_id
    assert event_record.run_id == result.run_id
    assert event_record.normalized_event["root_id"] == "om_root"
    assert event_record.normalized_event["parent_id"] == "om_parent"
    assert event_record.normalized_event["thread_id"] == "omt_1"
    assert result.run_id is not None
    run_record = await db_session.get(RunRecord, result.run_id)
    assert isinstance(run_record, RunRecord)
    assert run_record.run_metadata["bridge"]["root_id"] == "om_root"
    assert run_record.run_metadata["bridge"]["parent_id"] == "om_parent"
    assert run_record.run_metadata["bridge"]["thread_id"] == "omt_1"
    assert len(run_record.input_parts) == 1
    prompt = run_record.input_parts[0]["text"]
    assert prompt.startswith("<lark_bridge_event>")
    assert "<metadata>" in prompt
    assert "<tenant_key>tenant-1</tenant_key>" in prompt
    assert "<chat_id>oc_1</chat_id>" in prompt
    assert "<message_id>om_1</message_id>" in prompt
    assert "<root_id>om_root</root_id>" in prompt
    assert "<parent_id>om_parent</parent_id>" in prompt
    assert "<thread_id>omt_1</thread_id>" in prompt
    assert "<sender_id>ou_1</sender_id>" in prompt
    assert "<message>" in prompt
    assert "<content>hello</content>" in prompt
    assert "<output>" in prompt
    assert prompt.count("<message_id>om_1</message_id>") == 2
    assert prompt.count("<root_id>om_root</root_id>") == 2
    assert prompt.count("<parent_id>om_parent</parent_id>") == 2
    assert prompt.count("<thread_id>omt_1</thread_id>") == 2
    assert "<idempotency_key>bridge-lark-event-1</idempotency_key>" in prompt
    assert "<recommended_command>" in prompt
    assert "--reply-in-thread" in prompt
    assert "&lt;reply&gt;" in prompt

    duplicate = await controller.handle_inbound_message(
        db_session,
        settings,
        runtime_state,
        RunDispatcher(None),
        message,
    )

    assert duplicate.status == BridgeEventStatus.DUPLICATE
    assert duplicate.duplicate is True
    assert duplicate.session_id == result.session_id
    assert duplicate.run_id == result.run_id


async def test_bridge_controller_escapes_xml_prompt_values(db_session: AsyncSession) -> None:
    controller = BridgeController()
    message = BridgeInboundMessage(
        adapter=BridgeAdapterType.LARK,
        tenant_key="tenant<&1",
        event_id="event'1",
        message_id='om_"1',
        root_id="om_<root>",
        parent_id="om_&parent",
        thread_id="omt_'1",
        chat_id="oc_1",
        sender_id="ou_1",
        content_text="hello <world> & friends",
    )

    prompt = controller._build_agent_prompt(message)

    assert "<tenant_key>tenant&lt;&amp;1</tenant_key>" in prompt
    assert "<message_id>om_&quot;1</message_id>" in prompt
    assert "<root_id>om_&lt;root&gt;</root_id>" in prompt
    assert "<parent_id>om_&amp;parent</parent_id>" in prompt
    assert "<thread_id>omt_&apos;1</thread_id>" in prompt
    assert "<event_id>event&apos;1</event_id>" in prompt
    assert "<content>hello &lt;world&gt; &amp; friends</content>" in prompt
    assert "&lt;reply&gt;" in prompt


async def test_bridge_controller_includes_previous_messages_snapshot_from_message_metadata(
    db_session: AsyncSession,
) -> None:
    runtime_state = create_runtime_state()
    controller = BridgeController()
    settings = ClawSettings(api_token="test-token", _env_file=None)  # noqa: S106
    snapshot = BridgePreviousMessagesSnapshot(
        items=[
            BridgePreviousMessageSnapshotItem(
                speaker="self",
                relation="parent",
                message_id="om_parent",
                sender_id="cli_self",
                sender_type="app",
                message_type="text",
                create_time="1000",
                content_text="scheduled task asked for approval",
            )
        ]
    )

    result = await controller.handle_inbound_message(
        db_session,
        settings,
        runtime_state,
        RunDispatcher(None),
        BridgeInboundMessage(
            adapter=BridgeAdapterType.LARK,
            tenant_key="tenant-1",
            event_id="event-1",
            message_id="om_1",
            parent_id="om_parent",
            chat_id="oc_1",
            content_text="approved",
            metadata={"previous_messages_snapshot": snapshot.model_dump(mode="json")},
        ),
    )

    assert result.run_id is not None
    run_record = await db_session.get(RunRecord, result.run_id)
    event_result = await db_session.execute(select(BridgeEventRecord).where(BridgeEventRecord.event_id == "event-1"))
    event_record = event_result.scalar_one()
    assert isinstance(run_record, RunRecord)
    prompt = run_record.input_parts[0]["text"]
    assert '<previous_messages_snapshot source="lark" max_messages="1" truncated="false">' in prompt
    assert 'speaker="self"' in prompt
    assert 'relation="parent"' in prompt
    assert "scheduled task asked for approval" in prompt
    assert "Messages marked speaker=&quot;self&quot; were sent by this Lark bridge bot/app" in prompt
    assert run_record.run_metadata["bridge"]["previous_messages_snapshot"]["items"][0]["message_id"] == "om_parent"
    assert event_record.normalized_event["previous_messages_snapshot"]["items"][0]["speaker"] == "self"


async def test_bridge_controller_steer_skips_previous_messages_snapshot(db_session: AsyncSession) -> None:
    runtime_state = create_runtime_state()
    controller = BridgeController()
    settings = ClawSettings(api_token="test-token", _env_file=None)  # noqa: S106
    first = await controller.handle_inbound_message(
        db_session,
        settings,
        runtime_state,
        RunDispatcher(None),
        BridgeInboundMessage(
            adapter=BridgeAdapterType.LARK,
            tenant_key="tenant-1",
            event_id="event-1",
            message_id="om_1",
            chat_id="oc_1",
            content_text="first",
        ),
    )
    assert first.run_id is not None
    session = await db_session.get(SessionRecord, first.session_id)
    run = await db_session.get(RunRecord, first.run_id)
    assert isinstance(session, SessionRecord)
    assert isinstance(run, RunRecord)
    session.active_run_id = first.run_id
    run.status = "running"
    await db_session.commit()
    snapshot = BridgePreviousMessagesSnapshot(
        items=[
            BridgePreviousMessageSnapshotItem(
                speaker="self",
                relation="parent",
                message_id="om_parent",
                content_text="context that steer should skip",
            )
        ]
    )

    result = await controller.handle_inbound_message(
        db_session,
        settings,
        runtime_state,
        RunDispatcher(None),
        BridgeInboundMessage(
            adapter=BridgeAdapterType.LARK,
            tenant_key="tenant-1",
            event_id="event-2",
            message_id="om_2",
            chat_id="oc_1",
            content_text="second",
            metadata={"previous_messages_snapshot": snapshot.model_dump(mode="json")},
        ),
    )

    handle = runtime_state.get_run_handle(first.run_id)
    assert result.status == BridgeEventStatus.STEERED
    assert handle is not None
    steered_prompt = handle.steering_inputs[0][0]["text"]
    assert "context that steer should skip" not in steered_prompt
    assert "<previous_messages_snapshot" not in steered_prompt


async def test_bridge_controller_reuses_chat_session(db_session: AsyncSession) -> None:
    runtime_state = create_runtime_state()
    controller = BridgeController()
    settings = ClawSettings(api_token="test-token", _env_file=None)  # noqa: S106

    first = await controller.handle_inbound_message(
        db_session,
        settings,
        runtime_state,
        RunDispatcher(None),
        BridgeInboundMessage(
            adapter=BridgeAdapterType.LARK,
            tenant_key="tenant-1",
            event_id="event-1",
            message_id="om_1",
            chat_id="oc_1",
            content_text="first",
        ),
    )
    second = await controller.handle_inbound_message(
        db_session,
        settings,
        runtime_state,
        RunDispatcher(None),
        BridgeInboundMessage(
            adapter=BridgeAdapterType.LARK,
            tenant_key="tenant-1",
            event_id="event-2",
            message_id="om_2",
            chat_id="oc_1",
            content_text="second",
        ),
    )

    assert first.session_id == second.session_id
    assert first.run_id != second.run_id
    assert second.run_id is not None
    run_record = await db_session.get(RunRecord, second.run_id)
    assert isinstance(run_record, RunRecord)
    assert run_record.trigger_type == "bridge"


async def test_bridge_controller_steers_active_conversation_session(db_session: AsyncSession) -> None:
    runtime_state = create_runtime_state()
    controller = BridgeController()
    settings = ClawSettings(api_token="test-token", _env_file=None)  # noqa: S106

    first = await controller.handle_inbound_message(
        db_session,
        settings,
        runtime_state,
        RunDispatcher(None),
        BridgeInboundMessage(
            adapter=BridgeAdapterType.LARK,
            tenant_key="tenant-1",
            event_id="event-1",
            message_id="om_1",
            chat_id="oc_1",
            content_text="first",
        ),
    )
    assert first.run_id is not None
    session = await db_session.get(SessionRecord, first.session_id)
    assert isinstance(session, SessionRecord)
    session.active_run_id = first.run_id
    run = await db_session.get(RunRecord, first.run_id)
    assert isinstance(run, RunRecord)
    run.status = "running"
    await db_session.commit()

    second = await controller.handle_inbound_message(
        db_session,
        settings,
        runtime_state,
        RunDispatcher(None),
        BridgeInboundMessage(
            adapter=BridgeAdapterType.LARK,
            tenant_key="tenant-1",
            event_id="event-2",
            message_id="om_2",
            chat_id="oc_1",
            content_text="second",
        ),
    )

    handle = runtime_state.get_run_handle(first.run_id)
    event_record_result = await db_session.execute(
        select(BridgeEventRecord).where(BridgeEventRecord.event_id == "event-2")
    )
    event_record = event_record_result.scalar_one()
    run_count = len((await db_session.execute(select(RunRecord))).scalars().all())

    assert second.status == BridgeEventStatus.STEERED
    assert second.session_id == first.session_id
    assert second.run_id == first.run_id
    assert isinstance(handle, object)
    assert handle is not None
    assert len(handle.steering_inputs) == 1
    steered_prompt = handle.steering_inputs[0][0]["text"]
    assert "<content>second</content>" in steered_prompt
    assert event_record.status == BridgeEventStatus.STEERED
    assert event_record.run_id == first.run_id
    assert run_count == 1


async def test_bridge_controller_retries_failed_bridge_run(db_session: AsyncSession) -> None:
    runtime_state = create_runtime_state()
    controller = BridgeController()
    settings = ClawSettings(api_token="test-token", _env_file=None)  # noqa: S106
    session = SessionRecord(id="session-1", profile_name="default", head_success_run_id="success-run")
    success_run = RunRecord(
        id="success-run",
        session_id="session-1",
        sequence_no=1,
        status="completed",
        trigger_type="bridge",
        input_parts=[{"type": "text", "text": "success"}],
    )
    failed_run = RunRecord(
        id="failed-run",
        session_id="session-1",
        sequence_no=2,
        status="failed",
        trigger_type="bridge",
        input_parts=[{"type": "text", "text": "retry me"}],
        run_metadata={"bridge": {"chat_id": "oc_1", "message_id": "om_1"}},
    )
    db_session.add_all([session, success_run, failed_run])
    await db_session.commit()

    result = await controller.handle_inbound_action(
        db_session,
        settings,
        runtime_state,
        RunDispatcher(None),
        BridgeInboundAction(
            adapter=BridgeAdapterType.LARK,
            tenant_key="tenant-1",
            event_id="action-1",
            action_type="session_recovery",
            token="recovery:session-1:failed-run",  # noqa: S106
            metadata={"action": "retry"},
        ),
    )

    assert result.status == BridgeEventStatus.QUEUED
    assert result.run_id is not None
    retry_run = await db_session.get(RunRecord, result.run_id)
    refreshed_session = await db_session.get(SessionRecord, "session-1")
    assert isinstance(retry_run, RunRecord)
    assert isinstance(refreshed_session, SessionRecord)
    assert retry_run.input_parts[0]["type"] == failed_run.input_parts[0]["type"]
    assert retry_run.input_parts[0]["text"] == failed_run.input_parts[0]["text"]
    assert retry_run.restore_from_run_id == "success-run"
    assert retry_run.run_metadata["bridge"]["chat_id"] == "oc_1"
    assert retry_run.run_metadata["recovery"]["mode"] == "retry"
    assert retry_run.trigger_type == "bridge"
    assert refreshed_session.head_success_run_id == "success-run"


async def test_bridge_controller_reset_and_retries_failed_bridge_run(db_session: AsyncSession) -> None:
    runtime_state = create_runtime_state()
    controller = BridgeController()
    settings = ClawSettings(api_token="test-token", _env_file=None)  # noqa: S106
    session = SessionRecord(id="session-1", profile_name="default", head_success_run_id="success-run")
    success_run = RunRecord(
        id="success-run",
        session_id="session-1",
        sequence_no=1,
        status="completed",
        trigger_type="bridge",
        input_parts=[{"type": "text", "text": "success"}],
    )
    failed_run = RunRecord(
        id="failed-run",
        session_id="session-1",
        sequence_no=2,
        status="failed",
        trigger_type="bridge",
        input_parts=[{"type": "text", "text": "retry clean"}],
    )
    db_session.add_all([session, success_run, failed_run])
    await db_session.commit()

    result = await controller.handle_inbound_action(
        db_session,
        settings,
        runtime_state,
        RunDispatcher(None),
        BridgeInboundAction(
            adapter=BridgeAdapterType.LARK,
            tenant_key="tenant-1",
            event_id="action-1",
            action_type="session_recovery",
            token="recovery:session-1:failed-run",  # noqa: S106
            metadata={"action": "reset_and_retry"},
        ),
    )

    assert result.status == BridgeEventStatus.QUEUED
    assert result.run_id is not None
    retry_run = await db_session.get(RunRecord, result.run_id)
    refreshed_session = await db_session.get(SessionRecord, "session-1")
    assert isinstance(retry_run, RunRecord)
    assert isinstance(refreshed_session, SessionRecord)
    assert retry_run.input_parts[0]["type"] == failed_run.input_parts[0]["type"]
    assert retry_run.input_parts[0]["text"] == failed_run.input_parts[0]["text"]
    assert retry_run.restore_from_run_id is None
    assert retry_run.run_metadata["restore_state"] is False
    assert retry_run.run_metadata["recovery"]["mode"] == "reset_and_retry"
    assert refreshed_session.head_success_run_id == "success-run"
