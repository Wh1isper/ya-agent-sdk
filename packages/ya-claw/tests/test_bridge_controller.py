from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from ya_claw.bridge.context_snapshot import BridgePreviousMessageSnapshotItem, BridgePreviousMessagesSnapshot
from ya_claw.bridge.controller import BridgeController, _bridge_delivery_key
from ya_claw.bridge.models import BridgeAdapterType, BridgeEventStatus, BridgeInboundAction, BridgeInboundMessage
from ya_claw.config import ClawSettings
from ya_claw.db.engine import create_engine, create_session_factory
from ya_claw.execution.dispatcher import RunDispatcher
from ya_claw.orm.tables import (
    BridgeConversationRecord,
    BridgeEventRecord,
    RunInputInboxRecord,
    RunRecord,
    SessionRecord,
)
from ya_claw.runtime_state import create_runtime_state


@pytest.fixture
async def db_engine(tmp_path: Path, initialize_sqlite_database: Callable[..., None]) -> AsyncEngine:
    database_url = f"sqlite+aiosqlite:///{(tmp_path / 'bridge.sqlite3').resolve()}"
    initialize_sqlite_database(database_url, profile_names=("default", "general"))
    engine = create_engine(database_url)
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


async def test_bridge_controller_steer_skips_previous_messages_snapshot(
    db_session: AsyncSession,
    bind_recording_input_ingress: Callable[..., list[list[dict[str, object]]]],
) -> None:
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
    batches = bind_recording_input_ingress(runtime_state, first.run_id)
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

    assert result.status == BridgeEventStatus.STEERED
    steered_prompt = batches[0][0]["text"]
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
    assert second.run_id == first.run_id
    assert second.run_id is not None
    run_record = await db_session.get(RunRecord, second.run_id)
    assert isinstance(run_record, RunRecord)
    assert run_record.trigger_type == "bridge"
    assert len(run_record.input_parts) == 2
    assert "<content>first</content>" in run_record.input_parts[0]["text"]
    assert "<content>second</content>" in run_record.input_parts[1]["text"]


async def test_bridge_controller_steers_active_conversation_session(
    db_session: AsyncSession,
    bind_recording_input_ingress: Callable[..., list[list[dict[str, object]]]],
) -> None:
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
    batches = bind_recording_input_ingress(runtime_state, first.run_id)

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

    event_record_result = await db_session.execute(
        select(BridgeEventRecord).where(BridgeEventRecord.event_id == "event-2")
    )
    event_record = event_record_result.scalar_one()
    run_count = len((await db_session.execute(select(RunRecord))).scalars().all())

    assert second.status == BridgeEventStatus.STEERED
    assert second.session_id == first.session_id
    assert second.run_id == first.run_id
    assert len(batches) == 1
    steered_prompt = batches[0][0]["text"]
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


async def test_bridge_controller_uses_github_profile_prompt_and_shared_default_workspace(
    db_session: AsyncSession,
) -> None:
    runtime_state = create_runtime_state()
    controller = BridgeController()
    settings = ClawSettings(
        api_token="test-token",  # noqa: S106
        bridge_github_default_profile="general",
        _env_file=None,
    )
    metadata = {
        "github": {
            "repository": "acme/widgets",
            "resource_kind": "issue",
            "resource_number": 7,
            "resource_url": "https://github.com/acme/widgets/issues/7",
            "reason": "mention",
            "subject_title": "Fix widget",
        }
    }

    first = await controller.handle_inbound_message(
        db_session,
        settings,
        runtime_state,
        RunDispatcher(None),
        BridgeInboundMessage(
            adapter=BridgeAdapterType.GITHUB,
            tenant_key="github:api.github.com:100",
            event_id="github:101:2026-08-26T08:00:00Z",
            message_id="github:101:2026-08-26T08:00:00Z",
            thread_id="101",
            chat_id="github:42:issue:7",
            event_type="github.notification",
            sender_id="alice",
            chat_type="issue",
            message_type="notification",
            content_text="GitHub notification update.",
            create_time="2026-08-26T08:00:00Z",
            metadata=metadata,
        ),
    )
    second = await controller.handle_inbound_message(
        db_session,
        settings,
        runtime_state,
        RunDispatcher(None),
        BridgeInboundMessage(
            adapter=BridgeAdapterType.GITHUB,
            tenant_key="github:api.github.com:100",
            event_id="github:101:2026-08-26T08:01:00Z",
            message_id="github:101:2026-08-26T08:01:00Z",
            thread_id="101",
            chat_id="github:42:issue:7",
            event_type="github.notification",
            sender_id="alice",
            chat_type="issue",
            message_type="notification",
            content_text="GitHub notification update again.",
            create_time="2026-08-26T08:01:00Z",
            metadata=metadata,
        ),
    )

    assert first.session_id == second.session_id
    session = await db_session.get(SessionRecord, first.session_id)
    assert isinstance(session, SessionRecord)
    assert session.profile_name == "general"
    assert "workspace" not in session.session_metadata
    assert session.session_metadata["bridge"]["adapter_metadata"] == metadata
    assert first.run_id is not None
    run = await db_session.get(RunRecord, first.run_id)
    assert isinstance(run, RunRecord)
    prompt = run.input_parts[0]["text"]
    assert prompt.startswith("<github_bridge_event>")
    assert "<repository>acme/widgets</repository>" in prompt
    assert "<resource_kind>issue</resource_kind>" in prompt
    assert "gh issue view 7 --repo acme/widgets --comments" in prompt
    assert "gh issue comment 7 --repo acme/widgets --body &apos;&lt;reply&gt;&apos;" in prompt
    assert run.run_metadata["bridge"]["adapter_metadata"] == metadata


async def test_bridge_controller_recovers_received_github_event(db_session: AsyncSession) -> None:
    runtime_state = create_runtime_state()
    controller = BridgeController()
    settings = ClawSettings(
        api_token="test-token",  # noqa: S106
        bridge_github_default_profile="general",
        _env_file=None,
    )
    message = BridgeInboundMessage(
        adapter=BridgeAdapterType.GITHUB,
        tenant_key="github:api.github.com:100",
        event_id="github:recover:2026-08-26T08:00:00Z",
        message_id="github:recover:2026-08-26T08:00:00Z",
        thread_id="recover",
        chat_id="github:42:issue:7",
        event_type="github.notification",
        sender_id="alice",
        chat_type="issue",
        message_type="notification",
        content_text="Recover this GitHub notification.",
        create_time="2026-08-26T08:00:00Z",
        metadata={
            "github": {
                "repository": "acme/widgets",
                "resource_kind": "issue",
                "resource_number": 7,
                "resource_url": "https://github.com/acme/widgets/issues/7",
                "reason": "mention",
                "subject_title": "Fix widget",
            }
        },
    )
    db_session.add(
        BridgeEventRecord(
            id="received-github-event",
            adapter=message.adapter,
            tenant_key=message.tenant_key,
            event_id=message.event_id,
            external_message_id=message.message_id,
            external_chat_id=message.chat_id,
            event_type=message.event_type,
            status=BridgeEventStatus.RECEIVED,
            raw_event={},
            normalized_event=message.model_dump(mode="json"),
        )
    )
    await db_session.commit()

    recovered = await controller.handle_inbound_message(
        db_session,
        settings,
        runtime_state,
        RunDispatcher(None),
        message,
    )
    duplicate = await controller.handle_inbound_message(
        db_session,
        settings,
        runtime_state,
        RunDispatcher(None),
        message,
    )

    assert recovered.status == BridgeEventStatus.QUEUED
    assert recovered.run_id is not None
    assert duplicate.status == BridgeEventStatus.DUPLICATE
    event_record = await db_session.get(BridgeEventRecord, "received-github-event")
    assert isinstance(event_record, BridgeEventRecord)
    assert event_record.status == BridgeEventStatus.QUEUED
    assert event_record.run_id == recovered.run_id


async def test_bridge_controller_retries_running_github_steer_after_metadata_only_failure(
    db_session: AsyncSession,
) -> None:
    runtime_state = create_runtime_state()
    controller = BridgeController()
    settings = ClawSettings(api_token="test-token", _env_file=None)  # noqa: S106
    metadata = {
        "github": {
            "repository": "acme/widgets",
            "resource_kind": "issue",
            "resource_number": 7,
            "resource_url": "https://github.com/acme/widgets/issues/7",
            "reason": "mention",
            "subject_title": "Fix widget",
        }
    }

    def github_message(event_id: str, text: str) -> BridgeInboundMessage:
        return BridgeInboundMessage(
            adapter=BridgeAdapterType.GITHUB,
            tenant_key="github:api.github.com:100",
            event_id=event_id,
            message_id=event_id,
            thread_id="recover-steer",
            chat_id="github:42:issue:7",
            event_type="github.notification",
            sender_id="alice",
            chat_type="issue",
            message_type="notification",
            content_text=text,
            create_time="2026-08-26T08:00:00Z",
            metadata=metadata,
        )

    first = github_message("github:recover-steer:2026-08-26T08:00:00Z", "First update")
    second = github_message("github:recover-steer:2026-08-26T08:01:00Z", "Second update")
    first_result = await controller.handle_inbound_message(
        db_session,
        settings,
        runtime_state,
        RunDispatcher(None),
        first,
    )
    assert first_result.run_id is not None
    run = await db_session.get(RunRecord, first_result.run_id)
    assert isinstance(run, RunRecord)
    run.status = "running"
    run.run_metadata = {**run.run_metadata, "bridge": {**run.run_metadata["bridge"], "event_id": second.event_id}}
    db_session.add(
        BridgeEventRecord(
            id="failed-github-steer",
            adapter=second.adapter,
            tenant_key=second.tenant_key,
            event_id=second.event_id,
            external_message_id=second.message_id,
            external_chat_id=second.chat_id,
            event_type=second.event_type,
            status=BridgeEventStatus.FAILED,
            raw_event={},
            normalized_event=second.model_dump(mode="json"),
        )
    )
    await db_session.commit()

    retried = await controller.handle_inbound_message(
        db_session,
        settings,
        runtime_state,
        RunDispatcher(None),
        second,
    )

    assert retried.status == BridgeEventStatus.STEERED
    inbox_rows = (await db_session.execute(select(RunInputInboxRecord))).scalars().all()
    assert len(inbox_rows) == 1
    assert inbox_rows[0].run_id == run.id
    assert inbox_rows[0].input_parts[0]["text"].startswith("<github_bridge_event>")

    third = github_message("github:recover-steer:2026-08-26T08:02:00Z", "Third update")
    run.status = "completed"
    db_session.add(
        BridgeEventRecord(
            id="failed-github-inbox-only",
            adapter=third.adapter,
            tenant_key=third.tenant_key,
            event_id=third.event_id,
            external_message_id=third.message_id,
            external_chat_id=third.chat_id,
            event_type=third.event_type,
            status=BridgeEventStatus.FAILED,
            raw_event={},
            normalized_event=third.model_dump(mode="json"),
        )
    )
    db_session.add(
        RunInputInboxRecord(
            id="github-inbox-only",
            run_id=run.id,
            delivery_key=_bridge_delivery_key(third),
            input_parts=[{"type": "text", "text": "Third update"}],
        )
    )
    await db_session.commit()

    inbox_only_recovery = await controller.handle_inbound_message(
        db_session,
        settings,
        runtime_state,
        RunDispatcher(None),
        third,
    )

    assert inbox_only_recovery.status == BridgeEventStatus.DUPLICATE
    recovered_event = await db_session.get(BridgeEventRecord, "failed-github-inbox-only")
    assert isinstance(recovered_event, BridgeEventRecord)
    assert recovered_event.status == BridgeEventStatus.STEERED
    assert recovered_event.run_id == run.id
    run_count = await db_session.scalar(select(func.count()).select_from(RunRecord))
    assert run_count == 1
