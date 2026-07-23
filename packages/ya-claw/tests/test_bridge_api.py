from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from ya_claw.app import create_app
from ya_claw.config import get_settings
from ya_claw.orm.tables import BridgeConversationRecord, BridgeEventRecord, RunRecord, SessionRecord


@pytest.fixture(autouse=True)
def clear_claw_settings(
    monkeypatch,
    tmp_path: Path,
    initialize_sqlite_database: Callable[[str], None],
) -> None:
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
        "YA_CLAW_BRIDGE_DISPATCH_MODE",
    ):
        monkeypatch.delenv(env_name, raising=False)

    monkeypatch.setenv("YA_CLAW_API_TOKEN", "test-token")
    monkeypatch.setenv("YA_CLAW_DATA_DIR", str(tmp_path / "runtime-data"))
    monkeypatch.setenv("YA_CLAW_WORKSPACE_DIR", str(tmp_path / "workspace"))
    monkeypatch.setenv("YA_CLAW_WORKSPACE_PROVIDER_BACKEND", "local")
    monkeypatch.setenv("YA_CLAW_AUTO_SEED_PROFILES", "false")
    monkeypatch.setenv("YA_CLAW_SCHEDULE_DISPATCH_ENABLED", "false")
    monkeypatch.setenv("YA_CLAW_HEARTBEAT_ENABLED", "false")
    monkeypatch.setenv("YA_CLAW_BRIDGE_DISPATCH_MODE", "manual")

    get_settings.cache_clear()
    settings = get_settings()
    initialize_sqlite_database(settings.resolved_database_url)
    yield
    get_settings.cache_clear()


def _auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer test-token"}


def test_bridge_lists_conversations_and_events() -> None:
    app = create_app()
    with TestClient(app) as client:
        session_record = SessionRecord(id="session-1", profile_name="default", active_run_id="run-1")
        run_record = RunRecord(
            id="run-1",
            session_id="session-1",
            sequence_no=1,
            status="running",
            trigger_type="bridge",
            input_parts=[],
        )
        conversation = BridgeConversationRecord(
            id="conversation-1",
            adapter="lark",
            tenant_key="tenant-1",
            external_chat_id="oc_1",
            session_id="session-1",
            profile_name="default",
            conversation_metadata={"chat_type": "group"},
            last_event_at=datetime.now(UTC),
        )
        event = BridgeEventRecord(
            id="event-record-1",
            adapter="lark",
            tenant_key="tenant-1",
            event_id="event-1",
            external_message_id="om_1",
            external_chat_id="oc_1",
            conversation_id="conversation-1",
            session_id="session-1",
            run_id="run-1",
            event_type="im.message.receive_v1",
            status="steered",
            raw_event={"schema": "2.0"},
            normalized_event={"content_text": "hello"},
        )

        async def seed_records() -> None:
            session_factory = app.state.db_session_factory
            async with session_factory() as db_session:
                db_session.add_all([session_record, run_record, conversation, event])
                await db_session.commit()

        client.portal.call(seed_records)

        conversations_response = client.get("/api/v1/bridges/conversations", headers=_auth_headers())
        events_response = client.get("/api/v1/bridges/events", headers=_auth_headers())

    assert conversations_response.status_code == 200
    conversations_payload = conversations_response.json()
    assert conversations_payload["conversations"][0]["id"] == "conversation-1"
    assert conversations_payload["conversations"][0]["event_count"] == 1
    assert conversations_payload["conversations"][0]["latest_event_status"] == "steered"
    assert conversations_payload["conversations"][0]["active_run_id"] == "run-1"
    assert conversations_payload["conversations"][0]["metadata"] == {"chat_type": "group"}

    assert events_response.status_code == 200
    events_payload = events_response.json()
    assert events_payload["events"][0]["id"] == "event-record-1"
    assert events_payload["events"][0]["conversation_id"] == "conversation-1"
    assert events_payload["events"][0]["run_status"] == "running"
    assert events_payload["events"][0]["status"] == "steered"
    assert events_payload["events"][0]["raw_event"] == {"schema": "2.0"}
    assert events_payload["events"][0]["normalized_event"] == {"content_text": "hello"}


def test_bridge_events_filter_by_status() -> None:
    app = create_app()
    with TestClient(app) as client:
        session_record = SessionRecord(id="session-1", profile_name="default")
        conversation = BridgeConversationRecord(
            id="conversation-1",
            adapter="lark",
            tenant_key="tenant-1",
            external_chat_id="oc_1",
            session_id="session-1",
            profile_name="default",
        )
        steered_event = BridgeEventRecord(
            id="event-record-1",
            adapter="lark",
            tenant_key="tenant-1",
            event_id="event-1",
            external_message_id="om_1",
            external_chat_id="oc_1",
            conversation_id="conversation-1",
            event_type="im.message.receive_v1",
            status="steered",
        )
        failed_event = BridgeEventRecord(
            id="event-record-2",
            adapter="lark",
            tenant_key="tenant-1",
            event_id="event-2",
            external_message_id="om_2",
            external_chat_id="oc_1",
            conversation_id="conversation-1",
            event_type="im.message.receive_v1",
            status="failed",
            error_message="boom",
        )

        async def seed_records() -> None:
            session_factory = app.state.db_session_factory
            async with session_factory() as db_session:
                db_session.add_all([session_record, conversation, steered_event, failed_event])
                await db_session.commit()

        client.portal.call(seed_records)

        response = client.get("/api/v1/bridges/events?status=failed", headers=_auth_headers())

    assert response.status_code == 200
    payload = response.json()
    assert [event["id"] for event in payload["events"]] == ["event-record-2"]
    assert payload["events"][0]["error_message"] == "boom"
