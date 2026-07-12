from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from inline_snapshot import snapshot
from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from ya_claw.app import create_app
from ya_claw.config import get_settings
from ya_claw.db.engine import create_engine, create_session_factory
from ya_claw.orm.base import Base
from ya_claw.orm.tables import RunRecord, SessionMemoryStateRecord, SessionRecord


@pytest.fixture(autouse=True)
def clear_claw_settings(monkeypatch, tmp_path: Path) -> None:
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
    yield
    get_settings.cache_clear()


def _auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer test-token"}


def _create_schema() -> None:
    async def _run() -> None:
        settings = get_settings()
        engine = create_engine(settings.resolved_database_url)
        try:
            async with engine.begin() as connection:
                await connection.run_sync(Base.metadata.create_all)
        finally:
            await engine.dispose()

    asyncio.run(_run())


def _mark_run_completed(session_id: str, run_id: str, *, output_text: str | None = None) -> None:
    async def _run() -> None:
        settings = get_settings()
        for _ in range(5):
            engine = create_engine(settings.resolved_database_url)
            session_factory = create_session_factory(engine)
            now = datetime.now(UTC)
            try:
                async with session_factory() as db_session:
                    session_record = await db_session.get(SessionRecord, session_id)
                    run_record = await db_session.get(RunRecord, run_id)
                    assert isinstance(session_record, SessionRecord)
                    assert isinstance(run_record, RunRecord)
                    run_record.status = "completed"
                    run_record.output_text = output_text
                    run_record.started_at = now - timedelta(seconds=2)
                    run_record.finished_at = now - timedelta(seconds=1)
                    run_record.committed_at = now
                    session_record.active_run_id = None
                    session_record.head_success_run_id = run_id
                    await db_session.commit()
                    return
            except OperationalError:
                await asyncio.sleep(0.1)
            finally:
                await engine.dispose()
        raise AssertionError("failed to mark run completed due to persistent database lock")

    asyncio.run(_run())


def test_session_and_run_endpoints_support_rerun_controls_and_events() -> None:
    _create_schema()

    app = create_app()
    with TestClient(app) as client:
        app.state.execution_supervisor = None
        create_session_response = client.post(
            "/api/v1/sessions",
            headers=_auth_headers(),
            json={
                "profile_name": "general",
                "metadata": {"source": "api"},
                "input_parts": [{"type": "text", "text": "hello from api"}],
            },
        )
        assert create_session_response.status_code == 201
        session_payload = create_session_response.json()["session"]
        first_run_payload = create_session_response.json()["run"]
        assert isinstance(first_run_payload, dict)
        assert session_payload["status"] == "queued"

        session_steer_response = client.post(
            f"/api/v1/sessions/{session_payload['id']}/steer",
            headers=_auth_headers(),
            json={"input_parts": [{"type": "text", "text": "focus on tests"}]},
        )
        assert session_steer_response.status_code == 409

        steer_response = client.post(
            f"/api/v1/runs/{first_run_payload['id']}/steer",
            headers=_auth_headers(),
            json={"input_parts": [{"type": "text", "text": "focus on tests"}]},
        )
        assert steer_response.status_code == 200

        interrupt_response = client.post(
            f"/api/v1/runs/{first_run_payload['id']}/interrupt",
            headers=_auth_headers(),
        )
        assert interrupt_response.status_code == 200
        assert interrupt_response.json()["termination_reason"] == "interrupt"

        rerun_response = client.post(
            f"/api/v1/sessions/{session_payload['id']}/runs",
            headers=_auth_headers(),
            json={
                "restore_from_run_id": first_run_payload["id"],
                "input_parts": [{"type": "text", "text": "retry after interrupt"}],
            },
        )
        assert rerun_response.status_code == 201
        rerun_payload = rerun_response.json()

        list_sessions_response = client.get("/api/v1/sessions", headers=_auth_headers())
        assert list_sessions_response.status_code == 200
        sessions_payload = list_sessions_response.json()
        list_sessions_page_response = client.get(
            "/api/v1/sessions/page?limit=1&include_latest_output=false",
            headers=_auth_headers(),
        )
        assert list_sessions_page_response.status_code == 200
        sessions_page_payload = list_sessions_page_response.json()
        invalid_sessions_page_response = client.get(
            "/api/v1/sessions/page?before_id=incomplete-cursor",
            headers=_auth_headers(),
        )
        assert invalid_sessions_page_response.status_code == 422

        session_detail_response = client.get(
            f"/api/v1/sessions/{session_payload['id']}?include_message=true",
            headers=_auth_headers(),
        )
        assert session_detail_response.status_code == 200
        detail_payload = session_detail_response.json()

        run_detail_response = client.get(
            f"/api/v1/runs/{rerun_payload['id']}?include_message=true",
            headers=_auth_headers(),
        )
        assert run_detail_response.status_code == 200

        run_events_response = client.get(f"/api/v1/runs/{first_run_payload['id']}/events", headers=_auth_headers())
        assert run_events_response.status_code == 200

    assert len(sessions_payload) == 1
    assert sessions_payload[0]["id"] == session_payload["id"]
    assert sessions_payload[0]["run_count"] == 2
    assert sessions_payload[0]["head_run_id"] == rerun_payload["id"]
    assert sessions_payload[0]["latest_run"]["id"] == rerun_payload["id"]
    assert sessions_page_payload["total"] == 1
    assert sessions_page_payload["limit"] == 1
    assert sessions_page_payload["has_more"] is False
    assert sessions_page_payload["sessions"][0]["latest_run"]["id"] == rerun_payload["id"]
    assert sessions_page_payload["sessions"][0]["latest_run"]["output_text"] is None
    assert detail_payload["session"]["runs"][0]["id"] == rerun_payload["id"]
    assert detail_payload["session"]["runs_limit"] == 20
    assert detail_payload["session"]["runs_has_more"] is False
    assert detail_payload["session"]["runs"][0]["message"] is None
    assert run_detail_response.json()["session"]["id"] == session_payload["id"]
    assert run_detail_response.json()["run"]["input_preview"] == "retry after interrupt"
    assert run_detail_response.json()["state"] is None
    assert run_detail_response.json()["message"] is None or isinstance(run_detail_response.json()["message"], list)
    assert "ya_claw.run_queued" in run_events_response.text
    assert "ya_claw.run_interrupted" in run_events_response.text


def test_submit_uses_session_events_for_streaming_and_run_create_rejects_running_session() -> None:
    _create_schema()

    app = create_app()
    settings = get_settings()
    with TestClient(app) as client:
        app.state.execution_supervisor = None
        create_session_response = client.post(
            "/api/v1/sessions",
            headers=_auth_headers(),
            json={"input_parts": [{"type": "text", "text": "first"}]},
        )
        assert create_session_response.status_code == 201
        session_id = create_session_response.json()["session"]["id"]
        run_id = create_session_response.json()["run"]["id"]

        submit_response = client.post(
            f"/api/v1/sessions/{session_id}/submit",
            headers=_auth_headers(),
            json={"input_parts": [{"type": "text", "text": "second"}]},
        )
        assert submit_response.status_code == 202
        assert submit_response.json()["delivery"] == "merged"
        assert app.state.runtime_state.get_session_run_handle(session_id) is not None
        assert app.state.runtime_state.get_run_handle(run_id) is not None

        async def _mark_running() -> None:
            session_factory = app.state.db_session_factory
            async with session_factory() as db_session:
                run_record = await db_session.get(RunRecord, run_id)
                assert isinstance(run_record, RunRecord)
                session_record = await db_session.get(SessionRecord, session_id)
                assert isinstance(session_record, SessionRecord)
                run_record.status = "running"
                session_record.active_run_id = run_id
                await db_session.commit()

        asyncio.run(_mark_running())

        create_run_response = client.post(
            f"/api/v1/sessions/{session_id}/runs",
            headers=_auth_headers(),
            json={"input_parts": [{"type": "text", "text": "run should reject"}]},
        )
        create_run_stream_response = client.post(
            f"/api/v1/sessions/{session_id}/runs:stream",
            headers=_auth_headers(),
            json={"input_parts": [{"type": "text", "text": "run stream should reject"}]},
        )
        assert create_run_response.status_code == 409
        assert create_run_stream_response.status_code == 409

    async def _assert_no_side_effects() -> None:
        engine = create_engine(settings.resolved_database_url)
        session_factory = create_session_factory(engine)
        try:
            async with session_factory() as db_session:
                runs = (
                    (await db_session.execute(select(RunRecord).where(RunRecord.session_id == session_id)))
                    .scalars()
                    .all()
                )
                assert len(runs) == 1
                assert runs[0].id == run_id
                assert runs[0].input_parts == [
                    {"type": "text", "text": "first", "metadata": None},
                    {"type": "text", "text": "second", "metadata": None},
                ]
                assert app.state.runtime_state.consume_steering_inputs(run_id) == []
        finally:
            await engine.dispose()

    asyncio.run(_assert_no_side_effects())


def test_session_create_uses_single_workspace_response_shape() -> None:
    _create_schema()

    with TestClient(create_app()) as client:
        create_session_response = client.post(
            "/api/v1/sessions",
            headers=_auth_headers(),
            json={"input_parts": [{"type": "text", "text": "hello from workspace session"}]},
        )

    assert create_session_response.status_code == 201
    payload = create_session_response.json()
    assert payload["session"]["metadata"] == {}
    assert payload["session"]["memory_state"] is None
    assert sorted(payload["session"]) == snapshot([
        "active_run_id",
        "created_at",
        "head_run_id",
        "head_success_run_id",
        "id",
        "latest_run",
        "memory_state",
        "metadata",
        "parent_session_id",
        "profile_name",
        "run_count",
        "session_type",
        "source_session_id",
        "status",
        "status_detail",
        "status_reason",
        "updated_at",
        "workspace_state",
    ])
    assert sorted(payload["run"]) == snapshot([
        "committed_at",
        "created_at",
        "error_message",
        "finished_at",
        "has_message",
        "has_state",
        "id",
        "input_parts",
        "input_preview",
        "message",
        "metadata",
        "output_text",
        "profile_name",
        "restore_from_run_id",
        "sequence_no",
        "session_id",
        "started_at",
        "status",
        "termination_reason",
        "trigger_type",
    ])


def test_session_detail_can_include_message_and_paginate_runs() -> None:
    _create_schema()

    with TestClient(create_app()) as client:
        create_session_response = client.post("/api/v1/sessions", headers=_auth_headers(), json={})
        assert create_session_response.status_code == 201
        session_id = create_session_response.json()["session"]["id"]

        first_run_response = client.post(
            "/api/v1/runs",
            headers=_auth_headers(),
            json={
                "session_id": session_id,
                "input_parts": [{"type": "text", "text": "hello-1"}],
            },
        )
        assert first_run_response.status_code == 201
        first_run_id = first_run_response.json()["id"]

    _mark_run_completed(session_id, first_run_id)

    with TestClient(create_app()) as client:
        second_run_response = client.post(
            f"/api/v1/sessions/{session_id}/runs",
            headers=_auth_headers(),
            json={
                "input_parts": [{"type": "text", "text": "hello-2"}],
            },
        )
        assert second_run_response.status_code == 201
        second_run_id = second_run_response.json()["id"]

    _mark_run_completed(session_id, second_run_id)

    settings = get_settings()
    first_run_dir = settings.run_store_dir / first_run_id
    second_run_dir = settings.run_store_dir / second_run_id
    first_run_dir.mkdir(parents=True, exist_ok=True)
    second_run_dir.mkdir(parents=True, exist_ok=True)
    (first_run_dir / "message.json").write_text(
        json.dumps([{"type": "message", "content": "first"}]),
        encoding="utf-8",
    )
    (second_run_dir / "message.json").write_text(
        json.dumps([{"type": "message", "content": "second"}]),
        encoding="utf-8",
    )

    with TestClient(create_app()) as client:
        session_response = client.get(
            f"/api/v1/sessions/{session_id}?runs_limit=1&include_message=true",
            headers=_auth_headers(),
        )
        page_2_response = client.get(
            f"/api/v1/sessions/{session_id}?runs_limit=1&before_sequence_no=2&include_message=true",
            headers=_auth_headers(),
        )

    assert session_response.status_code == 200
    assert session_response.json()["session"]["runs"][0]["id"] == second_run_id
    assert session_response.json()["session"]["runs"][0]["message"] == [{"type": "message", "content": "second"}]
    assert session_response.json()["message"] == [{"type": "message", "content": "second"}]
    assert session_response.json()["state"] is None
    assert session_response.json()["session"]["runs_has_more"] is True
    assert session_response.json()["session"]["runs_next_before_sequence_no"] == 2

    assert page_2_response.status_code == 200
    assert page_2_response.json()["session"]["runs"][0]["id"] == first_run_id
    assert page_2_response.json()["session"]["runs"][0]["message"] == [{"type": "message", "content": "first"}]
    assert page_2_response.json()["session"]["runs_has_more"] is False


def test_run_get_rejects_non_array_message_blob() -> None:
    _create_schema()

    with TestClient(create_app()) as client:
        create_session_response = client.post("/api/v1/sessions", headers=_auth_headers(), json={})
        assert create_session_response.status_code == 201
        session_id = create_session_response.json()["session"]["id"]

        create_run_response = client.post(
            "/api/v1/runs",
            headers=_auth_headers(),
            json={
                "session_id": session_id,
                "input_parts": [{"type": "text", "text": "hello"}],
            },
        )
        assert create_run_response.status_code == 201
        run_id = create_run_response.json()["id"]

    _mark_run_completed(session_id, run_id)

    settings = get_settings()
    run_dir = settings.run_store_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "message.json").write_text(json.dumps({"events": []}), encoding="utf-8")

    with TestClient(create_app()) as client:
        run_response = client.get(
            f"/api/v1/runs/{run_id}?include_message=true",
            headers=_auth_headers(),
        )

    assert run_response.status_code == 500
    assert "top-level JSON array" in run_response.json()["detail"]


def test_run_get_exposes_session_state_and_message() -> None:
    _create_schema()

    with TestClient(create_app()) as client:
        create_session_response = client.post("/api/v1/sessions", headers=_auth_headers(), json={})
        assert create_session_response.status_code == 201
        session_id = create_session_response.json()["session"]["id"]

        create_run_response = client.post(
            "/api/v1/runs",
            headers=_auth_headers(),
            json={
                "session_id": session_id,
                "input_parts": [{"type": "text", "text": "hello"}],
            },
        )
        assert create_run_response.status_code == 201
        run_id = create_run_response.json()["id"]

    _mark_run_completed(session_id, run_id)

    settings = get_settings()
    run_dir = settings.run_store_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "state.json").write_text(json.dumps({"state": "ready"}), encoding="utf-8")
    (run_dir / "message.json").write_text(json.dumps([]), encoding="utf-8")

    with TestClient(create_app()) as client:
        run_response = client.get(
            f"/api/v1/runs/{run_id}?include_message=true",
            headers=_auth_headers(),
        )

    assert run_response.status_code == 200
    assert run_response.json()["session"]["id"] == session_id
    assert run_response.json()["state"] == {"state": "ready"}
    assert run_response.json()["message"] == []
    assert run_response.json()["run"]["has_state"] is True
    assert run_response.json()["run"]["has_message"] is True

    with TestClient(create_app()) as client:
        session_response = client.get(
            f"/api/v1/sessions/{session_id}?include_message=true",
            headers=_auth_headers(),
        )

    assert session_response.status_code == 200
    assert session_response.json()["state"] == {"state": "ready"}
    assert session_response.json()["message"] == []


def test_session_detail_can_include_input_parts_for_run_replay() -> None:
    _create_schema()

    with TestClient(create_app()) as client:
        create_session_response = client.post(
            "/api/v1/sessions",
            headers=_auth_headers(),
            json={
                "input_parts": [
                    {"type": "mode", "mode": "plan"},
                    {"type": "text", "text": "render this exactly"},
                    {
                        "type": "url",
                        "url": "https://example.com/image.png",
                        "kind": "image",
                        "filename": "image.png",
                    },
                ]
            },
        )
        assert create_session_response.status_code == 201
        session_id = create_session_response.json()["session"]["id"]

        default_response = client.get(f"/api/v1/sessions/{session_id}", headers=_auth_headers())
        include_response = client.get(
            f"/api/v1/sessions/{session_id}?include_input_parts=true",
            headers=_auth_headers(),
        )

    assert default_response.status_code == 200
    assert default_response.json()["session"]["runs"][0]["input_parts"] is None
    assert include_response.status_code == 200
    assert include_response.json()["session"]["runs"][0]["input_parts"] == [
        {"type": "mode", "mode": "plan", "params": None, "metadata": None},
        {"type": "text", "text": "render this exactly", "metadata": None},
        {
            "type": "url",
            "url": "https://example.com/image.png",
            "kind": "image",
            "filename": "image.png",
            "storage": "ephemeral",
            "metadata": None,
        },
    ]


def test_session_turns_return_completed_runs_with_raw_input_and_output() -> None:
    _create_schema()

    with TestClient(create_app()) as client:
        create_session_response = client.post("/api/v1/sessions", headers=_auth_headers(), json={})
        assert create_session_response.status_code == 201
        session_id = create_session_response.json()["session"]["id"]

        first_run_response = client.post(
            "/api/v1/runs",
            headers=_auth_headers(),
            json={"session_id": session_id, "input_parts": [{"type": "text", "text": "completed-1"}]},
        )
        assert first_run_response.status_code == 201
        first_run_id = first_run_response.json()["id"]

    _mark_run_completed(session_id, first_run_id, output_text="answer-1")

    with TestClient(create_app()) as client:
        failed_run_response = client.post(
            f"/api/v1/sessions/{session_id}/runs",
            headers=_auth_headers(),
            json={"input_parts": [{"type": "text", "text": "failed"}]},
        )
        assert failed_run_response.status_code == 201
        failed_run_id = failed_run_response.json()["id"]

    def _mark_failed() -> None:
        async def _run() -> None:
            settings = get_settings()
            for _ in range(5):
                engine = create_engine(settings.resolved_database_url)
                session_factory = create_session_factory(engine)
                try:
                    async with session_factory() as db_session:
                        session_record = await db_session.get(SessionRecord, session_id)
                        run_record = await db_session.get(RunRecord, failed_run_id)
                        assert isinstance(session_record, SessionRecord)
                        assert isinstance(run_record, RunRecord)
                        run_record.status = "failed"
                        run_record.error_message = "boom"
                        session_record.active_run_id = None
                        await db_session.commit()
                        return
                except OperationalError:
                    await asyncio.sleep(0.1)
                finally:
                    await engine.dispose()
            raise AssertionError("failed to mark run failed due to persistent database lock")

        asyncio.run(_run())

    _mark_failed()

    with TestClient(create_app()) as client:
        second_run_response = client.post(
            f"/api/v1/sessions/{session_id}/runs",
            headers=_auth_headers(),
            json={"input_parts": [{"type": "text", "text": "completed-2"}]},
        )
        assert second_run_response.status_code == 201
        second_run_id = second_run_response.json()["id"]

    _mark_run_completed(session_id, second_run_id, output_text="answer-2")

    with TestClient(create_app()) as client:
        page_1_response = client.get(
            f"/api/v1/sessions/{session_id}/turns?limit=1",
            headers=_auth_headers(),
        )
        page_2_response = client.get(
            f"/api/v1/sessions/{session_id}/turns?limit=1&before_sequence_no=3",
            headers=_auth_headers(),
        )
        page_2_cursor_response = client.get(
            f"/api/v1/sessions/{session_id}/turns?limit=1&cursor={second_run_id}",
            headers=_auth_headers(),
        )

    assert page_1_response.status_code == 200
    page_1_payload = page_1_response.json()
    assert page_1_payload["session_id"] == session_id
    assert page_1_payload["has_more"] is True
    assert page_1_payload["next_cursor"] == second_run_id
    assert page_1_payload["next_before_sequence_no"] == 3
    assert len(page_1_payload["turns"]) == 1
    assert page_1_payload["turns"][0]["run_id"] == second_run_id
    assert page_1_payload["turns"][0]["input_parts"] == [{"type": "text", "text": "completed-2", "metadata": None}]
    assert page_1_payload["turns"][0]["output_text"] == "answer-2"

    assert page_2_response.status_code == 200
    page_2_payload = page_2_response.json()
    assert page_2_payload["has_more"] is False
    assert page_2_payload["next_cursor"] is None
    assert [turn["run_id"] for turn in page_2_payload["turns"]] == [first_run_id]
    assert page_2_payload["turns"][0]["output_text"] == "answer-1"

    assert page_2_cursor_response.status_code == 200
    page_2_cursor_payload = page_2_cursor_response.json()
    assert page_2_cursor_payload["has_more"] is False
    assert page_2_cursor_payload["next_cursor"] is None
    assert [turn["run_id"] for turn in page_2_cursor_payload["turns"]] == [first_run_id]
    assert page_2_cursor_payload["turns"][0]["output_text"] == "answer-1"


def test_list_sessions_hides_memory_sessions_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YA_CLAW_AGENCY_ENABLED", "false")
    get_settings.cache_clear()
    _create_schema()

    settings = get_settings()

    async def _run() -> None:
        engine = create_engine(settings.resolved_database_url)
        session_factory = create_session_factory(engine)
        try:
            async with session_factory() as db_session:
                source_session = SessionRecord(id="source-session", profile_name="general", session_metadata={})
                memory_session = SessionRecord(
                    id="memory-session",
                    parent_session_id="source-session",
                    profile_name="general",
                    session_type="memory",
                    source_session_id="source-session",
                    session_metadata={"memory": {"source_session_id": "source-session"}},
                )
                memory_state = SessionMemoryStateRecord(
                    source_session_id="source-session",
                    memory_session_id="memory-session",
                    extract_count=3,
                    turns_since_extract=1,
                    extracts_since_summary=2,
                )
                db_session.add_all([source_session, memory_session, memory_state])
                await db_session.commit()
        finally:
            await engine.dispose()

    asyncio.run(_run())

    with TestClient(create_app()) as client:
        default_response = client.get("/api/v1/sessions", headers=_auth_headers())
        internal_response = client.get("/api/v1/sessions?include_internal=true", headers=_auth_headers())

    assert default_response.status_code == 200
    default_payload = default_response.json()
    assert [session["id"] for session in default_payload] == ["source-session"]
    assert default_payload[0]["memory_state"]["extract_count"] == 3
    assert default_payload[0]["memory_state"]["turns_since_extract"] == 1
    assert internal_response.status_code == 200
    assert {session["id"] for session in internal_response.json()} == {"source-session", "memory-session"}


def test_list_sessions_include_internal_exposes_agency_session() -> None:
    _create_schema()

    settings = get_settings()

    async def _run() -> None:
        engine = create_engine(settings.resolved_database_url)
        session_factory = create_session_factory(engine)
        try:
            async with session_factory() as db_session:
                source_session = SessionRecord(id="source-session", profile_name="general", session_metadata={})
                agency_session = SessionRecord(
                    id="agency-session",
                    profile_name="general",
                    session_type="agency",
                    source_session_id="19aafc63e85a06fb38321a895de724d0",
                    session_metadata={
                        "agency": {
                            "kind": "claw_agency_session",
                            "scope": "global",
                            "scope_key": "agency:global",
                            "version": 1,
                        }
                    },
                )
                db_session.add_all([source_session, agency_session])
                await db_session.commit()
        finally:
            await engine.dispose()

    asyncio.run(_run())

    app = create_app()
    with TestClient(app) as client:
        app.state.agency_dispatcher = None
        default_response = client.get("/api/v1/sessions", headers=_auth_headers())
        internal_response = client.get("/api/v1/sessions?include_internal=true", headers=_auth_headers())

    assert default_response.status_code == 200
    assert [session["id"] for session in default_response.json()] == ["source-session"]
    assert internal_response.status_code == 200
    internal_ids = {session["id"] for session in internal_response.json()}
    assert {"source-session", "agency-session"}.issubset(internal_ids)
    assert any(
        session["session_type"] == "agency"
        and session["metadata"].get("agency", {}).get("scope_key") == "agency:global"
        for session in internal_response.json()
    )


def test_memory_api_enqueues_jobs_exposes_state_and_uses_filetree_for_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YA_CLAW_MEMORY_ENABLED", "true")
    monkeypatch.setenv("YA_CLAW_AGENCY_ENABLED", "false")
    get_settings.cache_clear()
    _create_schema()

    app = create_app()
    with TestClient(app) as client:
        app.state.execution_supervisor = None
        create_session_response = client.post("/api/v1/sessions", headers=_auth_headers(), json={})
        assert create_session_response.status_code == 201
        session_id = create_session_response.json()["session"]["id"]

        create_run_response = client.post(
            f"/api/v1/sessions/{session_id}/runs",
            headers=_auth_headers(),
            json={"input_parts": [{"type": "text", "text": "remember this"}]},
        )
        assert create_run_response.status_code == 201
        run_id = create_run_response.json()["id"]

    _mark_run_completed(session_id, run_id, output_text="done")

    app = create_app()
    with TestClient(app) as client:
        app.state.execution_supervisor = None
        extract_response = client.post(
            f"/api/v1/sessions/{session_id}/memory:extract",
            headers=_auth_headers(),
            json={"reason": "api_smoke", "run_ids": [run_id]},
        )
        assert extract_response.status_code == 202
        extract_payload = extract_response.json()
        assert extract_payload["accepted"] is True
        assert extract_payload["kind"] == "extract"
        assert isinstance(extract_payload["run_id"], str)

        summary_response = client.post(
            f"/api/v1/sessions/{session_id}/memory:summarize",
            headers=_auth_headers(),
            json={"reason": "api_smoke_summary"},
        )
        assert summary_response.status_code == 202
        assert summary_response.json()["kind"] == "summary"
        assert summary_response.json()["run_id"] is None

        session_response = client.get(f"/api/v1/sessions/{session_id}", headers=_auth_headers())
        list_response = client.get("/api/v1/sessions", headers=_auth_headers())
        internal_response = client.get("/api/v1/sessions?include_internal=true", headers=_auth_headers())
        invalid_run_response = client.post(
            f"/api/v1/sessions/{session_id}/memory:extract",
            headers=_auth_headers(),
            json={"reason": "bad_run", "run_ids": ["missing-run"]},
        )
        removed_memory_read_response = client.get(
            f"/api/v1/sessions/{session_id}/memory",
            headers=_auth_headers(),
        )

    assert session_response.status_code == 200
    memory_state = session_response.json()["session"]["memory_state"]
    assert memory_state["memory_session_id"] is not None
    assert memory_state["last_extract_run_id"] == extract_payload["run_id"]
    assert memory_state["pending_summary"] is True
    assert memory_state["metadata"]["pending_requests"][0]["kind"] == "summary"
    assert list_response.status_code == 200
    assert [session["id"] for session in list_response.json()] == [session_id]
    assert list_response.json()[0]["memory_state"]["last_extract_run_id"] == extract_payload["run_id"]
    assert internal_response.status_code == 200
    assert {session["session_type"] for session in internal_response.json()} == {"conversation", "memory"}
    assert invalid_run_response.status_code == 422
    assert removed_memory_read_response.status_code == 404


def test_run_trace_extracts_tool_call_and_response_with_trimming() -> None:
    _create_schema()

    with TestClient(create_app()) as client:
        create_session_response = client.post("/api/v1/sessions", headers=_auth_headers(), json={})
        assert create_session_response.status_code == 201
        session_id = create_session_response.json()["session"]["id"]

        create_run_response = client.post(
            "/api/v1/runs",
            headers=_auth_headers(),
            json={"session_id": session_id, "input_parts": [{"type": "text", "text": "use tools"}]},
        )
        assert create_run_response.status_code == 201
        run_id = create_run_response.json()["id"]

    _mark_run_completed(session_id, run_id, output_text="done")

    settings = get_settings()
    run_dir = settings.run_store_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "message.json").write_text(
        json.dumps([
            {"type": "TEXT_MESSAGE_CHUNK", "messageId": "m1", "delta": "hello"},
            {
                "type": "TOOL_CALL_CHUNK",
                "toolCallId": "tool-1",
                "toolCallName": "shell_exec",
                "delta": '{"command":"echo hello"}',
            },
            {
                "type": "TOOL_CALL_RESULT",
                "toolCallId": "tool-1",
                "messageId": "tool-1:result",
                "role": "tool",
                "content": "x" * 300,
            },
        ]),
        encoding="utf-8",
    )

    with TestClient(create_app()) as client:
        trace_response = client.get(
            f"/api/v1/runs/{run_id}/trace?max_item_chars=256&max_total_chars=280",
            headers=_auth_headers(),
        )

    assert trace_response.status_code == 200
    payload = trace_response.json()
    assert payload["run_id"] == run_id
    assert payload["session_id"] == session_id
    assert payload["truncated"] is True
    assert payload["item_count"] == 2
    assert payload["trace"][0] == {
        "sequence_no": 1,
        "type": "tool_call",
        "tool_call_id": "tool-1",
        "tool_name": "shell_exec",
        "message_id": None,
        "role": None,
        "content": '{"command":"echo hello"}',
        "truncated": False,
    }
    assert payload["trace"][1]["type"] == "tool_response"
    assert payload["trace"][1]["tool_call_id"] == "tool-1"
    assert payload["trace"][1]["message_id"] == "tool-1:result"
    assert payload["trace"][1]["role"] == "tool"
    assert len(payload["trace"][1]["content"]) == 256
    assert payload["trace"][1]["truncated"] is True
