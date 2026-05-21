from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from ya_claw.app import create_app
from ya_claw.config import get_settings
from ya_claw.db.engine import create_engine, create_session_factory
from ya_claw.orm.base import Base


@pytest.fixture(autouse=True)
def clear_claw_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
        "YA_CLAW_AGENCY_ENABLED",
        "YA_CLAW_AGENCY_TIMER_INTERVAL_SECONDS",
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


def _fire_kinds() -> list[str]:
    async def _run() -> list[str]:
        from sqlalchemy import select
        from ya_claw.orm.tables import AgencyFireRecord

        settings = get_settings()
        engine = create_engine(settings.resolved_database_url)
        try:
            session_factory = create_session_factory(engine)
            async with session_factory() as db_session:
                result = await db_session.execute(select(AgencyFireRecord.kind).order_by(AgencyFireRecord.created_at))
                return [str(item) for item in result.scalars().all()]
        finally:
            await engine.dispose()

    return asyncio.run(_run())


def test_agency_config_status_fires_and_no_product_trigger() -> None:
    _create_schema()

    app = create_app()
    with TestClient(app) as client:
        app.state.execution_supervisor = None
        config_response = client.get("/api/v1/agency/config", headers=_auth_headers())
        assert config_response.status_code == 200
        config = config_response.json()
        assert config["enabled"] is False
        assert config["singleton_scope_key"] == "agency:global"
        assert config["risk_policy"] == {"max_auto_action_risk": "extra_high"}
        assert config["next_fire_at"] is None

        status_response = client.get("/api/v1/agency/status", headers=_auth_headers())
        assert status_response.status_code == 200
        status = status_response.json()
        assert status["agency_session_id"] == config["agency_session_id"]
        assert status["state"] in {"idle", "queued", "running"}
        assert status["next_fire_at"] is None
        assert isinstance(status["pending_fire_count"], int)

        fires_response = client.get("/api/v1/agency/fires", headers=_auth_headers())
        assert fires_response.status_code == 200
        assert isinstance(fires_response.json()["fires"], list)

        trigger_response = client.post(
            "/api/v1/agency:trigger",
            headers=_auth_headers(),
            json={"kind": "message_observed"},
        )
        assert trigger_response.status_code == 404


def test_clear_agency_resets_singleton_session_and_workspace_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("YA_CLAW_AGENCY_ENABLED", "false")
    get_settings.cache_clear()
    _create_schema()

    workspace_dir = tmp_path / "workspace"
    agency_md = workspace_dir / "AGENCY.md"
    agency_action_log = workspace_dir / "agency" / "ACTION_LOG.md"

    app = create_app()
    with TestClient(app) as client:
        app.state.execution_supervisor = None
        config_response = client.get("/api/v1/agency/config", headers=_auth_headers())
        assert config_response.status_code == 200
        old_session_id = config_response.json()["agency_session_id"]
        agency_action_log.parent.mkdir(parents=True, exist_ok=True)
        agency_md.write_text("old agency index", encoding="utf-8")
        agency_action_log.write_text("old agency log", encoding="utf-8")

        clear_response = client.post("/api/v1/agency:clear", headers=_auth_headers())
        assert clear_response.status_code == 202
        clear_payload = clear_response.json()
        assert clear_payload["accepted"] is True
        assert clear_payload["cleared_session_id"] == old_session_id
        assert clear_payload["new_agency_session_id"] != old_session_id
        assert clear_payload["archived_run_ids"] == []
        assert clear_payload["deleted_fire_count"] == 0
        assert clear_payload["agency_session"]["id"] == clear_payload["new_agency_session_id"]
        assert "# Agency" in agency_md.read_text(encoding="utf-8")
        assert "# Agency Action Log" in agency_action_log.read_text(encoding="utf-8")

        status_response = client.get("/api/v1/agency/status", headers=_auth_headers())
        assert status_response.status_code == 200
        status_payload = status_response.json()
        assert status_payload["agency_session_id"] == clear_payload["new_agency_session_id"]
        assert status_payload["agency_session"]["run_count"] == 0
        assert status_payload["pending_fire_count"] == 0

        fires_response = client.get("/api/v1/agency/fires", headers=_auth_headers())
        assert fires_response.status_code == 200
        assert fires_response.json()["fires"] == []


def test_clear_agency_cancels_active_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YA_CLAW_AGENCY_ENABLED", "true")
    get_settings.cache_clear()
    _create_schema()

    async def _seed_active_run() -> None:
        from ya_claw.agency.lifecycle import AgencyLifecycle
        from ya_claw.controller.models import TriggerType
        from ya_claw.db.engine import create_session_factory
        from ya_claw.orm.tables import RunRecord
        from ya_claw.runtime_state import InMemoryRuntimeState

        settings = get_settings()
        engine = create_engine(settings.resolved_database_url)
        try:
            session_factory = create_session_factory(engine)
            async with session_factory() as db_session:
                lifecycle = AgencyLifecycle(settings=settings, runtime_state=InMemoryRuntimeState())
                agency_session = await lifecycle.ensure_agency_session(db_session)
                run = RunRecord(
                    id="activeagencyrun000000000000000001",
                    session_id=agency_session.id,
                    sequence_no=1,
                    status="running",
                    trigger_type=TriggerType.AGENCY.value,
                    input_parts=[],
                    run_metadata={"agency": {"kind": "episode"}},
                )
                db_session.add(run)
                agency_session.head_run_id = run.id
                await db_session.commit()
        finally:
            await engine.dispose()

    asyncio.run(_seed_active_run())

    app = create_app()
    with TestClient(app) as client:
        app.state.execution_supervisor = None
        clear_response = client.post("/api/v1/agency:clear", headers=_auth_headers())
        assert clear_response.status_code == 202
        payload = clear_response.json()
        assert payload["cleared_session_id"] is not None
        assert "activeagencyrun000000000000000001" in payload["archived_run_ids"]


def test_session_submit_copies_message_to_agency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YA_CLAW_AGENCY_ENABLED", "true")
    get_settings.cache_clear()
    _create_schema()

    app = create_app()
    with TestClient(app) as client:
        app.state.execution_supervisor = None
        create_session = client.post("/api/v1/sessions", headers=_auth_headers(), json={})
        assert create_session.status_code == 201
        session_id = create_session.json()["session"]["id"]

        submit_response = client.post(
            f"/api/v1/sessions/{session_id}/submit",
            headers=_auth_headers(),
            json={"input_parts": [{"type": "text", "text": "hello agency"}]},
        )
        assert submit_response.status_code == 202
        assert submit_response.json()["delivery"] == "submitted"

        fires_response = client.get("/api/v1/agency/fires", headers=_auth_headers())
        assert fires_response.status_code == 200
        fires = fires_response.json()["fires"]
        assert len(fires) == 1
        fire = fires[0]
        assert fire["kind"] == "message_observed"
        assert fire["source_session_id"] == session_id
        assert fire["source_run_id"] == submit_response.json()["run_id"]
        assert fire["payload"]["input_parts"][0]["text"] == "hello agency"
        assert _fire_kinds() == ["message_observed"]


def test_agency_source_session_submit_api_rejects_completed_agency_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YA_CLAW_AGENCY_ENABLED", "true")
    get_settings.cache_clear()
    _create_schema()

    async def _seed() -> tuple[str, str, str]:
        from ya_claw.agency.lifecycle import AgencyLifecycle
        from ya_claw.controller.models import TriggerType
        from ya_claw.orm.tables import RunRecord, SessionRecord
        from ya_claw.runtime_state import InMemoryRuntimeState

        settings = get_settings()
        engine = create_engine(settings.resolved_database_url)
        try:
            session_factory = create_session_factory(engine)
            async with session_factory() as db_session:
                source_session = SessionRecord(id="source-session", profile_name="general", session_metadata={})
                lifecycle = AgencyLifecycle(settings=settings, runtime_state=InMemoryRuntimeState())
                agency_session = await lifecycle.ensure_agency_session(db_session)
                agency_run = RunRecord(
                    id="agencycompletedrun0000000000001",
                    session_id=agency_session.id,
                    sequence_no=1,
                    status="completed",
                    trigger_type=TriggerType.AGENCY.value,
                    input_parts=[],
                    run_metadata={"agency": {"fire_ids": ["fire-1"]}},
                )
                db_session.add_all([source_session, agency_run])
                agency_session.active_run_id = agency_run.id
                agency_session.head_run_id = agency_run.id
                await db_session.commit()
                return source_session.id, agency_session.id, agency_run.id
        finally:
            await engine.dispose()

    source_session_id, agency_session_id, agency_run_id = asyncio.run(_seed())

    app = create_app()
    with TestClient(app) as client:
        app.state.execution_supervisor = None
        response = client.post(
            "/api/v1/agency/source-session:submit",
            headers=_auth_headers(),
            json={
                "source_session_id": source_session_id,
                "prompt": "Please review Agency findings and update the thread.",
                "metadata": {"fire_ids": ["fire-1"]},
                "handoff_kind": "reminder",
                "agency_session_id": agency_session_id,
                "agency_run_id": agency_run_id,
            },
        )

    assert response.status_code == 403
