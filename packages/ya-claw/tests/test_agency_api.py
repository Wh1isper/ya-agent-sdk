from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from ya_claw.app import create_app
from ya_claw.config import get_settings
from ya_claw.db.engine import create_engine
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
    monkeypatch.setenv("YA_CLAW_AGENCY_TIMER_INTERVAL_SECONDS", "3600")

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


def test_agency_config_status_and_fires() -> None:
    _create_schema()

    app = create_app()
    with TestClient(app) as client:
        app.state.execution_supervisor = None
        config_response = client.get("/api/v1/agency/config", headers=_auth_headers())
        assert config_response.status_code == 200
        config = config_response.json()
        assert config["enabled"] is False
        assert config["singleton_scope_key"] == "agency:global"
        assert "budget_defaults" not in config
        assert "deny_external_actions" not in config
        assert config["risk_policy"] == {"max_auto_action_risk": "extra_high"}

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
        trigger_response = client.post(
            "/api/v1/agency:trigger",
            headers=_auth_headers(),
            json={"kind": "manual", "client_token": "clear-1", "prompt": "Seed agency."},
        )
        assert trigger_response.status_code == 409

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


def test_agency_trigger_uses_optional_source_session_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YA_CLAW_AGENCY_ENABLED", "true")
    get_settings.cache_clear()
    _create_schema()

    app = create_app()
    with TestClient(app) as client:
        app.state.execution_supervisor = None
        create_session = client.post("/api/v1/sessions", headers=_auth_headers(), json={})
        assert create_session.status_code == 201
        source_session_id = create_session.json()["session"]["id"]

        trigger_response = client.post(
            "/api/v1/agency:trigger",
            headers=_auth_headers(),
            json={
                "kind": "manual",
                "source_session_id": source_session_id,
                "client_token": "manual-1",
                "prompt": "Review memory changes.",
            },
        )
        assert trigger_response.status_code == 202
        trigger_payload = trigger_response.json()
        assert trigger_payload["delivery"] in {"submitted", "merged", "steered"}
        assert trigger_payload["fire"]["source_session_id"] == source_session_id
        assert trigger_payload["fire"]["payload"]["prompt"] == "Review memory changes."

        fires_response = client.get("/api/v1/agency/fires", headers=_auth_headers())
        assert fires_response.status_code == 200
        fires = fires_response.json()["fires"]
        fire = next(item for item in fires if item["id"] == trigger_payload["fire"]["id"])
        assert fire["kind"] == "manual"
        assert fire["source_session_id"] == source_session_id
        if trigger_payload["run_id"] is not None:
            assert fire["run_id"] == trigger_payload["run_id"]
        else:
            assert fire["run_id"] is None or isinstance(fire["run_id"], str)

        status_response = client.get("/api/v1/agency/status", headers=_auth_headers())
        assert status_response.status_code == 200
        status = status_response.json()
        assert status["state"] in {"idle", "queued", "running"}
        assert status["agency_session"]["id"] == trigger_payload["agency_session_id"]
