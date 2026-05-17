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
        "YA_CLAW_AGENCY_TICK_SECONDS",
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
    monkeypatch.setenv("YA_CLAW_AGENCY_TICK_SECONDS", "3600")

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
        assert config["enabled"] is True
        assert config["singleton_scope_key"] == "agency:global"
        assert config["budget_defaults"]["external_actions"] == "deny"
        assert config["deny_external_actions"] is True
        assert config["risk_policy"]["max_auto_action_risk"] == "extra_high"

        status_response = client.get("/api/v1/agency/status", headers=_auth_headers())
        assert status_response.status_code == 200
        status = status_response.json()
        assert status["agency_session_id"] == config["agency_session_id"]
        assert status["state"] in {"idle", "queued", "running"}
        assert isinstance(status["pending_fire_count"], int)

        fires_response = client.get("/api/v1/agency/fires", headers=_auth_headers())
        assert fires_response.status_code == 200
        assert isinstance(fires_response.json()["fires"], list)


def test_agency_trigger_uses_optional_source_session_context() -> None:
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
