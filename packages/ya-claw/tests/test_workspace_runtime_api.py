from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from ya_claw.app import create_app
from ya_claw.config import get_settings
from ya_claw.db.engine import create_engine, create_session_factory
from ya_claw.orm.base import Base
from ya_claw.orm.tables import SessionRecord


@pytest.fixture(autouse=True)
def clear_claw_settings(monkeypatch, tmp_path: Path) -> None:
    for env_name in (
        "YA_CLAW_API_TOKEN",
        "YA_CLAW_DATABASE_URL",
        "YA_CLAW_DATA_DIR",
        "YA_CLAW_WEB_DIST_DIR",
        "YA_CLAW_WORKSPACE_DIR",
        "YA_CLAW_WORKSPACE_PROVIDER_BACKEND",
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


def _create_session_with_sandbox() -> str:
    async def _run() -> str:
        settings = get_settings()
        engine = create_engine(settings.resolved_database_url)
        session_factory = create_session_factory(engine)
        try:
            async with session_factory() as db_session:
                session_record = SessionRecord(
                    id="session-workspace-1",
                    profile_name="default",
                    session_metadata={
                        "sandbox": {
                            "provider": "docker",
                            "scope": "session",
                            "status": "ready",
                            "ready_state": "ready",
                            "container_ref": "ya-claw-workspace-ref",
                            "container_id": "container-1",
                            "verified_container_id": "container-1",
                            "image": "python:3.11",
                            "cwd": "/workspace",
                            "retention_policy": "stop_on_idle",
                            "idle_ttl_seconds": 3600,
                            "last_used_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                        }
                    },
                )
                db_session.add(session_record)
                await db_session.commit()
                return session_record.id
        finally:
            await engine.dispose()

    return asyncio.run(_run())


def test_workspace_runtime_api_exposes_local_backend() -> None:
    _create_schema()

    with TestClient(create_app()) as client:
        response = client.get("/api/v1/workspace/runtime", headers=_auth_headers())

    assert response.status_code == 200
    payload = response.json()
    assert payload["backend"] == "local"
    assert payload["status"] == "ready"
    assert payload["execution_location"] == "local"
    assert payload["workspace"]["exists"] is True
    assert payload["workspace"]["writable"] is True
    assert payload["capabilities"]["file_browse"] is True
    assert payload["capabilities"]["shell"] is True
    assert payload["capabilities"]["sandbox_prepare"] is False


def test_session_workspace_api_and_session_response_expose_sandbox_state() -> None:
    _create_schema()
    session_id = _create_session_with_sandbox()

    with TestClient(create_app()) as client:
        workspace_response = client.get(f"/api/v1/sessions/{session_id}/workspace", headers=_auth_headers())
        sandbox_response = client.get(f"/api/v1/sessions/{session_id}/sandbox", headers=_auth_headers())
        session_response = client.get(f"/api/v1/sessions/{session_id}", headers=_auth_headers())
        list_response = client.get("/api/v1/sessions", headers=_auth_headers())

    assert workspace_response.status_code == 200
    workspace_payload = workspace_response.json()
    assert workspace_payload["binding"]["provider"] == "local"
    assert workspace_payload["sandbox_state"]["container_id"] == "container-1"
    assert workspace_payload["sandbox_state"]["ready_state"] == "ready"

    assert sandbox_response.status_code == 200
    assert sandbox_response.json()["container_ref"] == "ya-claw-workspace-ref"

    assert session_response.status_code == 200
    session_payload = session_response.json()["session"]
    assert session_payload["workspace_state"]["sandbox_state"]["container_id"] == "container-1"

    assert list_response.status_code == 200
    list_payload = list_response.json()
    assert list_payload[0]["workspace_state"]["sandbox_state"]["ready_state"] == "ready"


def test_local_backend_rejects_manual_sandbox_lifecycle() -> None:
    _create_schema()
    session_id = _create_session_with_sandbox()

    with TestClient(create_app()) as client:
        prepare_response = client.post(f"/api/v1/sessions/{session_id}/sandbox:prepare", headers=_auth_headers())
        stop_response = client.post(f"/api/v1/sessions/{session_id}/sandbox:stop", headers=_auth_headers())

    assert prepare_response.status_code == 409
    assert stop_response.status_code == 409
