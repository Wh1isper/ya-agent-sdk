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
def clear_claw_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    for env_name in (
        "YA_CLAW_API_TOKEN",
        "YA_CLAW_DATABASE_URL",
        "YA_CLAW_DATA_DIR",
        "YA_CLAW_WEB_DIST_DIR",
        "YA_CLAW_WORKSPACE_DIR",
        "YA_CLAW_PROFILE_SEED_FILE",
        "YA_CLAW_AUTO_SEED_PROFILES",
        "YA_CLAW_SCHEDULE_DISPATCH_ENABLED",
        "YA_CLAW_WORKFLOW_DISPATCH_ENABLED",
        "YA_CLAW_HEARTBEAT_ENABLED",
    ):
        monkeypatch.delenv(env_name, raising=False)

    monkeypatch.setenv("YA_CLAW_API_TOKEN", "test-token")
    monkeypatch.setenv("YA_CLAW_DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'workflow-api.sqlite3').resolve()}")
    monkeypatch.setenv("YA_CLAW_DATA_DIR", str(tmp_path / "runtime-data"))
    monkeypatch.setenv("YA_CLAW_WORKSPACE_DIR", str(tmp_path / "workspace"))
    monkeypatch.setenv("YA_CLAW_AUTO_SEED_PROFILES", "false")
    monkeypatch.setenv("YA_CLAW_WORKSPACE_PROVIDER_BACKEND", "local")
    monkeypatch.setenv("YA_CLAW_BRIDGE_DISPATCH_MODE", "manual")
    monkeypatch.setenv("YA_CLAW_SCHEDULE_DISPATCH_ENABLED", "false")
    monkeypatch.setenv("YA_CLAW_WORKFLOW_DISPATCH_ENABLED", "false")
    monkeypatch.setenv("YA_CLAW_HEARTBEAT_ENABLED", "false")

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer test-token"}


def _agent_headers() -> dict[str, str]:
    return {
        **_auth_headers(),
        "X-YA-Claw-Session-Id": "session-1",
        "X-YA-Claw-Run-Id": "run-1",
        "X-YA-Claw-Profile-Name": "default",
    }


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


def _definition() -> dict[str, object]:
    return {
        "schema": "ya-claw.workflow.v1",
        "name": "API Workflow",
        "tags": ["api"],
        "inputs": {"type": "object", "required": ["topic"], "properties": {"topic": {"type": "string"}}},
        "nodes": {"a": {"prompt": "Handle {{ inputs.topic }}"}},
        "result": {"from_node": "a"},
    }


def test_workflow_api_crud_trigger_events_and_filters() -> None:
    _create_schema()

    with TestClient(create_app()) as client:
        create_response = client.post(
            "/api/v1/agent/workflows",
            headers=_agent_headers(),
            json={"definition": _definition()},
        )
        assert create_response.status_code == 201
        workflow = create_response.json()
        assert workflow["owner_kind"] == "agent"
        assert workflow["owner_session_id"] == "session-1"
        assert workflow["scope"] == "session"

        other_response = client.post(
            "/api/v1/workflows",
            headers=_auth_headers(),
            json={"definition": {**_definition(), "name": "Other Workflow", "tags": ["other"]}},
        )
        assert other_response.status_code == 201

        list_response = client.get(
            "/api/v1/workflows?only_current_session=true&current_session_id=session-1&tags=api",
            headers=_auth_headers(),
        )
        assert list_response.status_code == 200
        assert [item["id"] for item in list_response.json()["workflows"]] == [workflow["id"]]

        tag_only_response = client.get("/api/v1/workflows?tags=api", headers=_auth_headers())
        assert tag_only_response.status_code == 200
        assert [item["id"] for item in tag_only_response.json()["workflows"]] == [workflow["id"]]

        repeated_tags_response = client.get(
            "/api/v1/workflows?tags=api&tags=missing",
            headers=_auth_headers(),
        )
        assert repeated_tags_response.status_code == 200
        assert repeated_tags_response.json()["workflows"] == []

        trigger_response = client.post(
            f"/api/v1/agent/workflows/{workflow['id']}:trigger",
            headers=_agent_headers(),
            json={"inputs": {"topic": "api"}, "profile_name": "default"},
        )
        assert trigger_response.status_code == 201
        run = trigger_response.json()
        assert run["trigger_kind"] == "agent"
        assert run["supervisor_session_id"] == "session-1"

        runs_response = client.get(
            "/api/v1/workflow-runs?only_supervised_by_current_session=true&current_session_id=session-1",
            headers=_auth_headers(),
        )
        assert runs_response.status_code == 200
        assert [item["id"] for item in runs_response.json()["workflow_runs"]] == [run["id"]]

        events_response = client.get(f"/api/v1/workflow-runs/{run['id']}/events", headers=_auth_headers())
        assert events_response.status_code == 200
        assert events_response.json()["events"][0]["event_type"] == "workflow_queued"

        archive_response = client.post(f"/api/v1/workflows/{workflow['id']}:archive", headers=_auth_headers())
        assert archive_response.status_code == 200
        assert archive_response.json()["status"] == "archived"
