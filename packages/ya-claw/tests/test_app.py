from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from ya_claw.app import create_app
from ya_claw.config import get_settings


@pytest.fixture(autouse=True)
def clear_claw_settings(monkeypatch, tmp_path: Path) -> None:
    for env_name in (
        "YA_CLAW_API_TOKEN",
        "YA_CLAW_DATABASE_URL",
        "YA_CLAW_DATA_DIR",
        "YA_CLAW_WEB_DIST_DIR",
        "YA_CLAW_WORKSPACE_DIR",
        "YA_CLAW_AUTO_SEED_PROFILES",
        "YA_CLAW_SCHEDULE_DISPATCH_ENABLED",
        "YA_CLAW_HEARTBEAT_ENABLED",
        "YA_CLAW_AGENCY_ENABLED",
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
    monkeypatch.setenv("YA_CLAW_AGENCY_ENABLED", "false")
    monkeypatch.setenv("YA_CLAW_BRIDGE_DISPATCH_MODE", "manual")

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer test-token"}


def test_create_app_requires_api_token(monkeypatch) -> None:
    monkeypatch.setenv("YA_CLAW_API_TOKEN", "")
    get_settings.cache_clear()

    with pytest.raises(RuntimeError, match="YA_CLAW_API_TOKEN"):
        create_app()


def test_healthz() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok", "runtime_state": "ok"}


def test_docs_and_openapi_are_public() -> None:
    with TestClient(create_app()) as client:
        docs_response = client.get("/docs")
        openapi_response = client.get("/openapi.json")

    assert docs_response.status_code == 200
    assert "Swagger UI" in docs_response.text
    assert openapi_response.status_code == 200
    assert openapi_response.json()["info"]["title"] == "YA Claw"
    assert openapi_response.json()["info"]["version"] != "0.1.0"


def test_root_requires_authorization() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/")

    assert response.status_code == 401
    assert response.json() == {"detail": "Bearer token required."}


def test_frontend_bundle_loads_without_authorization(monkeypatch, tmp_path: Path) -> None:
    web_dist_dir = tmp_path / "web-dist"
    web_dist_dir.mkdir()
    (web_dist_dir / "index.html").write_text("<html><body>claw shell</body></html>", encoding="utf-8")
    assets_dir = web_dist_dir / "assets"
    assets_dir.mkdir()
    (assets_dir / "app.js").write_text("console.log('ready')", encoding="utf-8")

    monkeypatch.setenv("YA_CLAW_WEB_DIST_DIR", str(web_dist_dir))
    get_settings.cache_clear()

    with TestClient(create_app()) as app_client:
        root_response = app_client.get("/")
        asset_response = app_client.get("/assets/app.js")
        api_response = app_client.get("/api/v1/claw/info")

    assert root_response.status_code == 200
    assert "claw shell" in root_response.text
    assert asset_response.status_code == 200
    assert "console.log('ready')" in asset_response.text
    assert api_response.status_code == 401

    get_settings.cache_clear()


def test_index_without_frontend_bundle() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/", headers=_auth_headers())

    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "YA Claw"
    assert payload["surfaces"] == ["profiles", "sessions", "runs", "schedules", "bridges"]


def test_serves_frontend_bundle(monkeypatch, tmp_path: Path) -> None:
    web_dist_dir = tmp_path / "web-dist"
    web_dist_dir.mkdir()
    (web_dist_dir / "index.html").write_text("<html><body>claw shell</body></html>", encoding="utf-8")
    assets_dir = web_dist_dir / "assets"
    assets_dir.mkdir()
    (assets_dir / "app.js").write_text("console.log('ready')", encoding="utf-8")

    monkeypatch.setenv("YA_CLAW_WEB_DIST_DIR", str(web_dist_dir))
    get_settings.cache_clear()

    with TestClient(create_app()) as app_client:
        root_response = app_client.get("/", headers=_auth_headers())
        asset_response = app_client.get("/assets/app.js", headers=_auth_headers())
        spa_response = app_client.get("/sessions", headers=_auth_headers())

    assert root_response.status_code == 200
    assert "claw shell" in root_response.text
    assert asset_response.status_code == 200
    assert "console.log('ready')" in asset_response.text
    assert spa_response.status_code == 200
    assert "claw shell" in spa_response.text

    get_settings.cache_clear()
