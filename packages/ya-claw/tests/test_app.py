from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic_ai.capabilities import AbstractCapability
from ya_claw.app import create_app
from ya_claw.config import get_settings


@dataclass
class _AppPluginCapability(AbstractCapability[Any]):
    result_limit: int = 10

    @classmethod
    def get_serialization_name(cls) -> str:
        return "test.app_plugin"


class _AppPluginEntryPoint:
    name = "test.app_plugin"
    value = "test_app:_AppPluginCapability"
    dist = SimpleNamespace(name="test-app-plugin", version="1.0.0")

    def load(self) -> object:
        return _AppPluginCapability


@pytest.fixture(autouse=True)
def clear_claw_settings(monkeypatch, tmp_path: Path) -> None:
    for env_name in (
        "YA_CLAW_API_TOKEN",
        "YA_CLAW_DATABASE_URL",
        "YA_CLAW_DATA_DIR",
        "YA_CLAW_WEB_DIST_DIR",
        "YA_CLAW_WORKSPACE_DIR",
        "YA_CLAW_AUTO_SEED_PROFILES",
        "YA_CLAW_CAPABILITY_PLUGIN_MANIFEST",
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


def test_create_app_rejects_invalid_explicit_plugin_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "plugins.toml"
    manifest_path.write_text("schema_version = 2\n", encoding="utf-8")
    monkeypatch.setenv("YA_CLAW_CAPABILITY_PLUGIN_MANIFEST", str(manifest_path))
    get_settings.cache_clear()

    with pytest.raises(ValueError, match="Invalid capability plugin manifest"):
        create_app()


def test_create_app_rejects_plugin_grant_argument_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "plugins.toml"
    manifest_path.write_text(
        """\
schema_version = 1
entry_points = ["test.app_plugin"]

[[capabilities]]
name = "test.app_plugin"
arguments = { unexpected = true }
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "ya_agent_sdk.capabilities.catalog.importlib.metadata.entry_points",
        lambda **_kwargs: [_AppPluginEntryPoint()],
    )
    monkeypatch.setenv("YA_CLAW_CAPABILITY_PLUGIN_MANIFEST", str(manifest_path))
    get_settings.cache_clear()

    with pytest.raises(ValueError, match=r"grant 0.*invalid arguments.*unexpected"):
        create_app()


def test_create_app_without_plugin_manifest_does_not_scan_entry_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_scan(**_kwargs: object) -> object:
        raise AssertionError("an omitted manifest must not scan installed entry points")

    monkeypatch.setattr(
        "ya_agent_sdk.capabilities.catalog.importlib.metadata.entry_points",
        reject_scan,
    )

    app = create_app()

    assert app.state.capability_plugins.manifest.entry_points == ()


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
    assert payload["surfaces"] == ["profiles", "sessions", "runs", "schedules", "workflows", "bridges"]


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
