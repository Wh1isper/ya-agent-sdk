from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from ya_claw.app import create_app
from ya_claw.config import get_settings


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
    ):
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setenv("YA_CLAW_API_TOKEN", "test-token")
    monkeypatch.setenv("YA_CLAW_DATA_DIR", str(tmp_path / "runtime-data"))
    monkeypatch.setenv("YA_CLAW_WORKSPACE_DIR", str(tmp_path / "workspace"))
    monkeypatch.setenv("YA_CLAW_WORKSPACE_PROVIDER_BACKEND", "local")
    monkeypatch.setenv("YA_CLAW_PROFILE_SEED_FILE", str(tmp_path / "profiles.yaml"))
    monkeypatch.setenv("YA_CLAW_AUTO_SEED_PROFILES", "false")
    get_settings.cache_clear()
    initialize_sqlite_database(get_settings().resolved_database_url)
    yield
    get_settings.cache_clear()


def _auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer test-token"}


def _profile_payload(*, model: str = "test") -> dict[str, object]:
    return {
        "schema_version": 2,
        "agent": {
            "model": model,
            "name": "custom",
            "instructions": "You are a custom profile.",
            "model_settings": {"temperature": 0},
            "capabilities": ["FilesystemCapability", "ShellCapability"],
        },
        "host": {
            "model_config_preset": "gpt5_270k",
            "tool_groups": ["session"],
            "need_user_approve_mcps": ["context7"],
            "enabled_mcps": ["context7"],
            "mcp_servers": {
                "context7": {
                    "transport": "streamable_http",
                    "url": "https://mcp.context7.com/mcp",
                    "required": False,
                }
            },
            "workspace_backend_hint": "local",
        },
        "subagents": [
            {
                "schema_version": 1,
                "route": "debugger",
                "execution_modes": ["foreground", "background"],
                "durability": "restart",
                "agent": {
                    "model": "test",
                    "name": "debugger",
                    "description": "Debug runtime issues",
                    "instructions": "Return a root cause.",
                    "capabilities": ["FilesystemCapability"],
                },
            }
        ],
        "enabled": True,
        "source_type": "api",
    }


def test_profile_crud_and_seed_api(tmp_path: Path) -> None:
    seed_file = tmp_path / "profiles.yaml"
    seed_file.write_text(
        """
version: 2
profiles:
  - schema_version: 2
    name: seeded
    agent:
      model: test
      name: seeded
      capabilities: [FilesystemCapability]
    host:
      tool_groups: [session]
    enabled: true
    source_type: seed
    source_version: '2'
""".strip(),
        encoding="utf-8",
    )

    with TestClient(create_app()) as client:
        response = client.put(
            "/api/v1/profiles/custom",
            headers=_auth_headers(),
            json=_profile_payload(),
        )
        assert response.status_code == 200
        detail = response.json()
        assert detail["schema_version"] == 2
        assert detail["name"] == "custom"
        assert detail["model"] == "test"
        assert detail["agent"]["instructions"] == "You are a custom profile."
        assert detail["host"]["tool_groups"] == ["session"]
        assert detail["subagents"][0]["route"] == "debugger"
        assert detail["host"]["mcp_servers"]["context7"]["required"] is False

        legacy = client.put(
            "/api/v1/profiles/legacy",
            headers=_auth_headers(),
            json={"model": "test", "builtin_toolsets": ["core"]},
        )
        assert legacy.status_code == 422

        mixed = _profile_payload()
        mixed["include_builtin_subagents"] = False
        mixed_response = client.put(
            "/api/v1/profiles/mixed",
            headers=_auth_headers(),
            json=mixed,
        )
        assert mixed_response.status_code == 422

        unknown = _profile_payload()
        unknown["future_field"] = "ignored-no-more"
        unknown_response = client.put(
            "/api/v1/profiles/unknown",
            headers=_auth_headers(),
            json=unknown,
        )
        assert unknown_response.status_code == 422

        mismatched = _profile_payload()
        mismatched["agent"] = {"model": "test", "name": "other"}
        mismatch_response = client.put(
            "/api/v1/profiles/custom",
            headers=_auth_headers(),
            json=mismatched,
        )
        assert mismatch_response.status_code == 422

        list_response = client.get("/api/v1/profiles", headers=_auth_headers())
        assert list_response.status_code == 200
        assert [item["name"] for item in list_response.json()] == ["custom"]

        seed_response = client.post(
            "/api/v1/profiles/seed",
            headers=_auth_headers(),
            json={"prune_missing": False},
        )
        assert seed_response.status_code == 200
        assert seed_response.json()["seeded_names"] == ["seeded"]

        seeded = client.get(
            "/api/v1/profiles/seeded",
            headers=_auth_headers(),
        )
        assert seeded.status_code == 200
        assert seeded.json()["agent"]["capabilities"] == [{"name": "FilesystemCapability", "arguments": None}]

        delete_response = client.delete(
            "/api/v1/profiles/custom",
            headers=_auth_headers(),
        )
        assert delete_response.status_code == 204
        assert (
            client.get(
                "/api/v1/profiles/custom",
                headers=_auth_headers(),
            ).status_code
            == 404
        )
