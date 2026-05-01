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
def clear_claw_settings(monkeypatch, tmp_path: Path) -> None:
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


def test_profile_crud_and_seed_api(tmp_path: Path) -> None:
    _create_schema()
    seed_file = tmp_path / "profiles.yaml"
    seed_file.write_text(
        """
profiles:
  - name: seeded
    model: gateway@openai-responses:gpt-5.5
    model_settings_preset: openai_responses_high
    model_config_preset: gpt5_270k
    builtin_toolsets: [core, web]
    enabled_mcps: [context7]
    mcp_servers:
      context7:
        transport: streamable_http
        url: https://mcp.context7.com/mcp
        description: Library docs
        required: false
    unified_subagents: true
    subagents:
      - name: explorer
        description: Explore the codebase
        system_prompt: |
          You explore the codebase.
""".strip(),
        encoding="utf-8",
    )

    with TestClient(create_app()) as client:
        put_response = client.put(
            "/api/v1/profiles/custom",
            headers=_auth_headers(),
            json={
                "model": "gateway@openai-responses:gpt-5.5",
                "model_settings_preset": "openai_responses_high",
                "model_config_preset": "gpt5_270k",
                "system_prompt": "You are a custom profile.",
                "toolsets": ["filesystem", "shell"],
                "need_user_approve_mcps": ["context7"],
                "enabled_mcps": ["context7", "github"],
                "disabled_mcps": ["github"],
                "mcp_servers": {
                    "context7": {
                        "transport": "streamable_http",
                        "url": "https://mcp.context7.com/mcp",
                        "description": "Library docs",
                        "required": False,
                    },
                    "github": {
                        "transport": "streamable_http",
                        "url": "https://mcp.github.example/mcp",
                    },
                },
                "subagents": [
                    {
                        "name": "debugger",
                        "description": "Debug runtime issues",
                        "system_prompt": "You debug runtime issues.",
                        "model": "inherit",
                    }
                ],
                "include_builtin_subagents": False,
                "unified_subagents": True,
                "workspace_backend_hint": "local",
                "enabled": True,
            },
        )
        assert put_response.status_code == 200
        assert put_response.json()["name"] == "custom"
        assert put_response.json()["system_prompt"] == "You are a custom profile."
        assert put_response.json()["builtin_toolsets"] == ["filesystem", "shell"]
        assert put_response.json()["toolsets"] == ["filesystem", "shell"]
        assert put_response.json()["need_user_approve_mcps"] == ["context7"]
        assert put_response.json()["enabled_mcps"] == ["context7", "github"]
        assert put_response.json()["disabled_mcps"] == ["github"]
        assert put_response.json()["mcp_servers"] == {
            "context7": {
                "transport": "streamable_http",
                "url": "https://mcp.context7.com/mcp",
                "headers": {},
                "description": "Library docs",
                "required": False,
            },
            "github": {
                "transport": "streamable_http",
                "url": "https://mcp.github.example/mcp",
                "headers": {},
                "description": "",
                "required": True,
            },
        }

        invalid_mcp_response = client.put(
            "/api/v1/profiles/invalid-mcp",
            headers=_auth_headers(),
            json={
                "model": "gateway@openai-responses:gpt-5.5",
                "mcp_servers": {"github": {"transport": "stdio", "url": "https://example.test/mcp"}},
            },
        )
        assert invalid_mcp_response.status_code == 422

        list_response = client.get("/api/v1/profiles", headers=_auth_headers())
        assert list_response.status_code == 200
        assert [item["name"] for item in list_response.json()] == ["custom"]

        get_response = client.get("/api/v1/profiles/custom", headers=_auth_headers())
        assert get_response.status_code == 200
        assert get_response.json()["unified_subagents"] is True
        assert get_response.json()["subagents"][0]["name"] == "debugger"
        assert get_response.json()["subagents"][0]["model"] == "inherit"
        assert get_response.json()["builtin_toolsets"] == ["filesystem", "shell"]
        assert get_response.json()["toolsets"] == ["filesystem", "shell"]
        assert get_response.json()["mcp_servers"]["context7"]["url"] == "https://mcp.context7.com/mcp"

        seed_response = client.post(
            "/api/v1/profiles/seed",
            headers=_auth_headers(),
            json={"prune_missing": False},
        )
        assert seed_response.status_code == 200
        assert seed_response.json()["seeded_names"] == ["seeded"]

        list_after_seed_response = client.get("/api/v1/profiles", headers=_auth_headers())
        assert list_after_seed_response.status_code == 200
        assert [item["name"] for item in list_after_seed_response.json()] == ["custom", "seeded"]

        seeded_get_response = client.get("/api/v1/profiles/seeded", headers=_auth_headers())
        assert seeded_get_response.status_code == 200
        assert seeded_get_response.json()["subagents"][0]["name"] == "explorer"
        assert seeded_get_response.json()["builtin_toolsets"] == ["core", "web"]
        assert seeded_get_response.json()["enabled_mcps"] == ["context7"]
        assert seeded_get_response.json()["mcp_servers"]["context7"]["required"] is False

        seed_file.write_text(
            """
profiles:
  - name: seeded
    model: gateway@openai-responses:gpt-5.5
    builtin_toolsets: [core, shell]
    enabled_mcps: []
    subagents:
      - name: debugger
        description: Debug runtime issues
        system_prompt: |
          You debug runtime issues.
        model: inherit
        model_settings_preset: inherit
        model_config_preset: inherit
""".strip(),
            encoding="utf-8",
        )
        update_seed_response = client.post(
            "/api/v1/profiles/seed",
            headers=_auth_headers(),
            json={"prune_missing": False},
        )
        assert update_seed_response.status_code == 200
        assert update_seed_response.json()["seeded_names"] == ["seeded"]

        updated_seeded_response = client.get("/api/v1/profiles/seeded", headers=_auth_headers())
        assert updated_seeded_response.status_code == 200
        assert updated_seeded_response.json()["model"] == "gateway@openai-responses:gpt-5.5"
        assert updated_seeded_response.json()["builtin_toolsets"] == ["core", "shell"]
        assert updated_seeded_response.json()["enabled_mcps"] == []
        assert [item["name"] for item in updated_seeded_response.json()["subagents"]] == ["debugger"]
        assert updated_seeded_response.json()["subagents"][0]["model"] == "inherit"
        assert updated_seeded_response.json()["subagents"][0]["model_settings_preset"] == "inherit"
        assert updated_seeded_response.json()["subagents"][0]["model_config_preset"] == "inherit"

        delete_response = client.delete("/api/v1/profiles/custom", headers=_auth_headers())
        assert delete_response.status_code == 204

        missing_response = client.get("/api/v1/profiles/custom", headers=_auth_headers())
        assert missing_response.status_code == 404
