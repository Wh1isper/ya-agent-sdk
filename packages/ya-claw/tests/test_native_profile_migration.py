from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from pydantic_ai import AgentSpec
from ya_agent_sdk.subagents import SubagentSpec
from ya_claw.profile_spec import ClawProfileHostConfig


def test_native_profile_migration_converts_legacy_rows_and_drops_old_columns(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database_path = tmp_path / "legacy-profiles.sqlite3"
    monkeypatch.setenv(
        "YA_CLAW_DATABASE_URL",
        f"sqlite+aiosqlite:///{database_path}",
    )
    config = Config(str(Path(__file__).parents[1] / "ya_claw" / "alembic.ini"))
    command.upgrade(config, "20260818_000015")

    engine = sa.create_engine(f"sqlite:///{database_path}")
    legacy_metadata = sa.MetaData()
    legacy_profiles = sa.Table("profiles", legacy_metadata, autoload_with=engine)
    with engine.begin() as connection:
        connection.execute(
            legacy_profiles.insert().values(
                name="legacy",
                model="test",
                model_settings_preset=None,
                model_settings_override={"temperature": 0.2},
                model_config_preset=None,
                model_config_override={"request_limit": 7},
                system_prompt="Legacy instructions",
                builtin_toolsets=["filesystem", "session", "background"],
                subagents=[
                    {
                        "name": "worker",
                        "description": "Legacy worker",
                        "model": "inherit",
                        "system_prompt": "Inspect files",
                        "tools": ["view"],
                        "optional_tools": [],
                    }
                ],
                include_builtin_subagents=False,
                unified_subagents=True,
                need_user_approve_tools=["shell_exec"],
                need_user_approve_mcps=["remote"],
                enabled_mcps=["remote"],
                disabled_mcps=[],
                mcp_servers={},
                workspace_backend_hint="docker",
                enabled=True,
                source_type="user",
                source_version="1",
            )
        )

    command.upgrade(config, "head")

    inspector = sa.inspect(engine)
    column_names = {column["name"] for column in inspector.get_columns("profiles")}
    assert {"agent_spec", "host_config", "subagent_specs"} <= column_names
    assert {
        "model",
        "system_prompt",
        "builtin_toolsets",
        "subagents",
        "include_builtin_subagents",
        "unified_subagents",
    }.isdisjoint(column_names)

    native_metadata = sa.MetaData()
    native_profiles = sa.Table("profiles", native_metadata, autoload_with=engine)
    with engine.connect() as connection:
        row = connection.execute(sa.select(native_profiles).where(native_profiles.c.name == "legacy")).mappings().one()
    engine.dispose()

    agent = AgentSpec.model_validate(row["agent_spec"])
    host = ClawProfileHostConfig.model_validate(row["host_config"])
    children = tuple(SubagentSpec.model_validate(item) for item in row["subagent_specs"])
    assert agent.model == "test"
    assert agent.instructions == "Legacy instructions"
    assert agent.model_settings == {"temperature": 0.2}
    assert [capability.name for capability in agent.capabilities] == ["FilesystemCapability"]
    assert host.model_config_override == {"request_limit": 7}
    assert host.tool_groups == ("session",)
    assert host.need_user_approve_tools == ("shell_exec",)
    assert host.need_user_approve_mcps == ("remote",)
    assert host.workspace_backend_hint == "docker"
    assert len(children) == 1
    assert children[0].route == "worker"
    assert children[0].agent.model == "test"
    assert children[0].agent.instructions == "Inspect files"
    assert children[0].durability.value == "restart"


def test_native_profile_migration_rejects_implicit_builtin_children() -> None:
    migration = importlib.import_module("ya_claw.alembic.versions.20260818_000016_native_profiles")

    with pytest.raises(RuntimeError, match=r"materialize.*explicit subagents"):
        migration._migrate_profile({
            "name": "legacy-builtins",
            "model": "test",
            "include_builtin_subagents": True,
            "unified_subagents": True,
            "builtin_toolsets": ["core"],
            "subagents": [],
        })


def test_native_profile_migration_rejects_generated_subagent_tools() -> None:
    migration = importlib.import_module("ya_claw.alembic.versions.20260818_000016_native_profiles")

    with pytest.raises(RuntimeError, match=r"unified_subagents=true"):
        migration._migrate_profile({
            "name": "legacy-generated-tools",
            "model": "test",
            "include_builtin_subagents": False,
            "unified_subagents": False,
            "builtin_toolsets": ["core"],
            "subagents": [],
        })
