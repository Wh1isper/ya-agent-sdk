from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine
from ya_claw.config import ClawSettings
from ya_claw.db.engine import create_engine, create_session_factory
from ya_claw.execution.profile import (
    ProfileResolver,
    capture_execution_profile_descriptor,
    resolved_profile_from_descriptor,
    resolved_profile_from_subagent_plan,
)
from ya_claw.execution.subagents import resolve_claw_subagent_plan
from ya_claw.orm.tables import ProfileRecord


@pytest.fixture
async def db_engine(
    tmp_path: Path,
    initialize_sqlite_database: Callable[[str], None],
) -> AsyncEngine:
    database_url = f"sqlite+aiosqlite:///{(tmp_path / 'profile.sqlite3').resolve()}"
    initialize_sqlite_database(database_url)
    engine = create_engine(database_url)
    try:
        yield engine
    finally:
        await engine.dispose()


def _settings(tmp_path: Path, seed_file: Path) -> ClawSettings:
    return ClawSettings(
        api_token="test-token",  # noqa: S106
        data_dir=tmp_path / "runtime-data",
        workspace_dir=tmp_path / "workspace",
        profile_seed_file=seed_file,
    )


def _native_seed(*, instructions: str = "Execute carefully.") -> str:
    return f"""
version: 2
profiles:
  - schema_version: 2
    name: default
    agent:
      model: test
      name: default
      instructions: {instructions!r}
      model_settings:
        temperature: 0
      capabilities:
        - FilesystemCapability
        - ShellCapability
    host:
      model_config_preset: gpt5_270k
      model_config_override:
        security:
          shell_review:
            enabled: true
            model: test
            on_needs_approval: deny
            risk_threshold: extra_high
      tool_groups: [session, workflow]
      need_user_approve_tools: [shell_exec]
      need_user_approve_mcps: [context7]
      enabled_mcps: [context7]
      mcp_servers:
        context7:
          transport: streamable_http
          url: https://mcp.context7.com/mcp
          required: false
      workspace_backend_hint: docker
    subagents:
      - schema_version: 1
        route: explorer
        execution_modes: [foreground, background]
        durability: restart
        agent:
          model: test
          name: explorer
          description: Explore the workspace
          instructions: Return evidence.
          metadata:
            claw:
              model_config_preset: gpt5_270k
              tool_groups: [session]
              need_user_approve_tools: [shell_exec]
              need_user_approve_mcps: [context7]
              enabled_mcps: [context7]
              mcp_servers:
                context7:
                  transport: streamable_http
                  url: https://mcp.context7.com/mcp
                  required: false
              workspace_backend_hint: docker
          capabilities:
            - FilesystemCapability
    enabled: true
    source_type: seed
    source_version: '2'
""".strip()


async def test_profile_resolver_seeds_native_profile_documents(
    tmp_path: Path,
    db_engine: AsyncEngine,
) -> None:
    seed_file = tmp_path / "profiles.yaml"
    seed_file.write_text(_native_seed(), encoding="utf-8")
    session_factory = create_session_factory(db_engine)
    resolver = ProfileResolver(
        settings=_settings(tmp_path, seed_file),
        session_factory=session_factory,
    )

    assert await resolver.seed_profiles() == ["default"]
    profile = await resolver.resolve("default")

    assert profile.agent_spec.model == "test"
    assert profile.agent_spec.instructions == "Execute carefully."
    assert [item.name for item in profile.agent_spec.capabilities] == [
        "FilesystemCapability",
        "ShellCapability",
    ]
    assert profile.host_tool_groups == ("session", "workflow")
    assert profile.approval_tools == frozenset({"shell_exec"})
    assert profile.approval_mcps == frozenset({"context7"})
    assert profile.workspace_backend_hint == "docker"
    assert profile.shell_review is not None
    assert profile.shell_review.enabled is True
    assert [spec.route for spec in profile.subagent_specs] == ["explorer"]

    plan = resolve_claw_subagent_plan(profile.subagent_specs[0])
    restored_child = resolved_profile_from_subagent_plan(plan)
    assert restored_child.host_tool_groups == ("session",)
    assert restored_child.approval_tools == frozenset({"shell_exec"})
    assert restored_child.mcp_servers["context7"]["required"] is False

    async with session_factory() as db_session:
        record = await db_session.get(ProfileRecord, "default")
        assert isinstance(record, ProfileRecord)
        assert record.agent_spec["model"] == "test"
        assert record.host_config["tool_groups"] == ["session", "workflow"]
        assert record.subagent_specs[0]["route"] == "explorer"
        assert record.source_checksum is not None


async def test_execution_profile_descriptor_is_exact_after_catalog_mutation(
    tmp_path: Path,
    db_engine: AsyncEngine,
) -> None:
    seed_file = tmp_path / "profiles.yaml"
    seed_file.write_text(_native_seed(instructions="Accepted instructions."), encoding="utf-8")
    session_factory = create_session_factory(db_engine)
    resolver = ProfileResolver(
        settings=_settings(tmp_path, seed_file),
        session_factory=session_factory,
    )
    await resolver.seed_profiles()
    async with session_factory() as db_session:
        descriptor = await capture_execution_profile_descriptor(db_session, "default")
        record = await db_session.get(ProfileRecord, "default")
        assert isinstance(record, ProfileRecord)
        record.agent_spec = {**record.agent_spec, "instructions": "Mutated instructions."}
        record.enabled = False
        await db_session.commit()

    restored = resolved_profile_from_descriptor(descriptor.model_dump(mode="json"))
    assert restored.agent_spec.instructions == "Accepted instructions."
    assert restored.metadata["profile_descriptor_id"] == descriptor.descriptor_id
    with pytest.raises(ValueError, match="does not match its fingerprint"):
        resolved_profile_from_descriptor(
            descriptor.model_copy(
                update={"agent_spec": {**descriptor.agent_spec, "instructions": "tampered"}}
            ).model_dump(mode="json")
        )


async def test_profile_resolver_replaces_seeded_native_definition(
    tmp_path: Path,
    db_engine: AsyncEngine,
) -> None:
    seed_file = tmp_path / "profiles.yaml"
    seed_file.write_text(_native_seed(instructions="Old instructions."), encoding="utf-8")
    session_factory = create_session_factory(db_engine)
    resolver = ProfileResolver(
        settings=_settings(tmp_path, seed_file),
        session_factory=session_factory,
    )
    await resolver.seed_profiles()

    seed_file.write_text(_native_seed(instructions="New instructions."), encoding="utf-8")
    await resolver.seed_profiles()

    profile = await resolver.resolve("default")
    assert profile.agent_spec.instructions == "New instructions."
    assert profile.metadata["source_version"] == "2"


async def test_profile_resolver_rejects_legacy_seed_schema(
    tmp_path: Path,
    db_engine: AsyncEngine,
) -> None:
    seed_file = tmp_path / "profiles.yaml"
    seed_file.write_text(
        "version: 1\nprofiles:\n  - name: legacy\n    model: test\n",
        encoding="utf-8",
    )
    resolver = ProfileResolver(
        settings=_settings(tmp_path, seed_file),
        session_factory=create_session_factory(db_engine),
    )

    with pytest.raises(ValueError, match="legacy profile documents are not accepted"):
        await resolver.seed_profiles()


async def test_profile_resolver_rejects_stdio_mcp(
    tmp_path: Path,
    db_engine: AsyncEngine,
) -> None:
    seed_file = tmp_path / "profiles.yaml"
    seed_file.write_text(
        """
version: 2
profiles:
  - schema_version: 2
    name: invalid
    agent: {model: test, name: invalid}
    host:
      mcp_servers:
        bad: {transport: stdio, command: npx}
""".strip(),
        encoding="utf-8",
    )
    resolver = ProfileResolver(
        settings=_settings(tmp_path, seed_file),
        session_factory=create_session_factory(db_engine),
    )

    with pytest.raises(ValueError, match="unsupported transport"):
        await resolver.seed_profiles()


async def test_missing_default_profile_raises_clear_error(
    tmp_path: Path,
    db_engine: AsyncEngine,
) -> None:
    settings = ClawSettings(
        api_token="test-token",  # noqa: S106
        data_dir=tmp_path / "runtime-data",
        workspace_dir=tmp_path / "workspace",
    )
    resolver = ProfileResolver(
        settings=settings,
        session_factory=create_session_factory(db_engine),
    )

    with pytest.raises(ValueError, match="Execution profile 'default' could not be resolved"):
        await resolver.resolve(None)
