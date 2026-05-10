from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine
from ya_claw.config import ClawSettings
from ya_claw.db.engine import create_engine, create_session_factory
from ya_claw.execution.profile import ProfileResolver
from ya_claw.orm.base import Base
from ya_claw.orm.tables import ProfileRecord


@pytest.fixture
async def db_engine(tmp_path: Path) -> AsyncEngine:
    engine = create_engine(f"sqlite+aiosqlite:///{(tmp_path / 'profile.sqlite3').resolve()}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    try:
        yield engine
    finally:
        await engine.dispose()


async def test_profile_resolver_seeds_profiles_from_yaml(tmp_path: Path, db_engine: AsyncEngine) -> None:
    seed_file = tmp_path / "profiles.yaml"
    seed_file.write_text(
        """
profiles:
  - name: default
    model: gateway@openai-responses:gpt-5.5
    model_settings_preset: openai_responses_high
    model_config_preset: gpt5_270k
    security:
      shell_review:
        enabled: true
        model: gateway@openai-responses:gpt-5.5-mini
        model_settings: openai_responses_low
        on_needs_approval: deny
        risk_threshold: extra_high
        unattended_risk_threshold: high
    system_prompt: |
      You are the profile-scoped execution agent.
    builtin_toolsets: [filesystem, shell]
    need_user_approve_mcps: [context7]
    enabled_mcps: [context7, github]
    disabled_mcps: [github]
    mcp_servers:
      context7:
        transport: streamable_http
        url: https://mcp.context7.com/mcp
        description: Library docs
        required: false
      github:
        transport: streamable_http
        url: https://mcp.github.example/mcp
    unified_subagents: true
    workspace_backend_hint: docker
    subagents:
      - name: explorer
        description: Explore the codebase
        system_prompt: |
          You explore the codebase.
      - name: searcher
        description: Search the web
        system_prompt: |
          You search the web.
        model: gateway@openai-responses:gpt-5.5-mini
        model_settings_preset: openai_responses_high
        model_config_preset: gpt5_270k
      - name: executor
        description: Execute tasks
        system_prompt: |
          You execute tasks.
        model: inherit
        model_settings_preset: inherit
        model_config_preset: inherit
""".strip(),
        encoding="utf-8",
    )
    settings = ClawSettings(
        api_token="test-token",  # noqa: S106
        data_dir=tmp_path / "runtime-data",
        workspace_dir=tmp_path / "workspace",
        profile_seed_file=seed_file,
    )
    session_factory = create_session_factory(db_engine)
    resolver = ProfileResolver(settings=settings, session_factory=session_factory)

    seeded_names = await resolver.seed_profiles()
    resolved_profile = await resolver.resolve("default")

    assert seeded_names == ["default"]
    assert resolved_profile.model == "gateway@openai-responses:gpt-5.5"
    assert resolved_profile.shell_review is not None
    assert resolved_profile.shell_review.enabled is True
    assert resolved_profile.shell_review.model == "gateway@openai-responses:gpt-5.5-mini"
    assert isinstance(resolved_profile.shell_review.model_settings, dict)
    assert resolved_profile.shell_review.model_settings["openai_reasoning_effort"] == "low"
    assert resolved_profile.shell_review.risk_threshold == "extra_high"
    assert resolved_profile.shell_review.unattended_risk_threshold == "high"
    assert resolved_profile.system_prompt == "You are the profile-scoped execution agent."
    assert resolved_profile.builtin_toolsets == ["filesystem", "shell"]
    assert resolved_profile.need_user_approve_mcps == ["context7"]
    assert resolved_profile.enabled_mcps == ["context7", "github"]
    assert resolved_profile.disabled_mcps == ["github"]
    assert resolved_profile.mcp_servers == {
        "context7": {
            "transport": "streamable_http",
            "args": [],
            "env": {},
            "url": "https://mcp.context7.com/mcp",
            "headers": {},
            "description": "Library docs",
            "required": False,
        },
        "github": {
            "transport": "streamable_http",
            "args": [],
            "env": {},
            "url": "https://mcp.github.example/mcp",
            "headers": {},
            "description": "",
            "required": True,
        },
    }
    assert resolved_profile.unified_subagents is True
    assert resolved_profile.workspace_backend_hint == "docker"
    assert [config.name for config in resolved_profile.subagent_configs] == ["explorer", "searcher", "executor"]
    assert resolved_profile.subagent_configs[0].tools is None
    assert resolved_profile.subagent_configs[1].model == "gateway@openai-responses:gpt-5.5-mini"
    assert isinstance(resolved_profile.subagent_configs[1].model_settings, dict)
    assert isinstance(resolved_profile.subagent_configs[1].model_cfg, dict)
    assert resolved_profile.subagent_configs[2].model == "inherit"
    assert resolved_profile.subagent_configs[2].model_settings is None
    assert resolved_profile.subagent_configs[2].model_cfg is None

    async with session_factory() as db_session:
        record = await db_session.get(ProfileRecord, "default")
        assert isinstance(record, ProfileRecord)
        assert record.source_checksum is not None
        assert record.model_config_override is not None
        assert record.model_config_override["security"]["shell_review"]["model_settings"] == "openai_responses_low"
        assert record.model_config_override["security"]["shell_review"]["risk_threshold"] == "extra_high"
        assert record.model_config_override["security"]["shell_review"]["unattended_risk_threshold"] == "high"
        assert record.builtin_toolsets == ["filesystem", "shell"]
        assert record.need_user_approve_mcps == ["context7"]
        assert record.enabled_mcps == ["context7", "github"]
        assert record.disabled_mcps == ["github"]
        assert record.mcp_servers["context7"]["transport"] == "streamable_http"
        assert record.unified_subagents is True
        assert [item["name"] for item in record.subagents] == ["explorer", "searcher", "executor"]


async def test_profile_resolver_updates_existing_seeded_profile_and_subagents(
    tmp_path: Path,
    db_engine: AsyncEngine,
) -> None:
    seed_file = tmp_path / "profiles.yaml"
    seed_file.write_text(
        """
profiles:
  - name: default
    model: gateway@openai-responses:gpt-5.5
    system_prompt: old prompt
    builtin_toolsets: [core]
    subagents:
      - name: explorer
        description: Explore the codebase
        system_prompt: old explorer prompt
""".strip(),
        encoding="utf-8",
    )
    settings = ClawSettings(
        api_token="test-token",  # noqa: S106
        data_dir=tmp_path / "runtime-data",
        workspace_dir=tmp_path / "workspace",
        profile_seed_file=seed_file,
    )
    session_factory = create_session_factory(db_engine)
    resolver = ProfileResolver(settings=settings, session_factory=session_factory)

    assert await resolver.seed_profiles() == ["default"]

    seed_file.write_text(
        """
profiles:
  - name: default
    model: gateway@openai-responses:gpt-5.5
    system_prompt: updated prompt
    builtin_toolsets: [core, web]
    subagents:
      - name: debugger
        description: Debug runtime issues
        system_prompt: updated debugger prompt
        model: inherit
        model_settings_preset: inherit
        model_config_preset: inherit
""".strip(),
        encoding="utf-8",
    )

    assert await resolver.seed_profiles() == ["default"]
    resolved_profile = await resolver.resolve("default")

    assert resolved_profile.model == "gateway@openai-responses:gpt-5.5"
    assert resolved_profile.system_prompt == "updated prompt"
    assert resolved_profile.builtin_toolsets == ["core", "web"]
    assert [config.name for config in resolved_profile.subagent_configs] == ["debugger"]
    assert resolved_profile.subagent_configs[0].system_prompt == "updated debugger prompt"
    assert resolved_profile.subagent_configs[0].model == "inherit"
    assert resolved_profile.subagent_configs[0].model_settings is None
    assert resolved_profile.subagent_configs[0].model_cfg is None

    async with session_factory() as db_session:
        record = await db_session.get(ProfileRecord, "default")
        assert isinstance(record, ProfileRecord)
        assert record.model == "gateway@openai-responses:gpt-5.5"
        assert record.system_prompt == "updated prompt"
        assert [item["name"] for item in record.subagents] == ["debugger"]
        assert record.subagents[0]["system_prompt"] == "updated debugger prompt"
        assert record.source_type == "seed"
        assert record.source_checksum is not None


async def test_profile_resolver_rejects_stdio_mcp_from_yaml(
    tmp_path: Path,
    db_engine: AsyncEngine,
) -> None:
    seed_file = tmp_path / "profiles.yaml"
    seed_file.write_text(
        """
profiles:
  - name: invalid
    model: gateway@openai-responses:gpt-5.5
    mcp_servers:
      github:
        transport: stdio
        command: npx
""".strip(),
        encoding="utf-8",
    )
    settings = ClawSettings(
        api_token="test-token",  # noqa: S106
        data_dir=tmp_path / "runtime-data",
        workspace_dir=tmp_path / "workspace",
        profile_seed_file=seed_file,
    )
    session_factory = create_session_factory(db_engine)
    resolver = ProfileResolver(settings=settings, session_factory=session_factory)

    with pytest.raises(ValueError, match="unsupported transport"):
        await resolver.seed_profiles()


async def test_profile_resolver_accepts_legacy_toolsets_alias_from_yaml(
    tmp_path: Path,
    db_engine: AsyncEngine,
) -> None:
    seed_file = tmp_path / "profiles.yaml"
    seed_file.write_text(
        """
profiles:
  - name: legacy
    model: gateway@openai-responses:gpt-5.5
    toolsets: [core, web]
""".strip(),
        encoding="utf-8",
    )
    settings = ClawSettings(
        api_token="test-token",  # noqa: S106
        data_dir=tmp_path / "runtime-data",
        workspace_dir=tmp_path / "workspace",
        profile_seed_file=seed_file,
    )
    session_factory = create_session_factory(db_engine)
    resolver = ProfileResolver(settings=settings, session_factory=session_factory)

    await resolver.seed_profiles()
    resolved_profile = await resolver.resolve("legacy")

    assert resolved_profile.builtin_toolsets == ["core", "web"]


async def test_profile_resolver_preserves_explicit_empty_builtin_toolsets(
    tmp_path: Path,
    db_engine: AsyncEngine,
) -> None:
    seed_file = tmp_path / "profiles.yaml"
    seed_file.write_text(
        """
profiles:
  - name: empty-tools
    model: gateway@openai-responses:gpt-5.5
    builtin_toolsets: []
""".strip(),
        encoding="utf-8",
    )
    settings = ClawSettings(
        api_token="test-token",  # noqa: S106
        data_dir=tmp_path / "runtime-data",
        workspace_dir=tmp_path / "workspace",
        profile_seed_file=seed_file,
    )
    session_factory = create_session_factory(db_engine)
    resolver = ProfileResolver(settings=settings, session_factory=session_factory)

    await resolver.seed_profiles()
    resolved_profile = await resolver.resolve("empty-tools")

    assert resolved_profile.builtin_toolsets == []


async def test_missing_default_profile_raises_clear_error(
    tmp_path: Path,
    db_engine: AsyncEngine,
) -> None:
    settings = ClawSettings(
        api_token="test-token",  # noqa: S106
        data_dir=tmp_path / "runtime-data",
        workspace_dir=tmp_path / "workspace",
    )
    session_factory = create_session_factory(db_engine)
    resolver = ProfileResolver(settings=settings, session_factory=session_factory)

    try:
        await resolver.resolve(None)
    except ValueError as exc:
        assert str(exc) == "Execution profile 'default' could not be resolved."
    else:
        raise AssertionError("Expected missing default profile to raise ValueError")
