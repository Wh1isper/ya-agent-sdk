"""Tests for yaacli.config module."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from yaacli.config import (
    DEFAULT_COMMANDS,
    CommandDefinition,
    ConfigManager,
    GeneralConfig,
    ToolsConfig,
    YaacliConfig,
)

# =============================================================================
# Config Model Tests
# =============================================================================


def test_default_config() -> None:
    """Test default configuration values."""
    config = YaacliConfig()

    # General - model defaults to empty string (not configured)
    assert config.general.model == ""
    assert config.general.is_configured is False
    assert config.is_configured is False
    assert config.general.model_settings is None
    assert config.general.agent_stream_resume_on_error is True
    assert config.general.agent_stream_resume_max_attempts == 3
    assert config.model_profiles == {}
    assert config.general.agent_stream_resume_prompt.startswith("The previous streaming model request failed")

    # Display and local retention
    assert config.display.code_theme == "dark"
    assert config.display.max_output_lines == 1000
    assert config.display.max_output_blocks == 1000
    assert config.display.max_output_bytes == 4 * 1024 * 1024
    assert config.display.max_stream_render_bytes == 512 * 1024
    assert config.display.max_prompt_history == 500
    assert config.media.max_pending_attachments == 8
    assert config.media.max_pending_attachment_bytes == 20 * 1024 * 1024

    # Tools and security
    assert config.tools.need_approval == []
    assert config.security.shell_review.enabled is False


def test_general_config_with_preset() -> None:
    """Test GeneralConfig with preset model_settings."""
    config = GeneralConfig(
        model="openai-chat:gpt-4o",
        model_settings="openai_high",
    )

    assert config.model == "openai-chat:gpt-4o"
    assert config.model_settings == "openai_high"


def test_general_config_with_dict_settings() -> None:
    """Test GeneralConfig with dict model_settings."""
    settings: dict[str, Any] = {
        "max_tokens": 8192,
        "anthropic_thinking": {"type": "enabled", "budget_tokens": 16000},
    }
    config = GeneralConfig(model_settings=settings)

    assert isinstance(config.model_settings, dict)
    assert config.model_settings["max_tokens"] == 8192


def test_general_config_accepts_former_loop_iteration_setting() -> None:
    """Test migration from the former /loop iteration config key."""
    config = GeneralConfig.model_validate({"max_loop_iterations": 7})
    assert config.max_goal_iterations == 7


def test_tools_config() -> None:
    """Test ToolsConfig (project-only config)."""
    config = ToolsConfig(need_approval=["shell_sandbox", "file_write"])
    assert config.need_approval == ["shell_sandbox", "file_write"]


# =============================================================================
# ConfigManager Tests
# =============================================================================


def test_load_defaults(config_manager: ConfigManager, clean_env: None) -> None:
    """Test loading with no config files."""
    config = config_manager.load()

    assert config.general.model == ""
    assert config.is_configured is False
    assert config.tools.need_approval == []
    assert config_manager.loaded_sources == []


def test_load_global_config(
    config_manager: ConfigManager,
    temp_config_dir: Path,
    clean_env: None,
) -> None:
    """Test loading global config (model + TUI settings)."""
    config_file = temp_config_dir / "config.toml"
    config_file.write_text("""
[general]
model = "openai-chat:gpt-4o"
model_settings = "openai_high"

[display]
code_theme = "light"
""")

    config = config_manager.load()

    assert config.general.model == "openai-chat:gpt-4o"
    assert config.general.model_settings == "openai_high"
    assert config.display.code_theme == "light"
    assert config.tools.need_approval == []
    assert config.security.shell_review.enabled is False


def test_load_global_model_profiles_config(
    config_manager: ConfigManager,
    temp_config_dir: Path,
    clean_env: None,
) -> None:
    """Test loading selectable model profiles from global config.toml."""
    config_file = temp_config_dir / "config.toml"
    config_file.write_text("""
[general]
model = "anthropic:claude-sonnet-4-5"
model_settings = "anthropic_adaptive_high"
model_cfg = "claude_200k"

[model_profiles.fast]
label = "Fast"
model = "openai-responses:gpt-5-mini"
model_settings = "openai_responses_low"
model_cfg = "gpt5_270k"
""")

    config = config_manager.load()

    assert config.general.model == "anthropic:claude-sonnet-4-5"
    assert "fast" in config.model_profiles
    fast = config.model_profiles["fast"]
    assert fast.label == "Fast"
    assert fast.model == "openai-responses:gpt-5-mini"
    assert fast.model_settings == "openai_responses_low"
    assert fast.model_cfg == "gpt5_270k"


def test_load_global_security_shell_review_config(
    config_manager: ConfigManager,
    temp_config_dir: Path,
    clean_env: None,
) -> None:
    """Test loading security.shell_review from global config.toml."""
    config_file = temp_config_dir / "config.toml"
    config_file.write_text("""
[general]
model = "openai-chat:gpt-4o"

[security.shell_review]
enabled = true
model = "gateway@openai-responses:gpt-5.4-mini"
model_settings = "openai_responses_low"
on_needs_approval = "defer"
risk_threshold = "extra_high"
""")

    config = config_manager.load()

    assert config.security.shell_review.enabled is True
    assert config.security.shell_review.model == "gateway@openai-responses:gpt-5.4-mini"
    assert config.security.shell_review.model_settings == "openai_responses_low"
    assert config.security.shell_review.on_needs_approval == "defer"
    assert config.security.shell_review.risk_threshold == "extra_high"


def test_load_project_tools_config(
    config_manager: ConfigManager,
    temp_project_dir: Path,
    clean_env: None,
) -> None:
    """Test loading project tools.toml (tools only)."""
    project_config_dir = temp_project_dir / ".yaacli"
    project_config_dir.mkdir()
    config_file = project_config_dir / "tools.toml"
    config_file.write_text("""
[tools]
need_approval = ["shell_sandbox", "file_write"]
""")

    config = config_manager.load()

    assert config.tools.need_approval == ["shell_sandbox", "file_write"]
    # Model should be empty (not configured)
    assert config.general.model == ""
    assert config.is_configured is False


def test_global_and_project_config(
    config_manager: ConfigManager,
    temp_config_dir: Path,
    temp_project_dir: Path,
    clean_env: None,
) -> None:
    """Test global (model+TUI) and project (tools) configs together."""
    global_config = temp_config_dir / "config.toml"
    global_config.write_text("""
[general]
model = "openai-chat:gpt-4o"

[display]
code_theme = "light"
""")

    project_config_dir = temp_project_dir / ".yaacli"
    project_config_dir.mkdir()
    project_config = project_config_dir / "tools.toml"
    project_config.write_text("""
[tools]
need_approval = ["dangerous_tool"]
""")

    config = config_manager.load()

    assert config.general.model == "openai-chat:gpt-4o"
    assert config.display.code_theme == "light"
    assert config.tools.need_approval == ["dangerous_tool"]


def test_global_tools_config(
    config_manager: ConfigManager,
    temp_config_dir: Path,
    clean_env: None,
) -> None:
    """Test global tools.toml loading."""
    global_tools = temp_config_dir / "tools.toml"
    global_tools.write_text("""
[tools]
need_approval = ["shell_sandbox", "file_write"]
""")

    config = config_manager.load()

    assert config.tools.need_approval == ["shell_sandbox", "file_write"]


def test_project_tools_overrides_global(
    config_manager: ConfigManager,
    temp_config_dir: Path,
    temp_project_dir: Path,
    clean_env: None,
) -> None:
    """Test project tools.toml overrides global tools.toml."""
    # Global tools
    global_tools = temp_config_dir / "tools.toml"
    global_tools.write_text("""
[tools]
need_approval = ["global_tool"]
""")

    # Project tools (should override)
    project_config_dir = temp_project_dir / ".yaacli"
    project_config_dir.mkdir()
    project_tools = project_config_dir / "tools.toml"
    project_tools.write_text("""
[tools]
need_approval = ["project_tool"]
""")

    config = config_manager.load()

    # Project tools takes priority
    assert config.tools.need_approval == ["project_tool"]


def test_load_global_oauth_refresh_config(
    config_manager: ConfigManager,
    temp_config_dir: Path,
    clean_env: None,
) -> None:
    """Test loading OAuth proactive refresh config."""
    config_file = temp_config_dir / "config.toml"
    config_file.write_text("""
[general]
model = "oauth@codex:gpt-5.5"

[oauth_refresh]
enabled = true
interval_seconds = 1200
failure_retry_seconds = 30
refresh_on_startup = false
""")

    config = config_manager.load()

    assert config.oauth_refresh.enabled is True
    assert config.oauth_refresh.interval_seconds == 1200
    assert config.oauth_refresh.failure_retry_seconds == 30
    assert config.oauth_refresh.refresh_on_startup is False


def test_oauth_refresh_config_requires_positive_intervals(
    config_manager: ConfigManager,
    temp_config_dir: Path,
    clean_env: None,
) -> None:
    """Test OAuth proactive refresh interval validation."""
    config_file = temp_config_dir / "config.toml"
    config_file.write_text("""
[oauth_refresh]
interval_seconds = 0
failure_retry_seconds = -1
""")

    with pytest.raises(ValidationError):
        config_manager.load()


def test_env_overrides_tui_only(
    config_manager: ConfigManager,
    temp_config_dir: Path,
    clean_env: None,
) -> None:
    """Test environment variables override TUI settings only."""
    global_config = temp_config_dir / "config.toml"
    global_config.write_text("""
[general]
model = "openai-chat:gpt-4o"

[display]
code_theme = "dark"
""")

    os.environ["YAACLI_CODE_THEME"] = "light"
    os.environ["YAACLI_AGENT_STREAM_RESUME_MAX_ATTEMPTS"] = "3"
    os.environ["YAACLI_OAUTH_REFRESH_INTERVAL_SECONDS"] = "900"
    os.environ["YAACLI_OAUTH_REFRESH_ON_STARTUP"] = "false"

    config = config_manager.load()

    assert config.display.code_theme == "light"
    assert config.general.model == "openai-chat:gpt-4o"
    assert config.general.agent_stream_resume_max_attempts == 3
    assert config.oauth_refresh.interval_seconds == 900
    assert config.oauth_refresh.refresh_on_startup is False


def test_project_config_overrides_global(
    config_manager: ConfigManager,
    temp_config_dir: Path,
    temp_project_dir: Path,
    clean_env: None,
) -> None:
    """Test that project config.toml overrides global config.toml entirely."""
    # Global config
    global_config = temp_config_dir / "config.toml"
    global_config.write_text("""
[general]
model = "openai-chat:gpt-4o"
max_requests = 500

[display]
code_theme = "dark"
""")

    # Project config (replaces global entirely)
    project_config_dir = temp_project_dir / ".yaacli"
    project_config_dir.mkdir()
    project_config = project_config_dir / "config.toml"
    project_config.write_text("""
[general]
model = "anthropic:claude-sonnet-4-5"
""")

    config = config_manager.load()

    # Project model takes over
    assert config.general.model == "anthropic:claude-sonnet-4-5"
    # max_requests defaults (not from global)
    assert config.general.max_requests == 1000
    # display defaults (not from global)
    assert config.display.code_theme == "dark"


def test_tools_toml_ignores_non_tools(
    config_manager: ConfigManager,
    temp_project_dir: Path,
    clean_env: None,
) -> None:
    """Test that tools.toml ignores non-tools sections."""
    project_config_dir = temp_project_dir / ".yaacli"
    project_config_dir.mkdir()
    config_file = project_config_dir / "tools.toml"
    config_file.write_text("""
[general]
model = "should-be-ignored"

[tools]
need_approval = ["test_tool"]
""")

    config = config_manager.load()

    # Model should be empty (tools.toml's general section ignored)
    assert config.general.model == ""
    assert config.tools.need_approval == ["test_tool"]


def test_reload(
    config_manager: ConfigManager,
    temp_config_dir: Path,
    clean_env: None,
) -> None:
    """Test config reload."""
    config1 = config_manager.load()
    assert config1.general.model == ""

    config_file = temp_config_dir / "config.toml"
    config_file.write_text('[general]\nmodel = "openai-chat:gpt-4o"')

    assert config_manager.config.general.model == ""

    config2 = config_manager.reload()
    assert config2.general.model == "openai-chat:gpt-4o"


def test_save_default_config(
    config_manager: ConfigManager,
    temp_config_dir: Path,
) -> None:
    """Test save_default_config."""
    config_file = temp_config_dir / "config.toml"
    assert not config_file.exists()

    result = config_manager.save_default_config()
    assert result == config_file
    assert config_file.exists()
    assert "[general]" in config_file.read_text()

    assert config_manager.save_default_config() is None
    assert config_manager.save_default_config(force=True) == config_file


def test_save_project_config(
    config_manager: ConfigManager,
    temp_project_dir: Path,
) -> None:
    """Test save_project_config."""
    config_file = temp_project_dir / ".yaacli" / "tools.toml"
    assert not config_file.exists()

    result = config_manager.save_project_config()
    assert result == config_file
    assert config_file.exists()
    content = config_file.read_text()
    assert "[tools]" in content
    assert "need_approval" in content


# =============================================================================
# MCP Config Tests
# =============================================================================


def test_load_mcp_config_none(
    config_manager: ConfigManager,
    clean_env: None,
) -> None:
    """Test load_mcp_config returns None when no mcp.json exists."""
    result = config_manager.load_mcp_config()
    assert result is None


def test_load_mcp_config_global(
    config_manager: ConfigManager,
    temp_config_dir: Path,
    clean_env: None,
) -> None:
    """Test loading MCP config from global directory."""
    mcp_file = temp_config_dir / "mcp.json"
    mcp_file.write_text("""
{
    "servers": {
        "github": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"]
        }
    }
}
""")

    result = config_manager.load_mcp_config()
    assert result is not None
    assert "github" in result.servers
    assert result.servers["github"].command == "npx"


def test_load_mcp_config_project_priority(
    config_manager: ConfigManager,
    temp_config_dir: Path,
    temp_project_dir: Path,
    clean_env: None,
) -> None:
    """Test project mcp.json takes priority over global."""
    global_mcp = temp_config_dir / "mcp.json"
    global_mcp.write_text('{"servers": {"global_server": {"transport": "stdio", "command": "global"}}}')

    project_config_dir = temp_project_dir / ".yaacli"
    project_config_dir.mkdir()
    project_mcp = project_config_dir / "mcp.json"
    project_mcp.write_text('{"servers": {"project_server": {"transport": "stdio", "command": "project"}}}')

    result = config_manager.load_mcp_config()
    assert result is not None
    assert "project_server" in result.servers
    assert "global_server" not in result.servers


def test_get_mcp_config_file(
    config_manager: ConfigManager,
    temp_config_dir: Path,
    temp_project_dir: Path,
) -> None:
    """Test get_mcp_config_file returns correct path."""
    assert config_manager.get_mcp_config_file() is None

    global_mcp = temp_config_dir / "mcp.json"
    global_mcp.write_text("{}")
    assert config_manager.get_mcp_config_file() == global_mcp

    project_config_dir = temp_project_dir / ".yaacli"
    project_config_dir.mkdir()
    project_mcp = project_config_dir / "mcp.json"
    project_mcp.write_text("{}")
    assert config_manager.get_mcp_config_file() == project_mcp


# =============================================================================
# Commands Tests
# =============================================================================


def test_default_commands() -> None:
    """Test that default commands include init."""
    assert "init" in DEFAULT_COMMANDS
    assert DEFAULT_COMMANDS["init"].mode == "act"
    assert DEFAULT_COMMANDS["init"].description == "Initialize AGENTS.md"


def test_get_commands_returns_defaults() -> None:
    """Test get_commands returns default commands."""
    config = YaacliConfig()
    commands = config.get_commands()

    assert "init" in commands


def test_get_commands_merges_user_commands() -> None:
    """Test that user commands are merged with defaults."""
    config = YaacliConfig(
        commands={
            "custom": CommandDefinition(
                prompt="Custom prompt",
                description="Custom command",
            )
        }
    )
    commands = config.get_commands()

    assert "init" in commands  # Default
    assert "custom" in commands  # User-defined
    assert commands["custom"].prompt == "Custom prompt"


def test_user_command_overrides_default() -> None:
    """Test that user commands can override defaults."""
    config = YaacliConfig(
        commands={
            "init": CommandDefinition(
                prompt="Custom init prompt",
                description="Custom init",
            )
        }
    )
    commands = config.get_commands()

    assert commands["init"].prompt == "Custom init prompt"
    assert commands["init"].description == "Custom init"


def test_load_commands_from_config_file(
    temp_config_dir: Path,
    temp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test loading commands from config.toml."""
    monkeypatch.chdir(temp_project_dir)

    global_config = temp_config_dir / "config.toml"
    global_config.write_text("""
[general]
model = "anthropic:claude-sonnet-4-5"

[commands.commit]
description = "Commit changes"
mode = "act"
prompt = "Please commit"

[commands.review]
description = "Review code"
prompt = "Please review"
""")

    config_manager = ConfigManager(config_dir=temp_config_dir)
    config = config_manager.load()

    commands = config.get_commands()
    assert "init" in commands  # Default
    assert "commit" in commands  # From config
    assert "review" in commands  # From config
    assert commands["commit"].mode == "act"
    assert commands["review"].prompt == "Please review"


# =============================================================================
# Gitignore Tests (file tree context filtering)
# =============================================================================


def test_ensure_config_dir_creates_gitignore(tmp_path: Path) -> None:
    """ensure_config_dir creates .gitignore to exclude ephemeral dirs from file tree."""
    config_dir = tmp_path / "config"
    cm = ConfigManager(config_dir=config_dir)
    cm.ensure_config_dir()

    gitignore = config_dir / ".gitignore"
    assert gitignore.exists()
    content = gitignore.read_text()
    assert "sessions/" in content
    assert "message_history/" in content
    assert "worktrees/" in content
    assert "state.json" in content


def test_ensure_config_dir_gitignore_idempotent(tmp_path: Path) -> None:
    """Running ensure_config_dir multiple times does not duplicate entries."""
    config_dir = tmp_path / "config"
    cm = ConfigManager(config_dir=config_dir)
    cm.ensure_config_dir()
    cm.ensure_config_dir()
    cm.ensure_config_dir()

    gitignore = config_dir / ".gitignore"
    lines = [line for line in gitignore.read_text().splitlines() if line.strip()]
    assert len(lines) == len(set(lines)), f"Duplicate entries found: {lines}"


def test_ensure_config_dir_gitignore_appends_missing(tmp_path: Path) -> None:
    """If .gitignore exists but is incomplete, missing entries are appended."""
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    gitignore = config_dir / ".gitignore"
    gitignore.write_text("sessions/\ncustom_ignore/\n")

    cm = ConfigManager(config_dir=config_dir)
    cm.ensure_config_dir()

    content = gitignore.read_text()
    assert "sessions/" in content
    assert "message_history/" in content
    assert "worktrees/" in content
    assert "state.json" in content
    assert "custom_ignore/" in content  # User entries preserved
