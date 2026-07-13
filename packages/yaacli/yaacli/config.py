"""Configuration management for yaacli.

Configuration files are loaded with project-level priority (no merging):

1. **config.toml** (model + TUI settings + security runtime settings):
   - Global: ~/.yaacli/config.toml
   - Project: .yaacli/config.toml (overrides global entirely)
   - Contains: model, model_settings, display, steering, session, subagents, env, security.shell_review

2. **tools.toml** (tool permissions and project tool overrides):
   - Global: ~/.yaacli/tools.toml
   - Project: .yaacli/tools.toml (overrides global entirely)
   - Contains: need_approval list and optional tool overrides

3. **mcp.json** (MCP server configurations):
   - Global: ~/.yaacli/mcp.json
   - Project: .yaacli/mcp.json (overrides global entirely)

4. **Environment variables** (YAACLI_*):
   - TUI configuration overrides only (merged on top of config.toml)
   - Does not affect model settings
"""

from __future__ import annotations

import tomllib
from importlib import resources
from pathlib import Path
from typing import Any, Literal, Self, TypedDict

from pydantic import BaseModel, Field, PositiveInt, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from ya_agent_sdk.mcp import MCPConfig, MCPServerConfig, load_mcp_config_file

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent

__all__ = [
    "CommandDefinition",
    "ConfigManager",
    "EnvSettings",
    "GeneralConfig",
    "MCPConfig",
    "MCPServerConfig",
    "ModelProfileConfig",
    "SessionConfig",
    "SubagentOverride",
    "SubagentsConfig",
    "ToolsConfig",
    "YaacliConfig",
    "get_config_manager",
    "load_config",
]

# =============================================================================
# Configuration Models
# =============================================================================


class GeneralConfig(BaseModel):
    """General agent configuration (global only)."""

    model: str = ""
    """Default model for main agent. Format: 'provider:model_name'. Empty means not configured."""

    model_settings: str | dict[str, Any] | None = None
    """Model settings: preset name (e.g., 'anthropic_high') or dict of actual values."""

    model_cfg: str | dict[str, Any] | None = None
    """Model config for context management: preset name (e.g., 'claude_200k', 'gpt5_1m', 'gemini_1m') or dict."""

    max_requests: int = 1000
    """Maximum requests per session."""

    agent_stream_resume_on_error: bool = True
    """Resume failed streaming attempts from recovered message history."""

    agent_stream_resume_max_attempts: int = 3
    """Maximum total streaming attempts when resume is enabled."""

    agent_stream_resume_prompt: str = (
        "The previous streaming model request failed before the agent finished. "
        "Continue the task from the available conversation history. Avoid repeating completed work."
    )
    """Prompt sent when resuming a failed stream attempt."""

    max_goal_iterations: int = 10
    """Maximum iterations for /goal command."""

    system_prompt_file: str = ""
    """Path to custom system prompt file. Empty uses built-in default."""

    @model_validator(mode="before")
    @classmethod
    def migrate_loop_config(cls, data: object) -> object:
        """Accept the former /loop iteration setting for existing configs."""
        if isinstance(data, dict) and "max_goal_iterations" not in data and "max_loop_iterations" in data:
            data = dict(data)
            data["max_goal_iterations"] = data["max_loop_iterations"]
        return data

    @property
    def is_configured(self) -> bool:
        """Check if model is configured."""
        return bool(self.model)


class ModelProfileConfig(BaseModel):
    """Selectable model profile configuration."""

    label: str | None = None
    """Human-friendly label shown in the model selector."""

    model: str
    """Model for main agent. Format: 'provider:model_name'."""

    model_settings: str | dict[str, Any] | None = None
    """Model settings: preset name or dict of actual values."""

    model_cfg: str | dict[str, Any] | None = None
    """Model config for context management: preset name or dict."""


class DisplayConfig(BaseModel):
    """Display and rendering configuration."""

    code_theme: Literal["dark", "light"] = "dark"
    """Code highlighting theme."""

    max_tool_result_lines: int = 5
    """Maximum lines to show for tool results."""

    max_arg_length: int = 100
    """Maximum length for tool argument display."""

    max_output_lines: PositiveInt = 1000
    """Maximum rendered lines retained in the transcript."""

    max_output_blocks: PositiveInt = 1000
    """Maximum rendered blocks retained in the transcript."""

    max_output_bytes: PositiveInt = 4 * 1024 * 1024
    """Maximum UTF-8 bytes retained in the rendered transcript."""

    max_stream_render_bytes: PositiveInt = 512 * 1024
    """Maximum raw streamed UTF-8 bytes retained and rendered by the UI."""

    max_prompt_history: PositiveInt = 500
    """Maximum submitted prompts retained for input history navigation."""

    show_token_usage: bool = True
    """Show token usage in status bar."""

    show_elapsed_time: bool = True
    """Show elapsed time."""


class ShellReviewConfig(BaseModel):
    """Shell command safety review configuration."""

    enabled: bool = False
    model: str | None = None
    model_settings: str | dict[str, Any] | None = None
    on_needs_approval: str = "defer"
    risk_threshold: str = "high"

    @model_validator(mode="after")
    def validate_model_when_enabled(self) -> Self:
        if self.enabled and (self.model is None or self.model.strip() == ""):
            msg = "security.shell_review.model is required when shell review is enabled."
            raise ValueError(msg)
        return self


class ToolsConfig(BaseModel):
    """Tool permission configuration."""

    need_approval: list[str] = Field(default_factory=list)
    """Tools requiring user approval before execution."""

    need_approval_mcps: list[str] = Field(default_factory=list)
    """MCP servers requiring user approval for all tools."""


class SecurityConfig(BaseModel):
    """Security runtime configuration."""

    shell_review: ShellReviewConfig = Field(default_factory=ShellReviewConfig)
    """Shell command safety review configuration."""


class SubagentOverride(BaseModel):
    """Override settings for a specific subagent."""

    model: str | None = None
    """Override model for this subagent."""

    model_settings: str | dict[str, Any] | None = None
    """Override model settings: preset name or dict of actual values."""


class SubagentsConfig(BaseModel):
    """Subagent configuration.

    Subagents are loaded from ~/.yaacli/subagents/
    which is initialized by `yaacli setup`.
    """

    disabled: list[str] = Field(default_factory=list)
    """Subagents to disable (by name)."""

    overrides: dict[str, SubagentOverride] = Field(default_factory=dict)
    """Override settings for specific subagents."""


class CommandDefinition(BaseModel):
    """Definition for a custom slash command.

    Custom commands trigger a predefined prompt when invoked via /name.
    """

    prompt: str
    """The prompt text to send to the agent."""

    mode: Literal["act", "plan"] | None = None
    """Optional mode to switch to before executing (act or plan)."""

    description: str = ""
    """Description shown in /help output."""


# Default commands provided out of the box (minimal set)
# Additional commands like /commit, /review can be added in config.toml
DEFAULT_COMMANDS: dict[str, CommandDefinition] = {
    "init": CommandDefinition(
        prompt="Please initialize the project's AGENTS.md file.",
        mode="act",
        description="Initialize AGENTS.md",
    ),
}


class S3Config(BaseModel):
    """S3 configuration for media upload."""

    enabled: bool = False
    """Enable S3 media upload."""

    bucket: str = ""
    """S3 bucket name."""

    region: str = "us-east-1"
    """AWS region."""

    access_key_id: str | None = None
    """AWS access key ID. None uses default credential chain."""

    secret_access_key: str | None = None
    """AWS secret access key. None uses default credential chain."""

    endpoint_url: str | None = None
    """Custom S3 endpoint URL for S3-compatible services (MinIO, R2, Ceph, etc.)."""

    prefix: str = ""
    """Object key prefix for uploaded files. e.g., 'uploads/' or 'uploads'"""

    url_mode: Literal["cdn", "presign"] = "presign"
    """URL generation mode: 'cdn' for CDN mapping, 'presign' for presigned URLs."""

    cdn_base_url: str | None = None
    """CDN base URL (required if url_mode='cdn'). e.g., 'https://cdn.example.com'"""

    presign_expires_seconds: int = 3600
    """Presigned URL expiration time in seconds (default: 1 hour)."""

    force_path_style: bool = False
    """Use path-style URLs. Required for some S3-compatible services (MinIO, Ceph, etc.)."""


class MediaConfig(BaseModel):
    """Media handling configuration."""

    s3: S3Config = Field(default_factory=S3Config)
    """S3 configuration for media upload."""

    max_pending_attachments: PositiveInt = 8
    """Maximum clipboard attachments queued for one prompt."""

    max_pending_attachment_bytes: PositiveInt = 20 * 1024 * 1024
    """Maximum total bytes of clipboard attachments queued for one prompt."""


class SessionConfig(BaseModel):
    """Saved session retention configuration."""

    max_turns_per_session: PositiveInt = 20
    """Maximum saved turns retained inside each session."""

    max_sessions: PositiveInt = 100
    """Maximum saved sessions retained globally."""

    max_session_age_days: PositiveInt | None = None
    """Optional maximum session age in days. Older sessions are pruned on save."""


class OAuthRefreshConfig(BaseModel):
    """OAuth proactive refresh configuration."""

    enabled: bool = True
    interval_seconds: PositiveInt = 1800
    failure_retry_seconds: PositiveInt = 60
    refresh_on_startup: bool = True


class YaacliConfig(BaseModel):
    """Complete yaacli configuration."""

    # From global config
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    display: DisplayConfig = Field(default_factory=DisplayConfig)
    subagents: SubagentsConfig = Field(default_factory=SubagentsConfig)
    media: MediaConfig = Field(default_factory=MediaConfig)
    """Media handling configuration (S3 upload, etc.)."""
    session: SessionConfig = Field(default_factory=SessionConfig)
    """Saved session retention configuration."""
    oauth_refresh: OAuthRefreshConfig = Field(default_factory=OAuthRefreshConfig)
    """OAuth proactive refresh configuration."""
    env: dict[str, str] = Field(default_factory=dict)
    """Environment variable overrides for the CLI process (e.g., API keys)."""
    shell_env: dict[str, str] = Field(default_factory=dict)
    """Environment variables injected into shell command execution.

    Separate from [env] to isolate CLI process env (API keys, etc.)
    from shell subprocess env. These are passed to AgentContext.shell_env.
    """
    include_os_env: bool = True
    """Whether shell subprocesses include the parent process environment.

    When True (default), os.environ is merged as the base layer when
    shell_env or per-call env is provided. When False, only the explicitly
    configured shell_env (and per-call env) is used.
    Set to False to prevent CLI process env vars (API keys, etc.)
    from leaking into shell subprocesses.
    """
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    """Security runtime configuration."""

    model_profiles: dict[str, ModelProfileConfig] = Field(default_factory=dict)
    """Selectable model profiles for the /model command."""

    # From project config
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    # Custom slash commands
    commands: dict[str, CommandDefinition] = Field(default_factory=dict)
    """Custom slash commands (merged with defaults)."""

    def get_commands(self) -> dict[str, CommandDefinition]:
        """Get all commands (defaults + user-defined, user overrides defaults)."""
        result = dict(DEFAULT_COMMANDS)
        result.update(self.commands)
        return result

    @property
    def is_configured(self) -> bool:
        """Check if minimum required configuration is present."""
        return self.general.is_configured


# =============================================================================
# Environment Settings (TUI only, using pydantic-settings)
# =============================================================================


class EnvSettings(BaseSettings):
    """TUI settings from environment variables.

    Only TUI-related settings, not model configuration.
    """

    model_config = SettingsConfigDict(
        env_prefix="YAACLI_",
        env_file=(_PACKAGE_ROOT / ".env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Display
    code_theme: Literal["dark", "light"] | None = None
    show_token_usage: bool | None = None
    show_elapsed_time: bool | None = None

    # Steering
    steering_enabled: bool | None = None
    steering_prefix: str | None = None

    # Session
    session_dir: str | None = None
    auto_save_history: bool | None = None
    auto_restore: bool | None = None
    max_turns_per_session: PositiveInt | None = None
    max_sessions: PositiveInt | None = None
    max_session_age_days: PositiveInt | None = None

    # Agent stream recovery
    agent_stream_resume_on_error: bool | None = None
    agent_stream_resume_max_attempts: int | None = None
    agent_stream_resume_prompt: str | None = None

    # OAuth refresh
    oauth_refresh_enabled: bool | None = None
    oauth_refresh_interval_seconds: PositiveInt | None = None
    oauth_refresh_failure_retry_seconds: PositiveInt | None = None
    oauth_refresh_on_startup: bool | None = None


# =============================================================================
# ConfigManager
# =============================================================================


class ConfigManager:
    """Manages configuration loading from global, project, and environment sources."""

    DEFAULT_CONFIG_DIR = Path.home() / ".yaacli"
    PROJECT_CONFIG_DIR = ".yaacli"

    def __init__(
        self,
        config_dir: Path | None = None,
        project_dir: Path | None = None,
    ) -> None:
        self._config_dir = config_dir or self.DEFAULT_CONFIG_DIR
        self._project_dir = project_dir or Path.cwd()
        self._config: YaacliConfig | None = None
        self._loaded_sources: list[str] = []

    @property
    def config(self) -> YaacliConfig:
        """Get current configuration, loading if necessary."""
        if self._config is None:
            self._config = self.load()
        return self._config

    @property
    def config_dir(self) -> Path:
        """Get global config directory."""
        return self._config_dir

    @property
    def project_dir(self) -> Path:
        """Get project directory."""
        return self._project_dir

    @property
    def loaded_sources(self) -> list[str]:
        """Get list of loaded configuration sources."""
        return self._loaded_sources.copy()

    def load(self) -> YaacliConfig:
        """Load configuration from all sources.

        Priority (higher wins, no merging between levels):
        1. config.toml: Project > Global
        2. tools.toml: Project > Global
        3. Environment overrides (TUI settings only, merged on top of config.toml)
        """
        self._loaded_sources = []
        merged: dict[str, Any] = {}

        # Layer 1: config.toml (project takes priority over global, no merge)
        project_config_file = self._project_dir / self.PROJECT_CONFIG_DIR / "config.toml"
        global_config_file = self._config_dir / "config.toml"

        if project_config_file.exists():
            with open(project_config_file, "rb") as f:
                config = tomllib.load(f)
            merged.update(config)
            self._loaded_sources.append(str(project_config_file))
        elif global_config_file.exists():
            with open(global_config_file, "rb") as f:
                config = tomllib.load(f)
            merged.update(config)
            self._loaded_sources.append(str(global_config_file))

        # Layer 2: Environment overrides (TUI only, merged)
        env_overrides = self._load_env_overrides()
        if env_overrides:
            merged = _deep_merge(merged, env_overrides)
            self._loaded_sources.append("environment")

        # Layer 3: tools.toml (project takes priority over global, no merge)
        project_tools_file = self._project_dir / self.PROJECT_CONFIG_DIR / "tools.toml"
        global_tools_file = self._config_dir / "tools.toml"

        if project_tools_file.exists():
            with open(project_tools_file, "rb") as f:
                tools_config = tomllib.load(f)
            if "tools" in tools_config:
                merged["tools"] = tools_config["tools"]
            self._loaded_sources.append(str(project_tools_file))
        elif global_tools_file.exists():
            with open(global_tools_file, "rb") as f:
                tools_config = tomllib.load(f)
            if "tools" in tools_config:
                merged["tools"] = tools_config["tools"]
            self._loaded_sources.append(str(global_tools_file))

        self._config = YaacliConfig.model_validate(merged)
        return self._config

    def reload(self) -> YaacliConfig:
        """Force reload configuration."""
        self._config = None
        return self.load()

    def _load_env_overrides(self) -> dict[str, Any]:
        """Load TUI settings from environment using pydantic-settings."""
        env = EnvSettings()
        overrides: dict[str, Any] = {}

        # Display
        display: dict[str, Any] = {}
        if env.code_theme is not None:
            display["code_theme"] = env.code_theme
        if env.show_token_usage is not None:
            display["show_token_usage"] = env.show_token_usage
        if env.show_elapsed_time is not None:
            display["show_elapsed_time"] = env.show_elapsed_time
        if display:
            overrides["display"] = display

        # Steering
        steering: dict[str, Any] = {}
        if env.steering_enabled is not None:
            steering["enabled"] = env.steering_enabled
        if env.steering_prefix is not None:
            steering["prefix"] = env.steering_prefix
        if steering:
            overrides["steering"] = steering

        # Session
        session: dict[str, Any] = {}
        if env.session_dir is not None:
            session["session_dir"] = env.session_dir
        if env.auto_save_history is not None:
            session["auto_save_history"] = env.auto_save_history
        if env.auto_restore is not None:
            session["auto_restore"] = env.auto_restore
        if env.max_turns_per_session is not None:
            session["max_turns_per_session"] = env.max_turns_per_session
        if env.max_sessions is not None:
            session["max_sessions"] = env.max_sessions
        if env.max_session_age_days is not None:
            session["max_session_age_days"] = env.max_session_age_days
        if session:
            overrides["session"] = session

        # Agent stream recovery
        general: dict[str, Any] = {}
        if env.agent_stream_resume_on_error is not None:
            general["agent_stream_resume_on_error"] = env.agent_stream_resume_on_error
        if env.agent_stream_resume_max_attempts is not None:
            general["agent_stream_resume_max_attempts"] = env.agent_stream_resume_max_attempts
        if env.agent_stream_resume_prompt is not None:
            general["agent_stream_resume_prompt"] = env.agent_stream_resume_prompt
        if general:
            overrides["general"] = general

        # OAuth refresh
        oauth_refresh: dict[str, Any] = {}
        if env.oauth_refresh_enabled is not None:
            oauth_refresh["enabled"] = env.oauth_refresh_enabled
        if env.oauth_refresh_interval_seconds is not None:
            oauth_refresh["interval_seconds"] = env.oauth_refresh_interval_seconds
        if env.oauth_refresh_failure_retry_seconds is not None:
            oauth_refresh["failure_retry_seconds"] = env.oauth_refresh_failure_retry_seconds
        if env.oauth_refresh_on_startup is not None:
            oauth_refresh["refresh_on_startup"] = env.oauth_refresh_on_startup
        if oauth_refresh:
            overrides["oauth_refresh"] = oauth_refresh

        return overrides

    def load_mcp_config(self) -> MCPConfig | None:
        """Load MCP configuration from mcp.json.

        Project config takes priority over global config (no merging).

        Returns:
            MCPConfig if found, None otherwise.
        """
        # Check project first
        project_mcp = self._project_dir / self.PROJECT_CONFIG_DIR / "mcp.json"
        if project_mcp.exists():
            return load_mcp_config_file(project_mcp)

        # Fall back to global
        global_mcp = self._config_dir / "mcp.json"
        if global_mcp.exists():
            return load_mcp_config_file(global_mcp)

        return None

    def get_mcp_config_file(self) -> Path | None:
        """Get path to active MCP config file (project or global)."""
        project_mcp = self._project_dir / self.PROJECT_CONFIG_DIR / "mcp.json"
        if project_mcp.exists():
            return project_mcp
        global_mcp = self._config_dir / "mcp.json"
        if global_mcp.exists():
            return global_mcp
        return None

    # Entries to exclude from file tree context in ~/.yaacli/
    _TREEIGNORE_DIRS = frozenset({"sessions", "message_history", "worktrees"})
    _TREEIGNORE_FILES = frozenset({"state.json"})

    def ensure_config_dir(self) -> None:
        """Create global config directory structure."""
        self._config_dir.mkdir(parents=True, exist_ok=True)
        (self._config_dir / "subagents").mkdir(exist_ok=True)
        self._ensure_gitignore()

    def _ensure_gitignore(self) -> None:
        """Ensure .gitignore exists in config dir to exclude ephemeral data from file tree context.

        The file tree generator reads .gitignore per allowed_path root.
        This keeps session/history directories out of the agent's context.
        """
        gitignore_path = self._config_dir / ".gitignore"
        needed = {f"{d}/" for d in self._TREEIGNORE_DIRS} | set(self._TREEIGNORE_FILES)
        if gitignore_path.exists():
            existing = set(gitignore_path.read_text().splitlines())
            missing = needed - existing
            if not missing:
                return
            # Append missing entries
            with gitignore_path.open("a") as f:
                for entry in sorted(missing):
                    f.write(f"\n{entry}")
        else:
            gitignore_path.write_text("\n".join(sorted(needed)) + "\n")

    def ensure_project_config_dir(self) -> None:
        """Create project config directory."""
        project_config_dir = self._project_dir / self.PROJECT_CONFIG_DIR
        project_config_dir.mkdir(parents=True, exist_ok=True)

    def save_default_config(self, force: bool = False) -> Path | None:
        """Save default global configuration."""
        config_file = self._config_dir / "config.toml"
        if config_file.exists() and not force:
            return None

        self.ensure_config_dir()
        config_file.write_text(_load_template("config.toml"))
        return config_file

    def save_project_config(self, force: bool = False) -> Path | None:
        """Save default project configuration."""
        self.ensure_project_config_dir()
        config_file = self._project_dir / self.PROJECT_CONFIG_DIR / "tools.toml"
        if config_file.exists() and not force:
            return None

        config_file.write_text(_load_template("tools.toml"))
        return config_file

    def get_global_config_file(self) -> Path:
        """Get path to global config file."""
        return self._config_dir / "config.toml"

    def get_project_config_file(self) -> Path:
        """Get path to project tools config file."""
        return self._project_dir / self.PROJECT_CONFIG_DIR / "tools.toml"

    def get_sessions_dir(self) -> Path:
        """Get sessions directory for session-based storage.

        Returns:
            Path to the sessions directory under global config.
            Format: ~/.yaacli/sessions/
        """
        return self._config_dir / "sessions"


# =============================================================================
# Worktree Metadata
# =============================================================================


class WorktreeMetadata(TypedDict):
    """Metadata for a git worktree managed by yaacli.

    Stored as JSON in ~/.yaacli/worktrees/{project_hash}/metadata.json.
    """

    git_root: str
    """Absolute path to the original git repository root."""

    created_at: str
    """ISO 8601 timestamp of when this worktree group was first created."""


# =============================================================================
# Internal Utilities
# =============================================================================


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_template(name: str) -> str:
    """Load a template file."""
    template_files = resources.files("yaacli").joinpath("templates")
    return template_files.joinpath(name).read_text(encoding="utf-8")


# =============================================================================
# Convenience Functions
# =============================================================================


def load_config(
    config_dir: Path | None = None,
    project_dir: Path | None = None,
) -> YaacliConfig:
    """Load configuration from all sources.

    Convenience function that creates a ConfigManager and loads config.

    Args:
        config_dir: Optional custom global config directory.
        project_dir: Optional custom project directory.

    Returns:
        Loaded YaacliConfig.
    """
    manager = ConfigManager(config_dir=config_dir, project_dir=project_dir)
    return manager.load()


def get_config_manager(
    config_dir: Path | None = None,
    project_dir: Path | None = None,
) -> ConfigManager:
    """Get a ConfigManager instance.

    Args:
        config_dir: Optional custom global config directory.
        project_dir: Optional custom project directory.

    Returns:
        ConfigManager instance.
    """
    return ConfigManager(config_dir=config_dir, project_dir=project_dir)
