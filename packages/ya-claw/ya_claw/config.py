from __future__ import annotations

import os
import socket
from collections.abc import Callable
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal, cast
from uuid import uuid4

from dotenv import dotenv_values, load_dotenv
from pydantic import AliasChoices, Field, PositiveInt, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from ya_agent_sdk.environment.virtual_path import normalize_virtual_path as normalize_agent_virtual_path

from ya_claw.bridge.models import BridgeAdapterType, BridgeDispatchMode
from ya_claw.workspace import DockerExtraMount

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATABASE_FILENAME = "ya_claw.sqlite3"
_DEFAULT_DATA_DIR = Path("~/.ya-claw/data")
_DEFAULT_RUN_STORE_DIRNAME = "run-store"
_DEFAULT_WORKSPACE_DIRNAME = "workspace"
_DEFAULT_WORKSPACE_DOCKER_IMAGE = "ghcr.io/wh1isper/ya-claw-workspace:latest"
_SERVICE_PACKAGE_NAME = "ya-claw"
_UNKNOWN_BUILD_VALUE = "unknown"


def _default_instance_id() -> str:
    hostname = socket.gethostname().split(".", 1)[0] or "host"
    return f"{hostname}-{os.getpid()}-{uuid4().hex[:8]}"


def _parse_env_var_names(value: str) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for raw_name in value.split(","):
        name = raw_name.strip()
        if name == "" or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def _split_docker_extra_mount(item: str) -> tuple[str, str, str]:
    for mode in (":rw", ":ro"):
        if item.endswith(mode):
            body = item[: -len(mode)]
            parsed_mode = mode[1:]
            break
    else:
        body = item
        parsed_mode = "rw"

    container_marker = body.rfind(":/")
    if container_marker < 0:
        raise ValueError("Docker extra mounts must use host_path:container_path[:mode] entries")
    return body[:container_marker], body[container_marker + 1 :], parsed_mode


def _parse_docker_extra_mounts(value: str) -> list[DockerExtraMount]:
    mounts: list[DockerExtraMount] = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if item == "":
            continue
        host_path_raw, container_path_raw, mode = _split_docker_extra_mount(item)
        host_path = Path(host_path_raw).expanduser()
        container_path = normalize_agent_virtual_path(container_path_raw)
        if str(host_path).strip() == "":
            raise ValueError("Docker extra mount host_path must not be empty")
        if not container_path.is_absolute():
            raise ValueError(f"Docker extra mount container_path must be absolute: {container_path}")
        if mode not in {"rw", "ro"}:
            raise ValueError(f"Docker extra mount mode must be 'rw' or 'ro': {mode}")
        mounts.append(DockerExtraMount(host_path=host_path, container_path=container_path, mode=mode))
    return mounts


def _normalized_optional_str(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def load_runtime_environment() -> dict[str, str]:
    package_env_file = (_PACKAGE_ROOT / ".env").expanduser()
    cwd_env_file = Path(".env").expanduser()

    merged: dict[str, str] = {}
    for env_file in (package_env_file, cwd_env_file):
        if env_file.exists():
            merged.update({
                key: value
                for key, value in dotenv_values(env_file).items()
                if isinstance(key, str) and isinstance(value, str)
            })

    for env_file in (cwd_env_file, package_env_file):
        if env_file.exists():
            load_dotenv(env_file, override=False, encoding="utf-8")

    return merged


class ClawSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="YA_CLAW_",
        env_file=(_PACKAGE_ROOT / ".env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "YA Claw"
    environment: str = "development"
    host: str = "127.0.0.1"
    port: int = 9042
    reload: bool = False
    log_level: str = "INFO"
    public_base_url: str = "http://127.0.0.1:9042"
    service_version: str | None = None
    service_commit: str | None = None
    service_build: str | None = None
    service_image: str | None = None
    instance_id: str = Field(default_factory=_default_instance_id)
    web_dist_dir: Path | None = None
    api_token: SecretStr | None = None
    data_dir: Path = Field(default_factory=lambda: _DEFAULT_DATA_DIR)
    workspace_dir: Path | None = None
    allow_origins: list[str] = Field(default_factory=lambda: ["http://127.0.0.1:5173", "http://localhost:5173"])

    database_url: str | None = None
    database_echo: bool = False
    database_pool_size: int = 5
    database_max_overflow: int = 10
    database_pool_recycle_seconds: int = 3600

    workspace_provider_backend: Literal["local", "docker"] = "docker"
    workspace_download_max_bytes: PositiveInt = 100 * 1024 * 1024
    workspace_provider_docker_image: str = _DEFAULT_WORKSPACE_DOCKER_IMAGE
    workspace_provider_docker_host_workspace_dir: Path | None = None
    workspace_provider_docker_uid: int | None = None
    workspace_provider_docker_gid: int | None = None
    workspace_provider_docker_container_cache_dir: Path | None = None
    workspace_provider_docker_extra_mounts: str = ""
    workspace_provider_docker_exec_user: str = "auto"
    workspace_provider_docker_home: str = "/home/claw"
    workspace_provider_docker_retention_policy: Literal["stop_on_idle", "keep_warm"] = "stop_on_idle"
    workspace_provider_docker_idle_ttl_seconds: PositiveInt = 3600
    shell_sandbox_enabled: bool = True
    shell_sandbox_backend: Literal[
        "auto",
        "linux_bwrap_seccomp",
        "macos_seatbelt",
        "windows_restricted_token",
        "docker",
        "podman",
        "nsjail",
        "raw_host",
    ] = "auto"
    shell_sandbox_network: Literal["blocked", "restricted", "proxy", "full"] = "full"
    shell_sandbox_allow_raw_host: bool = False
    workspace_env_vars: str = ""
    bridge_dispatch_mode: BridgeDispatchMode = BridgeDispatchMode.EMBEDDED
    bridge_enabled_adapters: str = ""
    bridge_lark_enabled: bool = False
    bridge_lark_app_id: str | None = None
    bridge_lark_app_secret: SecretStr | None = None
    bridge_lark_default_profile: str | None = None
    bridge_lark_event_types: str = (
        "im.chat.member.bot.added_v1,im.chat.member.user.added_v1,im.message.receive_v1,"
        "drive.notice.comment_add_v1,card.action.trigger"
    )
    bridge_lark_reply_identity: Literal["bot", "user"] = "bot"
    bridge_lark_domain: str = "https://open.feishu.cn"
    bridge_lark_previous_messages_enabled: bool = True
    bridge_lark_previous_messages_limit: PositiveInt = 6
    default_profile: str = "default"
    agent_stream_resume_on_error: bool = True
    agent_stream_resume_max_attempts: int = 3
    agent_stream_resume_prompt: str = (
        "The previous streaming model request failed before the agent finished. "
        "Continue the task from the available conversation history. Avoid repeating completed work."
    )
    oauth_refresh_enabled: bool = True
    oauth_refresh_interval_seconds: PositiveInt = 1800
    oauth_refresh_failure_retry_seconds: PositiveInt = 60
    oauth_refresh_on_startup: bool = True
    profile_seed_file: Path | None = None
    auto_seed_profiles: bool = False
    schedule_dispatch_enabled: bool = True
    schedule_tick_seconds: int = 5
    schedule_max_due_per_tick: int = 20
    workflow_dispatch_enabled: bool = True
    workflow_tick_seconds: int = 5
    workflow_max_runs_per_tick: int = 20
    heartbeat_enabled: bool = False
    heartbeat_interval_seconds: int = 300
    heartbeat_profile: str | None = None
    heartbeat_prompt: str = "Run heartbeat according to HEARTBEAT.md."
    heartbeat_on_active: Literal["skip", "queue"] = "skip"
    unattended_shell_review_risk_threshold: Literal["low", "medium", "high", "extra_high"] | None = None
    agency_enabled: bool = False
    agency_idle_after_seconds: int = 600
    agency_cooldown_seconds: int = 1800
    agency_profile: str | None = None
    agency_timer_interval_seconds: int = 3600
    agency_fire_batch_limit: int = 20
    agency_memory_capture_enabled: bool = True
    agency_context_max_chars: int = 8000
    agency_recent_files_limit: int = 5
    agency_index_target_chars: int = 16_000
    agency_index_max_chars: int = 32_000
    agency_action_log_recent_chars: int = 32_000
    agency_unattended_shell_review_risk_threshold: Literal["low", "medium", "high", "extra_high"] | None = "extra_high"
    memory_enabled: bool = True
    memory_extract_every_turns: int = 5
    memory_summary_every_extracts: int = 4
    memory_extract_on_compact: bool = True
    memory_extract_on_summarize: bool = True
    memory_inject_enabled: bool = True
    memory_context_max_chars: int = Field(
        default=8000,
        validation_alias=AliasChoices(
            "memory_context_max_chars",
            "memory_summary_max_chars",
            "YA_CLAW_MEMORY_CONTEXT_MAX_CHARS",
            "YA_CLAW_MEMORY_SUMMARY_MAX_CHARS",
        ),
    )
    memory_recent_extracts_limit: int = 5
    memory_profile: str | None = None
    session_prune_enabled: bool = False
    session_prune_interval_seconds: int = 86400
    session_prune_startup_delay_seconds: int = 300
    session_prune_batch_size: int = 1000
    session_prune_run_keep_recent: int = 10
    session_prune_run_older_than_days: int = 0
    session_prune_generated_sessions_enabled: bool = False
    session_prune_schedule_keep_recent: int = 10
    session_prune_schedule_older_than_days: int = 30
    session_prune_once_schedules_hide_after_days: int = 7
    session_prune_heartbeat_keep_recent: int = 10
    session_prune_heartbeat_older_than_days: int = 7
    session_prune_fire_records_older_than_days: int = 0
    session_prune_orphans_enabled: bool = True
    shutdown_timeout_seconds: PositiveInt | None = 30

    auto_migrate: bool = True

    @property
    def resolved_service_version(self) -> str:
        configured_version = _normalized_optional_str(self.service_version)
        if configured_version is not None:
            return configured_version
        try:
            return version(_SERVICE_PACKAGE_NAME)
        except PackageNotFoundError:
            return _UNKNOWN_BUILD_VALUE

    @property
    def resolved_service_commit(self) -> str | None:
        return _normalized_optional_str(self.service_commit)

    @property
    def resolved_service_build(self) -> str | None:
        return _normalized_optional_str(self.service_build)

    @property
    def resolved_service_image(self) -> str | None:
        return _normalized_optional_str(self.service_image)

    @property
    def resolved_service_revision(self) -> str:
        commit = self.resolved_service_commit
        if commit is None:
            return self.resolved_service_version
        return f"{self.resolved_service_version}+{commit[:12]}"

    @property
    def runtime_root(self) -> Path:
        return self.data_dir.expanduser().parent

    @property
    def runtime_data_dir(self) -> Path:
        return self.data_dir.expanduser()

    @property
    def resolved_workspace_dir(self) -> Path:
        if self.workspace_dir is not None:
            return self.workspace_dir.expanduser()
        return self.runtime_data_dir / _DEFAULT_WORKSPACE_DIRNAME

    @property
    def resolved_profile_seed_file(self) -> Path | None:
        if self.profile_seed_file is None:
            return None
        return self.profile_seed_file.expanduser()

    @property
    def run_store_dir(self) -> Path:
        return self.runtime_data_dir / _DEFAULT_RUN_STORE_DIRNAME

    @property
    def resolved_workspace_provider_docker_host_workspace_dir(self) -> Path:
        if self.workspace_provider_docker_host_workspace_dir is not None:
            return self.workspace_provider_docker_host_workspace_dir.expanduser()
        return self.resolved_workspace_dir

    @property
    def resolved_workspace_provider_docker_uid(self) -> int:
        if isinstance(self.workspace_provider_docker_uid, int):
            return self.workspace_provider_docker_uid
        getuid = getattr(os, "getuid", None)
        if callable(getuid):
            return cast(Callable[[], int], getuid)()
        return 1000

    @property
    def resolved_workspace_provider_docker_gid(self) -> int:
        if isinstance(self.workspace_provider_docker_gid, int):
            return self.workspace_provider_docker_gid
        getgid = getattr(os, "getgid", None)
        if callable(getgid):
            return cast(Callable[[], int], getgid)()
        return 1000

    @property
    def resolved_workspace_provider_docker_container_cache_dir(self) -> Path:
        if self.workspace_provider_docker_container_cache_dir is not None:
            return self.workspace_provider_docker_container_cache_dir.expanduser()
        return self.runtime_data_dir / "docker-workspace-containers"

    @property
    def resolved_workspace_provider_docker_extra_mounts(self) -> list[DockerExtraMount]:
        return _parse_docker_extra_mounts(self.workspace_provider_docker_extra_mounts)

    @property
    def resolved_workspace_provider_docker_exec_user(self) -> str:
        return self.workspace_provider_docker_exec_user.strip() or "auto"

    @property
    def resolved_workspace_provider_docker_exec_default_env(self) -> dict[str, str]:
        return {"HOME": self.workspace_provider_docker_home.strip() or "/home/claw", "USER": "claw"}

    @property
    def resolved_workspace_provider_docker_retention_policy(self) -> Literal["stop_on_idle", "keep_warm"]:
        return self.workspace_provider_docker_retention_policy

    @property
    def resolved_workspace_provider_docker_idle_ttl_seconds(self) -> int:
        return int(self.workspace_provider_docker_idle_ttl_seconds)

    @property
    def resolved_bridge_enabled_adapters(self) -> set[BridgeAdapterType]:
        raw_adapters = [item.strip() for item in self.bridge_enabled_adapters.split(",") if item.strip()]
        resolved_adapters = {BridgeAdapterType(adapter) for adapter in raw_adapters}
        if self.bridge_lark_enabled:
            resolved_adapters.add(BridgeAdapterType.LARK)
        return resolved_adapters

    @property
    def resolved_bridge_lark_event_types(self) -> list[str]:
        return [item.strip() for item in self.bridge_lark_event_types.split(",") if item.strip()]

    @property
    def resolved_bridge_lark_profile(self) -> str:
        if isinstance(self.bridge_lark_default_profile, str) and self.bridge_lark_default_profile.strip() != "":
            return self.bridge_lark_default_profile.strip()
        return self.default_profile

    @property
    def resolved_heartbeat_profile(self) -> str:
        if isinstance(self.heartbeat_profile, str) and self.heartbeat_profile.strip() != "":
            return self.heartbeat_profile.strip()
        return self.default_profile

    @property
    def resolved_agency_profile(self) -> str:
        if isinstance(self.agency_profile, str) and self.agency_profile.strip() != "":
            return self.agency_profile.strip()
        return self.default_profile

    @property
    def heartbeat_guidance_path(self) -> Path:
        return self.resolved_workspace_dir / "HEARTBEAT.md"

    @property
    def bridge_lark_app_secret_value(self) -> str | None:
        if self.bridge_lark_app_secret is None:
            return None
        normalized_value = self.bridge_lark_app_secret.get_secret_value().strip()
        return normalized_value or None

    @property
    def resolved_lark_cli_environment(self) -> dict[str, str]:
        environment: dict[str, str] = {}
        app_id = os.environ.get("LARKSUITE_CLI_APP_ID") or os.environ.get("LARK_APP_ID") or self.bridge_lark_app_id
        app_secret = (
            os.environ.get("LARKSUITE_CLI_APP_SECRET")
            or os.environ.get("LARK_APP_SECRET")
            or self.bridge_lark_app_secret_value
        )
        brand = os.environ.get("LARKSUITE_CLI_BRAND") or "feishu"
        default_as = os.environ.get("LARKSUITE_CLI_DEFAULT_AS") or self.bridge_lark_reply_identity
        strict_mode = os.environ.get("LARKSUITE_CLI_STRICT_MODE") or self.bridge_lark_reply_identity
        has_lark_cli_credentials = False
        if isinstance(app_id, str) and app_id.strip() != "":
            has_lark_cli_credentials = True
            environment["LARK_APP_ID"] = app_id.strip()
            environment["LARKSUITE_CLI_APP_ID"] = ""
        if isinstance(app_secret, str) and app_secret.strip() != "":
            has_lark_cli_credentials = True
            environment["LARK_APP_SECRET"] = app_secret.strip()
            environment["LARKSUITE_CLI_APP_SECRET"] = ""
        if has_lark_cli_credentials and brand.strip() != "":
            environment["LARKSUITE_CLI_BRAND"] = brand.strip()
        if has_lark_cli_credentials and default_as.strip() != "":
            environment["LARKSUITE_CLI_DEFAULT_AS"] = default_as.strip()
        if has_lark_cli_credentials and strict_mode.strip() != "":
            environment["LARKSUITE_CLI_STRICT_MODE"] = strict_mode.strip()
        return environment

    @property
    def resolved_forwarded_workspace_environment(self) -> dict[str, str]:
        environment: dict[str, str] = {}
        for name in _parse_env_var_names(self.workspace_env_vars):
            value = os.environ.get(name)
            if isinstance(value, str):
                environment[name] = value
        return environment

    @property
    def resolved_workspace_environment(self) -> dict[str, str]:
        return {
            **self.resolved_lark_cli_environment,
            **self.resolved_forwarded_workspace_environment,
        }

    @property
    def api_token_value(self) -> str | None:
        if self.api_token is None:
            return None

        normalized_value = self.api_token.get_secret_value().strip()
        return normalized_value or None

    def require_api_token(self) -> str:
        api_token = self.api_token_value
        if api_token is None:
            raise RuntimeError("YA_CLAW_API_TOKEN must be configured before starting YA Claw.")
        return api_token

    @property
    def database_path(self) -> Path:
        return self.runtime_root / _DEFAULT_DATABASE_FILENAME

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url

        return f"sqlite+aiosqlite:///{self.database_path.resolve()}"

    def ensure_runtime_directories(self) -> None:
        self.runtime_data_dir.mkdir(parents=True, exist_ok=True)
        self.run_store_dir.mkdir(parents=True, exist_ok=True)
        self.resolved_workspace_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> ClawSettings:
    load_runtime_environment()
    return ClawSettings()
