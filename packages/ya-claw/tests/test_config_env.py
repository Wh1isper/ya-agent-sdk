from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from ya_claw import config as config_module
from ya_claw.bridge import BridgeAdapterType, BridgeDispatchMode
from ya_claw.config import ClawSettings


def test_load_runtime_environment_exports_non_prefixed_provider_variables(
    tmp_path: Path,
    monkeypatch,
) -> None:
    package_root = tmp_path / "package-root"
    package_root.mkdir(parents=True, exist_ok=True)
    package_env_file = package_root / ".env"
    package_env_file.write_text(
        "YA_CLAW_API_TOKEN=package-token\nGATEWAY_API_KEY=package-key\n",
        encoding="utf-8",
    )

    cwd = tmp_path / "cwd"
    cwd.mkdir(parents=True, exist_ok=True)
    cwd_env_file = cwd / ".env"
    cwd_env_file.write_text(
        "GATEWAY_API_KEY=cwd-key\nGATEWAY_BASE_URL=https://gateway.example.test\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(cwd)
    monkeypatch.setattr(config_module, "_PACKAGE_ROOT", package_root)
    monkeypatch.delenv("YA_CLAW_API_TOKEN", raising=False)
    monkeypatch.delenv("GATEWAY_API_KEY", raising=False)
    monkeypatch.delenv("GATEWAY_BASE_URL", raising=False)
    config_module.get_settings.cache_clear()

    loaded = config_module.load_runtime_environment()
    settings = config_module.get_settings()

    assert loaded["YA_CLAW_API_TOKEN"] == "package-token"  # noqa: S105
    assert loaded["GATEWAY_API_KEY"] == "cwd-key"
    assert loaded["GATEWAY_BASE_URL"] == "https://gateway.example.test"
    assert os.environ["GATEWAY_API_KEY"] == "cwd-key"
    assert os.environ["GATEWAY_BASE_URL"] == "https://gateway.example.test"
    assert settings.api_token_value == "package-token"  # noqa: S105

    config_module.get_settings.cache_clear()


def test_load_runtime_environment_preserves_existing_process_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    package_root = tmp_path / "package-root"
    package_root.mkdir(parents=True, exist_ok=True)
    package_env_file = package_root / ".env"
    package_env_file.write_text(
        "GATEWAY_API_KEY=package-key\n",
        encoding="utf-8",
    )

    cwd = tmp_path / "cwd"
    cwd.mkdir(parents=True, exist_ok=True)
    cwd_env_file = cwd / ".env"
    cwd_env_file.write_text(
        "GATEWAY_API_KEY=cwd-key\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(cwd)
    monkeypatch.setattr(config_module, "_PACKAGE_ROOT", package_root)
    monkeypatch.setenv("GATEWAY_API_KEY", "process-key")
    config_module.get_settings.cache_clear()

    loaded = config_module.load_runtime_environment()

    assert loaded["GATEWAY_API_KEY"] == "cwd-key"
    assert os.environ["GATEWAY_API_KEY"] == "process-key"

    config_module.get_settings.cache_clear()


def test_settings_use_official_workspace_image_by_default(monkeypatch) -> None:
    monkeypatch.delenv("YA_CLAW_WORKSPACE_PROVIDER_DOCKER_IMAGE", raising=False)
    settings = ClawSettings(api_token="test-token", _env_file=None)  # noqa: S106

    assert settings.workspace_provider_docker_image == "ghcr.io/wh1isper/ya-claw-workspace:latest"


def test_settings_service_build_metadata_can_be_configured() -> None:
    settings = ClawSettings(
        api_token="test-token",  # noqa: S106
        service_version="dev",
        service_commit="abcdef1234567890",
        service_build="dev-42-1",
        service_image="ghcr.io/example/ya-claw:dev",
        _env_file=None,
    )

    assert settings.resolved_service_version == "dev"
    assert settings.resolved_service_commit == "abcdef1234567890"
    assert settings.resolved_service_build == "dev-42-1"
    assert settings.resolved_service_image == "ghcr.io/example/ya-claw:dev"
    assert settings.resolved_service_revision == "dev+abcdef123456"


def test_settings_session_prune_defaults() -> None:
    settings = ClawSettings(api_token="test-token", _env_file=None)  # noqa: S106

    assert settings.session_prune_enabled is False
    assert settings.session_prune_run_keep_recent == 10
    assert settings.session_prune_generated_sessions_enabled is False
    assert settings.session_prune_schedule_keep_recent == 10
    assert settings.session_prune_heartbeat_keep_recent == 10
    assert settings.session_prune_fire_records_older_than_days == 0
    assert settings.session_prune_orphans_enabled is True


def test_settings_default_workspace_docker_host_workspace_dir_uses_workspace_dir(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    settings = ClawSettings(
        api_token="test-token",  # noqa: S106
        workspace_dir=workspace_dir,
        _env_file=None,
    )

    assert settings.resolved_workspace_provider_docker_host_workspace_dir == workspace_dir


def test_settings_workspace_docker_host_workspace_dir_can_be_configured(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "service-workspace"
    host_workspace_dir = tmp_path / "host-workspace"
    settings = ClawSettings(
        api_token="test-token",  # noqa: S106
        workspace_dir=workspace_dir,
        workspace_provider_docker_host_workspace_dir=host_workspace_dir,
        _env_file=None,
    )

    assert settings.resolved_workspace_provider_docker_host_workspace_dir == host_workspace_dir


def test_settings_default_workspace_docker_identity_uses_process_uid_gid(monkeypatch) -> None:
    monkeypatch.delenv("YA_CLAW_WORKSPACE_PROVIDER_DOCKER_UID", raising=False)
    monkeypatch.delenv("YA_CLAW_WORKSPACE_PROVIDER_DOCKER_GID", raising=False)
    with patch.object(os, "getuid", return_value=1234), patch.object(os, "getgid", return_value=2345):
        settings = ClawSettings(api_token="test-token", _env_file=None)  # noqa: S106
        assert settings.resolved_workspace_provider_docker_uid == 1234
        assert settings.resolved_workspace_provider_docker_gid == 2345


def test_settings_workspace_docker_identity_can_be_configured() -> None:
    settings = ClawSettings(
        api_token="test-token",  # noqa: S106
        workspace_provider_docker_uid=3456,
        workspace_provider_docker_gid=4567,
        _env_file=None,
    )

    assert settings.resolved_workspace_provider_docker_uid == 3456
    assert settings.resolved_workspace_provider_docker_gid == 4567


def test_settings_default_workspace_docker_exec_user_and_home() -> None:
    settings = ClawSettings(api_token="test-token", _env_file=None)  # noqa: S106

    assert settings.resolved_workspace_provider_docker_exec_user == "auto"
    assert settings.resolved_workspace_provider_docker_exec_default_env == {"HOME": "/home/claw", "USER": "claw"}


def test_settings_workspace_docker_exec_user_and_home_can_be_configured() -> None:
    settings = ClawSettings(
        api_token="test-token",  # noqa: S106
        workspace_provider_docker_exec_user="root",
        workspace_provider_docker_home="/custom-home",
        _env_file=None,
    )

    assert settings.resolved_workspace_provider_docker_exec_user == "root"
    assert settings.resolved_workspace_provider_docker_exec_default_env == {"HOME": "/custom-home", "USER": "claw"}


def test_settings_default_workspace_docker_container_cache_dir(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    settings = ClawSettings(
        api_token="test-token",  # noqa: S106
        data_dir=data_dir,
        _env_file=None,
    )

    assert settings.resolved_workspace_provider_docker_container_cache_dir == data_dir / "docker-workspace-containers"


def test_settings_workspace_docker_container_cache_dir_can_be_configured(tmp_path: Path) -> None:
    cache_dir = tmp_path / "custom-cache"
    settings = ClawSettings(
        api_token="test-token",  # noqa: S106
        workspace_provider_docker_container_cache_dir=cache_dir,
        _env_file=None,
    )

    assert settings.resolved_workspace_provider_docker_container_cache_dir == cache_dir


def test_settings_resolves_bridge_and_lark_cli_environment(monkeypatch) -> None:
    monkeypatch.delenv("LARKSUITE_CLI_APP_ID", raising=False)
    monkeypatch.delenv("LARKSUITE_CLI_APP_SECRET", raising=False)
    monkeypatch.delenv("LARKSUITE_CLI_BRAND", raising=False)
    monkeypatch.delenv("LARKSUITE_CLI_DEFAULT_AS", raising=False)
    monkeypatch.delenv("LARKSUITE_CLI_STRICT_MODE", raising=False)
    monkeypatch.delenv("LARK_APP_ID", raising=False)
    monkeypatch.delenv("LARK_APP_SECRET", raising=False)
    settings = ClawSettings(
        api_token="test-token",  # noqa: S106
        bridge_enabled_adapters="lark",
        bridge_lark_app_id="cli_test",
        bridge_lark_app_secret="secret-value",  # noqa: S106
        bridge_lark_default_profile="lark-profile",
        _env_file=None,
    )

    assert settings.bridge_dispatch_mode == BridgeDispatchMode.EMBEDDED
    assert settings.resolved_bridge_enabled_adapters == {BridgeAdapterType.LARK}
    assert settings.resolved_bridge_lark_event_types == [
        "im.chat.member.bot.added_v1",
        "im.chat.member.user.added_v1",
        "im.message.receive_v1",
        "drive.notice.comment_add_v1",
    ]
    assert settings.resolved_bridge_lark_profile == "lark-profile"
    assert settings.resolved_lark_cli_environment == {
        "LARKSUITE_CLI_APP_ID": "",
        "LARKSUITE_CLI_APP_SECRET": "",
        "LARKSUITE_CLI_BRAND": "feishu",
        "LARKSUITE_CLI_DEFAULT_AS": "bot",
        "LARKSUITE_CLI_STRICT_MODE": "bot",
        "LARK_APP_ID": "cli_test",
        "LARK_APP_SECRET": "secret-value",
    }
    assert settings.resolved_workspace_environment == {
        "LARKSUITE_CLI_APP_ID": "",
        "LARKSUITE_CLI_APP_SECRET": "",
        "LARKSUITE_CLI_BRAND": "feishu",
        "LARKSUITE_CLI_DEFAULT_AS": "bot",
        "LARKSUITE_CLI_STRICT_MODE": "bot",
        "LARK_APP_ID": "cli_test",
        "LARK_APP_SECRET": "secret-value",
    }


def test_settings_lark_cli_environment_prefers_official_process_environment(monkeypatch) -> None:
    monkeypatch.setenv("LARKSUITE_CLI_APP_ID", "official-cli")
    monkeypatch.setenv("LARKSUITE_CLI_APP_SECRET", "official-secret")
    monkeypatch.setenv("LARKSUITE_CLI_BRAND", "lark")
    monkeypatch.setenv("LARKSUITE_CLI_DEFAULT_AS", "user")
    monkeypatch.setenv("LARKSUITE_CLI_STRICT_MODE", "user")
    monkeypatch.setenv("LARK_APP_ID", "legacy-cli")
    monkeypatch.setenv("LARK_APP_SECRET", "legacy-secret")
    settings = ClawSettings(
        api_token="test-token",  # noqa: S106
        bridge_lark_app_id="settings-cli",
        bridge_lark_app_secret="settings-secret",  # noqa: S106
        _env_file=None,
    )

    assert settings.resolved_lark_cli_environment == {
        "LARKSUITE_CLI_APP_ID": "",
        "LARKSUITE_CLI_APP_SECRET": "",
        "LARKSUITE_CLI_BRAND": "lark",
        "LARKSUITE_CLI_DEFAULT_AS": "user",
        "LARKSUITE_CLI_STRICT_MODE": "user",
        "LARK_APP_ID": "official-cli",
        "LARK_APP_SECRET": "official-secret",
    }


def test_settings_lark_cli_environment_accepts_legacy_process_environment(monkeypatch) -> None:
    monkeypatch.delenv("LARKSUITE_CLI_APP_ID", raising=False)
    monkeypatch.delenv("LARKSUITE_CLI_APP_SECRET", raising=False)
    monkeypatch.delenv("LARKSUITE_CLI_BRAND", raising=False)
    monkeypatch.delenv("LARKSUITE_CLI_DEFAULT_AS", raising=False)
    monkeypatch.delenv("LARKSUITE_CLI_STRICT_MODE", raising=False)
    monkeypatch.setenv("LARK_APP_ID", "process-cli")
    monkeypatch.setenv("LARK_APP_SECRET", "process-secret")
    settings = ClawSettings(
        api_token="test-token",  # noqa: S106
        bridge_lark_app_id="settings-cli",
        bridge_lark_app_secret="settings-secret",  # noqa: S106
        _env_file=None,
    )

    assert settings.resolved_lark_cli_environment == {
        "LARKSUITE_CLI_APP_ID": "",
        "LARKSUITE_CLI_APP_SECRET": "",
        "LARKSUITE_CLI_BRAND": "feishu",
        "LARKSUITE_CLI_DEFAULT_AS": "bot",
        "LARKSUITE_CLI_STRICT_MODE": "bot",
        "LARK_APP_ID": "process-cli",
        "LARK_APP_SECRET": "process-secret",
    }


def test_settings_resolves_docker_extra_mounts(tmp_path: Path) -> None:
    home_dir = tmp_path / "home"
    cache_dir = tmp_path / "cache"
    settings = ClawSettings(
        api_token="test-token",  # noqa: S106
        workspace_provider_docker_extra_mounts=f"{home_dir}:/home/claw:rw,{cache_dir}:/cache:ro",
        _env_file=None,
    )

    mounts = settings.resolved_workspace_provider_docker_extra_mounts

    assert [(mount.host_path, mount.container_path, mount.mode) for mount in mounts] == [
        (home_dir, Path("/home/claw"), "rw"),
        (cache_dir, Path("/cache"), "ro"),
    ]


def test_settings_rejects_invalid_docker_extra_mounts() -> None:
    settings = ClawSettings(
        api_token="test-token",  # noqa: S106
        workspace_provider_docker_extra_mounts="relative-host:relative-container:rw",
        _env_file=None,
    )

    try:
        _ = settings.resolved_workspace_provider_docker_extra_mounts
    except ValueError as exc:
        assert "container_path must be absolute" in str(exc)
    else:
        raise AssertionError("Expected invalid Docker extra mount to raise ValueError")


def test_settings_resolves_explicit_workspace_environment(monkeypatch) -> None:
    monkeypatch.setenv("MY_TOOL_API_KEY", "tool-secret")
    monkeypatch.setenv("MY_TOOL_ENDPOINT", "https://tool.example.test")
    monkeypatch.delenv("MISSING_TOOL_ENV", raising=False)
    settings = ClawSettings(
        api_token="test-token",  # noqa: S106
        workspace_env_vars=" MY_TOOL_API_KEY,MY_TOOL_ENDPOINT,MY_TOOL_API_KEY,MISSING_TOOL_ENV ",
        _env_file=None,
    )

    assert settings.resolved_forwarded_workspace_environment == {
        "MY_TOOL_API_KEY": "tool-secret",
        "MY_TOOL_ENDPOINT": "https://tool.example.test",
    }
    assert settings.resolved_workspace_environment == {
        "MY_TOOL_API_KEY": "tool-secret",
        "MY_TOOL_ENDPOINT": "https://tool.example.test",
    }


def test_settings_workspace_environment_combines_lark_alias_and_explicit_forwarding(monkeypatch) -> None:
    monkeypatch.setenv("MY_TOOL_API_KEY", "tool-secret")
    monkeypatch.delenv("LARKSUITE_CLI_APP_ID", raising=False)
    monkeypatch.delenv("LARKSUITE_CLI_APP_SECRET", raising=False)
    monkeypatch.delenv("LARKSUITE_CLI_BRAND", raising=False)
    monkeypatch.delenv("LARKSUITE_CLI_DEFAULT_AS", raising=False)
    monkeypatch.delenv("LARKSUITE_CLI_STRICT_MODE", raising=False)
    monkeypatch.delenv("LARK_APP_ID", raising=False)
    monkeypatch.delenv("LARK_APP_SECRET", raising=False)
    settings = ClawSettings(
        api_token="test-token",  # noqa: S106
        bridge_lark_app_id="cli_test",
        bridge_lark_app_secret="secret-value",  # noqa: S106
        workspace_env_vars="MY_TOOL_API_KEY,LARK_APP_ID",
        _env_file=None,
    )

    assert settings.resolved_workspace_environment == {
        "LARKSUITE_CLI_APP_ID": "",
        "LARKSUITE_CLI_APP_SECRET": "",
        "LARKSUITE_CLI_BRAND": "feishu",
        "LARKSUITE_CLI_DEFAULT_AS": "bot",
        "LARKSUITE_CLI_STRICT_MODE": "bot",
        "LARK_APP_ID": "cli_test",
        "LARK_APP_SECRET": "secret-value",
        "MY_TOOL_API_KEY": "tool-secret",
    }
