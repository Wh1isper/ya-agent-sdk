"""Direct tests for configuration and process-environment persistence policy."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError
from yaacli import cli as cli_module
from yaacli.cli import load_env_from_config, load_package_env_files
from yaacli.config import ConfigManager, NotificationConfig, SessionConfig, YaacliConfig


def test_session_config_defaults() -> None:
    config = SessionConfig()

    assert config.session_dir is None
    assert config.auto_save_history is True
    assert config.auto_restore is False
    assert config.max_turns_per_session == 20
    assert config.max_sessions == 100
    assert config.max_session_age_days is None


def test_packaged_config_template_exposes_terminal_bell_policy() -> None:
    template_path = Path(__file__).resolve().parents[1] / "yaacli" / "templates" / "config.toml"
    notification_template = tomllib.loads(template_path.read_text(encoding="utf-8"))["notifications"]
    defaults = NotificationConfig()

    assert notification_template["bell_on_turn_complete"] is defaults.bell_on_turn_complete
    assert notification_template["bell_on_user_action_required"] is defaults.bell_on_user_action_required


def test_packaged_config_template_exposes_session_persistence_policy() -> None:
    template_path = Path(__file__).resolve().parents[1] / "yaacli" / "templates" / "config.toml"
    template_text = template_path.read_text(encoding="utf-8")
    session_template = tomllib.loads(template_text)["session"]
    defaults = SessionConfig()

    assert '# session_dir = "~/.yaacli/sessions"' in template_text
    assert session_template["auto_save_history"] is defaults.auto_save_history
    assert session_template["auto_restore"] is defaults.auto_restore
    assert session_template["max_turns_per_session"] == defaults.max_turns_per_session
    assert session_template["max_sessions"] == defaults.max_sessions
    assert "# max_session_age_days = 90" in template_text


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_turns_per_session", 0),
        ("max_turns_per_session", -1),
        ("max_sessions", 0),
        ("max_sessions", -1),
        ("max_session_age_days", 0),
        ("max_session_age_days", -1),
    ],
)
def test_session_config_rejects_non_positive_limits(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        SessionConfig.model_validate({field: value})


@pytest.mark.parametrize("shell_review", [{}, {"model": ""}, {"model": "   "}])
def test_enabled_shell_review_requires_non_blank_model(shell_review: dict[str, str]) -> None:
    with pytest.raises(ValidationError, match=r"security\.shell_review\.model is required"):
        YaacliConfig.model_validate({
            "security": {
                "shell_review": {
                    "enabled": True,
                    **shell_review,
                }
            }
        })


def test_load_env_from_config_does_not_override_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YAACLI_TEST_EXISTING", "from-process")
    monkeypatch.delenv("YAACLI_TEST_NEW", raising=False)
    config = YaacliConfig(
        env={
            "YAACLI_TEST_EXISTING": "from-config",
            "YAACLI_TEST_NEW": "from-config",
        }
    )

    load_env_from_config(config)

    assert cli_module.os.environ["YAACLI_TEST_EXISTING"] == "from-process"
    assert cli_module.os.environ["YAACLI_TEST_NEW"] == "from-config"


def test_supported_process_environment_overrides_are_applied(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = {
        "YAACLI_SHOW_TOKEN_USAGE": "false",
        "YAACLI_SHOW_ELAPSED_TIME": "false",
        "YAACLI_AUTO_SAVE_HISTORY": "false",
        "YAACLI_AUTO_RESTORE": "true",
        "YAACLI_MAX_TURNS_PER_SESSION": "7",
        "YAACLI_MAX_SESSIONS": "11",
        "YAACLI_MAX_SESSION_AGE_DAYS": "13",
        "YAACLI_AGENT_STREAM_RESUME_ON_ERROR": "false",
        "YAACLI_AGENT_STREAM_RESUME_PROMPT": "Resume without repeating work.",
        "YAACLI_OAUTH_REFRESH_ENABLED": "false",
        "YAACLI_OAUTH_REFRESH_FAILURE_RETRY_SECONDS": "17",
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)

    config = ConfigManager(
        config_dir=tmp_path / "global",
        project_dir=tmp_path / "project",
    ).load()

    assert config.display.show_token_usage is False
    assert config.display.show_elapsed_time is False
    assert config.session.auto_save_history is False
    assert config.session.auto_restore is True
    assert config.session.max_turns_per_session == 7
    assert config.session.max_sessions == 11
    assert config.session.max_session_age_days == 13
    assert config.general.agent_stream_resume_on_error is False
    assert config.general.agent_stream_resume_prompt == "Resume without repeating work."
    assert config.oauth_refresh.enabled is False
    assert config.oauth_refresh.failure_retry_seconds == 17


def test_package_env_precedes_working_directory_env_without_overriding_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "package"
    working_dir = tmp_path / "workspace"
    package_root.mkdir()
    working_dir.mkdir()
    (package_root / ".env").write_text(
        "YAACLI_DOTENV_SHARED=from-package\n"
        "YAACLI_DOTENV_PACKAGE_ONLY=from-package\n"
        "YAACLI_DOTENV_PROCESS=from-package\n"
    )
    (working_dir / ".env").write_text(
        "YAACLI_DOTENV_SHARED=from-workspace\n"
        "YAACLI_DOTENV_WORKSPACE_ONLY=from-workspace\n"
        "YAACLI_DOTENV_PROCESS=from-workspace\n"
    )
    for key in (
        "YAACLI_DOTENV_SHARED",
        "YAACLI_DOTENV_PACKAGE_ONLY",
        "YAACLI_DOTENV_WORKSPACE_ONLY",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("YAACLI_DOTENV_PROCESS", "from-process")
    monkeypatch.setattr(cli_module, "_PACKAGE_ROOT", package_root)
    monkeypatch.chdir(working_dir)

    load_package_env_files()

    assert cli_module.os.environ["YAACLI_DOTENV_SHARED"] == "from-package"
    assert cli_module.os.environ["YAACLI_DOTENV_PACKAGE_ONLY"] == "from-package"
    assert cli_module.os.environ["YAACLI_DOTENV_WORKSPACE_ONLY"] == "from-workspace"
    assert cli_module.os.environ["YAACLI_DOTENV_PROCESS"] == "from-process"
