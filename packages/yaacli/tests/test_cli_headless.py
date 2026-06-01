from __future__ import annotations

from unittest.mock import MagicMock

from click.testing import CliRunner
from yaacli.cli import cli


def test_cli_headless_forwards_session_and_profile(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    config = MagicMock()
    config_manager = MagicMock()
    monkeypatch.setattr("yaacli.cli._prepare_cli_runtime", MagicMock(return_value=(config_manager, config)))
    run_headless = MagicMock(return_value="session-1")
    monkeypatch.setattr("yaacli.cli.asyncio.run", lambda coro: run_headless(coro))
    monkeypatch.setattr(
        "yaacli.cli._run_headless_prompt",
        MagicMock(return_value="headless-coro"),
    )

    result = CliRunner().invoke(
        cli,
        ["-p", "hello", "--session", "session-0", "--profile", "fast"],
    )

    assert result.exit_code == 0
    from yaacli import cli as cli_module

    cli_module._run_headless_prompt.assert_called_once_with(
        config,
        config_manager,
        "hello",
        working_dir=cli_module.Path.cwd(),
        session_id="session-0",
        model_profile_id="fast",
    )


def test_cli_help_includes_sessions_and_profile_options() -> None:
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "--session" in result.output
    assert "--profile" in result.output
    assert "sessions" in result.output

    sessions_result = CliRunner().invoke(cli, ["sessions", "--help"])
    assert sessions_result.exit_code == 0
    assert "list" in sessions_result.output
    assert "show" in sessions_result.output
    assert "delete" in sessions_result.output
