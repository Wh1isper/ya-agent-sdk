from __future__ import annotations

import json
import runpy
from unittest.mock import AsyncMock, MagicMock

import pytest
from click.testing import CliRunner
from yaacli.cli import cli


class UnprintableCliError(RuntimeError):
    def __str__(self) -> str:
        raise ValueError("broken __str__")

    def __repr__(self) -> str:
        raise ValueError("broken __repr__")


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
        worker=False,
    )


def test_cli_headless_forwards_worker(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    config = MagicMock()
    config_manager = MagicMock()
    monkeypatch.setattr("yaacli.cli._prepare_cli_runtime", MagicMock(return_value=(config_manager, config)))
    run_headless = MagicMock(return_value="session-1")
    monkeypatch.setattr("yaacli.cli.asyncio.run", lambda coro: run_headless(coro))
    monkeypatch.setattr(
        "yaacli.cli._run_headless_prompt",
        MagicMock(return_value="headless-coro"),
    )

    result = CliRunner().invoke(cli, ["-p", "hello", "--worker"])

    assert result.exit_code == 0
    from yaacli import cli as cli_module

    cli_module._run_headless_prompt.assert_called_once_with(
        config,
        config_manager,
        "hello",
        working_dir=cli_module.Path.cwd(),
        session_id=None,
        model_profile_id=None,
        worker=True,
    )


def test_cli_worker_requires_headless_prompt() -> None:
    result = CliRunner().invoke(cli, ["--worker"])
    assert result.exit_code != 0
    assert "--worker requires --prompt/-p headless mode" in result.output


def test_cli_headless_keeps_human_output_on_stderr(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    config = MagicMock()
    config_manager = MagicMock()

    def prepare_runtime(_verbose: bool):  # type: ignore[no-untyped-def]
        print("configuration diagnostic")
        return config_manager, config

    def run_headless(_coro: object) -> str:
        print(json.dumps({"type": "RUN_STARTED"}))
        return "session-1"

    monkeypatch.setattr("yaacli.cli._prepare_cli_runtime", prepare_runtime)
    monkeypatch.setattr("yaacli.cli._run_headless_prompt", MagicMock(return_value="headless-coro"))
    monkeypatch.setattr("yaacli.cli.asyncio.run", run_headless)

    result = CliRunner().invoke(cli, ["-p", "hello"])

    assert result.exit_code == 0
    assert [json.loads(line) for line in result.stdout.splitlines()] == [{"type": "RUN_STARTED"}]
    assert "configuration diagnostic" in result.stderr
    assert "Session: session-1" in result.stderr
    assert "To resume this session:" in result.stderr


def test_cli_headless_fatal_output_does_not_corrupt_ndjson(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    config = MagicMock()
    config_manager = MagicMock()
    monkeypatch.setattr("yaacli.cli._prepare_cli_runtime", MagicMock(return_value=(config_manager, config)))
    monkeypatch.setattr("yaacli.cli._run_headless_prompt", MagicMock(return_value="headless-coro"))

    def fail_after_event(_coro: object) -> None:
        print(json.dumps({"type": "RUN_STARTED"}))
        raise RuntimeError("headless failed")

    monkeypatch.setattr("yaacli.cli.asyncio.run", fail_after_event)

    result = CliRunner().invoke(cli, ["-p", "hello"])

    assert result.exit_code == 1
    assert [json.loads(line) for line in result.stdout.splitlines()] == [{"type": "RUN_STARTED"}]
    assert "FATAL ERROR" in result.stderr
    assert "Message: headless failed" in result.stderr
    assert "FATAL ERROR" not in result.stdout


def test_cli_headless_fatal_survives_broken_exception_formatters(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    config = MagicMock()
    config_manager = MagicMock()
    monkeypatch.setattr("yaacli.cli._prepare_cli_runtime", MagicMock(return_value=(config_manager, config)))
    monkeypatch.setattr("yaacli.cli._run_headless_prompt", MagicMock(return_value="headless-coro"))
    monkeypatch.setattr("yaacli.cli.asyncio.run", MagicMock(side_effect=UnprintableCliError()))

    result = CliRunner().invoke(cli, ["-p", "hello"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "FATAL ERROR" in result.stderr
    assert "Error type: UnprintableCliError" in result.stderr
    assert "Message: <UnprintableCliError: exception text unavailable>" in result.stderr


def test_cli_tui_fatal_output_stays_on_stdout(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    config = MagicMock()
    config_manager = MagicMock()
    monkeypatch.setattr("yaacli.cli._prepare_cli_runtime", MagicMock(return_value=(config_manager, config)))
    monkeypatch.setattr("yaacli.cli._run_tui", MagicMock(return_value="tui-coro"))
    monkeypatch.setattr("yaacli.cli.asyncio.run", MagicMock(side_effect=RuntimeError("tui failed")))

    result = CliRunner().invoke(cli)

    assert result.exit_code == 1
    assert "FATAL ERROR" in result.stdout
    assert "Message: tui failed" in result.stdout
    assert "FATAL ERROR" not in result.stderr


def test_python_module_entrypoint_invokes_cli(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    cli_mock = MagicMock()
    monkeypatch.setattr("yaacli.cli.cli", cli_mock)

    runpy.run_module("yaacli", run_name="__main__")

    cli_mock.assert_called_once_with()


def test_cli_help_includes_sessions_and_profile_options() -> None:
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "--session" in result.output
    assert "--profile" in result.output
    assert "--worker" in result.output
    assert "sessions" in result.output

    sessions_result = CliRunner().invoke(cli, ["sessions", "--help"])
    assert sessions_result.exit_code == 0
    assert "list" in sessions_result.output
    assert "show" in sessions_result.output
    assert "delete" in sessions_result.output


@pytest.mark.asyncio
async def test_run_tui_profile_override_is_not_persisted(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI --profile applies to one invocation; interactive /model owns persistence."""
    from yaacli import cli as cli_module

    config = MagicMock()
    config_manager = MagicMock()
    profile = MagicMock()
    fake_app = MagicMock()
    fake_app.__aenter__ = AsyncMock(return_value=fake_app)
    fake_app.__aexit__ = AsyncMock(return_value=None)
    fake_app._switch_model_profile = AsyncMock()
    fake_app.run = AsyncMock()
    fake_app.has_session_data = False
    monkeypatch.setattr("yaacli.model_profiles.get_model_profile", MagicMock(return_value=profile))
    monkeypatch.setattr("yaacli.app.TUIApp", MagicMock(return_value=fake_app))

    result = await cli_module._run_tui(
        config,
        config_manager,
        False,
        model_profile_id="fast",
    )

    assert result is None
    fake_app._switch_model_profile.assert_awaited_once_with(profile, persist=False)


def test_cli_tui_forwards_session_and_profile(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    config = MagicMock()
    config_manager = MagicMock()
    monkeypatch.setattr("yaacli.cli._prepare_cli_runtime", MagicMock(return_value=(config_manager, config)))
    run_tui = MagicMock(return_value="session-1")
    monkeypatch.setattr("yaacli.cli.asyncio.run", lambda coro: run_tui(coro))
    monkeypatch.setattr("yaacli.cli._run_tui", MagicMock(return_value="tui-coro"))

    result = CliRunner().invoke(cli, ["--session", "session-0", "--profile", "fast"])

    assert result.exit_code == 0
    assert "Session: session-1" in result.output
    assert "yaacli --session session-1" in result.output
    from yaacli import cli as cli_module

    cli_module._run_tui.assert_called_once_with(
        config,
        config_manager,
        False,
        working_dir=cli_module.Path.cwd(),
        session_id="session-0",
        model_profile_id="fast",
    )


def test_cli_tui_resume_command_quotes_noncanonical_session_id(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    config = MagicMock()
    config_manager = MagicMock()
    session_id = "session; echo unexpected"
    monkeypatch.setattr("yaacli.cli._prepare_cli_runtime", MagicMock(return_value=(config_manager, config)))
    run_tui = MagicMock(return_value=session_id)
    monkeypatch.setattr("yaacli.cli.asyncio.run", lambda coro: run_tui(coro))
    monkeypatch.setattr("yaacli.cli._run_tui", MagicMock(return_value="tui-coro"))

    result = CliRunner().invoke(cli)

    assert result.exit_code == 0
    assert "yaacli --session 'session; echo unexpected'" in result.output
