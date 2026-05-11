from __future__ import annotations

import secrets
from pathlib import Path

import ya_claw.cli as claw_cli
from click.testing import CliRunner
from ya_claw.config import ClawSettings

runner = CliRunner()
TEST_API_TOKEN = secrets.token_hex(16)


def test_db_upgrade_invokes_upgrade(monkeypatch) -> None:
    revisions: list[str] = []

    monkeypatch.setattr(
        claw_cli.cli_application, "upgrade_database", lambda revision="head": revisions.append(revision)
    )

    result = runner.invoke(claw_cli.cli, ["db", "upgrade"])

    assert result.exit_code == 0
    assert revisions == ["head"]
    assert "Database upgraded to head." in result.output


def test_serve_runs_auto_migrate_with_default_sqlite(monkeypatch, tmp_path) -> None:
    calls: list[tuple[str, object]] = []
    settings = ClawSettings(
        host="127.0.0.1",
        port=9042,
        reload=False,
        auto_migrate=True,
        api_token=TEST_API_TOKEN,
        database_url=None,
        data_dir=tmp_path,
        workspace_dir=tmp_path / "workspace",
    )

    monkeypatch.setattr(claw_cli, "get_settings", lambda: settings)
    monkeypatch.setattr(
        claw_cli.cli_application,
        "upgrade_database",
        lambda revision="head": calls.append(("migrate", revision)),
    )
    monkeypatch.setattr(claw_cli.uvicorn, "run", lambda *args, **kwargs: calls.append(("serve", kwargs)))

    result = runner.invoke(claw_cli.cli, ["serve"])

    assert result.exit_code == 0
    assert calls[0] == ("migrate", "head")
    assert calls[1][0] == "serve"


def test_serve_requires_api_token(monkeypatch, tmp_path) -> None:
    settings = ClawSettings(
        host="127.0.0.1",
        port=9042,
        reload=False,
        auto_migrate=True,
        api_token=None,
        database_url=None,
        data_dir=tmp_path,
        workspace_dir=tmp_path / "workspace",
    )

    monkeypatch.setattr(claw_cli, "get_settings", lambda: settings)

    result = runner.invoke(claw_cli.cli, ["serve"])

    assert result.exit_code == 1
    assert "YA_CLAW_API_TOKEN must be configured before starting YA Claw." in result.output


def test_serve_skips_auto_migrate_when_disabled(monkeypatch, tmp_path) -> None:
    calls: list[tuple[str, object]] = []
    settings = ClawSettings(
        host="127.0.0.1",
        port=9042,
        reload=False,
        auto_migrate=True,
        api_token=TEST_API_TOKEN,
        database_url=None,
        data_dir=tmp_path,
        workspace_dir=tmp_path / "workspace",
    )

    monkeypatch.setattr(claw_cli, "get_settings", lambda: settings)
    monkeypatch.setattr(
        claw_cli.cli_application,
        "upgrade_database",
        lambda revision="head": calls.append(("migrate", revision)),
    )
    monkeypatch.setattr(claw_cli.uvicorn, "run", lambda *args, **kwargs: calls.append(("serve", kwargs)))

    result = runner.invoke(claw_cli.cli, ["serve", "--no-migrate"])

    assert result.exit_code == 0
    assert calls == [
        (
            "serve",
            {
                "factory": True,
                "host": "127.0.0.1",
                "port": 9042,
                "reload": False,
                "log_level": "info",
                "timeout_graceful_shutdown": 30,
            },
        )
    ]


def test_serve_passes_shutdown_timeout_to_uvicorn(monkeypatch, tmp_path) -> None:
    calls: list[tuple[str, object]] = []
    settings = ClawSettings(
        host="127.0.0.1",
        port=9042,
        reload=False,
        auto_migrate=False,
        api_token=TEST_API_TOKEN,
        database_url=None,
        data_dir=tmp_path,
        workspace_dir=tmp_path / "workspace",
        shutdown_timeout_seconds=600,
    )

    monkeypatch.setattr(claw_cli, "get_settings", lambda: settings)
    monkeypatch.setattr(claw_cli.uvicorn, "run", lambda *args, **kwargs: calls.append(("serve", kwargs)))

    result = runner.invoke(claw_cli.cli, ["serve"])

    assert result.exit_code == 0
    assert calls == [
        (
            "serve",
            {
                "factory": True,
                "host": "127.0.0.1",
                "port": 9042,
                "reload": False,
                "log_level": "info",
                "timeout_graceful_shutdown": 600,
            },
        )
    ]


def test_start_runs_migrate_seed_and_serve(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, object]] = []
    seed_file = tmp_path / "profiles.yaml"
    seed_file.write_text("version: 1\nprofiles: []\n", encoding="utf-8")
    settings = ClawSettings(
        host="127.0.0.1",
        port=9042,
        reload=True,
        auto_migrate=True,
        auto_seed_profiles=True,
        api_token=TEST_API_TOKEN,
        database_url=None,
        data_dir=tmp_path,
        workspace_dir=tmp_path / "workspace",
        profile_seed_file=seed_file,
    )

    monkeypatch.setattr(claw_cli, "get_settings", lambda: settings)
    monkeypatch.setattr(
        claw_cli.cli_application,
        "upgrade_database",
        lambda revision="head": calls.append(("migrate", revision)),
    )
    monkeypatch.setattr(
        claw_cli.cli_application,
        "seed_profiles",
        lambda *, prune_missing, migrate, seed_file: (
            calls.append(("seed", (prune_missing, migrate, seed_file))) or ["default"]
        ),
    )
    monkeypatch.setattr(
        claw_cli.cli_application,
        "serve",
        lambda *, host, port, reload, migrate: calls.append(("serve", (host, port, reload, migrate))),
    )

    result = runner.invoke(claw_cli.cli, ["start", "--host", "127.0.0.2", "--port", "9999"])

    assert result.exit_code == 0
    assert calls == [
        ("migrate", "head"),
        ("seed", (False, False, None)),
        ("serve", ("127.0.0.2", 9999, False, False)),
    ]
    assert "Database migrations applied." in result.output
    assert "Seeded 1 profile(s): default" in result.output


def test_start_requires_api_token(monkeypatch, tmp_path: Path) -> None:
    settings = ClawSettings(
        api_token=None,
        data_dir=tmp_path,
        workspace_dir=tmp_path / "workspace",
    )

    monkeypatch.setattr(claw_cli, "get_settings", lambda: settings)

    result = runner.invoke(claw_cli.cli, ["start"])

    assert result.exit_code == 1
    assert "YA_CLAW_API_TOKEN must be configured before starting YA Claw." in result.output


def test_start_reports_missing_seed_file(monkeypatch, tmp_path: Path) -> None:
    settings = ClawSettings(
        auto_migrate=False,
        auto_seed_profiles=True,
        api_token=TEST_API_TOKEN,
        data_dir=tmp_path,
        workspace_dir=tmp_path / "workspace",
        profile_seed_file=tmp_path / "missing.yaml",
    )

    monkeypatch.setattr(claw_cli, "get_settings", lambda: settings)

    result = runner.invoke(claw_cli.cli, ["start"])

    assert result.exit_code == 1
    assert "Profile seed file is not configured or does not exist." in result.output


def test_profiles_seed_command(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[bool, bool, str | None]] = []
    settings = ClawSettings(
        api_token=TEST_API_TOKEN,
        data_dir=tmp_path,
        workspace_dir=tmp_path / "workspace",
        profile_seed_file=tmp_path / "profiles.yaml",
    )

    monkeypatch.setattr(claw_cli, "get_settings", lambda: settings)
    monkeypatch.setattr(
        claw_cli.cli_application,
        "seed_profiles",
        lambda *, prune_missing, migrate, seed_file: calls.append((prune_missing, migrate, seed_file)) or ["default"],
    )

    result = runner.invoke(
        claw_cli.cli, ["profiles", "seed", "--prune-missing", "--seed-file", str(tmp_path / "profiles.yaml")]
    )

    assert result.exit_code == 0
    assert calls == [(True, True, str(tmp_path / "profiles.yaml"))]
    assert "Seeded 1 profile(s): default" in result.output


def test_bridge_ls_command() -> None:
    result = runner.invoke(claw_cli.cli, ["bridge", "ls"])

    assert result.exit_code == 0
    assert "No bridge adapters registered yet." in result.output


def test_bridge_run_command() -> None:
    result = runner.invoke(claw_cli.cli, ["bridge", "run", "lark"])

    assert result.exit_code == 0
    assert "Bridge adapter run requested: lark" in result.output


def test_bridge_serve_command() -> None:
    result = runner.invoke(claw_cli.cli, ["bridge", "serve", "lark"])

    assert result.exit_code == 0
    assert "Bridge adapter service requested: lark" in result.output
