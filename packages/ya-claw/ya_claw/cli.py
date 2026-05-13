from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click
import uvicorn
from alembic import command
from alembic.config import Config
from loguru import logger

from ya_claw.bridge.cli import bridge
from ya_claw.config import ClawSettings, get_settings
from ya_claw.db.engine import create_engine, create_session_factory
from ya_claw.execution.profile import ProfileResolver
from ya_claw.logging import configure_claw_logging, redact_url


class ClawCliApplication:
    def settings(self) -> ClawSettings:
        return get_settings()

    def alembic_config(self) -> Config:
        ini_path = Path(__file__).parent / "alembic.ini"
        return Config(str(ini_path))

    def resolved_database_url(self) -> str:
        settings = self.settings()
        return settings.resolved_database_url

    def upgrade_database(self, revision: str = "head") -> None:
        self.resolved_database_url()
        command.upgrade(self.alembic_config(), revision)

    def downgrade_database(self, revision: str = "-1") -> None:
        self.resolved_database_url()
        command.downgrade(self.alembic_config(), revision)

    def create_revision(self, message: str) -> None:
        self.resolved_database_url()
        command.revision(self.alembic_config(), message=message, autogenerate=True)

    def show_current(self) -> None:
        self.resolved_database_url()
        command.current(self.alembic_config(), verbose=True)

    def show_history(self) -> None:
        self.resolved_database_url()
        command.history(self.alembic_config(), verbose=True)

    def seed_profiles(
        self,
        *,
        prune_missing: bool,
        migrate: bool,
        seed_file: str | None,
    ) -> list[str]:
        settings = self.settings()
        effective_settings = settings.model_copy(
            update={"profile_seed_file": Path(seed_file).expanduser()} if isinstance(seed_file, str) else {}
        )
        resolved_seed_file = effective_settings.resolved_profile_seed_file
        if resolved_seed_file is None or not resolved_seed_file.exists():
            raise click.ClickException("Profile seed file is not configured or does not exist.")

        if migrate:
            self.upgrade_database()

        async def _run() -> list[str]:
            engine = create_engine(effective_settings.resolved_database_url)
            session_factory = create_session_factory(engine)
            try:
                resolver = ProfileResolver(settings=effective_settings, session_factory=session_factory)
                return await resolver.seed_profiles(prune_missing=prune_missing)
            finally:
                await engine.dispose()

        return asyncio.run(_run())

    def serve(
        self,
        host: str | None,
        port: int | None,
        reload: bool | None,
        migrate: bool | None,
        *,
        data_dir: str | None = None,
        sqlite_path: str | None = None,
        workspace_root: str | None = None,
        runtime_lock_file: str | None = None,
        ready_json: bool = False,
    ) -> None:
        self.apply_runtime_overrides(
            host=host,
            port=port,
            data_dir=data_dir,
            sqlite_path=sqlite_path,
            workspace_root=workspace_root,
        )
        settings = self.settings()
        configure_claw_logging(settings.log_level)
        logger.info("YA Claw serve requested")

        try:
            settings.require_api_token()
        except RuntimeError as exc:
            raise click.ClickException(str(exc)) from exc

        resolved_host = settings.host
        resolved_port = settings.port
        resolved_reload = settings.reload if reload is None else reload
        resolved_migrate = settings.auto_migrate if migrate is None else migrate

        logger.info(
            "Resolved serve options host={} port={} reload={} migrate={} log_level={} shutdown_timeout_seconds={}",
            resolved_host,
            resolved_port,
            resolved_reload,
            resolved_migrate,
            settings.log_level,
            settings.shutdown_timeout_seconds,
        )

        if resolved_migrate:
            logger.info("Applying database migrations before serving")
            self.upgrade_database()
            click.echo("Database migrations applied.", err=True)

        if resolved_reload or resolved_port != 0:
            logger.info("Starting uvicorn server")
            uvicorn.run(
                "ya_claw.app:create_app",
                factory=True,
                host=resolved_host,
                port=resolved_port,
                reload=resolved_reload,
                log_level=settings.log_level.lower(),
                timeout_graceful_shutdown=settings.shutdown_timeout_seconds,
            )
            return

        self.serve_with_ready_socket(
            settings=settings,
            host=resolved_host,
            port=resolved_port,
            runtime_lock_file=runtime_lock_file,
            ready_json=ready_json,
        )

    def serve_with_ready_socket(
        self,
        *,
        settings: ClawSettings,
        host: str,
        port: int,
        runtime_lock_file: str | None,
        ready_json: bool,
    ) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.set_inheritable(True)
        actual_host, actual_port = _socket_host_port(sock, fallback_host=host)
        self.apply_runtime_overrides(host=actual_host, port=actual_port)
        settings = self.settings()
        base_url = f"http://{actual_host}:{actual_port}"
        os.environ["YA_CLAW_PUBLIC_BASE_URL"] = base_url
        clear_settings_cache()
        settings = self.settings()

        ready_payload = self.ready_payload(settings=settings, base_url=base_url)
        runtime_lock_path = Path(runtime_lock_file).expanduser() if isinstance(runtime_lock_file, str) else None

        server = _ReadyUvicornServer(
            config=uvicorn.Config(
                "ya_claw.app:create_app",
                factory=True,
                host=actual_host,
                port=actual_port,
                log_level=settings.log_level.lower(),
                timeout_graceful_shutdown=settings.shutdown_timeout_seconds,
            ),
            ready_callback=lambda: self.emit_ready_payload(
                payload=ready_payload,
                runtime_lock_file=runtime_lock_path,
                ready_json=ready_json,
            ),
        )
        try:
            asyncio.run(server.serve(sockets=[sock]))
        finally:
            if runtime_lock_path is not None and runtime_lock_path.exists():
                runtime_lock_path.unlink()

    def emit_ready_payload(
        self,
        *,
        payload: dict[str, Any],
        runtime_lock_file: Path | None,
        ready_json: bool,
    ) -> None:
        encoded_payload = json.dumps(payload, sort_keys=True)
        if runtime_lock_file is not None:
            runtime_lock_file.parent.mkdir(parents=True, exist_ok=True)
            runtime_lock_file.write_text(encoded_payload + "\n", encoding="utf-8")
        if ready_json:
            click.echo(encoded_payload, err=False)
            sys.stdout.flush()

    def ready_payload(self, *, settings: ClawSettings, base_url: str) -> dict[str, Any]:
        return {
            "type": "ya_clawd.ready",
            "pid": os.getpid(),
            "base_url": base_url,
            "version": settings.resolved_service_version,
            "service_revision": settings.resolved_service_revision,
            "instance_id": settings.instance_id,
            "data_dir": str(settings.runtime_data_dir),
            "workspace_dir": str(settings.resolved_workspace_dir),
            "database_url": redact_url(settings.resolved_database_url),
            "created_at": datetime.now(UTC).isoformat(),
        }

    def apply_runtime_overrides(
        self,
        *,
        host: str | None = None,
        port: int | None = None,
        data_dir: str | None = None,
        sqlite_path: str | None = None,
        workspace_root: str | None = None,
    ) -> None:
        if isinstance(host, str) and host.strip() != "":
            os.environ["YA_CLAW_HOST"] = host.strip()
        if isinstance(port, int):
            os.environ["YA_CLAW_PORT"] = str(port)
        if isinstance(data_dir, str) and data_dir.strip() != "":
            os.environ["YA_CLAW_DATA_DIR"] = str(Path(data_dir).expanduser())
        if isinstance(sqlite_path, str) and sqlite_path.strip() != "":
            sqlite_file = Path(sqlite_path).expanduser()
            sqlite_file.parent.mkdir(parents=True, exist_ok=True)
            os.environ["YA_CLAW_DATABASE_URL"] = f"sqlite+aiosqlite:///{sqlite_file.resolve()}"
        if isinstance(workspace_root, str) and workspace_root.strip() != "":
            os.environ["YA_CLAW_WORKSPACE_DIR"] = str(Path(workspace_root).expanduser())
        clear_settings_cache()

    def version_payload(self) -> dict[str, Any]:
        settings = self.settings()
        return {
            "name": "ya-clawd",
            "version": settings.resolved_service_version,
            "service_revision": settings.resolved_service_revision,
            "service_commit": settings.resolved_service_commit,
            "service_build": settings.resolved_service_build,
            "service_image": settings.resolved_service_image,
            "desktop_compatibility": {
                "contract": "claw-desktop.v1",
            },
        }

    def doctor_payload(self) -> dict[str, Any]:
        settings = self.settings()
        data_dir = settings.runtime_data_dir
        workspace_dir = settings.resolved_workspace_dir
        runtime_root = settings.runtime_root
        profile_seed_file = settings.resolved_profile_seed_file
        checks = {
            "api_token": settings.api_token_value is not None,
            "runtime_root_parent_writable": _path_parent_writable(runtime_root),
            "data_dir_parent_writable": _path_parent_writable(data_dir),
            "workspace_dir_parent_writable": _path_parent_writable(workspace_dir),
            "profile_seed_file_exists": profile_seed_file.exists() if profile_seed_file is not None else None,
        }
        return {
            "name": "ya-clawd",
            "ok": all(value for value in checks.values() if isinstance(value, bool)),
            "checks": checks,
            "settings": {
                "environment": settings.environment,
                "host": settings.host,
                "port": settings.port,
                "public_base_url": settings.public_base_url,
                "runtime_root": str(runtime_root),
                "data_dir": str(data_dir),
                "workspace_dir": str(workspace_dir),
                "database_url": redact_url(settings.resolved_database_url),
                "workspace_provider_backend": settings.workspace_provider_backend,
                "bridge_dispatch_mode": settings.bridge_dispatch_mode.value,
                "default_profile": settings.default_profile,
                "profile_seed_file": str(profile_seed_file) if profile_seed_file is not None else None,
            },
        }


class _ReadyUvicornServer(uvicorn.Server):
    def __init__(self, *, config: uvicorn.Config, ready_callback: Any) -> None:
        super().__init__(config=config)
        self._ready_callback = ready_callback
        self._ready_emitted = False

    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        await super().startup(sockets=sockets)
        if not self.should_exit and not self._ready_emitted:
            self._ready_emitted = True
            self._ready_callback()


def clear_settings_cache() -> None:
    cache_clear = getattr(get_settings, "cache_clear", None)
    if callable(cache_clear):
        cache_clear()


def _socket_host_port(sock: socket.socket, *, fallback_host: str) -> tuple[str, int]:
    sockname = sock.getsockname()
    if not isinstance(sockname, tuple) or len(sockname) < 2:
        raise click.ClickException("Failed to resolve bound socket address.")
    raw_host = sockname[0]
    host = raw_host if isinstance(raw_host, str) and raw_host not in {"0.0.0.0", "::"} else fallback_host  # noqa: S104
    return host, int(sockname[1])


def _path_parent_writable(path: Path) -> bool:
    parent = path.expanduser().parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
        return os.access(parent, os.W_OK)
    except OSError:
        return False


cli_application = ClawCliApplication()


@click.group()
def cli() -> None:
    """YA Claw management CLI."""


@cli.command()
@click.option("--host", default=None, help="Bind host for the HTTP server.")
@click.option("--port", default=None, type=int, help="Bind port for the HTTP server. Use 0 for a random port.")
@click.option("--reload/--no-reload", default=None, help="Enable or disable code reload.")
@click.option("--migrate/--no-migrate", default=None, help="Run database migrations before starting the server.")
@click.option("--data-dir", default=None, help="Runtime data directory override.")
@click.option("--sqlite-path", default=None, help="SQLite database file path override.")
@click.option("--workspace-root", default=None, help="Local workspace root directory override.")
@click.option("--runtime-lock-file", default=None, help="Write local runtime ready metadata to this file.")
@click.option("--ready-json/--no-ready-json", default=False, help="Write a ya_clawd.ready JSON line to stdout.")
def serve(
    host: str | None,
    port: int | None,
    reload: bool | None,
    migrate: bool | None,
    data_dir: str | None,
    sqlite_path: str | None,
    workspace_root: str | None,
    runtime_lock_file: str | None,
    ready_json: bool,
) -> None:
    cli_application.serve(
        host=host,
        port=port,
        reload=reload,
        migrate=migrate,
        data_dir=data_dir,
        sqlite_path=sqlite_path,
        workspace_root=workspace_root,
        runtime_lock_file=runtime_lock_file,
        ready_json=ready_json,
    )


@cli.command()
@click.option("--host", default=None, help="Bind host for the HTTP server.")
@click.option("--port", default=None, type=int, help="Bind port for the HTTP server.")
def start(host: str | None, port: int | None) -> None:
    """Production startup: migrate, seed, then serve."""
    cli_application.apply_runtime_overrides(host=host, port=port)
    settings = cli_application.settings()

    try:
        settings.require_api_token()
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    if settings.auto_migrate:
        cli_application.upgrade_database()
        click.echo("Database migrations applied.")

    if settings.auto_seed_profiles:
        seeded_names = cli_application.seed_profiles(prune_missing=False, migrate=False, seed_file=None)
        if seeded_names:
            click.echo(f"Seeded {len(seeded_names)} profile(s): {', '.join(seeded_names)}")

    cli_application.serve(host=host, port=port, reload=False, migrate=False)


@cli.command("migrate")
@click.option("--revision", default="head", help="Target revision (default: head).")
def migrate(revision: str) -> None:
    """Apply database migrations."""
    cli_application.upgrade_database(revision)
    click.echo(f"Database upgraded to {revision}.")


@cli.command("seed-profiles")
@click.option(
    "--prune-missing/--keep-missing",
    default=False,
    help="Delete seeded DB profiles that are missing from the seed file.",
)
@click.option("--migrate/--no-migrate", default=True, help="Run database migrations before seeding profiles.")
@click.option("--seed-file", default=None, help="Override the configured profile seed YAML path.")
def seed_profiles(prune_missing: bool, migrate: bool, seed_file: str | None) -> None:
    """Seed AgentProfile rows from a YAML seed file."""
    seeded_names = cli_application.seed_profiles(
        prune_missing=prune_missing,
        migrate=migrate,
        seed_file=seed_file,
    )
    click.echo(f"Seeded {len(seeded_names)} profile(s): {', '.join(seeded_names)}")


@cli.command("version")
@click.option("--json-output/--text", default=False, help="Write version metadata as JSON.")
def version(json_output: bool) -> None:
    """Show ya-clawd version metadata."""
    payload = cli_application.version_payload()
    if json_output:
        click.echo(json.dumps(payload, sort_keys=True))
        return
    click.echo(payload["service_revision"])


@cli.command("doctor")
@click.option("--json-output/--text", default=False, help="Write diagnostics as JSON.")
def doctor(json_output: bool) -> None:
    """Check local runtime configuration for desktop packaging."""
    payload = cli_application.doctor_payload()
    if json_output:
        click.echo(json.dumps(payload, sort_keys=True))
        return
    click.echo(f"ya-clawd ok={payload['ok']}")
    checks = payload["checks"]
    for name, value in checks.items():
        click.echo(f"{name}: {value}")


@cli.group()
def db() -> None:
    """Database migration and management commands."""


@cli.group()
def profiles() -> None:
    """Profile management commands."""


@profiles.command("seed")
@click.option(
    "--prune-missing/--keep-missing",
    default=False,
    help="Delete seeded DB profiles that are missing from the seed file.",
)
@click.option("--migrate/--no-migrate", default=True, help="Run database migrations before seeding profiles.")
@click.option("--seed-file", default=None, help="Override the configured profile seed YAML path.")
def profiles_seed(prune_missing: bool, migrate: bool, seed_file: str | None) -> None:
    seeded_names = cli_application.seed_profiles(
        prune_missing=prune_missing,
        migrate=migrate,
        seed_file=seed_file,
    )
    click.echo(f"Seeded {len(seeded_names)} profile(s): {', '.join(seeded_names)}")


@db.command("upgrade")
@click.option("--revision", default="head", help="Target revision (default: head).")
def db_upgrade(revision: str) -> None:
    cli_application.upgrade_database(revision)
    click.echo(f"Database upgraded to {revision}.")


@db.command("downgrade")
@click.option("--revision", default="-1", help="Target revision (default: -1, one step back).")
def db_downgrade(revision: str) -> None:
    cli_application.downgrade_database(revision)
    click.echo(f"Database downgraded to {revision}.")


@db.command("revision")
@click.argument("message")
def db_revision(message: str) -> None:
    cli_application.create_revision(message)
    click.echo(f"Migration generated: {message}")


@db.command("current")
def db_current() -> None:
    cli_application.show_current()


@db.command("history")
def db_history() -> None:
    cli_application.show_history()


cli.add_command(bridge)
