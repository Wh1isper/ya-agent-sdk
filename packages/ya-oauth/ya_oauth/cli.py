from __future__ import annotations

import os
import stat
from pathlib import Path

import click

from ya_oauth.codex import CodexOAuthClient, redact_record
from ya_oauth.store import DEFAULT_AUTH_PATH, OAuthStore


@click.group()
def cli() -> None:
    """Manage OAuth credentials for YA model providers."""


@cli.command()
@click.argument("provider", type=click.Choice(["codex"]))
@click.option(
    "--auth-file", type=click.Path(path_type=Path), default=None, help="Auth file path. Defaults to ~/.yaai/auth.json."
)
def login(provider: str, auth_file: Path | None) -> None:
    """Log in to an OAuth provider."""
    store = OAuthStore(auth_file)
    if provider == "codex":
        client = CodexOAuthClient(store=store)
        try:
            device_code = client.request_device_code()
            click.echo("Open this URL in your browser and sign in to ChatGPT:")
            click.echo(device_code.verification_url)
            click.echo("")
            click.echo("Enter this one-time code:")
            click.echo(device_code.user_code)
            click.echo("")
            click.echo("Waiting for browser authorization...")
            token_code = client.poll_device_token(device_code)
            record = client.exchange_device_code(token_code)
            store.set_provider("codex", record)
            email = record.account.email or "unknown account"
            click.echo(f"Logged in to codex as {email}.")
        finally:
            client.close()


@cli.command()
@click.argument("provider", type=click.Choice(["codex"]), required=False)
@click.option(
    "--auth-file", type=click.Path(path_type=Path), default=None, help="Auth file path. Defaults to ~/.yaai/auth.json."
)
def status(provider: str | None, auth_file: Path | None) -> None:
    """Show OAuth provider login status."""
    store = OAuthStore(auth_file)
    auth = store.load()
    provider_names = [provider] if provider else sorted(auth.providers)
    if not provider_names:
        click.echo("No OAuth providers are logged in.")
        return
    for provider_name in provider_names:
        record = auth.providers.get(provider_name)
        if record is None:
            click.echo(f"{provider_name}: not logged in")
            continue
        account = record.account
        identity = account.email or account.chatgpt_user_id or "unknown account"
        plan = f", plan={account.chatgpt_plan_type}" if account.chatgpt_plan_type else ""
        refreshed = record.last_refresh_at.isoformat() if record.last_refresh_at else "never"
        click.echo(f"{provider_name}: logged in as {identity}{plan}, last_refresh_at={refreshed}")


@cli.command(name="refresh")
@click.argument("provider", type=click.Choice(["codex"]))
@click.option(
    "--auth-file", type=click.Path(path_type=Path), default=None, help="Auth file path. Defaults to ~/.yaai/auth.json."
)
def refresh_cmd(provider: str, auth_file: Path | None) -> None:
    """Refresh OAuth credentials."""
    store = OAuthStore(auth_file)
    if provider == "codex":
        client = CodexOAuthClient(store=store)
        try:
            source = client.make_token_source()
            snapshot = _run_sync(source.refresh_token())
            identity = snapshot.account.email or snapshot.account.chatgpt_user_id or "unknown account"
            click.echo(f"Refreshed codex credentials for {identity}.")
        finally:
            client.close()


@cli.command()
@click.argument("provider", type=click.Choice(["codex"]))
@click.option(
    "--auth-file", type=click.Path(path_type=Path), default=None, help="Auth file path. Defaults to ~/.yaai/auth.json."
)
@click.option("--revoke/--no-revoke", default=True, help="Revoke provider tokens before deleting local credentials.")
def logout(provider: str, auth_file: Path | None, revoke: bool) -> None:
    """Log out from an OAuth provider."""
    store = OAuthStore(auth_file)
    record = store.get_provider(provider)
    if record is None:
        click.echo(f"{provider}: not logged in")
        return
    if provider == "codex" and revoke:
        client = CodexOAuthClient(store=store)
        try:
            client.revoke_record(record)
        finally:
            client.close()
    store.delete_provider(provider)
    click.echo(f"Logged out from {provider}.")


@cli.command()
@click.option(
    "--auth-file", type=click.Path(path_type=Path), default=None, help="Auth file path. Defaults to ~/.yaai/auth.json."
)
def doctor(auth_file: Path | None) -> None:
    """Inspect OAuth store health without printing tokens."""
    path = (auth_file or DEFAULT_AUTH_PATH).expanduser()
    store = OAuthStore(path)
    auth = store.load()
    click.echo(f"Auth file: {path}")
    click.echo(f"Providers: {', '.join(sorted(auth.providers)) if auth.providers else 'none'}")
    _print_mode("Directory", path.parent, expected=0o700)
    if path.exists():
        _print_mode("File", path, expected=0o600)
    for provider_name, record in sorted(auth.providers.items()):
        safe_record = redact_record(record)
        account = safe_record.get("account", {})
        click.echo(f"{provider_name}: account={account}")


def _print_mode(label: str, path: Path, *, expected: int) -> None:
    mode = stat.S_IMODE(os.stat(path).st_mode)
    status_text = "ok" if mode == expected else f"expected {expected:o}"
    click.echo(f"{label} mode: {mode:o} ({status_text})")


def _run_sync(awaitable):  # type: ignore[no-untyped-def]
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    raise RuntimeError(f"Cannot run ya-oauth CLI command inside active event loop: {loop}")
