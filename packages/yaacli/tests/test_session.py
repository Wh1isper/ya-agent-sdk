"""Tests for durable YAACLI session identity and restoration."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ya_agent_sdk.context import ResumableState
from yaacli.config import ConfigManager
from yaacli.durable.application import SessionApplicationService
from yaacli.durable.restoration import restore_resumable_state_safely
from yaacli.durable.sqlite import SQLiteSessionStore
from yaacli.session import TUIContext


def _config() -> MagicMock:
    config = MagicMock()
    config.general.max_requests = 10
    config.display.max_output_lines = 500
    config.display.mouse_support = True
    config.session.auto_restore = False
    return config


def _app_with_store(
    tmp_path: Path,
    *,
    initial_session_id: str | None = None,
):
    from yaacli.app.tui import TUIApp

    config_manager = MagicMock()
    config_manager.get_sessions_dir.return_value = tmp_path / "sessions"
    store = SQLiteSessionStore(tmp_path / "sessions.sqlite3")
    app = TUIApp(
        config=_config(),
        config_manager=config_manager,
        initial_session_id=initial_session_id,
    )
    app._durable_store = store
    app._session_service = SessionApplicationService(store, MagicMock())
    return app, store


def test_restore_resumable_state_preserves_current_approval_policy() -> None:
    """Persisted conversation state must never weaken the current runtime policy."""
    ctx = TUIContext(
        need_user_approve_tools=["shell"],
        need_user_approve_mcps=["filesystem"],
    )
    saved = ResumableState(need_user_approve_tools=[], need_user_approve_mcps=[])

    restore_resumable_state_safely(saved, ctx)

    assert ctx.need_user_approve_tools == ["shell"]
    assert ctx.need_user_approve_mcps == ["filesystem"]


def test_get_sessions_dir(tmp_path: Path) -> None:
    """The legacy directory setting remains the parent for durable databases."""
    config_dir = tmp_path / ".yaacli"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text('model = "anthropic:test"\n')

    manager = ConfigManager(config_dir=config_dir, project_dir=tmp_path)

    assert manager.get_sessions_dir() == config_dir / "sessions"


def test_session_id_is_12_char_hex() -> None:
    from yaacli.app.tui import TUIApp

    config_manager = MagicMock()
    config_manager.get_sessions_dir.return_value = MagicMock(exists=lambda: False)

    app = TUIApp(config=_config(), config_manager=config_manager)

    assert re.fullmatch(r"[0-9a-f]{12}", app.session_id)


def test_session_id_unique_per_instance() -> None:
    from yaacli.app.tui import TUIApp

    config_manager = MagicMock()
    config_manager.get_sessions_dir.return_value = MagicMock(exists=lambda: False)

    first = TUIApp(config=_config(), config_manager=config_manager)
    second = TUIApp(config=_config(), config_manager=config_manager)

    assert first.session_id != second.session_id


@pytest.mark.asyncio
async def test_load_session_exact_match(tmp_path: Path) -> None:
    app, store = _app_with_store(tmp_path)
    try:
        store.create_session(str(tmp_path), session_id="abc123def456")

        restored = await app._load_session("abc123def456")

        assert restored is True
        assert app.session_id == "abc123def456"
        assert any("abc123def456 restored" in line for line in app._output_lines)
    finally:
        store.close()


@pytest.mark.asyncio
async def test_load_session_prefix_match(tmp_path: Path) -> None:
    app, store = _app_with_store(tmp_path)
    try:
        store.create_session(str(tmp_path), session_id="abc123def456")

        restored = await app._load_session("abc1")

        assert restored is True
        assert app.session_id == "abc123def456"
    finally:
        store.close()


@pytest.mark.asyncio
async def test_load_session_ambiguous(tmp_path: Path) -> None:
    app, store = _app_with_store(tmp_path)
    try:
        store.create_session(str(tmp_path), session_id="abc123aaaaaa")
        store.create_session(str(tmp_path), session_id="abc123bbbbbb")

        with patch.object(app, "_append_system_output") as output:
            restored = await app._load_session("abc")

        assert restored is False
        assert any("Ambiguous session prefix" in str(call) for call in output.call_args_list)
    finally:
        store.close()


@pytest.mark.asyncio
async def test_explicit_startup_session_not_found_aborts(tmp_path: Path) -> None:
    app, store = _app_with_store(tmp_path, initial_session_id="missing")
    try:
        with pytest.raises(
            RuntimeError,
            match="Unable to restore requested session: missing",
        ):
            await app._restore_startup_session()
    finally:
        store.close()


@pytest.mark.asyncio
async def test_load_session_not_found(tmp_path: Path) -> None:
    app, store = _app_with_store(tmp_path)
    try:
        with patch.object(app, "_append_system_output") as output:
            restored = await app._load_session("nonexistent")

        assert restored is False
        assert any("not found" in str(call).lower() for call in output.call_args_list)
    finally:
        store.close()
