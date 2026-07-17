"""Direct tests for session retention and automatic restore policy."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from yaacli.app.tui import TUIApp
from yaacli.config import SessionConfig, YaacliConfig
from yaacli.sessions import save_session_turn, trim_sessions


class DummyConfigManager:
    def __init__(self, sessions_dir: Path) -> None:
        self._sessions_dir = sessions_dir

    def get_sessions_dir(self) -> Path:
        return self._sessions_dir


def _save_session(
    manager: DummyConfigManager,
    *,
    session_id: str,
    working_dir: Path,
    updated_at: str | None,
) -> Path:
    save_session_turn(
        config_manager=manager,  # type: ignore[arg-type]
        session_id=session_id,
        working_dir=working_dir,
        message_history_json=b"[]",
        context_state_json="{}",
        display_messages=[],
        output_text=None,
        save_reason="test",
        turn_id="turn-1",
        max_turns=20,
        max_sessions=100,
    )
    session_dir = manager.get_sessions_dir() / session_id
    metadata_file = session_dir / "metadata.json"
    metadata = json.loads(metadata_file.read_text())
    if updated_at is None:
        metadata.pop("updated_at", None)
    else:
        metadata["updated_at"] = updated_at
    metadata_file.write_text(json.dumps(metadata))
    return session_dir


def test_max_session_age_prunes_old_sessions_but_protects_current_session(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    manager = DummyConfigManager(sessions_dir)
    old_session = _save_session(
        manager,
        session_id="old-session",
        working_dir=tmp_path,
        updated_at="2000-01-01T00:00:00+00:00",
    )
    current_session = _save_session(
        manager,
        session_id="current-session",
        working_dir=tmp_path,
        updated_at="2000-01-01T00:00:00+00:00",
    )

    trim_sessions(
        sessions_dir,
        max_sessions=100,
        max_session_age_days=30,
        protected_session_id="current-session",
    )

    assert not old_session.exists()
    assert current_session.is_dir()


def test_max_session_age_uses_filesystem_mtime_when_metadata_timestamp_is_missing(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    manager = DummyConfigManager(sessions_dir)
    session_dir = _save_session(
        manager,
        session_id="mtime-session",
        working_dir=tmp_path,
        updated_at=None,
    )
    old_timestamp = datetime(2000, 1, 1, tzinfo=UTC).timestamp()
    os.utime(session_dir, (old_timestamp, old_timestamp))

    trim_sessions(sessions_dir, max_sessions=100, max_session_age_days=30)

    assert not session_dir.exists()


@pytest.mark.asyncio
async def test_auto_restore_selects_newest_session_for_current_workspace(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    current_workspace = tmp_path / "current-workspace"
    other_workspace = tmp_path / "other-workspace"
    current_workspace.mkdir()
    other_workspace.mkdir()
    manager = DummyConfigManager(sessions_dir)
    _save_session(
        manager,
        session_id="current-old",
        working_dir=current_workspace,
        updated_at="2026-01-01T00:00:00+00:00",
    )
    _save_session(
        manager,
        session_id="current-new",
        working_dir=current_workspace,
        updated_at="2026-01-03T00:00:00+00:00",
    )
    _save_session(
        manager,
        session_id="other-newest",
        working_dir=other_workspace,
        updated_at="2026-01-04T00:00:00+00:00",
    )
    config = YaacliConfig(session=SessionConfig(auto_restore=True))
    config_manager = MagicMock()
    config_manager.get_sessions_dir.return_value = sessions_dir
    app = TUIApp(config=config, config_manager=config_manager, working_dir=current_workspace)

    with patch.object(app, "_load_session", new=AsyncMock(return_value=True)) as load_session:
        restored = await app._restore_startup_session()

    assert restored is True
    load_session.assert_awaited_once_with("current-new")
