"""Tests for session management (save/load/prune/list)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from ya_agent_sdk.context import ResumableState
from yaacli.config import ConfigManager
from yaacli.session import TUIContext
from yaacli.sessions import restore_resumable_state_safely, save_session_turn


def _write_v2_session(session_dir: Path, *, updated_at: str) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    turn_id = "turn-1"
    turn_dir = session_dir / "turns" / turn_id
    turn_dir.mkdir(parents=True)
    (turn_dir / "message_history.json").write_text("[]")
    (turn_dir / "context_state.json").write_text("{}")
    (turn_dir / "display_messages.json").write_text("[]")
    (turn_dir / "metadata.json").write_text(
        json.dumps({"turn_id": turn_id, "session_id": session_dir.name, "updated_at": updated_at})
    )
    (session_dir / "metadata.json").write_text(
        json.dumps({
            "schema_version": 2,
            "session_id": session_dir.name,
            "head_turn_id": turn_id,
            "updated_at": updated_at,
        })
    )


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


# =============================================================================
# ConfigManager.get_sessions_dir Tests
# =============================================================================


def test_get_sessions_dir(tmp_path: Path) -> None:
    """Test get_sessions_dir returns correct path."""
    config_dir = tmp_path / ".yaacli"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text('model = "anthropic:test"\n')

    mgr = ConfigManager(config_dir=config_dir, project_dir=tmp_path)
    sessions_dir = mgr.get_sessions_dir()

    assert sessions_dir == config_dir / "sessions"


# =============================================================================
# Session ID Tests
# =============================================================================


def test_session_id_is_12_char_hex() -> None:
    """Test that session_id is a 12-character hex string."""
    from yaacli.app.tui import TUIApp

    config = MagicMock()
    config.general.max_requests = 10
    config.display.max_output_lines = 500
    config.display.mouse_support = True
    config_manager = MagicMock()
    config_manager.get_sessions_dir.return_value = MagicMock(exists=lambda: False)

    app = TUIApp(config=config, config_manager=config_manager)

    assert len(app.session_id) == 12
    assert re.match(r"^[0-9a-f]{12}$", app.session_id)


def test_save_session_turn_uses_precomputed_message_count(tmp_path: Path) -> None:
    """Saving should not reread and revalidate history when the caller knows its count."""
    config_manager = MagicMock()
    config_manager.get_sessions_dir.return_value = tmp_path / "sessions"

    with patch("yaacli.sessions._read_message_count", side_effect=AssertionError("history was reparsed")):
        turn_dir = save_session_turn(
            config_manager=config_manager,
            session_id="counted-session",
            working_dir=tmp_path,
            message_history_json=b"not parsed by the save path",
            message_count=37,
            context_state_json="{}",
            display_messages=[],
            output_text=None,
            save_reason="test",
        )

    metadata = json.loads((turn_dir / "metadata.json").read_text())
    assert metadata["message_count"] == 37


def test_session_id_unique_per_instance() -> None:
    """Test that each TUIApp instance gets a unique session_id."""
    from yaacli.app.tui import TUIApp

    config = MagicMock()
    config.general.max_requests = 10
    config.display.max_output_lines = 500
    config.display.mouse_support = True
    config_manager = MagicMock()
    config_manager.get_sessions_dir.return_value = MagicMock(exists=lambda: False)

    app1 = TUIApp(config=config, config_manager=config_manager)
    app2 = TUIApp(config=config, config_manager=config_manager)

    assert app1.session_id != app2.session_id


# =============================================================================
# Prune Sessions Tests
# =============================================================================


def test_prune_sessions_no_op_under_limit(tmp_path: Path) -> None:
    """Test prune does nothing when under limit."""
    from yaacli.app.tui import TUIApp

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    # Create 3 sessions
    for i in range(3):
        _write_v2_session(
            sessions_dir / f"session{i:04d}",
            updated_at=f"2026-01-{i + 1:02d}T00:00:00+00:00",
        )

    config = MagicMock()
    config.general.max_requests = 10
    config.display.max_output_lines = 500
    config.display.mouse_support = True
    config_manager = MagicMock()
    config_manager.get_sessions_dir.return_value = sessions_dir

    app = TUIApp(config=config, config_manager=config_manager)
    app._prune_sessions(sessions_dir, max_sessions=5)

    assert len(list(sessions_dir.iterdir())) == 3


def test_prune_sessions_removes_oldest(tmp_path: Path) -> None:
    """Test prune removes oldest sessions when over limit."""
    from yaacli.app.tui import TUIApp

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    # Create 5 sessions with different timestamps
    for i in range(5):
        _write_v2_session(
            sessions_dir / f"session{i:04d}",
            updated_at=f"2026-01-{i + 1:02d}T00:00:00+00:00",
        )

    config = MagicMock()
    config.general.max_requests = 10
    config.display.max_output_lines = 500
    config.display.mouse_support = True
    config_manager = MagicMock()
    config_manager.get_sessions_dir.return_value = sessions_dir

    app = TUIApp(config=config, config_manager=config_manager)
    app._prune_sessions(sessions_dir, max_sessions=3)

    remaining = sorted(d.name for d in sessions_dir.iterdir())
    assert len(remaining) == 3
    # Oldest (session0000, session0001) should be removed
    assert remaining == ["session0002", "session0003", "session0004"]


def test_tui_session_list_uses_confirmed_sessions_only(tmp_path: Path) -> None:
    """The in-TUI list must use the same classifier as CLI and retention."""
    from yaacli.app.tui import TUIApp

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    _write_v2_session(
        sessions_dir / "valid123",
        updated_at="2026-02-01T00:00:00+00:00",
    )
    unknown = sessions_dir / "skills"
    unknown.mkdir()
    (unknown / "metadata.json").write_text(json.dumps({"updated_at": ["invalid", "type"]}))
    config = MagicMock()
    config.general.max_requests = 10
    config.display.max_output_lines = 500
    config.display.mouse_support = True
    config_manager = MagicMock()
    config_manager.get_sessions_dir.return_value = sessions_dir
    app = TUIApp(config=config, config_manager=config_manager)

    app._list_sessions()

    output = "\n".join(app._output_lines)
    assert "valid123" in output
    assert "skills" not in output


def test_prune_sessions_never_deletes_unknown_directories(tmp_path: Path) -> None:
    """Retention counts only confirmed sessions and preserves unrelated config directories."""
    from yaacli.app.tui import TUIApp

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    for name in ("skills", "subagents", "config-cache"):
        unknown_dir = sessions_dir / name
        unknown_dir.mkdir()
        (unknown_dir / "keep.txt").write_text(name)
    for i in range(3):
        _write_v2_session(
            sessions_dir / f"session{i:04d}",
            updated_at=f"2026-02-{i + 1:02d}T00:00:00+00:00",
        )

    config = MagicMock()
    config.general.max_requests = 10
    config.display.max_output_lines = 500
    config.display.mouse_support = True
    config_manager = MagicMock()
    config_manager.get_sessions_dir.return_value = sessions_dir

    app = TUIApp(config=config, config_manager=config_manager)
    app._prune_sessions(sessions_dir, max_sessions=2)

    remaining = {path.name for path in sessions_dir.iterdir()}
    assert {"skills", "subagents", "config-cache"} <= remaining
    assert {"session0001", "session0002"} <= remaining
    assert "session0000" not in remaining


def test_prune_sessions_nonexistent_dir(tmp_path: Path) -> None:
    """Test prune handles non-existent sessions directory."""
    from yaacli.app.tui import TUIApp

    config = MagicMock()
    config.general.max_requests = 10
    config.display.max_output_lines = 500
    config.display.mouse_support = True
    config_manager = MagicMock()

    app = TUIApp(config=config, config_manager=config_manager)
    # Should not raise
    app._prune_sessions(tmp_path / "nonexistent")


def test_save_session_snapshot_includes_extra_usages_for_error_recovery(tmp_path: Path) -> None:
    """Error recovery snapshots should preserve extra_usages in exported state."""
    from yaacli.app.tui import TUIApp

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    config = MagicMock()
    config.general.max_requests = 10
    config.display.max_output_lines = 500
    config.display.mouse_support = True
    config_manager = MagicMock()
    config_manager.get_sessions_dir.return_value = sessions_dir

    app = TUIApp(config=config, config_manager=config_manager)
    app._message_history = [MagicMock()]
    app._runtime = MagicMock()
    state = MagicMock()
    state.model_dump_json.return_value = "{}"
    app._runtime.ctx.export_state.return_value = state

    with patch("yaacli.app.tui.ModelMessagesTypeAdapter.dump_json", return_value=b"[]"):
        saved = app._save_session_snapshot(include_extra_usages=True, save_reason="error")

    assert saved is True
    app._runtime.ctx.export_state.assert_called_once_with(include_extra_usages=True)

    metadata = json.loads((sessions_dir / app.session_id / "metadata.json").read_text())
    assert metadata["session_id"] == app.session_id
    assert metadata["last_save_reason"] == "error"


# =============================================================================
# Load Session Tests
# =============================================================================


@pytest.mark.asyncio
async def test_load_session_exact_match(tmp_path: Path) -> None:
    """Test loading session with exact ID match."""
    from yaacli.app.tui import TUIApp

    sessions_dir = tmp_path / "sessions"
    session_dir = sessions_dir / "abc123def456"
    _write_v2_session(session_dir, updated_at="2026-01-01T00:00:00+00:00")

    config = MagicMock()
    config.general.max_requests = 10
    config.display.max_output_lines = 500
    config.display.mouse_support = True
    config_manager = MagicMock()
    config_manager.get_sessions_dir.return_value = sessions_dir

    app = TUIApp(config=config, config_manager=config_manager)

    with patch.object(app, "_load_history", new_callable=AsyncMock, return_value=True) as mock_load:
        await app._load_session("abc123def456")

    mock_load.assert_awaited_once_with(str(session_dir), target_session_id="abc123def456")


@pytest.mark.asyncio
async def test_load_session_prefix_match(tmp_path: Path) -> None:
    """Test loading session with prefix match."""
    from yaacli.app.tui import TUIApp

    sessions_dir = tmp_path / "sessions"
    session_dir = sessions_dir / "abc123def456"
    _write_v2_session(session_dir, updated_at="2026-01-01T00:00:00+00:00")

    config = MagicMock()
    config.general.max_requests = 10
    config.display.max_output_lines = 500
    config.display.mouse_support = True
    config_manager = MagicMock()
    config_manager.get_sessions_dir.return_value = sessions_dir

    app = TUIApp(config=config, config_manager=config_manager)

    with patch.object(app, "_load_history", new_callable=AsyncMock, return_value=True) as mock_load:
        await app._load_session("abc1")

    mock_load.assert_awaited_once_with(str(session_dir), target_session_id="abc123def456")


@pytest.mark.asyncio
async def test_load_session_ambiguous(tmp_path: Path) -> None:
    """Test loading session with ambiguous prefix."""
    from yaacli.app.tui import TUIApp

    sessions_dir = tmp_path / "sessions"
    first = sessions_dir / "abc123aaaaaa"
    second = sessions_dir / "abc123bbbbbb"
    _write_v2_session(first, updated_at="2026-01-01T00:00:00+00:00")
    _write_v2_session(second, updated_at="2026-01-02T00:00:00+00:00")

    config = MagicMock()
    config.general.max_requests = 10
    config.display.max_output_lines = 500
    config.display.mouse_support = True
    config_manager = MagicMock()
    config_manager.get_sessions_dir.return_value = sessions_dir

    app = TUIApp(config=config, config_manager=config_manager)

    with patch.object(app, "_append_system_output") as mock_output:
        await app._load_session("abc")

    # Should mention "Ambiguous"
    calls = [str(c) for c in mock_output.call_args_list]
    assert any("Ambiguous" in c for c in calls)


@pytest.mark.asyncio
async def test_explicit_startup_session_not_found_aborts(tmp_path: Path) -> None:
    """An explicit --session miss must fail instead of silently creating a new session."""
    from yaacli.app.tui import TUIApp

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    config_manager = MagicMock()
    config_manager.get_sessions_dir.return_value = sessions_dir
    app = TUIApp(config=MagicMock(), config_manager=config_manager, initial_session_id="missing")

    with pytest.raises(RuntimeError, match="Unable to restore requested session: missing"):
        await app._restore_startup_session()


@pytest.mark.asyncio
async def test_explicit_startup_corrupt_session_aborts(tmp_path: Path) -> None:
    """An explicitly requested corrupt session must fail rather than change session IDs."""
    from yaacli.app.tui import TUIApp

    sessions_dir = tmp_path / "sessions"
    corrupt_dir = sessions_dir / "corrupt123"
    corrupt_dir.mkdir(parents=True)
    (corrupt_dir / "message_history.json").write_text("not-json")
    config_manager = MagicMock()
    config_manager.get_sessions_dir.return_value = sessions_dir
    app = TUIApp(config=MagicMock(), config_manager=config_manager, initial_session_id="corrupt123")
    original_id = app.session_id

    with pytest.raises(RuntimeError, match="Unable to restore requested session: corrupt123"):
        await app._restore_startup_session()

    assert app.session_id == original_id


@pytest.mark.asyncio
async def test_load_session_not_found(tmp_path: Path) -> None:
    """Test loading session that doesn't exist."""
    from yaacli.app.tui import TUIApp

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    config = MagicMock()
    config.general.max_requests = 10
    config.display.max_output_lines = 500
    config.display.mouse_support = True
    config_manager = MagicMock()
    config_manager.get_sessions_dir.return_value = sessions_dir

    app = TUIApp(config=config, config_manager=config_manager)

    with patch.object(app, "_append_system_output") as mock_output:
        await app._load_session("nonexistent")

    calls = [str(c) for c in mock_output.call_args_list]
    assert any("not found" in c for c in calls)
