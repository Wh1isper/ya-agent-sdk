"""Durable YAACLI automatic-restore policy tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from yaacli.app.tui import TUIApp
from yaacli.config import SessionConfig, YaacliConfig
from yaacli.durable.application import SessionApplicationService, build_runtime_descriptor
from yaacli.durable.models import RevisionPayload
from yaacli.durable.sqlite import SQLiteSessionStore


async def test_auto_restore_selects_newest_durable_session_for_workspace(tmp_path: Path) -> None:
    current_workspace = tmp_path / "current-workspace"
    other_workspace = tmp_path / "other-workspace"
    current_workspace.mkdir()
    other_workspace.mkdir()
    store = SQLiteSessionStore(tmp_path / "sessions.sqlite3")
    try:
        service = SessionApplicationService(store)
        descriptor = build_runtime_descriptor(agent_spec={"name": "test"})
        for session_id, workspace in (
            ("current-old", current_workspace),
            ("other", other_workspace),
            ("current-new", current_workspace),
        ):
            service.create_session(str(workspace.resolve()), session_id=session_id)
            service.import_snapshot(
                session_id,
                descriptor=descriptor,
                payload=RevisionPayload(),
                source="test",
            )
        app = TUIApp(
            config=YaacliConfig(session=SessionConfig(auto_restore=True)),
            config_manager=MagicMock(),
            working_dir=current_workspace,
        )
        app._durable_store = store
        app._session_service = SessionApplicationService(store, MagicMock())

        with patch.object(app, "_load_session", new=AsyncMock(return_value=True)) as load_session:
            restored = await app._restore_startup_session()

        assert restored is True
        load_session.assert_awaited_once_with("current-new")
    finally:
        store.close()
