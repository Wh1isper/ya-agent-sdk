from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic_ai.models.test import TestModel
from ya_agent_sdk.agents.main import create_agent
from yaacli.app.tui import TUIApp
from yaacli.config import ConfigManager
from yaacli.durable.capabilities import DurableInboxPumpCapability
from yaacli.environment import TUIEnvironment
from yaacli.session import TUIContext


async def test_tui_turn_uses_durable_application_service(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text(
        """
[general]
model = "test"

[session]
auto_restore = false
session_dir = "{session_dir}"
""".format(session_dir=(tmp_path / "sessions").as_posix()),
        encoding="utf-8",
    )
    manager = ConfigManager(config_dir=config_dir)
    config = manager.load()

    def minimal_runtime_factory(*args: Any, **kwargs: Any):
        del args
        binding_ref = kwargs["durable_binding_ref"]
        return create_agent(
            TestModel(call_tools=[], custom_output_text="durable tui answer"),
            capabilities=[DurableInboxPumpCapability()],
            context_type=TUIContext,
            context_kwargs={"durable_binding_ref": binding_ref},
            env=TUIEnvironment,
            env_kwargs={"allowed_paths": [tmp_path], "default_path": tmp_path},
            agent_name="yaacli_main_v2",
        )

    monkeypatch.setattr("yaacli.app.tui.create_tui_runtime", minimal_runtime_factory)
    app = TUIApp(config, manager, working_dir=tmp_path)

    async with app:
        assert await app._restore_startup_session() is False
        assert app._launch_agent("hello durable tui") is True
        task = app._agent_task
        assert task is not None
        await task

        assert app.has_session_data is True
        assert app._message_history
        revision = app._current_revision()
        assert revision.terminal == {
            "status": "completed",
            "output": "durable tui answer",
        }, app._output_lines
        assert app._last_session_output == "durable tui answer"
        assert app._durable_store is not None
        session = app._durable_store.get_session(app.session_id)
        assert session is not None
        assert session.head_revision_id is not None
