from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
from ya_agent_sdk.agents.main import create_agent
from yaacli.app.state import TUIPhase
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


async def test_tui_durable_cancel_is_a_normal_terminal_result(
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
    model_started = asyncio.Event()

    async def blocked_response(
        _messages: list[ModelMessage],
        _info: Any,
    ) -> AsyncIterator[str]:
        model_started.set()
        await asyncio.Event().wait()
        yield "unreachable"

    def minimal_runtime_factory(*args: Any, **kwargs: Any):
        del args
        binding_ref = kwargs["durable_binding_ref"]
        return create_agent(
            FunctionModel(stream_function=blocked_response),
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
        assert app._launch_agent("cancel this durable turn") is True
        task = app._agent_task
        assert task is not None
        await asyncio.wait_for(model_started.wait(), timeout=5)
        logical_run_id = app._active_logical_run_id
        assert logical_run_id is not None
        assert app._session_service is not None
        assert app._durable_store is not None
        app.runtime.ctx.goal_task = "cancelled goal"
        app.runtime.ctx.goal_iteration = 2

        app._cancel_foreground()
        assert app.phase is TUIPhase.CANCELLING
        task.cancel()  # Exercise the waiter-cancellation race seen from the public cancel path.
        await asyncio.wait_for(task, timeout=5)

        revision = app._durable_store.get_revision_for_run(logical_run_id)
        assert revision is not None
        assert revision.terminal == {
            "status": "cancelled",
            "reason": "user_interrupted",
        }
        assert any("Cancelled · durable cancellation recorded" in line for line in app._output_lines)
        assert not any("[ERROR]" in line or "RuntimeError" in line for line in app._output_lines)
        cancellation_events = [
            event for event in app._display_replay.snapshot() if str(event.get("name", "")).endswith("run_cancelled")
        ]
        assert len(cancellation_events) == 1
        assert app._last_snapshot_saved is True
        assert app._active_logical_run_id is None
        assert app.phase is TUIPhase.IDLE
        assert app.runtime.ctx.goal_active is False
        assert any("[Goal] Cancelled at iteration 2" in line for line in app._output_lines)


async def test_tui_unexpected_worker_cancellation_projects_interrupted_revision(
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
    model_started = asyncio.Event()

    async def blocked_response(
        _messages: list[ModelMessage],
        _info: Any,
    ) -> AsyncIterator[str]:
        model_started.set()
        await asyncio.Event().wait()
        yield "unreachable"

    def minimal_runtime_factory(*args: Any, **kwargs: Any):
        del args
        binding_ref = kwargs["durable_binding_ref"]
        return create_agent(
            FunctionModel(stream_function=blocked_response),
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
        assert app._launch_agent("interrupt this durable turn") is True
        task = app._agent_task
        assert task is not None
        await asyncio.wait_for(model_started.wait(), timeout=5)
        logical_run_id = app._active_logical_run_id
        assert logical_run_id is not None
        assert app._durable_store is not None
        assert app._execution_worker is not None
        run = app._durable_store.get_run(logical_run_id)
        assert run is not None
        execution_task = app._execution_worker.coordinator._tasks[run.execution_id]
        app.runtime.ctx.goal_task = "interrupted goal"
        app.runtime.ctx.goal_iteration = 3

        execution_task.cancel()
        await asyncio.wait_for(task, timeout=5)

        revision = app._durable_store.get_revision_for_run(logical_run_id)
        assert revision is not None
        assert revision.terminal["status"] == "interrupted"
        assert "interrupted before its active segment completed" in str(revision.terminal["reason"])
        assert any("Agent run ended with status interrupted" in line for line in app._output_lines)
        assert not any(str(event.get("name", "")).endswith("run_cancelled") for event in app._display_replay.snapshot())
        assert app._last_snapshot_saved is True
        assert app._active_logical_run_id is None
        assert app.phase is TUIPhase.IDLE
        assert app.runtime.ctx.goal_active is False
        assert any("[Goal] Stopped after an error at iteration 3" in line for line in app._output_lines)


async def test_tui_failed_run_restores_empty_canonical_revision(
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
    partial_streamed = asyncio.Event()
    release_failure = asyncio.Event()

    async def failing_response(
        _messages: list[ModelMessage],
        _info: Any,
    ) -> AsyncIterator[str]:
        yield "uncommitted partial output"
        partial_streamed.set()
        await release_failure.wait()
        raise RuntimeError("model exploded")

    def minimal_runtime_factory(*args: Any, **kwargs: Any):
        del args
        binding_ref = kwargs["durable_binding_ref"]
        return create_agent(
            FunctionModel(stream_function=failing_response),
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
        assert app._launch_agent("fail this durable turn") is True
        task = app._agent_task
        assert task is not None
        await asyncio.wait_for(partial_streamed.wait(), timeout=5)
        logical_run_id = app._active_logical_run_id
        assert logical_run_id is not None
        assert app._durable_store is not None
        app.runtime.ctx.note_manager.set("transient", "not committed")
        app.runtime.ctx.goal_task = "failed goal"
        app.runtime.ctx.goal_iteration = 4

        release_failure.set()
        await asyncio.wait_for(task, timeout=5)

        revision = app._durable_store.get_revision_for_run(logical_run_id)
        assert revision is not None
        assert revision.terminal["status"] == "failed"
        assert revision.terminal["error"] == "model exploded"
        assert revision.resumable_state == {}
        assert revision.display_projection == []
        replay = app._display_replay.snapshot()
        assert sum(event.get("type") == "RUN_ERROR" for event in replay) == 1
        assert not any(event.get("type") in {"RUN_STARTED", "TEXT_MESSAGE_CHUNK"} for event in replay)
        assert app.runtime.ctx.note_manager.get("transient") is None
        assert any("Agent run ended with status failed: model exploded" in line for line in app._output_lines)
        assert app._last_snapshot_saved is True
        assert app._active_logical_run_id is None
        assert app.phase is TUIPhase.IDLE
        assert app.runtime.ctx.goal_active is False
        assert any("[Goal] Stopped after an error at iteration 4" in line for line in app._output_lines)
