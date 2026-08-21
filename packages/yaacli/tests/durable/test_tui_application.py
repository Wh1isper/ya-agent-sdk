from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from prompt_toolkit.widgets import TextArea
from pydantic_ai import Tool
from pydantic_ai.capabilities import Toolset as NativeToolsetCapability
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import DeltaToolCall, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset
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


async def test_tui_terminal_replay_preserves_steering_pair_and_interleaved_tool_call(
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
    release_model = asyncio.Event()
    final_messages: list[ModelMessage] = []
    model_calls = 0
    toolset = FunctionToolset[TUIContext](id="visible-tool")

    async def show_value(value: str) -> str:
        return f"visible tool result: {value}"

    toolset.add_tool(Tool(show_value))

    async def stream_response(
        messages: list[ModelMessage],
        _info: Any,
    ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            model_started.set()
            await release_model.wait()
            yield {
                0: DeltaToolCall(
                    name="show_value",
                    json_args='{"value":"kept"}',
                    tool_call_id="visible-tool-call",
                )
            }
            return
        final_messages.extend(messages)
        yield "completed with visible tool history"

    def minimal_runtime_factory(*args: Any, **kwargs: Any):
        del args
        binding_ref = kwargs["durable_binding_ref"]
        return create_agent(
            FunctionModel(stream_function=stream_response),
            capabilities=[
                NativeToolsetCapability(toolset, id="visible-tool"),
                DurableInboxPumpCapability(),
            ],
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
        assert app._launch_agent("run one visible tool") is True
        task = app._agent_task
        assert task is not None
        await asyncio.wait_for(model_started.wait(), timeout=5)
        logical_run_id = app._active_logical_run_id
        assert logical_run_id is not None

        input_area = TextArea(text="keep the tool visible", multiline=True)
        app._submit_input(input_area.text, input_area)

        assert input_area.buffer.text == ""
        assert sum(line.endswith("keep the tool visible") for line in app._output_lines) == 1
        release_model.set()
        await asyncio.wait_for(task, timeout=5)

        assert model_calls == 2
        assert "keep the tool visible" in str(final_messages)
        assert app._durable_store is not None
        revision = app._durable_store.get_revision_for_run(logical_run_id)
        assert revision is not None
        assert revision.terminal["status"] == "completed"
        replay = revision.display_projection
        assert sum(event.get("type") == "CUSTOM" and event.get("name") == "yaacli.user_input" for event in replay) == 1
        assert (
            sum(event.get("type") == "CUSTOM" and event.get("name") == "yaacli.steering_applied" for event in replay)
            == 1
        )
        assert sum(event.get("type") == "TOOL_CALL_CHUNK" for event in replay) == 1
        assert sum(event.get("type") == "TOOL_CALL_RESULT" for event in replay) == 1

        output = "\n".join(app._output_lines)
        assert sum(line.endswith("keep the tool visible") for line in app._output_lines) == 1
        assert output.count("Guidance injected") == 1
        assert sum("Calling:" in line and "show_value" in line for line in app._output_lines) == 1
        assert sum("Complete:" in line and "show_value" in line for line in app._output_lines) == 1
        assert "visible tool result: kept" in output
