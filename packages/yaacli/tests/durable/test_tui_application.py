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

    def reject_historical_descriptor_scan(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("TUI startup must not eagerly scan historical child descriptors")

    startup_logs: list[str] = []
    monkeypatch.setattr("yaacli.app.tui.create_tui_runtime", minimal_runtime_factory)
    monkeypatch.setattr(
        "yaacli.app.tui.FileSubagentExecutionStore.list_referenced_descriptors",
        reject_historical_descriptor_scan,
    )
    monkeypatch.setattr(
        "yaacli.app.tui.logger.info",
        lambda message, *args: startup_logs.append(message % args),
    )
    app = TUIApp(config, manager, working_dir=tmp_path)

    async with app:
        stage_logs = [line for line in startup_logs if line.startswith("Startup stage ")]
        assert [line.split(" completed in ", 1)[0] for line in stage_logs] == [
            "Startup stage runtime-sources",
            "Startup stage durable-store-and-plans",
            "Startup stage active-runtime",
            "Startup stage application-services",
            "Startup stage background-services",
        ]
        assert any(line.startswith("TUI startup completed in ") for line in startup_logs)
        assert await app._restore_startup_session() is False
        app._append_block("historical tool output before completed run")
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
        assert any("historical tool output before completed run" in line for line in app._output_lines)
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
        yield "cancelled partial output"
        model_started.set()
        await asyncio.Event().wait()

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
        app._append_block("historical tool output before cancelled run")
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
        steering_text = "remember this cancelled guidance"
        input_area = TextArea(text=steering_text, multiline=True)
        app._submit_input(input_area.text, input_area)
        assert input_area.buffer.text == ""
        assert sum("Guidance sent to the active run." in line for line in app._output_lines) == 1

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
        history = str(revision.message_history)
        assert history.count("cancel this durable turn") == 1
        assert history.count("cancelled partial output") == 1
        assert (
            sum(
                event.get("type") == "CUSTOM" and event.get("name") == "yaacli.steering_accepted"
                for event in revision.display_projection
            )
            == 1
        )
        assert not any(event.get("name") == "yaacli.steering_applied" for event in revision.display_projection)
        persisted_steering = app._durable_store.list_inputs(logical_run_id)[1]
        assert persisted_steering.state.value == "rejected"
        assert persisted_steering.content == [steering_text]
        assert steering_text not in str(revision.display_projection)
        assert steering_text not in str(app._display_replay.snapshot())
        assert sum("Guidance sent to the active run." in line for line in app._output_lines) == 1
        assert any("historical tool output before cancelled run" in line for line in app._output_lines)
        assert any("cancelled partial output" in line for line in app._output_lines)
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


async def test_tui_cancelled_run_replays_observed_incomplete_tool_call(
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
    tool_started = asyncio.Event()
    toolset = FunctionToolset[TUIContext](id="blocking-tool")

    async def blocking_tool(value: str) -> str:
        tool_started.set()
        await asyncio.Event().wait()
        return value

    toolset.add_tool(Tool(blocking_tool))

    async def stream_response(
        _messages: list[ModelMessage],
        _info: Any,
    ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        yield {
            0: DeltaToolCall(
                name="blocking_tool",
                json_args='{"value":"unfinished"}',
                tool_call_id="unfinished-tool-call",
            )
        }

    def minimal_runtime_factory(*args: Any, **kwargs: Any):
        del args
        binding_ref = kwargs["durable_binding_ref"]
        return create_agent(
            FunctionModel(stream_function=stream_response),
            capabilities=[
                NativeToolsetCapability(toolset, id="blocking-tool"),
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
        assert app._launch_agent("start one blocking tool") is True
        task = app._agent_task
        assert task is not None
        await asyncio.wait_for(tool_started.wait(), timeout=5)
        logical_run_id = app._active_logical_run_id
        assert logical_run_id is not None
        assert any("Calling:" in line and "blocking_tool" in line for line in app._output_lines)

        app._cancel_foreground()
        await asyncio.wait_for(task, timeout=5)

        assert app._durable_store is not None
        revision = app._durable_store.get_revision_for_run(logical_run_id)
        assert revision is not None
        assert revision.terminal == {
            "status": "cancelled",
            "reason": "user_interrupted",
        }
        observed_tool_events = [
            event for event in revision.display_projection if event.get("toolCallName") == "blocking_tool"
        ]
        assert len(observed_tool_events) == 1
        assert observed_tool_events[0]["type"] == "TOOL_CALL_CHUNK"
        observed_tool_call_id = observed_tool_events[0]["toolCallId"]
        assert not any(
            event.get("type") == "TOOL_CALL_RESULT" and event.get("toolCallId") == observed_tool_call_id
            for event in revision.display_projection
        )
        assert any("Calling:" in line and "blocking_tool" in line for line in app._output_lines)

        app._restore_output_from_display_events(revision.display_projection)

        assert any("Calling:" in line and "blocking_tool" in line for line in app._output_lines)
        session_id = app.session_id

    reattached_app = TUIApp(config, manager, working_dir=tmp_path)
    async with reattached_app:
        assert await reattached_app._load_session(session_id) is True
        assert any("Calling:" in line and "blocking_tool" in line for line in reattached_app._output_lines)


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
        yield "interrupted partial output"
        model_started.set()
        await asyncio.Event().wait()

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
        assert str(revision.message_history).count("interrupt this durable turn") == 1
        assert str(revision.message_history).count("interrupted partial output") == 1
        assert any("interrupted partial output" in line for line in app._output_lines)
        assert any("Agent run ended with status interrupted" in line for line in app._output_lines)
        assert not any(str(event.get("name", "")).endswith("run_cancelled") for event in app._display_replay.snapshot())
        assert app._last_snapshot_saved is True
        assert app._active_logical_run_id is None
        assert app.phase is TUIPhase.IDLE
        assert app.runtime.ctx.goal_active is False
        assert any("[Goal] Stopped after an error at iteration 3" in line for line in app._output_lines)


async def test_tui_failed_run_restores_recoverable_state_and_next_turn_history(
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
    model_calls = 0
    continuation_messages: list[ModelMessage] = []

    async def failing_response(
        messages: list[ModelMessage],
        _info: Any,
    ) -> AsyncIterator[str]:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            yield "recoverable partial output"
            partial_streamed.set()
            await release_failure.wait()
            raise RuntimeError("model exploded")
        continuation_messages.extend(messages)
        yield "continued after recovered failure"

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
        app._append_block("historical tool output before failed run")
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
        steering_text = "remember this failed guidance"
        input_area = TextArea(text=steering_text, multiline=True)
        app._submit_input(input_area.text, input_area)
        assert input_area.buffer.text == ""
        assert sum("Guidance sent to the active run." in line for line in app._output_lines) == 1

        release_failure.set()
        await asyncio.wait_for(task, timeout=5)

        revision = app._durable_store.get_revision_for_run(logical_run_id)
        assert revision is not None
        assert revision.terminal["status"] == "failed"
        assert revision.terminal["error"] == "model exploded"
        assert "transient" in str(revision.resumable_state)
        history = str(revision.message_history)
        assert history.count("fail this durable turn") == 1
        assert history.count("recoverable partial output") == 1
        assert (
            sum(
                event.get("type") == "CUSTOM" and event.get("name") == "yaacli.steering_accepted"
                for event in revision.display_projection
            )
            == 1
        )
        assert not any(event.get("name") == "yaacli.steering_applied" for event in revision.display_projection)
        assert any(event.get("type") == "RUN_STARTED" for event in revision.display_projection)
        assert any(
            event.get("type") == "TEXT_MESSAGE_CHUNK" and "recoverable partial output" in str(event.get("delta"))
            for event in revision.display_projection
        )
        assert app._durable_store.list_inputs(logical_run_id)[1].state.value == "rejected"
        replay = app._display_replay.snapshot()
        assert sum(event.get("type") == "RUN_ERROR" for event in replay) == 1
        assert any(event.get("type") == "RUN_STARTED" for event in replay)
        assert any(
            event.get("type") == "TEXT_MESSAGE_CHUNK" and "recoverable partial output" in str(event.get("delta"))
            for event in replay
        )
        assert sum("Guidance sent to the active run." in line for line in app._output_lines) == 1
        assert any("historical tool output before failed run" in line for line in app._output_lines)
        assert any("recoverable partial output" in line for line in app._output_lines)
        assert app.runtime.ctx.note_manager.get("transient") == "not committed"
        assert any("Agent run ended with status failed: model exploded" in line for line in app._output_lines)
        assert app._last_snapshot_saved is True
        assert app._active_logical_run_id is None
        assert app.phase is TUIPhase.IDLE
        assert app.runtime.ctx.goal_active is False
        assert any("[Goal] Stopped after an error at iteration 4" in line for line in app._output_lines)

        assert app._launch_agent("continue after failed turn") is True
        continuation_task = app._agent_task
        assert continuation_task is not None
        await asyncio.wait_for(continuation_task, timeout=5)
        continued = str(continuation_messages)
        assert continued.count("fail this durable turn") == 1
        assert continued.count("recoverable partial output") == 1
        assert continued.count("continue after failed turn") == 1


async def test_tui_failed_run_keeps_observed_tool_history(
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
    model_calls = 0
    toolset = FunctionToolset[TUIContext](id="observed-tool")

    async def show_value(value: str) -> str:
        return f"observed tool result: {value}"

    toolset.add_tool(Tool(show_value))

    async def stream_response(
        _messages: list[ModelMessage],
        _info: Any,
    ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            yield {
                0: DeltaToolCall(
                    name="show_value",
                    json_args='{"value":"kept after failure"}',
                    tool_call_id="observed-tool-call",
                )
            }
            return
        yield "uncommitted output after tool"
        partial_streamed.set()
        await release_failure.wait()
        raise RuntimeError("model failed after visible tool")

    def minimal_runtime_factory(*args: Any, **kwargs: Any):
        del args
        binding_ref = kwargs["durable_binding_ref"]
        return create_agent(
            FunctionModel(stream_function=stream_response),
            capabilities=[
                NativeToolsetCapability(toolset, id="observed-tool"),
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
        assert app._launch_agent("use a tool and then fail") is True
        task = app._agent_task
        assert task is not None
        await asyncio.wait_for(partial_streamed.wait(), timeout=5)
        logical_run_id = app._active_logical_run_id
        assert logical_run_id is not None
        assert any("observed tool result: kept after failure" in line for line in app._output_lines)

        release_failure.set()
        await asyncio.wait_for(task, timeout=5)

        assert app._durable_store is not None
        revision = app._durable_store.get_revision_for_run(logical_run_id)
        assert revision is not None
        assert revision.terminal["status"] == "failed"
        assert any(event.get("type") == "TOOL_CALL_CHUNK" for event in revision.display_projection)
        assert any(event.get("type") == "TOOL_CALL_RESULT" for event in revision.display_projection)
        assert "uncommitted output after tool" in str(revision.message_history)
        output = "\n".join(app._output_lines)
        assert "Calling:" in output
        assert "Complete:" in output
        assert "observed tool result: kept after failure" in output
        assert "uncommitted output after tool" in output
        assert "model failed after visible tool" in output


async def test_tui_failed_run_preserves_native_applied_steering_pair(
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
    first_model_started = asyncio.Event()
    release_first_model = asyncio.Event()
    second_model_started = asyncio.Event()
    release_second_failure = asyncio.Event()
    model_calls = 0

    async def stream_response(
        _messages: list[ModelMessage],
        _info: Any,
    ) -> AsyncIterator[str]:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            first_model_started.set()
            await release_first_model.wait()
            yield "uncommitted output before injected failure"
            return
        second_model_started.set()
        yield "uncommitted second output"
        await release_second_failure.wait()
        raise RuntimeError("model failed after steering injection")

    def minimal_runtime_factory(*args: Any, **kwargs: Any):
        del args
        binding_ref = kwargs["durable_binding_ref"]
        return create_agent(
            FunctionModel(stream_function=stream_response),
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
        assert app._launch_agent("fail after applying guidance") is True
        task = app._agent_task
        assert task is not None
        await asyncio.wait_for(first_model_started.wait(), timeout=5)
        logical_run_id = app._active_logical_run_id
        assert logical_run_id is not None
        assert app._durable_store is not None

        steering_text = "apply this before failing"
        input_area = TextArea(text=steering_text, multiline=True)
        app._submit_input(input_area.text, input_area)
        assert input_area.buffer.text == ""
        release_first_model.set()
        await asyncio.wait_for(second_model_started.wait(), timeout=5)
        for _ in range(100):
            if any(event.get("name") == "yaacli.steering_applied" for event in app._display_replay.snapshot()):
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("Native steering application was not projected")
        assert app._durable_store.list_inputs(logical_run_id)[1].state.value == "applied"
        release_second_failure.set()
        await asyncio.wait_for(task, timeout=5)

        assert model_calls == 2
        revision = app._durable_store.get_revision_for_run(logical_run_id)
        assert revision is not None
        assert revision.terminal["status"] == "failed"
        assert revision.terminal["error"] == "model failed after steering injection"
        assert (
            sum(
                event.get("type") == "CUSTOM" and event.get("name") == "yaacli.steering_accepted"
                for event in revision.display_projection
            )
            == 1
        )
        assert (
            sum(
                event.get("type") == "CUSTOM" and event.get("name") == "yaacli.steering_applied"
                for event in revision.display_projection
            )
            == 1
        )
        assert any(event.get("type") == "RUN_STARTED" for event in revision.display_projection)
        assert any(
            event.get("type") == "TEXT_MESSAGE_CHUNK" and "uncommitted second output" in str(event.get("delta"))
            for event in revision.display_projection
        )
        assert app._durable_store.list_inputs(logical_run_id)[1].state.value == "applied"

        output = "\n".join(app._output_lines)
        assert sum("Guidance sent to the active run." in line for line in app._output_lines) == 1
        assert output.count("Guidance injected") == 1
        assert "uncommitted output before injected failure" in output
        assert "uncommitted second output" in output
        assert "model failed after steering injection" in output


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
        assert sum("Guidance sent to the active run." in line for line in app._output_lines) == 1
        release_model.set()
        await asyncio.wait_for(task, timeout=5)

        assert model_calls == 2
        assert "keep the tool visible" in str(final_messages)
        assert app._durable_store is not None
        revision = app._durable_store.get_revision_for_run(logical_run_id)
        assert revision is not None
        assert revision.terminal["status"] == "completed"
        replay = revision.display_projection
        assert (
            sum(event.get("type") == "CUSTOM" and event.get("name") == "yaacli.steering_accepted" for event in replay)
            == 1
        )
        assert (
            sum(event.get("type") == "CUSTOM" and event.get("name") == "yaacli.steering_applied" for event in replay)
            == 1
        )
        assert sum(event.get("type") == "TOOL_CALL_CHUNK" for event in replay) == 1
        assert sum(event.get("type") == "TOOL_CALL_RESULT" for event in replay) == 1

        output = "\n".join(app._output_lines)
        assert sum("Guidance sent to the active run." in line for line in app._output_lines) == 1
        assert output.count("Guidance injected") == 1
        assert sum("Calling:" in line and "show_value" in line for line in app._output_lines) == 1
        assert sum("Complete:" in line and "show_value" in line for line in app._output_lines) == 1
        assert "visible tool result: kept" in output
