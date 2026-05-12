"""Integration tests for TUIApp.

Tests core logic that requires mocking the TUI environment.
Focus on testable components and state transitions.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from prompt_toolkit import Application
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.widgets import TextArea
from pydantic_ai import BinaryContent
from ya_agent_environment.shell import BackgroundProcess

# Import the components we're testing
from yaacli.app import TUIApp, TUIMode, TUIState
from yaacli.app.tui import PendingAttachment, _is_benign_contextvar_cleanup_error
from yaacli.clipboard import ClipboardImage, ClipboardImageReadResult
from yaacli.config import CommandDefinition, GeneralConfig, ModelProfileConfig, YaacliConfig
from yaacli.model_profiles import build_model_profiles


@dataclass
class MockConfig:
    """Minimal mock config for testing."""

    general: Any = field(
        default_factory=lambda: MagicMock(
            max_requests=10,
            mode="act",
        )
    )
    display: Any = field(
        default_factory=lambda: MagicMock(
            max_lines=500,
            mouse=True,
        )
    )
    commands: dict[str, CommandDefinition] = field(default_factory=dict)

    def get_commands(self) -> dict[str, CommandDefinition]:
        return self.commands


@dataclass
class MockConfigManager:
    """Minimal mock config manager for testing."""

    global_config_dir: Any = field(default_factory=lambda: MagicMock())
    project_config_dir: Any = field(default_factory=lambda: MagicMock())
    config_dir: Path = field(default_factory=lambda: Path.cwd() / ".yaacli-test-config")

    def get_sessions_dir(self) -> Any:
        return MagicMock(exists=lambda: False)

    def get_mcp_config(self) -> None:
        return None

    def load_custom_commands(self) -> dict:
        return {}


def _make_contextvar_cleanup_error() -> ValueError:
    return ValueError(
        "<Token var=<ContextVar name='pydantic_ai.current_run_context' default=None at 0x0> "
        "at 0x0> was created in a different Context"
    )


async def _raise_on_cancel(exc: Exception) -> None:
    try:
        await asyncio.sleep(3600)
    except asyncio.CancelledError:
        raise exc from None


async def _sleep_forever() -> None:
    await asyncio.sleep(3600)


async def _ignore_cancel_until_released(release: asyncio.Event) -> None:
    try:
        await asyncio.sleep(3600)
    except asyncio.CancelledError:
        await release.wait()


# =============================================================================
# TUIMode/TUIState Tests
# =============================================================================


def test_tui_mode_values():
    """Test TUIMode enum values."""
    assert TUIMode.ACT.value == "act"
    assert TUIMode.PLAN.value == "plan"


def test_tui_state_values():
    """Test TUIState enum values."""
    assert TUIState.IDLE.value == "idle"
    assert TUIState.RUNNING.value == "running"


def test_tui_mode_is_string():
    """Test that TUIMode values can be used as strings."""
    # TUIMode inherits from str, so .value gives the string
    assert TUIMode.ACT.value == "act"
    assert TUIMode.PLAN.value == "plan"
    # Can compare with string due to str inheritance
    assert TUIMode.ACT == "act"
    assert TUIMode.PLAN == "plan"


# =============================================================================
# TUIApp Initialization Tests
# =============================================================================


def test_tui_app_initial_state():
    """Test TUIApp initial state."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)

    # Check initial state
    assert app.mode == TUIMode.ACT
    assert app.state == TUIState.IDLE
    assert app._agent_phase == "idle"


def test_tui_app_mode_switching():
    """Test mode switching."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)

    # Initial mode
    assert app.mode == TUIMode.ACT

    # Switch to PLAN
    app.switch_mode(TUIMode.PLAN)
    assert app.mode == TUIMode.PLAN

    # Switch back to ACT
    app.switch_mode(TUIMode.ACT)
    assert app.mode == TUIMode.ACT


def test_tui_app_mode_switch_no_change_when_same():
    """Test mode switch does nothing when already in that mode."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)

    # Already in ACT mode
    app.switch_mode(TUIMode.ACT)
    assert app.mode == TUIMode.ACT


# =============================================================================
# Output Management Tests
# =============================================================================


def test_tui_app_output_cache_invalidation():
    """Test output generation counter is bumped on invalidation."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)

    # Initial state - generation is 0
    assert app._output_generation == 0

    # Invalidate cache bumps generation
    app._invalidate_output_cache()
    assert app._output_generation == 1


def test_tui_app_append_output():
    """Test appending output lines."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)

    # Append some lines
    app._append_output("Line 1")
    app._append_output("Line 2")

    assert len(app._output_lines) == 2
    assert app._output_lines[0] == "Line 1"
    assert app._output_lines[1] == "Line 2"
    assert app._output_generation > 0
    assert len(app._block_line_counts) == 2
    assert app._total_line_count == 2


def test_tui_app_output_line_limit():
    """Test output line trimming at max limit."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)
    app._max_output_lines = 10  # Set low limit for testing

    # Add more lines than limit
    for i in range(15):
        app._append_output(f"Line {i}")

    # Should be trimmed to max_output_lines
    assert len(app._output_lines) == 10
    # Oldest lines should be removed
    assert app._output_lines[0] == "Line 5"
    assert app._output_lines[-1] == "Line 14"


def test_tui_app_show_tasks_handles_naive_background_process_timestamp():
    """Background process rendering supports naive timestamps from shell metadata."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)

    runtime = MagicMock()
    runtime.ctx = MagicMock()
    runtime.ctx.task_manager = MagicMock()
    runtime.ctx.task_manager.list_all.return_value = []
    runtime.env = MagicMock()
    runtime.env.resources = {}

    proc = BackgroundProcess(
        process_id="proc-1",
        command="sleep 1",
        cwd=".",
        started_at=datetime.now() - timedelta(seconds=5),
    )
    runtime.env.shell = MagicMock()
    runtime.env.shell.active_background_processes = {proc.process_id: proc}
    app._runtime = runtime

    app._show_tasks()

    output = "\n".join(app._output_lines)
    assert "Background Processes" in output
    assert "proc-1" in output
    assert "running (" in output


# =============================================================================
# Virtual Viewport Tests
# =============================================================================


def test_get_visible_text_single_block():
    """Test _get_visible_text with a single block fully visible."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)
    app._append_output("line1\nline2\nline3")

    # Request all 3 lines
    result = app._get_visible_text(0, 3)
    assert result == "line1\nline2\nline3"


def test_get_visible_text_partial_block():
    """Test _get_visible_text slicing into the middle of a block."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)
    app._append_output("a\nb\nc\nd\ne")  # 5 lines

    # Request lines 1-3 (0-indexed display lines)
    result = app._get_visible_text(1, 4)
    assert result == "b\nc\nd"


def test_get_visible_text_across_blocks():
    """Test _get_visible_text spanning multiple blocks."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)
    app._append_output("block0-line0\nblock0-line1")  # 2 lines
    app._append_output("block1-line0")  # 1 line
    app._append_output("block2-line0\nblock2-line1\nblock2-line2")  # 3 lines

    # Total: 6 lines. Request lines 1-4 (crosses block boundary)
    result = app._get_visible_text(1, 5)
    assert "block0-line1" in result
    assert "block1-line0" in result
    assert "block2-line0" in result
    assert "block2-line1" in result


def test_get_visible_text_empty():
    """Test _get_visible_text with no content."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)
    result = app._get_visible_text(0, 10)
    assert result == ""


def test_get_visible_text_beyond_range():
    """Test _get_visible_text when range exceeds available content."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)
    app._append_output("only-line")

    # Request way beyond what exists
    result = app._get_visible_text(0, 100)
    assert result == "only-line"


def test_update_block_line_count_tracking():
    """Test _update_block maintains correct line counts."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)
    app._append_output("one-line")  # 1 line
    app._append_output("another")  # 1 line

    assert app._total_line_count == 2
    assert app._block_line_counts == [1, 1]

    # Update first block to have 3 lines
    app._update_block(0, "line1\nline2\nline3")
    assert app._block_line_counts[0] == 3
    assert app._total_line_count == 4  # 3 + 1
    assert app._output_lines[0] == "line1\nline2\nline3"


def test_update_block_out_of_range():
    """Test _update_block silently ignores invalid index."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)
    app._append_output("content")

    prev_gen = app._output_generation
    prev_count = app._total_line_count

    # Should not crash or change state
    app._update_block(99, "new-content")
    assert app._output_generation == prev_gen
    assert app._total_line_count == prev_count


def test_append_block_bookkeeping():
    """Test _append_block maintains all counters consistently."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)
    assert app._output_generation == 0
    assert app._total_line_count == 0

    app._append_block("a\nb")  # 2 lines
    assert app._output_generation == 1
    assert app._total_line_count == 2
    assert len(app._block_line_counts) == 1
    assert app._block_line_counts[0] == 2

    app._append_block("c")  # 1 line
    assert app._output_generation == 2
    assert app._total_line_count == 3
    assert len(app._block_line_counts) == 2


def test_output_trimming_preserves_line_counts():
    """Test that trimming old blocks keeps line counts in sync."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)
    app._max_output_lines = 3

    app._append_output("a\nb")  # 2 lines, block count [2]
    app._append_output("c")  # 1 line, block count [2, 1]
    app._append_output("d\ne\nf")  # 3 lines, block count [2, 1, 3] -> trim -> [1, 3]

    # After trim: first block (2 lines) removed
    assert len(app._output_lines) == len(app._block_line_counts)
    assert app._total_line_count == sum(app._block_line_counts)


# =============================================================================
# Streaming Text Tests
# =============================================================================


def test_tui_app_streaming_text_lifecycle():
    """Test streaming text start/update/finalize."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)
    # Mock the prompt_toolkit app with proper output size
    mock_output = MagicMock()
    mock_output.get_size.return_value = MagicMock(columns=80, rows=24)
    app._app = MagicMock(output=mock_output)

    # Start streaming
    app._start_streaming_text("Hello")
    assert app._streaming_text == "Hello"
    assert app._streaming_line_index == 0
    assert len(app._output_lines) == 1

    # Update streaming - this renders markdown so needs proper width
    app._update_streaming_text(" World")
    assert app._streaming_text == "Hello World"

    # Finalize
    app._finalize_streaming_text()
    assert app._streaming_text == ""
    assert app._streaming_line_index is None


def test_tui_app_empty_streaming_text_does_not_append_blank_line():
    """Test empty streaming text waits for content before appending a block."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)
    mock_output = MagicMock()
    mock_output.get_size.return_value = MagicMock(columns=80, rows=24)
    app._app = MagicMock(output=mock_output)

    app._start_streaming_text("")
    assert app._streaming_line_index == 0
    assert app._output_lines == []
    assert app._block_line_counts == []
    assert app._total_line_count == 0

    app._update_streaming_text("Hello")
    assert len(app._output_lines) == 1
    assert app._block_line_counts == [1]
    assert app._total_line_count == 1
    assert "Hello" in app._output_lines[0]

    app._finalize_streaming_text()
    assert app._streaming_text == ""
    assert app._streaming_line_index is None


def test_tui_app_streaming_thinking_lifecycle():
    """Test streaming thinking start/update/finalize."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)
    # Mock the prompt_toolkit app with proper output size
    mock_output = MagicMock()
    mock_output.get_size.return_value = MagicMock(columns=80, rows=24)
    app._app = MagicMock(output=mock_output)

    # Start streaming thinking
    app._start_streaming_thinking("Thinking...")
    assert app._streaming_thinking == "Thinking..."
    assert app._streaming_thinking_line_index == 0

    # Update
    app._update_streaming_thinking(" more")
    assert app._streaming_thinking == "Thinking... more"

    # Finalize
    app._finalize_streaming_thinking()
    assert app._streaming_thinking == ""
    assert app._streaming_thinking_line_index is None


# =============================================================================
# HITL State Tests
# =============================================================================


def test_tui_app_hitl_initial_state():
    """Test HITL initial state."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)

    assert app._hitl_pending is False
    assert app._approval_event is None
    assert app._approval_result is None
    assert len(app._pending_approvals) == 0
    assert app._current_approval_index == 0


def test_tui_app_hitl_reset():
    """Test HITL state reset."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)

    # Set some HITL state
    app._hitl_pending = True
    app._pending_approvals = [MagicMock(), MagicMock()]
    app._current_approval_index = 1
    app._approval_result = True
    # Don't set _approval_event for this test

    # Reset
    app._reset_hitl_state()

    assert app._hitl_pending is False
    assert len(app._pending_approvals) == 0
    assert app._current_approval_index == 0
    # When no event exists, result remains unchanged after reset


def test_tui_app_hitl_reset_with_event():
    """Test HITL state reset when event exists."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)

    # Set HITL state with an event
    app._hitl_pending = True
    app._approval_event = asyncio.Event()
    app._approval_result = True

    # Reset should set result to False and set the event
    app._reset_hitl_state()

    assert app._hitl_pending is False
    assert app._approval_result is False
    assert app._approval_reason == "Cancelled"
    assert app._approval_event is None  # Cleared after reset


# =============================================================================
# Steering Message Tests
# =============================================================================


def test_tui_app_steering_add():
    """Test adding steering messages."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)
    app._app = MagicMock()
    app._state = TUIState.RUNNING  # Only add when running

    app._add_steering_message("Do this instead")

    assert len(app._steering_items) == 1
    _, text, status = app._steering_items[0]
    assert text == "Do this instead"
    assert status == "pending"


def test_tui_app_steering_ack():
    """Test acknowledging steering messages."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)
    app._app = MagicMock()
    app._state = TUIState.RUNNING

    # Add a message
    app._add_steering_message("Do this")

    # Acknowledge it - content_preview must contain the original text
    app._ack_steering_by_content("Please Do this instead")

    _, _, status = app._steering_items[0]
    assert status == "acked"


# =============================================================================
# Subagent State Tests
# =============================================================================


def test_tui_app_subagent_state_tracking():
    """Test subagent state tracking."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)

    # Initially empty
    assert len(app._subagent_states) == 0

    # Add subagent state
    app._subagent_states["sub-1"] = {
        "line_index": 0,
        "tool_names": ["search", "view"],
    }

    assert "sub-1" in app._subagent_states
    assert app._subagent_states["sub-1"]["tool_names"] == ["search", "view"]


# =============================================================================
# Tool Message Tests
# =============================================================================


def test_tui_app_tool_message_tracking():
    """Test tool message tracking."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)

    # Initially empty
    assert len(app._tool_messages) == 0
    assert len(app._printed_tool_calls) == 0


# =============================================================================
# History Tests
# =============================================================================


def test_tui_app_prompt_history():
    """Test prompt history tracking."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)

    # Initially empty
    assert len(app._prompt_history) == 0
    assert app._history_index == -1

    # Add to history
    app._prompt_history.append("First prompt")
    app._prompt_history.append("Second prompt")

    assert len(app._prompt_history) == 2
    assert app._prompt_history[0] == "First prompt"


@pytest.mark.asyncio
async def test_tui_app_slash_commands_are_added_to_prompt_history():
    """Slash commands should be available through prompt history navigation."""
    config = MockConfig(
        commands={
            "commit": CommandDefinition(
                prompt="Create a git commit for the current changes.",
                mode="act",
                description="Commit changes",
            )
        }
    )
    config_manager = MockConfigManager()
    app = TUIApp(config=config, config_manager=config_manager)
    input_area = TextArea(multiline=True)

    async def fake_run_agent(prompt: str, attachments: Any = None) -> None:
        return None

    app._run_agent = fake_run_agent  # type: ignore[method-assign]
    app._submit_input("/commit polish tests", input_area)

    command_task = next(iter(app._managed_tasks))
    await command_task
    if app._agent_task is not None:
        await app._agent_task

    assert app._prompt_history == ["/commit polish tests"]

    key_bindings = app._setup_input_keybindings(input_area)
    handle_up = next(binding.handler for binding in key_bindings.bindings if binding.keys == (Keys.Up,))
    handle_up(MagicMock())

    assert input_area.buffer.text == "/commit polish tests"


@pytest.mark.asyncio
async def test_tui_app_shell_commands_are_added_to_prompt_history():
    """Shell commands should be available through prompt history navigation."""
    config = MockConfig()
    config_manager = MockConfigManager()
    app = TUIApp(config=config, config_manager=config_manager)
    input_area = TextArea(multiline=True)

    async def fake_execute_shell_command(command_str: str) -> None:
        return None

    app._execute_shell_command = fake_execute_shell_command  # type: ignore[method-assign]
    app._submit_input("!git status", input_area)

    command_task = next(iter(app._managed_tasks))
    await command_task

    assert app._prompt_history == ["!git status"]


def test_tui_app_input_keybindings_are_eager():
    """Ensure focused input keys win over TextArea defaults."""
    config = MockConfig()
    config_manager = MockConfigManager()
    app = TUIApp(config=config, config_manager=config_manager)
    input_area = TextArea(multiline=True)

    key_bindings = app._setup_input_keybindings(input_area)
    eager_handlers = {
        tuple(key.value for key in binding.keys): binding.handler.__name__
        for binding in key_bindings.bindings
        if bool(binding.eager())
    }

    assert eager_handlers[("up",)] == "handle_up"
    assert eager_handlers[("c-p",)] == "handle_ctrl_p"
    assert eager_handlers[("down",)] == "handle_down"
    assert eager_handlers[("c-n",)] == "handle_ctrl_n"
    assert eager_handlers[("c-m",)] == "handle_enter"
    assert eager_handlers[("c-j",)] == "handle_ctrl_j"


def test_tui_app_focused_input_keybindings_win_in_application_registry():
    """Ensure prompt_toolkit resolves focused input bindings as active matches."""
    config = MockConfig()
    config_manager = MockConfigManager()
    app = TUIApp(config=config, config_manager=config_manager)
    input_area = TextArea(multiline=True)

    class TestBufferControl(BufferControl):
        pass

    original_control = input_area.control
    input_area.control = TestBufferControl(
        buffer=original_control.buffer,
        input_processors=original_control.input_processors,
        include_default_input_processors=False,
        lexer=original_control.lexer,
        focus_on_click=original_control.focus_on_click,
        key_bindings=app._setup_input_keybindings(input_area),
    )
    input_area.window.content = input_area.control

    layout = Layout(
        HSplit([
            Window(content=FormattedTextControl("output")),
            input_area,
        ]),
        focused_element=input_area,
    )
    pt_app: Application[None] = Application(
        layout=layout,
        key_bindings=app._setup_keybindings(input_area),
        output=DummyOutput(),
    )
    registry = pt_app.key_processor._bindings

    active_handlers = {}
    for key in (Keys.Up, Keys.ControlP, Keys.Down, Keys.ControlN, Keys.ControlM, Keys.ControlJ):
        active = [binding for binding in registry.get_bindings_for_keys((key,)) if binding.filter()]
        active_handlers[key.value] = active[-1].handler.__name__

    assert active_handlers == {
        "up": "handle_up",
        "c-p": "handle_ctrl_p",
        "down": "handle_down",
        "c-n": "handle_ctrl_n",
        "c-m": "handle_enter",
        "c-j": "handle_ctrl_j",
    }


@pytest.mark.asyncio
async def test_tui_app_submit_custom_slash_command():
    """Submitting a custom slash command expands and runs its configured prompt."""
    config = MockConfig(
        commands={
            "commit": CommandDefinition(
                prompt="Create a git commit for the current changes.",
                mode="act",
                description="Commit changes",
            )
        }
    )
    config_manager = MockConfigManager()
    app = TUIApp(config=config, config_manager=config_manager)
    input_area = TextArea(multiline=True)
    captured_prompts: list[str] = []

    async def fake_run_agent(prompt: str, attachments: Any = None) -> None:
        captured_prompts.append(prompt)

    app._run_agent = fake_run_agent  # type: ignore[method-assign]

    app._submit_input("/commit polish tests", input_area)
    assert input_area.buffer.text == ""

    command_task = next(iter(app._managed_tasks))
    await command_task
    assert app._agent_task is not None
    await app._agent_task

    assert captured_prompts == ["Create a git commit for the current changes.\n\nUser instruction: polish tests"]


@pytest.mark.asyncio
async def test_tui_app_run_async_accepts_custom_slash_command_enter() -> None:
    """Ensure prompt_toolkit run_async submits slash commands from terminal input."""
    config = MockConfig(
        commands={
            "commit": CommandDefinition(
                prompt="Create a git commit for the current changes.",
                mode="act",
                description="Commit changes",
            )
        }
    )
    config_manager = MockConfigManager()
    app = TUIApp(config=config, config_manager=config_manager)
    captured_prompts: list[str] = []

    async def fake_run_agent(prompt: str, attachments: Any = None) -> None:
        captured_prompts.append(prompt)
        if app._app:
            app._app.exit()

    app._run_agent = fake_run_agent  # type: ignore[method-assign]

    with create_pipe_input() as pipe_input:
        with create_app_session(input=pipe_input, output=DummyOutput()):
            run_task = asyncio.create_task(app.run())
            await asyncio.sleep(0.1)
            pipe_input.send_text("/commit polish tests\r")
            await asyncio.wait_for(run_task, timeout=2)

    assert captured_prompts == ["Create a git commit for the current changes.\n\nUser instruction: polish tests"]


# =============================================================================
# Session Usage Tests
# =============================================================================


def test_tui_app_session_usage_tracking():
    """Test session usage tracking."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)

    # Initial state
    assert app._session_usage.is_empty()


def test_tui_app_context_token_tracking():
    """Test context token tracking."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)

    # Initial state
    assert app._current_context_tokens == 0
    assert app._context_window_size == 200000

    # Update tokens
    app._current_context_tokens = 5000
    assert app._current_context_tokens == 5000


# =============================================================================
# UI State Tests
# =============================================================================


def test_tui_app_input_mode():
    """Test input mode tracking."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)

    # Default mode
    assert app._input_mode == "send"


def test_tui_app_mouse_enabled():
    """Test mouse mode tracking."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)

    # Default enabled
    assert app._mouse_enabled is True


def test_tui_app_ctrl_c_handling():
    """Test double Ctrl+C exit tracking."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)

    assert app._last_ctrl_c_time == 0.0
    assert app._ctrl_c_exit_timeout == 2.0


def test_detects_benign_contextvar_cleanup_error():
    """Known pydantic-ai cleanup errors should be recognized precisely."""
    assert _is_benign_contextvar_cleanup_error(_make_contextvar_cleanup_error())
    assert not _is_benign_contextvar_cleanup_error(ValueError("plain value error"))
    assert not _is_benign_contextvar_cleanup_error(RuntimeError("wrong type"))


def test_tui_app_task_done_suppresses_benign_contextvar_cleanup_error():
    """Task completion callback should ignore benign cleanup errors."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)
    task = MagicMock()
    task.cancelled.return_value = False
    task.exception.return_value = _make_contextvar_cleanup_error()

    app._on_agent_task_done(task)

    assert app._output_lines == []
    assert app.state == TUIState.IDLE


@pytest.mark.asyncio
async def test_tui_app_run_agent_suppresses_benign_contextvar_cleanup_error():
    """Top-level agent loop should not surface benign cleanup errors to the UI."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)
    runtime = MagicMock()
    runtime.ctx = MagicMock(loop_active=False)
    runtime.ctx.steering_messages = []
    app._runtime = runtime
    app._execute_stream = AsyncMock(side_effect=_make_contextvar_cleanup_error())
    app._check_pending_bus_messages = MagicMock()

    await app._run_agent("hello")

    assert app._output_lines == []
    assert app.state == TUIState.IDLE


@pytest.mark.asyncio
async def test_tui_app_cancel_agent_task_suppresses_benign_contextvar_cleanup_error():
    """Shutdown should absorb the known ContextVar cleanup race."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)
    task = asyncio.create_task(_raise_on_cancel(_make_contextvar_cleanup_error()))
    app._agent_task = task

    await asyncio.sleep(0)
    await app._cancel_agent_task()

    assert task.cancelled() is False
    assert task.done() is True
    assert app._agent_task is None


@pytest.mark.asyncio
async def test_tui_app_cancel_managed_tasks_cleans_up_fire_and_forget_tasks():
    """Shutdown should cancel tracked fire-and-forget tasks."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)
    task = app._track_managed_task(asyncio.create_task(_sleep_forever()))

    await asyncio.sleep(0)
    await app._cancel_managed_tasks()

    assert task.cancelled() is True
    assert app._managed_tasks == set()


@pytest.mark.asyncio
async def test_tui_app_cancel_managed_tasks_retains_timed_out_tasks(monkeypatch: pytest.MonkeyPatch):
    """Shutdown should keep references to managed tasks that outlive the cancellation wait."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)
    release = asyncio.Event()
    task = app._track_managed_task(asyncio.create_task(_ignore_cancel_until_released(release)))
    monkeypatch.setattr("yaacli.app.tui._SHUTDOWN_MANAGED_TASKS_TIMEOUT", 0.001)

    await asyncio.sleep(0)
    await app._cancel_managed_tasks()

    assert task in app._managed_tasks

    release.set()
    await task
    await asyncio.sleep(0)
    assert app._managed_tasks == set()


@pytest.mark.asyncio
async def test_tui_app_cancel_agent_task_retains_timed_out_task(monkeypatch: pytest.MonkeyPatch):
    """Shutdown should keep the agent task reference when cancellation outlives the wait."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)
    release = asyncio.Event()
    task = asyncio.create_task(_ignore_cancel_until_released(release))
    app._agent_task = task
    monkeypatch.setattr("yaacli.app.tui._SHUTDOWN_AGENT_TASK_TIMEOUT", 0.001)

    await asyncio.sleep(0)
    await app._cancel_agent_task()

    assert app._agent_task is task

    release.set()
    await task
    await asyncio.sleep(0)
    assert app._agent_task is None


def test_tui_app_recover_tui_screen_resets_redraws_and_invalidates() -> None:
    """TUI recovery should clear terminal artifacts, reset layout, redraw, and invalidate."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)
    app._screen_recovery_scheduled = True
    app._app = MagicMock()
    app._app.renderer = MagicMock()

    app._recover_tui_screen()

    assert app._screen_recovery_scheduled is False
    app._app.renderer.clear.assert_called_once_with()
    app._app.reset.assert_called_once_with()
    app._app._redraw.assert_called_once_with()
    app._app.invalidate.assert_called_once_with()


def test_tui_app_schedule_tui_recovery_schedules_once() -> None:
    """TUI recovery should be deferred and coalesced."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)
    loop = MagicMock()

    app._schedule_tui_recovery(loop)
    app._schedule_tui_recovery(loop)

    assert app._screen_recovery_scheduled is True
    loop.call_soon.assert_called_once_with(app._recover_tui_screen)


def test_tui_app_build_user_prompt_with_binary_attachment() -> None:
    """Clipboard attachments should become BinaryContent in the user prompt."""
    config = MockConfig()
    config_manager = MockConfigManager()
    app = TUIApp(config=config, config_manager=config_manager)

    prompt = app._build_user_prompt(
        "",
        attachments=[PendingAttachment(data=b"png-bytes", media_type="image/png", size_bytes=9)],
    )

    assert isinstance(prompt, list)
    assert prompt[0] == "Please analyze the attached image."
    assert isinstance(prompt[1], BinaryContent)
    assert prompt[1].data == b"png-bytes"
    assert prompt[1].media_type == "image/png"


@pytest.mark.asyncio
async def test_tui_app_handle_bracketed_paste_inserts_plain_text() -> None:
    """Bracketed paste should always insert plain text into the input buffer."""
    config = MockConfig()
    config_manager = MockConfigManager()
    app = TUIApp(config=config, config_manager=config_manager)
    input_area = TextArea(multiline=True)

    await app._handle_bracketed_paste("hello\r\nworld", input_area)

    assert app._pending_attachments == []
    assert input_area.buffer.text == "hello\nworld"


@pytest.mark.asyncio
async def test_tui_app_paste_clipboard_image_attaches_image() -> None:
    """Explicit image paste should queue clipboard image data as an attachment."""
    config = MockConfig()
    config_manager = MockConfigManager()
    app = TUIApp(config=config, config_manager=config_manager)

    with patch("yaacli.app.tui.read_clipboard_image", new=AsyncMock()) as mock_read:
        mock_read.return_value = ClipboardImageReadResult(
            image=ClipboardImage(data=b"image-bytes", media_type="image/png")
        )
        await app._paste_clipboard_image()

    assert len(app._pending_attachments) == 1
    assert app._pending_attachments[0].data == b"image-bytes"
    assert any("Attached image/png" in line for line in app._output_lines)


def test_tui_app_setup_keybindings_marks_ctrl_v_as_eager() -> None:
    """Ctrl+V image paste binding should outrank prompt_toolkit default handlers."""
    config = MockConfig()
    config_manager = MockConfigManager()
    app = TUIApp(config=config, config_manager=config_manager)
    input_area = TextArea(multiline=True)

    kb = app._setup_keybindings(input_area)
    binding = next(b for b in kb.bindings if b.keys == (Keys.ControlV,))

    assert binding.eager()


@pytest.mark.asyncio
async def test_tui_app_paste_clipboard_image_reports_error() -> None:
    """Explicit image paste should surface clipboard errors."""
    config = MockConfig()
    config_manager = MockConfigManager()
    app = TUIApp(config=config, config_manager=config_manager)

    with patch("yaacli.app.tui.read_clipboard_image", new=AsyncMock()) as mock_read:
        mock_read.return_value = ClipboardImageReadResult(image=None, error="Clipboard unavailable")
        await app._paste_clipboard_image()

    assert app._pending_attachments == []
    assert any("Clipboard unavailable" in line for line in app._output_lines)


@pytest.mark.asyncio
async def test_tui_app_handle_command_paste_image_invokes_clipboard_paste() -> None:
    """Slash command should trigger explicit clipboard image paste."""
    config = MockConfig()
    config_manager = MockConfigManager()
    app = TUIApp(config=config, config_manager=config_manager)

    with patch.object(app, "_paste_clipboard_image", new=AsyncMock()) as mock_paste:
        await app._handle_command_inner("/paste-image")

    mock_paste.assert_awaited_once()
    assert any("/paste-image" in line for line in app._output_lines)


@pytest.mark.asyncio
async def test_tui_app_submit_input_allows_attachment_only_message() -> None:
    """Submitting with clipboard attachments and no text should start an agent turn."""
    config = MockConfig()
    config_manager = MockConfigManager()
    app = TUIApp(config=config, config_manager=config_manager)
    input_area = TextArea(multiline=True)
    attachment = PendingAttachment(data=b"img", media_type="image/png", size_bytes=3)
    app._pending_attachments.append(attachment)
    captured_runs: list[tuple[str, list[PendingAttachment] | None]] = []

    async def fake_run_agent(
        prompt: str,
        attachments: list[PendingAttachment] | None = None,
    ) -> None:
        captured_runs.append((prompt, attachments))

    app._run_agent = fake_run_agent  # type: ignore[method-assign]

    with patch.object(app, "_append_user_input") as mock_append_user_input:
        app._submit_input("", input_area)

    mock_append_user_input.assert_called_once_with("", [attachment])
    assert app._pending_attachments == []
    assert app._agent_task is not None
    await app._agent_task
    assert captured_runs == [("", [attachment])]


def test_tui_app_clear_session_clears_pending_attachments() -> None:
    """Clearing the session should drop queued clipboard images."""
    config = MockConfig()
    config_manager = MockConfigManager()
    app = TUIApp(config=config, config_manager=config_manager)
    app._pending_attachments.append(PendingAttachment(data=b"img", media_type="image/png", size_bytes=3))

    app._clear_session()

    assert app._pending_attachments == []


def test_tui_app_load_history_clears_pending_attachments(tmp_path: Path) -> None:
    """Loading a session should reset queued clipboard images."""
    config = MockConfig()
    config_manager = MockConfigManager()
    app = TUIApp(config=config, config_manager=config_manager)
    app._pending_attachments.append(PendingAttachment(data=b"img", media_type="image/png", size_bytes=3))

    load_dir = tmp_path / "session"
    load_dir.mkdir()
    (load_dir / "message_history.json").write_bytes(b"[]")

    app._load_history(str(load_dir))

    assert app._pending_attachments == []
    assert app._message_history == []


@pytest.mark.asyncio
async def test_tui_app_run_agent_reports_saved_recovery_session():
    """Agent errors should surface recovery guidance when session data is saved."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)
    runtime = MagicMock()
    runtime.ctx = MagicMock(loop_active=False)
    runtime.ctx.steering_messages = []
    app._runtime = runtime
    app._execute_stream = AsyncMock(side_effect=RuntimeError("peer closed connection"))
    app._check_pending_bus_messages = MagicMock()

    with patch.object(TUIApp, "has_session_data", new_callable=PropertyMock, return_value=True):
        await app._run_agent("hello")

    joined_output = "\n".join(app._output_lines)
    assert "Session state saved." in joined_output
    assert f"/session {app.session_id}" in joined_output
    assert app.state == TUIState.IDLE


@pytest.mark.asyncio
async def test_tui_app_model_command_opens_in_tui_selector() -> None:
    """/model should open the embedded TUI selector without terminal dialogs."""
    config = YaacliConfig(
        general=GeneralConfig(model="openai:gpt-4o"),
        model_profiles={
            "sonnet": ModelProfileConfig(label="Sonnet", model="anthropic:claude-sonnet-4-5"),
        },
    )
    config_manager = MockConfigManager()
    app = TUIApp(config=config, config_manager=config_manager)

    await app._handle_command_inner("/model")

    assert app._model_selector_open is True
    assert [profile.id for profile in app._model_selector_profiles] == ["default", "sonnet"]
    assert app._model_selector_index == 0
    assert any("/model" in line for line in app._output_lines)


def test_tui_app_model_selector_movement_wraps() -> None:
    """Embedded model selector movement should stay inside TUI state."""
    config = YaacliConfig(
        general=GeneralConfig(model="openai:gpt-4o"),
        model_profiles={
            "sonnet": ModelProfileConfig(label="Sonnet", model="anthropic:claude-sonnet-4-5"),
            "gemini": ModelProfileConfig(label="Gemini", model="google-gla:gemini-2.5-pro"),
        },
    )
    config_manager = MockConfigManager()
    app = TUIApp(config=config, config_manager=config_manager)
    app._model_selector_open = True
    app._model_selector_profiles = build_model_profiles(config)
    app._model_selector_index = 0

    app._move_model_selector(-1)
    assert app._model_selector_index == 2

    app._move_model_selector(1)
    assert app._model_selector_index == 0


def test_tui_app_model_selector_text_marks_current_and_highlighted_profiles() -> None:
    """Embedded model selector output should mark active and highlighted profiles."""
    config = YaacliConfig(
        general=GeneralConfig(model="openai:gpt-4o"),
        model_profiles={
            "sonnet": ModelProfileConfig(label="Sonnet", model="anthropic:claude-sonnet-4-5"),
        },
    )
    config_manager = MockConfigManager()
    app = TUIApp(config=config, config_manager=config_manager)
    profiles = build_model_profiles(config)
    app._active_model_profile = profiles[1]
    app._model_selector_open = True
    app._model_selector_profiles = profiles
    app._model_selector_index = 1

    rendered = app._get_model_selector_text().value

    assert "Select model profile" in rendered
    assert "> * Sonnet: anthropic:claude-sonnet-4-5" in rendered
    assert "Enter: use" in rendered
