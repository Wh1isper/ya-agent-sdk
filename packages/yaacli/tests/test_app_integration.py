"""Integration tests for TUIApp.

Tests core logic that requires mocking the TUI environment.
Focus on testable components and state transitions.
"""

from __future__ import annotations

import asyncio
import json
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from prompt_toolkit import Application
from prompt_toolkit.application import create_app_session
from prompt_toolkit.completion import Completion
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import ConditionalContainer, FloatContainer, HSplit, Layout, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.widgets import TextArea
from pydantic_ai import BinaryContent, DeferredToolRequests, DeferredToolResults, PartStartEvent, ToolDenied
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import RunUsage
from ya_agent_environment import ShellBackgroundResetError
from ya_agent_environment.shell import BackgroundProcess
from ya_agent_sdk.agents.main import AgentInterrupted
from ya_agent_sdk.context import AvailableSkill, BusMessage, ResumableState, StreamEvent, TaskManager, TaskStatus
from ya_agent_sdk.context.agent import AgentInfo
from ya_agent_sdk.events import TaskEvent

# Import the components we're testing
from yaacli.app import BUSY_CONTROL_COMMANDS, TUIApp, TUIState
from yaacli.app.state import TUIPhase
from yaacli.app.tui import (
    PendingAttachment,
    _BoundedOutputTail,
    _drain_direct_shell_stream,
    _format_direct_shell_truncation_note,
    _format_elapsed_duration,
    _is_benign_contextvar_cleanup_error,
)
from yaacli.background import BackgroundMonitor, BackgroundTaskInfo, BackgroundTaskResult
from yaacli.clipboard import ClipboardImage, ClipboardImageReadResult
from yaacli.config import CommandDefinition, DisplayConfig, GeneralConfig, ModelProfileConfig, YaacliConfig
from yaacli.model_profiles import ResolvedModelProfile, build_model_profiles
from yaacli.session import TUIContext
from yaacli.sessions import SessionInfo, save_session_turn
from yaacli.theme import prompt_toolkit_style_rules


@dataclass
class MockConfig:
    """Minimal mock config for testing."""

    general: object = field(
        default_factory=lambda: MagicMock(
            max_requests=10,
        )
    )
    display: object = field(
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

    global_config_dir: object = field(default_factory=lambda: MagicMock())
    project_config_dir: object = field(default_factory=lambda: MagicMock())
    config_dir: Path = field(default_factory=lambda: Path.cwd() / ".yaacli-test-config")

    def get_sessions_dir(self) -> object:
        return MagicMock(exists=lambda: False)

    def get_mcp_config(self) -> None:
        return None

    def load_custom_commands(self) -> dict:
        return {}


def _session_info(
    session_id: str,
    *,
    working_dir: str = "/workspace",
    input_text: str | None = "last input",
    output_text: str | None = "last output",
    updated_at: str = "2026-01-01T12:34:56+00:00",
) -> SessionInfo:
    return SessionInfo(
        id=session_id,
        path=Path("/sessions") / session_id,
        updated_at=updated_at,
        created_at="2026-01-01T00:00:00+00:00",
        working_dir=working_dir,
        model_profile_id=None,
        model_label=None,
        model=None,
        input_text=input_text,
        output_text=output_text,
        message_count=2,
        display_event_count=3,
        metadata={},
        head_turn_id="turn-1",
        turn_count=1,
    )


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
# TUIState Tests
# =============================================================================


def test_tui_state_values():
    """Test TUIState enum values."""
    assert TUIState.IDLE.value == "idle"
    assert TUIState.RUNNING.value == "running"


# =============================================================================
# TUIApp Initialization Tests
# =============================================================================


def test_tui_app_initial_state():
    """Test TUIApp initial state."""
    config = MockConfig()
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)

    # Check initial state
    assert app.state == TUIState.IDLE
    assert app._agent_phase == "idle"


def test_tui_app_applies_explicit_light_theme_to_all_renderers() -> None:
    config = YaacliConfig(display=DisplayConfig(code_theme="light"))
    app = TUIApp(config=config, config_manager=MockConfigManager())

    renderer = app._event_renderer
    renderer.tracker.start_call("existing", "view")
    renderer.start_thinking("in flight")
    app._configure_theme(query_terminal=False)

    assert app._event_renderer is renderer
    assert "existing" in renderer.tracker.tool_calls
    assert renderer.get_current_thinking() == "in flight"
    assert app._theme.variant == "light"
    assert app._get_code_theme() == "ansi_light"
    assert app._event_renderer._code_theme == "ansi_light"
    assert app._event_renderer._max_tool_result_lines == 5
    assert app._event_renderer._max_arg_length == 100
    assert "bg:ansiwhite" in prompt_toolkit_style_rules(app._theme)["model-selector"]


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


def test_tui_app_show_processes_handles_naive_background_process_timestamp():
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

    app._show_processes()

    output = "\n".join(app._output_lines)
    assert "Background Processes (1 running)" in output
    assert "proc-1" in output
    assert "running (" in output


def test_tui_app_show_agents_separates_subagents_from_processes() -> None:
    """The agents view shows subagent lifecycle data without shell processes."""
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    started_at = datetime.now().astimezone() - timedelta(seconds=5)
    completed_at = started_at + timedelta(seconds=3)
    monitor = MagicMock(spec=BackgroundMonitor)
    monitor.active_tasks = {}
    monitor.task_infos = {
        "executor-bg-1": BackgroundTaskInfo(
            agent_id="executor-bg-1",
            subagent_name="executor",
            prompt="Run focused tests",
            started_at=started_at,
        )
    }
    monitor.task_results = {
        "executor-bg-1": BackgroundTaskResult(
            agent_id="executor-bg-1",
            subagent_name="executor",
            status="completed",
            content="done",
            completed_at=completed_at,
        )
    }
    app._get_background_monitor = MagicMock(return_value=monitor)  # type: ignore[method-assign]

    app._show_agents()

    output = "\n".join(app._output_lines)
    assert "Background Subagents (0 running, 1 finished)" in output
    assert "executor-bg-1" in output
    assert "completed" in output
    assert "Run focused tests" in output
    assert "Background Processes" not in output


def test_tui_app_background_inspection_commands_have_specific_empty_states() -> None:
    agents_app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    processes_app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())

    agents_app._show_agents()
    processes_app._show_processes()

    assert "No background subagents." in agents_app._output_lines[-1]
    assert "No active background processes." in processes_app._output_lines[-1]


@pytest.mark.asyncio
async def test_tui_session_selector_renders_metadata_previews_and_current_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    app._session_id = "current-session"
    entries = [
        _session_info("older-session", working_dir="/workspace/older"),
        _session_info(
            "current-session",
            working_dir="/workspace/current",
            input_text="fix\nthis selector",
            output_text="done\x1b[31m safely",
        ),
    ]
    monkeypatch.setattr("yaacli.app.tui.list_sessions", lambda _: entries)

    await app._show_session_selector()

    assert app._session_selector_open is True
    assert app._session_selector_index == 1
    title = "".join(text for _, text in app._get_session_selector_title())
    rendered = "".join(text for _, text in app._get_session_selector_text())
    assert title == "Sessions · 2"
    assert "Up/Down navigate" in rendered
    assert "SESSION" in rendered
    assert "UPDATED" in rendered
    assert "WORKSPACE" in rendered
    assert "> * current-session" in rendered
    assert "DETAILS  current-session" in rendered
    assert "Directory   /workspace/current" in rendered
    assert "Last input  fix this selector" in rendered
    assert "Last output done [31m safely" in rendered
    assert "\x1b" not in rendered
    assert app._get_session_selector_height() == rendered.count("\n") + 1
    assert any(style == "class:session-selector.selection" for style, _ in app._get_session_selector_text())


def test_tui_session_selector_adapts_columns_and_scroll_hints_to_terminal_width() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    app._session_selector_open = True
    app._session_selector_entries = [
        _session_info(f"session-{index:02d}", working_dir=f"/workspace/{index}") for index in range(12)
    ]
    app._session_selector_index = 6
    app._get_terminal_width = MagicMock(return_value=40)  # type: ignore[method-assign]

    lines = app._session_selector_lines()
    rendered = "\n".join("".join(text for _, text in line) for line in lines)

    assert app._get_session_selector_width() == 36
    assert "UPDATED" in rendered
    assert "WORKSPACE" not in rendered
    assert "newer sessions" in rendered
    assert "older sessions" in rendered
    assert all(len("".join(text for _, text in line)) <= 32 for line in lines)

    app._get_terminal_width = MagicMock(return_value=36)  # type: ignore[method-assign]
    boundary_lines = app._session_selector_lines()
    assert all(len("".join(text for _, text in line)) <= 28 for line in boundary_lines)

    app._get_terminal_width = MagicMock(return_value=200)  # type: ignore[method-assign]
    assert app._get_session_selector_width() == 110


def test_tui_session_selector_fits_short_terminals_and_keeps_selection_visible() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    app._session_selector_open = True
    app._session_selector_entries = [_session_info(f"session-{index:02d}") for index in range(12)]
    app._session_selector_index = 6

    for terminal_height in (20, 13, 8, 5):
        app._get_terminal_height = MagicMock(return_value=terminal_height)  # type: ignore[method-assign]
        lines = app._session_selector_lines()
        selected_lines = [text for line in lines for style, text in line if style == "class:session-selector.selection"]

        assert len(lines) <= terminal_height - 3
        assert any("session-06" in text for text in selected_lines)

    app._get_terminal_height = MagicMock(return_value=20)  # type: ignore[method-assign]
    assert "DETAILS" in "".join(text for _, text in app._get_session_selector_text())
    app._get_terminal_height = MagicMock(return_value=13)  # type: ignore[method-assign]
    assert "DETAILS" not in "".join(text for _, text in app._get_session_selector_text())


def test_tui_session_selector_keybindings_move_wrap_and_load() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    app._session_selector_open = True
    app._session_selector_entries = [_session_info("session-a"), _session_info("session-b")]
    app._session_selector_index = 0
    app._schedule_command = MagicMock()  # type: ignore[method-assign]
    input_area = TextArea(multiline=True)
    key_bindings = app._setup_input_keybindings(input_area)
    handle_up = next(binding.handler for binding in key_bindings.bindings if binding.keys == (Keys.Up,))
    handle_down = next(binding.handler for binding in key_bindings.bindings if binding.keys == (Keys.Down,))
    handle_enter = next(binding.handler for binding in key_bindings.bindings if binding.keys == (Keys.ControlM,))

    handle_up(MagicMock())
    assert app._session_selector_index == 1
    handle_down(MagicMock())
    assert app._session_selector_index == 0

    app._session_selector_index = 1
    handle_enter(MagicMock())

    assert app._session_selector_open is False
    assert app._session_selector_entries == []
    app._schedule_command.assert_called_once_with("/session session-b")

    app._session_selector_open = True
    app._session_selector_entries = [_session_info("session-a")]
    application_bindings = app._setup_keybindings(input_area)
    handle_escape = next(binding.handler for binding in application_bindings.bindings if binding.keys == (Keys.Escape,))
    handle_escape(MagicMock())
    assert app._session_selector_open is False

    app._session_selector_open = True
    app._session_selector_entries = [_session_info("session-a")]
    handle_ctrl_c = next(
        binding.handler for binding in application_bindings.bindings if binding.keys == (Keys.ControlC,)
    )
    handle_ctrl_c(MagicMock())
    assert app._session_selector_open is False


@pytest.mark.asyncio
async def test_tui_session_command_without_id_opens_selector() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    app._show_session_selector = AsyncMock()  # type: ignore[method-assign]

    await app._handle_command_inner("/session")

    app._show_session_selector.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_tui_scheduled_session_command_opens_selector_as_foreground_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    monkeypatch.setattr("yaacli.app.tui.list_sessions", lambda _: [_session_info("session-a")])

    app._schedule_command("/session")
    command_task = app._foreground_command_task
    assert command_task is not None
    await command_task

    assert app._session_selector_open is True
    assert app._session_selector_entries[0].id == "session-a"


@pytest.mark.asyncio
async def test_tui_session_selector_rejects_busy_non_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    list_mock = MagicMock(return_value=[_session_info("session-a")])
    monkeypatch.setattr("yaacli.app.tui.list_sessions", list_mock)
    app._set_phase(TUIPhase.THINKING)

    await app._show_session_selector()

    list_mock.assert_not_called()
    assert app._session_selector_open is False
    assert any("after foreground work finishes" in line for line in app._output_lines)


@pytest.mark.asyncio
async def test_tui_auto_restore_skips_unloadable_shallow_legacy_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = YaacliConfig()
    config.session.auto_restore = True
    app = TUIApp(config=config, config_manager=MockConfigManager(), working_dir=tmp_path)
    candidates = [
        _session_info("broken-newest", working_dir=str(tmp_path), updated_at="2026-01-02T00:00:00+00:00"),
        _session_info("valid-older", working_dir=str(tmp_path), updated_at="2026-01-01T00:00:00+00:00"),
    ]
    monkeypatch.setattr("yaacli.app.tui.list_sessions", lambda _: candidates)
    app._load_session = AsyncMock(side_effect=[False, True])  # type: ignore[method-assign]

    restored = await app._restore_startup_session()

    assert restored is True
    assert [call.args[0] for call in app._load_session.await_args_list] == ["broken-newest", "valid-older"]


@pytest.mark.asyncio
async def test_tui_model_selector_applies_gateway_websocket_responses_profile(monkeypatch, tmp_path: Path) -> None:
    """The model selector should not raise for gateway@openai-responses-ws profiles."""
    monkeypatch.setenv("GATEWAY_API_KEY", "test-key")
    monkeypatch.setenv("GATEWAY_BASE_URL", "https://example.com/v1")
    config = YaacliConfig(
        general=GeneralConfig(model="anthropic:claude-sonnet-4-5"),
        model_profiles={
            "ws": ModelProfileConfig(
                label="Responses WS Gateway",
                model="gateway@openai-responses-ws:gpt-5",
                model_settings="openai_responses_default",
                model_cfg="gpt5_270k",
            )
        },
    )
    config_manager = MockConfigManager(config_dir=tmp_path)
    app = TUIApp(config=config, config_manager=config_manager)

    runtime = MagicMock()
    runtime.agent = MagicMock()
    runtime.ctx = MagicMock()
    runtime.ctx.model_cfg = MagicMock()
    runtime.ctx.get_model_extra_headers.return_value = {"unused": "header"}
    app._runtime = runtime
    app._model_selector_open = True
    app._model_selector_profiles = build_model_profiles(config)
    app._model_selector_index = 1

    await app._apply_model_selector_selection()

    assert runtime.agent.model.model_name == "gpt-5"
    assert runtime.agent.model.provider.name == "openai"
    assert app._active_model_profile == ResolvedModelProfile(
        id="ws",
        label="Responses WS Gateway",
        model="gateway@openai-responses-ws:gpt-5",
        model_settings="openai_responses_default",
        model_cfg="gpt5_270k",
    )
    assert app._model_selector_open is False
    persisted_state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert persisted_state["model_profile"]["selected_profile_id"] == "ws"


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


@pytest.mark.asyncio
@pytest.mark.parametrize("approval_kind", ["approval", "call"])
async def test_tui_app_hitl_cancel_takes_priority_over_approval_input(approval_kind: str) -> None:
    """/cancel must never approve or supply a deferred tool result."""
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    input_area = TextArea(text="/cancel", multiline=True)
    agent_task = asyncio.create_task(_sleep_forever())
    app._agent_task = agent_task
    app._set_phase(TUIPhase.THINKING)
    app._set_phase(TUIPhase.AWAITING_APPROVAL)
    app._hitl_pending = True
    app._approval_kind = approval_kind
    app._approval_event = asyncio.Event()
    key_bindings = app._setup_input_keybindings(input_area)
    handle_enter = next(binding.handler for binding in key_bindings.bindings if binding.keys == (Keys.ControlM,))

    handle_enter(MagicMock())
    await asyncio.sleep(0)

    assert input_area.buffer.text == ""
    assert app.phase == TUIPhase.CANCELLING
    assert app._approval_result is None
    assert app._approval_event.is_set() is False
    with pytest.raises(asyncio.CancelledError):
        await agent_task


@pytest.mark.parametrize("approval_kind", ["approval", "call"])
@pytest.mark.parametrize(
    "command",
    [
        "/agents",
        "/attachments",
        "/integrate",
        "/cost",
        "/help",
        "/paste-image",
        "/perf",
        "/process",
        "/remove-image all",
        "/tool call-1",
    ],
)
def test_tui_app_hitl_safe_busy_commands_do_not_resolve_or_steer(
    approval_kind: str,
    command: str,
) -> None:
    """Safe slash commands retain their command meaning throughout HITL."""
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    input_area = TextArea(text=command, multiline=True)
    app._set_phase(TUIPhase.THINKING)
    app._set_phase(TUIPhase.AWAITING_APPROVAL)
    app._hitl_pending = True
    app._approval_kind = approval_kind
    app._approval_event = asyncio.Event()
    app._schedule_command = MagicMock()  # type: ignore[method-assign]
    app._add_steering_message = MagicMock()  # type: ignore[method-assign]
    key_bindings = app._setup_input_keybindings(input_area)
    handle_enter = next(binding.handler for binding in key_bindings.bindings if binding.keys == (Keys.ControlM,))

    handle_enter(MagicMock())

    app._schedule_command.assert_called_once_with(command)
    app._add_steering_message.assert_not_called()
    assert app._approval_result is None
    assert app._approval_event.is_set() is False
    assert input_area.buffer.text == ""


def test_tui_app_hitl_view_expands_request_and_preserves_shell_review_context() -> None:
    """Real Enter routing for view must show full args, risk, and reviewer reason."""
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    input_area = TextArea(text="v", multiline=True)
    request = ToolCallPart(
        tool_name="shell",
        args={"command": "printf '%s' " + "x" * 600},
        tool_call_id="approval-1",
    )
    app._set_phase(TUIPhase.THINKING)
    app._set_phase(TUIPhase.AWAITING_APPROVAL)
    app._hitl_pending = True
    app._approval_kind = "approval"
    app._approval_event = asyncio.Event()
    app._pending_approvals = [request]
    app._current_deferred_request = request
    app._current_deferred_metadata = {
        "reviewer": "shell_command_reviewer",
        "risk_level": "high",
        "reason": "Command can modify protected files",
    }
    key_bindings = app._setup_input_keybindings(input_area)
    handle_enter = next(binding.handler for binding in key_bindings.bindings if binding.keys == (Keys.ControlM,))

    handle_enter(MagicMock())

    output = "\n".join(app._output_lines)
    assert app._approval_expanded is True
    assert app._approval_event.is_set() is False
    assert "Risk: high" in output
    assert "Reason: Command can modify protected files" in output
    assert "more chars" not in output


def test_tui_app_hitl_nondecision_text_steers_without_resolving_approval() -> None:
    """Real Enter routing must preserve approval while sending ordinary text as steering."""
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    input_area = TextArea(text="focus on the failing test", multiline=True)
    app._set_phase(TUIPhase.THINKING)
    app._set_phase(TUIPhase.AWAITING_APPROVAL)
    app._hitl_pending = True
    app._approval_kind = "approval"
    app._approval_event = asyncio.Event()
    app._add_steering_message = MagicMock()  # type: ignore[method-assign]
    key_bindings = app._setup_input_keybindings(input_area)
    handle_enter = next(binding.handler for binding in key_bindings.bindings if binding.keys == (Keys.ControlM,))

    handle_enter(MagicMock())

    app._add_steering_message.assert_called_once_with("focus on the failing test")
    assert input_area.buffer.text == ""
    assert app._approval_result is None
    assert app._approval_event.is_set() is False
    assert app.phase == TUIPhase.AWAITING_APPROVAL
    assert any("approval is still pending" in line for line in app._output_lines)


@pytest.mark.parametrize(
    ("text", "expected_result", "expected_reason"),
    [
        ("y", True, None),
        ("approve", True, None),
        ("n", False, "User rejected"),
        ("reject unsafe operation", False, "unsafe operation"),
    ],
)
def test_tui_app_hitl_real_enter_accepts_only_explicit_approval_decisions(
    text: str,
    expected_result: bool,
    expected_reason: str | None,
) -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    input_area = TextArea(text=text, multiline=True)
    app._set_phase(TUIPhase.THINKING)
    app._set_phase(TUIPhase.AWAITING_APPROVAL)
    app._hitl_pending = True
    app._approval_kind = "approval"
    app._approval_event = asyncio.Event()
    key_bindings = app._setup_input_keybindings(input_area)
    handle_enter = next(binding.handler for binding in key_bindings.bindings if binding.keys == (Keys.ControlM,))

    handle_enter(MagicMock())

    assert app._approval_result is expected_result
    assert app._approval_reason == expected_reason
    assert app._approval_event.is_set() is True
    assert input_area.buffer.text == ""


def test_tui_app_hitl_empty_enter_keeps_approval_pending() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    input_area = TextArea(multiline=True)
    app._set_phase(TUIPhase.THINKING)
    app._set_phase(TUIPhase.AWAITING_APPROVAL)
    app._hitl_pending = True
    app._approval_kind = "approval"
    app._approval_event = asyncio.Event()
    key_bindings = app._setup_input_keybindings(input_area)
    handle_enter = next(binding.handler for binding in key_bindings.bindings if binding.keys == (Keys.ControlM,))

    handle_enter(MagicMock())

    assert app._approval_result is None
    assert app._approval_event.is_set() is False
    assert app.phase == TUIPhase.AWAITING_APPROVAL
    assert any("Type y to approve" in line for line in app._output_lines)


@pytest.mark.parametrize(
    ("text", "expected_result", "expected_reason"),
    [
        ("provided value", True, "provided value"),
        ("/deny unavailable", False, "unavailable"),
    ],
)
def test_tui_app_hitl_real_enter_handles_deferred_call_results(
    text: str,
    expected_result: bool,
    expected_reason: str,
) -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    input_area = TextArea(text=text, multiline=True)
    app._set_phase(TUIPhase.THINKING)
    app._set_phase(TUIPhase.AWAITING_APPROVAL)
    app._hitl_pending = True
    app._approval_kind = "call"
    app._approval_event = asyncio.Event()
    key_bindings = app._setup_input_keybindings(input_area)
    handle_enter = next(binding.handler for binding in key_bindings.bindings if binding.keys == (Keys.ControlM,))

    handle_enter(MagicMock())

    assert app._approval_result is expected_result
    assert app._approval_reason == expected_reason
    assert app._approval_event.is_set() is True
    assert input_area.buffer.text == ""


def test_tui_app_hitl_unrecognized_slash_text_can_be_a_deferred_call_result() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    input_area = TextArea(text="/home/user/result.json", multiline=True)
    app._set_phase(TUIPhase.THINKING)
    app._set_phase(TUIPhase.AWAITING_APPROVAL)
    app._hitl_pending = True
    app._approval_kind = "call"
    app._approval_event = asyncio.Event()
    app._schedule_command = MagicMock()  # type: ignore[method-assign]
    key_bindings = app._setup_input_keybindings(input_area)
    handle_enter = next(binding.handler for binding in key_bindings.bindings if binding.keys == (Keys.ControlM,))

    handle_enter(MagicMock())

    app._schedule_command.assert_not_called()
    assert app._approval_result is True
    assert app._approval_reason == "/home/user/result.json"
    assert app._approval_event.is_set() is True
    assert input_area.buffer.text == ""


# =============================================================================
# Persistent Task Pane Tests
# =============================================================================


def test_tui_app_task_pane_is_hidden_without_tasks_and_preserves_output_budget() -> None:
    """An empty task pane consumes no rows, leaving the viewport the remaining space."""
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    terminal_size = MagicMock(rows=24, columns=80)
    app._app = MagicMock()
    app._app.output.get_size.return_value = terminal_size

    assert app._get_tasks() == []
    assert app._get_task_height() == 0
    assert app._get_viewport_height() == 19

    task_manager = TaskManager()
    task_manager.create("Active work", "Verify compact layout", active_form="Verifying compact layout")
    runtime = MagicMock()
    runtime.ctx.task_manager = task_manager
    app._runtime = runtime

    assert app._get_task_height() == 1
    assert app._get_viewport_height() == 18
    app._task_pane_expanded = True
    assert app._get_task_height() == 2
    assert app._get_viewport_height() == 17


@pytest.mark.asyncio
async def test_tui_app_real_layout_mounts_hidden_task_pane_and_completion_menu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The run() layout must mount both the conditional pane and visible completion float."""
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    app._theme_terminal_resolved = True
    app._restore_startup_session = AsyncMock(return_value=False)  # type: ignore[method-assign]
    app._cancel_agent_task = AsyncMock()  # type: ignore[method-assign]
    app._cancel_managed_tasks = AsyncMock()  # type: ignore[method-assign]
    captured: dict[str, object] = {}
    prompt_app = MagicMock()
    prompt_app._handle_exception = MagicMock()
    prompt_app.run_async = AsyncMock()

    def capture_application(**kwargs: object) -> MagicMock:
        captured.update(kwargs)
        return prompt_app

    monkeypatch.setattr("yaacli.app.tui.Application", capture_application)

    await app.run()

    layout = captured["layout"]
    assert isinstance(layout, Layout)
    root = layout.container
    assert isinstance(root, FloatContainer)
    task_container = root.content.children[1]
    assert isinstance(task_container, ConditionalContainer)
    assert task_container.filter() is False
    assert isinstance(root.floats[0].content, ConditionalContainer)
    assert isinstance(root.floats[1].content, ConditionalContainer)
    assert root.floats[0].content.filter() is False
    assert root.floats[1].content.filter() is False
    assert any(isinstance(item.content, CompletionsMenu) for item in root.floats)


def test_tui_app_task_pane_shows_task_list_and_statuses() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    task_manager = TaskManager()
    completed = task_manager.create("Finished work", "Already done")
    active = task_manager.create("Implement UI", "Update the task pane", active_form="Implementing UI")
    blocked = task_manager.create("Verify UI", "Run the tests")
    task_manager.create("Write docs", "Update documentation")
    task_manager.update(completed.id, status=TaskStatus.COMPLETED)
    task_manager.update(active.id, status=TaskStatus.IN_PROGRESS)
    task_manager.update(blocked.id, add_blocked_by=[active.id])

    runtime = MagicMock()
    runtime.ctx.task_manager = task_manager
    app._runtime = runtime

    fragments = app._get_task_text()
    rendered = "".join(text for _, text in fragments)

    assert "Tasks: 1/4 done | 1 active | 2 pending" in rendered
    assert "F2: expand" in rendered
    assert "[active]" not in rendered
    assert app._get_task_height() == 1

    input_area = TextArea(multiline=True)
    key_bindings = app._setup_keybindings(input_area)
    handle_f2 = next(binding.handler for binding in key_bindings.bindings if binding.keys == (Keys.F2,))
    handle_f2(MagicMock())

    assert app._task_pane_expanded is True
    expanded = "".join(text for _, text in app._get_task_text())
    assert "[active] #2 Implementing UI" in expanded
    assert "[blocked] #3 Verify UI (by #2)" in expanded
    assert "[pending] #4 Write docs" in expanded
    assert "[done] #1 Finished work" in expanded
    assert app._get_task_height() == 5

    handle_f2(MagicMock())
    assert app._task_pane_expanded is False
    assert app._get_task_height() == 1


def test_tui_app_task_pane_limits_completed_history() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    task_manager = TaskManager()
    for index in range(12):
        task = task_manager.create(f"Completed {index + 1}", "Done")
        task_manager.update(task.id, status=TaskStatus.COMPLETED)

    runtime = MagicMock()
    runtime.ctx.task_manager = task_manager
    app._runtime = runtime

    rendered = "".join(text for _, text in app._get_task_text())

    assert "10 hidden" in rendered
    assert "[done]" not in rendered
    assert app._get_task_height() == 1

    app._task_pane_expanded = True
    expanded = "".join(text for _, text in app._get_task_text())
    assert "10 hidden" in expanded
    assert "[done] #11 Completed 11" in expanded
    assert "[done] #12 Completed 12" in expanded
    assert "Completed 10" not in expanded
    assert app._get_task_height() == 3


def test_tui_app_task_event_updates_pane_without_appending_panel() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())

    app._handle_stream_event(
        StreamEvent(
            agent_id="main",
            agent_name="main",
            event=TaskEvent(event_id="task-1"),
        )
    )

    assert app._output_lines == []


# =============================================================================
# Steering Message Tests
# =============================================================================


def test_tui_app_steering_sends_without_dedicated_pane() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    runtime = MagicMock()
    app._runtime = runtime

    app._add_steering_message("Do this instead")

    runtime.ctx.send_message.assert_called_once()
    message = runtime.ctx.send_message.call_args.args[0]
    assert isinstance(message, BusMessage)
    assert message.content == "Do this instead"
    assert not hasattr(app, "_steering_items")


def test_tui_app_status_counts_only_unconsumed_user_steering() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    ctx = TUIContext()
    ctx.message_bus.subscribe(ctx.agent_id)
    ctx.message_bus.send(BusMessage(id="user-1", content="first secret", source="user", target=ctx.agent_id))
    ctx.message_bus.send(BusMessage(id="background-1", content="result", source="subagent-1", target=ctx.agent_id))
    ctx.message_bus.send(BusMessage(id="user-2", content="second secret", source="user", target=ctx.agent_id))
    app._runtime = MagicMock(ctx=ctx)

    pending_status = "".join(text for _style, text in app._get_status_text())

    assert app._get_pending_steering_count() == 2
    assert "steering 2 pending" in pending_status
    assert "first secret" not in pending_status
    assert "second secret" not in pending_status

    assert ctx.message_bus.mark_consumed(ctx.agent_id, {"user-1"}) == 1
    assert app._get_pending_steering_count() == 1
    assert "steering 1 pending" in "".join(text for _style, text in app._get_status_text())

    assert ctx.message_bus.mark_consumed(ctx.agent_id, {"user-2"}) == 1
    ctx.steering_messages.extend(["first secret", "second secret"])
    applied_status = "".join(text for _style, text in app._get_status_text())
    assert app._get_pending_steering_count() == 0
    assert "steering" not in applied_status
    assert [message.id for message in ctx.message_bus.peek(ctx.agent_id)] == ["background-1"]


def test_tui_app_terminal_cleanup_discards_user_bus_messages_but_preserves_background() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    ctx = TUIContext(steering_messages=["unconsumed guidance"])
    ctx.message_bus.subscribe(ctx.agent_id)
    ctx.message_bus.send(BusMessage(id="user-1", content="guidance", source="user", target=ctx.agent_id))
    ctx.message_bus.send(BusMessage(id="background-1", content="result", source="subagent-1", target=ctx.agent_id))
    app._runtime = MagicMock(ctx=ctx)

    assert app._get_pending_steering_count() == 1
    app._clear_unconsumed_user_steering()

    assert app._get_pending_steering_count() == 0
    assert ctx.steering_messages == []
    assert [message.id for message in ctx.message_bus.peek(ctx.agent_id)] == ["background-1"]
    assert [message.id for message in ctx.consume_messages()] == ["background-1"]


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
                description="Commit changes",
            )
        }
    )
    config_manager = MockConfigManager()
    app = TUIApp(config=config, config_manager=config_manager)
    input_area = TextArea(multiline=True)

    async def fake_run_agent(prompt: str, attachments: object = None) -> None:
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


def test_direct_shell_tail_is_bounded_and_reports_truncation() -> None:
    """Direct shell output keeps only its tail and makes discarded bytes visible."""
    tail = _BoundedOutputTail(max_bytes=8)
    tail.append(b"discard-me")
    tail.append(b"-tail")

    assert tail.retained_bytes == 8
    assert tail.total_bytes == len(b"discard-me-tail")
    assert tail.truncated is True
    assert tail.text() == b"discard-me-tail"[-8:].decode()

    note = _format_direct_shell_truncation_note(tail, stream_name="stdout")
    assert "stdout truncated for diagnostics" in note
    assert "last 8 of 15 streamed bytes" in note


@pytest.mark.asyncio
async def test_direct_shell_stream_decodes_split_utf8_and_flushes_partial_line() -> None:
    """Split code points and final non-newline output should be emitted exactly once."""
    stream = asyncio.StreamReader()
    stream.feed_data(b"price: \xe2")
    stream.feed_data(b"\x82\xac\npartial")
    stream.feed_eof()
    tail = _BoundedOutputTail(max_bytes=1024)
    chunks: list[str] = []

    await _drain_direct_shell_stream(stream, tail, chunks.append)

    assert chunks == ["price: €\n", "partial"]
    assert tail.text() == "price: €\npartial"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group semantics")
@pytest.mark.asyncio
async def test_tui_app_direct_shell_terminates_the_entire_posix_process_group() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    process = MagicMock(pid=4321, returncode=None)
    process.wait = AsyncMock(return_value=0)

    with patch("yaacli.app.tui.os.killpg") as killpg:
        await app._terminate_direct_shell_process(process)

    assert [item.args for item in killpg.call_args_list] == [
        (4321, signal.SIGTERM),
        (4321, signal.SIGKILL),
    ]
    process.wait.assert_awaited_once()


@pytest.mark.asyncio
async def test_tui_app_direct_shell_timeout_terminates_and_releases_foreground(monkeypatch) -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    stdout = asyncio.StreamReader()
    stderr = asyncio.StreamReader()
    stdout.feed_eof()
    stderr.feed_eof()
    process = MagicMock(stdout=stdout, stderr=stderr, pid=4321, returncode=None)

    async def wait_forever() -> int:
        await asyncio.sleep(3600)
        return 0

    process.wait = AsyncMock(side_effect=wait_forever)
    terminate = AsyncMock()
    app._terminate_direct_shell_process = terminate  # type: ignore[method-assign]
    monkeypatch.setattr("yaacli.app.tui._DIRECT_SHELL_TIMEOUT", 0.01)

    with patch("yaacli.app.tui.asyncio.create_subprocess_shell", new=AsyncMock(return_value=process)):
        await app._execute_shell_command("sleep forever")

    terminate.assert_awaited_once_with(process)
    assert app._direct_shell_task is None
    assert app._direct_shell_command is None
    assert app.phase == TUIPhase.IDLE
    assert any("Command timed out (0s)" in line for line in app._output_lines)


@pytest.mark.asyncio
async def test_tui_app_direct_shell_drains_large_stdout_and_stderr_with_bounded_tails(monkeypatch) -> None:
    """Large direct-command streams finish, preserve tails, and report truncation."""
    monkeypatch.setattr("yaacli.app.tui._DIRECT_SHELL_OUTPUT_TAIL_BYTES", 256)
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    script = (
        "import sys; "
        "sys.stdout.buffer.write(b'OUT-START' + b'x' * 131072 + b'OUT-END\\n'); "
        "sys.stdout.flush(); "
        "sys.stderr.buffer.write(b'ERR-START' + b'y' * 131072 + b'ERR-END\\n'); "
        "sys.stderr.flush(); "
        "sys.exit(7)"
    )
    command = (
        subprocess.list2cmdline([sys.executable, "-c", script])
        if sys.platform == "win32"
        else shlex.join([sys.executable, "-c", script])
    )

    await asyncio.wait_for(app._execute_shell_command(command), timeout=10)

    output = "\n".join(app._output_lines)
    streamed_output = "\n".join(app._output_lines[1:])
    stdout_note = next(block for block in app._output_lines if "stdout truncated" in block)
    stderr_note = next(block for block in app._output_lines if "stderr truncated" in block)
    assert "stdout truncated for diagnostics; retained the last 256" in stdout_note
    assert "stderr truncated for diagnostics; retained the last 256" in stderr_note
    assert "OUT-END" not in stdout_note
    assert "ERR-END" not in stderr_note
    assert streamed_output.count("OUT-END") == 1
    assert streamed_output.count("ERR-END") == 1
    assert "Exit code: 7" in output


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


@pytest.mark.asyncio
async def test_tui_app_completion_keys_navigate_and_accept_before_input_actions() -> None:
    """Focused eager keys must operate an active completion instead of history or submit."""
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    input_area = TextArea(text="/he", multiline=True)
    input_area.buffer.cursor_position = len(input_area.buffer.text)
    input_area.buffer._set_completions([
        Completion("/help", start_position=-3),
        Completion("/health", start_position=-3),
    ])
    input_bindings = app._setup_input_keybindings(input_area)
    global_bindings = app._setup_keybindings(input_area)
    handle_down = next(binding.handler for binding in input_bindings.bindings if binding.keys == (Keys.Down,))
    handle_up = next(binding.handler for binding in input_bindings.bindings if binding.keys == (Keys.Up,))
    handle_enter = next(binding.handler for binding in input_bindings.bindings if binding.keys == (Keys.ControlM,))
    handle_tab = next(binding.handler for binding in global_bindings.bindings if binding.keys == (Keys.ControlI,))

    handle_tab(MagicMock())
    assert input_area.buffer.complete_state is not None
    assert input_area.buffer.complete_state.current_completion.text == "/help"
    handle_down(MagicMock())
    assert input_area.buffer.complete_state.current_completion.text == "/health"
    handle_up(MagicMock())
    assert input_area.buffer.complete_state.current_completion.text == "/help"
    handle_enter(MagicMock())
    await asyncio.sleep(0)

    assert input_area.buffer.text == "/help"
    assert input_area.buffer.complete_state is None
    assert app._agent_task is None


def test_tui_app_tab_starts_slash_completion_before_toggling_mode() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    input_area = TextArea(text="/he", multiline=True)
    input_area.buffer.cursor_position = len(input_area.buffer.text)
    input_area.buffer.start_completion = MagicMock()  # type: ignore[method-assign]
    key_bindings = app._setup_keybindings(input_area)
    handle_tab = next(binding.handler for binding in key_bindings.bindings if binding.keys == (Keys.ControlI,))

    handle_tab(MagicMock())

    input_area.buffer.start_completion.assert_called_once_with(select_first=True)
    assert app._input_mode == "send"


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
async def test_tui_app_submits_unrecognized_slash_prefix_as_an_ordinary_prompt() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    app._launch_agent = MagicMock()  # type: ignore[method-assign]
    input_area = TextArea(multiline=True)
    prompt = "/home/user/project/file.txt is the input"

    app._submit_input(prompt, input_area)
    dispatch_task = app._foreground_command_task
    assert dispatch_task is not None
    assert input_area.buffer.text == ""
    await dispatch_task

    app._launch_agent.assert_called_once_with(prompt, [])
    assert app._prompt_history == [prompt]
    assert any(prompt in line for line in app._output_lines)


@pytest.mark.asyncio
async def test_tui_app_slash_prompt_snapshots_attachments_before_skill_refresh() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    runtime = MagicMock()
    runtime.ctx = TUIContext()
    app._runtime = runtime
    refresh_started = asyncio.Event()
    release_refresh = asyncio.Event()

    async def refresh_context(ctx: TUIContext) -> None:
        refresh_started.set()
        await release_refresh.wait()

    app._skill_toolset = MagicMock()
    app._skill_toolset.refresh_context = AsyncMock(side_effect=refresh_context)
    app._launch_agent = MagicMock()  # type: ignore[method-assign]
    placeholder = app._format_attachment_placeholder(1, "image/png", 3)
    original = PendingAttachment(
        data=b"old",
        media_type="image/png",
        size_bytes=3,
        placeholder=placeholder,
    )
    app._pending_attachments = [original]
    prompt = "/home/user/project/file.txt is the input"
    input_area = TextArea(text=f"{placeholder} {prompt}", multiline=True)
    app._input_area = input_area

    app._submit_input(input_area.buffer.text, input_area)
    dispatch_task = app._foreground_command_task
    assert dispatch_task is not None
    await asyncio.wait_for(refresh_started.wait(), timeout=1)

    try:
        assert app._pending_attachments == []
        clipboard_result = ClipboardImageReadResult(image=ClipboardImage(data=b"new", media_type="image/png"))
        with patch("yaacli.app.tui.read_clipboard_image", new=AsyncMock(return_value=clipboard_result)):
            await app._paste_clipboard_image(input_area)

        assert [item.data for item in app._pending_attachments] == [b"new"]
        assert app._pending_attachments[0].placeholder in input_area.buffer.text
    finally:
        release_refresh.set()
        await dispatch_task

    app._launch_agent.assert_called_once_with(prompt, [original])
    assert [item.data for item in app._pending_attachments] == [b"new"]
    assert app._pending_attachments[0].placeholder in input_area.buffer.text


@pytest.mark.asyncio
async def test_tui_app_submit_multiple_skill_prefixes_as_agent_prompt() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    app._runtime = MagicMock()
    app._runtime.ctx.available_skills = {
        "lark-cli": AvailableSkill(name="lark-cli", description="Lark", path="/skills/lark-cli"),
        "agent-builder": AvailableSkill(
            name="agent-builder",
            description="Agents",
            path="/skills/agent-builder",
        ),
    }
    app._launch_agent = MagicMock()  # type: ignore[method-assign]
    input_area = TextArea(multiline=True)

    app._submit_input("/lark-cli /agent-builder Build an agent", input_area)
    dispatch_task = app._foreground_command_task
    assert dispatch_task is not None
    await dispatch_task

    app._launch_agent.assert_called_once()
    prompt = app._launch_agent.call_args.args[0]
    assert app._launch_agent.call_args.kwargs["session_input"] == "/lark-cli /agent-builder Build an agent"
    assert '<skill name="lark-cli" path="/skills/lark-cli" />' in prompt
    assert '<skill name="agent-builder" path="/skills/agent-builder" />' in prompt
    assert prompt.endswith("Build an agent")
    assert any("/lark-cli /agent-builder Build an agent" in line for line in app._output_lines)


@pytest.mark.asyncio
async def test_tui_app_refreshes_skill_catalog_before_slash_classification() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    runtime = MagicMock()
    runtime.ctx = TUIContext()
    app._runtime = runtime
    app._skill_toolset = MagicMock()

    async def refresh_context(ctx: TUIContext) -> None:
        ctx.available_skills = {
            "hot-skill": AvailableSkill(
                name="hot-skill",
                description="Added after startup",
                path="/skills/hot-skill",
            )
        }

    app._skill_toolset.refresh_context = AsyncMock(side_effect=refresh_context)
    app._launch_agent = MagicMock()  # type: ignore[method-assign]
    input_area = TextArea(multiline=True)

    app._submit_input("/hot-skill Use the new workflow", input_area)
    dispatch_task = app._foreground_command_task
    assert dispatch_task is not None
    assert app.phase == TUIPhase.COMMAND_RUNNING
    await dispatch_task

    app._skill_toolset.refresh_context.assert_awaited_once_with(runtime.ctx)
    app._launch_agent.assert_called_once()
    prompt = app._launch_agent.call_args.args[0]
    assert '<skill name="hot-skill" path="/skills/hot-skill" />' in prompt
    assert prompt.endswith("Use the new workflow")


@pytest.mark.asyncio
async def test_tui_app_submit_custom_slash_command():
    """Submitting a custom slash command expands and runs its configured prompt."""
    config = MockConfig(
        commands={
            "commit": CommandDefinition(
                prompt="Create a git commit for the current changes.",
                description="Commit changes",
            )
        }
    )
    config_manager = MockConfigManager()
    app = TUIApp(config=config, config_manager=config_manager)
    input_area = TextArea(multiline=True)
    captured_prompts: list[str] = []

    async def fake_run_agent(prompt: str, attachments: object = None) -> None:
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
                description="Commit changes",
            )
        }
    )
    config_manager = MockConfigManager()
    app = TUIApp(config=config, config_manager=config_manager)
    captured_prompts: list[str] = []

    async def fake_run_agent(prompt: str, attachments: object = None) -> None:
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


@pytest.mark.asyncio
async def test_tui_app_run_async_accepts_absolute_path_prompt_enter() -> None:
    """A leading absolute path must remain typeable and submit as ordinary text."""
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    captured_prompts: list[str] = []
    prompt = "/home/user/project/file.txt is the input"

    async def fake_run_agent(text: str, attachments: object = None) -> None:
        captured_prompts.append(text)
        if app._app:
            app._app.exit()

    app._run_agent = fake_run_agent  # type: ignore[method-assign]

    with create_pipe_input() as pipe_input:
        with create_app_session(input=pipe_input, output=DummyOutput()):
            run_task = asyncio.create_task(app.run())
            await asyncio.sleep(0.1)
            pipe_input.send_text(f"{prompt}\r")
            await asyncio.wait_for(run_task, timeout=2)

    assert captured_prompts == [prompt]


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
async def test_tui_app_deferred_continuations_share_one_request_budget() -> None:
    """Repeated HITL continuations must not reset max_requests within one turn."""
    config = MockConfig()
    config.general.max_requests = 2
    app = TUIApp(config=config, config_manager=MockConfigManager())
    runtime = MagicMock()
    runtime.ctx = MagicMock(loop_active=False)
    runtime.ctx.steering_messages = []
    app._runtime = runtime

    deferred = DeferredToolRequests(approvals=[ToolCallPart(tool_name="edit", args={}, tool_call_id="approval-1")])
    first = MagicMock(output=deferred)
    first.usage.requests = 1
    second = MagicMock(output=deferred)
    second.usage.requests = 1
    app._execute_stream = AsyncMock(side_effect=[first, second])
    app._request_user_action = AsyncMock(return_value=DeferredToolResults())  # type: ignore[method-assign]
    app._check_pending_bus_messages = MagicMock()

    await app._run_agent("hello")

    assert [call.kwargs["request_limit"] for call in app._execute_stream.await_args_list] == [2, 1]
    assert app._request_user_action.await_count == 1
    assert any("cumulative model request limit of 2" in line for line in app._output_lines)
    replay = app._display_replay.snapshot()
    assert [event.get("type") for event in replay].count("RUN_ERROR") == 1
    assert not any(event.get("type") == "RUN_FINISHED" for event in replay)


@pytest.mark.asyncio
async def test_tui_app_successful_run_save_failure_keeps_single_finished_terminal_event() -> None:
    """Post-run persistence failures are warnings, not contradictory RUN_ERROR events."""
    config = MockConfig()
    config.session = MagicMock(auto_save_history=True)
    app = TUIApp(config=config, config_manager=MockConfigManager())
    runtime = MagicMock()
    runtime.ctx = MagicMock(loop_active=False)
    runtime.ctx.steering_messages = ["unconsumed guidance"]
    app._runtime = runtime
    result = MagicMock()
    result.output = "done"
    app._execute_stream = AsyncMock(return_value=result)
    steering_at_save: list[list[str]] = []

    async def fail_snapshot(**_: object) -> bool:
        steering_at_save.append(list(runtime.ctx.steering_messages))
        raise OSError("disk full")

    app._save_session_snapshot_async = AsyncMock(side_effect=fail_snapshot)  # type: ignore[method-assign]
    app._check_pending_bus_messages = MagicMock()

    await app._run_agent("hello")

    assert steering_at_save == [[]]
    event_types = [str(event.get("type")) for event in app._display_replay.snapshot()]
    assert event_types.count("RUN_FINISHED") == 1
    assert "RUN_ERROR" not in event_types
    assert any("snapshot could not be saved" in line for line in app._output_lines)
    assert app.phase == TUIPhase.IDLE


@pytest.mark.asyncio
async def test_tui_app_failed_run_persists_run_error_before_snapshot() -> None:
    """A durable failed snapshot must already contain its terminal RUN_ERROR."""
    config = MockConfig()
    config.session = MagicMock(auto_save_history=True)
    app = TUIApp(config=config, config_manager=MockConfigManager())
    runtime = MagicMock()
    runtime.ctx = MagicMock(loop_active=False)
    runtime.ctx.steering_messages = ["unconsumed guidance"]
    app._runtime = runtime
    app._execute_stream = AsyncMock(side_effect=RuntimeError("provider failed"))
    persisted_replay: list[dict[str, object]] = []
    steering_at_save: list[list[str]] = []

    async def capture_snapshot(**_: object) -> bool:
        persisted_replay.extend(app._display_replay.snapshot())
        steering_at_save.append(list(runtime.ctx.steering_messages))
        return True

    app._save_session_snapshot_async = AsyncMock(side_effect=capture_snapshot)  # type: ignore[method-assign]
    app._check_pending_bus_messages = MagicMock()

    await app._run_agent("hello")

    assert persisted_replay[-1]["type"] == "RUN_ERROR"
    assert persisted_replay[-1]["message"] == "provider failed"
    assert steering_at_save == [[]]
    app._save_session_snapshot_async.assert_awaited_once_with(
        include_usage_ledger=True,
        save_reason="error",
    )
    assert app.phase == TUIPhase.IDLE


@pytest.mark.asyncio
async def test_tui_app_cancelled_run_saves_partial_snapshot_and_clears_steering() -> None:
    """Cancelling active execution preserves recovery state and isolates future turns."""
    config = MockConfig()
    config.session = MagicMock(auto_save_history=True)
    app = TUIApp(config=config, config_manager=MockConfigManager())
    ctx = TUIContext(steering_messages=["unconsumed guidance"])
    ctx.message_bus.subscribe(ctx.agent_id)
    ctx.message_bus.send(BusMessage(content="guidance", source="user", target=ctx.agent_id))
    ctx.message_bus.send(BusMessage(content="result", source="subagent-1", target=ctx.agent_id))
    runtime = MagicMock(ctx=ctx)
    app._runtime = runtime
    stream_started = asyncio.Event()

    async def wait_for_cancellation(*_: object, **__: object) -> None:
        stream_started.set()
        await asyncio.sleep(3600)

    app._execute_stream = AsyncMock(side_effect=wait_for_cancellation)
    steering_in_exported_state: list[list[str]] = []
    pending_bus_sources_at_export: list[list[str]] = []

    async def capture_snapshot(**_: object) -> bool:
        state = ctx.export_state(include_usage_ledger=True)
        steering_in_exported_state.append(state.steering_messages)
        pending_bus_sources_at_export.append([message.source for message in ctx.message_bus.peek(ctx.agent_id)])
        return True

    app._save_session_snapshot_async = AsyncMock(side_effect=capture_snapshot)  # type: ignore[method-assign]
    app._check_pending_bus_messages = MagicMock()

    run_task = asyncio.create_task(app._run_agent("hello"))
    await stream_started.wait()
    run_task.cancel()
    await run_task

    app._save_session_snapshot_async.assert_awaited_once_with(
        include_usage_ledger=True,
        save_reason="cancelled",
    )
    replay = app._display_replay.snapshot()
    assert replay[-1]["name"].endswith("run_cancelled")
    assert replay[-1]["value"] == {"reason": "user_interrupted"}
    assert steering_in_exported_state == [[]]
    assert pending_bus_sources_at_export == [["subagent-1"]]
    assert runtime.ctx.steering_messages == []
    assert any(line == "[Cancelled · partial state saved]" for line in app._output_lines)
    app._check_pending_bus_messages.assert_not_called()
    assert app.phase == TUIPhase.IDLE


@pytest.mark.asyncio
async def test_tui_app_cancel_after_run_finished_does_not_reclassify_or_resave() -> None:
    """Persistence cancellation after RUN_FINISHED must not emit run_cancelled or save twice."""
    config = MockConfig()
    config.session = MagicMock(auto_save_history=True)
    app = TUIApp(config=config, config_manager=MockConfigManager())
    runtime = MagicMock()
    runtime.ctx = MagicMock(loop_active=False)
    runtime.ctx.steering_messages = []
    app._runtime = runtime
    result = MagicMock()
    result.output = "done"
    app._execute_stream = AsyncMock(return_value=result)
    save_started = asyncio.Event()

    async def wait_for_cancellation(**_: object) -> bool:
        save_started.set()
        await asyncio.sleep(3600)
        return True

    app._save_session_snapshot_async = AsyncMock(side_effect=wait_for_cancellation)  # type: ignore[method-assign]
    app._check_pending_bus_messages = MagicMock()

    run_task = asyncio.create_task(app._run_agent("hello"))
    await save_started.wait()
    run_task.cancel()
    await run_task

    replay = app._display_replay.snapshot()
    assert [event.get("type") for event in replay].count("RUN_FINISHED") == 1
    assert not any(event.get("name") == "run_cancelled" for event in replay)
    assert app._save_session_snapshot_async.await_count == 1
    assert not any(line.startswith("[Cancelled") for line in app._output_lines)
    assert any("persistence was interrupted" in line for line in app._output_lines)
    assert app.phase == TUIPhase.IDLE


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


def test_tui_app_build_user_prompt_has_no_mode_control_context() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    app._load_guidance_files = MagicMock(return_value=(None, None))  # type: ignore[method-assign]

    prompt = app._build_user_prompt("inspect the implementation")

    assert prompt == "inspect the implementation"


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
    input_area = TextArea(multiline=True)

    with patch("yaacli.app.tui.read_clipboard_image", new=AsyncMock()) as mock_read:
        mock_read.return_value = ClipboardImageReadResult(
            image=ClipboardImage(data=b"image-bytes", media_type="image/png")
        )
        await app._paste_clipboard_image(input_area)

    assert len(app._pending_attachments) == 1
    assert app._pending_attachments[0].data == b"image-bytes"
    assert app._pending_attachments[0].placeholder == "[Attached image 1: image/png 11B]"
    assert input_area.buffer.text == "[Attached image 1: image/png 11B] "
    assert any("Attached image/png" in line for line in app._output_lines)


@pytest.mark.asyncio
async def test_tui_app_active_run_image_paste_preserves_draft_for_next_turn() -> None:
    """Binary clipboard input cannot steer and must not alter the active compose draft."""
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    input_area = TextArea(text="keep this draft", multiline=True)
    app._set_phase(TUIPhase.THINKING)
    app._add_steering_message = MagicMock()  # type: ignore[method-assign]

    with patch("yaacli.app.tui.read_clipboard_image", new=AsyncMock()) as mock_read:
        mock_read.return_value = ClipboardImageReadResult(
            image=ClipboardImage(data=b"image-bytes", media_type="image/png")
        )
        await app._paste_clipboard_image(input_area)

    assert input_area.buffer.text == "keep this draft"
    assert len(app._pending_attachments) == 1
    assert app._pending_attachments[0].placeholder == ""
    app._add_steering_message.assert_not_called()
    assert any("for the next turn" in line for line in app._output_lines)


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
async def test_tui_app_real_paste_image_command_delivers_attachment_on_next_prompt() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    input_area = TextArea(text="/paste-image", multiline=True)
    app._input_area = input_area
    clipboard_result = ClipboardImageReadResult(image=ClipboardImage(data=b"img", media_type="image/png"))

    with patch("yaacli.app.tui.read_clipboard_image", new=AsyncMock(return_value=clipboard_result)):
        app._submit_input("/paste-image", input_area)
        command_task = app._foreground_command_task
        assert command_task is not None
        await command_task

    assert len(app._pending_attachments) == 1
    attachment = app._pending_attachments[0]
    assert attachment.placeholder
    assert attachment.placeholder in input_area.buffer.text

    captured_runs: list[tuple[str, list[PendingAttachment] | None]] = []

    async def fake_run_agent(
        prompt: str,
        attachments: list[PendingAttachment] | None = None,
    ) -> None:
        captured_runs.append((prompt, attachments))

    app._run_agent = fake_run_agent  # type: ignore[method-assign]
    input_area.buffer.insert_text("next prompt")
    app._submit_input(input_area.buffer.text.strip(), input_area)
    assert app._agent_task is not None
    await app._agent_task

    assert len(captured_runs) == 1
    prompt, attachments = captured_runs[0]
    assert prompt == "next prompt"
    assert attachments is not None
    assert [item.data for item in attachments] == [b"img"]


@pytest.mark.asyncio
async def test_tui_app_paste_image_without_compose_area_queues_placeholderless_attachment() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    clipboard_result = ClipboardImageReadResult(image=ClipboardImage(data=b"img", media_type="image/png"))

    with patch("yaacli.app.tui.read_clipboard_image", new=AsyncMock(return_value=clipboard_result)):
        await app._handle_command_inner("/paste-image")

    assert len(app._pending_attachments) == 1
    assert app._pending_attachments[0].placeholder == ""


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


@pytest.mark.asyncio
async def test_tui_app_submit_input_strips_attachment_placeholder() -> None:
    """Submitted prompt text should omit generated clipboard image placeholders."""
    config = MockConfig()
    config_manager = MockConfigManager()
    app = TUIApp(config=config, config_manager=config_manager)
    input_area = TextArea(multiline=True)
    attachment = PendingAttachment(
        data=b"img",
        media_type="image/png",
        size_bytes=3,
        placeholder="[Attached image 1: image/png 3B]",
    )
    app._pending_attachments.append(attachment)
    captured_runs: list[tuple[str, list[PendingAttachment] | None]] = []

    async def fake_run_agent(
        prompt: str,
        attachments: list[PendingAttachment] | None = None,
    ) -> None:
        captured_runs.append((prompt, attachments))

    app._run_agent = fake_run_agent  # type: ignore[method-assign]

    with patch.object(app, "_append_user_input") as mock_append_user_input:
        app._submit_input("Please inspect [Attached image 1: image/png 3B]", input_area)

    mock_append_user_input.assert_called_once_with("Please inspect", [attachment])
    assert app._pending_attachments == []
    assert app._agent_task is not None
    await app._agent_task
    assert captured_runs == [("Please inspect", [attachment])]


def test_tui_app_rejects_input_while_session_clear_is_in_progress() -> None:
    """A new turn must not race with asynchronous session cleanup."""
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    input_area = TextArea(text="new request")
    app._session_clear_in_progress = True

    app._submit_input("new request", input_area)

    assert input_area.buffer.text == "new request"
    assert app._agent_task is None
    assert any("Session clear is still in progress" in line for line in app._output_lines)


@pytest.mark.asyncio
async def test_tui_app_attachment_chips_remain_unique_after_remove_and_readd() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    input_area = TextArea(multiline=True)
    app._input_area = input_area
    clipboard_results = [
        ClipboardImageReadResult(image=ClipboardImage(data=value, media_type="image/png"))
        for value in (b"a", b"b", b"c")
    ]

    with patch("yaacli.app.tui.read_clipboard_image", new=AsyncMock(side_effect=clipboard_results)):
        await app._paste_clipboard_image(input_area)
        await app._paste_clipboard_image(input_area)
        removed = app._remove_pending_attachment(0)
        await app._paste_clipboard_image(input_area)

    assert removed is not None and removed.data == b"a"
    assert [item.placeholder for item in app._pending_attachments] == [
        "[Attached image 2: image/png 1B]",
        "[Attached image 3: image/png 1B]",
    ]

    retained, deleted = app._pending_attachments
    submitted_text = input_area.buffer.text.replace(deleted.placeholder, "")
    captured_runs: list[tuple[str, list[PendingAttachment] | None]] = []

    async def fake_run_agent(
        prompt: str,
        attachments: list[PendingAttachment] | None = None,
    ) -> None:
        captured_runs.append((prompt, attachments))

    app._run_agent = fake_run_agent  # type: ignore[method-assign]
    app._submit_input(submitted_text, input_area)
    assert app._agent_task is not None
    await app._agent_task

    assert len(captured_runs) == 1
    prompt, attachments = captured_runs[0]
    assert prompt == ""
    assert attachments == [retained]
    assert [item.data for item in attachments] == [b"b"]


@pytest.mark.asyncio
async def test_tui_app_clear_session_clears_pending_attachments() -> None:
    """Clearing the session should drop queued clipboard images."""
    config = MockConfig()
    config_manager = MockConfigManager()
    app = TUIApp(config=config, config_manager=config_manager)
    app._pending_attachments.append(PendingAttachment(data=b"img", media_type="image/png", size_bytes=3))

    await app._clear_session()

    assert app._pending_attachments == []


@pytest.mark.asyncio
async def test_tui_app_clear_session_resets_conversation_state() -> None:
    """Clearing should remove all conversation state while retaining runtime policy."""
    config = MockConfig()
    config_manager = MockConfigManager()
    app = TUIApp(config=config, config_manager=config_manager)
    ctx = TUIContext()
    ctx.provider_session_id = "provider-session"
    ctx.provider_thread_id = "provider-thread"
    ctx.deferred_tool_metadata["call-1"] = {"tool": "shell"}
    ctx.handoff_message = "old handoff"
    ctx.force_inject_instructions = True
    ctx.shell_env["OLD_SESSION"] = "1"
    ctx.update_usage_snapshot_entry(
        agent_id="main",
        agent_name="main",
        model_id="test-model",
        usage=RunUsage(input_tokens=7, output_tokens=3),
        source="test",
    )
    ctx.shell_review_records.append("old review")
    ctx.user_prompts = "old prompt"
    ctx.previous_assistant_response_reference = "old response"
    ctx.steering_messages.append("old steering")
    ctx.tool_id_wrapper.upsert_tool_call_id("old-tool-call")
    ctx.agent_stream_queues["old-agent"].put_nowait(MagicMock())
    ctx.subagent_history["explorer-1"] = []
    ctx.agent_registry["explorer-1"] = AgentInfo(agent_id="explorer-1", agent_name="explorer", parent_agent_id="main")
    ctx.auto_load_files.append("old.py")
    ctx.task_manager.create("Inspect issue", "Investigate stale state after clear")
    ctx.note_manager.set("old-note", "old value")
    ctx.tool_search_loaded_tools.append("search")
    ctx.tool_search_loaded_namespaces.append("web")
    ctx.tool_tags.add("shell")
    ctx.message_bus.subscribe("main")
    ctx.message_bus.send(BusMessage(content="old message", source="explorer-1", target="main"))
    ctx.goal_task = "old goal"
    ctx.goal_iteration = 3
    ctx.need_user_approve_tools = ["shell"]
    ctx.need_user_approve_mcps = ["filesystem"]

    runtime = MagicMock()
    runtime.ctx = ctx
    runtime.env = None
    app._runtime = runtime
    app._pending_bus_check_needed = True
    app._goal_usage_start_breakdown = app._session_usage.token_breakdown
    app._goal_usage_report_pending = True
    usage_snapshot = ctx.build_usage_snapshot()
    app._session_usage.set_run_snapshot(usage_snapshot)
    app._session_usage.commit_run_snapshot(usage_snapshot.run_id)
    app._session_usage.finalize_run_snapshots(usage_snapshot.run_id)

    await app._clear_session()

    cleared_ctx = runtime.ctx
    assert cleared_ctx is not ctx
    assert cleared_ctx.provider_session_id is None
    assert cleared_ctx.provider_thread_id is None
    assert cleared_ctx.deferred_tool_metadata == {}
    assert cleared_ctx.handoff_message is None
    assert cleared_ctx.force_inject_instructions is False
    assert cleared_ctx.shell_env == {}
    assert cleared_ctx.usage_snapshot_entries == {}
    assert list(cleared_ctx.shell_review_records) == []
    assert cleared_ctx.user_prompts is None
    assert cleared_ctx.previous_assistant_response_reference is None
    assert cleared_ctx.steering_messages == []
    assert cleared_ctx.tool_id_wrapper._tool_call_maps == {}
    assert cleared_ctx.agent_stream_queues == {}
    assert cleared_ctx.subagent_history == {}
    assert cleared_ctx.agent_registry == {}
    assert cleared_ctx.auto_load_files == []
    assert cleared_ctx.task_manager.list_all() == []
    assert cleared_ctx.note_manager.list_all() == []
    assert cleared_ctx.tool_search_loaded_tools == []
    assert cleared_ctx.tool_search_loaded_namespaces == []
    assert cleared_ctx.tool_tags == set()
    assert len(cleared_ctx.message_bus) == 0
    assert cleared_ctx.message_bus.subscriber_count == 0
    assert cleared_ctx.goal_active is False
    assert cleared_ctx.goal_iteration == 0
    assert cleared_ctx.need_user_approve_tools == ["shell"]
    assert cleared_ctx.need_user_approve_mcps == ["filesystem"]
    assert app._pending_bus_check_needed is False
    assert app._goal_usage_start_breakdown is None
    assert app._goal_usage_report_pending is False
    assert app._session_usage.total_input_tokens == 7
    assert app._session_usage.total_output_tokens == 3


@pytest.mark.asyncio
async def test_tui_app_clear_session_discards_deferred_shell_notification() -> None:
    """A shell completion from the old session must not enter the fresh context."""
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    ctx = TUIContext()
    runtime = MagicMock()
    runtime.ctx = ctx
    runtime.env = None
    app._runtime = runtime
    monitor = BackgroundMonitor()
    monitor.enqueue_shell_message(
        BusMessage(content="shell output ready", source="shell-monitor", target="main"),
        process_id="proc-1",
        kind="output",
    )
    app._get_background_monitor = MagicMock(return_value=monitor)  # type: ignore[method-assign]

    await app._clear_session()

    assert app._agent_task is None
    assert app.phase == TUIPhase.IDLE
    assert app._background_results_ready is False
    assert runtime.ctx.message_bus.has_pending("main") is False
    assert monitor.has_pending_messages is False


@pytest.mark.asyncio
async def test_tui_app_new_dispatch_restores_shell_env_without_old_background_ready() -> None:
    """Real /new dispatch should retain runtime policy but discard old shell readiness."""
    config = YaacliConfig(shell_env={"CONFIGURED_BASE": "enabled"})
    app = TUIApp(config=config, config_manager=MockConfigManager())
    old_ctx = TUIContext(shell_env={"OLD_SESSION": "stale"})
    runtime = MagicMock(ctx=old_ctx, env=None)
    app._runtime = runtime
    old_session_id = app.session_id
    monitor = BackgroundMonitor()
    shell = MagicMock()
    shell.reset_background_processes = AsyncMock()
    monitor._shell = shell
    monitor.enqueue_shell_message(
        BusMessage(content="shell output ready", source="shell-monitor", target="main"),
        process_id="proc-new",
        kind="completion",
    )
    app._get_background_monitor = MagicMock(return_value=monitor)  # type: ignore[method-assign]
    input_area = TextArea(text="/new")

    app._submit_input("/new", input_area)
    command_task = app._foreground_command_task
    assert command_task is not None
    await command_task

    assert app.session_id != old_session_id
    assert runtime.ctx is not old_ctx
    assert runtime.ctx.shell_env == {"CONFIGURED_BASE": "enabled"}
    assert app.phase == TUIPhase.IDLE
    assert app._background_results_ready is False
    assert runtime.ctx.message_bus.has_pending("main") is False
    assert monitor.has_pending_messages is False
    shell.reset_background_processes.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_tui_app_cancelled_new_rolls_identity_forward_before_next_save() -> None:
    """Cancelling /new after isolation starts must never reuse the old durable identity."""
    config = YaacliConfig(shell_env={"CONFIGURED_BASE": "enabled"})
    app = TUIApp(config=config, config_manager=MockConfigManager())
    old_ctx = TUIContext(shell_env={"OLD_SESSION": "stale"})
    runtime = MagicMock(ctx=old_ctx, env=None)
    app._runtime = runtime
    app._message_history = [ModelRequest(parts=[UserPromptPart(content="old history")])]
    old_session_id = app.session_id
    monitor = BackgroundMonitor()
    reset_started = asyncio.Event()

    async def slow_reset() -> None:
        reset_started.set()
        await asyncio.Event().wait()

    monitor.reset_session_state = slow_reset  # type: ignore[method-assign]
    app._get_background_monitor = MagicMock(return_value=monitor)  # type: ignore[method-assign]
    input_area = TextArea(text="/new")

    app._submit_input("/new", input_area)
    command_task = app._foreground_command_task
    assert command_task is not None
    await asyncio.wait_for(reset_started.wait(), timeout=1)
    new_session_id = app.session_id
    assert new_session_id != old_session_id

    app._cancel_foreground()
    with pytest.raises(asyncio.CancelledError):
        await command_task
    await asyncio.sleep(0)

    assert runtime.ctx is not old_ctx
    assert runtime.ctx.shell_env == {"CONFIGURED_BASE": "enabled"}
    assert app._message_history is None
    assert app.session_id == new_session_id
    assert app.phase == TUIPhase.IDLE
    assert app._session_clear_in_progress is False

    app._message_history = [ModelRequest(parts=[UserPromptPart(content="new history")])]
    with patch("yaacli.app.tui.save_session_turn", return_value=Path("turn")) as save_turn:
        assert app._save_session_snapshot(save_reason="test") is True
    assert save_turn.call_args.kwargs["session_id"] == new_session_id


@pytest.mark.asyncio
async def test_tui_app_new_rolls_back_when_shell_process_cleanup_fails() -> None:
    """A live process with a failed kill hook must prevent a new-session commit."""
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    old_ctx = TUIContext(user_prompts="keep me")
    runtime = MagicMock(ctx=old_ctx, env=None)
    app._runtime = runtime
    old_history = [ModelRequest(parts=[UserPromptPart(content="old history")])]
    app._message_history = old_history
    old_session_id = app.session_id
    monitor = BackgroundMonitor()
    reset_error = ShellBackgroundResetError({"proc-failed": RuntimeError("kill failed")})
    monitor.reset_session_state = AsyncMock(side_effect=reset_error)  # type: ignore[method-assign]
    app._get_background_monitor = MagicMock(return_value=monitor)  # type: ignore[method-assign]

    await app._start_new_session()

    assert app.session_id == old_session_id
    assert runtime.ctx is old_ctx
    assert app._message_history is old_history
    assert app.phase == TUIPhase.IDLE
    assert app._session_clear_in_progress is False
    assert any("New session was not started" in line for line in app._output_lines)


@pytest.mark.asyncio
async def test_tui_app_clear_session_still_clears_context_when_monitor_reset_fails() -> None:
    """Background cleanup errors must not leave the old conversation active."""
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    ctx = TUIContext(subagent_history={"old-agent": []}, user_prompts="old prompt")
    runtime = MagicMock()
    runtime.ctx = ctx
    runtime.env = None
    app._runtime = runtime
    app._message_history = [ModelRequest(parts=[UserPromptPart(content="old history")])]
    monitor = BackgroundMonitor()
    release = asyncio.Event()

    async def publish_late_result() -> None:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            await release.wait()
        monitor.record_task_result(
            BackgroundTaskResult(
                agent_id="old-agent",
                subagent_name="executor",
                status="completed",
                content="stale result",
            )
        )
        if monitor.should_deliver_task_result_message("old-agent"):
            monitor.enqueue_message(BusMessage(content="stale result", source="old-agent", target="main"))
        monitor.notify_completion("old-agent")

    stale_task = asyncio.create_task(publish_late_result())
    monitor.register_task("old-agent", stale_task)
    monitor.set_completion_callback(app._on_background_task_complete)
    await asyncio.sleep(0)
    monitor.reset_session_state = AsyncMock(side_effect=RuntimeError("reset failed"))  # type: ignore[method-assign]
    app._get_background_monitor = MagicMock(return_value=monitor)  # type: ignore[method-assign]

    await app._clear_session()
    release.set()
    await stale_task
    await asyncio.sleep(0)

    assert app._message_history is None
    assert runtime.ctx is not ctx
    assert runtime.ctx.subagent_history == {}
    assert runtime.ctx.user_prompts is None
    assert runtime.ctx.message_bus.has_pending("main") is False
    assert monitor.task_results == {}
    assert monitor.has_pending_messages is False
    assert app._session_clear_in_progress is False


@pytest.mark.asyncio
async def test_tui_app_load_history_does_not_commit_when_shell_cleanup_fails(tmp_path: Path) -> None:
    """A restore candidate remains uncommitted until old processes are terminated."""
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    old_ctx = TUIContext(user_prompts="active context")
    runtime = MagicMock(ctx=old_ctx, env=None)
    app._runtime = runtime
    old_history = [ModelRequest(parts=[UserPromptPart(content="active history")])]
    app._message_history = old_history
    old_session_id = app.session_id
    monitor = BackgroundMonitor()
    reset_error = ShellBackgroundResetError({"proc-failed": RuntimeError("kill failed")})
    monitor.reset_session_state = AsyncMock(side_effect=reset_error)  # type: ignore[method-assign]
    app._get_background_monitor = MagicMock(return_value=monitor)  # type: ignore[method-assign]
    load_dir = tmp_path / "candidate"
    load_dir.mkdir()
    (load_dir / "message_history.json").write_bytes(b"[]")

    loaded = await app._load_history(str(load_dir), target_session_id="candidate123")

    assert loaded is False
    assert app.session_id == old_session_id
    assert runtime.ctx is old_ctx
    assert app._message_history is old_history
    assert app._session_clear_in_progress is False
    assert any("Session was not loaded" in line for line in app._output_lines)


@pytest.mark.asyncio
async def test_tui_app_load_history_clears_pending_attachments(tmp_path: Path) -> None:
    """Loading a session should reset queued clipboard images."""
    config = MockConfig()
    config_manager = MockConfigManager()
    app = TUIApp(config=config, config_manager=config_manager)
    app._runtime = MagicMock(ctx=TUIContext(), env=None)
    original_session_id = app.session_id
    app._pending_attachments.append(PendingAttachment(data=b"img", media_type="image/png", size_bytes=3))

    load_dir = tmp_path / "session"
    load_dir.mkdir()
    (load_dir / "message_history.json").write_bytes(b"[]")

    await app._load_history(str(load_dir))

    assert app._pending_attachments == []
    assert app._message_history == []
    assert app.session_id == original_session_id


@pytest.mark.asyncio
async def test_tui_app_load_history_reads_schema_v2_head_as_atomic_snapshot(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    config_manager = MagicMock()
    config_manager.get_sessions_dir.return_value = sessions_dir
    config_manager.config_dir = tmp_path / "config"
    turn_dir = save_session_turn(
        config_manager=config_manager,
        session_id="atomic-session",
        working_dir=tmp_path,
        message_history_json=b"[]",
        context_state_json=ResumableState().model_dump_json(),
        display_messages=[],
        output_text="done",
        save_reason="test",
        turn_id="turn-0",
        max_turns=1,
        max_sessions=10,
    )
    session_dir = turn_dir.parents[1]
    assert not (session_dir / "message_history.json").exists()

    app = TUIApp(config=MockConfig(), config_manager=config_manager)
    app._runtime = MagicMock(ctx=TUIContext(), env=None)

    loaded = await app._load_history(str(session_dir), target_session_id="atomic-session")

    assert loaded is True
    assert app.session_id == "atomic-session"
    assert app._message_history == []
    assert any("display_messages.json (0 events)" in line for line in app._output_lines)


@pytest.mark.asyncio
async def test_tui_app_load_history_isolates_context_and_background_state(tmp_path: Path) -> None:
    """A valid restore must switch all conversation state without rebuilding runtime resources."""
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    old_ctx = TUIContext(need_user_approve_tools=["shell"], need_user_approve_mcps=["filesystem"])
    old_ctx.provider_session_id = "provider-old"
    old_ctx.provider_thread_id = "thread-old"
    old_ctx.shell_env = {"OLD_SESSION": "secret"}
    old_ctx.steering_messages = ["old steering"]
    old_ctx.subagent_history = {"old-agent": []}
    old_ctx.goal_task = "old goal"
    old_bus = old_ctx.message_bus
    runtime_env = object()
    runtime = MagicMock(ctx=old_ctx, env=runtime_env)
    app._runtime = runtime
    app._message_history = [ModelRequest(parts=[UserPromptPart(content="old history")])]
    app._display_replay.append({"type": "TEXT_MESSAGE_CHUNK", "messageId": "old", "delta": "old output"})
    original_session_id = app.session_id

    monitor = BackgroundMonitor()
    monitor.set_message_bus(old_bus, old_ctx.agent_id)
    stale_task = asyncio.create_task(_sleep_forever())
    monitor.register_task("old-agent", stale_task)
    app._get_background_monitor = MagicMock(return_value=monitor)  # type: ignore[method-assign]

    load_dir = tmp_path / "session-b"
    load_dir.mkdir()
    (load_dir / "message_history.json").write_bytes(b"[]")

    loaded = await app._load_history(str(load_dir), target_session_id="target123456")

    assert loaded is True
    assert app.session_id == "target123456"
    assert app.session_id != original_session_id
    assert app._runtime is runtime
    assert runtime.env is runtime_env
    assert runtime.ctx is not old_ctx
    assert runtime.ctx.provider_session_id is None
    assert runtime.ctx.provider_thread_id is None
    assert runtime.ctx.shell_env == {}
    assert runtime.ctx.steering_messages == []
    assert runtime.ctx.subagent_history == {}
    assert runtime.ctx.goal_active is False
    assert runtime.ctx.need_user_approve_tools == ["shell"]
    assert runtime.ctx.need_user_approve_mcps == ["filesystem"]
    assert runtime.ctx.message_bus is not old_bus
    assert monitor._bus is runtime.ctx.message_bus
    assert stale_task.cancelled()
    assert monitor.active_tasks == {}
    assert app._message_history == []


@pytest.mark.asyncio
async def test_tui_app_load_history_rolls_forward_when_background_reset_fails(tmp_path: Path) -> None:
    """After tombstoning old subagents, cleanup failure must not reactivate the old context."""
    config = YaacliConfig(shell_env={"CONFIGURED_BASE": "enabled"})
    app = TUIApp(config=config, config_manager=MockConfigManager())
    old_ctx = TUIContext(shell_env={"OLD_SESSION": "value"})
    runtime = MagicMock(ctx=old_ctx, env=object())
    app._runtime = runtime
    monitor = BackgroundMonitor()
    monitor.reset_session_state = AsyncMock(side_effect=RuntimeError("reset failed"))  # type: ignore[method-assign]
    app._get_background_monitor = MagicMock(return_value=monitor)  # type: ignore[method-assign]

    load_dir = tmp_path / "session-b"
    load_dir.mkdir()
    (load_dir / "message_history.json").write_bytes(b"[]")
    (load_dir / "context_state.json").write_text(
        ResumableState(shell_env={"TARGET_SESSION": "value"}).model_dump_json()
    )

    loaded = await app._load_history(str(load_dir), target_session_id="target123456")

    assert loaded is True
    assert runtime.ctx is not old_ctx
    assert runtime.ctx.shell_env == {"CONFIGURED_BASE": "enabled", "TARGET_SESSION": "value"}
    assert app.session_id == "target123456"
    assert monitor._bus is runtime.ctx.message_bus
    assert app._session_clear_in_progress is False


@pytest.mark.asyncio
async def test_tui_app_load_history_cancellation_commits_isolation_then_reraises(tmp_path: Path) -> None:
    """Cancellation after tombstoning must not restore the old conversation."""
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    old_ctx = TUIContext(shell_env={"OLD_SESSION": "value"})
    runtime = MagicMock(ctx=old_ctx, env=object())
    app._runtime = runtime
    monitor = BackgroundMonitor()
    monitor.reset_session_state = AsyncMock(side_effect=asyncio.CancelledError)  # type: ignore[method-assign]
    app._get_background_monitor = MagicMock(return_value=monitor)  # type: ignore[method-assign]

    load_dir = tmp_path / "session-b"
    load_dir.mkdir()
    (load_dir / "message_history.json").write_bytes(b"[]")

    with pytest.raises(asyncio.CancelledError):
        await app._load_history(str(load_dir), target_session_id="target123456")

    assert runtime.ctx is not old_ctx
    assert runtime.ctx.shell_env == {}
    assert app.session_id == "target123456"
    assert monitor._bus is runtime.ctx.message_bus
    assert app._session_clear_in_progress is False


@pytest.mark.asyncio
async def test_tui_app_saves_and_restores_display_messages(tmp_path: Path) -> None:
    """Session restore should rebuild visible output from display_messages.json."""
    config = MockConfig()
    sessions_dir = tmp_path / "sessions"
    config_manager = MockConfigManager()
    config_manager.get_sessions_dir = MagicMock(return_value=sessions_dir)
    app = TUIApp(config=config, config_manager=config_manager)
    app._runtime = MagicMock()
    app._runtime.ctx.export_state.return_value.model_dump_json.return_value = "{}"
    app._message_history = []
    app._last_session_input = "last user input"
    app._last_session_output = "last assistant output"
    app._display_replay.append({"type": "TEXT_MESSAGE_CHUNK", "messageId": "m1", "delta": "hello"})

    app._save_session_snapshot(save_reason="test")

    assert app.has_session_data is True
    save_dir = sessions_dir / app.session_id
    metadata = json.loads((save_dir / "metadata.json").read_text())
    assert metadata["input_text"] == "last user input"
    assert metadata["output_text"] == "last assistant output"
    display_file = next((save_dir / "turns").iterdir()) / "display_messages.json"
    assert display_file.exists()
    assert json.loads(display_file.read_text()) == [{"type": "TEXT_MESSAGE_CHUNK", "messageId": "m1", "delta": "hello"}]

    load_dir = tmp_path / "load"
    load_dir.mkdir()
    (load_dir / "message_history.json").write_bytes(b"[]")
    (load_dir / "display_messages.json").write_text(display_file.read_text())

    restored = TUIApp(config=config, config_manager=config_manager)
    restored._runtime = MagicMock(ctx=TUIContext(), env=None)
    await restored._load_history(str(load_dir))

    assert restored._message_history == []
    assert restored._display_replay.snapshot() == [{"type": "TEXT_MESSAGE_CHUNK", "messageId": "m1", "delta": "hello"}]
    assert any("hello" in line for line in restored._output_lines)


@pytest.mark.parametrize(
    ("display_payload", "max_bytes"),
    [
        (json.dumps([{"type": "TEXT_MESSAGE_CHUNK", "messageId": "new", "delta": "new"}]), 1),
        ("{invalid", 1024),
    ],
)
@pytest.mark.asyncio
async def test_tui_app_skipped_display_replay_clears_prior_session_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    display_payload: str,
    max_bytes: int,
) -> None:
    """Unsafe replay data must not leave the previous session visible or retained."""
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    app._runtime = MagicMock(ctx=TUIContext(), env=None)
    app._message_history = [ModelRequest(parts=[UserPromptPart(content="old history")])]
    app._display_replay.append({"type": "TEXT_MESSAGE_CHUNK", "messageId": "old", "delta": "prior-session-output"})
    app._restore_output_from_display_events(app._display_replay.snapshot())

    load_dir = tmp_path / "session-b"
    load_dir.mkdir()
    (load_dir / "message_history.json").write_bytes(b"[]")
    (load_dir / "display_messages.json").write_text(display_payload)
    monkeypatch.setattr("yaacli.app.tui._MAX_DISPLAY_REPLAY_LOAD_BYTES", max_bytes)

    await app._load_history(str(load_dir))

    assert app._message_history == []
    assert app._display_replay.snapshot() == []
    assert all("prior-session-output" not in line for line in app._output_lines)


@pytest.mark.asyncio
async def test_tui_app_state_restore_failure_keeps_prior_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    app._message_history = [ModelRequest(parts=[UserPromptPart(content="old history")])]
    app._display_replay.append({"type": "TEXT_MESSAGE_CHUNK", "messageId": "old", "delta": "prior-session-output"})
    app._restore_output_from_display_events(app._display_replay.snapshot())
    prior_history = app._message_history
    prior_replay = app._display_replay.snapshot()
    prior_output = list(app._output_lines)
    active_ctx = TUIContext(shell_env={"EXISTING": "value"}, steering_messages=["old steering"])
    app._runtime = MagicMock(ctx=active_ctx, env=None)
    monitor = MagicMock(spec=BackgroundMonitor)
    app._get_background_monitor = MagicMock(return_value=monitor)  # type: ignore[method-assign]
    failing_state = MagicMock()

    def fail_after_partial_restore(ctx: MagicMock) -> None:
        ctx.shell_env = {**ctx.shell_env, "INCOMING": "leak"}
        ctx.steering_messages = ["new steering"]
        raise RuntimeError("incompatible state")

    failing_state.restore.side_effect = fail_after_partial_restore
    monkeypatch.setattr(ResumableState, "model_validate_json", MagicMock(return_value=failing_state))

    load_dir = tmp_path / "session-b"
    load_dir.mkdir()
    (load_dir / "message_history.json").write_bytes(b"[]")
    (load_dir / "context_state.json").write_text("{}")

    await app._load_history(str(load_dir))

    assert app._message_history is prior_history
    assert app._display_replay.snapshot() == prior_replay
    assert app._output_lines[:-1] == prior_output
    assert "Error loading session: incompatible state" in app._output_lines[-1]
    assert app.runtime.ctx is active_ctx
    assert app.runtime.ctx.shell_env == {"EXISTING": "value"}
    assert app.runtime.ctx.steering_messages == ["old steering"]
    monitor.begin_session_reset.assert_not_called()
    monitor.reset_session_state.assert_not_called()


def test_tui_app_persist_stream_recoverable_state_updates_memory_only_on_interrupt():
    """Recoverable stream state should update in-memory history without saving session files."""
    config = MockConfig()
    config_manager = MockConfigManager()
    app = TUIApp(config=config, config_manager=config_manager)
    app._runtime = MagicMock()
    app._runtime.agent.model.model_name = "test-model"
    app._session_usage = MagicMock()
    app._session_usage.has_run_snapshot = True
    app._auto_save_history = MagicMock()
    app._save_session_snapshot = MagicMock()

    history = [
        ModelRequest(parts=[UserPromptPart(content="hello")]),
        ModelResponse(
            parts=[TextPart(content="partial")],
            metadata={"ya_agent_sdk": {"partial": True, "reason": "stream_interrupted"}},
        ),
    ]
    stream = MagicMock()
    stream.run = MagicMock()
    stream.run.usage.total_tokens = 123
    stream.recoverable_messages.return_value = history
    stream.exception = AgentInterrupted("Agent execution was interrupted")

    assert app._persist_stream_recoverable_state(stream) is True

    assert app._message_history == history
    assert app._last_run is stream.run
    app._auto_save_history.assert_not_called()
    app._save_session_snapshot.assert_not_called()


@pytest.mark.asyncio
async def test_tui_app_run_agent_reports_saved_recovery_session():
    """Agent errors should surface recovery guidance when session data is saved."""
    config = MockConfig()
    config.session = MagicMock(auto_save_history=True)
    config_manager = MockConfigManager()

    app = TUIApp(config=config, config_manager=config_manager)
    runtime = MagicMock()
    runtime.ctx = MagicMock(loop_active=False)
    runtime.ctx.steering_messages = []
    app._runtime = runtime
    app._execute_stream = AsyncMock(side_effect=RuntimeError("peer closed connection"))
    app._save_session_snapshot_async = AsyncMock(return_value=True)  # type: ignore[method-assign]
    app._check_pending_bus_messages = MagicMock()

    with patch.object(TUIApp, "has_session_data", new_callable=PropertyMock, return_value=True):
        await app._run_agent("hello")

    joined_output = "\n".join(app._output_lines)
    assert "Session state saved." in joined_output
    assert f"/session {app.session_id}" in joined_output
    assert app.state == TUIState.IDLE


@pytest.mark.asyncio
async def test_tui_app_model_command_opens_in_tui_selector_from_submit_dispatch() -> None:
    """Real /model submission should let its command owner open the selector."""
    config = YaacliConfig(
        general=GeneralConfig(model="openai-chat:gpt-4o"),
        model_profiles={
            "sonnet": ModelProfileConfig(label="Sonnet", model="anthropic:claude-sonnet-4-5"),
        },
    )
    config_manager = MockConfigManager()
    app = TUIApp(config=config, config_manager=config_manager)
    input_area = TextArea(text="/model", multiline=True)

    app._submit_input(input_area.text, input_area)

    command_task = app._foreground_command_task
    assert command_task is not None
    assert app.phase == TUIPhase.COMMAND_RUNNING
    await command_task

    assert input_area.text == ""
    assert app.phase == TUIPhase.IDLE
    assert app._model_selector_open is True
    assert [profile.id for profile in app._model_selector_profiles] == ["default", "sonnet"]
    assert app._model_selector_index == 0
    assert any("/model" in line for line in app._output_lines)
    assert not any("available after foreground work" in line for line in app._output_lines)


@pytest.mark.asyncio
async def test_tui_app_model_selector_rejects_non_owner_during_command_dispatch() -> None:
    """COMMAND_RUNNING does not authorize tasks other than the reserved command owner."""
    config = YaacliConfig(general=GeneralConfig(model="openai-chat:gpt-4o"))
    app = TUIApp(config=config, config_manager=MockConfigManager())
    command_owner = asyncio.create_task(_sleep_forever())
    app._foreground_command_task = command_owner
    app._set_phase(TUIPhase.COMMAND_RUNNING)

    try:
        await app._show_model_selector()
    finally:
        command_owner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await command_owner

    assert app._model_selector_open is False
    assert any("available after foreground work" in line for line in app._output_lines)


def test_tui_app_model_selector_escape_closes_without_toggling_mouse_mode() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    input_area = TextArea(multiline=True)
    app._model_selector_open = True
    app._model_selector_profiles = [MagicMock()]
    app._model_selector_index = 0
    app._mouse_enabled = True
    key_bindings = app._setup_keybindings(input_area)
    handle_escape = next(binding.handler for binding in key_bindings.bindings if binding.keys == (Keys.Escape,))

    handle_escape(MagicMock())

    assert app._model_selector_open is False
    assert app._model_selector_profiles == []
    assert app._model_selector_index == 0
    assert app._mouse_enabled is True


def test_tui_app_model_selector_movement_wraps() -> None:
    """Embedded model selector movement should stay inside TUI state."""
    config = YaacliConfig(
        general=GeneralConfig(model="openai-chat:gpt-4o"),
        model_profiles={
            "sonnet": ModelProfileConfig(label="Sonnet", model="anthropic:claude-sonnet-4-5"),
            "gemini": ModelProfileConfig(label="Gemini", model="google:gemini-2.5-pro"),
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
        general=GeneralConfig(model="openai-chat:gpt-4o"),
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


# =============================================================================
# Explicit Interaction Lifecycle Tests
# =============================================================================


def test_tui_app_clear_command_preserves_conversation_context() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    history = [ModelRequest(parts=[UserPromptPart(content="keep me")])]
    app._message_history = history
    app._append_output("old transcript")

    app._clear_transcript()

    assert app._message_history is history
    assert all("old transcript" not in line for line in app._output_lines)
    assert any("Conversation context is unchanged" in line for line in app._output_lines)


@pytest.mark.asyncio
async def test_tui_app_new_command_resets_context_and_changes_session_id() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    runtime = MagicMock()
    runtime.ctx = TUIContext(user_prompts="old prompt")
    runtime.env = None
    app._runtime = runtime
    app._get_background_monitor = MagicMock(return_value=None)  # type: ignore[method-assign]
    app._message_history = [ModelRequest(parts=[UserPromptPart(content="old")])]
    old_session_id = app.session_id

    await app._start_new_session()

    assert app.session_id != old_session_id
    assert app._message_history is None
    assert runtime.ctx.user_prompts is None
    assert any("New session" in line for line in app._output_lines)


@pytest.mark.asyncio
async def test_tui_app_unknown_slash_command_suggests_without_running_agent() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())

    await app._handle_command_inner("/hlep")

    assert app._agent_task is None
    assert any("Did you mean: /help" in line for line in app._output_lines)


@pytest.mark.asyncio
async def test_tui_app_submit_slash_command_preserves_visible_queued_attachments() -> None:
    """A visible chip is ignored for command routing without losing its binary."""
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    placeholder = "[Attached image 1: image/png 3B]"
    command_text = f"{placeholder} /attachments"
    input_area = TextArea(text=command_text, multiline=True)
    attachment = PendingAttachment(
        data=b"one",
        media_type="image/png",
        size_bytes=3,
        placeholder=placeholder,
    )
    app._pending_attachments.append(attachment)

    app._submit_input(command_text, input_area)
    command_task = app._foreground_command_task
    assert command_task is not None
    await command_task

    assert len(app._pending_attachments) == 1
    assert app._pending_attachments[0].data == attachment.data
    assert app._pending_attachments[0].placeholder == ""
    assert any("Queued images (1)" in line for line in app._output_lines)

    captured_runs: list[tuple[str, list[PendingAttachment] | None]] = []

    async def fake_run_agent(
        prompt: str,
        attachments: list[PendingAttachment] | None = None,
    ) -> None:
        captured_runs.append((prompt, attachments))

    app._run_agent = fake_run_agent  # type: ignore[method-assign]
    prompt_input = TextArea(text="hello", multiline=True)
    app._submit_input("hello", prompt_input)
    assert app._agent_task is not None
    await app._agent_task

    assert len(captured_runs) == 1
    prompt, attachments = captured_runs[0]
    assert prompt == "hello"
    assert attachments is not None
    assert [item.data for item in attachments] == [b"one"]


def test_tui_app_deleted_attachment_chip_drops_binary_before_control_dispatch() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    app._pending_attachments = [
        PendingAttachment(
            data=b"private",
            media_type="image/png",
            size_bytes=7,
            placeholder="[Attached image 1: image/png 7B]",
        )
    ]
    input_area = TextArea(text="/help", multiline=True)
    app._schedule_command = MagicMock()  # type: ignore[method-assign]

    app._submit_input("/help", input_area)

    app._schedule_command.assert_called_once_with("/help")
    assert app._pending_attachments == []


@pytest.mark.asyncio
async def test_tui_app_attachment_commands_list_and_remove_images() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    first = PendingAttachment(data=b"one", media_type="image/png", size_bytes=3)
    second = PendingAttachment(data=b"two", media_type="image/jpeg", size_bytes=3)
    app._pending_attachments.extend([first, second])

    await app._handle_command_inner("/attachments")
    await app._handle_command_inner("/remove-image 1")

    assert app._pending_attachments == [second]
    output = "\n".join(app._output_lines)
    assert "Queued images (2)" in output
    assert "Removed image/png" in output


def test_tui_app_tool_command_shows_complete_cross_turn_history_result() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    call_id = "call-123456"
    app._message_history = [
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="shell",
                    args={"command": "printf result"},
                    tool_call_id=call_id,
                )
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="shell",
                    content="line 1\nline 2\ncomplete result",
                    tool_call_id=call_id,
                )
            ]
        ),
    ]

    app._show_tool_result("call-123")

    output = "\n".join(app._output_lines)
    assert app._tool_messages == {}
    assert f"Tool shell [{call_id}]" in output
    assert '"command": "printf result"' in output
    assert "complete result" in output


@pytest.mark.asyncio
async def test_tui_app_deferred_flow_resolves_approvals_and_calls() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    deferred = DeferredToolRequests(
        approvals=[ToolCallPart(tool_name="edit", args={"path": "a.py"}, tool_call_id="approve-1")],
        calls=[ToolCallPart(tool_name="ask_user", args={"question": "value?"}, tool_call_id="call-1")],
    )
    responses = iter([(True, None), (True, "provided value")])

    async def fake_wait_for_input() -> tuple[bool, str | None]:
        return next(responses)

    app._wait_for_approval_input = fake_wait_for_input  # type: ignore[method-assign]
    app._set_phase(TUIPhase.THINKING)

    results = await app._request_user_action(deferred)

    assert results.approvals["approve-1"] is True
    call_result = results.calls["call-1"]
    assert isinstance(call_result, RetryPromptPart)
    assert call_result.content == "provided value"
    assert app.phase == TUIPhase.TOOL_CALLING


@pytest.mark.asyncio
async def test_tui_app_deferred_flow_collects_structured_clarifying_answers() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    deferred = DeferredToolRequests(
        calls=[
            ToolCallPart(
                tool_name="ask_user_question",
                args={
                    "questions": [
                        {
                            "question": "Which format?",
                            "header": "Format",
                            "options": [
                                {"label": "Summary", "description": "Brief"},
                                {"label": "Detailed", "description": "Complete"},
                            ],
                            "multiSelect": False,
                        },
                        {
                            "question": "Which sections?",
                            "header": "Sections",
                            "options": [
                                {"label": "Intro", "description": "Opening"},
                                {"label": "Conclusion", "description": "Ending"},
                            ],
                            "multiSelect": True,
                        },
                    ]
                },
                tool_call_id="question-1",
            )
        ]
    )
    responses = iter([(True, "2"), (True, "1, 2")])

    async def fake_wait_for_input() -> tuple[bool, str | None]:
        return next(responses)

    app._wait_for_approval_input = fake_wait_for_input  # type: ignore[method-assign]
    app._set_phase(TUIPhase.THINKING)

    results = await app._request_user_action(deferred)

    call_result = results.calls["question-1"]
    assert isinstance(call_result, dict)
    assert call_result["answers"] == {
        "Which format?": "Detailed",
        "Which sections?": ["Intro", "Conclusion"],
    }
    assert app.phase == TUIPhase.TOOL_CALLING


@pytest.mark.asyncio
async def test_tui_app_deferred_flow_records_explicit_denials() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    deferred = DeferredToolRequests(
        approvals=[ToolCallPart(tool_name="shell", args={}, tool_call_id="approve-1")],
        calls=[ToolCallPart(tool_name="ask_user", args={}, tool_call_id="call-1")],
    )
    responses = iter([(False, "unsafe"), (False, "not available")])

    async def fake_wait_for_input() -> tuple[bool, str | None]:
        return next(responses)

    app._wait_for_approval_input = fake_wait_for_input  # type: ignore[method-assign]
    app._set_phase(TUIPhase.THINKING)

    results = await app._request_user_action(deferred)

    approval = results.approvals["approve-1"]
    assert isinstance(approval, ToolDenied)
    assert approval.message == "unsafe"
    call_result = results.calls["call-1"]
    assert isinstance(call_result, RetryPromptPart)
    assert "denied by user: not available" in str(call_result.content)


def test_tui_app_rejects_invalid_phase_transition_without_mutating_coarse_state() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())

    app._set_phase(TUIPhase.TOOL_CALLING)

    assert app.phase == TUIPhase.IDLE
    assert app.state == TUIState.IDLE


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "0s"),
        (59.9, "59s"),
        (60, "1m 00s"),
        (185, "3m 05s"),
        (3600, "1h 00m 00s"),
        (7384, "2h 03m 04s"),
    ],
)
def test_format_elapsed_duration_uses_compact_units(seconds: float, expected: str) -> None:
    assert _format_elapsed_duration(seconds) == expected


def test_tui_app_status_height_wraps_to_terminal_width() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    app._app = MagicMock()
    app._app.output.get_size.return_value = MagicMock(rows=24, columns=12)
    app._get_terminal_width = lambda: 12  # type: ignore[method-assign]
    app._get_status_text = lambda: [("class:status-bar", "1234567890123456789012345")]  # type: ignore[method-assign]

    assert app._get_status_height() == 3
    assert app._get_viewport_height() == 17


def test_tui_app_status_height_respects_explicit_newlines() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    app._get_terminal_width = lambda: 20  # type: ignore[method-assign]
    app._get_status_text = lambda: [("class:status-bar", "first\nsecond")]

    assert app._get_status_height() == 2


def test_tui_app_status_height_wraps_wide_characters_at_cell_boundaries() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    app._get_terminal_width = lambda: 3  # type: ignore[method-assign]
    app._get_status_text = lambda: [("class:status-bar", "中中中")]

    assert app._get_status_height() == 3


def test_tui_app_status_starts_with_phase_without_an_agent_mode_badge() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())

    fragments = app._get_status_text()
    status = "".join(text for _style, text in fragments)

    assert status.startswith(" Ready")
    assert "ACT" not in status
    assert "PLAN" not in status
    assert all("mode-" not in style for style, _text in fragments)


@pytest.mark.parametrize(
    ("width", "show_token_usage", "shows_model", "shows_context"),
    [
        (99, True, False, True),
        (99, False, False, False),
        (100, True, True, True),
        (100, False, True, False),
    ],
)
def test_tui_app_status_respects_compact_and_token_usage_settings(
    width: int,
    show_token_usage: bool,
    shows_model: bool,
    shows_context: bool,
) -> None:
    config = YaacliConfig(display=DisplayConfig(show_token_usage=show_token_usage))
    app = TUIApp(config=config, config_manager=MockConfigManager())
    app._get_terminal_width = lambda: width  # type: ignore[method-assign]
    app._current_context_tokens = 50_000
    app._context_window_size = 200_000

    status = "".join(text for _style, text in app._get_status_text())

    assert (app._format_active_model_label() in status) is shows_model
    assert ("ctx 25%" in status) is shows_context


@pytest.mark.parametrize("phase", [TUIPhase.SAVING, TUIPhase.CANCELLING])
def test_tui_app_cleanup_phases_advertise_wait_instead_of_send_or_steer(phase: TUIPhase) -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    if phase == TUIPhase.CANCELLING:
        app._set_phase(TUIPhase.THINKING)
    app._set_phase(phase)

    status = "".join(text for _style, text in app._get_status_text())
    assert "Please wait for foreground cleanup" in status
    assert "Enter steers" not in status
    assert "Enter send" not in status
    assert app._get_prompt() == "[scroll:wait] > "


_BUSY_PHASE_PATHS = [
    pytest.param((TUIPhase.THINKING,), id="thinking"),
    pytest.param((TUIPhase.THINKING, TUIPhase.TOOL_CALLING), id="tool-calling"),
    pytest.param((TUIPhase.THINKING, TUIPhase.AWAITING_APPROVAL), id="awaiting-approval"),
    pytest.param((TUIPhase.THINKING, TUIPhase.STREAMING_OUTPUT), id="streaming-output"),
    pytest.param((TUIPhase.SHELL_RUNNING,), id="shell-running"),
    pytest.param((TUIPhase.COMMAND_RUNNING,), id="command-running"),
    pytest.param((TUIPhase.SAVING,), id="saving"),
    pytest.param((TUIPhase.THINKING, TUIPhase.CANCELLING), id="cancelling"),
]


@pytest.mark.parametrize("phase_path", _BUSY_PHASE_PATHS)
@pytest.mark.parametrize("command", sorted(BUSY_CONTROL_COMMANDS))
def test_tui_app_busy_control_commands_never_become_steering(
    command: str,
    phase_path: tuple[TUIPhase, ...],
) -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    input_area = TextArea(text=command, multiline=True)
    for phase in phase_path:
        app._set_phase(phase)
    app._schedule_command = MagicMock()  # type: ignore[method-assign]
    app._add_steering_message = MagicMock()  # type: ignore[method-assign]

    app._submit_input(command, input_area)

    app._schedule_command.assert_called_once_with(command)
    app._add_steering_message.assert_not_called()
    assert input_area.buffer.text == ""


@pytest.mark.parametrize("phase_path", _BUSY_PHASE_PATHS)
@pytest.mark.parametrize(
    "command",
    [
        "/clear",
        "/dump",
        "/exit",
        "/goal finish this",
        "/load snapshot",
        "/model",
        "/new",
        "/review",
        "/session abc123",
    ],
)
def test_tui_app_idle_only_commands_are_rejected_while_busy_without_losing_draft(
    command: str,
    phase_path: tuple[TUIPhase, ...],
) -> None:
    config = MockConfig(commands={"review": CommandDefinition(prompt="Review the current changes")})
    app = TUIApp(config=config, config_manager=MockConfigManager())
    input_area = TextArea(text=command, multiline=True)
    for phase in phase_path:
        app._set_phase(phase)
    app._schedule_command = MagicMock()  # type: ignore[method-assign]
    app._add_steering_message = MagicMock()  # type: ignore[method-assign]

    app._submit_input(command, input_area)

    app._schedule_command.assert_not_called()
    app._add_steering_message.assert_not_called()
    assert input_area.buffer.text == command
    phase_label = phase_path[-1].name.replace("_", " ").lower()
    assert any(f"is unavailable while foreground work is {phase_label}" in line for line in app._output_lines)


@pytest.mark.parametrize("phase_path", _BUSY_PHASE_PATHS)
def test_tui_app_busy_unrecognized_slash_text_follows_ordinary_input_routing(
    phase_path: tuple[TUIPhase, ...],
) -> None:
    text = "/home/user/project/file.txt"
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    input_area = TextArea(text=text, multiline=True)
    for phase in phase_path:
        app._set_phase(phase)
    app._add_steering_message = MagicMock()  # type: ignore[method-assign]

    app._submit_input(text, input_area)

    if app._accepts_steering():
        app._add_steering_message.assert_called_once_with(text)
        assert input_area.buffer.text == ""
    else:
        app._add_steering_message.assert_not_called()
        assert input_area.buffer.text == text
        assert any("Please wait or use /cancel" in line for line in app._output_lines)


@pytest.mark.parametrize("removed_command", ["/act", "/background", "/plan", "/tasks"])
def test_tui_app_removed_commands_are_ordinary_steering(removed_command: str) -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    input_area = TextArea(text=removed_command, multiline=True)
    app._set_phase(TUIPhase.THINKING)
    app._add_steering_message = MagicMock()  # type: ignore[method-assign]

    app._submit_input(removed_command, input_area)

    app._add_steering_message.assert_called_once_with(removed_command)
    assert input_area.buffer.text == ""
    assert removed_command not in app._command_words()


@pytest.mark.parametrize("phase_path", _BUSY_PHASE_PATHS)
def test_tui_app_busy_direct_shell_syntax_is_rejected_without_leaking_to_model(
    phase_path: tuple[TUIPhase, ...],
) -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    input_area = TextArea(text="!printf secret", multiline=True)
    for phase in phase_path:
        app._set_phase(phase)
    app._add_steering_message = MagicMock()  # type: ignore[method-assign]

    app._submit_input("!printf secret", input_area)

    app._add_steering_message.assert_not_called()
    assert app._direct_shell_task is None
    assert input_area.buffer.text == "!printf secret"
    assert any("Direct shell commands are unavailable" in line for line in app._output_lines)


@pytest.mark.asyncio
async def test_tui_app_active_integrate_command_delivers_to_current_run() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    input_area = TextArea(text="/integrate", multiline=True)
    app._set_phase(TUIPhase.THINKING)
    app._deliver_background_messages = MagicMock(return_value=True)  # type: ignore[method-assign]
    app._launch_agent = MagicMock()  # type: ignore[method-assign]
    app._add_steering_message = MagicMock()  # type: ignore[method-assign]

    app._submit_input("/integrate", input_area)
    command_tasks = tuple(app._managed_tasks)
    await asyncio.gather(*command_tasks)

    app._deliver_background_messages.assert_called_once_with()
    app._launch_agent.assert_not_called()
    app._add_steering_message.assert_not_called()
    assert app._background_results_ready is True
    assert app._pending_bus_check_needed is True
    assert input_area.buffer.text == ""
    assert any("delivered to the active run" in line for line in app._output_lines)


@pytest.mark.asyncio
async def test_tui_app_active_integrate_command_with_no_results_stays_local() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    input_area = TextArea(text="/integrate", multiline=True)
    app._set_phase(TUIPhase.THINKING)
    app._deliver_background_messages = MagicMock(return_value=False)  # type: ignore[method-assign]
    app._launch_agent = MagicMock()  # type: ignore[method-assign]
    app._add_steering_message = MagicMock()  # type: ignore[method-assign]

    app._submit_input("/integrate", input_area)
    await asyncio.gather(*tuple(app._managed_tasks))

    app._launch_agent.assert_not_called()
    app._add_steering_message.assert_not_called()
    assert app._background_results_ready is False
    assert any("No background results are ready" in line for line in app._output_lines)


@pytest.mark.asyncio
async def test_tui_app_idle_integrate_command_starts_explicit_integration_turn() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    input_area = TextArea(text="/integrate", multiline=True)
    app._deliver_background_messages = MagicMock(return_value=True)  # type: ignore[method-assign]
    app._launch_agent = MagicMock()  # type: ignore[method-assign]

    app._submit_input("/integrate", input_area)
    command_task = app._foreground_command_task
    assert command_task is not None
    await command_task

    app._launch_agent.assert_called_once_with("")
    assert any("Integrating background results" in line for line in app._output_lines)


@pytest.mark.asyncio
async def test_tui_app_cleanup_integrate_command_keeps_results_queued() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    app._set_phase(TUIPhase.SAVING)
    app._background_results_ready = True
    app._deliver_background_messages = MagicMock(return_value=True)  # type: ignore[method-assign]
    app._launch_agent = MagicMock()  # type: ignore[method-assign]

    app._integrate_background_results()

    app._deliver_background_messages.assert_not_called()
    app._launch_agent.assert_not_called()
    assert app._background_results_ready is True
    assert any("remain queued" in line for line in app._output_lines)


def test_tui_app_user_steering_does_not_count_as_background_result() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    ctx = TUIContext()
    ctx.message_bus.subscribe(ctx.agent_id)
    runtime = MagicMock(ctx=ctx)
    runtime.env.resources = {}
    app._runtime = runtime
    ctx.message_bus.send(BusMessage(content="guidance", source="user", target=ctx.agent_id))

    assert app._deliver_background_messages() is False

    ctx.message_bus.send(BusMessage(content="result", source="subagent-1", target=ctx.agent_id))
    assert app._deliver_background_messages() is True


@pytest.mark.parametrize("phase", [TUIPhase.SAVING, TUIPhase.CANCELLING])
def test_tui_app_cleanup_phase_takes_priority_over_stale_hitl_state(phase: TUIPhase) -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    input_area = TextArea(text="provided value", multiline=True)
    app._set_phase(TUIPhase.THINKING)
    app._set_phase(TUIPhase.AWAITING_APPROVAL)
    app._set_phase(phase)
    app._hitl_pending = True
    app._approval_kind = "call"
    app._approval_event = asyncio.Event()
    app._add_steering_message = MagicMock()  # type: ignore[method-assign]
    key_bindings = app._setup_input_keybindings(input_area)
    handle_enter = next(binding.handler for binding in key_bindings.bindings if binding.keys == (Keys.ControlM,))

    handle_enter(MagicMock())

    app._add_steering_message.assert_not_called()
    assert app._approval_result is None
    assert app._approval_event.is_set() is False
    assert input_area.buffer.text == "provided value"
    assert any("Please wait or use /cancel" in line for line in app._output_lines)


@pytest.mark.parametrize("approval_kind", ["approval", "call"])
@pytest.mark.parametrize("control_text", ["/new", "/review", "!printf secret"])
def test_tui_app_hitl_rejects_idle_only_controls_without_resolving_request(
    approval_kind: str,
    control_text: str,
) -> None:
    config = MockConfig(commands={"review": CommandDefinition(prompt="Review the current changes")})
    app = TUIApp(config=config, config_manager=MockConfigManager())
    input_area = TextArea(text=control_text, multiline=True)
    app._set_phase(TUIPhase.THINKING)
    app._set_phase(TUIPhase.AWAITING_APPROVAL)
    app._hitl_pending = True
    app._approval_kind = approval_kind
    app._approval_event = asyncio.Event()
    app._schedule_command = MagicMock()  # type: ignore[method-assign]
    app._add_steering_message = MagicMock()  # type: ignore[method-assign]
    key_bindings = app._setup_input_keybindings(input_area)
    handle_enter = next(binding.handler for binding in key_bindings.bindings if binding.keys == (Keys.ControlM,))

    handle_enter(MagicMock())

    app._schedule_command.assert_not_called()
    app._add_steering_message.assert_not_called()
    assert app._approval_result is None
    assert app._approval_event.is_set() is False
    assert input_area.buffer.text == control_text


@pytest.mark.parametrize("phase", [None, TUIPhase.THINKING])
def test_tui_app_attachment_chip_cannot_hide_slash_control_input(phase: TUIPhase | None) -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    placeholder = app._format_attachment_placeholder(1, "image/png", 3)
    app._pending_attachments = [
        PendingAttachment(
            data=b"img",
            media_type="image/png",
            size_bytes=3,
            placeholder=placeholder,
        )
    ]
    command_text = f"{placeholder} /help"
    input_area = TextArea(text=command_text, multiline=True)
    if phase is not None:
        app._set_phase(phase)
    app._schedule_command = MagicMock()  # type: ignore[method-assign]
    app._add_steering_message = MagicMock()  # type: ignore[method-assign]

    app._submit_input(command_text, input_area)

    app._schedule_command.assert_called_once_with("/help")
    app._add_steering_message.assert_not_called()
    assert input_area.buffer.text == ""
    assert len(app._pending_attachments) == 1
    assert app._pending_attachments[0].data == b"img"
    assert app._pending_attachments[0].placeholder == ""


@pytest.mark.asyncio
async def test_tui_app_attachment_chip_cannot_hide_direct_shell_input() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    placeholder = app._format_attachment_placeholder(1, "image/png", 3)
    app._pending_attachments = [
        PendingAttachment(
            data=b"img",
            media_type="image/png",
            size_bytes=3,
            placeholder=placeholder,
        )
    ]
    command_text = f"{placeholder} !pwd"
    input_area = TextArea(text=command_text, multiline=True)
    commands: list[str] = []

    async def capture_shell(command: str) -> None:
        commands.append(command)

    app._execute_shell_command = capture_shell  # type: ignore[method-assign]

    app._submit_input(command_text, input_area)
    assert app._direct_shell_task is not None
    await app._direct_shell_task

    assert commands == ["pwd"]
    assert input_area.buffer.text == ""
    assert len(app._pending_attachments) == 1
    assert app._pending_attachments[0].placeholder == ""


@pytest.mark.parametrize(
    "phase",
    [
        TUIPhase.THINKING,
        TUIPhase.TOOL_CALLING,
        TUIPhase.AWAITING_APPROVAL,
        TUIPhase.STREAMING_OUTPUT,
    ],
)
async def test_tui_app_running_input_is_immediate_steering_without_queue(phase: TUIPhase) -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    input_area = TextArea(multiline=True)
    app._set_phase(TUIPhase.THINKING)
    if phase != TUIPhase.THINKING:
        app._set_phase(phase)
    app._add_steering_message = MagicMock()  # type: ignore[method-assign]

    app._submit_input("focus on the failing test", input_area)

    app._add_steering_message.assert_called_once_with("focus on the failing test")
    assert input_area.buffer.text == ""
    assert app._agent_task is None
    assert not hasattr(app, "_queued_prompts")
    assert any("Guidance sent to the active run" in line for line in app._output_lines)


@pytest.mark.asyncio
async def test_tui_app_shell_claim_blocks_back_to_back_prompt_and_second_shell() -> None:
    """A shell submission must own the foreground before its coroutine gets CPU time."""
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    first_input = TextArea(multiline=True)
    second_input = TextArea(text="next prompt", multiline=True)
    third_input = TextArea(text="!second", multiline=True)
    release = asyncio.Event()
    calls: list[str] = []

    async def fake_shell(command: str) -> None:
        calls.append(command)
        await release.wait()

    app._execute_shell_command = fake_shell  # type: ignore[method-assign]
    app._submit_input("!first", first_input)
    first_task = app._direct_shell_task

    assert first_task is not None
    assert app.phase == TUIPhase.SHELL_RUNNING
    app._submit_input("next prompt", second_input)
    app._submit_input("!second", third_input)

    assert app._agent_task is None
    assert app._direct_shell_task is first_task
    assert second_input.buffer.text == "next prompt"
    assert third_input.buffer.text == "!second"
    release.set()
    await first_task
    assert calls == ["first"]


def test_tui_app_saving_phase_rejects_new_prompt() -> None:
    """Submitting while persistence owns the foreground must not launch an agent."""
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    input_area = TextArea(text="do not race save", multiline=True)
    app._set_phase(TUIPhase.SAVING)

    app._submit_input("do not race save", input_area)

    assert app._agent_task is None
    assert input_area.buffer.text == "do not race save"
    assert any("saving" in line.lower() for line in app._output_lines)


@pytest.mark.asyncio
async def test_tui_app_idle_slash_command_blocks_immediate_prompt_dispatch() -> None:
    """Session-mutating slash commands must reserve dispatch before their coroutine runs."""
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    command_input = TextArea(text="/new", multiline=True)
    prompt_input = TextArea(text="do not race new", multiline=True)
    release = asyncio.Event()

    async def fake_handle_inner(command: str) -> None:
        assert command == "/new"
        await release.wait()

    app._handle_command_inner = fake_handle_inner  # type: ignore[method-assign]
    app._submit_input("/new", command_input)
    command_task = app._foreground_command_task

    assert command_task is not None
    assert app.phase == TUIPhase.COMMAND_RUNNING
    assert app.state == TUIState.RUNNING
    app._submit_input("do not race new", prompt_input)
    assert app._agent_task is None
    assert prompt_input.buffer.text == "do not race new"
    assert any("command running" in line.lower() for line in app._output_lines)

    release.set()
    await command_task
    assert app._foreground_command_task is None
    assert app.phase == TUIPhase.IDLE


@pytest.mark.asyncio
async def test_tui_app_non_agent_command_blocks_exit_and_cancels_from_authoritative_phase() -> None:
    """Slow slash commands share the same busy, Ctrl+D, and cancellation boundary."""
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    input_area = TextArea(multiline=True)
    started = asyncio.Event()

    async def slow_command(command: str) -> None:
        assert command == "/new"
        started.set()
        await asyncio.sleep(3600)

    app._handle_command_inner = slow_command  # type: ignore[method-assign]
    app._schedule_command("/new")
    command_task = app._foreground_command_task
    assert command_task is not None
    await started.wait()
    assert app.phase == TUIPhase.COMMAND_RUNNING

    key_bindings = app._setup_keybindings(input_area)
    handle_ctrl_d = next(binding.handler for binding in key_bindings.bindings if binding.keys == (Keys.ControlD,))
    exit_event = MagicMock()
    handle_ctrl_d(exit_event)

    exit_event.app.exit.assert_not_called()
    assert any("Foreground work is active" in line for line in app._output_lines)

    app._cancel_foreground()
    assert app.phase == TUIPhase.CANCELLING
    with pytest.raises(asyncio.CancelledError):
        await command_task
    await asyncio.sleep(0)

    assert app._foreground_command_task is None
    assert app.phase == TUIPhase.IDLE


@pytest.mark.asyncio
async def test_tui_app_idle_cancel_command_does_not_cancel_itself() -> None:
    """An idle /cancel command should report idle instead of cancelling its own task."""
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())

    app._schedule_command("/cancel")
    command_task = app._foreground_command_task
    assert command_task is not None
    await command_task
    await asyncio.sleep(0)

    assert not command_task.cancelled()
    assert app._foreground_command_task is None
    assert any("Nothing is running" in line for line in app._output_lines)
    assert app.phase == TUIPhase.IDLE


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("task_field", "phase"),
    [
        ("_direct_shell_task", TUIPhase.SHELL_RUNNING),
        ("_agent_task", TUIPhase.THINKING),
    ],
)
async def test_tui_app_repeated_cancel_does_not_interrupt_cleanup(
    task_field: str,
    phase: TUIPhase,
) -> None:
    """Repeated /cancel must not add another cancellation request to an active task."""
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    task = asyncio.create_task(_sleep_forever())
    setattr(app, task_field, task)
    app._set_phase(phase)

    app._cancel_foreground()
    assert task.cancelling() == 1
    app._cancel_foreground()

    assert task.cancelling() == 1
    assert any("already in progress" in line for line in app._output_lines)
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_tui_app_saving_rejects_cancel_and_ctrl_c_exit() -> None:
    """Persistence is non-cancellable and Ctrl+C cannot enter the idle exit path."""
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    input_area = TextArea(multiline=True)
    agent_task = asyncio.create_task(_sleep_forever())
    app._agent_task = agent_task
    app._set_phase(TUIPhase.SAVING)

    app._cancel_foreground()
    key_bindings = app._setup_keybindings(input_area)
    handle_ctrl_c = next(binding.handler for binding in key_bindings.bindings if binding.keys == (Keys.ControlC,))
    event = MagicMock()
    handle_ctrl_c(event)

    assert agent_task.cancelling() == 0
    assert app.phase == TUIPhase.SAVING
    assert sum("snapshot is being saved" in line for line in app._output_lines) == 2
    event.app.exit.assert_not_called()

    agent_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await agent_task


@pytest.mark.asyncio
async def test_tui_app_prestart_command_cancellation_releases_foreground() -> None:
    """Cancelling before command coroutine startup must release timer and phase."""
    config = MockConfig(commands={"commit": CommandDefinition(prompt="commit changes")})
    app = TUIApp(config=config, config_manager=MockConfigManager())
    input_area = TextArea(text="/commit", multiline=True)

    app._submit_input("/commit", input_area)
    command_task = app._foreground_command_task
    assert command_task is not None
    app._cancel_foreground()
    with pytest.raises(asyncio.CancelledError):
        await command_task
    await asyncio.sleep(0)

    assert command_task.cancelled()
    assert app._foreground_command_task is None
    assert app._agent_task is None
    assert app._run_started_at is None
    assert app.phase == TUIPhase.IDLE


@pytest.mark.asyncio
async def test_tui_app_custom_command_reserves_before_async_dispatch() -> None:
    """A custom slash command and immediate prompt cannot launch concurrent agent turns."""
    config = MockConfig(commands={"commit": CommandDefinition(prompt="commit changes")})
    app = TUIApp(config=config, config_manager=MockConfigManager())
    command_input = TextArea(text="/commit", multiline=True)
    steering_input = TextArea(text="focus tests", multiline=True)
    release = asyncio.Event()
    prompts: list[str] = []

    async def fake_run_agent(prompt: str, attachments: object = None) -> None:
        prompts.append(prompt)
        await release.wait()

    app._run_agent = fake_run_agent  # type: ignore[method-assign]
    app._add_steering_message = MagicMock()  # type: ignore[method-assign]
    app._submit_input("/commit", command_input)
    command_task = app._foreground_command_task

    assert command_task is not None
    assert app.phase == TUIPhase.THINKING
    assert app._run_started_at is not None
    app._submit_input("focus tests", steering_input)
    app._add_steering_message.assert_called_once_with("focus tests")

    await command_task
    assert app._agent_task is not None
    release.set()
    await app._agent_task
    assert prompts == ["commit changes"]


def test_tui_app_live_text_sets_streaming_phase_but_replay_does_not() -> None:
    """Only live main-agent text should mutate the interaction phase."""
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    app._set_phase(TUIPhase.THINKING)
    app._handle_stream_event(
        StreamEvent(
            agent_id="main",
            agent_name="main",
            event=PartStartEvent(index=0, part=TextPart(content="hello")),
        )
    )
    assert app.phase == TUIPhase.STREAMING_OUTPUT

    restored = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    restored._restore_output_from_display_events([
        {"type": "TEXT_MESSAGE_START", "messageId": "old"},
        {"type": "TEXT_MESSAGE_CHUNK", "messageId": "old", "delta": "history"},
    ])
    assert restored.phase == TUIPhase.IDLE


@pytest.mark.asyncio
async def test_tui_app_elapsed_timer_starts_when_agent_task_is_launched() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    release = asyncio.Event()

    async def fake_run_agent(prompt: str, attachments: object = None) -> None:
        await release.wait()

    app._run_agent = fake_run_agent  # type: ignore[method-assign]
    submitted_at = time.monotonic()

    app._launch_agent("hello")

    assert app.phase == TUIPhase.THINKING
    assert app._run_started_at is not None
    assert app._run_started_at >= submitted_at
    run_started_at = app._run_started_at
    for phase in (TUIPhase.TOOL_CALLING, TUIPhase.STREAMING_OUTPUT, TUIPhase.SAVING):
        app._set_phase(phase)
        assert app._run_started_at == run_started_at
    assert app._agent_task is not None
    release.set()
    await app._agent_task


@pytest.mark.asyncio
async def test_tui_app_immediate_agent_cancellation_clears_timer_and_phase() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    release = asyncio.Event()

    async def fake_run_agent(prompt: str, attachments: object = None) -> None:
        await release.wait()

    app._run_agent = fake_run_agent  # type: ignore[method-assign]
    app._launch_agent("hello")
    assert app._agent_task is not None

    app._agent_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await app._agent_task
    await asyncio.sleep(0)

    assert app._run_started_at is None
    assert app.phase == TUIPhase.IDLE
