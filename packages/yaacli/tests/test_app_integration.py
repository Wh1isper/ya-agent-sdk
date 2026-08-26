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
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prompt_toolkit import Application
from prompt_toolkit.completion import Completion
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import ConditionalContainer, FloatContainer, HSplit, Layout, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.utils import Event
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
from ya_agent_environment.shell import BackgroundProcess
from ya_agent_sdk.context import AvailableSkill, StreamEvent, TaskManager, TaskStatus
from ya_agent_sdk.events import (
    NamespaceStatus,
    NamespaceStatusEvent,
    TaskEvent,
)
from ya_agent_sdk.usage import CostEstimate

# Import the components we're testing
from yaacli.app import BUSY_CONTROL_COMMANDS, TUIApp, TUIState
from yaacli.app.shell import ComposeSnapshot
from yaacli.app.state import TUIPhase
from yaacli.app.tui import (
    USER_INPUT_TIMEOUT_PROMPT,
    PendingAttachment,
    _BoundedOutputTail,
    _drain_direct_shell_stream,
    _format_direct_shell_truncation_note,
    _format_elapsed_duration,
    _is_benign_contextvar_cleanup_error,
)
from yaacli.clipboard import ClipboardImage, ClipboardImageReadResult
from yaacli.config import (
    CommandDefinition,
    DisplayConfig,
    GeneralConfig,
    ModelProfileConfig,
    NotificationConfig,
    ToolsConfig,
    YaacliConfig,
)
from yaacli.durable.models import InputState, SessionStatus, SessionSummary
from yaacli.model_profiles import ResolvedModelProfile, build_model_profiles
from yaacli.session import TUIContext
from yaacli.shell_monitor import ShellNotification
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
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
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


def _stub_agent_turn_acceptance(app: TUIApp) -> None:
    app._accept_agent_turn = MagicMock(  # type: ignore[method-assign]
        return_value=MagicMock(logical_run_id="accepted-test-run")
    )


def _session_info(
    session_id: str,
    *,
    working_dir: str = "/workspace",
    input_text: str | None = "last input",
    output_text: str | None = "last output",
    updated_at: str = "2026-01-01T12:34:56+00:00",
) -> SessionSummary:
    return SessionSummary(
        session_id=session_id,
        workspace_ref=working_dir,
        status=SessionStatus.active,
        head_revision_id="revision-1",
        created_at=datetime.fromisoformat("2026-01-01T00:00:00+00:00"),
        updated_at=datetime.fromisoformat(updated_at),
        input_preview=input_text,
        output_preview=output_text,
        message_count=2,
        display_event_count=3,
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
async def test_tui_model_selector_activates_registered_gateway_websocket_plan(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The selector activates the exact prebuilt plan for a gateway websocket profile."""
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
                instructions="Use the websocket profile instructions.",
            )
        },
    )
    config_manager = MockConfigManager(config_dir=tmp_path)
    app = TUIApp(config=config, config_manager=config_manager)

    runtime = MagicMock()
    runtime.capabilities = []
    runtime.ctx.injected_context_tags = ()
    runtime.ctx.model_cfg.context_window = 270000
    plan = MagicMock(runtime=runtime)
    worker = MagicMock()
    worker.activate = AsyncMock(return_value=plan)
    skill_toolset = MagicMock()
    skill_toolset.refresh_context = AsyncMock()
    app._execution_worker = worker
    app._skill_toolsets = {"ws": skill_toolset}
    app._model_selector_open = True
    app._model_selector_profiles = build_model_profiles(config)
    app._model_selector_index = 1

    await app._apply_model_selector_selection()

    worker.activate.assert_awaited_once_with("ws")
    skill_toolset.refresh_context.assert_awaited_once_with(runtime.ctx)
    assert app._runtime is runtime
    assert app._active_model_profile == ResolvedModelProfile(
        id="ws",
        label="Responses WS Gateway",
        model="gateway@openai-responses-ws:gpt-5",
        model_settings="openai_responses_default",
        model_cfg="gpt5_270k",
        instructions="Use the websocket profile instructions.",
    )
    assert app._context_window_size == 270000
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
        "/cost",
        "/help",
        "/paste-image",
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

    monkeypatch.setattr("yaacli.app.shell.Application", capture_application)

    await app.run()

    app._cancel_agent_task.assert_not_awaited()
    app._cancel_managed_tasks.assert_not_awaited()
    assert "min_redraw_interval" not in captured
    assert app._invalidate_interval == 1 / 24
    assert app._stream_render_interval == 1 / 15

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


@pytest.mark.asyncio
async def test_tui_app_handoff_restores_draft_and_submits_after_initial_render() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    app._theme_terminal_resolved = True
    app._restore_startup_session = AsyncMock(return_value=False)  # type: ignore[method-assign]
    input_area = TextArea(multiline=True)
    application = MagicMock()
    application._handle_exception = MagicMock()
    application.after_render = Event(application)
    events: list[str] = []

    async def run_async() -> None:
        events.append("initial-render")
        application.after_render.fire()
        await asyncio.sleep(0)

    application.run_async = run_async
    app._build_tui_shell = MagicMock(return_value=(application, input_area))  # type: ignore[method-assign]
    app._submit_input = MagicMock(side_effect=lambda *_args: events.append("submit"))  # type: ignore[method-assign]
    snapshot = ComposeSnapshot(
        text="queued prompt",
        cursor_position=4,
        submit_when_ready=True,
        input_mode="send",
        mouse_enabled=False,
    )

    await app.run(initial_compose=snapshot)

    assert input_area.buffer.text == "queued prompt"
    assert input_area.buffer.cursor_position == 4
    assert app._input_mode == "send"
    assert app._mouse_enabled is False
    application.output.disable_mouse_support.assert_called_once_with()
    app._submit_input.assert_called_once_with("queued prompt", input_area)
    assert events == ["initial-render", "submit"]


@pytest.mark.asyncio
async def test_tui_app_does_not_submit_handoff_draft_when_initial_render_fails() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    app._theme_terminal_resolved = True
    app._restore_startup_session = AsyncMock(return_value=False)  # type: ignore[method-assign]
    input_area = TextArea(multiline=True)
    application = MagicMock()
    application._handle_exception = MagicMock()
    application.after_render = Event(application)

    async def run_async() -> None:
        raise RuntimeError("initial render failed")

    application.run_async = run_async
    app._build_tui_shell = MagicMock(return_value=(application, input_area))  # type: ignore[method-assign]
    app._submit_input = MagicMock()  # type: ignore[method-assign]
    snapshot = ComposeSnapshot(text="queued prompt", cursor_position=4, submit_when_ready=True)

    with pytest.raises(RuntimeError, match="initial render failed"):
        await app.run(initial_compose=snapshot)
    await asyncio.sleep(0)

    assert input_area.buffer.text == "queued prompt"
    app._submit_input.assert_not_called()


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


def test_tui_app_reports_unavailable_optional_mcp_once_per_status() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    stream_event = StreamEvent(
        agent_id="main",
        agent_name="main",
        event=NamespaceStatusEvent(
            event_id="mcp-init",
            namespace_status={"offline": NamespaceStatus.skipped},
        ),
    )

    app._handle_stream_event(stream_event)
    app._handle_stream_event(stream_event)
    app._handle_stream_event(
        StreamEvent(
            agent_id="main",
            agent_name="main",
            event=NamespaceStatusEvent(
                event_id="mcp-recovered",
                namespace_status={"offline": NamespaceStatus.connected},
            ),
        )
    )
    app._handle_stream_event(stream_event)

    assert len(app._output_lines) == 2
    assert all(
        "Optional MCP server 'offline' failed to connect; continuing without it." in line for line in app._output_lines
    )


# =============================================================================
# Steering Message Tests
# =============================================================================


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
    _stub_agent_turn_acceptance(app)
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

    async def run_shell() -> None:
        with patch("yaacli.app.tui.asyncio.create_subprocess_shell", new=AsyncMock(return_value=process)):
            await app._execute_shell_command("sleep forever")

    task = asyncio.create_task(run_shell())
    task.add_done_callback(app._release_direct_shell_task)
    await task
    await asyncio.sleep(0)

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
        assert app._pending_attachments == [original]
        clipboard_result = ClipboardImageReadResult(image=ClipboardImage(data=b"new", media_type="image/png"))
        with patch("yaacli.app.tui.read_clipboard_image", new=AsyncMock(return_value=clipboard_result)):
            await app._paste_clipboard_image(input_area)

        assert [item.data for item in app._pending_attachments] == [b"old", b"new"]
        assert app._pending_attachments[-1].placeholder in input_area.buffer.text
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
    _stub_agent_turn_acceptance(app)

    app._submit_input("/commit polish tests", input_area)
    assert input_area.buffer.text == ""

    command_task = next(iter(app._managed_tasks))
    await command_task
    assert app._agent_task is not None
    await app._agent_task

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


def test_tui_app_user_action_bell_can_be_disabled() -> None:
    app = TUIApp(
        config=MockConfig(notifications=NotificationConfig(bell_on_user_action_required=False)),
        config_manager=MockConfigManager(),
    )
    app._app = MagicMock()
    app._tui_running = True

    app._notify_user_action_required()

    app._app.output.write_raw.assert_not_called()
    app._app.output.flush.assert_not_called()


def test_tui_app_completion_bell_is_scoped_to_terminal_ownership() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    app._app = MagicMock()

    app._notify_turn_complete()

    app._app.output.write_raw.assert_not_called()
    app._app.output.flush.assert_not_called()

    app._tui_running = True
    app._shutdown_requested = True
    app._notify_turn_complete()

    app._app.output.write_raw.assert_not_called()
    app._app.output.flush.assert_not_called()

    app._shutdown_requested = False
    app._notify_turn_complete()

    app._app.output.write_raw.assert_called_once_with("\a")
    app._app.output.flush.assert_called_once_with()


def test_tui_app_completion_bell_can_be_disabled() -> None:
    """The opt-out must not emit a terminal bell."""
    app = TUIApp(
        config=MockConfig(notifications=NotificationConfig(bell_on_turn_complete=False)),
        config_manager=MockConfigManager(),
    )
    app._app = MagicMock()
    app._tui_running = True

    app._notify_turn_complete()

    app._app.output.write_raw.assert_not_called()
    app._app.output.flush.assert_not_called()


def test_tui_app_shutdown_status_does_not_write_after_terminal_release(
    capsys: pytest.CaptureFixture[str],
) -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    app._app = MagicMock()

    app._show_shutdown_status("closing durable worker")

    assert app._shutdown_status == "closing durable worker"
    app._app.invalidate.assert_not_called()
    assert capsys.readouterr().err == ""


def test_tui_app_exit_request_closes_notification_ingress_before_terminal_release() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    application = MagicMock()
    monitor = MagicMock()
    app._get_shell_monitor = MagicMock(return_value=monitor)  # type: ignore[method-assign]

    def assert_shutdown_requested() -> None:
        assert app._shutdown_requested is True
        monitor.set_notification_callback.assert_called_once_with(None)

    application.exit.side_effect = assert_shutdown_requested

    app._request_exit(application)

    monitor.set_notification_callback.assert_called_once_with(None)
    application.exit.assert_called_once_with()
    assert app._shutdown_status == "exit requested"


def test_tui_app_shutdown_rejects_stale_shell_notifications() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    monitor = MagicMock()
    notification = ShellNotification(process_id="process-1", kind="completion")
    app._get_shell_monitor = MagicMock(return_value=monitor)  # type: ignore[method-assign]
    app._launch_agent = MagicMock()  # type: ignore[method-assign]
    app._shutdown_requested = True

    app._on_shell_notification(notification)
    app._route_pending_shell_notifications()

    assert app._output_lines == []
    monitor.pending.assert_not_called()
    app._launch_agent.assert_not_called()


@pytest.mark.asyncio
async def test_tui_app_exit_owns_task_cleanup_once() -> None:
    """The context exit, not run(), owns the single task-cleanup pass."""
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    app._show_shutdown_status = MagicMock()  # type: ignore[method-assign]
    app._cancel_agent_task = AsyncMock()  # type: ignore[method-assign]
    app._cancel_managed_tasks = AsyncMock()  # type: ignore[method-assign]

    await app.__aexit__(None, None, None)

    app._cancel_agent_task.assert_awaited_once_with()
    app._cancel_managed_tasks.assert_awaited_once_with()
    status_messages = [call.args[0] for call in app._show_shutdown_status.call_args_list]
    assert status_messages == ["starting shutdown", "shutdown complete"]


@pytest.mark.asyncio
async def test_tui_app_exit_continues_cleanup_after_stage_failure() -> None:
    """A failed task-cleanup stage must not strand later runtime resources."""
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    app._show_shutdown_status = MagicMock()  # type: ignore[method-assign]
    app._cancel_agent_task = AsyncMock(side_effect=ValueError("agent cleanup failed"))  # type: ignore[method-assign]
    app._cancel_managed_tasks = AsyncMock()  # type: ignore[method-assign]
    supervisor = MagicMock()
    supervisor.shutdown = AsyncMock()
    app._oauth_refresh_supervisor = supervisor
    stack = MagicMock()
    stack.__aexit__ = AsyncMock(return_value=None)
    app._exit_stack = stack

    with pytest.raises(ValueError, match="agent cleanup failed"):
        await app.__aexit__(None, None, None)

    app._cancel_managed_tasks.assert_awaited_once_with()
    supervisor.shutdown.assert_awaited_once_with()
    stack.__aexit__.assert_awaited_once_with(None, None, None)
    assert app._exit_stack is None
    assert app._oauth_refresh_supervisor is None
    assert app._show_shutdown_status.call_args_list[-1].args == ("shutdown complete",)


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
    _stub_agent_turn_acceptance(app)
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
    _stub_agent_turn_acceptance(app)

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
    _stub_agent_turn_acceptance(app)

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
    _stub_agent_turn_acceptance(app)
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
    _stub_agent_turn_acceptance(app)
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
async def test_tui_app_hitl_pauses_timer_rings_bell_and_resumes_after_answer() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    app._app = MagicMock()
    app._tui_running = True
    app._app.output.get_size.return_value = MagicMock(columns=120, rows=40)
    app._run_started_at = 100.0
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
                        }
                    ]
                },
                tool_call_id="question-1",
            )
        ]
    )
    now = 110.0

    with patch("yaacli.app.tui.time.monotonic", side_effect=lambda: now):
        app._set_phase(TUIPhase.THINKING)
        task = asyncio.create_task(app._request_user_action(deferred))
        await asyncio.sleep(0)

        assert app.phase == TUIPhase.AWAITING_APPROVAL
        assert app._run_timer_paused_at == 110.0
        assert " · 10s" in "".join(text for _style, text in app._get_status_text())
        app._app.output.write_raw.assert_called_once_with("\a")
        app._app.output.flush.assert_called_once_with()

        now = 210.0
        assert " · 10s" in "".join(text for _style, text in app._get_status_text())
        assert app._approval_event is not None
        app._approval_result = True
        app._approval_reason = "2"
        app._approval_event.set()
        results = await task

        assert app._run_timer_paused_at is None
        assert app._run_started_at == 200.0
        now = 215.0
        assert " · 15s" in "".join(text for _style, text in app._get_status_text())

    assert results.calls["question-1"] == {
        "questions": [
            {
                "question": "Which format?",
                "header": "Format",
                "options": [
                    {"label": "Summary", "description": "Brief"},
                    {"label": "Detailed", "description": "Complete"},
                ],
                "multiSelect": False,
            }
        ],
        "answers": {"Which format?": "Detailed"},
    }


@pytest.mark.asyncio
async def test_tui_app_hitl_cancellation_resumes_timer() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    app._run_started_at = 100.0
    deferred = DeferredToolRequests(approvals=[ToolCallPart(tool_name="edit", args={}, tool_call_id="approval-1")])
    now = 110.0

    with patch("yaacli.app.tui.time.monotonic", side_effect=lambda: now):
        app._set_phase(TUIPhase.THINKING)
        task = asyncio.create_task(app._request_user_action(deferred))
        await asyncio.sleep(0)
        assert app._run_timer_paused_at == 110.0

        now = 210.0
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert app._run_timer_paused_at is None
    assert app._run_started_at == 200.0


@pytest.mark.asyncio
async def test_tui_app_cancel_interrupts_active_structured_question_once() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    service = MagicMock()
    service.cancel = AsyncMock()
    app._session_service = service
    app._active_logical_run_id = "question-run"
    deferred = DeferredToolRequests(
        calls=[
            ToolCallPart(
                tool_name="ask_user_question",
                args={
                    "questions": [
                        {
                            "question": "How should this change be handled?",
                            "header": "Change",
                            "options": [
                                {"label": "Include", "description": "Include it"},
                                {"label": "Exclude", "description": "Leave it out"},
                            ],
                            "multiSelect": False,
                        }
                    ]
                },
                tool_call_id="question-cancel",
            )
        ]
    )
    app._set_phase(TUIPhase.THINKING)
    question_task = asyncio.create_task(app._request_user_action(deferred))
    app._agent_task = question_task
    await asyncio.sleep(0)

    assert app.phase == TUIPhase.AWAITING_APPROVAL
    assert app._approval_event is not None

    app._cancel_foreground()
    app._cancel_foreground()

    assert app.phase == TUIPhase.CANCELLING
    assert question_task.cancelling() == 1
    with pytest.raises(asyncio.CancelledError):
        await question_task
    await asyncio.sleep(0)

    service.cancel.assert_awaited_once_with("question-run", reason="user_interrupted")
    assert app._approval_event is None
    assert sum("Cancelling durable agent run" in line for line in app._output_lines) == 1
    assert any("Cancellation is already in progress" in line for line in app._output_lines)


@pytest.mark.asyncio
async def test_tui_app_empty_deferred_batch_does_not_pause_or_ring() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    app._app = MagicMock()
    app._run_started_at = 100.0

    results = await app._request_user_action(DeferredToolRequests())

    assert results == DeferredToolResults()
    assert app._run_started_at == 100.0
    assert app._run_timer_paused_at is None
    app._app.output.write_raw.assert_not_called()
    app._app.output.flush.assert_not_called()


def test_tui_app_run_timer_excludes_multiple_paused_intervals() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    app._run_started_at = 100.0
    now = 110.0

    with patch("yaacli.app.tui.time.monotonic", side_effect=lambda: now):
        app._pause_run_timer()
        now = 210.0
        app._resume_run_timer()
        now = 220.0
        app._pause_run_timer()
        now = 270.0
        app._resume_run_timer()

    assert app._run_started_at == 250.0
    assert app._run_timer_paused_at is None


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

    async def fake_wait_for_input(*, timeout_seconds: float | None = None) -> tuple[bool, str | None]:
        assert timeout_seconds == 120.0
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
async def test_tui_app_structured_question_timeout_rejects_call_and_continues() -> None:
    app = TUIApp(
        config=MockConfig(tools=ToolsConfig(user_input_timeout_seconds=0.01)),
        config_manager=MockConfigManager(),
    )
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
                        }
                    ]
                },
                tool_call_id="question-timeout",
            )
        ]
    )
    app._set_phase(TUIPhase.THINKING)

    results = await app._request_user_action(deferred)

    call_result = results.calls["question-timeout"]
    assert isinstance(call_result, RetryPromptPart)
    assert call_result.content == (
        "The user did not respond before the clarification request timed out. "
        "Do not wait for or request the same input again. Continue the task using your best judgment "
        "and make reasonable assumptions where needed."
    )
    assert call_result.content == USER_INPUT_TIMEOUT_PROMPT
    assert call_result.tool_name == "ask_user_question"
    assert call_result.tool_call_id == "question-timeout"
    assert app._approval_event is None
    assert app.phase == TUIPhase.TOOL_CALLING
    assert "Timed out: ask_user_question" in "\n".join(app._output_lines)


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
    assert app._output_lines == []


def test_tui_app_steering_acceptance_failure_retains_editor() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    input_area = TextArea(text="keep this guidance", multiline=True)
    service = MagicMock()
    service.accept_input.side_effect = RuntimeError("database unavailable")
    app._session_service = service
    app._active_logical_run_id = "active-run"
    app._set_phase(TUIPhase.THINKING)

    app._submit_input("keep this guidance", input_area)

    assert input_area.buffer.text == "keep this guidance"
    service.accept_input.assert_called_once()
    assert not any("Guidance sent" in line for line in app._output_lines)


@pytest.mark.asyncio
async def test_tui_app_turn_acceptance_failure_retains_prompt_and_attachments() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    input_area = TextArea(text="keep this prompt", multiline=True)
    attachment = PendingAttachment(data=b"img", media_type="image/png", size_bytes=3)
    app._pending_attachments.append(attachment)
    app._accept_agent_turn = MagicMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("database unavailable")
    )

    with patch.object(app, "_append_user_input") as append_user_input:
        app._submit_input("keep this prompt", input_area)

    assert input_area.buffer.text == "keep this prompt"
    assert app._pending_attachments == [attachment]
    append_user_input.assert_not_called()


@pytest.mark.asyncio
async def test_tui_app_slash_prompt_acceptance_failure_retains_draft_and_attachments() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    attachment = PendingAttachment(data=b"img", media_type="image/png", size_bytes=3)
    app._pending_attachments = [attachment]
    input_area = TextArea(text="/not-a-command keep this", multiline=True)
    app._launch_agent = MagicMock(return_value=False)  # type: ignore[method-assign]

    app._submit_input(input_area.buffer.text, input_area)
    task = app._foreground_command_task
    assert task is not None
    await task

    assert input_area.buffer.text == "/not-a-command keep this"
    assert app._pending_attachments == [attachment]
    assert app._prompt_history == []


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/goal keep this", "/commit keep this"])
async def test_tui_app_agent_command_acceptance_failure_retains_draft(command: str) -> None:
    config = MockConfig(commands={"commit": CommandDefinition(prompt="commit changes")})
    app = TUIApp(config=config, config_manager=MockConfigManager())
    app._runtime = MagicMock()
    app._runtime.ctx = TUIContext()
    attachment = PendingAttachment(data=b"img", media_type="image/png", size_bytes=3)
    app._pending_attachments = [attachment]
    input_area = TextArea(text=command, multiline=True)
    app._launch_agent = MagicMock(return_value=False)  # type: ignore[method-assign]

    app._submit_input(command, input_area)
    task = app._foreground_command_task
    assert task is not None
    await task

    assert input_area.buffer.text == command
    assert app._pending_attachments == [attachment]
    assert app._prompt_history == []
    assert app.runtime.ctx.goal_active is False


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


def test_tui_app_terminal_last_write_preserves_rejected_steering_draft() -> None:
    """A terminal write wins without raising or locking the session."""
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    input_area = TextArea(text="late guidance", multiline=True)
    service = MagicMock()
    service.accept_input.return_value = MagicMock(state=InputState.rejected)
    app._session_service = service
    app._active_logical_run_id = "run-1"
    app._set_phase(TUIPhase.THINKING)
    app._set_phase(TUIPhase.STREAMING_OUTPUT)

    app._submit_input(input_area.text, input_area)

    assert app.phase is TUIPhase.STREAMING_OUTPUT
    assert input_area.buffer.text == "late guidance"
    assert any("active run already finished" in line.lower() for line in app._output_lines)
    assert not any("[ERROR]" in line for line in app._output_lines)


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
    assert sum("durable commit is in progress" in line for line in app._output_lines) == 2
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
    _stub_agent_turn_acceptance(app)
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
    _stub_agent_turn_acceptance(app)
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
    _stub_agent_turn_acceptance(app)
    service = MagicMock()
    service.cancel = AsyncMock()
    app._session_service = service
    app._launch_agent("hello")
    assert app._agent_task is not None

    app._agent_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await app._agent_task
    await asyncio.sleep(0)

    assert app._run_started_at is None
    assert app._active_logical_run_id is None
    service.cancel.assert_awaited_once_with(
        "accepted-test-run",
        reason="agent_task_cancelled_before_start",
    )
    assert app.phase == TUIPhase.IDLE


def test_tui_app_status_places_sdk_cost_immediately_after_context() -> None:
    config = YaacliConfig(display=DisplayConfig(show_token_usage=True))
    app = TUIApp(config=config, config_manager=MockConfigManager())
    app._get_terminal_width = lambda: 120  # type: ignore[method-assign]
    app._current_context_tokens = 50_000
    app._context_window_size = 200_000
    app._session_usage.add(
        "main",
        "model-a",
        RunUsage(requests=1, input_tokens=10, output_tokens=2),
        cost_estimate=CostEstimate(total_amount=Decimal("0.007"), priced_requests=1),
    )

    status = "".join(text for _style, text in app._get_status_text())

    assert " · ctx 25% · cost ~$0.0070" in status


async def test_parent_and_child_hitl_requests_are_serialized_by_owner() -> None:
    app = TUIApp(config=MockConfig(), config_manager=MockConfigManager())
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    observed: list[str] = []

    async def collect(deferred: DeferredToolRequests) -> DeferredToolResults:
        tool_name = deferred.calls[0].tool_name
        observed.append(tool_name)
        if tool_name == "child_call":
            first_started.set()
            await release_first.wait()
        return DeferredToolResults()

    app._collect_deferred_user_actions = collect  # type: ignore[method-assign]
    child = DeferredToolRequests(
        calls=[
            ToolCallPart(
                tool_name="child_call",
                args={},
                tool_call_id="child-call",
            )
        ]
    )
    parent = DeferredToolRequests(
        calls=[
            ToolCallPart(
                tool_name="parent_call",
                args={},
                tool_call_id="parent-call",
            )
        ]
    )

    child_task = asyncio.create_task(app._request_user_action(child, owner="subagent:child"))
    await asyncio.wait_for(first_started.wait(), timeout=1)
    parent_task = asyncio.create_task(app._request_user_action(parent, owner="main"))
    await asyncio.sleep(0)

    app._reset_hitl_state(owner="main")
    assert app._hitl_owner == "subagent:child"
    assert observed == ["child_call"]

    release_first.set()
    await child_task
    await parent_task

    assert observed == ["child_call", "parent_call"]
    assert app._hitl_owner is None
