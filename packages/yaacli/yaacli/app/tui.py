"""TUI Application for yaacli.

This module provides the main TUI application with:
- prompt_toolkit based UI with dual-pane layout
- Agent execution with streaming output
- Steering message injection during execution
- Mode switching (ACT/PLAN) via /act and /plan slash commands
- Scrollable output with keyboard and mouse support
- Input mode switching (send/edit) with Tab key
- Double Ctrl+C exit confirmation

Example:
    from yaacli.app import TUIApp

    async with TUIApp(config, config_manager) as app:
        await app.run()

"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import signal
import sys
import time
import traceback
import uuid
from collections.abc import Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, cast

from prompt_toolkit import Application
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea
from pydantic_ai import BinaryContent, DeferredToolRequests, DeferredToolResults, ToolDenied, UsageLimits, UserContent
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessagesTypeAdapter,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
)
from pydantic_ai.models import Model
from rich.table import Table
from rich.text import Text
from ya_agent_sdk.agents.main import AgentRuntime, stream_agent
from ya_agent_sdk.context import PROJECT_GUIDANCE_TAG, USER_RULES_TAG, BusMessage, ResumableState, StreamEvent
from ya_agent_sdk.events import (
    CompactCompleteEvent,
    CompactFailedEvent,
    CompactStartEvent,
    FileChangeEvent,
    HandoffCompleteEvent,
    HandoffFailedEvent,
    HandoffStartEvent,
    MessageReceivedEvent,
    ModelRequestStartEvent,
    NoteEvent,
    SubagentCompleteEvent,
    SubagentStartEvent,
    TaskEvent,
    ToolCallsStartEvent,
    UsageSnapshotEvent,
)
from ya_agent_sdk.utils import get_latest_request_usage

# Import state management from app.state (re-export TUIMode, TUIState for backward compatibility)
from yaacli.app.state import TUIMode
from yaacli.background import BACKGROUND_MONITOR_KEY, BackgroundMonitor, BackgroundTaskInfo
from yaacli.browser import BrowserManager
from yaacli.clipboard import ClipboardImageReadResult, read_clipboard_image
from yaacli.config import ConfigManager, YaacliConfig
from yaacli.display import EventRenderer, RichRenderer, ToolMessage
from yaacli.environment import TUIEnvironment
from yaacli.events import ContextUpdateEvent, LoopCompleteEvent, LoopCompleteReason, LoopIterationEvent
from yaacli.hooks import emit_context_update
from yaacli.logging import configure_tui_logging, get_logger
from yaacli.perf import perf_log_report, perf_report, perf_timer
from yaacli.runtime import create_tui_runtime
from yaacli.session import TUIContext
from yaacli.usage import SessionUsage

if TYPE_CHECKING:
    from prompt_toolkit.key_binding import KeyPressEvent
    from y_agent_environment import BackgroundProcess

logger = get_logger(__name__)

_SHUTDOWN_AGENT_TASK_TIMEOUT = 8.0
_SHUTDOWN_MANAGED_TASKS_TIMEOUT = 5.0
_DIRECT_SHELL_TERMINATE_TIMEOUT = 2.0


# =============================================================================
# Utilities
# =============================================================================


def _safe_exception_str(e: BaseException) -> str:
    """Safely convert an exception to a string.

    Some exceptions (e.g., pydantic-ai's ModelAPIError with message=None)
    have __str__ that returns None instead of a string, causing str() to
    raise TypeError. This helper falls back to repr() in such cases.

    Args:
        e: The exception to convert.

    Returns:
        A string representation of the exception.
    """
    try:
        result = str(e)
    except Exception:
        result = repr(e)

    # Guard against __str__ returning "None" for exceptions created with None arg
    # e.g., Exception(None), RuntimeError(None), anthropic.APIConnectionError(message=None)
    if not result or result == "None":
        result = repr(e)

    return result


def _is_benign_contextvar_cleanup_error(e: BaseException | None) -> bool:
    """Check if an exception matches pydantic-ai's known ContextVar cleanup race."""
    if not isinstance(e, ValueError):
        return False

    message = _safe_exception_str(e)
    return "was created in a different Context" in message and "ContextVar" in message


def _get_elapsed_seconds(started_at: datetime) -> float:
    """Calculate elapsed seconds for naive or aware timestamps."""
    if started_at.tzinfo is None:
        return (datetime.now() - started_at).total_seconds()

    return (datetime.now(UTC) - started_at.astimezone(UTC)).total_seconds()


# =============================================================================
# Constants
# =============================================================================

STEERING_TEMPLATE = """<steering>
{{ content }}
</steering>

<system-reminder>
The user has provided additional guidance during task execution.
Review the <steering> content carefully, consider how it affects your current approach,
and adjust your work accordingly while continuing toward the goal.
</system-reminder>"""


# TUIState kept for backward compatibility (used in tests and status bar)
class TUIState(StrEnum):
    """TUI application state (legacy, use TUIStateMachine for new code)."""

    IDLE = "idle"
    RUNNING = "running"


@dataclass(frozen=True)
class PendingAttachment:
    """Pending binary attachment queued from clipboard paste."""

    data: bytes
    media_type: str
    size_bytes: int


# =============================================================================
# TUI Application
# =============================================================================


@dataclass
class TUIApp:
    """Main TUI application class.

    Manages the lifecycle of:
    - BrowserManager (optional)
    - AgentRuntime (env + ctx + agent)
    - prompt_toolkit Application

    Usage:
        async with TUIApp(config, config_manager) as app:
            await app.run()
    """

    config: YaacliConfig
    config_manager: ConfigManager
    verbose: bool = False
    working_dir: Path = field(default_factory=Path.cwd)

    # Runtime state
    _mode: TUIMode = field(default=TUIMode.ACT, init=False)
    _state: TUIState = field(default=TUIState.IDLE, init=False)
    _agent_phase: str = field(default="idle", init=False)  # "idle", "thinking", "tools"

    # Resources (initialized in __aenter__)
    _exit_stack: AsyncExitStack | None = field(default=None, init=False, repr=False)
    _browser: BrowserManager | None = field(default=None, init=False)
    _runtime: AgentRuntime[TUIContext, str | DeferredToolRequests, TUIEnvironment] | None = field(
        default=None, init=False
    )

    # UI components
    _app: Application[None] | None = field(default=None, init=False, repr=False)
    _output_lines: list[str] = field(default_factory=list, init=False)
    _max_output_lines: int = field(default=500, init=False)  # Overridden from config.display

    # Virtual viewport rendering (only parse ANSI for visible lines)
    _scroll_offset: int = field(default=0, init=False)  # Display line offset from top
    _block_line_counts: list[int] = field(default_factory=list, init=False)  # Line count per output block
    _total_line_count: int = field(default=0, init=False)  # Sum of all block line counts
    _output_generation: int = field(default=0, init=False)  # Bumped on any content change
    _viewport_cache_key: tuple[int, int, int] | None = field(default=None, init=False)
    _output_ansi_cache: ANSI | None = field(default=None, init=False)  # Cached visible ANSI
    _renderer: RichRenderer = field(default_factory=RichRenderer, init=False)
    _event_renderer: EventRenderer = field(default_factory=EventRenderer, init=False)

    # Session
    _session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12], init=False)

    # Agent execution
    _agent_task: asyncio.Task[None] | None = field(default=None, init=False)
    _managed_tasks: set[asyncio.Task[Any]] = field(default_factory=set, init=False, repr=False)
    _last_run: Any | None = field(default=None, init=False)  # AgentRun from last execution
    _message_history: list[Any] | None = field(default=None, init=False)  # Conversation history

    # Tool tracking
    _tool_messages: dict[str, ToolMessage] = field(default_factory=dict, init=False)
    _printed_tool_calls: set[str] = field(default_factory=set, init=False)

    # Subagent state tracking: agent_id -> {"line_index": int, "tool_names": list[str]}
    _subagent_states: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)

    # Steering pane
    _steering_items: list[tuple[str, str, str]] = field(default_factory=list, init=False)
    _max_steering_lines: int = field(default=5, init=False)

    # Input mode: "send" (Enter sends) or "edit" (Enter inserts newline)
    _input_mode: str = field(default="send", init=False)

    # Mouse support mode
    _mouse_enabled: bool = field(default=True, init=False)

    # Double Ctrl+C exit
    _last_ctrl_c_time: float = field(default=0.0, init=False)
    _ctrl_c_exit_timeout: float = field(default=2.0, init=False)

    # Prompt history for up/down navigation
    _prompt_history: list[str] = field(default_factory=list, init=False)
    _history_index: int = field(default=-1, init=False)
    _current_input_backup: str = field(default="", init=False)
    _pending_attachments: list[PendingAttachment] = field(default_factory=list, init=False)

    # Shutdown visibility
    _shutdown_status: str | None = field(default=None, init=False)
    _tui_running: bool = field(default=False, init=False)

    # Streaming text tracking for markdown rendering
    _streaming_text: str = field(default="", init=False)
    _streaming_line_index: int | None = field(default=None, init=False)

    # Streaming thinking tracking for extended thinking display
    _streaming_thinking: str = field(default="", init=False)
    _streaming_thinking_line_index: int | None = field(default=None, init=False)

    # Real-time context usage tracking
    _current_context_tokens: int = field(default=0, init=False)
    _context_window_size: int = field(default=200000, init=False)

    # Session-level usage tracking
    _session_usage: SessionUsage = field(default_factory=SessionUsage, init=False)

    # UI refresh throttling
    _last_invalidate_time: float = field(default=0.0, init=False)
    _invalidate_interval: float = field(default=0.016, init=False)  # ~60fps max

    # Streaming render throttle (separate from UI invalidation)
    _last_stream_render_time: float = field(default=0.0, init=False)
    _stream_render_interval: float = field(default=0.08, init=False)  # ~12fps for markdown re-render

    # HITL (Human-in-the-Loop) approval state
    _hitl_pending: bool = field(default=False, init=False)
    _approval_event: asyncio.Event | None = field(default=None, init=False)
    _approval_result: bool | None = field(default=None, init=False)  # True=approve, False=reject
    _approval_reason: str | None = field(default=None, init=False)
    _pending_approvals: list[ToolCallPart] = field(default_factory=list, init=False)
    _current_approval_index: int = field(default=0, init=False)

    # Background task completion tracking
    _pending_bus_check_needed: bool = field(default=False, init=False)

    # Deferred screen recovery scheduling
    _screen_recovery_scheduled: bool = field(default=False, init=False)

    @property
    def mode(self) -> TUIMode:
        """Current agent mode."""
        return self._mode

    @property
    def state(self) -> TUIState:
        """Current application state."""
        return self._state

    @property
    def runtime(self) -> AgentRuntime[TUIContext, str | DeferredToolRequests, TUIEnvironment]:
        """Get agent runtime (must be entered first)."""
        if self._runtime is None:
            raise RuntimeError("TUIApp not entered. Use 'async with app:' first.")
        return self._runtime

    def _track_managed_task(self, task: asyncio.Task[Any]) -> asyncio.Task[Any]:
        """Track a fire-and-forget task so it can be cancelled on shutdown."""
        self._managed_tasks.add(task)
        task.add_done_callback(self._on_managed_task_done)
        return task

    def _on_managed_task_done(self, task: asyncio.Future[Any]) -> None:
        """Release a managed task and consume exceptions from fire-and-forget work."""
        self._managed_tasks.discard(cast(asyncio.Task[Any], task))
        self._log_managed_task_exception(task, during_shutdown=False)

    def _log_managed_task_exception(self, task: asyncio.Future[Any], *, during_shutdown: bool) -> None:
        """Log managed task exceptions after the task has completed."""
        if task.cancelled():
            return

        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return

        if exc is None:
            return
        if _is_benign_contextvar_cleanup_error(exc):
            logger.debug(
                "Suppressed ContextVar cleanup error during managed task%s: %s",
                " shutdown" if during_shutdown else "",
                _safe_exception_str(exc),
            )
            return

        logger.debug(
            "Managed task ended with exception%s: %s",
            " during shutdown" if during_shutdown else "",
            _safe_exception_str(exc),
        )

    def _show_shutdown_status(self, message: str) -> None:
        """Show shutdown progress in the TUI, with stderr fallback after it exits."""
        self._shutdown_status = message
        logger.info("Shutdown: %s", message)

        if self._app is not None and self._tui_running:
            self._app.invalidate()
            return

        print(f"Shutdown: {message}", file=sys.stderr, flush=True)

    def _on_agent_task_shutdown_done(self, task: asyncio.Future[None]) -> None:
        """Release a timed-out agent task after cancellation eventually completes."""
        if self._agent_task is task:
            self._agent_task = None

    async def _cancel_agent_task(self) -> None:
        """Cancel and await the current agent task with a bounded deadline."""
        task = self._agent_task
        if task is None:
            return

        try:
            if not task.done():
                self._show_shutdown_status("cancelling agent task")
                task.cancel()
                done, pending = await asyncio.wait({task}, timeout=_SHUTDOWN_AGENT_TASK_TIMEOUT)
                if pending:
                    logger.warning(
                        "Agent task did not finish within %.1fs during shutdown",
                        _SHUTDOWN_AGENT_TASK_TIMEOUT,
                    )
                    self._show_shutdown_status("agent task cleanup timed out; continuing shutdown")
                    task.add_done_callback(self._on_agent_task_shutdown_done)
                for completed in done:
                    if not completed.cancelled():
                        exc = completed.exception()
                        if exc is not None:
                            raise exc
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if _is_benign_contextvar_cleanup_error(e):
                logger.debug("Suppressed ContextVar cleanup error during agent shutdown: %s", _safe_exception_str(e))
            else:
                raise
        finally:
            if self._agent_task is task and task.done():
                self._agent_task = None

    async def _cancel_managed_tasks(self) -> None:
        """Cancel and await fire-and-forget UI tasks created by the TUI."""
        tasks = [task for task in self._managed_tasks if not task.done()]
        if not tasks:
            self._managed_tasks.clear()
            return

        self._show_shutdown_status(f"cancelling {len(tasks)} UI task(s)")
        for task in tasks:
            task.cancel()

        done, pending = await asyncio.wait(tasks, timeout=_SHUTDOWN_MANAGED_TASKS_TIMEOUT)
        if pending:
            logger.warning(
                "%d managed task(s) did not finish within %.1fs during shutdown",
                len(pending),
                _SHUTDOWN_MANAGED_TASKS_TIMEOUT,
            )
            self._show_shutdown_status("UI task cleanup timed out; continuing shutdown")

        self._managed_tasks.difference_update(done)

    def _recover_tui_screen(self) -> None:
        """Force-clear and fully redraw the TUI after terminal corruption."""
        self._screen_recovery_scheduled = False

        if not self._app:
            return

        try:
            self._app.renderer.clear()
        except Exception:
            logger.debug("Failed to clear TUI renderer during recovery", exc_info=True)

        try:
            self._app.reset()
        except Exception:
            logger.debug("Failed to reset TUI application during recovery", exc_info=True)

        try:
            self._app._redraw()
        except Exception:
            logger.debug("Failed to redraw TUI during recovery", exc_info=True)

        try:
            self._app.invalidate()
        except Exception:
            logger.debug("Failed to invalidate TUI during recovery", exc_info=True)

    def _schedule_tui_recovery(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Schedule screen recovery on the next event-loop tick."""
        if self._screen_recovery_scheduled:
            return

        if loop is None:
            loop = asyncio.get_running_loop()

        self._screen_recovery_scheduled = True
        loop.call_soon(self._recover_tui_screen)

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def __aenter__(self) -> TUIApp:
        """Initialize resources."""
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        # Configure TUI logging (queue for internal use, file for verbose mode)
        log_queue: asyncio.Queue[object] = asyncio.Queue()

        # Start browser manager (optional)
        self._browser = BrowserManager(self.config.browser)
        await self._exit_stack.enter_async_context(self._browser)

        # Load MCP config
        mcp_config = self.config_manager.load_mcp_config()

        # Create runtime
        self._runtime = create_tui_runtime(
            config=self.config,
            mcp_config=mcp_config,
            browser_manager=self._browser,
            working_dir=self.working_dir,
            config_dir=self.config_manager.config_dir,
        )
        await self._exit_stack.enter_async_context(self._runtime)

        # Register application-level injected context tags for compact stripping
        self._runtime.ctx.injected_context_tags = (
            *self._runtime.ctx.injected_context_tags,
            PROJECT_GUIDANCE_TAG,
            USER_RULES_TAG,
        )

        # Initialize context window size from model config
        if self._runtime.ctx.model_cfg.context_window:
            self._context_window_size = self._runtime.ctx.model_cfg.context_window

        # Apply display config
        self._max_output_lines = self.config.display.max_output_lines

        logger.info("TUIApp initialized")
        configure_tui_logging(log_queue, verbose=self.verbose)

        # Set core_toolset on BackgroundMonitor so it can find the delegate tool
        bg_monitor = self._get_background_monitor()
        if bg_monitor and self._runtime:
            bg_monitor.set_core_toolset(self._runtime.core_toolset)
            bg_monitor.set_completion_callback(self._on_background_task_complete)

            # Start shell process completion monitoring
            if self._runtime.env.shell is not None:
                bg_monitor.start_shell_monitor(
                    shell=self._runtime.env.shell,
                    bus=self._runtime.ctx.message_bus,
                    agent_id=self._runtime.ctx.agent_id,
                )

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        """Cleanup resources."""
        # Clear completion callback
        self._show_shutdown_status("starting shutdown")
        bg_monitor = self._get_background_monitor()
        if bg_monitor:
            bg_monitor.set_completion_callback(None)

        # Cancel any running agent task and tracked fire-and-forget tasks
        await self._cancel_agent_task()
        await self._cancel_managed_tasks()

        # Give event loop a chance to process pending cleanups
        await asyncio.sleep(0)

        if self._exit_stack:
            try:
                self._show_shutdown_status("closing runtime resources")
                result = await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)
                self._exit_stack = None
                self._show_shutdown_status("shutdown complete")
                return result
            except (RuntimeError, GeneratorExit, BaseExceptionGroup) as e:
                # Suppress MCP stdio client cleanup errors
                # These occur due to async generator lifecycle issues in pydantic-ai/mcp
                logger.debug("Suppressed cleanup error: %s", e)
                self._exit_stack = None
                return None
        self._show_shutdown_status("shutdown complete")
        return None

    # =========================================================================
    # Mode Management
    # =========================================================================

    def switch_mode(self, mode: TUIMode) -> None:
        """Switch agent operating mode."""
        if self._state == TUIState.RUNNING:
            self._append_output("[Cannot switch mode while agent is running]")
            return

        if self._mode != mode:
            old_mode = self._mode
            self._mode = mode
            self._append_output(f"[Mode switched: {old_mode.value} -> {mode.value}]")
            if self._app:
                self._app.invalidate()

    # =========================================================================
    # Output Management
    # =========================================================================

    def _throttled_invalidate(self) -> None:
        """Invalidate UI with throttling to prevent excessive redraws."""
        if not self._app:
            return
        now = time.time()
        if now - self._last_invalidate_time >= self._invalidate_interval:
            self._last_invalidate_time = now
            self._app.invalidate()

    def _invalidate_output_cache(self) -> None:
        """Mark output cache as invalid (must be called after modifying _output_lines).

        O(1) operation - just bumps the generation counter.
        Line counts are maintained incrementally by _append_block and _update_block.
        """
        self._output_generation += 1

    def _append_block(self, content: str) -> None:
        """Append an output block with incremental bookkeeping.

        Handles line count tracking and generation bump.
        All code that appends to _output_lines should use this method.
        """
        line_count = content.count("\n") + 1
        self._output_lines.append(content)
        self._block_line_counts.append(line_count)
        self._total_line_count += line_count
        self._output_generation += 1

    def _update_block(self, idx: int, new_content: str) -> None:
        """Update an output block in-place with incremental line count tracking.

        This is O(1) for line count update (just delta between old and new).
        Used by streaming text/thinking updates.
        """
        if idx >= len(self._output_lines):
            return
        old_count = self._block_line_counts[idx]
        new_count = new_content.count("\n") + 1
        self._output_lines[idx] = new_content
        self._block_line_counts[idx] = new_count
        self._total_line_count += new_count - old_count
        self._output_generation += 1

    def _append_output(self, text: str) -> None:
        """Append text to output buffer with auto-scroll when running."""
        self._append_block(text)

        # Trim old lines to prevent memory issues
        if len(self._output_lines) > self._max_output_lines:
            trim_count = len(self._output_lines) - self._max_output_lines
            # Subtract line counts of removed blocks
            for i in range(trim_count):
                self._total_line_count -= self._block_line_counts[i]
            self._output_lines = self._output_lines[trim_count:]
            self._block_line_counts = self._block_line_counts[trim_count:]
            # Adjust streaming indices to account for removed blocks
            if self._streaming_line_index is not None:
                self._streaming_line_index -= trim_count
                if self._streaming_line_index < 0:
                    # Streaming block was trimmed away - reset
                    self._streaming_text = ""
                    self._streaming_line_index = None
            if self._streaming_thinking_line_index is not None:
                self._streaming_thinking_line_index -= trim_count
                if self._streaming_thinking_line_index < 0:
                    # Thinking block was trimmed away - reset
                    self._streaming_thinking = ""
                    self._streaming_thinking_line_index = None

        # Auto-scroll to bottom when agent is running
        if self._state == TUIState.RUNNING:
            self._scroll_to_bottom()
        # Invalidate app to refresh display (throttled during streaming)
        self._throttled_invalidate()

    def _get_viewport_height(self) -> int:
        """Get visible output height in lines."""
        if self._app and self._app.output:
            terminal_size = self._app.output.get_size()
            # Reserve: status bar (2) + steering (dynamic) + input area (5) + margins
            return max(5, terminal_size.rows - 9)
        return 40

    def _scroll_to_bottom(self) -> None:
        """Scroll output to bottom.

        Uses cached line count for performance - O(1) operation.
        """
        visible_height = self._get_viewport_height()
        bottom_padding = 4
        if self._total_line_count > visible_height:
            self._scroll_offset = self._total_line_count - visible_height + bottom_padding
        else:
            self._scroll_offset = 0

    def _get_output_text(self) -> ANSI:
        """Get formatted output for display using virtual viewport.

        Only joins and parses ANSI for the visible portion of output,
        making this O(viewport) instead of O(total_content).
        Critical for performance since prompt_toolkit calls this on every redraw.
        """
        with perf_timer("get_output_text"):
            if not self._output_lines:
                return ANSI("")

            vh = self._get_viewport_height()
            cache_key = (self._scroll_offset, vh, self._output_generation)
            if cache_key == self._viewport_cache_key and self._output_ansi_cache is not None:
                return self._output_ansi_cache

            visible = self._get_visible_text(self._scroll_offset, self._scroll_offset + vh)
            self._output_ansi_cache = ANSI(visible)
            self._viewport_cache_key = cache_key
            return self._output_ansi_cache

    def _get_visible_text(self, start_line: int, end_line: int) -> str:
        """Extract only the text visible in the given line range.

        Scans output blocks using pre-computed line counts to find
        the overlapping blocks, then slices them to the exact visible range.
        O(total_blocks) scan + O(visible_blocks) string operations.
        The scan is cheap (integer additions on at most _max_output_lines blocks).
        """
        if not self._output_lines:
            return ""

        cum = 0
        parts: list[str] = []

        for i in range(len(self._output_lines)):
            block_count = self._block_line_counts[i]
            block_end = cum + block_count

            if block_end <= start_line:
                cum = block_end
                continue
            if cum >= end_line:
                break

            # This block overlaps with visible range
            block_lines = self._output_lines[i].split("\n")
            local_start = max(0, start_line - cum)
            local_end = min(block_count, end_line - cum)

            if local_start > 0 or local_end < block_count:
                block_lines = block_lines[local_start:local_end]

            parts.append("\n".join(block_lines))
            cum = block_end

        return "\n".join(parts)

    def _get_max_scroll(self) -> int:
        """Calculate maximum scroll position. O(1) using cached line count."""
        return max(0, self._total_line_count - self._get_viewport_height())

    def _get_terminal_width(self) -> int:
        """Get current terminal width for Rich rendering."""
        if self._app and self._app.output:
            return self._app.output.get_size().columns
        return 120

    def _get_code_theme(self) -> str:
        """Get the code theme for markdown/syntax highlighting."""
        # TODO: Make configurable via config
        return "monokai"

    def _start_streaming_text(self, initial_content: str = "") -> None:
        """Start tracking a new streaming text block."""
        self._streaming_text = initial_content
        self._streaming_line_index = len(self._output_lines)
        self._last_stream_render_time = 0.0  # Reset throttle for new block
        if initial_content:
            # Add placeholder that will be updated.
            # Empty text blocks should not create a visible blank line.
            self._append_block(initial_content)

    def _update_streaming_text(self, delta: str) -> None:
        """Update the current streaming text block with delta.

        Throttles markdown re-rendering to ~12fps to avoid expensive Rich
        markdown parsing on every single token delta.
        """
        self._streaming_text += delta

        # Throttle: skip expensive markdown re-render if too soon
        now = time.time()
        if now - self._last_stream_render_time < self._stream_render_interval:
            return  # Delta buffered in _streaming_text, will render on next interval or finalize
        self._last_stream_render_time = now

        # Re-render markdown for the complete text so far
        if self._streaming_line_index is not None:
            with perf_timer("stream_render_markdown"):
                rendered = self._renderer.render_markdown(
                    self._streaming_text,
                    code_theme=self._get_code_theme(),
                    width=self._get_terminal_width(),
                ).rstrip("\n")
            if self._streaming_line_index < len(self._output_lines):
                self._update_block(self._streaming_line_index, rendered)
            elif rendered:
                self._append_block(rendered)
            if self._state == TUIState.RUNNING:
                self._scroll_to_bottom()
            self._throttled_invalidate()

    def _finalize_streaming_text(self) -> None:
        """Finalize the current streaming text block."""
        if self._streaming_text and self._streaming_line_index is not None:
            # Final render with complete text and dynamic width
            rendered = self._renderer.render_markdown(
                self._streaming_text,
                code_theme=self._get_code_theme(),
                width=self._get_terminal_width(),
            ).rstrip("\n")
            if self._streaming_line_index < len(self._output_lines):
                self._update_block(self._streaming_line_index, rendered)
            elif rendered:
                self._append_block(rendered)
        self._streaming_text = ""
        self._streaming_line_index = None

    def _start_streaming_thinking(self, initial_content: str = "") -> None:
        """Start tracking a new streaming thinking block."""
        self._streaming_thinking = initial_content
        self._streaming_thinking_line_index = len(self._output_lines)
        self._last_stream_render_time = 0.0  # Reset throttle for new block
        # Render initial content with thinking style
        rendered = self._event_renderer.render_thinking(initial_content, width=self._get_terminal_width()).rstrip("\n")
        self._append_block(rendered)
        self._throttled_invalidate()

    def _update_streaming_thinking(self, delta: str) -> None:
        """Update current streaming thinking with delta.

        Throttled similarly to streaming text to avoid excessive re-rendering.
        """
        self._streaming_thinking += delta

        # Throttle: skip expensive re-render if too soon
        now = time.time()
        if now - self._last_stream_render_time < self._stream_render_interval:
            return  # Delta buffered, will render on next interval or finalize
        self._last_stream_render_time = now

        # Re-render thinking for the complete text so far
        if self._streaming_thinking_line_index is not None and self._streaming_thinking_line_index < len(
            self._output_lines
        ):
            rendered = self._event_renderer.render_thinking(
                self._streaming_thinking,
                width=self._get_terminal_width(),
            ).rstrip("\n")
            self._update_block(self._streaming_thinking_line_index, rendered)
            if self._state == TUIState.RUNNING:
                self._scroll_to_bottom()
            self._throttled_invalidate()

    def _finalize_streaming_thinking(self) -> None:
        """Finalize the current streaming thinking block."""
        if self._streaming_thinking and self._streaming_thinking_line_index is not None:
            # Final render
            rendered = self._event_renderer.render_thinking(
                self._streaming_thinking,
                width=self._get_terminal_width(),
            ).rstrip("\n")
            if self._streaming_thinking_line_index < len(self._output_lines):
                self._update_block(self._streaming_thinking_line_index, rendered)
        self._streaming_thinking = ""
        self._streaming_thinking_line_index = None

    def _format_size_bytes(self, size_bytes: int) -> str:
        """Format byte size for compact UI display."""
        if size_bytes < 1024:
            return f"{size_bytes}B"
        if size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.0f}KB"
        return f"{size_bytes / (1024 * 1024):.1f}MB"

    def _format_attachment_description(self, attachment: PendingAttachment) -> str:
        """Format a single attachment description."""
        return f"{attachment.media_type} {self._format_size_bytes(attachment.size_bytes)}"

    def _format_pending_attachments_label(self) -> str:
        """Format pending attachment status label."""
        if not self._pending_attachments:
            return ""

        count = len(self._pending_attachments)
        total_size = sum(item.size_bytes for item in self._pending_attachments)
        noun = "image" if count == 1 else "images"
        return f"Attach: {count} {noun} ({self._format_size_bytes(total_size)})"

    def _consume_pending_attachments(self) -> list[PendingAttachment]:
        """Take the queued clipboard attachments for the next agent turn."""
        attachments = list(self._pending_attachments)
        self._pending_attachments.clear()
        return attachments

    def _reset_pending_attachments(self) -> None:
        """Clear queued clipboard attachments and related compose state."""
        self._pending_attachments.clear()
        self._history_index = -1
        self._current_input_backup = ""

    def _append_user_input(self, text: str, attachments: Sequence[PendingAttachment] | None = None) -> None:
        """Render user input with styled prompt indicator and attachment markers."""
        width = self._get_terminal_width()
        from rich.text import Text as RichText

        attachment_list = list(attachments or [])
        display_text = text
        if not display_text and attachment_list:
            count = len(attachment_list)
            noun = "image" if count == 1 else "images"
            display_text = f"[Attached {count} {noun}]"

        user_text = RichText()
        user_text.append("> ", style="bold green")
        user_text.append(display_text)
        for attachment in attachment_list:
            user_text.append("\n")
            user_text.append("  [Attached] ", style="dim cyan")
            user_text.append(self._format_attachment_description(attachment), style="dim cyan")
        rendered = self._renderer.render(user_text, width=width).rstrip("\n")
        self._append_output(rendered)

    def _append_error_output(self, e: BaseException) -> None:
        """Render error message with traceback to fit terminal width."""
        width = self._get_terminal_width()
        from rich.text import Text as RichText

        error_type = type(e).__name__
        error_msg = _safe_exception_str(e)

        self._append_output("")

        # Error header
        header = RichText()
        header.append("[ERROR] ", style="bold red")
        header.append(error_type, style="bold red")
        self._append_output(self._renderer.render(header, width=width).rstrip("\n"))

        # Error message (with word wrap)
        msg_text = RichText()
        msg_text.append("  ", style="dim")
        msg_text.append(error_msg)
        self._append_output(self._renderer.render(msg_text, width=width).rstrip("\n"))

        # Traceback (formatted, dimmed)
        tb_lines = traceback.format_exception(e)
        if tb_lines:
            tb_str = "".join(tb_lines).rstrip()
            for line in tb_str.splitlines():
                tb_text = RichText()
                tb_text.append(line, style="dim")
                self._append_output(self._renderer.render(tb_text, width=width).rstrip("\n"))

        self._append_output("")

        # Hint
        hint = RichText()
        hint.append("(History not saved - you can retry the same message)", style="dim")
        self._append_output(self._renderer.render(hint, width=width).rstrip("\n"))

    # =========================================================================
    # Steering Pane
    # =========================================================================

    def _get_steering_text(self) -> ANSI:
        """Get formatted steering messages for the steering pane."""
        if not self._steering_items:
            return ANSI(" [Steering messages will appear here during agent execution]")

        lines = []
        for _, text, status in reversed(self._steering_items[-self._max_steering_lines :]):
            if status == "acked":
                lines.append(f"[v] {text}")
            else:
                lines.append(f">>> {text}")

        return ANSI("\n".join(lines))

    def _get_steering_height(self) -> int:
        """Get dynamic height for steering pane."""
        if not self._steering_items:
            return 1
        return min(len(self._steering_items), self._max_steering_lines)

    def _add_steering_message(self, message: str) -> None:
        """Add a steering message to UI and send to message bus.

        This method:
        1. Adds the message to UI list with 'pending' status
        2. Sends formatted message to message bus via ctx.send_message

        The UI status will be updated to 'acked' when MessageReceivedEvent
        is received (event-driven UI update).
        """
        # Add to UI list with pending status (use content as key for matching)
        self._steering_items.append((message, message, "pending"))
        if self._app:
            self._app.invalidate()

        # Send to message bus with TUI-specific formatting
        self._send_steering_message(message)

    def _send_steering_message(self, message: str) -> None:
        """Send steering message to the message bus with TUI formatting."""
        try:
            self.runtime.ctx.send_message(
                BusMessage(
                    content=message,
                    source="user",
                    target="main",
                    template=STEERING_TEMPLATE,
                )
            )
            logger.debug("Steering message sent: %s", message[:50])
        except Exception:
            logger.exception("Failed to send steering message")

    def _ack_steering_by_content(self, content_preview: str) -> None:
        """Mark steering messages as acknowledged by matching content.

        Called when MessageReceivedEvent is received. Matches messages
        by content preview.
        """
        for i, (key, text, status) in enumerate(self._steering_items):
            # Match if the preview contains part of the original message
            if status == "pending" and text.strip() in content_preview:
                self._steering_items[i] = (key, text, "acked")
                break  # Only ack one message per event

        if self._app:
            self._app.invalidate()

    # =========================================================================
    # Status Bar
    # =========================================================================

    def _get_status_text(self) -> list[tuple[str, str]]:
        """Get formatted status bar text."""
        if self._shutdown_status:
            return [
                ("class:status-bar.warning", " SHUTDOWN "),
                ("class:status-bar", " | "),
                ("class:status-bar", self._shutdown_status),
            ]

        mode_style = f"class:status-bar.mode-{self._mode.value}"
        state_text = "RUNNING" if self._state == TUIState.RUNNING else "IDLE"
        attachment_label = self._format_pending_attachments_label()

        # Calculate context usage percentage
        if self._current_context_tokens > 0 and self._context_window_size > 0:
            context_pct = f"{self._current_context_tokens / self._context_window_size * 100:.0f}"
        else:
            context_pct = "--"

        # Build status based on state
        if self._state == TUIState.RUNNING:
            # Check if waiting for HITL approval
            if self._hitl_pending:
                approval_progress = f"{self._current_approval_index + 1}/{len(self._pending_approvals)}"
                return [
                    (mode_style, f" {self._mode.value.upper()} "),
                    ("class:status-bar", " | "),
                    ("class:status-bar.warning", f"Approval: {approval_progress}"),
                    ("class:status-bar", " | "),
                    ("class:status-bar", f"Context: {context_pct}%"),
                    ("class:status-bar", " | "),
                    ("class:status-bar", "Enter/Y: Approve | Text: Reject | Ctrl+C: Cancel"),
                ]
            else:
                # Show phase-specific status
                phase_display = {"thinking": "Thinking...", "tools": "Running tools..."}.get(
                    self._agent_phase, "Running..."
                )
                parts = [
                    (mode_style, f" {self._mode.value.upper()} "),
                    ("class:status-bar", " | "),
                    ("class:status-bar", phase_display),
                    ("class:status-bar", " | "),
                    ("class:status-bar", f"Context: {context_pct}%"),
                ]
                ctx = self.runtime.ctx
                if ctx.loop_active:
                    parts.extend([
                        ("class:status-bar", " | "),
                        ("class:status-bar.warning", f"Loop: {ctx.loop_iteration}/{ctx.loop_max_iterations}"),
                    ])
                bg_label = self._format_background_label()
                if bg_label:
                    parts.extend([
                        ("class:status-bar", " | "),
                        ("class:status-bar.warning", bg_label),
                    ])
                if attachment_label:
                    parts.extend([
                        ("class:status-bar", " | "),
                        ("class:status-bar.warning", attachment_label),
                    ])
                parts.extend([
                    ("class:status-bar", " | "),
                    ("class:status-bar", "Ctrl+C: Interrupt "),
                ])
                return parts
        else:
            # IDLE: show input mode and scroll hint
            if self._input_mode == "send":
                input_mode_text = "Enter:Send | Tab:Multiline"
            else:
                input_mode_text = "Enter:Newline | Tab:Send"

            scroll_hint = "Fn+Up/Down: Scroll" if sys.platform == "darwin" else "Ctrl+Up/Down: Scroll"

            parts = [
                (mode_style, f" {self._mode.value.upper()} "),
                ("class:status-bar", " | "),
                ("class:status-bar", f"State: {state_text}"),
                ("class:status-bar", " | "),
                ("class:status-bar", f"Context: {context_pct}%"),
            ]
            bg_label = self._format_background_label()
            if bg_label:
                parts.extend([
                    ("class:status-bar", " | "),
                    ("class:status-bar.warning", bg_label),
                ])
            if attachment_label:
                parts.extend([
                    ("class:status-bar", " | "),
                    ("class:status-bar.warning", attachment_label),
                ])
            parts.extend([
                ("class:status-bar", " | "),
                ("class:status-bar", input_mode_text),
                ("class:status-bar", " | "),
                ("class:status-bar", scroll_hint),
                ("class:status-bar", " | "),
                ("class:status-bar", "Ctrl+C: Exit "),
            ])
            return parts

    def _get_prompt(self) -> str:
        """Get the input prompt based on current state."""
        state_indicator = "*" if self._state == TUIState.RUNNING else ">"
        mouse_mode = "scroll" if self._mouse_enabled else "select"
        return f"[{mouse_mode}] {state_indicator} "

    # =========================================================================
    # Agent Execution
    # =========================================================================

    def _load_guidance_files(self) -> tuple[str | None, str | None]:
        """Load project guidance (AGENTS.md) and user rules (RULES.md).

        Returns:
            Tuple of (project_guidance, user_rules), each can be None if not found.
        """
        project_guidance = None
        user_rules = None

        # Load AGENTS.md from working directory
        agents_path = self.working_dir / "AGENTS.md"
        if agents_path.exists() and agents_path.is_file():
            try:
                content = agents_path.read_text(encoding="utf-8")
                if content.strip():
                    project_guidance = (
                        f"<{PROJECT_GUIDANCE_TAG} name={agents_path.name}>\n{content}\n</{PROJECT_GUIDANCE_TAG}>"
                    )
                    logger.debug(f"Loaded project guidance from {agents_path}")
            except Exception as e:
                logger.warning(f"Failed to read {agents_path}: {e}")

        # Load RULES.md from user config directory
        rules_path = self.config_manager.config_dir / "RULES.md"
        if rules_path.exists() and rules_path.is_file():
            try:
                content = rules_path.read_text(encoding="utf-8")
                if content.strip():
                    user_rules = f"<{USER_RULES_TAG} location={rules_path.absolute().as_posix()}>\n{content}\n</{USER_RULES_TAG}>"
                    logger.debug(f"Loaded user rules from {rules_path}")
            except Exception as e:
                logger.warning(f"Failed to read {rules_path}: {e}")

        return project_guidance, user_rules

    def _build_user_prompt(
        self,
        user_input: str,
        attachments: Sequence[PendingAttachment] | None = None,
    ) -> str | list[UserContent]:
        """Build the full user prompt with optional clipboard image attachments."""
        project_guidance, user_rules = self._load_guidance_files()
        attachment_list = list(attachments or [])

        mode_reminder: str | None = None
        if self._mode == TUIMode.PLAN:
            mode_reminder = (
                "<mode-reminder>\n"
                "You are currently in PLAN mode. In this mode:\n"
                "- Do NOT make any modifications to existing code or files\n"
                "- Do NOT execute commands that change system state\n"
                "- Focus on analysis, discussion, and planning\n"
                "- You MAY create new draft files (e.g., in .drafts/ or .handoff/) to save context, "
                "discussion results, plans, or design documents\n"
                "You can ask user to switch back to ACT mode by typing '/act'\n"
                "</mode-reminder>"
            )

        if not attachment_list and not project_guidance and not user_rules and not mode_reminder:
            return user_input

        if not attachment_list:
            parts: list[UserContent] = [user_input]
            if project_guidance:
                parts.append(project_guidance)
            if user_rules:
                parts.append(user_rules)
            if mode_reminder:
                parts.append(mode_reminder)
            return parts

        prompt_parts: list[UserContent] = []
        if user_input:
            prompt_parts.append(user_input)
        elif len(attachment_list) == 1:
            prompt_parts.append("Please analyze the attached image.")
        else:
            prompt_parts.append("Please analyze the attached images.")

        for attachment in attachment_list:
            prompt_parts.append(BinaryContent(data=attachment.data, media_type=attachment.media_type))

        if project_guidance:
            prompt_parts.append(project_guidance)
        if user_rules:
            prompt_parts.append(user_rules)
        if mode_reminder:
            prompt_parts.append(mode_reminder)

        return prompt_parts

    def _get_background_monitor(self) -> BackgroundMonitor | None:
        """Get BackgroundMonitor from environment resources."""
        if self._runtime and self._runtime.env and self._runtime.env.resources:
            resource = self._runtime.env.resources.get(BACKGROUND_MONITOR_KEY)
            if isinstance(resource, BackgroundMonitor):
                return resource
        return None

    def _get_background_task_count(self) -> int:
        """Get the number of active background tasks."""
        monitor = self._get_background_monitor()
        if monitor is None:
            return 0
        return len(monitor.active_tasks)

    def _get_background_process_count(self) -> int:
        """Get the number of active background shell processes."""
        try:
            if self._runtime and self._runtime.env and self._runtime.env.shell:
                return len(self._runtime.env.shell.active_background_processes)
        except RuntimeError:
            pass
        return 0

    def _format_background_label(self) -> str:
        """Format background indicator label combining tasks and processes.

        Returns empty string if nothing is running. Examples:
        - "BG: 2 tasks" (only subagent tasks)
        - "BG: 3 procs" (only shell processes)
        - "BG: 2 tasks, 3 procs" (both)
        """
        task_count = self._get_background_task_count()
        proc_count = self._get_background_process_count()
        if task_count == 0 and proc_count == 0:
            return ""
        parts: list[str] = []
        if task_count > 0:
            parts.append(f"{task_count} task{'s' if task_count != 1 else ''}")
        if proc_count > 0:
            parts.append(f"{proc_count} proc{'s' if proc_count != 1 else ''}")
        return f"BG: {', '.join(parts)}"

    def _on_background_task_complete(self, agent_id: str) -> None:
        """Callback invoked when a background task completes.

        This is called synchronously from the asyncio event loop when
        SpawnDelegateTool finishes. If the agent is idle, queued background
        results are redelivered and a new agent turn is scheduled.

        Args:
            agent_id: The ID of the completed background agent.
        """
        # Only trigger if agent is idle - if running, we set a flag to redeliver
        # after the current turn completes (see _check_pending_bus_messages)
        if self._state != TUIState.IDLE:
            logger.debug("Background task %s completed while agent running, will check after turn", agent_id)
            self._pending_bus_check_needed = True
            return

        if not self._deliver_background_messages():
            logger.debug("Background task %s completed but no pending messages", agent_id)
            return

        logger.info("Background task %s completed, triggering agent turn", agent_id)

        # Show UI notification that background task completed
        self._append_system_output(f"Background task completed: {agent_id}")

        # Set state atomically BEFORE create_task to prevent race
        self._state = TUIState.RUNNING

        # Schedule agent turn - empty prompt, the bus message IS the input
        self._agent_task = asyncio.create_task(self._run_agent(""))
        self._agent_task.add_done_callback(self._on_agent_task_done)

    def _on_agent_task_done(self, task: asyncio.Task[None]) -> None:
        """Callback when agent task completes - handles uncaught exceptions."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            if _is_benign_contextvar_cleanup_error(exc):
                logger.debug("Suppressed benign ContextVar cleanup error from agent task: %s", _safe_exception_str(exc))
                return
            # Exception was not caught in _run_agent - display it
            logger.error("Uncaught exception in agent task: %s: %s", type(exc).__name__, _safe_exception_str(exc))
            self._append_error_output(exc)
            self._agent_phase = "idle"
            self._state = TUIState.IDLE
            if self._app:
                self._app.invalidate()
            return

    def _deliver_background_messages(self) -> bool:
        """Redeliver queued background notifications into the current main-agent bus."""
        monitor = self._get_background_monitor()
        ctx = self.runtime.ctx
        delivered = 0
        if monitor is not None:
            for snapshot in monitor.drain_usage_snapshots():
                self._session_usage.set_run_snapshot(snapshot)
                self._session_usage.commit_run_snapshot(snapshot.run_id)
            delivered = monitor.deliver_pending_messages(ctx.message_bus, ctx.agent_id)
        return delivered > 0 or ctx.message_bus.has_pending(ctx.agent_id)

    def _check_pending_bus_messages(self) -> None:
        """Check for pending bus messages and trigger agent turn if needed.

        Called after agent execution completes to redeliver messages that
        arrived while the main agent was still running.
        """
        # Only proceed if flag was set (background task completed during run)
        if not self._pending_bus_check_needed:
            return
        self._pending_bus_check_needed = False

        # Must be idle now
        if self._state != TUIState.IDLE:
            return

        if not self._deliver_background_messages():
            return

        logger.info("Pending bus messages detected after agent turn, triggering new turn")

        # Show UI notification
        self._append_system_output("Processing pending messages from background tasks...")

        # Set state atomically BEFORE create_task to prevent race
        self._state = TUIState.RUNNING

        # Schedule agent turn - empty prompt, the bus message IS the input
        self._agent_task = asyncio.create_task(self._run_agent(""))
        self._agent_task.add_done_callback(self._on_agent_task_done)

    async def _run_agent(
        self,
        user_input: str,
        attachments: Sequence[PendingAttachment] | None = None,
    ) -> None:
        """Execute agent with HITL inner loop for tool approvals."""
        self._state = TUIState.RUNNING
        turn_attachments = list(attachments or [])
        self._pending_bus_check_needed = False
        self._tool_messages.clear()
        self._printed_tool_calls.clear()
        self._subagent_states.clear()
        self._event_renderer.clear()
        cancelled = False

        try:
            # Initial agent execution
            result = await self._execute_stream(user_input, turn_attachments)

            # HITL inner loop: keep processing until we get final str output
            while result and isinstance(result.output, DeferredToolRequests):
                deferred = result.output
                if not deferred.approvals:
                    # No approvals needed, just continue
                    break

                # Collect user approval decisions
                user_response = await self._request_user_action(deferred)

                # Resume agent with approval results
                result = await self._execute_stream(user_response)

            # Agent completed successfully

        except asyncio.CancelledError:
            cancelled = True
            self._finalize_streaming_text()
            self._finalize_streaming_thinking()
            self._append_output("[Cancelled]")
            # Don't update message history on cancel - allows retry
        except Exception as e:
            if _is_benign_contextvar_cleanup_error(e):
                logger.debug("Suppressed benign ContextVar cleanup error in agent run: %s", _safe_exception_str(e))
            else:
                self._finalize_streaming_text()
                self._finalize_streaming_thinking()
                self._append_error_output(e)
                if self.has_session_data:
                    self._append_system_output(
                        "Session state saved. Enter your next prompt to continue from the current context."
                    )
                    self._append_system_output(
                        f"After restarting, run /session {self._session_id} to restore this session."
                    )
                logger.exception("Agent execution failed")
        finally:
            # Finalize any remaining streaming text/thinking
            self._finalize_streaming_text()
            self._finalize_streaming_thinking()
            # Reset all HITL state
            self._reset_hitl_state()
            # NOTE: Do NOT call consume_messages() here.
            # It would swallow background subagent results that arrived after
            # the last LLM request. The inject_bus_messages filter already
            # tracks consumed IDs for idempotency, so messages won't duplicate.
            self._steering_items.clear()
            # Clear user steering messages that were not consumed this turn.
            # These are messages injected via bus from user during execution.
            # If not cleared, they would leak into unrelated future tasks.
            self.runtime.ctx.steering_messages.clear()
            # Clean up loop state if still active (cancelled or error)
            ctx = self.runtime.ctx
            if ctx.loop_active:
                self._append_system_output("[Loop] Cancelled")
                ctx.reset_loop()
            self._agent_phase = "idle"
            self._state = TUIState.IDLE
            if self._app:
                self._app.invalidate()
            # Check if we need to trigger a new turn for pending bus messages
            # (e.g., background task completed while we were running).
            # Skip this if the user explicitly cancelled (Ctrl+C) -- they
            # intended to stop execution, not restart it immediately.
            if not cancelled:
                self._check_pending_bus_messages()

    def _reset_hitl_state(self) -> None:
        """Reset all HITL-related state variables.

        Called after agent execution completes (success, error, or cancel)
        to ensure clean state for next execution.
        """
        self._hitl_pending = False
        self._pending_approvals.clear()
        self._current_approval_index = 0
        self._approval_result = None
        self._approval_reason = None
        # Don't set _approval_event to None here as it may still be awaited
        # Instead, set it if it exists to unblock any waiting coroutine
        if self._approval_event and not self._approval_event.is_set():
            # Signal cancellation by setting result to False
            self._approval_result = False
            self._approval_reason = "Cancelled"
            self._approval_event.set()
        self._approval_event = None

    async def _execute_stream(
        self,
        prompt: str | DeferredToolResults,
        attachments: Sequence[PendingAttachment] | None = None,
    ) -> Any:
        """Execute a single agent stream and return the result.

        Args:
            prompt: User prompt string or DeferredToolResults from approval.

        Returns:
            AgentRunResult with output (str or DeferredToolRequests).
        """
        # Clear tracking for new stream
        self._tool_messages.clear()
        self._printed_tool_calls.clear()
        self._subagent_states.clear()

        # Build user prompt if string input
        if isinstance(prompt, str):
            user_prompt = self._build_user_prompt(prompt, attachments)
            deferred_results = None
        else:
            user_prompt = ""
            deferred_results = prompt

        async with stream_agent(
            self.runtime,  # type: ignore[arg-type] # TUIContext is subclass of AgentContext
            user_prompt=user_prompt if user_prompt else None,
            message_history=self._message_history,
            deferred_tool_results=deferred_results,
            usage_limits=UsageLimits(request_limit=self.config.general.max_requests),
            post_node_hook=emit_context_update,
            resume_on_error=self.config.general.agent_stream_resume_on_error,
            resume_max_attempts=self.config.general.agent_stream_resume_max_attempts,
            resume_prompt=self.config.general.agent_stream_resume_prompt,
        ) as stream:
            async for event in stream:
                self._handle_stream_event(event)

            # Always try to save message history from the run, even on error.
            # This preserves conversation context so user can continue
            # with a new prompt after an error, instead of losing all context.
            # Wrapped in try/except since run fields may be incomplete on error.
            self._last_run = stream.run
            try:
                if stream.run:
                    self._message_history = list(stream.run.all_messages())
                    # Update context usage from run
                    usage = stream.run.usage()
                    latest_usage = get_latest_request_usage(self._message_history)
                    self._current_context_tokens = latest_usage.total_tokens if latest_usage else usage.total_tokens

                    # Accumulate main-agent usage when no realtime snapshot was emitted.
                    if not self._session_usage.has_run_snapshot:
                        model_id = cast(Model, self.runtime.agent.model).model_name
                        self._session_usage.add("main", model_id, usage)
                    self._session_usage.commit_run_snapshot()
            except Exception:
                logger.debug("Failed to save message history from errored run", exc_info=True)

            # Persist the latest recoverable state before surfacing stream errors.
            # Error paths include the usage ledger to preserve crash-recovery context.
            try:
                stream.raise_if_exception()
            except Exception:
                self._save_session_snapshot(include_usage_ledger=True, save_reason="error")
                raise

            self._auto_save_history()
            return stream.run.result if stream.run else None

    async def _request_user_action(
        self,
        deferred: DeferredToolRequests,
    ) -> DeferredToolResults:
        """Collect approval decisions from user for pending tool calls.

        Args:
            deferred: DeferredToolRequests containing tools needing approval.

        Returns:
            DeferredToolResults with approval decisions.
        """
        results = DeferredToolResults()

        if not deferred.approvals:
            return results

        self._hitl_pending = True
        self._pending_approvals = list(deferred.approvals)
        self._current_approval_index = 0

        self._append_output("")
        self._append_output(f"[Tool approval required: {len(deferred.approvals)} tool(s)]")

        for idx, tool_call in enumerate(deferred.approvals):
            self._current_approval_index = idx
            # Display approval panel
            self._display_approval_panel(
                tool_call,
                idx + 1,
                len(deferred.approvals),
                deferred.metadata.get(tool_call.tool_call_id),
            )

            # Wait for user decision (blocking with asyncio.Event)
            approved, reason = await self._wait_for_approval_input()

            if approved:
                results.approvals[tool_call.tool_call_id] = True
                self._append_output(f"  [Approved: {tool_call.tool_name}]")
            else:
                results.approvals[tool_call.tool_call_id] = ToolDenied(reason or "User rejected")
                self._append_output(f"  [Rejected: {tool_call.tool_name} - {reason or 'User rejected'}]")

        self._hitl_pending = False
        self._pending_approvals.clear()
        self._append_output("")

        return results

    async def _wait_for_approval_input(self) -> tuple[bool, str | None]:
        """Wait for user's approval decision.

        Returns:
            Tuple of (approved: bool, reason: str | None).
        """
        self._approval_event = asyncio.Event()
        self._approval_result = None
        self._approval_reason = None

        if self._app:
            self._app.invalidate()

        # Wait for key handler to set the event
        await self._approval_event.wait()

        approved = self._approval_result if self._approval_result is not None else False
        reason = f"User not approved with response: `{self._approval_reason}`"

        self._approval_event = None
        return approved, reason

    def _format_args_for_display(self, args: Any, max_str_len: int = 500, max_lines: int = 30) -> str:
        """Format tool arguments for display with smart truncation.

        Args:
            args: Tool arguments (can be dict, JSON string, or any object)
            max_str_len: Maximum length for string values before truncation
            max_lines: Maximum number of lines in output

        Returns:
            Formatted JSON string or fallback representation
        """

        def truncate_strings(obj: Any, max_len: int) -> Any:
            """Recursively truncate long strings in nested structures."""
            if isinstance(obj, str):
                if len(obj) > max_len:
                    return obj[:max_len] + f"... ({len(obj) - max_len} more chars)"
                return obj
            elif isinstance(obj, dict):
                return {k: truncate_strings(v, max_len) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [truncate_strings(item, max_len) for item in obj]
            return obj

        try:
            # If args is a string, try to parse it as JSON first
            if isinstance(args, str):
                try:
                    parsed = json.loads(args)
                    args = parsed
                except json.JSONDecodeError:
                    # Not valid JSON, treat as plain string
                    if len(args) > max_str_len:
                        return args[:max_str_len] + f"\n... ({len(args) - max_str_len} more chars)"
                    return args

            # Truncate long strings in the structure
            truncated = truncate_strings(args, max_str_len)

            # Format as pretty JSON
            formatted = json.dumps(truncated, indent=2, ensure_ascii=False)

            # Limit total lines
            lines = formatted.split("\n")
            if len(lines) > max_lines:
                formatted = "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"

            return formatted

        except Exception:
            # Ultimate fallback: convert to string
            result = str(args)
            if len(result) > max_str_len:
                result = result[:max_str_len] + f"... ({len(result) - max_str_len} more chars)"
            return result

    def _display_approval_panel(
        self,
        tool_call: ToolCallPart,
        index: int,
        total: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Display approval panel for a tool call."""
        from rich.console import Group
        from rich.panel import Panel
        from rich.syntax import Syntax

        content_parts: list[Any] = [
            Text(f"Tool {index} of {total}", style="bold cyan"),
            Text(""),
            Text(f"Tool: {tool_call.tool_name}", style="bold yellow"),
        ]

        if metadata and metadata.get("reviewer") == "shell_command_reviewer":
            reason = metadata.get("reason")
            risk_level = metadata.get("risk_level")
            if reason or risk_level:
                content_parts.append(Text(""))
                content_parts.append(Text("Shell review:", style="bold cyan"))
                if risk_level:
                    content_parts.append(Text(f"Risk: {risk_level}", style="yellow"))
                if reason:
                    content_parts.append(Text(f"Reason: {reason}", style="yellow"))

        if tool_call.args:
            content_parts.append(Text(""))
            content_parts.append(Text("Arguments:", style="bold cyan"))
            formatted_args = self._format_args_for_display(tool_call.args)
            code_theme = self.config.display.code_theme or "monokai"
            # Determine if it looks like JSON for syntax highlighting
            is_json_like = formatted_args.strip().startswith(("{", "["))
            syntax = Syntax(formatted_args, "json" if is_json_like else "text", theme=code_theme)
            content_parts.append(syntax)

        panel = Panel(
            Group(*content_parts),
            title="[yellow]Tool Approval Required[/yellow]",
            subtitle="[dim]Enter/Y: Approve | Any text: Reject with reason | Ctrl+C: Cancel[/dim]",
            border_style="yellow",
            padding=(1, 2),
        )

        # Render panel to ANSI and append
        rendered = self._renderer.render(panel, width=self._get_terminal_width())
        self._append_output(rendered.rstrip())

    # =========================================================================
    # Subagent Event Handling
    # =========================================================================

    def _handle_subagent_start(self, event: SubagentStartEvent) -> None:
        """Handle subagent start event - create progress line."""
        agent_id = event.agent_id
        agent_name = event.agent_name

        # Create progress line
        text = Text()
        text.append(f"[{agent_id}] ", style="cyan")
        text.append("Running...", style="dim")
        rendered = self._renderer.render(text, width=self._get_terminal_width())

        # Track state with line index for later update
        line_index = len(self._output_lines)
        self._subagent_states[agent_id] = {
            "line_index": line_index,
            "tool_names": [],
            "agent_name": agent_name,
        }
        self._append_block(rendered.rstrip())
        self._throttled_invalidate()

    def _handle_subagent_complete(self, event: SubagentCompleteEvent) -> None:
        """Handle subagent complete event - update progress line to summary."""
        agent_id = event.agent_id

        if agent_id not in self._subagent_states:
            # Start event was missed, just show completion
            text = Text()
            if event.success:
                text.append(f"[{agent_id}] ", style="cyan")
                text.append("Done ", style="bold green")
                text.append(f"({event.duration_seconds:.1f}s)", style="dim")
                if event.request_count > 0:
                    text.append(f" | {event.request_count} reqs", style="dim")
            else:
                text.append(f"[{agent_id}] ", style="cyan")
                text.append("Failed ", style="bold red")
                text.append(f"({event.duration_seconds:.1f}s)", style="dim")
                if event.error:
                    text.append(f" | {event.error[:50]}", style="dim red")
            rendered = self._renderer.render(text, width=self._get_terminal_width())
            self._append_output(rendered.rstrip())
            return

        state = self._subagent_states[agent_id]
        line_index = state["line_index"]

        # Build summary line
        text = Text()
        if event.success:
            text.append(f"[{agent_id}] ", style="cyan")
            text.append("Done ", style="bold green")
            text.append(f"({event.duration_seconds:.1f}s)", style="dim")
            if event.request_count > 0:
                text.append(f" | {event.request_count} reqs", style="dim")
            if event.result_preview:
                # Truncate result preview
                preview = event.result_preview.replace("\n", " ")[:60]
                if len(event.result_preview) > 60:
                    preview += "..."
                text.append(f' | "{preview}"', style="dim italic")
        else:
            text.append(f"[{agent_id}] ", style="cyan")
            text.append("Failed ", style="bold red")
            text.append(f"({event.duration_seconds:.1f}s)", style="dim")
            if event.error:
                error_preview = event.error[:50]
                text.append(f" | {error_preview}", style="dim red")

        rendered = self._renderer.render(text, width=self._get_terminal_width())

        # Update the line in place
        if line_index < len(self._output_lines):
            self._update_block(line_index, rendered.rstrip())

        # Clean up state
        del self._subagent_states[agent_id]
        self._throttled_invalidate()

    def _update_subagent_progress_line(self, agent_id: str) -> None:
        """Update subagent progress line with current tool list."""
        if agent_id not in self._subagent_states:
            return

        state = self._subagent_states[agent_id]
        line_index = state["line_index"]
        tool_names = state["tool_names"]

        # Build progress line
        text = Text()
        text.append(f"[{agent_id}] ", style="cyan")
        text.append("Running... ", style="dim")

        if tool_names:
            # Show last few tools
            recent_tools = tool_names[-3:]  # Last 3 tools
            tools_str = ", ".join(recent_tools)
            if len(tool_names) > 3:
                tools_str = f"...{tools_str}"
            text.append(tools_str, style="dim yellow")
            text.append(f" ({len(tool_names)} tools)", style="dim")

        rendered = self._renderer.render(text, width=self._get_terminal_width())

        # Update the line in place
        if line_index < len(self._output_lines):
            self._update_block(line_index, rendered.rstrip())
            self._throttled_invalidate()

    @staticmethod
    def _is_background_agent(agent_id: str) -> bool:
        """Check if an agent_id belongs to a background subagent."""
        return "-bg-" in agent_id

    def _handle_stream_event(self, event: StreamEvent) -> None:
        """Handle a stream event from agent execution."""
        message_event = event.event
        agent_id = event.agent_id

        # Suppress all events from background subagents.
        # Their results are delivered via message bus, not streamed.
        if self._is_background_agent(agent_id):
            return

        # Handle subagent lifecycle events (from any agent)
        if isinstance(message_event, SubagentStartEvent):
            # Suppress background subagent start events
            if self._is_background_agent(message_event.agent_id):
                return
            self._handle_subagent_start(message_event)
            return

        if isinstance(message_event, SubagentCompleteEvent):
            # Suppress background subagent complete events
            if self._is_background_agent(message_event.agent_id):
                return
            self._handle_subagent_complete(message_event)
            return

        # For subagent events (not main), only track tool calls silently
        if agent_id != "main" and agent_id in self._subagent_states:
            if isinstance(message_event, FunctionToolCallEvent):
                # Track tool name for progress display
                tool_name = message_event.part.tool_name
                self._subagent_states[agent_id]["tool_names"].append(tool_name)
                self._update_subagent_progress_line(agent_id)
            # Ignore all other subagent events (text streaming, tool results, etc.)
            return

        if isinstance(message_event, UsageSnapshotEvent):
            if message_event.snapshot is not None:
                self._session_usage.set_run_snapshot(message_event.snapshot)
            return

        # Main agent events - normal processing
        if isinstance(message_event, PartStartEvent) and isinstance(message_event.part, TextPart):
            # Start new streaming text block
            self._finalize_streaming_text()  # Finalize any previous
            self._finalize_streaming_thinking()  # Finalize any thinking
            self._start_streaming_text(message_event.part.content)

        elif isinstance(message_event, PartStartEvent) and isinstance(message_event.part, ThinkingPart):
            # Start new streaming thinking block (extended thinking from model)
            self._finalize_streaming_text()  # Finalize any active text (interleaved thinking)
            self._finalize_streaming_thinking()  # Finalize any previous thinking
            self._start_streaming_thinking(message_event.part.content)

        elif isinstance(message_event, PartDeltaEvent) and isinstance(message_event.delta, TextPartDelta):
            # Update streaming text with delta
            if self._streaming_line_index is not None:
                self._update_streaming_text(message_event.delta.content_delta)
            else:
                # Fallback if no streaming started
                self._start_streaming_text(message_event.delta.content_delta)

        elif isinstance(message_event, PartDeltaEvent) and isinstance(message_event.delta, ThinkingPartDelta):
            # Update streaming thinking with delta
            if message_event.delta.content_delta:
                if self._streaming_thinking_line_index is not None:
                    self._update_streaming_thinking(message_event.delta.content_delta)
                else:
                    # Fallback if no streaming started
                    self._start_streaming_thinking(message_event.delta.content_delta)

        elif isinstance(message_event, PartStartEvent):
            # Other part types (ToolCallPart, FilePart, etc.) - finalize active streams
            self._finalize_streaming_text()
            self._finalize_streaming_thinking()

        elif isinstance(message_event, PartEndEvent):
            # Part completed - finalize the corresponding stream
            if isinstance(message_event.part, TextPart):
                self._finalize_streaming_text()
            elif isinstance(message_event.part, ThinkingPart):
                self._finalize_streaming_thinking()

        elif isinstance(message_event, FunctionToolCallEvent):
            # Finalize any streaming text before tool call
            self._finalize_streaming_text()
            self._finalize_streaming_thinking()

            tool_call_id = message_event.part.tool_call_id
            tool_name = message_event.part.tool_name
            self._tool_messages[tool_call_id] = ToolMessage(
                tool_call_id=tool_call_id,
                name=tool_name,
                args=message_event.part.args,
            )
            self._event_renderer.tracker.start_call(tool_call_id, tool_name, message_event.part.args)
            rendered = self._event_renderer.render_tool_call_start(tool_name, tool_call_id)
            self._append_output(rendered.rstrip())

        elif isinstance(message_event, FunctionToolResultEvent):
            tool_call_id = message_event.tool_call_id
            if tool_call_id in self._tool_messages:
                tool_msg = self._tool_messages[tool_call_id]
                result_content = self._extract_tool_result(message_event)
                tool_msg.content = result_content
                self._event_renderer.tracker.complete_call(tool_call_id, result_content)

                if tool_call_id not in self._printed_tool_calls:
                    # Get duration from tracker
                    duration = 0.0
                    if tool_call_id in self._event_renderer.tracker.tool_calls:
                        duration = self._event_renderer.tracker.tool_calls[tool_call_id].duration()
                    rendered = self._event_renderer.render_tool_call_complete(
                        tool_msg, duration=duration, width=self._get_terminal_width()
                    )
                    self._append_output(rendered.rstrip())
                    self._printed_tool_calls.add(tool_call_id)

        # Handle SDK events (compact, handoff)
        elif isinstance(message_event, CompactStartEvent):
            self._finalize_streaming_text()
            self._finalize_streaming_thinking()
            rendered = self._event_renderer.render_compact_start(message_event.message_count)
            self._append_output(rendered.rstrip())

        elif isinstance(message_event, CompactCompleteEvent):
            rendered = self._event_renderer.render_compact_complete(
                message_event.original_message_count,
                message_event.compacted_message_count,
                message_event.summary_markdown,
            )
            self._append_output(rendered.rstrip())

        elif isinstance(message_event, CompactFailedEvent):
            rendered = self._event_renderer.render_compact_failed(message_event.error)
            self._append_output(rendered.rstrip())

        elif isinstance(message_event, HandoffStartEvent):
            self._finalize_streaming_text()
            self._finalize_streaming_thinking()
            rendered = self._event_renderer.render_handoff_start(message_event.message_count)
            self._append_output(rendered.rstrip())

        elif isinstance(message_event, HandoffCompleteEvent):
            rendered = self._event_renderer.render_handoff_complete(message_event.handoff_content)
            self._append_output(rendered.rstrip())

        elif isinstance(message_event, HandoffFailedEvent):
            rendered = self._event_renderer.render_handoff_failed(message_event.error)
            self._append_output(rendered.rstrip())

        # Handle task/memory state events
        elif isinstance(message_event, TaskEvent):
            rendered = self._event_renderer.render_task_event(message_event)
            self._append_output(rendered.rstrip())

        elif isinstance(message_event, NoteEvent):
            rendered = self._event_renderer.render_note_event(message_event)
            self._append_output(rendered.rstrip())

        elif isinstance(message_event, FileChangeEvent):
            rendered = self._event_renderer.render_file_change_event(message_event, width=self._get_terminal_width())
            if rendered:
                self._append_output(rendered.rstrip())

        # Handle TUI-specific events
        elif isinstance(message_event, LoopIterationEvent):
            self._append_system_output(f"[Loop] Iteration {message_event.iteration}/{message_event.max_iterations}")

        elif isinstance(message_event, LoopCompleteEvent):
            if message_event.reason == LoopCompleteReason.verified:
                self._append_system_output(f"[Loop] Task completed in {message_event.iteration} iteration(s)")
            elif message_event.reason == LoopCompleteReason.max_iterations:
                self._append_system_output(
                    f"[Loop] Reached max iterations ({message_event.iteration}). "
                    "Task may be incomplete. You can run /loop again to continue."
                )

        elif isinstance(message_event, ContextUpdateEvent):
            self._current_context_tokens = message_event.total_tokens
            if message_event.context_window_size > 0:
                self._context_window_size = message_event.context_window_size

        # Handle SDK lifecycle events for status bar
        elif isinstance(message_event, ModelRequestStartEvent):
            self._agent_phase = "thinking"

        elif isinstance(message_event, ToolCallsStartEvent):
            self._finalize_streaming_text()
            self._finalize_streaming_thinking()
            self._agent_phase = "tools"

        elif isinstance(message_event, MessageReceivedEvent):
            # Update UI status to acked for matched steering messages
            for msg_info in message_event.messages:
                if msg_info.source == "user":
                    self._ack_steering_by_content(msg_info.content_text)

            # Render user messages
            user_messages = [m for m in message_event.messages if m.source == "user"]
            if user_messages:
                previews = [m.content_text for m in user_messages]
                rendered = self._event_renderer.render_steering_injected(previews)
                self._append_output(rendered.rstrip())

        if self._app:
            self._app.invalidate()

    async def _paste_clipboard_image(self) -> None:
        """Attach an image from the system clipboard when available."""
        try:
            clipboard_result = await read_clipboard_image()
        except Exception as e:
            logger.exception("Failed to inspect clipboard image during explicit paste")
            clipboard_result = ClipboardImageReadResult(image=None, error=_safe_exception_str(e))

        image = clipboard_result.image
        if image is not None:
            attachment = PendingAttachment(
                data=image.data,
                media_type=image.media_type,
                size_bytes=len(image.data),
            )
            self._pending_attachments.append(attachment)
            self._history_index = -1
            self._current_input_backup = ""
            self._append_system_output(f"Attached {self._format_attachment_description(attachment)} from clipboard")
            if self._app:
                self._app.invalidate()
            return

        error = clipboard_result.error or "No clipboard image available."
        self._append_system_output(error)
        if self._app:
            self._app.invalidate()

    async def _handle_bracketed_paste(self, pasted_text: str, input_area: TextArea) -> None:
        """Handle terminal paste as plain text."""
        normalized_text = pasted_text.replace("\r\n", "\n").replace("\r", "\n")
        if normalized_text:
            input_area.buffer.insert_text(normalized_text)
            if self._app:
                self._app.invalidate()

    def _add_prompt_history(self, text: str) -> None:
        """Record submitted text for prompt history navigation."""
        if not text:
            return
        self._history_index = -1
        self._current_input_backup = ""
        if not self._prompt_history or self._prompt_history[-1] != text:
            self._prompt_history.append(text)

    def _submit_input(self, text: str, input_area: TextArea) -> None:
        """Submit the current input buffer content."""
        if self._state == TUIState.RUNNING:
            if text:
                self._add_prompt_history(text)
                self._add_steering_message(text)
                input_area.buffer.reset()
                return

            if self._pending_attachments:
                input_area.buffer.reset()
                self._append_system_output("Queued clipboard image for the next prompt after the current run finishes.")
                return

            input_area.buffer.reset()
            return

        if text.startswith("/"):
            self._add_prompt_history(text)
            input_area.buffer.reset()
            self._track_managed_task(asyncio.create_task(self._handle_command(text)))
            return

        if text.startswith("!"):
            self._add_prompt_history(text)
            input_area.buffer.reset()
            self._track_managed_task(asyncio.create_task(self._execute_shell_command(text[1:])))
            return

        attachments = self._consume_pending_attachments()
        if not text and not attachments:
            input_area.buffer.reset()
            return

        self._add_prompt_history(text)

        input_area.buffer.reset()
        self._append_user_input(text, attachments)
        self._agent_task = asyncio.create_task(self._run_agent(text, attachments))
        self._agent_task.add_done_callback(self._on_agent_task_done)

    def _extract_tool_result(self, event: FunctionToolResultEvent) -> str:
        """Extract result content from tool result event."""
        try:
            result = event.result
            if hasattr(result, "content"):
                content = result.content
                if isinstance(content, str):
                    return content
                rv = getattr(content, "return_value", None)
                if rv is not None:
                    return str(rv)
                return str(content)
            return str(result)
        except Exception:
            return "<result>"

    # =========================================================================
    # UI Setup
    # =========================================================================

    def _setup_input_keybindings(self, input_area: TextArea) -> KeyBindings:
        """Set up key bindings owned by the focused input control."""
        kb = KeyBindings()

        def previous_history() -> None:
            if not self._prompt_history:
                return
            # First time pressing up: backup current input
            if self._history_index == -1:
                self._current_input_backup = input_area.buffer.text
                self._history_index = len(self._prompt_history)
            # Move to previous item
            if self._history_index > 0:
                self._history_index -= 1
                input_area.buffer.text = self._prompt_history[self._history_index]
                input_area.buffer.cursor_position = len(input_area.buffer.text)

        def next_history() -> None:
            if self._history_index == -1:
                return
            # Move to next item
            self._history_index += 1
            if self._history_index >= len(self._prompt_history):
                # Reached end, restore original input
                self._history_index = -1
                input_area.buffer.text = self._current_input_backup
            else:
                input_area.buffer.text = self._prompt_history[self._history_index]
            input_area.buffer.cursor_position = len(input_area.buffer.text)

        @kb.add("up", eager=True)
        def handle_up(event: KeyPressEvent) -> None:
            """Navigate to previous prompt in history."""
            previous_history()

        @kb.add("c-p", eager=True)
        def handle_ctrl_p(event: KeyPressEvent) -> None:
            """Navigate to previous prompt in history using a TTY-stable key."""
            previous_history()

        @kb.add("down", eager=True)
        def handle_down(event: KeyPressEvent) -> None:
            """Navigate to next prompt in history."""
            next_history()

        @kb.add("c-n", eager=True)
        def handle_ctrl_n(event: KeyPressEvent) -> None:
            """Navigate to next prompt in history using a TTY-stable key."""
            next_history()

        def submit_or_newline() -> None:
            if self._input_mode == "send":
                text = input_area.buffer.text.strip()

                if self._hitl_pending and self._approval_event and not self._approval_event.is_set():
                    input_area.buffer.reset()
                    if text.lower() in ("", "y", "yes"):
                        self._approval_result = True
                        self._approval_reason = None
                    else:
                        self._approval_result = False
                        self._approval_reason = text if text else None
                    self._approval_event.set()
                    return

                self._submit_input(text, input_area)
            else:
                input_area.buffer.insert_text("\n")

        @kb.add("enter", eager=True)
        def handle_enter(event: KeyPressEvent) -> None:
            """Handle Enter based on current input mode."""
            submit_or_newline()

        @kb.add("c-j", eager=True)
        def handle_ctrl_j(event: KeyPressEvent) -> None:
            """Handle terminals that emit Ctrl+J for Enter."""
            submit_or_newline()

        return kb

    def _setup_keybindings(self, input_area: TextArea) -> KeyBindings:
        """Set up application-wide keyboard bindings."""
        kb = KeyBindings()

        @kb.add("c-c")
        def handle_ctrl_c(event: KeyPressEvent) -> None:
            """Handle Ctrl+C - cancel running task or double-press to exit."""
            current_time = time.time()

            if self._state == TUIState.RUNNING:
                # Running: request cancellation (state change handled by _run_agent finally)
                # Only cancel once - repeated cancellation can orphan internal tasks
                # and cause ContextVar errors (pydantic-ai wrap_task cleanup interrupted)
                if self._agent_task and not self._agent_task.done() and not self._agent_task.cancelling():
                    self._agent_task.cancel()
                    self._append_output("[Cancelling...]")
            else:
                # Idle: double-press to exit, single-press to clear input
                if current_time - self._last_ctrl_c_time < self._ctrl_c_exit_timeout:
                    self._show_shutdown_status("exit requested")
                    event.app.exit()
                else:
                    self._append_output("[Press Ctrl+C again to exit, or Ctrl+D to exit immediately]")
                    self._last_ctrl_c_time = current_time
                    # Clear input area on first Ctrl+C
                    input_area.buffer.reset()

        @kb.add("c-d")
        def handle_ctrl_d(event: KeyPressEvent) -> None:
            """Handle Ctrl+D - exit."""
            self._show_shutdown_status("exit requested")
            event.app.exit()

        # Scroll functions
        def _scroll_up(event: KeyPressEvent) -> None:
            """Scroll output up."""
            self._scroll_offset = max(0, self._scroll_offset - 10)
            if self._app:
                self._app.invalidate()

        def _scroll_down(event: KeyPressEvent) -> None:
            """Scroll output down."""
            max_scroll = self._get_max_scroll()
            self._scroll_offset = min(self._scroll_offset + 10, max_scroll)
            if self._app:
                self._app.invalidate()

        # Register scroll keybindings
        kb.add("pageup")(_scroll_up)
        kb.add("pagedown")(_scroll_down)
        if sys.platform == "darwin":
            kb.add("s-up")(_scroll_up)
            kb.add("s-down")(_scroll_down)
        else:
            kb.add("c-up")(_scroll_up)
            kb.add("c-down")(_scroll_down)

        @kb.add("c-l")
        def handle_ctrl_l(event: KeyPressEvent) -> None:
            """Scroll to bottom of output."""
            self._scroll_to_bottom()
            if self._app:
                self._app.invalidate()

        @kb.add("c-u")
        def handle_ctrl_u(event: KeyPressEvent) -> None:
            """Clear input line."""
            input_area.buffer.reset()
            self._history_index = -1

        @kb.add("escape")
        def handle_escape(event: KeyPressEvent) -> None:
            """Toggle mouse support mode."""
            self._mouse_enabled = not self._mouse_enabled
            if self._app and self._app.output:
                if self._mouse_enabled:
                    self._app.output.enable_mouse_support()
                else:
                    self._app.output.disable_mouse_support()

        @kb.add(Keys.BracketedPaste, eager=True)
        def handle_bracketed_paste(event: KeyPressEvent) -> None:
            """Handle terminal paste events as plain text."""
            pasted_text = event.data or ""
            if self._app:
                self._app.create_background_task(self._handle_bracketed_paste(pasted_text, input_area))
            else:
                self._track_managed_task(asyncio.create_task(self._handle_bracketed_paste(pasted_text, input_area)))

        @kb.add("c-v", eager=True)
        def handle_paste_image(event: KeyPressEvent) -> None:
            """Attach an image from the system clipboard."""
            # Use eager matching so this app-level binding wins over prompt_toolkit's
            # default buffer/control Ctrl+V handlers on macOS terminals.
            if self._app:
                self._app.create_background_task(self._paste_clipboard_image())
            else:
                self._track_managed_task(asyncio.create_task(self._paste_clipboard_image()))

        @kb.add("tab")
        def handle_tab(event: KeyPressEvent) -> None:
            """Toggle input mode between send and edit."""
            if self._input_mode == "send":
                self._input_mode = "edit"
            else:
                self._input_mode = "send"
            if self._app:
                self._app.invalidate()

        @kb.add("c-o")
        def handle_newline(event: KeyPressEvent) -> None:
            """Insert newline with Ctrl+O (works in both modes)."""
            input_area.buffer.insert_text("\n")

        # Word navigation (Option+Arrow on macOS)
        @kb.add("escape", "b")
        def handle_word_left(event: KeyPressEvent) -> None:
            """Move cursor to previous word."""
            buff = input_area.buffer
            pos = buff.document.find_previous_word_beginning(count=1)
            if pos:
                buff.cursor_position += pos

        @kb.add("escape", "f")
        def handle_word_right(event: KeyPressEvent) -> None:
            """Move cursor to next word."""
            buff = input_area.buffer
            pos = buff.document.find_next_word_ending(count=1)
            if pos:
                buff.cursor_position += pos

        return kb

    def _setup_style(self) -> Style:
        """Set up UI styles."""
        return Style.from_dict({
            "status-bar": "bg:ansiblue fg:white",
            "status-bar.mode-act": "bg:ansigreen fg:black bold",
            "status-bar.mode-plan": "bg:ansiblue fg:white bold",
            "steering-pane": "bg:ansibrightblack fg:ansicyan",
            "input-area": "",
        })

    # =========================================================================
    # Command Handling
    # =========================================================================

    async def _handle_command(self, command: str) -> None:
        """Handle slash commands."""
        try:
            await self._handle_command_inner(command)
        except Exception as e:
            logger.exception("Command failed: %s", command)
            self._append_error_output(e)
        finally:
            self._scroll_to_bottom()
            if self._app:
                self._app.invalidate()

    async def _handle_command_inner(self, command: str) -> None:
        """Inner command dispatch (exceptions caught by _handle_command)."""
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        # Built-in system commands (cannot be overridden)
        match cmd:
            case "/help":
                self._append_user_input(command)
                self._show_help()
            case "/clear":
                self._append_user_input(command)
                self._clear_session()
            case "/cost":
                self._append_user_input(command)
                self._show_cost()
            case "/perf":
                self._append_user_input(command)
                self._append_system_output(perf_report())
            case "/dump":
                self._append_user_input(command)
                self._dump_history(args.strip() if args else None)
            case "/session":
                self._append_user_input(command)
                if not args.strip():
                    self._list_sessions()
                else:
                    self._load_session(args.strip())
            case "/load":
                self._append_user_input(command)
                if not args.strip():
                    self._append_system_output("Usage: /load <folder>")
                    self._append_system_output("To restore a session by ID, use /session <id>")
                else:
                    self._load_history(args.strip())
            case "/exit":
                self._append_user_input(command)
                if self._app:
                    self._show_shutdown_status("exit requested")
                    self._app.exit()
            case "/act":
                self._append_user_input(command)
                self.switch_mode(TUIMode.ACT)
                self._append_system_output("Mode changed to ACT")
            case "/plan":
                self._append_user_input(command)
                self.switch_mode(TUIMode.PLAN)
                self._append_system_output("Mode changed to PLAN")
            case "/tasks":
                self._append_user_input(command)
                self._show_tasks()
            case "/paste-image":
                self._append_user_input(command)
                await self._paste_clipboard_image()
            case "/loop":
                self._append_user_input(command)
                ctx = self.runtime.ctx
                if ctx.loop_active:
                    self._append_system_output("Loop is already running. Use Ctrl+C to stop it first.")
                elif not args.strip():
                    self._append_system_output("Usage: /loop <task description>")
                else:
                    task = args.strip()
                    ctx.loop_task = task
                    ctx.loop_iteration = 0
                    ctx.loop_max_iterations = self.config.general.max_loop_iterations
                    self._append_system_output(
                        f"[Loop] Starting loop mode ({ctx.loop_max_iterations} max iterations). Ctrl+C to stop."
                    )
                    self._agent_task = asyncio.create_task(self._run_agent(task))
                    self._agent_task.add_done_callback(self._on_agent_task_done)
            case _:
                # Check custom commands
                cmd_name = cmd[1:]  # Remove leading /
                commands = self.config.get_commands()
                if cmd_name in commands:
                    cmd_def = commands[cmd_name]
                    # Switch mode if specified
                    if cmd_def.mode:
                        new_mode = TUIMode.ACT if cmd_def.mode == "act" else TUIMode.PLAN
                        self.switch_mode(new_mode)
                    # Append user instruction to prompt if provided
                    prompt = cmd_def.prompt
                    if args.strip():
                        prompt = f"{prompt}\n\nUser instruction: {args.strip()}"
                    # Show expanded prompt instead of command name
                    self._append_user_input(prompt)
                    self._agent_task = asyncio.create_task(self._run_agent(prompt))
                    self._agent_task.add_done_callback(self._on_agent_task_done)
                else:
                    # Unknown command - treat as regular prompt input
                    # This handles cases like /mnt/dev (file paths) gracefully
                    self._append_user_input(command)
                    self._agent_task = asyncio.create_task(self._run_agent(command))
                    self._agent_task.add_done_callback(self._on_agent_task_done)

    async def _terminate_direct_shell_process(self, process: asyncio.subprocess.Process | None) -> None:
        """Terminate a direct !command subprocess during timeout or shutdown."""
        if process is None or process.returncode is not None:
            return

        if os.name == "posix" and process.pid is not None:
            with contextlib.suppress(ProcessLookupError, OSError):
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        else:
            with contextlib.suppress(ProcessLookupError):
                process.terminate()

        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(process.wait(), timeout=_DIRECT_SHELL_TERMINATE_TIMEOUT)

        if process.returncode is not None:
            return

        if os.name == "posix" and process.pid is not None:
            with contextlib.suppress(ProcessLookupError, OSError):
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        else:
            with contextlib.suppress(ProcessLookupError):
                process.kill()

        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(process.wait(), timeout=_DIRECT_SHELL_TERMINATE_TIMEOUT)

    async def _execute_shell_command(self, command_str: str) -> None:
        """Execute a shell command directly and display output."""
        if not command_str.strip():
            self._append_system_output("Usage: !<command>")
            return

        # Show command being executed
        cmd_text = Text()
        cmd_text.append("$ ", style="bold cyan")
        cmd_text.append(command_str, style="cyan")
        self._append_output(self._renderer.render(cmd_text).rstrip())

        start_time = time.time()
        process: asyncio.subprocess.Process | None = None

        try:
            if os.name == "posix":
                process = await asyncio.create_subprocess_shell(
                    command_str,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.working_dir,
                    env=os.environ.copy(),
                    start_new_session=True,
                )
            else:
                process = await asyncio.create_subprocess_shell(
                    command_str,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.working_dir,
                    env=os.environ.copy(),
                )

            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)
            elapsed = time.time() - start_time

            # Display stdout (limit to 100 lines)
            if stdout:
                stdout_text = stdout.decode("utf-8", errors="replace").strip()
                if stdout_text:
                    lines = stdout_text.split("\n")
                    if len(lines) > 100:
                        lines = lines[:100]
                        lines.append(f"... ({len(stdout_text.split(chr(10))) - 100} more lines)")
                    self._append_output("\n".join(lines))

            # Display stderr in red (limit to 50 lines)
            if stderr:
                stderr_text = stderr.decode("utf-8", errors="replace").strip()
                if stderr_text:
                    lines = stderr_text.split("\n")
                    if len(lines) > 50:
                        lines = lines[:50]
                        lines.append("... (truncated)")
                    err_output = Text("\n".join(lines), style="red")
                    self._append_output(self._renderer.render(err_output).rstrip())

            # Show exit code if non-zero
            if process.returncode != 0:
                self._append_system_output(f"Exit code: {process.returncode}")

            # Show elapsed time
            self._append_output(f"({elapsed:.1f}s)")

        except TimeoutError:
            await self._terminate_direct_shell_process(process)
            self._append_system_output("Command timed out (300s)")
        except asyncio.CancelledError:
            await self._terminate_direct_shell_process(process)
            raise
        except Exception as e:
            self._append_system_output(f"Error: {type(e).__name__}: {e}")
        finally:
            if self._app:
                self._app.invalidate()

    def _show_help(self) -> None:
        """Display help text."""
        from rich.table import Table

        lines = []

        # Header
        header = Text("Available Commands", style="bold cyan")
        lines.append(self._renderer.render(header).rstrip())

        # System commands
        sys_table = Table(show_header=False, box=None, padding=(0, 2))
        sys_table.add_column("Command", style="green")
        sys_table.add_column("Description")
        sys_table.add_row("/help", "Show this help")
        sys_table.add_row("/clear", "Clear output and history")
        sys_table.add_row("/cost", "Show cost summary")
        sys_table.add_row("/tasks", "Show background tasks and processes")
        sys_table.add_row("/perf", "Show performance stats (YAACLI_PERF=1)")
        sys_table.add_row("/session [id]", "List sessions or restore by ID")
        sys_table.add_row("/paste-image", "Attach image from system clipboard")
        sys_table.add_row("/dump [folder]", "Export session to folder")
        sys_table.add_row("/load <folder>", "Load session from folder")
        sys_table.add_row("/act", "Switch to ACT mode")
        sys_table.add_row("/plan", "Switch to PLAN mode")
        sys_table.add_row("/loop <task>", "Run task in autonomous loop until complete")
        sys_table.add_row("/exit", "Exit TUI")
        lines.append(self._renderer.render(sys_table).rstrip())

        # Custom commands
        commands = self.config.get_commands()
        if commands:
            custom_header = Text("\nCustom Commands", style="bold cyan")
            lines.append(self._renderer.render(custom_header).rstrip())

            custom_table = Table(show_header=False, box=None, padding=(0, 2))
            custom_table.add_column("Command", style="yellow")
            custom_table.add_column("Description")
            for name, cmd_def in sorted(commands.items()):
                desc = cmd_def.description or "(no description)"
                custom_table.add_row(f"/{name} [instruction]", desc)
            lines.append(self._renderer.render(custom_table).rstrip())

        # Shell
        shell_header = Text("\nShell", style="bold cyan")
        lines.append(self._renderer.render(shell_header).rstrip())
        lines.append("  !<cmd>         Execute shell command directly")

        # Key bindings
        kb_header = Text("\nKey Bindings", style="bold cyan")
        lines.append(self._renderer.render(kb_header).rstrip())

        kb_table = Table(show_header=False, box=None, padding=(0, 2))
        kb_table.add_column("Key", style="yellow")
        kb_table.add_column("Action")
        kb_table.add_row("Ctrl+C", "Cancel / double-press exit")
        kb_table.add_row("Ctrl+D", "Exit")
        kb_table.add_row("Ctrl+V", "Attach image from clipboard")
        kb_table.add_row("Tab", "Toggle input mode")
        kb_table.add_row("Escape", "Toggle mouse mode")
        kb_table.add_row("Up/Down, Ctrl+P/N", "Browse history")
        kb_table.add_row("PageUp/PageDown", "Scroll output")
        lines.append(self._renderer.render(kb_table).rstrip())

        self._append_output("\n".join(lines))

    def _clear_session(self) -> None:
        """Clear output and message history.

        Resets:
        - Output lines and streaming state
        - Conversation history
        - Status bar context percentage
        - Scroll position

        Preserves:
        - Session usage (token/cost tracking)
        """
        self._output_lines.clear()
        # Reset virtual viewport state
        self._block_line_counts.clear()
        self._output_ansi_cache = None
        self._viewport_cache_key = None
        self._output_generation = 0
        self._total_line_count = 0
        self._scroll_offset = 0
        # Reset streaming state
        self._streaming_text = ""
        self._streaming_line_index = None
        self._streaming_thinking = ""
        self._streaming_thinking_line_index = None
        self._printed_tool_calls.clear()
        self._tool_messages.clear()
        self._subagent_states.clear()
        self._steering_items.clear()
        self._reset_pending_attachments()
        # Clear conversation history
        self._message_history = None
        self._last_run = None
        # Reset status bar context (but keep usage)
        self._current_context_tokens = 0
        # Show help after clear
        self._show_help()

    def _show_cost(self) -> None:
        """Show token usage summary for the current session."""
        summary = self._session_usage.format_summary()
        self._append_system_output(summary)

    def _show_tasks(self) -> None:
        """Show all background tasks, subagents, and processes."""
        lines: list[str] = []
        has_content = False

        # --- Section 1: Agent Tasks (from task_manager) ---
        task_manager = None
        all_tasks = []
        try:
            task_manager = self.runtime.ctx.task_manager
            all_tasks = task_manager.list_all()
        except RuntimeError:
            pass

        if all_tasks and task_manager:
            has_content = True
            header = Text("Agent Tasks", style="bold cyan")
            lines.append(self._renderer.render(header).rstrip())

            table = Table(show_header=True, box=None, padding=(0, 2))
            table.add_column("ID", style="dim")
            table.add_column("Status", style="bold")
            table.add_column("Subject")
            table.add_column("Owner", style="dim")

            status_styles = {
                "pending": "yellow",
                "in_progress": "cyan",
                "completed": "green",
            }

            for task in all_tasks:
                status_text = Text(task.status.value, style=status_styles.get(task.status.value, ""))
                blocked_suffix = ""
                if task.blocked_by:
                    active_blockers = [
                        bid for bid in task.blocked_by if (b := task_manager.get(bid)) and b.status != "completed"
                    ]
                    if active_blockers:
                        blocked_suffix = f" (blocked by {','.join(active_blockers)})"
                subject = task.subject + blocked_suffix
                owner = task.owner or ""
                table.add_row(f"T{task.id}", status_text, subject, owner)

            lines.append(self._renderer.render(table).rstrip())

        # --- Section 2: Background Subagents ---
        bg_monitor = self._get_background_monitor()
        bg_active: dict[str, asyncio.Task[Any]] = {}
        bg_infos: dict[str, BackgroundTaskInfo] = {}
        if bg_monitor:
            bg_active = bg_monitor.active_tasks
            bg_infos = bg_monitor.task_infos

        # Show all known background subagents (running + recently completed)
        if bg_infos:
            has_content = True
            if lines:
                lines.append("")
            header = Text("Background Subagents", style="bold cyan")
            lines.append(self._renderer.render(header).rstrip())

            table = Table(show_header=True, box=None, padding=(0, 2))
            table.add_column("Agent ID", style="dim")
            table.add_column("Subagent", style="bold")
            table.add_column("Status")
            table.add_column("Elapsed", style="dim")
            table.add_column("Prompt", style="dim")

            now = datetime.now(UTC)
            for agent_id, info in bg_infos.items():
                is_running = agent_id in bg_active and not bg_active[agent_id].done()
                if is_running:
                    status_text = Text("running", style="cyan")
                else:
                    status_text = Text("completed", style="green")
                elapsed = now - info.started_at
                elapsed_str = f"{int(elapsed.total_seconds())}s"
                prompt_preview = info.prompt[:60] + "..." if len(info.prompt) > 60 else info.prompt
                name = info.subagent_name
                if info.is_resume:
                    name += " (resume)"
                table.add_row(agent_id, name, status_text, elapsed_str, prompt_preview)

            lines.append(self._renderer.render(table).rstrip())

        # --- Section 3: Background Processes (from Shell ABC) ---
        bg_processes: dict[str, BackgroundProcess] = {}
        try:
            if self._runtime and self._runtime.env and self._runtime.env.shell:
                bg_processes = self._runtime.env.shell.active_background_processes
        except RuntimeError:
            pass

        if bg_processes:
            has_content = True
            if lines:
                lines.append("")
            header = Text("Background Processes", style="bold cyan")
            lines.append(self._renderer.render(header).rstrip())

            table = Table(show_header=True, box=None, padding=(0, 2))
            table.add_column("ID", style="dim")
            table.add_column("Status", style="bold")
            table.add_column("Command")
            table.add_column("PID", style="dim")

            for _proc_id, proc in bg_processes.items():
                elapsed = _get_elapsed_seconds(proc.started_at)
                status_text = Text(f"running ({elapsed:.0f}s)", style="cyan")
                pid_str = str(proc.pid) if proc.pid is not None else "-"
                table.add_row(proc.process_id, status_text, proc.command, pid_str)

            lines.append(self._renderer.render(table).rstrip())

        if not has_content:
            self._append_system_output("No active tasks, subagents, or processes.")
        else:
            self._append_output("\n".join(lines))

    def _dump_history(self, folder_path: str | None) -> None:
        """Dump session state to a folder.

        Creates a folder containing:
        - message_history.json: The conversation history
        - context_state.json: The agent context state (subagent history, etc.)

        Args:
            folder_path: Target folder path. Defaults to ".yaacli-session".
        """
        if not self._message_history:
            self._append_system_output("No conversation history to dump")
            return

        dump_dir = Path(folder_path or ".yaacli-session").expanduser().resolve()
        try:
            # Create folder
            dump_dir.mkdir(parents=True, exist_ok=True)

            # Save message history
            history_file = dump_dir / "message_history.json"
            history_file.write_bytes(ModelMessagesTypeAdapter.dump_json(self._message_history, indent=2))

            # Save context state
            state_file = dump_dir / "context_state.json"
            state = self.runtime.ctx.export_state()
            state_file.write_text(state.model_dump_json(indent=2))

            self._append_system_output(f"Session dumped to {dump_dir}")
            self._append_system_output(f"  - message_history.json ({len(self._message_history)} messages)")
            self._append_system_output("  - context_state.json")
        except Exception as e:
            self._append_system_output(f"Error: {e}")

    def _load_history(self, folder_path: str) -> None:
        """Load session state from a folder.

        Loads from a folder containing:
        - message_history.json: The conversation history
        - context_state.json: The agent context state (optional)

        Args:
            folder_path: Source folder path.
        """
        load_dir = Path(folder_path).expanduser().resolve()

        if not load_dir.is_dir():
            self._append_system_output(f"Not a directory: {load_dir}")
            return

        history_file = load_dir / "message_history.json"
        state_file = load_dir / "context_state.json"

        if not history_file.exists():
            self._append_system_output(f"message_history.json not found in {load_dir}")
            return

        try:
            self._reset_pending_attachments()
            # Load message history
            history_data = history_file.read_bytes()
            history = ModelMessagesTypeAdapter.validate_json(history_data)
            self._message_history = history

            # Load context state if exists
            if state_file.exists():
                state_data = state_file.read_text()
                state = ResumableState.model_validate_json(state_data)
                state.restore(self.runtime.ctx)

                # Re-populate session usage from restored usage ledger entries.
                if self.runtime.ctx.usage_snapshot_entries:
                    snapshot = self.runtime.ctx.build_usage_snapshot()
                    self._session_usage.set_run_snapshot(snapshot)
                    self._session_usage.commit_run_snapshot()
                    # Clear after populating to avoid double counting on next run.
                    self.runtime.ctx.usage_snapshot_entries.clear()

                self._append_system_output(f"Session loaded from {load_dir}")
                self._append_system_output(f"  - message_history.json ({len(history)} messages)")
                self._append_system_output("  - context_state.json (restored)")
            else:
                self._append_system_output(f"Session loaded from {load_dir}")
                self._append_system_output(f"  - message_history.json ({len(history)} messages)")
                self._append_system_output("  - context_state.json (not found, skipped)")

            self._append_system_output("Next message will continue from loaded history.")
        except Exception as e:
            self._append_system_output(f"Error loading session: {e}")

    def _save_session_snapshot(
        self,
        *,
        include_usage_ledger: bool | None = None,
        save_reason: str,
        include_extra_usages: bool | None = None,
    ) -> bool:
        """Persist the current session to disk.

        Args:
            include_usage_ledger: Whether to include the usage ledger in exported state.
                Use True for error recovery snapshots.
            save_reason: Metadata tag describing why the snapshot was saved.
            include_extra_usages: Backward-compatible alias for include_usage_ledger.

        Returns:
            True when a snapshot was written, False when there is no message history.
        """
        if include_usage_ledger is None:
            include_usage_ledger = bool(include_extra_usages)

        if not self._message_history:
            return False

        sessions_dir = self.config_manager.get_sessions_dir()
        save_dir = sessions_dir / self._session_id
        save_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now(UTC).isoformat()

        # Save message history
        history_file = save_dir / "message_history.json"
        history_file.write_bytes(ModelMessagesTypeAdapter.dump_json(self._message_history, indent=2))

        # Save context state
        state_file = save_dir / "context_state.json"
        if include_extra_usages is None:
            state = self.runtime.ctx.export_state(include_usage_ledger=include_usage_ledger)
        else:
            state = self.runtime.ctx.export_state(include_extra_usages=include_extra_usages)
        state_file.write_text(state.model_dump_json(indent=2))

        # Save/update metadata
        metadata_file = save_dir / "metadata.json"
        if metadata_file.exists():
            metadata = json.loads(metadata_file.read_text())
            metadata["updated_at"] = now
        else:
            metadata = {
                "session_id": self._session_id,
                "working_dir": str(self.working_dir),
                "created_at": now,
                "updated_at": now,
            }
        metadata["last_save_reason"] = save_reason
        metadata_file.write_text(json.dumps(metadata, indent=2))

        logger.debug("Saved session snapshot to %s (reason=%s)", save_dir, save_reason)

        # Prune old sessions
        self._prune_sessions(sessions_dir)
        return True

    def _auto_save_history(self) -> None:
        """Auto-save session to session-specific directory after a successful run."""
        self._save_session_snapshot(include_usage_ledger=False, save_reason="success")

    @property
    def session_id(self) -> str:
        """Get the current session ID."""
        return self._session_id

    @property
    def has_session_data(self) -> bool:
        """Check if this session has any saved data."""
        sessions_dir = self.config_manager.get_sessions_dir()
        return (sessions_dir / self._session_id / "message_history.json").exists()

    def _prune_sessions(self, sessions_dir: Path, max_sessions: int = 100) -> None:
        """Remove old sessions beyond the retention limit.

        Keeps the most recent `max_sessions` sessions, deleting the rest.
        Sessions are sorted by updated_at from metadata.json, falling back
        to directory mtime.

        Args:
            sessions_dir: Path to ~/.yaacli/sessions/
            max_sessions: Maximum number of sessions to retain.
        """
        if not sessions_dir.exists():
            return

        session_dirs = [d for d in sessions_dir.iterdir() if d.is_dir()]
        if len(session_dirs) <= max_sessions:
            return

        def _get_session_time(d: Path) -> str:
            metadata_file = d / "metadata.json"
            if metadata_file.exists():
                try:
                    metadata = json.loads(metadata_file.read_text())
                    return metadata.get("updated_at", "")
                except (json.JSONDecodeError, OSError):
                    pass
            # Fallback to directory mtime
            return datetime.fromtimestamp(d.stat().st_mtime, tz=UTC).isoformat()

        # Sort by time ascending (oldest first)
        session_dirs.sort(key=_get_session_time)

        # Remove oldest sessions
        to_remove = session_dirs[: len(session_dirs) - max_sessions]
        for d in to_remove:
            try:
                shutil.rmtree(d)
                logger.debug(f"Pruned old session: {d.name}")
            except OSError as e:
                logger.warning(f"Failed to prune session {d.name}: {e}")

    def _list_sessions(self, max_display: int = 20) -> None:
        """List recent sessions.

        Shows session ID, timestamp, and working directory.

        Args:
            max_display: Maximum number of sessions to show.
        """
        from rich.table import Table

        sessions_dir = self.config_manager.get_sessions_dir()
        if not sessions_dir.exists():
            self._append_system_output("No sessions found.")
            return

        session_dirs = [d for d in sessions_dir.iterdir() if d.is_dir()]
        if not session_dirs:
            self._append_system_output("No sessions found.")
            return

        # Collect session info
        sessions: list[dict[str, str]] = []
        for d in session_dirs:
            metadata_file = d / "metadata.json"
            if metadata_file.exists():
                try:
                    metadata = json.loads(metadata_file.read_text())
                    sessions.append({
                        "id": d.name,
                        "updated_at": metadata.get("updated_at", "unknown"),
                        "working_dir": metadata.get("working_dir", "unknown"),
                    })
                except (json.JSONDecodeError, OSError):
                    sessions.append({"id": d.name, "updated_at": "unknown", "working_dir": "unknown"})
            else:
                sessions.append({"id": d.name, "updated_at": "unknown", "working_dir": "unknown"})

        # Sort by updated_at descending (newest first)
        sessions.sort(key=lambda s: s["updated_at"], reverse=True)

        # Mark current session
        current_id = self._session_id

        table = Table(show_header=True, box=None, padding=(0, 2))
        table.add_column("Session ID", style="cyan")
        table.add_column("Updated", style="dim")
        table.add_column("Working Dir", style="dim")

        for s in sessions[:max_display]:
            sid = s["id"]
            marker = f"{sid} *" if sid == current_id else sid
            # Shorten timestamp for display
            updated = s["updated_at"][:19].replace("T", " ") if s["updated_at"] != "unknown" else "unknown"
            table.add_row(marker, updated, s["working_dir"])

        self._append_system_output(
            f"Sessions ({len(sessions)} total, showing latest {min(len(sessions), max_display)}):"
        )
        self._append_output(self._renderer.render(table).rstrip())
        self._append_system_output("Use /session <id> to restore. (* = current session)")

    def _load_session(self, session_id: str) -> None:
        """Load a session by ID (supports prefix matching).

        Args:
            session_id: Full or prefix of session ID.
        """
        sessions_dir = self.config_manager.get_sessions_dir()
        if not sessions_dir.exists():
            self._append_system_output("No sessions found.")
            return

        # Exact match first
        target = sessions_dir / session_id
        if target.is_dir():
            self._load_history(str(target))
            self._session_id = session_id
            return

        # Prefix match
        matches = [d for d in sessions_dir.iterdir() if d.is_dir() and d.name.startswith(session_id)]
        if len(matches) == 1:
            self._load_history(str(matches[0]))
            self._session_id = matches[0].name
            return
        elif len(matches) > 1:
            self._append_system_output(f"Ambiguous session ID '{session_id}'. Matches:")
            for m in sorted(matches, key=lambda d: d.name):
                self._append_system_output(f"  {m.name}")
        else:
            self._append_system_output(f"Session not found: {session_id}")

    def _append_system_output(self, text: str) -> None:
        """Append system message to output."""
        sys_text = Text()
        sys_text.append("[SYS] ", style="bold yellow")
        sys_text.append(text)
        self._append_output(self._renderer.render(sys_text).rstrip())

    # =========================================================================
    # Main Run Loop
    # =========================================================================

    async def run(self) -> None:
        """Run the TUI application."""
        # Welcome message
        title = Text("YAACLI CLI", style="bold magenta")
        self._append_output(self._renderer.render(title).rstrip())
        self._append_output(f"Model: {self.config.general.model}")
        self._append_output(f"Mode: {self._mode.value.upper()}")
        self._append_output(f"Config dir: {self.config_manager.config_dir}")
        self._append_output("This is the global YAACLI config directory.")
        self._append_output("")  # blank line before help
        self._show_help()

        # Show session ID
        self._append_output(f"Session: {self._session_id}")

        # Create scrollable FormattedTextControl with mouse support
        tui_ref = self

        class ScrollableFormattedTextControl(FormattedTextControl):
            """FormattedTextControl that handles mouse scroll events."""

            def mouse_handler(self, mouse_event: MouseEvent) -> object:
                """Handle mouse scroll events."""
                if mouse_event.event_type == MouseEventType.SCROLL_UP:
                    tui_ref._scroll_offset = max(0, tui_ref._scroll_offset - 3)
                    if tui_ref._app:
                        tui_ref._app.invalidate()
                    return None
                elif mouse_event.event_type == MouseEventType.SCROLL_DOWN:
                    max_scroll = tui_ref._get_max_scroll()
                    tui_ref._scroll_offset = min(tui_ref._scroll_offset + 3, max_scroll)
                    if tui_ref._app:
                        tui_ref._app.invalidate()
                    return None
                return super().mouse_handler(mouse_event)

        # Create output control and window (no ScrollablePane - virtual viewport handles scrolling)
        output_control = ScrollableFormattedTextControl(self._get_output_text)
        output_window = Window(
            content=output_control,
            wrap_lines=False,
        )

        # Steering pane
        steering_control = FormattedTextControl(self._get_steering_text)
        steering_window = Window(
            content=steering_control,
            height=self._get_steering_height,
            style="class:steering-pane",
            wrap_lines=True,
        )

        # Status bar
        status_bar = Window(
            content=FormattedTextControl(self._get_status_text),
            height=2,
            style="class:status-bar",
            wrap_lines=True,
        )

        # Input area with mouse scroll support
        class ScrollableBufferControl(BufferControl):
            """BufferControl that handles mouse scroll events for input area."""

            def mouse_handler(self, mouse_event: MouseEvent) -> object:
                """Handle mouse scroll events to scroll input text."""
                # Get the window that contains this control
                if mouse_event.event_type == MouseEventType.SCROLL_UP:
                    # Move cursor up by 1 line to scroll content
                    buff = self.buffer
                    if buff:
                        doc = buff.document
                        if doc.cursor_position_row > 0:
                            buff.cursor_up()
                        return None
                elif mouse_event.event_type == MouseEventType.SCROLL_DOWN:
                    buff = self.buffer
                    if buff:
                        doc = buff.document
                        if doc.cursor_position_row < doc.line_count - 1:
                            buff.cursor_down()
                        return None
                return super().mouse_handler(mouse_event)

        input_area = TextArea(
            multiline=True,
            prompt=self._get_prompt,
            style="class:input-area",
            focusable=True,
            height=5,
            scrollbar=True,
        )

        # Replace the buffer control with our scrollable version
        original_control = input_area.control
        scrollable_control = ScrollableBufferControl(
            buffer=original_control.buffer,
            input_processors=original_control.input_processors,
            include_default_input_processors=False,
            lexer=original_control.lexer,
            focus_on_click=original_control.focus_on_click,
            key_bindings=self._setup_input_keybindings(input_area),
        )
        input_area.window.content = scrollable_control
        input_area.control = scrollable_control

        # Layout: Output | Steering | Status | Input
        layout = Layout(
            HSplit([
                output_window,
                steering_window,
                status_bar,
                input_area,
            ]),
            focused_element=input_area,
        )

        # Key bindings
        kb = self._setup_keybindings(input_area)

        # Create application
        self._app = Application(
            layout=layout,
            key_bindings=kb,
            style=self._setup_style(),
            full_screen=True,
            mouse_support=True,
        )

        # Override prompt_toolkit's exception handler to prevent "Press ENTER to
        # continue..." messages that flash on screen and corrupt the TUI display.
        #
        # prompt_toolkit's Application._handle_exception is registered as the asyncio
        # event loop exception handler during run_async(). When unhandled asyncio
        # exceptions occur (e.g., from httpx, third-party callbacks, GC'd tasks),
        # it exits full-screen, prints the traceback, waits for Enter, then redraws.
        #
        # We replace this with a handler that logs the error silently and triggers
        # a TUI redraw, so the user experience is uninterrupted.
        original_handle_exception = self._app._handle_exception

        def _quiet_exception_handler(loop: asyncio.AbstractEventLoop, context: dict[str, object]) -> None:
            message = context.get("message", "Unhandled asyncio exception")
            exception = context.get("exception")
            task = context.get("task") or context.get("future")
            handle = context.get("handle")

            details: list[str] = []
            if task is not None:
                details.append(f"task={task!r}")
            if handle is not None:
                details.append(f"handle={handle!r}")
            detail_suffix = f" ({', '.join(details)})" if details else ""

            if isinstance(exception, BaseException):
                if _is_benign_contextvar_cleanup_error(exception):
                    logger.debug(
                        "Suppressed asyncio cleanup error: %s%s", _safe_exception_str(exception), detail_suffix
                    )
                    self._schedule_tui_recovery(loop)
                    return
                logger.error("asyncio: %s%s: %s", message, detail_suffix, exception)
            else:
                logger.error("asyncio: %s%s", message, detail_suffix)
            # Recover on the next loop tick so redraw does not interleave with
            # the current exception handling output.
            self._schedule_tui_recovery(loop)

        self._app._handle_exception = _quiet_exception_handler  # type: ignore[assignment]

        # Run with error handling
        try:
            self._tui_running = True
            await self._app.run_async()
        except Exception as e:
            # Re-raise to be caught by cli.py with proper error display
            raise RuntimeError(f"TUI crashed: {e}") from e
        finally:
            self._tui_running = False
            self._show_shutdown_status("leaving TUI")
            # Restore original prompt_toolkit exception handler
            self._app._handle_exception = original_handle_exception  # type: ignore[assignment]
            # Log performance report on shutdown
            perf_log_report()
            # Ensure agent task and tracked fire-and-forget tasks are fully cancelled
            # and awaited before __aexit__.
            await self._cancel_agent_task()
            await self._cancel_managed_tasks()
