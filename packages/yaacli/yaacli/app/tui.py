"""TUI Application for yaacli.

This module provides the main TUI application with:
- prompt_toolkit based UI with dual-pane layout
- Agent execution with streaming output
- Steering message injection during execution
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
import codecs
import contextlib
import difflib
import json
import os
import re
import signal
import sys
import time
import traceback
import uuid
from collections.abc import Callable, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, cast

from prompt_toolkit import Application
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI, StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import ConditionalContainer, Float, FloatContainer, HSplit, Layout, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.widgets import Box, Frame, TextArea
from pydantic import BaseModel
from pydantic_ai import (
    AgentRunResult,
    BinaryContent,
    DeferredToolRequests,
    DeferredToolResults,
    ModelSettings,
    ToolDenied,
    UsageLimits,
    UserContent,
)
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    OutputToolCallEvent,
    OutputToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models import Model
from pydantic_ai.run import AgentRun
from rich.table import Table
from rich.text import Text
from ya_agent_environment import ShellBackgroundResetError
from ya_agent_sdk.agents.main import AgentRuntime, AgentStreamer, stream_agent
from ya_agent_sdk.agents.models import infer_model
from ya_agent_sdk.context import (
    PROJECT_GUIDANCE_TAG,
    USER_RULES_TAG,
    AgentContext,
    BusMessage,
    MessageBus,
    NoteManager,
    StreamEvent,
    Task,
    TaskManager,
    TaskStatus,
)
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
from ya_agent_sdk.presets import resolve_model_settings
from ya_agent_sdk.toolsets.core.interaction import (
    ASK_USER_QUESTION_KIND,
    AskUserQuestionTool,
    UserQuestion,
    UserQuestionAnswers,
    format_user_question_answers,
    parse_ask_user_question_args,
    parse_user_question_answer,
)
from ya_agent_sdk.toolsets.skills import SHARED_SKILLS_DIR_NAME, SkillToolset
from ya_agent_sdk.utils import get_latest_request_usage

# Import state management from app.state.
from ya_agent_stream_protocol.agui import AguiReplayConfig, is_subagent_event, validate_display_events
from ya_agent_stream_protocol.sdk import AguiAdapterConfig, AguiEventAdapter
from ya_oauth_provider import OAuthRefreshSupervisor, create_oauth_refresh_supervisor_for_models

from yaacli.app.commands import (
    BUILTIN_COMMAND_HELP,
    BUILTIN_COMMANDS,
    BUSY_CONTROL_COMMANDS,
    SlashCommandCompleter,
    format_skill_invocation,
    parse_skill_invocation,
)
from yaacli.app.state import TUIPhase, TUIStateMachine
from yaacli.background import BACKGROUND_MONITOR_KEY, BackgroundMonitor, BackgroundTaskInfo, BackgroundTaskResult
from yaacli.clipboard import ClipboardImageReadResult, read_clipboard_image
from yaacli.config import ConfigManager, YaacliConfig
from yaacli.display import EventRenderer, RichRenderer, ToolMessage
from yaacli.display_replay import MAX_DISPLAY_REPLAY_LOAD_BYTES, BoundedDisplayReplay, load_display_replay
from yaacli.environment import TUIEnvironment
from yaacli.errors import safe_exception_str as _safe_exception_str
from yaacli.events import ContextUpdateEvent, GoalCompleteEvent, GoalCompleteReason, GoalIterationEvent
from yaacli.hooks import emit_context_update
from yaacli.logging import configure_tui_logging, get_logger
from yaacli.model_profiles import (
    ResolvedModelProfile,
    build_model_profiles,
    format_model_profile_choice,
    format_model_profile_label,
    get_startup_model_profile,
    resolve_profile_model_cfg,
    save_selected_model_profile_id,
)
from yaacli.perf import perf_log_report, perf_report, perf_timer
from yaacli.rendering.transcript import BlockId, BoundedTextAccumulator, TranscriptLimits, TranscriptStore
from yaacli.runtime import create_tui_runtime
from yaacli.session import TUIContext, TUIResumableState
from yaacli.sessions import (
    SessionInfo,
    get_head_artifact_paths,
    get_session_info,
    list_sessions,
    read_head_artifacts,
    resolve_session_dir,
    restore_resumable_state_safely,
    save_session_turn,
    trim_sessions,
)
from yaacli.theme import (
    ResolvedTheme,
    ThemePreference,
    fallback_theme,
    prompt_toolkit_style_rules,
    resolve_theme,
)
from yaacli.usage import SessionUsage, TokenUsageBreakdown

YAACLI_AGUI_ADAPTER_CONFIG = AguiAdapterConfig(run_event_prefix="yaacli", stream_metadata_prefix="yaacli")
YAACLI_AGUI_REPLAY_CONFIG = AguiReplayConfig(
    agent_id_field="yaacliAgentId",
    main_agent_id="main",
    drop_subagent_detail_events=True,
)

if TYPE_CHECKING:
    from prompt_toolkit.key_binding import KeyPressEvent
    from ya_agent_environment import BackgroundProcess

logger = get_logger(__name__)

_SHUTDOWN_AGENT_TASK_TIMEOUT = 8.0
_SHUTDOWN_MANAGED_TASKS_TIMEOUT = 5.0
_DIRECT_SHELL_TERMINATE_TIMEOUT = 2.0
_DIRECT_SHELL_TIMEOUT = 300.0
_DIRECT_SHELL_READ_CHUNK_BYTES = 16 * 1024
_DIRECT_SHELL_OUTPUT_TAIL_BYTES = 64 * 1024
_DIRECT_SHELL_LIVE_FRAGMENT_CHARS = 16 * 1024
_DEFAULT_MAX_TURNS_PER_SESSION = 20
_DEFAULT_MAX_SESSIONS = 100
_DEFAULT_MAX_PENDING_ATTACHMENTS = 8
_DEFAULT_MAX_PENDING_ATTACHMENT_BYTES = 20 * 1024 * 1024
_MAX_RETAINED_TOOL_RESULT_CHARS = 64 * 1024
_MAX_RETAINED_TOOL_ARG_CHARS = 64 * 1024
_MAX_DISPLAY_REPLAY_LOAD_BYTES = MAX_DISPLAY_REPLAY_LOAD_BYTES
_SESSION_SELECTOR_MAX_VISIBLE = 8
_SESSION_SELECTOR_MAX_WIDTH = 110
_SESSION_SELECTOR_MIN_WIDTH = 24
_TOOL_RESULT_TRUNCATION_SUFFIX = "\n... [tool result truncated for display]"
_TOOL_ARG_TRUNCATION_SUFFIX = "\n... [tool arguments truncated for display]"


@dataclass
class _BoundedOutputTail:
    """Keep only a fixed-size tail while accounting for all drained bytes."""

    max_bytes: int
    _buffer: bytearray = field(default_factory=bytearray)
    total_bytes: int = 0

    @property
    def truncated(self) -> bool:
        """Whether bytes were discarded before the retained tail."""
        return self.total_bytes > len(self._buffer)

    @property
    def retained_bytes(self) -> int:
        """Return the number of bytes currently retained."""
        return len(self._buffer)

    def append(self, chunk: bytes) -> None:
        """Append a chunk, discarding the oldest bytes past ``max_bytes``."""
        self.total_bytes += len(chunk)
        if self.max_bytes <= 0:
            self._buffer.clear()
            return

        if len(chunk) >= self.max_bytes:
            self._buffer[:] = chunk[-self.max_bytes :]
            return

        excess = len(self._buffer) + len(chunk) - self.max_bytes
        if excess > 0:
            del self._buffer[:excess]
        self._buffer.extend(chunk)

    def text(self) -> str:
        """Decode the bounded tail without retaining the original stream."""
        return self._buffer.decode("utf-8", errors="replace")


async def _drain_direct_shell_stream(
    stream: asyncio.StreamReader | None,
    tail: _BoundedOutputTail,
    on_chunk: Callable[[str], None] | None = None,
) -> None:
    """Drain a subprocess pipe with incremental UTF-8 and line-aware live chunks."""
    if stream is None:
        return

    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    pending = ""

    def emit_ready(*, final: bool) -> None:
        nonlocal pending
        if on_chunk is None:
            pending = ""
            return
        while "\n" in pending:
            newline_index = pending.index("\n") + 1
            on_chunk(pending[:newline_index])
            pending = pending[newline_index:]
        while len(pending) > _DIRECT_SHELL_LIVE_FRAGMENT_CHARS:
            on_chunk(pending[:_DIRECT_SHELL_LIVE_FRAGMENT_CHARS])
            pending = pending[_DIRECT_SHELL_LIVE_FRAGMENT_CHARS:]
        if final and pending:
            on_chunk(pending)
            pending = ""

    while chunk := await stream.read(_DIRECT_SHELL_READ_CHUNK_BYTES):
        tail.append(chunk)
        pending += decoder.decode(chunk, final=False)
        emit_ready(final=False)
    pending += decoder.decode(b"", final=True)
    emit_ready(final=True)


def _format_direct_shell_truncation_note(tail: _BoundedOutputTail, *, stream_name: str) -> str:
    """Return truncation statistics without replaying output that was already rendered live."""
    if not tail.truncated:
        return ""
    return (
        f"... ({stream_name} truncated for diagnostics; retained the last "
        f"{tail.retained_bytes:,} of {tail.total_bytes:,} streamed bytes)"
    )


# =============================================================================
# Utilities
# =============================================================================


def _agui_event_timestamp_seconds(event: dict[str, Any]) -> float | None:
    """Return an AGUI event timestamp as Unix seconds when available."""
    timestamp = event.get("timestamp")
    if timestamp is None or isinstance(timestamp, bool):
        return None
    if isinstance(timestamp, int | float):
        return float(timestamp) / 1000.0
    if not isinstance(timestamp, str):
        return None

    value = timestamp.strip()
    if not value:
        return None
    try:
        return float(value) / 1000.0
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _bounded_tool_result(value: str) -> str:
    """Bound display-only tool state; full results remain in model history."""
    if len(value) <= _MAX_RETAINED_TOOL_RESULT_CHARS:
        return value
    keep = _MAX_RETAINED_TOOL_RESULT_CHARS - len(_TOOL_RESULT_TRUNCATION_SUFFIX)
    return value[: max(0, keep)] + _TOOL_RESULT_TRUNCATION_SUFFIX


def _bounded_tool_args(value: str | dict[str, Any] | None) -> str | dict[str, Any] | None:
    """Bound display-only tool arguments without mutating the source event."""
    if value is None:
        return None
    serialized = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    if len(serialized) <= _MAX_RETAINED_TOOL_ARG_CHARS:
        return value
    keep = _MAX_RETAINED_TOOL_ARG_CHARS - len(_TOOL_ARG_TRUNCATION_SUFFIX)
    return serialized[: max(0, keep)] + _TOOL_ARG_TRUNCATION_SUFFIX


def _positive_int_config(value: object, default: int) -> int:
    return value if isinstance(value, int) and value > 0 else default


def _optional_positive_int_config(value: object) -> int | None:
    return value if isinstance(value, int) and value > 0 else None


def _single_line_session_preview(value: str | None) -> str | None:
    """Normalize untrusted session metadata for one terminal line."""
    if value is None:
        return None
    printable = "".join(character if character.isprintable() else " " for character in value)
    normalized = " ".join(printable.split())
    return normalized or None


def _truncate_display_text(value: str, max_width: int) -> str:
    """Truncate text to a terminal cell width without splitting wide glyphs."""
    if max_width <= 0:
        return ""
    if sum(max(0, get_cwidth(character)) for character in value) <= max_width:
        return value
    suffix = "..."
    if max_width <= len(suffix):
        return suffix[:max_width]
    available = max_width - len(suffix)
    width = 0
    retained: list[str] = []
    for character in value:
        character_width = max(0, get_cwidth(character))
        if width + character_width > available:
            break
        retained.append(character)
        width += character_width
    return f"{''.join(retained).rstrip()}{suffix}"


def _pad_display_text(value: str, width: int) -> str:
    """Truncate and right-pad text to an exact terminal cell width."""
    truncated = _truncate_display_text(value, width)
    display_width = sum(max(0, get_cwidth(character)) for character in truncated)
    return f"{truncated}{' ' * max(0, width - display_width)}"


def _format_session_timestamp(value: str) -> str:
    """Format an ISO session timestamp for the compact selector table."""
    normalized = _single_line_session_preview(value) or "unknown"
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError:
        return _truncate_display_text(normalized.replace("T", " "), 12)
    return parsed.strftime("%b %d %H:%M")


def _session_detail_line(label: str, value: str | None, max_width: int) -> StyleAndTextTuples:
    preview = _single_line_session_preview(value)
    if max_width <= 12:
        compact = preview or "Not available"
        style = "class:session-selector.detail-value" if preview is not None else "class:session-selector.empty"
        return [(style, _truncate_display_text(compact, max_width))]
    label_text = f"{label:<12}"
    value_width = max(1, max_width - len(label_text))
    if preview is None:
        return [
            ("class:session-selector.detail-label", label_text),
            ("class:session-selector.empty", _truncate_display_text("Not available", value_width)),
        ]
    return [
        ("class:session-selector.detail-label", label_text),
        ("class:session-selector.detail-value", _truncate_display_text(preview, value_width)),
    ]


def _completed_result_request_count(result: AgentRunResult[Any]) -> int:
    """Return a conservative model-request count for one completed stream."""
    requests = result.usage.requests
    return requests if isinstance(requests, int) and requests > 0 else 1


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


def _format_elapsed_duration(seconds: float) -> str:
    """Format elapsed seconds as compact seconds, minutes, or hours."""
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {remaining_seconds:02d}s"
    if minutes:
        return f"{minutes}m {remaining_seconds:02d}s"
    return f"{remaining_seconds}s"


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

BACKGROUND_WAKEUP_PROMPT = """<system-reminder>
A background task is ready. Review the background notification and use the relevant tool if more detail is needed.
</system-reminder>"""
USER_INPUT_TIMEOUT_PROMPT = (
    "The user did not respond before the clarification request timed out. "
    "Do not wait for or request the same input again. Continue the task using your best judgment "
    "and make reasonable assumptions where needed."
)


class _UserInputTimeoutError(TimeoutError):
    """Raised when a structured user question is left unanswered."""


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
    placeholder: str = ""


# =============================================================================
# TUI Application
# =============================================================================


@dataclass
class TUIApp:
    """Main TUI application class.

    Manages the lifecycle of:
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
    initial_session_id: str | None = None

    # Runtime state
    _state: TUIState = field(default=TUIState.IDLE, init=False)
    _state_machine: TUIStateMachine = field(default_factory=TUIStateMachine, init=False, repr=False)
    _agent_phase: str = field(default="idle", init=False)  # compatibility for older integrations
    _phase_started_at: float = field(default_factory=time.monotonic, init=False)
    _run_started_at: float | None = field(default=None, init=False)
    _run_timer_paused_at: float | None = field(default=None, init=False)

    # Resources (initialized in __aenter__)
    _exit_stack: AsyncExitStack | None = field(default=None, init=False, repr=False)
    _runtime: AgentRuntime[TUIContext, str | DeferredToolRequests, TUIEnvironment] | None = field(
        default=None, init=False
    )
    _skill_toolset: SkillToolset | None = field(default=None, init=False, repr=False)
    _oauth_refresh_supervisor: OAuthRefreshSupervisor | None = field(default=None, init=False, repr=False)

    # UI components
    _app: Application[None] | None = field(default=None, init=False, repr=False)
    _transcript: TranscriptStore = field(default_factory=TranscriptStore, init=False, repr=False)
    # Compatibility views backed by _transcript. Do not mutate directly.
    _output_lines: list[str] = field(default_factory=list, init=False)
    _max_output_lines: int = field(default=500, init=False)
    _max_output_blocks: int = field(default=1000, init=False)
    _max_output_bytes: int = field(default=4 * 1024 * 1024, init=False)
    _max_stream_render_bytes: int = field(default=512 * 1024, init=False)

    # Virtual viewport rendering (only parse ANSI for visible lines)
    _scroll_offset: int = field(default=0, init=False)  # Display line offset from top
    _follow_latest: bool = field(default=True, init=False)  # Auto-scroll while the viewport is at the bottom
    _block_line_counts: list[int] = field(default_factory=list, init=False)  # Line count per output block
    _total_line_count: int = field(default=0, init=False)  # Sum of all block line counts
    _output_generation: int = field(default=0, init=False)  # Bumped on any content change
    _viewport_cache_key: tuple[int, int, int] | None = field(default=None, init=False)
    _output_ansi_cache: ANSI | None = field(default=None, init=False)  # Cached visible ANSI
    _renderer: RichRenderer = field(default_factory=RichRenderer, init=False)
    _event_renderer: EventRenderer = field(default_factory=EventRenderer, init=False)
    _theme: ResolvedTheme = field(default_factory=lambda: fallback_theme("auto"), init=False)
    _theme_terminal_resolved: bool = field(default=False, init=False)
    _display_replay: BoundedDisplayReplay = field(
        default_factory=lambda: BoundedDisplayReplay(config=YAACLI_AGUI_REPLAY_CONFIG), init=False
    )
    _display_adapter: AguiEventAdapter | None = field(default=None, init=False)

    # Session
    _session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12], init=False)
    _last_session_input: str | None = field(default=None, init=False)
    _last_session_output: str | None = field(default=None, init=False)
    _session_selector_open: bool = field(default=False, init=False)
    _session_selector_entries: list[SessionInfo] = field(default_factory=list, init=False)
    _session_selector_index: int = field(default=0, init=False)

    # Agent execution
    _agent_task: asyncio.Task[None] | None = field(default=None, init=False)
    _foreground_command_task: asyncio.Task[None] | None = field(default=None, init=False)
    _direct_shell_task: asyncio.Task[None] | None = field(default=None, init=False)
    _direct_shell_command: str | None = field(default=None, init=False)
    _managed_tasks: set[asyncio.Task[Any]] = field(default_factory=set, init=False, repr=False)
    _last_run: AgentRun[TUIContext, str | DeferredToolRequests] | None = field(default=None, init=False)
    _message_history: list[Any] | None = field(default=None, init=False)  # Conversation history
    _session_save_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _last_snapshot_saved: bool | None = field(default=None, init=False)

    # Tool tracking
    _tool_messages: dict[str, ToolMessage] = field(default_factory=dict, init=False)
    _printed_tool_calls: set[str] = field(default_factory=set, init=False)

    # Subagent state tracking: agent_id -> {"line_index": int, "tool_names": list[str]}
    _subagent_states: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)

    # Persistent task pane
    _max_visible_tasks: int = field(default=5, init=False)
    _max_visible_completed_tasks: int = field(default=2, init=False)
    _task_pane_expanded: bool = field(default=False, init=False)
    _output_window: Window | None = field(default=None, init=False, repr=False)

    # Input mode: "send" (Enter sends) or "edit" (Enter inserts newline)
    _input_mode: str = field(default="send", init=False)

    # Mouse support mode
    _mouse_enabled: bool = field(default=True, init=False)

    # Double Ctrl+C exit
    _last_ctrl_c_time: float = field(default=0.0, init=False)
    _ctrl_c_exit_timeout: float = field(default=2.0, init=False)

    # Prompt history for up/down navigation
    _prompt_history: list[str] = field(default_factory=list, init=False)
    _max_prompt_history: int = field(default=500, init=False)
    _history_index: int = field(default=-1, init=False)
    _current_input_backup: str = field(default="", init=False)
    _pending_attachments: list[PendingAttachment] = field(default_factory=list, init=False)
    _next_attachment_id: int = field(default=1, init=False)
    _input_area: TextArea | None = field(default=None, init=False, repr=False)

    # Shutdown visibility
    _shutdown_status: str | None = field(default=None, init=False)
    _tui_running: bool = field(default=False, init=False)

    # Streaming text tracking for markdown rendering
    _streaming_text: str = field(default="", init=False)
    _streaming_text_buffer: BoundedTextAccumulator | None = field(default=None, init=False, repr=False)
    _streaming_line_index: int | None = field(default=None, init=False)
    _streaming_block_id: BlockId | None = field(default=None, init=False)

    # Streaming thinking tracking for extended thinking display
    _streaming_thinking: str = field(default="", init=False)
    _streaming_thinking_buffer: BoundedTextAccumulator | None = field(default=None, init=False, repr=False)
    _streaming_thinking_line_index: int | None = field(default=None, init=False)
    _streaming_thinking_block_id: BlockId | None = field(default=None, init=False)

    # Real-time context usage tracking
    _current_context_tokens: int = field(default=0, init=False)
    _context_window_size: int = field(default=200000, init=False)

    # Session-level usage tracking
    _session_usage: SessionUsage = field(default_factory=SessionUsage, init=False)

    # Goal usage tracking
    _goal_usage_start_breakdown: TokenUsageBreakdown | None = field(default=None, init=False)
    _goal_usage_report_pending: bool = field(default=False, init=False)

    # Model profile state
    _active_model_profile: ResolvedModelProfile | None = field(default=None, init=False)
    _model_selector_open: bool = field(default=False, init=False)
    _model_selector_profiles: list[ResolvedModelProfile] = field(default_factory=list, init=False)
    _model_selector_index: int = field(default=0, init=False)

    # UI refresh throttling
    _last_invalidate_time: float = field(default=0.0, init=False)
    _invalidate_interval: float = field(default=1 / 30, init=False)  # Smooth 30fps redraw cadence
    _pending_invalidate_handle: asyncio.TimerHandle | None = field(default=None, init=False, repr=False)

    # Streaming render throttle (separate from UI invalidation)
    _last_stream_render_time: float = field(default=0.0, init=False)
    _stream_render_interval: float = field(default=1 / 30, init=False)  # Typewriter-like Markdown cadence
    _pending_stream_render_handle: asyncio.TimerHandle | None = field(default=None, init=False, repr=False)

    # HITL (Human-in-the-Loop) approval state
    _hitl_pending: bool = field(default=False, init=False)
    _approval_event: asyncio.Event | None = field(default=None, init=False)
    _approval_result: bool | None = field(default=None, init=False)  # True=approve, False=reject
    _approval_reason: str | None = field(default=None, init=False)
    _pending_approvals: list[ToolCallPart] = field(default_factory=list, init=False)
    _current_approval_index: int = field(default=0, init=False)
    _approval_expanded: bool = field(default=False, init=False)
    _approval_kind: str = field(default="approval", init=False)
    _current_deferred_request: ToolCallPart | None = field(default=None, init=False)
    _current_deferred_metadata: dict[str, Any] | None = field(default=None, init=False)

    # Background task completion tracking
    _pending_bus_check_needed: bool = field(default=False, init=False)
    _background_results_ready: bool = field(default=False, init=False)
    _pending_background_wakeup_kinds: set[str] = field(default_factory=set, init=False)
    _session_clear_in_progress: bool = field(default=False, init=False)

    # Deferred screen recovery scheduling
    _screen_recovery_scheduled: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        """Initialize bounded stores from config while tolerating test doubles."""
        display = getattr(self.config, "display", None)
        self._max_output_lines = _positive_int_config(
            getattr(display, "max_output_lines", None), self._max_output_lines
        )
        self._max_output_blocks = _positive_int_config(
            getattr(display, "max_output_blocks", None), self._max_output_blocks
        )
        self._max_output_bytes = _positive_int_config(
            getattr(display, "max_output_bytes", None), self._max_output_bytes
        )
        self._max_stream_render_bytes = _positive_int_config(
            getattr(display, "max_stream_render_bytes", None), self._max_stream_render_bytes
        )
        self._max_prompt_history = _positive_int_config(
            getattr(display, "max_prompt_history", None), self._max_prompt_history
        )
        self._configure_theme(query_terminal=False)
        self._transcript.configure(self._transcript_limits())
        self._output_lines = self._transcript.blocks
        self._block_line_counts = self._transcript.line_counts

    @property
    def state(self) -> TUIState:
        """Current application state."""
        return self._state

    @property
    def phase(self) -> TUIPhase:
        """Return the authoritative interaction phase."""
        return self._state_machine.phase

    def _set_phase(self, phase: TUIPhase) -> None:
        """Transition interaction state and keep the legacy coarse state synchronized."""
        previous = self._state_machine.phase
        if not self._state_machine.transition(phase):
            logger.warning("Rejected invalid TUI phase transition: %s -> %s", previous.name, phase.name)
            return
        if previous != phase:
            self._phase_started_at = time.monotonic()
        self._state = TUIState.IDLE if phase in {TUIPhase.IDLE, TUIPhase.BACKGROUND_RESULT_READY} else TUIState.RUNNING
        if self._app:
            self._app.invalidate()

    def _pause_run_timer(self) -> None:
        """Freeze the foreground run timer while waiting for human input."""
        if self._run_started_at is not None and self._run_timer_paused_at is None:
            self._run_timer_paused_at = time.monotonic()

    def _resume_run_timer(self) -> None:
        """Resume the foreground run timer without counting the paused interval."""
        paused_at = self._run_timer_paused_at
        if paused_at is None:
            return
        if self._run_started_at is not None:
            self._run_started_at += max(0.0, time.monotonic() - paused_at)
        self._run_timer_paused_at = None

    def _is_foreground_busy(self) -> bool:
        """Whether an agent, approval, save, cancellation, or direct shell owns the foreground."""
        return self._state_machine.is_running or self._state == TUIState.RUNNING

    def _is_agent_running(self) -> bool:
        """Whether the current foreground activity belongs to an agent turn."""
        return self._state_machine.is_agent_running or (self._state == TUIState.RUNNING and self.phase == TUIPhase.IDLE)

    def _accepts_steering(self) -> bool:
        """Whether ordinary compose text is sent to the active agent run."""
        return self.phase in {
            TUIPhase.THINKING,
            TUIPhase.TOOL_CALLING,
            TUIPhase.AWAITING_APPROVAL,
            TUIPhase.STREAMING_OUTPUT,
        }

    def _get_configured_model(self) -> str | None:
        """Return a serializable configured model name for session metadata."""
        if isinstance(self.config, YaacliConfig):
            return self.config.general.model
        return None

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
        self._configure_theme(query_terminal=True)
        logger.debug("Resolved terminal theme: %s (%s)", self._theme.variant, self._theme.source)

        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        # Configure stderr-safe logging; verbose mode uses a bounded rotating file.

        # Load MCP config
        mcp_config = self.config_manager.load_mcp_config()

        self._active_model_profile = get_startup_model_profile(self.config, self.config_manager.config_dir)

        # Create runtime and preload the same effective skill catalog used by
        # model instructions so slash completion works before the first turn.
        self._skill_toolset = SkillToolset(toolset_id="skills", extra_dir_names=[SHARED_SKILLS_DIR_NAME])
        self._runtime = create_tui_runtime(
            config=self.config,
            mcp_config=mcp_config,
            working_dir=self.working_dir,
            config_dir=self.config_manager.config_dir,
            model_profile=self._active_model_profile,
            enable_user_input=self.config.tools.enable_user_input,
            skill_toolset=self._skill_toolset,
        )
        await self._exit_stack.enter_async_context(self._runtime)
        await self._skill_toolset.refresh_context(self._runtime.ctx)

        # Register application-level injected context tags for compact stripping
        self._runtime.ctx.injected_context_tags = (
            *self._runtime.ctx.injected_context_tags,
            PROJECT_GUIDANCE_TAG,
            USER_RULES_TAG,
        )

        self._oauth_refresh_supervisor = self._create_oauth_refresh_supervisor()
        if self._oauth_refresh_supervisor is not None:
            await self._oauth_refresh_supervisor.start()
            logger.info(
                "OAuth refresh supervisor started providers=%s", sorted(self._oauth_refresh_supervisor.provider_names)
            )

        # Initialize context window size from model config
        if self._runtime.ctx.model_cfg.context_window:
            self._context_window_size = self._runtime.ctx.model_cfg.context_window

        # Apply display retention config.
        self._max_output_lines = self.config.display.max_output_lines
        self._max_output_blocks = self.config.display.max_output_blocks
        self._max_output_bytes = self.config.display.max_output_bytes
        self._max_stream_render_bytes = self.config.display.max_stream_render_bytes
        self._max_prompt_history = self.config.display.max_prompt_history
        self._transcript.configure(self._transcript_limits())
        self._sync_transcript_state()

        logger.info("TUIApp initialized")
        configure_tui_logging(verbose=self.verbose)

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
        if self._pending_invalidate_handle is not None:
            self._pending_invalidate_handle.cancel()
            self._pending_invalidate_handle = None
        self._cancel_pending_stream_render()

        # Cancel any running agent task and tracked fire-and-forget tasks
        await self._cancel_agent_task()
        await self._cancel_managed_tasks()
        if self._oauth_refresh_supervisor is not None:
            await self._oauth_refresh_supervisor.shutdown()
            self._oauth_refresh_supervisor = None

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
    # Output Management
    # =========================================================================

    def _emit_terminal_bell(self, *, failure_context: str) -> None:
        """Send a terminal BEL without prompt_toolkit's global bell suppression."""
        if self._app is None:
            return

        try:
            # Do not use Output.bell(): prompt_toolkit intentionally suppresses it
            # when PROMPT_TOOLKIT_BELL is false. YAACLI notification settings are
            # authoritative for these user-facing alerts.
            self._app.output.write_raw("\a")
            self._app.output.flush()
        except Exception:
            logger.debug("Failed to emit %s bell", failure_context, exc_info=True)

    def _notify_turn_complete(self) -> None:
        """Emit the configured terminal notification for a successful agent turn."""
        if not self.config.notifications.bell_on_turn_complete:
            return
        self._emit_terminal_bell(failure_context="completion")

    def _notify_user_action_required(self) -> None:
        """Emit the configured terminal notification when HITL input is required."""
        if not self.config.notifications.bell_on_user_action_required:
            return
        self._emit_terminal_bell(failure_context="user-action")

    def _throttled_invalidate(self) -> None:
        """Coalesce redraws while preserving the final trailing update."""
        if not self._app:
            return
        now = time.monotonic()
        elapsed = now - self._last_invalidate_time
        if elapsed >= self._invalidate_interval:
            if self._pending_invalidate_handle is not None:
                self._pending_invalidate_handle.cancel()
                self._pending_invalidate_handle = None
            self._last_invalidate_time = now
            self._app.invalidate()
            return
        if self._pending_invalidate_handle is not None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._app.invalidate()
            return
        self._pending_invalidate_handle = loop.call_later(
            self._invalidate_interval - elapsed,
            self._run_trailing_invalidate,
        )

    def _run_trailing_invalidate(self) -> None:
        self._pending_invalidate_handle = None
        if self._app is not None:
            self._last_invalidate_time = time.monotonic()
            self._app.invalidate()

    def _transcript_limits(self) -> TranscriptLimits:
        return TranscriptLimits(
            max_lines=self._max_output_lines,
            max_blocks=self._max_output_blocks,
            max_bytes=self._max_output_bytes,
        )

    def _sync_transcript_state(self) -> None:
        """Synchronize compatibility counters and live block indices."""
        self._total_line_count = self._transcript.total_lines
        self._streaming_line_index = self._transcript.index_of(self._streaming_block_id)
        self._streaming_thinking_line_index = self._transcript.index_of(self._streaming_thinking_block_id)

    def _invalidate_output_cache(self) -> None:
        """Mark the viewport cache as invalid."""
        self._output_generation += 1

    def _append_block(self, content: str) -> BlockId:
        """Append through the single bounded transcript entry point."""
        self._transcript.configure(self._transcript_limits())
        block_id = self._transcript.append(content)
        self._sync_transcript_state()
        self._output_generation += 1
        return block_id

    def _update_block(self, idx: int, new_content: str) -> None:
        """Compatibility update by retained block index."""
        self._transcript.configure(self._transcript_limits())
        if self._transcript.replace_at(idx, new_content):
            self._sync_transcript_state()
            self._output_generation += 1

    def _update_block_by_id(self, block_id: BlockId | None, new_content: str) -> bool:
        """Replace a live block by stable ID, surviving older-block eviction."""
        if block_id is None:
            return False
        self._transcript.configure(self._transcript_limits())
        updated = self._transcript.replace(block_id, new_content)
        self._sync_transcript_state()
        if updated:
            self._output_generation += 1
        return updated

    def _append_output(self, text: str) -> None:
        """Append text to the bounded transcript and auto-scroll when running."""
        self._append_block(text)

        # Auto-scroll to bottom when the agent is running and the user is following new output.
        if self._is_foreground_busy() and self._follow_latest:
            self._scroll_to_bottom()
        # Invalidate app to refresh display (throttled during streaming)
        self._throttled_invalidate()

    def _record_display_event(self, event: dict[str, Any]) -> None:
        """Persist a display-layer event for replay/session restore."""
        self._display_replay.append(event)

    def _record_display_events(self, events: Sequence[dict[str, Any]]) -> None:
        """Persist display-layer events for replay/session restore."""
        for event in events:
            self._record_display_event(event)

    def _record_display_system_event(self, event_name: str, payload: Any) -> None:
        """Persist a YAACLI custom display event."""
        adapter = self._display_adapter or AguiEventAdapter(
            session_id=self._session_id, run_id=self._session_id, config=YAACLI_AGUI_ADAPTER_CONFIG
        )
        self._handle_and_record_display_events([adapter.build_run_custom_event(event_name, cast(Any, payload))])

    def _handle_and_record_display_events(self, events: Sequence[dict[str, Any]]) -> None:
        self._record_display_events(events)
        if self.phase in {TUIPhase.THINKING, TUIPhase.TOOL_CALLING, TUIPhase.AWAITING_APPROVAL} and any(
            str(event.get("type", "")) in {"TEXT_MESSAGE_START", "TEXT_MESSAGE_CHUNK"}
            and not is_subagent_event(event, agent_id_field="yaacliAgentId")
            for event in events
        ):
            self._set_phase(TUIPhase.STREAMING_OUTPUT)
        self._handle_display_events(events)

    def _reset_output_blocks(self) -> None:
        """Clear rendered output blocks and viewport bookkeeping."""
        self._cancel_pending_stream_render()
        self._transcript.clear()
        self._output_ansi_cache = None
        self._viewport_cache_key = None
        self._output_generation = 0
        self._total_line_count = 0
        self._streaming_block_id = None
        self._streaming_thinking_block_id = None
        self._scroll_offset = 0
        self._follow_latest = True

    def _handle_display_events(self, events: Sequence[dict[str, Any]]) -> None:
        """Render display-layer events into the TUI output buffer."""
        width = self._get_terminal_width()
        for event in events:
            event_type = str(event.get("type", ""))
            if is_subagent_event(event, agent_id_field="yaacliAgentId"):
                if event_type == "TOOL_CALL_CHUNK":
                    tool_name = str(event.get("toolCallName") or event.get("tool_call_name") or "")
                    tool_call_id = str(event.get("toolCallId") or event.get("tool_call_id") or "")
                    agent_id = str(event.get("yaacliAgentId") or event.get("yaacli_agent_id") or "")
                    if agent_id in self._subagent_states and tool_call_id and tool_call_id not in self._tool_messages:
                        self._tool_messages[tool_call_id] = ToolMessage(
                            tool_call_id=tool_call_id,
                            name=tool_name,
                            args=_bounded_tool_args(str(event.get("delta") or "")),
                        )
                        if tool_name:
                            state = self._subagent_states[agent_id]
                            state["tool_names"] = [*state["tool_names"][-2:], tool_name]
                            state["tool_count"] = int(state.get("tool_count", 0)) + 1
                            self._update_subagent_progress_line(agent_id)
                continue
            if event_type == "TEXT_MESSAGE_START":
                self._finalize_streaming_text()
                self._finalize_streaming_thinking()
                self._start_streaming_text("")
                continue
            if event_type == "TEXT_MESSAGE_CHUNK":
                delta = event.get("delta")
                if isinstance(delta, str) and delta:
                    if self._streaming_line_index is not None:
                        self._update_streaming_text(delta)
                    else:
                        self._append_block(
                            self._renderer.render_markdown(
                                delta,
                                code_theme=self._get_code_theme(),
                                width=width,
                            ).rstrip("\n")
                        )
                continue
            if event_type == "TEXT_MESSAGE_END":
                self._finalize_streaming_text()
                continue
            if event_type == "REASONING_MESSAGE_START":
                self._finalize_streaming_text()
                self._finalize_streaming_thinking()
                self._start_streaming_thinking("")
                continue
            if event_type == "REASONING_MESSAGE_CHUNK":
                delta = event.get("delta")
                if isinstance(delta, str) and delta:
                    if self._streaming_thinking_line_index is not None:
                        self._update_streaming_thinking(delta)
                    else:
                        self._append_block(self._event_renderer.render_thinking(delta, width=width).rstrip("\n"))
                continue
            if event_type == "REASONING_MESSAGE_END":
                self._finalize_streaming_thinking()
                continue
            if event_type == "TOOL_CALL_START":
                agent_id = str(event.get("yaacliAgentId") or "main")
                if agent_id != "main":
                    continue
                self._finalize_streaming_text()
                self._finalize_streaming_thinking()
                tool_name = event.get("toolCallName") or event.get("tool_call_name") or "tool"
                tool_call_id = str(event.get("toolCallId") or event.get("tool_call_id") or "")
                if tool_call_id and tool_call_id not in self._tool_messages:
                    self._tool_messages[tool_call_id] = ToolMessage(
                        tool_call_id=tool_call_id,
                        name=str(tool_name),
                    )
                    self._event_renderer.tracker.start_call(
                        tool_call_id,
                        str(tool_name),
                        start_time=_agui_event_timestamp_seconds(event),
                    )
                    self._append_block(
                        self._event_renderer.render_tool_call_start(str(tool_name), tool_call_id).rstrip()
                    )
                continue
            if event_type == "TOOL_CALL_CHUNK":
                agent_id = str(event.get("yaacliAgentId") or "main")
                if agent_id != "main":
                    continue
                self._finalize_streaming_text()
                self._finalize_streaming_thinking()
                tool_name = event.get("toolCallName") or event.get("tool_call_name") or "tool"
                tool_call_id = str(event.get("toolCallId") or event.get("tool_call_id") or "")
                if tool_call_id and tool_call_id not in self._tool_messages:
                    display_args = _bounded_tool_args(str(event.get("delta") or ""))
                    self._tool_messages[tool_call_id] = ToolMessage(
                        tool_call_id=tool_call_id,
                        name=str(tool_name),
                        args=display_args,
                    )
                    self._event_renderer.tracker.start_call(
                        tool_call_id,
                        str(tool_name),
                        display_args,
                        start_time=_agui_event_timestamp_seconds(event),
                    )
                    self._append_block(
                        self._event_renderer.render_tool_call_start(str(tool_name), tool_call_id).rstrip()
                    )
                elif tool_call_id and tool_call_id in self._tool_messages:
                    existing_tool_msg = self._tool_messages[tool_call_id]
                    if not existing_tool_msg.args:
                        existing_tool_msg.args = _bounded_tool_args(str(event.get("delta") or ""))
                    if tool_call_id in self._event_renderer.tracker.tool_calls:
                        self._event_renderer.tracker.tool_calls[tool_call_id].args = existing_tool_msg.args
                continue
            if event_type == "TOOL_CALL_RESULT":
                agent_id = str(event.get("yaacliAgentId") or "main")
                if agent_id != "main":
                    continue
                tool_call_id = str(event.get("toolCallId") or event.get("tool_call_id") or "")
                existing_tool_msg = self._tool_messages.get(tool_call_id)
                tool_msg = existing_tool_msg or ToolMessage(
                    tool_call_id=tool_call_id,
                    name=str(event.get("toolCallName") or event.get("tool_call_name") or "tool"),
                )
                full_content = str(event.get("content") or "")
                tool_msg.content = _bounded_tool_result(full_content)
                self._event_renderer.tracker.complete_call(
                    tool_call_id,
                    tool_msg.content,
                    end_time=_agui_event_timestamp_seconds(event),
                )
                if tool_call_id not in self._printed_tool_calls:
                    duration = 0.0
                    if tool_call_id in self._event_renderer.tracker.tool_calls:
                        duration = self._event_renderer.tracker.tool_calls[tool_call_id].duration()
                    self._append_block(
                        self._event_renderer.render_tool_call_complete(
                            tool_msg,
                            duration=duration,
                            width=width,
                        ).rstrip()
                    )
                    self._printed_tool_calls.add(tool_call_id)
                continue
            if event_type == "CUSTOM" and event.get("name") == "yaacli.user_input":
                value = event.get("value")
                if isinstance(value, dict):
                    text = str(value.get("text") or "")
                    attachments = value.get("attachments")
                    attachment_count = len(attachments) if isinstance(attachments, list) else 0
                    display_text = text
                    if not display_text and attachment_count:
                        noun = "image" if attachment_count == 1 else "images"
                        display_text = f"[Attached {attachment_count} {noun}]"
                    from rich.text import Text as RichText

                    user_text = RichText()
                    user_text.append("> ", style="bold green")
                    user_text.append(display_text)
                    self._append_block(self._renderer.render(user_text, width=width).rstrip("\n"))

        if self._is_foreground_busy() and self._follow_latest:
            self._scroll_to_bottom()
        self._throttled_invalidate()

    def _restore_output_from_display_events(self, events: Sequence[dict[str, Any]]) -> None:
        """Rebuild visible output from compacted display-layer events."""
        self._reset_output_blocks()
        self._streaming_text = ""
        self._streaming_text_buffer = None
        self._streaming_line_index = None
        self._streaming_thinking = ""
        self._streaming_thinking_buffer = None
        self._streaming_thinking_line_index = None
        self._handle_display_events(events)
        self._finalize_streaming_text()
        self._finalize_streaming_thinking()
        self._scroll_to_bottom()
        if self._app:
            self._app.invalidate()

    def _get_viewport_height(self) -> int:
        """Get the actual rendered output height, with a safe pre-render fallback."""
        if self._output_window is not None and self._output_window.render_info is not None:
            return max(3, self._output_window.render_info.window_height)
        if self._app and self._app.output:
            terminal_size = self._app.output.get_size()
            task_rows = self._get_task_height() if self._get_tasks() else 0
            status_rows = self._get_status_height()
            input_rows = 3 if terminal_size.rows < 28 else 5
            return max(3, terminal_size.rows - task_rows - status_rows - input_rows - 1)
        return 40

    def _scroll_to_bottom(self) -> None:
        """Scroll output to bottom and follow subsequent output."""
        visible_height = self._get_viewport_height()
        bottom_padding = 4
        if self._total_line_count > visible_height:
            self._scroll_offset = self._total_line_count - visible_height + bottom_padding
        else:
            self._scroll_offset = 0
        self._follow_latest = True

    def _scroll_output(self, delta: int) -> None:
        """Scroll by *delta* lines and update whether new output should be followed."""
        if delta > 0 and self._follow_latest:
            self._scroll_to_bottom()
            return

        max_scroll = self._get_max_scroll()
        self._scroll_offset = min(max(0, self._scroll_offset + delta), max_scroll)
        if delta < 0:
            self._follow_latest = False
        elif delta > 0 and self._scroll_offset >= max_scroll:
            self._scroll_to_bottom()

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
        """Extract a visible line range from the indexed transcript."""
        return self._transcript.visible_text(start_line, end_line)

    def _get_max_scroll(self) -> int:
        """Calculate maximum scroll position. O(1) using cached line count."""
        return max(0, self._total_line_count - self._get_viewport_height())

    def _get_terminal_width(self) -> int:
        """Get current terminal width for Rich rendering."""
        if self._app and self._app.output:
            return self._app.output.get_size().columns
        return 120

    def _get_terminal_height(self) -> int:
        """Get current terminal height for responsive floating layouts."""
        if self._app and self._app.output:
            return self._app.output.get_size().rows
        return 40

    def _get_code_theme(self) -> str:
        """Get the syntax-highlighting theme for the resolved terminal theme."""
        return self._theme.syntax_theme

    def _configured_theme_preference(self) -> ThemePreference:
        """Read the theme preference while tolerating lightweight test configs."""
        display = getattr(self.config, "display", None)
        preference = getattr(display, "code_theme", "auto")
        if isinstance(preference, str) and preference in {"auto", "dark", "light"}:
            return cast(ThemePreference, preference)
        return "auto"

    def _configure_theme(self, *, query_terminal: bool) -> None:
        """Resolve the color theme and apply it to Rich event rendering."""
        preference = self._configured_theme_preference()
        self._theme = resolve_theme(preference) if query_terminal else fallback_theme(preference)
        display = getattr(self.config, "display", None)
        self._event_renderer.configure_rendering(
            code_theme=self._theme.syntax_theme,
            max_tool_result_lines=_positive_int_config(getattr(display, "max_tool_result_lines", None), 2),
            max_arg_length=_positive_int_config(getattr(display, "max_arg_length", None), 50),
        )
        self._theme_terminal_resolved = query_terminal

    def _new_stream_accumulator(self) -> BoundedTextAccumulator:
        return BoundedTextAccumulator(
            max_bytes=min(self._max_stream_render_bytes, self._max_output_bytes),
            max_lines=self._max_output_lines,
        )

    def _cancel_pending_stream_render(self) -> None:
        """Cancel a coalesced text or reasoning preview render."""
        if self._pending_stream_render_handle is not None:
            self._pending_stream_render_handle.cancel()
            self._pending_stream_render_handle = None

    def _request_stream_render(self, render: Callable[[], None]) -> None:
        """Render now when due, otherwise guarantee one trailing preview frame."""
        now = time.monotonic()
        delay = self._stream_render_interval - (now - self._last_stream_render_time)
        if delay <= 0:
            self._cancel_pending_stream_render()
            render()
            return
        if self._pending_stream_render_handle is not None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._pending_stream_render_handle = loop.call_later(
            delay,
            self._run_trailing_stream_render,
            render,
        )

    def _run_trailing_stream_render(self, render: Callable[[], None]) -> None:
        """Commit the latest coalesced stream state on the event loop."""
        self._pending_stream_render_handle = None
        render()

    def _render_streaming_text_preview(self) -> None:
        """Render every text delta accumulated through the current frame."""
        if self._streaming_text_buffer is None:
            return
        self._streaming_text = self._streaming_text_buffer.text()
        if not self._streaming_text:
            return
        with perf_timer("stream_render_markdown"):
            rendered = self._renderer.render_markdown(
                self._streaming_text,
                code_theme=self._get_code_theme(),
                width=self._get_terminal_width(),
            ).rstrip("\n")
        self._last_stream_render_time = time.monotonic()
        if not self._update_block_by_id(self._streaming_block_id, rendered) and rendered:
            self._streaming_block_id = self._append_block(rendered)
            self._sync_transcript_state()
        if self._is_foreground_busy() and self._follow_latest:
            self._scroll_to_bottom()
        self._throttled_invalidate()

    def _start_streaming_text(self, initial_content: str = "") -> None:
        """Start a bounded streaming text block."""
        self._cancel_pending_stream_render()
        self._streaming_text_buffer = self._new_stream_accumulator()
        self._streaming_text_buffer.append(initial_content)
        self._streaming_text = self._streaming_text_buffer.text()
        self._streaming_block_id = None
        self._streaming_line_index = len(self._output_lines)
        self._last_stream_render_time = 0.0
        if initial_content:
            self._streaming_block_id = self._append_block(initial_content)
            self._sync_transcript_state()

    def _update_streaming_text(self, delta: str) -> None:
        """Append a fragment and schedule a smooth Markdown preview frame."""
        if self._streaming_text_buffer is None:
            self._streaming_text_buffer = self._new_stream_accumulator()
        self._streaming_text_buffer.append(delta)
        self._request_stream_render(self._render_streaming_text_preview)

    def _finalize_streaming_text(self) -> None:
        """Synchronously commit complete text before a part or tool boundary."""
        self._cancel_pending_stream_render()
        if self._streaming_text_buffer is not None and self._streaming_text_buffer.text():
            self._render_streaming_text_preview()
        self._streaming_text = ""
        self._streaming_text_buffer = None
        self._streaming_block_id = None
        self._streaming_line_index = None

    def _render_streaming_thinking_preview(self) -> None:
        """Render every reasoning delta accumulated through the current frame."""
        if self._streaming_thinking_buffer is None:
            return
        self._streaming_thinking = self._streaming_thinking_buffer.text()
        if not self._streaming_thinking:
            return
        rendered = self._event_renderer.render_thinking(
            self._streaming_thinking,
            width=self._get_terminal_width(),
        ).rstrip("\n")
        self._last_stream_render_time = time.monotonic()
        if not self._update_block_by_id(self._streaming_thinking_block_id, rendered) and rendered:
            self._streaming_thinking_block_id = self._append_block(rendered)
            self._sync_transcript_state()
        if self._is_foreground_busy() and self._follow_latest:
            self._scroll_to_bottom()
        self._throttled_invalidate()

    def _start_streaming_thinking(self, initial_content: str = "") -> None:
        """Start a bounded reasoning block."""
        self._cancel_pending_stream_render()
        self._streaming_thinking_buffer = self._new_stream_accumulator()
        self._streaming_thinking_buffer.append(initial_content)
        self._streaming_thinking = self._streaming_thinking_buffer.text()
        self._streaming_thinking_block_id = None
        self._streaming_thinking_line_index = len(self._output_lines)
        self._last_stream_render_time = 0.0
        if initial_content:
            rendered = self._event_renderer.render_thinking(
                initial_content,
                width=self._get_terminal_width(),
            ).rstrip("\n")
            self._streaming_thinking_block_id = self._append_block(rendered)
            self._sync_transcript_state()
            self._throttled_invalidate()

    def _update_streaming_thinking(self, delta: str) -> None:
        """Append reasoning and schedule a smooth lightweight preview frame."""
        if self._streaming_thinking_buffer is None:
            self._streaming_thinking_buffer = self._new_stream_accumulator()
        self._streaming_thinking_buffer.append(delta)
        self._request_stream_render(self._render_streaming_thinking_preview)

    def _finalize_streaming_thinking(self) -> None:
        """Synchronously commit complete reasoning before the next boundary."""
        self._cancel_pending_stream_render()
        if self._streaming_thinking_buffer is not None and self._streaming_thinking_buffer.text():
            self._render_streaming_thinking_preview()
        self._streaming_thinking = ""
        self._streaming_thinking_buffer = None
        self._streaming_thinking_block_id = None
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

    def _format_attachment_placeholder(self, index: int, media_type: str, size_bytes: int) -> str:
        """Format the visible compose-buffer placeholder for a queued image."""
        return f"[Attached image {index}: {media_type} {self._format_size_bytes(size_bytes)}]"

    def _insert_attachment_placeholder(self, input_area: TextArea, placeholder: str) -> None:
        """Insert an attachment placeholder into the current compose buffer."""
        buffer = input_area.buffer
        prefix = "" if not buffer.text or buffer.text.endswith((" ", "\n")) else " "
        suffix = "" if placeholder.endswith(" ") else " "
        buffer.insert_text(f"{prefix}{placeholder}{suffix}")

    def _strip_attachment_placeholders(
        self,
        text: str,
        attachments: Sequence[PendingAttachment],
    ) -> str:
        """Remove generated attachment placeholders from submitted prompt text."""
        cleaned_text = text
        for attachment in attachments:
            if attachment.placeholder:
                cleaned_text = cleaned_text.replace(attachment.placeholder, "")
        return cleaned_text.strip()

    def _format_pending_attachments_label(self) -> str:
        """Format pending attachment status label."""
        if not self._pending_attachments:
            return ""

        count = len(self._pending_attachments)
        total_size = sum(item.size_bytes for item in self._pending_attachments)
        noun = "image" if count == 1 else "images"
        return f"Attach: {count} {noun} ({self._format_size_bytes(total_size)})"

    def _synchronize_pending_attachments(self, compose_text: str) -> None:
        """Drop attachments whose visible compose chips were deleted by the user."""
        self._pending_attachments = [
            attachment
            for attachment in self._pending_attachments
            if not attachment.placeholder or attachment.placeholder in compose_text
        ]

    def _detach_pending_attachment_placeholders(self) -> None:
        """Keep queued binaries after a non-prompt action clears the compose buffer."""
        self._pending_attachments = [
            replace(attachment, placeholder="") if attachment.placeholder else attachment
            for attachment in self._pending_attachments
        ]

    def _consume_pending_attachments(self) -> list[PendingAttachment]:
        """Take the queued clipboard attachments for the next agent turn."""
        attachments = list(self._pending_attachments)
        self._pending_attachments.clear()
        self._next_attachment_id = 1
        return attachments

    def _remove_pending_attachment(self, index: int | None = None) -> PendingAttachment | None:
        """Remove one queued attachment, defaulting to the most recent."""
        if not self._pending_attachments:
            return None
        resolved_index = len(self._pending_attachments) - 1 if index is None else index
        if resolved_index < 0 or resolved_index >= len(self._pending_attachments):
            return None
        attachment = self._pending_attachments.pop(resolved_index)
        if self._input_area is not None and attachment.placeholder:
            text = self._input_area.buffer.text.replace(attachment.placeholder, "")
            self._input_area.buffer.text = " ".join(text.split()) if "\n" not in text else text
            self._input_area.buffer.cursor_position = len(self._input_area.buffer.text)
        return attachment

    def _reset_pending_attachments(self) -> None:
        """Clear queued clipboard attachments and related compose state."""
        self._pending_attachments.clear()
        self._next_attachment_id = 1
        self._history_index = -1
        self._current_input_backup = ""

    def _append_user_input(self, text: str, attachments: Sequence[PendingAttachment] | None = None) -> None:
        """Render user input with styled prompt indicator and attachment markers."""
        attachment_list = list(attachments or [])
        adapter = self._display_adapter or AguiEventAdapter(
            session_id=self._session_id, run_id=self._session_id, config=YAACLI_AGUI_ADAPTER_CONFIG
        )
        self._record_display_event(
            adapter.build_run_custom_event(
                "user_input",
                {
                    "text": text,
                    "attachments": [
                        {"media_type": item.media_type, "size_bytes": item.size_bytes} for item in attachment_list
                    ],
                },
            )
        )
        width = self._get_terminal_width()
        from rich.text import Text as RichText

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

    def _append_error_output(self, e: BaseException, *, saved: bool | None = None) -> None:
        """Render a concise error card; verbose mode includes the traceback."""
        self._record_display_system_event(
            "error",
            {"type": type(e).__name__, "message": _safe_exception_str(e)},
        )
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

        if self.verbose:
            tb_lines = traceback.format_exception(e)
            if tb_lines:
                tb_str = "".join(tb_lines).rstrip()
                for line in tb_str.splitlines():
                    tb_text = RichText()
                    tb_text.append(line, style="dim")
                    self._append_output(self._renderer.render(tb_text, width=width).rstrip("\n"))

        self._append_output("")
        hint = RichText()
        if saved is True:
            hint.append(f"Recovery snapshot saved · /session {self._session_id}", style="dim green")
        elif saved is False:
            hint.append("No recoverable state was available to save", style="dim yellow")
        else:
            hint.append("Run with --verbose for traceback; details are also in yaacli.log", style="dim")
        self._append_output(self._renderer.render(hint, width=width).rstrip("\n"))

    # =========================================================================
    # Task Pane
    # =========================================================================

    def _get_tasks(self) -> list[Task]:
        """Return the current task snapshot for the persistent task pane."""
        if self._runtime is None:
            return []
        return self._runtime.ctx.task_manager.list_all()

    def _visible_tasks(self, tasks: list[Task]) -> list[Task]:
        """Prioritize unfinished work and retain only recent completions."""
        active = [task for task in tasks if task.status == TaskStatus.IN_PROGRESS]
        pending = [task for task in tasks if task.status == TaskStatus.PENDING]
        unfinished = [*active, *pending]
        visible = unfinished[: self._max_visible_tasks]
        remaining = self._max_visible_tasks - len(visible)
        if remaining <= 0:
            return visible

        completed = [task for task in tasks if task.status == TaskStatus.COMPLETED]
        completed_limit = min(remaining, self._max_visible_completed_tasks)
        if completed_limit:
            visible.extend(completed[-completed_limit:])
        return visible

    def _get_task_text(self) -> list[tuple[str, str]]:
        """Render the persistent task list and aggregate task status."""
        tasks = self._get_tasks()
        if not tasks:
            return [("class:task-pane.summary", " Tasks: none")]

        completed = sum(task.status == TaskStatus.COMPLETED for task in tasks)
        active = sum(task.status == TaskStatus.IN_PROGRESS for task in tasks)
        pending = len(tasks) - completed - active
        visible_tasks = self._visible_tasks(tasks)
        hidden = len(tasks) - len(visible_tasks)

        summary = f" Tasks: {completed}/{len(tasks)} done | {active} active | {pending} pending"
        if hidden:
            summary += f" | {hidden} hidden"

        fragments: list[tuple[str, str]] = [
            (
                "class:task-pane.summary",
                summary + " | F2: expand" if not self._task_pane_expanded else summary + " | F2: collapse",
            )
        ]
        if not self._task_pane_expanded:
            return fragments

        incomplete_ids = {task.id for task in tasks if task.status != TaskStatus.COMPLETED}
        for task in visible_tasks:
            fragments.append(("class:task-pane", "\n "))
            active_blockers = [task_id for task_id in task.blocked_by if task_id in incomplete_ids]
            if task.status == TaskStatus.COMPLETED:
                label = "done"
                text = task.subject
                style = "class:task-pane.status-completed"
            elif task.status == TaskStatus.IN_PROGRESS:
                label = "active"
                text = task.active_form or task.subject
                style = "class:task-pane.status-active"
            elif active_blockers:
                label = "blocked"
                text = f"{task.subject} (by #{', #'.join(active_blockers)})"
                style = "class:task-pane.status-blocked"
            else:
                label = "pending"
                text = task.subject
                style = "class:task-pane.status-pending"
            fragments.append((style, f"[{label}] #{task.id} {text}"))

        return fragments

    def _get_task_height(self) -> int:
        """Return a compact one-line summary unless the user explicitly expands it."""
        tasks = self._get_tasks()
        if not tasks:
            return 0
        return 1 + len(self._visible_tasks(tasks)) if self._task_pane_expanded else 1

    # =========================================================================
    # Steering
    # =========================================================================

    def _add_steering_message(self, message: str) -> None:
        """Send additional user guidance while the agent is running."""
        self._send_steering_message(message)

    def _get_pending_steering_count(self) -> int:
        """Count user steering messages not yet consumed by a model request."""
        if self._runtime is None or not isinstance(self._runtime.ctx, TUIContext):
            return 0
        ctx = self._runtime.ctx
        return sum(message.source == "user" for message in ctx.message_bus.peek(ctx.agent_id))

    def _clear_unconsumed_user_steering(self) -> None:
        """Discard unread user guidance without consuming background results."""
        ctx = self.runtime.ctx
        ctx.steering_messages.clear()
        pending_user_ids = {message.id for message in ctx.message_bus.peek(ctx.agent_id) if message.source == "user"}
        if pending_user_ids:
            ctx.message_bus.mark_consumed(ctx.agent_id, pending_user_ids)

    def _send_steering_message(self, message: str) -> None:
        """Send steering guidance to the message bus with TUI formatting."""
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

    def _create_oauth_refresh_supervisor(self) -> OAuthRefreshSupervisor | None:
        if not self.config.oauth_refresh.enabled:
            return None
        models = [profile.model for profile in build_model_profiles(self.config)]
        return create_oauth_refresh_supervisor_for_models(
            models,
            interval_seconds=self.config.oauth_refresh.interval_seconds,
            failure_retry_seconds=self.config.oauth_refresh.failure_retry_seconds,
            refresh_on_startup=self.config.oauth_refresh.refresh_on_startup,
        )

    def _format_active_model_label(self) -> str:
        """Format the active model label for status and welcome output."""
        if self._active_model_profile is not None:
            return format_model_profile_label(self._active_model_profile)
        if self.config.general.model:
            return self.config.general.model
        return "unconfigured"

    # =========================================================================
    # Status Bar
    # =========================================================================

    def _get_status_text(self) -> StyleAndTextTuples:
        """Render priority-ordered status content that wraps on narrow terminals."""
        if self._shutdown_status:
            return [("class:status-bar.warning", " SHUTDOWN "), ("class:status-bar", self._shutdown_status)]

        width = self._get_terminal_width()
        compact = width < 100
        phase_labels = {
            TUIPhase.IDLE: "Ready",
            TUIPhase.THINKING: "Thinking",
            TUIPhase.TOOL_CALLING: "Tools",
            TUIPhase.AWAITING_APPROVAL: "Approval",
            TUIPhase.STREAMING_OUTPUT: "Streaming",
            TUIPhase.SHELL_RUNNING: "Shell",
            TUIPhase.COMMAND_RUNNING: "Command",
            TUIPhase.SAVING: "Saving",
            TUIPhase.CANCELLING: "Cancelling",
            TUIPhase.BACKGROUND_RESULT_READY: "Background ready",
        }
        parts: StyleAndTextTuples = [
            (
                "class:status-bar.warning"
                if self.phase in {TUIPhase.AWAITING_APPROVAL, TUIPhase.CANCELLING}
                else "class:status-bar",
                f" {phase_labels[self.phase]}",
            ),
        ]

        # Alerts precede hints and diagnostics so narrow terminals never hide them.
        if not self._follow_latest:
            parts.append(("class:status-bar.warning", " · HISTORY · Ctrl+L latest"))
        pending_steering = self._get_pending_steering_count()
        if pending_steering:
            parts.append(("class:status-bar.warning", f" · steering {pending_steering} pending"))
        if self._background_results_ready:
            parts.append(("class:status-bar.warning", " · /integrate ready"))
        if self._pending_attachments:
            label = f"attach {len(self._pending_attachments)}" if compact else self._format_pending_attachments_label()
            parts.append(("class:status-bar.warning", f" · {label}"))
        bg_label = self._format_background_label()
        if bg_label:
            parts.append(("class:status-bar.warning", f" · {bg_label}"))

        if self.phase == TUIPhase.AWAITING_APPROVAL:
            progress = f"{self._current_approval_index + 1}/{len(self._pending_approvals)}"
            if self._approval_kind == "call":
                action = "result | /deny | view"
            elif self._approval_kind == "question":
                action = "number(s) | text"
            else:
                action = "y | n | view"
            parts.append(("class:status-bar", f" · {progress} · {action} · Ctrl+C cancel"))
        elif self.phase == TUIPhase.SHELL_RUNNING and self._direct_shell_command:
            command = self._direct_shell_command if not compact else self._direct_shell_command[:24]
            parts.append(("class:status-bar", f" · {command} · Ctrl+C cancel"))
        elif self._accepts_steering():
            parts.append(("class:status-bar", " · Enter steers active run · Ctrl+C cancels"))
        elif self.phase == TUIPhase.COMMAND_RUNNING:
            parts.append(("class:status-bar", " · Slash command active · Ctrl+C cancels"))
        elif self.phase in {TUIPhase.SAVING, TUIPhase.CANCELLING}:
            parts.append(("class:status-bar", " · Please wait for foreground cleanup"))
        else:
            input_hint = "Enter send" if self._input_mode == "send" else "Enter newline"
            parts.append(("class:status-bar", f" · {input_hint} · Tab mode"))

        if not compact:
            parts.append(("class:status-bar", f" · {self._format_active_model_label()}"))
        if self.config.display.show_token_usage:
            context_pct = (
                f"{self._current_context_tokens / self._context_window_size * 100:.0f}%"
                if self._current_context_tokens > 0 and self._context_window_size > 0
                else "--"
            )
            parts.append(("class:status-bar", f" · ctx {context_pct}"))
            parts.append(("class:status-bar", f" · cost {self._session_usage.format_status_cost()}"))
        if self.config.display.show_elapsed_time and self._is_foreground_busy():
            started_at = self._run_started_at if self._run_started_at is not None else self._phase_started_at
            elapsed_at = self._run_timer_paused_at if self._run_timer_paused_at is not None else time.monotonic()
            elapsed = _format_elapsed_duration(elapsed_at - started_at)
            parts.append(("class:status-bar", f" · {elapsed}"))
        return parts

    def _get_status_height(self) -> int:
        """Return cell-aware wrapped rows for the current status fragments."""
        width = max(1, self._get_terminal_width())
        rows = 1
        column = 0
        for fragment in self._get_status_text():
            text = fragment[1]
            for character in text:
                if character == "\n":
                    rows += 1
                    column = 0
                    continue
                cell_width = max(0, get_cwidth(character))
                if cell_width == 0:
                    continue
                if column > 0 and column + cell_width > width:
                    rows += 1
                    column = 0
                if cell_width > width:
                    # A wide glyph is indivisible; account for one occupied row
                    # even when the reported terminal width is pathologically small.
                    column = width
                else:
                    column += cell_width
        return rows

    def _get_prompt(self) -> str:
        """Expose the current compose semantics directly in the prompt."""
        mouse_mode = "scroll" if self._mouse_enabled else "select"
        if self.phase == TUIPhase.AWAITING_APPROVAL:
            if self._approval_kind == "call":
                action = "result"
            elif self._approval_kind == "question":
                action = "answer"
            else:
                action = "approve"
        elif self._accepts_steering():
            action = "steer"
        elif self.phase == TUIPhase.SHELL_RUNNING:
            action = "shell"
        elif self.phase in {TUIPhase.COMMAND_RUNNING, TUIPhase.SAVING, TUIPhase.CANCELLING}:
            action = "wait"
        else:
            action = "send" if self._input_mode == "send" else "edit"
        return f"[{mouse_mode}:{action}] > "

    def _selector_is_open(self) -> bool:
        """Return whether an input-owning floating selector is open."""
        return self._model_selector_open or self._session_selector_open

    async def _show_model_selector(self) -> None:
        """Open the in-TUI model profile selector."""
        current_task = asyncio.current_task()
        owns_command_dispatch = (
            self.phase == TUIPhase.COMMAND_RUNNING
            and current_task is not None
            and self._foreground_command_task is current_task
        )
        if self._is_foreground_busy() and not owns_command_dispatch:
            self._append_system_output("Model selection is available after foreground work finishes.")
            return

        profiles = build_model_profiles(self.config)
        if not profiles:
            self._append_system_output("No model profiles are configured.")
            return

        current_id = self._active_model_profile.id if self._active_model_profile else profiles[0].id
        current_index = next((idx for idx, profile in enumerate(profiles) if profile.id == current_id), 0)
        self._close_session_selector()
        self._model_selector_profiles = profiles
        self._model_selector_index = current_index
        self._model_selector_open = True
        if self._app:
            self._app.invalidate()

    def _close_model_selector(self) -> None:
        """Close the in-TUI model selector."""
        self._model_selector_open = False
        self._model_selector_profiles = []
        self._model_selector_index = 0
        if self._app:
            self._app.invalidate()

    def _move_model_selector(self, delta: int) -> None:
        """Move the active selection in the model selector."""
        if not self._model_selector_open or not self._model_selector_profiles:
            return
        count = len(self._model_selector_profiles)
        self._model_selector_index = (self._model_selector_index + delta) % count
        if self._app:
            self._app.invalidate()

    async def _apply_model_selector_selection(self) -> None:
        """Apply the currently highlighted model profile."""
        if not self._model_selector_open or not self._model_selector_profiles:
            return

        index = max(0, min(self._model_selector_index, len(self._model_selector_profiles) - 1))
        selected_profile = self._model_selector_profiles[index]
        self._close_model_selector()
        await self._switch_model_profile(selected_profile)

    def _get_model_selector_text(self) -> ANSI:
        """Render the in-TUI model selector."""
        if not self._model_selector_open or not self._model_selector_profiles:
            return ANSI("")

        max_visible = 8
        total = len(self._model_selector_profiles)
        start = max(0, min(self._model_selector_index - max_visible // 2, total - max_visible))
        end = min(total, start + max_visible)

        lines = [
            " Select model profile",
            " Up/Down: move | Enter: use | Esc: cancel",
            "",
        ]
        for idx in range(start, end):
            profile = self._model_selector_profiles[idx]
            cursor = ">" if idx == self._model_selector_index else " "
            active = "*" if self._active_model_profile and profile.id == self._active_model_profile.id else " "
            lines.append(f" {cursor} {active} {format_model_profile_choice(profile)}")

        if start > 0:
            lines.insert(3, "     ...")
        if end < total:
            lines.append("     ...")

        return ANSI("\n".join(lines))

    def _get_model_selector_height(self) -> int:
        """Return the model selector window height."""
        if not self._model_selector_open or not self._model_selector_profiles:
            return 1
        max_visible = 8
        total = len(self._model_selector_profiles)
        visible_items = min(total, max_visible)
        overflow_lines = int(total > max_visible and self._model_selector_index >= max_visible // 2)
        overflow_lines += int(total > max_visible and self._model_selector_index < total - max_visible // 2 - 1)
        return visible_items + 3 + overflow_lines

    async def _show_session_selector(self) -> None:
        """Open the metadata-only session selector without blocking the event loop."""
        current_task = asyncio.current_task()
        owns_command_dispatch = (
            self.phase == TUIPhase.COMMAND_RUNNING
            and current_task is not None
            and self._foreground_command_task is current_task
        )
        if self._is_foreground_busy() and not owns_command_dispatch:
            self._append_system_output("Session selection is available after foreground work finishes.")
            return

        try:
            entries = await asyncio.to_thread(list_sessions, self.config_manager)
        except (OSError, ValueError) as exc:
            logger.warning("Failed to list sessions: %s", exc)
            self._append_system_output(f"Unable to list sessions: {exc}")
            return
        if not entries:
            self._append_system_output("No sessions found.")
            return

        self._close_model_selector()
        self._session_selector_entries = entries
        self._session_selector_index = next(
            (index for index, entry in enumerate(entries) if entry.id == self._session_id),
            0,
        )
        self._session_selector_open = True
        if self._app:
            self._app.invalidate()

    def _close_session_selector(self) -> None:
        """Close the in-TUI session selector."""
        self._session_selector_open = False
        self._session_selector_entries = []
        self._session_selector_index = 0
        if self._app:
            self._app.invalidate()

    def _move_session_selector(self, delta: int) -> None:
        """Move the active selection in the session selector."""
        if not self._session_selector_open or not self._session_selector_entries:
            return
        count = len(self._session_selector_entries)
        self._session_selector_index = (self._session_selector_index + delta) % count
        if self._app:
            self._app.invalidate()

    def _apply_session_selector_selection(self) -> None:
        """Load the currently highlighted session through normal command dispatch."""
        if not self._session_selector_open or not self._session_selector_entries:
            return
        index = max(0, min(self._session_selector_index, len(self._session_selector_entries) - 1))
        session_id = self._session_selector_entries[index].id
        self._close_session_selector()
        self._schedule_command(f"/session {session_id}")

    def _get_session_selector_width(self) -> int:
        """Return a centered modal width that stays usable on narrow terminals."""
        terminal_width = self._get_terminal_width()
        if terminal_width <= _SESSION_SELECTOR_MIN_WIDTH:
            return max(12, terminal_width)
        return min(_SESSION_SELECTOR_MAX_WIDTH, max(_SESSION_SELECTOR_MIN_WIDTH, terminal_width - 4))

    def _get_session_selector_title(self) -> StyleAndTextTuples:
        """Return the framed selector title and current result count."""
        return [
            ("class:session-selector.title", "Sessions"),
            ("class:session-selector.count", f" · {len(self._session_selector_entries)}"),
        ]

    def _session_selector_lines(self) -> list[StyleAndTextTuples]:
        """Build the styled session table and selected-session details."""
        if not self._session_selector_open or not self._session_selector_entries:
            return []

        total = len(self._session_selector_entries)
        body_budget = max(1, self._get_terminal_height() - 3)
        if body_budget >= 11:
            show_details = True
            show_scroll_hints = True
            show_shortcuts = True
            show_header = True
            show_top_separator = True
            max_visible = min(_SESSION_SELECTOR_MAX_VISIBLE, max(1, body_budget - 10))
        elif body_budget >= 6:
            show_details = False
            show_scroll_hints = True
            show_shortcuts = True
            show_header = True
            show_top_separator = True
            max_visible = min(_SESSION_SELECTOR_MAX_VISIBLE, max(1, body_budget - 5))
        else:
            show_details = False
            show_scroll_hints = False
            show_shortcuts = body_budget >= 3
            show_header = body_budget >= 2
            show_top_separator = False
            reserved_rows = int(show_shortcuts) + int(show_header)
            max_visible = min(_SESSION_SELECTOR_MAX_VISIBLE, max(1, body_budget - reserved_rows))

        start = max(
            0,
            min(
                self._session_selector_index - max_visible // 2,
                total - max_visible,
            ),
        )
        end = min(total, start + max_visible)
        line_width = max(8, self._get_session_selector_width() - 4)
        show_updated = line_width >= 24
        show_workspace = line_width >= 64
        updated_width = 12
        prefix_width = 4
        if show_workspace:
            session_width = min(24, max(12, line_width // 3))
            workspace_width = max(1, line_width - prefix_width - session_width - updated_width - 2)
        elif show_updated:
            session_width = max(4, line_width - prefix_width - updated_width - 1)
            workspace_width = 0
        else:
            session_width = max(4, line_width - prefix_width)
            workspace_width = 0

        header = f"{'':{prefix_width}}{'SESSION':<{session_width}}"
        if show_updated:
            header += f" {'UPDATED':<{updated_width}}"
        if show_workspace:
            header += f" {'WORKSPACE':<{workspace_width}}"

        if line_width >= 54:
            shortcut_line: StyleAndTextTuples = [
                ("class:session-selector.key", "Up/Down"),
                ("class:session-selector.hint", " navigate   "),
                ("class:session-selector.key", "Enter"),
                ("class:session-selector.hint", " load   "),
                ("class:session-selector.key", "Esc"),
                ("class:session-selector.hint", " close   "),
                ("class:session-selector.current", "*"),
                ("class:session-selector.hint", " current"),
            ]
        elif line_width >= 30:
            shortcut_line = [
                ("class:session-selector.key", "Up/Down"),
                ("class:session-selector.hint", "  "),
                ("class:session-selector.key", "Enter"),
                ("class:session-selector.hint", "  "),
                ("class:session-selector.key", "Esc"),
                ("class:session-selector.hint", "  "),
                ("class:session-selector.current", "*"),
                ("class:session-selector.hint", " current"),
            ]
        else:
            shortcut_line = [("class:session-selector.hint", _truncate_display_text("Up/Down  Enter  Esc", line_width))]

        lines: list[StyleAndTextTuples] = []
        if show_shortcuts:
            lines.append(shortcut_line)
        if show_top_separator:
            lines.append([("class:session-selector.separator", "─" * line_width)])
        if show_header:
            lines.append([("class:session-selector.header", _pad_display_text(header, line_width))])
        if show_scroll_hints and start > 0:
            lines.append([
                ("class:session-selector.scroll", _pad_display_text(f"  ↑ {start} newer sessions", line_width))
            ])

        for index in range(start, end):
            entry = self._session_selector_entries[index]
            cursor = ">" if index == self._session_selector_index else " "
            current = "*" if entry.id == self._session_id else " "
            session_id = _single_line_session_preview(entry.id) or "unknown"
            row = f"{cursor} {current} {_pad_display_text(session_id, session_width)}"
            if show_updated:
                row += f" {_pad_display_text(_format_session_timestamp(entry.updated_at), updated_width)}"
            if show_workspace:
                directory = _single_line_session_preview(entry.working_dir) or "unknown"
                row += f" {_pad_display_text(directory, workspace_width)}"
            if index == self._session_selector_index:
                style = "class:session-selector.selection"
            elif entry.id == self._session_id:
                style = "class:session-selector.current"
            else:
                style = "class:session-selector.row"
            lines.append([(style, _pad_display_text(row, line_width))])

        if show_scroll_hints and end < total:
            lines.append([
                (
                    "class:session-selector.scroll",
                    _pad_display_text(f"  ↓ {total - end} older sessions", line_width),
                )
            ])

        if show_details:
            selected_index = max(0, min(self._session_selector_index, total - 1))
            selected = self._session_selector_entries[selected_index]
            selected_id = _single_line_session_preview(selected.id) or "unknown"
            detail_id = _truncate_display_text(selected_id, max(1, line_width - len("DETAILS  ")))
            lines.extend([
                [("class:session-selector.separator", "─" * line_width)],
                [
                    ("class:session-selector.section", "DETAILS"),
                    ("class:session-selector.detail-id", f"  {detail_id}"),
                ],
                _session_detail_line("Directory", selected.working_dir, line_width),
                _session_detail_line("Last input", selected.input_text, line_width),
                _session_detail_line("Last output", selected.output_text, line_width),
            ])
        return lines

    def _get_session_selector_text(self) -> StyleAndTextTuples:
        """Render the session selector without interpreting preview text as ANSI."""
        lines = self._session_selector_lines()
        fragments: StyleAndTextTuples = []
        for index, line in enumerate(lines):
            fragments.extend(line)
            if index < len(lines) - 1:
                fragments.append(("", "\n"))
        return fragments

    def _get_session_selector_height(self) -> int:
        """Return the exact session selector body height."""
        return max(1, len(self._session_selector_lines()))

    async def _switch_model_profile(self, profile: ResolvedModelProfile, *, persist: bool = True) -> None:
        """Switch the current runtime to a model profile."""
        if self._is_foreground_busy():
            self._append_system_output("Model selection is available after foreground work finishes.")
            return

        model_settings = resolve_model_settings(profile.model_settings)
        model_cfg = resolve_profile_model_cfg(profile.model_cfg)

        model_extra_headers = (
            self.runtime.ctx.get_model_extra_headers() if profile.model.startswith("oauth@codex:") else None
        )
        self.runtime.agent.model = infer_model(profile.model, extra_headers=model_extra_headers)
        self.runtime.agent.model_settings = cast(ModelSettings, model_settings)
        self.runtime.ctx.model_cfg = model_cfg
        self.runtime.ctx.model_profile_instructions = profile.instructions
        self._active_model_profile = profile
        if model_cfg.context_window:
            self._context_window_size = model_cfg.context_window
        else:
            self._context_window_size = 200000

        if persist:
            save_selected_model_profile_id(self.config_manager.config_dir, profile.id)
        self._append_system_output(f"Switched model to {format_model_profile_label(profile)}")

        if self._app:
            self._app.invalidate()

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

        if not attachment_list and not project_guidance and not user_rules:
            return user_input

        if not attachment_list:
            parts: list[UserContent] = [user_input]
            if project_guidance:
                parts.append(project_guidance)
            if user_rules:
                parts.append(user_rules)
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
        """Handle subagent completion or shell output/completion readiness.

        This is called synchronously from the asyncio event loop. If the main
        agent is idle, its queued background notification starts an empty turn
        immediately; otherwise the turn that owns the foreground wakes it when
        it finishes.

        Args:
            agent_id: Background agent ID or shell process ID.
        """
        monitor = self._get_background_monitor()
        if monitor is not None and monitor.is_current_task_discarded():
            self._drain_background_usage(monitor)
            logger.debug("Discarded late background completion after session clear: %s", agent_id)
            return

        if self._session_clear_in_progress:
            logger.debug("Discarding background completion during session clear: %s", agent_id)
            return

        shell_kind = monitor.pending_shell_notification_kind(agent_id) if monitor is not None else None
        result = monitor.task_results.get(agent_id) if monitor is not None else None
        status = result.status if result is not None else "completed"
        if shell_kind == "output":
            self._append_system_output(f"Background shell output ready: {agent_id}")
        elif shell_kind == "completion":
            self._append_system_output(f"Background shell process completed: {agent_id}")
        elif status == "failed":
            detail = f": {result.error}" if result is not None and result.error else ""
            self._append_system_output(f"Background task failed: {agent_id}{detail}")
        elif status == "cancelled":
            self._append_system_output(f"Background task cancelled: {agent_id}")
        else:
            self._append_system_output(f"Background task completed: {agent_id}")

        if shell_kind is not None:
            self._pending_background_wakeup_kinds.add("shell")
        elif monitor is not None and (agent_id in monitor.task_infos or agent_id in monitor.task_results):
            self._pending_background_wakeup_kinds.add("subagent")
        else:
            self._pending_background_wakeup_kinds.add("other")

        self._pending_bus_check_needed = True
        self._background_results_ready = True
        if self._is_foreground_busy():
            logger.debug("Background task %s will wake the main agent after foreground work", agent_id)
            return

        self._wake_main_agent_for_background_results()

    def _build_background_wakeup_prompt(self) -> str:
        """Build a compact reminder from monitor-verified wakeup provenance."""
        kinds = self._pending_background_wakeup_kinds
        if not kinds:
            return BACKGROUND_WAKEUP_PROMPT

        descriptions: list[str] = []
        if "subagent" in kinds:
            descriptions.append(
                "- An asynchronous subagent result is available; use wait_subagent for its retained result if needed."
            )
        if "shell" in kinds:
            descriptions.append(
                "- The monitored shell has unread output or a completed process; use shell_wait if needed."
            )
        if "other" in kinds:
            descriptions.append(
                "- Another background notification is available; review the notification before taking action."
            )

        return "\n".join([
            "<system-reminder>",
            "Background work is ready. Review the notification(s):",
            *descriptions,
            "</system-reminder>",
        ])

    def _wake_main_agent_for_background_results(self) -> None:
        """Start one main-agent turn when a deliverable background event is queued."""
        if self._is_foreground_busy():
            return
        if not self._deliver_background_messages():
            self._background_results_ready = False
            self._pending_bus_check_needed = False
            self._pending_background_wakeup_kinds.clear()
            if self.phase == TUIPhase.BACKGROUND_RESULT_READY:
                self._set_phase(TUIPhase.IDLE)
            logger.debug("Background completion had no deliverable notification")
            return

        self._background_results_ready = True
        logger.info("Background result available, waking main agent")
        # The bus notification is delivered independently of this reminder, so
        # future bus-buffer changes cannot turn the wake-up into a blank turn.
        # _launch_agent does not consume the compose buffer, preserving a draft.
        prompt = self._build_background_wakeup_prompt()
        self._pending_background_wakeup_kinds.clear()
        self._launch_agent(prompt)

    def _on_agent_task_done(self, task: asyncio.Task[None]) -> None:
        """Recover interaction state if the owning task exits outside normal cleanup."""
        if self._agent_task is not task:
            if not task.cancelled():
                task.exception()
            return
        self._agent_task = None
        if task.cancelled():
            self._run_started_at = None
            self._run_timer_paused_at = None
            self._agent_phase = "idle"
            self._reset_hitl_state()
            self._set_phase(TUIPhase.BACKGROUND_RESULT_READY if self._background_results_ready else TUIPhase.IDLE)
            return
        exc = task.exception()
        if exc is not None:
            if _is_benign_contextvar_cleanup_error(exc):
                logger.debug("Suppressed benign ContextVar cleanup error from agent task: %s", _safe_exception_str(exc))
            else:
                logger.error("Uncaught exception in agent task: %s: %s", type(exc).__name__, _safe_exception_str(exc))
                self._append_error_output(exc)
            self._run_started_at = None
            self._run_timer_paused_at = None
            self._agent_phase = "idle"
            self._reset_hitl_state()
            self._set_phase(TUIPhase.BACKGROUND_RESULT_READY if self._background_results_ready else TUIPhase.IDLE)

        if self._background_results_ready:
            self._wake_main_agent_for_background_results()

    def _drain_background_usage(self, monitor: BackgroundMonitor) -> None:
        """Commit queued background usage without delivering conversation messages."""
        drained_snapshots = monitor.drain_usage_snapshots()
        for snapshot in drained_snapshots:
            self._session_usage.set_run_snapshot(snapshot)
            self._session_usage.commit_run_snapshot(snapshot.run_id)
        for run_id in monitor.drain_retired_usage_run_ids():
            self._session_usage.finalize_run_snapshots(run_id)
        for snapshot in drained_snapshots:
            if not monitor.can_publish_late_usage_snapshot(snapshot.run_id):
                self._session_usage.finalize_run_snapshots(snapshot.run_id)

    def _deliver_background_messages(self) -> bool:
        """Redeliver queued background notifications into the current main-agent bus."""
        monitor = self._get_background_monitor()
        ctx = self.runtime.ctx
        delivered = 0
        if monitor is not None:
            self._drain_background_usage(monitor)
            delivered = monitor.deliver_pending_messages(ctx.message_bus, ctx.agent_id)
        # User-authored steering shares the bus but is not a background result.
        # Do not let pending guidance make /integrate claim that work arrived.
        pending_background = any(message.source != "user" for message in ctx.message_bus.peek(ctx.agent_id))
        return delivered > 0 or pending_background

    def _check_pending_bus_messages(self) -> None:
        """Wake the main agent for messages that arrived during a foreground turn.

        Called after agent execution completes to redeliver messages that
        arrived while the main agent was still running.
        """
        # Only proceed if flag was set (background task completed during run)
        if not self._pending_bus_check_needed:
            return
        self._pending_bus_check_needed = False
        if not self._deliver_background_messages():
            self._background_results_ready = False
            self._pending_background_wakeup_kinds.clear()
            if self.phase == TUIPhase.BACKGROUND_RESULT_READY:
                self._set_phase(TUIPhase.IDLE)
            return

        self._background_results_ready = True
        # _run_agent() invokes this method from its finally block, before the
        # owning task has transitioned to done. _launch_agent() correctly
        # rejects a second foreground task at that point, so defer the wakeup
        # to _on_agent_task_done(), after it releases the old task handle.
        if self._agent_task is None or self._agent_task.done():
            self._wake_main_agent_for_background_results()

    def _mark_goal_usage_report_pending(self) -> None:
        """Mark the active goal usage report to be printed after final usage persistence."""
        if self._goal_usage_start_breakdown is not None:
            self._goal_usage_report_pending = True

    def _append_goal_usage_report_if_pending(self) -> None:
        """Append the token delta for the just-finished goal, if pending."""
        if not self._goal_usage_report_pending:
            return

        start_breakdown = self._goal_usage_start_breakdown
        self._goal_usage_start_breakdown = None
        self._goal_usage_report_pending = False

        if start_breakdown is None:
            return

        delta = self._session_usage.token_breakdown.delta_since(start_breakdown)
        self._append_system_output(
            "[Goal] Total tokens used this goal: "
            f"{delta.total_tokens:,} tokens "
            "("
            f"input: {delta.input_tokens:,}, "
            f"cache read: {delta.cache_read_tokens:,}, "
            f"cache write: {delta.cache_write_tokens:,}, "
            f"output: {delta.output_tokens:,}"
            ")"
        )

    def _finish_active_goal(self, reason: GoalCompleteReason) -> None:
        """Finish active goal mode from the TUI layer with an explicit reason."""
        ctx = self.runtime.ctx
        if not isinstance(ctx, TUIContext) or not ctx.goal_active:
            return

        event = GoalCompleteEvent(
            event_id=f"goal-{uuid.uuid4().hex[:8]}",
            iteration=ctx.goal_iteration,
            reason=reason,
            task=ctx.goal_task or "",
        )
        stream_event = StreamEvent(agent_id="main", agent_name="main", event=event)

        if self._display_adapter is not None:
            self._handle_and_record_display_events(self._display_adapter.adapt_stream_event(stream_event))
            self._handle_stream_event(stream_event, render_display=False)
        else:
            self._handle_stream_event(stream_event)

        ctx.reset_goal()

    def _launch_agent(
        self,
        user_input: str,
        attachments: Sequence[PendingAttachment] | None = None,
        *,
        session_input: str | None = None,
    ) -> None:
        """Start an agent task after synchronously claiming foreground ownership."""
        if self._agent_task is not None and not self._agent_task.done():
            self._append_system_output("An agent run already owns the foreground.")
            return
        current_task = asyncio.current_task()
        command_reserved = current_task is not None and self._foreground_command_task is current_task
        if self._is_foreground_busy() and not command_reserved:
            self._append_system_output("Foreground work is already in progress.")
            return
        if self._run_started_at is None:
            self._run_started_at = time.monotonic()
            self._run_timer_paused_at = None
        self._set_phase(TUIPhase.THINKING)
        run = (
            self._run_agent(user_input, attachments)
            if session_input is None
            else self._run_agent(user_input, attachments, session_input=session_input)
        )
        task = asyncio.create_task(run)
        self._agent_task = task
        task.add_done_callback(self._on_agent_task_done)

    async def _run_agent(
        self,
        user_input: str,
        attachments: Sequence[PendingAttachment] | None = None,
        *,
        session_input: str | None = None,
    ) -> None:
        """Execute an agent turn, including all deferred approvals and calls."""
        if self._run_started_at is None:
            self._run_started_at = time.monotonic()
            self._run_timer_paused_at = None
        self._set_phase(TUIPhase.THINKING)
        if self._background_results_ready:
            self._deliver_background_messages()
            self._background_results_ready = False
            self._pending_background_wakeup_kinds.clear()
        turn_attachments = list(attachments or [])
        self._pending_bus_check_needed = False
        self._last_snapshot_saved = None
        self._last_session_input = session_input if session_input is not None else user_input
        self._last_session_output = None
        auto_save_history = bool(getattr(getattr(self.config, "session", None), "auto_save_history", False))
        self._tool_messages.clear()
        self._printed_tool_calls.clear()
        self._subagent_states.clear()
        self._event_renderer.clear()
        cancelled = False
        reported_error = False
        run_finished = False
        run_id = uuid.uuid4().hex[:12]
        self._display_adapter = AguiEventAdapter(
            session_id=self._session_id, run_id=run_id, config=YAACLI_AGUI_ADAPTER_CONFIG
        )
        self._handle_and_record_display_events([self._display_adapter.build_run_started_event()])

        try:
            # Initial agent execution. Every deferred continuation belongs to
            # this same foreground turn and therefore shares one request budget.
            max_model_requests = self.config.general.max_requests
            cumulative_model_requests = 0
            result = await self._execute_stream(
                user_input,
                turn_attachments,
                request_limit=max_model_requests,
            )

            # Resolve every deferred approval and call before reporting completion.
            while result and isinstance(result.output, DeferredToolRequests):
                deferred = result.output
                if not deferred.approvals and not deferred.calls:
                    raise RuntimeError("Agent returned an empty DeferredToolRequests payload.")
                cumulative_model_requests += _completed_result_request_count(result)
                remaining_model_requests = max_model_requests - cumulative_model_requests
                if remaining_model_requests <= 0:
                    raise RuntimeError(
                        "TUI deferred continuation exhausted the cumulative "
                        f"model request limit of {max_model_requests}."
                    )
                user_response = await self._request_user_action(deferred)
                result = await self._execute_stream(
                    user_response,
                    request_limit=remaining_model_requests,
                )

            if result is None or not isinstance(result.output, str):
                raise RuntimeError("Agent completed without a final text result.")
            output = result.output
            self._last_session_output = output
            self._handle_and_record_display_events([
                self._display_adapter.build_run_finished_event(result={"output_text": output})
            ])
            run_finished = True
            self._notify_turn_complete()
            # Steering that did not reach the completed run must not be
            # serialized into a snapshot and replayed by a future turn.
            self._clear_unconsumed_user_steering()
            if auto_save_history:
                try:
                    self._last_snapshot_saved = await self._save_session_snapshot_async(
                        include_usage_ledger=False,
                        save_reason="success",
                    )
                except Exception as save_error:
                    self._last_snapshot_saved = False
                    logger.exception("Agent completed, but session persistence failed")
                    self._append_system_output(
                        "Response completed, but the session snapshot could not be saved: "
                        f"{type(save_error).__name__}: {_safe_exception_str(save_error)}"
                    )

        except asyncio.CancelledError:
            self._finalize_streaming_text()
            self._finalize_streaming_thinking()
            if run_finished:
                # The response already has its terminal event. Persistence
                # cancellation must not reclassify the completed run or start
                # a second "cancelled" snapshot.
                self._last_snapshot_saved = False
                logger.warning("Session persistence was interrupted after the response completed")
                self._append_system_output(
                    "Response completed, but session persistence was interrupted before confirmation."
                )
            else:
                cancelled = True
                if self._display_adapter is not None:
                    self._handle_and_record_display_events([
                        self._display_adapter.build_run_custom_event("run_cancelled", {"reason": "user_interrupted"})
                    ])
                # Keep the partial execution state, but discard steering that
                # the cancelled run never consumed before exporting it.
                self._clear_unconsumed_user_steering()
                if auto_save_history:
                    try:
                        self._last_snapshot_saved = await self._save_session_snapshot_async(
                            include_usage_ledger=True,
                            save_reason="cancelled",
                        )
                    except Exception as save_error:
                        self._last_snapshot_saved = False
                        logger.exception("Cancelled run persistence failed")
                        self._append_system_output(
                            "The run was cancelled, but its partial snapshot could not be saved: "
                            f"{type(save_error).__name__}: {_safe_exception_str(save_error)}"
                        )
                saved_text = "partial state saved" if self._last_snapshot_saved else "no recoverable state"
                self._append_output(f"[Cancelled · {saved_text}]")
        except Exception as e:
            if _is_benign_contextvar_cleanup_error(e):
                logger.debug("Suppressed benign ContextVar cleanup error in agent run: %s", _safe_exception_str(e))
            else:
                reported_error = True
                self._finalize_streaming_text()
                self._finalize_streaming_thinking()
                self._handle_and_record_display_events([
                    self._display_adapter.build_run_error_event(
                        message=_safe_exception_str(e),
                        code=type(e).__name__,
                    )
                ])
                # Failed runs must not persist unconsumed steering into a
                # later restored interaction.
                self._clear_unconsumed_user_steering()
                if auto_save_history:
                    try:
                        self._last_snapshot_saved = await self._save_session_snapshot_async(
                            include_usage_ledger=True,
                            save_reason="error",
                        )
                    except Exception as save_error:
                        self._last_snapshot_saved = False
                        logger.exception("Failed run persistence failed")
                        self._append_system_output(
                            "The run failed, and its recovery snapshot could not be saved: "
                            f"{type(save_error).__name__}: {_safe_exception_str(save_error)}"
                        )
                self._append_error_output(e, saved=self._last_snapshot_saved)
                if self._last_snapshot_saved:
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
            # Clear user steering messages that were not consumed this turn.
            # These are messages injected via bus from user during execution.
            # If not cleared, they would leak into unrelated future tasks.
            self._clear_unconsumed_user_steering()
            # Finish active goal state explicitly. Verified and max-iteration
            # endings are handled by the goal guard; if the guard did not end
            # goal mode, the run stopped without accepted completion.
            ctx = self.runtime.ctx
            if isinstance(ctx, TUIContext) and ctx.goal_active:
                if cancelled:
                    goal_stop_reason = GoalCompleteReason.cancelled
                elif reported_error:
                    goal_stop_reason = GoalCompleteReason.error
                else:
                    goal_stop_reason = GoalCompleteReason.unverified_stop
                self._finish_active_goal(goal_stop_reason)
            self._append_goal_usage_report_if_pending()
            self._agent_phase = "idle"
            self._run_started_at = None
            self._run_timer_paused_at = None
            self._set_phase(TUIPhase.BACKGROUND_RESULT_READY if self._background_results_ready else TUIPhase.IDLE)
            # Surface pending bus messages for the next prompt or an explicit
            # /integrate turn (for example, a task completed while running).
            # After an explicit cancellation, leave them queued instead of
            # redelivering them into the just-cancelled interaction.
            self._display_adapter = None
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
        self._approval_kind = "approval"
        self._current_deferred_request = None
        self._current_deferred_metadata = None
        self._approval_expanded = False
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
        *,
        request_limit: int | None = None,
    ) -> AgentRunResult[str | DeferredToolRequests] | None:
        """Execute a single agent stream and return the result.

        Args:
            prompt: User prompt string or DeferredToolResults from approval.

        Returns:
            AgentRunResult with output (str or DeferredToolRequests).
        """
        # Reset per-stream rendering markers while retaining tool details for /tool.
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
            usage_limits=UsageLimits(
                request_limit=self.config.general.max_requests if request_limit is None else request_limit
            ),
            post_node_hook=emit_context_update,
            resume_on_error=self.config.general.agent_stream_resume_on_error,
            resume_max_attempts=self.config.general.agent_stream_resume_max_attempts,
            transport_resume_max_attempts=self.config.general.agent_stream_transport_resume_max_attempts,
            resume_prompt=self.config.general.agent_stream_resume_prompt,
        ) as stream:
            try:
                async for event in stream:
                    if self._display_adapter is not None:
                        display_events = self._display_adapter.adapt_stream_event(event)
                        self._handle_and_record_display_events(display_events)
                        self._handle_stream_event(event, render_display=False)
                    else:
                        self._handle_stream_event(event)

                stream.raise_if_exception()
            except BaseException:
                self._persist_stream_recoverable_state(stream)
                raise

            self._persist_stream_recoverable_state(stream)
            monitor = self._get_background_monitor()
            if monitor is not None:
                monitor.acknowledge_enqueued_task_results(self.runtime.ctx.message_bus, self.runtime.ctx.agent_id)
            return stream.run.result if stream.run else None

    def _persist_stream_recoverable_state(
        self, stream: AgentStreamer[AgentContext, str | DeferredToolRequests]
    ) -> bool:
        """Persist stream recoverable messages and usage to in-memory session state."""
        self._last_run = stream.run
        if stream.run is None:
            return False
        try:
            message_history = stream.recoverable_messages()
            self._message_history = message_history
            usage = stream.run.usage
            latest_usage = get_latest_request_usage(message_history)
            self._current_context_tokens = latest_usage.total_tokens if latest_usage else usage.total_tokens

            # Accumulate main-agent usage when no realtime snapshot was emitted.
            if not self._session_usage.has_run_snapshot:
                model_id = cast(Model, self.runtime.agent.model).model_name
                self._session_usage.add("main", model_id, usage)
            committed_run_ids = self._session_usage.commit_run_snapshot()
            monitor = self._get_background_monitor()
            for run_id in committed_run_ids:
                if monitor is not None:
                    monitor.observe_usage_run(run_id)
                if monitor is None or not monitor.can_publish_late_usage_snapshot(run_id):
                    self._session_usage.finalize_run_snapshots(run_id)
            if monitor is not None:
                for run_id in monitor.drain_retired_usage_run_ids():
                    self._session_usage.finalize_run_snapshots(run_id)
        except Exception:
            logger.debug("Failed to persist recoverable stream state", exc_info=True)
            return False
        return True

    async def _request_user_action(
        self,
        deferred: DeferredToolRequests,
    ) -> DeferredToolResults:
        """Collect user actions without charging HITL wait time to the run timer."""
        if not deferred.approvals and not deferred.calls:
            return DeferredToolResults()

        self._pause_run_timer()
        try:
            return await self._collect_deferred_user_actions(deferred)
        finally:
            self._resume_run_timer()

    async def _collect_deferred_user_actions(
        self,
        deferred: DeferredToolRequests,
    ) -> DeferredToolResults:
        """Collect explicit decisions/results for every deferred tool request."""
        results = DeferredToolResults()
        requests = [*deferred.approvals, *deferred.calls]
        self._hitl_pending = True
        self._pending_approvals = requests
        self._current_approval_index = 0
        self._set_phase(TUIPhase.AWAITING_APPROVAL)
        self._notify_user_action_required()
        self._append_output("")
        self._append_output(
            f"[User action required: {len(deferred.approvals)} approval(s), {len(deferred.calls)} deferred call(s)]"
        )

        current_index = 0
        for tool_call in deferred.approvals:
            self._approval_kind = "approval"
            self._current_approval_index = current_index
            self._current_deferred_request = tool_call
            self._current_deferred_metadata = deferred.metadata.get(tool_call.tool_call_id)
            self._approval_expanded = False
            self._display_approval_panel(
                tool_call,
                current_index + 1,
                len(requests),
                self._current_deferred_metadata,
            )
            approved, reason = await self._wait_for_approval_input()
            if approved:
                results.approvals[tool_call.tool_call_id] = True
                self._append_output(f"  [Approved: {tool_call.tool_name}]")
            else:
                denial_reason = reason or "User rejected"
                results.approvals[tool_call.tool_call_id] = ToolDenied(message=denial_reason)
                self._append_output(f"  [Rejected: {tool_call.tool_name} - {denial_reason}]")
            current_index += 1

        for tool_call in deferred.calls:
            self._current_approval_index = current_index
            self._current_deferred_request = tool_call
            self._current_deferred_metadata = deferred.metadata.get(tool_call.tool_call_id)
            self._approval_expanded = False

            metadata_kind = (
                self._current_deferred_metadata.get("kind")
                if isinstance(self._current_deferred_metadata, dict)
                else None
            )
            if tool_call.tool_name == AskUserQuestionTool.name or metadata_kind == ASK_USER_QUESTION_KIND:
                self._approval_kind = "question"
                try:
                    answers = await self._collect_user_question_answers(tool_call)
                except _UserInputTimeoutError:
                    results.calls[tool_call.tool_call_id] = RetryPromptPart(
                        content=USER_INPUT_TIMEOUT_PROMPT,
                        tool_name=tool_call.tool_name,
                        tool_call_id=tool_call.tool_call_id,
                    )
                    self._append_output(f"  [Timed out: {tool_call.tool_name}]")
                else:
                    results.calls[tool_call.tool_call_id] = format_user_question_answers(answers)
                    self._append_output(f"  [Answered: {tool_call.tool_name}]")
                current_index += 1
                continue

            self._approval_kind = "call"
            self._display_approval_panel(
                tool_call,
                current_index + 1,
                len(requests),
                self._current_deferred_metadata,
            )
            provided, response = await self._wait_for_approval_input()
            content = response or "User declined to provide a deferred tool result."
            if not provided:
                content = f"Deferred tool call denied by user: {content}"
            results.calls[tool_call.tool_call_id] = RetryPromptPart(
                content=content,
                tool_name=tool_call.tool_name,
                tool_call_id=tool_call.tool_call_id,
            )
            label = "Provided result" if provided else "Denied deferred call"
            self._append_output(f"  [{label}: {tool_call.tool_name}]")
            current_index += 1

        self._hitl_pending = False
        self._pending_approvals.clear()
        self._current_deferred_request = None
        self._current_deferred_metadata = None
        self._set_phase(TUIPhase.TOOL_CALLING)
        self._append_output("")
        return results

    async def _wait_for_approval_input(
        self,
        *,
        timeout_seconds: float | None = None,
    ) -> tuple[bool, str | None]:
        """Wait for an explicit approval, denial, or deferred-call response."""
        event = asyncio.Event()
        self._approval_event = event
        self._approval_result = None
        self._approval_reason = None
        if self._app:
            self._app.invalidate()
        try:
            if timeout_seconds is None:
                await event.wait()
            else:
                try:
                    await asyncio.wait_for(event.wait(), timeout=timeout_seconds)
                except TimeoutError as error:
                    self._approval_result = False
                    self._approval_reason = USER_INPUT_TIMEOUT_PROMPT
                    raise _UserInputTimeoutError from error
            approved = self._approval_result if self._approval_result is not None else False
            return approved, self._approval_reason
        finally:
            if self._approval_event is event:
                self._approval_event = None

    async def _collect_user_question_answers(self, tool_call: ToolCallPart) -> UserQuestionAnswers:
        """Render and collect all questions from one structured deferred call."""
        request = parse_ask_user_question_args(tool_call.args)
        answers: dict[str, str | list[str]] = {}
        total = len(request.questions)

        for index, question in enumerate(request.questions, start=1):
            while True:
                self._display_user_question(question, index=index, total=total)
                provided, response = await self._wait_for_approval_input(
                    timeout_seconds=self.config.tools.user_input_timeout_seconds
                )
                if not provided or not response:
                    continue
                try:
                    answers[question.question] = parse_user_question_answer(question, response)
                    break
                except ValueError as error:
                    self._append_system_output(str(error))

        return UserQuestionAnswers(questions=request.questions, answers=answers)

    def _display_user_question(self, question: UserQuestion, *, index: int, total: int) -> None:
        """Display one structured clarifying question and its options."""
        from rich.console import Group
        from rich.panel import Panel

        content: list[Any] = [
            Text(f"{question.header}: {question.question}", style="bold cyan"),
            Text(""),
        ]
        for option_index, option in enumerate(question.options, start=1):
            line = Text()
            line.append(f"{option_index}. ", style="bold yellow")
            line.append(option.label, style="bold")
            line.append(f" — {option.description}", style="dim")
            content.append(line)

        selection_hint = (
            "numbers separated by commas, or free text" if question.multi_select else "a number, or free text"
        )
        timeout_seconds = float(self.config.tools.user_input_timeout_seconds)
        panel = Panel(
            Group(*content),
            title=f"[yellow]Clarifying Question {index}/{total}[/yellow]",
            subtitle=(f"[dim]Enter {selection_hint} | Timeout: {timeout_seconds:g}s | Ctrl+C: Cancel[/dim]"),
            border_style="yellow",
            padding=(1, 2),
        )
        rendered = self._renderer.render(panel, width=self._get_terminal_width())
        self._append_output(rendered.rstrip())

    def _format_args_for_display(self, args: object, max_str_len: int = 500, max_lines: int = 30) -> str:
        """Format tool arguments for display with smart truncation.

        Args:
            args: Tool arguments (can be dict, JSON string, or any object)
            max_str_len: Maximum length for string values before truncation
            max_lines: Maximum number of lines in output

        Returns:
            Formatted JSON string or fallback representation
        """

        def truncate_strings(obj: object, max_len: int) -> object:
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
            formatted_args = self._format_args_for_display(
                tool_call.args,
                max_str_len=1_000_000 if self._approval_expanded else 500,
                max_lines=10_000 if self._approval_expanded else 30,
            )
            code_theme = self._get_code_theme()
            # Determine if it looks like JSON for syntax highlighting
            is_json_like = formatted_args.strip().startswith(("{", "["))
            syntax = Syntax(formatted_args, "json" if is_json_like else "text", theme=code_theme)
            content_parts.append(syntax)

        if self._approval_kind == "call":
            title = "[yellow]Deferred Tool Result Required[/yellow]"
            subtitle = "[dim]Enter result text | /deny <reason> | view: show full args | Ctrl+C: Cancel[/dim]"
        else:
            title = "[yellow]Tool Approval Required[/yellow]"
            subtitle = "[dim]y: Approve | n or reject <reason>: Reject | view: show full args | Ctrl+C: Cancel[/dim]"
        panel = Panel(
            Group(*content_parts),
            title=title,
            subtitle=subtitle,
            border_style="yellow",
            padding=(1, 2),
        )

        # Render panel to ANSI and append
        rendered = self._renderer.render(panel, width=self._get_terminal_width())
        self._append_output(rendered.rstrip())

    def _show_full_deferred_request(self) -> None:
        """Render the current approval/call without argument truncation."""
        request = self._current_deferred_request
        if request is None:
            return
        self._approval_expanded = True
        self._display_approval_panel(
            request,
            self._current_approval_index + 1,
            len(self._pending_approvals),
            self._current_deferred_metadata,
        )

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

        block_id = self._append_block(rendered.rstrip())
        self._subagent_states[agent_id] = {
            "block_id": block_id,
            "line_index": self._transcript.index_of(block_id),
            "tool_names": [],
            "tool_count": 0,
            "agent_name": agent_name,
        }
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

        # Update by stable ID, with index fallback for legacy/test state.
        block_id = state.get("block_id")
        stable_update_failed = not isinstance(block_id, int) or not self._update_block_by_id(
            BlockId(block_id), rendered.rstrip()
        )
        if stable_update_failed and isinstance(line_index, int) and line_index < len(self._output_lines):
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
        tool_count = state.get("tool_count", len(tool_names))

        # Build progress line
        text = Text()
        text.append(f"[{agent_id}] ", style="cyan")
        text.append("Running... ", style="dim")

        if tool_names:
            # Show last few tools
            recent_tools = tool_names[-3:]  # Last 3 tools
            tools_str = ", ".join(recent_tools)
            if tool_count > 3:
                tools_str = f"...{tools_str}"
            text.append(tools_str, style="dim yellow")
            text.append(f" ({tool_count} tools)", style="dim")

        rendered = self._renderer.render(text, width=self._get_terminal_width())

        # Update by stable ID, with index fallback for legacy/test state.
        block_id = state.get("block_id")
        updated = isinstance(block_id, int) and self._update_block_by_id(BlockId(block_id), rendered.rstrip())
        if not updated and isinstance(line_index, int) and line_index < len(self._output_lines):
            self._update_block(line_index, rendered.rstrip())
            updated = True
        if updated:
            self._throttled_invalidate()

    @staticmethod
    def _is_background_agent(agent_id: str) -> bool:
        """Check if an agent_id belongs to a background subagent."""
        return "-bg-" in agent_id

    def _handle_stream_event(self, event: StreamEvent, *, render_display: bool = True) -> None:
        """Handle non-display state updates from agent execution."""
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
                tool_call_id = message_event.part.tool_call_id
                if tool_call_id not in self._tool_messages:
                    tool_name = message_event.part.tool_name
                    self._tool_messages[tool_call_id] = ToolMessage(
                        tool_call_id=tool_call_id,
                        name=tool_name,
                        args=message_event.part.args,
                    )
                    state = self._subagent_states[agent_id]
                    state["tool_names"] = [*state["tool_names"][-2:], tool_name]
                    state["tool_count"] = int(state.get("tool_count", 0)) + 1
                    self._update_subagent_progress_line(agent_id)
            # Ignore all other subagent events (text streaming, tool results, etc.)
            return

        if isinstance(message_event, UsageSnapshotEvent):
            if message_event.snapshot is not None:
                self._session_usage.set_run_snapshot(message_event.snapshot)
            return

        if not render_display and isinstance(
            message_event,
            PartStartEvent
            | PartDeltaEvent
            | PartEndEvent
            | FunctionToolCallEvent
            | OutputToolCallEvent
            | FunctionToolResultEvent
            | OutputToolResultEvent,
        ):
            # AGUI is the sole owner of display and tool-render state. The raw
            # SDK path continues below only when no display adapter is active.
            return

        # Main agent events - normal processing
        if isinstance(message_event, PartStartEvent) and isinstance(message_event.part, TextPart):
            self._set_phase(TUIPhase.STREAMING_OUTPUT)
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
            self._set_phase(TUIPhase.STREAMING_OUTPUT)
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

        elif isinstance(message_event, FunctionToolCallEvent | OutputToolCallEvent):
            # Finalize any streaming text before tool call
            self._finalize_streaming_text()
            self._finalize_streaming_thinking()

            tool_call_id = message_event.part.tool_call_id
            tool_name = message_event.part.tool_name
            display_args = _bounded_tool_args(message_event.part.args)
            self._tool_messages[tool_call_id] = ToolMessage(
                tool_call_id=tool_call_id,
                name=tool_name,
                args=display_args,
            )
            self._event_renderer.tracker.start_call(tool_call_id, tool_name, display_args)
            rendered = self._event_renderer.render_tool_call_start(tool_name, tool_call_id)
            self._append_output(rendered.rstrip())

        elif isinstance(message_event, FunctionToolResultEvent | OutputToolResultEvent):
            tool_call_id = message_event.tool_call_id
            if tool_call_id in self._tool_messages:
                tool_msg = self._tool_messages[tool_call_id]
                full_content = self._extract_tool_result(message_event)
                result_content = _bounded_tool_result(full_content)
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
            # The task pane reads the current TaskManager snapshot directly.
            # Do not append task snapshots to the transcript.
            pass

        elif isinstance(message_event, NoteEvent):
            rendered = self._event_renderer.render_note_event(message_event)
            self._append_output(rendered.rstrip())

        elif isinstance(message_event, FileChangeEvent):
            rendered = self._event_renderer.render_file_change_event(message_event, width=self._get_terminal_width())
            if rendered:
                self._append_output(rendered.rstrip())

        # Handle TUI-specific events
        elif isinstance(message_event, GoalIterationEvent):
            self._append_system_output(f"[Goal] Iteration {message_event.iteration}/{message_event.max_iterations}")

        elif isinstance(message_event, GoalCompleteEvent):
            if message_event.reason == GoalCompleteReason.verified:
                self._append_system_output(f"[Goal] Task completed in {message_event.iteration} iteration(s)")
            elif message_event.reason == GoalCompleteReason.max_iterations:
                self._append_system_output(
                    f"[Goal] Reached max iterations ({message_event.iteration}). "
                    "Task may be incomplete. You can run /goal again to continue."
                )
            elif message_event.reason == GoalCompleteReason.cancelled:
                self._append_system_output(
                    f"[Goal] Cancelled at iteration {message_event.iteration}. Task may be incomplete."
                )
            elif message_event.reason == GoalCompleteReason.error:
                self._append_system_output(
                    f"[Goal] Stopped after an error at iteration {message_event.iteration}. Task may be incomplete."
                )
            elif message_event.reason == GoalCompleteReason.unverified_stop:
                self._append_system_output(
                    f"[Goal] Stopped without verified completion at iteration {message_event.iteration}. "
                    "Task may be incomplete. You can run /goal again to continue."
                )
            self._mark_goal_usage_report_pending()

        elif isinstance(message_event, ContextUpdateEvent):
            self._current_context_tokens = message_event.total_tokens
            if message_event.context_window_size > 0:
                self._context_window_size = message_event.context_window_size

        # Handle SDK lifecycle events for status bar
        elif isinstance(message_event, ModelRequestStartEvent):
            self._agent_phase = "thinking"
            self._set_phase(TUIPhase.THINKING)

        elif isinstance(message_event, ToolCallsStartEvent):
            self._finalize_streaming_text()
            self._finalize_streaming_thinking()
            self._agent_phase = "tools"
            self._set_phase(TUIPhase.TOOL_CALLING)

        elif isinstance(message_event, MessageReceivedEvent):
            # Render user messages after they have been injected into the run.
            user_messages = [m for m in message_event.messages if m.source == "user"]
            if user_messages:
                previews = [m.content_text for m in user_messages]
                rendered = self._event_renderer.render_steering_injected(previews)
                self._append_output(rendered.rstrip())

        self._throttled_invalidate()

    async def _paste_clipboard_image(self, input_area: TextArea | None = None) -> None:
        """Attach an image from the system clipboard when available."""
        target_input_area = input_area if input_area is not None else self._input_area
        try:
            clipboard_result = await read_clipboard_image()
        except Exception as e:
            logger.exception("Failed to inspect clipboard image during explicit paste")
            clipboard_result = ClipboardImageReadResult(image=None, error=_safe_exception_str(e))

        image = clipboard_result.image
        if image is not None:
            size_bytes = len(image.data)
            media = getattr(self.config, "media", None)
            max_attachments = _positive_int_config(
                getattr(media, "max_pending_attachments", None), _DEFAULT_MAX_PENDING_ATTACHMENTS
            )
            max_attachment_bytes = _positive_int_config(
                getattr(media, "max_pending_attachment_bytes", None), _DEFAULT_MAX_PENDING_ATTACHMENT_BYTES
            )
            queued_bytes = sum(item.size_bytes for item in self._pending_attachments)
            if len(self._pending_attachments) >= max_attachments:
                self._append_system_output(f"Attachment limit reached ({max_attachments} per prompt).")
                return
            if size_bytes > max_attachment_bytes or queued_bytes + size_bytes > max_attachment_bytes:
                self._append_system_output(
                    f"Attachment byte limit exceeded ({self._format_size_bytes(max_attachment_bytes)} per prompt)."
                )
                return
            attachment_id = self._next_attachment_id
            self._next_attachment_id += 1
            placeholder = (
                ""
                if self._is_agent_running() or target_input_area is None
                else self._format_attachment_placeholder(
                    attachment_id,
                    image.media_type,
                    size_bytes,
                )
            )
            attachment = PendingAttachment(
                data=image.data,
                media_type=image.media_type,
                size_bytes=size_bytes,
                placeholder=placeholder,
            )
            self._pending_attachments.append(attachment)
            if target_input_area is not None and placeholder:
                self._insert_attachment_placeholder(target_input_area, placeholder)
            self._history_index = -1
            self._current_input_backup = ""
            if self._is_agent_running():
                self._append_system_output(
                    f"Queued {self._format_attachment_description(attachment)} for the next turn; /remove-image removes it."
                )
            else:
                self._append_system_output(
                    f"Attached {self._format_attachment_description(attachment)} from clipboard; /remove-image removes it."
                )
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
            if len(self._prompt_history) > self._max_prompt_history:
                del self._prompt_history[: len(self._prompt_history) - self._max_prompt_history]

    def _command_starts_agent(self, command: str) -> bool:
        """Return whether a slash command can start a foreground agent turn."""
        parts = command.split(maxsplit=1)
        command_name = parts[0].lower() if parts else ""
        args = parts[1].strip() if len(parts) > 1 else ""
        builtin_name = command_name.removeprefix("/")
        if builtin_name in BUILTIN_COMMANDS:
            return (command_name == "/goal" and bool(args)) or command_name == "/integrate"
        return builtin_name in self.config.get_commands()

    def _is_known_slash_command(self, command_name: str) -> bool:
        """Return whether a slash-prefixed token belongs to the local control plane."""
        if not command_name.startswith("/"):
            return False
        name = command_name.removeprefix("/")
        return name in BUILTIN_COMMANDS or name in self.config.get_commands()

    def _route_busy_control_input(self, text: str, input_area: TextArea) -> bool:
        """Route recognized slash commands and shell syntax before ordinary input.

        Safe commands execute without taking ownership from the current
        foreground task. Commands that require an idle foreground, including
        custom prompt commands, are rejected without clearing the draft. An
        unrecognized slash prefix remains ordinary user input.
        """
        command_name = text.split(maxsplit=1)[0].lower() if text.strip() else ""
        if self._is_known_slash_command(command_name):
            if command_name in BUSY_CONTROL_COMMANDS:
                self._detach_pending_attachment_placeholders()
                input_area.buffer.reset()
                self._add_prompt_history(text)
                self._schedule_command(text)
            else:
                phase_label = self.phase.name.replace("_", " ").lower()
                self._append_system_output(
                    f"Command {command_name} is unavailable while foreground work is {phase_label}. "
                    "Wait for it to finish or use /cancel."
                )
            return True

        if text.startswith("!"):
            phase_label = self.phase.name.replace("_", " ").lower()
            self._append_system_output(
                f"Direct shell commands are unavailable while foreground work is {phase_label}. "
                "Wait for it to finish or use /cancel."
            )
            return True

        return False

    def _release_foreground_command(self, task: asyncio.Future[None]) -> None:
        """Release a command reservation, including pre-start cancellation paths."""
        if self._foreground_command_task is not task:
            return
        self._foreground_command_task = None
        if self._agent_task is None or self._agent_task.done():
            self._run_started_at = None
            self._run_timer_paused_at = None
            if self.phase in {TUIPhase.THINKING, TUIPhase.COMMAND_RUNNING, TUIPhase.CANCELLING}:
                self._set_phase(TUIPhase.BACKGROUND_RESULT_READY if self._background_results_ready else TUIPhase.IDLE)
        if self._background_results_ready and not task.cancelled():
            self._wake_main_agent_for_background_results()

    def _release_direct_shell_task(self, task: asyncio.Future[None]) -> None:
        """Release shell ownership and wake only after the shell task is done."""
        if self._direct_shell_task is not task:
            return
        self._direct_shell_command = None
        self._direct_shell_task = None
        if self.phase in {TUIPhase.SHELL_RUNNING, TUIPhase.CANCELLING}:
            self._set_phase(TUIPhase.BACKGROUND_RESULT_READY if self._background_results_ready else TUIPhase.IDLE)
        if self._background_results_ready and not task.cancelled():
            self._wake_main_agent_for_background_results()

    def _schedule_command(self, command: str) -> None:
        """Schedule a slash command after synchronously reserving idle dispatch."""
        reserve_foreground = not self._is_foreground_busy() and (
            self._foreground_command_task is None or self._foreground_command_task.done()
        )
        starts_agent = reserve_foreground and self._command_starts_agent(command)
        if reserve_foreground:
            if starts_agent:
                self._run_started_at = time.monotonic()
                self._run_timer_paused_at = None
                self._set_phase(TUIPhase.THINKING)
            else:
                self._set_phase(TUIPhase.COMMAND_RUNNING)
        task = asyncio.create_task(self._handle_command(command))
        if reserve_foreground:
            self._foreground_command_task = task
            task.add_done_callback(self._release_foreground_command)
        self._track_managed_task(task)

    def _schedule_skill_or_prompt(self, text: str, attachments: list[PendingAttachment]) -> None:
        """Refresh skill discovery before classifying slash-prefixed user input."""
        self._set_phase(TUIPhase.COMMAND_RUNNING)
        task = asyncio.create_task(self._handle_skill_or_prompt(text, attachments))
        self._foreground_command_task = task
        task.add_done_callback(self._release_foreground_command)
        self._track_managed_task(task)

    async def _handle_skill_or_prompt(self, text: str, attachments: list[PendingAttachment]) -> None:
        """Dispatch a catalog-grounded skill invocation or an ordinary prompt."""
        try:
            if self._skill_toolset is not None and self._runtime is not None:
                try:
                    await self._skill_toolset.refresh_context(self._runtime.ctx)
                except Exception:
                    logger.exception("Failed to refresh skill catalog before slash dispatch")

            available_skills = self._available_skills()
            skill_invocation = parse_skill_invocation(
                text,
                available_skills,
                command_names=self._command_words(),
            )
            if skill_invocation is None:
                self._launch_agent(text, attachments)
                return

            agent_prompt = format_skill_invocation(skill_invocation, available_skills)
            self._launch_agent(agent_prompt, attachments, session_input=text)
        except Exception as error:
            logger.exception("Slash dispatch failed: %s", text)
            self._append_error_output(error)
        finally:
            self._scroll_to_bottom()
            if self._app:
                self._app.invalidate()

    def _submit_input(self, text: str, input_area: TextArea) -> None:
        """Submit input according to the explicit interaction phase."""
        if self._session_clear_in_progress:
            self._append_system_output("Session clear is still in progress. Please retry after it finishes.")
            return

        # Reconcile visible chips before classifying control syntax. A chip the
        # user deleted must remove its binary even when the submitted text is a
        # slash or shell command rather than a normal prompt.
        self._synchronize_pending_attachments(text)
        semantic_text = self._strip_attachment_placeholders(text, self._pending_attachments).strip()
        command_dispatch_pending = (
            self._foreground_command_task is not None and not self._foreground_command_task.done()
        )
        if command_dispatch_pending and not self._is_foreground_busy():
            self._append_system_output("A command is being dispatched. Please wait for it to finish.")
            return

        if self._is_foreground_busy():
            if self._route_busy_control_input(semantic_text, input_area):
                return

            if self._accepts_steering():
                guidance = semantic_text
                self._detach_pending_attachment_placeholders()
                input_area.buffer.reset()
                if guidance:
                    self._add_prompt_history(guidance)
                    self._add_steering_message(guidance)
                    self._append_system_output("Guidance sent to the active run.")
                elif self._pending_attachments:
                    self._append_system_output(
                        "Images remain attached for the next agent turn; binary input cannot steer an active run."
                    )
                return

            phase_label = self.phase.name.replace("_", " ").lower()
            self._append_system_output(f"Foreground work is {phase_label}. Please wait or use /cancel.")
            return

        if semantic_text.startswith("/"):
            self._add_prompt_history(semantic_text)
            input_area.buffer.reset()
            command_name = semantic_text.split(maxsplit=1)[0].lower()
            if self._is_known_slash_command(command_name):
                self._detach_pending_attachment_placeholders()
                self._schedule_command(semantic_text)
            else:
                # Skill directories can change while the TUI is running. Claim
                # the current attachments and foreground ownership now, then
                # refresh before deciding whether this is an explicit skill
                # selection or an ordinary prompt.
                attachments = self._consume_pending_attachments()
                self._append_user_input(semantic_text, attachments)
                self._schedule_skill_or_prompt(semantic_text, attachments)
            return

        if semantic_text.startswith("!"):
            command = semantic_text[1:]
            if not command.strip():
                self._detach_pending_attachment_placeholders()
                input_area.buffer.reset()
                self._append_system_output("Usage: !<command>")
                return
            self._add_prompt_history(semantic_text)
            self._detach_pending_attachment_placeholders()
            input_area.buffer.reset()
            self._direct_shell_command = command
            self._set_phase(TUIPhase.SHELL_RUNNING)
            task = asyncio.create_task(self._execute_shell_command(command))
            self._direct_shell_task = task
            task.add_done_callback(self._release_direct_shell_task)
            self._track_managed_task(task)
            return

        # Only a normal prompt reconciles compose chips and consumes binary
        # attachments. Slash commands, direct shell commands, and active-run
        # steering leave the queue intact unless /remove-image changes it.
        self._synchronize_pending_attachments(text)
        attachments = self._consume_pending_attachments()
        submitted_text = self._strip_attachment_placeholders(text, attachments)
        if not submitted_text and not attachments:
            input_area.buffer.reset()
            return

        self._add_prompt_history(submitted_text)
        input_area.buffer.reset()
        self._append_user_input(submitted_text, attachments)
        self._launch_agent(submitted_text, attachments)

    def _extract_tool_result(self, event: FunctionToolResultEvent | OutputToolResultEvent) -> str:
        """Extract result content from tool result event."""
        try:
            part = event.part
            if hasattr(part, "content"):
                content = part.content
                if isinstance(content, str):
                    return content
                rv = getattr(content, "return_value", None)
                if rv is not None:
                    return str(rv)
                return str(content)
            return str(part)
        except Exception:
            return "<result>"

    # =========================================================================
    # UI Setup
    # =========================================================================

    def _setup_input_keybindings(self, input_area: TextArea) -> KeyBindings:
        """Set up key bindings owned by the focused input control."""
        kb = KeyBindings()

        def previous_history() -> None:
            if self._session_selector_open:
                self._move_session_selector(-1)
                return
            if self._model_selector_open:
                self._move_model_selector(-1)
                return
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
            if self._session_selector_open:
                self._move_session_selector(1)
                return
            if self._model_selector_open:
                self._move_model_selector(1)
                return
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
            """Navigate selector/completion, multiline text, or prompt history."""
            if self._selector_is_open():
                previous_history()
                return
            completion_state = input_area.buffer.complete_state
            if completion_state is not None and completion_state.completions:
                input_area.buffer.complete_previous()
                return
            if input_area.buffer.document.cursor_position_row > 0:
                input_area.buffer.cursor_up()
            else:
                previous_history()

        @kb.add("c-p", eager=True)
        def handle_ctrl_p(event: KeyPressEvent) -> None:
            """Navigate to the previous completion or prompt history item."""
            completion_state = input_area.buffer.complete_state
            if completion_state is not None and completion_state.completions:
                input_area.buffer.complete_previous()
                return
            previous_history()

        @kb.add("down", eager=True)
        def handle_down(event: KeyPressEvent) -> None:
            """Navigate selector/completion, multiline text, or prompt history."""
            document = input_area.buffer.document
            if self._selector_is_open():
                next_history()
                return
            completion_state = input_area.buffer.complete_state
            if completion_state is not None and completion_state.completions:
                input_area.buffer.complete_next()
                return
            if document.cursor_position_row < document.line_count - 1:
                input_area.buffer.cursor_down()
            else:
                next_history()

        @kb.add("c-n", eager=True)
        def handle_ctrl_n(event: KeyPressEvent) -> None:
            """Navigate to the next completion or prompt history item."""
            completion_state = input_area.buffer.complete_state
            if completion_state is not None and completion_state.completions:
                input_area.buffer.complete_next()
                return
            next_history()

        def submit_or_newline() -> None:
            if self._session_selector_open:
                self._apply_session_selector_selection()
                return
            if self._model_selector_open:
                if self._app:
                    self._app.create_background_task(self._apply_model_selector_selection())
                else:
                    self._track_managed_task(asyncio.create_task(self._apply_model_selector_selection()))
                return

            completion_state = input_area.buffer.complete_state
            if completion_state is not None and completion_state.current_completion is not None:
                input_area.buffer.apply_completion(completion_state.current_completion)
                return

            text = input_area.buffer.text.strip()
            self._synchronize_pending_attachments(text)
            semantic_text = self._strip_attachment_placeholders(text, self._pending_attachments).strip()
            if (
                self.phase == TUIPhase.AWAITING_APPROVAL
                and self._hitl_pending
                and self._approval_event
                and not self._approval_event.is_set()
            ):
                lowered = semantic_text.lower()
                command_name = lowered.split(maxsplit=1)[0] if lowered else ""
                if command_name == "/cancel":
                    input_area.buffer.reset()
                    self._cancel_foreground()
                    return
                if self._approval_kind == "call" and (lowered == "/deny" or lowered.startswith("/deny ")):
                    self._approval_result = False
                    self._approval_reason = semantic_text[5:].strip() or "User denied the deferred call"
                    input_area.buffer.reset()
                    self._approval_event.set()
                    return
                if self._route_busy_control_input(semantic_text, input_area):
                    return
                if self._approval_kind == "question":
                    if not semantic_text:
                        self._append_system_output("Enter an option number or a free-text answer.")
                        return
                    self._approval_result = True
                    self._approval_reason = semantic_text
                    input_area.buffer.reset()
                    self._approval_event.set()
                    return
                if lowered in {"view", "v"}:
                    input_area.buffer.reset()
                    self._show_full_deferred_request()
                    return
                if self._approval_kind == "call":
                    if semantic_text:
                        self._approval_result = True
                        self._approval_reason = semantic_text
                    else:
                        self._append_system_output("Enter a result, /deny <reason>, or view.")
                        return
                else:
                    if lowered in {"y", "yes", "approve"}:
                        self._approval_result = True
                        self._approval_reason = None
                    elif lowered in {"n", "no", "reject"} or lowered.startswith("reject "):
                        self._approval_result = False
                        self._approval_reason = semantic_text[6:].strip() or "User rejected"
                    elif semantic_text:
                        input_area.buffer.reset()
                        self._add_prompt_history(semantic_text)
                        self._add_steering_message(semantic_text)
                        self._append_system_output("Guidance sent; approval is still pending.")
                        return
                    else:
                        self._append_system_output("Type y to approve, n/reject <reason> to reject, or view.")
                        return
                input_area.buffer.reset()
                self._approval_event.set()
                return

            if self._input_mode == "send":
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

            if self._session_selector_open:
                self._close_session_selector()
                return
            if self._model_selector_open:
                self._close_model_selector()
                return

            command_active = self._foreground_command_task is not None and not self._foreground_command_task.done()
            if self._is_foreground_busy() or command_active:
                self._cancel_foreground()
                return

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
            """Exit only from a safe, empty idle compose state."""
            if self._is_foreground_busy():
                self._append_system_output("Foreground work is active. Use Ctrl+C to cancel it first.")
                return
            if input_area.buffer.text or self._pending_attachments:
                self._append_system_output("A draft or attachments exist. Clear them before exiting.")
                return
            self._show_shutdown_status("exit requested")
            event.app.exit()

        # Scroll functions
        def _scroll_up(event: KeyPressEvent) -> None:
            """Scroll output up and stop following new output."""
            self._scroll_output(-10)
            if self._app:
                self._app.invalidate()

        def _scroll_down(event: KeyPressEvent) -> None:
            """Scroll output down, following new output again at the bottom."""
            self._scroll_output(10)
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

        @kb.add("f2")
        def handle_toggle_tasks(event: KeyPressEvent) -> None:
            """Expand or collapse the bounded task pane."""
            if self._get_tasks():
                self._task_pane_expanded = not self._task_pane_expanded
                self._scroll_to_bottom()
                if self._app:
                    self._app.invalidate()

        @kb.add("c-x")
        def handle_remove_attachment(event: KeyPressEvent) -> None:
            """Remove the most recently queued attachment."""
            removed = self._remove_pending_attachment()
            if removed is not None:
                self._append_system_output(f"Removed {self._format_attachment_description(removed)}")

        @kb.add("c-u")
        def handle_ctrl_u(event: KeyPressEvent) -> None:
            """Clear input line."""
            input_area.buffer.reset()
            self._history_index = -1

        @kb.add("escape")
        def handle_escape(event: KeyPressEvent) -> None:
            """Close an active selector or toggle mouse support mode."""
            if self._session_selector_open:
                self._close_session_selector()
                return
            if self._model_selector_open:
                self._close_model_selector()
                return

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
                self._app.create_background_task(self._paste_clipboard_image(input_area))
            else:
                self._track_managed_task(asyncio.create_task(self._paste_clipboard_image(input_area)))

        @kb.add("tab")
        def handle_tab(event: KeyPressEvent) -> None:
            """Navigate slash completion, otherwise toggle send/edit mode."""
            if self._selector_is_open():
                return
            buffer = input_area.buffer
            completion_state = buffer.complete_state
            if completion_state is not None and completion_state.completions:
                buffer.complete_next()
                return
            text_before_cursor = buffer.document.text_before_cursor
            session_fragment = text_before_cursor.removeprefix("/session ")
            skill_prefix, skill_separator, skill_fragment = text_before_cursor.rpartition(" ")
            selected_skill_tokens = skill_prefix.split()
            skill_words = set(self._skill_words())
            skill_completion_context = (
                bool(skill_separator)
                and skill_fragment.startswith("/")
                and bool(selected_skill_tokens)
                and all(token in skill_words for token in selected_skill_tokens)
            )
            slash_completion_context = text_before_cursor.startswith("/") and (
                " " not in text_before_cursor
                or (text_before_cursor.startswith("/session ") and " " not in session_fragment)
                or skill_completion_context
            )
            if slash_completion_context:
                buffer.start_completion(select_first=True)
                return
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
        """Set up UI styles for the resolved terminal theme."""
        return Style.from_dict(prompt_toolkit_style_rules(self._theme))

    # =========================================================================
    # Command Handling
    # =========================================================================

    def _command_words(self) -> list[str]:
        """Return built-in and configured command names for help and completion."""
        custom = self.config.get_commands()
        return [f"/{name}" for name in sorted(BUILTIN_COMMANDS | custom.keys())]

    def _available_skills(self) -> dict[str, Any]:
        """Return the current SDK skill catalog when the runtime is ready."""
        if self._runtime is None:
            return {}
        return self._runtime.ctx.available_skills

    def _skill_words(self) -> list[str]:
        """Return slash-safe names from the effective SDK skill catalog."""
        return [
            f"/{name}" for name in sorted(self._available_skills()) if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name)
        ]

    def _session_completion_ids(self) -> list[str]:
        """Return current session IDs for contextual /session completion."""
        try:
            return [info.id for info in list_sessions(self.config_manager)]
        except (OSError, ValueError):
            logger.debug("Failed to list sessions for completion", exc_info=True)
            return []

    def _clear_transcript(self) -> None:
        """Clear only rendered output, preserving conversation and runtime state."""
        self._reset_output_blocks()
        self._scroll_offset = 0
        self._append_system_output("Transcript cleared. Conversation context is unchanged; use /new to reset it.")

    async def _start_new_session(self) -> None:
        """Reset conversation state and atomically roll forward session identity."""
        if self._session_clear_in_progress:
            self._append_system_output("A session switch is already in progress.")
            return

        old_session_id = self._session_id
        new_session_id = uuid.uuid4().hex[:12]
        # Publish the new identity before the first cleanup await. Once the old
        # conversation is tombstoned, cancellation must never pair a fresh
        # context with the old durable session ID.
        self._session_id = new_session_id
        cancelled: asyncio.CancelledError | None = None
        try:
            await self._clear_session()
        except ShellBackgroundResetError as exc:
            # Process ownership is unresolved, so keep the durable identity and
            # conversation context that were active before /new.
            self._session_id = old_session_id
            self._set_phase(TUIPhase.IDLE)
            self._append_system_output(f"New session was not started: {_safe_exception_str(exc)}")
            return
        except asyncio.CancelledError as exc:
            cancelled = exc

        self._display_adapter = None
        self._set_phase(TUIPhase.IDLE)
        self._append_system_output(f"New session {new_session_id} started (previous: {old_session_id}).")
        if cancelled is not None:
            raise cancelled

    def _cancel_foreground(self) -> None:
        """Request cancellation once without interrupting persistence cleanup."""
        if self.phase == TUIPhase.SAVING:
            self._append_system_output("A session snapshot is being saved; waiting for persistence to finish.")
            return

        current_task = asyncio.current_task()
        candidates = (
            (self._direct_shell_task, "shell command"),
            (self._agent_task, "agent run"),
            (self._foreground_command_task, "command dispatch"),
        )
        for task, label in candidates:
            if task is None or task is current_task or task.done():
                continue
            if task.cancelling():
                self._append_system_output("Cancellation is already in progress.")
                return
            self._set_phase(TUIPhase.CANCELLING)
            task.cancel()
            self._append_system_output(f"Cancelling {label}...")
            return
        self._append_system_output("Nothing is running.")

    def _integrate_background_results(self) -> None:
        """Integrate ready background messages into this or a fresh agent turn."""
        current_task = asyncio.current_task()
        command_reserved = current_task is not None and self._foreground_command_task is current_task
        if self._is_foreground_busy() and not command_reserved:
            if not self._accepts_steering():
                self._append_system_output("Background results remain queued until foreground work finishes.")
                return
            if not self._deliver_background_messages():
                self._background_results_ready = False
                self._pending_bus_check_needed = False
                self._pending_background_wakeup_kinds.clear()
                self._append_system_output("No background results are ready.")
                return
            # Delivery to the active run is not a future wake-up. Its queued
            # bus messages remain available for the next model request, while
            # later completions establish fresh wakeup provenance.
            self._background_results_ready = True
            self._pending_bus_check_needed = True
            self._pending_background_wakeup_kinds.clear()
            self._append_system_output(
                "Background results delivered to the active run; they will be used by its next model request."
            )
            return
        if not self._deliver_background_messages():
            self._background_results_ready = False
            self._pending_background_wakeup_kinds.clear()
            if self.phase == TUIPhase.BACKGROUND_RESULT_READY:
                self._set_phase(TUIPhase.IDLE)
            self._append_system_output("No background results are ready.")
            return
        # The messages are already queued on the main-agent bus for the turn
        # we are about to start. Do not let the command done callback schedule
        # a second wake-up before that new owner begins running.
        self._background_results_ready = False
        self._pending_bus_check_needed = False
        self._pending_background_wakeup_kinds.clear()
        self._append_system_output("Integrating background results...")
        self._launch_agent("")

    def _show_attachments(self) -> None:
        """List attachments queued for the next prompt."""
        if not self._pending_attachments:
            self._append_system_output("No images are queued.")
            return
        self._append_system_output(f"Queued images ({len(self._pending_attachments)}):")
        for index, attachment in enumerate(self._pending_attachments, start=1):
            self._append_output(f"  {index}. {self._format_attachment_description(attachment)}")
        self._append_system_output("Use /remove-image <number> or /remove-image all.")

    def _remove_attachment_command(self, args: str) -> None:
        """Remove one or all queued attachments from the compose state."""
        value = args.strip().lower()
        if value == "all":
            count = len(self._pending_attachments)
            self._reset_pending_attachments()
            self._append_system_output(f"Removed {count} queued image(s).")
            return
        if value:
            try:
                index = int(value) - 1
            except ValueError:
                self._append_system_output("Usage: /remove-image [number|all]")
                return
        else:
            index = None
        removed = self._remove_pending_attachment(index)
        if removed is None:
            self._append_system_output("No matching queued image.")
        else:
            self._append_system_output(f"Removed {self._format_attachment_description(removed)}.")

    @staticmethod
    def _serialize_tool_args(args: object) -> str | dict[str, Any] | None:
        """Retain full tool arguments in the query index."""
        if args is None or isinstance(args, str | dict):
            return args
        if isinstance(args, BaseModel):
            payload = args.model_dump(mode="json")
            if isinstance(payload, dict):
                return payload
            return json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        return json.dumps(args, ensure_ascii=False, indent=2, default=str)

    @staticmethod
    def _serialize_tool_result(content: object) -> str:
        """Serialize retained tool output without applying display truncation."""
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False, indent=2, default=str)

    def _tool_query_index(self) -> dict[str, ToolMessage]:
        """Build an authoritative cross-turn tool index from model history."""
        index: dict[str, ToolMessage] = {}
        for message in self._message_history or []:
            if isinstance(message, ModelResponse):
                for part in message.parts:
                    if not isinstance(part, ToolCallPart) or not part.tool_call_id:
                        continue
                    index[part.tool_call_id] = ToolMessage(
                        tool_call_id=part.tool_call_id,
                        name=part.tool_name,
                        args=self._serialize_tool_args(part.args),
                    )
            elif isinstance(message, ModelRequest):
                for part in message.parts:
                    if not isinstance(part, ToolReturnPart | RetryPromptPart) or not part.tool_call_id:
                        continue
                    existing = index.get(part.tool_call_id)
                    if existing is None:
                        existing = ToolMessage(
                            tool_call_id=part.tool_call_id,
                            name=part.tool_name or "tool",
                        )
                        index[part.tool_call_id] = existing
                    existing.content = self._serialize_tool_result(part.content)

        for tool_call_id, live_message in self._tool_messages.items():
            existing = index.get(tool_call_id)
            if existing is None:
                index[tool_call_id] = live_message.model_copy(deep=True)
                continue
            if existing.args is None:
                existing.args = live_message.args
            if existing.content is None:
                existing.content = live_message.content
        return index

    def _show_tool_result(self, requested_id: str) -> None:
        """Render a complete tool result retained in any loaded turn."""
        tool_messages = self._tool_query_index()
        requested_id = requested_id.strip()
        if not requested_id:
            recent = list(tool_messages)[-5:]
            hint = ", ".join(recent) if recent else "none"
            self._append_system_output(f"Usage: /tool <call-id>. Recent IDs: {hint}")
            return
        matches = [call_id for call_id in tool_messages if call_id == requested_id or call_id.startswith(requested_id)]
        if len(matches) != 1:
            if matches:
                self._append_system_output(f"Ambiguous tool call ID: {', '.join(matches)}")
            else:
                self._append_system_output(f"Tool call not found: {requested_id}")
            return
        call_id = matches[0]
        message = tool_messages[call_id]
        self._append_system_output(f"Tool {message.name} [{call_id}]")
        if message.args:
            args_text = (
                json.dumps(message.args, ensure_ascii=False, indent=2)
                if isinstance(message.args, dict)
                else message.args
            )
            self._append_output(f"Arguments:\n{args_text}")
        self._append_output(message.content or "[no output]")

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
                self._clear_transcript()
            case "/new":
                await self._start_new_session()
            case "/cancel":
                self._cancel_foreground()
            case "/integrate":
                self._integrate_background_results()
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
                    await self._show_session_selector()
                else:
                    await self._load_session(args.strip())
            case "/load":
                self._append_user_input(command)
                if not args.strip():
                    self._append_system_output("Usage: /load <folder>")
                    self._append_system_output("To restore a session by ID, use /session <id>")
                else:
                    await self._load_history(args.strip())
            case "/exit":
                self._append_user_input(command)
                if self._app:
                    self._show_shutdown_status("exit requested")
                    self._app.exit()
            case "/model":
                self._append_user_input(command)
                await self._show_model_selector()
            case "/agents":
                self._append_user_input(command)
                self._show_agents()
            case "/process":
                self._append_user_input(command)
                self._show_processes()
            case "/attachments":
                self._show_attachments()
            case "/remove-image":
                self._remove_attachment_command(args)
            case "/tool":
                self._show_tool_result(args)
            case "/paste-image":
                self._append_user_input(command)
                await self._paste_clipboard_image()
            case "/goal":
                self._append_user_input(command)
                ctx = self.runtime.ctx
                if ctx.goal_active:
                    self._append_system_output("Goal is already running. Use Ctrl+C to stop it first.")
                elif not args.strip():
                    self._append_system_output("Usage: /goal <task description>")
                else:
                    task = args.strip()
                    ctx.reset_goal()
                    ctx.goal_task = task
                    ctx.goal_iteration = 0
                    ctx.goal_max_iterations = self.config.general.max_goal_iterations
                    self._goal_usage_start_breakdown = self._session_usage.token_breakdown
                    self._goal_usage_report_pending = False
                    self._append_system_output(
                        f"[Goal] Starting goal mode ({ctx.goal_max_iterations} max iterations). Ctrl+C to stop."
                    )
                    self._launch_agent(task)
            case _:
                # Check custom commands
                cmd_name = cmd[1:]  # Remove leading /
                commands = self.config.get_commands()
                if cmd_name in commands:
                    cmd_def = commands[cmd_name]
                    # Append user instruction to prompt if provided
                    prompt = cmd_def.prompt
                    if args.strip():
                        prompt = f"{prompt}\n\nUser instruction: {args.strip()}"
                    # Show expanded prompt instead of command name
                    self._append_user_input(prompt)
                    self._launch_agent(prompt)
                else:
                    candidates = [name.removeprefix("/") for name in self._command_words()]
                    suggestions = difflib.get_close_matches(cmd_name, candidates, n=3, cutoff=0.45)
                    if suggestions:
                        rendered = ", ".join(f"/{name}" for name in suggestions)
                        self._append_system_output(f"Unknown command {cmd}. Did you mean: {rendered}?")
                    else:
                        self._append_system_output(f"Unknown command {cmd}. Use /help to list commands.")

    async def _terminate_direct_shell_process(self, process: asyncio.subprocess.Process | None) -> None:
        """Terminate a direct !command subprocess group during timeout or shutdown."""
        if process is None:
            return

        if os.name == "posix" and process.pid is not None:
            # start_new_session=True makes the shell PID the process-group ID.
            # Use that stable ID rather than getpgid(pid), because the shell can
            # exit before a background child holding a pipe open does.
            process_group_id = process.pid
            with contextlib.suppress(ProcessLookupError, OSError):
                os.killpg(process_group_id, signal.SIGTERM)

            if process.returncode is None:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(process.wait(), timeout=_DIRECT_SHELL_TERMINATE_TIMEOUT)

            # A timed-out command must not leave a child alive merely because
            # its shell exited first. This is harmless when the group is gone.
            with contextlib.suppress(ProcessLookupError, OSError):
                os.killpg(process_group_id, signal.SIGKILL)
            return

        if process.returncode is not None:
            return

        with contextlib.suppress(ProcessLookupError):
            process.terminate()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(process.wait(), timeout=_DIRECT_SHELL_TERMINATE_TIMEOUT)

        if process.returncode is not None:
            return

        with contextlib.suppress(ProcessLookupError):
            process.kill()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(process.wait(), timeout=_DIRECT_SHELL_TERMINATE_TIMEOUT)

    async def _execute_shell_command(self, command_str: str) -> None:
        """Execute a foreground shell command with exclusive task ownership."""
        if not command_str.strip():
            self._append_system_output("Usage: !<command>")
            return

        current_task = asyncio.current_task()
        if current_task is None:
            raise RuntimeError("Direct shell execution requires an asyncio task.")
        existing_owner = self._direct_shell_task
        if existing_owner is not None and existing_owner is not current_task and not existing_owner.done():
            self._append_system_output("A direct shell command already owns the foreground.")
            return
        if existing_owner is None:
            if self._is_foreground_busy():
                self._append_system_output("Foreground work is already in progress.")
                return
            self._direct_shell_task = current_task
            self._direct_shell_command = command_str
            self._set_phase(TUIPhase.SHELL_RUNNING)

        # Show command being executed
        cmd_text = Text()
        cmd_text.append("$ ", style="bold cyan")
        cmd_text.append(command_str, style="cyan")
        self._append_output(self._renderer.render(cmd_text).rstrip())

        start_time = time.time()
        process: asyncio.subprocess.Process | None = None
        drain_tasks: list[asyncio.Task[None]] = []

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

            # Drain both pipes concurrently. Unlike communicate(), this never retains
            # the complete command output in memory and therefore stays bounded even
            # when a command produces many gigabytes on stdout or stderr.
            stdout_tail = _BoundedOutputTail(_DIRECT_SHELL_OUTPUT_TAIL_BYTES)
            stderr_tail = _BoundedOutputTail(_DIRECT_SHELL_OUTPUT_TAIL_BYTES)

            def make_live_appender(*, style: str | None = None) -> Callable[[str], None]:
                current_block_id: BlockId | None = None
                current_line = ""

                def render_line(value: str) -> str:
                    if style is None:
                        return value
                    return self._renderer.render(Text(value, style=style)).rstrip("\n")

                def append_chunk(chunk: str) -> None:
                    nonlocal current_block_id, current_line
                    remaining = chunk
                    while remaining:
                        newline_index = remaining.find("\n")
                        line_complete = newline_index >= 0
                        fragment = remaining[: newline_index + 1] if line_complete else remaining
                        remaining = remaining[newline_index + 1 :] if line_complete else ""
                        content = fragment[:-1] if line_complete else fragment
                        if line_complete and content.endswith("\r"):
                            content = content[:-1]
                        current_line += content
                        encoded = current_line.encode("utf-8")
                        if len(encoded) > _DIRECT_SHELL_OUTPUT_TAIL_BYTES:
                            marker = "... [live line truncated] "
                            tail_budget = max(0, _DIRECT_SHELL_OUTPUT_TAIL_BYTES - len(marker.encode()))
                            current_line = marker + encoded[-tail_budget:].decode("utf-8", errors="ignore")
                        rendered = render_line(current_line)
                        if not self._update_block_by_id(current_block_id, rendered):
                            current_block_id = self._append_block(rendered)
                        if line_complete:
                            current_block_id = None
                            current_line = ""
                    if self._follow_latest:
                        self._scroll_to_bottom()
                    self._throttled_invalidate()

                return append_chunk

            append_stdout = make_live_appender()
            append_stderr = make_live_appender(style="red")

            stdout_drain = asyncio.create_task(_drain_direct_shell_stream(process.stdout, stdout_tail, append_stdout))
            stderr_drain = asyncio.create_task(_drain_direct_shell_stream(process.stderr, stderr_tail, append_stderr))
            drain_tasks = [stdout_drain, stderr_drain]
            await asyncio.wait_for(
                asyncio.gather(process.wait(), *drain_tasks),
                timeout=_DIRECT_SHELL_TIMEOUT,
            )
            elapsed = time.time() - start_time

            stdout_note = _format_direct_shell_truncation_note(stdout_tail, stream_name="stdout")
            if stdout_note:
                self._append_output(stdout_note)
            stderr_note = _format_direct_shell_truncation_note(stderr_tail, stream_name="stderr")
            if stderr_note:
                self._append_output(self._renderer.render(Text(stderr_note, style="red")).rstrip("\n"))

            # Show exit code if non-zero
            if process.returncode != 0:
                self._append_system_output(f"Exit code: {process.returncode}")

            # Show elapsed time
            self._append_output(f"({elapsed:.1f}s)")

        except TimeoutError:
            await self._terminate_direct_shell_process(process)
            self._append_system_output(f"Command timed out ({_DIRECT_SHELL_TIMEOUT:.0f}s)")
        except asyncio.CancelledError:
            await self._terminate_direct_shell_process(process)
            self._append_system_output("Shell command cancelled.")
            raise
        except Exception as e:
            await self._terminate_direct_shell_process(process)
            self._append_system_output(f"Error: {type(e).__name__}: {e}")
        finally:
            for task in drain_tasks:
                if not task.done():
                    task.cancel()
            if drain_tasks:
                await asyncio.gather(*drain_tasks, return_exceptions=True)
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
        for name, description in BUILTIN_COMMAND_HELP.items():
            sys_table.add_row(f"/{name}", description)
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

    def _reset_context_session_state(self, ctx: TUIContext) -> None:
        """Reset conversation-scoped AgentContext state while preserving runtime policy."""
        ctx.provider_session_id = None
        ctx.provider_thread_id = None
        ctx.deferred_tool_metadata = {}
        ctx.handoff_message = None
        ctx.force_inject_instructions = False
        ctx.shell_env = dict(self.config.shell_env) if isinstance(self.config, YaacliConfig) else {}
        ctx.usage_snapshot_entries = {}
        ctx.shell_review_records.clear()
        ctx.user_prompts = None
        ctx.previous_assistant_response_reference = None
        ctx.steering_messages = []
        ctx.tool_id_wrapper.clear()
        ctx.agent_stream_queues = {}
        ctx.subagent_history = {}
        ctx.agent_registry = {}
        ctx.auto_load_files = []
        ctx.task_manager = TaskManager()
        ctx.note_manager = NoteManager()
        ctx.tool_search_loaded_tools = []
        ctx.tool_search_loaded_namespaces = []
        ctx.tool_tags = set()
        ctx.message_bus = MessageBus()
        ctx.reset_goal()

    def _reset_tui_session_state(self) -> None:
        """Reset conversation-scoped display and interaction state."""
        self._reset_output_blocks()
        self._streaming_text = ""
        self._streaming_text_buffer = None
        self._streaming_line_index = None
        self._streaming_block_id = None
        self._streaming_thinking = ""
        self._streaming_thinking_buffer = None
        self._streaming_thinking_line_index = None
        self._streaming_thinking_block_id = None
        self._prompt_history.clear()
        self._history_index = -1
        self._current_input_backup = ""
        self._printed_tool_calls.clear()
        self._tool_messages.clear()
        self._subagent_states.clear()
        self._display_replay.clear()
        self._event_renderer.clear()
        self._reset_pending_attachments()
        self._pending_bus_check_needed = False
        self._background_results_ready = False
        self._pending_background_wakeup_kinds.clear()
        self._reset_hitl_state()
        self._goal_usage_start_breakdown = None
        self._goal_usage_report_pending = False
        self._agent_phase = "idle"
        self._run_started_at = None
        self._run_timer_paused_at = None
        self._display_adapter = None
        self._message_history = None
        self._last_run = None
        self._last_session_input = None
        self._last_session_output = None
        self._session_selector_open = False
        self._session_selector_entries = []
        self._session_selector_index = 0
        self._current_context_tokens = 0

    async def _clear_session(self) -> None:
        """Start a clean conversation while preserving reusable runtime configuration."""
        if self._session_clear_in_progress:
            return
        self._session_clear_in_progress = True
        monitor = self._get_background_monitor()
        commit_fresh_context = True
        try:
            if monitor is not None:
                # Establish the isolation boundary before any operation that
                # can fail or yield to an old background task.
                monitor.begin_session_reset()
                try:
                    self._drain_background_usage(monitor)
                except Exception:
                    logger.exception("Failed to preserve background usage before session clear")
                try:
                    await monitor.reset_session_state()
                except ShellBackgroundResetError:
                    # A live process may still exist. Do not commit a new
                    # conversation until its retained handle can be killed.
                    commit_fresh_context = False
                    raise
                except Exception:
                    logger.exception("Failed to fully reset background session state")
                try:
                    self._drain_background_usage(monitor)
                except Exception:
                    logger.exception("Failed to preserve final background usage during session clear")
        finally:
            try:
                if monitor is not None:
                    monitor.finish_session_reset()
                if commit_fresh_context:
                    # Non-shell cleanup errors roll forward after tombstoning;
                    # shell termination failure is the only fatal boundary.
                    self._reset_tui_session_state()
                    self._session_usage.clear()
                    if self._runtime is not None:
                        ctx = self._runtime.ctx
                        if isinstance(ctx, TUIContext):
                            fresh_ctx = ctx.prepare_new_run()
                            self._reset_context_session_state(fresh_ctx)
                            self._runtime.ctx = fresh_ctx
                            if monitor is not None:
                                monitor.set_message_bus(fresh_ctx.message_bus, fresh_ctx.agent_id)
            finally:
                self._session_clear_in_progress = False

    def _show_cost(self) -> None:
        """Show token usage summary for the current session."""
        summary = self._session_usage.format_summary()
        self._append_system_output(summary)

    def _show_agents(self) -> None:
        """Show running and recently completed background subagents."""
        monitor = self._get_background_monitor()
        active: dict[str, asyncio.Task[Any]] = {}
        infos: dict[str, BackgroundTaskInfo] = {}
        results: dict[str, BackgroundTaskResult] = {}
        if monitor is not None:
            active = monitor.active_tasks
            infos = monitor.task_infos
            results = monitor.task_results

        if not infos:
            self._append_system_output("No background subagents.")
            return

        running_count = sum(agent_id in active and not active[agent_id].done() for agent_id in infos)
        header = Text(
            f"Background Subagents ({running_count} running, {len(infos) - running_count} finished)",
            style="bold cyan",
        )
        table = Table(show_header=True, box=None, padding=(0, 2))
        table.add_column("Agent ID", style="dim")
        table.add_column("Subagent", style="bold")
        table.add_column("Status")
        table.add_column("Elapsed", style="dim")
        table.add_column("Prompt", style="dim")

        now = datetime.now(UTC)
        status_styles = {"completed": "green", "failed": "red", "cancelled": "yellow"}
        for agent_id, info in sorted(infos.items()):
            is_running = agent_id in active and not active[agent_id].done()
            result = results.get(agent_id)
            if is_running:
                status_text = Text("running", style="cyan")
                ended_at = now
            elif result is not None:
                status_text = Text(result.status, style=status_styles.get(result.status, ""))
                ended_at = result.completed_at
            else:
                status_text = Text("finished", style="dim")
                ended_at = now
            elapsed = ended_at - info.started_at
            elapsed_str = f"{max(0, int(elapsed.total_seconds()))}s"
            prompt_preview = info.prompt[:60] + "..." if len(info.prompt) > 60 else info.prompt
            name = f"{info.subagent_name} (resume)" if info.is_resume else info.subagent_name
            table.add_row(agent_id, name, status_text, elapsed_str, prompt_preview)

        self._append_output(f"{self._renderer.render(header).rstrip()}\n{self._renderer.render(table).rstrip()}")

    def _show_processes(self) -> None:
        """Show active background shell processes."""
        processes: dict[str, BackgroundProcess] = {}
        try:
            if self._runtime and self._runtime.env and self._runtime.env.shell:
                processes = self._runtime.env.shell.active_background_processes
        except RuntimeError:
            pass

        if not processes:
            self._append_system_output("No active background processes.")
            return

        header = Text(f"Background Processes ({len(processes)} running)", style="bold cyan")
        table = Table(show_header=True, box=None, padding=(0, 2))
        table.add_column("ID", style="dim")
        table.add_column("Status", style="bold")
        table.add_column("Command")
        table.add_column("PID", style="dim")

        for process_id, process in sorted(processes.items()):
            elapsed = _get_elapsed_seconds(process.started_at)
            status_text = Text(f"running ({elapsed:.0f}s)", style="cyan")
            pid_str = str(process.pid) if process.pid is not None else "-"
            table.add_row(process_id, status_text, process.command, pid_str)

        self._append_output(f"{self._renderer.render(header).rstrip()}\n{self._renderer.render(table).rstrip()}")

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

            display_file = dump_dir / "display_messages.json"
            display_file.write_text(json.dumps(self._display_replay.snapshot(), ensure_ascii=False, indent=2))

            # Save context state
            state_file = dump_dir / "context_state.json"
            state = self._export_session_state(include_usage_ledger=False)
            state_file.write_text(state.model_dump_json(indent=2))

            self._append_system_output(f"Session dumped to {dump_dir}")
            self._append_system_output(f"  - message_history.json ({len(self._message_history)} messages)")
            self._append_system_output(f"  - display_messages.json ({len(self._display_replay.snapshot())} events)")
            self._append_system_output("  - context_state.json")
        except Exception as e:
            self._append_system_output(f"Error: {e}")

    async def _load_history(self, folder_path: str, *, target_session_id: str | None = None) -> bool:
        """Transactionally load history into an isolated conversation context.

        ``/load`` keeps the current durable session ID. ``/session`` supplies
        ``target_session_id`` so context, transcript, replay, and identity all
        change in the same no-await commit.
        """
        if self._session_clear_in_progress:
            self._append_system_output("A session switch is already in progress.")
            return False

        load_dir = Path(folder_path).expanduser().resolve()
        if not load_dir.is_dir():
            self._append_system_output(f"Not a directory: {load_dir}")
            return False

        history_file = load_dir / "message_history.json"
        display_file = load_dir / "display_messages.json"
        state_file = load_dir / "context_state.json"
        history_payload: bytes | None = None
        state_payload: bytes | None = None
        display_payload: bytes | None = None
        atomic_session_snapshot = False
        if not history_file.exists():
            try:
                snapshot = read_head_artifacts(
                    self.config_manager,
                    load_dir.name,
                    max_display_messages_bytes=_MAX_DISPLAY_REPLAY_LOAD_BYTES,
                )
            except (FileNotFoundError, ValueError):
                snapshot = None
            if snapshot is not None:
                atomic_session_snapshot = True
                history_payload = snapshot.message_history_json
                state_payload = snapshot.context_state_json
                display_payload = snapshot.display_messages_json

        if history_payload is None and not history_file.exists():
            self._append_system_output(f"message_history.json not found in {load_dir}")
            return False

        try:
            # Stage every fallible artifact and context operation before
            # touching the active conversation or its background monitor.
            history = ModelMessagesTypeAdapter.validate_json(
                history_payload if history_payload is not None else history_file.read_bytes()
            )
            if atomic_session_snapshot:
                state = TUIResumableState.model_validate_json(state_payload) if state_payload is not None else None
            else:
                state = TUIResumableState.model_validate_json(state_file.read_text()) if state_file.exists() else None
            display_events: list[dict[str, Any]] = []
            display_warning: str | None = None
            if atomic_session_snapshot:
                if display_payload is None:
                    display_warning = (
                        "Display replay is too large to restore safely; conversation history was still loaded."
                    )
                else:
                    try:
                        display_events = validate_display_events(json.loads(display_payload.decode("utf-8")))
                    except Exception as exc:
                        logger.warning("Failed to restore atomic display replay for %s: %s", load_dir, exc)
                        display_warning = (
                            "Display replay is invalid and was skipped; conversation history was still loaded."
                        )
            elif display_file.exists():
                try:
                    loaded_display_events = load_display_replay(
                        display_file,
                        max_bytes=_MAX_DISPLAY_REPLAY_LOAD_BYTES,
                    )
                    if loaded_display_events is None:
                        display_warning = (
                            "Display replay is too large to restore safely; conversation history was still loaded."
                        )
                    else:
                        display_events = loaded_display_events
                except Exception as exc:
                    logger.warning("Failed to restore display replay from %s: %s", display_file, exc)
                    display_warning = (
                        "Display replay is invalid and was skipped; conversation history was still loaded."
                    )

            replacement_replay = BoundedDisplayReplay(config=YAACLI_AGUI_REPLAY_CONFIG)
            replacement_replay.extend_snapshot(display_events)

            old_ctx = self.runtime.ctx
            if not isinstance(old_ctx, TUIContext):
                raise TypeError(f"Expected TUIContext, got {type(old_ctx).__name__}")
            candidate_ctx = old_ctx.prepare_new_run()
            self._reset_context_session_state(candidate_ctx)
            if state is not None:
                restore_resumable_state_safely(state, candidate_ctx)
            restored_usage = state.session_usage_snapshot if state is not None else None
            if restored_usage is None and candidate_ctx.usage_snapshot_entries:
                restored_usage = candidate_ctx.build_usage_snapshot()
        except Exception as exc:
            self._append_system_output(f"Error loading session: {exc}")
            return False

        monitor = self._get_background_monitor()
        cancelled: asyncio.CancelledError | None = None
        shell_cleanup_error: ShellBackgroundResetError | None = None
        self._session_clear_in_progress = True
        try:
            if monitor is not None:
                # Tombstone old subagents and revoke their inherited shell lease
                # before the first await.
                monitor.begin_session_reset()
                try:
                    self._drain_background_usage(monitor)
                except Exception:
                    logger.exception("Failed to preserve background usage before session switch")
                try:
                    await monitor.reset_session_state()
                except ShellBackgroundResetError as exc:
                    shell_cleanup_error = exc
                    logger.exception("Shell process cleanup blocked session switch")
                except asyncio.CancelledError as exc:
                    cancelled = exc
                    logger.debug("Session switch cancelled while resetting old subagents; committing isolated state")
                except Exception:
                    logger.exception("Failed to fully reset background subagent state during session switch")
                try:
                    self._drain_background_usage(monitor)
                except Exception:
                    logger.exception("Failed to preserve final background usage during session switch")
        finally:
            try:
                if monitor is not None:
                    monitor.finish_session_reset()

                if shell_cleanup_error is None:
                    # No await is permitted in this commit: observers see either
                    # the complete old session or the complete restored session.
                    committed_session_id = target_session_id or self._session_id
                    self._reset_tui_session_state()
                    self.runtime.ctx = candidate_ctx
                    if monitor is not None:
                        monitor.set_message_bus(candidate_ctx.message_bus, candidate_ctx.agent_id)
                    self._message_history = history
                    self._display_replay = replacement_replay
                    self._session_id = committed_session_id
                    self._set_phase(TUIPhase.IDLE)
                    self._restore_output_from_display_events(replacement_replay.snapshot())
                    self._session_usage.clear()

                    if restored_usage is not None:
                        self._session_usage.set_run_snapshot(restored_usage)
                        self._session_usage.commit_run_snapshot(restored_usage.run_id)
                        self._session_usage.finalize_run_snapshots(restored_usage.run_id)
                        # Avoid counting restored ledger entries again on the next run.
                        candidate_ctx.usage_snapshot_entries.clear()
            finally:
                self._session_clear_in_progress = False

        if shell_cleanup_error is not None:
            self._append_system_output(f"Session was not loaded: {_safe_exception_str(shell_cleanup_error)}")
            return False
        if display_warning is not None:
            self._append_system_output(display_warning)
        self._append_system_output(f"Session loaded from {load_dir}")
        self._append_system_output(f"  - message_history.json ({len(history)} messages)")
        if atomic_session_snapshot or display_file.exists():
            self._append_system_output(f"  - display_messages.json ({len(replacement_replay.snapshot())} events)")
        self._append_system_output(
            "  - context_state.json (restored)" if state is not None else "  - context_state.json (not found, skipped)"
        )
        self._append_system_output("Next message will continue from loaded history.")
        if cancelled is not None:
            raise cancelled
        return True

    def _export_session_state(
        self,
        *,
        include_usage_ledger: bool,
        include_extra_usages: bool | None = None,
    ) -> TUIResumableState:
        """Export resumable context plus cumulative session usage."""
        if include_extra_usages is None:
            base_state = self.runtime.ctx.export_state(include_usage_ledger=include_usage_ledger)
        else:
            base_state = self.runtime.ctx.export_state(include_extra_usages=include_extra_usages)
        usage_snapshot = (
            None
            if self._session_usage.is_empty()
            else self._session_usage.export_snapshot(run_id=f"session:{self._session_id}")
        )
        return TUIResumableState(
            **base_state.model_dump(),
            session_usage_snapshot=usage_snapshot,
        )

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

        if not self._message_history and not self._display_replay.snapshot():
            return False

        state = self._export_session_state(
            include_usage_ledger=include_usage_ledger,
            include_extra_usages=include_extra_usages,
        )

        profile = self._active_model_profile
        turn_dir = save_session_turn(
            config_manager=self.config_manager,
            session_id=self._session_id,
            working_dir=self.working_dir,
            model_profile_id=profile.id if profile is not None else None,
            model_label=profile.label if profile is not None else None,
            model=profile.model if profile is not None else self._get_configured_model(),
            message_history_json=ModelMessagesTypeAdapter.dump_json(self._message_history or [], indent=2),
            context_state_json=state.model_dump_json(indent=2),
            display_messages=self._display_replay.snapshot(),
            input_text=self._last_session_input,
            output_text=self._last_session_output,
            save_reason=save_reason,
            max_turns=_positive_int_config(
                getattr(getattr(self.config, "session", None), "max_turns_per_session", None),
                _DEFAULT_MAX_TURNS_PER_SESSION,
            ),
            max_sessions=_positive_int_config(
                getattr(getattr(self.config, "session", None), "max_sessions", None), _DEFAULT_MAX_SESSIONS
            ),
            max_session_age_days=_optional_positive_int_config(
                getattr(getattr(self.config, "session", None), "max_session_age_days", None)
            ),
        )

        logger.debug("Saved session snapshot to %s (reason=%s)", turn_dir, save_reason)
        return True

    async def _save_session_snapshot_async(
        self,
        *,
        include_usage_ledger: bool,
        save_reason: str,
    ) -> bool:
        """Prepare a consistent snapshot, then serialize/write it off-loop."""
        async with self._session_save_lock:
            display_messages = self._display_replay.snapshot()
            message_history = list(self._message_history or [])
            if not message_history and not display_messages:
                return False

            # Read mutable runtime structures on the event-loop thread. The
            # prepared Pydantic state and shallow message list are stable while
            # this turn awaits the worker write.
            state = self._export_session_state(include_usage_ledger=include_usage_ledger)
            max_turns = _positive_int_config(
                getattr(getattr(self.config, "session", None), "max_turns_per_session", None),
                _DEFAULT_MAX_TURNS_PER_SESSION,
            )
            max_sessions = _positive_int_config(
                getattr(getattr(self.config, "session", None), "max_sessions", None), _DEFAULT_MAX_SESSIONS
            )
            max_session_age_days = _optional_positive_int_config(
                getattr(getattr(self.config, "session", None), "max_session_age_days", None)
            )

            profile = self._active_model_profile
            input_text = self._last_session_input
            output_text = self._last_session_output

            def persist() -> Path:
                return save_session_turn(
                    config_manager=self.config_manager,
                    session_id=self._session_id,
                    working_dir=self.working_dir,
                    model_profile_id=profile.id if profile is not None else None,
                    model_label=profile.label if profile is not None else None,
                    model=profile.model if profile is not None else self._get_configured_model(),
                    message_history_json=ModelMessagesTypeAdapter.dump_json(message_history, indent=2),
                    context_state_json=state.model_dump_json(indent=2),
                    display_messages=display_messages,
                    input_text=input_text,
                    output_text=output_text,
                    save_reason=save_reason,
                    max_turns=max_turns,
                    max_sessions=max_sessions,
                    max_session_age_days=max_session_age_days,
                )

            previous_phase = self.phase
            self._set_phase(TUIPhase.SAVING)
            worker = asyncio.create_task(asyncio.to_thread(persist))
            try:
                turn_dir = await asyncio.shield(worker)
            except asyncio.CancelledError:
                # A worker thread cannot be cancelled. Keep the save lock until
                # it finishes so a newer writer cannot race the same session.
                with contextlib.suppress(Exception):
                    await worker
                raise
            finally:
                if self.phase == TUIPhase.SAVING:
                    self._set_phase(previous_phase)
            logger.debug("Saved session snapshot to %s (reason=%s)", turn_dir, save_reason)
            return True

    async def _auto_save_history(self) -> None:
        """Auto-save a successful turn when session persistence is enabled."""
        if self.config.session.auto_save_history:
            self._last_snapshot_saved = await self._save_session_snapshot_async(
                include_usage_ledger=False,
                save_reason="success",
            )

    @property
    def session_id(self) -> str:
        """Get the current session ID."""
        return self._session_id

    @property
    def has_session_data(self) -> bool:
        """Check whether the current session has a legacy or schema-v2 snapshot."""
        try:
            paths = get_head_artifact_paths(self.config_manager, self._session_id)
        except (FileNotFoundError, ValueError):
            return False
        return paths.message_history_file is not None and paths.message_history_file.exists()

    def _prune_sessions(self, sessions_dir: Path, max_sessions: int = 100) -> None:
        """Remove old sessions beyond the retention limit.

        Keeps the most recent `max_sessions` sessions, deleting the rest.
        Sessions are sorted by updated_at from metadata.json, falling back
        to directory mtime.

        Args:
            sessions_dir: Path to ~/.yaacli/sessions/
            max_sessions: Maximum number of sessions to retain.
        """
        try:
            trim_sessions(
                sessions_dir,
                max_sessions=max_sessions,
                max_session_age_days=_optional_positive_int_config(
                    getattr(getattr(self.config, "session", None), "max_session_age_days", None)
                ),
                protected_session_id=self._session_id,
            )
        except OSError as e:
            logger.warning("Failed to prune sessions in %s: %s", sessions_dir, e)

    def _list_sessions(self, max_display: int = 20) -> None:
        """List recent sessions.

        Shows session ID, timestamp, and working directory.

        Args:
            max_display: Maximum number of sessions to show.
        """
        from rich.table import Table

        try:
            sessions = list_sessions(self.config_manager)
        except (OSError, ValueError) as exc:
            logger.warning("Failed to list sessions: %s", exc)
            self._append_system_output(f"Unable to list sessions: {exc}")
            return
        if not sessions:
            self._append_system_output("No sessions found.")
            return

        current_id = self._session_id
        table = Table(show_header=True, box=None, padding=(0, 2))
        table.add_column("Session ID", style="cyan")
        table.add_column("Updated", style="dim")
        table.add_column("Working Dir", style="dim")

        for session in sessions[:max_display]:
            marker = f"{session.id} *" if session.id == current_id else session.id
            updated = session.updated_at[:19].replace("T", " ")
            table.add_row(marker, updated, session.working_dir or "unknown")

        self._append_system_output(
            f"Sessions ({len(sessions)} total, showing latest {min(len(sessions), max_display)}):"
        )
        self._append_output(self._renderer.render(table).rstrip())
        self._append_system_output("Use /session <id> to restore. (* = current session)")

    async def _load_session(self, session_id: str) -> bool:
        """Load a saved session by exact ID or unambiguous prefix."""
        try:
            target = resolve_session_dir(self.config_manager, session_id)
            info = get_session_info(self.config_manager, target.name)
        except (FileNotFoundError, ValueError) as exc:
            self._append_system_output(str(exc))
            return False

        if not await self._load_history(str(target), target_session_id=target.name):
            return False

        self._last_session_input = info.input_text
        self._last_session_output = info.output_text
        if info.working_dir is not None and Path(info.working_dir).resolve() != self.working_dir.resolve():
            self._append_system_output(
                f"Workspace warning: session was saved in {info.working_dir}; current workspace is {self.working_dir}."
            )
        active_model = (
            self._active_model_profile.model if self._active_model_profile is not None else self._get_configured_model()
        )
        if info.model and active_model and info.model != active_model:
            self._append_system_output(f"Model warning: session used {info.model}; continuing with {active_model}.")
        return True

    async def _restore_startup_session(self) -> bool:
        """Restore an explicitly requested session or the newest workspace match."""
        requested = self.initial_session_id
        if requested:
            if await self._load_session(requested):
                return True
            raise RuntimeError(f"Unable to restore requested session: {requested}")
        if not isinstance(self.config, YaacliConfig) or not self.config.session.auto_restore:
            return False

        workspace = self.working_dir.resolve()
        candidates = [
            info
            for info in list_sessions(self.config_manager)
            if info.working_dir is not None and Path(info.working_dir).resolve() == workspace
        ]
        for candidate in candidates:
            if await self._load_session(candidate.id):
                return True
        return False

    def _append_system_output(self, text: str) -> None:
        """Append system message to output, wrapped to the current terminal width."""
        sys_text = Text()
        sys_text.append("[SYS] ", style="bold yellow")
        sys_text.append(text)
        self._append_output(self._renderer.render(sys_text, width=self._get_terminal_width()).rstrip())

    # =========================================================================
    # Main Run Loop
    # =========================================================================

    async def run(self) -> None:
        """Run the TUI application."""
        if not self._theme_terminal_resolved:
            self._configure_theme(query_terminal=True)
            logger.debug("Resolved terminal theme: %s (%s)", self._theme.variant, self._theme.source)

        restored_session = await self._restore_startup_session()

        # Welcome message
        title = Text("YAACLI CLI", style="bold magenta")
        self._append_output(self._renderer.render(title).rstrip())
        self._append_output(f"Model: {self._format_active_model_label()}")
        self._append_output("Type /help for commands; F2 expands tasks; Ctrl+L returns to live output.")

        # Show session ID
        self._append_output(f"Session: {self._session_id}{' (restored)' if restored_session else ''}")

        # Create scrollable FormattedTextControl with mouse support
        tui_ref = self

        class ScrollableFormattedTextControl(FormattedTextControl):
            """FormattedTextControl that handles mouse scroll events."""

            def mouse_handler(self, mouse_event: MouseEvent) -> object:
                """Handle mouse scroll events."""
                if mouse_event.event_type == MouseEventType.SCROLL_UP:
                    tui_ref._scroll_output(-3)
                    if tui_ref._app:
                        tui_ref._app.invalidate()
                    return None
                elif mouse_event.event_type == MouseEventType.SCROLL_DOWN:
                    tui_ref._scroll_output(3)
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
        self._output_window = output_window

        # Persistent task pane: hidden when empty and one-line compact by default.
        task_control = FormattedTextControl(self._get_task_text)
        task_window = ConditionalContainer(
            Window(
                content=task_control,
                height=self._get_task_height,
                style="class:task-pane",
                wrap_lines=False,
            ),
            filter=Condition(lambda: bool(self._get_tasks())),
        )

        # Model selector is a floating overlay and never consumes output rows.
        model_selector_control = FormattedTextControl(self._get_model_selector_text)
        model_selector_window = ConditionalContainer(
            Window(
                content=model_selector_control,
                height=self._get_model_selector_height,
                style="class:model-selector",
                wrap_lines=False,
            ),
            filter=Condition(lambda: self._model_selector_open),
        )

        session_selector_control = FormattedTextControl(self._get_session_selector_text)
        session_selector_body = Box(
            Window(
                content=session_selector_control,
                height=self._get_session_selector_height,
                style="class:session-selector",
                wrap_lines=False,
            ),
            padding_left=1,
            padding_right=1,
            style="class:session-selector",
        )
        session_selector_window = ConditionalContainer(
            Frame(
                session_selector_body,
                title=self._get_session_selector_title,
                style="class:session-selector.frame",
            ),
            filter=Condition(lambda: self._session_selector_open),
        )

        # Status bar
        status_bar = Window(
            content=FormattedTextControl(self._get_status_text),
            height=self._get_status_height,
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
            height=lambda: 3 if self._app and self._app.output.get_size().rows < 28 else 5,
            scrollbar=True,
            completer=SlashCommandCompleter(
                self._command_words,
                self._session_completion_ids,
                self._skill_words,
            ),
            complete_while_typing=True,
        )

        self._input_area = input_area

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

        # Layout: floating selectors over a stable Output | Tasks | Status | Input body.
        body = HSplit([
            output_window,
            task_window,
            status_bar,
            input_area,
        ])
        root = FloatContainer(
            content=body,
            floats=[
                Float(top=1, left=2, right=2, content=model_selector_window),
                Float(top=1, width=self._get_session_selector_width, content=session_selector_window),
                Float(
                    xcursor=True,
                    ycursor=True,
                    content=CompletionsMenu(max_height=8, scroll_offset=1, display_arrows=True),
                ),
            ],
        )
        layout = Layout(root, focused_element=input_area)

        # Key bindings
        kb = self._setup_keybindings(input_area)

        # Create application
        self._app = Application(
            layout=layout,
            key_bindings=kb,
            style=self._setup_style(),
            full_screen=True,
            mouse_support=True,
            min_redraw_interval=self._invalidate_interval,
            refresh_interval=1.0,
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
