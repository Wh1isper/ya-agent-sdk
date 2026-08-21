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
from collections.abc import Awaitable, Callable, Sequence
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
from pydantic import BaseModel, JsonValue, TypeAdapter
from pydantic_ai import (
    AgentSpec,
    BinaryContent,
    DeferredToolRequests,
    DeferredToolResults,
    EnqueuedMessagesEvent,
    ToolDenied,
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
from pydantic_ai.run import AgentRun
from pydantic_core import to_jsonable_python
from rich.table import Table
from rich.text import Text
from ya_agent_sdk.agents.main import AgentRuntime
from ya_agent_sdk.context import (
    PROJECT_GUIDANCE_TAG,
    USER_RULES_TAG,
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
    ModelRequestCompleteEvent,
    ModelRequestStartEvent,
    NamespaceStatus,
    NamespaceStatusEvent,
    NoteEvent,
    SubagentCompleteEvent,
    SubagentStartEvent,
    TaskEvent,
    ToolCallsStartEvent,
    UsageSnapshotEvent,
)
from ya_agent_sdk.subagents import (
    DelegationCapability,
    SubagentDeliveryState,
    SubagentExecutionMode,
    SubagentExecutionRecord,
    SubagentExecutionService,
    SubagentExecutionState,
)
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
from yaacli.clipboard import ClipboardImageReadResult, read_clipboard_image
from yaacli.config import ConfigManager, YaacliConfig
from yaacli.display import EventRenderer, RichRenderer, ToolMessage
from yaacli.display_replay import BoundedDisplayReplay
from yaacli.durable.application import (
    SessionApplicationService,
    build_runtime_descriptor,
)
from yaacli.durable.executor import LocalExecutionWorker
from yaacli.durable.models import (
    ActionBatch,
    InputState,
    LogicalRunRecord,
    LogicalRunStatus,
    RevisionPayload,
    RevisionRecord,
    RuntimeDescriptor,
    SessionRecord,
    SessionSummary,
)
from yaacli.durable.projections import (
    MAX_STEERING_PREVIEW_MESSAGES,
    single_line_steering_preview,
    steering_projection_key,
)
from yaacli.durable.restoration import restore_resumable_state_safely
from yaacli.durable.sqlite import SQLiteSessionStore
from yaacli.durable.subagents import SQLiteSubagentExecutionStore
from yaacli.environment import TUIEnvironment
from yaacli.errors import safe_exception_str as _safe_exception_str
from yaacli.events import GoalCompleteEvent, GoalCompleteReason, GoalIterationEvent
from yaacli.logging import configure_tui_logging, get_logger
from yaacli.model_profiles import (
    ResolvedModelProfile,
    build_model_profiles,
    format_model_profile_choice,
    format_model_profile_label,
    get_startup_model_profile,
    save_selected_model_profile_id,
)
from yaacli.perf import perf_log_report, perf_report, perf_timer
from yaacli.rendering.transcript import BlockId, BoundedTextAccumulator, TranscriptLimits, TranscriptStore
from yaacli.runtime import (
    RuntimeSourceSnapshot,
    build_main_runtime_manifest,
    build_runtime_agent_spec,
    compile_child_plan_manifest,
    compile_runtime_sources,
    create_tui_runtime,
    restore_main_runtime_manifest,
)
from yaacli.session import TUIContext, TUIResumableState
from yaacli.shell_monitor import SHELL_MONITOR_KEY, ShellMonitor, ShellNotification
from yaacli.streaming.subagent_tracker import format_subagent_display_id
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
_DEFAULT_MAX_PENDING_ATTACHMENTS = 8
_DEFAULT_MAX_PENDING_ATTACHMENT_BYTES = 20 * 1024 * 1024
_MAX_RETAINED_TOOL_RESULT_CHARS = 64 * 1024
_MAX_RETAINED_TOOL_ARG_CHARS = 64 * 1024
_SESSION_SELECTOR_MAX_VISIBLE = 8
_SESSION_SELECTOR_MAX_WIDTH = 110
_SESSION_SELECTOR_MIN_WIDTH = 24
_RESIZE_SETTLE_SECONDS = 0.15
_RESIZE_STREAM_RENDER_INTERVAL = 1 / 8
_MEDIUM_STREAM_RENDER_BYTES = 32 * 1024
_LARGE_STREAM_RENDER_BYTES = 128 * 1024
_MEDIUM_STREAM_RENDER_INTERVAL = 0.1
_LARGE_STREAM_RENDER_INTERVAL = 0.2
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


def _format_session_timestamp(value: str | datetime) -> str:
    """Format a session timestamp for the compact selector table."""
    raw_value = value.isoformat() if isinstance(value, datetime) else value
    normalized = _single_line_session_preview(raw_value) or "unknown"
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


@dataclass(slots=True)
class _TUISubagentDeferredResolver:
    app: TUIApp

    async def resolve(
        self,
        record: SubagentExecutionRecord,
        requests: DeferredToolRequests,
    ) -> DeferredToolResults:
        self.app._append_system_output(f"Subagent {record.route} requires user action: {record.execution_id}")
        try:
            return await self.app._request_user_action(
                requests,
                owner=f"subagent:{record.execution_id}",
            )
        finally:
            if not self.app._is_foreground_busy():
                self.app._set_phase(TUIPhase.IDLE)


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
    _skill_toolsets: dict[str, SkillToolset] = field(default_factory=dict, init=False, repr=False)
    _runtime_descriptors_by_profile: dict[str, RuntimeDescriptor] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _runtime_sources: RuntimeSourceSnapshot | None = field(default=None, init=False, repr=False)
    _runtime_behavior_base: str = field(default="", init=False, repr=False)
    _oauth_refresh_supervisor: OAuthRefreshSupervisor | None = field(default=None, init=False, repr=False)
    _durable_store: SQLiteSessionStore | None = field(default=None, init=False, repr=False)
    _execution_worker: LocalExecutionWorker | None = field(default=None, init=False, repr=False)
    _session_service: SessionApplicationService | None = field(default=None, init=False, repr=False)
    _active_logical_run_id: str | None = field(default=None, init=False)

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
    _viewport_cache_key: tuple[int, int, int, int] | None = field(default=None, init=False)
    _output_ansi_cache: ANSI | None = field(default=None, init=False)  # Cached visible ANSI
    _renderer: RichRenderer = field(default_factory=RichRenderer, init=False)
    _event_renderer: EventRenderer = field(default_factory=EventRenderer, init=False)
    _theme: ResolvedTheme = field(default_factory=lambda: fallback_theme("auto"), init=False)
    _theme_terminal_resolved: bool = field(default=False, init=False)
    _display_replay: BoundedDisplayReplay = field(
        default_factory=lambda: BoundedDisplayReplay(config=YAACLI_AGUI_REPLAY_CONFIG), init=False
    )
    _display_adapter: AguiEventAdapter | None = field(default=None, init=False)
    _projected_steering_receipt_keys: set[str] = field(default_factory=set, init=False, repr=False)
    _projected_steering_keys: set[str] = field(default_factory=set, init=False, repr=False)

    # Session
    _session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12], init=False)
    _last_session_input: str | None = field(default=None, init=False)
    _last_session_output: str | None = field(default=None, init=False)
    _session_selector_open: bool = field(default=False, init=False)
    _session_selector_entries: list[SessionSummary] = field(default_factory=list, init=False)
    _session_selector_index: int = field(default=0, init=False)

    # Agent execution
    _agent_task: asyncio.Task[None] | None = field(default=None, init=False)
    _agent_task_logical_run_id: str | None = field(default=None, init=False)
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
    _namespace_status: dict[str, NamespaceStatus] = field(default_factory=dict, init=False)

    # Subagent state tracking: agent_id -> {"line_index": int, "tool_names": list[str]}
    _subagent_states: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)
    _background_subagent_ids: set[str] = field(default_factory=set, init=False)

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
    _invalidate_interval: float = field(default=1 / 15, init=False)  # Terminal-friendly 15fps redraw cadence
    _pending_invalidate_handle: asyncio.TimerHandle | None = field(default=None, init=False, repr=False)

    # Terminal resize coalescing
    _last_terminal_size: tuple[int, int] | None = field(default=None, init=False)
    _resize_active: bool = field(default=False, init=False)
    _pending_resize_settle_handle: asyncio.TimerHandle | None = field(default=None, init=False, repr=False)

    # Streaming render throttle (separate from UI invalidation)
    _last_stream_render_time: float = field(default=0.0, init=False)
    _stream_render_interval: float = field(default=1 / 15, init=False)  # Base Markdown preview cadence
    _pending_stream_render_handle: asyncio.TimerHandle | None = field(default=None, init=False, repr=False)
    _pending_stream_render_deadline: float | None = field(default=None, init=False, repr=False)
    _pending_stream_render_callback: Callable[[], None] | None = field(default=None, init=False, repr=False)

    # HITL (Human-in-the-Loop) approval state
    _hitl_pending: bool = field(default=False, init=False)
    _hitl_owner: str | None = field(default=None, init=False, repr=False)
    _hitl_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _approval_event: asyncio.Event | None = field(default=None, init=False)
    _approval_result: bool | None = field(default=None, init=False)  # True=approve, False=reject
    _approval_reason: str | None = field(default=None, init=False)
    _pending_approvals: list[ToolCallPart] = field(default_factory=list, init=False)
    _current_approval_index: int = field(default=0, init=False)
    _approval_expanded: bool = field(default=False, init=False)
    _approval_kind: str = field(default="approval", init=False)
    _current_deferred_request: ToolCallPart | None = field(default=None, init=False)
    _current_deferred_metadata: dict[str, Any] | None = field(default=None, init=False)

    # Durable background readiness projection
    _shell_notification_task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    _subagent_execution_store: SQLiteSubagentExecutionStore | None = field(default=None, init=False, repr=False)
    _subagent_execution_service: SubagentExecutionService | None = field(default=None, init=False, repr=False)
    _projected_subagent_completion_ids: set[str] = field(default_factory=set, init=False, repr=False)
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
        self._state = TUIState.IDLE if phase is TUIPhase.IDLE else TUIState.RUNNING
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
        """Initialize one durable worker shared by every TUI turn."""
        self._configure_theme(query_terminal=True)
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()
        try:
            return await self._initialize_runtime()
        except BaseException as error:
            await self.__aexit__(type(error), error, error.__traceback__)
            raise

    async def _initialize_runtime(self) -> TUIApp:
        """Build runtime resources under the lifecycle cleanup boundary."""
        logger.debug("Resolved terminal theme: %s (%s)", self._theme.variant, self._theme.source)
        mcp_config = self.config_manager.load_mcp_config()
        capability_plugins = self.config_manager.load_capability_plugin_config()
        capability_catalog = capability_plugins.catalog
        self._active_model_profile = get_startup_model_profile(self.config, self.config_manager.config_dir)
        sources = compile_runtime_sources(
            self.config,
            config_dir=self.config_manager.config_dir,
            include_subagents=True,
        )
        self._runtime_sources = sources
        database_path = self.config_manager.get_session_database_path()
        self._durable_store = SQLiteSessionStore(database_path)
        child_store = SQLiteSubagentExecutionStore(database_path)
        try:
            retained_child_descriptors = child_store.list_referenced_descriptors()
        finally:
            child_store.close_sync()

        configured_profiles = build_model_profiles(self.config)
        runtime_profiles: list[ResolvedModelProfile | None] = [*configured_profiles] or [None]
        descriptors: list[RuntimeDescriptor] = []
        self._runtime_descriptors_by_profile.clear()
        for profile in runtime_profiles:
            child_manifest = compile_child_plan_manifest(
                self.config,
                profile=profile,
                sources=sources,
                retained_descriptors=retained_child_descriptors,
                capability_catalog=capability_catalog,
            )
            main_manifest = build_main_runtime_manifest(
                self.config,
                mcp_config,
                profile=profile,
                sources=sources,
                working_dir=self.working_dir,
                config_dir=self.config_manager.config_dir,
                subagent_default_mode=SubagentExecutionMode.background,
                enable_user_input=self.config.tools.enable_user_input,
                frontend="tui",
                hitl_policy="wait",
            )
            descriptor = build_runtime_descriptor(
                agent_spec=build_runtime_agent_spec(
                    self.config,
                    profile=profile,
                    capability_plugins=capability_plugins,
                ),
                main_plan_manifest=main_manifest,
                child_plan_manifest=child_manifest,
                host_envelope={
                    "schema_version": 3,
                    "workspace_ref": main_manifest.workspace_ref,
                    "model_profile_id": profile.id if profile is not None else None,
                    "frontend": "tui",
                },
            )
            descriptors.append(descriptor)
            self._runtime_descriptors_by_profile[profile.id if profile is not None else "default"] = descriptor

        active_profile_key = self._active_model_profile.id if self._active_model_profile is not None else "default"
        active_descriptor = self._runtime_descriptors_by_profile[active_profile_key]

        async def event_sink(event: StreamEvent) -> None:
            self._handle_execution_stream_event(event)

        def runtime_factory(
            descriptor: RuntimeDescriptor,
            binding_ref: str,
        ) -> AgentRuntime[TUIContext, Any, TUIEnvironment]:
            (
                runtime_config,
                runtime_mcp,
                runtime_profile,
                runtime_sources,
                runtime_workspace,
                runtime_config_dir,
            ) = restore_main_runtime_manifest(
                descriptor.main_plan_manifest,
                current_config=self.config,
                current_mcp_config=mcp_config,
            )
            mode_value = descriptor.main_plan_manifest.subagent_default_mode
            mode = SubagentExecutionMode(mode_value) if mode_value is not None else None
            skill_toolset = SkillToolset(
                toolset_id="skills",
                extra_dir_names=[SHARED_SKILLS_DIR_NAME],
            )
            self._skill_toolsets[descriptor.descriptor_id] = skill_toolset
            return create_tui_runtime(
                config=runtime_config,
                mcp_config=runtime_mcp,
                working_dir=runtime_workspace,
                system_prompt=runtime_sources.system_prompt,
                child_plan_manifest=descriptor.child_plan_manifest,
                config_dir=runtime_config_dir,
                model_profile=runtime_profile,
                subagent_default_mode=mode,
                enable_user_input=descriptor.main_plan_manifest.enable_user_input,
                skill_toolset=skill_toolset,
                durable_binding_ref=binding_ref,
                durable_database_path=database_path,
                subagent_deferred_resolver=_TUISubagentDeferredResolver(self),
                agent_spec=AgentSpec.model_validate(descriptor.agent_spec),
                capability_catalog=capability_catalog,
                agent_name="yaacli_main_v2",
            )

        try:
            self._execution_worker = await LocalExecutionWorker.create(
                store=self._durable_store,
                state_path=database_path,
                active_descriptor=active_descriptor,
                available_descriptors=descriptors,
                runtime_factory=runtime_factory,
                event_sink=event_sink,
                display_projection_provider=lambda: cast(list[JsonValue], self._display_replay.snapshot()),
            )
        except BaseException:
            self._durable_store.close()
            self._durable_store = None
            raise
        self._runtime = cast(
            AgentRuntime[TUIContext, str | DeferredToolRequests, TUIEnvironment],
            self._execution_worker.runtime,
        )
        self._skill_toolset = self._skill_toolsets[active_descriptor.descriptor_id]
        self._subagent_execution_store = SQLiteSubagentExecutionStore(database_path)
        for capability in self._runtime.capabilities:
            if isinstance(capability, DelegationCapability):
                self._subagent_execution_service = capability.service
                break
        self._session_service = SessionApplicationService(self._durable_store, self._execution_worker.coordinator)
        await self._skill_toolset.refresh_context(self._runtime.ctx)
        self._runtime.ctx.injected_context_tags = (
            *self._runtime.ctx.injected_context_tags,
            PROJECT_GUIDANCE_TAG,
            USER_RULES_TAG,
        )

        self._oauth_refresh_supervisor = self._create_oauth_refresh_supervisor()
        if self._oauth_refresh_supervisor is not None:
            await self._oauth_refresh_supervisor.start()
            logger.info(
                "OAuth refresh supervisor started providers=%s",
                sorted(self._oauth_refresh_supervisor.provider_names),
            )

        if self._runtime.ctx.model_cfg.context_window:
            self._context_window_size = self._runtime.ctx.model_cfg.context_window
        self._max_output_lines = self.config.display.max_output_lines
        self._max_output_blocks = self.config.display.max_output_blocks
        self._max_output_bytes = self.config.display.max_output_bytes
        self._max_stream_render_bytes = self.config.display.max_stream_render_bytes
        self._max_prompt_history = self.config.display.max_prompt_history
        self._transcript.configure(self._transcript_limits())
        self._sync_transcript_state()

        logger.info("TUIApp initialized with durable session execution")
        configure_tui_logging(verbose=self.verbose)
        shell_monitor = self._get_shell_monitor()
        if shell_monitor is not None and self._runtime is not None:
            shell_monitor.set_notification_callback(self._on_shell_notification)
            if self._runtime.env.shell is not None:
                shell_monitor.start(self._runtime.env.shell)
        projection_task = asyncio.create_task(
            self._poll_subagent_completion_projection(),
            name="yaacli-subagent-completion-projection",
        )
        self._track_managed_task(projection_task)
        return self

    async def _run_shutdown_stage(self, name: str, operation: Callable[[], Awaitable[Any]]) -> Any:
        """Run one shutdown stage and log its duration for slow-exit diagnosis."""
        started_at = time.monotonic()
        try:
            return await operation()
        finally:
            logger.info("Shutdown stage %s completed in %.3fs", name, time.monotonic() - started_at)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        """Cleanup resources once after the prompt-toolkit run loop has stopped."""
        self._show_shutdown_status("starting shutdown")
        shell_monitor = self._get_shell_monitor()
        if shell_monitor is not None:
            shell_monitor.set_notification_callback(None)
        if self._pending_invalidate_handle is not None:
            self._pending_invalidate_handle.cancel()
            self._pending_invalidate_handle = None
        if self._pending_resize_settle_handle is not None:
            self._pending_resize_settle_handle.cancel()
            self._pending_resize_settle_handle = None
        self._resize_active = False
        self._cancel_pending_stream_render()

        errors: list[BaseException] = []

        async def attempt_shutdown_stage(
            name: str,
            operation: Callable[[], Awaitable[Any]],
            *,
            suppress_mcp_cleanup_errors: bool = False,
        ) -> Any:
            try:
                return await self._run_shutdown_stage(name, operation)
            except BaseException as error:
                if suppress_mcp_cleanup_errors and isinstance(
                    error,
                    RuntimeError | GeneratorExit | BaseExceptionGroup,
                ):
                    # MCP stdio clients can raise these during async-generator
                    # teardown after their useful resources are already closed.
                    logger.debug("Suppressed cleanup error: %s", error)
                else:
                    errors.append(error)
                return None

        result: bool | None = None
        try:
            await attempt_shutdown_stage("agent-task", self._cancel_agent_task)
            await attempt_shutdown_stage("managed-tasks", self._cancel_managed_tasks)
            if self._subagent_execution_store is not None:
                subagent_store = self._subagent_execution_store
                self._subagent_execution_store = None
                await attempt_shutdown_stage("subagent-projection-store", subagent_store.close)
            if self._oauth_refresh_supervisor is not None:
                supervisor = self._oauth_refresh_supervisor
                self._oauth_refresh_supervisor = None
                await attempt_shutdown_stage("oauth-refresh", supervisor.shutdown)

            # Give cancellation callbacks one event-loop turn before closing the
            # runtime resources that they may still reference.
            await asyncio.sleep(0)

            if self._execution_worker is not None:
                worker = self._execution_worker
                self._execution_worker = None
                self._show_shutdown_status("closing durable worker")
                await attempt_shutdown_stage(
                    "durable-worker",
                    worker.close,
                    suppress_mcp_cleanup_errors=True,
                )
                self._runtime = None

            if self._exit_stack:
                stack = self._exit_stack
                self._exit_stack = None
                stack_result = await attempt_shutdown_stage(
                    "application-resources",
                    lambda: stack.__aexit__(exc_type, exc_val, exc_tb),
                )
                if isinstance(stack_result, bool) or stack_result is None:
                    result = stack_result
        finally:
            if self._durable_store is not None:
                self._durable_store.close()
                self._durable_store = None
            self._session_service = None
            perf_log_report()

        self._show_shutdown_status("shutdown complete")
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("Multiple errors occurred during YAACLI shutdown", errors)
        return result

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

    def _observe_terminal_size(self, width: int, height: int) -> None:
        """Coalesce a resize burst and guarantee one settled redraw."""
        current = (width, height)
        if self._last_terminal_size is None:
            self._last_terminal_size = current
            return
        if current == self._last_terminal_size:
            return

        self._last_terminal_size = current
        self._resize_active = True
        if self._pending_stream_render_callback is not None:
            self._request_stream_render(self._pending_stream_render_callback)
        if self._pending_resize_settle_handle is not None:
            self._pending_resize_settle_handle.cancel()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._resize_active = False
            self._pending_resize_settle_handle = None
            return
        self._pending_resize_settle_handle = loop.call_later(
            _RESIZE_SETTLE_SECONDS,
            self._finish_terminal_resize,
        )

    def _finish_terminal_resize(self) -> None:
        """Commit the final frame after terminal dimensions stop changing."""
        self._pending_resize_settle_handle = None
        self._resize_active = False
        if self._pending_stream_render_callback is not None:
            self._request_stream_render(self._pending_stream_render_callback)
        if self._app is not None:
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
            if event_type == "RUN_STARTED":
                # Tool render state is derived from one run's replay events. Reset it
                # at run boundaries so canonical replay can reconstruct every call,
                # including providers that reuse tool call IDs across turns.
                self._tool_messages.clear()
                self._printed_tool_calls.clear()
                self._event_renderer.clear()
                continue
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
            if event_type == "CUSTOM" and event.get("name") == "yaacli.steering_accepted":
                value = event.get("value")
                if isinstance(value, dict):
                    projection_key = value.get("projection_key")
                    text = value.get("text")
                    if (
                        isinstance(projection_key, str)
                        and projection_key not in self._projected_steering_receipt_keys
                        and isinstance(text, str)
                        and text
                    ):
                        self._projected_steering_receipt_keys.add(projection_key)
                        from rich.text import Text as RichText

                        user_text = RichText()
                        user_text.append("> ", style="bold green")
                        user_text.append(text)
                        self._append_block(self._renderer.render(user_text, width=width).rstrip("\n"))
                continue
            if event_type == "CUSTOM" and event.get("name") == "yaacli.steering_applied":
                value = event.get("value")
                if isinstance(value, dict):
                    projection_key = value.get("projection_key")
                    messages = value.get("messages")
                    if (
                        isinstance(projection_key, str)
                        and projection_key not in self._projected_steering_keys
                        and isinstance(messages, list)
                    ):
                        previews = [
                            single_line_steering_preview(message)
                            for message in messages[:MAX_STEERING_PREVIEW_MESSAGES]
                            if isinstance(message, str)
                        ]
                        previews = [preview for preview in previews if preview]
                        if previews:
                            self._projected_steering_keys.add(projection_key)
                            self._append_block(self._event_renderer.render_steering_injected(previews).rstrip("\n"))
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
        self._projected_steering_receipt_keys.clear()
        self._projected_steering_keys.clear()
        # The transcript was cleared, so live tool state must not suppress the
        # corresponding canonical replay events as already rendered.
        self._tool_messages.clear()
        self._printed_tool_calls.clear()
        self._event_renderer.clear()
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
            width = self._get_terminal_width()
            self._observe_terminal_size(width, self._get_terminal_height())
            cache_key = (self._scroll_offset, vh, width, self._output_generation)
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
        self._pending_stream_render_deadline = None
        self._pending_stream_render_callback = None

    def _effective_stream_render_interval(self) -> float:
        """Reduce expensive full-Markdown preview frequency as input grows."""
        retained_bytes = max(
            self._streaming_text_buffer.retained_bytes if self._streaming_text_buffer is not None else 0,
            self._streaming_thinking_buffer.retained_bytes if self._streaming_thinking_buffer is not None else 0,
        )
        interval = self._stream_render_interval
        if retained_bytes >= _LARGE_STREAM_RENDER_BYTES:
            interval = max(interval, _LARGE_STREAM_RENDER_INTERVAL)
        elif retained_bytes >= _MEDIUM_STREAM_RENDER_BYTES:
            interval = max(interval, _MEDIUM_STREAM_RENDER_INTERVAL)
        if self._resize_active:
            interval = max(interval, _RESIZE_STREAM_RENDER_INTERVAL)
        return interval

    def _request_stream_render(self, render: Callable[[], None]) -> None:
        """Render when due and move an existing frame when its cadence changes."""
        now = time.monotonic()
        deadline = self._last_stream_render_time + self._effective_stream_render_interval()
        delay = deadline - now
        if delay <= 0:
            self._cancel_pending_stream_render()
            render()
            return
        if self._pending_stream_render_handle is not None:
            pending_deadline = self._pending_stream_render_deadline
            if pending_deadline is not None and abs(pending_deadline - deadline) < 1e-9:
                return
            self._pending_stream_render_handle.cancel()
            self._pending_stream_render_handle = None
            self._pending_stream_render_deadline = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._pending_stream_render_deadline = deadline
        self._pending_stream_render_callback = render
        self._pending_stream_render_handle = loop.call_later(
            delay,
            self._run_trailing_stream_render,
            render,
        )

    def _run_trailing_stream_render(self, render: Callable[[], None]) -> None:
        """Commit the latest coalesced stream state on the event loop."""
        self._pending_stream_render_handle = None
        self._pending_stream_render_deadline = None
        self._pending_stream_render_callback = None
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

    def _consume_attachment_snapshot(self, attachments: Sequence[PendingAttachment]) -> None:
        """Remove only attachments accepted with an asynchronously classified prompt."""
        accepted = {id(item) for item in attachments}
        self._pending_attachments = [item for item in self._pending_attachments if id(item) not in accepted]
        if not self._pending_attachments:
            self._next_attachment_id = 1

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

    def _add_steering_message(self, message: str) -> bool:
        """Durably accept additional guidance while the agent is running."""
        return self._send_steering_message(message)

    def _get_pending_steering_count(self) -> int:
        """Count persisted active-run inputs not yet applied by Pydantic AI."""
        if self._durable_store is None or self._active_logical_run_id is None:
            return 0
        return sum(
            item.order_index > 0 and item.state.value in {"accepted", "enqueued"}
            for item in self._durable_store.list_inputs(self._active_logical_run_id)
        )

    def _clear_unconsumed_user_steering(self) -> None:
        """Persisted accepted input is never silently discarded."""

    def _send_steering_message(self, message: str) -> bool:
        """Persist steering before acknowledging it in the UI."""
        if self._session_service is None or self._active_logical_run_id is None:
            self._append_system_output("No active agent run accepts steering.")
            return False

        service = self._session_service
        try:
            record = service.accept_input(
                self._active_logical_run_id,
                [message],
                origin="user",
            )
        except Exception as exc:
            self._append_error_output(exc)
            return False

        # Durable acceptance is the user-visible receive boundary. Record it
        # only after persistence succeeds; EnqueuedMessagesEvent later records
        # the distinct injection boundary with the same stable projection key.
        self._record_display_system_event(
            "steering_accepted",
            {
                "projection_key": steering_projection_key(self._session_id, record.input_id),
                "text": message,
            },
        )

        async def dispatch() -> None:
            try:
                await service.dispatch_pending()
            except Exception:
                logger.exception("Durable steering dispatch failed; the accepted input remains pending")

        logger.debug("Durable steering accepted (chars=%d)", len(message))
        self._track_managed_task(asyncio.create_task(dispatch()))
        return True

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

    def _durable_session_infos(self, *, limit: int = 100) -> list[SessionSummary]:
        """Return durable session projections for the selector."""
        if self._session_service is None:
            return []
        return list(self._session_service.list_session_summaries(limit=limit))

    async def _show_session_selector(self) -> None:
        """Open the durable session selector without blocking the event loop."""
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
            entries = self._durable_session_infos()
        except (OSError, ValueError) as exc:
            logger.warning("Failed to list durable sessions: %s", exc)
            self._append_system_output(f"Unable to list sessions: {exc}")
            return
        if not entries:
            self._append_system_output("No sessions found.")
            return

        self._close_model_selector()
        self._session_selector_entries = entries
        self._session_selector_index = next(
            (index for index, entry in enumerate(entries) if entry.session_id == self._session_id),
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
        session_id = self._session_selector_entries[index].session_id
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
            current = "*" if entry.session_id == self._session_id else " "
            session_id = _single_line_session_preview(entry.session_id) or "unknown"
            row = f"{cursor} {current} {_pad_display_text(session_id, session_width)}"
            if show_updated:
                row += f" {_pad_display_text(_format_session_timestamp(entry.updated_at), updated_width)}"
            if show_workspace:
                directory = _single_line_session_preview(entry.workspace_ref) or "unknown"
                row += f" {_pad_display_text(directory, workspace_width)}"
            if index == self._session_selector_index:
                style = "class:session-selector.selection"
            elif entry.session_id == self._session_id:
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
            selected_id = _single_line_session_preview(selected.session_id) or "unknown"
            detail_id = _truncate_display_text(selected_id, max(1, line_width - len("DETAILS  ")))
            lines.extend([
                [("class:session-selector.separator", "─" * line_width)],
                [
                    ("class:session-selector.section", "DETAILS"),
                    ("class:session-selector.detail-id", f"  {detail_id}"),
                ],
                _session_detail_line("Directory", selected.workspace_ref, line_width),
                _session_detail_line("Last input", selected.input_preview, line_width),
                _session_detail_line("Last output", selected.output_preview, line_width),
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

        if self._execution_worker is None:
            raise RuntimeError("Durable worker is not initialized")
        try:
            descriptor = self._runtime_descriptors_by_profile[profile.id]
        except KeyError as exc:
            raise RuntimeError(f"Model profile {profile.id!r} has no registered runtime plan") from exc
        plan = self._execution_worker.activate(descriptor.descriptor_id)
        self._runtime = cast(
            AgentRuntime[TUIContext, str | DeferredToolRequests, TUIEnvironment],
            plan.runtime,
        )
        self._skill_toolset = self._skill_toolsets[descriptor.descriptor_id]
        self._subagent_execution_service = None
        for capability in self._runtime.capabilities:
            if isinstance(capability, DelegationCapability):
                self._subagent_execution_service = capability.service
                break
        await self._skill_toolset.refresh_context(self._runtime.ctx)
        self._runtime.ctx.injected_context_tags = tuple(
            dict.fromkeys((
                *self._runtime.ctx.injected_context_tags,
                PROJECT_GUIDANCE_TAG,
                USER_RULES_TAG,
            ))
        )
        self._active_model_profile = profile
        self._context_window_size = self._runtime.ctx.model_cfg.context_window or 200000

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

    def _get_shell_monitor(self) -> ShellMonitor | None:
        """Return the environment-owned shell monitor when the runtime is entered."""
        if self._runtime is None or self._runtime.env.resources is None:
            return None
        resource = self._runtime.env.resources.get(SHELL_MONITOR_KEY)
        return resource if isinstance(resource, ShellMonitor) else None

    def _get_background_process_count(self) -> int:
        """Return the number of active environment shell processes."""
        try:
            if self._runtime is not None and self._runtime.env.shell is not None:
                return len(self._runtime.env.shell.active_background_processes)
        except RuntimeError:
            pass
        return 0

    def _format_background_label(self) -> str:
        """Format the status label for environment shell processes."""
        process_count = self._get_background_process_count()
        if process_count == 0:
            return ""
        suffix = "proc" if process_count == 1 else "procs"
        return f"BG: {process_count} {suffix}"

    async def _poll_subagent_completion_projection(self) -> None:
        """Project durable background completion readiness into the active session UI."""
        try:
            while True:
                await self._refresh_subagent_completion_projection()
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Durable subagent completion projection stopped")
            raise

    async def _refresh_subagent_completion_projection(self) -> None:
        """Read durable records once without transporting any model input."""
        execution_store = self._subagent_execution_store
        session_store = self._durable_store
        if execution_store is None or session_store is None:
            return
        records = await execution_store.list(owner_scope_id=self._session_id)
        for record in records:
            if record.execution_id in self._projected_subagent_completion_ids:
                continue
            if (
                record.mode is not SubagentExecutionMode.background
                or not record.terminal
                or record.delivery_state is not SubagentDeliveryState.pending
                or record.parent_logical_run_id is None
            ):
                continue
            parent_run = session_store.get_run(record.parent_logical_run_id)
            if parent_run is None or parent_run.session_id != self._session_id:
                continue
            self._projected_subagent_completion_ids.add(record.execution_id)
            self._append_system_output(self._format_subagent_completion(record))

    @staticmethod
    def _format_subagent_completion(record: SubagentExecutionRecord) -> str:
        if record.state is SubagentExecutionState.succeeded:
            status = "completed"
        elif record.state is SubagentExecutionState.cancelled:
            status = "was cancelled"
        else:
            status = "failed"
        return (
            f"Background subagent {record.route} {status}: "
            f"{record.execution_id}. The result is ready for the next agent turn."
        )

    def _on_shell_notification(self, notification: ShellNotification) -> None:
        """Expose shell readiness in the UI and route it through durable input."""
        if self._session_clear_in_progress:
            return
        if notification.kind == "output":
            self._append_system_output(f"Background shell output ready: {notification.process_id}")
        else:
            self._append_system_output(f"Background shell process completed: {notification.process_id}")
        self._route_pending_shell_notifications()

    def _route_pending_shell_notifications(self) -> None:
        """Persist shell readiness into an active run or start one idle turn."""
        monitor = self._get_shell_monitor()
        if monitor is None or self._session_clear_in_progress:
            return
        current_delivery = self._shell_notification_task
        if current_delivery is not None and not current_delivery.done():
            return
        notifications = monitor.pending()
        if not notifications:
            return
        prompt = "\n\n".join(notification.prompt() for notification in notifications)

        if self._active_logical_run_id is not None and self._session_service is not None:
            service = self._session_service
            try:
                service.accept_input(
                    self._active_logical_run_id,
                    [prompt],
                    origin="feature",
                )
            except Exception:
                logger.debug(
                    "Active run closed before shell readiness could be accepted",
                    exc_info=True,
                )
                return
            for notification in notifications:
                monitor.acknowledge(
                    notification.process_id,
                    expected=notification,
                )

            async def dispatch() -> None:
                try:
                    await service.dispatch_pending()
                except Exception:
                    logger.exception("Shell readiness dispatch failed; the accepted input remains pending")
                finally:
                    self._shell_notification_task = None
                    if not self._is_foreground_busy():
                        self._route_pending_shell_notifications()

            task = asyncio.create_task(dispatch(), name="yaacli-shell-notification")
            self._shell_notification_task = task
            self._track_managed_task(task)
            return

        if self._is_foreground_busy():
            return
        if self._launch_agent(prompt, session_input="Background shell readiness"):
            for notification in notifications:
                monitor.acknowledge(notification.process_id, expected=notification)

    def _on_agent_task_done(self, task: asyncio.Task[None]) -> None:
        """Recover interaction state if the owning task exits outside normal cleanup."""
        if self._agent_task is not task:
            if not task.cancelled():
                task.exception()
            return
        logical_run_id = self._agent_task_logical_run_id
        self._agent_task = None
        self._agent_task_logical_run_id = None
        if task.cancelled():
            if (
                logical_run_id is not None
                and self._active_logical_run_id == logical_run_id
                and self._session_service is not None
            ):
                service = self._session_service
                try:
                    service.accept_cancel(logical_run_id, reason="agent_task_cancelled_before_start")
                except Exception:
                    logger.exception("Failed to persist cancellation for an accepted agent turn")
                else:

                    async def dispatch_cancel() -> None:
                        try:
                            await service.dispatch_pending()
                        except Exception:
                            logger.exception("Accepted agent cancellation remains pending dispatch")

                    self._track_managed_task(asyncio.create_task(dispatch_cancel()))
                self._active_logical_run_id = None
            self._run_started_at = None
            self._run_timer_paused_at = None
            self._agent_phase = "idle"
            self._reset_hitl_state(owner="main")
            self._set_phase(TUIPhase.IDLE)
            self._route_pending_shell_notifications()
            return
        exc = task.exception()
        if exc is not None:
            if _is_benign_contextvar_cleanup_error(exc):
                logger.debug(
                    "Suppressed benign ContextVar cleanup error from agent task: %s",
                    _safe_exception_str(exc),
                )
            else:
                logger.error(
                    "Uncaught exception in agent task: %s: %s",
                    type(exc).__name__,
                    _safe_exception_str(exc),
                )
                self._append_error_output(exc)
            self._run_started_at = None
            self._run_timer_paused_at = None
            self._agent_phase = "idle"
            self._reset_hitl_state(owner="main")
            self._set_phase(TUIPhase.IDLE)
        self._route_pending_shell_notifications()

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

    def _accept_agent_turn(
        self,
        user_input: str,
        attachments: Sequence[PendingAttachment] | None,
    ) -> LogicalRunRecord:
        """Serialize and durably accept one foreground turn."""
        if self._session_service is None or self._durable_store is None:
            raise RuntimeError("Durable session service is not initialized")
        if self._durable_store.get_session(self._session_id) is None:
            self._session_service.create_session(str(self.working_dir.resolve()), session_id=self._session_id)
        prompt = self._build_user_prompt(user_input, attachments)
        prompt_items: list[UserContent] = [prompt] if isinstance(prompt, str) else list(prompt)
        content = cast(
            list[JsonValue],
            TypeAdapter(list[UserContent]).dump_python(prompt_items, mode="json"),
        )
        return self._session_service.accept_turn(
            self._session_id,
            content,
            descriptor=self._build_runtime_descriptor(self.runtime),
        )

    def _launch_agent(
        self,
        user_input: str,
        attachments: Sequence[PendingAttachment] | None = None,
        *,
        session_input: str | None = None,
    ) -> bool:
        """Durably accept a turn before claiming the foreground and acknowledging it."""
        if self._agent_task is not None and not self._agent_task.done():
            self._append_system_output("An agent run already owns the foreground.")
            return False
        current_task = asyncio.current_task()
        command_reserved = current_task is not None and self._foreground_command_task is current_task
        if self._is_foreground_busy() and not command_reserved:
            self._append_system_output("Foreground work is already in progress.")
            return False
        try:
            logical_run = self._accept_agent_turn(user_input, attachments)
        except Exception as exc:
            self._append_error_output(exc)
            return False

        self._active_logical_run_id = logical_run.logical_run_id
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
        self._agent_task_logical_run_id = logical_run.logical_run_id
        task.add_done_callback(self._on_agent_task_done)
        return True

    async def _run_agent(
        self,
        user_input: str,
        attachments: Sequence[PendingAttachment] | None = None,
        *,
        session_input: str | None = None,
    ) -> None:
        """Execute one product logical run through the durable application service."""
        if self._session_service is None or self._durable_store is None:
            raise RuntimeError("Durable session service is not initialized")
        if self._run_started_at is None:
            self._run_started_at = time.monotonic()
            self._run_timer_paused_at = None
        self._set_phase(TUIPhase.THINKING)
        self._last_session_input = session_input if session_input is not None else user_input
        self._last_session_output = None
        self._last_snapshot_saved = None
        self._tool_messages.clear()
        self._printed_tool_calls.clear()
        self._subagent_states.clear()
        self._background_subagent_ids.clear()
        self._event_renderer.clear()
        cancelled = False
        reported_error = False
        run_id = uuid.uuid4().hex[:12]
        self._display_adapter = AguiEventAdapter(
            session_id=self._session_id,
            run_id=run_id,
            config=YAACLI_AGUI_ADAPTER_CONFIG,
        )
        self._handle_and_record_display_events([self._display_adapter.build_run_started_event()])

        wait_task: asyncio.Task[Any] | None = None
        try:
            logical_run_id = self._active_logical_run_id
            if logical_run_id is None:
                raise RuntimeError("No durable logical run was accepted for this agent task")
            logical_run = self._durable_store.get_run(logical_run_id)
            if logical_run is None:
                raise RuntimeError(f"Accepted logical run {logical_run_id!r} is unavailable")
            try:
                await self._session_service.dispatch_pending()
            except Exception:
                logger.exception("Initial turn dispatch failed; the accepted run remains pending")
            wait_task = asyncio.create_task(
                self._session_service.wait(logical_run.logical_run_id),
                name=f"yaacli-run:{logical_run.logical_run_id}",
            )
            handled_batches: set[str] = set()
            while not wait_task.done():
                current = self._durable_store.get_run(logical_run.logical_run_id)
                if (
                    current is not None
                    and current.status is LogicalRunStatus.suspended
                    and current.pending_action_batch_id is not None
                    and current.pending_action_batch_id not in handled_batches
                ):
                    batch = self._durable_store.get_action_batch(current.pending_action_batch_id)
                    if batch is None:
                        raise RuntimeError(f"Pending action batch {current.pending_action_batch_id!r} is unavailable")
                    handled_batches.add(batch.batch_id)
                    await self._resolve_durable_action_batch(batch)
                    continue
                await asyncio.sleep(0.05)
            await wait_task
            revision = self._require_run_revision(logical_run.logical_run_id)
            status = self._project_terminal_revision(revision)
            cancelled = status is LogicalRunStatus.cancelled
        except asyncio.CancelledError:
            logical_run_id = self._active_logical_run_id
            if logical_run_id is None:
                raise
            try:
                revision = await self._settle_run_revision_after_cancellation(
                    logical_run_id,
                    wait_task=wait_task,
                )
                status = self._project_terminal_revision(revision)
                cancelled = status is LogicalRunStatus.cancelled
            except Exception as exc:
                if _is_benign_contextvar_cleanup_error(exc):
                    logger.debug(
                        "Suppressed benign ContextVar cleanup error in agent run: %s",
                        _safe_exception_str(exc),
                    )
                else:
                    reported_error = True
                    self._report_durable_run_error(exc)
        except Exception as exc:
            logical_run_id = self._active_logical_run_id
            revision = (
                self._durable_store.get_revision_for_run(logical_run_id)
                if self._durable_store is not None and logical_run_id is not None
                else None
            )
            if revision is not None:
                try:
                    status = self._project_terminal_revision(revision)
                    cancelled = status is LogicalRunStatus.cancelled
                except Exception as terminal_exc:
                    reported_error = True
                    self._report_durable_run_error(terminal_exc)
            elif _is_benign_contextvar_cleanup_error(exc):
                logger.debug(
                    "Suppressed benign ContextVar cleanup error in agent run: %s",
                    _safe_exception_str(exc),
                )
            else:
                reported_error = True
                self._report_durable_run_error(exc)
        finally:
            if wait_task is not None and not wait_task.done():
                wait_task.cancel()
                with contextlib.suppress(BaseException):
                    await wait_task
            self._active_logical_run_id = None
            self._finalize_streaming_text()
            self._finalize_streaming_thinking()
            self._reset_hitl_state(owner="main")
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
            self._set_phase(TUIPhase.IDLE)
            self._display_adapter = None

    def _require_run_revision(self, logical_run_id: str) -> RevisionRecord:
        if self._durable_store is None:
            raise RuntimeError("Durable store is not initialized")
        revision = self._durable_store.get_revision_for_run(logical_run_id)
        if revision is None:
            raise RuntimeError(f"Logical run {logical_run_id!r} terminated without publishing a revision")
        return revision

    async def _settle_run_revision_after_cancellation(
        self,
        logical_run_id: str,
        *,
        wait_task: asyncio.Task[Any] | None,
    ) -> RevisionRecord:
        """Wait through task cancellation until durable terminal truth is committed."""
        if self._session_service is None:
            raise RuntimeError("Durable session service is not initialized")
        settlement_task = wait_task
        while True:
            if settlement_task is None or settlement_task.cancelled():
                settlement_task = asyncio.create_task(
                    self._session_service.wait(logical_run_id),
                    name=f"yaacli-settle:{logical_run_id}",
                )
            try:
                await asyncio.shield(settlement_task)
            except asyncio.CancelledError:
                continue
            except Exception:
                if self._durable_store is None or self._durable_store.get_revision_for_run(logical_run_id) is None:
                    raise
            break
        return self._require_run_revision(logical_run_id)

    def _project_terminal_revision(self, revision: RevisionRecord) -> LogicalRunStatus:
        """Restore and project one canonical durable terminal revision."""
        self._restore_revision(revision)
        self._restore_output_from_display_events(self._display_replay.snapshot())
        self._last_snapshot_saved = True
        status_value = revision.terminal.get("status")
        try:
            status = LogicalRunStatus(status_value)
        except ValueError as exc:
            raise RuntimeError(f"Agent run published unknown terminal status {status_value!r}") from exc

        if status is LogicalRunStatus.cancelled:
            reason = revision.terminal.get("reason")
            if not isinstance(reason, str) or not reason:
                reason = "cancelled"
            if self._display_adapter is not None:
                self._handle_and_record_display_events([
                    self._display_adapter.build_run_custom_event("run_cancelled", {"reason": reason})
                ])
            self._append_output("[Cancelled · durable cancellation recorded]")
            return status

        if status is not LogicalRunStatus.completed:
            detail = revision.terminal.get("error") or revision.terminal.get("reason")
            message = f"Agent run ended with status {status.value}"
            if isinstance(detail, str) and detail:
                message = f"{message}: {detail}"
            raise RuntimeError(message)

        output_value = revision.terminal.get("output")
        if not isinstance(output_value, str):
            raise TypeError("Agent completed without a final text result.")
        self._last_session_output = output_value
        if self._display_adapter is not None:
            self._handle_and_record_display_events([
                self._display_adapter.build_run_finished_event(result={"output_text": output_value})
            ])
        self._notify_turn_complete()
        return status

    def _report_durable_run_error(self, exc: Exception) -> None:
        self._finalize_streaming_text()
        self._finalize_streaming_thinking()
        if self._display_adapter is not None:
            self._handle_and_record_display_events([
                self._display_adapter.build_run_error_event(
                    message=_safe_exception_str(exc),
                    code=type(exc).__name__,
                )
            ])
        if self._last_snapshot_saved is not True:
            self._last_snapshot_saved = self.has_session_data
        self._append_error_output(exc, saved=self._last_snapshot_saved)
        logger.error(
            "Durable agent execution failed",
            exc_info=(type(exc), exc, exc.__traceback__),
        )

    def _build_runtime_descriptor(
        self,
        runtime: AgentRuntime[TUIContext, Any, TUIEnvironment],
    ) -> RuntimeDescriptor:
        if self._execution_worker is None or runtime is not self._execution_worker.runtime:
            raise RuntimeError("The active runtime is not registered by the durable worker")
        return self._execution_worker.descriptor

    def _current_revision(self) -> RevisionRecord:
        if self._durable_store is None:
            raise RuntimeError("Durable store is not initialized")
        session = self._durable_store.get_session(self._session_id)
        if session is None or session.head_revision_id is None:
            raise RuntimeError(f"Session {self._session_id!r} has no committed revision")
        revision = self._durable_store.get_revision(session.head_revision_id)
        if revision is None:
            raise RuntimeError(f"Revision {session.head_revision_id!r} is unavailable")
        return revision

    def _restore_revision(self, revision: RevisionRecord) -> None:
        self._message_history = ModelMessagesTypeAdapter.validate_python(revision.message_history)
        latest_usage = get_latest_request_usage(self._message_history)
        self._current_context_tokens = latest_usage.total_tokens if latest_usage is not None else 0
        state = TUIResumableState.model_validate(revision.resumable_state)
        restore_resumable_state_safely(state, self.runtime.ctx)
        display_events = validate_display_events(revision.display_projection)
        replay = BoundedDisplayReplay(config=YAACLI_AGUI_REPLAY_CONFIG)
        replay.extend_snapshot(display_events)
        self._display_replay = replay

    async def _resolve_durable_action_batch(self, batch: ActionBatch) -> None:
        session_service = self._session_service
        if session_service is None:
            raise RuntimeError("Durable session service is not initialized")
        pending = [item for item in batch.items if item.state.value == "pending"]
        if not pending:
            return
        approvals = [
            TypeAdapter(ToolCallPart).validate_python(item.request)
            for item in pending
            if item.decision_kind == "approval"
        ]
        calls = [
            TypeAdapter(ToolCallPart).validate_python(item.request)
            for item in pending
            if item.decision_kind == "external_result"
        ]
        deferred = DeferredToolRequests(approvals=approvals, calls=calls)
        by_call_id = {item.tool_call_id: item for item in pending}
        approval_ids = {item.tool_call_id for item in pending if item.decision_kind == "approval"}

        async def persist_result(tool_call_id: str, result: object) -> None:
            item = by_call_id[tool_call_id]
            if tool_call_id in approval_ids:
                decision: dict[str, JsonValue]
                if result is True:
                    decision = {"approved": True}
                else:
                    message = result.message if isinstance(result, ToolDenied) else "Tool call denied"
                    decision = {"approved": False, "message": message}
            else:
                content = result.content if isinstance(result, RetryPromptPart) else result
                decision = {"result": cast(JsonValue, to_jsonable_python(content))}
            session_service.accept_action(
                item.action_item_id,
                decision,
                actor="tui-user",
            )

        await self._request_user_action(deferred, on_result=persist_result)
        try:
            await session_service.dispatch_pending()
        except Exception:
            logger.exception("Durable action dispatch failed; accepted decisions remain pending")

    def _reset_hitl_state(self, *, owner: str | None = None) -> None:
        """Reset one matching HITL interaction, or force-reset all host HITL."""
        if owner is not None and self._hitl_owner != owner:
            return
        self._hitl_pending = False
        self._hitl_owner = None
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

    async def _request_user_action(
        self,
        deferred: DeferredToolRequests,
        *,
        owner: str = "main",
        on_result: Callable[[str, object], Awaitable[None]] | None = None,
    ) -> DeferredToolResults:
        """Collect user actions without charging HITL wait time to the run timer."""
        if not deferred.approvals and not deferred.calls:
            return DeferredToolResults()

        async with self._hitl_lock:
            self._hitl_owner = owner
            self._pause_run_timer()
            try:
                if on_result is None:
                    return await self._collect_deferred_user_actions(deferred)
                return await self._collect_deferred_user_actions(deferred, on_result=on_result)
            finally:
                self._resume_run_timer()
                self._reset_hitl_state(owner=owner)

    async def _collect_deferred_user_actions(
        self,
        deferred: DeferredToolRequests,
        *,
        on_result: Callable[[str, object], Awaitable[None]] | None = None,
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
                result: bool | ToolDenied = True
                results.approvals[tool_call.tool_call_id] = True
                confirmation = f"  [Approved: {tool_call.tool_name}]"
            else:
                denial_reason = reason or "User rejected"
                result = ToolDenied(message=denial_reason)
                results.approvals[tool_call.tool_call_id] = result
                confirmation = f"  [Rejected: {tool_call.tool_name} - {denial_reason}]"
            if on_result is not None:
                await on_result(tool_call.tool_call_id, result)
            self._append_output(confirmation)
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
                    call_result: Any = RetryPromptPart(
                        content=USER_INPUT_TIMEOUT_PROMPT,
                        tool_name=tool_call.tool_name,
                        tool_call_id=tool_call.tool_call_id,
                    )
                    confirmation = f"  [Timed out: {tool_call.tool_name}]"
                else:
                    call_result = format_user_question_answers(answers)
                    confirmation = f"  [Answered: {tool_call.tool_name}]"
                results.calls[tool_call.tool_call_id] = call_result
                if on_result is not None:
                    await on_result(tool_call.tool_call_id, call_result)
                self._append_output(confirmation)
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
            call_result = RetryPromptPart(
                content=content,
                tool_name=tool_call.tool_name,
                tool_call_id=tool_call.tool_call_id,
            )
            results.calls[tool_call.tool_call_id] = call_result
            if on_result is not None:
                await on_result(tool_call.tool_call_id, call_result)
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
        display_id = format_subagent_display_id(agent_id, agent_name)

        # Create progress line
        text = Text()
        text.append(f"[{display_id}] ", style="cyan")
        text.append("Running...", style="dim")
        rendered = self._renderer.render(text, width=self._get_terminal_width())

        block_id = self._append_block(rendered.rstrip())
        self._subagent_states[agent_id] = {
            "block_id": block_id,
            "line_index": self._transcript.index_of(block_id),
            "tool_names": [],
            "tool_count": 0,
            "agent_name": agent_name,
            "display_id": display_id,
        }
        self._throttled_invalidate()

    def _handle_subagent_complete(self, event: SubagentCompleteEvent) -> None:
        """Handle subagent complete event - update progress line to summary."""
        agent_id = event.agent_id

        if agent_id not in self._subagent_states:
            # Start event was missed, just show completion
            text = Text()
            display_id = format_subagent_display_id(agent_id, event.agent_name)
            if event.success:
                text.append(f"[{display_id}] ", style="cyan")
                text.append("Done ", style="bold green")
                text.append(f"({event.duration_seconds:.1f}s)", style="dim")
                if event.request_count > 0:
                    text.append(f" | {event.request_count} reqs", style="dim")
            else:
                text.append(f"[{display_id}] ", style="cyan")
                text.append("Failed ", style="bold red")
                text.append(f"({event.duration_seconds:.1f}s)", style="dim")
                if event.error:
                    text.append(f" | {event.error[:50]}", style="dim red")
            rendered = self._renderer.render(text, width=self._get_terminal_width())
            self._append_output(rendered.rstrip())
            return

        state = self._subagent_states[agent_id]
        line_index = state["line_index"]
        display_id = state.get("display_id")
        if not isinstance(display_id, str):
            display_id = format_subagent_display_id(agent_id, event.agent_name)

        # Build summary line
        text = Text()
        if event.success:
            text.append(f"[{display_id}] ", style="cyan")
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
            text.append(f"[{display_id}] ", style="cyan")
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
        display_id = state.get("display_id")
        if not isinstance(display_id, str):
            agent_name = state.get("agent_name")
            display_id = format_subagent_display_id(
                agent_id,
                agent_name if isinstance(agent_name, str) else "subagent",
            )

        # Build progress line
        text = Text()
        text.append(f"[{display_id}] ", style="cyan")
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

    def _handle_execution_stream_event(self, event: StreamEvent) -> None:
        """Apply mode policy before adapting or persisting one execution event."""
        message_event = event.event
        if event.agent_id == "main" and isinstance(message_event, EnqueuedMessagesEvent):
            self._project_applied_steering(message_event)
        if isinstance(message_event, SubagentStartEvent | SubagentCompleteEvent) and (
            message_event.mode == SubagentExecutionMode.background.value
        ):
            self._handle_stream_event(event)
            return
        if event.agent_id in self._background_subagent_ids:
            return
        if self._display_adapter is not None:
            display_events = self._display_adapter.adapt_stream_event(event)
            self._handle_and_record_display_events(display_events)
            self._handle_stream_event(event, render_display=False)
            return
        self._handle_stream_event(event)

    def _handle_stream_event(self, event: StreamEvent, *, render_display: bool = True) -> None:
        """Handle non-display state updates from agent execution."""
        message_event = event.event
        agent_id = event.agent_id

        # Handle durable subagent lifecycle events before ordinary stream parts.
        if isinstance(message_event, SubagentStartEvent):
            if message_event.mode == SubagentExecutionMode.background.value:
                self._background_subagent_ids.add(message_event.agent_id)
                return
            self._handle_subagent_start(message_event)
            return

        if isinstance(message_event, SubagentCompleteEvent):
            if message_event.mode == SubagentExecutionMode.background.value:
                self._background_subagent_ids.discard(message_event.agent_id)
                return
            self._handle_subagent_complete(message_event)
            return

        if agent_id in self._background_subagent_ids:
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

        if isinstance(message_event, ModelRequestCompleteEvent):
            if message_event.context_tokens > 0:
                self._current_context_tokens = message_event.context_tokens
            if message_event.context_window_size > 0:
                self._context_window_size = message_event.context_window_size
            return

        if isinstance(message_event, UsageSnapshotEvent):
            if message_event.snapshot is not None:
                self._session_usage.set_run_snapshot(message_event.snapshot)
            return

        if isinstance(message_event, NamespaceStatusEvent):
            for namespace, status in message_event.namespace_status.items():
                previous_status = self._namespace_status.get(namespace)
                self._namespace_status[namespace] = status
                if status in {NamespaceStatus.skipped, NamespaceStatus.error} and status != previous_status:
                    rendered = self._event_renderer.render_mcp_unavailable(namespace, status)
                    self._append_output(rendered.rstrip())
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

        # Handle SDK lifecycle events for status bar
        elif isinstance(message_event, ModelRequestStartEvent):
            self._agent_phase = "thinking"
            self._set_phase(TUIPhase.THINKING)

        elif isinstance(message_event, ToolCallsStartEvent):
            self._finalize_streaming_text()
            self._finalize_streaming_thinking()
            self._agent_phase = "tools"
            self._set_phase(TUIPhase.TOOL_CALLING)

        self._throttled_invalidate()

    def _project_applied_steering(self, event: EnqueuedMessagesEvent) -> None:
        """Project one applied durable user input into the replayable TUI stream."""
        store = self._durable_store
        logical_run_id = self._active_logical_run_id
        if store is None or logical_run_id is None:
            return
        # The native event is the application confirmation. Product state may
        # still be enqueued until the durable capability's next reconciliation
        # hook, so correlate both sides of that narrow projection window.
        for item in store.list_inputs(
            logical_run_id,
            states=(InputState.enqueued, InputState.applied),
        ):
            projection_key = steering_projection_key(self._session_id, item.input_id)
            if (
                item.order_index <= 0
                or item.origin != "user"
                or item.native_enqueue_id != event.enqueue_id
                or projection_key in self._projected_steering_keys
            ):
                continue
            previews = [
                single_line_steering_preview(content)
                for content in item.content[:MAX_STEERING_PREVIEW_MESSAGES]
                if isinstance(content, str)
            ]
            if previews:
                self._record_display_system_event(
                    "steering_applied",
                    {"projection_key": projection_key, "messages": previews},
                )
            return

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
            return command_name == "/goal" and bool(args)
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
                self._set_phase(TUIPhase.IDLE)
        if not task.cancelled():
            self._route_pending_shell_notifications()

    def _release_direct_shell_task(self, task: asyncio.Future[None]) -> None:
        """Release shell ownership and wake only after the shell task is done."""
        if self._direct_shell_task is not task:
            return
        self._direct_shell_command = None
        self._direct_shell_task = None
        if self.phase in {TUIPhase.SHELL_RUNNING, TUIPhase.CANCELLING}:
            self._set_phase(TUIPhase.IDLE)
        if not task.cancelled():
            self._route_pending_shell_notifications()

    def _schedule_command(self, command: str, *, pending_input: TextArea | None = None) -> None:
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
        task = asyncio.create_task(self._handle_command(command, pending_input=pending_input))
        if reserve_foreground:
            self._foreground_command_task = task
            task.add_done_callback(self._release_foreground_command)
        self._track_managed_task(task)

    def _schedule_skill_or_prompt(
        self,
        text: str,
        attachments: list[PendingAttachment],
        input_area: TextArea,
        compose_snapshot: str,
    ) -> None:
        """Refresh skill discovery before classifying slash-prefixed user input."""
        self._set_phase(TUIPhase.COMMAND_RUNNING)
        task = asyncio.create_task(self._handle_skill_or_prompt(text, attachments, input_area, compose_snapshot))
        self._foreground_command_task = task
        task.add_done_callback(self._release_foreground_command)
        self._track_managed_task(task)

    async def _handle_skill_or_prompt(
        self,
        text: str,
        attachments: list[PendingAttachment],
        input_area: TextArea,
        compose_snapshot: str,
    ) -> None:
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
                launched = self._launch_agent(text, attachments)
            else:
                agent_prompt = format_skill_invocation(skill_invocation, available_skills)
                launched = self._launch_agent(agent_prompt, attachments, session_input=text)
            if launched:
                self._consume_attachment_snapshot(attachments)
                self._add_prompt_history(text)
                current = input_area.buffer.text
                if current.startswith(compose_snapshot):
                    input_area.buffer.text = current[len(compose_snapshot) :].lstrip()
                    input_area.buffer.cursor_position = len(input_area.buffer.text)
                self._append_user_input(text, attachments)
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
                if guidance:
                    if self._add_steering_message(guidance):
                        self._detach_pending_attachment_placeholders()
                        input_area.buffer.reset()
                        self._add_prompt_history(guidance)
                elif self._pending_attachments:
                    self._append_system_output(
                        "Images remain attached for the next agent turn; binary input cannot steer an active run."
                    )
                return

            phase_label = self.phase.name.replace("_", " ").lower()
            self._append_system_output(f"Foreground work is {phase_label}. Please wait or use /cancel.")
            return

        if semantic_text.startswith("/"):
            command_name = semantic_text.split(maxsplit=1)[0].lower()
            if self._is_known_slash_command(command_name):
                if self._command_starts_agent(semantic_text):
                    self._schedule_command(semantic_text, pending_input=input_area)
                else:
                    self._add_prompt_history(semantic_text)
                    self._detach_pending_attachment_placeholders()
                    input_area.buffer.reset()
                    self._schedule_command(semantic_text)
            else:
                # Skill directories can change while the TUI is running. Keep
                # the draft and attachments until the resulting turn is durable.
                attachments = list(self._pending_attachments)
                self._schedule_skill_or_prompt(
                    semantic_text,
                    attachments,
                    input_area,
                    input_area.buffer.text,
                )
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
        attachments = list(self._pending_attachments)
        submitted_text = self._strip_attachment_placeholders(text, attachments)
        if not submitted_text and not attachments:
            input_area.buffer.reset()
            return

        if self._launch_agent(submitted_text, attachments):
            self._consume_pending_attachments()
            self._add_prompt_history(submitted_text)
            input_area.buffer.reset()
            self._append_user_input(submitted_text, attachments)

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
        """Return durable session IDs for contextual /session completion."""
        try:
            return [info.session_id for info in self._durable_session_infos()]
        except (OSError, ValueError):
            logger.debug("Failed to list sessions for completion", exc_info=True)
            return []

    def _clear_transcript(self) -> None:
        """Clear only rendered output, preserving conversation and runtime state."""
        self._reset_output_blocks()
        self._scroll_offset = 0
        self._append_system_output("Transcript cleared. Conversation context is unchanged; use /new to reset it.")

    async def _start_new_session(self) -> None:
        """Tombstone the current product session and create a clean durable one."""
        if self._session_clear_in_progress:
            self._append_system_output("A session switch is already in progress.")
            return
        if self._session_service is None or self._durable_store is None:
            raise RuntimeError("Durable session service is not initialized")
        old_session_id = self._session_id
        old_session = self._durable_store.get_session(old_session_id)
        if old_session is not None:
            if old_session.active_execution_id is not None:
                self._append_system_output("The active run must finish or be cancelled before starting a new session.")
                return
            self._durable_store.tombstone_session(old_session_id)
        new_session = self._session_service.create_session(str(self.working_dir.resolve()))
        await self._clear_session()
        self._session_id = new_session.session_id
        self._display_adapter = None
        self._set_phase(TUIPhase.IDLE)
        self._append_system_output(f"New session {new_session.session_id} started (previous: {old_session_id}).")

    def _cancel_foreground(self) -> None:
        """Request cancellation once without interrupting persistence cleanup."""
        if self.phase == TUIPhase.SAVING:
            self._append_system_output("A durable commit is in progress; waiting for it to finish.")
            return
        if self._active_logical_run_id is not None and self._session_service is not None:
            self._set_phase(TUIPhase.CANCELLING)
            logical_run_id = self._active_logical_run_id
            service = self._session_service

            async def cancel_run() -> None:
                await service.cancel(logical_run_id, reason="user_interrupted")

            self._track_managed_task(asyncio.create_task(cancel_run()))
            self._append_system_output("Cancelling durable agent run...")
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

    async def _handle_command(self, command: str, *, pending_input: TextArea | None = None) -> None:
        """Handle slash commands."""
        try:
            if pending_input is None:
                await self._handle_command_inner(command)
            else:
                await self._handle_command_inner(command, pending_input=pending_input)
        except Exception as e:
            logger.exception("Command failed: %s", command)
            self._append_error_output(e)
        finally:
            self._scroll_to_bottom()
            if self._app:
                self._app.invalidate()

    async def _handle_command_inner(self, command: str, *, pending_input: TextArea | None = None) -> None:
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
                await self._show_agents()
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
                if pending_input is None:
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
                    if self._launch_agent(task):
                        if pending_input is not None:
                            self._add_prompt_history(command)
                            self._detach_pending_attachment_placeholders()
                            pending_input.buffer.reset()
                            self._append_user_input(command)
                        self._append_system_output(
                            f"[Goal] Starting goal mode ({ctx.goal_max_iterations} max iterations). Ctrl+C to stop."
                        )
                    else:
                        ctx.reset_goal()
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
                    # Show the expanded prompt only after its durable turn exists.
                    if pending_input is None:
                        self._append_user_input(prompt)
                        self._launch_agent(prompt)
                    elif self._launch_agent(prompt):
                        self._add_prompt_history(command)
                        self._detach_pending_attachment_placeholders()
                        pending_input.buffer.reset()
                        self._append_user_input(prompt)
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
        ctx.tool_id_wrapper.clear()
        ctx.agent_stream_queues = {}
        ctx.agent_stream_info = {}
        ctx.files_to_inspect = []
        ctx.task_manager = TaskManager()
        ctx.note_manager = NoteManager()
        ctx.tool_proxy.loaded_tools = []
        ctx.tool_proxy.loaded_namespaces = []
        ctx.tool_tags = set()
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
        self._background_subagent_ids.clear()
        self._projected_steering_receipt_keys.clear()
        self._projected_steering_keys.clear()
        self._display_replay.clear()
        self._event_renderer.clear()
        self._reset_pending_attachments()
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
        """Reset frontend and resumable context state without replacing worker authority."""
        if self._session_clear_in_progress:
            return
        self._session_clear_in_progress = True
        try:
            monitor = self._get_shell_monitor()
            if monitor is not None:
                await monitor.reset_session_state()
            self._reset_tui_session_state()
            self._session_usage.clear()
            if self._runtime is not None:
                self._reset_context_session_state(self._runtime.ctx)
        finally:
            self._session_clear_in_progress = False

    def _show_cost(self) -> None:
        """Show token usage summary for the current session."""
        summary = self._session_usage.format_summary()
        self._append_system_output(summary)

    async def _show_agents(self) -> None:
        """Show durable subagent executions linked to the current session."""
        if self._durable_store is None:
            raise RuntimeError("Durable store is not initialized")
        store = SQLiteSubagentExecutionStore(self.config_manager.get_session_database_path())
        try:
            records = list(await store.list(owner_scope_id=self._session_id))
        finally:
            await store.close()
        if not records:
            self._append_system_output("No durable subagent executions for this session.")
            return

        running_states = {"pending", "running", "suspended"}
        running_count = sum(record.state.value in running_states for record in records)
        header = Text(
            f"Durable Subagents ({running_count} active, {len(records) - running_count} terminal)",
            style="bold cyan",
        )
        table = Table(show_header=True, box=None, padding=(0, 2))
        table.add_column("Execution ID", style="dim")
        table.add_column("Route", style="bold")
        table.add_column("Mode")
        table.add_column("Status")
        table.add_column("Elapsed", style="dim")
        table.add_column("Prompt", style="dim")
        now = datetime.now(UTC)
        status_styles = {
            "pending": "cyan",
            "running": "cyan",
            "suspended": "yellow",
            "succeeded": "green",
            "failed": "red",
            "cancelled": "yellow",
            "lost": "red",
        }
        for record in sorted(records, key=lambda item: item.created_at):
            ended_at = record.completed_at or now
            started_at = record.started_at or record.created_at
            elapsed = max(0, int((ended_at - started_at).total_seconds()))
            prompt = (
                record.prompt
                if isinstance(record.prompt, str)
                else json.dumps(record.prompt, ensure_ascii=False, default=str)
            )
            if len(prompt) > 60:
                prompt = prompt[:57] + "..."
            route = f"{record.route} (resume)" if record.resumed_from else record.route
            table.add_row(
                record.execution_id,
                route,
                record.mode.value,
                Text(record.state.value, style=status_styles.get(record.state.value, "")),
                f"{elapsed}s",
                prompt,
            )
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
        """Export the committed durable head as an offline transfer bundle."""
        if not self.has_session_data:
            self._append_system_output("No committed conversation history to dump")
            return
        dump_dir = Path(folder_path or ".yaacli-session").expanduser().resolve()
        try:
            revision = self._current_revision()
            dump_dir.mkdir(parents=True, exist_ok=True)
            (dump_dir / "message_history.json").write_text(
                json.dumps(revision.message_history, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (dump_dir / "context_state.json").write_text(
                json.dumps(revision.resumable_state, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (dump_dir / "display_messages.json").write_text(
                json.dumps(revision.display_projection, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (dump_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "session_id": revision.session_id,
                        "revision_id": revision.revision_id,
                        "working_dir": str(self.working_dir.resolve()),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            self._append_system_output(f"Durable session exported to {dump_dir}")
        except Exception as exc:
            self._append_system_output(f"Error: {exc}")

    async def _load_history(self, folder_path: str, *, target_session_id: str | None = None) -> bool:
        """Import one offline bundle into the durable product store."""
        if target_session_id is not None:
            raise ValueError("Offline import cannot replace an existing session identity")
        if self._session_service is None or self._durable_store is None:
            raise RuntimeError("Durable session service is not initialized")
        load_dir = Path(folder_path).expanduser().resolve()
        history_file = load_dir / "message_history.json"
        state_file = load_dir / "context_state.json"
        display_file = load_dir / "display_messages.json"
        if not load_dir.is_dir() or not history_file.is_file():
            self._append_system_output(f"Offline bundle requires {history_file}")
            return False
        try:
            history = ModelMessagesTypeAdapter.validate_json(history_file.read_bytes())
            history_payload = cast(
                list[JsonValue],
                ModelMessagesTypeAdapter.dump_python(history, mode="json"),
            )
            state_payload: dict[str, JsonValue] = {}
            if state_file.is_file():
                state = TUIResumableState.model_validate_json(state_file.read_text(encoding="utf-8"))
                state_payload = cast(dict[str, JsonValue], state.model_dump(mode="json"))
            display_payload: list[JsonValue] = []
            if display_file.is_file():
                display_payload = cast(
                    list[JsonValue],
                    validate_display_events(json.loads(display_file.read_text(encoding="utf-8"))),
                )
        except Exception as exc:
            self._append_system_output(f"Invalid offline session bundle: {exc}")
            return False

        session = self._durable_store.get_session(self._session_id)
        if session is None:
            session = self._session_service.create_session(str(self.working_dir.resolve()), session_id=self._session_id)
        revision = self._session_service.import_snapshot(
            session.session_id,
            descriptor=build_runtime_descriptor(
                agent_spec={"name": "yaacli_main_v2", "model": "offline-import"},
                host_envelope={
                    "schema_version": 2,
                    "workspace_ref": str(self.working_dir.resolve()),
                    "import_source": str(load_dir),
                },
            ),
            payload=RevisionPayload(
                message_history=history_payload,
                resumable_state=state_payload,
                display_projection=display_payload,
                terminal={"status": "completed", "output": None},
            ),
            source=str(load_dir),
        )
        await self._clear_session()
        self._session_id = session.session_id
        self._restore_revision(revision)
        if revision.display_projection:
            self._restore_output_from_display_events(self._display_replay.snapshot())
        self._append_system_output(f"Offline session imported from {load_dir}")
        return True

    @property
    def session_id(self) -> str:
        """Get the current session ID."""
        return self._session_id

    @property
    def has_session_data(self) -> bool:
        """Whether the current durable session has a committed revision."""
        if self._durable_store is None:
            return False
        session = self._durable_store.get_session(self._session_id)
        return session is not None and session.head_revision_id is not None

    def _list_sessions(self, max_display: int = 20) -> None:
        """List recent durable sessions."""
        sessions = self._durable_session_infos(limit=max_display)
        if not sessions:
            self._append_system_output("No sessions found.")
            return
        table = Table(show_header=True, box=None, padding=(0, 2))
        table.add_column("Session ID", style="cyan")
        table.add_column("Updated", style="dim")
        table.add_column("Working Dir", style="dim")
        for session in sessions:
            marker = f"{session.session_id} *" if session.session_id == self._session_id else session.session_id
            table.add_row(
                marker,
                _format_session_timestamp(session.updated_at),
                session.workspace_ref or "unknown",
            )
        self._append_system_output(f"Sessions (showing latest {len(sessions)}):")
        self._append_output(self._renderer.render(table).rstrip())
        self._append_system_output("Use /session <id> to restore. (* = current session)")

    def _resolve_durable_session(self, session_id: str) -> SessionRecord:
        if self._session_service is None:
            raise RuntimeError("Durable session service is not initialized")
        return self._session_service.resolve_session(session_id)

    async def _load_session(self, session_id: str) -> bool:
        """Restore a durable session by exact ID or unambiguous prefix."""
        try:
            target = self._resolve_durable_session(session_id)
        except (KeyError, ValueError) as exc:
            self._append_system_output(str(exc))
            return False
        if target.active_execution_id is not None:
            self._append_system_output(
                f"Session {target.session_id} has an active execution and cannot be switched in-place."
            )
            return False
        await self._clear_session()
        self._session_id = target.session_id
        if target.head_revision_id is not None:
            if self._durable_store is None:
                raise RuntimeError("Durable store is not initialized")
            revision = self._durable_store.get_revision(target.head_revision_id)
            if revision is None:
                raise RuntimeError(f"Session head revision {target.head_revision_id!r} is unavailable")
            self._restore_revision(revision)
            if revision.display_projection:
                self._restore_output_from_display_events(self._display_replay.snapshot())
            output = revision.terminal.get("output")
            self._last_session_output = output if isinstance(output, str) else None
        self._set_phase(TUIPhase.IDLE)
        self._append_system_output(f"Session {target.session_id} restored.")
        if Path(target.workspace_ref).resolve() != self.working_dir.resolve():
            self._append_system_output(
                f"Workspace warning: session belongs to {target.workspace_ref}; "
                f"current workspace is {self.working_dir}."
            )
        return True

    async def _restore_startup_session(self) -> bool:
        """Restore an explicitly requested session or newest workspace head."""
        if self._session_service is None or self._durable_store is None:
            raise RuntimeError("Durable session service is not initialized")
        if self.initial_session_id:
            if await self._load_session(self.initial_session_id):
                return True
            raise RuntimeError(f"Unable to restore requested session: {self.initial_session_id}")
        if self.config.session.auto_restore:
            workspace = str(self.working_dir.resolve())
            for candidate in self._durable_store.list_sessions(limit=1000):
                if (
                    candidate.workspace_ref == workspace
                    and candidate.head_revision_id is not None
                    and await self._load_session(candidate.session_id)
                ):
                    return True
        session = self._session_service.create_session(str(self.working_dir.resolve()), session_id=self._session_id)
        self._session_id = session.session_id
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
            # Foreground and managed task cleanup is owned by __aexit__. Keeping
            # it in one lifecycle boundary avoids paying each bounded wait twice.
