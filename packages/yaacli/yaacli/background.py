"""Background monitor for CLI.

BackgroundMonitor is a BaseResource that manages both background subagent
tasks and shell process completion monitoring. It tracks asyncio tasks,
provides callback-based completion notification, holds a reference to the
core toolset for accessing the delegate tool, and polls Shell for process
completions to auto-wake the agent.

Design:
- Callback-based: No polling for subagents. TUI registers a callback that's invoked on completion.
- Polling-based: Shell process completions detected by diffing active_background_processes.
- Event-driven: Background tool calls notify_completion() when done.
- Race-free: TUI callback checks state atomically before scheduling agent turn.

Example:
    # Register with environment
    env.resources.set(BACKGROUND_MONITOR_KEY, BackgroundMonitor())

    # Set core toolset and completion callback after runtime is entered
    monitor.set_core_toolset(runtime.core_toolset)
    monitor.set_completion_callback(on_background_complete)

    # Start shell monitoring
    monitor.start_shell_monitor(shell=env.shell, bus=ctx.message_bus, agent_id="main")

    # Access from tool
    monitor = ctx.deps.resources.get_typed(BACKGROUND_MONITOR_KEY, BackgroundMonitor)
    delegate = monitor.get_delegate_tool()
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import tempfile
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from pydantic_ai import RunContext
from pydantic_ai.usage import RunUsage
from ya_agent_environment import BaseResource
from ya_agent_sdk.context.bus import BusMessage, MessageBus
from ya_agent_sdk.usage import UsageAgentTotal, UsageSnapshot

from yaacli.logging import get_logger

if TYPE_CHECKING:
    from ya_agent_environment.shell import Shell
    from ya_agent_sdk.toolsets.core.base import BaseTool, Toolset

logger = get_logger(__name__)

BACKGROUND_MONITOR_KEY = "background_monitor"
DELEGATE_BACKEND_TOOL_NAME = "__delegate_backend"


@runtime_checkable
class DelegateBackendTool(Protocol):
    """Protocol for unified delegate backend helpers used by background tools."""

    @staticmethod
    def _get_roster_instruction(ctx: RunContext[Any]) -> str | None: ...

    @staticmethod
    def _can_delegate(ctx: RunContext[Any]) -> bool: ...


_SHELL_POLL_INTERVAL = 1.0  # seconds
_SHUTDOWN_BACKGROUND_TASKS_TIMEOUT = 5.0
_DEFAULT_MAX_COMPLETED_TASKS = 100
_DEFAULT_MAX_TASK_PROMPT_CHARS = 4_000
_DEFAULT_MAX_TASK_RESULT_CHARS = 16_000
_TASK_PREVIEW_SUFFIX = "\n… [background preview truncated]"
_MAX_PENDING_MESSAGE_CHARS = 16_000
_MAX_USAGE_SNAPSHOT_GROUPS = 64
_USAGE_OVERFLOW_KEY = "__background_usage_overflow__"


@dataclass
class BackgroundTaskInfo:
    """Metadata for a background subagent task."""

    agent_id: str
    subagent_name: str
    prompt: str
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    is_resume: bool = False
    prompt_truncated: bool = False
    usage_run_id: str | None = None


@dataclass
class BackgroundTaskResult:
    """Cached terminal result for a background subagent task."""

    agent_id: str
    subagent_name: str
    status: Literal["completed", "failed", "cancelled"]
    content: str | None = None
    error: str | None = None
    completed_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    content_truncated: bool = False
    error_truncated: bool = False
    content_size_chars: int | None = None
    error_size_chars: int | None = None


@dataclass
class PendingBackgroundMessage:
    """Queued background notification waiting for TUI-managed delivery."""

    message: BusMessage
    shell_process_id: str | None = None
    shell_kind: Literal["output", "completion"] | None = None


class BackgroundMonitor(BaseResource):
    """Manages background subagent tasks and shell process monitoring.

    This resource has two responsibilities:

    1. **Subagent task tracking** (existing): Tracks asyncio tasks spawned by
       SpawnDelegateTool, provides callback-based completion notification, and
       holds a reference to the core toolset for accessing the delegate tool.

    2. **Shell process monitoring** (new): Polls Shell.active_background_processes
       to detect process completions. On completion, sends a bus message as
       wake-up trigger and invokes the completion callback.

    Lifecycle:
    - Created and registered during TUIEnvironment._setup()
    - core_toolset and completion_callback set after runtime entered (TUIApp.__aenter__)
    - start_shell_monitor() called to begin polling shell processes
    - Tasks registered by SpawnDelegateTool during agent execution
    - notify_completion() called when tasks complete
    - All tasks cancelled on close() (TUIEnvironment._teardown)
    """

    def __init__(
        self,
        *,
        max_completed_tasks: int = _DEFAULT_MAX_COMPLETED_TASKS,
        max_task_prompt_chars: int = _DEFAULT_MAX_TASK_PROMPT_CHARS,
        max_task_result_chars: int = _DEFAULT_MAX_TASK_RESULT_CHARS,
    ) -> None:
        # Completed entries are only evicted after their result has been
        # delivered. This preserves at-least-once delivery semantics while
        # bounding retained prompt/result payloads and delivered history.
        self._max_completed_tasks = max(0, max_completed_tasks)
        self._max_task_prompt_chars = max(0, max_task_prompt_chars)
        self._max_task_result_chars = max(0, max_task_result_chars)
        # --- Subagent task tracking ---
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._task_info: dict[str, BackgroundTaskInfo] = {}
        self._task_results: dict[str, BackgroundTaskResult] = {}
        self._task_result_artifacts: dict[str, tuple[Path | None, Path | None]] = {}
        self._artifact_dir: Path | None = None
        self._completed_task_order: OrderedDict[str, None] = OrderedDict()
        self._delivered_task_results: set[str] = set()
        self._enqueued_task_results: set[str] = set()
        self._waiting_task_results: set[str] = set()
        self._usage_run_task_ids: dict[str, set[str]] = {}
        self._late_usage_run_ids: set[str] = set()
        self._current_usage_run_id: str | None = None
        self._retired_usage_run_ids: set[str] = set()
        self._task_message_tokens: dict[str, str] = {}
        self._core_toolset: Toolset[Any] | None = None
        self._completion_callback: Callable[[str], None] | None = None

        # --- Shell process monitoring ---
        self._shell: Shell | None = None
        self._bus: MessageBus | None = None
        self._agent_id: str | None = None
        self._poll_task: asyncio.Task[None] | None = None
        self._known_active: set[str] = set()

        # --- Output monitoring ---
        self._monitored_processes: set[str] = set()
        self._notified_pending: set[str] = set()

        # --- Wake-up redelivery ---
        self._pending_messages: list[PendingBackgroundMessage] = []
        self._pending_usage_snapshots: list[UsageSnapshot] = []

    # =========================================================================
    # Subagent task management
    # =========================================================================

    def set_core_toolset(self, toolset: Toolset[Any] | None) -> None:
        """Set the core toolset reference for delegate tool access.

        Called by TUIApp after the runtime is entered.

        Args:
            toolset: The core Toolset from AgentRuntime.
        """
        self._core_toolset = toolset

    def set_completion_callback(self, callback: Callable[[str], None] | None) -> None:
        """Set callback to invoke when a background task or shell process completes.

        The callback receives an identifier (agent_id for tasks, process_id for
        shell processes). Called from the asyncio event loop, so it's safe to
        schedule tasks.

        Args:
            callback: Function to call on completion, or None to clear.
        """
        self._completion_callback = callback

    def notify_completion(self, agent_id: str) -> None:
        """Notify that a background task has completed.

        Called by SpawnDelegateTool when a task finishes (success or failure).
        This invokes the registered callback if one exists.

        Args:
            agent_id: The ID of the completed background agent.
        """
        logger.debug("Background task completed: %s", agent_id)
        if self._completion_callback:
            try:
                self._completion_callback(agent_id)
            except Exception:
                logger.exception("Error in completion callback for %s", agent_id)

    def enqueue_message(self, message: BusMessage) -> None:
        """Queue a bounded background notification for TUI-managed delivery."""
        self._enqueue_pending_message(PendingBackgroundMessage(message=_bounded_bus_message(message)))

    def _enqueue_pending_message(self, pending: PendingBackgroundMessage) -> None:
        """Replace a repeated message id instead of retaining duplicate payloads."""
        for index, existing in enumerate(self._pending_messages):
            if existing.message.id == pending.message.id:
                self._pending_messages[index] = pending
                return
        self._pending_messages.append(pending)

    def enqueue_shell_message(
        self,
        message: BusMessage,
        *,
        process_id: str,
        kind: Literal["output", "completion"],
    ) -> None:
        """Queue a shell notification tied to the current shell buffer state.

        Shell wakeups are delivery triggers, while shell output and completion
        results live in Shell output buffers. The buffer can be drained by
        shell_wait() or inject_background_results() before the TUI redelivers
        this notification, so delivery validates that the wakeup is still useful.
        """
        self._enqueue_pending_message(
            PendingBackgroundMessage(
                message=_bounded_bus_message(message),
                shell_process_id=process_id,
                shell_kind=kind,
            )
        )

    def deliver_pending_messages(self, bus: MessageBus, agent_id: str) -> int:
        """Deliver queued background notifications to the message bus.

        The TUI calls this immediately before starting the wake-up turn. This
        avoids losing messages when the main SDK context exits and clears bus state.
        Shell wakeups are dropped if their process output was already drained.
        """
        if not self._pending_messages:
            return 0
        bus.subscribe(agent_id)
        delivered = 0
        remaining: list[PendingBackgroundMessage] = []
        for pending in self._pending_messages:
            message = pending.message
            target = message.target
            if target is not None and target != agent_id:
                remaining.append(pending)
                continue
            if self.is_task_result_delivered(message.source) and pending.shell_process_id is None:
                logger.debug(
                    "Dropping delivered background task notification: source=%s target=%s",
                    message.source,
                    message.target,
                )
                continue
            if not self._is_pending_message_deliverable(pending):
                logger.debug(
                    "Dropping stale background notification: source=%s target=%s process_id=%s kind=%s",
                    message.source,
                    message.target,
                    pending.shell_process_id,
                    pending.shell_kind,
                )
                continue
            bus.send(message)
            is_task_result_notification = (
                pending.shell_process_id is None
                and message.id == self.get_task_result_message_id(message.source)
                and message.source in self._task_results
            )
            if is_task_result_notification:
                self.mark_task_result_enqueued(message.source)
            # Count only notifications that remain unread for this subscriber.
            # The same message may have already been delivered directly to the
            # active run's bus and consumed by inject_bus_messages. In that case
            # this queued copy is only a fallback and must not trigger an empty
            # wake-up turn.
            if any(peeked.id == message.id for peeked in bus.peek(agent_id)):
                delivered += 1
        self._pending_messages = remaining
        return delivered

    def _is_pending_message_deliverable(self, pending: PendingBackgroundMessage) -> bool:
        """Return whether a queued notification still has work for the agent."""
        if pending.shell_process_id is None or pending.shell_kind is None:
            return True
        if self._shell is None:
            return True

        try:
            buf = self._shell._output_buffers.get(pending.shell_process_id)
        except AttributeError:
            return True
        if buf is None:
            return False

        if pending.shell_kind == "output":
            return len(buf.stdout) > 0 or len(buf.stderr) > 0
        if pending.shell_kind == "completion":
            return bool(buf.completed)
        return True

    def enqueue_usage_snapshot(self, snapshot: UsageSnapshot) -> None:
        """Queue a compact usage snapshot for delivery when the TUI wakes the main agent."""
        compacted = _compact_usage_snapshot(snapshot)
        for index, pending in enumerate(self._pending_usage_snapshots):
            if pending.run_id == compacted.run_id:
                self._pending_usage_snapshots[index] = compacted
                return
        self._pending_usage_snapshots.append(compacted)

    def drain_usage_snapshots(self) -> list[UsageSnapshot]:
        """Return and clear queued usage snapshots."""
        snapshots = list(self._pending_usage_snapshots)
        self._pending_usage_snapshots.clear()
        return snapshots

    @property
    def has_pending_messages(self) -> bool:
        """Whether queued background notifications are waiting for TUI delivery."""
        return bool(self._pending_messages)

    @property
    def has_pending_usage_snapshots(self) -> bool:
        """Whether queued usage snapshots are waiting for TUI delivery."""
        return bool(self._pending_usage_snapshots)

    def get_delegate_tool(self, tool_name: str | None = None) -> BaseTool | None:
        """Get the delegate backend tool instance from the core toolset.

        Args:
            tool_name: Preferred backend tool name. When omitted, the hidden
                backend is preferred and the visible blocking delegate is used
                as a compatibility fallback.

        Returns:
            The delegate backend BaseTool instance, or None if not available.
        """
        if self._core_toolset is None:
            return None

        names = (tool_name,) if tool_name else (DELEGATE_BACKEND_TOOL_NAME, "delegate")
        for name in names:
            try:
                return self._core_toolset._get_tool_instance(name)
            except Exception as exc:
                logger.debug("Delegate backend tool not found: %s (%s)", name, exc)
        return None

    def get_delegate_roster_instruction(
        self,
        ctx: RunContext[Any],
        tool_name: str | None = None,
    ) -> str | None:
        """Return available subagent roster text from the delegate backend."""
        delegate = self.get_delegate_tool(tool_name)
        if not isinstance(delegate, DelegateBackendTool):
            return None
        return delegate._get_roster_instruction(ctx)

    def can_delegate(self, ctx: RunContext[Any], tool_name: str | None = None) -> bool:
        """Return whether the delegate backend has at least one callable target."""
        delegate = self.get_delegate_tool(tool_name)
        if delegate is None:
            return False
        if not isinstance(delegate, DelegateBackendTool):
            return True
        return delegate._can_delegate(ctx)

    def has_delegate_backend_tool(self, tool_name: str | None = None) -> bool:
        """Check if a delegate backend tool exists."""
        return self.get_delegate_tool(tool_name) is not None

    @property
    def has_delegate_tool(self) -> bool:
        """Check if a delegate backend tool exists."""
        return self.has_delegate_backend_tool()

    @property
    def has_active_tasks(self) -> bool:
        """Check if there are any active (non-completed) background tasks."""
        return any(not t.done() for t in self._tasks.values())

    def is_task_active(self, agent_id: str) -> bool:
        """Return whether one background agent execution is still running."""
        task = self._tasks.get(agent_id)
        return task is not None and not task.done()

    @property
    def active_tasks(self) -> dict[str, asyncio.Task[Any]]:
        """Active background tasks, keyed by agent_id (copy)."""
        return dict(self._tasks)

    @property
    def task_infos(self) -> dict[str, BackgroundTaskInfo]:
        """All background task metadata, keyed by agent_id (copy)."""
        return dict(self._task_info)

    @property
    def task_results(self) -> dict[str, BackgroundTaskResult]:
        """Terminal background task results, keyed by agent_id (copy)."""
        return dict(self._task_results)

    def record_task_result(self, result: BackgroundTaskResult) -> None:
        """Synchronously cache a result; use the async variant in agent tasks."""
        content, content_truncated, error, error_truncated = self._bound_result(result)
        if content_truncated or error_truncated:
            self._ensure_artifact_dir()
        content_artifact = (
            self._write_task_artifact(result.agent_id, "content", result.content) if content_truncated else None
        )
        error_artifact = self._write_task_artifact(result.agent_id, "error", result.error) if error_truncated else None
        self._store_task_result(
            result,
            content=content,
            error=error,
            content_truncated=content_truncated,
            error_truncated=error_truncated,
            content_artifact=content_artifact,
            error_artifact=error_artifact,
        )

    async def record_task_result_async(self, result: BackgroundTaskResult) -> None:
        """Spool oversized payloads off-loop, then atomically publish previews."""
        content, content_truncated, error, error_truncated = self._bound_result(result)
        if content_truncated or error_truncated:
            self._ensure_artifact_dir()
        writes = [
            asyncio.to_thread(self._write_task_artifact, result.agent_id, "content", result.content)
            if content_truncated
            else _return_none(),
            asyncio.to_thread(self._write_task_artifact, result.agent_id, "error", result.error)
            if error_truncated
            else _return_none(),
        ]
        content_artifact, error_artifact = await asyncio.gather(*writes)
        self._store_task_result(
            result,
            content=content,
            error=error,
            content_truncated=content_truncated,
            error_truncated=error_truncated,
            content_artifact=content_artifact,
            error_artifact=error_artifact,
        )

    def _bound_result(self, result: BackgroundTaskResult) -> tuple[str | None, bool, str | None, bool]:
        content, content_truncated = _bounded_task_text(result.content, self._max_task_result_chars)
        error, error_truncated = _bounded_task_text(result.error, self._max_task_result_chars)
        return content, content_truncated, error, error_truncated

    def _store_task_result(
        self,
        result: BackgroundTaskResult,
        *,
        content: str | None,
        error: str | None,
        content_truncated: bool,
        error_truncated: bool,
        content_artifact: Path | None,
        error_artifact: Path | None,
    ) -> None:
        self._delete_task_artifacts(result.agent_id)
        if content_artifact is not None or error_artifact is not None:
            self._task_result_artifacts[result.agent_id] = (content_artifact, error_artifact)
        self._task_results[result.agent_id] = replace(
            result,
            content=content,
            error=error,
            content_truncated=result.content_truncated or content_truncated,
            error_truncated=result.error_truncated or error_truncated,
            content_size_chars=len(result.content) if result.content is not None else None,
            error_size_chars=len(result.error) if result.error is not None else None,
        )
        self._completed_task_order[result.agent_id] = None
        self._completed_task_order.move_to_end(result.agent_id)
        self._prune_completed_tasks()

    def _ensure_artifact_dir(self) -> Path:
        if self._artifact_dir is None:
            self._artifact_dir = Path(tempfile.mkdtemp(prefix="yaacli-background-results-"))
            self._artifact_dir.chmod(0o700)
        return self._artifact_dir

    def _write_task_artifact(self, agent_id: str, kind: str, value: str | None) -> Path | None:
        if value is None:
            return None
        artifact_dir = self._ensure_artifact_dir()
        path = artifact_dir / f"{uuid.uuid4().hex}-{kind}.txt"
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as artifact:
            artifact.write(value)
        logger.debug("Spooled oversized background result for %s to a temporary artifact", agent_id)
        return path

    def _delete_task_artifacts(self, agent_id: str) -> None:
        artifacts = self._task_result_artifacts.pop(agent_id, None)
        if artifacts is None:
            return
        for path in artifacts:
            if path is not None:
                with contextlib.suppress(OSError):
                    path.unlink(missing_ok=True)
        if not self._task_result_artifacts and self._artifact_dir is not None:
            with contextlib.suppress(OSError):
                self._artifact_dir.rmdir()
                self._artifact_dir = None

    def _materialize_task_result(self, agent_id: str) -> BackgroundTaskResult | None:
        result = self._task_results.get(agent_id)
        if result is None:
            return None
        artifacts = self._task_result_artifacts.get(agent_id)
        if artifacts is None:
            return result
        content_path, error_path = artifacts
        try:
            content = content_path.read_text(encoding="utf-8") if content_path is not None else result.content
            error = error_path.read_text(encoding="utf-8") if error_path is not None else result.error
        except OSError:
            logger.exception("Failed to read spooled background result for %s", agent_id)
            return result
        return replace(result, content=content, error=error, content_truncated=False, error_truncated=False)

    def _prune_completed_tasks(self) -> None:
        """Discard only old *delivered* terminal entries over the retention cap."""
        while len(self._completed_task_order) > self._max_completed_tasks:
            evictable = next(
                (agent_id for agent_id in self._completed_task_order if agent_id in self._delivered_task_results), None
            )
            if evictable is None:
                # An undelivered result must stay reachable even if many tasks
                # finish before the main agent has a chance to wake up.
                return
            self._completed_task_order.pop(evictable, None)
            self._task_results.pop(evictable, None)
            self._delete_task_artifacts(evictable)
            self._task_info.pop(evictable, None)
            self._task_message_tokens.pop(evictable, None)
            self._delivered_task_results.discard(evictable)

    def begin_task_result_wait(self, agent_id: str) -> None:
        """Mark that the main agent is actively waiting for a task result."""
        self._waiting_task_results.add(agent_id)

    def end_task_result_wait(self, agent_id: str) -> None:
        """Clear active wait state for a task result."""
        self._waiting_task_results.discard(agent_id)

    def get_task_result_message_id(self, agent_id: str) -> str:
        """Return a per-execution bus message id for a background task result."""
        token = self._task_message_tokens.get(agent_id)
        suffix = f":{token}" if token is not None else ""
        return f"background-task-result:{agent_id}{suffix}"

    def mark_task_result_enqueued(self, agent_id: str) -> None:
        """Record that a result notification was placed on the active bus."""
        self._enqueued_task_results.add(agent_id)

    def acknowledge_enqueued_task_results(self, bus: MessageBus, target: str) -> None:
        """Acknowledge only notifications no longer unread after an agent turn."""
        unread_message_ids = {message.id for message in bus.peek(target)}
        for agent_id in list(self._enqueued_task_results):
            if self.get_task_result_message_id(agent_id) not in unread_message_ids:
                self.mark_task_result_delivered(agent_id)

    def mark_task_result_delivered(
        self, agent_id: str, bus: MessageBus | None = None, target: str | None = None
    ) -> None:
        """Acknowledge successful result delivery and apply bounded retention.

        This also drops queued fallback notifications and, for wait_subagent,
        marks the stable active-run bus message as consumed to prevent duplicate
        result delivery. Full artifacts remain available until the task entry is
        actually evicted by the completed-task retention policy.
        """
        message_id = self.get_task_result_message_id(agent_id)
        self._enqueued_task_results.discard(agent_id)
        self._delivered_task_results.add(agent_id)
        self._prune_completed_tasks()
        self._pending_messages = [
            pending
            for pending in self._pending_messages
            if pending.shell_process_id is not None or pending.message.id != message_id
        ]
        if bus is not None and target is not None:
            bus.mark_consumed(target, {message_id})

    def is_task_result_delivered(self, agent_id: str) -> bool:
        """Return whether a task result notification was successfully delivered."""
        return agent_id in self._delivered_task_results

    def should_deliver_task_result_message(self, agent_id: str) -> bool:
        """Return whether completion should still be sent through the message bus."""
        return agent_id not in self._delivered_task_results and agent_id not in self._waiting_task_results

    def get_task_result_preview(self, agent_id: str) -> BackgroundTaskResult | None:
        """Return the bounded in-memory result used for notifications/UI."""
        return self._task_results.get(agent_id)

    def get_task_result(self, agent_id: str) -> BackgroundTaskResult | None:
        """Return a terminal result, materializing any spooled full payload."""
        return self._materialize_task_result(agent_id)

    async def wait_for_agent(
        self,
        agent_id: str,
        timeout: float | None = None,
    ) -> BackgroundTaskResult | None:
        """Wait for a background subagent to finish without cancelling it on timeout.

        Args:
            agent_id: Background subagent id to wait for.
            timeout: Maximum seconds to wait. None waits indefinitely.

        Returns:
            Cached terminal result when available; None for timeout or unknown id.
        """
        result = self._materialize_task_result(agent_id)
        if result is not None:
            return result

        task = self._tasks.get(agent_id)
        if task is None:
            return None

        done, _pending = await asyncio.wait({task}, timeout=timeout)
        if not done:
            return None

        result = self._materialize_task_result(agent_id)
        if result is not None:
            return result
        return self._record_missing_task_result(agent_id)

    async def wait_for_agents(
        self,
        agent_ids: list[str],
        timeout: float | None = None,
    ) -> dict[str, BackgroundTaskResult | None]:
        """Wait for multiple background subagents and return cached results by id."""
        results: dict[str, BackgroundTaskResult | None] = {}
        pending_ids = [agent_id for agent_id in agent_ids if agent_id not in self._task_results]

        if pending_ids:
            tasks = [self._tasks[agent_id] for agent_id in pending_ids if agent_id in self._tasks]
            if tasks:
                await asyncio.wait(tasks, timeout=timeout)

        for agent_id in agent_ids:
            result = self._materialize_task_result(agent_id)
            task = self._tasks.get(agent_id)
            if result is None and task is not None and task.done():
                result = self._record_missing_task_result(agent_id)
            results[agent_id] = result
        return results

    def _record_missing_task_result(self, agent_id: str) -> BackgroundTaskResult:
        """Record a failed result for a task that finished without a terminal result."""
        info = self._task_info.get(agent_id)
        result = BackgroundTaskResult(
            agent_id=agent_id,
            subagent_name=info.subagent_name if info is not None else agent_id.rsplit("-bg-", 1)[0],
            status="failed",
            error="Background task finished without recording a result.",
        )
        self.record_task_result(result)
        return result

    def known_task_ids(self) -> list[str]:
        """Return ids for running or cached-result background subagents."""
        return sorted(set(self._tasks) | set(self._task_results))

    def register_task(
        self,
        agent_id: str,
        task: asyncio.Task[Any],
        *,
        subagent_name: str = "",
        prompt: str = "",
        is_resume: bool = False,
        usage_run_id: str | None = None,
    ) -> None:
        """Register a background task for tracking.

        The asyncio task is auto-removed when it completes.
        Task info is preserved for display purposes.

        Args:
            agent_id: Unique identifier for the background subagent.
            task: The asyncio.Task running the subagent.
            subagent_name: Name of the subagent (e.g., "searcher").
            prompt: The prompt sent to the subagent.
            is_resume: Whether this is resuming a previous conversation.
            usage_run_id: Parent usage run that can receive a late cumulative snapshot.
        """
        previous_task = self._tasks.get(agent_id)
        if previous_task is not None:
            if not previous_task.done():
                raise ValueError(f"Background agent {agent_id!r} is already running")
            previous_info = self._task_info.get(agent_id)
            self._on_task_done(
                agent_id,
                previous_info.usage_run_id if previous_info is not None else None,
                previous_task,
            )

        previous_message_id = self.get_task_result_message_id(agent_id)
        self._pending_messages = [
            pending
            for pending in self._pending_messages
            if pending.shell_process_id is not None or pending.message.id != previous_message_id
        ]
        self._delivered_task_results.discard(agent_id)
        self._enqueued_task_results.discard(agent_id)
        self._waiting_task_results.discard(agent_id)
        self._completed_task_order.pop(agent_id, None)
        self._task_results.pop(agent_id, None)
        self._delete_task_artifacts(agent_id)
        self._task_message_tokens[agent_id] = uuid.uuid4().hex
        self._tasks[agent_id] = task
        prompt_preview, prompt_truncated = _bounded_task_text(prompt, self._max_task_prompt_chars)
        self._task_info[agent_id] = BackgroundTaskInfo(
            agent_id=agent_id,
            subagent_name=subagent_name,
            prompt=prompt_preview or "",
            is_resume=is_resume,
            prompt_truncated=prompt_truncated,
            usage_run_id=usage_run_id,
        )
        if usage_run_id is not None:
            self.observe_usage_run(usage_run_id)
            self._usage_run_task_ids.setdefault(usage_run_id, set()).add(agent_id)
            self._late_usage_run_ids.add(usage_run_id)
        task.add_done_callback(lambda completed: self._on_task_done(agent_id, usage_run_id, completed))
        logger.debug("Registered background task: %s (%s)", agent_id, subagent_name)

    def _on_task_done(
        self,
        agent_id: str,
        usage_run_id: str | None,
        completed_task: asyncio.Task[Any],
    ) -> None:
        if self._tasks.get(agent_id) is not completed_task:
            return
        self._tasks.pop(agent_id, None)
        if usage_run_id is None:
            return
        task_ids = self._usage_run_task_ids.get(usage_run_id)
        if task_ids is None:
            return
        task_ids.discard(agent_id)
        if not task_ids:
            self._usage_run_task_ids.pop(usage_run_id, None)
            if usage_run_id != self._current_usage_run_id:
                self._retired_usage_run_ids.add(usage_run_id)

    def observe_usage_run(self, run_id: str) -> None:
        """Advance the active context usage run and retire inactive predecessors."""
        if run_id == self._current_usage_run_id:
            return
        previous_run_id = self._current_usage_run_id
        self._current_usage_run_id = run_id
        self._retired_usage_run_ids.discard(run_id)
        if previous_run_id is not None and not self.has_tasks_for_usage_run(previous_run_id):
            self._retired_usage_run_ids.add(previous_run_id)

    def drain_retired_usage_run_ids(self) -> set[str]:
        """Return retired runs only after their final snapshots are drained."""
        pending_run_ids = {snapshot.run_id for snapshot in self._pending_usage_snapshots}
        run_ids = {
            run_id
            for run_id in self._retired_usage_run_ids
            if not self.has_tasks_for_usage_run(run_id) and run_id not in pending_run_ids
        }
        self._retired_usage_run_ids.difference_update(run_ids)
        self._late_usage_run_ids.difference_update(run_ids)
        return run_ids

    def has_tasks_for_usage_run(self, run_id: str) -> bool:
        """Return whether a usage run still has a background producer."""
        return bool(self._usage_run_task_ids.get(run_id))

    def can_publish_late_usage_snapshot(self, run_id: str) -> bool:
        """Return whether this context run may spawn another cumulative update."""
        return run_id in self._late_usage_run_ids

    def get_context_instruction(self) -> str | None:
        """Return context instruction about active background tasks.

        Returns XML describing running background tasks, or None if none active.
        """
        running = [(aid, t) for aid, t in self._tasks.items() if not t.done()]
        if not running:
            return None

        lines = ["<background-tasks>"]
        for agent_id, _ in running:
            lines.append(f'  <task agent-id="{agent_id}" status="running"/>')
        lines.append("</background-tasks>")
        return "\n".join(lines)

    async def wait_for_all(self, timeout: float | None = None) -> None:
        """Wait for all background tasks to complete.

        Args:
            timeout: Maximum seconds to wait. None for no timeout.
        """
        tasks = list(self._tasks.values())
        if not tasks:
            return
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout,
            )

    # =========================================================================
    # Output monitoring for shell processes
    # =========================================================================

    def register_monitored_process(self, process_id: str) -> None:
        """Register a shell process for output monitoring.

        When the process has new unread output in its OutputBuffer, a bus
        message and completion callback are triggered to wake up the agent.
        After the agent drains the output (e.g. via shell_wait), the
        notification state resets so the next batch of output triggers
        another notification.

        The process must already exist in Shell._output_buffers.

        Args:
            process_id: The process ID returned by shell.start().
        """
        self._monitored_processes.add(process_id)
        logger.debug("Registered process for output monitoring: %s", process_id)

    def _check_monitored_output(self) -> None:
        """Check monitored processes for new unread output.

        For each monitored process:
        - If output buffer has data and not yet notified: send notification,
          mark as notified_pending.
        - If output buffer is empty and was notified: clear notified_pending
          (output was drained, ready for next notification).
        - If process completed: remove from monitored set (handled by _check_shell).
        """
        if self._shell is None:
            return

        for pid in list(self._monitored_processes):
            buf = self._shell._output_buffers.get(pid)
            if buf is None:
                # Buffer removed (process consumed or killed) -- stop monitoring
                self._monitored_processes.discard(pid)
                self._notified_pending.discard(pid)
                continue

            has_output = len(buf.stdout) > 0 or len(buf.stderr) > 0

            if has_output and pid not in self._notified_pending:
                # New output detected -- notify and mark pending
                self._notify_monitored_output(pid)
                self._notified_pending.add(pid)
            elif not has_output and pid in self._notified_pending:
                # Output was drained -- ready for next notification
                self._notified_pending.discard(pid)

    def _notify_monitored_output(self, process_id: str) -> None:
        """Send bus message and invoke callback for new output on a monitored process."""
        if self._bus is not None and self._agent_id is not None:
            from ya_agent_sdk.context.bus import BusMessage

            command = self._get_process_command(process_id)
            content = f"Background shell process has new output: {process_id}"
            if command:
                content += f" ({command})"
            content += ". Use shell_wait(process_id, timeout_seconds=0) to read it."

            self.enqueue_shell_message(
                BusMessage(
                    content=content,
                    source="shell-monitor",
                    target=self._agent_id,
                ),
                process_id=process_id,
                kind="output",
            )

        if self._completion_callback:
            try:
                self._completion_callback(process_id)
            except Exception:
                logger.exception("Error in completion callback for monitored output %s", process_id)

    # =========================================================================
    # Shell process completion monitoring
    # =========================================================================

    def start_shell_monitor(
        self,
        shell: Shell,
        bus: MessageBus,
        agent_id: str,
        *,
        poll_interval: float = _SHELL_POLL_INTERVAL,
    ) -> None:
        """Start polling shell for background process completions.

        Takes a snapshot of currently active processes as the baseline,
        then starts a polling loop that detects new completions.

        Args:
            shell: The Shell instance to monitor.
            bus: The MessageBus for sending wake-up notifications.
            agent_id: The agent ID to target bus messages to (usually "main").
            poll_interval: Seconds between polls (default: 1.0).
        """
        if self._poll_task is not None:
            logger.warning("Shell monitor already running, ignoring start_shell_monitor()")
            return

        self._shell = shell
        self._bus = bus
        self._agent_id = agent_id

        # Snapshot current active processes as baseline
        self._known_active = set(shell.active_background_processes.keys())

        self._poll_task = asyncio.create_task(self._poll_loop(poll_interval))
        logger.info(
            "Shell monitor started (poll_interval=%.1fs, baseline=%d processes)",
            poll_interval,
            len(self._known_active),
        )

    async def _poll_loop(self, interval: float) -> None:
        """Background polling loop for shell process completions and output monitoring."""
        try:
            while True:
                await asyncio.sleep(interval)
                self._check_shell()
                self._check_monitored_output()
        except asyncio.CancelledError:
            logger.debug("Shell monitor polling loop cancelled")
            raise

    def _check_shell(self) -> None:
        """Check shell for process completions and notify.

        Compares current active_background_processes with known_active set.
        - New processes (in current but not known): add to known set
        - Completed processes (in known but not current): send notification
        """
        if self._shell is None:
            return

        try:
            current_active = set(self._shell.active_background_processes.keys())
        except Exception:
            logger.debug("Failed to read active_background_processes", exc_info=True)
            return

        # Detect new processes
        new_pids = current_active - self._known_active
        for pid in new_pids:
            logger.debug("Shell monitor: new process detected: %s", pid)

        # Detect completed processes
        completed_pids = self._known_active - current_active
        for pid in completed_pids:
            logger.info("Shell monitor: process completed: %s", pid)
            # Stop output monitoring for completed processes
            self._monitored_processes.discard(pid)
            self._notified_pending.discard(pid)
            self._notify_shell_completion(pid)

        # Update known set
        self._known_active = current_active

    def _notify_shell_completion(self, process_id: str) -> None:
        """Send bus message and invoke callback for a completed shell process.

        The bus message serves as a wake-up trigger. Actual stdout/stderr/exit_code
        data is consumed by inject_background_results filter via shell tools.
        """
        # Send bus message as wake-up notification
        if self._bus is not None and self._agent_id is not None:
            from ya_agent_sdk.context.bus import BusMessage

            # Try to get command name from shell for a more informative message
            command = self._get_process_command(process_id)
            content = f"Background shell process completed: {process_id}"
            if command:
                content += f" ({command})"

            self.enqueue_shell_message(
                BusMessage(
                    content=content,
                    source="shell-monitor",
                    target=self._agent_id,
                ),
                process_id=process_id,
                kind="completion",
            )

        # Invoke completion callback (same as subagent completion)
        if self._completion_callback:
            try:
                self._completion_callback(process_id)
            except Exception:
                logger.exception("Error in completion callback for shell process %s", process_id)

    def _get_process_command(self, process_id: str) -> str | None:
        """Try to get the command string for a process from shell metadata."""
        if self._shell is None:
            return None
        try:
            # Check _background_processes dict which keeps metadata even after task completes
            proc = self._shell._background_processes.get(process_id)
            if proc is not None:
                return proc.command
        except AttributeError:
            pass
        return None

    @property
    def is_shell_monitor_running(self) -> bool:
        """Check if the shell monitoring loop is active."""
        return self._poll_task is not None and not self._poll_task.done()

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def close(self) -> None:
        """Cancel all background tasks, stop shell monitor, and clean up."""
        # Stop shell monitor
        if self._poll_task is not None:
            self._poll_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._poll_task
            self._poll_task = None
            logger.debug("Shell monitor stopped")

        # Cancel subagent tasks
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            _done, pending = await asyncio.wait(tasks, timeout=_SHUTDOWN_BACKGROUND_TASKS_TIMEOUT)
            if pending:
                logger.warning(
                    "%d background task(s) did not finish within %.1fs during shutdown",
                    len(pending),
                    _SHUTDOWN_BACKGROUND_TASKS_TIMEOUT,
                )
            logger.debug("Cancelled %d background tasks", len(tasks))

        # Clear all state
        self._tasks.clear()
        self._task_info.clear()
        self._task_results.clear()
        self._task_result_artifacts.clear()
        if self._artifact_dir is not None:
            with contextlib.suppress(OSError):
                for path in self._artifact_dir.iterdir():
                    path.unlink(missing_ok=True)
                self._artifact_dir.rmdir()
            self._artifact_dir = None
        self._completed_task_order.clear()
        self._delivered_task_results.clear()
        self._enqueued_task_results.clear()
        self._waiting_task_results.clear()
        self._usage_run_task_ids.clear()
        self._late_usage_run_ids.clear()
        self._current_usage_run_id = None
        self._retired_usage_run_ids.clear()
        self._task_message_tokens.clear()
        self._core_toolset = None
        self._completion_callback = None
        self._shell = None
        self._bus = None
        self._agent_id = None
        self._known_active.clear()
        self._monitored_processes.clear()
        self._notified_pending.clear()
        self._pending_messages.clear()
        self._pending_usage_snapshots.clear()


async def _return_none() -> None:
    return None


def _bounded_task_text(value: str | None, limit: int) -> tuple[str | None, bool]:
    """Return a bounded in-memory preview and whether data was omitted."""
    if value is None or len(value) <= limit:
        return value, False
    if limit <= len(_TASK_PREVIEW_SUFFIX):
        return value[:limit], True
    return value[: limit - len(_TASK_PREVIEW_SUFFIX)] + _TASK_PREVIEW_SUFFIX, True


def _bounded_bus_message(message: BusMessage) -> BusMessage:
    """Bound queued text while retaining the message id/source/target contract."""
    if isinstance(message.content, str):
        content, truncated = _bounded_task_text(message.content, _MAX_PENDING_MESSAGE_CHARS)
        if not truncated:
            return message
        return message.model_copy(update={"content": content})
    # Background notifications are textual in normal operation. Bound a
    # malformed multimodal sequence too, while leaving the first items intact.
    content = list(islice(message.content, 32))
    if len(content) == len(message.content):
        return message
    content.append(_TASK_PREVIEW_SUFFIX)
    return message.model_copy(update={"content": content})


def _compact_usage_snapshot(snapshot: UsageSnapshot) -> UsageSnapshot:
    """Bound group cardinality while preserving every snapshot's numeric total."""
    max_direct = _MAX_USAGE_SNAPSHOT_GROUPS - 1
    agent_items = list(snapshot.agent_usages.items())
    model_items = list(snapshot.model_usages.items())
    agents = dict(agent_items[:max_direct])
    models = {model_id: _bounded_run_usage(usage) for model_id, usage in model_items[:max_direct]}

    if len(agent_items) > max_direct:
        agent_overflow = RunUsage()
        for _agent_id, entry in agent_items[max_direct:]:
            agent_overflow.incr(_bounded_run_usage(entry.usage))
        agents[_USAGE_OVERFLOW_KEY] = UsageAgentTotal(
            agent_name="background overflow",
            model_id=_USAGE_OVERFLOW_KEY,
            usage=agent_overflow,
        )
    if len(model_items) > max_direct:
        model_overflow = RunUsage()
        for _model_id, usage in model_items[max_direct:]:
            model_overflow.incr(_bounded_run_usage(usage))
        models[_USAGE_OVERFLOW_KEY] = model_overflow

    # entries are redundant with agent_usages for SessionUsage and can otherwise
    # carry one object per event/source for a long background run.
    bounded_agents = {
        agent_id: entry.model_copy(update={"usage": _bounded_run_usage(entry.usage)})
        for agent_id, entry in agents.items()
    }
    return snapshot.model_copy(update={"entries": [], "agent_usages": bounded_agents, "model_usages": models})


def _bounded_run_usage(usage: RunUsage) -> RunUsage:
    """Keep usage totals while bounding provider-specific detail cardinality."""
    compacted = RunUsage() + usage
    if len(compacted.details) <= _MAX_USAGE_SNAPSHOT_GROUPS:
        return compacted
    detail_items = list(compacted.details.items())
    retained = dict(detail_items[: _MAX_USAGE_SNAPSHOT_GROUPS - 1])
    retained[_USAGE_OVERFLOW_KEY] = sum(value for _key, value in detail_items[_MAX_USAGE_SNAPSHOT_GROUPS - 1 :])
    compacted.details = retained
    return compacted
