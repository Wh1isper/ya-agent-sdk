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
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from pydantic_ai import RunContext
from ya_agent_environment import BaseResource
from ya_agent_sdk.context.bus import BusMessage, MessageBus
from ya_agent_sdk.usage import UsageSnapshot

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


@dataclass
class BackgroundTaskInfo:
    """Metadata for a background subagent task."""

    agent_id: str
    subagent_name: str
    prompt: str
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    is_resume: bool = False


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

    def __init__(self) -> None:
        # --- Subagent task tracking ---
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._task_info: dict[str, BackgroundTaskInfo] = {}
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
        """Queue a background notification for delivery when the TUI can wake the main agent."""
        self._pending_messages.append(PendingBackgroundMessage(message=message))

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
        self._pending_messages.append(
            PendingBackgroundMessage(
                message=message,
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
        """Queue a usage snapshot for delivery when the TUI wakes the main agent."""
        self._pending_usage_snapshots.append(snapshot)

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

    @property
    def active_tasks(self) -> dict[str, asyncio.Task[Any]]:
        """Active background tasks, keyed by agent_id (copy)."""
        return dict(self._tasks)

    @property
    def task_infos(self) -> dict[str, BackgroundTaskInfo]:
        """All background task metadata, keyed by agent_id (copy)."""
        return dict(self._task_info)

    def register_task(
        self,
        agent_id: str,
        task: asyncio.Task[Any],
        *,
        subagent_name: str = "",
        prompt: str = "",
        is_resume: bool = False,
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
        """
        self._tasks[agent_id] = task
        self._task_info[agent_id] = BackgroundTaskInfo(
            agent_id=agent_id,
            subagent_name=subagent_name,
            prompt=prompt,
            is_resume=is_resume,
        )
        task.add_done_callback(lambda _t: self._tasks.pop(agent_id, None))
        logger.debug("Registered background task: %s (%s)", agent_id, subagent_name)

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
