"""Lifecycle resource for monitored background shell processes.

The monitor observes the environment shell and reports readiness to the TUI. It
never owns conversation state and never transports model input; the TUI routes
notifications through the durable session inbox.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from ya_agent_environment import BaseResource, Shell, ShellBackgroundResetError

from yaacli.logging import get_logger

logger = get_logger(__name__)

SHELL_MONITOR_KEY = "shell_monitor"
_SHELL_POLL_INTERVAL = 1.0


@dataclass(frozen=True, slots=True)
class ShellNotification:
    """One shell readiness signal awaiting durable delivery."""

    process_id: str
    kind: Literal["output", "completion"]
    command: str | None = None

    def prompt(self) -> str:
        command = f" ({self.command})" if self.command else ""
        if self.kind == "output":
            detail = f"Background shell process {self.process_id}{command} has unread output."
        else:
            detail = f"Background shell process {self.process_id}{command} completed."
        return (
            "<system-reminder>\n"
            f"{detail} Use shell_wait(process_id={self.process_id!r}, timeout_seconds=0) "
            "to inspect the buffered result if needed.\n"
            "</system-reminder>"
        )


class ShellMonitor(BaseResource):
    """Detect unread output and completion for explicitly monitored processes."""

    def __init__(self) -> None:
        self._shell: Shell | None = None
        self._callback: Callable[[ShellNotification], None] | None = None
        self._poll_task: asyncio.Task[None] | None = None
        self._known_active: set[str] = set()
        self._completion_notified: set[str] = set()
        self._monitored_processes: set[str] = set()
        self._output_notified: set[str] = set()
        self._pending: OrderedDict[str, ShellNotification] = OrderedDict()
        self._resetting = False

    def set_notification_callback(
        self,
        callback: Callable[[ShellNotification], None] | None,
    ) -> None:
        """Set the synchronous UI callback for newly ready notifications."""
        self._callback = callback

    def start(
        self,
        shell: Shell,
        *,
        poll_interval: float = _SHELL_POLL_INTERVAL,
    ) -> None:
        """Start observing one environment-owned shell."""
        if self._poll_task is not None and not self._poll_task.done():
            raise RuntimeError("Shell monitor is already running")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        self._shell = shell
        self._known_active = set(shell.active_background_processes)
        self._poll_task = asyncio.create_task(
            self._poll_loop(poll_interval),
            name="yaacli-shell-monitor",
        )

    def register(self, process_id: str) -> None:
        """Observe unread output for a process already started by the shell."""
        if self._resetting:
            raise RuntimeError("Cannot monitor a process while the session is resetting")
        self._monitored_processes.add(process_id)

    def get_pending(self, process_id: str) -> ShellNotification | None:
        """Return a pending notification only while its shell state is still useful."""
        notification = self._pending.get(process_id)
        if notification is None:
            return None
        if self._is_deliverable(notification):
            return notification
        self._pending.pop(process_id, None)
        return None

    def acknowledge(
        self,
        process_id: str,
        *,
        expected: ShellNotification | None = None,
    ) -> None:
        """Mark one unchanged notification as handed to the durable boundary."""
        current = self._pending.get(process_id)
        if expected is None or current == expected:
            self._pending.pop(process_id, None)

    def pending(self) -> tuple[ShellNotification, ...]:
        """Return all currently deliverable notifications in readiness order."""
        return tuple(
            notification
            for process_id in tuple(self._pending)
            if (notification := self.get_pending(process_id)) is not None
        )

    @property
    def is_running(self) -> bool:
        return self._poll_task is not None and not self._poll_task.done()

    async def reset_session_state(self) -> None:
        """Terminate environment shell work and discard old-session readiness."""
        if self._resetting:
            return
        self._resetting = True
        shell = self._shell
        try:
            if shell is not None:
                shell.revoke_session_access()
                try:
                    await shell.reset_background_processes()
                except asyncio.CancelledError:
                    raise
                except ShellBackgroundResetError:
                    raise
                except Exception as exc:
                    raise ShellBackgroundResetError({"shell-backend": exc}) from exc
        finally:
            self._known_active.clear()
            self._completion_notified.clear()
            self._monitored_processes.clear()
            self._output_notified.clear()
            self._pending.clear()
            self._resetting = False

    async def _poll_loop(self, interval: float) -> None:
        try:
            while True:
                await asyncio.sleep(interval)
                self._check_processes()
                self._check_output()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Shell monitor stopped after an unexpected polling failure")
            raise

    def _check_processes(self) -> None:
        shell = self._shell
        if shell is None or self._resetting:
            return
        try:
            active = set(shell.active_background_processes)
            buffers = dict(shell._output_buffers)
        except (AttributeError, RuntimeError):
            logger.debug("Unable to inspect background shell state", exc_info=True)
            return

        completed_from_buffers = {
            process_id for process_id, buffer in buffers.items() if process_id not in active and bool(buffer.completed)
        }
        completed = (self._known_active - active) | completed_from_buffers
        for process_id in completed - self._completion_notified:
            self._completion_notified.add(process_id)
            self._monitored_processes.discard(process_id)
            self._output_notified.discard(process_id)
            self._notify(
                ShellNotification(
                    process_id=process_id,
                    kind="completion",
                    command=self._get_command(process_id),
                )
            )

        known_to_shell = active | set(buffers)
        self._completion_notified.intersection_update(known_to_shell)
        self._known_active = active

    def _check_output(self) -> None:
        shell = self._shell
        if shell is None or self._resetting:
            return
        for process_id in tuple(self._monitored_processes):
            buffer = shell._output_buffers.get(process_id)
            if buffer is None:
                self._monitored_processes.discard(process_id)
                self._output_notified.discard(process_id)
                self._pending.pop(process_id, None)
                continue
            has_output = bool(buffer.stdout or buffer.stderr)
            if has_output and process_id not in self._output_notified:
                self._output_notified.add(process_id)
                self._notify(
                    ShellNotification(
                        process_id=process_id,
                        kind="output",
                        command=self._get_command(process_id),
                    )
                )
            elif not has_output:
                self._output_notified.discard(process_id)
                pending = self._pending.get(process_id)
                if pending is not None and pending.kind == "output":
                    self._pending.pop(process_id, None)

    def _notify(self, notification: ShellNotification) -> None:
        if self._resetting:
            return
        self._pending[notification.process_id] = notification
        self._pending.move_to_end(notification.process_id)
        callback = self._callback
        if callback is None:
            return
        try:
            callback(notification)
        except Exception:
            logger.exception(
                "Shell readiness callback failed for %s",
                notification.process_id,
            )

    def _is_deliverable(self, notification: ShellNotification) -> bool:
        shell = self._shell
        if shell is None:
            return False
        buffer = shell._output_buffers.get(notification.process_id)
        if buffer is None:
            return False
        if notification.kind == "output":
            return bool(buffer.stdout or buffer.stderr)
        return bool(buffer.completed)

    def _get_command(self, process_id: str) -> str | None:
        shell = self._shell
        if shell is None:
            return None
        process = shell._background_processes.get(process_id)
        return process.command if process is not None else None

    async def close(self) -> None:
        """Stop monitoring without taking ownership of shell teardown."""
        if self._poll_task is not None:
            self._poll_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._poll_task
            self._poll_task = None
        self._callback = None
        self._shell = None
        self._known_active.clear()
        self._completion_notified.clear()
        self._monitored_processes.clear()
        self._output_notified.clear()
        self._pending.clear()
