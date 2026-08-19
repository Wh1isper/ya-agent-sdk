from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from ya_agent_sdk.environment.local import LocalShell
from yaacli.shell_monitor import ShellMonitor, ShellNotification


async def _wait_for_notification(
    monitor: ShellMonitor,
    process_id: str,
    *,
    kind: str,
    timeout: float = 2.0,
) -> ShellNotification:
    async def wait() -> ShellNotification:
        while True:
            notification = monitor.get_pending(process_id)
            if notification is not None and notification.kind == kind:
                return notification
            await asyncio.sleep(0.01)

    return await asyncio.wait_for(wait(), timeout=timeout)


@pytest.mark.asyncio
async def test_shell_monitor_reports_unread_output(tmp_path: Path) -> None:
    shell = LocalShell(default_cwd=tmp_path, allowed_paths=[tmp_path])
    monitor = ShellMonitor()
    callbacks: list[ShellNotification] = []
    monitor.set_notification_callback(callbacks.append)
    monitor.start(shell, poll_interval=0.01)
    try:
        process_id = await shell.start("echo ready; sleep 1")
        monitor.register(process_id)

        notification = await _wait_for_notification(
            monitor,
            process_id,
            kind="output",
        )

        assert notification.process_id == process_id
        assert notification.command == "echo ready; sleep 1"
        assert "shell_wait" in notification.prompt()
        assert callbacks[-1] == notification
        assert not hasattr(monitor, "_bus")

        monitor.acknowledge(process_id, expected=notification)
        assert monitor.get_pending(process_id) is None
    finally:
        await monitor.close()
        await shell.close()


@pytest.mark.asyncio
async def test_shell_monitor_completion_supersedes_output_notification(tmp_path: Path) -> None:
    shell = LocalShell(default_cwd=tmp_path, allowed_paths=[tmp_path])
    monitor = ShellMonitor()
    monitor.start(shell, poll_interval=0.01)
    try:
        process_id = await shell.start("printf done")
        monitor.register(process_id)

        notification = await _wait_for_notification(
            monitor,
            process_id,
            kind="completion",
        )

        assert notification.kind == "completion"
        assert "completed" in notification.prompt()
        monitor.acknowledge(process_id, expected=notification)
        await shell.wait_process(process_id, timeout=1)
        assert monitor.get_pending(process_id) is None
    finally:
        await monitor.close()
        await shell.close()


@pytest.mark.asyncio
async def test_shell_monitor_reset_terminates_and_discards_session_processes(
    tmp_path: Path,
) -> None:
    shell = LocalShell(default_cwd=tmp_path, allowed_paths=[tmp_path])
    monitor = ShellMonitor()
    monitor.start(shell, poll_interval=0.01)
    try:
        process_id = await shell.start("sleep 30")
        monitor.register(process_id)

        await monitor.reset_session_state()

        assert not shell.active_background_processes
        assert monitor.pending() == ()
        with pytest.raises(KeyError):
            await shell.wait_process(process_id, timeout=0)

        next_process = await shell.start("printf reusable")
        stdout, _stderr, running, exit_code = await shell.wait_process(
            next_process,
            timeout=1,
        )
        assert "reusable" in stdout
        assert not running
        assert exit_code == 0
    finally:
        await monitor.close()
        await shell.close()


def test_shell_notification_prompt_is_structured_and_specific() -> None:
    notification = ShellNotification(
        process_id="proc-1",
        kind="output",
        command="python server.py",
    )

    prompt = notification.prompt()

    assert prompt.startswith("<system-reminder>")
    assert "proc-1" in prompt
    assert "python server.py" in prompt
    assert "timeout_seconds=0" in prompt
