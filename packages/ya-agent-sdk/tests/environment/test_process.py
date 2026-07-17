from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import signal
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import ya_agent_sdk.environment.process as process_module
from ya_agent_sdk.environment.process import (
    kill_process_tree,
    process_group_kwargs,
    send_process_tree_signal,
    terminate_process_tree,
)


def _process_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True

    proc_stat = Path(f"/proc/{pid}/stat")
    if proc_stat.exists():
        try:
            parts = proc_stat.read_text().split()
        except (FileNotFoundError, ProcessLookupError):
            return False
        if len(parts) >= 3 and parts[2] == "Z":
            return False

    return True


async def _wait_until_stopped(pid: int, *, timeout: float = 2.0) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if not _process_is_running(pid):
            return True
        await asyncio.sleep(0.05)
    return not _process_is_running(pid)


async def _wait_for_pidfile(pidfile: Path, *, timeout: float = 2.0) -> int:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        with contextlib.suppress(FileNotFoundError, ValueError):
            raw_pid = pidfile.read_text().strip()
            pid = int(raw_pid)
            if _process_is_running(pid):
                return pid
        await asyncio.sleep(0.05)
    raise AssertionError(f"child PID file was not populated with a running process ID: {pidfile}")


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group signal test")
def test_send_process_tree_signal_propagates_permission_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only a missing process group may be treated as a successful public signal."""
    process = MagicMock(spec=asyncio.subprocess.Process)
    process.pid = 1234
    monkeypatch.setattr(os, "killpg", MagicMock(side_effect=PermissionError("denied")))

    with pytest.raises(PermissionError, match="denied"):
        send_process_tree_signal(process, signal.SIGTERM, process_group_id=1234)


def test_send_process_tree_signal_non_posix_propagates_os_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-POSIX public signals must not report permission failures as success."""
    process = MagicMock(spec=asyncio.subprocess.Process)
    process.pid = 1234
    process.send_signal.side_effect = PermissionError("denied")
    monkeypatch.setattr(process_module.os, "name", "nt")

    with pytest.raises(PermissionError, match="denied"):
        send_process_tree_signal(process, signal.SIGTERM)


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group lifecycle test")
async def test_terminate_process_tree_does_not_signal_reaped_numeric_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Graceful leader completion must not be followed by stale-PGID escalation."""
    process = MagicMock(spec=asyncio.subprocess.Process)
    process.pid = 1234
    process.returncode = None

    async def _wait() -> int:
        process.returncode = 0
        return 0

    process.wait.side_effect = _wait
    killpg = MagicMock()
    monkeypatch.setattr(os, "killpg", killpg)

    await terminate_process_tree(process, process_group_id=1234)

    killpg.assert_called_once_with(1234, signal.SIGTERM)


@pytest.mark.skipif(os.name != "posix", reason="POSIX shell process tree test")
async def test_kill_process_tree_terminates_spawned_child_process(tmp_path: Path) -> None:
    pidfile = tmp_path / "child.pid"
    process = await asyncio.create_subprocess_shell(
        f"sleep 100 & echo $! > {shlex.quote(str(pidfile))}; wait",
        cwd=tmp_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        **process_group_kwargs(),
    )
    child_pid: int | None = None

    try:
        child_pid = await _wait_for_pidfile(pidfile)

        await kill_process_tree(process)

        assert await _wait_until_stopped(child_pid)
    finally:
        if child_pid is not None and _process_is_running(child_pid):
            with contextlib.suppress(ProcessLookupError, OSError):
                os.kill(child_pid, signal.SIGKILL)
        if process.returncode is None:
            process.kill()
            await process.wait()
