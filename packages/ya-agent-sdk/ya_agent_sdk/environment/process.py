from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from typing import Any


def process_group_kwargs() -> dict[str, Any]:
    """Return subprocess kwargs that isolate a command tree for lifecycle control."""
    if os.name == "posix":
        return {"start_new_session": True}
    return {}


def send_process_tree_signal(process: asyncio.subprocess.Process, sig: int) -> None:
    """Send a signal to the whole process tree when process groups are available."""
    if process.pid is None:
        return

    if os.name == "posix":
        with contextlib.suppress(ProcessLookupError, OSError):
            os.killpg(os.getpgid(process.pid), sig)
            return

    with contextlib.suppress(ProcessLookupError, OSError):
        process.send_signal(sig)


async def terminate_process_tree(
    process: asyncio.subprocess.Process,
    *,
    timeout: float = 5.0,
) -> None:
    """Terminate a process tree gracefully, then force kill if it keeps running."""
    if process.returncode is not None:
        return

    send_process_tree_signal(process, signal.SIGTERM)
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout)
        return
    except TimeoutError:
        pass

    await kill_process_tree(process)


async def kill_process_tree(process: asyncio.subprocess.Process) -> None:
    """Force kill a process tree and wait for the root process to be reaped."""
    if process.returncode is None:
        if os.name == "posix":
            send_process_tree_signal(process, signal.SIGKILL)
        else:
            with contextlib.suppress(ProcessLookupError, OSError):
                process.kill()
    with contextlib.suppress(ProcessLookupError, OSError):
        await process.wait()
