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


def send_process_tree_signal(
    process: asyncio.subprocess.Process,
    sig: int,
    *,
    process_group_id: int | None = None,
) -> None:
    """Send a signal to the whole process tree when process groups are available."""
    if process.pid is None:
        return

    if os.name == "posix":
        try:
            group_id = process_group_id if process_group_id is not None else os.getpgid(process.pid)
            os.killpg(group_id, sig)
        except ProcessLookupError:
            pass
        return

    with contextlib.suppress(ProcessLookupError):
        process.send_signal(sig)


async def terminate_process_tree(
    process: asyncio.subprocess.Process,
    *,
    timeout: float = 5.0,
    process_group_id: int | None = None,
) -> None:
    """Terminate a live owned tree, escalating only while its leader is retained.

    A numeric POSIX PGID is not an ownership handle after its leader is reaped.
    Callers that require residual-member cleanup must keep a stable guardian
    alive until ``kill_process_tree`` has completed.
    """
    if os.name == "posix" and process_group_id is not None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process_group_id, signal.SIGTERM)
    elif process.returncode is None:
        send_process_tree_signal(process, signal.SIGTERM, process_group_id=process_group_id)

    try:
        await asyncio.wait_for(process.wait(), timeout=timeout)
    except TimeoutError:
        await kill_process_tree(process, process_group_id=process_group_id)


async def kill_process_tree(
    process: asyncio.subprocess.Process,
    *,
    process_group_id: int | None = None,
) -> None:
    """Force kill a process tree and wait for the root process to be reaped."""
    if os.name == "posix":
        group_id = process_group_id
        if group_id is None and process.pid is not None:
            try:
                group_id = os.getpgid(process.pid)
            except ProcessLookupError:
                group_id = None
        if group_id is not None:
            # Only ESRCH means the group is already gone. Permission and other
            # failures leave ownership uncertain and must reach the caller.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(group_id, signal.SIGKILL)
    elif process.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            process.kill()

    with contextlib.suppress(ProcessLookupError):
        await process.wait()
