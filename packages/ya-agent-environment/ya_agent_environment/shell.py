"""Shell abstraction for environment module.

This module provides an abstract base class for shell command execution,
including support for background process management with streaming output
via OutputBuffer.
"""

import asyncio
import contextlib
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from html import escape as _xml_escape
from pathlib import Path
from typing import Protocol
from uuid import uuid4

# --- OutputBuffer limits ---
# Max lines per stream (stdout/stderr) retained in the bounded deque.
_MAX_BUFFER_LINES = 200
# Max characters per line before truncation (guards against binary/minified blobs).
_MAX_LINE_LENGTH = 4096

# --- Completed results limits (for filter consumption) ---
# Per-process cap on output bytes when constructing CompletedProcess for the filter.
_MAX_COMPLETED_OUTPUT_BYTES = 1 * 1024 * 1024  # 1 MB per stream
# Max queued CompletedProcess results returned by consume_completed_results().
_MAX_COMPLETED_RESULTS = 50


def _truncate_to_bytes(text: str, max_bytes: int) -> tuple[str, bool]:
    """Truncate a string so its UTF-8 encoding fits within max_bytes.

    Returns (truncated_text, was_truncated). Uses binary slice then
    decodes back, ignoring partial chars at the boundary.
    """
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text, False
    truncated = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return truncated, True


def _truncate_line(line: str, max_length: int = _MAX_LINE_LENGTH) -> str:
    """Truncate a single line to max_length characters."""
    if len(line) <= max_length:
        return line
    return line[:max_length]


class ReadableStream(Protocol):
    """Protocol for an async readable byte stream.

    Matches asyncio.StreamReader.readline() and can be implemented
    by any async byte-line source (subprocess pipes, network streams, etc.).
    """

    async def readline(self) -> bytes: ...


class WritableStream(Protocol):
    """Protocol for an async writable byte stream.

    Used for stdin access to background processes.  Implementations
    should flush/drain on write and handle broken pipes gracefully.
    """

    async def write(self, data: bytes) -> None: ...

    async def close(self) -> None: ...


class StdinAdapter:
    """WritableStream adapter for asyncio.StreamWriter (subprocess stdin).

    Wraps an asyncio.StreamWriter to implement the WritableStream protocol,
    with graceful handling of broken pipes when the process has exited.
    """

    def __init__(self, writer: asyncio.StreamWriter) -> None:
        self._writer = writer
        self._closed = False

    async def write(self, data: bytes) -> None:
        """Write data and flush.  Silently ignores writes after close.

        On pipe/connection errors, marks the adapter as closed to
        prevent further futile writes, then re-raises so the caller
        knows the write failed (e.g., process exited, SSH dropped).
        """
        if self._closed:
            return
        try:
            self._writer.write(data)
            await self._writer.drain()
        except (BrokenPipeError, ConnectionResetError, OSError):
            self._closed = True
            raise

    async def close(self) -> None:
        """Close the stdin stream (sends EOF to the process)."""
        if self._closed:
            return
        self._closed = True
        try:
            self._writer.close()
            await self._writer.wait_closed()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass  # Process already exited


@dataclass
class ExecutionHandle:
    """Handle returned by _create_process() for the ABC to manage.

    Provides stream access and lifecycle control over a background process.
    The ABC's concrete start() uses this to set up reader tasks and manage
    the process lifecycle uniformly.

    For subprocess-based backends, stdout/stderr are the subprocess pipe
    StreamReaders directly.  For non-streaming backends (Docker, RPC),
    use asyncio.StreamReader as an adapter with feed_data/feed_eof.
    """

    stdout: ReadableStream
    stderr: ReadableStream
    wait: Callable[[], Awaitable[int]]
    kill: Callable[[], Awaitable[None]]
    stdin: WritableStream | None = None
    pid: int | None = None
    send_signal: Callable[[int], Awaitable[None]] | None = None


@dataclass
class OutputBuffer:
    """Streaming output buffer for a background process.

    Reader tasks append lines to the bounded deques.  drain_output()
    consumes all accumulated lines and clears the deques.  The bounded
    deque (maxlen) ensures memory usage per process is capped even if
    the consumer drains slowly.
    """

    stdout: deque[str] = field(default_factory=lambda: deque(maxlen=_MAX_BUFFER_LINES))
    stderr: deque[str] = field(default_factory=lambda: deque(maxlen=_MAX_BUFFER_LINES))
    exit_code: int | None = None
    completed: bool = False


@dataclass
class BackgroundProcess:
    """Metadata for a background shell process.

    Tracks information about a shell command running in the background.
    The actual process lifecycle is managed by an asyncio.Task held
    in Shell._background_tasks.
    """

    process_id: str
    command: str
    cwd: str | None
    pid: int | None = None
    started_at: datetime = field(default_factory=datetime.now)


@dataclass
class CompletedProcess:
    """Result of a completed background shell process.

    Built on-the-fly from OutputBuffer by consume_completed_results()
    for one-time consumption by the filter layer.  Output is capped at
    _MAX_COMPLETED_OUTPUT_BYTES to bound memory usage.
    """

    process_id: str
    command: str
    cwd: str | None
    exit_code: int
    stdout: str
    stderr: str
    truncated: bool
    completed_at: datetime = field(default_factory=datetime.now)


class Shell(ABC):
    """Abstract base class for shell command execution.

    Supports both synchronous (execute) and background (start/wait/kill)
    command execution.  Background processes stream their output into an
    OutputBuffer that can be drained incrementally (via drain_output /
    wait_process) or consumed in bulk when completed (via
    consume_completed_results for filter injection).
    """

    def __init__(
        self,
        default_cwd: Path | None = None,
        allowed_paths: list[Path] | None = None,
        default_timeout: float = 30.0,
        skip_instructions: bool = False,
    ):
        """Initialize Shell.

        Args:
            default_cwd: Default working directory for command execution.
                If None, no default working directory is set; callers must
                provide an explicit cwd for each command.
                Always included in allowed_paths when set.
            allowed_paths: Directories allowed as working directories.
                If None, defaults to [default_cwd] when default_cwd is set,
                or [] when default_cwd is None.
            default_timeout: Default timeout in seconds.
            skip_instructions: If True, get_context_instructions returns None.
        """
        self._default_cwd = default_cwd.resolve() if default_cwd is not None else None

        # Build allowed_paths, ensuring default_cwd is included when set
        if allowed_paths is None:
            self._allowed_paths = [self._default_cwd] if self._default_cwd is not None else []
        else:
            resolved_paths = [p.resolve() for p in allowed_paths]
            if self._default_cwd is not None and self._default_cwd not in resolved_paths:
                resolved_paths.append(self._default_cwd)
            self._allowed_paths = resolved_paths

        self._default_timeout = default_timeout
        self._skip_instructions = skip_instructions

        # Background process tracking
        self._background_processes: dict[str, BackgroundProcess] = {}
        self._background_tasks: dict[str, asyncio.Task[int]] = {}
        self._output_buffers: dict[str, OutputBuffer] = {}
        self._stdin_streams: dict[str, WritableStream] = {}
        self._signal_handlers: dict[str, Callable[[int], Awaitable[None]]] = {}

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    async def execute(
        self,
        command: str,
        *,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> tuple[int, str, str]:
        """Execute a command and return (exit_code, stdout, stderr).

        Args:
            command: Command string to execute via shell.
            timeout: Timeout in seconds. None means no timeout -- the command
                runs until it completes or is cancelled. The tool layer is
                responsible for providing an explicit timeout (e.g. 180s default).
            env: Environment variables.
            cwd: Working directory (relative or absolute path).

        Returns:
            Tuple of (exit_code, stdout, stderr).
        """
        ...

    @staticmethod
    async def _read_stream(
        stream: ReadableStream,
        target: deque[str],
    ) -> None:
        """Read lines from stream and append to target deque."""
        while True:
            try:
                line_bytes = await stream.readline()
            except ValueError:
                # Line exceeded stream buffer limit (~64KB) without a
                # newline.  The data has already been discarded from
                # the internal buffer by StreamReader, so just note
                # the truncation and continue reading.
                target.append("[line too long, truncated]")
                continue
            if not line_bytes:
                break
            line = line_bytes.decode("utf-8", errors="replace").rstrip("\n")
            target.append(_truncate_line(line))

    async def start(
        self,
        command: str,
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> str:
        """Start a command in the background with streaming output.

        Calls _create_process() to obtain an ExecutionHandle, then sets up
        reader tasks that continuously drain stdout/stderr into bounded
        deques, and a main task that awaits process completion.

        Reader tasks run continuously to prevent subprocess pipe buffers
        from filling up and blocking the child process.  Output lines are
        truncated and stored in bounded deques to cap memory usage.

        Args:
            command: Command string to execute via shell.
            env: Environment variables.
            cwd: Working directory (relative or absolute path).

        Returns:
            A process_id string for use with wait_process / kill_process.
        """
        process_id, buf = self._setup_background_process(command, cwd)

        try:
            handle = await self._create_process(command, env=env, cwd=cwd)
        except Exception:
            # Clean up tracking on failure to create process
            self._output_buffers.pop(process_id, None)
            self._background_processes.pop(process_id, None)
            raise

        async def _run() -> int:
            """Main task: start readers, wait for process, return exit code."""
            stdout_task = asyncio.create_task(
                self._read_stream(handle.stdout, buf.stdout),
                name=f"bg-stdout-{process_id}",
            )
            stderr_task = asyncio.create_task(
                self._read_stream(handle.stderr, buf.stderr),
                name=f"bg-stderr-{process_id}",
            )
            try:
                exit_code = await handle.wait()
            except asyncio.CancelledError:
                await handle.kill()
                raise
            else:
                # Wait for readers to drain remaining buffered output
                await asyncio.gather(stdout_task, stderr_task)
                return exit_code
            finally:
                # Ensure reader tasks are always cleaned up
                stdout_task.cancel()
                stderr_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.gather(stdout_task, stderr_task)

        # Update metadata with PID if available
        if handle.pid is not None:
            self._background_processes[process_id].pid = handle.pid

        # Track stdin stream if available
        if handle.stdin is not None:
            self._stdin_streams[process_id] = handle.stdin

        # Track signal handler if available
        if handle.send_signal is not None:
            self._signal_handlers[process_id] = handle.send_signal

        task = asyncio.create_task(_run(), name=f"bg-shell-{process_id}")
        self._register_background_task(process_id, task)
        return process_id

    @abstractmethod
    async def _create_process(
        self,
        command: str,
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> ExecutionHandle:
        """Create a background process and return an ExecutionHandle.

        Subclasses implement this to create the actual subprocess or
        connection.  The ABC's start() manages all lifecycle concerns
        (reader tasks, buffer tracking, cancellation cleanup).

        For subprocess-based backends, stdout/stderr are the subprocess
        pipe StreamReaders directly.  For non-streaming backends (Docker,
        RPC), use asyncio.StreamReader as an adapter with feed_data/feed_eof.

        Args:
            command: Command string to execute via shell.
            env: Environment variables.
            cwd: Working directory (relative or absolute path).

        Returns:
            ExecutionHandle with stream access and lifecycle callbacks.
        """
        ...

    # ------------------------------------------------------------------
    # Background process helpers (called by start() implementations)
    # ------------------------------------------------------------------

    def _generate_process_id(self) -> str:
        """Generate a unique process ID for background processes."""
        return uuid4().hex[:12]

    def _setup_background_process(
        self,
        command: str,
        cwd: str | None,
    ) -> tuple[str, OutputBuffer]:
        """Create process_id, metadata, and output buffer.

        Called by start() implementations before creating the subprocess.

        Args:
            command: The command string (for metadata).
            cwd: The working directory (for metadata).

        Returns:
            Tuple of (process_id, output_buffer).
        """
        process_id = self._generate_process_id()
        meta = BackgroundProcess(process_id=process_id, command=command, cwd=cwd)
        buf = OutputBuffer()
        self._background_processes[process_id] = meta
        self._output_buffers[process_id] = buf
        return process_id, buf

    def _register_background_task(
        self,
        process_id: str,
        task: asyncio.Task[int],
    ) -> None:
        """Register the main task for a background process.

        Sets up a done callback that marks the output buffer as completed
        with the exit_code when the task finishes.  Cancelled tasks are
        left unmarked so kill_process can handle cleanup.

        Args:
            process_id: Unique identifier for this background process.
            task: The asyncio.Task whose result is the exit_code.
        """
        self._background_tasks[process_id] = task

        def _on_done(_t: asyncio.Task[int]) -> None:
            self._background_tasks.pop(process_id, None)

            buf = self._output_buffers.get(process_id)
            if buf is None:
                # Buffer already consumed (by kill or explicit drain)
                self._background_processes.pop(process_id, None)
                return

            if _t.cancelled():
                # Don't mark completed; kill_process handles cleanup
                return

            try:
                exit_code = _t.result()
            except Exception:
                buf.completed = True
                buf.exit_code = -1
                return

            buf.completed = True
            buf.exit_code = exit_code

        task.add_done_callback(_on_done)

    # ------------------------------------------------------------------
    # Output draining
    # ------------------------------------------------------------------

    def drain_output(self, process_id: str) -> tuple[str, str, bool, int | None]:
        """Drain buffered output for a background process.

        Consumes all lines currently in the buffer deques and returns
        them joined by newlines.  If the process has completed, also
        removes the buffer and metadata (the result has been consumed).

        Args:
            process_id: The process ID returned by start().

        Returns:
            Tuple of (stdout, stderr, is_running, exit_code).
            - is_running: True if process is still running.
            - exit_code: None if still running.

        Raises:
            KeyError: If process_id has no output buffer (never started,
                already fully consumed, or already killed).
        """
        buf = self._output_buffers.get(process_id)
        if buf is None:
            raise KeyError(f"No output buffer for process: {process_id}")

        stdout = "\n".join(buf.stdout) if buf.stdout else ""
        stderr = "\n".join(buf.stderr) if buf.stderr else ""
        buf.stdout.clear()
        buf.stderr.clear()

        is_running = not buf.completed
        exit_code = buf.exit_code

        # Completed: clean up tracking (agent explicitly consumed)
        if buf.completed:
            self._output_buffers.pop(process_id, None)
            self._background_processes.pop(process_id, None)
            self._stdin_streams.pop(process_id, None)
            self._signal_handlers.pop(process_id, None)

        return stdout, stderr, is_running, exit_code

    # ------------------------------------------------------------------
    # Wait / Kill / Close
    # ------------------------------------------------------------------

    async def wait_process(
        self,
        process_id: str,
        *,
        timeout: float,
    ) -> tuple[str, str, bool, int | None]:
        """Wait for a background process and drain its output.

        When timeout is 0, drains the buffer immediately without waiting
        (a single poll).  When timeout > 0, waits up to *timeout* seconds
        for the process to complete, then drains whatever is available.

        The process is never killed on timeout -- it continues running
        in the background.  The caller can poll again later or kill it.

        Args:
            process_id: The process ID returned by start().
            timeout: Maximum seconds to wait.  0 means drain immediately.

        Returns:
            Tuple of (stdout, stderr, is_running, exit_code).

        Raises:
            KeyError: If process_id is not found (never started or
                already consumed / killed).
        """
        buf = self._output_buffers.get(process_id)
        if buf is None:
            raise KeyError(f"No background process with id: {process_id}")

        # Wait for completion if requested and not yet done
        if not buf.completed and timeout > 0:
            task = self._background_tasks.get(process_id)
            if task is not None:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
                # Yield to let done callbacks execute
                await asyncio.sleep(0)

        return self.drain_output(process_id)

    async def kill_process(self, process_id: str) -> tuple[str, str]:
        """Kill a background process and return its buffered output.

        Cancels the async task, drains any remaining output from the
        buffer, and removes all tracking state.

        Args:
            process_id: The process ID returned by start().

        Returns:
            Tuple of (stdout, stderr) -- final buffered output.

        Raises:
            KeyError: If process_id is not found.
        """
        task = self._background_tasks.get(process_id)
        buf = self._output_buffers.get(process_id)

        if task is None and buf is None:
            raise KeyError(f"No background process with id: {process_id}")

        # Drain current buffer before cancel
        stdout = "\n".join(buf.stdout) if buf and buf.stdout else ""
        stderr = "\n".join(buf.stderr) if buf and buf.stderr else ""

        try:
            # Close stdin before cancelling
            stdin = self._stdin_streams.pop(process_id, None)
            if stdin is not None:
                with contextlib.suppress(Exception):
                    await stdin.close()

            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        finally:
            # Always clean up tracking, even if cancel/close raises
            self._background_tasks.pop(process_id, None)
            self._background_processes.pop(process_id, None)
            self._output_buffers.pop(process_id, None)
            self._signal_handlers.pop(process_id, None)

        return stdout, stderr

    async def close(self) -> None:
        """Clean up resources owned by this Shell.

        Kills all remaining background processes and cleans up tracking state.
        Subclasses can override this to clean up additional resources
        (e.g., persistent shell sessions, SSH connections).
        Always call super().close() when overriding.
        """
        for pid in list(self._background_tasks):
            with contextlib.suppress(Exception):
                await self.kill_process(pid)
        # Also clean up any completed-but-unconsumed buffers
        self._background_tasks.clear()
        self._background_processes.clear()
        self._output_buffers.clear()
        self._stdin_streams.clear()
        self._signal_handlers.clear()

    # ------------------------------------------------------------------
    # Stdin interaction
    # ------------------------------------------------------------------

    async def write_stdin(self, process_id: str, data: str) -> None:
        """Write text to a background process's stdin.

        Args:
            process_id: The process ID returned by start().
            data: Text to write (encoded as UTF-8).

        Raises:
            KeyError: If process_id not found or process has no stdin.
        """
        stream = self._stdin_streams.get(process_id)
        if stream is None:
            if process_id in self._output_buffers:
                raise KeyError(f"Process {process_id} does not support stdin")
            raise KeyError(f"No background process with id: {process_id}")
        await stream.write(data.encode("utf-8"))

    async def close_stdin(self, process_id: str) -> None:
        """Close stdin for a background process (sends EOF).

        After closing, the process will receive EOF on its stdin.
        This is idempotent -- closing an already-closed stdin is a no-op.

        Args:
            process_id: The process ID returned by start().
        """
        stream = self._stdin_streams.pop(process_id, None)
        if stream is not None:
            with contextlib.suppress(Exception):
                await stream.close()

    # ------------------------------------------------------------------
    # Signal sending
    # ------------------------------------------------------------------

    async def send_signal(self, process_id: str, sig: int) -> None:
        """Send a signal to a background process.

        Uses the signal handler provided by the ExecutionHandle.
        For local processes this typically maps to os.kill(); for sandbox
        processes this may send a signal message over the exec transport.

        Common signals (from the ``signal`` module)::

            signal.SIGINT  (2)  -- Interrupt (like Ctrl+C)
            signal.SIGTERM (15) -- Graceful termination
            signal.SIGKILL (9)  -- Forced kill
            signal.SIGCONT (18) -- Resume a stopped process

        Args:
            process_id: The process ID returned by start().
            sig: Signal number to send (use ``signal`` module constants).

        Raises:
            KeyError: If process_id is not found, has no signal handler,
                or has already completed.
        """
        handler = self._signal_handlers.get(process_id)
        if handler is None:
            if process_id in self._output_buffers:
                raise KeyError(f"Process {process_id} does not support signals")
            raise KeyError(f"No background process with id: {process_id}")

        # Reject signals for completed processes to avoid signaling reused PIDs
        if process_id not in self._background_tasks:
            raise KeyError(f"Process {process_id} has already completed")

        await handler(sig)

    # ------------------------------------------------------------------
    # Completed results for filter consumption
    # ------------------------------------------------------------------

    def consume_completed_results(self) -> list[CompletedProcess]:
        """Consume all completed-but-unconsumed background results.

        Scans output buffers for completed processes, drains their output,
        constructs CompletedProcess objects, and removes the buffers.
        Each result is returned exactly once (one-time consumption).

        Called by the background shell filter to inject results into
        the conversation.

        Returns:
            List of CompletedProcess, ordered by discovery.
        """
        completed_pids = [pid for pid, buf in self._output_buffers.items() if buf.completed]

        if not completed_pids:
            return []

        # Only consume up to the cap to avoid silently dropping results.
        # Remaining completed processes stay in the buffer for the next call.
        pids_to_consume = completed_pids[:_MAX_COMPLETED_RESULTS]

        results: list[CompletedProcess] = []
        for pid in pids_to_consume:
            buf = self._output_buffers.pop(pid)
            meta = self._background_processes.pop(pid, None)
            self._stdin_streams.pop(pid, None)
            self._signal_handlers.pop(pid, None)

            stdout = "\n".join(buf.stdout) if buf.stdout else ""
            stderr = "\n".join(buf.stderr) if buf.stderr else ""

            stdout, stdout_trunc = _truncate_to_bytes(stdout, _MAX_COMPLETED_OUTPUT_BYTES)
            stderr, stderr_trunc = _truncate_to_bytes(stderr, _MAX_COMPLETED_OUTPUT_BYTES)

            results.append(
                CompletedProcess(
                    process_id=pid,
                    command=meta.command if meta else "unknown",
                    cwd=meta.cwd if meta else None,
                    exit_code=buf.exit_code if buf.exit_code is not None else -1,
                    stdout=stdout,
                    stderr=stderr,
                    truncated=stdout_trunc or stderr_trunc,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Status / properties
    # ------------------------------------------------------------------

    @property
    def active_background_processes(self) -> dict[str, BackgroundProcess]:
        """Return a snapshot of currently running background processes.

        Only includes processes whose task is still running (not yet
        completed or killed).  Completed-but-unconsumed processes are
        excluded -- use consume_completed_results() for those.
        """
        return {pid: p for pid, p in self._background_processes.items() if pid in self._background_tasks}

    @property
    def has_active_background_processes(self) -> bool:
        """Check if there are any active background processes."""
        return bool(self._background_tasks)

    @property
    def has_background_activity(self) -> bool:
        """Check if there are any active or completed-but-unconsumed background processes."""
        return bool(self._background_tasks) or any(buf.completed for buf in self._output_buffers.values())

    def background_status_summary(self) -> str | None:
        """Return a brief XML summary of background process status.

        Includes active (running) processes and completed-but-unconsumed
        results.  Returns None if there is no background activity.

        This method is used by:
        - inject_background_results filter (alongside full results)
        - AgentContext.get_context_instructions (user prompt briefing)
        """
        active = {pid: p for pid, p in self._background_processes.items() if pid in self._background_tasks}
        completed_bufs = {pid: buf for pid, buf in self._output_buffers.items() if buf.completed}

        if not active and not completed_bufs:
            return None

        parts: list[str] = ["<background-processes>"]

        if active:
            for proc in active.values():
                elapsed = (datetime.now() - proc.started_at).total_seconds()
                parts.append(
                    f'  <process id="{_xml_escape(proc.process_id)}" status="running" '
                    f'command="{_xml_escape(proc.command)}" elapsed="{elapsed:.0f}s" />'
                )

        if completed_bufs:
            for pid, buf in completed_bufs.items():
                meta = self._background_processes.get(pid)
                cmd = meta.command if meta else "unknown"
                ec = buf.exit_code if buf.exit_code is not None else -1
                status = "completed" if ec == 0 else f"failed (exit={ec})"
                parts.append(
                    f'  <process id="{_xml_escape(pid)}" status="{_xml_escape(status)}" command="{_xml_escape(cmd)}" />'
                )

        parts.append("</background-processes>")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Context instructions
    # ------------------------------------------------------------------

    async def get_context_instructions(self) -> str | None:
        """Return instructions for the agent about shell capabilities."""
        if self._skip_instructions:
            return None
        parts: list[str] = ["<shell-execution>"]

        if self._allowed_paths:
            paths_str = "\n".join(f"    <path>{p}</path>" for p in self._allowed_paths)
            parts.append(f"  <allowed-working-directories>\n{paths_str}\n  </allowed-working-directories>")

        if self._default_cwd is not None:
            parts.append(f"  <default-working-directory>{self._default_cwd}</default-working-directory>")

        parts.append(f"  <default-timeout>{self._default_timeout}s</default-timeout>")
        parts.append("  <note>Commands will be executed with the working directory validated.</note>")
        parts.append("</shell-execution>")
        return "\n".join(parts)
