"""Shell abstraction for environment module.

This module provides an abstract base class for shell command execution,
including support for background process management with streaming output
via OutputBuffer.
"""

import asyncio
import codecs
import contextlib
from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable, Iterator
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from html import escape as _xml_escape
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast, final, runtime_checkable
from uuid import uuid4

from .exceptions import ShellTimeoutError
from .output import BoundedTextAccumulator, truncate_utf8_head_tail

_T = TypeVar("_T")

# --- OutputBuffer limits ---
# Max lines per stream (stdout/stderr) retained in the bounded deque.
_MAX_BUFFER_LINES = 200
# Max characters per line before truncation (guards against binary/minified blobs).
_MAX_LINE_LENGTH = 4096
_LINE_TRUNCATION_MARKER = "...[truncated]..."
_STREAM_READ_CHUNK_BYTES = 64 * 1024

# --- Completed results limits (for filter consumption) ---
# Per-process cap on output bytes when constructing CompletedProcess for the filter.
_MAX_COMPLETED_OUTPUT_BYTES = 1 * 1024 * 1024  # 1 MB per stream
# Max queued CompletedProcess results returned by consume_completed_results().
_MAX_COMPLETED_RESULTS = 50
# Max completed results retained for an explicit wait after filter injection.
_MAX_RETAINED_COMPLETED_RESULTS = 50
# Aggregate UTF-8 bytes retained across completed stdout/stderr streams.
_MAX_RETAINED_COMPLETED_OUTPUT_BYTES = 16 * 1024 * 1024  # 16 MB per Shell


def _combine_status_summaries(*summaries: str | None) -> str | None:
    """Join non-empty status summaries."""
    present = [summary for summary in summaries if summary]
    return "\n".join(present) if present else None


class ReadableStream(Protocol):
    """Protocol for a legacy asynchronously readable byte-line stream."""

    async def readline(self) -> bytes: ...


@runtime_checkable
class ChunkReadableStream(Protocol):
    """Optional stream capability for size-safe chunked reads."""

    async def read(self, n: int = -1) -> bytes: ...


class _LegacyLineOverflowError(ValueError):
    """Signal that a readline-only stream discarded an oversized line."""


async def _read_stream_chunk(stream: ReadableStream, max_bytes: int) -> bytes:
    """Read one bounded chunk, preserving compatibility with line-only custom streams."""
    if isinstance(stream, ChunkReadableStream):
        return await stream.read(max_bytes)
    try:
        return await stream.readline()
    except ValueError as exc:
        raise _LegacyLineOverflowError from exc


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
    StreamReaders directly and ``communicate`` may expose the native full-output
    operation for foreground execution. For non-streaming backends (Docker,
    RPC), use asyncio.StreamReader as an adapter with feed_data/feed_eof.
    """

    stdout: ReadableStream
    stderr: ReadableStream
    wait: Callable[[], Awaitable[int]]
    kill: Callable[[], Awaitable[None]]
    stdin: WritableStream | None = None
    pid: int | None = None
    send_signal: Callable[[int], Awaitable[None]] | None = None
    communicate: Callable[[], Awaitable[tuple[bytes, bytes]]] | None = None


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

    Built from OutputBuffer for filter delivery and explicit wait retention.
    Filter-facing instances are capped at _MAX_COMPLETED_OUTPUT_BYTES per
    stream, while a separate full-buffer instance is retained under aggregate
    count and byte budgets until explicitly read.
    """

    process_id: str
    command: str
    cwd: str | None
    exit_code: int
    stdout: str
    stderr: str
    truncated: bool
    completed_at: datetime = field(default_factory=datetime.now)


def _completed_result_output_bytes(result: CompletedProcess) -> int:
    """Return retained stdout/stderr size in UTF-8 bytes."""
    return len(result.stdout.encode("utf-8")) + len(result.stderr.encode("utf-8"))


class ReadyState(StrEnum):
    """Readiness state for deferred environment capabilities."""

    NOT_STARTED = "not_started"
    STARTING = "starting"
    READY = "ready"
    FAILED = "failed"


class ShellSessionAccessError(RuntimeError):
    """Raised when work from a retired session tries to access a shared shell."""


class ShellBackgroundResetError(RuntimeError):
    """Raised when one or more owned shell executions could not be terminated."""

    def __init__(self, failures: dict[str, BaseException]) -> None:
        self.failures = dict(failures)
        process_ids = ", ".join(sorted(failures))
        super().__init__(f"Failed to terminate owned shell execution(s): {process_ids}")


class Shell(ABC):
    """Abstract base class for shell command execution.

    Supports both synchronous (execute) and background (start/wait/kill)
    command execution.  Background processes stream their output into an
    OutputBuffer that can be drained incrementally (via drain_output /
    wait_process) or consumed in bulk when completed (via
    consume_completed_results for filter injection).
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Reject runtime overrides that bypass the owned execute boundary."""
        super().__init_subclass__(**kwargs)
        if cls.execute is not Shell.execute:
            raise TypeError(
                f"{cls.__name__} must implement _create_process() and must not override final Shell.execute()"
            )

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
        self._foreground_tasks: dict[str, asyncio.Task[tuple[int, str, str]]] = {}
        self._execution_handles: dict[str, ExecutionHandle] = {}
        self._background_cleanup_errors: dict[str, BaseException] = {}
        self._output_buffers: dict[str, OutputBuffer] = {}
        self._retained_completed_results: OrderedDict[str, CompletedProcess] = OrderedDict()
        self._retained_completed_output_bytes = 0
        self._stdin_streams: dict[str, WritableStream] = {}
        self._signal_handlers: dict[str, Callable[[int], Awaitable[None]]] = {}
        self._background_lifecycle_lock = asyncio.Lock()

        # Session-scoped work receives the current generation through a
        # ContextVar. Child tasks inherit it automatically. Host tasks without
        # a lease remain unrestricted so the reusable backend can serve the
        # next session after the generation advances.
        self._session_generation = 0
        self._session_generation_var: ContextVar[int | None] = ContextVar(
            f"shell_session_generation_{id(self)}",
            default=None,
        )

    @property
    def ready_state(self) -> ReadyState:
        """Return readiness state for shells with deferred setup."""
        return ReadyState.READY

    async def ensure_ready(self) -> None:
        """Ensure the shell backend is ready for command execution."""
        self.assert_session_access()

    def capture_session_access(self) -> int:
        """Capture the generation for work that may start on a later loop turn."""
        self.assert_session_access()
        return self._session_generation

    @contextlib.contextmanager
    def session_access_scope(self, generation: int | None = None) -> Iterator[None]:
        """Bind this task and its child tasks to a captured shell session."""
        self.assert_session_access()
        bound_generation = self._session_generation if generation is None else generation
        if bound_generation != self._session_generation:
            raise ShellSessionAccessError("Shell access belongs to a retired session")
        token = self._session_generation_var.set(bound_generation)
        try:
            yield
        finally:
            self._session_generation_var.reset(token)

    def revoke_session_access(self) -> None:
        """Retire all previously bound session leases without closing the backend."""
        self._session_generation += 1

    def assert_session_access(self) -> None:
        """Reject shell access inherited from a retired session task."""
        generation = self._session_generation_var.get()
        if generation is not None and generation != self._session_generation:
            raise ShellSessionAccessError("Shell access belongs to a retired session")

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @final
    async def execute(
        self,
        command: str,
        *,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> tuple[int, str, str]:
        """Execute a foreground command through the shell-owned lifecycle.

        Process creation, execution, and cancellation remain visible to a
        concurrent session reset. Subclasses implement ``_create_process`` and
        must not override this ownership boundary.
        """
        self.assert_session_access()
        operation_id = f"foreground-{self._generate_process_id()}"
        effective_timeout = self._resolve_execute_timeout(timeout)
        loop = asyncio.get_running_loop()
        deadline = None if effective_timeout is None else loop.time() + effective_timeout

        async def _create_registered_task() -> asyncio.Task[tuple[int, str, str]]:
            async with self._background_lifecycle_lock:
                self.assert_session_access()
                handle = await self._create_execution_handle_owned(
                    operation_id,
                    command,
                    env=env,
                    cwd=cwd,
                )
                self._execution_handles[operation_id] = handle
                task = asyncio.create_task(
                    self._run_foreground_execution(operation_id, handle),
                    name=f"foreground-shell-{operation_id}",
                )
                self._foreground_tasks[operation_id] = task
                task.add_done_callback(lambda _done: self._foreground_tasks.pop(operation_id, None))
                return task

        if deadline is None:
            task = await _create_registered_task()
        else:
            timeout_scope = asyncio.timeout_at(deadline)
            try:
                async with timeout_scope:
                    task = await _create_registered_task()
            except TimeoutError as exc:
                if not timeout_scope.expired():
                    raise
                raise ShellTimeoutError(command, cast(float, effective_timeout)) from exc

        try:
            if deadline is None:
                return await asyncio.shield(task)
            remaining = deadline - loop.time()
            if remaining <= 0:
                await self._cancel_foreground_execution(task)
                raise ShellTimeoutError(command, cast(float, effective_timeout))
            return await asyncio.wait_for(asyncio.shield(task), timeout=remaining)
        except TimeoutError as exc:
            await self._cancel_foreground_execution(task)
            raise ShellTimeoutError(command, cast(float, effective_timeout)) from exc
        except asyncio.CancelledError:
            await self._cancel_foreground_execution(task)
            raise

    def _resolve_execute_timeout(self, timeout: float | None) -> float | None:
        """Resolve a public execute timeout; subclasses may preserve backend defaults."""
        return timeout

    @staticmethod
    async def _read_stream_fully(stream: ReadableStream) -> str:
        """Read a foreground stream in chunks without imposing a line-length limit."""
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        chunks: list[str] = []
        while True:
            data = await _read_stream_chunk(stream, _STREAM_READ_CHUNK_BYTES)
            if not data:
                chunks.append(decoder.decode(b"", final=True))
                return "".join(chunks)
            chunks.append(decoder.decode(data))

    async def _run_foreground_execution(
        self,
        operation_id: str,
        handle: ExecutionHandle,
    ) -> tuple[int, str, str]:
        """Drain and wait for one foreground handle under shell ownership."""
        stdout_task: asyncio.Task[str] | None = None
        stderr_task: asyncio.Task[str] | None = None
        try:
            if handle.communicate is not None:
                stdout_bytes, stderr_bytes = await handle.communicate()
                exit_code = await handle.wait()
                stdout = stdout_bytes.decode("utf-8", errors="replace")
                stderr = stderr_bytes.decode("utf-8", errors="replace")
            else:
                stdout_task = asyncio.create_task(
                    self._read_stream_fully(handle.stdout),
                    name=f"foreground-stdout-{operation_id}",
                )
                stderr_task = asyncio.create_task(
                    self._read_stream_fully(handle.stderr),
                    name=f"foreground-stderr-{operation_id}",
                )
                exit_code = await handle.wait()
                stdout, stderr = await asyncio.gather(stdout_task, stderr_task)
        except asyncio.CancelledError:
            await self._kill_execution_handle(operation_id, handle)
            raise
        except BaseException:
            await self._kill_execution_handle(operation_id, handle)
            raise
        else:
            self._execution_handles.pop(operation_id, None)
            self._background_cleanup_errors.pop(operation_id, None)
            return exit_code, stdout, stderr
        finally:
            reader_tasks = [task for task in (stdout_task, stderr_task) if task is not None]
            for reader_task in reader_tasks:
                reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.gather(*reader_tasks)

    async def _cancel_foreground_execution(self, task: asyncio.Task[tuple[int, str, str]]) -> None:
        """Cancel an owned foreground task and wait for termination to finish."""
        task.cancel()
        # The execution task re-raises its expected cancellation only after its
        # kill hook has completed successfully.
        with contextlib.suppress(asyncio.CancelledError):
            await self._await_owned_future(task)

    @staticmethod
    async def _read_stream(
        stream: ReadableStream,
        target: deque[str],
    ) -> None:
        """Read chunks into bounded logical lines without relying on ``readline()`` limits."""
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        line = BoundedTextAccumulator(_MAX_LINE_LENGTH, marker=_LINE_TRUNCATION_MARKER)

        def _append_text(value: str) -> None:
            segments = value.split("\n")
            for segment in segments[:-1]:
                line.append(segment)
                target.append(line.finish())
            line.append(segments[-1])

        def _flush_partial_line() -> None:
            _append_text(decoder.decode(b"", final=True))
            if not line.empty:
                target.append(line.finish())

        try:
            while True:
                try:
                    data = await _read_stream_chunk(stream, _STREAM_READ_CHUNK_BYTES)
                except _LegacyLineOverflowError:
                    # Preserve the legacy line-only stream behavior when its
                    # implementation discards an over-limit line.
                    target.append("[line too long, truncated]")
                    continue
                if not data:
                    _flush_partial_line()
                    return
                _append_text(decoder.decode(data))
        except asyncio.CancelledError:
            _flush_partial_line()
            raise

    async def start(
        self,
        command: str,
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> str:
        """Create and register a background process atomically with session reset."""
        self.assert_session_access()
        async with self._background_lifecycle_lock:
            # The lease may have been revoked while this task waited behind a
            # reset that was already in progress.
            self.assert_session_access()
            return await self._start_background_process(command, env=env, cwd=cwd)

    async def _start_background_process(
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
        self.assert_session_access()
        process_id, buf = self._setup_background_process(command, cwd)

        try:
            handle = await self._create_execution_handle_owned(
                process_id,
                command,
                env=env,
                cwd=cwd,
            )
        except BaseException:
            # A failed termination retains its handle and metadata for reset.
            if process_id not in self._execution_handles:
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
                # Wait for readers to drain remaining buffered output.
                await asyncio.gather(stdout_task, stderr_task)
            except asyncio.CancelledError:
                # Keep the handle when termination fails so reset/kill can retry.
                await self._kill_execution_handle(process_id, handle)
                raise
            except BaseException:
                # A failed wait/read path still owns a potentially live process.
                await self._kill_execution_handle(process_id, handle)
                raise
            else:
                self._execution_handles.pop(process_id, None)
                self._background_cleanup_errors.pop(process_id, None)
                return exit_code
            finally:
                # Ensure reader tasks are always cleaned up.
                stdout_task.cancel()
                stderr_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.gather(stdout_task, stderr_task)

        self._execution_handles[process_id] = handle

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

    async def _create_execution_handle_owned(
        self,
        operation_id: str,
        command: str,
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> ExecutionHandle:
        """Create a handle without allowing caller cancellation to lose it."""
        creation_task = asyncio.create_task(
            self._create_process(command, env=env, cwd=cwd),
            name=f"shell-create-{operation_id}",
        )
        handle, cancelled_error = await self._await_owned_future(creation_task)
        if cancelled_error is None:
            return handle

        self._execution_handles[operation_id] = handle
        await self._kill_execution_handle(operation_id, handle)
        raise cancelled_error

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

        For subprocess-based backends, stdout/stderr are the subprocess pipe
        StreamReaders directly and communicate may expose native full-output
        collection. For non-streaming backends (Docker, RPC), use
        asyncio.StreamReader as an adapter with feed_data/feed_eof.

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
        self._pop_retained_completed_result(process_id)
        return process_id, buf

    def _forget_background_process(self, process_id: str) -> None:
        """Drop all lifecycle state after a process is known to be terminated."""
        self._background_tasks.pop(process_id, None)
        self._foreground_tasks.pop(process_id, None)
        self._execution_handles.pop(process_id, None)
        self._background_cleanup_errors.pop(process_id, None)
        self._background_processes.pop(process_id, None)
        self._output_buffers.pop(process_id, None)
        self._stdin_streams.pop(process_id, None)
        self._signal_handlers.pop(process_id, None)
        self._pop_retained_completed_result(process_id)

    async def _kill_execution_handle(self, process_id: str, handle: ExecutionHandle) -> None:
        """Kill a process and retain ownership when its backend hook fails."""

        async def _kill() -> None:
            await handle.kill()

        kill_task = asyncio.create_task(_kill(), name=f"shell-kill-{process_id}")
        try:
            _, cancelled_error = await self._await_owned_future(kill_task)
        except BaseException as exc:
            self._background_cleanup_errors[process_id] = exc
            raise

        self._execution_handles.pop(process_id, None)
        self._background_cleanup_errors.pop(process_id, None)
        if cancelled_error is not None:
            raise cancelled_error

    @staticmethod
    async def _await_owned_future(
        future: asyncio.Future[_T],
    ) -> tuple[_T, asyncio.CancelledError | None]:
        """Finish shell-owned work despite repeated caller cancellation."""
        cancelled_error: asyncio.CancelledError | None = None
        while True:
            try:
                return await asyncio.shield(future), cancelled_error
            except asyncio.CancelledError as exc:
                if cancelled_error is None:
                    cancelled_error = exc
                if future.done():
                    return future.result(), cancelled_error

    def _record_reset_task_failures(
        self,
        task_items: list[tuple[str, asyncio.Task[Any]]],
        task_results: list[Any],
    ) -> None:
        """Retain errors only when a process handle still requires cleanup."""
        for (process_id, _task), result in zip(task_items, task_results, strict=True):
            if (
                isinstance(result, BaseException)
                and not isinstance(result, asyncio.CancelledError)
                and process_id in self._execution_handles
            ):
                self._background_cleanup_errors.setdefault(process_id, result)

    def _pop_retained_completed_result(self, process_id: str) -> CompletedProcess | None:
        """Remove a retained result and update its aggregate byte accounting."""
        result = self._retained_completed_results.pop(process_id, None)
        if result is not None:
            self._retained_completed_output_bytes -= _completed_result_output_bytes(result)
        return result

    def _retain_completed_result(self, result: CompletedProcess) -> None:
        """Retain a terminal result within count and aggregate byte limits."""
        self._pop_retained_completed_result(result.process_id)
        self._retained_completed_results[result.process_id] = result
        self._retained_completed_output_bytes += _completed_result_output_bytes(result)

        while (
            len(self._retained_completed_results) > _MAX_RETAINED_COMPLETED_RESULTS
            or self._retained_completed_output_bytes > _MAX_RETAINED_COMPLETED_OUTPUT_BYTES
        ):
            oldest_process_id = next(iter(self._retained_completed_results))
            self._pop_retained_completed_result(oldest_process_id)

    def _finalize_background_task(self, process_id: str, task: asyncio.Task[int]) -> None:
        """Record completion state for a finished background task.

        Done callbacks normally perform this bookkeeping.  Synchronous readers
        such as consume_completed_results() also call this method for tasks that
        are already done, avoiding timing dependence on callback scheduling.
        """
        if not task.done():
            return

        self._background_tasks.pop(process_id, None)

        buf = self._output_buffers.get(process_id)
        if buf is None:
            # Buffer already consumed (by kill, wait/drain, or filter injection).
            self._background_processes.pop(process_id, None)
            return

        if buf.completed:
            return

        if task.cancelled():
            # Don't mark completed; kill_process handles cleanup.
            return

        try:
            exit_code = task.result()
        except Exception:
            if process_id in self._background_cleanup_errors:
                # The execution task ended because its kill hook failed. Keep
                # lifecycle status active until a retry confirms termination.
                return
            buf.completed = True
            buf.exit_code = -1
            return

        buf.completed = True
        buf.exit_code = exit_code

    def _refresh_completed_tasks(self) -> None:
        """Synchronously finalize background tasks that are already done."""
        for process_id, task in list(self._background_tasks.items()):
            if task.done():
                self._finalize_background_task(process_id, task)

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
            self._finalize_background_task(process_id, _t)

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
        self.assert_session_access()
        buf = self._output_buffers.get(process_id)
        if buf is None:
            retained = self._pop_retained_completed_result(process_id)
            if retained is None:
                raise KeyError(f"No output buffer for process: {process_id}")
            return retained.stdout, retained.stderr, False, retained.exit_code

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
        self.assert_session_access()
        buf = self._output_buffers.get(process_id)
        if buf is None:
            retained = self._pop_retained_completed_result(process_id)
            if retained is None:
                raise KeyError(f"No background process with id: {process_id}")
            return retained.stdout, retained.stderr, False, retained.exit_code

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
        self.assert_session_access()
        task = self._background_tasks.get(process_id)
        handle = self._execution_handles.get(process_id)
        buf = self._output_buffers.get(process_id)

        if task is None and handle is None and buf is None:
            raise KeyError(f"No background process with id: {process_id}")

        stdout = ""
        stderr = ""
        terminated = False

        try:
            stdin = self._stdin_streams.pop(process_id, None)
            if stdin is not None:
                with contextlib.suppress(Exception):
                    await stdin.close()

            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                # Cancellation before _run() first executes bypasses its kill
                # handler, so terminate any handle that remains owned.
                handle = self._execution_handles.get(process_id)
                if handle is not None:
                    await self._kill_execution_handle(process_id, handle)
            elif handle is not None:
                await self._kill_execution_handle(process_id, handle)
            terminated = True
            if buf is not None:
                stdout = "\n".join(buf.stdout) if buf.stdout else ""
                stderr = "\n".join(buf.stderr) if buf.stderr else ""
        finally:
            if terminated:
                self._forget_background_process(process_id)

        return stdout, stderr

    async def reset_background_processes(self) -> None:
        """Terminate all owned execution and forget background state while remaining reusable."""
        self.assert_session_access()
        cleanup = asyncio.create_task(
            self._reset_background_processes_locked(),
            name=f"shell-reset-{id(self)}",
        )
        _, cancelled_error = await self._await_owned_future(cleanup)
        if cancelled_error is not None:
            raise cancelled_error

    async def _reset_background_processes_locked(self) -> None:
        """Serialize process creation and the complete reset snapshot."""
        async with self._background_lifecycle_lock:
            await self._reset_background_processes_unlocked()

    async def _reset_background_processes_unlocked(self) -> None:
        """Reset tracked work while the execution lifecycle lock is held.

        Failed process termination is reported and its execution handle remains
        owned by the shell so a later reset can retry safely.
        """
        task_items: list[tuple[str, asyncio.Task[Any]]] = [
            *self._background_tasks.items(),
            *self._foreground_tasks.items(),
        ]
        task_process_ids = {process_id for process_id, _task in task_items}
        retry_handle_items = [
            (process_id, handle)
            for process_id, handle in self._execution_handles.items()
            if process_id not in task_process_ids
        ]
        tracked_process_ids = (
            set(self._background_processes)
            | set(self._background_tasks)
            | set(self._foreground_tasks)
            | set(self._execution_handles)
            | set(self._output_buffers)
            | set(self._background_cleanup_errors)
        )
        stdin_streams = list(self._stdin_streams.values())

        # Every active execution receives cancellation before the first await.
        for _process_id, task in task_items:
            task.cancel()

        async def _close_stream(stream: WritableStream) -> None:
            with contextlib.suppress(Exception):
                await stream.close()

        retry_tasks = [
            asyncio.create_task(
                self._kill_execution_handle(process_id, handle),
                name=f"bg-shell-reset-{process_id}",
            )
            for process_id, handle in retry_handle_items
        ]
        stream_tasks = [asyncio.create_task(_close_stream(stream)) for stream in stdin_streams]
        cleanup = asyncio.gather(
            *(task for _process_id, task in task_items),
            *retry_tasks,
            *stream_tasks,
            return_exceptions=True,
        )
        results, cancelled_error = await self._await_owned_future(cleanup)
        self._record_reset_task_failures(task_items, results[: len(task_items)])

        failed_process_ids = tracked_process_ids & set(self._background_cleanup_errors)
        for process_id in tracked_process_ids - failed_process_ids:
            self._forget_background_process(process_id)

        # Terminal caches and process I/O never cross a session boundary, even
        # when a failed execution handle must remain for a retry.
        self._retained_completed_results.clear()
        self._retained_completed_output_bytes = 0
        self._stdin_streams.clear()
        self._signal_handlers.clear()

        if failed_process_ids:
            failures = {process_id: self._background_cleanup_errors[process_id] for process_id in failed_process_ids}
            raise ShellBackgroundResetError(failures)
        if cancelled_error is not None:
            raise cancelled_error

    async def close(self) -> None:
        """Clean up resources owned by this Shell.

        Kills all remaining background processes and cleans up tracking state.
        Subclasses can override this to clean up additional resources
        (e.g., persistent shell sessions, SSH connections).
        Always call super().close() when overriding.
        """
        await self.reset_background_processes()

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
        self.assert_session_access()
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
        self.assert_session_access()
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
        self.assert_session_access()
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
        """Deliver newly completed background results to the filter once.

        Scans output buffers for completed processes and constructs separate
        CompletedProcess instances for filter delivery and explicit wait.
        Each capped filter result is returned exactly once. The original
        buffered stdout/stderr remains in a count- and byte-bounded terminal
        cache until wait_process() or drain_output() explicitly reads it.

        Called by the background shell filter to inject results into
        the conversation.

        Returns:
            List of CompletedProcess, ordered by discovery.
        """
        self.assert_session_access()
        self._refresh_completed_tasks()
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
            command = meta.command if meta else "unknown"
            cwd = meta.cwd if meta else None
            exit_code = buf.exit_code if buf.exit_code is not None else -1

            retained_result = CompletedProcess(
                process_id=pid,
                command=command,
                cwd=cwd,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                truncated=False,
            )
            self._retain_completed_result(retained_result)

            injected_stdout, stdout_trunc = truncate_utf8_head_tail(stdout, _MAX_COMPLETED_OUTPUT_BYTES)
            injected_stderr, stderr_trunc = truncate_utf8_head_tail(stderr, _MAX_COMPLETED_OUTPUT_BYTES)
            results.append(
                CompletedProcess(
                    process_id=pid,
                    command=command,
                    cwd=cwd,
                    exit_code=exit_code,
                    stdout=injected_stdout,
                    stderr=injected_stderr,
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
        self.assert_session_access()
        active_ids = set(self._background_tasks) | set(self._background_cleanup_errors)
        return {pid: process for pid, process in self._background_processes.items() if pid in active_ids}

    @property
    def has_active_background_processes(self) -> bool:
        """Check if there are any active or termination-failed processes."""
        self.assert_session_access()
        return bool(self._background_tasks) or bool(self._background_cleanup_errors)

    @property
    def has_background_activity(self) -> bool:
        """Check for running, unread, or termination-failed background work."""
        self.assert_session_access()
        self._refresh_completed_tasks()
        return (
            bool(self._background_tasks)
            or bool(self._background_cleanup_errors)
            or any(buf.completed for buf in self._output_buffers.values())
        )

    @property
    def has_retained_completed_results(self) -> bool:
        """Check for injected completed results still available to explicit wait."""
        self.assert_session_access()
        return bool(self._retained_completed_results)

    def background_status_summary(self) -> str | None:
        """Return active and newly completed background process status."""
        self.assert_session_access()
        self._refresh_completed_tasks()
        cleanup_failed = set(self._background_cleanup_errors)
        active = {pid: p for pid, p in self._background_processes.items() if pid in self._background_tasks}
        completed_bufs = {
            pid: buf for pid, buf in self._output_buffers.items() if buf.completed and pid not in cleanup_failed
        }

        if not active and not completed_bufs and not cleanup_failed:
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

        for pid in cleanup_failed - set(completed_bufs):
            meta = self._background_processes.get(pid)
            cmd = meta.command if meta else "unknown"
            parts.append(
                f'  <process id="{_xml_escape(pid)}" status="termination-failed" command="{_xml_escape(cmd)}" />'
            )

        parts.append("</background-processes>")
        return "\n".join(parts)

    def background_status_summary_with_retained_results(self) -> str | None:
        """Return virtual shell status plus injected results available to wait_process()."""
        return _combine_status_summaries(
            self.background_status_summary(),
            self._retained_completed_status_summary(),
        )

    def _retained_completed_status_summary(self) -> str | None:
        """Return status for retained results owned by this Shell instance."""
        if not self._retained_completed_results:
            return None

        parts = ["<background-processes>"]
        for result in self._retained_completed_results.values():
            status = "completed" if result.exit_code == 0 else f"failed (exit={result.exit_code})"
            parts.append(
                f'  <process id="{_xml_escape(result.process_id)}" status="{_xml_escape(status)}" '
                f'command="{_xml_escape(result.command)}" result="available" />'
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


class DeferredShell(Shell):
    """Shell proxy that resolves a concrete shell on first command use.

    DeferredShell keeps environment setup lightweight while preserving the
    regular Shell API. Context instructions are pure-read and should not trigger
    heavy backend setup; command execution and background process creation call
    ensure_ready() and delegate to the resolved shell.
    """

    def __init__(
        self,
        *,
        default_cwd: Path | None = None,
        allowed_paths: list[Path] | None = None,
        default_timeout: float = 30.0,
        skip_instructions: bool = False,
    ) -> None:
        super().__init__(
            default_cwd=default_cwd,
            allowed_paths=allowed_paths,
            default_timeout=default_timeout,
            skip_instructions=skip_instructions,
        )
        self._resolved_shell: Shell | None = None
        self._resolve_lock: asyncio.Lock = asyncio.Lock()
        self._ready_state: ReadyState = ReadyState.NOT_STARTED
        self._ready_error: BaseException | None = None

    @property
    def ready_state(self) -> ReadyState:
        """Return readiness state for the deferred shell backend."""
        return self._ready_state

    @property
    def resolved_shell(self) -> Shell | None:
        """Return the resolved concrete shell when ready."""
        return self._resolved_shell

    @property
    def ready_error(self) -> BaseException | None:
        """Return the last readiness error, if any."""
        return self._ready_error

    async def ensure_ready(self) -> None:
        """Resolve the concrete shell if needed."""
        await self.resolve_shell()

    def _resolve_execute_timeout(self, timeout: float | None) -> float | None:
        """Apply the deferred backend's configured default timeout."""
        effective = self._default_timeout if timeout is None else timeout
        return effective if effective > 0 else None

    async def resolve_shell(self) -> Shell:
        """Return the concrete shell, resolving it once with concurrency safety."""
        self.assert_session_access()
        if self._resolved_shell is not None:
            return self._resolved_shell

        async with self._resolve_lock:
            if self._resolved_shell is not None:
                return self._resolved_shell
            self._ready_state = ReadyState.STARTING
            self._ready_error = None
            try:
                shell = await self._resolve_shell()
            except BaseException as exc:
                self._ready_state = ReadyState.FAILED
                self._ready_error = exc
                raise
            self._resolved_shell = shell
            self._ready_state = ReadyState.READY
            return shell

    @abstractmethod
    async def _resolve_shell(self) -> Shell:
        """Create or return the concrete shell backend."""
        ...

    async def start(
        self,
        command: str,
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> str:
        """Start a background command atomically with proxy and backend reset."""
        self.assert_session_access()
        async with self._background_lifecycle_lock:
            self.assert_session_access()
            shell = await self.resolve_shell()
            self.assert_session_access()
            return await shell.start(command, env=env, cwd=cwd)

    async def _create_process(
        self,
        command: str,
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> ExecutionHandle:
        """Create a background process with the resolved shell."""
        shell = await self.resolve_shell()
        return await shell._create_process(command, env=env, cwd=cwd)

    def drain_output(self, process_id: str) -> tuple[str, str, bool, int | None]:
        """Drain output from proxy-owned or resolved-shell background processes."""
        try:
            return super().drain_output(process_id)
        except KeyError:
            if self._resolved_shell is not None:
                return self._resolved_shell.drain_output(process_id)
            raise

    async def wait_process(
        self,
        process_id: str,
        *,
        timeout: float,
    ) -> tuple[str, str, bool, int | None]:
        """Wait for proxy-owned or resolved-shell background processes."""
        try:
            return await super().wait_process(process_id, timeout=timeout)
        except KeyError:
            if self._resolved_shell is not None:
                return await self._resolved_shell.wait_process(process_id, timeout=timeout)
            raise

    async def kill_process(self, process_id: str) -> tuple[str, str]:
        """Kill proxy-owned or resolved-shell background processes."""
        try:
            return await super().kill_process(process_id)
        except KeyError:
            if self._resolved_shell is not None:
                return await self._resolved_shell.kill_process(process_id)
            raise

    async def write_stdin(self, process_id: str, data: str) -> None:
        """Write stdin to proxy-owned or resolved-shell background processes."""
        try:
            await super().write_stdin(process_id, data)
        except KeyError:
            if self._resolved_shell is not None:
                await self._resolved_shell.write_stdin(process_id, data)
                return
            raise

    async def close_stdin(self, process_id: str) -> None:
        """Close stdin for proxy-owned or resolved-shell background processes."""
        try:
            await super().close_stdin(process_id)
        except KeyError:
            if self._resolved_shell is not None:
                await self._resolved_shell.close_stdin(process_id)
                return
            raise

    async def send_signal(self, process_id: str, sig: int) -> None:
        """Send signal to proxy-owned or resolved-shell background processes."""
        try:
            await super().send_signal(process_id, sig)
        except KeyError:
            if self._resolved_shell is not None:
                await self._resolved_shell.send_signal(process_id, sig)
                return
            raise

    def consume_completed_results(self) -> list[CompletedProcess]:
        """Consume completed results from proxy and resolved shell."""
        results = super().consume_completed_results()
        if self._resolved_shell is not None:
            results.extend(self._resolved_shell.consume_completed_results())
        return results

    @property
    def active_background_processes(self) -> dict[str, BackgroundProcess]:
        """Return active background processes from proxy and resolved shell."""
        active = dict(super().active_background_processes)
        if self._resolved_shell is not None:
            active.update(self._resolved_shell.active_background_processes)
        return active

    @property
    def has_active_background_processes(self) -> bool:
        """Return whether proxy or resolved shell has active background processes."""
        return super().has_active_background_processes or (
            self._resolved_shell.has_active_background_processes if self._resolved_shell is not None else False
        )

    @property
    def has_background_activity(self) -> bool:
        """Return whether proxy or resolved shell has background activity."""
        return super().has_background_activity or (
            self._resolved_shell.has_background_activity if self._resolved_shell is not None else False
        )

    @property
    def has_retained_completed_results(self) -> bool:
        """Return whether proxy or resolved shell has retained completed results."""
        return super().has_retained_completed_results or (
            self._resolved_shell.has_retained_completed_results if self._resolved_shell is not None else False
        )

    def background_status_summary(self) -> str | None:
        """Return background status for proxy and resolved shell."""
        own = super().background_status_summary()
        delegated = self._resolved_shell.background_status_summary() if self._resolved_shell is not None else None
        if own and delegated:
            return f"{own}\n{delegated}"
        return own or delegated

    def background_status_summary_with_retained_results(self) -> str | None:
        """Return virtual status plus retained results for proxy and resolved shell."""
        return _combine_status_summaries(
            self.background_status_summary(),
            self._retained_completed_status_summary(),
            (self._resolved_shell._retained_completed_status_summary() if self._resolved_shell is not None else None),
        )

    async def _reset_background_processes_locked(self) -> None:
        """Reset proxy and backend in the same lock order used by start()."""
        errors: list[BaseException] = []
        async with self._background_lifecycle_lock:
            try:
                await super()._reset_background_processes_unlocked()
            except BaseException as exc:
                errors.append(exc)
            if self._resolved_shell is not None:
                try:
                    await self._resolved_shell.reset_background_processes()
                except BaseException as exc:
                    errors.append(exc)
        if errors:
            raise errors[0]

    async def close(self) -> None:
        """Close proxy and resolved shell resources."""
        await super().close()
        if self._resolved_shell is not None:
            await self._resolved_shell.close()

    async def get_context_instructions(self) -> str | None:
        """Return concrete instructions when ready, otherwise deferred instructions."""
        if self._resolved_shell is not None:
            return await self._resolved_shell.get_context_instructions()
        return await self.get_deferred_context_instructions()

    async def get_deferred_context_instructions(self) -> str | None:
        """Return pure-read instructions before backend resolution."""
        return await super().get_context_instructions()
