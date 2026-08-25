"""Fast interactive startup with a lightweight shell and cold runtime child."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import tomllib
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NoReturn, cast

from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI, StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea

from yaacli.app.shell import (
    ComposeSnapshot,
    LeasedVt100Output,
    TUIShell,
    TUIShellCallbacks,
    build_tui_shell,
    create_leased_output,
)
from yaacli.theme import ResolvedTheme, ThemePreference, prompt_toolkit_style_rules, resolve_theme

_CHILD_FD_ENV = "YAACLI_TUI_STARTUP_FD"
_CHILD_STDOUT_FD_ENV = "YAACLI_TUI_STDOUT_FD"
_CHILD_STDERR_FD_ENV = "YAACLI_TUI_STDERR_FD"
_PROTOCOL_VERSION = 1
_MAX_MESSAGE_BYTES = 1024 * 1024
_CHILD_CLEANUP_GRACE_SECONDS = 16.0


class FastTUISetupRequired(RuntimeError):
    """Signal that the ordinary setup wizard must own the terminal."""


class FastTUIRuntimeError(RuntimeError):
    """Runtime-child failure with an optional remote traceback."""

    def __init__(self, message: str, *, child_traceback: str | None = None) -> None:
        super().__init__(message)
        self.child_traceback = child_traceback


@dataclass(frozen=True, slots=True)
class FastTUIRequest:
    """Interactive invocation fields required by the runtime child."""

    verbose: bool
    cwd: str
    session_id: str | None = None
    model_profile_id: str | None = None

    @classmethod
    def from_payload(cls, payload: object) -> FastTUIRequest:
        """Validate one startup request received from the parent process."""
        if not isinstance(payload, dict):
            raise TypeError("Startup request must be a JSON object")
        allowed_fields = {"verbose", "cwd", "session_id", "model_profile_id"}
        unknown_fields = set(payload) - allowed_fields
        if unknown_fields:
            raise ValueError(f"Unknown startup request fields: {sorted(unknown_fields)}")

        verbose = payload.get("verbose")
        cwd = payload.get("cwd")
        if not isinstance(verbose, bool):
            raise TypeError("Startup request verbose flag must be a boolean")
        if not isinstance(cwd, str) or not cwd:
            raise TypeError("Startup request working directory must be a non-empty string")
        optional_strings: dict[str, str | None] = {}
        for field_name in ("session_id", "model_profile_id"):
            value = payload.get(field_name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"Startup request {field_name} must be a string or null")
            optional_strings[field_name] = value
        return cls(
            verbose=verbose,
            cwd=cwd,
            session_id=optional_strings["session_id"],
            model_profile_id=optional_strings["model_profile_id"],
        )


@dataclass(frozen=True, slots=True)
class FastTUIResult:
    """Result returned after the runtime child releases the terminal."""

    session_id: str | None = None


class _JsonSocket:
    """Bounded newline-delimited JSON over one private non-blocking socketpair."""

    def __init__(self, sock: socket.socket) -> None:
        self._socket = sock
        self._socket.setblocking(False)
        self._buffer = bytearray()

    @staticmethod
    def _encode(payload: dict[str, object]) -> bytes:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
        if len(encoded) > _MAX_MESSAGE_BYTES:
            raise ValueError("Startup protocol message exceeds the size limit")
        return encoded

    async def send(self, payload: dict[str, object]) -> None:
        loop = asyncio.get_running_loop()
        await loop.sock_sendall(self._socket, self._encode(payload))

    async def receive(self) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        while b"\n" not in self._buffer:
            chunk = await loop.sock_recv(self._socket, 65536)
            if not chunk:
                raise EOFError("Runtime child closed the startup channel")
            self._buffer.extend(chunk)
            if len(self._buffer) > _MAX_MESSAGE_BYTES:
                raise ValueError("Startup protocol message exceeds the size limit")
        raw, _, remainder = self._buffer.partition(b"\n")
        self._buffer[:] = remainder
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise TypeError("Startup protocol message must be a JSON object")
        return cast(dict[str, Any], payload)


class _SuppressedTerminalOutput:
    """Restore trusted terminal descriptors after bootstrap output was redirected."""

    def __init__(self, *, stdout_fd: int, stderr_fd: int) -> None:
        self._stdout_fd = stdout_fd
        self._stderr_fd = stderr_fd
        self._restored = False

    def restore(self) -> None:
        if self._restored:
            return
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(self._stdout_fd, sys.stdout.fileno())
        os.dup2(self._stderr_fd, sys.stderr.fileno())
        os.close(self._stdout_fd)
        os.close(self._stderr_fd)
        self._restored = True

    def close(self) -> None:
        if self._restored:
            return
        os.close(self._stdout_fd)
        os.close(self._stderr_fd)
        self._restored = True


class _StartupController:
    """Minimal controller for the canonical shell while the child initializes."""

    def __init__(self, request: FastTUIRequest) -> None:
        self.request = request
        self.application: Any = None
        self.shell: TUIShell | None = None
        self.pending_submit = False
        self.cancel_requested = False
        self.input_mode = "send"
        self.mouse_enabled = True
        self._last_ctrl_c_time = 0.0
        profile = request.model_profile_id or "loading"
        session = request.session_id or "loading"
        self._output = ANSI(
            "\x1b[1;35mYAACLI CLI\x1b[0m\n"
            f"Model: {profile}\n"
            "Type /help for commands; F2 expands tasks; Ctrl+L returns to live output.\n"
            f"Session: {session}"
        )

    def _invalidate(self) -> None:
        if self.application is not None:
            self.application.invalidate()

    def callbacks(self) -> TUIShellCallbacks:
        return TUIShellCallbacks(
            output_text=lambda: self._output,
            scroll_output=lambda _amount: None,
            task_text=lambda: [],
            task_height=lambda: 0,
            has_tasks=lambda: False,
            model_selector_text=lambda: [],
            model_selector_height=lambda: 1,
            model_selector_open=lambda: False,
            session_selector_text=lambda: [],
            session_selector_height=lambda: 1,
            session_selector_width=lambda: min(110, max(24, shutil.get_terminal_size((120, 40)).columns - 4)),
            session_selector_title=lambda: [],
            session_selector_open=lambda: False,
            status_text=self.status_text,
            status_height=lambda: 1,
            prompt=self.prompt,
            terminal_height=lambda: shutil.get_terminal_size((120, 40)).lines,
            input_key_bindings=self.input_key_bindings,
            application_key_bindings=self.application_key_bindings,
        )

    def status_text(self) -> StyleAndTextTuples:
        if self.pending_submit:
            return [
                ("class:status-bar.warning", " Starting runtime | prompt queued locally; Ctrl+C edits "),
            ]
        return [("class:status-bar", " Starting runtime | input is editable ")]

    def prompt(self) -> str:
        mouse_mode = "scroll" if self.mouse_enabled else "select"
        state = "queued" if self.pending_submit else self.input_mode
        return f"[{state} | {mouse_mode}] > "

    def input_key_bindings(self, input_area: TextArea) -> KeyBindings:
        bindings = KeyBindings()

        def submit_or_newline() -> None:
            if self.pending_submit:
                return
            if self.input_mode == "edit":
                input_area.buffer.insert_text("\n")
                return
            if input_area.buffer.text.strip():
                self.pending_submit = True
                self._invalidate()

        @bindings.add("enter", eager=True)
        def handle_enter(_event: object) -> None:
            submit_or_newline()

        @bindings.add("c-j", eager=True)
        def handle_ctrl_j(_event: object) -> None:
            submit_or_newline()

        @bindings.add("c-o", eager=True)
        def handle_ctrl_o(_event: object) -> None:
            if not self.pending_submit:
                input_area.buffer.insert_text("\n")

        return bindings

    def application_key_bindings(self, input_area: TextArea) -> KeyBindings:
        bindings = KeyBindings()

        @bindings.add("c-c")
        def handle_ctrl_c(_event: object) -> None:
            if self.pending_submit:
                self.pending_submit = False
                self._invalidate()
                return
            now = time.monotonic()
            if input_area.buffer.text:
                input_area.buffer.reset()
                self._last_ctrl_c_time = now
                return
            if now - self._last_ctrl_c_time < 2.0:
                self.cancel_requested = True
                self.application.exit()
            else:
                self._last_ctrl_c_time = now

        @bindings.add("c-d")
        def handle_ctrl_d(_event: object) -> None:
            if not input_area.buffer.text and not self.pending_submit:
                self.cancel_requested = True
                self.application.exit()

        @bindings.add("tab")
        def handle_tab(_event: object) -> None:
            if self.pending_submit:
                return
            self.input_mode = "edit" if self.input_mode == "send" else "send"
            self._invalidate()

        @bindings.add("c-u")
        def handle_ctrl_u(_event: object) -> None:
            if not self.pending_submit:
                input_area.buffer.reset()

        @bindings.add("escape")
        def handle_escape(_event: object) -> None:
            self.mouse_enabled = not self.mouse_enabled
            if self.mouse_enabled:
                self.application.output.enable_mouse_support()
            else:
                self.application.output.disable_mouse_support()
            self._invalidate()

        @bindings.add(Keys.ControlL)
        def handle_ctrl_l(_event: object) -> None:
            self._invalidate()

        return bindings


def can_use_fast_tui(*, cwd: Path | None = None) -> bool:
    """Return whether this invocation can safely perform a terminal handoff."""
    if os.name != "posix" or not sys.stdin.isatty() or not sys.stdout.isatty():
        return False
    terminal = os.environ.get("TERM", "")
    if not terminal or terminal == "dumb":
        return False
    working_dir = (cwd or Path.cwd()).resolve()
    return (working_dir / ".yaacli" / "config.toml").is_file() or (Path.home() / ".yaacli" / "config.toml").is_file()


def _read_startup_theme_preference(cwd: Path) -> ThemePreference:
    environment_value = os.environ.get("YAACLI_CODE_THEME", "").lower()
    if environment_value in {"auto", "dark", "light"}:
        return cast(ThemePreference, environment_value)
    for path in (cwd / ".yaacli" / "config.toml", Path.home() / ".yaacli" / "config.toml"):
        try:
            with path.open("rb") as stream:
                payload = tomllib.load(stream)
        except (OSError, tomllib.TOMLDecodeError):
            continue
        display = payload.get("display")
        if not isinstance(display, dict):
            continue
        value = display.get("code_theme")
        if isinstance(value, str) and value in {"auto", "dark", "light"}:
            return cast(ThemePreference, value)
    return "auto"


def _restore_alternate_screen(output: LeasedVt100Output) -> None:
    output.suppress_quit_alternate_screen = False
    output.quit_alternate_screen()
    output.flush()


def _runtime_error_from_message(
    message: dict[str, Any],
    *,
    fallback: str,
) -> FastTUIRuntimeError:
    error_type = message.get("error_type", "RuntimeError")
    detail = message.get("message", fallback)
    child_traceback = message.get("traceback")
    return FastTUIRuntimeError(
        f"{error_type}: {detail}",
        child_traceback=child_traceback if isinstance(child_traceback, str) else None,
    )


async def _run_parent_shell(
    request: FastTUIRequest,
    process: subprocess.Popen[bytes],
    channel: _JsonSocket,
    output: LeasedVt100Output,
) -> FastTUIResult:
    controller = _StartupController(request)
    theme = resolve_theme(_read_startup_theme_preference(Path(request.cwd)))
    await channel.send({
        "type": "start",
        "version": _PROTOCOL_VERSION,
        "request": asdict(request),
        "theme_variant": theme.variant,
    })
    shell = build_tui_shell(
        controller.callbacks(),
        style=Style.from_dict(prompt_toolkit_style_rules(theme)),
        output=output,
        input_read_only=Condition(lambda: controller.pending_submit),
    )
    controller.shell = shell
    controller.application = shell.application
    startup_message: dict[str, Any] | None = None
    lease_acquired = False
    lease_released = False

    async def monitor_child() -> None:
        nonlocal startup_message
        try:
            startup_message = await channel.receive()
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            startup_message = {"type": "error", "error_type": type(error).__name__, "message": str(error)}
        if shell.application.is_running:
            shell.application.exit()

    monitor_task: asyncio.Task[None] | None = None

    def pre_run() -> None:
        nonlocal monitor_task
        monitor_task = asyncio.create_task(monitor_child(), name="yaacli-runtime-child-startup")

    try:
        try:
            await shell.application.run_async(pre_run=pre_run)
        except (EOFError, KeyboardInterrupt):
            controller.cancel_requested = True

        if controller.cancel_requested:
            if monitor_task is not None:
                monitor_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await monitor_task
            with contextlib.suppress(OSError):
                await channel.send({"type": "cancel"})
            await _graceful_cancel_process(process)
            raise KeyboardInterrupt

        if monitor_task is not None:
            await monitor_task
        if startup_message is None:
            raise FastTUIRuntimeError("Runtime child did not report startup state")
        message_type = startup_message.get("type")
        if message_type == "setup_required":
            await asyncio.to_thread(_wait_for_process, process)
            raise FastTUISetupRequired
        if message_type != "ready":
            await asyncio.to_thread(_wait_for_process, process)
            raise _runtime_error_from_message(
                startup_message,
                fallback="Runtime child failed during startup",
            )

        snapshot = shell.capture_compose(
            submit_when_ready=controller.pending_submit,
            input_mode=controller.input_mode,
            mouse_enabled=controller.mouse_enabled,
        )
        await channel.send({"type": "handoff", "compose": asdict(snapshot)})

        lease_message = await channel.receive()
        if lease_message.get("type") != "lease_acquired":
            await asyncio.to_thread(_wait_for_process, process)
            raise _runtime_error_from_message(
                lease_message,
                fallback="Runtime child failed before acquiring the terminal",
            )
        lease_acquired = True
        release_message = await channel.receive()
        if release_message.get("type") != "lease_released":
            await asyncio.to_thread(_wait_for_process, process)
            raise _runtime_error_from_message(
                release_message,
                fallback="Runtime child failed while releasing the terminal",
            )
        lease_released = True

        terminal_message = await channel.receive()
        return_code = await asyncio.to_thread(_wait_for_process, process)
        if terminal_message.get("type") != "completed" or return_code != 0:
            raise _runtime_error_from_message(
                terminal_message,
                fallback=f"Runtime child exited with status {return_code}",
            )
        return FastTUIResult(session_id=terminal_message.get("session_id"))
    except BaseException:
        if not lease_released:
            child_released_lease = False
            if lease_acquired:
                child_released_lease = await _cancel_child_after_handoff(process, channel)
            if not child_released_lease:
                _restore_alternate_screen(output)
        raise


async def _cancel_child_after_handoff(
    process: subprocess.Popen[bytes],
    channel: _JsonSocket,
    *,
    grace_seconds: float = _CHILD_CLEANUP_GRACE_SECONDS,
) -> bool:
    """Cancel the terminal owner and wait for its release acknowledgement before recovery."""
    with contextlib.suppress(OSError, EOFError):
        await channel.send({"type": "cancel"})
    loop = asyncio.get_running_loop()
    deadline = loop.time() + grace_seconds
    lease_released = False
    while process.poll() is None and loop.time() < deadline:
        if lease_released:
            await asyncio.sleep(0.025)
            continue
        try:
            message = await asyncio.wait_for(channel.receive(), timeout=min(0.1, deadline - loop.time()))
        except TimeoutError:
            continue
        except (EOFError, OSError):
            break
        lease_released = message.get("type") == "lease_released"
    if process.poll() is None:
        await asyncio.to_thread(_terminate_process, process)
    return lease_released


async def _graceful_cancel_process(
    process: subprocess.Popen[bytes],
    *,
    grace_seconds: float = _CHILD_CLEANUP_GRACE_SECONDS,
) -> int:
    """Allow a runtime child to close its resources before escalating to signals."""
    deadline = asyncio.get_running_loop().time() + grace_seconds
    while process.poll() is None and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.025)
    if process.poll() is not None:
        return cast(int, process.returncode)
    return await asyncio.to_thread(_terminate_process, process)


def _wait_for_process(process: subprocess.Popen[bytes], *, timeout: float = 2.0) -> int:
    """Wait for a child and terminate it if protocol completion does not exit."""
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        return _terminate_process(process)


def _terminate_process(process: subprocess.Popen[bytes]) -> int:
    """Terminate one startup child without leaving the caller blocked indefinitely."""
    if process.poll() is not None:
        return cast(int, process.returncode)
    process.terminate()
    try:
        return process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.wait()


def _runtime_child_command() -> list[str]:
    """Return an isolated launcher that cannot import YAACLI from the workspace."""
    return [sys.executable, "-I", "-m", "yaacli.tui_startup", "--child"]


def run_fast_tui(request: FastTUIRequest) -> FastTUIResult:
    """Run the lightweight shell, then transfer the terminal to a cold child."""
    if not can_use_fast_tui(cwd=Path(request.cwd)):
        raise RuntimeError("Fast TUI startup is unavailable for this terminal")

    parent_socket: socket.socket | None = None
    child_socket: socket.socket | None = None
    process: subprocess.Popen[bytes] | None = None
    stdout_fd: int | None = None
    stderr_fd: int | None = None
    try:
        parent_socket, child_socket = socket.socketpair()
        stdout_fd = os.dup(sys.stdout.fileno())
        stderr_fd = os.dup(sys.stderr.fileno())
        env = os.environ.copy()
        env.pop("PYTHONHOME", None)
        env.pop("PYTHONPATH", None)
        env[_CHILD_FD_ENV] = str(child_socket.fileno())
        env[_CHILD_STDOUT_FD_ENV] = str(stdout_fd)
        env[_CHILD_STDERR_FD_ENV] = str(stderr_fd)
        process = subprocess.Popen(  # noqa: S603
            _runtime_child_command(),
            env=env,
            pass_fds=(child_socket.fileno(), stdout_fd, stderr_fd),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=False,
        )
        child_socket.close()
        child_socket = None
        os.close(stdout_fd)
        stdout_fd = None
        os.close(stderr_fd)
        stderr_fd = None

        channel = _JsonSocket(parent_socket)
        output = create_leased_output(enter=True, leave=False)
        return asyncio.run(_run_parent_shell(request, process, channel, output))
    finally:
        if child_socket is not None:
            child_socket.close()
        if parent_socket is not None:
            parent_socket.close()
        if stdout_fd is not None:
            os.close(stdout_fd)
        if stderr_fd is not None:
            os.close(stderr_fd)
        if process is not None and process.poll() is None:
            _terminate_process(process)


async def _run_runtime_child(
    request: FastTUIRequest,
    channel: _JsonSocket,
    suppressed_output: _SuppressedTerminalOutput,
    handoff_future: asyncio.Future[dict[str, Any]],
    runtime_theme: ResolvedTheme,
) -> None:
    from yaacli.app import TUIApp
    from yaacli.cli import _prepare_cli_runtime
    from yaacli.model_profiles import get_model_profile

    config_manager, config = _prepare_cli_runtime(request.verbose, allow_setup_wizard=False)
    model_profile = None
    if request.model_profile_id:
        model_profile = get_model_profile(config, request.model_profile_id)
        if model_profile is None:
            raise ValueError(f"Unknown model profile: {request.model_profile_id}")

    app = TUIApp(
        config=config,
        config_manager=config_manager,
        verbose=request.verbose,
        working_dir=Path(request.cwd),
        initial_session_id=request.session_id,
        query_terminal_on_enter=False,
    )
    async with app:
        app._apply_resolved_theme(runtime_theme, terminal_resolved=True)
        if model_profile is not None:
            await app._switch_model_profile(model_profile, persist=False)
        await app.prepare_startup_session()
        await channel.send({"type": "ready"})
        handoff = await handoff_future
        if handoff.get("type") != "handoff":
            raise ValueError("Invalid terminal handoff message")
        compose = ComposeSnapshot.from_payload(handoff.get("compose"))

        suppressed_output.restore()
        output = create_leased_output(enter=False, leave=True)
        await channel.send({"type": "lease_acquired"})
        try:
            await app.run(initial_compose=compose, output=output, resolved_theme=runtime_theme)
        finally:
            output.quit_alternate_screen()
            output.flush()
            await channel.send({"type": "lease_released"})
        session_id = app.session_id if app.has_session_data else None

    await channel.send({"type": "completed", "session_id": session_id})


async def _cancel_runtime_task(runtime_task: asyncio.Task[None]) -> None:
    runtime_task.cancel()
    with contextlib.suppress(asyncio.CancelledError, OSError, EOFError):
        await runtime_task


async def _run_child_session(
    request: FastTUIRequest,
    channel: _JsonSocket,
    suppressed_output: _SuppressedTerminalOutput,
    runtime_theme: ResolvedTheme,
) -> None:
    """Run the runtime while continuously treating the parent socket as its liveness lease."""
    loop = asyncio.get_running_loop()
    handoff_future: asyncio.Future[dict[str, Any]] = loop.create_future()
    runtime_task = asyncio.create_task(
        _run_runtime_child(request, channel, suppressed_output, handoff_future, runtime_theme),
        name="yaacli-runtime-child",
    )
    control_task: asyncio.Task[dict[str, Any]] | None = asyncio.create_task(
        channel.receive(),
        name="yaacli-runtime-child-control",
    )
    liveness_task: asyncio.Task[dict[str, Any]] | None = None
    try:
        done, _pending = await asyncio.wait({runtime_task, control_task}, return_when=asyncio.FIRST_COMPLETED)
        if runtime_task in done:
            await runtime_task
            return

        control_message = control_task.result()
        control_task = None
        if control_message.get("type") == "cancel":
            await _cancel_runtime_task(runtime_task)
            return

        handoff_future.set_result(control_message)
        liveness_task = asyncio.create_task(channel.receive(), name="yaacli-runtime-parent-liveness")
        done, _pending = await asyncio.wait({runtime_task, liveness_task}, return_when=asyncio.FIRST_COMPLETED)
        if runtime_task in done:
            await runtime_task
            return

        try:
            liveness_message = liveness_task.result()
        except (EOFError, OSError):
            await _cancel_runtime_task(runtime_task)
            return
        liveness_task = None
        if liveness_message.get("type") != "cancel":
            await _cancel_runtime_task(runtime_task)
            raise ValueError("Unexpected runtime child control message")
        await _cancel_runtime_task(runtime_task)
    finally:
        for receive_task in (control_task, liveness_task):
            if receive_task is not None and not receive_task.done():
                receive_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await receive_task
        if not runtime_task.done():
            await _cancel_runtime_task(runtime_task)


async def _run_child_main() -> int:
    fd_text = os.environ.pop(_CHILD_FD_ENV, None)
    stdout_fd_text = os.environ.pop(_CHILD_STDOUT_FD_ENV, None)
    stderr_fd_text = os.environ.pop(_CHILD_STDERR_FD_ENV, None)
    if fd_text is None or stdout_fd_text is None or stderr_fd_text is None:
        raise RuntimeError("Missing startup child file descriptors")
    fd = int(fd_text)
    stdout_fd = int(stdout_fd_text)
    stderr_fd = int(stderr_fd_text)
    for inherited_fd in (fd, stdout_fd, stderr_fd):
        os.set_inheritable(inherited_fd, False)

    sock = socket.socket(fileno=fd)
    channel = _JsonSocket(sock)
    suppressed_output = _SuppressedTerminalOutput(stdout_fd=stdout_fd, stderr_fd=stderr_fd)
    loop = asyncio.get_running_loop()
    child_task = asyncio.current_task()
    installed_signal_handlers: list[signal.Signals] = []
    if child_task is not None:
        for child_signal in (signal.SIGHUP, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError, RuntimeError):
                loop.add_signal_handler(child_signal, child_task.cancel)
                installed_signal_handlers.append(child_signal)
    try:
        start = await channel.receive()
        if start.get("type") != "start" or start.get("version") != _PROTOCOL_VERSION:
            raise RuntimeError("Unsupported startup protocol")
        request = FastTUIRequest.from_payload(start.get("request"))
        theme_variant = start.get("theme_variant")
        if theme_variant not in {"dark", "light"}:
            raise ValueError("Invalid startup theme")
        runtime_theme = resolve_theme(cast(ThemePreference, theme_variant))
        await _run_child_session(request, channel, suppressed_output, runtime_theme)
        return 0
    except asyncio.CancelledError:
        return 130
    except BaseException as error:
        from yaacli.cli import SetupWizardRequired

        if isinstance(error, SetupWizardRequired):
            payload: dict[str, object] = {"type": "setup_required"}
        else:
            payload = {
                "type": "error",
                "error_type": type(error).__name__,
                "message": str(error) or repr(error),
                "traceback": "".join(traceback.format_exception(error))[-32768:],
            }
        with contextlib.suppress(OSError, EOFError):
            await channel.send(payload)
        return 1
    finally:
        for child_signal in installed_signal_handlers:
            loop.remove_signal_handler(child_signal)
        suppressed_output.close()
        sock.close()


def _child_main() -> None:
    raise SystemExit(asyncio.run(_run_child_main()))


def main() -> NoReturn:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if not args.child:
        parser.error("This module is an internal YAACLI entry point")
    _child_main()
    raise SystemExit(0)


if __name__ == "__main__":
    main()
