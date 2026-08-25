"""Tests for YAACLI's lightweight TUI startup boundary."""

from __future__ import annotations

import asyncio
import json
import os
import select
import socket
import subprocess
import sys
import textwrap
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from click.testing import CliRunner
from prompt_toolkit.output.vt100 import Vt100_Output
from prompt_toolkit.selection import SelectionState, SelectionType
from prompt_toolkit.widgets import TextArea
from yaacli.app.shell import ComposeSnapshot, LeasedVt100Output
from yaacli.cli import cli
from yaacli.tui_startup import (
    FastTUIRequest,
    FastTUIResult,
    FastTUISetupRequired,
    _JsonSocket,
    _run_child_session,
    _run_parent_shell,
    _runtime_child_command,
    can_use_fast_tui,
)


def test_compose_snapshot_round_trips_editing_state() -> None:
    source = TextArea(multiline=True)
    source.buffer.text = "hello\nworld"
    source.buffer.cursor_position = 3
    source.buffer.selection_state = SelectionState(
        original_cursor_position=9,
        type=SelectionType.LINES,
    )

    captured = ComposeSnapshot.capture(
        source,
        submit_when_ready=True,
        input_mode="edit",
        mouse_enabled=False,
    )
    decoded = ComposeSnapshot.from_payload(json.loads(json.dumps(asdict(captured))))
    restored = TextArea(multiline=True)
    decoded.restore(restored)

    assert restored.buffer.text == "hello\nworld"
    assert restored.buffer.cursor_position == 3
    assert restored.buffer.selection_state is not None
    assert restored.buffer.selection_state.original_cursor_position == 9
    assert restored.buffer.selection_state.type == SelectionType.LINES
    assert decoded.submit_when_ready is True
    assert decoded.input_mode == "edit"
    assert decoded.mouse_enabled is False


@pytest.mark.parametrize(
    ("payload", "error_type"),
    [
        ({"text": 1}, TypeError),
        ({"cursor_position": True}, TypeError),
        ({"selection_type": "unknown"}, ValueError),
        ({"input_mode": "unknown"}, ValueError),
        ({"unexpected": "field"}, ValueError),
    ],
)
def test_compose_snapshot_rejects_invalid_protocol_payloads(
    payload: dict[str, object],
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        ComposeSnapshot.from_payload(payload)


def test_fast_tui_request_validates_protocol_payload() -> None:
    request = FastTUIRequest.from_payload({
        "verbose": True,
        "cwd": "/workspace",
        "session_id": "session-1",
        "model_profile_id": "fast",
    })

    assert request == FastTUIRequest(
        verbose=True,
        cwd="/workspace",
        session_id="session-1",
        model_profile_id="fast",
    )
    with pytest.raises(TypeError, match="verbose"):
        FastTUIRequest.from_payload({"verbose": 1, "cwd": "/workspace"})
    with pytest.raises(ValueError, match="Unknown"):
        FastTUIRequest.from_payload({"verbose": False, "cwd": "/workspace", "extra": True})


@pytest.mark.asyncio
async def test_json_socket_preserves_buffered_messages_and_enforces_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left, right = socket.socketpair()
    try:
        sender = _JsonSocket(left)
        receiver = _JsonSocket(right)
        await sender.send({"type": "ready"})
        await sender.send({"type": "completed", "session_id": "session-1"})

        assert await receiver.receive() == {"type": "ready"}
        assert await receiver.receive() == {"type": "completed", "session_id": "session-1"}

        monkeypatch.setattr("yaacli.tui_startup._MAX_MESSAGE_BYTES", 8)
        with pytest.raises(ValueError, match="size limit"):
            await sender.send({"too": "large"})
    finally:
        left.close()
        right.close()


def test_leased_output_suppresses_only_unowned_screen_transitions(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(Vt100_Output, "enter_alternate_screen", lambda _self: calls.append("enter"))
    monkeypatch.setattr(Vt100_Output, "quit_alternate_screen", lambda _self: calls.append("quit"))
    output = object.__new__(LeasedVt100Output)

    output.alternate_screen_active = True
    output.suppress_enter_alternate_screen = True
    output.suppress_quit_alternate_screen = False
    output.enter_alternate_screen()
    output.quit_alternate_screen()
    output.quit_alternate_screen()

    output.suppress_enter_alternate_screen = False
    output.suppress_quit_alternate_screen = True
    output.enter_alternate_screen()
    output.enter_alternate_screen()
    output.quit_alternate_screen()

    assert calls == ["quit", "enter"]


def test_fast_tui_requires_posix_tty_term_and_existing_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stdin = MagicMock()
    stdout = MagicMock()
    stdin.isatty.return_value = True
    stdout.isatty.return_value = True
    monkeypatch.setattr("yaacli.tui_startup.os.name", "posix")
    monkeypatch.setattr("yaacli.tui_startup.sys.stdin", stdin)
    monkeypatch.setattr("yaacli.tui_startup.sys.stdout", stdout)
    monkeypatch.setattr("yaacli.tui_startup.Path.home", lambda: tmp_path / "home")
    monkeypatch.setenv("TERM", "xterm-256color")

    assert can_use_fast_tui(cwd=tmp_path) is False

    config_path = tmp_path / ".yaacli" / "config.toml"
    config_path.parent.mkdir()
    config_path.write_text("[general]\n", encoding="utf-8")
    assert can_use_fast_tui(cwd=tmp_path) is True

    monkeypatch.setenv("TERM", "dumb")
    assert can_use_fast_tui(cwd=tmp_path) is False


@pytest.mark.asyncio
async def test_parent_cancel_does_not_wait_for_runtime_readiness(monkeypatch: pytest.MonkeyPatch) -> None:
    monitor_started = asyncio.Event()
    monitor_cancelled = asyncio.Event()

    class HangingChannel:
        async def receive(self) -> dict[str, object]:
            monitor_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                monitor_cancelled.set()
                raise

        async def send(self, _payload: dict[str, object]) -> None:
            return None

    class CancelledApplication:
        async def run_async(self, *, pre_run: object) -> None:
            assert callable(pre_run)
            pre_run()
            await monitor_started.wait()
            raise EOFError

        def exit(self) -> None:
            return None

        def invalidate(self) -> None:
            return None

    shell = MagicMock()
    shell.application = CancelledApplication()
    shell.input_area = TextArea(multiline=True)
    process = MagicMock(spec=subprocess.Popen)
    output = MagicMock(spec=LeasedVt100Output)
    cancel_process = AsyncMock(return_value=130)
    restore = MagicMock()
    monkeypatch.setattr("yaacli.tui_startup.build_tui_shell", MagicMock(return_value=shell))
    monkeypatch.setattr("yaacli.tui_startup._graceful_cancel_process", cancel_process)
    monkeypatch.setattr("yaacli.tui_startup._restore_alternate_screen", restore)

    with pytest.raises(KeyboardInterrupt):
        await asyncio.wait_for(
            _run_parent_shell(
                FastTUIRequest(verbose=False, cwd=os.getcwd()),
                process,
                HangingChannel(),  # type: ignore[arg-type]
                output,
            ),
            timeout=1,
        )

    assert monitor_cancelled.is_set()
    cancel_process.assert_awaited_once_with(process)
    restore.assert_called_once_with(output)


@pytest.mark.asyncio
async def test_child_cancel_interrupts_runtime_initialization(monkeypatch: pytest.MonkeyPatch) -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def fake_runtime_child(*_args: object) -> None:
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    class CancelChannel:
        async def receive(self) -> dict[str, object]:
            await started.wait()
            return {"type": "cancel"}

    monkeypatch.setattr("yaacli.tui_startup._run_runtime_child", fake_runtime_child)

    await _run_child_session(
        FastTUIRequest(verbose=False, cwd=os.getcwd()),
        CancelChannel(),  # type: ignore[arg-type]
        MagicMock(),
        MagicMock(),
    )

    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_child_parent_eof_after_handoff_cancels_terminal_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    handoff_received = asyncio.Event()
    cancelled = asyncio.Event()

    async def fake_runtime_child(
        _request: object,
        _channel: object,
        _output: object,
        handoff_future: asyncio.Future[dict[str, object]],
        _theme: object,
    ) -> None:
        assert (await handoff_future)["type"] == "handoff"
        handoff_received.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    class ParentDiesChannel:
        def __init__(self) -> None:
            self.calls = 0

        async def receive(self) -> dict[str, object]:
            self.calls += 1
            if self.calls == 1:
                return {"type": "handoff", "compose": {}}
            await handoff_received.wait()
            raise EOFError("parent exited")

    monkeypatch.setattr("yaacli.tui_startup._run_runtime_child", fake_runtime_child)

    await _run_child_session(
        FastTUIRequest(verbose=False, cwd=os.getcwd()),
        ParentDiesChannel(),  # type: ignore[arg-type]
        MagicMock(),
        MagicMock(),
    )

    assert cancelled.is_set()


def test_cli_routes_interactive_tty_through_fast_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    run_fast = MagicMock(return_value=FastTUIResult(session_id="session-1"))
    prepare_runtime = MagicMock()
    monkeypatch.setattr("yaacli.tui_startup.can_use_fast_tui", MagicMock(return_value=True))
    monkeypatch.setattr("yaacli.tui_startup.run_fast_tui", run_fast)
    monkeypatch.setattr("yaacli.cli._prepare_cli_runtime", prepare_runtime)

    result = CliRunner().invoke(cli, ["--session", "session-0", "--profile", "fast"])

    assert result.exit_code == 0
    request = run_fast.call_args.args[0]
    assert request.session_id == "session-0"
    assert request.model_profile_id == "fast"
    assert "Session: session-1" in result.output
    prepare_runtime.assert_not_called()


def test_cli_worktree_invocation_uses_direct_lifecycle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = MagicMock()
    config_manager = MagicMock()
    run_fast = MagicMock()
    monkeypatch.setattr("yaacli.tui_startup.can_use_fast_tui", MagicMock(return_value=True))
    monkeypatch.setattr("yaacli.tui_startup.run_fast_tui", run_fast)
    monkeypatch.setattr("yaacli.cli._prepare_cli_runtime", MagicMock(return_value=(config_manager, config)))
    monkeypatch.setattr(
        "yaacli.cli._create_worktree",
        MagicMock(return_value=(tmp_path / "worktree", "feature/startup", False)),
    )
    monkeypatch.setattr("yaacli.cli._run_tui", MagicMock(return_value="tui-coro"))
    monkeypatch.setattr("yaacli.cli.asyncio.run", MagicMock(return_value=None))

    result = CliRunner().invoke(cli, ["--worktree"])

    assert result.exit_code == 0
    run_fast.assert_not_called()


def test_isolated_child_module_resolution_ignores_workspace_shadow(tmp_path: Path) -> None:
    assert _runtime_child_command()[1:] == ["-I", "-m", "yaacli.tui_startup", "--child"]
    fake_package = tmp_path / "yaacli"
    fake_package.mkdir()
    (fake_package / "__init__.py").write_text("", encoding="utf-8")
    marker = tmp_path / "hijacked"
    (fake_package / "tui_startup.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('hijacked')\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import pathlib, yaacli.tui_startup as module; print(pathlib.Path(module.__file__).resolve())",
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert marker.exists() is False
    assert not Path(result.stdout.strip()).is_relative_to(fake_package)


def test_cli_setup_required_falls_back_to_direct_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    config = MagicMock()
    config_manager = MagicMock()
    monkeypatch.setattr("yaacli.tui_startup.can_use_fast_tui", MagicMock(return_value=True))
    monkeypatch.setattr("yaacli.tui_startup.run_fast_tui", MagicMock(side_effect=FastTUISetupRequired))
    prepare_runtime = MagicMock(return_value=(config_manager, config))
    monkeypatch.setattr("yaacli.cli._prepare_cli_runtime", prepare_runtime)
    monkeypatch.setattr("yaacli.cli._run_tui", MagicMock(return_value="tui-coro"))
    monkeypatch.setattr("yaacli.cli.asyncio.run", MagicMock(return_value=None))

    result = CliRunner().invoke(cli)

    assert result.exit_code == 0
    prepare_runtime.assert_called_once_with(False)


def _run_pty_harness(  # noqa: C901
    harness_path: Path,
    child_path: Path,
    *,
    input_after_first_paint: bytes = b"",
    timeout: float = 10.0,
) -> tuple[int, bytes, bool]:
    master_fd, slave_fd = os.openpty()
    environment = os.environ.copy()
    environment["TERM"] = "xterm-256color"
    environment["YAACLI_CODE_THEME"] = "dark"
    process = subprocess.Popen(
        [sys.executable, str(harness_path), str(child_path)],
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        env=environment,
        close_fds=True,
    )
    os.close(slave_fd)
    output = bytearray()
    sent_input = False
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            ready, _, _ = select.select([master_fd], [], [], 0.05)
            if ready:
                try:
                    chunk = os.read(master_fd, 65536)
                except OSError:
                    chunk = b""
                if chunk:
                    output.extend(chunk)
                    if not sent_input and b"YAACLI CLI" in output:
                        if input_after_first_paint:
                            os.write(master_fd, input_after_first_paint)
                        sent_input = True
            if process.poll() is not None:
                while True:
                    ready, _, _ = select.select([master_fd], [], [], 0)
                    if not ready:
                        break
                    try:
                        chunk = os.read(master_fd, 65536)
                    except OSError:
                        break
                    if not chunk:
                        break
                    output.extend(chunk)
                break
        else:
            process.kill()
            process.wait()
            pytest.fail(f"PTY harness timed out. Output: {bytes(output)!r}")
        return process.wait(), bytes(output), sent_input
    finally:
        os.close(master_fd)


def _write_pty_harness(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """
            from __future__ import annotations

            import asyncio
            import os
            import socket
            import subprocess
            import sys

            from yaacli.app.shell import create_leased_output
            from yaacli.tui_startup import FastTUIRequest, _JsonSocket, _run_parent_shell

            parent_socket, child_socket = socket.socketpair()
            environment = os.environ.copy()
            environment["FAKE_STARTUP_FD"] = str(child_socket.fileno())
            process = subprocess.Popen(
                [sys.executable, sys.argv[1]],
                env=environment,
                pass_fds=(child_socket.fileno(),),
            )
            child_socket.close()
            output = create_leased_output(enter=True, leave=False)
            try:
                asyncio.run(
                    _run_parent_shell(
                        FastTUIRequest(verbose=False, cwd=os.getcwd()),
                        process,
                        _JsonSocket(parent_socket),
                        output,
                    )
                )
            except RuntimeError as error:
                print(f"HARNESS_ERROR={error}", flush=True)
            else:
                print("HARNESS_OK", flush=True)
            finally:
                parent_socket.close()
            """
        ),
        encoding="utf-8",
    )


@pytest.mark.skipif(os.name != "posix", reason="PTY terminal handoff requires POSIX")
def test_real_pty_handoff_preserves_draft_and_uses_one_screen_lease(tmp_path: Path) -> None:
    harness_path = tmp_path / "harness.py"
    child_path = tmp_path / "ready_child.py"
    _write_pty_harness(harness_path)
    child_path.write_text(
        textwrap.dedent(
            """
            from __future__ import annotations

            import asyncio
            import json
            import os
            import socket
            import sys

            from yaacli.app.shell import ComposeSnapshot, create_leased_output
            from yaacli.tui_startup import _JsonSocket

            async def main() -> None:
                sock = socket.socket(fileno=int(os.environ["FAKE_STARTUP_FD"]))
                channel = _JsonSocket(sock)
                await channel.receive()
                await asyncio.sleep(0.75)
                await channel.send({"type": "ready"})
                handoff = await channel.receive()
                snapshot = ComposeSnapshot.from_payload(handoff.get("compose"))
                marker = json.dumps({
                    "text": snapshot.text,
                    "cursor_position": snapshot.cursor_position,
                    "submit_when_ready": snapshot.submit_when_ready,
                }, separators=(",", ":"))
                os.write(sys.stdout.fileno(), f"\\r\\nHANDOFF={marker}\\r\\n".encode())
                output = create_leased_output(enter=False, leave=True)
                await channel.send({"type": "lease_acquired"})
                output.quit_alternate_screen()
                output.flush()
                await channel.send({"type": "lease_released"})
                await channel.send({"type": "completed", "session_id": None})
                sock.close()

            asyncio.run(main())
            """
        ),
        encoding="utf-8",
    )

    return_code, output, sent_input = _run_pty_harness(
        harness_path,
        child_path,
        input_after_first_paint=b"draft\x1b[D\x1b[D\r",
    )

    assert return_code == 0, output.decode(errors="replace")
    assert sent_input is True
    assert b' HANDOFF={"text":"draft","cursor_position":3,"submit_when_ready":true}' in output.replace(b"\r\n", b" ")
    assert output.count(b"\x1b[?1049h") == 1
    assert output.count(b"\x1b[?1049l") == 1
    assert b"HARNESS_OK" in output


@pytest.mark.skipif(os.name != "posix", reason="PTY terminal handoff requires POSIX")
def test_real_pty_child_startup_failure_restores_terminal(tmp_path: Path) -> None:
    harness_path = tmp_path / "harness.py"
    child_path = tmp_path / "failing_child.py"
    _write_pty_harness(harness_path)
    child_path.write_text(
        textwrap.dedent(
            """
            from __future__ import annotations

            import asyncio
            import os
            import socket

            from yaacli.tui_startup import _JsonSocket

            async def main() -> None:
                sock = socket.socket(fileno=int(os.environ["FAKE_STARTUP_FD"]))
                channel = _JsonSocket(sock)
                await channel.receive()
                await asyncio.sleep(0.25)
                await channel.send({
                    "type": "error",
                    "error_type": "FakeStartupError",
                    "message": "runtime failed",
                })
                sock.close()

            asyncio.run(main())
            raise SystemExit(1)
            """
        ),
        encoding="utf-8",
    )

    return_code, output, _sent_input = _run_pty_harness(harness_path, child_path)

    assert return_code == 0, output.decode(errors="replace")
    assert output.count(b"\x1b[?1049h") == 1
    assert output.count(b"\x1b[?1049l") == 1
    assert b"HARNESS_ERROR=FakeStartupError: runtime failed" in output


@pytest.mark.skipif(os.name != "posix", reason="PTY terminal handoff requires POSIX")
def test_real_pty_runtime_failure_releases_screen_once(tmp_path: Path) -> None:
    harness_path = tmp_path / "harness.py"
    child_path = tmp_path / "runtime_failure_child.py"
    _write_pty_harness(harness_path)
    child_path.write_text(
        textwrap.dedent(
            """
            from __future__ import annotations

            import asyncio
            import os
            import socket

            from yaacli.app.shell import create_leased_output
            from yaacli.tui_startup import _JsonSocket

            async def main() -> None:
                sock = socket.socket(fileno=int(os.environ["FAKE_STARTUP_FD"]))
                channel = _JsonSocket(sock)
                await channel.receive()
                await channel.send({"type": "ready"})
                await channel.receive()
                output = create_leased_output(enter=False, leave=True)
                await channel.send({"type": "lease_acquired"})
                output.quit_alternate_screen()
                output.flush()
                await channel.send({"type": "lease_released"})
                await channel.send({
                    "type": "error",
                    "error_type": "FakeRuntimeError",
                    "message": "runtime crashed",
                })
                sock.close()

            asyncio.run(main())
            raise SystemExit(1)
            """
        ),
        encoding="utf-8",
    )

    return_code, output, _sent_input = _run_pty_harness(harness_path, child_path)

    assert return_code == 0, output.decode(errors="replace")
    assert output.count(b"\x1b[?1049h") == 1
    assert output.count(b"\x1b[?1049l") == 1
    assert b"HARNESS_ERROR=FakeRuntimeError: runtime crashed" in output
