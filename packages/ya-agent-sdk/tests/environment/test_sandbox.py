"""Tests for SandboxEnvironment and DockerShell.

These tests require Docker to be installed.
The entire module is skipped when docker package is unavailable.
"""

import asyncio
import contextlib
import errno
import os
import shlex
import signal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from ya_agent_environment import ShellExecutionError, ShellTimeoutError

# Skip all tests in this module if docker is not installed
docker = pytest.importorskip("docker")

import ya_agent_sdk.environment.sandbox as sandbox_module  # noqa: E402
from ya_agent_sdk.environment.local import VirtualMount  # noqa: E402
from ya_agent_sdk.environment.sandbox import DeferredDockerShell, DockerShell, SandboxEnvironment  # noqa: E402
from ya_agent_sdk.environment.virtual_path import normalize_virtual_path  # noqa: E402

# --- DockerShell Tests ---


def _mock_docker_exec_process(
    monkeypatch: pytest.MonkeyPatch,
    *,
    exit_code: int = 0,
    stdout: bytes = b"",
    stderr: bytes = b"",
) -> tuple[MagicMock, AsyncMock]:
    process = MagicMock()
    process.stdout = asyncio.StreamReader()
    process.stderr = asyncio.StreamReader()
    process.stdin = MagicMock()
    process.stdin.drain = AsyncMock()
    process.stdin.wait_closed = AsyncMock()
    process.pid = 1234
    process.returncode = exit_code
    process.communicate = AsyncMock(return_value=(stdout, stderr))
    process.wait = AsyncMock(return_value=exit_code)
    create_process = AsyncMock(return_value=process)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)

    async def _register(
        _self: DockerShell,
        _process: object,
        _pidfile: str,
        _token: str,
        registration: sandbox_module._DockerExecRegistration,
        _handshake_lock: asyncio.Lock,
    ) -> None:
        registration.identity = ("1234", "5678")

    monkeypatch.setattr(DockerShell, "_register_container_exec", _register)
    monkeypatch.setattr(DockerShell, "_confirm_container_exec_completion", AsyncMock())
    return process, create_process


def test_docker_shell_initialization() -> None:
    """Should initialize with container_id and container_workdir."""

    shell = DockerShell(
        container_id="test123",
        container_workdir="/app",
        default_timeout=60.0,
    )
    assert shell._container_id == "test123"
    assert shell._container_workdir == "/app"
    assert shell._default_timeout == 60.0
    assert shell._exec_user is None
    assert shell._default_env == {}


async def test_docker_shell_execute_empty_command() -> None:
    """Should raise error for empty command."""

    shell = DockerShell(container_id="test123")
    with pytest.raises(ShellExecutionError):
        await shell.execute("")


async def test_docker_shell_missing_cli_reports_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing host CLI should explain that mounting docker.sock is insufficient."""
    shell = DockerShell(container_id="test123")

    async def missing_cli(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError(errno.ENOENT, "No such file or directory", "docker")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", missing_cli)

    with pytest.raises(ShellExecutionError) as exc_info:
        await shell.execute("echo hello")

    assert isinstance(exc_info.value.__cause__, FileNotFoundError)
    assert "Docker CLI executable 'docker' was not found" in str(exc_info.value)
    assert "access to the Docker socket alone is not sufficient" in str(exc_info.value)


async def test_docker_shell_execute_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Foreground execution should use and remotely confirm the owned exec handle."""
    shell = DockerShell(container_id="test123")
    _process, create_process = _mock_docker_exec_process(monkeypatch, stdout=b"hello\n")
    confirm_completion = AsyncMock()
    monkeypatch.setattr(shell, "_confirm_container_exec_completion", confirm_completion)

    code, stdout, stderr = await shell.execute("echo hello")

    assert (code, stdout, stderr) == (0, "hello\n", "")
    confirm_completion.assert_awaited_once()
    args = create_process.await_args.args
    assert args[:6] == ("docker", "exec", "-i", "-w", "/workspace", "test123")
    assert args[-7:-4] == ("/bin/sh", "-c", sandbox_module._DOCKER_EXEC_WRAPPER)
    assert args[-4] == "ya-agent-exec"
    assert args[-3].startswith("/tmp/.ya-agent-exec-")  # noqa: S108
    assert len(args[-2]) == 32
    assert all(character in "0123456789abcdef" for character in args[-2])
    assert args[-1] == "echo hello"


async def test_docker_shell_timeout_can_interrupt_slow_registration_after_handle_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registration delay is inside Shell's timeout-owned handle lifecycle."""
    shell = DockerShell(container_id="test123")
    process, _create_process = _mock_docker_exec_process(monkeypatch)
    process.returncode = None
    never_registered = asyncio.Event()

    async def _stall_registration(
        _process: object,
        _pidfile: str,
        _token: str,
        _registration: sandbox_module._DockerExecRegistration,
        _handshake_lock: asyncio.Lock,
    ) -> None:
        await never_registered.wait()

    abort_unregistered = AsyncMock()
    terminate_container_exec = AsyncMock()
    stop_local_docker_exec = AsyncMock()
    monkeypatch.setattr(shell, "_register_container_exec", _stall_registration)
    monkeypatch.setattr(shell, "_abort_unregistered_container_exec", abort_unregistered)
    monkeypatch.setattr(shell, "_terminate_container_exec", terminate_container_exec)
    monkeypatch.setattr(shell, "_stop_local_docker_exec", stop_local_docker_exec)
    monkeypatch.setattr(sandbox_module, "_DOCKER_EXEC_KILL_REGISTRATION_GRACE", 0.01)

    with pytest.raises(ShellTimeoutError):
        await shell.execute("sleep 10", timeout=0.01)

    abort_unregistered.assert_awaited_once_with(process)
    terminate_container_exec.assert_not_awaited()
    stop_local_docker_exec.assert_awaited_once_with(process)
    assert shell._execution_handles == {}


async def test_docker_shell_registration_persists_identity_before_acknowledgement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact-channel identity remains available if releasing the remote command fails."""
    shell = DockerShell(container_id="test123")
    process = MagicMock()
    wait_for_registration = AsyncMock(return_value=("42", "123456"))
    acknowledge = AsyncMock(side_effect=RuntimeError("ack response lost"))
    monkeypatch.setattr(shell, "_wait_for_container_exec_registration", wait_for_registration)
    monkeypatch.setattr(shell, "_ack_container_exec_registration", acknowledge)

    registration = sandbox_module._DockerExecRegistration()
    await shell._register_container_exec(
        process,
        "/tmp/exec.pid",  # noqa: S108
        "registration-token",
        registration,
        asyncio.Lock(),
    )

    assert registration.identity == ("42", "123456")
    assert isinstance(registration.error, RuntimeError)
    assert str(registration.error) == "ack response lost"
    wait_for_registration.assert_awaited_once_with(process, "registration-token")
    acknowledge.assert_awaited_once_with(
        process,
        "/tmp/exec.pid",  # noqa: S108
        "registration-token",
        registration.identity,
    )


async def test_docker_shell_registration_uses_exact_transport_and_strips_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registration consumes only the exact exec frame before exposing command stderr."""
    shell = DockerShell(container_id="test123")
    process = MagicMock()
    process.stderr = asyncio.StreamReader()
    process.stderr.feed_data(b"ya-agent-exec transport-nonce 42 123456\ncommand stderr\n")
    process.stderr.feed_eof()
    process.stdin = MagicMock()
    process.stdin.drain = AsyncMock()
    process.returncode = None
    helper_result = MagicMock(returncode=0, stderr=b"")
    run_helper = AsyncMock(return_value=helper_result)
    monkeypatch.setattr(shell, "_run_docker_helper", run_helper)
    registration = sandbox_module._DockerExecRegistration()

    await shell._register_container_exec(
        process,
        "/tmp/exec.pid",  # noqa: S108
        "transport-nonce",
        registration,
        asyncio.Lock(),
    )

    assert registration.identity == ("42", "123456")
    assert registration.error is None
    process.stdin.write.assert_called_once_with(b"ya-agent-ack transport-nonce\n")
    process.stdin.drain.assert_awaited_once_with()
    assert await process.stderr.readline() == b"command stderr\n"
    helper_args = run_helper.await_args.args[0]
    assert helper_args[-4:] == ["/tmp/exec.pid", "transport-nonce", "42", "123456"]  # noqa: S108


async def test_docker_shell_cancel_waits_for_exec_termination(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancellation should signal docker exec and retain ownership until it exits."""
    shell = DockerShell(container_id="test123")
    terminate_container_exec = AsyncMock()
    monkeypatch.setattr(shell, "_terminate_container_exec", terminate_container_exec)
    monkeypatch.setattr(shell, "_wait_for_container_exec_registration", AsyncMock())
    process, _create_process = _mock_docker_exec_process(monkeypatch)
    process.returncode = None
    communicate_started = asyncio.Event()
    exited = asyncio.Event()

    async def _communicate() -> tuple[bytes, bytes]:
        communicate_started.set()
        await exited.wait()
        return b"", b""

    async def _wait() -> int:
        await exited.wait()
        return -15

    def _terminate() -> None:
        process.returncode = -15
        exited.set()

    process.communicate = AsyncMock(side_effect=_communicate)
    process.wait = AsyncMock(side_effect=_wait)
    process.terminate.side_effect = _terminate

    execute_task = asyncio.create_task(shell.execute("sleep 10"))
    await communicate_started.wait()
    execute_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await execute_task
    terminate_container_exec.assert_awaited_once()
    process.terminate.assert_called_once_with()
    assert shell._execution_handles == {}
    assert shell._foreground_tasks == {}


async def test_docker_shell_remote_kill_failure_is_retained_for_reset_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unconfirmed remote kill must retain the handle until a retry succeeds."""
    shell = DockerShell(container_id="test123")
    terminate_container_exec = AsyncMock(side_effect=[RuntimeError("remote termination unconfirmed"), None])
    monkeypatch.setattr(shell, "_terminate_container_exec", terminate_container_exec)
    monkeypatch.setattr(shell, "_wait_for_container_exec_registration", AsyncMock())
    process, _create_process = _mock_docker_exec_process(monkeypatch)
    process.returncode = None
    communicate_started = asyncio.Event()
    exited = asyncio.Event()

    async def _communicate() -> tuple[bytes, bytes]:
        communicate_started.set()
        await exited.wait()
        return b"", b""

    async def _wait() -> int:
        await exited.wait()
        return -15

    def _terminate() -> None:
        process.returncode = -15
        exited.set()

    process.communicate = AsyncMock(side_effect=_communicate)
    process.wait = AsyncMock(side_effect=_wait)
    process.terminate.side_effect = _terminate

    execute_task = asyncio.create_task(shell.execute("sleep 10"))
    await communicate_started.wait()
    execute_task.cancel()

    with pytest.raises(RuntimeError, match="remote termination unconfirmed"):
        await execute_task
    assert len(shell._execution_handles) == 1

    await shell.reset_background_processes()

    assert terminate_container_exec.await_count == 2
    assert shell._execution_handles == {}
    assert shell._background_cleanup_errors == {}


async def test_docker_shell_registration_failure_retains_handle_when_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed handshake must retain ownership when exact-channel abort fails."""
    shell = DockerShell(container_id="test123")
    process, _create_process = _mock_docker_exec_process(monkeypatch)
    process.returncode = None

    async def _fail_registration(
        _process: object,
        _pidfile: str,
        _token: str,
        registration: sandbox_module._DockerExecRegistration,
        _handshake_lock: asyncio.Lock,
    ) -> None:
        registration.error = RuntimeError("registration unavailable")

    monkeypatch.setattr(shell, "_register_container_exec", AsyncMock(side_effect=_fail_registration))
    abort_unregistered = AsyncMock(side_effect=[RuntimeError("cleanup unavailable"), None])
    terminate_container_exec = AsyncMock()
    stop_local_docker_exec = AsyncMock()
    monkeypatch.setattr(shell, "_abort_unregistered_container_exec", abort_unregistered)
    monkeypatch.setattr(shell, "_terminate_container_exec", terminate_container_exec)
    monkeypatch.setattr(shell, "_stop_local_docker_exec", stop_local_docker_exec)

    with pytest.raises(RuntimeError, match="cleanup unavailable"):
        await shell.execute("sleep 10")

    assert process.communicate.await_count == 0
    assert len(shell._execution_handles) == 1

    await shell.reset_background_processes()

    assert abort_unregistered.await_count == 2
    terminate_container_exec.assert_not_awaited()
    stop_local_docker_exec.assert_awaited_once_with(process)
    assert shell._execution_handles == {}
    assert shell._background_cleanup_errors == {}


async def test_docker_shell_cleanup_retry_remembers_remote_termination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A local CLI stop failure must not make a retry depend on a deleted remote marker."""
    shell = DockerShell(container_id="test123")
    process, _create_process = _mock_docker_exec_process(monkeypatch)
    process.returncode = None
    monkeypatch.setattr(shell, "_wait_for_container_exec_registration", AsyncMock())
    terminate_container_exec = AsyncMock()
    stop_local_docker_exec = AsyncMock(side_effect=[RuntimeError("local stop failed"), None])
    monkeypatch.setattr(shell, "_terminate_container_exec", terminate_container_exec)
    monkeypatch.setattr(shell, "_stop_local_docker_exec", stop_local_docker_exec)
    communicate_started = asyncio.Event()
    never_complete = asyncio.Event()

    async def _communicate() -> tuple[bytes, bytes]:
        communicate_started.set()
        await never_complete.wait()
        return b"", b""

    process.communicate = AsyncMock(side_effect=_communicate)
    execute_task = asyncio.create_task(shell.execute("sleep 10"))
    await communicate_started.wait()
    execute_task.cancel()

    with pytest.raises(RuntimeError, match="local stop failed"):
        await execute_task
    assert len(shell._execution_handles) == 1

    await shell.reset_background_processes()

    terminate_container_exec.assert_awaited_once()
    assert stop_local_docker_exec.await_count == 2
    assert shell._execution_handles == {}


async def test_docker_shell_send_signal_targets_remote_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public signals must use the remote marker rather than the local docker CLI PID."""
    shell = DockerShell(container_id="test123")
    process, create_process = _mock_docker_exec_process(monkeypatch)
    process.returncode = None
    monkeypatch.setattr(shell, "_wait_for_container_exec_registration", AsyncMock())
    signal_container_exec = AsyncMock()
    monkeypatch.setattr(shell, "_signal_container_exec", signal_container_exec)

    handle = await shell._create_process("sleep 10")
    assert handle.send_signal is not None
    await handle.send_signal(getattr(signal, "SIGCONT", 18))

    pidfile = create_process.await_args.args[-3]
    signal_container_exec.assert_awaited_once_with(pidfile, ("1234", "5678"), "SIGCONT")
    process.terminate.assert_not_called()
    process.kill.assert_not_called()


async def test_docker_shell_send_signal_uses_linux_numbers_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows host signal aliases must not shadow remote Linux signal numbers."""
    shell = DockerShell(container_id="test123")
    process, create_process = _mock_docker_exec_process(monkeypatch)
    process.returncode = None
    monkeypatch.setattr(shell, "_wait_for_container_exec_registration", AsyncMock())
    signal_container_exec = AsyncMock()
    monkeypatch.setattr(shell, "_signal_container_exec", signal_container_exec)
    monkeypatch.setattr(sandbox_module.os, "name", "nt")

    handle = await shell._create_process("sleep 10")
    assert handle.send_signal is not None
    await handle.send_signal(1)

    pidfile = create_process.await_args.args[-3]
    signal_container_exec.assert_awaited_once_with(pidfile, ("1234", "5678"), "SIGHUP")


async def test_docker_shell_send_sigkill_uses_verified_termination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SIGKILL must verify remote termination before stopping the local CLI."""
    shell = DockerShell(container_id="test123")
    process, create_process = _mock_docker_exec_process(monkeypatch)
    process.returncode = None
    monkeypatch.setattr(shell, "_wait_for_container_exec_registration", AsyncMock())
    terminate_container_exec = AsyncMock()
    stop_local_docker_exec = AsyncMock()
    monkeypatch.setattr(shell, "_terminate_container_exec", terminate_container_exec)
    monkeypatch.setattr(shell, "_stop_local_docker_exec", stop_local_docker_exec)

    handle = await shell._create_process("sleep 10")
    assert handle.send_signal is not None
    await handle.send_signal(getattr(signal, "SIGKILL", 9))

    pidfile = create_process.await_args.args[-3]
    terminate_container_exec.assert_awaited_once_with(pidfile, ("1234", "5678"))
    stop_local_docker_exec.assert_awaited_once_with(process)


async def test_docker_shell_rejects_unsupported_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Signals without safe remote guardian semantics must fail explicitly."""
    shell = DockerShell(container_id="test123")
    process, _create_process = _mock_docker_exec_process(monkeypatch)
    process.returncode = None
    monkeypatch.setattr(shell, "_wait_for_container_exec_registration", AsyncMock())

    handle = await shell._create_process("sleep 10")
    assert handle.send_signal is not None
    with pytest.raises(ValueError, match="Unsupported Docker exec signal"):
        await handle.send_signal(getattr(signal, "SIGALRM", 14))


async def test_docker_shell_unregistered_abort_does_not_trust_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-registration cleanup closes the exact ACK channel and never reads a marker identity."""
    shell = DockerShell(container_id="test123")
    process = MagicMock()
    process.returncode = None
    process.stdin = MagicMock()
    process.stdin.wait_closed = AsyncMock()
    process.wait = AsyncMock(return_value=125)
    run_helper = AsyncMock()
    monkeypatch.setattr(shell, "_run_docker_helper", run_helper)

    await shell._abort_unregistered_container_exec(process)

    process.stdin.close.assert_called_once_with()
    process.stdin.wait_closed.assert_awaited_once_with()
    process.wait.assert_awaited_once_with()
    run_helper.assert_not_awaited()


@pytest.mark.skipif(not Path("/proc/self/status").exists(), reason="Linux /proc lifecycle test")
async def test_docker_exec_rejects_pre_ack_marker_identity_substitution(tmp_path: Path) -> None:
    """A sibling marker identity cannot replace the exact exec transport identity."""
    pidfile = tmp_path / "exec.pid"
    command_marker = tmp_path / "command-started.txt"
    handshake_nonce = "substitution-token"
    unrelated = await asyncio.create_subprocess_exec("sleep", "10", start_new_session=True)
    process = await asyncio.create_subprocess_exec(
        "/bin/sh",
        "-c",
        sandbox_module._DOCKER_EXEC_WRAPPER,
        "ya-agent-exec",
        str(pidfile),
        handshake_nonce,
        f"printf started > {shlex.quote(str(command_marker))}",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )

    try:
        assert process.stderr is not None
        registration_frame = await asyncio.wait_for(process.stderr.readline(), timeout=2.0)
        frame_parts = registration_frame.decode().split()
        assert frame_parts[:2] == ["ya-agent-exec", handshake_nonce]
        trusted_target, trusted_starttime = frame_parts[2:]

        unrelated_stat = Path(f"/proc/{unrelated.pid}/stat").read_text().split(") ", 1)[1].split()
        unrelated_starttime = unrelated_stat[19]
        pidfile.write_text(f"active {unrelated.pid} {unrelated_starttime} {handshake_nonce}\n")

        validation = await asyncio.create_subprocess_exec(
            "/bin/sh",
            "-c",
            sandbox_module._DOCKER_EXEC_VALIDATE,
            "ya-agent-validate",
            str(pidfile),
            handshake_nonce,
            trusted_target,
            trusted_starttime,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await validation.communicate()

        assert validation.returncode != 0
        assert unrelated.returncode is None
        assert process.stdin is not None
        process.stdin.close()
        await process.stdin.wait_closed()
        await asyncio.wait_for(process.wait(), timeout=2.0)
        assert not command_marker.exists()
    finally:
        if process.returncode is None and process.pid is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
        if unrelated.returncode is None and unrelated.pid is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(unrelated.pid, signal.SIGKILL)
            await unrelated.wait()


@pytest.mark.skipif(not Path("/proc/self/status").exists(), reason="Linux /proc lifecycle test")
async def test_docker_exec_guardian_outlives_command_leader_and_kills_descendant(tmp_path: Path) -> None:
    """The remote wrapper marker must survive a leader exit until the whole group is killed."""
    pidfile = tmp_path / "exec.pid"
    handshake_nonce = "guardian-token"
    leaked_marker = tmp_path / "leaked.txt"
    command = f"(trap '' HUP TERM; sleep 0.4; printf leaked > {shlex.quote(str(leaked_marker))}) & exit 0"
    process = await asyncio.create_subprocess_exec(
        "/bin/sh",
        "-c",
        sandbox_module._DOCKER_EXEC_WRAPPER,
        "ya-agent-exec",
        str(pidfile),
        handshake_nonce,
        command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )

    try:
        deadline = asyncio.get_running_loop().time() + 2.0
        while not pidfile.exists():
            if asyncio.get_running_loop().time() >= deadline:
                pytest.fail("Docker exec guardian did not publish its marker")
            await asyncio.sleep(0.01)

        marker_state, target, starttime, marker_token = pidfile.read_text().split()
        assert marker_state == "active"
        assert marker_token == handshake_nonce
        assert process.stderr is not None
        registration_frame = await process.stderr.readline()
        assert registration_frame == f"ya-agent-exec {handshake_nonce} {target} {starttime}\n".encode()
        assert process.stdin is not None
        process.stdin.write(f"ya-agent-ack {handshake_nonce}\n".encode())
        await process.stdin.drain()
        await asyncio.sleep(0.05)
        assert process.returncode is None

        cleanup = await asyncio.create_subprocess_exec(
            "/bin/sh",
            "-c",
            sandbox_module._DOCKER_EXEC_KILL_TREE,
            "ya-agent-kill",
            str(pidfile),
            target,
            starttime,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _stdout, cleanup_stderr = await cleanup.communicate()
        assert cleanup.returncode == 0, cleanup_stderr.decode("utf-8", errors="replace")
        await asyncio.wait_for(process.wait(), timeout=2.0)
        await asyncio.sleep(0.45)

        assert not leaked_marker.exists()
        assert not pidfile.exists()
    finally:
        if process.returncode is None and process.pid is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()


@pytest.mark.skipif(not Path("/proc/self/status").exists(), reason="Linux /proc lifecycle test")
@pytest.mark.parametrize("tamper", ["delete", "forge_done", "garbage"])
async def test_docker_exec_cleanup_uses_trusted_identity_after_marker_tampering(
    tmp_path: Path,
    tamper: str,
) -> None:
    """User-controlled marker changes must not bypass trusted group cleanup."""
    pidfile = tmp_path / "exec.pid"
    handshake_nonce = "tamper-token"
    quoted_pidfile = shlex.quote(str(pidfile))
    if tamper == "delete":
        tamper_command = f"rm -f {quoted_pidfile}"
    elif tamper == "forge_done":
        tamper_command = (
            f"read state target starttime marker_token < {quoted_pidfile}; "
            f'printf \'done %s %s %s\\n\' "$target" "$starttime" "$marker_token" > {quoted_pidfile}'
        )
    else:
        tamper_command = f"printf garbage > {quoted_pidfile}"
    command = f"{tamper_command}; trap '' HUP TERM; while :; do sleep 1; done"
    process = await asyncio.create_subprocess_exec(
        "/bin/sh",
        "-c",
        sandbox_module._DOCKER_EXEC_WRAPPER,
        "ya-agent-exec",
        str(pidfile),
        handshake_nonce,
        command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )

    try:
        deadline = asyncio.get_running_loop().time() + 2.0
        while not pidfile.exists():
            if asyncio.get_running_loop().time() >= deadline:
                pytest.fail("Docker exec guardian did not publish its marker")
            await asyncio.sleep(0.01)
        marker_state, target, starttime, marker_token = pidfile.read_text().split()
        assert marker_state == "active"
        assert marker_token == handshake_nonce
        assert process.stderr is not None
        registration_frame = await process.stderr.readline()
        assert registration_frame == f"ya-agent-exec {handshake_nonce} {target} {starttime}\n".encode()
        assert process.stdin is not None
        process.stdin.write(f"ya-agent-ack {handshake_nonce}\n".encode())
        await process.stdin.drain()
        await asyncio.sleep(0.05)
        assert process.returncode is None

        cleanup = await asyncio.create_subprocess_exec(
            "/bin/sh",
            "-c",
            sandbox_module._DOCKER_EXEC_KILL_TREE,
            "ya-agent-kill",
            str(pidfile),
            target,
            starttime,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _stdout, cleanup_stderr = await cleanup.communicate()
        assert cleanup.returncode == 0, cleanup_stderr.decode("utf-8", errors="replace")
        await asyncio.wait_for(process.wait(), timeout=2.0)
        assert not pidfile.exists()
    finally:
        if process.returncode is None and process.pid is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()


@pytest.mark.parametrize(
    ("cwd", "expected_workdir"),
    [("subdir", "/workspace/subdir"), ("/tmp", "/tmp")],  # noqa: S108
)
async def test_docker_shell_execute_with_cwd(
    monkeypatch: pytest.MonkeyPatch,
    cwd: str,
    expected_workdir: str,
) -> None:
    """Foreground execution should pass resolved workdir to docker exec."""
    shell = DockerShell(container_id="test123", container_workdir="/workspace")
    _process, create_process = _mock_docker_exec_process(monkeypatch)

    await shell.execute("ls", cwd=cwd)

    args = list(create_process.await_args.args)
    assert args[args.index("-w") + 1] == expected_workdir


async def test_docker_shell_execute_with_exec_user_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Foreground execution should preserve user and merged environment options."""
    shell = DockerShell(
        container_id="test123",
        exec_user="1000:1000",
        default_env={"HOME": "/home/claw", "USER": "claw", "FOO": "default"},
    )
    _process, create_process = _mock_docker_exec_process(monkeypatch)

    await shell.execute("env", env={"FOO": "override", "BAR": "baz"})

    args = list(create_process.await_args.args)
    assert args[args.index("--user") + 1] == "1000:1000"
    env_pairs = {args[index + 1] for index, value in enumerate(args) if value == "-e"}
    assert env_pairs == {
        "HOME=/home/claw",
        "USER=claw",
        "FOO=override",
        "BAR=baz",
    }


def test_docker_shell_builds_background_exec_args_with_user_and_default_env() -> None:
    shell = DockerShell(
        container_id="test123",
        container_workdir="/workspace",
        exec_user="1000:1000",
        default_env={"HOME": "/home/claw", "USER": "claw"},
    )

    args = shell._build_docker_exec_args("echo hello", env={"FOO": "bar"}, cwd="subdir")

    assert args == [
        "docker",
        "exec",
        "-i",
        "--user",
        "1000:1000",
        "-e",
        "HOME=/home/claw",
        "-e",
        "USER=claw",
        "-e",
        "FOO=bar",
        "-w",
        "/workspace/subdir",
        "test123",
        "/bin/sh",
        "-c",
        "echo hello",
    ]


async def test_docker_shell_get_context_instructions() -> None:
    """Should return docker-specific instructions."""
    shell = DockerShell(
        container_id="abc123",
        container_workdir="/workspace",
        default_timeout=30.0,
    )
    instructions = await shell.get_context_instructions()

    assert instructions is not None
    assert "docker-exec" in instructions
    assert "abc123" in instructions
    assert "/workspace" in instructions


# --- SandboxEnvironment Tests ---


def test_sandbox_environment_requires_mounts() -> None:
    """Should raise ValueError if mounts is empty."""
    with pytest.raises(ValueError, match="At least one mount is required"):
        SandboxEnvironment(mounts=[], image="python:3.11")


def test_sandbox_environment_requires_shell_or_docker() -> None:
    """Should raise ValueError if no shell backend can be determined."""
    with pytest.raises(ValueError, match="Either shell, container_id, or image must be provided"):
        SandboxEnvironment(mounts=[VirtualMount(Path("/tmp"), Path("/workspace"))])  # noqa: S108


def test_sandbox_environment_rejects_work_dir_outside_mounts(tmp_path: Path) -> None:
    """Should reject work_dir that is not under any mount's virtual_path."""
    with pytest.raises(ValueError, match=r"work_dir .* is not under any mount"):
        SandboxEnvironment(
            mounts=[VirtualMount(tmp_path, Path("/workspace"))],
            work_dir="/other",
            image="python:3.11",
        )


def test_sandbox_environment_rejects_work_dir_traversal(tmp_path: Path) -> None:
    """Should reject work_dir with path traversal that escapes mounts."""
    with pytest.raises(ValueError, match=r"work_dir .* is not under any mount"):
        SandboxEnvironment(
            mounts=[VirtualMount(tmp_path, Path("/workspace"))],
            work_dir="/workspace/../etc",
            image="python:3.11",
        )


def test_sandbox_environment_initialization_with_container_id(tmp_path: Path) -> None:
    """Should initialize with existing container_id."""
    env = SandboxEnvironment(
        mounts=[VirtualMount(tmp_path, Path("/app"))],
        container_id="existing123",
        cleanup_on_exit=False,
    )
    assert env._container_id == "existing123"
    assert env._work_dir == "/app"
    assert env._cleanup_on_exit is False
    assert env._docker_exec_user is None
    assert env._docker_exec_default_env == {}


def test_sandbox_environment_initialization_with_image(tmp_path: Path) -> None:
    """Should initialize with image for new container."""
    env = SandboxEnvironment(
        mounts=[VirtualMount(tmp_path, Path("/workspace"))],
        image="python:3.11",
        cleanup_on_exit=True,
    )
    assert env._image == "python:3.11"
    assert env._cleanup_on_exit is True


def test_sandbox_environment_initialization_with_custom_shell(tmp_path: Path) -> None:
    """Should initialize with custom shell backend."""
    mock_shell = MagicMock(spec=["execute", "get_context_instructions", "close"])
    env = SandboxEnvironment(
        mounts=[VirtualMount(tmp_path, Path("/workspace"))],
        shell=mock_shell,
    )
    assert env._custom_shell is mock_shell


def test_sandbox_environment_custom_work_dir(tmp_path: Path) -> None:
    """Should use custom work_dir when provided."""
    env = SandboxEnvironment(
        mounts=[
            VirtualMount(tmp_path / "a", Path("/workspace/a")),
            VirtualMount(tmp_path / "b", Path("/workspace/b")),
        ],
        work_dir="/workspace/b",
        image="python:3.11",
    )
    assert env._work_dir == "/workspace/b"


def test_sandbox_environment_default_work_dir(tmp_path: Path) -> None:
    """Should default work_dir to first mount's virtual_path."""
    env = SandboxEnvironment(
        mounts=[VirtualMount(tmp_path, Path("/myworkspace"))],
        image="python:3.11",
    )
    assert env._work_dir == "/myworkspace"


async def test_sandbox_environment_properties_before_enter(tmp_path: Path) -> None:
    """Should raise error when accessing properties before entering context."""
    from ya_agent_environment import EnvironmentNotEnteredError

    env = SandboxEnvironment(
        mounts=[VirtualMount(tmp_path, Path("/workspace"))],
        container_id="test123",
    )
    with pytest.raises(EnvironmentNotEnteredError):
        _ = env.file_operator
    with pytest.raises(EnvironmentNotEnteredError):
        _ = env.shell


async def test_sandbox_environment_enter_with_existing_container(tmp_path: Path) -> None:
    """Should verify container and create operators on enter."""
    env = SandboxEnvironment(
        mounts=[VirtualMount(tmp_path, Path("/workspace"))],
        container_id="existing123",
        cleanup_on_exit=False,
    )

    mock_container = MagicMock()
    mock_container.status = "running"

    mock_client = MagicMock()
    mock_client.containers.get.return_value = mock_container
    env._client = mock_client

    async with env:
        assert env.file_operator is not None
        assert env.shell is not None
        # File operator should use virtual path
        assert env._file_operator._default_path == normalize_virtual_path("/workspace")
        assert env._shell._container_id == "existing123"
        assert env._shell._container_workdir == "/workspace"

    # Verify container was not stopped (cleanup_on_exit=False)
    assert mock_container.stop.call_count == 0


async def test_sandbox_environment_enter_creates_new_container(tmp_path: Path) -> None:
    """Should create container when entering with image."""
    env = SandboxEnvironment(
        mounts=[VirtualMount(tmp_path, Path("/workspace"))],
        image="python:3.11",
        cleanup_on_exit=True,
    )

    mock_container = MagicMock()
    mock_container.id = "new123"

    mock_client = MagicMock()
    mock_client.containers.run.return_value = mock_container
    mock_client.containers.get.return_value = mock_container
    env._client = mock_client

    async with env:
        assert env._container_id == "new123"
        assert env._created_container is True

    # Verify container was stopped and removed (cleanup_on_exit=True)
    mock_container.stop.assert_called_once()
    mock_container.remove.assert_called_once()


async def test_sandbox_environment_removes_container_when_stop_fails(tmp_path: Path) -> None:
    """Cleanup should still remove a container that is already stopped or cannot be stopped."""
    env = SandboxEnvironment(
        mounts=[VirtualMount(tmp_path, Path("/workspace"))],
        image="python:3.11",
        cleanup_on_exit=True,
    )

    mock_container = MagicMock()
    mock_container.id = "new123"
    mock_container.stop.side_effect = docker.errors.APIError("container is already stopped")

    mock_client = MagicMock()
    mock_client.containers.run.return_value = mock_container
    mock_client.containers.get.return_value = mock_container
    env._client = mock_client

    async with env:
        assert env._container_id == "new123"

    mock_container.stop.assert_called_once_with(timeout=10)
    mock_container.remove.assert_called_once_with(force=True)


async def test_sandbox_environment_passes_docker_exec_options_to_shell(tmp_path: Path) -> None:
    """Should pass docker exec options to created DockerShell."""
    env = SandboxEnvironment(
        mounts=[VirtualMount(tmp_path, Path("/workspace"))],
        container_id="test123",
        docker_exec_user="1000:1000",
        docker_exec_default_env={"HOME": "/home/claw"},
    )

    mock_container = MagicMock()
    mock_container.status = "running"

    mock_client = MagicMock()
    mock_client.containers.get.return_value = mock_container
    env._client = mock_client

    async with env:
        assert env._shell._exec_user == "1000:1000"
        assert env._shell._default_env == {"HOME": "/home/claw"}


async def test_sandbox_environment_file_operator_uses_virtual_paths(tmp_path: Path) -> None:
    """Should configure file operator with virtual paths mapped to host_dir."""
    env = SandboxEnvironment(
        mounts=[VirtualMount(tmp_path, Path("/workspace"))],
        container_id="test123",
    )

    mock_container = MagicMock()
    mock_container.status = "running"

    mock_client = MagicMock()
    mock_client.containers.get.return_value = mock_container
    env._client = mock_client

    async with env:
        # Write a file using file operator (relative path)
        await env.file_operator.write_file("test.txt", "hello")
        # Actual file should be on host
        assert (tmp_path / "test.txt").read_text() == "hello"

        # Read back using absolute virtual path
        content = await env.file_operator.read_file("/workspace/test.txt")
        assert content == "hello"


async def test_sandbox_environment_multi_mount(tmp_path: Path) -> None:
    """Should support multiple mounts."""
    host_a = tmp_path / "project"
    host_b = tmp_path / "config"
    host_a.mkdir()
    host_b.mkdir()

    env = SandboxEnvironment(
        mounts=[
            VirtualMount(host_a, Path("/workspace/project")),
            VirtualMount(host_b, Path("/workspace/config")),
        ],
        work_dir="/workspace/project",
        container_id="test123",
    )

    mock_container = MagicMock()
    mock_container.status = "running"

    mock_client = MagicMock()
    mock_client.containers.get.return_value = mock_container
    env._client = mock_client

    async with env:
        # Write to project mount (default, relative path)
        await env.file_operator.write_file("main.py", "code")
        assert (host_a / "main.py").read_text() == "code"

        # Write to config mount (absolute path)
        await env.file_operator.write_file("/workspace/config/app.json", "{}")
        assert (host_b / "app.json").read_text() == "{}"


async def test_sandbox_environment_tmp_dir_enabled(tmp_path: Path) -> None:
    """Should create tmp directory when enabled."""
    env = SandboxEnvironment(
        mounts=[VirtualMount(tmp_path, Path("/workspace"))],
        container_id="test123",
        enable_tmp_dir=True,
        tmp_base_dir=tmp_path,
    )

    mock_container = MagicMock()
    mock_container.status = "running"

    mock_client = MagicMock()
    mock_client.containers.get.return_value = mock_container
    env._client = mock_client

    async with env:
        assert env.tmp_dir is not None
        assert env.tmp_dir.exists()
        tmp_dir = env.tmp_dir

    # Tmp dir should be cleaned up after exit
    assert not tmp_dir.exists()


async def test_sandbox_environment_tmp_dir_disabled(tmp_path: Path) -> None:
    """Should not create tmp directory when disabled."""
    env = SandboxEnvironment(
        mounts=[VirtualMount(tmp_path, Path("/workspace"))],
        container_id="test123",
        enable_tmp_dir=False,
    )

    mock_container = MagicMock()
    mock_container.status = "running"

    mock_client = MagicMock()
    mock_client.containers.get.return_value = mock_container
    env._client = mock_client

    async with env:
        assert env.tmp_dir is None


async def test_sandbox_environment_get_context_instructions(tmp_path: Path) -> None:
    """Should return instructions with virtual paths, no mount-mapping."""
    env = SandboxEnvironment(
        mounts=[VirtualMount(tmp_path, Path("/workspace"))],
        container_id="test123",
    )

    mock_container = MagicMock()
    mock_container.status = "running"

    mock_client = MagicMock()
    mock_client.containers.get.return_value = mock_container
    env._client = mock_client

    async with env:
        instructions = await env.get_context_instructions()

        assert instructions is not None
        assert "/workspace" in instructions
        # Should NOT have mount-mapping (paths are symmetric now)
        assert "mount-mapping" not in instructions
        # Should NOT expose host path
        assert str(tmp_path) not in instructions


async def test_sandbox_environment_cross_session_sharing(tmp_path: Path) -> None:
    """Should support cross-session container sharing with cleanup_on_exit=False."""
    mount = VirtualMount(tmp_path, Path("/workspace"))

    # First session
    env1 = SandboxEnvironment(
        mounts=[mount],
        container_id="shared123",
        cleanup_on_exit=False,
    )

    mock_container = MagicMock()
    mock_container.status = "running"

    mock_client = MagicMock()
    mock_client.containers.get.return_value = mock_container
    env1._client = mock_client

    async with env1:
        await env1.file_operator.write_file("session1.txt", "from session 1")

    assert mock_container.stop.call_count == 0

    # Second session
    env2 = SandboxEnvironment(
        mounts=[mount],
        container_id="shared123",
        cleanup_on_exit=False,
    )
    env2._client = mock_client

    async with env2:
        content = await env2.file_operator.read_file("session1.txt")
        assert content == "from session 1"


async def test_sandbox_environment_with_custom_shell(tmp_path: Path) -> None:
    """Should use custom shell backend when provided."""
    mock_shell = MagicMock()
    mock_shell.execute = MagicMock(return_value=(0, "output", ""))
    mock_shell.get_context_instructions = MagicMock(return_value="custom shell")
    mock_shell.close = AsyncMock()

    env = SandboxEnvironment(
        mounts=[VirtualMount(tmp_path, Path("/workspace"))],
        shell=mock_shell,
    )

    async with env:
        assert env.shell is mock_shell
        await env.file_operator.write_file("test.txt", "custom shell test")
        assert (tmp_path / "test.txt").read_text() == "custom shell test"
        assert env._created_container is False


async def test_sandbox_environment_create_container_mounts_tmp_dir(tmp_path: Path) -> None:
    """Should mount tmp_dir into Docker container when creating a new one."""
    env = SandboxEnvironment(
        mounts=[VirtualMount(tmp_path, Path("/workspace"))],
        image="python:3.11",
        enable_tmp_dir=True,
        tmp_base_dir=tmp_path,
    )

    mock_container = MagicMock()
    mock_container.id = "new123"

    mock_client = MagicMock()
    mock_client.containers.run.return_value = mock_container
    mock_client.containers.get.return_value = mock_container
    env._client = mock_client

    async with env:
        # Verify containers.run was called with tmp_dir in volumes
        call_kwargs = mock_client.containers.run.call_args[1]
        volumes = call_kwargs["volumes"]
        # Should have 2 volumes: the mount + tmp_dir
        assert len(volumes) == 2
        # tmp_dir should be mounted at the same path inside container
        assert env.tmp_dir is not None
        assert str(env.tmp_dir) in volumes


async def test_sandbox_environment_auto_start_stopped_container(tmp_path: Path) -> None:
    """Should auto-start a stopped container instead of raising error."""
    env = SandboxEnvironment(
        mounts=[VirtualMount(tmp_path, Path("/workspace"))],
        container_id="stopped123",
        cleanup_on_exit=False,
    )

    mock_container = MagicMock()
    # Simulate: first reload returns "exited", after start() returns "running"
    mock_container.status = "exited"

    def _reload_side_effect() -> None:
        # After start() is called, status changes to running
        if mock_container.start.called:
            mock_container.status = "running"

    mock_container.reload.side_effect = _reload_side_effect

    mock_client = MagicMock()
    mock_client.containers.get.return_value = mock_container
    env._client = mock_client

    async with env:
        # Should have auto-started the container
        mock_container.start.assert_called_once()
        assert env.file_operator is not None
        assert env.shell is not None


async def test_sandbox_environment_unrecoverable_container_state(tmp_path: Path) -> None:
    """Should raise error for containers in unrecoverable state."""
    env = SandboxEnvironment(
        mounts=[VirtualMount(tmp_path, Path("/workspace"))],
        container_id="dead123",
        cleanup_on_exit=False,
    )

    mock_container = MagicMock()
    mock_container.status = "removing"

    mock_client = MagicMock()
    mock_client.containers.get.return_value = mock_container
    env._client = mock_client

    with pytest.raises(RuntimeError, match="unrecoverable state"):
        async with env:
            pass


async def test_sandbox_environment_lazy_shell_does_not_verify_container_on_enter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """lazy_shell should defer Docker verification until shell use."""
    env = SandboxEnvironment(
        mounts=[VirtualMount(tmp_path, Path("/workspace"))],
        container_id="existing123",
        cleanup_on_exit=False,
        lazy_shell=True,
    )

    mock_container = MagicMock()
    mock_container.status = "running"
    mock_container.exec_run.return_value = MagicMock(exit_code=0, output=(b"ok", b""))

    mock_client = MagicMock()
    mock_client.containers.get.return_value = mock_container
    env._client = mock_client
    _mock_docker_exec_process(monkeypatch, stdout=b"ok")

    async with env:
        assert isinstance(env.shell, DeferredDockerShell)
        assert env.container_id == "existing123"
        assert mock_client.containers.get.call_count == 0

        instructions = await env.get_context_instructions()
        assert "status>not_started" in instructions
        assert mock_client.containers.get.call_count == 0

        code, stdout, stderr = await env.shell.execute("echo ok")
        assert (code, stdout, stderr) == (0, "ok", "")
        assert mock_client.containers.get.call_count >= 1


async def test_sandbox_environment_lazy_shell_creates_container_on_first_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """lazy_shell should create image-backed containers on first command."""
    env = SandboxEnvironment(
        mounts=[VirtualMount(tmp_path, Path("/workspace"))],
        image="python:3.11",
        cleanup_on_exit=True,
        lazy_shell=True,
    )

    mock_container = MagicMock()
    mock_container.id = "new123"
    mock_container.status = "running"
    mock_container.exec_run.return_value = MagicMock(exit_code=0, output=(b"ok", b""))

    mock_client = MagicMock()
    mock_client.containers.run.return_value = mock_container
    mock_client.containers.get.return_value = mock_container
    env._client = mock_client
    _mock_docker_exec_process(monkeypatch, stdout=b"ok")

    async with env:
        assert env.container_id is None
        assert mock_client.containers.run.call_count == 0
        code, stdout, stderr = await env.shell.execute("echo ok")
        assert (code, stdout, stderr) == (0, "ok", "")
        assert env.container_id == "new123"
        mock_client.containers.run.assert_called_once()

    mock_container.stop.assert_called_once()
    mock_container.remove.assert_called_once()
