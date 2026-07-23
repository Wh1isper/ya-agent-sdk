"""Sandbox environment implementation.

This module provides a sandboxed environment that:
- Uses VirtualLocalFileOperator for path-mapped file operations
- Uses a sandboxed shell (Docker by default, pluggable)
- Presents a symmetric path space to the agent

Architecture:
    - File operations: Local filesystem at host_dir, presented as work_dir
    - Shell execution: Sandboxed shell (e.g., Docker) at work_dir
    - Both file ops and shell see the same path space (e.g., /workspace)
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePath
from uuid import uuid4

import docker
import docker.errors
from ya_agent_environment import (
    DeferredShell,
    Environment,
    ExecutionHandle,
    ResourceFactory,
    ResourceRegistryState,
    Shell,
    ShellExecutionError,
    StdinAdapter,
)

from ya_agent_sdk.environment.local import VirtualLocalFileOperator, VirtualMount
from ya_agent_sdk.environment.virtual_path import VirtualPath, normalize_virtual_path

_DOCKER_CLI_EXECUTABLE = "docker"
_DOCKER_CLI_MISSING_MESSAGE = (
    "Docker CLI executable 'docker' was not found in the shell host runtime. "
    "Install the Docker CLI and ensure it is on PATH; access to the Docker socket alone is not sufficient."
)
_DOCKER_EXEC_MARKER_MISSING = 10
_DOCKER_EXEC_MARKER_ACTIVE = 11
_DOCKER_EXEC_KILL_REGISTRATION_GRACE = 1.0


@dataclass
class _DockerExecRegistration:
    """Mutable registration state shared with an immediately returned handle."""

    identity: tuple[str, str] | None = None
    error: Exception | None = None
    aborted: bool = False


async def _await_docker_exec_registration(
    task: asyncio.Task[None],
    registration: _DockerExecRegistration,
) -> tuple[str, str]:
    """Wait for one exact-channel handshake without transferring task cancellation."""
    await asyncio.shield(task)
    if registration.error is not None:
        raise registration.error
    if registration.identity is None:
        raise RuntimeError("Container exec completed without a trusted process identity")
    return registration.identity


class _DockerExecReadableStream:
    """Hide the registration frame before exposing one Docker exec stream."""

    def __init__(
        self,
        stream: asyncio.StreamReader,
        registration_task: asyncio.Task[None],
        registration: _DockerExecRegistration,
    ) -> None:
        self._stream = stream
        self._registration_task = registration_task
        self._registration = registration

    async def readline(self) -> bytes:
        await _await_docker_exec_registration(self._registration_task, self._registration)
        return await self._stream.readline()


class _DockerExecStdin:
    """Prevent user input from racing the guardian registration ACK."""

    def __init__(
        self,
        stream: StdinAdapter,
        registration_task: asyncio.Task[None],
        registration: _DockerExecRegistration,
    ) -> None:
        self._stream = stream
        self._registration_task = registration_task
        self._registration = registration

    async def write(self, data: bytes) -> None:
        await _await_docker_exec_registration(self._registration_task, self._registration)
        await self._stream.write(data)

    async def close(self) -> None:
        await _await_docker_exec_registration(self._registration_task, self._registration)
        await self._stream.close()


_DOCKER_EXEC_WRAPPER = """
pidfile=$1
token=$2
command=$3
stat_line=$(cat "/proc/$$/stat" 2>/dev/null) || exit 125
stat_fields=${stat_line##*) }
set -- $stat_fields
process_group=$3
session_id=$4
shift 19
starttime=$1
if [ "$process_group" != "$$" ] || [ "$session_id" != "$$" ]; then
    exit 125
fi
printf 'active %s %s %s\\n' "$$" "$starttime" "$token" > "$pidfile" || exit 125
# Identity is published over this exact docker exec's stderr pipe. User code
# remains blocked until the host validates it and replies over the same exec's
# stdin pipe, so same-user siblings cannot substitute marker identity or forge
# the release acknowledgement.
printf 'ya-agent-exec %s %s %s\\n' "$token" "$$" "$starttime" >&2 || exit 125
IFS= read -r acknowledgement || exit 125
if [ "$acknowledgement" != "ya-agent-ack $token" ]; then
    exit 125
fi
# The wrapper is the stable remote group/session leader. Catch terminating
# signals supported by DockerShell so group signaling reaches the command tree
# without allowing the ownership marker to disappear first.
trap ':' HUP INT TERM QUIT USR1 USR2
group_has_other_live_members() {
    for status_file in /proc/[0-9]*/status; do
        member_pid=
        state=
        member_group=
        member_session=
        while read -r key value rest; do
            if [ "$key" = "Pid:" ]; then
                member_pid=$value
            elif [ "$key" = "State:" ]; then
                state=$value
            elif [ "$key" = "NSpgid:" ]; then
                for part in $value $rest; do member_group=$part; done
            elif [ "$key" = "NSsid:" ]; then
                for part in $value $rest; do member_session=$part; done
            fi
        done < "$status_file" 2>/dev/null || continue
        if [ "$member_pid" != "$$" ] && [ "$member_group" = "$$" ] && \
           [ "$member_session" = "$$" ] && [ "$state" != "Z" ]; then
            return 0
        fi
    done
    return 1
}
/bin/sh -c "$command"
command_status=$?
empty_scans=0
while [ "$empty_scans" -lt 2 ]; do
    if group_has_other_live_members; then
        empty_scans=0
    else
        empty_scans=$((empty_scans + 1))
    fi
    if [ "$empty_scans" -lt 2 ]; then sleep 0.02; fi
done
printf 'done %s %s %s\\n' "$$" "$starttime" "$token" > "$pidfile" || exit 125
exit "$command_status"
""".strip()

_DOCKER_EXEC_VALIDATE = """
pidfile=$1
expected_token=$2
expected_target=$3
expected_starttime=$4
if ! read -r marker_state target marker_starttime marker_token < "$pidfile" 2>/dev/null; then
    exit 10
fi
if [ "$marker_state" != "active" ] || [ "$target" != "$expected_target" ] || \
   [ "$marker_starttime" != "$expected_starttime" ] || [ "$marker_token" != "$expected_token" ]; then
    exit 4
fi
stat_line=$(cat "/proc/$target/stat" 2>/dev/null) || exit 4
stat_fields=${stat_line##*) }
set -- $stat_fields
process_group=$3
session_id=$4
shift 19
actual_starttime=$1
if [ "$process_group" != "$target" ] || [ "$session_id" != "$target" ] || \
   [ "$actual_starttime" != "$expected_starttime" ]; then
    exit 4
fi
""".strip()

_DOCKER_EXEC_CONFIRM_COMPLETE = """
pidfile=$1
target=$2
expected_starttime=$3
case "$target" in ''|*[!0-9]*) exit 2 ;; esac
case "$expected_starttime" in ''|*[!0-9]*) exit 2 ;; esac
group_has_live_members() {
    for status_file in /proc/[0-9]*/status; do
        state=
        member_group=
        member_session=
        while read -r key value rest; do
            if [ "$key" = "State:" ]; then
                state=$value
            elif [ "$key" = "NSpgid:" ]; then
                for part in $value $rest; do member_group=$part; done
            elif [ "$key" = "NSsid:" ]; then
                for part in $value $rest; do member_session=$part; done
            fi
        done < "$status_file" 2>/dev/null || continue
        if [ "$member_group" = "$target" ] && [ "$member_session" = "$target" ] && \
           [ "$state" != "Z" ]; then
            return 0
        fi
    done
    return 1
}
if [ -e "/proc/$target/stat" ]; then
    stat_line=$(cat "/proc/$target/stat" 2>/dev/null) || exit 4
    stat_fields=${stat_line##*) }
    set -- $stat_fields
    process_group=$3
    session_id=$4
    shift 19
    actual_starttime=$1
    if [ "$process_group" != "$target" ] || [ "$session_id" != "$target" ] || \
       [ "$actual_starttime" != "$expected_starttime" ]; then
        exit 4
    fi
fi
if group_has_live_members; then
    exit 11
fi
rm -f "$pidfile"
""".strip()

_DOCKER_EXEC_KILL_TREE = """
pidfile=$1
target=$2
expected_starttime=$3
case "$target" in ''|*[!0-9]*) exit 2 ;; esac
case "$expected_starttime" in ''|*[!0-9]*) exit 2 ;; esac
group_has_live_members() {
    for status_file in /proc/[0-9]*/status; do
        state=
        member_group=
        member_session=
        while read -r key value rest; do
            if [ "$key" = "State:" ]; then
                state=$value
            elif [ "$key" = "NSpgid:" ]; then
                for part in $value $rest; do member_group=$part; done
            elif [ "$key" = "NSsid:" ]; then
                for part in $value $rest; do member_session=$part; done
            fi
        done < "$status_file" 2>/dev/null || continue
        if [ "$member_group" = "$target" ] && [ "$member_session" = "$target" ] && \
           [ "$state" != "Z" ]; then
            return 0
        fi
    done
    return 1
}
if [ -e "/proc/$target/stat" ]; then
    stat_line=$(cat "/proc/$target/stat" 2>/dev/null) || exit 4
    stat_fields=${stat_line##*) }
    set -- $stat_fields
    process_group=$3
    session_id=$4
    shift 19
    actual_starttime=$1
    if [ "$process_group" != "$target" ] || [ "$session_id" != "$target" ] || \
       [ "$actual_starttime" != "$expected_starttime" ]; then
        exit 4
    fi
fi
if ! group_has_live_members; then
    rm -f "$pidfile"
    exit 0
fi
if ! kill -STOP "-$target" 2>/dev/null && group_has_live_members; then
    exit 5
fi
if ! kill -KILL "-$target" 2>/dev/null && group_has_live_members; then
    exit 5
fi
attempt=0
while group_has_live_members; do
    attempt=$((attempt + 1))
    if [ "$attempt" -ge 100 ]; then
        exit 3
    fi
    sleep 0.05
done
rm -f "$pidfile"
""".strip()

_DOCKER_EXEC_SIGNAL_TREE = """
pidfile=$1
target=$2
expected_starttime=$3
signal_name=$4
case "$target" in ''|*[!0-9]*) exit 2 ;; esac
case "$expected_starttime" in ''|*[!0-9]*) exit 2 ;; esac
stat_line=$(cat "/proc/$target/stat" 2>/dev/null) || exit 4
stat_fields=${stat_line##*) }
set -- $stat_fields
process_group=$3
session_id=$4
shift 19
actual_starttime=$1
if [ "$process_group" != "$target" ] || [ "$session_id" != "$target" ] || \
   [ "$actual_starttime" != "$expected_starttime" ]; then
    exit 4
fi
kill -"$signal_name" "-$target"
""".strip()

_DOCKER_SUPPORTED_SIGNALS = frozenset({
    "SIGHUP",
    "SIGINT",
    "SIGQUIT",
    "SIGKILL",
    "SIGTERM",
    "SIGSTOP",
    "SIGCONT",
    "SIGUSR1",
    "SIGUSR2",
})
_DOCKER_LINUX_SIGNAL_NAMES_BY_NUMBER = {
    1: "SIGHUP",
    2: "SIGINT",
    3: "SIGQUIT",
    9: "SIGKILL",
    10: "SIGUSR1",
    12: "SIGUSR2",
    15: "SIGTERM",
    18: "SIGCONT",
    19: "SIGSTOP",
}


class DockerShell(Shell):
    """Shell implementation that executes commands inside a Docker container.

    Uses docker exec to run commands in the specified container.
    The working directory inside the container is specified by container_workdir.
    """

    def __init__(
        self,
        container_id: str,
        container_workdir: str = "/workspace",
        default_timeout: float = 30.0,
        exec_user: str | None = None,
        default_env: dict[str, str] | None = None,
    ):
        """Initialize DockerShell.

        Args:
            container_id: Docker container ID to execute commands in.
            container_workdir: Working directory inside the container.
            default_timeout: Default timeout in seconds.
            exec_user: Docker exec user, such as "1000:1000" or "root".
            default_env: Default environment variables for every docker exec.
        """
        # DockerShell doesn't use allowed_paths or default_cwd from base Shell
        # since path validation happens inside the container
        super().__init__(
            default_cwd=Path(container_workdir),
            allowed_paths=None,
            default_timeout=default_timeout,
        )
        self._container_id = container_id
        self._container_workdir = container_workdir
        self._exec_user = exec_user.strip() if isinstance(exec_user, str) and exec_user.strip() != "" else None
        self._default_env = dict(default_env or {})
        self._client: docker.DockerClient | None = None

    @property
    def client(self) -> docker.DockerClient:
        """Get Docker client with lazy initialization."""
        if self._client is None:
            self._client = docker.from_env()
        return self._client

    def _resolve_execute_timeout(self, timeout: float | None) -> float | None:
        """Use DockerShell's configured default; non-positive values disable timeout."""
        effective = self._default_timeout if timeout is None else timeout
        return effective if effective > 0 else None

    async def get_context_instructions(self) -> str | None:
        """Return instructions for the agent about shell capabilities."""
        exec_user_line = f"\n  <exec-user>{self._exec_user}</exec-user>" if self._exec_user is not None else ""
        return f"""<shell-execution>
  <type>docker-exec</type>
  <container-id>{self._container_id}</container-id>
  <container-workdir>{self._container_workdir}</container-workdir>{exec_user_line}
  <default-timeout>{self._default_timeout}s</default-timeout>
  <note>Commands are executed inside the Docker container via docker exec.</note>
</shell-execution>"""

    def _build_docker_exec_args(
        self,
        command: str,
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> list[str]:
        """Build the docker exec command-line arguments."""
        if cwd is not None:
            workdir = cwd if cwd.startswith("/") else f"{self._container_workdir}/{cwd}"
        else:
            workdir = self._container_workdir

        args: list[str] = [_DOCKER_CLI_EXECUTABLE, "exec", "-i"]
        if self._exec_user is not None:
            args.extend(["--user", self._exec_user])
        exec_env = self._build_exec_env(env)
        if exec_env:
            for k, v in exec_env.items():
                args.extend(["-e", f"{k}={v}"])
        args.extend(["-w", workdir, self._container_id, "/bin/sh", "-c", command])
        return args

    def _build_exec_env(self, env: dict[str, str] | None) -> dict[str, str] | None:
        merged_env = {**self._default_env, **dict(env or {})}
        return merged_env or None

    def _build_docker_helper_args(
        self,
        script: str,
        label: str,
        *helper_args: str,
    ) -> list[str]:
        """Build a separate docker exec invocation for lifecycle helpers."""
        args = [_DOCKER_CLI_EXECUTABLE, "exec"]
        if self._exec_user is not None:
            args.extend(["--user", self._exec_user])
        args.extend([
            self._container_id,
            "/bin/sh",
            "-c",
            script,
            label,
            *helper_args,
        ])
        return args

    def _build_docker_cleanup_args(
        self,
        pidfile: str,
        identity: tuple[str, str],
    ) -> list[str]:
        """Build a second docker exec that kills and verifies the owned process tree."""
        target, starttime = identity
        return self._build_docker_helper_args(
            _DOCKER_EXEC_KILL_TREE,
            "ya-agent-kill",
            pidfile,
            target,
            starttime,
        )

    @staticmethod
    async def _run_docker_helper(
        args: list[str],
        *,
        action: str,
    ) -> subprocess.CompletedProcess[bytes]:
        """Run one lifecycle helper without blocking the event loop."""

        def _run() -> subprocess.CompletedProcess[bytes]:
            return subprocess.run(  # noqa: S603
                args,
                capture_output=True,
                check=False,
                timeout=10.0,
            )

        try:
            return await asyncio.to_thread(_run)
        except FileNotFoundError as exc:
            raise RuntimeError(f"Failed to {action}: {_DOCKER_CLI_MISSING_MESSAGE}") from exc
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(f"Failed to {action}: {exc}") from exc

    @staticmethod
    def _helper_error_detail(result: subprocess.CompletedProcess[bytes]) -> str:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        return f": {stderr}" if stderr else ""

    @staticmethod
    def _parse_container_exec_identity(frame: bytes, expected_token: str) -> tuple[str, str]:
        try:
            parts = frame.decode("ascii", errors="strict").split()
        except UnicodeDecodeError as exc:
            raise RuntimeError("Container exec registration returned an invalid process identity") from exc
        if (
            len(parts) != 4
            or parts[0] != "ya-agent-exec"
            or parts[1] != expected_token
            or not all(part.isdigit() for part in parts[2:])
        ):
            raise RuntimeError("Container exec registration returned an invalid process identity")
        return parts[2], parts[3]

    async def _wait_for_container_exec_registration(
        self,
        process: asyncio.subprocess.Process,
        token: str,
    ) -> tuple[str, str]:
        """Read identity from this exact Docker exec's private stderr channel."""
        if process.stderr is None:
            raise RuntimeError("Container exec has no registration channel")
        try:
            frame = await asyncio.wait_for(process.stderr.readline(), timeout=10.0)
        except TimeoutError as exc:
            raise RuntimeError("Container exec did not publish its process-group identity") from exc
        if not frame:
            raise RuntimeError("Container exec exited before publishing its process-group identity")
        return self._parse_container_exec_identity(frame, token)

    async def _ack_container_exec_registration(
        self,
        process: asyncio.subprocess.Process,
        pidfile: str,
        token: str,
        identity: tuple[str, str],
    ) -> None:
        """Validate identity, then release user code over this exact exec's stdin."""
        target, starttime = identity
        result = await self._run_docker_helper(
            self._build_docker_helper_args(
                _DOCKER_EXEC_VALIDATE,
                "ya-agent-validate",
                pidfile,
                token,
                target,
                starttime,
            ),
            action="validate container exec registration",
        )
        if result.returncode != 0:
            detail = self._helper_error_detail(result)
            raise RuntimeError(f"Container exec registration acknowledgement failed{detail}")
        if process.stdin is None:
            raise RuntimeError("Container exec has no acknowledgement channel")
        process.stdin.write(f"ya-agent-ack {token}\n".encode("ascii"))
        await process.stdin.drain()

    @staticmethod
    async def _abort_unregistered_container_exec(process: asyncio.subprocess.Process) -> None:
        """Close the exact ACK channel so an unreleased guardian exits safely."""
        if process.returncode is not None:
            return
        if process.stdin is None:
            raise RuntimeError("Cannot abort an unregistered container exec without its stdin channel")
        await StdinAdapter(process.stdin).close()
        try:
            await asyncio.wait_for(process.wait(), timeout=5.0)
        except TimeoutError as exc:
            raise RuntimeError("Unregistered container exec did not exit after acknowledgement EOF") from exc

    @staticmethod
    async def _stop_local_docker_exec(process: asyncio.subprocess.Process) -> None:
        """Stop and reap the local docker CLI after remote termination is confirmed."""
        if process.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5.0)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            await process.wait()

    async def _register_container_exec(
        self,
        process: asyncio.subprocess.Process,
        pidfile: str,
        token: str,
        registration: _DockerExecRegistration,
        handshake_lock: asyncio.Lock,
    ) -> None:
        """Persist exact-channel identity before releasing the remote user command."""
        try:
            identity = await self._wait_for_container_exec_registration(process, token)
            registration.identity = identity
            async with handshake_lock:
                if registration.aborted:
                    raise RuntimeError("Container exec registration was aborted before acknowledgement")
                await self._ack_container_exec_registration(process, pidfile, token, identity)
        except Exception as exc:
            registration.error = exc

    async def _confirm_container_exec_completion(
        self,
        pidfile: str,
        identity: tuple[str, str],
    ) -> None:
        """Require the trusted remote group identity to have no live members."""
        target, starttime = identity
        args = self._build_docker_helper_args(
            _DOCKER_EXEC_CONFIRM_COMPLETE,
            "ya-agent-confirm",
            pidfile,
            target,
            starttime,
        )
        result = await self._run_docker_helper(
            args,
            action="confirm container exec completion",
        )
        if result.returncode != 0:
            detail = self._helper_error_detail(result)
            if result.returncode == _DOCKER_EXEC_MARKER_ACTIVE:
                raise RuntimeError(f"Container exec completed locally while its remote group remained active{detail}")
            raise RuntimeError(f"Container exec completion was not confirmed{detail}")

    async def _terminate_container_exec(
        self,
        pidfile: str,
        identity: tuple[str, str],
    ) -> None:
        """Kill an exact-channel-identified process group and verify it stopped."""
        result = await self._run_docker_helper(
            self._build_docker_cleanup_args(pidfile, identity),
            action="terminate container exec",
        )
        if result.returncode != 0:
            detail = self._helper_error_detail(result)
            raise RuntimeError(f"Container exec termination was not confirmed{detail}")

    async def _signal_container_exec(
        self,
        pidfile: str,
        identity: tuple[str, str],
        signal_name: str,
    ) -> None:
        """Send one validated symbolic signal to the owned remote process group."""
        target, starttime = identity
        args = self._build_docker_helper_args(
            _DOCKER_EXEC_SIGNAL_TREE,
            "ya-agent-signal",
            pidfile,
            target,
            starttime,
            signal_name.removeprefix("SIG"),
        )
        result = await self._run_docker_helper(
            args,
            action=f"signal container exec with {signal_name}",
        )
        if result.returncode != 0:
            detail = self._helper_error_detail(result)
            raise RuntimeError(f"Container exec signal {signal_name} was not confirmed{detail}")

    async def _create_process(  # noqa: C901
        self,
        command: str,
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> ExecutionHandle:
        """Create a background process in the Docker container.

        Uses `docker exec -i` subprocess to get native stdin/stdout/stderr
        pipes, enabling interactive input and real-time streaming.
        """
        if not command:
            raise ShellExecutionError("", stderr="Empty command")

        args = self._build_docker_exec_args(command, env=env, cwd=cwd)
        pidfile = f"/tmp/.ya-agent-exec-{uuid4().hex}.pid"  # noqa: S108
        token = uuid4().hex
        # Keep a stable in-container wrapper PID while the command runs. The
        # kill hook can terminate and verify the remote tree independently of
        # the local docker CLI process.
        args[-3:] = [
            "/bin/sh",
            "-c",
            _DOCKER_EXEC_WRAPPER,
            "ya-agent-exec",
            pidfile,
            token,
            command,
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise ShellExecutionError(command, stderr=_DOCKER_CLI_MISSING_MESSAGE) from exc
        except Exception as exc:
            raise ShellExecutionError(command, stderr=str(exc)) from exc

        stdout = process.stdout
        if stdout is None:
            stdout = asyncio.StreamReader()
            stdout.feed_eof()
        stderr = process.stderr
        if stderr is None:
            stderr = asyncio.StreamReader()
            stderr.feed_eof()

        # Return the handle immediately after the local CLI exists. Registration
        # runs under that handle so timeout/cancellation cannot spend the whole
        # probe deadline outside Shell ownership.
        registration = _DockerExecRegistration()
        handshake_lock = asyncio.Lock()
        registration_task = asyncio.create_task(
            self._register_container_exec(process, pidfile, token, registration, handshake_lock),
            name=f"docker-exec-register-{process.pid or id(process)}",
        )
        remote_termination_confirmed = False

        async def _await_registration() -> tuple[str, str]:
            return await _await_docker_exec_registration(registration_task, registration)

        async def _wait() -> int:
            identity = await _await_registration()
            await process.wait()
            await self._confirm_container_exec_completion(
                pidfile,
                identity,
            )
            return process.returncode or 0

        async def _kill() -> None:
            nonlocal remote_termination_confirmed
            if not remote_termination_confirmed:
                # Give the exact-channel handshake a short chance to publish
                # identity. If it has not, atomically forbid ACK and close the
                # same exec's stdin so the still-blocked guardian must exit.
                if not registration_task.done() and registration.identity is None:
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(
                            asyncio.shield(registration_task),
                            timeout=_DOCKER_EXEC_KILL_REGISTRATION_GRACE,
                        )
                abort_unregistered = False
                async with handshake_lock:
                    if registration.identity is None:
                        registration.aborted = True
                        abort_unregistered = True
                if abort_unregistered:
                    await self._abort_unregistered_container_exec(process)
                    registration_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await asyncio.shield(registration_task)
                else:
                    # Identity is from this exact exec. Let an in-flight ACK
                    # settle, then cleanup remains safe even if validation failed.
                    with contextlib.suppress(asyncio.CancelledError):
                        await asyncio.shield(registration_task)
                    identity = registration.identity
                    if identity is None:
                        raise RuntimeError("Container exec lost its trusted identity during cleanup")
                    await self._terminate_container_exec(pidfile, identity)
                remote_termination_confirmed = True
            await self._stop_local_docker_exec(process)

        async def _send_signal(sig: int) -> None:
            if os.name == "nt":
                signal_name = _DOCKER_LINUX_SIGNAL_NAMES_BY_NUMBER.get(sig)
                if signal_name is None:
                    raise ValueError(f"Unsupported Docker exec signal number: {sig}")
            else:
                try:
                    signal_name = signal.Signals(sig).name
                except ValueError as exc:
                    raise ValueError(f"Unsupported Docker exec signal number: {sig}") from exc
            if signal_name not in _DOCKER_SUPPORTED_SIGNALS:
                raise ValueError(f"Unsupported Docker exec signal: {signal_name}")
            if signal_name == "SIGKILL":
                await _kill()
                return
            identity = await _await_registration()
            await self._signal_container_exec(pidfile, identity, signal_name)

        async def _communicate() -> tuple[bytes, bytes]:
            await _await_registration()
            return await process.communicate()

        raw_stdin = StdinAdapter(process.stdin) if process.stdin is not None else None
        stdin = _DockerExecStdin(raw_stdin, registration_task, registration) if raw_stdin is not None else None

        return ExecutionHandle(
            stdout=_DockerExecReadableStream(stdout, registration_task, registration),
            stderr=_DockerExecReadableStream(stderr, registration_task, registration),
            wait=_wait,
            kill=_kill,
            stdin=stdin,
            pid=process.pid,
            send_signal=_send_signal,
            communicate=_communicate,
        )


class DeferredDockerShell(DeferredShell):
    """Docker shell that creates or verifies its container on first command use."""

    def __init__(self, environment: SandboxEnvironment) -> None:
        super().__init__(
            default_cwd=Path(environment._work_dir),
            default_timeout=environment._shell_timeout,
        )
        self._environment = environment

    async def _resolve_shell(self) -> Shell:
        return await self._environment.ensure_ready_shell()

    async def get_deferred_context_instructions(self) -> str | None:
        exec_user_line = (
            f"\n  <exec-user>{self._environment._docker_exec_user}</exec-user>"
            if self._environment._docker_exec_user is not None
            else ""
        )
        status = self.ready_state.value
        return f"""<shell-execution>
  <type>docker-exec</type>
  <container-workdir>{self._environment._work_dir}</container-workdir>{exec_user_line}
  <status>{status}</status>
  <default-timeout>{self._environment._shell_timeout}s</default-timeout>
  <note>The Docker container is prepared on first shell command.</note>
</shell-execution>"""


class SandboxEnvironment(Environment):
    """Sandboxed environment with virtual file operations and containerized shell.

    This environment provides:
    - File operations via VirtualLocalFileOperator (host I/O with virtual paths)
    - Shell execution via a sandboxed shell (Docker by default, pluggable)
    - Symmetric path space: both file ops and shell see the same paths
    - Multiple mount support for mapping several host directories

    The agent sees a unified virtual path space for both file operations and
    shell commands. Internally, file I/O happens on the host filesystem while
    shell commands execute in the sandbox.

    Example:
        Single mount with Docker:

        ```python
        async with SandboxEnvironment(
            mounts=[VirtualMount(Path("/home/user/project"), Path("/workspace"))],
            image="python:3.11",
        ) as env:
            await env.file_operator.write_file("test.py", "print('hello')")
            code, stdout, stderr = await env.shell.execute("python test.py")
        ```

        Multiple mounts:

        ```python
        async with SandboxEnvironment(
            mounts=[
                VirtualMount(Path("/home/user/project"), Path("/workspace/project")),
                VirtualMount(Path("/home/user/.config"), Path("/workspace/.config")),
            ],
            work_dir="/workspace/project",
            image="python:3.11",
        ) as env:
            ...
        ```

        Using a custom shell backend:

        ```python
        custom_shell = MySSHShell(host="remote", workdir="/workspace")
        async with SandboxEnvironment(
            mounts=[VirtualMount(Path("/home/user/project"), Path("/workspace"))],
            shell=custom_shell,
        ) as env:
            ...
        ```
    """

    def __init__(
        self,
        mounts: list[VirtualMount],
        work_dir: str | None = None,
        shell: Shell | None = None,
        container_id: str | None = None,
        image: str | None = None,
        cleanup_on_exit: bool = True,
        shell_timeout: float = 30.0,
        docker_exec_user: str | None = None,
        docker_exec_default_env: dict[str, str] | None = None,
        enable_tmp_dir: bool = True,
        lazy_shell: bool = False,
        resource_state: ResourceRegistryState | None = None,
        resource_factories: dict[str, ResourceFactory] | None = None,
    ):
        """Initialize SandboxEnvironment.

        Args:
            mounts: List of mount mappings from host paths to virtual paths.
                At least one mount is required.
            work_dir: Default working directory (virtual path) for shell commands.
                If None, uses the first mount's virtual_path.
            shell: Custom shell backend to use. If provided, container_id and
                image are ignored. The shell should use work_dir as its
                working directory for path symmetry.
            container_id: Existing Docker container ID to use.
                Ignored if shell is provided.
            image: Docker image to create a new container from.
                Required if neither shell nor container_id is provided.
                Ignored if shell is provided.
            cleanup_on_exit: Whether to stop/remove Docker container on exit.
                Only applies to Docker-managed containers.
            shell_timeout: Default timeout for shell commands.
                Only applies when creating a DockerShell (no custom shell).
            docker_exec_user: Docker exec user for DockerShell.
            docker_exec_default_env: Default environment variables for DockerShell.
            enable_tmp_dir: Whether to create a per-environment temporary directory
                below an existing writable shared mount.
            lazy_shell: Whether Docker container readiness is deferred until shell use.
            resource_state: Optional state to restore resources from.
            resource_factories: Optional dictionary of resource factories.

        Raises:
            ValueError: If mounts is empty or no shell backend can be determined.
        """
        if not mounts:
            raise ValueError("At least one mount is required")
        if shell is None and container_id is None and image is None:
            raise ValueError("Either shell, container_id, or image must be provided")

        super().__init__(
            resource_state=resource_state,
            resource_factories=resource_factories,
        )
        self._mounts = mounts
        raw_work_dir = work_dir if work_dir is not None else str(mounts[0].virtual_path)

        # Validate work_dir is absolute and under at least one mount's virtual_path
        normalized_work_dir = normalize_virtual_path(raw_work_dir)
        if not normalized_work_dir.is_absolute():
            raise ValueError(f"work_dir must be absolute, got: {raw_work_dir}")
        if not any(self._is_path_under(normalized_work_dir, m.virtual_path) for m in mounts):
            raise ValueError(
                f"work_dir '{raw_work_dir}' is not under any mount virtual path: "
                f"{[str(m.virtual_path) for m in mounts]}"
            )
        self._work_dir = str(normalized_work_dir)
        self._custom_shell = shell
        self._container_id = container_id
        self._image = image
        self._cleanup_on_exit = cleanup_on_exit
        self._shell_timeout = shell_timeout
        self._docker_exec_user = docker_exec_user
        self._docker_exec_default_env = dict(docker_exec_default_env or {})
        self._enable_tmp_dir = enable_tmp_dir
        self._lazy_shell = lazy_shell

        # Runtime state
        self._created_container: bool = False
        self._client: docker.DockerClient | None = None
        self._tmp_host_dir: Path | None = None
        self._ready_shell: DockerShell | None = None
        self._ready_lock: asyncio.Lock = asyncio.Lock()

    @staticmethod
    def _is_path_under(path: VirtualPath, root: PurePath) -> bool:
        """Check if path is equal to or under root using path semantics."""
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    @property
    def client(self) -> docker.DockerClient:
        """Get Docker client with lazy initialization."""
        if self._client is None:
            self._client = docker.from_env()
        return self._client

    @property
    def container_id(self) -> str | None:
        """Return the configured or discovered container ID."""
        return self._container_id

    @property
    def ready_container_id(self) -> str | None:
        """Return the verified ready container ID after shell readiness succeeds."""
        if self._ready_shell is None:
            return None
        if isinstance(self._container_id, str) and self._container_id.strip() != "":
            return self._container_id.strip()
        return None

    async def _setup(self) -> None:
        """Initialize file operator, shell, and container."""
        await self._remove_tmp_host_dir()
        # Ensure all host directories exist before creating shared temporary storage.
        for mount in self._mounts:
            mount.host_path.resolve().mkdir(parents=True, exist_ok=True)

        if self._enable_tmp_dir:
            self._setup_shared_tmp_dir()

        # Existing mounts expose the same paths to file operations and the shell.
        self._file_operator = VirtualLocalFileOperator(
            mounts=self._mounts,
            default_virtual_path=Path(self._work_dir),
        )

        # Create shell
        if self._custom_shell is not None:
            self._shell = self._custom_shell
        elif self._lazy_shell:
            self._shell = DeferredDockerShell(self)
        else:
            self._shell = await self.ensure_ready_shell()

    def _setup_shared_tmp_dir(self) -> None:
        """Create temporary storage below the mount containing ``work_dir``."""
        work_dir = normalize_virtual_path(self._work_dir)
        shared_mounts = [mount for mount in self._mounts if self._is_path_under(work_dir, mount.virtual_path)]
        if not shared_mounts:
            raise RuntimeError("Temporary storage requires a writable shared mount")
        mount = max(shared_mounts, key=lambda item: len(item.virtual_path.parts))
        relative_tmp = PurePath(".ya-agent") / "tmp" / uuid4().hex
        host_root = mount.host_path.resolve()
        control_dir = host_root / ".ya-agent"
        tmp_parent = control_dir / "tmp"
        for directory in (control_dir, tmp_parent):
            if directory.exists() or directory.is_symlink():
                try:
                    directory.resolve().relative_to(host_root)
                except ValueError as exc:
                    raise RuntimeError("Temporary storage escapes the shared mount") from exc
            directory.mkdir(exist_ok=True)
            try:
                directory.resolve().relative_to(host_root)
            except ValueError as exc:
                raise RuntimeError("Temporary storage escapes the shared mount") from exc
        tmp_host_dir = tmp_parent / relative_tmp.name
        tmp_host_dir.mkdir()
        self._tmp_host_dir = tmp_host_dir
        self._tmp_dir = mount.virtual_path / relative_tmp
        workspace_stat = host_root.stat()
        tmp_host_dir.chmod(workspace_stat.st_mode & 0o777)
        if sys.platform != "win32" and os.geteuid() == 0:
            os.chown(tmp_host_dir, workspace_stat.st_uid, workspace_stat.st_gid)

    async def _ensure_container(self) -> None:
        if self._container_id is None:
            await self._create_owned_container()
        else:
            await self._verify_container()

    async def _create_owned_container(self) -> str:
        """Create a container without letting caller cancellation lose its handle."""
        create_task = asyncio.create_task(self._create_container())
        cancellation: asyncio.CancelledError | None = None
        while True:
            try:
                container_id = await asyncio.shield(create_task)
                break
            except asyncio.CancelledError as exc:
                cancellation = exc
                if create_task.cancelled():
                    raise
                if create_task.done():
                    container_id = create_task.result()
                    break
        self._container_id = container_id
        self._created_container = True
        if cancellation is not None:
            raise cancellation
        return container_id

    async def ensure_ready_shell(self) -> DockerShell:
        """Ensure Docker container readiness and return a concrete DockerShell."""
        if self._ready_shell is not None:
            return self._ready_shell

        async with self._ready_lock:
            if self._ready_shell is not None:
                return self._ready_shell
            await self._ensure_container()
            if self._container_id is None:
                raise RuntimeError("container_id must be set when no custom shell is provided")
            await self._wait_for_container_ready(self._container_id)
            self._ready_shell = DockerShell(
                container_id=self._container_id,
                container_workdir=self._work_dir,
                default_timeout=self._shell_timeout,
                exec_user=self._docker_exec_user,
                default_env=self._docker_exec_default_env,
            )
            self._ready_shell._client = self._client
            return self._ready_shell

    async def _wait_for_container_ready(self, container_id: str) -> None:
        """Wait for Docker health checks to report readiness, when configured."""
        timeout_seconds = 60.0
        poll_interval_seconds = 0.25

        def _wait() -> None:
            deadline = time.monotonic() + timeout_seconds
            while True:
                try:
                    container = self.client.containers.get(container_id)
                    container.reload()
                except docker.errors.NotFound as e:
                    raise RuntimeError(f"Container not found: {container_id}") from e
                except docker.errors.APIError as e:
                    raise RuntimeError(f"Failed to inspect container health: {e}") from e

                attrs = container.attrs
                state = attrs.get("State") if isinstance(attrs, dict) else None
                health = state.get("Health") if isinstance(state, dict) else None
                health_status = health.get("Status") if isinstance(health, dict) else None
                if health_status is None or health_status == "healthy":
                    return
                if health_status == "unhealthy":
                    raise RuntimeError(f"Container {container_id} is unhealthy")
                if health_status != "starting":
                    raise RuntimeError(f"Container {container_id} has unexpected health status: {health_status}")
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"Container {container_id} did not become healthy within {timeout_seconds}s")
                time.sleep(poll_interval_seconds)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _wait)

    async def _teardown(self) -> None:
        """Clean up unowned handles, container, and shared temporary storage."""
        removed_created_container = False
        try:
            await self._close_ready_shell_if_unowned()
        finally:
            self._ready_shell = None
            try:
                # Cleanup container if we created it and cleanup_on_exit is True.
                if self._cleanup_on_exit and self._created_container and self._container_id is not None:
                    await self._stop_container()
                    removed_created_container = True
            finally:
                if removed_created_container:
                    self._container_id = None
                    self._created_container = False
                try:
                    await self._remove_tmp_host_dir()
                finally:
                    self._file_operator = None
                    self._shell = None

    async def _remove_tmp_host_dir(self) -> None:
        """Remove owned temporary storage, retaining ownership on failure."""
        owned_tmp = self._tmp_host_dir
        if owned_tmp is None:
            self._tmp_dir = None
            return
        try:
            await asyncio.to_thread(shutil.rmtree, owned_tmp)
        except FileNotFoundError:
            try:
                owned_tmp.lstat()
            except FileNotFoundError:
                pass
            else:
                raise
        self._tmp_host_dir = None
        self._tmp_dir = None

    async def _close_ready_shell_if_unowned(self) -> None:
        ready_shell = self._ready_shell
        if ready_shell is None:
            return
        if self._shell is ready_shell:
            return
        if isinstance(self._shell, DeferredShell) and self._shell.resolved_shell is ready_shell:
            return
        await ready_shell.close()

    async def _create_container(self) -> str:
        """Create and start a new container with the configured shared mounts."""
        if self._image is None:
            raise ValueError("Image must be provided to create a new container")

        image = self._image  # Capture for closure
        work_dir = self._work_dir
        mounts = self._mounts

        def _run_container() -> str:
            try:
                volumes = {str(m.host_path.resolve()): {"bind": str(m.virtual_path), "mode": "rw"} for m in mounts}
                container = self.client.containers.run(
                    image=image,
                    volumes=volumes,
                    working_dir=work_dir,
                    detach=True,
                    stdin_open=True,
                    tty=True,
                )
                container_id = container.id
                if container_id is None:
                    raise RuntimeError("Container was created but has no ID")
                return container_id
            except docker.errors.ImageNotFound as e:
                raise RuntimeError(f"Docker image not found: {image}") from e
            except docker.errors.APIError as e:
                raise RuntimeError(f"Failed to start container: {e}") from e

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _run_container)

    async def _verify_container(self) -> None:
        """Verify that the existing container is running, auto-starting if stopped."""
        container_id = self._container_id
        if container_id is None:
            raise RuntimeError("Container ID is not set")

        def _check_and_start_container() -> None:
            try:
                container = self.client.containers.get(container_id)
                container.reload()
                if container.status == "running":
                    return
                # Auto-start stopped/exited containers (handles restart scenarios)
                if container.status in ("exited", "created", "paused"):
                    container.start()
                    container.reload()
                    if container.status != "running":
                        raise RuntimeError(f"Container {container_id} failed to start (status: {container.status})")
                else:
                    raise RuntimeError(f"Container {container_id} is in unrecoverable state: {container.status}")
            except docker.errors.NotFound as e:
                raise RuntimeError(f"Container not found: {container_id}") from e
            except docker.errors.APIError as e:
                raise RuntimeError(f"Failed to verify/start container: {e}") from e

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _check_and_start_container)

    async def _stop_container(self) -> None:
        """Stop and remove the container."""
        container_id = self._container_id
        if container_id is None:
            return

        def _stop() -> None:
            try:
                container = self.client.containers.get(container_id)
            except docker.errors.NotFound:
                return  # Container already gone
            except docker.errors.APIError:
                return  # Best effort cleanup

            # Removal must still run when the container is already stopped or stop fails.
            with contextlib.suppress(docker.errors.NotFound, docker.errors.APIError):
                container.stop(timeout=10)
            with contextlib.suppress(docker.errors.NotFound, docker.errors.APIError):
                container.remove(force=True)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _stop)
