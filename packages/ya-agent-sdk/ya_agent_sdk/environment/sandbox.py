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
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ya_agent_environment import (
    Environment,
    EnvironmentNotEnteredError,
    ExecutionHandle,
    ResourceFactory,
    ResourceRegistryState,
    Shell,
    ShellExecutionError,
    ShellTimeoutError,
    StdinAdapter,
)

from ya_agent_sdk.environment.local import VirtualLocalFileOperator, VirtualMount

if TYPE_CHECKING:
    pass

import docker
import docker.errors


def _coerce_docker_exec_output(output: Any) -> tuple[bytes, bytes]:
    if isinstance(output, tuple) and len(output) == 2:
        out, err = output
        stdout_bytes = out if isinstance(out, bytes) else b""
        stderr_bytes = err if isinstance(err, bytes) else b""
        return stdout_bytes, stderr_bytes
    if isinstance(output, bytes):
        return output, b""
    if isinstance(output, Iterator):
        return b"".join(output), b""
    return b"", b""


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

    async def execute(
        self,
        command: str,
        *,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> tuple[int, str, str]:
        """Execute a command inside the Docker container.

        Args:
            command: Command string to execute via shell.
            timeout: Execution timeout in seconds.
            env: Environment variables for the command.
            cwd: Working directory (relative to container_workdir, or absolute).

        Returns:
            Tuple of (exit_code, stdout, stderr).
        """
        if not command:
            raise ShellExecutionError(command, stderr="Empty command")

        effective_timeout = timeout if timeout is not None else self._default_timeout

        # Determine working directory inside container
        if cwd is not None:
            workdir = cwd if cwd.startswith("/") else f"{self._container_workdir}/{cwd}"
        else:
            workdir = self._container_workdir

        def _exec_command() -> tuple[int, str, str]:
            try:
                container = self.client.containers.get(self._container_id)
                exec_kwargs: dict[str, Any] = {
                    "cmd": ["/bin/sh", "-c", command],
                    "stdout": True,
                    "stderr": True,
                    "demux": True,
                    "workdir": workdir,
                    "environment": self._build_exec_env(env),
                }
                if self._exec_user is not None:
                    exec_kwargs["user"] = self._exec_user
                result = container.exec_run(**exec_kwargs)

                exit_code = result.exit_code if isinstance(result.exit_code, int) else 0
                stdout_bytes, stderr_bytes = _coerce_docker_exec_output(result.output)

                stdout = stdout_bytes.decode("utf-8", errors="replace")
                stderr = stderr_bytes.decode("utf-8", errors="replace")
                return (exit_code, stdout, stderr)

            except docker.errors.NotFound as e:
                raise ShellExecutionError(
                    command,
                    stderr=f"Container not found: {self._container_id}",
                ) from e
            except docker.errors.APIError as e:
                raise ShellExecutionError(command, stderr=str(e)) from e

        try:
            loop = asyncio.get_running_loop()
            if effective_timeout > 0:
                return await asyncio.wait_for(
                    loop.run_in_executor(None, _exec_command),
                    timeout=effective_timeout,
                )
            else:
                return await loop.run_in_executor(None, _exec_command)
        except TimeoutError as e:
            raise ShellTimeoutError(command, effective_timeout) from e

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

        args: list[str] = ["docker", "exec", "-i"]
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

    async def _create_process(
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

        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as e:
            raise ShellExecutionError(command, stderr=str(e)) from e

        if process.stdout is None or process.stderr is None:
            raise ShellExecutionError(command, stderr="Failed to capture subprocess streams")

        async def _wait() -> int:
            await process.wait()
            return process.returncode or 0

        async def _kill() -> None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()

        async def _send_signal(sig: int) -> None:
            if process.pid is not None:
                with contextlib.suppress(ProcessLookupError, OSError):
                    os.kill(process.pid, sig)

        stdin = StdinAdapter(process.stdin) if process.stdin is not None else None

        return ExecutionHandle(
            stdout=process.stdout,
            stderr=process.stderr,
            wait=_wait,
            kill=_kill,
            stdin=stdin,
            pid=process.pid,
            send_signal=_send_signal,
        )


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
        tmp_base_dir: Path | None = None,
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
            enable_tmp_dir: Whether to create a session temporary directory.
            tmp_base_dir: Base directory for creating session temporary directory.
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
        normalized_work_dir = Path(os.path.normpath(raw_work_dir))
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
        self._tmp_base_dir = tmp_base_dir

        # Runtime state
        self._created_container: bool = False
        self._client: docker.DockerClient | None = None
        self._tmp_dir_obj: tempfile.TemporaryDirectory[str] | None = None

    @staticmethod
    def _is_path_under(path: Path, root: Path) -> bool:
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
        """Return the container ID (available after entering context)."""
        return self._container_id

    @property
    def tmp_dir(self) -> Path | None:
        """Return the session temporary directory path, or None if not enabled."""
        if self._tmp_dir_obj is None:
            return None
        return Path(self._tmp_dir_obj.name)

    async def _setup(self) -> None:
        """Initialize file operator, shell, and container."""
        # Create tmp directory if enabled
        tmp_dir_path: Path | None = None
        if self._enable_tmp_dir:
            self._tmp_dir_obj = tempfile.TemporaryDirectory(
                prefix="ya_agent_sandbox_",
                dir=str(self._tmp_base_dir) if self._tmp_base_dir else None,
            )
            tmp_dir_path = Path(self._tmp_dir_obj.name)

        # Ensure all host directories exist
        for mount in self._mounts:
            mount.host_path.resolve().mkdir(parents=True, exist_ok=True)

        # Create or verify Docker container (unless custom shell provided)
        if self._custom_shell is None:
            if self._container_id is None:
                # Create new container
                self._container_id = await self._create_container()
                self._created_container = True
            else:
                # Verify existing container is running
                await self._verify_container()

        # Create file operator (virtual paths mapped to host filesystem)
        self._file_operator = VirtualLocalFileOperator(
            mounts=self._mounts,
            default_virtual_path=Path(self._work_dir),
            tmp_dir=tmp_dir_path,
        )

        # Create shell
        if self._custom_shell is not None:
            self._shell = self._custom_shell
        else:
            if self._container_id is None:
                raise RuntimeError("container_id must be set when no custom shell is provided")
            self._shell = DockerShell(
                container_id=self._container_id,
                container_workdir=self._work_dir,
                default_timeout=self._shell_timeout,
                exec_user=self._docker_exec_user,
                default_env=self._docker_exec_default_env,
            )

    async def _teardown(self) -> None:
        """Clean up container and tmp directory.

        Note: Do NOT null _file_operator or _shell here.
        The base Environment.__aexit__ calls close() on them after
        _teardown returns.  Nulling here would skip close() and
        leak background processes.
        """
        # Cleanup container if we created it and cleanup_on_exit is True
        if self._cleanup_on_exit and self._created_container and self._container_id is not None:
            await self._stop_container()

        # Cleanup tmp directory
        if self._tmp_dir_obj is not None:
            self._tmp_dir_obj.cleanup()
            self._tmp_dir_obj = None

    async def _create_container(self) -> str:
        """Create and start a new container with all mounts and tmp_dir."""
        if self._image is None:
            raise ValueError("Image must be provided to create a new container")

        image = self._image  # Capture for closure
        work_dir = self._work_dir
        mounts = self._mounts
        tmp_dir = self.tmp_dir

        def _run_container() -> str:
            try:
                volumes = {str(m.host_path.resolve()): {"bind": str(m.virtual_path), "mode": "rw"} for m in mounts}
                # Also mount tmp_dir into container so shell can access tmp files
                if tmp_dir is not None:
                    volumes[str(tmp_dir)] = {"bind": str(tmp_dir), "mode": "rw"}
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
                container.stop(timeout=10)
                container.remove(force=True)
            except docker.errors.NotFound:
                pass  # Container already gone
            except docker.errors.APIError:
                pass  # Best effort cleanup

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _stop)

    async def get_context_instructions(self) -> str:
        """Return combined context instructions for file operations and shell.

        Since VirtualLocalFileOperator and shell share the same path space,
        no mount-mapping instructions are needed.

        Raises:
            EnvironmentNotEnteredError: If environment has not been entered yet.
        """
        if not self._file_operator or not self._shell:
            raise EnvironmentNotEnteredError("get_context_instructions")

        file_instructions = await self._file_operator.get_context_instructions()
        shell_instructions = await self._shell.get_context_instructions()

        parts = []
        if file_instructions:
            parts.append(file_instructions)
        if shell_instructions:
            parts.append(shell_instructions)

        return "\n\n".join(parts)
