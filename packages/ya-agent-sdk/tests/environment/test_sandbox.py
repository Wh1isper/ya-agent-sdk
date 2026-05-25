"""Tests for SandboxEnvironment and DockerShell.

These tests require Docker to be installed.
The entire module is skipped when docker package is unavailable.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from ya_agent_environment import ShellExecutionError

# Skip all tests in this module if docker is not installed
docker = pytest.importorskip("docker")

from ya_agent_sdk.environment.local import VirtualMount  # noqa: E402
from ya_agent_sdk.environment.sandbox import DeferredDockerShell, DockerShell, SandboxEnvironment  # noqa: E402
from ya_agent_sdk.environment.virtual_path import normalize_virtual_path  # noqa: E402

# --- DockerShell Tests ---


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


async def test_docker_shell_execute_success() -> None:
    """Should execute command and return results."""
    shell = DockerShell(container_id="test123")

    mock_container = MagicMock()
    mock_container.exec_run.return_value = MagicMock(
        exit_code=0,
        output=(b"hello\n", b""),
    )

    mock_client = MagicMock()
    mock_client.containers.get.return_value = mock_container
    shell._client = mock_client

    code, stdout, stderr = await shell.execute("echo hello")

    assert code == 0
    assert stdout == "hello\n"
    assert stderr == ""
    mock_container.exec_run.assert_called_once()


async def test_docker_shell_execute_with_cwd() -> None:
    """Should pass workdir to docker exec."""
    shell = DockerShell(container_id="test123", container_workdir="/workspace")

    mock_container = MagicMock()
    mock_container.exec_run.return_value = MagicMock(
        exit_code=0,
        output=(b"", b""),
    )

    mock_client = MagicMock()
    mock_client.containers.get.return_value = mock_container
    shell._client = mock_client

    await shell.execute("ls", cwd="subdir")

    mock_container.exec_run.assert_called_once()
    call_kwargs = mock_container.exec_run.call_args[1]
    assert call_kwargs["workdir"] == "/workspace/subdir"


async def test_docker_shell_execute_with_absolute_cwd() -> None:
    """Should use absolute cwd as-is."""
    shell = DockerShell(container_id="test123", container_workdir="/workspace")

    mock_container = MagicMock()
    mock_container.exec_run.return_value = MagicMock(
        exit_code=0,
        output=(b"", b""),
    )

    mock_client = MagicMock()
    mock_client.containers.get.return_value = mock_container
    shell._client = mock_client

    await shell.execute("ls", cwd="/tmp")  # noqa: S108

    call_kwargs = mock_container.exec_run.call_args[1]
    assert call_kwargs["workdir"] == "/tmp"  # noqa: S108


async def test_docker_shell_execute_with_env() -> None:
    """Should pass environment variables."""
    shell = DockerShell(container_id="test123")

    mock_container = MagicMock()
    mock_container.exec_run.return_value = MagicMock(
        exit_code=0,
        output=(b"", b""),
    )

    mock_client = MagicMock()
    mock_client.containers.get.return_value = mock_container
    shell._client = mock_client

    await shell.execute("env", env={"FOO": "bar"})

    call_kwargs = mock_container.exec_run.call_args[1]
    assert call_kwargs["environment"] == {"FOO": "bar"}


async def test_docker_shell_execute_with_exec_user_and_default_env() -> None:
    """Should pass docker exec user and merge default environment variables."""
    shell = DockerShell(
        container_id="test123",
        exec_user="1000:1000",
        default_env={"HOME": "/home/claw", "USER": "claw", "FOO": "default"},
    )

    mock_container = MagicMock()
    mock_container.exec_run.return_value = MagicMock(
        exit_code=0,
        output=(b"", b""),
    )

    mock_client = MagicMock()
    mock_client.containers.get.return_value = mock_container
    shell._client = mock_client

    await shell.execute("env", env={"FOO": "override", "BAR": "baz"})

    call_kwargs = mock_container.exec_run.call_args[1]
    assert call_kwargs["user"] == "1000:1000"
    assert call_kwargs["environment"] == {
        "HOME": "/home/claw",
        "USER": "claw",
        "FOO": "override",
        "BAR": "baz",
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


async def test_sandbox_environment_lazy_shell_does_not_verify_container_on_enter(tmp_path: Path) -> None:
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


async def test_sandbox_environment_lazy_shell_creates_container_on_first_command(tmp_path: Path) -> None:
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

    async with env:
        assert env.container_id is None
        assert mock_client.containers.run.call_count == 0
        code, stdout, stderr = await env.shell.execute("echo ok")
        assert (code, stdout, stderr) == (0, "ok", "")
        assert env.container_id == "new123"
        mock_client.containers.run.assert_called_once()

    mock_container.stop.assert_called_once()
    mock_container.remove.assert_called_once()
