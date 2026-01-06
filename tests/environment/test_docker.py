"""Tests for DockerEnvironment and DockerShell.

These tests require Docker to be installed and running.
Tests are marked with pytest.mark.docker to allow skipping when Docker is unavailable.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Skip all tests if docker is not installed
pytest.importorskip("docker")

from pai_agent_sdk.environment import ShellExecutionError
from pai_agent_sdk.environment.docker import DockerEnvironment, DockerShell

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


# --- DockerEnvironment Tests ---


def test_docker_environment_requires_container_or_image() -> None:
    """Should raise ValueError if neither container_id nor image provided."""
    with pytest.raises(ValueError, match="Either container_id or image must be provided"):
        DockerEnvironment(mount_dir=Path("/tmp"))  # noqa: S108


def test_docker_environment_initialization_with_container_id(tmp_path: Path) -> None:
    """Should initialize with existing container_id."""
    env = DockerEnvironment(
        mount_dir=tmp_path,
        container_id="existing123",
        container_workdir="/app",
        cleanup_on_exit=False,
    )
    assert env._container_id == "existing123"
    assert env._container_workdir == "/app"
    assert env._cleanup_on_exit is False


def test_docker_environment_initialization_with_image(tmp_path: Path) -> None:
    """Should initialize with image for new container."""
    env = DockerEnvironment(
        mount_dir=tmp_path,
        image="python:3.11",
        cleanup_on_exit=True,
    )
    assert env._image == "python:3.11"
    assert env._cleanup_on_exit is True


async def test_docker_environment_properties_before_enter(tmp_path: Path) -> None:
    """Should raise error when accessing properties before entering context."""
    env = DockerEnvironment(
        mount_dir=tmp_path,
        container_id="test123",
    )
    with pytest.raises(RuntimeError, match="Environment not entered"):
        _ = env.file_operator
    with pytest.raises(RuntimeError, match="Environment not entered"):
        _ = env.shell


async def test_docker_environment_enter_with_existing_container(tmp_path: Path) -> None:
    """Should verify container and create operators on enter."""
    env = DockerEnvironment(
        mount_dir=tmp_path,
        container_id="existing123",
        container_workdir="/workspace",
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
        assert env._file_operator._default_path == tmp_path.resolve()
        assert env._shell._container_id == "existing123"
        assert env._shell._container_workdir == "/workspace"

    # Verify container was not stopped (cleanup_on_exit=False)
    assert mock_container.stop.call_count == 0


async def test_docker_environment_enter_creates_new_container(tmp_path: Path) -> None:
    """Should create container when entering with image."""
    env = DockerEnvironment(
        mount_dir=tmp_path,
        image="python:3.11",
        container_workdir="/workspace",
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


async def test_docker_environment_file_operator_uses_mount_dir(tmp_path: Path) -> None:
    """Should configure file operator with mount_dir as default path."""
    env = DockerEnvironment(
        mount_dir=tmp_path,
        container_id="test123",
    )

    mock_container = MagicMock()
    mock_container.status = "running"

    mock_client = MagicMock()
    mock_client.containers.get.return_value = mock_container
    env._client = mock_client

    async with env:
        # Write a file using file operator
        await env.file_operator.write_file("test.txt", "hello")
        assert (tmp_path / "test.txt").read_text() == "hello"


async def test_docker_environment_tmp_dir_enabled(tmp_path: Path) -> None:
    """Should create tmp directory when enabled."""
    env = DockerEnvironment(
        mount_dir=tmp_path,
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


async def test_docker_environment_tmp_dir_disabled(tmp_path: Path) -> None:
    """Should not create tmp directory when disabled."""
    env = DockerEnvironment(
        mount_dir=tmp_path,
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


async def test_docker_environment_get_context_instructions(tmp_path: Path) -> None:
    """Should return combined context instructions."""
    env = DockerEnvironment(
        mount_dir=tmp_path,
        container_id="test123",
        container_workdir="/workspace",
    )

    mock_container = MagicMock()
    mock_container.status = "running"

    mock_client = MagicMock()
    mock_client.containers.get.return_value = mock_container
    env._client = mock_client

    async with env:
        instructions = await env.get_context_instructions()

        assert instructions is not None
        assert "docker-environment" in instructions
        assert "mount-mapping" in instructions
        assert str(tmp_path) in instructions
        assert "/workspace" in instructions


async def test_docker_environment_cross_session_sharing(tmp_path: Path) -> None:
    """Should support cross-session container sharing with cleanup_on_exit=False."""
    # First session - use existing container, don't cleanup
    env1 = DockerEnvironment(
        mount_dir=tmp_path,
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

    # Container should not be stopped
    assert mock_container.stop.call_count == 0

    # Second session - same container
    env2 = DockerEnvironment(
        mount_dir=tmp_path,
        container_id="shared123",
        cleanup_on_exit=False,
    )
    env2._client = mock_client

    async with env2:
        # File from previous session should exist
        content = await env2.file_operator.read_file("session1.txt")
        assert content == "from session 1"
