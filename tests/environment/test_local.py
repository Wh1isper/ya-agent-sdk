"""Tests for LocalFileOperator and LocalShell."""

from pathlib import Path

import pytest
from inline_snapshot import snapshot

from pai_agent_sdk.environment import (
    FileOperationError,
    LocalEnvironment,
    LocalFileOperator,
    LocalShell,
    PathNotAllowedError,
    ShellTimeoutError,
)

# --- LocalFileOperator Tests ---


def test_file_operator_default_initialization() -> None:
    """Should initialize with cwd as default."""
    op = LocalFileOperator()
    assert op._default_path == Path.cwd().resolve()


def test_file_operator_custom_allowed_paths(tmp_path: Path) -> None:
    """Should accept custom allowed paths."""
    op = LocalFileOperator(allowed_paths=[tmp_path], default_path=tmp_path)
    assert tmp_path.resolve() in op._allowed_paths


def test_file_operator_default_path_included_in_allowed(tmp_path: Path) -> None:
    """Should ensure default_path is in allowed_paths."""
    other_path = tmp_path / "other"
    other_path.mkdir()
    op = LocalFileOperator(allowed_paths=[other_path], default_path=tmp_path)
    assert tmp_path.resolve() in op._allowed_paths
    assert other_path.resolve() in op._allowed_paths


async def test_file_operator_read_file(tmp_path: Path) -> None:
    """Should read file content."""
    (tmp_path / "test.txt").write_text("hello world")
    op = LocalFileOperator(allowed_paths=[tmp_path], default_path=tmp_path)
    assert await op.read_file("test.txt") == "hello world"


async def test_file_operator_read_file_with_offset(tmp_path: Path) -> None:
    """Should read file content from character offset."""
    (tmp_path / "test.txt").write_text("hello world")
    op = LocalFileOperator(allowed_paths=[tmp_path], default_path=tmp_path)
    assert await op.read_file("test.txt", offset=6) == "world"


async def test_file_operator_read_file_with_length(tmp_path: Path) -> None:
    """Should read limited characters from file."""
    (tmp_path / "test.txt").write_text("hello world")
    op = LocalFileOperator(allowed_paths=[tmp_path], default_path=tmp_path)
    assert await op.read_file("test.txt", length=5) == "hello"


async def test_file_operator_read_file_with_offset_and_length(tmp_path: Path) -> None:
    """Should read substring with offset and length."""
    (tmp_path / "test.txt").write_text("hello world")
    op = LocalFileOperator(allowed_paths=[tmp_path], default_path=tmp_path)
    assert await op.read_file("test.txt", offset=3, length=5) == "lo wo"


async def test_file_operator_read_bytes(tmp_path: Path) -> None:
    """Should read file as bytes."""
    (tmp_path / "test.bin").write_bytes(b"\x00\x01\x02")
    op = LocalFileOperator(allowed_paths=[tmp_path], default_path=tmp_path)
    assert await op.read_bytes("test.bin") == b"\x00\x01\x02"


async def test_file_operator_read_bytes_with_offset(tmp_path: Path) -> None:
    """Should read bytes from offset."""
    (tmp_path / "test.bin").write_bytes(b"\x00\x01\x02\x03\x04")
    op = LocalFileOperator(allowed_paths=[tmp_path], default_path=tmp_path)
    assert await op.read_bytes("test.bin", offset=2) == b"\x02\x03\x04"


async def test_file_operator_read_bytes_with_length(tmp_path: Path) -> None:
    """Should read limited bytes."""
    (tmp_path / "test.bin").write_bytes(b"\x00\x01\x02\x03\x04")
    op = LocalFileOperator(allowed_paths=[tmp_path], default_path=tmp_path)
    assert await op.read_bytes("test.bin", length=3) == b"\x00\x01\x02"


async def test_file_operator_read_bytes_with_offset_and_length(tmp_path: Path) -> None:
    """Should read byte slice with offset and length."""
    (tmp_path / "test.bin").write_bytes(b"\x00\x01\x02\x03\x04")
    op = LocalFileOperator(allowed_paths=[tmp_path], default_path=tmp_path)
    assert await op.read_bytes("test.bin", offset=1, length=3) == b"\x01\x02\x03"


async def test_file_operator_write_file_string(tmp_path: Path) -> None:
    """Should write string content."""
    op = LocalFileOperator(allowed_paths=[tmp_path], default_path=tmp_path)
    await op.write_file("output.txt", "test content")
    assert (tmp_path / "output.txt").read_text() == "test content"


async def test_file_operator_write_file_bytes(tmp_path: Path) -> None:
    """Should write bytes content."""
    op = LocalFileOperator(allowed_paths=[tmp_path], default_path=tmp_path)
    await op.write_file("output.bin", b"\x00\x01\x02")
    assert (tmp_path / "output.bin").read_bytes() == b"\x00\x01\x02"


async def test_file_operator_append_file(tmp_path: Path) -> None:
    """Should append content to file."""
    (tmp_path / "test.txt").write_text("hello")
    op = LocalFileOperator(allowed_paths=[tmp_path], default_path=tmp_path)
    await op.append_file("test.txt", " world")
    assert (tmp_path / "test.txt").read_text() == "hello world"


async def test_file_operator_delete_file(tmp_path: Path) -> None:
    """Should delete file."""
    (tmp_path / "test.txt").write_text("hello")
    op = LocalFileOperator(allowed_paths=[tmp_path], default_path=tmp_path)
    await op.delete("test.txt")
    assert not (tmp_path / "test.txt").exists()


async def test_file_operator_list_dir(tmp_path: Path) -> None:
    """Should list directory contents."""
    (tmp_path / "a.txt").touch()
    (tmp_path / "b.txt").touch()
    (tmp_path / "subdir").mkdir()
    op = LocalFileOperator(allowed_paths=[tmp_path], default_path=tmp_path)
    entries = await op.list_dir(".")
    assert set(entries) == {"a.txt", "b.txt", "subdir"}


async def test_file_operator_exists(tmp_path: Path) -> None:
    """Should check path existence."""
    (tmp_path / "exists.txt").touch()
    op = LocalFileOperator(allowed_paths=[tmp_path], default_path=tmp_path)
    assert await op.exists("exists.txt") is True
    assert await op.exists("not_exists.txt") is False


async def test_file_operator_is_file_and_is_dir(tmp_path: Path) -> None:
    """Should distinguish files and directories."""
    (tmp_path / "file.txt").touch()
    (tmp_path / "subdir").mkdir()
    op = LocalFileOperator(allowed_paths=[tmp_path], default_path=tmp_path)
    assert await op.is_file("file.txt") is True
    assert await op.is_dir("file.txt") is False
    assert await op.is_file("subdir") is False
    assert await op.is_dir("subdir") is True


async def test_file_operator_mkdir(tmp_path: Path) -> None:
    """Should create directory."""
    op = LocalFileOperator(allowed_paths=[tmp_path], default_path=tmp_path)
    await op.mkdir("new_dir")
    assert (tmp_path / "new_dir").is_dir()


async def test_file_operator_mkdir_parents(tmp_path: Path) -> None:
    """Should create nested directories with parents=True."""
    op = LocalFileOperator(allowed_paths=[tmp_path], default_path=tmp_path)
    await op.mkdir("a/b/c", parents=True)
    assert (tmp_path / "a" / "b" / "c").is_dir()


async def test_file_operator_move(tmp_path: Path) -> None:
    """Should move file."""
    (tmp_path / "src.txt").write_text("content")
    op = LocalFileOperator(allowed_paths=[tmp_path], default_path=tmp_path)
    await op.move("src.txt", "dst.txt")
    assert not (tmp_path / "src.txt").exists()
    assert (tmp_path / "dst.txt").read_text() == "content"


async def test_file_operator_copy(tmp_path: Path) -> None:
    """Should copy file."""
    (tmp_path / "src.txt").write_text("content")
    op = LocalFileOperator(allowed_paths=[tmp_path], default_path=tmp_path)
    await op.copy("src.txt", "dst.txt")
    assert (tmp_path / "src.txt").read_text() == "content"
    assert (tmp_path / "dst.txt").read_text() == "content"


async def test_file_operator_path_not_allowed_error(tmp_path: Path) -> None:
    """Should raise PathNotAllowedError for paths outside allowed dirs."""
    op = LocalFileOperator(allowed_paths=[tmp_path], default_path=tmp_path)
    with pytest.raises(PathNotAllowedError):
        await op.read_file("/etc/passwd")


async def test_file_operator_file_operation_error(tmp_path: Path) -> None:
    """Should raise FileOperationError for failed operations."""
    op = LocalFileOperator(allowed_paths=[tmp_path], default_path=tmp_path)
    with pytest.raises(FileOperationError) as exc_info:
        await op.read_file("not_exists.txt")
    assert "file not found" in str(exc_info.value)


async def test_file_operator_get_context_instructions(tmp_path: Path) -> None:
    """Should return context instructions in XML format with file tree."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("# main")
    (tmp_path / "README.md").write_text("# readme")
    (tmp_path / ".git").mkdir()

    op = LocalFileOperator(allowed_paths=[tmp_path], default_path=tmp_path)
    instructions = await op.get_context_instructions()

    # Replace dynamic path with placeholder for snapshot
    assert instructions is not None
    normalized = instructions.replace(str(tmp_path), "/test/workspace")
    assert normalized == snapshot(
        """\
<file-system>
  <allowed-directories>
    <path>/test/workspace</path>
  </allowed-directories>
  <default-directory>/test/workspace</default-directory>
  <file-tree>
.git/ (skipped)
src/
  main.py
README.md
  </file-tree>
</file-system>"""
    )


# --- LocalShell Tests ---


def test_shell_default_initialization() -> None:
    """Should initialize with cwd as default."""
    shell = LocalShell()
    assert shell._default_cwd == Path.cwd().resolve()


def test_shell_default_cwd_included_in_allowed(tmp_path: Path) -> None:
    """Should ensure default_cwd is in allowed_paths."""
    other_path = tmp_path / "other"
    other_path.mkdir()
    shell = LocalShell(allowed_paths=[other_path], default_cwd=tmp_path)
    assert tmp_path.resolve() in shell._allowed_paths
    assert other_path.resolve() in shell._allowed_paths


async def test_shell_execute_simple_command(tmp_path: Path) -> None:
    """Should execute simple command."""
    shell = LocalShell(allowed_paths=[tmp_path], default_cwd=tmp_path)
    exit_code, stdout, _ = await shell.execute("echo hello")
    assert exit_code == 0
    assert "hello" in stdout


async def test_shell_execute_with_cwd(tmp_path: Path) -> None:
    """Should execute command in specified directory."""
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    shell = LocalShell(allowed_paths=[tmp_path], default_cwd=tmp_path)
    exit_code, stdout, _ = await shell.execute("pwd", cwd=str(subdir))
    assert exit_code == 0
    assert "subdir" in stdout


async def test_shell_execute_cwd_not_allowed(tmp_path: Path) -> None:
    """Should raise PathNotAllowedError for cwd outside allowed dirs."""
    shell = LocalShell(allowed_paths=[tmp_path], default_cwd=tmp_path)
    with pytest.raises(PathNotAllowedError):
        await shell.execute("ls", cwd="/etc")


async def test_shell_execute_timeout(tmp_path: Path) -> None:
    """Should raise ShellTimeoutError on timeout."""
    shell = LocalShell(allowed_paths=[tmp_path], default_cwd=tmp_path)
    with pytest.raises(ShellTimeoutError):
        await shell.execute("sleep 10", timeout=0.1)


async def test_shell_execute_command_not_found(tmp_path: Path) -> None:
    """Should return non-zero exit code for non-existent command."""
    shell = LocalShell(allowed_paths=[tmp_path], default_cwd=tmp_path)
    exit_code, _, stderr = await shell.execute("nonexistent_command_xyz")
    assert exit_code != 0
    assert "not found" in stderr.lower()


async def test_shell_execute_returns_stderr(tmp_path: Path) -> None:
    """Should capture stderr."""
    shell = LocalShell(allowed_paths=[tmp_path], default_cwd=tmp_path)
    exit_code, _, stderr = await shell.execute("ls nonexistent_file_xyz")
    assert exit_code != 0
    assert stderr


async def test_shell_get_context_instructions() -> None:
    """Should return context instructions in XML format."""
    shell = LocalShell(
        allowed_paths=[Path("/workspace")],
        default_cwd=Path("/workspace"),
        default_timeout=30.0,
    )
    instructions = await shell.get_context_instructions()
    assert instructions == snapshot(
        """\
<shell-execution>
  <allowed-working-directories>
    <path>/workspace</path>
  </allowed-working-directories>
  <default-working-directory>/workspace</default-working-directory>
  <default-timeout>30.0s</default-timeout>
  <note>Commands will be executed with the working directory validated.</note>
</shell-execution>"""
    )


# --- LocalEnvironment Tests ---


async def test_environment_async_context_manager(tmp_path: Path) -> None:
    """Should provide file_operator and shell via properties."""
    async with LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path) as env:
        assert isinstance(env.file_operator, LocalFileOperator)
        assert isinstance(env.shell, LocalShell)


async def test_environment_file_operations(tmp_path: Path) -> None:
    """Should be able to perform file operations within context."""
    async with LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path) as env:
        await env.file_operator.write_file("test.txt", "hello")
        assert await env.file_operator.read_file("test.txt") == "hello"


async def test_environment_shell_execution(tmp_path: Path) -> None:
    """Should be able to execute shell commands within context."""
    async with LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path) as env:
        exit_code, stdout, _ = await env.shell.execute("echo hello")
        assert exit_code == 0
        assert "hello" in stdout


async def test_environment_tmp_dir(tmp_path: Path) -> None:
    """Should create and cleanup tmp_dir."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        assert env.tmp_dir is not None
        assert env.tmp_dir.exists()
        assert "pai_agent_" in env.tmp_dir.name
        saved_tmp = env.tmp_dir

    assert not saved_tmp.exists()


async def test_environment_tmp_dir_disabled(tmp_path: Path) -> None:
    """Should not create tmp_dir when disabled."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        enable_tmp_dir=False,
    ) as env:
        assert env.tmp_dir is None


async def test_environment_tmp_routing(tmp_path: Path) -> None:
    """Should route tmp path operations to tmp_file_operator."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        assert env.tmp_dir is not None

        # Write to tmp_dir via main file_operator
        tmp_file = env.tmp_dir / "data.txt"
        await env.file_operator.write_file(str(tmp_file), "tmp content")
        assert tmp_file.exists()
        assert tmp_file.read_text() == "tmp content"

        # Read from tmp_dir
        content = await env.file_operator.read_file(str(tmp_file))
        assert content == "tmp content"
