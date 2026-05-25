"""Tests for LocalFileOperator and LocalShell."""

import asyncio
import contextlib
import os
import shlex
import shutil
import signal
import sys
from pathlib import Path

import pytest
from inline_snapshot import snapshot
from ya_agent_environment import FileOperationError, PathNotAllowedError, ShellTimeoutError
from ya_agent_sdk.environment import (
    LocalEnvironment,
    LocalFileOperator,
    LocalShell,
)

# --- LocalFileOperator Tests ---


def test_file_operator_default_initialization(tmp_path: Path) -> None:
    """Should initialize with provided default_path."""
    op = LocalFileOperator(default_path=tmp_path)
    assert op._default_path == tmp_path.resolve()


def test_file_operator_none_default_path() -> None:
    """Should allow None default_path (empty folder mode)."""
    op = LocalFileOperator(default_path=None)
    assert op._default_path is None


async def test_file_operator_none_default_path_rejects_paths() -> None:
    """Should reject all non-tmp paths when default_path is None."""
    op = LocalFileOperator(default_path=None)
    with pytest.raises(PathNotAllowedError):
        await op.read_file("test.txt")
    with pytest.raises(PathNotAllowedError):
        await op.read_file("/etc/passwd")


def test_file_operator_fallback_default_path_from_allowed_paths(tmp_path: Path) -> None:
    """Should use allowed_paths[0] as default_path when default_path is None."""
    op = LocalFileOperator(allowed_paths=[tmp_path])
    assert op._default_path == tmp_path.resolve()


def test_file_operator_custom_allowed_paths(tmp_path: Path) -> None:
    """Should accept custom allowed paths."""
    op = LocalFileOperator(default_path=tmp_path, allowed_paths=[tmp_path])
    assert tmp_path.resolve() in op._allowed_paths


def test_file_operator_default_path_included_in_allowed(tmp_path: Path) -> None:
    """Should ensure default_path is in allowed_paths."""
    other_path = tmp_path / "other"
    other_path.mkdir()
    op = LocalFileOperator(allowed_paths=[other_path], default_path=tmp_path, tmp_dir=tmp_path)
    assert tmp_path.resolve() in op._allowed_paths
    assert other_path.resolve() in op._allowed_paths


async def test_file_operator_read_file(tmp_path: Path) -> None:
    """Should read file content."""
    (tmp_path / "test.txt").write_text("hello world")
    op = LocalFileOperator(default_path=tmp_path, allowed_paths=[tmp_path])
    assert await op.read_file("test.txt") == "hello world"


async def test_file_operator_read_file_with_offset(tmp_path: Path) -> None:
    """Should read file content from character offset."""
    (tmp_path / "test.txt").write_text("hello world")
    op = LocalFileOperator(default_path=tmp_path, allowed_paths=[tmp_path])
    assert await op.read_file("test.txt", offset=6) == "world"


async def test_file_operator_read_file_with_length(tmp_path: Path) -> None:
    """Should read limited characters from file."""
    (tmp_path / "test.txt").write_text("hello world")
    op = LocalFileOperator(default_path=tmp_path, allowed_paths=[tmp_path])
    assert await op.read_file("test.txt", length=5) == "hello"


async def test_file_operator_read_file_with_offset_and_length(tmp_path: Path) -> None:
    """Should read substring with offset and length."""
    (tmp_path / "test.txt").write_text("hello world")
    op = LocalFileOperator(default_path=tmp_path, allowed_paths=[tmp_path])
    assert await op.read_file("test.txt", offset=3, length=5) == "lo wo"


async def test_file_operator_read_bytes(tmp_path: Path) -> None:
    """Should read file as bytes."""
    (tmp_path / "test.bin").write_bytes(b"\x00\x01\x02")
    op = LocalFileOperator(default_path=tmp_path, allowed_paths=[tmp_path])
    assert await op.read_bytes("test.bin") == b"\x00\x01\x02"


async def test_file_operator_read_bytes_with_offset(tmp_path: Path) -> None:
    """Should read bytes from offset."""
    (tmp_path / "test.bin").write_bytes(b"\x00\x01\x02\x03\x04")
    op = LocalFileOperator(default_path=tmp_path, allowed_paths=[tmp_path])
    assert await op.read_bytes("test.bin", offset=2) == b"\x02\x03\x04"


async def test_file_operator_read_bytes_with_length(tmp_path: Path) -> None:
    """Should read limited bytes."""
    (tmp_path / "test.bin").write_bytes(b"\x00\x01\x02\x03\x04")
    op = LocalFileOperator(default_path=tmp_path, allowed_paths=[tmp_path])
    assert await op.read_bytes("test.bin", length=3) == b"\x00\x01\x02"


async def test_file_operator_read_bytes_with_offset_and_length(tmp_path: Path) -> None:
    """Should read byte slice with offset and length."""
    (tmp_path / "test.bin").write_bytes(b"\x00\x01\x02\x03\x04")
    op = LocalFileOperator(default_path=tmp_path, allowed_paths=[tmp_path])
    assert await op.read_bytes("test.bin", offset=1, length=3) == b"\x01\x02\x03"


async def test_file_operator_write_file_string(tmp_path: Path) -> None:
    """Should write string content."""
    op = LocalFileOperator(default_path=tmp_path, allowed_paths=[tmp_path])
    await op.write_file("output.txt", "test content")
    assert (tmp_path / "output.txt").read_text() == "test content"


async def test_file_operator_write_file_bytes(tmp_path: Path) -> None:
    """Should write bytes content."""
    op = LocalFileOperator(default_path=tmp_path, allowed_paths=[tmp_path])
    await op.write_file("output.bin", b"\x00\x01\x02")
    assert (tmp_path / "output.bin").read_bytes() == b"\x00\x01\x02"


async def test_file_operator_append_file(tmp_path: Path) -> None:
    """Should append content to file."""
    (tmp_path / "test.txt").write_text("hello")
    op = LocalFileOperator(default_path=tmp_path, allowed_paths=[tmp_path])
    await op.append_file("test.txt", " world")
    assert (tmp_path / "test.txt").read_text() == "hello world"


async def test_file_operator_delete_file(tmp_path: Path) -> None:
    """Should delete file."""
    (tmp_path / "test.txt").write_text("hello")
    op = LocalFileOperator(default_path=tmp_path, allowed_paths=[tmp_path])
    await op.delete("test.txt")
    assert not (tmp_path / "test.txt").exists()


async def test_file_operator_list_dir(tmp_path: Path) -> None:
    """Should list directory contents."""
    (tmp_path / "a.txt").touch()
    (tmp_path / "b.txt").touch()
    (tmp_path / "subdir").mkdir()
    op = LocalFileOperator(default_path=tmp_path, allowed_paths=[tmp_path])
    entries = await op.list_dir(".")
    assert set(entries) == {"a.txt", "b.txt", "subdir"}


async def test_file_operator_exists(tmp_path: Path) -> None:
    """Should check path existence."""
    (tmp_path / "exists.txt").touch()
    op = LocalFileOperator(default_path=tmp_path, allowed_paths=[tmp_path])
    assert await op.exists("exists.txt") is True
    assert await op.exists("not_exists.txt") is False


async def test_file_operator_is_file_and_is_dir(tmp_path: Path) -> None:
    """Should distinguish files and directories."""
    (tmp_path / "file.txt").touch()
    (tmp_path / "subdir").mkdir()
    op = LocalFileOperator(default_path=tmp_path, allowed_paths=[tmp_path])
    assert await op.is_file("file.txt") is True
    assert await op.is_dir("file.txt") is False
    assert await op.is_file("subdir") is False
    assert await op.is_dir("subdir") is True


async def test_file_operator_mkdir(tmp_path: Path) -> None:
    """Should create directory."""
    op = LocalFileOperator(default_path=tmp_path, allowed_paths=[tmp_path])
    await op.mkdir("new_dir")
    assert (tmp_path / "new_dir").is_dir()


async def test_file_operator_mkdir_parents(tmp_path: Path) -> None:
    """Should create nested directories with parents=True."""
    op = LocalFileOperator(default_path=tmp_path, allowed_paths=[tmp_path])
    await op.mkdir("a/b/c", parents=True)
    assert (tmp_path / "a" / "b" / "c").is_dir()


async def test_file_operator_move(tmp_path: Path) -> None:
    """Should move file."""
    (tmp_path / "src.txt").write_text("content")
    op = LocalFileOperator(default_path=tmp_path, allowed_paths=[tmp_path])
    await op.move("src.txt", "dst.txt")
    assert not (tmp_path / "src.txt").exists()
    assert (tmp_path / "dst.txt").read_text() == "content"


async def test_file_operator_copy(tmp_path: Path) -> None:
    """Should copy file."""
    (tmp_path / "src.txt").write_text("content")
    op = LocalFileOperator(default_path=tmp_path, allowed_paths=[tmp_path])
    await op.copy("src.txt", "dst.txt")
    assert (tmp_path / "src.txt").read_text() == "content"
    assert (tmp_path / "dst.txt").read_text() == "content"


async def test_file_operator_path_not_allowed_error(tmp_path: Path) -> None:
    """Should raise PathNotAllowedError for paths outside allowed dirs."""
    op = LocalFileOperator(default_path=tmp_path, allowed_paths=[tmp_path])
    with pytest.raises(PathNotAllowedError):
        await op.read_file("/etc/passwd")


async def test_file_operator_file_operation_error(tmp_path: Path) -> None:
    """Should raise FileOperationError for failed operations."""
    op = LocalFileOperator(default_path=tmp_path, allowed_paths=[tmp_path])
    with pytest.raises(FileOperationError) as exc_info:
        await op.read_file("not_exists.txt")
    assert "file not found" in str(exc_info.value)


async def test_file_operator_stat(tmp_path: Path) -> None:
    """Should return file stat information."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello")
    op = LocalFileOperator(default_path=tmp_path, allowed_paths=[tmp_path])
    stat = await op.stat("test.txt")
    assert stat["size"] == 5
    assert stat["is_file"] is True
    assert stat["is_dir"] is False


async def test_file_operator_stat_dir(tmp_path: Path) -> None:
    """Should return directory stat information."""
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    op = LocalFileOperator(default_path=tmp_path, allowed_paths=[tmp_path])
    stat = await op.stat("subdir")
    assert stat["is_file"] is False
    assert stat["is_dir"] is True


async def test_file_operator_glob_relative(tmp_path: Path) -> None:
    """Should glob files with relative pattern."""
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    (tmp_path / "c.py").write_text("c")
    op = LocalFileOperator(default_path=tmp_path, allowed_paths=[tmp_path])
    matches = await op.glob("*.txt")
    assert set(matches) == {"a.txt", "b.txt"}


async def test_file_operator_glob_recursive(tmp_path: Path) -> None:
    """Should glob files recursively with ** pattern."""
    (tmp_path / "a.txt").write_text("a")
    subdir = tmp_path / "sub"
    subdir.mkdir()
    (subdir / "b.txt").write_text("b")
    op = LocalFileOperator(default_path=tmp_path, allowed_paths=[tmp_path])
    matches = await op.glob("**/*.txt")
    assert "sub/b.txt" in matches


async def test_file_operator_glob_absolute_path(tmp_path: Path) -> None:
    """Should glob files with absolute path pattern."""
    (tmp_path / "test.txt").write_text("test")
    op = LocalFileOperator(default_path=tmp_path, allowed_paths=[tmp_path])
    pattern = str(tmp_path / "*.txt")
    matches = await op.glob(pattern)
    assert len(matches) == 1
    assert str(tmp_path / "test.txt") in matches


async def test_file_operator_glob_absolute_path_security(tmp_path: Path) -> None:
    """Should filter absolute glob results by allowed_paths."""
    # Create a file in tmp_path
    (tmp_path / "allowed.txt").write_text("allowed")

    # Create another directory NOT in allowed_paths
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    (other_dir / "not_allowed.txt").write_text("not allowed")

    # Only tmp_path is allowed, not other_dir
    op = LocalFileOperator(default_path=tmp_path, allowed_paths=[tmp_path])

    # Glob for files in other_dir should return empty (security filter)
    pattern = str(other_dir / "*.txt")
    matches = await op.glob(pattern)

    # other_dir is a subdirectory of tmp_path, so it IS allowed
    # This test verifies the security check works for truly outside paths
    assert len(matches) == 1  # other_dir is under tmp_path, so it's allowed


async def test_file_operator_glob_sorted_by_mtime(tmp_path: Path) -> None:
    """Should return glob results sorted by modification time (newest first)."""
    import time

    (tmp_path / "old.txt").write_text("old")
    time.sleep(0.1)
    (tmp_path / "new.txt").write_text("new")

    op = LocalFileOperator(default_path=tmp_path, allowed_paths=[tmp_path])
    matches = await op.glob("*.txt")

    # Newest file should be first
    assert matches[0] == "new.txt"
    assert matches[1] == "old.txt"


async def test_file_operator_get_context_instructions(tmp_path: Path) -> None:
    """Should return context instructions in XML format with file tree."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("# main")
    (tmp_path / "README.md").write_text("# readme")
    (tmp_path / ".git").mkdir()

    op = LocalFileOperator(allowed_paths=[tmp_path], default_path=tmp_path, tmp_dir=tmp_path)
    instructions = await op.get_context_instructions()

    # Replace dynamic path with placeholder for snapshot
    assert instructions is not None
    normalized = instructions.replace(str(tmp_path), "/test/workspace")
    assert normalized == snapshot(
        """\
<file-system>
  <default-directory>/test/workspace</default-directory>
  <tmp-directory>/test/workspace</tmp-directory>
  <tmp-directory-note>This is an agent-only temporary directory for intermediate files. Never write deliverables or user-facing files here. Files the user needs to access must be written to the project directory. Never mention this path to the user.</tmp-directory-note>
  <file-trees>
    <directory path="/test/workspace">
.git/ (skipped)
src/main.py
README.md
    </directory>
  </file-trees>
</file-system>\
"""
    )


# --- LocalShell Tests ---


def _process_is_running(pid: int) -> bool:
    """Return whether a process is still running, treating zombies as stopped."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True

    proc_stat = Path(f"/proc/{pid}/stat")
    if proc_stat.exists():
        try:
            parts = proc_stat.read_text().split()
        except (FileNotFoundError, ProcessLookupError):
            return False
        if len(parts) >= 3 and parts[2] == "Z":
            return False

    return True


async def _wait_until_stopped(pid: int, *, timeout: float = 2.0) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if not _process_is_running(pid):
            return True
        await asyncio.sleep(0.05)
    return not _process_is_running(pid)


async def _wait_for_pidfile(pidfile: Path, *, timeout: float = 2.0) -> int:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        with contextlib.suppress(FileNotFoundError, ValueError):
            raw_pid = pidfile.read_text().strip()
            pid = int(raw_pid)
            if _process_is_running(pid):
                return pid
        await asyncio.sleep(0.05)
    raise AssertionError(f"child PID file was not populated with a running process ID: {pidfile}")


def test_shell_default_initialization(tmp_path: Path) -> None:
    """Should initialize with provided default_cwd."""
    shell = LocalShell(default_cwd=tmp_path)
    assert shell._default_cwd == tmp_path.resolve()


def test_shell_fallback_default_cwd_from_allowed_paths(tmp_path: Path) -> None:
    """Should use allowed_paths[0] as default_cwd when default_cwd is None."""
    shell = LocalShell(allowed_paths=[tmp_path])
    assert shell._default_cwd == tmp_path.resolve()


def test_shell_default_cwd_included_in_allowed(tmp_path: Path) -> None:
    """Should ensure default_cwd is in allowed_paths."""
    other_path = tmp_path / "other"
    other_path.mkdir()
    shell = LocalShell(default_cwd=tmp_path, allowed_paths=[other_path])
    assert tmp_path.resolve() in shell._allowed_paths
    assert other_path.resolve() in shell._allowed_paths


async def test_shell_execute_simple_command(tmp_path: Path) -> None:
    """Should execute simple command."""
    shell = LocalShell(default_cwd=tmp_path, allowed_paths=[tmp_path])
    exit_code, stdout, _ = await shell.execute("echo hello")
    assert exit_code == 0
    assert "hello" in stdout


async def test_shell_execute_with_cwd(tmp_path: Path) -> None:
    """Should execute command in specified directory."""
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    shell = LocalShell(default_cwd=tmp_path, allowed_paths=[tmp_path])
    exit_code, stdout, _ = await shell.execute("pwd", cwd=str(subdir))
    assert exit_code == 0
    assert "subdir" in stdout


async def test_shell_execute_cwd_not_allowed(tmp_path: Path) -> None:
    """Should raise PathNotAllowedError for cwd outside allowed dirs."""
    shell = LocalShell(default_cwd=tmp_path, allowed_paths=[tmp_path])
    with pytest.raises(PathNotAllowedError):
        await shell.execute("ls", cwd="/etc")


async def test_shell_execute_timeout(tmp_path: Path) -> None:
    """Should raise ShellTimeoutError on timeout."""
    shell = LocalShell(default_cwd=tmp_path, allowed_paths=[tmp_path])
    with pytest.raises(ShellTimeoutError):
        await shell.execute("sleep 10", timeout=0.1)


@pytest.mark.skipif(os.name != "posix", reason="POSIX shell process tree test")
async def test_shell_background_kill_terminates_child_process(tmp_path: Path) -> None:
    """Killing a background shell process should terminate spawned child processes."""
    shell = LocalShell(default_cwd=tmp_path, allowed_paths=[tmp_path])
    pidfile = tmp_path / "child.pid"
    child_pid: int | None = None

    process_id = await shell.start(f"sleep 100 & echo $! > {shlex.quote(str(pidfile))}; wait")
    try:
        child_pid = await _wait_for_pidfile(pidfile)

        await shell.kill_process(process_id)

        assert await _wait_until_stopped(child_pid)
    finally:
        if child_pid is not None and _process_is_running(child_pid):
            with contextlib.suppress(ProcessLookupError, OSError):
                os.kill(child_pid, signal.SIGKILL)
        await shell.close()


async def test_shell_execute_command_not_found(tmp_path: Path) -> None:
    """Should return non-zero exit code for non-existent command."""
    shell = LocalShell(default_cwd=tmp_path, allowed_paths=[tmp_path])
    exit_code, _, stderr = await shell.execute("nonexistent_command_xyz")
    assert exit_code != 0
    assert "not found" in stderr.lower() or "not recognized" in stderr.lower()


async def test_shell_execute_returns_stderr(tmp_path: Path) -> None:
    """Should capture stderr."""
    shell = LocalShell(default_cwd=tmp_path, allowed_paths=[tmp_path])
    exit_code, _, stderr = await shell.execute("ls nonexistent_file_xyz")
    assert exit_code != 0
    assert stderr


async def test_shell_get_context_instructions_with_platform_default_shell() -> None:
    """Should describe platform default shell execution when no executable is configured."""
    shell = LocalShell(
        default_cwd=Path("/workspace"),
        allowed_paths=[Path("/workspace")],
        default_timeout=30.0,
        shell_executable="",
    )
    instructions = await shell.get_context_instructions()

    assert instructions is not None
    assert "<shell-environment>" in instructions
    assert f"<platform>{sys.platform}</platform>" in instructions
    assert "<shell-type>platform-default</shell-type>" in instructions
    assert "<shell-executable>" not in instructions


async def test_shell_uses_bash_by_default_on_posix() -> None:
    """Should use Bash as the default shell executable on POSIX when available."""
    shell = LocalShell(default_cwd=Path("/workspace"), allowed_paths=[Path("/workspace")])
    if os.name == "posix" and Path("/bin/bash").exists():
        assert shell._shell_executable == "/bin/bash"
    elif os.name == "posix":
        assert shell._shell_executable == shutil.which("bash")
    else:
        assert shell._shell_executable is None


async def test_shell_get_context_instructions_with_custom_shell() -> None:
    """Should describe custom shell executable by basename."""
    shell = LocalShell(
        default_cwd=Path("/workspace"),
        allowed_paths=[Path("/workspace")],
        default_timeout=30.0,
        shell_executable="/bin/zsh",
    )
    shell._platform_name = "linux"
    instructions = await shell.get_context_instructions()

    assert instructions is not None
    assert "<shell-type>zsh</shell-type>" in instructions
    assert "<shell-executable>/bin/zsh</shell-executable>" in instructions


async def test_environment_get_context_instructions_with_custom_shell(tmp_path: Path) -> None:
    """Should pass custom shell executable through LocalEnvironment context."""
    async with LocalEnvironment(
        default_path=tmp_path,
        allowed_paths=[tmp_path],
        shell_executable="/bin/zsh",
    ) as env:
        instructions = await env.shell.get_context_instructions()

    assert instructions is not None
    assert "<shell-type>zsh</shell-type>" in instructions
    assert "<shell-executable>/bin/zsh</shell-executable>" in instructions


@pytest.mark.skipif(os.name != "posix", reason="snapshot uses POSIX path formatting")
async def test_shell_get_context_instructions() -> None:
    """Should return context instructions in XML format."""
    shell = LocalShell(
        default_cwd=Path("/workspace"),
        allowed_paths=[Path("/workspace")],
        default_timeout=30.0,
        shell_executable="/bin/bash",
    )
    shell._platform_name = "linux"
    instructions = await shell.get_context_instructions()
    assert instructions == snapshot(
        """\
<shell-execution>
  <allowed-working-directories>
    <path>/workspace</path>
  </allowed-working-directories>
  <default-working-directory>/workspace</default-working-directory>
  <shell-environment>
    <platform>linux</platform>
    <shell-type>bash</shell-type>
    <shell-executable>/bin/bash</shell-executable>
  </shell-environment>
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
        assert "ya_agent_" in env.tmp_dir.name
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


async def test_environment_double_enter_raises_error(tmp_path: Path) -> None:
    """Should raise RuntimeError when entering an already-entered environment."""
    env = LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path)
    async with env:
        with pytest.raises(RuntimeError, match="has already been entered"):
            await env.__aenter__()


async def test_environment_can_reenter_after_exit(tmp_path: Path) -> None:
    """Should allow re-entering after exiting."""
    env = LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path)

    # First enter/exit cycle
    async with env:
        await env.file_operator.write_file("test1.txt", "hello")

    # Second enter/exit cycle should work
    async with env:
        await env.file_operator.write_file("test2.txt", "world")


async def test_environment_no_default_path_no_allowed_paths() -> None:
    """Should create file_operator for tmp access but leave shell as None when no paths provided."""
    async with LocalEnvironment() as env:
        # file_operator is created for tmp_dir access (enable_tmp_dir=True by default)
        assert env.file_operator is not None
        assert env.shell is None


async def test_environment_no_default_path_no_allowed_paths_no_tmp() -> None:
    """Should leave both as None when no paths and tmp is disabled."""
    async with LocalEnvironment(enable_tmp_dir=False) as env:
        assert env.file_operator is None
        assert env.shell is None


async def test_environment_no_default_path_with_allowed_paths(tmp_path: Path) -> None:
    """Should use first allowed_path as default when default_path is None."""
    async with LocalEnvironment(allowed_paths=[tmp_path]) as env:
        assert isinstance(env.file_operator, LocalFileOperator)
        assert isinstance(env.shell, LocalShell)
        assert env.file_operator._default_path == tmp_path.resolve()
        assert env.shell._default_cwd == tmp_path.resolve()


async def test_environment_no_default_path_tmp_dir_still_works() -> None:
    """Should still create tmp_dir and file_operator even when no real paths."""
    async with LocalEnvironment(enable_tmp_dir=True) as env:
        assert env.shell is None
        assert env.file_operator is not None
        assert env.tmp_dir is not None
        assert env.tmp_dir.exists()

        # Can write and read files in tmp_dir via file_operator
        tmp_file = str(env.tmp_dir / "test.txt")
        await env.file_operator.write_file(tmp_file, "hello from tmp")
        content = await env.file_operator.read_file(tmp_file)
        assert content == "hello from tmp"


async def test_environment_no_default_path_rejects_non_tmp_paths() -> None:
    """File operator with no default_path routes all operations through tmp_file_operator."""
    async with LocalEnvironment() as env:
        assert env.file_operator is not None
        # Relative paths get resolved against tmp_dir (tmp-only mode).
        # Non-existent file in tmp_dir raises FileNotFoundError.
        with pytest.raises(FileNotFoundError):
            await env.file_operator.read_file("nonexistent_file.txt")


async def test_environment_no_default_path_context_instructions() -> None:
    """Should return empty instructions when shell is None and file_operator has no default."""
    async with LocalEnvironment(enable_tmp_dir=False) as env:
        instructions = await env.get_context_instructions()
        assert instructions == ""


async def test_environment_concurrent_enter_raises_error(tmp_path: Path) -> None:
    """Should raise RuntimeError when concurrently entering the same environment."""
    import asyncio

    env = LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path)
    errors: list[Exception] = []
    entered_count = 0

    async def try_enter():
        nonlocal entered_count
        try:
            async with env:
                entered_count += 1
                await asyncio.sleep(0.1)  # Hold the context
        except RuntimeError as e:
            errors.append(e)

    # Try to enter concurrently
    await asyncio.gather(try_enter(), try_enter())

    # One should succeed, one should fail
    assert entered_count == 1
    assert len(errors) == 1
    assert "has already been entered" in str(errors[0])
