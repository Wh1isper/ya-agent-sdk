"""Tests for VirtualMount and VirtualLocalFileOperator.

These tests do NOT require Docker and should run in all CI environments.
"""

from pathlib import Path

import pytest
from ya_agent_environment import FileOperationError, PathNotAllowedError
from ya_agent_sdk.environment.local import VirtualLocalFileOperator, VirtualMount
from ya_agent_sdk.environment.virtual_path import normalize_virtual_path

# --- VirtualMount Tests ---


def test_virtual_mount_requires_absolute_virtual_path() -> None:
    """Should reject relative virtual_path."""
    with pytest.raises(ValueError, match="virtual_path must be absolute"):
        VirtualMount(Path("/tmp"), Path("workspace"))  # noqa: S108


def test_virtual_mount_frozen() -> None:
    """Should be immutable."""
    mount = VirtualMount(Path("/tmp"), Path("/workspace"))  # noqa: S108
    with pytest.raises(AttributeError):
        mount.host_path = Path("/other")  # type: ignore[misc]


# --- VirtualLocalFileOperator Tests ---


def test_virtual_file_operator_init(tmp_path: Path) -> None:
    """Should initialize with mounts list."""
    op = VirtualLocalFileOperator(
        mounts=[VirtualMount(tmp_path, Path("/workspace"))],
    )
    assert op._mounts[0].host_path == tmp_path
    assert op._mounts[0].virtual_path == normalize_virtual_path("/workspace")
    assert op._default_path == normalize_virtual_path("/workspace")


def test_virtual_file_operator_empty_mounts() -> None:
    """Should allow empty mounts list (empty folder mode)."""
    op = VirtualLocalFileOperator(mounts=[])
    assert op._mounts == []
    assert op._default_path is None


async def test_virtual_file_operator_empty_mounts_rejects_paths() -> None:
    """Should reject all non-tmp paths when mounts is empty."""
    op = VirtualLocalFileOperator(mounts=[])
    with pytest.raises(PathNotAllowedError):
        await op.read_file("test.txt")
    with pytest.raises(PathNotAllowedError):
        await op.read_file("/workspace/test.txt")


def test_virtual_file_operator_custom_default_path(tmp_path: Path) -> None:
    """Should allow custom default_virtual_path."""
    op = VirtualLocalFileOperator(
        mounts=[
            VirtualMount(tmp_path / "a", Path("/workspace/a")),
            VirtualMount(tmp_path / "b", Path("/workspace/b")),
        ],
        default_virtual_path=Path("/workspace/b"),
    )
    assert op._default_path == normalize_virtual_path("/workspace/b")


async def test_virtual_file_operator_read_write(tmp_path: Path) -> None:
    """Should read/write files using virtual paths mapped to host."""
    op = VirtualLocalFileOperator(
        mounts=[VirtualMount(tmp_path, Path("/workspace"))],
    )

    # Write using relative path
    await op.write_file("test.txt", "hello world")
    assert (tmp_path / "test.txt").read_text() == "hello world"

    # Read back using relative path
    content = await op.read_file("test.txt")
    assert content == "hello world"


async def test_virtual_file_operator_absolute_virtual_paths(tmp_path: Path) -> None:
    """Should handle absolute virtual paths correctly."""
    op = VirtualLocalFileOperator(
        mounts=[VirtualMount(tmp_path, Path("/workspace"))],
    )

    # Write using absolute virtual path
    await op.write_file("/workspace/abs_test.txt", "absolute")
    assert (tmp_path / "abs_test.txt").read_text() == "absolute"

    # Read using absolute virtual path
    content = await op.read_file("/workspace/abs_test.txt")
    assert content == "absolute"


async def test_virtual_file_operator_nested_paths(tmp_path: Path) -> None:
    """Should handle nested directory paths."""
    op = VirtualLocalFileOperator(
        mounts=[VirtualMount(tmp_path, Path("/workspace"))],
    )

    await op.mkdir("subdir", parents=True)
    await op.write_file("subdir/nested.txt", "nested content")
    assert (tmp_path / "subdir" / "nested.txt").read_text() == "nested content"

    content = await op.read_file("subdir/nested.txt")
    assert content == "nested content"


async def test_virtual_file_operator_path_not_allowed(tmp_path: Path) -> None:
    """Should reject paths outside virtual mounts."""
    op = VirtualLocalFileOperator(
        mounts=[VirtualMount(tmp_path, Path("/workspace"))],
    )

    with pytest.raises(PathNotAllowedError):
        await op.read_file("/etc/passwd")


async def test_virtual_file_operator_symlink_escape_blocked(tmp_path: Path) -> None:
    """Should block symlinks that escape the mount boundary."""
    mount_root = tmp_path / "workspace"
    mount_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("top secret")

    # Create a symlink inside the mount that points outside
    (mount_root / "escape_link").symlink_to(outside)

    op = VirtualLocalFileOperator(
        mounts=[VirtualMount(mount_root, Path("/workspace"))],
    )

    with pytest.raises(PathNotAllowedError, match="escapes mount boundary"):
        await op.read_file("/workspace/escape_link/secret.txt")


async def test_virtual_file_operator_copytree_preserves_symlinks(tmp_path: Path) -> None:
    """Should copy symlinks as links, not dereference them (prevents escape via nested symlink)."""
    mount_root = tmp_path / "workspace"
    mount_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("top secret")

    # Create a directory with a symlink pointing outside
    src_dir = mount_root / "src"
    src_dir.mkdir()
    (src_dir / "normal.txt").write_text("normal")
    (src_dir / "escape_link").symlink_to(outside)

    op = VirtualLocalFileOperator(
        mounts=[VirtualMount(mount_root, Path("/workspace"))],
    )

    # Copy the directory - symlinks should be preserved as links, not dereferenced
    await op.copy("/workspace/src", "/workspace/dst")

    dst_dir = mount_root / "dst"
    assert (dst_dir / "normal.txt").read_text() == "normal"
    # The symlink should be copied as a symlink (not followed)
    assert (dst_dir / "escape_link").is_symlink()


async def test_virtual_file_operator_exists(tmp_path: Path) -> None:
    """Should check existence through virtual paths."""
    op = VirtualLocalFileOperator(
        mounts=[VirtualMount(tmp_path, Path("/workspace"))],
    )

    assert not await op.exists("nope.txt")

    (tmp_path / "exists.txt").write_text("yes")
    assert await op.exists("exists.txt")
    assert await op.is_file("exists.txt")


async def test_virtual_file_operator_list_dir(tmp_path: Path) -> None:
    """Should list directory contents through virtual paths."""
    op = VirtualLocalFileOperator(
        mounts=[VirtualMount(tmp_path, Path("/workspace"))],
    )

    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    (tmp_path / "subdir").mkdir()

    entries = await op.list_dir(".")
    assert "a.txt" in entries
    assert "b.txt" in entries
    assert "subdir" in entries


async def test_virtual_file_operator_stat(tmp_path: Path) -> None:
    """Should return stat for virtual path."""
    op = VirtualLocalFileOperator(
        mounts=[VirtualMount(tmp_path, Path("/workspace"))],
    )

    (tmp_path / "stat_test.txt").write_text("content")
    stat = await op.stat("stat_test.txt")
    assert stat["is_file"] is True
    assert stat["size"] == 7


async def test_virtual_file_operator_move_copy(tmp_path: Path) -> None:
    """Should move and copy files through virtual paths."""
    op = VirtualLocalFileOperator(
        mounts=[VirtualMount(tmp_path, Path("/workspace"))],
    )

    await op.write_file("original.txt", "data")

    # Copy
    await op.copy("original.txt", "copied.txt")
    assert (tmp_path / "copied.txt").read_text() == "data"

    # Move
    await op.move("copied.txt", "moved.txt")
    assert not (tmp_path / "copied.txt").exists()
    assert (tmp_path / "moved.txt").read_text() == "data"


async def test_virtual_file_operator_delete(tmp_path: Path) -> None:
    """Should delete files through virtual paths."""
    op = VirtualLocalFileOperator(
        mounts=[VirtualMount(tmp_path, Path("/workspace"))],
    )

    (tmp_path / "to_delete.txt").write_text("bye")
    assert await op.exists("to_delete.txt")

    await op.delete("to_delete.txt")
    assert not await op.exists("to_delete.txt")


async def test_virtual_file_operator_file_not_found(tmp_path: Path) -> None:
    """Should raise FileOperationError for missing files."""
    op = VirtualLocalFileOperator(
        mounts=[VirtualMount(tmp_path, Path("/workspace"))],
    )

    with pytest.raises(FileOperationError, match="file not found"):
        await op.read_file("nonexistent.txt")


async def test_virtual_file_operator_context_instructions(tmp_path: Path) -> None:
    """Should generate instructions with virtual paths."""
    op = VirtualLocalFileOperator(
        mounts=[VirtualMount(tmp_path, Path("/workspace"))],
    )

    # Create some files so tree has content
    (tmp_path / "test.py").write_text("pass")

    instructions = await op.get_context_instructions()
    assert instructions is not None
    assert "/workspace" in instructions
    # Should NOT contain host path
    assert str(tmp_path) not in instructions


# --- Multi-mount Tests ---


async def test_virtual_file_operator_multi_mount(tmp_path: Path) -> None:
    """Should route operations to correct mount based on longest prefix."""
    host_a = tmp_path / "project"
    host_b = tmp_path / "config"
    host_a.mkdir()
    host_b.mkdir()

    op = VirtualLocalFileOperator(
        mounts=[
            VirtualMount(host_a, Path("/workspace/project")),
            VirtualMount(host_b, Path("/workspace/config")),
        ],
        default_virtual_path=Path("/workspace/project"),
    )

    # Write to project mount (relative resolves against default)
    await op.write_file("main.py", "print('hello')")
    assert (host_a / "main.py").read_text() == "print('hello')"
    assert not (host_b / "main.py").exists()

    # Write to config mount (absolute path)
    await op.write_file("/workspace/config/settings.json", '{"key": "val"}')
    assert (host_b / "settings.json").read_text() == '{"key": "val"}'
    assert not (host_a / "settings.json").exists()

    # Read back from both mounts
    assert await op.read_file("main.py") == "print('hello')"
    assert await op.read_file("/workspace/config/settings.json") == '{"key": "val"}'


async def test_virtual_file_operator_multi_mount_path_isolation(tmp_path: Path) -> None:
    """Should reject paths outside all mounts."""
    host_a = tmp_path / "a"
    host_a.mkdir()

    op = VirtualLocalFileOperator(
        mounts=[VirtualMount(host_a, Path("/workspace/a"))],
        default_virtual_path=Path("/workspace/a"),
    )

    with pytest.raises(PathNotAllowedError):
        await op.read_file("/workspace/b/secret.txt")


async def test_virtual_file_operator_multi_mount_context_instructions(tmp_path: Path) -> None:
    """Should generate file trees for each mount."""
    host_a = tmp_path / "project"
    host_b = tmp_path / "config"
    host_a.mkdir()
    host_b.mkdir()
    (host_a / "main.py").write_text("pass")
    (host_b / "settings.json").write_text("{}")

    op = VirtualLocalFileOperator(
        mounts=[
            VirtualMount(host_a, Path("/workspace/project")),
            VirtualMount(host_b, Path("/workspace/config")),
        ],
    )

    instructions = await op.get_context_instructions()
    assert instructions is not None
    assert "/workspace/project" in instructions
    assert "/workspace/config" in instructions
    assert "main.py" in instructions
    assert "settings.json" in instructions
    # Should NOT contain host paths
    assert str(host_a) not in instructions
    assert str(host_b) not in instructions


async def test_virtual_file_operator_multi_mount_longest_prefix(tmp_path: Path) -> None:
    """Should match longest virtual prefix for nested mounts."""
    host_root = tmp_path / "root"
    host_nested = tmp_path / "nested"
    host_root.mkdir()
    host_nested.mkdir()

    op = VirtualLocalFileOperator(
        mounts=[
            VirtualMount(host_root, Path("/workspace")),
            VirtualMount(host_nested, Path("/workspace/special")),
        ],
    )

    # /workspace/special/file.txt should go to host_nested, not host_root/special/
    await op.write_file("/workspace/special/file.txt", "nested mount")
    assert (host_nested / "file.txt").read_text() == "nested mount"
    assert not (host_root / "special").exists()

    # /workspace/other.txt should go to host_root
    await op.write_file("/workspace/other.txt", "root mount")
    assert (host_root / "other.txt").read_text() == "root mount"
