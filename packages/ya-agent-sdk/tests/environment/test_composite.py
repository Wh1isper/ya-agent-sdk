"""Tests for CompositeFileOperator."""

import tempfile
from pathlib import Path

import pytest
from ya_agent_environment import FileOperator
from ya_agent_sdk.environment.composite import CompositeFileOperator, Mount, MountBackend
from ya_agent_sdk.environment.local import LocalFileOperator
from ya_agent_sdk.environment.virtual_path import normalize_virtual_path


@pytest.fixture
def mount_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Create two separate directories to act as mount backends."""
    workspace = tmp_path / "workspace_host"
    workspace.mkdir()
    remote = tmp_path / "remote_host"
    remote.mkdir()
    return workspace, remote


@pytest.fixture
def composite_op(mount_dirs: tuple[Path, Path]) -> CompositeFileOperator:
    """Create a CompositeFileOperator with two local mounts."""
    workspace, remote = mount_dirs

    workspace_op = LocalFileOperator(default_path=workspace, allowed_paths=[workspace])
    remote_op = LocalFileOperator(default_path=remote, allowed_paths=[remote])

    return CompositeFileOperator(
        mounts=[
            Mount(virtual_path=Path("/workspace"), backend=workspace_op, label="Agent workspace"),
            Mount(virtual_path=Path("/mnt/pc"), backend=remote_op, label="User PC"),
        ],
        default_mount=Path("/workspace"),
    )


@pytest.fixture
def readonly_composite_op(mount_dirs: tuple[Path, Path]) -> CompositeFileOperator:
    """Create a CompositeFileOperator with a read-only mount."""
    workspace, remote = mount_dirs

    workspace_op = LocalFileOperator(default_path=workspace, allowed_paths=[workspace])
    remote_op = LocalFileOperator(default_path=remote, allowed_paths=[remote])

    return CompositeFileOperator(
        mounts=[
            Mount(virtual_path=Path("/workspace"), backend=workspace_op, label="Agent workspace"),
            Mount(virtual_path=Path("/mnt/pc"), backend=remote_op, label="User PC", read_only=True),
        ],
        default_mount=Path("/workspace"),
    )


# --- Mount validation ---


def test_mount_requires_absolute_path(tmp_path: Path):
    """Mount virtual_path must be absolute."""
    op = LocalFileOperator(default_path=tmp_path)
    with pytest.raises(ValueError, match="absolute"):
        Mount(virtual_path=Path("relative/path"), backend=op)


def test_composite_requires_at_least_one_mount():
    """CompositeFileOperator requires at least one mount."""
    with pytest.raises(ValueError, match="At least one mount"):
        CompositeFileOperator(mounts=[])


# --- Basic routing ---


@pytest.mark.anyio
async def test_write_and_read_default_mount(composite_op: CompositeFileOperator, mount_dirs: tuple[Path, Path]):
    """Relative paths route to the default mount."""
    workspace, _ = mount_dirs

    await composite_op.write_file("hello.txt", "world")
    content = await composite_op.read_file("hello.txt")
    assert content == "world"

    # Verify it's actually on the workspace host directory
    assert (workspace / "hello.txt").read_text() == "world"


@pytest.mark.anyio
async def test_write_and_read_absolute_default_mount(
    composite_op: CompositeFileOperator, mount_dirs: tuple[Path, Path]
):
    """Absolute paths under default mount work correctly."""
    workspace, _ = mount_dirs

    await composite_op.write_file("/workspace/test.py", "print('hi')")
    content = await composite_op.read_file("/workspace/test.py")
    assert content == "print('hi')"
    assert (workspace / "test.py").read_text() == "print('hi')"


@pytest.mark.anyio
async def test_write_and_read_second_mount(composite_op: CompositeFileOperator, mount_dirs: tuple[Path, Path]):
    """Absolute paths under second mount route correctly."""
    _, remote = mount_dirs

    await composite_op.write_file("/mnt/pc/data.txt", "remote data")
    content = await composite_op.read_file("/mnt/pc/data.txt")
    assert content == "remote data"

    # Verify it's on the remote host directory
    assert (remote / "data.txt").read_text() == "remote data"


@pytest.mark.anyio
async def test_routing_isolation(composite_op: CompositeFileOperator, mount_dirs: tuple[Path, Path]):
    """Files written to one mount don't appear on the other."""
    workspace, remote = mount_dirs

    await composite_op.write_file("/workspace/only_here.txt", "workspace")
    await composite_op.write_file("/mnt/pc/only_there.txt", "remote")

    # Files exist on correct backends
    assert await composite_op.exists("/workspace/only_here.txt")
    assert await composite_op.exists("/mnt/pc/only_there.txt")

    # Files don't exist on wrong backends
    assert not await composite_op.exists("/mnt/pc/only_here.txt")
    assert not await composite_op.exists("/workspace/only_there.txt")


# --- Path not allowed ---


@pytest.mark.anyio
async def test_path_outside_all_mounts_raises(tmp_path: Path):
    """Paths not under any mount raise PathNotAllowedError."""
    from ya_agent_environment import PathNotAllowedError

    op = LocalFileOperator(default_path=tmp_path)
    composite = CompositeFileOperator(
        mounts=[Mount(virtual_path=Path("/workspace"), backend=op)],
    )

    with pytest.raises(PathNotAllowedError):
        await composite.read_file("/other/path/file.txt")


# --- Directory operations ---


@pytest.mark.anyio
async def test_mkdir_and_list_dir(composite_op: CompositeFileOperator):
    """mkdir and list_dir work through routing."""
    await composite_op.mkdir("/workspace/subdir", parents=True)
    await composite_op.write_file("/workspace/subdir/a.txt", "a")
    await composite_op.write_file("/workspace/subdir/b.txt", "b")

    entries = await composite_op.list_dir("/workspace/subdir")
    assert sorted(entries) == ["a.txt", "b.txt"]


@pytest.mark.anyio
async def test_list_dir_with_types(composite_op: CompositeFileOperator):
    """list_dir_with_types correctly identifies files and directories."""
    await composite_op.mkdir("/workspace/project/src", parents=True)
    await composite_op.write_file("/workspace/project/README.md", "# Hello")

    entries = await composite_op.list_dir_with_types("/workspace/project")
    entries_dict = dict(entries)
    assert entries_dict["src"] is True
    assert entries_dict["README.md"] is False


# --- File metadata ---


@pytest.mark.anyio
async def test_stat(composite_op: CompositeFileOperator):
    """stat returns correct information through routing."""
    await composite_op.write_file("/mnt/pc/info.txt", "some content")
    stat = await composite_op.stat("/mnt/pc/info.txt")
    assert stat["is_file"] is True
    assert stat["is_dir"] is False
    assert stat["size"] == len("some content")


@pytest.mark.anyio
async def test_is_file_and_is_dir(composite_op: CompositeFileOperator):
    """is_file and is_dir work through routing."""
    await composite_op.mkdir("/workspace/mydir")
    await composite_op.write_file("/workspace/myfile.txt", "content")

    assert await composite_op.is_dir("/workspace/mydir")
    assert not await composite_op.is_file("/workspace/mydir")
    assert await composite_op.is_file("/workspace/myfile.txt")
    assert not await composite_op.is_dir("/workspace/myfile.txt")


# --- Read-only mount ---


@pytest.mark.anyio
async def test_read_only_mount_allows_read(readonly_composite_op: CompositeFileOperator, mount_dirs: tuple[Path, Path]):
    """Read operations work on read-only mounts."""
    _, remote = mount_dirs

    # Pre-populate the remote directory
    (remote / "existing.txt").write_text("pre-existing")

    content = await readonly_composite_op.read_file("/mnt/pc/existing.txt")
    assert content == "pre-existing"


@pytest.mark.anyio
async def test_read_only_mount_blocks_write(readonly_composite_op: CompositeFileOperator):
    """Write operations on read-only mounts raise FileOperationError."""
    from ya_agent_environment import FileOperationError

    with pytest.raises(FileOperationError, match="read-only"):
        await readonly_composite_op.write_file("/mnt/pc/new.txt", "blocked")


@pytest.mark.anyio
async def test_read_only_mount_blocks_delete(
    readonly_composite_op: CompositeFileOperator, mount_dirs: tuple[Path, Path]
):
    """Delete operations on read-only mounts raise FileOperationError."""
    from ya_agent_environment import FileOperationError

    _, remote = mount_dirs
    (remote / "file.txt").write_text("data")

    with pytest.raises(FileOperationError, match="read-only"):
        await readonly_composite_op.delete("/mnt/pc/file.txt")


@pytest.mark.anyio
async def test_read_only_mount_blocks_mkdir(readonly_composite_op: CompositeFileOperator):
    """mkdir on read-only mounts raises FileOperationError."""
    from ya_agent_environment import FileOperationError

    with pytest.raises(FileOperationError, match="read-only"):
        await readonly_composite_op.mkdir("/mnt/pc/newdir")


# --- Cross-mount operations ---


@pytest.mark.anyio
async def test_cross_mount_copy(composite_op: CompositeFileOperator):
    """Copying files between mounts uses streaming."""
    await composite_op.write_file("/mnt/pc/source.txt", "cross-mount data")
    await composite_op.copy("/mnt/pc/source.txt", "/workspace/copied.txt")

    # Source still exists
    assert await composite_op.exists("/mnt/pc/source.txt")

    # Destination has correct content
    content = await composite_op.read_file("/workspace/copied.txt")
    assert content == "cross-mount data"


@pytest.mark.anyio
async def test_cross_mount_move(composite_op: CompositeFileOperator):
    """Moving files between mounts streams then deletes source."""
    await composite_op.write_file("/mnt/pc/moveme.txt", "moving data")
    await composite_op.move("/mnt/pc/moveme.txt", "/workspace/moved.txt")

    # Source is gone
    assert not await composite_op.exists("/mnt/pc/moveme.txt")

    # Destination has correct content
    content = await composite_op.read_file("/workspace/moved.txt")
    assert content == "moving data"


@pytest.mark.anyio
async def test_cross_mount_directory_copy_raises(composite_op: CompositeFileOperator):
    """Copying directories across mounts raises a clear error."""
    from ya_agent_environment import FileOperationError

    await composite_op.mkdir("/mnt/pc/mydir")
    await composite_op.write_file("/mnt/pc/mydir/file.txt", "content")

    with pytest.raises(FileOperationError, match="cross-mount directory copy is not supported"):
        await composite_op.copy("/mnt/pc/mydir", "/workspace/mydir")


@pytest.mark.anyio
async def test_cross_mount_directory_move_raises(composite_op: CompositeFileOperator):
    """Moving directories across mounts raises a clear error."""
    from ya_agent_environment import FileOperationError

    await composite_op.mkdir("/mnt/pc/movedir")
    await composite_op.write_file("/mnt/pc/movedir/file.txt", "content")

    with pytest.raises(FileOperationError, match="cross-mount directory move is not supported"):
        await composite_op.move("/mnt/pc/movedir", "/workspace/movedir")


@pytest.mark.anyio
async def test_same_mount_copy(composite_op: CompositeFileOperator):
    """Copying within same mount delegates directly to backend."""
    await composite_op.write_file("/workspace/original.txt", "same mount")
    await composite_op.copy("/workspace/original.txt", "/workspace/duplicate.txt")

    assert await composite_op.read_file("/workspace/duplicate.txt") == "same mount"
    assert await composite_op.exists("/workspace/original.txt")


@pytest.mark.anyio
async def test_same_mount_move(composite_op: CompositeFileOperator):
    """Moving within same mount delegates directly to backend."""
    await composite_op.write_file("/workspace/src.txt", "move within")
    await composite_op.move("/workspace/src.txt", "/workspace/dst.txt")

    assert not await composite_op.exists("/workspace/src.txt")
    assert await composite_op.read_file("/workspace/dst.txt") == "move within"


# --- Glob ---


@pytest.mark.anyio
async def test_glob_relative_pattern(composite_op: CompositeFileOperator):
    """Relative glob patterns search the default mount."""
    await composite_op.write_file("/workspace/a.py", "")
    await composite_op.write_file("/workspace/b.txt", "")
    await composite_op.write_file("/mnt/pc/c.py", "")

    results = await composite_op.glob("*.py")
    assert "a.py" in results
    assert "b.txt" not in results
    # Remote mount files should NOT appear in relative glob
    assert "c.py" not in results


@pytest.mark.anyio
async def test_glob_absolute_pattern(composite_op: CompositeFileOperator):
    """Absolute glob patterns search the specified mount."""
    await composite_op.write_file("/mnt/pc/x.py", "")
    await composite_op.write_file("/mnt/pc/y.txt", "")

    results = await composite_op.glob("/mnt/pc/*.py")
    assert any("x.py" in r for r in results)
    assert not any("y.txt" in r for r in results)


# --- Append ---


@pytest.mark.anyio
async def test_append_file(composite_op: CompositeFileOperator):
    """append_file works through routing."""
    await composite_op.write_file("/workspace/log.txt", "line1\n")
    await composite_op.append_file("/workspace/log.txt", "line2\n")

    content = await composite_op.read_file("/workspace/log.txt")
    assert content == "line1\nline2\n"


# --- Read with offset/length ---


@pytest.mark.anyio
async def test_read_file_with_offset_and_length(composite_op: CompositeFileOperator):
    """read_file supports offset and length parameters."""
    await composite_op.write_file("/workspace/data.txt", "0123456789")

    content = await composite_op.read_file("/workspace/data.txt", offset=3, length=4)
    assert content == "3456"


@pytest.mark.anyio
async def test_read_bytes(composite_op: CompositeFileOperator):
    """read_bytes works through routing."""
    await composite_op.write_file("/workspace/binary.bin", b"\x00\x01\x02\x03")

    data = await composite_op.read_bytes("/workspace/binary.bin")
    assert data == b"\x00\x01\x02\x03"


# --- Longest prefix matching ---


@pytest.mark.anyio
async def test_longest_prefix_mount_matching(tmp_path: Path):
    """When mounts overlap, the longest prefix wins."""
    general = tmp_path / "general"
    general.mkdir()
    specific = tmp_path / "specific"
    specific.mkdir()

    general_op = LocalFileOperator(default_path=general, allowed_paths=[general])
    specific_op = LocalFileOperator(default_path=specific, allowed_paths=[specific])

    composite = CompositeFileOperator(
        mounts=[
            Mount(virtual_path=Path("/data"), backend=general_op, label="General"),
            Mount(virtual_path=Path("/data/special"), backend=specific_op, label="Special"),
        ],
        default_mount=Path("/data"),
    )

    await composite.write_file("/data/general.txt", "general")
    await composite.write_file("/data/special/specific.txt", "specific")

    # Verify routing: general.txt goes to general backend
    assert (general / "general.txt").read_text() == "general"
    assert not (specific / "general.txt").exists()

    # specific.txt goes to specific backend (longest prefix match)
    assert (specific / "specific.txt").read_text() == "specific"
    assert not (general / "special" / "specific.txt").exists()


# --- Context instructions ---


@pytest.mark.anyio
async def test_get_context_instructions(composite_op: CompositeFileOperator):
    """get_context_instructions includes mount information."""
    await composite_op.write_file("/workspace/main.py", "print('hello')")

    instructions = await composite_op.get_context_instructions()
    assert instructions is not None
    assert "/workspace" in instructions
    assert "/mnt/pc" in instructions
    assert "Agent workspace" in instructions
    assert "User PC" in instructions


@pytest.mark.anyio
async def test_get_context_instructions_with_read_only(readonly_composite_op: CompositeFileOperator):
    """Context instructions show read-only status."""
    instructions = await readonly_composite_op.get_context_instructions()
    assert instructions is not None
    assert 'read-only="true"' in instructions


# --- Close ---


@pytest.mark.anyio
async def test_close_calls_backend_close(mount_dirs: tuple[Path, Path]):
    """close() cleans up all backend operators."""
    workspace, remote = mount_dirs

    workspace_op = LocalFileOperator(default_path=workspace, allowed_paths=[workspace])
    remote_op = LocalFileOperator(default_path=remote, allowed_paths=[remote])

    composite = CompositeFileOperator(
        mounts=[
            Mount(virtual_path=Path("/workspace"), backend=workspace_op),
            Mount(virtual_path=Path("/mnt/pc"), backend=remote_op),
        ],
    )

    # Should not raise
    await composite.close()


# --- Tmp file handling ---


@pytest.mark.anyio
async def test_tmp_file_operations(mount_dirs: tuple[Path, Path]):
    """Tmp files are handled by base class tmp routing, not by mounts."""
    workspace, remote = mount_dirs

    with tempfile.TemporaryDirectory(prefix="ya_test_") as tmp_dir:
        # Resolve to handle macOS /var -> /private/var symlink
        tmp_dir_resolved = Path(tmp_dir).resolve()

        workspace_op = LocalFileOperator(default_path=workspace, allowed_paths=[workspace])
        remote_op = LocalFileOperator(default_path=remote, allowed_paths=[remote])

        composite = CompositeFileOperator(
            mounts=[
                Mount(virtual_path=Path("/workspace"), backend=workspace_op),
                Mount(virtual_path=Path("/mnt/pc"), backend=remote_op),
            ],
            default_mount=Path("/workspace"),
            tmp_dir=tmp_dir_resolved,
        )

        # Write to tmp (use resolved path for consistency)
        tmp_file = str(tmp_dir_resolved / "test_output.txt")
        await composite.write_file(tmp_file, "tmp content")
        content = await composite.read_file(tmp_file)
        assert content == "tmp content"

        # Verify it's actually in the tmp directory, not in any mount
        assert Path(tmp_file).read_text() == "tmp content"
        assert not (workspace / "test_output.txt").exists()
        assert not (remote / "test_output.txt").exists()


# --- Mounts property ---


def test_mounts_property(composite_op: CompositeFileOperator):
    """mounts property returns a copy of the mount list."""
    mounts = composite_op.mounts
    assert len(mounts) == 2
    assert mounts[0].virtual_path == normalize_virtual_path("/workspace")
    assert mounts[1].virtual_path == normalize_virtual_path("/mnt/pc")

    # Verify it's a copy
    mounts.pop()
    assert len(composite_op.mounts) == 2


# --- MountBackend availability ---


class OfflineMountBackend(MountBackend):
    """Test backend that simulates an offline remote mount."""

    def __init__(self, file_op: FileOperator, available: bool = False):
        self._file_op = file_op
        self._available = available
        self._cached_tree: str | None = None

    @property
    def file_operator(self) -> FileOperator:
        return self._file_op

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def status(self) -> str:
        return "online" if self._available else "offline"

    @property
    def status_detail(self) -> str | None:
        return None if self._available else "last seen 2 minutes ago"

    @property
    def cached_file_tree(self) -> str | None:
        return self._cached_tree

    def set_available(self, available: bool) -> None:
        self._available = available

    def set_cached_tree(self, tree: str) -> None:
        self._cached_tree = tree


@pytest.mark.anyio
async def test_unavailable_backend_blocks_read(tmp_path: Path):
    """Operations on unavailable backends raise FileOperationError."""
    from ya_agent_environment import FileOperationError

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    remote = tmp_path / "remote"
    remote.mkdir()
    (remote / "file.txt").write_text("data")

    workspace_op = LocalFileOperator(default_path=workspace, allowed_paths=[workspace])
    remote_op = LocalFileOperator(default_path=remote, allowed_paths=[remote])
    offline_backend = OfflineMountBackend(remote_op, available=False)

    composite = CompositeFileOperator(
        mounts=[
            Mount(virtual_path=Path("/workspace"), backend=workspace_op),
            Mount(virtual_path=Path("/mnt/pc"), backend=offline_backend),
        ],
    )

    with pytest.raises(FileOperationError, match="offline"):
        await composite.read_file("/mnt/pc/file.txt")


@pytest.mark.anyio
async def test_unavailable_backend_error_includes_detail(tmp_path: Path):
    """Error message includes status_detail when available."""
    from ya_agent_environment import FileOperationError

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    remote = tmp_path / "remote"
    remote.mkdir()

    workspace_op = LocalFileOperator(default_path=workspace, allowed_paths=[workspace])
    remote_op = LocalFileOperator(default_path=remote, allowed_paths=[remote])
    offline_backend = OfflineMountBackend(remote_op, available=False)

    composite = CompositeFileOperator(
        mounts=[
            Mount(virtual_path=Path("/workspace"), backend=workspace_op),
            Mount(virtual_path=Path("/mnt/pc"), backend=offline_backend, label="User PC"),
        ],
    )

    with pytest.raises(FileOperationError, match="last seen 2 minutes ago"):
        await composite.read_file("/mnt/pc/anything.txt")


@pytest.mark.anyio
async def test_backend_comes_online(tmp_path: Path):
    """Operations succeed when backend becomes available again."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    remote = tmp_path / "remote"
    remote.mkdir()
    (remote / "data.txt").write_text("hello from PC")

    workspace_op = LocalFileOperator(default_path=workspace, allowed_paths=[workspace])
    remote_op = LocalFileOperator(default_path=remote, allowed_paths=[remote])
    backend = OfflineMountBackend(remote_op, available=False)

    composite = CompositeFileOperator(
        mounts=[
            Mount(virtual_path=Path("/workspace"), backend=workspace_op),
            Mount(virtual_path=Path("/mnt/pc"), backend=backend),
        ],
    )

    # Offline: should fail
    from ya_agent_environment import FileOperationError

    with pytest.raises(FileOperationError):
        await composite.read_file("/mnt/pc/data.txt")

    # Come online: should succeed
    backend.set_available(True)
    content = await composite.read_file("/mnt/pc/data.txt")
    assert content == "hello from PC"


@pytest.mark.anyio
async def test_context_instructions_shows_offline_status(tmp_path: Path):
    """Context instructions reflect backend status accurately."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "main.py").write_text("print('hi')")
    remote = tmp_path / "remote"
    remote.mkdir()

    workspace_op = LocalFileOperator(default_path=workspace, allowed_paths=[workspace])
    remote_op = LocalFileOperator(default_path=remote, allowed_paths=[remote])
    offline_backend = OfflineMountBackend(remote_op, available=False)
    offline_backend.set_cached_tree("project/\n  src/\n    main.py")

    composite = CompositeFileOperator(
        mounts=[
            Mount(virtual_path=Path("/workspace"), backend=workspace_op, label="Agent workspace"),
            Mount(virtual_path=Path("/mnt/pc"), backend=offline_backend, label="User PC"),
        ],
    )

    instructions = await composite.get_context_instructions()
    assert instructions is not None

    # Workspace should have live file tree and online status
    assert 'status="online"' in instructions
    assert "main.py" in instructions

    # Remote should show offline status with cached tree
    assert 'status="offline"' in instructions
    assert "currently offline" in instructions
    assert "cached-file-tree" in instructions
    assert "project/" in instructions


@pytest.mark.anyio
async def test_context_instructions_offline_without_cache(tmp_path: Path):
    """Context instructions handle offline backend with no cached tree."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    remote = tmp_path / "remote"
    remote.mkdir()

    workspace_op = LocalFileOperator(default_path=workspace, allowed_paths=[workspace])
    remote_op = LocalFileOperator(default_path=remote, allowed_paths=[remote])
    offline_backend = OfflineMountBackend(remote_op, available=False)
    # No cached tree set

    composite = CompositeFileOperator(
        mounts=[
            Mount(virtual_path=Path("/workspace"), backend=workspace_op),
            Mount(virtual_path=Path("/mnt/pc"), backend=offline_backend),
        ],
    )

    instructions = await composite.get_context_instructions()
    assert instructions is not None
    assert "currently offline" in instructions
    # No cached-file-tree element
    assert "cached-file-tree" not in instructions


@pytest.mark.anyio
async def test_mount_auto_wraps_file_operator(tmp_path: Path):
    """Plain FileOperator is auto-wrapped in LocalMountBackend."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    workspace_op = LocalFileOperator(default_path=workspace, allowed_paths=[workspace])
    mount = Mount(virtual_path=Path("/workspace"), backend=workspace_op)

    # Should auto-wrap
    assert mount.mount_backend.is_available is True
    assert mount.mount_backend.status == "online"
    assert mount.file_operator is workspace_op
