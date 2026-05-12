"""Tests for LocalTmpFileOperator and FileOperator."""

from collections.abc import AsyncIterator
from pathlib import Path

from ya_agent_environment import (
    DEFAULT_CHUNK_SIZE,
    FileOperator,
    FileStat,
    LocalTmpFileOperator,
)


async def test_local_tmp_file_operator_basic_operations(tmp_path: Path) -> None:
    """LocalTmpFileOperator should support basic file operations."""
    op = LocalTmpFileOperator(tmp_path)

    # Write and read text
    await op.write_file("test.txt", "Hello World")
    content = await op.read_file("test.txt")
    assert content == "Hello World"

    # Check exists
    assert await op.exists("test.txt")
    assert not await op.exists("nonexistent.txt")

    # Check is_file/is_dir
    assert await op.is_file("test.txt")
    assert not await op.is_dir("test.txt")


async def test_local_tmp_file_operator_read_bytes(tmp_path: Path) -> None:
    """LocalTmpFileOperator should support reading bytes."""
    op = LocalTmpFileOperator(tmp_path)

    # Write bytes
    await op.write_file("binary.bin", b"\x00\x01\x02\x03")
    content = await op.read_bytes("binary.bin")
    assert content == b"\x00\x01\x02\x03"


async def test_local_tmp_file_operator_read_with_offset(tmp_path: Path) -> None:
    """LocalTmpFileOperator should support offset and length."""
    op = LocalTmpFileOperator(tmp_path)

    await op.write_file("test.txt", "0123456789")

    # Read with offset
    content = await op.read_file("test.txt", offset=3)
    assert content == "3456789"

    # Read with length
    content = await op.read_file("test.txt", length=5)
    assert content == "01234"

    # Read with offset and length
    content = await op.read_file("test.txt", offset=2, length=4)
    assert content == "2345"


async def test_local_tmp_file_operator_append(tmp_path: Path) -> None:
    """LocalTmpFileOperator should support appending."""
    op = LocalTmpFileOperator(tmp_path)

    await op.write_file("test.txt", "Hello")
    await op.append_file("test.txt", " World")
    content = await op.read_file("test.txt")
    assert content == "Hello World"


async def test_local_tmp_file_operator_delete(tmp_path: Path) -> None:
    """LocalTmpFileOperator should support deleting files and dirs."""
    op = LocalTmpFileOperator(tmp_path)

    # Delete file
    await op.write_file("test.txt", "content")
    assert await op.exists("test.txt")
    await op.delete("test.txt")
    assert not await op.exists("test.txt")

    # Delete empty directory
    await op.mkdir("emptydir")
    assert await op.is_dir("emptydir")
    await op.delete("emptydir")
    assert not await op.exists("emptydir")


async def test_local_tmp_file_operator_mkdir(tmp_path: Path) -> None:
    """LocalTmpFileOperator should support creating directories."""
    op = LocalTmpFileOperator(tmp_path)

    await op.mkdir("newdir")
    assert await op.is_dir("newdir")

    # With parents
    await op.mkdir("a/b/c", parents=True)
    assert await op.is_dir("a/b/c")


async def test_local_tmp_file_operator_list_dir(tmp_path: Path) -> None:
    """LocalTmpFileOperator should list directory contents."""
    op = LocalTmpFileOperator(tmp_path)

    await op.write_file("a.txt", "a")
    await op.write_file("b.txt", "b")
    await op.mkdir("subdir")

    entries = await op.list_dir(".")
    assert sorted(entries) == ["a.txt", "b.txt", "subdir"]


async def test_local_tmp_file_operator_list_dir_with_types(tmp_path: Path) -> None:
    """LocalTmpFileOperator should list directory with type info."""
    op = LocalTmpFileOperator(tmp_path)

    await op.write_file("file1.txt", "content")
    await op.write_file("file2.txt", "content")
    await op.mkdir("dir1")
    await op.mkdir("dir2")

    entries = await op.list_dir_with_types(".")

    # Should be sorted alphabetically
    assert entries == [
        ("dir1", True),
        ("dir2", True),
        ("file1.txt", False),
        ("file2.txt", False),
    ]


async def test_local_tmp_file_operator_move(tmp_path: Path) -> None:
    """LocalTmpFileOperator should support moving files."""
    op = LocalTmpFileOperator(tmp_path)

    await op.write_file("source.txt", "content")
    await op.move("source.txt", "dest.txt")

    assert not await op.exists("source.txt")
    assert await op.exists("dest.txt")
    assert await op.read_file("dest.txt") == "content"


async def test_local_tmp_file_operator_copy(tmp_path: Path) -> None:
    """LocalTmpFileOperator should support copying files."""
    op = LocalTmpFileOperator(tmp_path)

    await op.write_file("source.txt", "content")
    await op.copy("source.txt", "dest.txt")

    assert await op.exists("source.txt")
    assert await op.exists("dest.txt")
    assert await op.read_file("dest.txt") == "content"


async def test_local_tmp_file_operator_stat(tmp_path: Path) -> None:
    """LocalTmpFileOperator should return file stats."""
    op = LocalTmpFileOperator(tmp_path)

    await op.write_file("test.txt", "hello")
    stat = await op.stat("test.txt")

    assert stat["size"] == 5
    assert stat["is_file"] is True
    assert stat["is_dir"] is False
    assert stat["mtime"] > 0


async def test_local_tmp_file_operator_glob(tmp_path: Path) -> None:
    """LocalTmpFileOperator should support glob patterns."""
    op = LocalTmpFileOperator(tmp_path)

    await op.write_file("a.txt", "a")
    await op.write_file("b.txt", "b")
    await op.write_file("c.py", "c")

    txt_files = await op.glob("*.txt")
    assert sorted(txt_files) == ["a.txt", "b.txt"]


async def test_local_tmp_file_operator_truncate_to_tmp(tmp_path: Path) -> None:
    """LocalTmpFileOperator should truncate large content and save to tmp."""
    op = LocalTmpFileOperator(tmp_path)

    # Small content - no truncation
    small = "Hello"
    result = await op.truncate_to_tmp(small, "small.txt", max_length=100)
    assert result == small

    # Large content - should truncate
    large = "x" * 200
    result = await op.truncate_to_tmp(large, "large.txt", max_length=100)

    assert isinstance(result, dict)
    assert len(result["content"]) <= 100
    assert "large.txt" in result["file_path"]
    assert "truncated" in result["message"].lower()


async def test_local_tmp_file_operator_is_managed_path(tmp_path: Path) -> None:
    """LocalTmpFileOperator should correctly identify managed paths."""
    op = LocalTmpFileOperator(tmp_path)

    # Path within tmp_dir
    is_managed, _path = op.is_managed_path("subdir/file.txt", tmp_path)
    assert is_managed is True

    # Path outside tmp_dir
    is_managed, _path = op.is_managed_path("/etc/passwd", tmp_path)
    assert is_managed is False


async def test_local_tmp_file_operator_tmp_dir_property(tmp_path: Path) -> None:
    """LocalTmpFileOperator should expose tmp_dir property."""
    op = LocalTmpFileOperator(tmp_path)
    assert op.tmp_dir == str(tmp_path)


async def test_local_tmp_operator_read_bytes_with_offset(tmp_path: Path) -> None:
    """LocalTmpFileOperator should support reading bytes with offset and length."""
    op = LocalTmpFileOperator(tmp_path)

    await op.write_file("test.bin", b"0123456789")

    # Read with offset
    content = await op.read_bytes("test.bin", offset=3)
    assert content == b"3456789"

    # Read with offset and length
    content = await op.read_bytes("test.bin", offset=3, length=4)
    assert content == b"3456"


async def test_local_tmp_operator_read_bytes_seek_optimization(tmp_path: Path) -> None:
    """Partial read should use seek instead of reading entire file."""
    op = LocalTmpFileOperator(tmp_path)

    # Create a larger file
    large_content = b"x" * 100000 + b"TARGET" + b"y" * 100000  # ~200KB
    await op.write_file("large.bin", large_content)

    # Read just the TARGET portion using offset/length
    # This should only read 6 bytes, not the entire 200KB
    content = await op.read_bytes("large.bin", offset=100000, length=6)
    assert content == b"TARGET"

    # Verify offset-only read
    content = await op.read_bytes("large.bin", offset=100000)
    assert content == b"TARGET" + b"y" * 100000


async def test_local_tmp_operator_write_bytes(tmp_path: Path) -> None:
    """LocalTmpFileOperator should support writing bytes."""
    op = LocalTmpFileOperator(tmp_path)

    await op.write_file("test.bin", b"\x00\x01\x02\x03")
    content = await op.read_bytes("test.bin")
    assert content == b"\x00\x01\x02\x03"


async def test_local_tmp_operator_append_bytes(tmp_path: Path) -> None:
    """LocalTmpFileOperator should support appending bytes."""
    op = LocalTmpFileOperator(tmp_path)

    await op.write_file("test.bin", b"hello")
    await op.append_file("test.bin", b"world")
    content = await op.read_bytes("test.bin")
    assert content == b"helloworld"


async def test_local_tmp_operator_append_text(tmp_path: Path) -> None:
    """LocalTmpFileOperator should support appending text."""
    op = LocalTmpFileOperator(tmp_path)

    await op.write_file("test.txt", "hello")
    await op.append_file("test.txt", "world")
    content = await op.read_file("test.txt")
    assert content == "helloworld"


async def test_local_tmp_operator_delete_dir(tmp_path: Path) -> None:
    """LocalTmpFileOperator should support deleting empty directories."""
    op = LocalTmpFileOperator(tmp_path)

    await op.mkdir("empty_dir")
    assert await op.is_dir("empty_dir")

    await op.delete("empty_dir")
    assert not await op.exists("empty_dir")


# --- Streaming tests ---


async def test_local_tmp_operator_read_bytes_stream(tmp_path: Path) -> None:
    """LocalTmpFileOperator should support streaming read."""
    op = LocalTmpFileOperator(tmp_path)

    # Create a file with known content
    content = b"0123456789" * 100  # 1000 bytes
    await op.write_file("test.bin", content)

    # Read as stream
    stream = op.read_bytes_stream("test.bin", chunk_size=256)
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    # Verify content
    result = b"".join(chunks)
    assert result == content

    # Verify chunking happened (with 256 byte chunks, should have multiple chunks)
    assert len(chunks) > 1


async def test_local_tmp_operator_read_bytes_stream_small_file(tmp_path: Path) -> None:
    """LocalTmpFileOperator streaming should work for small files."""
    op = LocalTmpFileOperator(tmp_path)

    content = b"small content"
    await op.write_file("small.bin", content)

    stream = op.read_bytes_stream("small.bin")
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    assert b"".join(chunks) == content


async def test_local_tmp_operator_write_bytes_stream(tmp_path: Path) -> None:
    """LocalTmpFileOperator should support streaming write."""
    op = LocalTmpFileOperator(tmp_path)

    # Create an async generator of chunks
    async def chunk_generator():
        yield b"chunk1"
        yield b"chunk2"
        yield b"chunk3"

    await op.write_bytes_stream("test.bin", chunk_generator())

    # Verify content
    content = await op.read_bytes("test.bin")
    assert content == b"chunk1chunk2chunk3"


async def test_local_tmp_operator_stream_roundtrip(tmp_path: Path) -> None:
    """Streaming read/write should preserve data integrity."""
    op = LocalTmpFileOperator(tmp_path)

    # Create a larger file to test proper chunking
    original = bytes(range(256)) * 1000  # 256KB
    await op.write_file("original.bin", original)

    # Stream read from original
    stream = op.read_bytes_stream("original.bin", chunk_size=4096)

    # Stream write to copy
    await op.write_bytes_stream("copy.bin", stream)

    # Verify copy matches original
    copy = await op.read_bytes("copy.bin")
    assert copy == original


async def test_local_tmp_operator_stream_default_chunk_size(tmp_path: Path) -> None:
    """Default chunk size should be DEFAULT_CHUNK_SIZE."""
    op = LocalTmpFileOperator(tmp_path)

    # Create a file larger than default chunk size
    content = b"x" * (DEFAULT_CHUNK_SIZE + 1000)
    await op.write_file("large.bin", content)

    stream = op.read_bytes_stream("large.bin")
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    # Should have at least 2 chunks with default chunk size
    assert len(chunks) >= 2
    assert b"".join(chunks) == content


# --- Cross-boundary streaming tests ---


class LocalFileOperator(FileOperator):
    """A local filesystem FileOperator for testing cross-boundary operations."""

    def __init__(self, default_path: Path, tmp_dir: Path, default_chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
        super().__init__(
            default_path=default_path,
            allowed_paths=[default_path, tmp_dir],
            tmp_dir=tmp_dir,
            default_chunk_size=default_chunk_size,
        )

    async def _read_file_impl(
        self, path: str, *, encoding: str = "utf-8", offset: int = 0, length: int | None = None
    ) -> str:
        import anyio

        resolved = self._default_path / path
        content = await anyio.Path(resolved).read_text(encoding=encoding)
        if offset > 0 or length is not None:
            end = None if length is None else offset + length
            content = content[offset:end]
        return content

    async def _read_bytes_impl(self, path: str, *, offset: int = 0, length: int | None = None) -> bytes:
        import anyio

        resolved = self._default_path / path
        content = await anyio.Path(resolved).read_bytes()
        if offset > 0 or length is not None:
            end = None if length is None else offset + length
            content = content[offset:end]
        return content

    async def _write_file_impl(self, path: str, content: str | bytes, *, encoding: str = "utf-8") -> None:
        import anyio

        resolved = self._default_path / path
        apath = anyio.Path(resolved)
        if isinstance(content, bytes):
            await apath.write_bytes(content)
        else:
            await apath.write_text(content, encoding=encoding)

    async def _append_file_impl(self, path: str, content: str | bytes, *, encoding: str = "utf-8") -> None:
        import anyio

        resolved = self._default_path / path
        existing = await anyio.Path(resolved).read_bytes() if await anyio.Path(resolved).exists() else b""
        new_content = existing + (content if isinstance(content, bytes) else content.encode(encoding))
        await anyio.Path(resolved).write_bytes(new_content)

    async def _delete_impl(self, path: str) -> None:
        import anyio

        resolved = self._default_path / path
        apath = anyio.Path(resolved)
        if await apath.is_dir():
            await apath.rmdir()
        else:
            await apath.unlink()

    async def _list_dir_impl(self, path: str) -> list[str]:
        import anyio

        resolved = self._default_path / path
        entries = [entry.name async for entry in anyio.Path(resolved).iterdir()]
        return sorted(entries)

    async def _exists_impl(self, path: str) -> bool:
        import anyio

        return await anyio.Path(self._default_path / path).exists()

    async def _is_file_impl(self, path: str) -> bool:
        import anyio

        return await anyio.Path(self._default_path / path).is_file()

    async def _is_dir_impl(self, path: str) -> bool:
        import anyio

        return await anyio.Path(self._default_path / path).is_dir()

    async def _mkdir_impl(self, path: str, *, parents: bool = False) -> None:
        import anyio

        await anyio.Path(self._default_path / path).mkdir(parents=parents, exist_ok=True)

    async def _move_impl(self, src: str, dst: str) -> None:
        import shutil

        import anyio

        src_resolved = self._default_path / src
        dst_resolved = self._default_path / dst
        await anyio.to_thread.run_sync(shutil.move, src_resolved, dst_resolved)

    async def _copy_impl(self, src: str, dst: str) -> None:
        import shutil

        import anyio

        src_resolved = self._default_path / src
        dst_resolved = self._default_path / dst
        await anyio.to_thread.run_sync(shutil.copy2, src_resolved, dst_resolved)

    async def _stat_impl(self, path: str) -> FileStat:
        import anyio

        resolved = self._default_path / path
        st = await anyio.Path(resolved).stat()
        return FileStat(
            size=st.st_size,
            mtime=st.st_mtime,
            is_file=await anyio.Path(resolved).is_file(),
            is_dir=await anyio.Path(resolved).is_dir(),
        )

    async def _glob_impl(self, pattern: str) -> list[str]:
        matches = []
        for p in self._default_path.glob(pattern):
            try:
                rel = p.relative_to(self._default_path)
                matches.append(str(rel))
            except ValueError:
                matches.append(str(p))
        return sorted(matches)

    # Override streaming for true streaming behavior
    async def _read_bytes_stream_impl(
        self,
        path: str,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> AsyncIterator[bytes]:
        import anyio

        resolved = self._default_path / path
        async with await anyio.open_file(resolved, "rb") as f:
            while True:
                chunk = await f.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    async def _write_bytes_stream_impl(
        self,
        path: str,
        stream: AsyncIterator[bytes],
    ) -> None:
        import anyio

        resolved = self._default_path / path
        async with await anyio.open_file(resolved, "wb") as f:
            async for chunk in stream:
                await f.write(chunk)


async def test_cross_boundary_copy_from_main_to_tmp(tmp_path: Path) -> None:
    """Copy from main filesystem to tmp should use streaming."""
    main_dir = tmp_path / "main"
    tmp_dir = tmp_path / "tmp"
    main_dir.mkdir()
    tmp_dir.mkdir()

    op = LocalFileOperator(main_dir, tmp_dir)

    # Create a large file in main dir
    content = b"x" * (DEFAULT_CHUNK_SIZE * 3)  # ~192KB
    await op.write_file("source.bin", content)

    # Copy to tmp (cross-boundary)
    await op.copy("source.bin", str(tmp_dir / "dest.bin"))

    # Verify content in tmp
    result = await op.read_bytes(str(tmp_dir / "dest.bin"))
    assert result == content

    # Original should still exist
    assert await op.exists("source.bin")


async def test_cross_boundary_copy_from_tmp_to_main(tmp_path: Path) -> None:
    """Copy from tmp to main filesystem should use streaming."""
    main_dir = tmp_path / "main"
    tmp_dir = tmp_path / "tmp"
    main_dir.mkdir()
    tmp_dir.mkdir()

    op = LocalFileOperator(main_dir, tmp_dir)

    # Create a large file in tmp dir
    content = b"y" * (DEFAULT_CHUNK_SIZE * 2)  # ~128KB
    await op._tmp_file_operator.write_file("source.bin", content)

    # Copy to main (cross-boundary)
    await op.copy(str(tmp_dir / "source.bin"), "dest.bin")

    # Verify content in main
    result = await op.read_bytes("dest.bin")
    assert result == content

    # Original should still exist in tmp
    assert await op._tmp_file_operator.exists("source.bin")


async def test_cross_boundary_move_from_main_to_tmp(tmp_path: Path) -> None:
    """Move from main filesystem to tmp should use streaming."""
    main_dir = tmp_path / "main"
    tmp_dir = tmp_path / "tmp"
    main_dir.mkdir()
    tmp_dir.mkdir()

    op = LocalFileOperator(main_dir, tmp_dir)

    # Create a file in main dir
    content = b"move_test" * 10000  # ~90KB
    await op.write_file("source.bin", content)

    # Move to tmp (cross-boundary)
    await op.move("source.bin", str(tmp_dir / "dest.bin"))

    # Verify content in tmp
    result = await op.read_bytes(str(tmp_dir / "dest.bin"))
    assert result == content

    # Original should be deleted
    assert not await op.exists("source.bin")


async def test_cross_boundary_move_from_tmp_to_main(tmp_path: Path) -> None:
    """Move from tmp to main filesystem should use streaming."""
    main_dir = tmp_path / "main"
    tmp_dir = tmp_path / "tmp"
    main_dir.mkdir()
    tmp_dir.mkdir()

    op = LocalFileOperator(main_dir, tmp_dir)

    # Create a file in tmp dir
    content = b"move_test_2" * 10000
    await op._tmp_file_operator.write_file("source.bin", content)

    # Move to main (cross-boundary)
    await op.move(str(tmp_dir / "source.bin"), "dest.bin")

    # Verify content in main
    result = await op.read_bytes("dest.bin")
    assert result == content

    # Original should be deleted from tmp
    assert not await op._tmp_file_operator.exists("source.bin")


async def test_file_operator_default_chunk_size(tmp_path: Path) -> None:
    """FileOperator should use default_chunk_size in cross-boundary operations."""
    main_dir = tmp_path / "main"
    tmp_dir = tmp_path / "tmp"
    main_dir.mkdir()
    tmp_dir.mkdir()

    # Use a custom small chunk size
    custom_chunk_size = 256

    # Track chunks read from main operator
    chunks_read: list[int] = []

    class TrackedFileOperator(LocalFileOperator):
        """FileOperator that tracks chunk sizes used in streaming."""

        async def _read_bytes_stream_impl(
            self,
            path: str,
            *,
            chunk_size: int = DEFAULT_CHUNK_SIZE,
        ) -> AsyncIterator[bytes]:
            import anyio

            chunks_read.append(chunk_size)
            resolved = self._default_path / path
            async with await anyio.open_file(resolved, "rb") as f:
                while True:
                    chunk = await f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk

    op = TrackedFileOperator(main_dir, tmp_dir, default_chunk_size=custom_chunk_size)

    # Create a file in main dir
    content = b"x" * 1024  # 1KB file
    await op.write_file("source.bin", content)

    # Copy to tmp (cross-boundary) - should use custom chunk size
    await op.copy("source.bin", str(tmp_dir / "dest.bin"))

    # Verify chunk size was used
    assert len(chunks_read) > 0
    assert chunks_read[-1] == custom_chunk_size

    # Verify content is correct
    result = await op.read_bytes(str(tmp_dir / "dest.bin"))
    assert result == content


# --- Tmp-only FileOperator (default_path=None) ---


class TmpOnlyFileOperator(FileOperator):
    """A FileOperator with no main filesystem, only tmp."""

    def __init__(self, tmp_dir: Path) -> None:
        super().__init__(
            default_path=None,
            tmp_dir=tmp_dir,
        )

    # All _impl methods raise -- they should never be called in tmp-only mode
    async def _read_file_impl(
        self, path: str, *, encoding: str = "utf-8", offset: int = 0, length: int | None = None
    ) -> str:
        raise RuntimeError("Should not be called in tmp-only mode")

    async def _read_bytes_impl(self, path: str, *, offset: int = 0, length: int | None = None) -> bytes:
        raise RuntimeError("Should not be called in tmp-only mode")

    async def _write_file_impl(self, path: str, content: str | bytes, *, encoding: str = "utf-8") -> None:
        raise RuntimeError("Should not be called in tmp-only mode")

    async def _append_file_impl(self, path: str, content: str | bytes, *, encoding: str = "utf-8") -> None:
        raise RuntimeError("Should not be called in tmp-only mode")

    async def _delete_impl(self, path: str) -> None:
        raise RuntimeError("Should not be called in tmp-only mode")

    async def _list_dir_impl(self, path: str) -> list[str]:
        raise RuntimeError("Should not be called in tmp-only mode")

    async def _exists_impl(self, path: str) -> bool:
        raise RuntimeError("Should not be called in tmp-only mode")

    async def _is_file_impl(self, path: str) -> bool:
        raise RuntimeError("Should not be called in tmp-only mode")

    async def _is_dir_impl(self, path: str) -> bool:
        raise RuntimeError("Should not be called in tmp-only mode")

    async def _mkdir_impl(self, path: str, *, parents: bool = False) -> None:
        raise RuntimeError("Should not be called in tmp-only mode")

    async def _move_impl(self, src: str, dst: str) -> None:
        raise RuntimeError("Should not be called in tmp-only mode")

    async def _copy_impl(self, src: str, dst: str) -> None:
        raise RuntimeError("Should not be called in tmp-only mode")

    async def _stat_impl(self, path: str) -> FileStat:
        raise RuntimeError("Should not be called in tmp-only mode")

    async def _glob_impl(self, pattern: str) -> list[str]:
        raise RuntimeError("Should not be called in tmp-only mode")


async def test_tmp_only_file_operator_routes_all_to_tmp(tmp_path: Path) -> None:
    """FileOperator with default_path=None should route all operations to tmp."""
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()

    op = TmpOnlyFileOperator(tmp_dir)

    # All operations go through tmp_file_operator, not _impl methods
    await op.write_file("test.txt", "Hello from tmp-only")
    content = await op.read_file("test.txt")
    assert content == "Hello from tmp-only"

    assert await op.exists("test.txt")
    assert await op.is_file("test.txt")
    assert not await op.is_dir("test.txt")

    await op.mkdir("subdir")
    assert await op.is_dir("subdir")

    entries = await op.list_dir(".")
    assert "test.txt" in entries
    assert "subdir" in entries


async def test_tmp_only_file_operator_allowed_paths_empty(tmp_path: Path) -> None:
    """FileOperator with default_path=None should have empty allowed_paths by default."""
    op = TmpOnlyFileOperator(tmp_path)
    assert op._allowed_paths == []
    assert op._default_path is None


async def test_tmp_only_file_operator_context_instructions(tmp_path: Path) -> None:
    """FileOperator with default_path=None should omit default-directory in instructions."""
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()

    op = TmpOnlyFileOperator(tmp_dir)
    instructions = await op.get_context_instructions()
    assert instructions is not None
    assert "default-directory" not in instructions
    assert "tmp-directory" in instructions


async def test_file_operator_allowed_paths_without_default(tmp_path: Path) -> None:
    """FileOperator with default_path=None and explicit allowed_paths."""
    extra = tmp_path / "extra"
    extra.mkdir()
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()

    # Reuse TmpOnlyFileOperator but override __init__ to pass allowed_paths
    op = TmpOnlyFileOperator.__new__(TmpOnlyFileOperator)
    FileOperator.__init__(op, default_path=None, tmp_dir=tmp_dir, allowed_paths=[extra])

    # allowed_paths should contain only the explicit paths, not try to add None
    assert len(op._allowed_paths) == 1
    assert op._allowed_paths[0] == extra.resolve()
