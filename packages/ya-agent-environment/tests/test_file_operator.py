"""Tests for backend-agnostic behavior provided by FileOperator."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from ya_agent_environment import FileEntry, FileStat

from .conftest import MockFileOperator


class ContentFileOperator(MockFileOperator):
    """Small concrete backend used to exercise base streaming fallbacks."""

    def __init__(self, content: bytes) -> None:
        super().__init__()
        self.content = content
        self.written: bytes | None = None

    async def read_bytes(
        self,
        path: str,
        *,
        offset: int = 0,
        length: int | None = None,
    ) -> bytes:
        del path
        end = None if length is None else offset + length
        return self.content[offset:end]

    async def write_file(
        self,
        path: str,
        content: str | bytes,
        *,
        encoding: str = "utf-8",
    ) -> None:
        del path
        self.written = content.encode(encoding) if isinstance(content, str) else content


async def test_read_bytes_stream_is_returned_without_await() -> None:
    """The common fallback is an AsyncIterator returned directly by the call."""
    op = ContentFileOperator(b"stream content")

    stream = op.read_bytes_stream("content.bin")

    assert isinstance(stream, AsyncIterator)
    assert b"".join([chunk async for chunk in stream]) == b"stream content"


async def test_read_bytes_stream_rejects_invalid_chunk_size_when_consumed() -> None:
    op = ContentFileOperator(b"content")

    stream = op.read_bytes_stream("content.bin", chunk_size=0)
    with pytest.raises(ValueError, match="chunk_size must be greater than zero"):
        _ = [chunk async for chunk in stream]


async def test_write_bytes_stream_common_fallback() -> None:
    op = ContentFileOperator(b"")

    async def chunks() -> AsyncIterator[bytes]:
        yield b"one"
        yield b"two"

    await op.write_bytes_stream("output.bin", chunks())

    assert op.written == b"onetwo"


def test_path_match_candidates_normalize_relative_and_absolute(tmp_path: Path) -> None:
    op = MockFileOperator()
    op._default_path = tmp_path.resolve()
    op._allowed_paths = [tmp_path.resolve()]
    absolute_path = (tmp_path / "docs" / "guide.md").as_posix()

    assert {"docs/guide.md", absolute_path} <= set(op.get_path_match_candidates("docs/guide.md"))
    assert {"docs/guide.md", absolute_path} <= set(op.get_path_match_candidates(absolute_path))


class TreeFileOperator(MockFileOperator):
    async def is_dir(self, path: str) -> bool:
        return path in {".", "docs"}

    async def is_file(self, path: str) -> bool:
        return path == "docs/readme.md"

    async def list_dir(self, path: str) -> list[str]:
        return {".": ["docs", ".hidden"], "docs": ["readme.md"]}.get(path, [])

    async def list_dir_with_types(self, path: str) -> list[tuple[str, bool]]:
        return {
            ".": [("docs", True), (".hidden", False)],
            "docs": [("readme.md", False)],
        }.get(path, [])

    async def stat(self, path: str) -> FileStat:
        is_dir = path in {".", "docs"}
        return FileStat(size=0 if is_dir else 4, mtime=1.0, is_file=not is_dir, is_dir=is_dir)


async def test_walk_files_uses_backend_public_methods() -> None:
    op = TreeFileOperator()

    entries: list[FileEntry] = [entry async for entry in op.walk_files()]

    assert [entry["path"] for entry in entries] == ["docs", "docs/readme.md"]
