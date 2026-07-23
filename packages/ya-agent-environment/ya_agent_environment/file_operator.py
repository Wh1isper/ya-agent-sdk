"""Backend-agnostic file operator abstraction."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from pathlib import Path, PurePath
from xml.etree import ElementTree as ET

from ya_agent_environment.protocols import DEFAULT_CHUNK_SIZE
from ya_agent_environment.types import FileEntry, FileStat

DEFAULT_INSTRUCTIONS_SKIP_DIRS: frozenset[str] = frozenset({"node_modules", ".git", ".venv", "__pycache__"})
DEFAULT_INSTRUCTIONS_MAX_DEPTH: int = 3


class FileOperator(ABC):
    """Abstract file-system backend.

    A file operator exposes one logical path space. Temporary-directory ownership
    and routing belong to :class:`Environment`; implementations only handle paths
    supported by their own backend.
    """

    def __init__(
        self,
        default_path: Path | PurePath | None = None,
        allowed_paths: Sequence[Path | PurePath] | None = None,
        instructions_paths: Sequence[Path | PurePath] | None = None,
        instructions_skip_dirs: frozenset[str] | None = None,
        instructions_max_depth: int = DEFAULT_INSTRUCTIONS_MAX_DEPTH,
        skip_instructions: bool = False,
        default_chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        self._default_path = self._normalize_config_path(default_path)
        if allowed_paths is None:
            self._allowed_paths = [self._default_path] if self._default_path is not None else []
        else:
            normalized = [self._normalize_config_path(path) for path in allowed_paths]
            if self._default_path is not None and self._default_path not in normalized:
                normalized.append(self._default_path)
            self._allowed_paths = [path for path in normalized if path is not None]

        if instructions_paths is None:
            self._instructions_paths = list(self._allowed_paths)
        else:
            self._instructions_paths = [
                path for path in (self._normalize_config_path(path) for path in instructions_paths) if path is not None
            ]
        self._instructions_skip_dirs = (
            instructions_skip_dirs if instructions_skip_dirs is not None else DEFAULT_INSTRUCTIONS_SKIP_DIRS
        )
        self._instructions_max_depth = instructions_max_depth
        self._skip_instructions = skip_instructions
        self._default_chunk_size = default_chunk_size

    @staticmethod
    def _normalize_config_path(path: Path | PurePath | None) -> Path | PurePath | None:
        if path is None:
            return None
        if isinstance(path, Path):
            return path.resolve()
        return path

    @staticmethod
    def _normalize_match_path(path: str) -> str:
        """Normalize an agent-facing path for pattern matching without resolving it."""
        normalized = path.replace("\\", "/")
        while "//" in normalized:
            normalized = normalized.replace("//", "/")
        if normalized.startswith("./"):
            normalized = normalized[2:]
        return normalized.rstrip("/") or "."

    def get_path_match_candidates(self, path: str) -> tuple[str, ...]:
        """Return equivalent path strings for agent-facing pattern matching."""
        normalized_path = self._normalize_match_path(path)
        candidates = {normalized_path}
        default_root = self._default_path
        if default_root is not None and not PurePath(normalized_path).is_absolute():
            default_root_path = self._normalize_match_path(str(default_root))
            if default_root_path != ".":
                candidates.add(
                    f"{default_root_path}/{normalized_path}" if normalized_path != "." else default_root_path
                )
        for root in [default_root, *self._allowed_paths]:
            if root is None:
                continue
            root_path = self._normalize_match_path(str(root))
            if root_path == ".":
                continue
            if normalized_path == root_path:
                candidates.add(".")
            elif normalized_path.startswith(f"{root_path}/"):
                candidates.add(normalized_path[len(root_path) + 1 :] or ".")
        return tuple(sorted(candidates))

    @abstractmethod
    async def read_file(
        self,
        path: str,
        *,
        encoding: str = "utf-8",
        offset: int = 0,
        length: int | None = None,
    ) -> str:
        """Read text from a file."""
        ...

    @abstractmethod
    async def read_bytes(
        self,
        path: str,
        *,
        offset: int = 0,
        length: int | None = None,
    ) -> bytes:
        """Read bytes from a file."""
        ...

    @abstractmethod
    async def write_file(
        self,
        path: str,
        content: str | bytes,
        *,
        encoding: str = "utf-8",
    ) -> None:
        """Write text or bytes to a file."""
        ...

    @abstractmethod
    async def append_file(
        self,
        path: str,
        content: str | bytes,
        *,
        encoding: str = "utf-8",
    ) -> None:
        """Append text or bytes to a file."""
        ...

    @abstractmethod
    async def delete(self, path: str) -> None:
        """Delete a file or empty directory."""
        ...

    @abstractmethod
    async def list_dir(self, path: str) -> list[str]:
        """List directory entries."""
        ...

    async def list_dir_with_types(self, path: str) -> list[tuple[str, bool]]:
        """List directory entries as ``(name, is_dir)`` pairs."""
        entries = await self.list_dir(path)
        result: list[tuple[str, bool]] = []
        for name in entries:
            entry_path = f"{path}/{name}" if path != "." else name
            result.append((name, await self.is_dir(entry_path)))
        return sorted(result, key=lambda item: item[0])

    @abstractmethod
    async def exists(self, path: str) -> bool:
        """Return whether a path exists."""
        ...

    @abstractmethod
    async def is_file(self, path: str) -> bool:
        """Return whether a path is a file."""
        ...

    @abstractmethod
    async def is_dir(self, path: str) -> bool:
        """Return whether a path is a directory."""
        ...

    @abstractmethod
    async def mkdir(self, path: str, *, parents: bool = False) -> None:
        """Create a directory."""
        ...

    @abstractmethod
    async def move(self, src: str, dst: str) -> None:
        """Move a file or directory within this backend."""
        ...

    @abstractmethod
    async def copy(self, src: str, dst: str) -> None:
        """Copy a file or directory within this backend."""
        ...

    @abstractmethod
    async def stat(self, path: str) -> FileStat:
        """Return file status."""
        ...

    async def walk_files(  # noqa: C901
        self,
        root: str = ".",
        *,
        max_depth: int | None = None,
        include_hidden: bool = False,
        follow_symlinks: bool = False,
    ) -> AsyncIterator[FileEntry]:
        """Walk files and directories through the logical path space."""
        del follow_symlinks
        root = root if root not in {"", "."} else "."
        try:
            root_is_dir = await self.is_dir(root)
            root_is_file = False if root_is_dir else await self.is_file(root)
        except Exception:
            return
        if root_is_file:
            try:
                file_stat = await self.stat(root)
            except Exception:
                file_stat = FileStat(size=0, mtime=0.0, is_file=True, is_dir=False)
            yield FileEntry(
                path=root,
                is_file=True,
                is_dir=False,
                size=file_stat.get("size"),
                mtime=file_stat.get("mtime"),
            )
            return
        if not root_is_dir:
            return

        async def walk_dir(path: str, depth: int) -> AsyncIterator[FileEntry]:
            try:
                children = await self.list_dir_with_types(path)
            except Exception:
                return
            for name, listed_as_dir in children:
                if not include_hidden and name.startswith("."):
                    continue
                child_path = name if path == "." else f"{path.rstrip('/')}/{name}"
                try:
                    child_stat = await self.stat(child_path)
                except Exception:
                    child_stat = FileStat(
                        size=0,
                        mtime=0.0,
                        is_file=not listed_as_dir,
                        is_dir=listed_as_dir,
                    )
                is_file = bool(child_stat.get("is_file", not listed_as_dir))
                is_dir = bool(child_stat.get("is_dir", listed_as_dir))
                yield FileEntry(
                    path=child_path,
                    is_file=is_file,
                    is_dir=is_dir,
                    size=child_stat.get("size"),
                    mtime=child_stat.get("mtime"),
                )
                if is_dir and (max_depth is None or depth + 1 < max_depth):
                    async for descendant in walk_dir(child_path, depth + 1):
                        yield descendant

        async for entry in walk_dir(root, 0):
            yield entry

    async def read_bytes_stream(
        self,
        path: str,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> AsyncIterator[bytes]:
        """Read bytes as an async iterator; call this method without ``await``."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        yield await self.read_bytes(path)

    async def write_bytes_stream(
        self,
        path: str,
        stream: AsyncIterator[bytes],
    ) -> None:
        """Write an async byte stream."""
        chunks: list[bytes] = []
        async for chunk in stream:
            chunks.append(chunk)
        await self.write_file(path, b"".join(chunks))

    async def get_context_instructions(self) -> str | None:
        """Return file-system context in XML format."""
        from ya_agent_environment.utils import generate_filetree

        if self._skip_instructions:
            return None
        root = ET.Element("file-system")
        if self._default_path is not None:
            default_dir = ET.SubElement(root, "default-directory")
            default_dir.text = str(self._default_path)

        file_trees = ET.SubElement(root, "file-trees")
        default_path = self._default_path
        for allowed_path in self._instructions_paths:
            if default_path is not None:
                try:
                    relative_path = str(allowed_path.relative_to(default_path))
                except ValueError:
                    relative_path = str(allowed_path)
            else:
                relative_path = str(allowed_path)
            tree = await generate_filetree(
                self,
                root_path=relative_path,
                max_depth=self._instructions_max_depth,
                skip_dirs=self._instructions_skip_dirs,
            )
            if tree and not tree.startswith("Directory not found"):
                directory = ET.SubElement(file_trees, "directory")
                directory.set("path", str(allowed_path))
                directory.text = f"\n{tree}\n    "
        ET.indent(root, space="  ")
        return ET.tostring(root, encoding="unicode")

    async def close(self) -> None:
        """Release backend resources."""
        return None
