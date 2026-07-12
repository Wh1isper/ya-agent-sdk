from __future__ import annotations

import asyncio
import base64
import binascii
import heapq
import json
import os
import stat
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import PurePosixPath, PureWindowsPath
from typing import BinaryIO

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ya_claw.controller.windows_workspace_files import (
    WindowsWorkspaceError,
)
from ya_claw.controller.windows_workspace_files import (
    open_regular_file as open_windows_regular_file,
)
from ya_claw.controller.windows_workspace_files import (
    pinned_directory as pinned_windows_directory,
)
from ya_claw.orm.tables import SessionRecord
from ya_claw.workspace import WorkspaceBinding, WorkspaceMountBinding, WorkspaceProvider
from ya_claw.workspace.file_models import (
    WorkspaceFileEntry,
    WorkspaceFileKind,
    WorkspaceFileListResponse,
    WorkspaceTextFileResponse,
)
from ya_claw.workspace.models import normalize_virtual_path, virtual_path_contains

MAX_WORKSPACE_TEXT_FILE_BYTES = 1024 * 1024
MAX_WORKSPACE_DIRECTORY_SCAN_ENTRIES = 100_000


@dataclass(slots=True)
class WorkspaceDownload:
    file: BinaryIO
    filename: str
    size_bytes: int
    max_bytes: int


class WorkspaceFilesController:
    async def list_files(
        self,
        *,
        db_session: AsyncSession,
        workspace_provider: WorkspaceProvider,
        session_id: str,
        path: str | None,
        include_hidden: bool,
        limit: int,
        offset: int,
        cursor: str | None,
    ) -> WorkspaceFileListResponse:
        binding = await self._resolve_session_binding(
            db_session=db_session,
            workspace_provider=workspace_provider,
            session_id=session_id,
        )
        virtual_path = _validated_virtual_path(path, default=str(binding.cwd))
        cursor_key, page_offset = _decode_directory_cursor(cursor) if cursor is not None else (None, offset)
        if cursor is not None and offset != 0:
            raise HTTPException(status_code=400, detail="Workspace cursor and a non-zero offset cannot be combined.")

        mount, relative_parts = _bound_mount_and_parts(binding, virtual_path)
        result_offset = 0 if cursor_key is not None else offset
        children = await asyncio.to_thread(
            _list_bound_directory,
            mount,
            relative_parts,
            virtual_path=virtual_path,
            include_hidden=include_hidden,
            result_limit=result_offset + limit + 1,
            after_key=cursor_key,
        )

        has_more = len(children) > result_offset + limit
        page = children[result_offset : result_offset + limit]
        items: list[WorkspaceFileEntry] = []
        for child_name, child_stat in page:
            items.append(
                WorkspaceFileEntry(
                    name=child_name,
                    path=(PurePosixPath(virtual_path) / child_name).as_posix(),
                    kind=_entry_kind(child_stat),
                    size_bytes=child_stat.st_size if stat.S_ISREG(child_stat.st_mode) else None,
                    modified_at=datetime.fromtimestamp(child_stat.st_mtime, tz=UTC),
                    hidden=child_name.startswith("."),
                )
            )

        next_offset = page_offset + len(items) if has_more else None
        next_cursor = (
            _encode_directory_cursor(_directory_sort_key(page[-1][0]), next_offset)
            if has_more and page and next_offset is not None
            else None
        )
        return WorkspaceFileListResponse(
            session_id=session_id,
            path=virtual_path,
            items=items,
            limit=limit,
            offset=page_offset,
            has_more=has_more,
            next_offset=next_offset,
            next_cursor=next_cursor,
            truncated=has_more,
        )

    async def read_text_file(
        self,
        *,
        db_session: AsyncSession,
        workspace_provider: WorkspaceProvider,
        session_id: str,
        path: str,
    ) -> WorkspaceTextFileResponse:
        binding = await self._resolve_session_binding(
            db_session=db_session,
            workspace_provider=workspace_provider,
            session_id=session_id,
        )
        virtual_path = _validated_virtual_path(path)
        mount, relative_parts = _bound_mount_and_parts(binding, virtual_path)
        data = await asyncio.to_thread(
            _read_regular_file,
            mount,
            relative_parts,
            virtual_path,
            max_bytes=MAX_WORKSPACE_TEXT_FILE_BYTES,
        )

        if len(data) > MAX_WORKSPACE_TEXT_FILE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Workspace text file exceeds the {MAX_WORKSPACE_TEXT_FILE_BYTES}-byte limit.",
            )
        if _looks_binary(data):
            raise HTTPException(status_code=415, detail=f"Workspace file '{virtual_path}' is not UTF-8 text.")
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=415, detail=f"Workspace file '{virtual_path}' is not UTF-8 text.") from exc

        return WorkspaceTextFileResponse(
            session_id=session_id,
            path=virtual_path,
            content=content,
            size_bytes=len(data),
        )

    async def open_download(
        self,
        *,
        db_session: AsyncSession,
        workspace_provider: WorkspaceProvider,
        session_id: str,
        path: str,
        max_bytes: int,
    ) -> WorkspaceDownload:
        binding = await self._resolve_session_binding(
            db_session=db_session,
            workspace_provider=workspace_provider,
            session_id=session_id,
        )
        virtual_path = _validated_virtual_path(path)
        mount, relative_parts = _bound_mount_and_parts(binding, virtual_path)
        file, size_bytes = await asyncio.to_thread(_open_regular_file, mount, relative_parts, virtual_path)
        if size_bytes > max_bytes:
            file.close()
            raise HTTPException(
                status_code=413,
                detail=f"Workspace file exceeds the {max_bytes}-byte download limit.",
            )
        return WorkspaceDownload(
            file=file,
            filename=PurePosixPath(virtual_path).name or "download",
            size_bytes=size_bytes,
            max_bytes=max_bytes,
        )

    async def _resolve_session_binding(
        self,
        *,
        db_session: AsyncSession,
        workspace_provider: WorkspaceProvider,
        session_id: str,
    ) -> WorkspaceBinding:
        session_record = await db_session.get(SessionRecord, session_id)
        if not isinstance(session_record, SessionRecord):
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' was not found.")
        metadata = dict(session_record.session_metadata) if isinstance(session_record.session_metadata, dict) else {}
        metadata.setdefault("session_id", session_record.id)
        return workspace_provider.resolve(metadata)


def _directory_sort_key(name: str) -> tuple[str, str]:
    return name.casefold(), name


def _encode_directory_cursor(key: tuple[str, str], offset: int) -> str:
    payload = json.dumps(
        {"v": 1, "key": [key[0], key[1]], "offset": offset},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_directory_cursor(cursor: str) -> tuple[tuple[str, str], int]:
    try:
        padding = "=" * (-len(cursor) % 4)
        payload = base64.b64decode(cursor + padding, altchars=b"-_", validate=True)
        decoded = json.loads(payload.decode("utf-8"))
        key = decoded["key"]
        offset = decoded["offset"]
        if (
            decoded.get("v") != 1
            or not isinstance(key, list)
            or len(key) != 2
            or not all(isinstance(part, str) for part in key)
            or not key[1]
            or "/" in key[1]
            or "\x00" in key[1]
            or key[0] != key[1].casefold()
            or not isinstance(offset, int)
            or isinstance(offset, bool)
            or offset < 0
            or offset > MAX_WORKSPACE_DIRECTORY_SCAN_ENTRIES
        ):
            raise ValueError
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Workspace pagination cursor is invalid.") from exc
    return (key[0], key[1]), offset


def _validated_virtual_path(path: str | None, *, default: str | None = None) -> str:
    raw_path = default if path is None else path
    if not isinstance(raw_path, str) or raw_path == "":
        raise HTTPException(status_code=400, detail="Workspace path must be a non-empty absolute virtual path.")
    if "\x00" in raw_path:
        raise HTTPException(status_code=400, detail="Workspace path must be an absolute POSIX virtual path.")
    if PureWindowsPath(raw_path).is_absolute():
        raise HTTPException(status_code=403, detail=f"Workspace path '{raw_path}' is outside readable mounts.")
    if "\\" in raw_path or not raw_path.startswith("/"):
        raise HTTPException(status_code=400, detail="Workspace path must be an absolute POSIX virtual path.")

    segments = raw_path.split("/")
    if any(segment in {".", ".."} for segment in segments):
        raise HTTPException(status_code=400, detail="Workspace path traversal is not allowed.")
    try:
        normalized = normalize_virtual_path(raw_path)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Workspace path is not a valid virtual path.") from exc

    # Do not silently reinterpret non-canonical input as another virtual path.
    canonical_input = raw_path.rstrip("/") or "/"
    if canonical_input != normalized:
        raise HTTPException(status_code=400, detail="Workspace path must be canonical.")
    return normalized


def _bound_mount_and_parts(
    binding: WorkspaceBinding,
    virtual_path: str,
) -> tuple[WorkspaceMountBinding, tuple[str, ...]]:
    mount = _mount_for_virtual_path(binding, virtual_path)
    mount_virtual_path = PurePosixPath(str(mount.virtual_path))
    relative_path = PurePosixPath(virtual_path).relative_to(mount_virtual_path)
    return mount, relative_path.parts


def _mount_for_virtual_path(binding: WorkspaceBinding, virtual_path: str) -> WorkspaceMountBinding:
    matching_mounts = [mount for mount in binding.mounts if virtual_path_contains(mount.virtual_path, virtual_path)]
    if not matching_mounts:
        raise HTTPException(status_code=403, detail=f"Workspace path '{virtual_path}' is outside readable mounts.")
    return max(matching_mounts, key=lambda mount: len(PurePosixPath(str(mount.virtual_path)).parts))


_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_OPEN_SUPPORTS_DIR_FD = os.open in getattr(os, "supports_dir_fd", ())
_SCANDIR_SUPPORTS_FD = os.scandir in getattr(os, "supports_fd", ())
_FILE_ATTRIBUTE_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
_PATH_FALLBACK_SUPPORTED = os.name == "nt"


def _secure_fd_operations_available(*, require_scandir: bool = False) -> bool:
    return bool(
        _O_DIRECTORY and _O_NOFOLLOW and _OPEN_SUPPORTS_DIR_FD and (not require_scandir or _SCANDIR_SUPPORTS_FD)
    )


def _require_secure_fd_operations(virtual_path: str, *, require_scandir: bool = False) -> None:
    if not _secure_fd_operations_available(require_scandir=require_scandir):
        # Descriptor-backend helpers must not silently degrade. Public dispatchers
        # select the checked Windows path backend when its platform contract applies.
        raise HTTPException(
            status_code=403,
            detail=f"Workspace path '{virtual_path}' cannot be opened safely on this platform.",
        )


def _open_mount_root(mount: WorkspaceMountBinding, virtual_path: str) -> int:
    _require_secure_fd_operations(virtual_path)
    flags = os.O_RDONLY | _O_CLOEXEC | _O_DIRECTORY | _O_NOFOLLOW
    try:
        descriptor = os.open(mount.host_path, flags)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Workspace path '{virtual_path}' was not found.") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=f"Workspace path '{virtual_path}' is not readable.") from exc
    except OSError as exc:
        raise HTTPException(
            status_code=403, detail=f"Workspace path '{virtual_path}' cannot be opened safely."
        ) from exc

    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise HTTPException(status_code=403, detail=f"Workspace path '{virtual_path}' cannot be opened safely.")
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _open_relative_directory(
    mount: WorkspaceMountBinding,
    relative_parts: tuple[str, ...],
    virtual_path: str,
) -> int:
    descriptor = _open_mount_root(mount, virtual_path)
    flags = os.O_RDONLY | _O_CLOEXEC | _O_DIRECTORY | _O_NOFOLLOW
    try:
        for segment in relative_parts:
            try:
                next_descriptor = os.open(segment, flags, dir_fd=descriptor)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=f"Workspace path '{virtual_path}' was not found.") from exc
            except PermissionError as exc:
                raise HTTPException(
                    status_code=403, detail=f"Workspace path '{virtual_path}' is not readable."
                ) from exc
            except OSError as exc:
                raise HTTPException(
                    status_code=403,
                    detail=f"Workspace path '{virtual_path}' cannot be opened safely.",
                ) from exc
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _list_bound_directory(
    mount: WorkspaceMountBinding,
    relative_parts: tuple[str, ...],
    *,
    virtual_path: str,
    include_hidden: bool,
    result_limit: int,
    after_key: tuple[str, str] | None,
) -> list[tuple[str, os.stat_result]]:
    if not _secure_fd_operations_available(require_scandir=True):
        if not _PATH_FALLBACK_SUPPORTED:
            _require_secure_fd_operations(virtual_path, require_scandir=True)
        return _list_bound_directory_by_path(
            mount,
            relative_parts,
            virtual_path=virtual_path,
            include_hidden=include_hidden,
            result_limit=result_limit,
            after_key=after_key,
        )
    directory_fd = _open_relative_directory(mount, relative_parts, virtual_path)
    try:
        return _list_directory_entries(
            directory_fd,
            virtual_path=virtual_path,
            include_hidden=include_hidden,
            result_limit=result_limit,
            after_key=after_key,
        )
    finally:
        os.close(directory_fd)


def _list_bound_directory_by_path(
    mount: WorkspaceMountBinding,
    relative_parts: tuple[str, ...],
    *,
    virtual_path: str,
    include_hidden: bool,
    result_limit: int,
    after_key: tuple[str, str] | None,
) -> list[tuple[str, os.stat_result]]:
    try:
        with pinned_windows_directory(mount.host_path, relative_parts) as directory_path:
            try:
                with os.scandir(directory_path) as iterator:
                    return heapq.nsmallest(
                        result_limit,
                        _stat_directory_entries(
                            iterator,
                            include_hidden=include_hidden,
                            after_key=after_key,
                        ),
                        key=lambda item: _directory_sort_key(item[0]),
                    )
            except PermissionError as exc:
                raise HTTPException(
                    status_code=403,
                    detail=f"Workspace path '{virtual_path}' is not readable.",
                ) from exc
            except OSError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Workspace path '{virtual_path}' cannot be listed.",
                ) from exc
    except WindowsWorkspaceError as exc:
        raise _windows_workspace_http_error(exc, virtual_path) from exc


def _list_directory_entries(
    directory_fd: int,
    *,
    virtual_path: str,
    include_hidden: bool,
    result_limit: int,
    after_key: tuple[str, str] | None,
) -> list[tuple[str, os.stat_result]]:
    _require_secure_fd_operations(virtual_path, require_scandir=True)
    try:
        with os.scandir(directory_fd) as iterator:
            entries = heapq.nsmallest(
                result_limit,
                _stat_directory_entries(
                    iterator,
                    include_hidden=include_hidden,
                    after_key=after_key,
                ),
                key=lambda item: _directory_sort_key(item[0]),
            )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=f"Workspace path '{virtual_path}' is not readable.") from exc
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"Workspace path '{virtual_path}' cannot be listed.") from exc
    return entries


def _stat_directory_entries(
    entries: Iterator[os.DirEntry[str]],
    *,
    include_hidden: bool,
    after_key: tuple[str, str] | None,
) -> Iterator[tuple[str, os.stat_result]]:
    for scanned_count, entry in enumerate(entries, start=1):
        if scanned_count > MAX_WORKSPACE_DIRECTORY_SCAN_ENTRIES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Workspace directory exceeds the {MAX_WORKSPACE_DIRECTORY_SCAN_ENTRIES}-entry safe scan limit."
                ),
            )
        if not include_hidden and entry.name.startswith("."):
            continue
        if after_key is not None and _directory_sort_key(entry.name) <= after_key:
            continue
        try:
            entry_stat = entry.stat(follow_symlinks=False)
        except FileNotFoundError:
            # A concurrent workspace update may remove an entry between scandir and stat.
            continue
        yield entry.name, entry_stat


def _open_regular_file_by_path(
    mount: WorkspaceMountBinding,
    relative_parts: tuple[str, ...],
    virtual_path: str,
) -> tuple[BinaryIO, int]:
    try:
        return open_windows_regular_file(mount.host_path, relative_parts)
    except WindowsWorkspaceError as exc:
        raise _windows_workspace_http_error(exc, virtual_path) from exc


def _windows_workspace_http_error(exc: WindowsWorkspaceError, virtual_path: str) -> HTTPException:
    if exc.reason == "missing":
        return HTTPException(status_code=404, detail=f"Workspace path '{virtual_path}' was not found.")
    if exc.reason == "not_regular":
        return HTTPException(status_code=400, detail=f"Workspace path '{virtual_path}' is not a regular file.")
    return HTTPException(status_code=403, detail=f"Workspace path '{virtual_path}' cannot be opened safely.")


def _open_regular_file(
    mount: WorkspaceMountBinding,
    relative_parts: tuple[str, ...],
    virtual_path: str,
) -> tuple[BinaryIO, int]:
    if not _secure_fd_operations_available():
        if not _PATH_FALLBACK_SUPPORTED:
            _require_secure_fd_operations(virtual_path)
        return _open_regular_file_by_path(mount, relative_parts, virtual_path)
    parent_parts = relative_parts[:-1] if relative_parts else ()
    parent_descriptor = _open_relative_directory(mount, parent_parts, virtual_path)
    if not relative_parts:
        descriptor = parent_descriptor
    else:
        flags = os.O_RDONLY | _O_CLOEXEC | _O_NOFOLLOW
        try:
            try:
                descriptor = os.open(relative_parts[-1], flags, dir_fd=parent_descriptor)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=f"Workspace path '{virtual_path}' was not found.") from exc
            except PermissionError as exc:
                raise HTTPException(
                    status_code=403, detail=f"Workspace file '{virtual_path}' is not readable."
                ) from exc
            except OSError as exc:
                raise HTTPException(
                    status_code=403,
                    detail=f"Workspace file '{virtual_path}' cannot be opened safely.",
                ) from exc
        finally:
            os.close(parent_descriptor)

    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise HTTPException(status_code=400, detail=f"Workspace path '{virtual_path}' is not a regular file.")
        return os.fdopen(descriptor, "rb"), file_stat.st_size
    except Exception:
        os.close(descriptor)
        raise


def _read_regular_file(
    mount: WorkspaceMountBinding,
    relative_parts: tuple[str, ...],
    virtual_path: str,
    *,
    max_bytes: int,
) -> bytes:
    file, size_bytes = _open_regular_file(mount, relative_parts, virtual_path)
    try:
        if size_bytes > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Workspace text file exceeds the {max_bytes}-byte limit.",
            )
        return file.read(max_bytes + 1)
    finally:
        file.close()


def _entry_kind(path_stat: os.stat_result) -> WorkspaceFileKind:
    file_attributes = getattr(path_stat, "st_file_attributes", 0)
    if stat.S_ISLNK(path_stat.st_mode) or bool(
        _FILE_ATTRIBUTE_REPARSE_POINT and file_attributes & _FILE_ATTRIBUTE_REPARSE_POINT
    ):
        return "symlink"
    if stat.S_ISDIR(path_stat.st_mode):
        return "directory"
    if stat.S_ISREG(path_stat.st_mode):
        return "file"
    return "other"


def _looks_binary(data: bytes) -> bool:
    if b"\x00" in data:
        return True
    disallowed_controls = sum(byte < 0x20 and byte not in {0x09, 0x0A, 0x0D} for byte in data)
    return disallowed_controls > 0
