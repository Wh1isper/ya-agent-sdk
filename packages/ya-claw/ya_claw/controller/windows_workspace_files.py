from __future__ import annotations

import ctypes
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from ctypes import wintypes
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, BinaryIO, Literal, cast

WindowsWorkspaceErrorReason = Literal["missing", "denied", "unsafe", "not_directory", "not_regular"]

_GENERIC_READ = 0x80000000
_FILE_LIST_DIRECTORY = 0x0001
_FILE_SHARE_READ = 0x00000001
_FILE_SHARE_WRITE = 0x00000002
_OPEN_EXISTING = 3
_FILE_ATTRIBUTE_DIRECTORY = 0x00000010
_FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
_FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
_FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
_INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

_ERROR_FILE_NOT_FOUND = 2
_ERROR_PATH_NOT_FOUND = 3
_ERROR_ACCESS_DENIED = 5
_ERROR_SHARING_VIOLATION = 32
_ERROR_LOCK_VIOLATION = 33


class _ByHandleFileInformation(ctypes.Structure):
    _fields_ = [
        ("file_attributes", wintypes.DWORD),
        ("creation_time", wintypes.FILETIME),
        ("last_access_time", wintypes.FILETIME),
        ("last_write_time", wintypes.FILETIME),
        ("volume_serial_number", wintypes.DWORD),
        ("file_size_high", wintypes.DWORD),
        ("file_size_low", wintypes.DWORD),
        ("number_of_links", wintypes.DWORD),
        ("file_index_high", wintypes.DWORD),
        ("file_index_low", wintypes.DWORD),
    ]


@dataclass(frozen=True, slots=True)
class _WindowsHandle:
    value: int
    final_path: str
    attributes: int
    size_bytes: int


class WindowsWorkspaceError(Exception):
    def __init__(self, reason: WindowsWorkspaceErrorReason) -> None:
        self.reason = reason
        super().__init__(reason)


@contextmanager
def pinned_directory(root: Path, relative_parts: tuple[str, ...]) -> Iterator[Path]:
    """Pin every directory component against rename while a pathname scan runs."""
    handles, directory_path = _open_directory_chain(root, relative_parts)
    try:
        yield directory_path
    finally:
        _close_handles(handles)


def open_regular_file(root: Path, relative_parts: tuple[str, ...]) -> tuple[BinaryIO, int]:
    if not relative_parts:
        raise WindowsWorkspaceError("not_regular")

    target_path = root.joinpath(*relative_parts)
    _classify_regular_path(target_path)
    directory_handles, parent_path = _open_directory_chain(root, relative_parts[:-1])
    file_handle: int | None = None
    try:
        opened_file = _open_handle(parent_path / relative_parts[-1], expect_directory=False)
        file_handle = opened_file.value
        _require_contained(directory_handles[0].final_path, opened_file.final_path)

        # open_osfhandle transfers ownership of the Win32 handle to the CRT descriptor.
        import msvcrt

        open_osfhandle = getattr(msvcrt, "open_osfhandle", None)
        if not callable(open_osfhandle):
            raise WindowsWorkspaceError("unsafe")
        descriptor = cast(int, open_osfhandle(file_handle, os.O_RDONLY | getattr(os, "O_BINARY", 0)))
        file_handle = None
        try:
            file = os.fdopen(descriptor, "rb")
        except Exception:
            os.close(descriptor)
            raise
        return file, opened_file.size_bytes
    finally:
        if file_handle is not None:
            _close_handle(file_handle)
        _close_handles(directory_handles)


def _classify_regular_path(path: Path) -> None:
    try:
        path_stat = path.lstat()
    except FileNotFoundError as exc:
        raise WindowsWorkspaceError("missing") from exc
    except PermissionError as exc:
        raise WindowsWorkspaceError("denied") from exc
    except OSError as exc:
        raise WindowsWorkspaceError("unsafe") from exc

    file_attributes = getattr(path_stat, "st_file_attributes", 0)
    if stat.S_ISLNK(path_stat.st_mode) or file_attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
        raise WindowsWorkspaceError("unsafe")
    if not stat.S_ISREG(path_stat.st_mode):
        raise WindowsWorkspaceError("not_regular")


def _open_directory_chain(root: Path, relative_parts: tuple[str, ...]) -> tuple[list[_WindowsHandle], Path]:
    handles: list[_WindowsHandle] = []
    current_path = root
    try:
        root_handle = _open_handle(current_path, expect_directory=True)
        handles.append(root_handle)
        root_final_path = root_handle.final_path
        for segment in relative_parts:
            current_path = current_path / segment
            current_handle = _open_handle(current_path, expect_directory=True)
            _require_contained(root_final_path, current_handle.final_path)
            handles.append(current_handle)
        return handles, current_path
    except Exception:
        _close_handles(handles)
        raise


def _open_handle(path: Path, *, expect_directory: bool) -> _WindowsHandle:
    kernel32, get_last_error = _windows_api()
    desired_access = _FILE_LIST_DIRECTORY if expect_directory else _GENERIC_READ
    handle = kernel32.CreateFileW(
        str(path),
        desired_access,
        _FILE_SHARE_READ | _FILE_SHARE_WRITE,
        None,
        _OPEN_EXISTING,
        _FILE_FLAG_BACKUP_SEMANTICS | _FILE_FLAG_OPEN_REPARSE_POINT,
        None,
    )
    if handle == _INVALID_HANDLE_VALUE:
        _raise_windows_error(get_last_error())
    if not isinstance(handle, int):
        raise WindowsWorkspaceError("unsafe")

    try:
        information = _ByHandleFileInformation()
        if not kernel32.GetFileInformationByHandle(handle, ctypes.byref(information)):
            _raise_windows_error(get_last_error())
        attributes = int(information.file_attributes)
        if attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
            raise WindowsWorkspaceError("unsafe")
        is_directory = bool(attributes & _FILE_ATTRIBUTE_DIRECTORY)
        if expect_directory and not is_directory:
            raise WindowsWorkspaceError("not_directory")
        if not expect_directory and is_directory:
            raise WindowsWorkspaceError("not_regular")
        final_path = _final_path(kernel32, get_last_error, handle)
        size_bytes = (int(information.file_size_high) << 32) | int(information.file_size_low)
        return _WindowsHandle(
            value=handle,
            final_path=final_path,
            attributes=attributes,
            size_bytes=size_bytes,
        )
    except Exception:
        _close_handle(handle)
        raise


def _final_path(kernel32: Any, get_last_error: Any, handle: int) -> str:
    capacity = 32_768
    buffer = ctypes.create_unicode_buffer(capacity)
    length = kernel32.GetFinalPathNameByHandleW(handle, buffer, capacity, 0)
    if length == 0:
        _raise_windows_error(get_last_error())
    if length >= capacity:
        capacity = int(length) + 1
        buffer = ctypes.create_unicode_buffer(capacity)
        length = kernel32.GetFinalPathNameByHandleW(handle, buffer, capacity, 0)
        if length == 0 or length >= capacity:
            _raise_windows_error(get_last_error())
    return buffer.value


def _require_contained(root_path: str, candidate_path: str) -> None:
    root = os.path.normcase(os.path.normpath(_strip_extended_prefix(root_path)))
    candidate = os.path.normcase(os.path.normpath(_strip_extended_prefix(candidate_path)))
    try:
        common_path = os.path.commonpath((root, candidate))
    except ValueError as exc:
        raise WindowsWorkspaceError("unsafe") from exc
    if common_path != root:
        raise WindowsWorkspaceError("unsafe")


def _strip_extended_prefix(path: str) -> str:
    if path.startswith("\\\\?\\UNC\\"):
        return "\\\\" + path[8:]
    if path.startswith("\\\\?\\"):
        return path[4:]
    return path


def _raise_windows_error(error_code: int) -> None:
    if error_code in {_ERROR_FILE_NOT_FOUND, _ERROR_PATH_NOT_FOUND}:
        raise WindowsWorkspaceError("missing")
    if error_code in {_ERROR_ACCESS_DENIED, _ERROR_SHARING_VIOLATION, _ERROR_LOCK_VIOLATION}:
        raise WindowsWorkspaceError("denied")
    raise WindowsWorkspaceError("unsafe")


def _close_handles(handles: list[_WindowsHandle]) -> None:
    for handle in reversed(handles):
        _close_handle(handle.value)


def _close_handle(handle: int) -> None:
    kernel32, _ = _windows_api()
    kernel32.CloseHandle(handle)


@lru_cache(maxsize=1)
def _windows_api() -> tuple[Any, Any]:
    if os.name != "nt":
        raise RuntimeError("Win32 workspace access is only available on Windows.")

    from ctypes import WinDLL, get_last_error

    kernel32 = WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.GetFileInformationByHandle.argtypes = [wintypes.HANDLE, wintypes.LPVOID]
    kernel32.GetFileInformationByHandle.restype = wintypes.BOOL
    kernel32.GetFinalPathNameByHandleW.argtypes = [
        wintypes.HANDLE,
        wintypes.LPWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
    ]
    kernel32.GetFinalPathNameByHandleW.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    return kernel32, get_last_error
