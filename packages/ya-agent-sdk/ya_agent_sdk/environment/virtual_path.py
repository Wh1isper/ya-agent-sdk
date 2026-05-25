"""Agent-facing virtual path helpers.

Virtual paths are paths presented to agents, containers, and remote backends.
They use POSIX semantics on every host platform, while host filesystem paths
continue to use pathlib.Path.
"""

import posixpath
from pathlib import PurePath, PurePosixPath

VirtualPath = PurePosixPath
VirtualPathLike = str | PurePath


def as_virtual_path(path: VirtualPathLike) -> VirtualPath:
    """Coerce an agent-facing path to POSIX separators."""
    return VirtualPath(str(path).replace("\\", "/"))


def normalize_virtual_path(path: VirtualPathLike) -> VirtualPath:
    """Normalize an agent-facing path without touching the host filesystem."""
    return VirtualPath(posixpath.normpath(as_virtual_path(path).as_posix()))


def is_virtual_path_relative_to(path: VirtualPathLike, root: VirtualPathLike) -> bool:
    """Return whether an agent-facing path is equal to or inside root."""
    normalized_path = normalize_virtual_path(path)
    normalized_root = normalize_virtual_path(root)
    try:
        normalized_path.relative_to(normalized_root)
        return True
    except ValueError:
        return False


def relative_virtual_path(path: VirtualPathLike, root: VirtualPathLike) -> VirtualPath:
    """Return path relative to root using POSIX virtual path semantics."""
    return normalize_virtual_path(path).relative_to(normalize_virtual_path(root))
