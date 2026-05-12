"""Type definitions for environment module."""

from typing import TypedDict


class FileStat(TypedDict):
    """File status information."""

    size: int
    """File size in bytes."""
    mtime: float
    """Modification time as Unix timestamp."""
    is_file: bool
    """True if path is a regular file."""
    is_dir: bool
    """True if path is a directory."""


class TruncatedResult(TypedDict):
    """Result from truncate_to_tmp operation."""

    content: str
    """The truncated content."""
    file_path: str
    """Path to the full content file in tmp directory."""
    message: str
    """Message indicating truncation occurred."""
