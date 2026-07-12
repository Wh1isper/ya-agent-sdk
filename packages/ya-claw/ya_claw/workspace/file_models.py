from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

WorkspaceFileKind = Literal["file", "directory", "symlink", "other"]


class WorkspaceFileEntry(BaseModel):
    """A single entry addressed only by its workspace virtual path."""

    name: str
    path: str
    kind: WorkspaceFileKind
    size_bytes: int | None = None
    modified_at: datetime | None = None
    hidden: bool = False


class WorkspaceFileListResponse(BaseModel):
    session_id: str
    path: str
    items: list[WorkspaceFileEntry] = Field(default_factory=list)
    limit: int
    offset: int = 0
    has_more: bool = False
    next_cursor: str | None = None
    # Kept for backwards compatibility with offset-based clients.
    next_offset: int | None = None
    # Kept for backwards compatibility. Equivalent to ``has_more``.
    truncated: bool = False


class WorkspaceTextFileResponse(BaseModel):
    session_id: str
    path: str
    content: str
    encoding: Literal["utf-8"] = "utf-8"
    size_bytes: int
