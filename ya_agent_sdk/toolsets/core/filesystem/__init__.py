"""Filesystem-related tools.

Tools for file and directory operations.
"""

from ya_agent_sdk.toolsets.core.base import BaseTool
from ya_agent_sdk.toolsets.core.filesystem._types import (
    BatchMkdirResponse,
    CopyResult,
    EditItem,
    FileInfo,
    FileInfoWithStats,
    GrepMatch,
    MkdirResult,
    MkdirSummary,
    MoveResult,
    PathPair,
    ViewMetadata,
    ViewReadingParams,
    ViewSegment,
    ViewTruncationInfo,
)
from ya_agent_sdk.toolsets.core.filesystem.edit import EditTool, MultiEditTool
from ya_agent_sdk.toolsets.core.filesystem.glob import GlobTool
from ya_agent_sdk.toolsets.core.filesystem.grep import GrepTool
from ya_agent_sdk.toolsets.core.filesystem.ls import ListTool
from ya_agent_sdk.toolsets.core.filesystem.mkdir import MkdirTool
from ya_agent_sdk.toolsets.core.filesystem.move_copy import CopyTool, MoveTool
from ya_agent_sdk.toolsets.core.filesystem.replace import ReplaceTool
from ya_agent_sdk.toolsets.core.filesystem.view import ViewTool

tools: list[type[BaseTool]] = [
    GlobTool,
    GrepTool,
    ListTool,
    ViewTool,
    EditTool,
    MultiEditTool,
    ReplaceTool,
    MkdirTool,
    MoveTool,
    CopyTool,
]

__all__ = [
    "BatchMkdirResponse",
    "CopyResult",
    "CopyTool",
    "EditItem",
    "EditTool",
    "FileInfo",
    "FileInfoWithStats",
    "GlobTool",
    "GrepMatch",
    "GrepTool",
    "ListTool",
    "MkdirResult",
    "MkdirSummary",
    "MkdirTool",
    "MoveResult",
    "MoveTool",
    "MultiEditTool",
    "PathPair",
    "ReplaceTool",
    "ViewMetadata",
    "ViewReadingParams",
    "ViewSegment",
    "ViewTool",
    "ViewTruncationInfo",
    "tools",
]
