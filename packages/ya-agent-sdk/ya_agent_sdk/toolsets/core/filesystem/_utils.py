"""Shared utilities for filesystem tools."""

from ya_agent_environment import FileOperator

# Size of the initial chunk read for binary detection (same heuristic as GNU grep)
_BINARY_CHECK_BYTES = 8192


async def is_binary_file(file_operator: FileOperator, file_path: str) -> bool:
    """Detect binary files by checking for null bytes in the first 8KB.

    This follows the same heuristic used by GNU grep: if the file contains
    a null byte (\\x00), it is considered binary.
    """
    chunk = await file_operator.read_bytes(file_path, length=_BINARY_CHECK_BYTES)
    return b"\x00" in chunk
