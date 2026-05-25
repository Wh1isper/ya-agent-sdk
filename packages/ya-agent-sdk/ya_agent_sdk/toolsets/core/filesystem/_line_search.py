"""Streaming line-oriented search helpers for grep."""

from __future__ import annotations

import inspect
import re
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass

from ya_agent_environment import FileOperator

from ya_agent_sdk.toolsets.core.filesystem import _ripgrep_core
from ya_agent_sdk.toolsets.core.filesystem._types import GrepMatch


@dataclass(frozen=True)
class LineSearchResult:
    """Result from searching one file."""

    matches: dict[str, GrepMatch]
    binary_detected: bool = False


async def iter_text_lines(
    file_operator: FileOperator,
    file_path: str,
    *,
    encoding: str = "utf-8",
    chunk_size: int = 65536,
) -> AsyncIterator[str]:
    """Yield decoded lines from read_bytes_stream without loading the full file."""
    pending = ""
    stream = file_operator.read_bytes_stream(file_path, chunk_size=chunk_size)
    if inspect.isawaitable(stream):
        stream = await stream
    async for chunk in stream:
        text = chunk.decode(encoding, errors="replace")
        pending += text
        lines = pending.splitlines(keepends=True)
        pending = lines.pop() if lines and not (lines[-1].endswith("\n") or lines[-1].endswith("\r")) else ""
        for line in lines:
            yield line
    if pending:
        yield pending


async def search_file_streaming(
    file_operator: FileOperator,
    file_path: str,
    regex_pattern: re.Pattern[str],
    *,
    context_lines: int,
    max_matches_per_file: int,
    chunk_size: int = 65536,
) -> LineSearchResult:
    """Search a file through FileOperator.read_bytes_stream.

    The output shape matches the existing GrepMatch model. Context after a
    match is completed by buffering pending matches until enough following
    lines have been read.
    """
    matches: dict[str, GrepMatch] = {}
    native_regex: _ripgrep_core.NativeRegex | None = None
    if _ripgrep_core.is_available():
        native_regex = _ripgrep_core.NativeRegex(regex_pattern.pattern)
    before_context: deque[tuple[int, str]] = deque(maxlen=max(0, context_lines))
    pending_matches: list[tuple[int, str, list[tuple[int, str]]]] = []
    file_matches = 0

    async def _flush_ready(current_line_number: int, *, final: bool = False) -> None:
        ready: list[tuple[int, str, list[tuple[int, str]]]] = []
        waiting: list[tuple[int, str, list[tuple[int, str]]]] = []
        for match_line_number, matching_line, context_before in pending_matches:
            if final or current_line_number - match_line_number >= context_lines:
                ready.append((match_line_number, matching_line, context_before))
            else:
                waiting.append((match_line_number, matching_line, context_before))
        pending_matches[:] = waiting
        for match_line_number, matching_line, context_before in ready:
            context_start = max(1, match_line_number - context_lines)
            context_lines_out = [line for _, line in context_before]
            context_lines_out.append(matching_line)
            after_lines = [
                line
                for line_number, line in before_context
                if match_line_number < line_number <= match_line_number + context_lines
            ]
            context_lines_out.extend(after_lines)
            matches[f"{file_path}:{match_line_number}"] = GrepMatch(
                file_path=file_path,
                line_number=match_line_number,
                matching_line=matching_line.rstrip("\n"),
                context="".join(context_lines_out),
                context_start_line=context_start,
            )

    async for line_number, line in _enumerate_async(
        iter_text_lines(file_operator, file_path, chunk_size=chunk_size), 1
    ):
        await _flush_ready(line_number)
        matched = native_regex.search(line) if native_regex is not None else regex_pattern.search(line) is not None
        if matched and (max_matches_per_file <= 0 or file_matches < max_matches_per_file):
            pending_matches.append((line_number, line, list(before_context)))
            file_matches += 1
        before_context.append((line_number, line))

    await _flush_ready(0, final=True)
    return LineSearchResult(matches=matches)


async def _enumerate_async(iterator: AsyncIterator[str], start: int = 0) -> AsyncIterator[tuple[int, str]]:
    """Async enumerate helper."""
    index = start
    async for item in iterator:
        yield index, item
        index += 1


__all__ = ["LineSearchResult", "iter_text_lines", "search_file_streaming"]
