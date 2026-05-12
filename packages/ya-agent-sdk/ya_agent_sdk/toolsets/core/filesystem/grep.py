"""Grep tool for content search."""

import json
import re
import uuid
from functools import cache
from pathlib import Path
from typing import Annotated, Any, cast

from pydantic import Field
from pydantic_ai import RunContext
from ya_agent_environment import FileOperator

from ya_agent_sdk._logger import get_logger
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.toolsets.core.base import BaseTool
from ya_agent_sdk.toolsets.core.filesystem._gitignore import filter_gitignored
from ya_agent_sdk.toolsets.core.filesystem._types import GrepMatch
from ya_agent_sdk.toolsets.core.filesystem._utils import is_binary_file

logger = get_logger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"
# Threshold to trigger soft truncation (drop context, limit line length)
_TRUNCATION_THRESHOLD = 30000
# Hard output size limit (aligned with glob) -- write to temp file if exceeded
_OUTPUT_HARD_LIMIT = 20000
# Max matching_line length in truncated output
_TRUNCATED_LINE_MAX = 300


@cache
def _load_instruction() -> str:
    """Load grep instruction from prompts/grep.md."""
    prompt_file = _PROMPTS_DIR / "grep.md"
    return prompt_file.read_text()


def _add_gitignore_info(results: dict[str, Any], gitignore_summary: list[str]) -> None:
    """Add gitignore exclusion info to results."""
    if gitignore_summary:
        results["<gitignore_excluded>"] = gitignore_summary
        note = "Some files excluded by .gitignore. Set include_ignored=true to include them."
        results.setdefault("<note>", "")
        if results["<note>"]:
            results["<note>"] += " "
        results["<note>"] += note


def _truncate_results(results: dict[str, Any]) -> dict[str, Any]:
    """Truncate results by dropping context and limiting matching_line length.

    Preserves all metadata keys (e.g. <skipped_large_files>, <note>, <gitignore_excluded>)
    from the original results.
    """
    logger.info("Results too long, dropping context")
    truncated: dict[str, Any] = {}
    for key, value in results.items():
        if key.startswith("<"):
            truncated[key] = value
        elif isinstance(value, dict) and "line_number" in value:
            matching_line = value["matching_line"]
            if len(matching_line) > _TRUNCATED_LINE_MAX:
                matching_line = matching_line[:_TRUNCATED_LINE_MAX] + "..."
            truncated[key] = {
                "file_path": value["file_path"],
                "line_number": value["line_number"],
                "matching_line": matching_line,
            }
    truncated["<system>"] = "Context dropped to reduce output size. Use `view` to read specific files."
    return truncated


async def _guard_output_size(
    results: dict[str, Any],
    file_operator: FileOperator,
) -> dict[str, Any]:
    """Ensure grep output stays within size limits.

    Two-phase approach:
    1. Soft truncation: drop context and limit matching_line length.
    2. Hard guard: write to temp file and return a bounded preview.
    """
    serialized = json.dumps(results, default=str, ensure_ascii=False)
    if len(serialized) <= _TRUNCATION_THRESHOLD:
        return results

    # Phase 1: soft truncation
    truncated = _truncate_results(results)
    serialized = json.dumps(truncated, default=str, ensure_ascii=False)
    if len(serialized) <= _OUTPUT_HARD_LIMIT:
        return truncated

    # Phase 2: write full truncated results to temp file, return bounded preview
    logger.info("Truncated results still too large (%d chars), writing to temp file", len(serialized))
    output_path: str | None = None
    try:
        output_file = f"grep-{uuid.uuid4().hex[:12]}.json"
        output_path = await file_operator.write_tmp_file(output_file, serialized)
    except Exception:
        logger.warning("Failed to write grep output to temp file", exc_info=True)

    # Extract match keys and metadata
    match_keys = [k for k in truncated if not k.startswith("<")]
    metadata = {k: v for k, v in truncated.items() if k.startswith("<")}

    # Build preview note
    if output_path is not None:
        system_msg = (
            f"Output too large ({len(serialized)} chars). Full results saved to temp file. Use `view` to read it."
        )
    else:
        system_msg = f"Output too large ({len(serialized)} chars). Failed to save temp file; showing truncated preview."

    # Build preview incrementally to guarantee it stays within the hard limit
    preview: dict[str, Any] = {**metadata}
    preview["<system>"] = system_msg
    preview["total_matches"] = len(match_keys)
    preview["showing"] = 0
    if output_path is not None:
        preview["output_file_path"] = output_path

    for key in match_keys:
        candidate = {**preview, key: truncated[key], "showing": preview["showing"] + 1}
        if len(json.dumps(candidate, default=str, ensure_ascii=False)) > _OUTPUT_HARD_LIMIT:
            break
        preview = candidate

    return preview


class GrepTool(BaseTool):
    """Tool for searching file contents using regular expressions."""

    name = "grep"
    description = "Search file contents using regex patterns. Returns matches with context lines."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        """Check if tool is available (requires file_operator)."""
        if ctx.deps.file_operator is None:
            logger.debug("GrepTool unavailable: file_operator is not configured")
            return False
        return True

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        """Load instruction from prompts/grep.md."""
        return _load_instruction()

    def _search_file_for_matches(
        self,
        file_path: str,
        content: str,
        regex_pattern: re.Pattern[str],
        context_lines: int,
        max_matches_per_file: int,
    ) -> dict[str, GrepMatch]:
        """Search a single file for regex matches."""
        matches: dict[str, GrepMatch] = {}
        file_matches = 0

        lines = content.splitlines(keepends=True)

        for i, line in enumerate(lines):
            if regex_pattern.search(line):
                if max_matches_per_file > 0 and file_matches >= max_matches_per_file:
                    break

                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)

                context = "".join(lines[start:end])

                match_data = GrepMatch(
                    file_path=file_path,
                    line_number=i + 1,
                    matching_line=line.rstrip("\n"),
                    context=context,
                    context_start_line=start + 1,
                )

                matches[f"{file_path}:{i + 1}"] = match_data
                file_matches += 1

        return matches

    async def _check_file_searchable(
        self,
        file_operator: FileOperator,
        file_path: str,
        max_file_size: int,
    ) -> str | None:
        """Check if a file is searchable. Returns skip reason or None if OK."""
        if await file_operator.is_dir(file_path):
            return "directory"

        if max_file_size > 0:
            try:
                stat = await file_operator.stat(file_path)
                if stat["size"] > max_file_size:
                    return "too_large"
            except Exception:
                logger.debug(f"Failed to stat file, skipping size check: {file_path}", exc_info=True)

        try:
            if await is_binary_file(file_operator, file_path):
                return "binary"
        except Exception:
            logger.debug(f"Failed to check binary status: {file_path}", exc_info=True)

        return None

    async def _search_files(
        self,
        file_operator: FileOperator,
        files: list[str],
        compiled_pattern: re.Pattern[str],
        context_lines: int,
        max_results: int,
        max_matches_per_file: int,
        max_file_size: int = 0,
    ) -> tuple[dict[str, Any], int, list[str]]:
        """Search files and return (results, match_count, skipped_large_files)."""
        results: dict[str, Any] = {}
        total_matches_found = 0
        skipped_large_files: list[str] = []

        for file_path in files:
            skip_reason = await self._check_file_searchable(file_operator, file_path, max_file_size)
            if skip_reason == "too_large":
                skipped_large_files.append(file_path)
                continue
            if skip_reason:
                continue

            try:
                content = await file_operator.read_file(file_path)
            except Exception as e:
                logger.warning(f"Failed to read file {file_path}: {e}")
                continue

            file_matches = self._search_file_for_matches(
                file_path, content, compiled_pattern, context_lines, max_matches_per_file
            )

            for match_key, match_data in file_matches.items():
                if max_results > 0 and total_matches_found >= max_results:
                    results["<system>"] = f"Hit global limit: {max_results} matches"
                    return results, total_matches_found, skipped_large_files

                results[match_key] = match_data
                total_matches_found += 1

        return results, total_matches_found, skipped_large_files

    async def call(
        self,
        ctx: RunContext[AgentContext],
        pattern: Annotated[str, Field(description="Regular expression pattern to search for")],
        include: Annotated[
            str,
            Field(description="Glob pattern to filter files (default: **/*)", default="**/*"),
        ] = "**/*",
        context_lines: Annotated[
            int,
            Field(description="Context lines before/after matches (default: 2)", default=2),
        ] = 2,
        max_results: Annotated[
            int,
            Field(description="Max total matches (default: 100, -1 for unlimited)", default=100),
        ] = 100,
        max_matches_per_file: Annotated[
            int,
            Field(description="Max matches per file (default: 20, -1 for unlimited)", default=20),
        ] = 20,
        max_files: Annotated[
            int,
            Field(description="Max files to search (default: 50, -1 for unlimited)", default=50),
        ] = 50,
        include_ignored: Annotated[
            bool,
            Field(description="Include files ignored by .gitignore (default: false)", default=False),
        ] = False,
    ) -> dict[str, Any] | str:
        """Search file contents using regular expressions."""
        file_operator = cast(FileOperator, ctx.deps.file_operator)

        try:
            compiled_pattern = re.compile(pattern, re.UNICODE)
        except re.error as e:
            return f"Error: Invalid regex pattern: {e}"

        files = await file_operator.glob(include)
        gitignore_summary: list[str] = []

        if not include_ignored:
            filter_result = await filter_gitignored(files, file_operator)
            files = filter_result.kept
            gitignore_summary = filter_result.get_ignored_summary(max_items=5)

        files_to_search = files[:max_files] if max_files > 0 else files

        max_file_size = ctx.deps.tool_config.grep_max_file_size

        results, total_matches, skipped_large_files = await self._search_files(
            file_operator,
            files_to_search,
            compiled_pattern,
            context_lines,
            max_results,
            max_matches_per_file,
            max_file_size,
        )

        logger.info(f"Total matches found: {total_matches}")
        if skipped_large_files:
            results["<skipped_large_files>"] = skipped_large_files
            results.setdefault("<note>", "")
            if results["<note>"]:
                results["<note>"] += " "
            results["<note>"] += (
                f"{len(skipped_large_files)} file(s) skipped due to size limit. "
                "Use shell `grep` command to search these files."
            )
        _add_gitignore_info(results, gitignore_summary)

        return await _guard_output_size(results, file_operator)


__all__ = ["GrepTool"]
