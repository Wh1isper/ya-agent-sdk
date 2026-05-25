"""Glob tool for file pattern matching."""

import json
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
from ya_agent_sdk.toolsets.core.filesystem._gitignore import GitignoreFilterResult
from ya_agent_sdk.toolsets.core.filesystem._search import collect_glob_candidates

logger = get_logger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"

DEFAULT_MAX_RESULTS = 500
OUTPUT_TRUNCATE_LIMIT = 20000


@cache
def _load_instruction() -> str:
    """Load glob instruction from prompts/glob.md."""
    prompt_file = _PROMPTS_DIR / "glob.md"
    return prompt_file.read_text()


class GlobTool(BaseTool):
    """Tool for finding files matching ripgrep-style glob patterns."""

    name = "glob"
    description = "Find files by ripgrep-style glob pattern. Returns paths sorted by modification time (newest first)."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        """Check if tool is available (requires file_operator)."""
        if ctx.deps.file_operator is None:
            logger.debug("GlobTool unavailable: file_operator is not configured")
            return False
        return True

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        """Load instruction from prompts/glob.md."""
        return _load_instruction()

    async def call(
        self,
        ctx: RunContext[AgentContext],
        pattern: Annotated[
            str,
            Field(
                description=(
                    "Ripgrep-style glob pattern to match files and directories. "
                    "Bare patterns like '*.py' match recursively; leading '/' anchors to the FileOperator root."
                )
            ),
        ],
        root: Annotated[
            str,
            Field(description="Logical root to search from (default: .)", default="."),
        ] = ".",
        include_ignored: Annotated[
            bool,
            Field(
                description="Include files ignored by .gitignore and nested ignore files (default: false)",
                default=False,
            ),
        ] = False,
        include_hidden: Annotated[
            bool,
            Field(description="Include hidden dot paths such as .git, .venv, and .env (default: false)", default=False),
        ] = False,
        max_results: Annotated[
            int,
            Field(
                description=f"Maximum number of results to return (default: {DEFAULT_MAX_RESULTS}). Use -1 for unlimited.",
                default=DEFAULT_MAX_RESULTS,
                ge=-1,
            ),
        ] = DEFAULT_MAX_RESULTS,
    ) -> list[str] | dict[str, Any]:
        """Find files matching the given glob pattern."""
        file_operator = cast(FileOperator, ctx.deps.file_operator)
        candidates, filter_result = await collect_glob_candidates(
            file_operator,
            pattern,
            root=root,
            include_ignored=include_ignored,
            include_hidden=include_hidden,
        )
        files = [candidate.path for candidate in candidates]

        if include_ignored:
            result = _apply_max_results(files, max_results)
        else:
            result = _build_filtered_result(files, filter_result, max_results)

        return await _guard_output_size(result, file_operator)


def _build_filtered_result(
    files: list[str],
    filter_result: GitignoreFilterResult | None,
    max_results: int,
) -> list[str] | dict[str, Any]:
    """Build result with gitignore metadata and max_results applied."""
    total_count = len(files)
    effective_limit = max_results if max_results >= 0 else total_count
    truncated = total_count > effective_limit
    kept = files[:effective_limit] if truncated else files
    ignored = filter_result.ignored if filter_result is not None else []

    if not ignored and not truncated:
        return kept

    response: dict[str, Any] = {"files": kept}
    notes: list[str] = []

    if ignored and filter_result is not None:
        response["gitignore_excluded"] = filter_result.get_ignored_summary(max_items=5)
        notes.append("Some files excluded by .gitignore. Set include_ignored=true to include them.")

    if truncated:
        response["truncated"] = True
        response["total_matches"] = total_count
        response["showing"] = effective_limit
        notes.append(
            f"Results truncated: showing {effective_limit} of {total_count} matches. "
            "Use a more specific pattern or increase max_results."
        )

    response["note"] = " ".join(notes)
    return response


def _apply_max_results(files: list[str], max_results: int) -> list[str] | dict[str, Any]:
    """Apply max_results limit to a file list."""
    if max_results < 0 or len(files) <= max_results:
        return files
    return {
        "files": files[:max_results],
        "truncated": True,
        "total_matches": len(files),
        "showing": max_results,
        "note": (
            f"Results truncated: showing {max_results} of {len(files)} matches. "
            "Use a more specific pattern or increase max_results."
        ),
    }


async def _guard_output_size(
    result: list[str] | dict[str, Any],
    file_operator: FileOperator,
) -> list[str] | dict[str, Any]:
    """Write result to temp file if serialized output exceeds the hard size limit."""
    serialized = json.dumps(result, ensure_ascii=False)
    if len(serialized) <= OUTPUT_TRUNCATE_LIMIT:
        return result

    # Extract file list and total count
    if isinstance(result, list):
        all_files = result
        total = len(result)
    else:
        all_files = result.get("files", [])
        total = result.get("total_matches", len(all_files))

    # Write full result to temp file (with fallback on failure)
    output_path: str | None = None
    try:
        output_file = f"glob-{uuid.uuid4().hex[:12]}.json"
        output_path = await file_operator.write_tmp_file(output_file, serialized)
    except Exception:
        logger.warning("Failed to write glob output to temp file", exc_info=True)

    # Build preview with exact serialization check to guarantee within limit
    preview: dict[str, Any] = {
        "files": [],
        "truncated": True,
        "total_matches": total,
        "showing": 0,
        "note": "",
    }
    if output_path is not None:
        preview["output_file_path"] = output_path
        preview["note"] = f"Output too large ({len(serialized)} chars). Full results saved to `output_file_path`."
    else:
        preview["note"] = (
            f"Output too large ({len(serialized)} chars). Failed to save temp file; showing truncated preview."
        )

    for f in all_files:
        candidate_files = preview["files"] + [f]
        candidate = {**preview, "files": candidate_files, "showing": len(candidate_files)}
        if len(json.dumps(candidate, ensure_ascii=False)) > OUTPUT_TRUNCATE_LIMIT:
            break
        preview = candidate

    return preview


__all__ = ["GlobTool"]
