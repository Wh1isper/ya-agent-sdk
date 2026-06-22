"""List tool for directory listing."""

import fnmatch
import json
import uuid
from functools import cache
from pathlib import Path
from typing import Annotated, Any, cast

import anyio
from pydantic import Field
from pydantic_ai import RunContext
from ya_agent_environment import FileOperator

from ya_agent_sdk._logger import get_logger
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.toolsets.core.base import BaseTool
from ya_agent_sdk.toolsets.core.filesystem._types import FileInfoWithStats

logger = get_logger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"
DEFAULT_MAX_RESULTS = 500
_METADATA_CONCURRENCY_LIMIT = 32
OUTPUT_TRUNCATE_LIMIT = 20000


def _child_path(parent_path: str, name: str) -> str:
    return f"{parent_path}/{name}" if parent_path != "." else name


def _should_ignore(name: str, ignore: list[str] | None) -> bool:
    if not ignore:
        return False
    return any(fnmatch.fnmatch(name, pattern) for pattern in ignore)


async def _build_entry(
    file_operator: FileOperator,
    parent_path: str,
    name: str,
    is_dir: bool,
) -> FileInfoWithStats:
    item_path = _child_path(parent_path, name)
    file_info: FileInfoWithStats = {
        "name": name,
        "path": item_path,
        "type": "directory" if is_dir else "file",
    }

    if not is_dir:
        try:
            stat = await file_operator.stat(item_path)
            file_info["size"] = stat["size"]
            file_info["modified"] = stat["mtime"]
        except Exception as e:
            file_info["error"] = f"Failed to get file stats: {e!s}"

    return file_info


async def _build_entries_from_typed_items(
    file_operator: FileOperator,
    parent_path: str,
    items: list[tuple[str, bool]],
) -> list[FileInfoWithStats]:
    entries: list[FileInfoWithStats] = []

    async def fill_entry(
        target_entries: list[FileInfoWithStats | None],
        index: int,
        name: str,
        is_dir: bool,
    ) -> None:
        target_entries[index] = await _build_entry(file_operator, parent_path, name, is_dir)

    for start in range(0, len(items), _METADATA_CONCURRENCY_LIMIT):
        batch = items[start : start + _METADATA_CONCURRENCY_LIMIT]
        batch_entries: list[FileInfoWithStats | None] = [None] * len(batch)

        async with anyio.create_task_group() as task_group:
            for index, (name, is_dir) in enumerate(batch):
                task_group.start_soon(fill_entry, batch_entries, index, name, is_dir)

        entries.extend(entry for entry in batch_entries if entry is not None)

    return entries


async def _build_entries_from_names(
    file_operator: FileOperator,
    parent_path: str,
    names: list[str],
) -> list[FileInfoWithStats]:
    entries: list[FileInfoWithStats] = []
    for name in names:
        item_path = _child_path(parent_path, name)
        is_dir = await file_operator.is_dir(item_path)
        entries.append(await _build_entry(file_operator, parent_path, name, is_dir))
    return entries


def _build_success_response(
    path: str,
    entries: list[FileInfoWithStats],
    total_entries: int,
) -> dict[str, Any]:
    response: dict[str, Any] = {
        "path": path,
        "entries": entries,
        "count": len(entries),
        "success": True,
    }

    if total_entries > len(entries):
        response["truncated"] = True
        response["total_entries"] = total_entries
        response["showing"] = len(entries)
        response["note"] = (
            f"Results truncated: showing {len(entries)} of {total_entries} entries. "
            "Use ignore patterns or increase max_results."
        )

    return response


async def _guard_output_size(
    result: dict[str, Any],
    file_operator: FileOperator,
) -> dict[str, Any]:
    serialized = json.dumps(result, ensure_ascii=False)
    if len(serialized) <= OUTPUT_TRUNCATE_LIMIT:
        return result

    output_path: str | None = None
    try:
        output_file = f"ls-{uuid.uuid4().hex[:12]}.json"
        output_path = await file_operator.write_tmp_file(output_file, serialized)
    except Exception:
        logger.warning("Failed to write ls output to temp file", exc_info=True)

    entries = result.get("entries", [])
    total_entries = result.get("total_entries", result.get("count", len(entries)))
    existing_note = result.get("note")
    if output_path is not None:
        note = f"Output too large ({len(serialized)} chars). Full response saved to `output_file_path`."
    else:
        note = f"Output too large ({len(serialized)} chars). Failed to save temp file; showing truncated preview."
    if existing_note:
        note = f"{existing_note} {note}"

    preview: dict[str, Any] = {
        "path": result["path"],
        "entries": [],
        "count": 0,
        "success": True,
        "truncated": True,
        "total_entries": total_entries,
        "showing": 0,
        "note": note,
    }
    if output_path is not None:
        preview["output_file_path"] = output_path

    for entry in entries:
        candidate_entries = preview["entries"] + [entry]
        candidate = {
            **preview,
            "entries": candidate_entries,
            "count": len(candidate_entries),
            "showing": len(candidate_entries),
        }
        if len(json.dumps(candidate, ensure_ascii=False)) > OUTPUT_TRUNCATE_LIMIT:
            break
        preview = candidate

    return preview


@cache
def _load_instruction() -> str:
    """Load ls instruction from prompts/ls.md."""
    prompt_file = _PROMPTS_DIR / "ls.md"
    return prompt_file.read_text()


class ListTool(BaseTool):
    """Tool for listing files and directories."""

    name = "ls"
    description = "List directory contents with file info (name, type, size, modified time)."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        """Check if tool is available (requires file_operator)."""
        if ctx.deps.file_operator is None:
            logger.debug("ListTool unavailable: file_operator is not configured")
            return False
        return True

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        """Load instruction from prompts/ls.md."""
        return _load_instruction()

    async def call(
        self,
        ctx: RunContext[AgentContext],
        path: Annotated[
            str,
            Field(description="Directory relative path"),
        ],
        ignore: Annotated[
            list[str] | None,
            Field(default=None, description="Glob patterns to ignore"),
        ] = None,
        max_results: Annotated[
            int,
            Field(
                description=f"Maximum entries to return (default: {DEFAULT_MAX_RESULTS}). Use -1 for unlimited.",
                default=DEFAULT_MAX_RESULTS,
                ge=-1,
            ),
        ] = DEFAULT_MAX_RESULTS,
    ) -> dict[str, Any]:
        """List directory contents with detailed information."""
        file_operator = cast(FileOperator, ctx.deps.file_operator)

        if not await file_operator.exists(path):
            return {"success": False, "error": f"Directory not found: {path}"}

        if not await file_operator.is_dir(path):
            return {"success": False, "error": f"Path is not a directory: {path}"}

        try:
            if ignore or max_results >= 0:
                names: list[str] = []
                total_entries = 0
                for name in await file_operator.list_dir(path):
                    if _should_ignore(name, ignore):
                        continue
                    total_entries += 1
                    if max_results < 0 or len(names) < max_results:
                        names.append(name)
                entries = await _build_entries_from_names(file_operator, path, names)
            else:
                items = await file_operator.list_dir_with_types(path)
                total_entries = len(items)
                entries = await _build_entries_from_typed_items(file_operator, path, items)

        except Exception as e:
            return {"success": False, "error": f"Failed to list directory: {e!s}"}

        response = _build_success_response(path, entries, total_entries)
        return await _guard_output_size(response, file_operator)


__all__ = ["ListTool"]
