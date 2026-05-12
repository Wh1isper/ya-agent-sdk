"""Delete tool for file operations."""

import posixpath
from functools import cache
from pathlib import Path
from typing import Annotated, cast

from pydantic import Field
from pydantic_ai import RunContext
from ya_agent_environment import FileOperator

from ya_agent_sdk._logger import get_logger
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.events import FileChange, FileChangeAction, FileChangeEvent
from ya_agent_sdk.toolsets.core.base import BaseTool
from ya_agent_sdk.toolsets.core.filesystem._types import DeleteResult

logger = get_logger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_PROTECTED_PATHS = frozenset({"", ".", "./", "..", "../", "/", "~"})


@cache
def _load_delete_instruction() -> str:
    """Load delete instruction from prompts/delete.md."""
    prompt_file = _PROMPTS_DIR / "delete.md"
    return prompt_file.read_text()


def _normalize_for_protection(path: str) -> str:
    """Normalize a path string for protected path checks."""
    stripped = path.strip()
    if stripped in {"", "/"}:
        return stripped
    return stripped.rstrip("/")


def _is_protected_path(path: str) -> bool:
    """Return true for paths that represent workspace/root anchors."""
    normalized = _normalize_for_protection(path)
    return normalized in _PROTECTED_PATHS


def _is_operator_root_path(file_operator: FileOperator, path: str) -> bool:
    """Return true when path resolves to the operator's default root."""
    default_path = getattr(file_operator, "_default_path", None)
    if default_path is None:
        return False

    try:
        target = Path(path).expanduser()
        if not target.is_absolute():
            target = Path(default_path) / target
        return target.resolve() == Path(default_path).resolve()
    except OSError:
        return False


def _join_child_path(parent: str, child: str) -> str:
    """Join a child name to a virtual filesystem path."""
    if parent in {"", "."}:
        return child
    return posixpath.join(parent, child)


class DeleteTool(BaseTool):
    """Tool for deleting files and directories."""

    name = "delete"
    description = "Delete files or directories. Supports batch operations, recursive deletion, and force mode."
    superseded_by_tags = frozenset({"shell"})

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        """Check if tool is available (requires file_operator)."""
        if ctx.deps.file_operator is None:
            logger.debug("DeleteTool unavailable: file_operator is not configured")
            return False
        return True

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        """Load instruction from prompts/delete.md."""
        return _load_delete_instruction()

    async def call(
        self,
        ctx: RunContext[AgentContext],
        paths: Annotated[
            list[str],
            Field(description="List of file or directory paths to delete"),
        ],
        recursive: Annotated[
            bool,
            Field(description="Delete directories and their contents recursively, equivalent to rm -r", default=False),
        ] = False,
        force: Annotated[
            bool,
            Field(description="Ignore missing paths, equivalent to rm -f", default=False),
        ] = False,
    ) -> list[DeleteResult]:
        """Delete files or directories."""
        file_operator = cast(FileOperator, ctx.deps.file_operator)
        results: list[DeleteResult] = []

        deleted_paths: list[str] = []
        for path in paths:
            result, deleted = await self._delete_one(file_operator, path, recursive=recursive, force=force)
            results.append(result)
            if deleted:
                deleted_paths.append(path)

        changes = [FileChange(path=path, action=FileChangeAction.deleted) for path in deleted_paths]
        if changes:
            await ctx.deps.emit_event(
                FileChangeEvent(
                    event_id=f"file-change-{ctx.deps.run_id[:8]}",
                    changes=changes,
                    tool_name="delete",
                )
            )

        return results

    async def _delete_one(
        self,
        file_operator: FileOperator,
        path: str,
        *,
        recursive: bool,
        force: bool,
    ) -> tuple[DeleteResult, bool]:
        """Delete one path and return a structured result plus change flag."""
        try:
            if _is_protected_path(path) or _is_operator_root_path(file_operator, path):
                return (
                    DeleteResult(path=path, success=False, message=f"Refusing to delete protected path: {path}"),
                    False,
                )

            if not await file_operator.exists(path):
                if force:
                    return DeleteResult(path=path, success=True, message=f"Path missing, ignored: {path}"), False
                return DeleteResult(path=path, success=False, message=f"Path not found: {path}"), False

            if await file_operator.is_dir(path):
                await self._delete_directory(file_operator, path, recursive=recursive)
            else:
                await file_operator.delete(path)

            return DeleteResult(path=path, success=True, message=f"Deleted {path}"), True
        except Exception as e:
            return DeleteResult(path=path, success=False, message=f"Error: {e!s}"), False

    async def _delete_directory(
        self,
        file_operator: FileOperator,
        path: str,
        *,
        recursive: bool,
    ) -> None:
        """Delete a directory, optionally walking its contents first."""
        entries = await file_operator.list_dir_with_types(path)
        if entries and not recursive:
            raise ValueError(f"Directory is not empty: {path}. Set recursive=True.")

        if recursive:
            for name, is_dir in entries:
                child_path = _join_child_path(path, name)
                if is_dir:
                    await self._delete_directory(file_operator, child_path, recursive=True)
                else:
                    await file_operator.delete(child_path)

        await file_operator.delete(path)


__all__ = ["DeleteTool"]
