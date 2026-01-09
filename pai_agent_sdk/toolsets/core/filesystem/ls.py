"""List tool for directory listing."""

import fnmatch
from functools import cache
from pathlib import Path
from typing import Annotated, Any, cast

from pydantic import Field
from pydantic_ai import RunContext

from pai_agent_sdk._logger import get_logger
from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.environment.base import FileOperator
from pai_agent_sdk.toolsets.core.base import BaseTool
from pai_agent_sdk.toolsets.core.filesystem._types import FileInfoWithStats

logger = get_logger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"


@cache
def _load_instruction() -> str:
    """Load ls instruction from prompts/ls.md."""
    prompt_file = _PROMPTS_DIR / "ls.md"
    return prompt_file.read_text()


class ListTool(BaseTool):
    """Tool for listing files and directories."""

    name = "ls"
    description = "List directory contents with file info (name, type, size, modified time)."

    def is_available(self) -> bool:
        """Check if tool is available (requires file_operator)."""
        if self.ctx.file_operator is None:
            logger.debug("ListTool unavailable: file_operator is not configured")
            return False
        return True

    def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
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
    ) -> dict[str, Any]:
        """List directory contents with detailed information."""
        file_operator = cast(FileOperator, ctx.deps.file_operator)

        if not await file_operator.exists(path):
            return {"success": False, "error": f"Directory not found: {path}"}

        if not await file_operator.is_dir(path):
            return {"success": False, "error": f"Path is not a directory: {path}"}

        def should_ignore(name: str) -> bool:
            if not ignore:
                return False
            return any(fnmatch.fnmatch(name, pattern) for pattern in ignore)

        entries: list[FileInfoWithStats] = []

        try:
            items = await file_operator.list_dir(path)

            for name in items:
                if should_ignore(name):
                    continue

                item_path = f"{path}/{name}" if path != "." else name
                is_dir = await file_operator.is_dir(item_path)

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

                entries.append(file_info)

        except Exception as e:
            return {"success": False, "error": f"Failed to list directory: {e!s}"}

        return {
            "path": path,
            "entries": entries,
            "count": len(entries),
            "success": True,
        }


__all__ = ["ListTool"]
