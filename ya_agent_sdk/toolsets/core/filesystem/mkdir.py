"""Mkdir tool for creating directories."""

from functools import cache
from pathlib import Path
from typing import Annotated, cast

from pydantic import Field
from pydantic_ai import RunContext
from y_agent_environment import FileOperator

from ya_agent_sdk._logger import get_logger
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.toolsets.core.base import BaseTool
from ya_agent_sdk.toolsets.core.filesystem._types import BatchMkdirResponse, MkdirResult, MkdirSummary

logger = get_logger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"


@cache
def _load_instruction() -> str:
    """Load mkdir instruction from prompts/mkdir.md."""
    prompt_file = _PROMPTS_DIR / "mkdir.md"
    return prompt_file.read_text()


class MkdirTool(BaseTool):
    """Tool for creating directories."""

    name = "mkdir"
    description = "Create multiple directories in batch within the working directory."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        """Check if tool is available (requires file_operator)."""
        if ctx.deps.file_operator is None:
            logger.debug("MkdirTool unavailable: file_operator is not configured")
            return False
        return True

    def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        """Load instruction from prompts/mkdir.md."""
        return _load_instruction()

    async def call(
        self,
        ctx: RunContext[AgentContext],
        paths: Annotated[list[str], Field(description="List of directory paths to create")],
        parents: Annotated[bool, Field(description="Create intermediate directories as needed")] = False,
    ) -> BatchMkdirResponse:
        """Create multiple directories."""
        file_operator = cast(FileOperator, ctx.deps.file_operator)

        if not paths:
            return BatchMkdirResponse(
                success=False,
                message="Error: No paths provided for directory creation",
                results=[],
                summary=MkdirSummary(total=0, successful=0, failed=0),
            )

        results: list[MkdirResult] = []
        successful_count = 0
        failed_count = 0

        for path in paths:
            try:
                await file_operator.mkdir(path, parents=parents)
                results.append(
                    MkdirResult(
                        path=path,
                        success=True,
                        message="Successfully created directory",
                    )
                )
                successful_count += 1
            except Exception as e:
                results.append(
                    MkdirResult(
                        path=path,
                        success=False,
                        message=f"Error creating directory: {e!s}",
                    )
                )
                failed_count += 1

        return BatchMkdirResponse(
            success=failed_count == 0,
            message=f"Batch mkdir completed: {successful_count} successful, {failed_count} failed",
            results=results,
            summary=MkdirSummary(
                total=len(paths),
                successful=successful_count,
                failed=failed_count,
            ),
        )


__all__ = ["MkdirTool"]
