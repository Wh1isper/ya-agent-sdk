"""Move and Copy tools for file operations."""

from functools import cache
from pathlib import Path
from typing import Annotated, cast

from agent_environment import FileOperator
from pydantic import Field
from pydantic_ai import RunContext

from ya_agent_sdk._logger import get_logger
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.toolsets.core.base import BaseTool
from ya_agent_sdk.toolsets.core.filesystem._types import CopyResult, MoveResult, PathPair

logger = get_logger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"


@cache
def _load_move_instruction() -> str:
    """Load move instruction from prompts/move.md."""
    prompt_file = _PROMPTS_DIR / "move.md"
    return prompt_file.read_text()


@cache
def _load_copy_instruction() -> str:
    """Load copy instruction from prompts/copy.md."""
    prompt_file = _PROMPTS_DIR / "copy.md"
    return prompt_file.read_text()


class MoveTool(BaseTool):
    """Tool for moving files and directories."""

    name = "move"
    description = "Move files or directories. Supports batch operations with multiple src/dst pairs."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        """Check if tool is available (requires file_operator)."""
        if ctx.deps.file_operator is None:
            logger.debug("MoveTool unavailable: file_operator is not configured")
            return False
        return True

    def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        """Load instruction from prompts/move.md."""
        return _load_move_instruction()

    async def call(
        self,
        ctx: RunContext[AgentContext],
        pairs: Annotated[
            list[PathPair],
            Field(description="List of {src, dst} pairs to move"),
        ],
        overwrite: Annotated[
            bool,
            Field(description="Allow overwriting existing destinations", default=False),
        ] = False,
    ) -> list[MoveResult]:
        """Move files or directories."""
        file_operator = cast(FileOperator, ctx.deps.file_operator)
        results: list[MoveResult] = []

        for pair in pairs:
            src = pair["src"]
            dst = pair["dst"]

            try:
                if not await file_operator.exists(src):
                    results.append(MoveResult(src=src, dst=dst, success=False, message=f"Source not found: {src}"))
                    continue

                if await file_operator.exists(dst) and not overwrite:
                    results.append(
                        MoveResult(
                            src=src, dst=dst, success=False, message=f"Destination exists: {dst}. Set overwrite=True."
                        )
                    )
                    continue

                await file_operator.move(src, dst)
                results.append(MoveResult(src=src, dst=dst, success=True, message=f"Moved {src} to {dst}"))
            except Exception as e:
                results.append(MoveResult(src=src, dst=dst, success=False, message=f"Error: {e!s}"))

        return results


class CopyTool(BaseTool):
    """Tool for copying files."""

    name = "copy"
    description = "Copy files. Supports batch operations with multiple src/dst pairs. Files only."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        """Check if tool is available (requires file_operator)."""
        if ctx.deps.file_operator is None:
            logger.debug("CopyTool unavailable: file_operator is not configured")
            return False
        return True

    def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        """Load instruction from prompts/copy.md."""
        return _load_copy_instruction()

    async def call(
        self,
        ctx: RunContext[AgentContext],
        pairs: Annotated[
            list[PathPair],
            Field(description="List of {src, dst} pairs to copy"),
        ],
        overwrite: Annotated[
            bool,
            Field(description="Allow overwriting existing destinations", default=False),
        ] = False,
    ) -> list[CopyResult]:
        """Copy files."""
        file_operator = cast(FileOperator, ctx.deps.file_operator)
        results: list[CopyResult] = []

        for pair in pairs:
            src = pair["src"]
            dst = pair["dst"]

            try:
                if not await file_operator.exists(src):
                    results.append(CopyResult(src=src, dst=dst, success=False, message=f"Source not found: {src}"))
                    continue

                if not await file_operator.is_file(src):
                    results.append(
                        CopyResult(src=src, dst=dst, success=False, message=f"Source is not a file: {src}. Files only.")
                    )
                    continue

                if await file_operator.exists(dst) and not overwrite:
                    results.append(
                        CopyResult(
                            src=src, dst=dst, success=False, message=f"Destination exists: {dst}. Set overwrite=True."
                        )
                    )
                    continue

                await file_operator.copy(src, dst)
                results.append(CopyResult(src=src, dst=dst, success=True, message=f"Copied {src} to {dst}"))
            except Exception as e:
                results.append(CopyResult(src=src, dst=dst, success=False, message=f"Error: {e!s}"))

        return results


__all__ = ["CopyTool", "MoveTool"]
