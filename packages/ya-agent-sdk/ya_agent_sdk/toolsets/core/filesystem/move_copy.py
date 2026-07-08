"""Move and Copy tools for file operations."""

from typing import Annotated, cast

from pydantic import Field
from pydantic_ai import RunContext
from ya_agent_environment import FileOperator

from ya_agent_sdk._logger import get_logger
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.events import FileChange, FileChangeAction, FileChangeEvent
from ya_agent_sdk.toolsets.core.base import BaseTool
from ya_agent_sdk.toolsets.core.filesystem._types import CopyResult, MoveResult, PathPair

logger = get_logger(__name__)


class MoveTool(BaseTool):
    """Tool for moving files and directories."""

    name = "move"
    description = "Move files or directories. Supports batch operations with multiple src/dst pairs."
    superseded_by_tags = frozenset({"shell"})

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        """Check if tool is available (requires file_operator)."""
        if ctx.deps.file_operator is None:
            logger.debug("MoveTool unavailable: file_operator is not configured")
            return False
        return True

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

        changes = [
            FileChange(path=r["src"], action=FileChangeAction.moved, destination=r["dst"])
            for r in results
            if r["success"]
        ]
        if changes:
            await ctx.deps.emit_event(
                FileChangeEvent(
                    event_id=f"file-change-{ctx.deps.run_id[:8]}",
                    changes=changes,
                    tool_name="move",
                )
            )

        return results


class CopyTool(BaseTool):
    """Tool for copying files."""

    name = "copy"
    description = "Copy files. Supports batch operations with multiple src/dst pairs. Files only."
    superseded_by_tags = frozenset({"shell"})

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        """Check if tool is available (requires file_operator)."""
        if ctx.deps.file_operator is None:
            logger.debug("CopyTool unavailable: file_operator is not configured")
            return False
        return True

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

        changes = [
            FileChange(path=r["src"], action=FileChangeAction.copied, destination=r["dst"])
            for r in results
            if r["success"]
        ]
        if changes:
            await ctx.deps.emit_event(
                FileChangeEvent(
                    event_id=f"file-change-{ctx.deps.run_id[:8]}",
                    changes=changes,
                    tool_name="copy",
                )
            )

        return results


__all__ = ["CopyTool", "MoveTool"]
