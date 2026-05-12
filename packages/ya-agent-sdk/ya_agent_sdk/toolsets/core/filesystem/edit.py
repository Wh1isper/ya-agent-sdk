"""Edit tools for file modification."""

from functools import cache
from pathlib import Path, PurePosixPath
from typing import Annotated, cast

from pydantic import Field
from pydantic_ai import RunContext
from ya_agent_environment import FileOperator

from ya_agent_sdk._logger import get_logger
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.events import FileChange, FileChangeAction, FileChangeEvent, TextReplacement
from ya_agent_sdk.toolsets.core.base import BaseTool
from ya_agent_sdk.toolsets.core.filesystem._types import EditItem
from ya_agent_sdk.toolsets.core.filesystem._utils import is_binary_file

_PROMPTS_DIR = Path(__file__).parent / "prompts"
logger = get_logger(__name__)


async def _check_edit_file_size(file_operator: FileOperator, file_path: str, max_size: int) -> str | None:
    """Check if file size is within the edit limit. Returns error message or None."""
    stat = await file_operator.stat(file_path)
    if stat["size"] > max_size:
        size_mb = stat["size"] / (1024 * 1024)
        limit_mb = max_size / (1024 * 1024)
        return (
            f"Error: File is too large to edit ({size_mb:.1f} MB). "
            f"Maximum supported size is {limit_mb:.0f} MB. "
            f"Use shell tools (e.g. `sed`) for large file operations."
        )
    return None


async def _validate_and_read_file(
    file_operator: FileOperator, file_path: str, max_size: int
) -> tuple[str, None] | tuple[None, str]:
    """Validate file (exists, not dir, size, binary) and read content.

    Returns (content, None) on success or (None, error_message) on failure.
    """
    if not await file_operator.exists(file_path):
        return None, f"Error: File not found: {file_path}"

    if await file_operator.is_dir(file_path):
        return None, f"Error: Path is a directory, not a file: {file_path}"

    if error := await _check_edit_file_size(file_operator, file_path, max_size):
        return None, error

    if await is_binary_file(file_operator, file_path):
        return None, f"Error: {file_path} appears to be a binary file. Edit tools only support text files."

    content = await file_operator.read_file(file_path)
    return content, None


@cache
def _load_edit_instruction() -> str:
    """Load edit instruction from prompts/edit.md."""
    prompt_file = _PROMPTS_DIR / "edit.md"
    return prompt_file.read_text()


@cache
def _load_multi_edit_instruction() -> str:
    """Load multi_edit instruction from prompts/multi_edit.md."""
    prompt_file = _PROMPTS_DIR / "multi_edit.md"
    return prompt_file.read_text()


class EditTool(BaseTool):
    """Tool for single find-and-replace edits."""

    name = "edit"
    description = "Performs exact string replacement in files. Use empty `old_string` to create new files."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        """Check if tool is available (requires file_operator)."""
        if ctx.deps.file_operator is None:
            logger.debug("EditTool unavailable: file_operator is not configured")
            return False
        return True

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        """Load instruction from prompts/edit.md."""
        return _load_edit_instruction()

    async def call(
        self,
        ctx: RunContext[AgentContext],
        file_path: Annotated[str, Field(description="Relative path to the file to edit")],
        old_string: Annotated[
            str,
            Field(description="Text to replace (exact match required). Empty string creates new file."),
        ],
        new_string: Annotated[str, Field(description="New text to replace the old text with")],
        replace_all: Annotated[
            bool,
            Field(
                default=False,
                description="Replace all occurrences. Default: false (replace first only).",
            ),
        ] = False,
    ) -> str:
        """Edit a file by performing a single find-and-replace operation."""
        file_operator = cast(FileOperator, ctx.deps.file_operator)

        if not old_string:
            if await file_operator.exists(file_path):
                return f"Error: File already exists: {file_path}. Use `write` tool to overwrite."
            # Auto-create parent directories if needed
            parent = str(PurePosixPath(file_path).parent)
            if parent and parent != ".":
                await file_operator.mkdir(parent, parents=True)
            await file_operator.write_file(file_path, new_string)
            await ctx.deps.emit_event(
                FileChangeEvent(
                    event_id=f"file-change-{ctx.deps.run_id[:8]}",
                    changes=[
                        FileChange(
                            path=file_path,
                            action=FileChangeAction.created,
                            replacements=[TextReplacement(old_string="", new_string=new_string)],
                        )
                    ],
                    tool_name="edit",
                )
            )
            return f"Successfully created new file: {file_path}"

        if not await file_operator.exists(file_path):
            return f"Error: File not found: {file_path}"

        content, error = await _validate_and_read_file(
            file_operator, file_path, ctx.deps.tool_config.edit_max_file_size
        )
        if error:
            return error
        content = cast(str, content)

        if old_string not in content:
            return "Error: Text not found. Ensure exact match including whitespace and indentation."

        if replace_all:
            content = content.replace(old_string, new_string)
        else:
            occurrences = content.count(old_string)
            if occurrences > 1:
                return f"Error: Text appears {occurrences} times. Add more context or use replace_all=true."
            content = content.replace(old_string, new_string, 1)

        await file_operator.write_file(file_path, content)
        await ctx.deps.emit_event(
            FileChangeEvent(
                event_id=f"file-change-{ctx.deps.run_id[:8]}",
                changes=[
                    FileChange(
                        path=file_path,
                        action=FileChangeAction.modified,
                        replacements=[TextReplacement(old_string=old_string, new_string=new_string)],
                    )
                ],
                tool_name="edit",
            )
        )
        return f"Successfully edited file: {file_path}"


class MultiEditTool(BaseTool):
    """Tool for multiple edits to a single file."""

    name = "multi_edit"
    description = "Perform multiple find-and-replace operations on a single file efficiently."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        """Check if tool is available (requires file_operator)."""
        if ctx.deps.file_operator is None:
            logger.debug("MultiEditTool unavailable: file_operator is not configured")
            return False
        return True

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        """Load instruction from prompts/multi_edit.md."""
        return _load_multi_edit_instruction()

    def _apply_edit(self, content: str, edit: EditItem, index: int) -> tuple[str, str | None]:
        """Apply a single edit to content. Returns (new_content, error_message)."""
        if edit.old_string not in content:
            return content, f"Error: Edit {index + 1}: Text not found. Ensure exact match."

        if edit.replace_all:
            return content.replace(edit.old_string, edit.new_string), None

        occurrences = content.count(edit.old_string)
        if occurrences > 1:
            return content, f"Error: Edit {index + 1}: Text appears {occurrences} times. Use replace_all=true."
        return content.replace(edit.old_string, edit.new_string, 1), None

    async def _load_or_create(
        self,
        file_operator: FileOperator,
        file_path: str,
        edits: list[EditItem],
        max_file_size: int,
    ) -> tuple[str, list[EditItem]] | str:
        """Load existing file content or create new file. Returns (content, remaining_edits) or error string."""
        if not edits[0].old_string:
            if await file_operator.exists(file_path):
                return f"Error: File already exists: {file_path}. Use `write` tool to overwrite."
            parent = str(PurePosixPath(file_path).parent)
            if parent and parent != ".":
                await file_operator.mkdir(parent, parents=True)
            return edits[0].new_string, edits[1:]

        if not await file_operator.exists(file_path):
            return f"Error: File not found: {file_path}"

        content, error = await _validate_and_read_file(file_operator, file_path, max_file_size)
        if error:
            return error
        content = cast(str, content)

        return content, edits

    async def call(
        self,
        ctx: RunContext[AgentContext],
        file_path: Annotated[str, Field(description="Relative path to the file to edit")],
        edits: Annotated[
            list[EditItem],
            Field(description="Array of edit operations to perform in sequence"),
        ],
    ) -> str:
        """Edit a file by performing multiple find-and-replace operations."""
        file_operator = cast(FileOperator, ctx.deps.file_operator)

        if not edits:
            return "Error: At least one edit operation must be provided."

        result = await self._load_or_create(file_operator, file_path, edits, ctx.deps.tool_config.edit_max_file_size)
        if isinstance(result, str):
            return result
        content, remaining_edits = result

        for i, edit in enumerate(remaining_edits):
            content, error = self._apply_edit(content, edit, i)
            if error:
                return error

        await file_operator.write_file(file_path, content)

        is_create = not edits[0].old_string
        await ctx.deps.emit_event(
            FileChangeEvent(
                event_id=f"file-change-{ctx.deps.run_id[:8]}",
                changes=[
                    FileChange(
                        path=file_path,
                        action=FileChangeAction.created if is_create else FileChangeAction.modified,
                        replacements=[TextReplacement(old_string=e.old_string, new_string=e.new_string) for e in edits],
                    )
                ],
                tool_name="multi_edit",
            )
        )

        if is_create:
            return f"Successfully created new file with {len(edits)} edits: {file_path}"
        return f"Successfully applied {len(edits)} edits to file: {file_path}"


__all__ = ["EditTool", "MultiEditTool"]
