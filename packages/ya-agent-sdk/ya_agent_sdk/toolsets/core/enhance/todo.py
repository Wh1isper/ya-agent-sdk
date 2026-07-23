"""TO-DO list tools for task planning and tracking.

These tools allow the agent to manage a session-level to-do list
stored in the file operator's temporary directory.
"""

from pathlib import Path
from typing import Annotated, Literal

import pydantic
from pydantic import BaseModel, Field
from pydantic_ai import RunContext
from ya_agent_environment import FileOperator

from ya_agent_sdk._logger import get_logger
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.toolsets.core.base import BaseTool

logger = get_logger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _get_todo_file_name(run_id: str) -> str:
    """Generate TO-DO file name with run_id to distinguish subagent todos."""
    return f"TO-DO-{run_id}.json"


def _get_todo_storage(context: AgentContext) -> tuple[FileOperator, str] | None:
    """Resolve TO-DO storage, returning ``None`` when tmp storage is unavailable."""
    try:
        file_operator = context.file_operator
        if file_operator is None or context.tmp_dir is None:
            return None
        path = str(context.resolve_tmp_path(_get_todo_file_name(context.run_id)))
        return file_operator, path
    except Exception:
        return None


class TodoItem(BaseModel):
    """A single TO-DO item."""

    id: str = Field(..., description="Unique identifier (e.g., TASK-1).")
    content: str = Field(..., description="Task description.")
    status: Literal["pending", "in_progress", "completed"] = Field(
        ...,
        description="Task status.",
    )
    priority: Literal["high", "medium", "low"] = Field(..., description="Task priority.")


TodoItemsTypeAdapter = pydantic.TypeAdapter(
    list[TodoItem],
    config=pydantic.ConfigDict(defer_build=True, ser_json_bytes="base64", val_json_bytes="base64"),
)


class TodoReadTool(BaseTool):
    """Tool for reading the TO-DO list."""

    name = "to_do_read"
    description = "Read the current session's to-do list."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        """Check if tool is available (requires file_operator and tmp_dir)."""
        if _get_todo_storage(ctx.deps) is None:
            logger.debug("TodoReadTool unavailable: file_operator or tmp_dir is not configured")
            return False
        return True

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        """Get instruction for this tool."""
        instruction_file = _PROMPTS_DIR / "to_do_read.md"
        if instruction_file.exists():
            return instruction_file.read_text()
        return None

    async def call(
        self,
        ctx: RunContext[AgentContext],
    ) -> str:
        storage = _get_todo_storage(ctx.deps)
        if storage is None:
            return "No TO-DO file found"
        file_op, todo_file = storage

        try:
            if not await file_op.exists(todo_file):
                return "No TO-DO file found"

            file_content = await file_op.read_file(todo_file)
            if not file_content.strip():
                return "No TO-DOs found"

            # Validate and return JSON string for consistent parsing
            todos = TodoItemsTypeAdapter.validate_json(file_content)
            return TodoItemsTypeAdapter.dump_json(todos).decode("utf-8")

        except Exception:
            # Remove the file if it is corrupted
            try:
                if await file_op.exists(todo_file):
                    await file_op.delete(todo_file)
            except Exception:  # noqa: S110
                pass
            return "Error reading to_do file, please try again."


class TodoWriteTool(BaseTool):
    """Tool for writing/updating the TO-DO list."""

    name = "to_do_write"
    description = "Replace the session's to-do list with an updated list."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        """Check if tool is available (requires file_operator and tmp_dir)."""
        if _get_todo_storage(ctx.deps) is None:
            logger.debug("TodoWriteTool unavailable: file_operator or tmp_dir is not configured")
            return False
        return True

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        """Get instruction for this tool."""
        instruction_file = _PROMPTS_DIR / "to_do_write.md"
        if instruction_file.exists():
            return instruction_file.read_text()
        return None

    async def call(
        self,
        ctx: RunContext[AgentContext],
        to_dos: Annotated[list[TodoItem], Field(description="The updated TO-DO list.")],
    ) -> str:
        storage = _get_todo_storage(ctx.deps)
        if storage is None:
            return "Error writing to_do file: temporary storage is not configured, please try again."
        file_op, todo_file = storage

        try:
            if not to_dos:
                if await file_op.exists(todo_file):
                    await file_op.delete(todo_file)
                return "TO-DO list cleared successfully."

            # Write TO-DOs to file and return JSON string for consistent parsing
            to_do_json = TodoItemsTypeAdapter.dump_json(to_dos)
            await file_op.write_file(todo_file, to_do_json)
            return to_do_json.decode("utf-8")

        except Exception as e:
            # If there is an error, remove the file
            try:
                if await file_op.exists(todo_file):
                    await file_op.delete(todo_file)
            except Exception:  # noqa: S110
                pass
            return f"Error writing to_do file: {e}, please try again."
