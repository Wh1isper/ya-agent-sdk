"""Session-backed async subagent tools for YA Claw runtime."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Protocol, runtime_checkable

from pydantic import Field
from pydantic_ai import RunContext
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.toolsets.core.base import BaseTool

from ya_claw.toolsets.session import CLAW_SELF_CLIENT_KEY


@runtime_checkable
class AsyncSubagentClient(Protocol):
    session_id: str
    run_id: str

    async def spawn_delegate(
        self,
        *,
        subagent_name: str,
        prompt: str,
        name: str | None,
        context: dict[str, Any] | None,
    ) -> dict[str, Any]: ...

    async def list_async_subagents(self, *, include_terminal: bool) -> dict[str, Any]: ...

    async def get_async_subagent(self, *, name_or_task_id: str) -> dict[str, Any]: ...

    async def steer_async_subagent(
        self,
        *,
        name_or_task_id: str,
        prompt: str | None,
        input_parts: list[dict[str, Any]] | None,
    ) -> dict[str, Any]: ...

    async def cancel_async_subagent(self, *, name_or_task_id: str, reason: str | None) -> dict[str, Any]: ...


class SpawnDelegateTool(BaseTool):
    """Spawn or resume a named session-backed async subagent."""

    name = "spawn_delegate"
    blocks_async_subagent = True
    description = "Spawn or resume a named YA Claw async subagent session. Returns immediately."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        return ctx.deps.agent_id == "main" and _get_client(ctx) is not None

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        if not self.is_available(ctx):
            return None
        return _async_subagent_instructions()

    async def call(
        self,
        ctx: RunContext[AgentContext],
        subagent_name: Annotated[str, Field(description="Configured subagent type to run")],
        prompt: Annotated[str, Field(description="Prompt to send to the async subagent")],
        name: Annotated[str | None, Field(description="Stable parent-session-local async subagent name")] = None,
        context: Annotated[
            dict[str, Any] | None, Field(description="Optional structured context for the child")
        ] = None,
    ) -> str:
        client = _get_client(ctx)
        if client is None:
            return "Error: YA Claw async subagent client is unavailable."
        try:
            payload = await client.spawn_delegate(
                subagent_name=subagent_name,
                prompt=prompt,
                name=name,
                context=context,
            )
            return _format_spawn_response(payload)
        except Exception as exc:
            return f"Error: {exc}"


class ListAsyncSubagentsTool(BaseTool):
    name = "list_async_subagents"
    blocks_async_subagent = True
    description = "List async subagents for the current YA Claw parent session."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        return ctx.deps.agent_id == "main" and _get_client(ctx) is not None

    async def call(
        self,
        ctx: RunContext[AgentContext],
        include_terminal: Annotated[
            bool, Field(description="Include completed, failed, and cancelled async subagents")
        ] = True,
    ) -> str:
        client = _get_client(ctx)
        if client is None:
            return "Error: YA Claw async subagent client is unavailable."
        try:
            return _dump_json(await client.list_async_subagents(include_terminal=include_terminal))
        except Exception as exc:
            return f"Error: {exc}"


class GetAsyncSubagentTool(BaseTool):
    name = "get_async_subagent"
    blocks_async_subagent = True
    description = "Get async subagent metadata, result summary, child session, latest run, and trace references."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        return ctx.deps.agent_id == "main" and _get_client(ctx) is not None

    async def call(
        self,
        ctx: RunContext[AgentContext],
        name: Annotated[str, Field(description="Async subagent name or task id")],
    ) -> str:
        client = _get_client(ctx)
        if client is None:
            return "Error: YA Claw async subagent client is unavailable."
        try:
            return _dump_json(await client.get_async_subagent(name_or_task_id=name))
        except Exception as exc:
            return f"Error: {exc}"


class SteerAsyncSubagentTool(BaseTool):
    name = "steer_async_subagent"
    blocks_async_subagent = True
    description = "Send input to a running async subagent. Queued or idle children return status and instructions."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        return ctx.deps.agent_id == "main" and _get_client(ctx) is not None

    async def call(
        self,
        ctx: RunContext[AgentContext],
        name: Annotated[str, Field(description="Async subagent name or task id")],
        prompt: Annotated[str | None, Field(description="Text steering prompt")] = None,
        input_parts: Annotated[
            list[dict[str, Any]] | None, Field(description="Optional raw YA Claw input parts")
        ] = None,
    ) -> str:
        client = _get_client(ctx)
        if client is None:
            return "Error: YA Claw async subagent client is unavailable."
        try:
            return _dump_json(
                await client.steer_async_subagent(name_or_task_id=name, prompt=prompt, input_parts=input_parts)
            )
        except Exception as exc:
            return f"Error: {exc}"


class CancelAsyncSubagentTool(BaseTool):
    name = "cancel_async_subagent"
    blocks_async_subagent = True
    description = "Request cancellation for a queued or running async subagent."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        return ctx.deps.agent_id == "main" and _get_client(ctx) is not None

    async def call(
        self,
        ctx: RunContext[AgentContext],
        name: Annotated[str, Field(description="Async subagent name or task id")],
        reason: Annotated[str | None, Field(description="Optional cancellation reason")] = None,
    ) -> str:
        client = _get_client(ctx)
        if client is None:
            return "Error: YA Claw async subagent client is unavailable."
        try:
            return _dump_json(await client.cancel_async_subagent(name_or_task_id=name, reason=reason))
        except Exception as exc:
            return f"Error: {exc}"


@lru_cache(maxsize=1)
def _async_subagent_instructions() -> str:
    path = Path(__file__).with_name("async_subagent_instructions.md")
    return path.read_text(encoding="utf-8").strip()


def _get_client(ctx: RunContext[AgentContext]) -> AsyncSubagentClient | None:
    if ctx.deps.resources is None:
        return None
    resource = ctx.deps.resources.get(CLAW_SELF_CLIENT_KEY)
    if isinstance(resource, AsyncSubagentClient):
        return resource
    return None


def _format_spawn_response(payload: dict[str, Any]) -> str:
    task = payload.get("task") if isinstance(payload, dict) else None
    if not isinstance(task, dict):
        return _dump_json(payload)
    attrs = {
        "task-id": task.get("task_id"),
        "name": task.get("name"),
        "session-id": task.get("task_session_id"),
        "run-id": task.get("task_run_id"),
        "subagent-name": task.get("subagent_name"),
        "status": task.get("status"),
    }
    attr_text = " ".join(
        f'{key}="{_xml_escape(value)}"' for key, value in attrs.items() if isinstance(value, str) and value
    )
    instruction = task.get("instruction")
    delivery = task.get("delivery")
    body = (
        instruction
        if isinstance(instruction, str) and instruction
        else "Result will wake the parent session when the child run completes."
    )
    if isinstance(delivery, str):
        body = f"Delivery: {delivery}. {body}"
    return f"<async-subagent {attr_text}>\n{_xml_escape(body)}\n</async-subagent>"


def _dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _xml_escape(value: object) -> str:
    return str(value).replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
