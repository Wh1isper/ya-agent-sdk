"""Agent-facing workflow tools for YA Claw runtime."""

from __future__ import annotations

from typing import Annotated, Any, Protocol, runtime_checkable

from pydantic import Field
from pydantic_ai import RunContext
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.toolsets.core.base import BaseTool

from ya_claw.toolsets.session import CLAW_SELF_CLIENT_KEY, SelfSessionClient, _dump_json


@runtime_checkable
class WorkflowClient(SelfSessionClient, Protocol):
    run_id: str
    profile_name: str | None

    async def list_workflows(self, *, params: dict[str, str]) -> dict[str, Any]: ...

    async def get_workflow(self, *, workflow_id: str) -> dict[str, Any]: ...

    async def create_workflow(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    async def update_workflow(self, *, workflow_id: str, payload: dict[str, Any]) -> dict[str, Any]: ...

    async def archive_workflow(self, *, workflow_id: str) -> dict[str, Any]: ...

    async def start_workflow(self, *, workflow_id: str, payload: dict[str, Any]) -> dict[str, Any]: ...

    async def list_workflow_runs(self, *, params: dict[str, str]) -> dict[str, Any]: ...

    async def get_workflow_run(self, *, workflow_run_id: str) -> dict[str, Any]: ...

    async def cancel_workflow_run(self, *, workflow_run_id: str, reason: str | None) -> dict[str, Any]: ...

    async def steer_workflow_node(
        self,
        *,
        workflow_run_id: str,
        node_id: str,
        input_parts: list[dict[str, Any]],
        prompt: str | None,
    ) -> dict[str, Any]: ...

    async def list_agent_presets(self, *, query: str | None) -> dict[str, Any]: ...


class ListWorkflowsTool(BaseTool):
    name = "list_workflows"
    description = "List YA Claw workflow definitions with filters for current-session relevance."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        return _get_workflow_client(ctx) is not None

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        if _get_workflow_client(ctx) is None:
            return None
        return (
            "List reusable workflow definitions. Use only_current_session=true for workflows related to this "
            "conversation, and broader filters for global reusable workflows."
        )

    async def call(
        self,
        ctx: RunContext[AgentContext],
        query: Annotated[str | None, Field(description="Search workflow name, description, and usage hints")] = None,
        tags: Annotated[list[str] | None, Field(description="Tags that must be present")] = None,
        status: Annotated[str | None, Field(description="Workflow status: draft, active, or archived")] = None,
        scope: Annotated[str | None, Field(description="Workflow scope: global or session")] = None,
        owner_kind: Annotated[str | None, Field(description="Owner kind filter: user, agent, api, or system")] = None,
        only_current_session: Annotated[
            bool, Field(description="Return current-session created, supervised, or touched workflows")
        ] = False,
        only_created_by_current_session: Annotated[
            bool, Field(description="Return definitions created by this session")
        ] = False,
        only_touched_by_current_session: Annotated[
            bool, Field(description="Return definitions touched by this session")
        ] = False,
        include_archived: Annotated[bool, Field(description="Include archived definitions")] = False,
        limit: Annotated[int, Field(description="Maximum workflows, clamped to 1..100")] = 20,
    ) -> str:
        client = _get_workflow_client(ctx)
        if client is None:
            return "Error: YA Claw workflow client is unavailable."
        params = _drop_empty({
            "query": query,
            "status": status,
            "scope": scope,
            "owner_kind": owner_kind,
            "only_current_session": _bool(only_current_session),
            "created_by_current_session": _bool(only_created_by_current_session),
            "touched_by_current_session": _bool(only_touched_by_current_session),
            "include_archived": _bool(include_archived),
            "limit": str(min(max(limit, 1), 100)),
        })
        if tags:
            params["tags"] = ",".join(tag for tag in tags if tag.strip() != "")
        try:
            return _dump_json(await client.list_workflows(params=params))
        except Exception as exc:
            return f"Error: {exc}"


class GetWorkflowTool(BaseTool):
    name = "get_workflow"
    description = "Get a workflow definition with its DAG body and latest run summary."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        return _get_workflow_client(ctx) is not None

    async def call(
        self, ctx: RunContext[AgentContext], workflow_id: Annotated[str, Field(description="Workflow ID")]
    ) -> str:
        client = _get_workflow_client(ctx)
        if client is None:
            return "Error: YA Claw workflow client is unavailable."
        try:
            return _dump_json(await client.get_workflow(workflow_id=workflow_id))
        except Exception as exc:
            return f"Error: {exc}"


class CreateWorkflowTool(BaseTool):
    name = "create_workflow"
    description = "Create a YA Claw workflow definition owned by the current session."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        return _get_workflow_client(ctx) is not None

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        if _get_workflow_client(ctx) is None:
            return None
        return (
            "Create durable workflows as JSON objects with schema ya-claw.workflow.v1, inputs, policy, nodes, "
            "and optional result.from_node. Agent-created workflows default to session scope."
        )

    async def call(
        self,
        ctx: RunContext[AgentContext],
        definition: Annotated[dict[str, Any], Field(description="Workflow definition JSON body")],
        name: Annotated[str | None, Field(description="Optional display name override")] = None,
        description: Annotated[str | None, Field(description="Optional description override")] = None,
        tags: Annotated[list[str] | None, Field(description="Optional workflow tags")] = None,
        scope: Annotated[str, Field(description="Scope: session or global")] = "session",
        status: Annotated[str, Field(description="Initial status: active or draft")] = "active",
    ) -> str:
        client = _get_workflow_client(ctx)
        if client is None:
            return "Error: YA Claw workflow client is unavailable."
        payload = _drop_none({
            "name": name,
            "description": description,
            "tags": tags,
            "scope": scope,
            "status": status,
            "definition": definition,
        })
        try:
            return _dump_json(await client.create_workflow(payload))
        except Exception as exc:
            return f"Error: {exc}"


class UpdateWorkflowTool(BaseTool):
    name = "update_workflow"
    description = "Update a YA Claw workflow definition."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        return _get_workflow_client(ctx) is not None

    async def call(
        self,
        ctx: RunContext[AgentContext],
        workflow_id: Annotated[str, Field(description="Workflow ID")],
        definition: Annotated[dict[str, Any] | None, Field(description="Replacement workflow definition body")] = None,
        name: Annotated[str | None, Field(description="Updated name")] = None,
        description: Annotated[str | None, Field(description="Updated description")] = None,
        tags: Annotated[list[str] | None, Field(description="Updated tags")] = None,
        status: Annotated[str | None, Field(description="Updated status: draft, active, archived")] = None,
        scope: Annotated[str | None, Field(description="Updated scope: global or session")] = None,
    ) -> str:
        client = _get_workflow_client(ctx)
        if client is None:
            return "Error: YA Claw workflow client is unavailable."
        payload = _drop_none({
            "definition": definition,
            "name": name,
            "description": description,
            "tags": tags,
            "status": status,
            "scope": scope,
        })
        try:
            return _dump_json(await client.update_workflow(workflow_id=workflow_id, payload=payload))
        except Exception as exc:
            return f"Error: {exc}"


class ArchiveWorkflowTool(BaseTool):
    name = "archive_workflow"
    description = "Archive a YA Claw workflow definition."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        return _get_workflow_client(ctx) is not None

    async def call(
        self, ctx: RunContext[AgentContext], workflow_id: Annotated[str, Field(description="Workflow ID")]
    ) -> str:
        client = _get_workflow_client(ctx)
        if client is None:
            return "Error: YA Claw workflow client is unavailable."
        try:
            return _dump_json(await client.archive_workflow(workflow_id=workflow_id))
        except Exception as exc:
            return f"Error: {exc}"


class StartWorkflowTool(BaseTool):
    name = "start_workflow"
    description = "Start a YA Claw workflow run supervised by the current session."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        return _get_workflow_client(ctx) is not None

    async def call(
        self,
        ctx: RunContext[AgentContext],
        workflow_id: Annotated[str, Field(description="Workflow ID")],
        inputs: Annotated[dict[str, Any] | None, Field(description="Workflow inputs object")] = None,
        profile_name: Annotated[str | None, Field(description="Default profile for Self nodes")] = None,
        metadata: Annotated[dict[str, Any] | None, Field(description="Optional run metadata")] = None,
    ) -> str:
        client = _get_workflow_client(ctx)
        if client is None:
            return "Error: YA Claw workflow client is unavailable."
        payload = {
            "inputs": dict(inputs or {}),
            "profile_name": profile_name or client.profile_name,
            "trigger_kind": "agent",
            "metadata": dict(metadata or {}),
            "inherit_shell_env": True,
            "shell_env": dict(ctx.deps.shell_env),
        }
        try:
            return _dump_json(await client.start_workflow(workflow_id=workflow_id, payload=payload))
        except Exception as exc:
            return f"Error: {exc}"


class ListWorkflowRunsTool(BaseTool):
    name = "list_workflow_runs"
    description = "List YA Claw workflow runs with current-session filtering flags."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        return _get_workflow_client(ctx) is not None

    async def call(
        self,
        ctx: RunContext[AgentContext],
        workflow_id: Annotated[str | None, Field(description="Optional workflow definition ID")] = None,
        status: Annotated[str | None, Field(description="Run status filter")] = None,
        trigger_kind: Annotated[str | None, Field(description="Trigger kind filter")] = None,
        only_current_session: Annotated[
            bool, Field(description="Return runs supervised by or touched by this session")
        ] = False,
        only_supervised_by_current_session: Annotated[
            bool, Field(description="Return runs supervised by this session")
        ] = False,
        only_touched_by_current_session: Annotated[
            bool, Field(description="Return runs with linked current-session node work")
        ] = False,
        include_completed: Annotated[bool, Field(description="Include terminal completed runs")] = True,
        limit: Annotated[int, Field(description="Maximum runs, clamped to 1..100")] = 20,
    ) -> str:
        client = _get_workflow_client(ctx)
        if client is None:
            return "Error: YA Claw workflow client is unavailable."
        params = _drop_empty({
            "workflow_id": workflow_id,
            "status": status,
            "trigger_kind": trigger_kind,
            "only_current_session": _bool(only_current_session),
            "only_supervised_by_current_session": _bool(only_supervised_by_current_session),
            "only_touched_by_current_session": _bool(only_touched_by_current_session),
            "include_completed": _bool(include_completed),
            "limit": str(min(max(limit, 1), 100)),
        })
        try:
            return _dump_json(await client.list_workflow_runs(params=params))
        except Exception as exc:
            return f"Error: {exc}"


class GetWorkflowRunTool(BaseTool):
    name = "get_workflow_run"
    description = "Get a workflow run with node statuses, linked Claw sessions/runs, result, and events."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        return _get_workflow_client(ctx) is not None

    async def call(
        self,
        ctx: RunContext[AgentContext],
        workflow_run_id: Annotated[str, Field(description="Workflow run ID")],
    ) -> str:
        client = _get_workflow_client(ctx)
        if client is None:
            return "Error: YA Claw workflow client is unavailable."
        try:
            return _dump_json(await client.get_workflow_run(workflow_run_id=workflow_run_id))
        except Exception as exc:
            return f"Error: {exc}"


class SteerWorkflowNodeTool(BaseTool):
    name = "steer_workflow_node"
    description = "Steer an active workflow node run."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        return _get_workflow_client(ctx) is not None

    async def call(
        self,
        ctx: RunContext[AgentContext],
        workflow_run_id: Annotated[str, Field(description="Workflow run ID")],
        node_id: Annotated[str, Field(description="Workflow node ID")],
        prompt: Annotated[str | None, Field(description="Text to steer into the node run")] = None,
        input_parts: Annotated[
            list[dict[str, Any]] | None, Field(description="Structured input parts for steering")
        ] = None,
    ) -> str:
        client = _get_workflow_client(ctx)
        if client is None:
            return "Error: YA Claw workflow client is unavailable."
        try:
            return _dump_json(
                await client.steer_workflow_node(
                    workflow_run_id=workflow_run_id,
                    node_id=node_id,
                    prompt=prompt,
                    input_parts=list(input_parts or []),
                )
            )
        except Exception as exc:
            return f"Error: {exc}"


class CancelWorkflowRunTool(BaseTool):
    name = "cancel_workflow_run"
    description = "Cancel an active YA Claw workflow run."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        return _get_workflow_client(ctx) is not None

    async def call(
        self,
        ctx: RunContext[AgentContext],
        workflow_run_id: Annotated[str, Field(description="Workflow run ID")],
        reason: Annotated[str | None, Field(description="Optional cancellation reason")] = None,
    ) -> str:
        client = _get_workflow_client(ctx)
        if client is None:
            return "Error: YA Claw workflow client is unavailable."
        try:
            return _dump_json(await client.cancel_workflow_run(workflow_run_id=workflow_run_id, reason=reason))
        except Exception as exc:
            return f"Error: {exc}"


class ListAgentPresetsTool(BaseTool):
    name = "list_agent_presets"
    description = "List available YA Claw profiles usable as workflow node agent presets."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        return _get_workflow_client(ctx) is not None

    async def call(
        self,
        ctx: RunContext[AgentContext],
        query: Annotated[str | None, Field(description="Optional profile name search")] = None,
    ) -> str:
        client = _get_workflow_client(ctx)
        if client is None:
            return "Error: YA Claw workflow client is unavailable."
        try:
            return _dump_json(await client.list_agent_presets(query=query))
        except Exception as exc:
            return f"Error: {exc}"


def _get_workflow_client(ctx: RunContext[AgentContext]) -> WorkflowClient | None:
    if ctx.deps.resources is None:
        return None
    resource = ctx.deps.resources.get(CLAW_SELF_CLIENT_KEY)
    if isinstance(resource, WorkflowClient):
        return resource
    return None


def _drop_none(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}


def _drop_empty(value: dict[str, str | None]) -> dict[str, str]:
    return {key: item for key, item in value.items() if isinstance(item, str) and item.strip() != ""}


def _bool(value: bool) -> str:
    return "true" if value else "false"
