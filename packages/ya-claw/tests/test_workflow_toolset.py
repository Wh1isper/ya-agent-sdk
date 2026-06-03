from __future__ import annotations

import json
from typing import Any

from ya_agent_environment import Environment
from ya_agent_sdk.context import AgentContext
from ya_claw.toolsets.schedule import CreateWorkflowScheduleTool
from ya_claw.toolsets.session import CLAW_SELF_CLIENT_KEY
from ya_claw.toolsets.workflow import (
    CreateWorkflowTool,
    ListWorkflowRunsTool,
    ListWorkflowsTool,
    StartWorkflowTool,
    WorkflowClient,
)


class EmptyEnvironment(Environment):
    async def _setup(self) -> None:
        return None

    async def _teardown(self) -> None:
        return None


class FakeRunContext:
    def __init__(self, deps: AgentContext) -> None:
        self.deps = deps


class FakeWorkflowClient:
    def __init__(self) -> None:
        self.session_id = "session-1"
        self.run_id = "run-1"
        self.profile_name = "default"
        self.calls: list[dict[str, Any]] = []

    def close(self) -> None:
        return None

    async def setup(self) -> None:
        return None

    def get_toolsets(self) -> list[Any]:
        return []

    async def list_session_turns(
        self, *, limit: int, before_sequence_no: int | None, cursor: str | None
    ) -> dict[str, Any]:
        return {}

    async def get_run_trace(self, *, run_id: str, max_item_chars: int, max_total_chars: int) -> dict[str, Any]:
        return {}

    async def list_schedules(
        self,
        *,
        schedule_id: str | None,
        include_disabled: bool,
        include_recent_runs: bool,
        limit: int,
    ) -> dict[str, Any]:
        self.calls.append({
            "method": "list_schedules",
            "schedule_id": schedule_id,
            "include_disabled": include_disabled,
            "include_recent_runs": include_recent_runs,
            "limit": limit,
        })
        return {"schedules": []}

    async def create_schedule(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"method": "create_schedule", "payload": payload})
        return {"id": payload.get("workflow_id") or "schedule-1", **payload}

    async def update_schedule(self, *, schedule_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"method": "update_schedule", "schedule_id": schedule_id, "payload": payload})
        return {"id": schedule_id, **payload}

    async def delete_schedule(self, *, schedule_id: str) -> dict[str, Any]:
        self.calls.append({"method": "delete_schedule", "schedule_id": schedule_id})
        return {"id": schedule_id, "status": "deleted"}

    async def trigger_schedule(self, *, schedule_id: str, prompt_override: str | None) -> dict[str, Any]:
        self.calls.append({
            "method": "trigger_schedule",
            "schedule_id": schedule_id,
            "prompt_override": prompt_override,
        })
        return {"id": "fire-1", "schedule_id": schedule_id}

    async def list_workflows(self, *, params: dict[str, str]) -> dict[str, Any]:
        self.calls.append({"method": "list_workflows", "params": params})
        return {"workflows": []}

    async def get_workflow(self, *, workflow_id: str) -> dict[str, Any]:
        return {"id": workflow_id}

    async def create_workflow(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"method": "create_workflow", "payload": payload})
        return {"id": "workflow-1", **payload}

    async def update_workflow(self, *, workflow_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {"id": workflow_id, **payload}

    async def archive_workflow(self, *, workflow_id: str) -> dict[str, Any]:
        return {"id": workflow_id, "status": "archived"}

    async def start_workflow(self, *, workflow_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"method": "start_workflow", "workflow_id": workflow_id, "payload": payload})
        return {"id": "workflow-run-1", "workflow_id": workflow_id, **payload}

    async def list_workflow_runs(self, *, params: dict[str, str]) -> dict[str, Any]:
        self.calls.append({"method": "list_workflow_runs", "params": params})
        return {"workflow_runs": []}

    async def get_workflow_run(self, *, workflow_run_id: str) -> dict[str, Any]:
        return {"id": workflow_run_id}

    async def cancel_workflow_run(self, *, workflow_run_id: str, reason: str | None) -> dict[str, Any]:
        return {"id": workflow_run_id, "reason": reason}

    async def steer_workflow_node(
        self,
        *,
        workflow_run_id: str,
        node_id: str,
        input_parts: list[dict[str, Any]],
        prompt: str | None,
    ) -> dict[str, Any]:
        return {"id": workflow_run_id, "node_id": node_id, "input_parts": input_parts, "prompt": prompt}

    async def list_agent_presets(self, *, query: str | None) -> dict[str, Any]:
        return {"profiles": [{"name": query or "default"}]}


def _context_with_client(client: WorkflowClient) -> FakeRunContext:
    env = EmptyEnvironment()
    ctx = AgentContext(agent_id="main", env=env)
    assert ctx.resources is not None
    ctx.resources.set(CLAW_SELF_CLIENT_KEY, client)
    return FakeRunContext(ctx)


def test_workflow_tools_are_available_with_workflow_client() -> None:
    ctx = _context_with_client(FakeWorkflowClient())

    assert ListWorkflowsTool().is_available(ctx) is True  # type: ignore[arg-type]
    assert CreateWorkflowTool().is_available(ctx) is True  # type: ignore[arg-type]
    assert StartWorkflowTool().is_available(ctx) is True  # type: ignore[arg-type]


async def test_list_workflows_passes_current_session_filter_flags() -> None:
    client = FakeWorkflowClient()
    ctx = _context_with_client(client)

    result = await ListWorkflowsTool().call(  # type: ignore[arg-type]
        ctx,
        query="research",
        tags=["report"],
        only_current_session=True,
        include_archived=True,
        limit=200,
    )

    assert json.loads(result) == {"workflows": []}
    assert client.calls == [
        {
            "method": "list_workflows",
            "params": {
                "query": "research",
                "only_current_session": "true",
                "created_by_current_session": "false",
                "touched_by_current_session": "false",
                "include_archived": "true",
                "limit": "100",
                "tags": "report",
            },
        }
    ]


async def test_create_and_start_workflow_payloads_use_agent_context_defaults() -> None:
    client = FakeWorkflowClient()
    ctx = _context_with_client(client)
    ctx.deps.shell_env = {"BASE_KEY": "base_value"}
    definition = {"schema": "ya-claw.workflow.v1", "nodes": {"a": {"prompt": "do it"}}}

    create_result = await CreateWorkflowTool().call(ctx, definition=definition, name="WF")  # type: ignore[arg-type]
    start_result = await StartWorkflowTool().call(ctx, workflow_id="workflow-1", inputs={"topic": "x"})  # type: ignore[arg-type]

    assert json.loads(create_result)["scope"] == "session"
    assert json.loads(start_result)["profile_name"] == "default"
    assert client.calls[0]["payload"]["definition"] == definition
    assert client.calls[1]["payload"] == {
        "inputs": {"topic": "x"},
        "profile_name": "default",
        "trigger_kind": "agent",
        "metadata": {},
        "inherit_shell_env": True,
        "shell_env": {"BASE_KEY": "base_value"},
    }


async def test_create_workflow_schedule_payload_targets_workflow() -> None:
    client = FakeWorkflowClient()
    ctx = _context_with_client(client)

    result = await CreateWorkflowScheduleTool().call(  # type: ignore[arg-type]
        ctx,
        name="daily workflow",
        workflow_id="workflow-1",
        cron="0 9 * * *",
        workflow_inputs_template={"topic": "{{ schedule.name }}"},
    )

    assert json.loads(result)["id"] == "workflow-1"
    assert client.calls == [
        {
            "method": "create_schedule",
            "payload": {
                "name": "daily workflow",
                "description": None,
                "prompt": "",
                "trigger_kind": "cron",
                "cron": "0 9 * * *",
                "timezone": "UTC",
                "enabled": True,
                "workflow_id": "workflow-1",
                "workflow_inputs_template": {"topic": "{{ schedule.name }}"},
                "owner_kind": "agent",
                "owner_session_id": "session-1",
                "owner_run_id": "run-1",
                "profile_name": "default",
            },
        }
    ]


async def test_list_workflow_runs_uses_requested_flags() -> None:
    client = FakeWorkflowClient()
    ctx = _context_with_client(client)

    await ListWorkflowRunsTool().call(  # type: ignore[arg-type]
        ctx,
        only_supervised_by_current_session=True,
        include_completed=False,
    )

    assert client.calls == [
        {
            "method": "list_workflow_runs",
            "params": {
                "only_current_session": "false",
                "only_supervised_by_current_session": "true",
                "only_touched_by_current_session": "false",
                "include_completed": "false",
                "limit": "20",
            },
        }
    ]
