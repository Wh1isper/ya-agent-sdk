"""Agent-facing schedule tools for YA Claw runtime."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Protocol, runtime_checkable

from pydantic import Field
from pydantic_ai import RunContext
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.toolsets.core.base import BaseTool

from ya_claw.toolsets.session import CLAW_SELF_CLIENT_KEY, SelfSessionClient, _dump_json


@runtime_checkable
class ScheduleClient(SelfSessionClient, Protocol):
    run_id: str
    profile_name: str | None

    async def list_schedules(
        self,
        *,
        schedule_id: str | None,
        include_disabled: bool,
        include_recent_runs: bool,
        limit: int,
    ) -> dict[str, Any]: ...

    async def create_schedule(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    async def update_schedule(self, *, schedule_id: str, payload: dict[str, Any]) -> dict[str, Any]: ...

    async def delete_schedule(self, *, schedule_id: str) -> dict[str, Any]: ...

    async def trigger_schedule(self, *, schedule_id: str, prompt_override: str | None) -> dict[str, Any]: ...


class ListSchedulesTool(BaseTool):
    name = "list_schedules"
    description = "List schedules owned by or related to the current YA Claw session."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        return _get_schedule_client(ctx) is not None

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        if _get_schedule_client(ctx) is None:
            return None
        return (
            "List schedules you own in the current YA Claw session. Use this before creating duplicate scheduled jobs."
        )

    async def call(
        self,
        ctx: RunContext[AgentContext],
        schedule_id: Annotated[str | None, Field(description="Optional schedule ID to inspect")] = None,
        include_disabled: Annotated[bool, Field(description="Include paused schedules")] = True,
        include_recent_runs: Annotated[bool, Field(description="Include last fire summary")] = True,
        limit: Annotated[int, Field(description="Maximum schedules to return, clamped to 1..100")] = 20,
    ) -> str:
        client = _get_schedule_client(ctx)
        if client is None:
            return "Error: YA Claw schedule client is unavailable."
        try:
            payload = await client.list_schedules(
                schedule_id=schedule_id,
                include_disabled=include_disabled,
                include_recent_runs=include_recent_runs,
                limit=min(max(limit, 1), 100),
            )
            return _dump_json(payload)
        except Exception as exc:
            return f"Error: {exc}"


class CreateScheduleTool(BaseTool):
    name = "create_schedule"
    description = "Create a recurring cron schedule owned by the current YA Claw session."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        return _get_schedule_client(ctx) is not None

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        if _get_schedule_client(ctx) is None:
            return None
        return (
            "Create recurring cron schedules with a plain text prompt. "
            "The schedule inherits the current profile. "
            "Use continue_current_session for timed messages in this conversation, "
            "start_from_current_session for recurring branches, and enabled=false to create a paused schedule."
        )

    async def call(
        self,
        ctx: RunContext[AgentContext],
        name: Annotated[str, Field(description="Schedule name")],
        prompt: Annotated[str, Field(description="Plain text prompt to run on each cron fire")],
        cron: Annotated[str, Field(description="Five-field cron expression, e.g. '0 9 * * *'")],
        timezone: Annotated[str, Field(description="IANA timezone, e.g. UTC or Asia/Shanghai")] = "UTC",
        enabled: Annotated[bool, Field(description="Whether the schedule is active")] = True,
        continue_current_session: Annotated[
            bool, Field(description="Continue the current session on each fire")
        ] = False,
        start_from_current_session: Annotated[
            bool,
            Field(description="Create a new session from current session's latest committed state on each fire"),
        ] = False,
        steer_when_running: Annotated[
            bool, Field(description="Steer the active run when current session is running")
        ] = False,
        description: Annotated[str | None, Field(description="Optional schedule description")] = None,
    ) -> str:
        client = _get_schedule_client(ctx)
        if client is None:
            return "Error: YA Claw schedule client is unavailable."
        try:
            payload = {
                "name": name,
                "description": description,
                "prompt": prompt,
                "trigger_kind": "cron",
                "cron": cron,
                "timezone": timezone,
                "enabled": enabled,
                "continue_current_session": continue_current_session,
                "start_from_current_session": start_from_current_session,
                "steer_when_running": steer_when_running,
                "owner_kind": "agent",
                "owner_session_id": client.session_id,
                "owner_run_id": client.run_id,
                "profile_name": client.profile_name,
            }
            return _dump_json(await client.create_schedule(payload))
        except Exception as exc:
            return f"Error: {exc}"


class CreateOnceScheduleTool(BaseTool):
    name = "create_once_schedule"
    description = "Create a one-time schedule owned by the current YA Claw session."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        return _get_schedule_client(ctx) is not None

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        if _get_schedule_client(ctx) is None:
            return None
        return (
            "Create one-time schedules with a plain text prompt and an absolute run_at datetime. "
            "The schedule inherits the current profile. "
            "Use continue_current_session for timed messages in this conversation, "
            "start_from_current_session for one-time branches, and enabled=false to create a paused schedule."
        )

    async def call(
        self,
        ctx: RunContext[AgentContext],
        name: Annotated[str, Field(description="Schedule name")],
        prompt: Annotated[str, Field(description="Plain text prompt to run once")],
        run_at: Annotated[datetime, Field(description="Absolute datetime to run at, with timezone if possible")],
        timezone: Annotated[str, Field(description="IANA timezone for display, e.g. UTC or Asia/Shanghai")] = "UTC",
        enabled: Annotated[bool, Field(description="Whether the schedule is active")] = True,
        continue_current_session: Annotated[
            bool, Field(description="Continue the current session when the one-time schedule fires")
        ] = False,
        start_from_current_session: Annotated[
            bool,
            Field(description="Create a new session from current session's latest committed state when it fires"),
        ] = False,
        steer_when_running: Annotated[
            bool, Field(description="Steer the active run when current session is running")
        ] = False,
        description: Annotated[str | None, Field(description="Optional schedule description")] = None,
    ) -> str:
        client = _get_schedule_client(ctx)
        if client is None:
            return "Error: YA Claw schedule client is unavailable."
        try:
            payload = {
                "name": name,
                "description": description,
                "prompt": prompt,
                "trigger_kind": "once",
                "run_at": run_at.isoformat(),
                "timezone": timezone,
                "enabled": enabled,
                "continue_current_session": continue_current_session,
                "start_from_current_session": start_from_current_session,
                "steer_when_running": steer_when_running,
                "owner_kind": "agent",
                "owner_session_id": client.session_id,
                "owner_run_id": client.run_id,
                "profile_name": client.profile_name,
            }
            return _dump_json(await client.create_schedule(payload))
        except Exception as exc:
            return f"Error: {exc}"


class CreateWorkflowScheduleTool(BaseTool):
    name = "create_workflow_schedule"
    description = "Create a recurring cron schedule that triggers a YA Claw workflow."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        return _get_schedule_client(ctx) is not None

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        if _get_schedule_client(ctx) is None:
            return None
        return (
            "Create recurring cron schedules that trigger workflows. "
            "Use workflow_inputs_template to map schedule context into workflow inputs; "
            "templates can reference {{ schedule.id }}, {{ schedule.name }}, and {{ scheduled_at }}."
        )

    async def call(
        self,
        ctx: RunContext[AgentContext],
        name: Annotated[str, Field(description="Schedule name")],
        workflow_id: Annotated[str, Field(description="Workflow definition ID to trigger")],
        cron: Annotated[str, Field(description="Five-field cron expression, e.g. '0 9 * * *'")],
        workflow_inputs_template: Annotated[
            dict[str, Any] | None,
            Field(description="Workflow input object template rendered on each fire"),
        ] = None,
        timezone: Annotated[str, Field(description="IANA timezone, e.g. UTC or Asia/Shanghai")] = "UTC",
        enabled: Annotated[bool, Field(description="Whether the schedule is active")] = True,
        description: Annotated[str | None, Field(description="Optional schedule description")] = None,
    ) -> str:
        client = _get_schedule_client(ctx)
        if client is None:
            return "Error: YA Claw schedule client is unavailable."
        try:
            payload = {
                "name": name,
                "description": description,
                "prompt": "",
                "trigger_kind": "cron",
                "cron": cron,
                "timezone": timezone,
                "enabled": enabled,
                "workflow_id": workflow_id,
                "workflow_inputs_template": dict(workflow_inputs_template or {}),
                "owner_kind": "agent",
                "owner_session_id": client.session_id,
                "owner_run_id": client.run_id,
                "profile_name": client.profile_name,
            }
            return _dump_json(await client.create_schedule(payload))
        except Exception as exc:
            return f"Error: {exc}"


class CreateOnceWorkflowScheduleTool(BaseTool):
    name = "create_once_workflow_schedule"
    description = "Create a one-time schedule that triggers a YA Claw workflow."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        return _get_schedule_client(ctx) is not None

    async def call(
        self,
        ctx: RunContext[AgentContext],
        name: Annotated[str, Field(description="Schedule name")],
        workflow_id: Annotated[str, Field(description="Workflow definition ID to trigger")],
        run_at: Annotated[datetime, Field(description="Absolute datetime to run at, with timezone if possible")],
        workflow_inputs_template: Annotated[
            dict[str, Any] | None,
            Field(description="Workflow input object template rendered on fire"),
        ] = None,
        timezone: Annotated[str, Field(description="IANA timezone for display, e.g. UTC or Asia/Shanghai")] = "UTC",
        enabled: Annotated[bool, Field(description="Whether the schedule is active")] = True,
        description: Annotated[str | None, Field(description="Optional schedule description")] = None,
    ) -> str:
        client = _get_schedule_client(ctx)
        if client is None:
            return "Error: YA Claw schedule client is unavailable."
        try:
            payload = {
                "name": name,
                "description": description,
                "prompt": "",
                "trigger_kind": "once",
                "run_at": run_at.isoformat(),
                "timezone": timezone,
                "enabled": enabled,
                "workflow_id": workflow_id,
                "workflow_inputs_template": dict(workflow_inputs_template or {}),
                "owner_kind": "agent",
                "owner_session_id": client.session_id,
                "owner_run_id": client.run_id,
                "profile_name": client.profile_name,
            }
            return _dump_json(await client.create_schedule(payload))
        except Exception as exc:
            return f"Error: {exc}"


class UpdateScheduleTool(BaseTool):
    name = "update_schedule"
    description = "Update a schedule owned by the current YA Claw session."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        return _get_schedule_client(ctx) is not None

    async def call(
        self,
        ctx: RunContext[AgentContext],
        schedule_id: Annotated[str, Field(description="Schedule ID")],
        name: Annotated[str | None, Field(description="Updated schedule name")] = None,
        prompt: Annotated[str | None, Field(description="Updated plain text prompt")] = None,
        trigger_kind: Annotated[str | None, Field(description="Updated trigger kind: cron or once")] = None,
        cron: Annotated[str | None, Field(description="Updated five-field cron expression")] = None,
        run_at: Annotated[datetime | None, Field(description="Updated one-time run datetime")] = None,
        timezone: Annotated[str | None, Field(description="Updated IANA timezone")] = None,
        enabled: Annotated[bool | None, Field(description="Set active or paused state")] = None,
        continue_current_session: Annotated[
            bool | None, Field(description="Update current-session continuation mode")
        ] = None,
        start_from_current_session: Annotated[bool | None, Field(description="Update source-session fork mode")] = None,
        steer_when_running: Annotated[bool | None, Field(description="Update active-run steering behavior")] = None,
        description: Annotated[str | None, Field(description="Updated description")] = None,
    ) -> str:
        client = _get_schedule_client(ctx)
        if client is None:
            return "Error: YA Claw schedule client is unavailable."
        payload = _drop_none({
            "name": name,
            "description": description,
            "prompt": prompt,
            "trigger_kind": trigger_kind,
            "cron": cron,
            "run_at": run_at.isoformat() if run_at is not None else None,
            "timezone": timezone,
            "enabled": enabled,
            "continue_current_session": continue_current_session,
            "start_from_current_session": start_from_current_session,
            "steer_when_running": steer_when_running,
        })
        try:
            return _dump_json(await client.update_schedule(schedule_id=schedule_id, payload=payload))
        except Exception as exc:
            return f"Error: {exc}"


class DeleteScheduleTool(BaseTool):
    name = "delete_schedule"
    description = "Delete a schedule owned by the current YA Claw session."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        return _get_schedule_client(ctx) is not None

    async def call(
        self, ctx: RunContext[AgentContext], schedule_id: Annotated[str, Field(description="Schedule ID")]
    ) -> str:
        client = _get_schedule_client(ctx)
        if client is None:
            return "Error: YA Claw schedule client is unavailable."
        try:
            return _dump_json(await client.delete_schedule(schedule_id=schedule_id))
        except Exception as exc:
            return f"Error: {exc}"


class TriggerScheduleTool(BaseTool):
    name = "trigger_schedule"
    description = "Manually trigger a schedule owned by the current YA Claw session."

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        return _get_schedule_client(ctx) is not None

    async def call(
        self,
        ctx: RunContext[AgentContext],
        schedule_id: Annotated[str, Field(description="Schedule ID")],
        prompt_override: Annotated[
            str | None, Field(description="Optional one-time plain text prompt override")
        ] = None,
    ) -> str:
        client = _get_schedule_client(ctx)
        if client is None:
            return "Error: YA Claw schedule client is unavailable."
        try:
            return _dump_json(await client.trigger_schedule(schedule_id=schedule_id, prompt_override=prompt_override))
        except Exception as exc:
            return f"Error: {exc}"


def _get_schedule_client(ctx: RunContext[AgentContext]) -> ScheduleClient | None:
    if ctx.deps.resources is None:
        return None
    resource = ctx.deps.resources.get(CLAW_SELF_CLIENT_KEY)
    if isinstance(resource, ScheduleClient):
        return resource
    return None


def _drop_none(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}
