"""Model-facing capability for the portable subagent execution service."""

from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

from pydantic import Field
from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelResponse
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset

from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.inputs import InputOrigin
from ya_agent_sdk.subagents.service import (
    SubagentExecutionService,
    SubagentRegistry,
)
from ya_agent_sdk.subagents.spec import (
    SubagentExecutionMode,
    SubagentExecutionRecord,
    SubagentExecutionState,
    SubagentHistoryPolicy,
)


@dataclass(kw_only=True)
class DelegationCapability(AbstractCapability[AgentContext]):
    """Expose one registry/service pair as delegation tools and instructions."""

    registry: SubagentRegistry
    service: SubagentExecutionService
    default_mode: SubagentExecutionMode = SubagentExecutionMode.foreground
    allow_mode_override: bool = False
    id: str | None = "delegation"

    _safe_at_runtime: ClassVar[bool] = False

    def __post_init__(self) -> None:
        if self.allow_mode_override:
            return
        incompatible_routes = [
            plan.spec.route for plan in self.registry.list() if self.default_mode not in plan.spec.execution_modes
        ]
        if incompatible_routes:
            routes = ", ".join(repr(route) for route in incompatible_routes)
            raise ValueError(
                f"Fixed {self.default_mode.value!r} delegation mode is not allowed by subagent routes: {routes}"
            )

    @classmethod
    def get_serialization_name(cls) -> None:
        """Delegation is host-injected authority, never a portable spec value."""
        return None

    def get_instructions(self) -> Any:
        async def instructions(_ctx: RunContext[AgentContext]) -> str | None:
            plans = self.registry.list()
            if not plans:
                return None
            lines = [
                "Use delegation only for bounded subtasks with useful independent value.",
                "The parent owns planning, integration, and final decisions.",
            ]
            if not self.allow_mode_override:
                if self.default_mode is SubagentExecutionMode.background:
                    lines.append(
                        "This host runs delegate asynchronously: it returns an execution handle immediately, "
                        "and the final result arrives through subagent completion delivery."
                    )
                else:
                    lines.append("This host runs delegate inline and returns the child result as the tool result.")
            lines.append("Available subagents:")
            for plan in plans:
                description = plan.normalized_agent_spec.description
                description_text = str(description) if description is not None else plan.spec.route
                if self.allow_mode_override:
                    modes = ", ".join(mode.value for mode in plan.spec.execution_modes)
                    mode_text = f"modes: {modes}"
                else:
                    mode_text = f"mode: {self.default_mode.value}"
                lines.append(f"- {plan.spec.route}: {description_text} ({mode_text})")
            return "\n".join(lines)

        return instructions

    async def close_runtime(self) -> None:
        """Release the execution service with the owning AgentRuntime."""
        await self.service.close()

    def get_toolset(self) -> AbstractToolset[AgentContext]:  # noqa: C901
        toolset = FunctionToolset[AgentContext](id=self.id or "delegation")

        async def _delegate(
            ctx: RunContext[AgentContext],
            subagent_name: str,
            prompt: str,
            effective_mode: SubagentExecutionMode,
            agent_id: str | None,
        ) -> str:
            if agent_id is None:
                plan = self.registry.get(subagent_name)
                history: tuple[dict[str, Any], ...] = ()
                if plan.spec.history is SubagentHistoryPolicy.parent_snapshot:
                    parent_messages = list(ctx.messages)
                    if parent_messages and isinstance(parent_messages[-1], ModelResponse):
                        parent_messages.pop()
                    bounded_messages = parent_messages[-plan.spec.history_message_limit :]
                    history = tuple(
                        ModelMessagesTypeAdapter.dump_python(
                            bounded_messages,
                            mode="json",
                        )
                    )
                handle = await self.service.spawn(
                    subagent_name,
                    prompt,
                    ctx.deps,
                    mode=effective_mode,
                    idempotency_key=_tool_operation_key(
                        ctx,
                        operation="spawn",
                        target=subagent_name,
                    ),
                    history=history,
                )
            else:
                previous = await self.service.get(
                    agent_id,
                    caller_scope_id=_caller_scope(ctx),
                )
                if previous.route != subagent_name:
                    raise ValueError(f"Execution {agent_id!r} belongs to {previous.route!r}, not {subagent_name!r}")
                handle = await self.service.resume(
                    agent_id,
                    prompt,
                    ctx.deps,
                    mode=effective_mode,
                    idempotency_key=_tool_operation_key(
                        ctx,
                        operation="resume",
                        target=agent_id,
                    ),
                )
            if effective_mode is SubagentExecutionMode.background:
                return json.dumps(
                    {
                        "execution_id": handle.execution_id,
                        "route": handle.route,
                        "mode": handle.mode.value,
                    },
                    sort_keys=True,
                )
            record = await self.service.wait(
                handle.execution_id,
                caller_scope_id=_caller_scope(ctx),
            )
            return _foreground_result(record)

        async def delegate(
            ctx: RunContext[AgentContext],
            subagent_name: Annotated[
                str,
                Field(description="Registered subagent route"),
            ],
            prompt: Annotated[
                str,
                Field(description="Bounded task and expected result"),
            ],
            agent_id: Annotated[
                str | None,
                Field(description="Prior execution handle returned by delegate to resume"),
            ] = None,
        ) -> str:
            """Delegate using the execution mode fixed by this host."""
            return await _delegate(ctx, subagent_name, prompt, self.default_mode, agent_id)

        async def delegate_with_mode(
            ctx: RunContext[AgentContext],
            subagent_name: Annotated[
                str,
                Field(description="Registered subagent route"),
            ],
            prompt: Annotated[
                str,
                Field(description="Bounded task and expected result"),
            ],
            mode: Annotated[
                SubagentExecutionMode | None,
                Field(description="Foreground waits; background returns a handle"),
            ] = None,
            agent_id: Annotated[
                str | None,
                Field(description="Prior execution handle returned by delegate to resume"),
            ] = None,
        ) -> str:
            """Delegate using an explicitly selected execution mode."""
            return await _delegate(ctx, subagent_name, prompt, mode or self.default_mode, agent_id)

        async def subagent_info(
            ctx: RunContext[AgentContext],
            execution_id: Annotated[
                str | None,
                Field(description="Execution handle returned by delegate; omit to list plans and executions"),
            ] = None,
        ) -> str:
            """Inspect registered plans or a specific child execution."""
            caller_scope_id = _caller_scope(ctx)
            if execution_id is not None:
                record = await self.service.get(
                    execution_id,
                    caller_scope_id=caller_scope_id,
                )
                return _model_record_json(record)
            return json.dumps(
                {
                    "plans": [
                        {
                            "route": plan.spec.route,
                            "descriptor_id": plan.descriptor_id,
                            "modes": [mode.value for mode in plan.spec.execution_modes],
                        }
                        for plan in self.registry.list()
                    ],
                    "executions": [
                        _model_record_payload(record)
                        for record in await self.service.list(caller_scope_id=caller_scope_id)
                    ],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )

        async def wait_subagent(
            ctx: RunContext[AgentContext],
            execution_id: Annotated[
                str | None,
                Field(description="Execution handle returned by delegate; omit to wait for all current children"),
            ] = None,
            timeout_seconds: Annotated[
                float | None,
                Field(description="Optional bounded wait timeout in seconds", gt=0),
            ] = None,
        ) -> str:
            """Wait once for one child or fan in all current child executions."""
            caller_scope_id = _caller_scope(ctx)
            if execution_id is not None:
                try:
                    record = await self.service.wait(
                        execution_id,
                        caller_scope_id=caller_scope_id,
                        timeout=timeout_seconds,
                    )
                except TimeoutError:
                    record = await self.service.get(
                        execution_id,
                        caller_scope_id=caller_scope_id,
                    )
                return _model_record_json(record)

            records = await self.service.list(caller_scope_id=caller_scope_id)
            pending_ids = [record.execution_id for record in records if not record.terminal]
            if pending_ids:
                waits = [self.service.wait(current_id, caller_scope_id=caller_scope_id) for current_id in pending_ids]
                with suppress(TimeoutError):
                    await asyncio.wait_for(asyncio.gather(*waits), timeout=timeout_seconds)
            latest = await self.service.list(caller_scope_id=caller_scope_id)
            return json.dumps(
                [_model_record_payload(record) for record in latest],
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )

        async def steer_subagent(
            ctx: RunContext[AgentContext],
            execution_id: Annotated[str, Field(description="Execution handle returned by delegate")],
            message: Annotated[str, Field(description="Structured child input")],
        ) -> str:
            """Send targeted input to a currently running child."""
            receipt = await self.service.steer(
                execution_id,
                message,
                caller_scope_id=_caller_scope(ctx),
                origin=InputOrigin.user,
                idempotency_key=_tool_operation_key(
                    ctx,
                    operation="steer",
                    target=execution_id,
                ),
            )
            return json.dumps(
                {
                    "execution_id": execution_id,
                    "disposition": receipt.disposition.value,
                },
                sort_keys=True,
            )

        async def cancel_subagent(
            ctx: RunContext[AgentContext],
            execution_id: Annotated[str, Field(description="Execution handle returned by delegate")],
        ) -> str:
            """Cancel a currently running child execution."""
            record = await self.service.cancel(
                execution_id,
                caller_scope_id=_caller_scope(ctx),
            )
            return _model_record_json(record)

        if self.allow_mode_override:
            delegate_with_mode.__name__ = "delegate"
            toolset.add_function(delegate_with_mode, takes_ctx=True)
        else:
            toolset.add_function(delegate, takes_ctx=True)
        toolset.add_function(subagent_info, takes_ctx=True)
        toolset.add_function(wait_subagent, takes_ctx=True)
        toolset.add_function(steer_subagent, takes_ctx=True)
        toolset.add_function(cancel_subagent, takes_ctx=True)
        return toolset

    async def before_model_request(
        self,
        ctx: RunContext[AgentContext],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Pump committed background completions into canonical native input."""
        await self.service.deliver_pending(ctx.deps)
        return request_context


def _caller_scope(ctx: RunContext[AgentContext]) -> str:
    scope_id = ctx.deps.delegation_scope_id
    if not isinstance(scope_id, str) or not scope_id:
        raise RuntimeError("Delegation tools require a stable caller scope")
    return scope_id


def _tool_operation_key(
    ctx: RunContext[AgentContext],
    *,
    operation: str,
    target: str,
) -> str:
    """Derive one replay-stable identity from the native durable tool call."""
    tool_call_id = ctx.tool_call_id
    if not isinstance(tool_call_id, str) or not tool_call_id:
        raise RuntimeError("Delegation tool execution requires a native tool_call_id")
    router = ctx.deps.input_router
    logical_run_id = router.logical_run_id if router is not None else ctx.deps.run_input_ledger.logical_run_id
    return f"delegation:{logical_run_id}:{tool_call_id}:{operation}:{target}"


_MODEL_RECORD_FIELDS = {
    "execution_id",
    "route",
    "mode",
    "state",
    "input_state",
    "delivery_state",
    "depth",
    "resumed_from",
    "created_at",
    "started_at",
    "completed_at",
    "output",
    "error",
    "usage",
    "segment_index",
    "deferred",
}


def _model_record_payload(record: SubagentExecutionRecord) -> dict[str, Any]:
    """Project one model-facing record without internal UUIDs or resumable state."""
    return record.model_dump(
        mode="json",
        include=_MODEL_RECORD_FIELDS,
    )


def _model_record_json(record: SubagentExecutionRecord) -> str:
    return json.dumps(
        _model_record_payload(record),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def _foreground_result(record: SubagentExecutionRecord) -> str:
    if record.state is SubagentExecutionState.succeeded:
        if isinstance(record.output, str):
            return record.output
        return json.dumps(record.output, ensure_ascii=False, sort_keys=True)
    if record.state is SubagentExecutionState.suspended:
        return _model_record_json(record)
    detail = record.error or f"execution ended as {record.state.value}"
    raise RuntimeError(f"Subagent {record.route!r} ({record.execution_id}) failed: {detail}")
