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
from ya_agent_sdk.subagents.projection import (
    DEFAULT_EXECUTION_PAGE_SIZE,
    DEFAULT_OUTPUT_PAGE_CHARS,
    MAX_EXECUTION_PAGE_SIZE,
    MAX_OUTPUT_PAGE_CHARS,
    model_record_json,
    model_record_payload,
)
from ya_agent_sdk.subagents.service import (
    SubagentExecutionService,
    SubagentRegistry,
)
from ya_agent_sdk.subagents.spec import (
    SubagentExecutionMode,
    SubagentExecutionRecord,
    SubagentHandle,
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
            plans, total_plans = self.registry.list_page(
                offset=0,
                limit=DEFAULT_EXECUTION_PAGE_SIZE,
            )
            if not plans:
                return None
            lines = [
                "Use delegation only for bounded subtasks with useful independent value.",
                "The parent owns planning, integration, and final decisions.",
            ]
            if not self.allow_mode_override:
                if self.default_mode is SubagentExecutionMode.background:
                    lines.append(
                        "This host runs delegate asynchronously: it returns execution_id immediately, "
                        "and the final result arrives through subagent completion delivery."
                    )
                else:
                    lines.append("This host runs delegate inline and returns a bounded structured result.")
            lines.append("Use resume_subagent with a terminal execution_id to create a linked continuation.")
            lines.append("Available subagents:")
            for plan in plans:
                description = plan.normalized_agent_spec.description
                description_text = str(description) if description is not None else plan.spec.route
                if len(description_text) > 300:
                    description_text = description_text[:300] + "... [truncated]"
                if self.allow_mode_override:
                    modes = ", ".join(mode.value for mode in plan.spec.execution_modes)
                    mode_text = f"modes: {modes}"
                else:
                    mode_text = f"mode: {self.default_mode.value}"
                lines.append(f"- {plan.spec.route}: {description_text} ({mode_text})")
            if total_plans > len(plans):
                lines.append(
                    f"{total_plans - len(plans)} additional routes are available; "
                    "use subagent_info plan_offset to inspect them."
                )
            return "\n".join(lines)

        return instructions

    async def close_runtime(self) -> None:
        """Release the execution service with the owning AgentRuntime."""
        await self.service.close()

    def get_toolset(self) -> AbstractToolset[AgentContext]:  # noqa: C901
        toolset = FunctionToolset[AgentContext](id=self.id or "delegation")

        async def _result(
            ctx: RunContext[AgentContext],
            handle: SubagentHandle,
        ) -> str:
            caller_scope_id = _caller_scope(ctx)
            if handle.mode is SubagentExecutionMode.background:
                record = await self.service.get(
                    handle.execution_id,
                    caller_scope_id=caller_scope_id,
                )
            else:
                record = await self.service.wait(
                    handle.execution_id,
                    caller_scope_id=caller_scope_id,
                )
            return model_record_json(record, include_output=True)

        async def _delegate(
            ctx: RunContext[AgentContext],
            subagent_name: str,
            prompt: str,
            effective_mode: SubagentExecutionMode,
        ) -> str:
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
            return await _result(ctx, handle)

        async def _resume(
            ctx: RunContext[AgentContext],
            execution_id: str,
            prompt: str,
            effective_mode: SubagentExecutionMode,
        ) -> str:
            handle = await self.service.resume(
                execution_id,
                prompt,
                ctx.deps,
                mode=effective_mode,
                idempotency_key=_tool_operation_key(
                    ctx,
                    operation="resume",
                    target=execution_id,
                ),
            )
            return await _result(ctx, handle)

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
        ) -> str:
            """Start a new subagent execution using the mode fixed by this host."""
            return await _delegate(ctx, subagent_name, prompt, self.default_mode)

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
                Field(description="Foreground waits; background returns immediately"),
            ] = None,
        ) -> str:
            """Start a new subagent execution using an explicitly selected mode."""
            return await _delegate(ctx, subagent_name, prompt, mode or self.default_mode)

        async def resume_subagent(
            ctx: RunContext[AgentContext],
            execution_id: Annotated[
                str,
                Field(description="Terminal execution_id returned by delegate or resume_subagent"),
            ],
            prompt: Annotated[
                str,
                Field(description="Bounded continuation task and expected result"),
            ],
        ) -> str:
            """Create a linked continuation from one terminal execution."""
            return await _resume(ctx, execution_id, prompt, self.default_mode)

        async def resume_subagent_with_mode(
            ctx: RunContext[AgentContext],
            execution_id: Annotated[
                str,
                Field(description="Terminal execution_id returned by delegate or resume_subagent"),
            ],
            prompt: Annotated[
                str,
                Field(description="Bounded continuation task and expected result"),
            ],
            mode: Annotated[
                SubagentExecutionMode | None,
                Field(description="Foreground waits; background returns immediately"),
            ] = None,
        ) -> str:
            """Create a linked continuation using an explicitly selected mode."""
            return await _resume(ctx, execution_id, prompt, mode or self.default_mode)

        async def subagent_info(
            ctx: RunContext[AgentContext],
            execution_id: Annotated[
                str | None,
                Field(description="Execution_id to inspect; omit to list plans and execution summaries"),
            ] = None,
            output_offset: Annotated[
                int,
                Field(description="Character offset into the selected execution output", ge=0),
            ] = 0,
            output_limit: Annotated[
                int,
                Field(
                    description="Maximum output characters to return for one execution",
                    ge=1,
                    le=MAX_OUTPUT_PAGE_CHARS,
                ),
            ] = DEFAULT_OUTPUT_PAGE_CHARS,
            execution_offset: Annotated[
                int,
                Field(description="Offset into execution summaries when listing", ge=0),
            ] = 0,
            execution_limit: Annotated[
                int,
                Field(
                    description="Maximum execution summaries to list",
                    ge=1,
                    le=MAX_EXECUTION_PAGE_SIZE,
                ),
            ] = DEFAULT_EXECUTION_PAGE_SIZE,
            plan_offset: Annotated[
                int,
                Field(description="Offset into registered plans when listing", ge=0),
            ] = 0,
            plan_limit: Annotated[
                int,
                Field(
                    description="Maximum registered plans to list",
                    ge=1,
                    le=MAX_EXECUTION_PAGE_SIZE,
                ),
            ] = DEFAULT_EXECUTION_PAGE_SIZE,
        ) -> str:
            """Inspect registered plans or one bounded page of child execution data."""
            caller_scope_id = _caller_scope(ctx)
            if execution_id is not None:
                record = await self.service.get(
                    execution_id,
                    caller_scope_id=caller_scope_id,
                )
                return model_record_json(
                    record,
                    include_output=True,
                    output_offset=output_offset,
                    output_limit=output_limit,
                )
            records, execution_total = await self.service.list_page(
                caller_scope_id=caller_scope_id,
                offset=execution_offset,
                limit=execution_limit,
            )
            plans, plan_total = self.registry.list_page(
                offset=plan_offset,
                limit=plan_limit,
            )
            return json.dumps(
                {
                    "plans": [
                        {
                            "route": plan.spec.route,
                            "modes": [mode.value for mode in plan.spec.execution_modes],
                        }
                        for plan in plans
                    ],
                    "plan_pagination": _pagination_payload(
                        offset=plan_offset,
                        limit=plan_limit,
                        total=plan_total,
                    ),
                    **_execution_page(
                        records,
                        offset=execution_offset,
                        limit=execution_limit,
                        total=execution_total,
                    ),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )

        async def wait_subagent(
            ctx: RunContext[AgentContext],
            execution_id: Annotated[
                str | None,
                Field(description="Execution_id to wait for; omit for one-shot fan-in"),
            ] = None,
            timeout_seconds: Annotated[
                float | None,
                Field(description="Optional bounded wait timeout in seconds", gt=0),
            ] = None,
            output_offset: Annotated[
                int,
                Field(description="Character offset into one selected execution output", ge=0),
            ] = 0,
            output_limit: Annotated[
                int,
                Field(
                    description="Maximum output characters to return for one execution",
                    ge=1,
                    le=MAX_OUTPUT_PAGE_CHARS,
                ),
            ] = DEFAULT_OUTPUT_PAGE_CHARS,
            execution_offset: Annotated[
                int,
                Field(description="Offset into fan-in execution summaries", ge=0),
            ] = 0,
            execution_limit: Annotated[
                int,
                Field(
                    description="Maximum fan-in execution summaries to return",
                    ge=1,
                    le=MAX_EXECUTION_PAGE_SIZE,
                ),
            ] = DEFAULT_EXECUTION_PAGE_SIZE,
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
                return model_record_json(
                    record,
                    include_output=True,
                    output_offset=output_offset,
                    output_limit=output_limit,
                )

            async def wait_for_current() -> None:
                pending_ids = await self.service.list_nonterminal_ids(caller_scope_id=caller_scope_id)
                for index in range(0, len(pending_ids), DEFAULT_EXECUTION_PAGE_SIZE):
                    batch = pending_ids[index : index + DEFAULT_EXECUTION_PAGE_SIZE]
                    await asyncio.gather(
                        *(self.service.wait(current_id, caller_scope_id=caller_scope_id) for current_id in batch)
                    )

            with suppress(TimeoutError):
                await asyncio.wait_for(wait_for_current(), timeout=timeout_seconds)
            latest, total = await self.service.list_page(
                caller_scope_id=caller_scope_id,
                offset=execution_offset,
                limit=execution_limit,
            )
            return json.dumps(
                _execution_page(
                    latest,
                    offset=execution_offset,
                    limit=execution_limit,
                    total=total,
                ),
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
                parent_ctx=ctx.deps,
            )
            return model_record_json(record)

        if self.allow_mode_override:
            delegate_with_mode.__name__ = "delegate"
            resume_subagent_with_mode.__name__ = "resume_subagent"
            toolset.add_function(delegate_with_mode, takes_ctx=True)
            toolset.add_function(resume_subagent_with_mode, takes_ctx=True)
        else:
            toolset.add_function(delegate, takes_ctx=True)
            toolset.add_function(resume_subagent, takes_ctx=True)
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


def _execution_page(
    records: tuple[SubagentExecutionRecord, ...],
    *,
    offset: int,
    limit: int,
    total: int,
) -> dict[str, Any]:
    return {
        "executions": [model_record_payload(record) for record in records],
        "pagination": _pagination_payload(
            offset=offset,
            limit=limit,
            total=total,
        ),
    }


def _pagination_payload(
    *,
    offset: int,
    limit: int,
    total: int,
) -> dict[str, int | None]:
    end = min(offset + limit, total)
    return {
        "offset": offset,
        "limit": limit,
        "total": total,
        "next_offset": end if end < total else None,
    }
