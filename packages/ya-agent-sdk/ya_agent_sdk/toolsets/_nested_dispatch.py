"""Shared native dispatch for nested tool calls."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from pydantic_ai import (
    ApprovalRequired,
    CallDeferred,
    DeferredToolRequests,
    ModelRetry,
    RunContext,
    ToolFailed,
)
from pydantic_ai.exceptions import ToolRetryError
from pydantic_ai.messages import RetryPromptPart, ToolCallPart
from pydantic_ai.tool_manager import ToolManager, ValidatedToolCall
from pydantic_ai.tools import ToolApproved, ToolDenied
from pydantic_ai.toolsets import ToolsetTool

from ya_agent_sdk.context import AgentContext


async def prepare_nested_tool(
    manager: ToolManager[AgentContext],
    ctx: RunContext[AgentContext],
    tool: ToolsetTool[AgentContext],
) -> ToolsetTool[AgentContext]:
    """Apply the active capability's preparation policy to one hidden tool."""
    capability = manager.root_capability
    if capability is None:
        return tool
    definitions = await capability.prepare_tools(ctx, [tool.tool_def])
    prepared = next((item for item in definitions if item.name == tool.tool_def.name), None)
    if prepared is None:
        raise ModelRetry(f"Tool {tool.tool_def.name!r} is not available in this agent context")
    return replace(tool, tool_def=prepared)


async def execute_nested_tool_call(
    manager: ToolManager[AgentContext],
    validated: ValidatedToolCall[AgentContext],
    call: ToolCallPart,
    *,
    require_resolution: bool = False,
) -> Any:
    """Execute one nested call while preserving native deferred semantics."""
    tool = validated.tool
    if tool is None:  # pragma: no cover - validated calls always retain their tool
        return await manager.execute_tool_call(validated, wrap_validation_errors=False)

    deferred: ApprovalRequired | CallDeferred | None = None
    if tool.tool_def.kind == "unapproved" and not validated.ctx.tool_call_approved:
        deferred = ApprovalRequired()
    elif tool.tool_def.kind == "external":
        deferred = CallDeferred()

    if deferred is None:
        try:
            return await manager.execute_tool_call(validated, wrap_validation_errors=False)
        except (CallDeferred, ApprovalRequired) as exc:
            deferred = exc

    return await _resolve_nested_deferred(
        manager,
        call,
        deferred,
        require_resolution=require_resolution,
    )


async def _resolve_nested_deferred(
    manager: ToolManager[AgentContext],
    call: ToolCallPart,
    deferred: ApprovalRequired | CallDeferred,
    *,
    require_resolution: bool,
) -> Any:
    requests = DeferredToolRequests(
        approvals=[call] if isinstance(deferred, ApprovalRequired) else [],
        calls=[call] if isinstance(deferred, CallDeferred) else [],
        metadata={call.tool_call_id: deferred.metadata} if deferred.metadata else {},
    )
    results = await manager.resolve_deferred_tool_calls(requests)
    if results is None:
        if require_resolution:
            raise RuntimeError(f"Nested deferred tool {call.tool_name!r} has no active resolver") from deferred
        raise deferred
    resolved = results.to_tool_call_results().get(call.tool_call_id)
    if resolved is None:
        if require_resolution:
            raise RuntimeError(f"Nested deferred tool {call.tool_name!r} was not resolved") from deferred
        raise deferred
    if isinstance(resolved, ToolDenied):
        return resolved
    if isinstance(resolved, ToolFailed):
        raise resolved from deferred
    if isinstance(resolved, ToolApproved):
        approved_call = replace(call, args=resolved.override_args) if resolved.override_args is not None else call
        approved = await manager.validate_tool_call(
            approved_call,
            approved=True,
            metadata=results.metadata.get(call.tool_call_id),
            wrap_validation_errors=False,
        )
        return await manager.execute_tool_call(approved, wrap_validation_errors=False)
    if isinstance(resolved, ModelRetry):
        retry = RetryPromptPart(
            content=resolved.message,
            tool_name=call.tool_name,
            tool_call_id=call.tool_call_id,
        )
        raise ToolRetryError(retry) from deferred
    if isinstance(resolved, RetryPromptPart):
        retry = replace(resolved, tool_name=call.tool_name, tool_call_id=call.tool_call_id)
        raise ToolRetryError(retry) from deferred
    return results.calls[call.tool_call_id]
