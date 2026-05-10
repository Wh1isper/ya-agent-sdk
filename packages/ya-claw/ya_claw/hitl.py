from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from pydantic_ai import DeferredToolRequests
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart

from ya_claw.controller.models import ActiveInteraction


def build_active_interactions(
    deferred_requests: DeferredToolRequests,
    *,
    run_id: str,
    session_id: str,
) -> list[ActiveInteraction]:
    tool_calls = list(deferred_requests.approvals or [])
    metadata_by_call = deferred_requests.metadata or {}
    total_count = len(tool_calls)
    created_at = datetime.now(UTC)
    interactions: list[ActiveInteraction] = []
    for index, tool_call in enumerate(tool_calls, start=1):
        metadata = dict(metadata_by_call.get(tool_call.tool_call_id) or {})
        interactions.append(
            ActiveInteraction(
                interaction_id=f"hitl_{run_id}_{index}",
                run_id=run_id,
                session_id=session_id,
                tool_call_id=tool_call.tool_call_id,
                tool_name=tool_call.tool_name,
                kind=_interaction_kind(metadata, tool_call),
                title=_interaction_title(metadata, tool_call),
                description=_interaction_description(metadata),
                arguments_preview=_preview_tool_args(tool_call.args),
                metadata=metadata,
                sequence_no=index,
                total_count=total_count,
                created_at=created_at,
            )
        )
    return interactions


def tool_name_by_call_id(messages: list[ModelMessage]) -> dict[str, str]:
    names: dict[str, str] = {}
    for message in messages:
        if not isinstance(message, ModelResponse):
            continue
        for part in message.parts:
            if isinstance(part, ToolCallPart):
                names[part.tool_call_id] = part.tool_name
    return names


def _interaction_kind(metadata: dict[str, Any], tool_call: ToolCallPart) -> str:
    reviewer = metadata.get("reviewer")
    if reviewer == "shell_command_reviewer":
        return "shell_review"
    if isinstance(metadata.get("mcp_server"), str):
        return "mcp_approval"
    return "tool_approval"


def _interaction_title(metadata: dict[str, Any], tool_call: ToolCallPart) -> str:
    if metadata.get("reviewer") == "shell_command_reviewer":
        return "Shell command approval required"
    full_name = metadata.get("full_name")
    if isinstance(full_name, str) and full_name.strip():
        return f"MCP tool approval required: {full_name.strip()}"
    return f"Tool approval required: {tool_call.tool_name}"


def _interaction_description(metadata: dict[str, Any]) -> str | None:
    value = metadata.get("reason")
    return value if isinstance(value, str) and value.strip() else None


def _preview_tool_args(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _preview_tool_args(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_preview_tool_args(item) for item in value[:20]]
    try:
        return json.loads(json.dumps(value, default=str))
    except TypeError:
        return str(value)
