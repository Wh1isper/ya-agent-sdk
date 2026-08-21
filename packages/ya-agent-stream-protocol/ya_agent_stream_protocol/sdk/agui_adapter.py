from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from ag_ui.core.events import (
    CustomEvent,
    ReasoningMessageChunkEvent,
    ReasoningMessageEndEvent,
    ReasoningMessageStartEvent,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    TextMessageChunkEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallChunkEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)
from pydantic import BaseModel
from pydantic_ai import (
    EnqueuedMessagesEvent,
    FinalResultEvent,
    FunctionToolResultEvent,
    OutputToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPartDelta,
    ThinkingPartDelta,
    ToolCallPartDelta,
)
from pydantic_ai.messages import RetryPromptPart, TextPart, ThinkingPart, ToolCallPart, ToolReturnPart
from ya_agent_sdk.context.agent import StreamEvent
from ya_agent_sdk.events import (
    AgentEvent,
    AgentExecutionCompleteEvent,
    AgentExecutionFailedEvent,
    AgentExecutionResumeEvent,
    AgentExecutionStartEvent,
    BackgroundShellCompleteEvent,
    BackgroundShellKilledEvent,
    BackgroundShellStartEvent,
    CompactCompleteEvent,
    CompactFailedEvent,
    CompactStartEvent,
    FileChangeEvent,
    HandoffCompleteEvent,
    HandoffFailedEvent,
    HandoffStartEvent,
    ModelRequestCompleteEvent,
    ModelRequestStartEvent,
    NamespaceStatusEvent,
    NoteEvent,
    SubagentCompleteEvent,
    SubagentStartEvent,
    TaskEvent,
    ToolCallsCompleteEvent,
    ToolCallsStartEvent,
    UsageSnapshotEvent,
)

from ya_agent_stream_protocol.agui.events import dump_agui_event
from ya_agent_stream_protocol.json_types import JsonObject, JsonValue


@dataclass(slots=True)
class AguiAdapterConfig:
    run_event_prefix: str
    agent_event_prefix: str = "ya_agent"
    stream_metadata_prefix: str | None = None


@dataclass(slots=True)
class PartCursor:
    kind: str
    part_id: str
    role: str | None = None
    tool_call_name: str | None = None
    emitted_chunk: bool = False


@dataclass(slots=True)
class AgentCursor:
    loop_index: int = 0
    parts: dict[int, PartCursor] = field(default_factory=dict)


class AguiEventAdapter:
    def __init__(self, *, session_id: str, run_id: str, config: AguiAdapterConfig) -> None:
        self._session_id = session_id
        self._run_id = run_id
        self._config = config
        self._agents: dict[str, AgentCursor] = {}

    def build_run_started_event(self, *, input_parts: list[JsonObject] | None = None) -> JsonObject:
        _ = input_parts
        return dump_agui_event(RunStartedEvent(thread_id=self._session_id, run_id=self._run_id))

    def build_run_finished_event(self, result: JsonValue = None) -> JsonObject:
        return dump_agui_event(RunFinishedEvent(thread_id=self._session_id, run_id=self._run_id, result=result))

    def build_run_error_event(self, *, message: str, code: str | None = None) -> JsonObject:
        return dump_agui_event(RunErrorEvent(message=message, code=code))

    def build_run_custom_event(self, event_name: str, payload: object) -> JsonObject:
        return dump_agui_event(
            CustomEvent(
                name=f"{self._config.run_event_prefix}.{event_name}",
                value=_serialize_value(payload),
            )
        )

    def adapt_stream_event(self, stream_event: StreamEvent) -> list[JsonObject]:
        cursor = self._agents.setdefault(stream_event.agent_id, AgentCursor())
        event = stream_event.event

        if isinstance(event, ModelRequestStartEvent):
            cursor.loop_index = event.loop_index

        if isinstance(event, PartStartEvent):
            return self._with_stream_metadata(stream_event, self._adapt_part_start(stream_event, cursor))
        if isinstance(event, PartDeltaEvent):
            return self._with_stream_metadata(stream_event, self._adapt_part_delta(stream_event, cursor))
        if isinstance(event, PartEndEvent):
            return self._with_stream_metadata(stream_event, self._adapt_part_end(stream_event, cursor))
        if isinstance(event, FunctionToolResultEvent | OutputToolResultEvent):
            return self._with_stream_metadata(stream_event, self._adapt_function_tool_result(stream_event))
        explicit_custom_event = self._adapt_explicit_custom_event(stream_event)
        if explicit_custom_event is not None:
            return [explicit_custom_event]
        return []

    def _adapt_explicit_custom_event(self, stream_event: StreamEvent) -> JsonObject | None:
        event = stream_event.event
        if isinstance(event, ModelRequestStartEvent):
            return self._custom_agent_event(
                event_name="model_request_start",
                stream_event=stream_event,
                payload={"loop_index": event.loop_index, "message_count": event.message_count},
            )
        if isinstance(event, ModelRequestCompleteEvent):
            return self._custom_agent_event(
                event_name="model_request_complete",
                stream_event=stream_event,
                payload={
                    "loop_index": event.loop_index,
                    "duration_seconds": event.duration_seconds,
                    "context_tokens": event.context_tokens,
                    "context_window_size": event.context_window_size,
                },
            )
        if isinstance(event, FinalResultEvent):
            return self._custom_agent_event(
                event_name="final_result",
                stream_event=stream_event,
                payload={"tool_name": event.tool_name, "tool_call_id": event.tool_call_id},
            )
        if isinstance(event, UsageSnapshotEvent):
            snapshot = event.snapshot
            return self._custom_agent_event(
                event_name="usage_snapshot",
                stream_event=stream_event,
                payload=(
                    {
                        "run_id": self._run_id,
                        "total_usage": _serialize_value(snapshot.total_usage),
                        "total_cost_estimate": _serialize_value(snapshot.total_cost_estimate),
                        "model_cost_estimates": _serialize_value(snapshot.model_cost_estimates),
                    }
                    if snapshot is not None
                    else None
                ),
            )
        if isinstance(event, SubagentStartEvent):
            return self._custom_agent_event(
                event_name="subagent_start",
                stream_event=stream_event,
                payload={
                    "execution_id": event.execution_id,
                    "mode": event.mode,
                    "agent_id": event.agent_id,
                    "agent_name": event.agent_name,
                    "prompt_preview": _bounded_text(event.prompt_preview, 200),
                },
            )
        if isinstance(event, SubagentCompleteEvent):
            return self._custom_agent_event(
                event_name="subagent_complete",
                stream_event=stream_event,
                payload={
                    "execution_id": event.execution_id,
                    "mode": event.mode,
                    "agent_id": event.agent_id,
                    "agent_name": event.agent_name,
                    "success": event.success,
                    "request_count": event.request_count,
                    "result_preview": _bounded_text(event.result_preview, 200),
                    "error": _bounded_text(event.error, 1_000),
                    "duration_seconds": event.duration_seconds,
                },
            )
        if isinstance(event, EnqueuedMessagesEvent):
            return self._custom_agent_event(
                event_name="enqueued_messages",
                stream_event=stream_event,
                payload={"message_count": len(event.messages)},
            )
        projection = _public_agent_event_projection(event)
        if projection is None:
            return None
        event_name, payload = projection
        return self._custom_agent_event(
            event_name=event_name,
            stream_event=stream_event,
            payload=payload,
        )

    def _adapt_part_start(self, stream_event: StreamEvent, cursor: AgentCursor) -> list[JsonObject]:
        event = cast(PartStartEvent, stream_event.event)
        part = event.part
        if isinstance(part, TextPart):
            message_id = part.id or self._part_id(stream_event.agent_id, cursor.loop_index, event.index, "text")
            cursor.parts[event.index] = PartCursor(kind="text", part_id=message_id, role="assistant")
            events = [
                dump_agui_event(
                    TextMessageStartEvent(message_id=message_id, role="assistant", name=stream_event.agent_name)
                )
            ]
            if part.content:
                events.append(
                    dump_agui_event(
                        TextMessageChunkEvent(
                            message_id=message_id,
                            role="assistant",
                            name=stream_event.agent_name,
                            delta=part.content,
                        )
                    )
                )
                cursor.parts[event.index].emitted_chunk = True
            return events
        if isinstance(part, ThinkingPart):
            message_id = part.id or self._part_id(stream_event.agent_id, cursor.loop_index, event.index, "reasoning")
            cursor.parts[event.index] = PartCursor(kind="reasoning", part_id=message_id, role="reasoning")
            events = [dump_agui_event(ReasoningMessageStartEvent(message_id=message_id, role="reasoning"))]
            if part.content:
                events.append(dump_agui_event(ReasoningMessageChunkEvent(message_id=message_id, delta=part.content)))
                cursor.parts[event.index].emitted_chunk = True
            return events
        if isinstance(part, ToolCallPart):
            tool_call_id = part.tool_call_id
            cursor.parts[event.index] = PartCursor(
                kind="tool_call",
                part_id=tool_call_id,
                tool_call_name=part.tool_name,
            )
            events = [dump_agui_event(ToolCallStartEvent(tool_call_id=tool_call_id, tool_call_name=part.tool_name))]
            chunk_delta = _stringify_tool_call_args(part.args)
            if chunk_delta is not None or part.tool_name:
                events.append(
                    dump_agui_event(
                        ToolCallChunkEvent(
                            tool_call_id=tool_call_id,
                            tool_call_name=part.tool_name,
                            delta=chunk_delta,
                        )
                    )
                )
                cursor.parts[event.index].emitted_chunk = True
            return events
        if isinstance(part, ToolReturnPart):
            return [self._tool_result_event(part)]
        if isinstance(part, RetryPromptPart):
            return [self._retry_prompt_event(stream_event, part)]
        return []

    def _adapt_part_delta(self, stream_event: StreamEvent, cursor: AgentCursor) -> list[JsonObject]:
        event = cast(PartDeltaEvent, stream_event.event)
        delta = event.delta
        if isinstance(delta, TextPartDelta):
            part_cursor = self._ensure_text_cursor(stream_event.agent_id, cursor, event.index)
            part_cursor.emitted_chunk = True
            return [
                dump_agui_event(
                    TextMessageChunkEvent(
                        message_id=part_cursor.part_id,
                        role="assistant",
                        name=stream_event.agent_name,
                        delta=delta.content_delta,
                    )
                )
            ]
        if isinstance(delta, ThinkingPartDelta):
            part_cursor = self._ensure_reasoning_cursor(stream_event.agent_id, cursor, event.index)
            events: list[JsonObject] = []
            if delta.content_delta:
                part_cursor.emitted_chunk = True
                events.append(
                    dump_agui_event(
                        ReasoningMessageChunkEvent(message_id=part_cursor.part_id, delta=delta.content_delta)
                    )
                )
            if getattr(delta, "signature_delta", None):
                events.append(
                    self._custom_agent_event(
                        "reasoning_signature_delta",
                        stream_event=stream_event,
                        payload={"message_id": part_cursor.part_id, "signature_delta": delta.signature_delta},
                    )
                )
            return events
        if isinstance(delta, ToolCallPartDelta):
            part_cursor = self._ensure_tool_call_cursor(stream_event.agent_id, cursor, event.index, delta.tool_call_id)
            if delta.tool_name_delta:
                part_cursor.tool_call_name = f"{part_cursor.tool_call_name or ''}{delta.tool_name_delta}" or None
            part_cursor.emitted_chunk = True
            return [
                dump_agui_event(
                    ToolCallChunkEvent(
                        tool_call_id=part_cursor.part_id,
                        tool_call_name=part_cursor.tool_call_name,
                        delta=_stringify_tool_call_args(delta.args_delta),
                    )
                )
            ]
        return []

    def _adapt_part_end(self, stream_event: StreamEvent, cursor: AgentCursor) -> list[JsonObject]:
        event = cast(PartEndEvent, stream_event.event)
        part = event.part
        part_cursor = cursor.parts.pop(event.index, None)
        if isinstance(part, TextPart):
            message_id = (
                part_cursor.part_id
                if part_cursor is not None
                else part.id or self._part_id(stream_event.agent_id, cursor.loop_index, event.index, "text")
            )
            events: list[JsonObject] = []
            emitted_chunk = part_cursor.emitted_chunk if part_cursor is not None else False
            if part.content and not emitted_chunk:
                events.append(
                    dump_agui_event(
                        TextMessageChunkEvent(
                            message_id=message_id,
                            role="assistant",
                            name=stream_event.agent_name,
                            delta=part.content,
                        )
                    )
                )
            events.append(dump_agui_event(TextMessageEndEvent(message_id=message_id)))
            return events
        if isinstance(part, ThinkingPart):
            message_id = (
                part_cursor.part_id
                if part_cursor is not None
                else part.id or self._part_id(stream_event.agent_id, cursor.loop_index, event.index, "reasoning")
            )
            events = []
            emitted_chunk = part_cursor.emitted_chunk if part_cursor is not None else False
            if part.content and not emitted_chunk:
                events.append(dump_agui_event(ReasoningMessageChunkEvent(message_id=message_id, delta=part.content)))
            events.append(dump_agui_event(ReasoningMessageEndEvent(message_id=message_id)))
            return events
        if isinstance(part, ToolCallPart):
            tool_call_id = part.tool_call_id if part_cursor is None else part_cursor.part_id
            events = []
            emitted_chunk = part_cursor.emitted_chunk if part_cursor is not None else False
            if not emitted_chunk:
                events.append(
                    dump_agui_event(
                        ToolCallChunkEvent(
                            tool_call_id=tool_call_id,
                            tool_call_name=part.tool_name,
                            delta=_stringify_tool_call_args(part.args),
                        )
                    )
                )
            events.append(dump_agui_event(ToolCallEndEvent(tool_call_id=tool_call_id)))
            return events
        if isinstance(part, ToolReturnPart):
            return [self._tool_result_event(part)]
        if isinstance(part, RetryPromptPart):
            return [self._retry_prompt_event(stream_event, part)]
        return []

    def _adapt_function_tool_result(self, stream_event: StreamEvent) -> list[JsonObject]:
        event = cast(FunctionToolResultEvent | OutputToolResultEvent, stream_event.event)
        part = event.part
        content = event.content if isinstance(event, FunctionToolResultEvent) else None
        if isinstance(part, ToolReturnPart):
            return [self._tool_result_event(part, content=content)]
        if isinstance(part, RetryPromptPart):
            return [self._retry_prompt_event(stream_event, part)]
        return []

    def _retry_prompt_event(self, stream_event: StreamEvent, part: RetryPromptPart) -> JsonObject:
        return self._custom_agent_event(
            "retry_prompt_part",
            stream_event=stream_event,
            payload={
                "tool_name": part.tool_name,
                "tool_call_id": part.tool_call_id,
            },
        )

    def _tool_result_event(self, part: ToolReturnPart, *, content: object = None) -> JsonObject:
        tool_call_id = part.tool_call_id
        return dump_agui_event(
            ToolCallResultEvent(
                message_id=f"{tool_call_id}:result",
                tool_call_id=tool_call_id,
                content=_stringify_tool_result(content if content is not None else part.content),
                role="tool",
            )
        )

    def _ensure_text_cursor(self, agent_id: str, cursor: AgentCursor, index: int) -> PartCursor:
        existing = cursor.parts.get(index)
        if existing is not None:
            return existing
        part_cursor = PartCursor(
            kind="text",
            part_id=self._part_id(agent_id, cursor.loop_index, index, "text"),
            role="assistant",
        )
        cursor.parts[index] = part_cursor
        return part_cursor

    def _ensure_reasoning_cursor(self, agent_id: str, cursor: AgentCursor, index: int) -> PartCursor:
        existing = cursor.parts.get(index)
        if existing is not None:
            return existing
        part_cursor = PartCursor(
            kind="reasoning",
            part_id=self._part_id(agent_id, cursor.loop_index, index, "reasoning"),
            role="reasoning",
        )
        cursor.parts[index] = part_cursor
        return part_cursor

    def _ensure_tool_call_cursor(
        self,
        agent_id: str,
        cursor: AgentCursor,
        index: int,
        tool_call_id: str | None,
    ) -> PartCursor:
        existing = cursor.parts.get(index)
        if existing is not None:
            if tool_call_id:
                existing.part_id = tool_call_id
            return existing
        part_cursor = PartCursor(
            kind="tool_call",
            part_id=tool_call_id or self._part_id(agent_id, cursor.loop_index, index, "tool_call"),
        )
        cursor.parts[index] = part_cursor
        return part_cursor

    def _part_id(self, agent_id: str, loop_index: int, part_index: int, kind: str) -> str:
        return f"{self._run_id}:{agent_id}:{loop_index}:{kind}:{part_index}"

    def _with_stream_metadata(self, stream_event: StreamEvent, events: list[JsonObject]) -> list[JsonObject]:
        prefix = self._config.stream_metadata_prefix
        if prefix is None:
            return events
        agent_id_key = f"{prefix}AgentId"
        agent_name_key = f"{prefix}AgentName"
        for event in events:
            event[agent_id_key] = stream_event.agent_id
            event[agent_name_key] = stream_event.agent_name
        return events

    def _custom_agent_event(self, event_name: str, *, stream_event: StreamEvent, payload: object) -> JsonObject:
        return dump_agui_event(
            CustomEvent(
                name=f"{self._config.agent_event_prefix}.{event_name}",
                value={
                    "run_id": self._run_id,
                    "session_id": self._session_id,
                    "agent_id": stream_event.agent_id,
                    "agent_name": stream_event.agent_name,
                    "payload": _serialize_value(payload),
                },
            )
        )


def _public_agent_event_projection(event: object) -> tuple[str, JsonValue] | None:  # noqa: C901
    if isinstance(event, CompactStartEvent):
        return "compact_start", {"message_count": event.message_count}
    if isinstance(event, CompactCompleteEvent):
        return "compact_complete", {
            "original_message_count": event.original_message_count,
            "compacted_message_count": event.compacted_message_count,
        }
    if isinstance(event, CompactFailedEvent):
        return "compact_failed", {
            "error": _bounded_text(event.error, 1_000),
            "message_count": event.message_count,
        }
    if isinstance(event, HandoffStartEvent):
        return "handoff_start", {"message_count": event.message_count}
    if isinstance(event, HandoffCompleteEvent):
        return "handoff_complete", {"original_message_count": event.original_message_count}
    if isinstance(event, HandoffFailedEvent):
        return "handoff_failed", {
            "error": _bounded_text(event.error, 1_000),
            "message_count": event.message_count,
        }
    if isinstance(event, AgentExecutionStartEvent):
        return "agent_execution_start", {
            "message_history_count": event.message_history_count,
            "attempt_index": event.attempt_index,
            "is_resume_attempt": event.is_resume_attempt,
        }
    if isinstance(event, AgentExecutionCompleteEvent):
        return "agent_execution_complete", {
            "total_loops": event.total_loops,
            "total_duration_seconds": event.total_duration_seconds,
            "final_message_count": event.final_message_count,
            "attempt_index": event.attempt_index,
        }
    if isinstance(event, AgentExecutionFailedEvent):
        return "agent_execution_failed", {
            "error": _bounded_text(event.error, 1_000),
            "error_type": event.error_type,
            "total_loops": event.total_loops,
            "total_duration_seconds": event.total_duration_seconds,
            "attempt_index": event.attempt_index,
            "recoverable": event.recoverable,
        }
    if isinstance(event, AgentExecutionResumeEvent):
        return "agent_execution_resume", {
            "attempt_index": event.attempt_index,
            "previous_attempt_index": event.previous_attempt_index,
            "error": _bounded_text(event.error, 1_000),
            "error_type": event.error_type,
            "message_history_count": event.message_history_count,
        }
    if isinstance(event, ToolCallsStartEvent):
        return "tool_calls_start", {"loop_index": event.loop_index}
    if isinstance(event, ToolCallsCompleteEvent):
        return "tool_calls_complete", {
            "loop_index": event.loop_index,
            "duration_seconds": event.duration_seconds,
        }
    if isinstance(event, NamespaceStatusEvent):
        return "namespace_status", {
            "namespace_status": {key: value.value for key, value in event.namespace_status.items()}
        }
    if isinstance(event, FileChangeEvent):
        return "file_change", {
            "tool_name": event.tool_name,
            "changes": [
                {
                    "path": change.path,
                    "action": change.action.value,
                    "destination": change.destination,
                }
                for change in event.changes
            ],
        }
    if isinstance(event, TaskEvent):
        return "task", {
            "tasks": [
                {
                    "id": task.id,
                    "subject": task.subject,
                    "status": task.status,
                    "active_form": task.active_form,
                    "owner": task.owner,
                    "blocked_by": list(task.blocked_by),
                    "blocks": list(task.blocks),
                }
                for task in event.tasks
            ]
        }
    if isinstance(event, NoteEvent):
        return "note", {"entries": dict(event.entries)}
    if isinstance(event, BackgroundShellStartEvent):
        return "background_shell_start", {"process_id": event.process_id}
    if isinstance(event, BackgroundShellCompleteEvent):
        return "background_shell_complete", {
            "process_id": event.process_id,
            "exit_code": event.exit_code,
        }
    if isinstance(event, BackgroundShellKilledEvent):
        return "background_shell_killed", {"process_id": event.process_id}
    if isinstance(event, AgentEvent):
        return None
    return None


def _stringify_tool_call_args(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(_serialize_value(value), ensure_ascii=False, separators=(",", ":"))


def _stringify_tool_result(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(_serialize_value(value), ensure_ascii=False)


def _bounded_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    suffix = "..."
    return f"{value[: max_chars - len(suffix)]}{suffix}"


def _serialize_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")  # type: ignore[return-value]
    if is_dataclass(value) and not isinstance(value, type):
        return _serialize_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): _serialize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize_value(item) for item in value]
    return str(value)
