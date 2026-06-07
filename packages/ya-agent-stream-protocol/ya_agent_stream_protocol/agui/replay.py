from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ya_agent_stream_protocol.json_types import JsonObject, JsonValue

_REPLAY_DROP_EVENT_TYPES = frozenset({
    "TEXT_MESSAGE_START",
    "TEXT_MESSAGE_END",
    "REASONING_MESSAGE_START",
    "REASONING_MESSAGE_END",
    "TOOL_CALL_START",
    "TOOL_CALL_END",
})
_SUBAGENT_DETAIL_EVENT_TYPES = frozenset({
    "TEXT_MESSAGE_START",
    "TEXT_MESSAGE_CHUNK",
    "TEXT_MESSAGE_END",
    "REASONING_MESSAGE_START",
    "REASONING_MESSAGE_CHUNK",
    "REASONING_MESSAGE_END",
    "TOOL_CALL_START",
    "TOOL_CALL_CHUNK",
    "TOOL_CALL_END",
    "TOOL_CALL_RESULT",
})


@dataclass(slots=True)
class AguiReplayConfig:
    agent_id_field: str | None = None
    main_agent_id: str = "main"
    drop_subagent_detail_events: bool = False


@dataclass(slots=True)
class AguiReplayBuffer:
    config: AguiReplayConfig = field(default_factory=AguiReplayConfig)
    events: list[JsonObject] = field(default_factory=list)
    _text_chunk_index: dict[str, int] = field(default_factory=dict)
    _reasoning_chunk_index: dict[str, int] = field(default_factory=dict)
    _tool_chunk_index: dict[str, int] = field(default_factory=dict)
    _chunk_fragments: dict[int, list[str]] = field(default_factory=dict)

    def append(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type", "")).strip()
        if event_type == "":
            return
        if self.is_subagent_detail_event(event):
            return
        if event_type in _REPLAY_DROP_EVENT_TYPES:
            return
        if event_type == "TEXT_MESSAGE_CHUNK":
            self._merge_text_chunk(event)
            return
        if event_type == "REASONING_MESSAGE_CHUNK":
            self._merge_reasoning_chunk(event)
            return
        if event_type == "TOOL_CALL_CHUNK":
            self._merge_tool_call_chunk(event)
            return
        self._append_passthrough_event(event)

    def extend_snapshot(self, events: list[dict[str, Any]]) -> None:
        self.clear()
        for event in events:
            self.append(event)

    def snapshot(self) -> list[JsonObject]:
        snapshot: list[JsonObject] = []
        for index, event in enumerate(self.events):
            event_copy = dict(event)
            fragments = self._chunk_fragments.get(index)
            if fragments:
                event_copy["delta"] = "".join(fragments)
            snapshot.append(event_copy)
        return snapshot

    def clear(self) -> None:
        self.events.clear()
        self._text_chunk_index.clear()
        self._reasoning_chunk_index.clear()
        self._tool_chunk_index.clear()
        self._chunk_fragments.clear()

    def is_subagent_event(self, event: dict[str, Any]) -> bool:
        agent_id_field = self.config.agent_id_field
        if agent_id_field is None:
            return False
        agent_id = _normalized_identifier(_event_field(event, agent_id_field, _camel_to_snake(agent_id_field)))
        return agent_id is not None and agent_id != self.config.main_agent_id

    def is_subagent_detail_event(self, event: dict[str, Any]) -> bool:
        if not self.config.drop_subagent_detail_events:
            return False
        event_type = str(event.get("type", "")).strip()
        return event_type in _SUBAGENT_DETAIL_EVENT_TYPES and self.is_subagent_event(event)

    def _merge_text_chunk(self, event: dict[str, Any]) -> None:
        message_id = _normalized_identifier(_event_field(event, "messageId", "message_id"))
        if message_id is None:
            self._append_passthrough_event(event)
            return
        existing_index = self._text_chunk_index.get(message_id)
        if existing_index is None:
            self._text_chunk_index[message_id] = self._append_chunk_event(event)
            return
        self._append_delta_fragment(existing_index, event.get("delta"))
        existing = self.events[existing_index]
        if existing.get("role") is None and event.get("role") is not None:
            existing["role"] = event.get("role")
        if existing.get("name") is None and event.get("name") is not None:
            existing["name"] = event.get("name")

    def _merge_reasoning_chunk(self, event: dict[str, Any]) -> None:
        message_id = _normalized_identifier(_event_field(event, "messageId", "message_id"))
        if message_id is None:
            self._append_passthrough_event(event)
            return
        existing_index = self._reasoning_chunk_index.get(message_id)
        if existing_index is None:
            self._reasoning_chunk_index[message_id] = self._append_chunk_event(event)
            return
        self._append_delta_fragment(existing_index, event.get("delta"))

    def _merge_tool_call_chunk(self, event: dict[str, Any]) -> None:
        tool_call_id = _normalized_identifier(_event_field(event, "toolCallId", "tool_call_id"))
        if tool_call_id is None:
            self._append_passthrough_event(event)
            return
        existing_index = self._tool_chunk_index.get(tool_call_id)
        if existing_index is None:
            self._tool_chunk_index[tool_call_id] = self._append_chunk_event(event)
            return
        self._append_delta_fragment(existing_index, event.get("delta"))
        existing = self.events[existing_index]
        tool_call_name = _event_field(event, "toolCallName", "tool_call_name")
        if existing.get("toolCallName") is None and tool_call_name is not None:
            existing["toolCallName"] = tool_call_name
        parent_message_id = _event_field(event, "parentMessageId", "parent_message_id")
        if existing.get("parentMessageId") is None and parent_message_id is not None:
            existing["parentMessageId"] = parent_message_id

    def _append_passthrough_event(self, event: dict[str, Any]) -> None:
        self._forget_previous_run_state_if_needed(event)
        self.events.append(_coerce_json_object(event))

    def _append_chunk_event(self, event: dict[str, Any]) -> int:
        self._forget_previous_run_state_if_needed(event)
        event_copy = _coerce_json_object(event)
        fragment = _delta_fragment(event_copy.get("delta"))
        if fragment is not None:
            event_copy["delta"] = ""
        index = len(self.events)
        self.events.append(event_copy)
        if fragment is not None:
            self._chunk_fragments[index] = [fragment]
        return index

    def _append_delta_fragment(self, index: int, value: object) -> None:
        fragment = _delta_fragment(value)
        if fragment is None:
            return
        self._chunk_fragments.setdefault(index, []).append(fragment)

    def _forget_previous_run_state_if_needed(self, event: dict[str, Any]) -> None:
        if event.get("type") != "RUN_STARTED":
            return
        self._text_chunk_index.clear()
        self._reasoning_chunk_index.clear()
        self._tool_chunk_index.clear()
        self._chunk_fragments = {
            index: fragments for index, fragments in self._chunk_fragments.items() if index < len(self.events)
        }


def compact_agui_events(events: list[dict[str, Any]], *, config: AguiReplayConfig | None = None) -> list[JsonObject]:
    replay = AguiReplayBuffer(config=config or AguiReplayConfig())
    for event in events:
        replay.append(event)
    return replay.snapshot()


def is_subagent_detail_event(
    event: dict[str, Any], *, agent_id_field: str = "yaacliAgentId", main_agent_id: str = "main"
) -> bool:
    replay = AguiReplayBuffer(
        config=AguiReplayConfig(
            agent_id_field=agent_id_field,
            main_agent_id=main_agent_id,
            drop_subagent_detail_events=True,
        )
    )
    return replay.is_subagent_detail_event(event)


def is_subagent_event(
    event: dict[str, Any], *, agent_id_field: str = "yaacliAgentId", main_agent_id: str = "main"
) -> bool:
    replay = AguiReplayBuffer(config=AguiReplayConfig(agent_id_field=agent_id_field, main_agent_id=main_agent_id))
    return replay.is_subagent_event(event)


def _coerce_json_object(event: dict[str, Any]) -> JsonObject:
    return dict(event)  # type: ignore[return-value]


def _event_field(event: dict[str, Any], camel_name: str, snake_name: str) -> JsonValue:
    if camel_name in event:
        return event[camel_name]  # type: ignore[return-value]
    return event.get(snake_name)  # type: ignore[return-value]


def _normalized_identifier(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _delta_fragment(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _camel_to_snake(value: str) -> str:
    output: list[str] = []
    for index, character in enumerate(value):
        if character.isupper() and index > 0:
            output.append("_")
        output.append(character.lower())
    return "".join(output)
