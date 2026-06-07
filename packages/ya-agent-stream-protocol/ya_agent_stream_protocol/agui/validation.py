from __future__ import annotations

from typing import Any

from ya_agent_stream_protocol.json_types import JsonObject, JsonValue


def validate_agui_events(
    raw_payload: object,
    *,
    payload_name: str = "AGUI events payload",
    allow_none: bool = False,
) -> list[JsonObject] | None:
    if raw_payload is None and allow_none:
        return None
    if not isinstance(raw_payload, list):
        raise TypeError(f"{payload_name} must be a top-level JSON array of AGUI event objects")
    parsed_events: list[JsonObject] = [event for event in raw_payload if isinstance(event, dict)]
    if len(parsed_events) != len(raw_payload):
        raise TypeError(f"{payload_name} must contain only AGUI event objects")
    return parsed_events


def parse_message_events(raw_message_payload: JsonValue) -> list[JsonObject] | None:
    return validate_agui_events(raw_message_payload, payload_name="message payload", allow_none=True)


def parse_required_message_events(
    raw_message_payload: JsonValue, *, payload_name: str = "message payload"
) -> list[JsonObject]:
    parsed = validate_agui_events(raw_message_payload, payload_name=payload_name, allow_none=False)
    if parsed is None:
        raise TypeError(f"{payload_name} must be a top-level JSON array of AGUI event objects")
    return parsed


def validate_display_events(raw_payload: object) -> list[JsonObject]:
    parsed = validate_agui_events(raw_payload, payload_name="display_messages payload", allow_none=False)
    if parsed is None:
        raise TypeError("display_messages payload must be a top-level JSON array of event objects")
    return parsed


def coerce_json_object(value: dict[str, Any]) -> JsonObject:
    return dict(value)  # type: ignore[return-value]
