from __future__ import annotations

import hashlib
from collections.abc import Sequence

from pydantic import JsonValue

from yaacli.durable.models import InputRecord, InputState

MAX_STEERING_PREVIEW_CHARS = 100
MAX_STEERING_PREVIEW_MESSAGES = 8
STEERING_ACCEPTED_EVENT_NAME = "yaacli.steering_accepted"
STEERING_APPLIED_EVENT_NAME = "yaacli.steering_applied"
DURABLE_STEERING_EVENT_NAMES = frozenset({
    STEERING_ACCEPTED_EVENT_NAME,
    STEERING_APPLIED_EVENT_NAME,
})


def _custom_event(name: str, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {"type": "CUSTOM", "name": name, "value": value}


def steering_projection_key(session_id: str, input_id: str) -> str:
    """Derive a replay-stable display identity without exposing a durable input ID."""
    key = hashlib.sha256(session_id.encode()).digest()
    digest = hashlib.blake2b(input_id.encode(), key=key, digest_size=16).hexdigest()
    return f"steering-{digest}"


def single_line_steering_preview(value: str) -> str:
    """Build a bounded display-only preview of applied steering input."""
    printable = "".join(character if character.isprintable() else " " for character in value)
    normalized = " ".join(printable.split())
    if len(normalized) <= MAX_STEERING_PREVIEW_CHARS:
        return normalized
    suffix = "..."
    return f"{normalized[: MAX_STEERING_PREVIEW_CHARS - len(suffix)].rstrip()}{suffix}"


def durable_steering_display_events(
    session_id: str,
    inputs: Sequence[InputRecord],
) -> list[JsonValue]:
    """Reconstruct run-scoped steering facts from canonical durable input rows."""
    events: list[JsonValue] = []
    for item in inputs:
        if item.order_index <= 0 or item.origin != "user":
            continue
        messages = [content for content in item.content if isinstance(content, str)]
        if not messages:
            continue
        projection_key = steering_projection_key(session_id, item.input_id)
        events.append(
            _custom_event(
                STEERING_ACCEPTED_EVENT_NAME,
                {"projection_key": projection_key},
            )
        )
        if item.state is not InputState.applied:
            continue
        previews: list[JsonValue] = [
            preview
            for message in messages[:MAX_STEERING_PREVIEW_MESSAGES]
            if (preview := single_line_steering_preview(message))
        ]
        if previews:
            events.append(
                _custom_event(
                    STEERING_APPLIED_EVENT_NAME,
                    {
                        "projection_key": projection_key,
                        "messages": previews,
                    },
                )
            )
    return events
