from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ya_agent_stream_protocol.json_types import JsonObject


@dataclass(slots=True)
class BufferedStreamEvent:
    id: str
    payload: JsonObject
    terminal: bool = False


def format_sse_event(event: BufferedStreamEvent) -> dict[str, str]:
    return {
        "id": event.id,
        "event": str(event.payload.get("type", "message")),
        "data": json.dumps(event.payload, ensure_ascii=False),
    }


def resolve_event_cursor(last_event_id: str | None) -> int:
    if last_event_id is None:
        return 0
    try:
        return max(int(last_event_id), 0)
    except ValueError:
        return 0


def build_buffered_stream_event(
    event_id: int | str, payload: dict[str, Any], *, terminal: bool = False
) -> BufferedStreamEvent:
    return BufferedStreamEvent(id=str(event_id), payload=dict(payload), terminal=terminal)  # type: ignore[arg-type]
