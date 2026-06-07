from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel

from ya_agent_stream_protocol.json_types import JsonObject


def dump_agui_event(event: BaseModel) -> JsonObject:
    payload = event.model_dump(mode="json", exclude_none=True, by_alias=True)
    payload.setdefault("timestamp", int(datetime.now(UTC).timestamp() * 1000))
    return payload  # type: ignore[return-value]
