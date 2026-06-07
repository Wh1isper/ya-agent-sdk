from __future__ import annotations

import json

import pytest
from ya_agent_stream_protocol.agui import (
    BufferedStreamEvent,
    format_sse_event,
    parse_message_events,
    resolve_event_cursor,
)


def test_parse_message_events_accepts_top_level_event_array() -> None:
    payload = [{"type": "TEXT_MESSAGE_CHUNK", "messageId": "m1", "delta": "hello"}]

    assert parse_message_events(payload) == payload


@pytest.mark.parametrize("payload", [{"events": []}, "[]", 123])
def test_parse_message_events_rejects_non_array_payload(payload: object) -> None:
    with pytest.raises(TypeError, match="top-level JSON array"):
        parse_message_events(payload)  # type: ignore[arg-type]


def test_parse_message_events_rejects_non_object_entries() -> None:
    with pytest.raises(TypeError, match="AGUI event objects"):
        parse_message_events([{"type": "message"}, "bad-entry"])


def test_sse_formatting_uses_agui_type_as_event_name() -> None:
    event = BufferedStreamEvent(id="2", payload={"type": "RUN_FINISHED", "result": {"output_text": "done"}})

    framed = format_sse_event(event)

    assert framed["id"] == "2"
    assert framed["event"] == "RUN_FINISHED"
    assert json.loads(framed["data"])["result"] == {"output_text": "done"}


@pytest.mark.parametrize(("last_event_id", "cursor"), [(None, 0), ("0", 0), ("1", 1), ("bad", 0), ("-1", 0)])
def test_resolve_event_cursor(last_event_id: str | None, cursor: int) -> None:
    assert resolve_event_cursor(last_event_id) == cursor
