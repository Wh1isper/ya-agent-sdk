from __future__ import annotations

import pytest
from ya_agent_stream_protocol.agui import parse_message_events


def test_parse_message_events_accepts_top_level_event_array() -> None:
    payload = [{"type": "message", "content": "hello"}]

    assert parse_message_events(payload) == payload


@pytest.mark.parametrize("payload", [{"events": []}, "invalid", 1])
def test_parse_message_events_rejects_non_array_payload(payload: object) -> None:
    with pytest.raises(TypeError, match="top-level JSON array"):
        parse_message_events(payload)


def test_parse_message_events_rejects_non_object_entries() -> None:
    with pytest.raises(TypeError, match="AGUI event objects"):
        parse_message_events([{"type": "message"}, "bad-entry"])
