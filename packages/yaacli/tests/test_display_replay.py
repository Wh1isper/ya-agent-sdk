from __future__ import annotations

import json

from ya_agent_stream_protocol.agui import AguiReplayConfig
from yaacli.display_replay import BoundedDisplayReplay, DisplayReplayLimits


def test_receiver_projection_bounds_chunks_without_changing_live_event() -> None:
    replay = BoundedDisplayReplay(
        limits=DisplayReplayLimits(
            max_runs=2,
            max_events=100,
            max_bytes=10_000,
            max_chunk_chars=32,
            max_event_payload_chars=16,
        )
    )
    event = {"type": "TEXT_MESSAGE_CHUNK", "messageId": "m1", "delta": "x" * 100}

    replay.append(event)

    assert event["delta"] == "x" * 100
    snapshot = replay.snapshot()
    assert len(snapshot[0]["delta"]) == 32
    assert snapshot[0]["yaacliReplayTruncated"] is True


def test_receiver_projection_bounds_tool_result_payload() -> None:
    replay = BoundedDisplayReplay(
        limits=DisplayReplayLimits(
            max_runs=2,
            max_events=100,
            max_bytes=10_000,
            max_chunk_chars=100,
            max_event_payload_chars=32,
        )
    )
    replay.append({
        "type": "TOOL_CALL_RESULT",
        "toolCallId": "tool-1",
        "content": "result" * 100,
    })

    event = replay.snapshot()[0]
    assert event["toolCallId"] == "tool-1"
    assert len(event["content"]) == 32
    assert event["yaacliReplayTruncated"] is True


def test_receiver_projection_retains_only_recent_runs() -> None:
    replay = BoundedDisplayReplay(
        config=AguiReplayConfig(),
        limits=DisplayReplayLimits(max_runs=2, max_events=100, max_bytes=10_000),
    )
    for run in range(5):
        replay.append({"type": "RUN_STARTED", "runId": f"run-{run}"})
        replay.append({"type": "TEXT_MESSAGE_CHUNK", "messageId": f"m-{run}", "delta": str(run)})
        replay.append({"type": "RUN_FINISHED", "runId": f"run-{run}"})

    snapshot = replay.snapshot()
    assert [event["runId"] for event in snapshot if event["type"] == "RUN_STARTED"] == ["run-3", "run-4"]


def test_receiver_projection_marks_chunk_truncated_after_exact_limit() -> None:
    replay = BoundedDisplayReplay(limits=DisplayReplayLimits(max_chunk_chars=8, max_event_payload_chars=8))
    replay.append({"type": "TEXT_MESSAGE_CHUNK", "messageId": "m1", "delta": "12345678"})
    replay.append({"type": "TEXT_MESSAGE_CHUNK", "messageId": "m1", "delta": "9"})

    event = replay.snapshot()[0]
    assert event["yaacliReplayTruncated"] is True
    assert event["delta"].endswith("...")
    assert len(event["delta"]) == 8


def test_receiver_projection_is_bounded_between_snapshots() -> None:
    max_bytes = 10_000
    replay = BoundedDisplayReplay(
        limits=DisplayReplayLimits(
            max_runs=10,
            max_events=100,
            max_bytes=max_bytes,
            max_chunk_chars=4096,
            max_event_payload_chars=4096,
        )
    )
    for index in range(255):
        replay.append({
            "type": "TEXT_MESSAGE_CHUNK",
            "messageId": f"message-{index}",
            "delta": "x" * 4096,
        })
        assert replay.retained_bytes <= max_bytes


def test_receiver_projection_accounts_for_json_escaping_and_bounds_chunk_identity() -> None:
    max_bytes = 3000
    replay = BoundedDisplayReplay(
        limits=DisplayReplayLimits(
            max_bytes=max_bytes,
            max_chunk_chars=500,
            max_event_payload_chars=500,
        )
    )
    long_id = "message-" + "i" * 1500
    long_run_id = "run-" + "r" * 1500
    replay.append({
        "type": "TEXT_MESSAGE_CHUNK",
        "runId": long_run_id,
        "messageId": long_id,
        "delta": "\x00" * 50,
    })
    replay.append({
        "type": "TEXT_MESSAGE_CHUNK",
        "runId": long_run_id,
        "messageId": long_id,
        "delta": "\x00" * 50,
        "role": "\x00" * 1024,
        "name": "\x00" * 1024,
    })

    assert replay.retained_bytes <= max_bytes
    assert all(len(part) <= 1024 for key in replay._chunk_chars for part in key)
    snapshot = replay.snapshot()
    assert (
        sum(len(json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) for event in snapshot)
        <= max_bytes
    )
    assert all(len(str(event.get("messageId", ""))) <= 1024 for event in snapshot)


def test_receiver_projection_enforces_total_event_and_byte_budgets() -> None:
    replay = BoundedDisplayReplay(
        limits=DisplayReplayLimits(
            max_runs=10,
            max_events=5,
            max_bytes=300,
            max_chunk_chars=100,
            max_event_payload_chars=100,
        )
    )
    for index in range(20):
        replay.append({"type": "CUSTOM", "name": "test", "value": f"{index}-" + "x" * 80})

    snapshot = replay.snapshot()
    assert len(snapshot) <= 5
    serialized_bytes = sum(
        len(json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) for event in snapshot
    )
    assert serialized_bytes <= 300
