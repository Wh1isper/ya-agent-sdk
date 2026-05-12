from __future__ import annotations

from pydantic_ai import PartDeltaEvent, PartEndEvent, PartStartEvent, TextPartDelta
from pydantic_ai.messages import TextPart
from ya_agent_sdk.context.agent import StreamEvent
from ya_agent_sdk.events import ModelRequestStartEvent
from ya_claw.agui_adapter import AguiEventAdapter, AguiReplayBuffer


def test_agui_adapter_maps_text_stream_events_and_compacts_replay() -> None:
    adapter = AguiEventAdapter(session_id="session-1", run_id="run-1")
    replay = AguiReplayBuffer()

    stream_events = [
        StreamEvent(
            agent_id="main",
            agent_name="main",
            event=ModelRequestStartEvent(event_id="run-1", loop_index=0, message_count=0),
        ),
        StreamEvent(
            agent_id="main",
            agent_name="main",
            event=PartStartEvent(index=0, part=TextPart(content="")),
        ),
        StreamEvent(
            agent_id="main",
            agent_name="main",
            event=PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="hello ")),
        ),
        StreamEvent(
            agent_id="main",
            agent_name="main",
            event=PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="world")),
        ),
        StreamEvent(
            agent_id="main",
            agent_name="main",
            event=PartEndEvent(index=0, part=TextPart(content="hello world")),
        ),
    ]

    live_events: list[dict[str, object]] = []
    for stream_event in stream_events:
        mapped = adapter.adapt_stream_event(stream_event)
        live_events.extend(mapped)
        for item in mapped:
            replay.append(item)

    assert live_events[0]["type"] == "CUSTOM"
    assert live_events[0]["name"] == "ya_agent.model_request_start"
    assert [event["type"] for event in live_events[1:]] == [
        "TEXT_MESSAGE_START",
        "TEXT_MESSAGE_CHUNK",
        "TEXT_MESSAGE_CHUNK",
        "TEXT_MESSAGE_END",
    ]

    replay.append(adapter.build_run_finished_event(result={"output_summary": "hello world"}))

    compacted = replay.snapshot()
    assert [event["type"] for event in compacted] == ["CUSTOM", "TEXT_MESSAGE_CHUNK", "RUN_FINISHED"]
    assert compacted[1]["delta"] == "hello world"
    assert "result" not in compacted[2]


def test_agui_replay_buffer_merges_tool_call_chunks() -> None:
    replay = AguiReplayBuffer()
    replay.append({
        "type": "TOOL_CALL_CHUNK",
        "toolCallId": "tool-1",
        "toolCallName": "delegate",
        "delta": '{"prompt":',
    })
    replay.append({
        "type": "TOOL_CALL_CHUNK",
        "toolCallId": "tool-1",
        "delta": '"hello"}',
    })
    replay.append({
        "type": "TOOL_CALL_RESULT",
        "toolCallId": "tool-1",
        "messageId": "tool-1:result",
        "content": "done",
        "role": "tool",
    })

    compacted = replay.snapshot()
    assert compacted[0]["type"] == "TOOL_CALL_CHUNK"
    assert compacted[0]["toolCallName"] == "delegate"
    assert compacted[0]["delta"] == '{"prompt":"hello"}'
    assert compacted[1]["type"] == "TOOL_CALL_RESULT"


def test_agui_replay_buffer_stores_chunk_fragments_until_snapshot() -> None:
    replay = AguiReplayBuffer()
    for index in range(5):
        replay.append({"type": "TEXT_MESSAGE_CHUNK", "messageId": "message-1", "delta": str(index)})

    assert replay.events == [{"type": "TEXT_MESSAGE_CHUNK", "messageId": "message-1", "delta": ""}]
    assert replay.snapshot()[0]["delta"] == "01234"


def test_agui_adapter_maps_run_lifecycle_events() -> None:
    adapter = AguiEventAdapter(session_id="session-1", run_id="run-1")

    queued = adapter.build_run_queued_event({"status": "queued"})
    started = adapter.build_run_started_event()
    finished = adapter.build_run_finished_event(result={"output_summary": "done"})
    errored = adapter.build_run_error_event(message="boom", code="error")

    assert queued["type"] == "CUSTOM"
    assert queued["name"] == "ya_claw.run_queued"
    assert started["type"] == "RUN_STARTED"
    assert started["runId"] == "run-1"
    assert finished["type"] == "RUN_FINISHED"
    assert "result" not in finished
    assert errored["type"] == "RUN_ERROR"
    assert errored["message"] == "boom"
