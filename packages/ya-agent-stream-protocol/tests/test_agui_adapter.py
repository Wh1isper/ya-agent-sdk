from __future__ import annotations

from pydantic_ai import PartDeltaEvent, PartEndEvent, PartStartEvent, TextPartDelta, ThinkingPartDelta
from pydantic_ai.messages import TextPart, ThinkingPart
from ya_agent_sdk.context.agent import StreamEvent
from ya_agent_sdk.events import ModelRequestStartEvent
from ya_agent_stream_protocol.agui import AguiReplayBuffer, AguiReplayConfig
from ya_agent_stream_protocol.sdk import AguiAdapterConfig, AguiEventAdapter

YAACLI_ADAPTER_CONFIG = AguiAdapterConfig(run_event_prefix="yaacli", stream_metadata_prefix="yaacli")
CLAW_ADAPTER_CONFIG = AguiAdapterConfig(run_event_prefix="ya_claw")
YAACLI_REPLAY_CONFIG = AguiReplayConfig(
    agent_id_field="yaacliAgentId",
    main_agent_id="main",
    drop_subagent_detail_events=True,
)


def test_agui_event_adapter_maps_text_stream_events_and_compacts_replay() -> None:
    adapter = AguiEventAdapter(session_id="session-1", run_id="run-1", config=YAACLI_ADAPTER_CONFIG)
    replay = AguiReplayBuffer(config=YAACLI_REPLAY_CONFIG)

    stream_events = [
        StreamEvent(
            agent_id="main",
            agent_name="main",
            event=ModelRequestStartEvent(event_id="run-1", loop_index=0, message_count=0),
        ),
        StreamEvent(agent_id="main", agent_name="main", event=PartStartEvent(index=0, part=TextPart(content=""))),
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
            agent_id="main", agent_name="main", event=PartEndEvent(index=0, part=TextPart(content="hello world"))
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
    assert live_events[1]["yaacliAgentId"] == "main"
    assert [event["type"] for event in live_events[1:]] == [
        "TEXT_MESSAGE_START",
        "TEXT_MESSAGE_CHUNK",
        "TEXT_MESSAGE_CHUNK",
        "TEXT_MESSAGE_END",
    ]

    replay.append(adapter.build_run_finished_event(result={"output_text": "hello world"}))

    compacted = replay.snapshot()
    assert [event["type"] for event in compacted] == ["CUSTOM", "TEXT_MESSAGE_CHUNK", "RUN_FINISHED"]
    assert compacted[1]["delta"] == "hello world"
    assert compacted[2]["result"] == {"output_text": "hello world"}


def test_agui_event_adapter_run_started_excludes_input_parts() -> None:
    adapter = AguiEventAdapter(session_id="session-1", run_id="run-1", config=CLAW_ADAPTER_CONFIG)

    event = adapter.build_run_started_event(input_parts=[{"type": "text", "text": "hello"}])

    assert event["type"] == "RUN_STARTED"
    assert "input" not in event


def test_agui_replay_buffer_keeps_runs_separate() -> None:
    replay = AguiReplayBuffer()
    replay.append({"type": "RUN_STARTED", "runId": "run-1"})
    replay.append({"type": "TEXT_MESSAGE_CHUNK", "messageId": "m1", "delta": "first"})
    replay.append({"type": "RUN_FINISHED", "runId": "run-1"})
    replay.append({"type": "RUN_STARTED", "runId": "run-2"})
    replay.append({"type": "TEXT_MESSAGE_CHUNK", "messageId": "m1", "delta": "second"})

    compacted = replay.snapshot()
    text_chunks = [event for event in compacted if event["type"] == "TEXT_MESSAGE_CHUNK"]
    assert [event["delta"] for event in text_chunks] == ["first", "second"]


def test_agui_replay_buffer_merges_tool_call_chunks() -> None:
    replay = AguiReplayBuffer()
    replay.append({
        "type": "TOOL_CALL_CHUNK",
        "toolCallId": "tool-1",
        "toolCallName": "delegate",
        "delta": '{"prompt":',
    })
    replay.append({"type": "TOOL_CALL_CHUNK", "toolCallId": "tool-1", "delta": '"hello"}'})
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


def test_agui_replay_buffer_drops_subagent_detail_events_when_configured() -> None:
    adapter = AguiEventAdapter(session_id="session-1", run_id="run-1", config=YAACLI_ADAPTER_CONFIG)
    replay = AguiReplayBuffer(config=YAACLI_REPLAY_CONFIG)

    stream_events = [
        StreamEvent(
            agent_id="worker-1",
            agent_name="worker",
            event=PartStartEvent(index=0, part=ThinkingPart(content="")),
        ),
        StreamEvent(
            agent_id="worker-1",
            agent_name="worker",
            event=PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="hidden thought")),
        ),
        StreamEvent(
            agent_id="worker-1",
            agent_name="worker",
            event=PartEndEvent(index=0, part=ThinkingPart(content="hidden thought")),
        ),
        StreamEvent(agent_id="worker-1", agent_name="worker", event=PartStartEvent(index=1, part=TextPart(content=""))),
        StreamEvent(
            agent_id="worker-1",
            agent_name="worker",
            event=PartDeltaEvent(index=1, delta=TextPartDelta(content_delta="hidden text")),
        ),
        StreamEvent(
            agent_id="worker-1", agent_name="worker", event=PartEndEvent(index=1, part=TextPart(content="hidden text"))
        ),
    ]

    live_events: list[dict[str, object]] = []
    for stream_event in stream_events:
        mapped = adapter.adapt_stream_event(stream_event)
        live_events.extend(mapped)
        for item in mapped:
            replay.append(item)

    assert any(event["type"] == "TEXT_MESSAGE_CHUNK" for event in live_events)
    assert any(event["type"] == "REASONING_MESSAGE_CHUNK" for event in live_events)
    assert replay.snapshot() == []


def test_agui_adapter_maps_run_custom_event_namespace() -> None:
    adapter = AguiEventAdapter(session_id="session-1", run_id="run-1", config=CLAW_ADAPTER_CONFIG)

    queued = adapter.build_run_custom_event("run_queued", {"status": "queued"})
    finished = adapter.build_run_finished_event(result={"output_text": "done"})
    errored = adapter.build_run_error_event(message="boom", code="error")

    assert queued["type"] == "CUSTOM"
    assert queued["name"] == "ya_claw.run_queued"
    assert finished["type"] == "RUN_FINISHED"
    assert finished["result"] == {"output_text": "done"}
    assert errored["type"] == "RUN_ERROR"
    assert errored["message"] == "boom"
