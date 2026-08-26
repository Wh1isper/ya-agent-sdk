from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal

from pydantic_ai import (
    EnqueuedMessagesEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPartDelta,
    ThinkingPartDelta,
)
from pydantic_ai.messages import ModelRequest, TextPart, ThinkingPart, UserPromptPart
from pydantic_ai.usage import RunUsage
from ya_agent_sdk.context.agent import StreamEvent
from ya_agent_sdk.events import (
    AgentEvent,
    AgentExecutionResumeEvent,
    AgentExecutionStartEvent,
    BackgroundShellStartEvent,
    CompactCompleteEvent,
    FileChange,
    FileChangeEvent,
    HandoffCompleteEvent,
    ModelRequestStartEvent,
    SubagentCompleteEvent,
    SubagentStartEvent,
    TaskEvent,
    TaskInfo,
    TextReplacement,
    UsageSnapshotEvent,
)
from ya_agent_sdk.usage import CostEstimate, UsageSnapshot
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


def test_agui_event_adapter_serializes_usage_cost_decimals_as_strings() -> None:
    adapter = AguiEventAdapter(session_id="session-1", run_id="run-1", config=CLAW_ADAPTER_CONFIG)
    snapshot = UsageSnapshot(
        run_id="run-1",
        total_usage=RunUsage(requests=1, input_tokens=10, output_tokens=2),
        total_cost_estimate=CostEstimate(
            input_amount=Decimal("0.001"),
            output_amount=Decimal("0.002"),
            total_amount=Decimal("0.003"),
            priced_requests=1,
        ),
    )

    [event] = adapter.adapt_stream_event(
        StreamEvent(
            agent_id="main",
            agent_name="main",
            event=UsageSnapshotEvent(event_id="usage-1", snapshot=snapshot),
        )
    )

    assert event["type"] == "CUSTOM"
    assert event["name"] == "ya_agent.usage_snapshot"
    value = event["value"]
    assert isinstance(value, dict)
    payload = value["payload"]
    assert isinstance(payload, dict)
    estimate = payload["total_cost_estimate"]
    assert isinstance(estimate, dict)
    assert estimate["total_amount"] == "0.003"
    assert estimate["priced_requests"] == 1


def test_agui_event_adapter_redacts_internal_subagent_and_enqueue_state() -> None:
    adapter = AguiEventAdapter(session_id="session-1", run_id="run-1", config=CLAW_ADAPTER_CONFIG)
    internal_parent_run_id = "550e8400-e29b-41d4-a716-446655440000"
    internal_enqueue_id = "9e7da301-bbb3-40f4-a570-76e8c25e73b8"
    sensitive_input = "private steering content"

    events = [
        *adapter.adapt_stream_event(
            StreamEvent(
                agent_id="worker-bg-a7b9",
                agent_name="worker",
                event=SubagentStartEvent(
                    event_id="worker-bg-a7b9",
                    execution_id="worker-bg-a7b9",
                    mode="background",
                    parent_logical_run_id=internal_parent_run_id,
                    agent_id="worker-bg-a7b9",
                    agent_name="worker",
                    prompt_preview="inspect code",
                ),
            )
        ),
        *adapter.adapt_stream_event(
            StreamEvent(
                agent_id="worker-bg-a7b9",
                agent_name="worker",
                event=SubagentCompleteEvent(
                    event_id="worker-bg-a7b9",
                    execution_id="worker-bg-a7b9",
                    mode="background",
                    parent_logical_run_id=internal_parent_run_id,
                    agent_id="worker-bg-a7b9",
                    agent_name="worker",
                    result_preview="done",
                ),
            )
        ),
        *adapter.adapt_stream_event(
            StreamEvent(
                agent_id="main",
                agent_name="main",
                event=EnqueuedMessagesEvent(
                    enqueue_id=internal_enqueue_id,
                    messages=(ModelRequest(parts=[UserPromptPart(content=sensitive_input)]),),
                ),
            )
        ),
    ]

    assert [event["name"] for event in events] == [
        "ya_agent.subagent_start",
        "ya_agent.subagent_complete",
        "ya_agent.enqueued_messages",
    ]
    serialized = json.dumps(events)
    assert internal_parent_run_id not in serialized
    assert internal_enqueue_id not in serialized
    assert sensitive_input not in serialized
    assert events[2]["value"]["payload"] == {"message_count": 1}  # type: ignore[index]


def test_agui_event_adapter_lifecycle_projection_is_fail_closed() -> None:
    adapter = AguiEventAdapter(session_id="session-1", run_id="run-1", config=CLAW_ADAPTER_CONFIG)
    secret = "TOP-SECRET-SENTINEL"  # noqa: S105
    internal_event_id = "internal-event-id"

    @dataclass
    class FutureSensitiveEvent(AgentEvent):
        secret_value: str = ""

    source_events = [
        AgentExecutionStartEvent(
            event_id=internal_event_id,
            user_prompt=secret,
            message_history_count=3,
        ),
        AgentExecutionResumeEvent(
            event_id=internal_event_id,
            resume_prompt=secret,
            message_history_count=4,
        ),
        CompactCompleteEvent(
            event_id=internal_event_id,
            summary_markdown=secret,
            condense_result={"secret": secret},
            original_message_count=10,
            compacted_message_count=2,
        ),
        HandoffCompleteEvent(
            event_id=internal_event_id,
            handoff_content=secret,
            original_message_count=5,
        ),
        FileChangeEvent(
            event_id=internal_event_id,
            changes=[
                FileChange(
                    path="safe.txt",
                    replacements=[TextReplacement(old_string=secret, new_string=secret)],
                )
            ],
            tool_name="edit",
        ),
        TaskEvent(
            event_id=internal_event_id,
            tasks=[TaskInfo(id="task-1", subject="safe", description=secret)],
        ),
        BackgroundShellStartEvent(
            event_id=internal_event_id,
            process_id="process-1",
            command=secret,
        ),
    ]

    projected = [
        item
        for source_event in source_events
        for item in adapter.adapt_stream_event(StreamEvent(agent_id="main", agent_name="main", event=source_event))
    ]
    serialized = json.dumps(projected)

    assert len(projected) == len(source_events)
    assert secret not in serialized
    assert internal_event_id not in serialized
    assert (
        adapter.adapt_stream_event(
            StreamEvent(
                agent_id="main",
                agent_name="main",
                event=FutureSensitiveEvent(event_id=internal_event_id, secret_value=secret),
            )
        )
        == []
    )


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


def test_agui_replay_buffer_retains_tool_start_without_a_chunk() -> None:
    replay = AguiReplayBuffer()
    replay.append({
        "type": "TOOL_CALL_START",
        "toolCallId": "tool-1",
        "toolCallName": "delegate",
    })

    assert replay.snapshot() == [
        {
            "type": "TOOL_CALL_START",
            "toolCallId": "tool-1",
            "toolCallName": "delegate",
        }
    ]


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
