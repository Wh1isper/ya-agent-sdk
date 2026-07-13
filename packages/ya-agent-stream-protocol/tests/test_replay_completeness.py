from __future__ import annotations

from ya_agent_stream_protocol.agui import AguiReplayBuffer


def test_shared_replay_preserves_complete_text_and_tool_payloads() -> None:
    replay = AguiReplayBuffer()
    text = "text-" * 100_000
    tool_result = "result-" * 100_000

    replay.append({"type": "TEXT_MESSAGE_CHUNK", "messageId": "message-1", "delta": text})
    replay.append({
        "type": "TOOL_CALL_RESULT",
        "toolCallId": "tool-1",
        "messageId": "tool-1:result",
        "content": tool_result,
    })

    snapshot = replay.snapshot()
    assert snapshot[0]["delta"] == text
    assert snapshot[1]["content"] == tool_result
