from __future__ import annotations

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ThinkingPart, ToolCallPart, UserPromptPart
from ya_agent_sdk.agents.main import (
    _extract_previous_assistant_response_reference,
    _truncate_previous_assistant_response_reference,
)


def test_extract_previous_assistant_response_reference_uses_latest_visible_text() -> None:
    history = [
        ModelRequest(parts=[UserPromptPart(content="What should we do?")]),
        ModelResponse(parts=[TextPart(content="1. Add tests\n2. Update docs")]),
        ModelRequest(parts=[UserPromptPart(content="1 and 2")]),
    ]

    assert _extract_previous_assistant_response_reference(history) == "1. Add tests\n2. Update docs"


def test_extract_previous_assistant_response_reference_skips_non_visible_parts() -> None:
    history = [
        ModelResponse(parts=[TextPart(content="visible answer")]),
        ModelResponse(
            parts=[
                ThinkingPart(content="private reasoning"),
                ToolCallPart(tool_name="shell", args={"command": "echo hi"}, tool_call_id="tool-1"),
            ]
        ),
    ]

    assert _extract_previous_assistant_response_reference(history) == "visible answer"


def test_truncate_previous_assistant_response_reference_bounds_long_text() -> None:
    text = "H" * 26000 + "T" * 10000

    truncated = _truncate_previous_assistant_response_reference(text)

    assert len(truncated) < len(text)
    assert "chars truncated from previous assistant response" in truncated
    assert truncated.startswith("H" * 100)
    assert truncated.endswith("T" * 100)
