"""Tests for ya_agent_sdk.filters._builders module."""

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from ya_agent_sdk.filters._builders import (
    KEEP_COMPACT,
    KEEP_HANDOFF,
    KEEP_TAG_KEY,
    build_context_restored_part,
    build_original_request_parts,
    build_steering_parts,
    has_keep_tag,
)

# =============================================================================
# build_original_request_parts tests
# =============================================================================


def test_build_original_request_parts_with_prompt() -> None:
    """Should return closed original-request XML when prompt is provided."""
    parts = build_original_request_parts("Build a CLI tool")
    assert len(parts) == 1
    assert parts[0].content.startswith("<original-request>")
    assert parts[0].content.endswith("</original-request>")
    assert "Build a CLI tool" in parts[0].content


def test_build_original_request_parts_none() -> None:
    """Should return empty list when prompt is None."""
    parts = build_original_request_parts(None)
    assert parts == []


# =============================================================================
# build_steering_parts tests
# =============================================================================


def test_build_steering_parts_with_messages() -> None:
    """Should return closed user-steering XML with prefixed messages."""
    parts = build_steering_parts(["Use click", "Add tests"])
    assert len(parts) == 1
    assert parts[0].content.startswith("<user-steering>")
    assert parts[0].content.endswith("</user-steering>")
    assert "[User Steering] Use click" in parts[0].content
    assert "[User Steering] Add tests" in parts[0].content


def test_build_steering_parts_none() -> None:
    """Should return empty list when messages is None."""
    assert build_steering_parts(None) == []


def test_build_steering_parts_empty() -> None:
    """Should return empty list when messages is empty."""
    assert build_steering_parts([]) == []


# =============================================================================
# build_context_restored_part tests
# =============================================================================


def test_build_context_restored_part() -> None:
    """Should return a UserPromptPart with context-restored XML."""
    part = build_context_restored_part()
    assert isinstance(part, UserPromptPart)
    assert "<context-restored>" in part.content
    assert "</context-restored>" in part.content
    assert "most authoritative source" in part.content


# =============================================================================
# has_keep_tag tests
# =============================================================================


def test_has_keep_tag_compact() -> None:
    """Should detect keep:compact tag on ModelResponse."""
    msg = ModelResponse(parts=[TextPart(content="summary")], metadata={KEEP_TAG_KEY: KEEP_COMPACT})
    assert has_keep_tag(msg) is True


def test_has_keep_tag_handoff() -> None:
    """Should detect keep:handoff tag on ModelRequest."""
    msg = ModelRequest(parts=[UserPromptPart(content="test")], metadata={KEEP_TAG_KEY: KEEP_HANDOFF})
    assert has_keep_tag(msg) is True


def test_has_keep_tag_no_metadata() -> None:
    """Should return False when metadata is None."""
    msg = ModelResponse(parts=[TextPart(content="text")])
    assert has_keep_tag(msg) is False


def test_has_keep_tag_empty_metadata() -> None:
    """Should return False when metadata has no keep key."""
    msg = ModelRequest(parts=[UserPromptPart(content="test")], metadata={"other": "value"})
    assert has_keep_tag(msg) is False
