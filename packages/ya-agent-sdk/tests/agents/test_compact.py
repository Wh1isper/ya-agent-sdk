from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic_ai import ThinkingPart, ToolOutput
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.messages import (
    BinaryContent,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai.usage import RunUsage
from ya_agent_sdk.agents import compact as compact_module
from ya_agent_sdk.agents import main as main_module
from ya_agent_sdk.agents.compact import (
    _COMPACT_STRIP_KEYS,
    _DEFAULT_INJECTED_TAGS,
    _INCOMPATIBLE_BETAS,
    _MAX_TOOL_RETURN_CHARS,
    CondenseResult,
    _find_last_user_turn_index,
    _is_media_content,
    _media_to_placeholder,
    _strip_incompatible_settings,
    _strip_injected_context,
    _trim_history_for_compact,
    _truncate_str,
    create_cache_friendly_compact_filter,
    create_compact_filter,
    get_compact_agent,
)
from ya_agent_sdk.agents.main import create_agent
from ya_agent_sdk.context import AgentContext, ModelConfig
from ya_agent_sdk.context.agent import PROJECT_GUIDANCE_TAG, USER_RULES_TAG
from ya_agent_sdk.environment.local import LocalEnvironment
from ya_agent_sdk.filters.auto_load_files import process_auto_load_files
from ya_agent_sdk.filters.runtime_instructions import inject_runtime_instructions


def _history_processors_from_agent(runtime) -> list:
    return [
        capability.processor
        for capability in runtime.agent.root_capability.capabilities
        if isinstance(capability, ProcessHistory)
    ]


# Full tag set including application-level tags for testing
_ALL_TAGS = (*_DEFAULT_INJECTED_TAGS, PROJECT_GUIDANCE_TAG, USER_RULES_TAG)

# =============================================================================
# _truncate_str tests
# =============================================================================


def test_truncate_str_short_content() -> None:
    """Short content should not be truncated."""
    content = "short content"
    assert _truncate_str(content) == content


def test_truncate_str_at_boundary() -> None:
    """Content exactly at max_chars should not be truncated."""
    content = "x" * _MAX_TOOL_RETURN_CHARS
    assert _truncate_str(content) == content


def test_truncate_str_long_content() -> None:
    """Long content should be truncated with head + marker + tail."""
    content = "H" * 300 + "M" * 400 + "T" * 300
    result = _truncate_str(content)

    assert len(result) < len(content)
    assert result.startswith("H" * 200)
    assert result.endswith("T" * 200)
    assert "[... 600 chars truncated ...]" in result


# =============================================================================
# _is_media_content tests
# =============================================================================


def test_is_media_content_image_url() -> None:
    assert _is_media_content(ImageUrl(url="https://example.com/image.png")) is True


def test_is_media_content_video_url() -> None:
    assert _is_media_content(VideoUrl(url="https://example.com/video.mp4")) is True


def test_is_media_content_binary_image() -> None:
    assert _is_media_content(BinaryContent(data=b"fake", media_type="image/png")) is True


def test_is_media_content_binary_video() -> None:
    assert _is_media_content(BinaryContent(data=b"fake", media_type="video/mp4")) is True


def test_is_media_content_string() -> None:
    assert _is_media_content("hello") is False


def test_is_media_content_binary_text() -> None:
    assert _is_media_content(BinaryContent(data=b"fake", media_type="application/pdf")) is False


# =============================================================================
# _media_to_placeholder tests
# =============================================================================


def test_media_to_placeholder_image_url() -> None:
    assert _media_to_placeholder(ImageUrl(url="https://example.com/img.png")) == "[image: https://example.com/img.png]"


def test_media_to_placeholder_video_url() -> None:
    assert _media_to_placeholder(VideoUrl(url="https://example.com/v.mp4")) == "[video: https://example.com/v.mp4]"


def test_media_to_placeholder_binary() -> None:
    assert (
        _media_to_placeholder(BinaryContent(data=b"x", media_type="image/png")) == "[image/png binary content removed]"
    )


# =============================================================================
# _trim_history_for_compact tests
# =============================================================================


def test_trim_truncates_large_tool_returns() -> None:
    """Large ToolReturnPart content should be truncated."""
    large_content = "a" * 1000
    history: list[ModelMessage] = [
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name="view", content=large_content, tool_call_id="call_1"),
            ]
        ),
    ]

    trimmed = _trim_history_for_compact(history)

    assert len(trimmed) == 1
    request = trimmed[0]
    assert isinstance(request, ModelRequest)
    part = request.parts[0]
    assert isinstance(part, ToolReturnPart)
    assert isinstance(part.content, str)
    assert len(part.content) < len(large_content)
    assert "[... " in part.content


def test_trim_preserves_small_tool_returns() -> None:
    """Small ToolReturnPart content should not be modified."""
    small_content = "OK"
    history: list[ModelMessage] = [
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name="shell", content=small_content, tool_call_id="call_1"),
            ]
        ),
    ]

    trimmed = _trim_history_for_compact(history)

    part = trimmed[0]
    assert isinstance(part, ModelRequest)
    assert part.parts[0].content == small_content  # type: ignore[union-attr]


def test_trim_replaces_image_content_in_user_prompt() -> None:
    """Image content in UserPromptPart should be replaced with placeholder."""
    history: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        "Describe this image",
                        ImageUrl(url="https://example.com/image.png"),
                    ]
                ),
            ]
        ),
    ]

    trimmed = _trim_history_for_compact(history)

    request = trimmed[0]
    assert isinstance(request, ModelRequest)
    part = request.parts[0]
    assert isinstance(part, UserPromptPart)
    assert isinstance(part.content, list)
    assert len(part.content) == 2
    assert part.content[0] == "Describe this image"
    assert part.content[1] == "[image: https://example.com/image.png]"


def test_trim_replaces_video_content_in_user_prompt() -> None:
    """Video content in UserPromptPart should be replaced with placeholder."""
    history: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        "Describe this video",
                        VideoUrl(url="https://example.com/video.mp4"),
                    ]
                ),
            ]
        ),
    ]

    trimmed = _trim_history_for_compact(history)

    request = trimmed[0]
    assert isinstance(request, ModelRequest)
    part = request.parts[0]
    assert isinstance(part, UserPromptPart)
    assert isinstance(part.content, list)
    assert len(part.content) == 2
    assert part.content[0] == "Describe this video"
    assert part.content[1] == "[video: https://example.com/video.mp4]"


def test_trim_replaces_binary_image_in_user_prompt() -> None:
    """BinaryContent images in UserPromptPart should be replaced with placeholder."""
    history: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        "Analyze this",
                        BinaryContent(data=b"fake-image-data", media_type="image/png"),
                    ]
                ),
            ]
        ),
    ]

    trimmed = _trim_history_for_compact(history)

    request = trimmed[0]
    assert isinstance(request, ModelRequest)
    part = request.parts[0]
    assert isinstance(part, UserPromptPart)
    assert isinstance(part.content, list)
    assert len(part.content) == 2
    assert part.content[0] == "Analyze this"
    assert part.content[1] == "[image/png binary content removed]"


def test_trim_replaces_all_media_in_list_with_placeholders() -> None:
    """When all content items are media, each should be replaced with placeholder."""
    history: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        ImageUrl(url="https://example.com/image.png"),
                    ]
                ),
            ]
        ),
    ]

    trimmed = _trim_history_for_compact(history)

    request = trimmed[0]
    assert isinstance(request, ModelRequest)
    part = request.parts[0]
    assert isinstance(part, UserPromptPart)
    assert isinstance(part.content, list)
    assert part.content[0] == "[image: https://example.com/image.png]"


def test_trim_handles_singleton_media_content() -> None:
    """Direct media content (not in a list) should be replaced with placeholder."""
    # This is technically a type violation (content should be str | Sequence[UserContent])
    # but we handle it defensively
    history: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(content=ImageUrl(url="https://example.com/image.png")),  # type: ignore[arg-type]
            ]
        ),
    ]

    trimmed = _trim_history_for_compact(history)

    request = trimmed[0]
    assert isinstance(request, ModelRequest)
    part = request.parts[0]
    assert isinstance(part, UserPromptPart)
    assert part.content == "[image: https://example.com/image.png]"


def test_trim_preserves_string_user_prompts() -> None:
    """String UserPromptPart content should not be modified."""
    history: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(content="Hello, how are you?"),
            ]
        ),
    ]

    trimmed = _trim_history_for_compact(history)

    request = trimmed[0]
    assert isinstance(request, ModelRequest)
    part = request.parts[0]
    assert isinstance(part, UserPromptPart)
    assert part.content == "Hello, how are you?"


def test_trim_preserves_model_responses() -> None:
    """ModelResponse messages should be kept as-is."""
    history: list[ModelMessage] = [
        ModelResponse(parts=[TextPart(content="I can help with that.")]),
    ]

    trimmed = _trim_history_for_compact(history)

    assert len(trimmed) == 1
    response = trimmed[0]
    assert isinstance(response, ModelResponse)
    assert response.parts[0].content == "I can help with that."  # type: ignore[union-attr]


def test_trim_handles_mixed_messages() -> None:
    """Should handle a realistic mixed message history."""
    history: list[ModelMessage] = [
        # User sends text
        ModelRequest(
            parts=[
                UserPromptPart(content="Read this file"),
            ]
        ),
        # Assistant responds with tool call
        ModelResponse(parts=[TextPart(content="Reading file...")]),
        # Tool return with large content
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name="view", content="x" * 2000, tool_call_id="call_1"),
            ]
        ),
        # User sends image + text
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        "What about this?",
                        ImageUrl(url="https://example.com/screenshot.png"),
                    ]
                ),
            ]
        ),
    ]

    trimmed = _trim_history_for_compact(history)

    assert len(trimmed) == 4

    # First message: preserved
    req0 = trimmed[0]
    assert isinstance(req0, ModelRequest)
    assert isinstance(req0.parts[0], UserPromptPart)
    assert req0.parts[0].content == "Read this file"

    # Second message: ModelResponse preserved
    assert isinstance(trimmed[1], ModelResponse)

    # Third message: tool return truncated
    req2 = trimmed[2]
    assert isinstance(req2, ModelRequest)
    part2 = req2.parts[0]
    assert isinstance(part2, ToolReturnPart)
    assert isinstance(part2.content, str)
    assert len(part2.content) < 2000
    assert "[... " in part2.content

    # Fourth message: image replaced with placeholder, text kept
    req3 = trimmed[3]
    assert isinstance(req3, ModelRequest)
    part3 = req3.parts[0]
    assert isinstance(part3, UserPromptPart)
    assert isinstance(part3.content, list)
    assert len(part3.content) == 2
    assert part3.content[0] == "What about this?"
    assert part3.content[1] == "[image: https://example.com/screenshot.png]"


# =============================================================================
# _strip_injected_context tests
# =============================================================================


def test_strip_injected_context_pure_runtime_context() -> None:
    """UserPromptPart with only runtime-context should be removed entirely."""
    part = UserPromptPart(
        content="<runtime-context>\n  <agent-id>main</agent-id>\n  <current-time>2026-04-02T19:02:51+08:00</current-time>\n</runtime-context>"
    )
    assert _strip_injected_context(part) is None


def test_strip_injected_context_pure_environment_context() -> None:
    """UserPromptPart with only environment-context should be removed entirely."""
    part = UserPromptPart(
        content="<environment-context>\n<file-system>\n  <default-directory>/home/user</default-directory>\n</file-system>\n</environment-context>"
    )
    assert _strip_injected_context(part) is None


def test_strip_injected_context_mixed_string_content() -> None:
    """Runtime-context should be stripped but other content preserved."""
    part = UserPromptPart(content="Hello world\n<runtime-context><agent-id>main</agent-id></runtime-context>")
    result = _strip_injected_context(part)
    assert result is not None
    assert result.content == "Hello world"


def test_strip_injected_context_plain_text() -> None:
    """Plain text content should be unchanged."""
    part = UserPromptPart(content="Just a normal message")
    result = _strip_injected_context(part)
    assert result is not None
    assert result.content == "Just a normal message"
    # Should return the same object (no copy needed)
    assert result is part


def test_strip_injected_context_list_with_instructions() -> None:
    """Instruction items should be filtered from list content."""
    part = UserPromptPart(
        content=[
            "user request text",
            "<project-guidance name=AGENTS.md>\n## Project Overview\n...long content...</project-guidance>",
            "<user-rules location=/home/.yaacli/RULES.md>\n## Preferences\n...</user-rules>",
        ]
    )
    result = _strip_injected_context(part, tags=_ALL_TAGS)
    assert result is not None
    assert isinstance(result.content, list)
    assert len(result.content) == 1
    assert result.content[0] == "user request text"


def test_strip_injected_context_list_all_instructions() -> None:
    """List with only instruction items should be removed entirely."""
    part = UserPromptPart(
        content=[
            "<project-guidance name=AGENTS.md>\ncontent</project-guidance>",
            "<user-rules location=/home/RULES.md>\nrules</user-rules>",
        ]
    )
    assert _strip_injected_context(part, tags=_ALL_TAGS) is None


def test_strip_injected_context_list_no_instructions() -> None:
    """List without instruction items should be unchanged."""
    part = UserPromptPart(
        content=[
            "user text",
            ImageUrl(url="https://example.com/img.png"),
        ]
    )
    result = _strip_injected_context(part)
    assert result is not None
    assert result is part  # Same object, no modification


def test_strip_injected_context_both_contexts_in_string() -> None:
    """Both runtime and environment context should be stripped from the same string."""
    part = UserPromptPart(
        content=(
            "<runtime-context><data>1</data></runtime-context>\n"
            "important message\n"
            "<environment-context><data>2</data></environment-context>"
        )
    )
    result = _strip_injected_context(part)
    assert result is not None
    assert result.content == "important message"


# =============================================================================
# _trim_history_for_compact: thinking part preservation tests
# =============================================================================


def test_trim_preserves_thinking_parts_from_response() -> None:
    """ThinkingPart should be preserved for provider reasoning round-trips."""
    history: list[ModelMessage] = [
        ModelResponse(
            parts=[
                ThinkingPart(content="Let me think about this..."),
                TextPart(content="Here is my answer."),
            ]
        ),
    ]

    trimmed = _trim_history_for_compact(history)

    assert len(trimmed) == 1
    response = trimmed[0]
    assert isinstance(response, ModelResponse)
    assert len(response.parts) == 2
    assert isinstance(response.parts[0], ThinkingPart)
    assert response.parts[0].content == "Let me think about this..."
    assert isinstance(response.parts[1], TextPart)
    assert response.parts[1].content == "Here is my answer."


def test_trim_preserves_thinking_and_tool_calls() -> None:
    """ThinkingPart and ToolCallPart should both be preserved."""
    history: list[ModelMessage] = [
        ModelResponse(
            parts=[
                ThinkingPart(content="I need to read a file..."),
                ToolCallPart(tool_name="view", args={"file_path": "test.py"}, tool_call_id="call_1"),
            ]
        ),
    ]

    trimmed = _trim_history_for_compact(history)

    response = trimmed[0]
    assert isinstance(response, ModelResponse)
    assert len(response.parts) == 2
    assert isinstance(response.parts[0], ThinkingPart)
    assert isinstance(response.parts[1], ToolCallPart)


def test_trim_preserves_response_without_thinking() -> None:
    """ModelResponse without ThinkingPart should not be modified."""
    original_response = ModelResponse(parts=[TextPart(content="Hello")])
    history: list[ModelMessage] = [original_response]

    trimmed = _trim_history_for_compact(history)

    assert trimmed[0] is original_response  # Same object, no unnecessary copy


# =============================================================================
# _trim_history_for_compact: injected context stripping tests
# =============================================================================


def test_trim_removes_runtime_context_user_prompt() -> None:
    """UserPromptPart with only runtime-context should be removed."""
    history: list[ModelMessage] = [
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name="shell", content="OK", tool_call_id="call_1"),
                UserPromptPart(
                    content="<runtime-context><agent-id>main</agent-id><current-time>2026-04-02</current-time></runtime-context>"
                ),
            ]
        ),
    ]

    trimmed = _trim_history_for_compact(history)

    request = trimmed[0]
    assert isinstance(request, ModelRequest)
    assert len(request.parts) == 1
    assert isinstance(request.parts[0], ToolReturnPart)


def test_trim_removes_environment_context_user_prompt() -> None:
    """UserPromptPart with only environment-context should be removed."""
    history: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(content="Hello"),
                UserPromptPart(
                    content="<environment-context><file-system><path>/home</path></file-system></environment-context>"
                ),
            ]
        ),
    ]

    trimmed = _trim_history_for_compact(history)

    request = trimmed[0]
    assert isinstance(request, ModelRequest)
    assert len(request.parts) == 1
    assert isinstance(request.parts[0], UserPromptPart)
    assert request.parts[0].content == "Hello"  # type: ignore[union-attr]


def test_trim_strips_instructions_from_list_user_prompt() -> None:
    """Instruction items should be filtered from list-type user prompt content."""
    history: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        "Fix the bug in main.py",
                        "<project-guidance name=AGENTS.md>\n## Overview\n...</project-guidance>",
                        "<user-rules location=/home/RULES.md>\n## Rules\n...</user-rules>",
                    ]
                ),
            ]
        ),
    ]

    trimmed = _trim_history_for_compact(history, injected_context_tags=_ALL_TAGS)

    request = trimmed[0]
    assert isinstance(request, ModelRequest)
    part = request.parts[0]
    assert isinstance(part, UserPromptPart)
    assert isinstance(part.content, list)
    assert len(part.content) == 1
    assert part.content[0] == "Fix the bug in main.py"


def test_trim_combined_all_stripping() -> None:
    """Should handle a realistic history with thinking, context, and instructions."""
    history: list[ModelMessage] = [
        # User message with instructions
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        "Help me refactor",
                        "<project-guidance name=AGENTS.md>\nProject info</project-guidance>",
                        "<user-rules location=RULES.md>\nPrefs</user-rules>",
                    ]
                ),
                UserPromptPart(content="<runtime-context><agent-id>main</agent-id></runtime-context>"),
                UserPromptPart(content="<environment-context><file-system>data</file-system></environment-context>"),
            ]
        ),
        # Response with thinking
        ModelResponse(
            parts=[
                ThinkingPart(content="Let me analyze the code..."),
                TextPart(content="I'll help you refactor."),
                ToolCallPart(tool_name="view", args={"file_path": "main.py"}, tool_call_id="call_1"),
            ]
        ),
        # Tool return with runtime context
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name="view", content="x" * 1000, tool_call_id="call_1"),
                UserPromptPart(content="<runtime-context><agent-id>main</agent-id></runtime-context>"),
            ]
        ),
    ]

    trimmed = _trim_history_for_compact(history, injected_context_tags=_ALL_TAGS)

    assert len(trimmed) == 3

    # First request: instructions and contexts stripped, user text preserved
    req0 = trimmed[0]
    assert isinstance(req0, ModelRequest)
    assert len(req0.parts) == 1  # Only the user text survives
    part0 = req0.parts[0]
    assert isinstance(part0, UserPromptPart)
    assert isinstance(part0.content, list)
    assert len(part0.content) == 1
    assert part0.content[0] == "Help me refactor"

    # Response: thinking, text, and tool call preserved
    resp = trimmed[1]
    assert isinstance(resp, ModelResponse)
    assert len(resp.parts) == 3
    assert isinstance(resp.parts[0], ThinkingPart)
    assert isinstance(resp.parts[1], TextPart)
    assert isinstance(resp.parts[2], ToolCallPart)

    # Tool return request: tool return truncated, runtime context removed
    req2 = trimmed[2]
    assert isinstance(req2, ModelRequest)
    assert len(req2.parts) == 1  # Only truncated tool return
    part2 = req2.parts[0]
    assert isinstance(part2, ToolReturnPart)
    assert len(part2.content) < 1000  # type: ignore[arg-type]


# =============================================================================
# _find_last_user_turn_index tests
# =============================================================================


def test_find_last_user_turn_index_empty_history() -> None:
    """Empty history should return None."""
    assert _find_last_user_turn_index([]) is None


def test_find_last_user_turn_index_only_tool_returns() -> None:
    """History with only tool return requests should return None."""
    history: list[ModelMessage] = [
        ModelRequest(parts=[ToolReturnPart(tool_name="shell", content="OK", tool_call_id="call_1")]),
    ]
    assert _find_last_user_turn_index(history) is None


def test_find_last_user_turn_index_single_user_turn() -> None:
    """Single user prompt request should return index 0."""
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="Hello")]),
    ]
    assert _find_last_user_turn_index(history) == 0


def test_find_last_user_turn_index_multiple_turns() -> None:
    """Should return the index of the last user turn."""
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="First question")]),
        ModelResponse(parts=[TextPart(content="Answer")]),
        ModelRequest(parts=[ToolReturnPart(tool_name="view", content="data", tool_call_id="c1")]),
        ModelResponse(parts=[TextPart(content="More")]),
        ModelRequest(parts=[UserPromptPart(content="Second question")]),
    ]
    assert _find_last_user_turn_index(history) == 4


def test_trim_preserves_last_turn_injected_context() -> None:
    """With preserve_last_turn=True, the last user turn should keep injected context."""
    history: list[ModelMessage] = [
        # Earlier turn
        ModelRequest(parts=[UserPromptPart(content="First question")]),
        ModelResponse(parts=[TextPart(content="Answer")]),
        # Last user turn with injected context
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        "My question",
                        "<project-guidance name=AGENTS.md>\nProject info</project-guidance>",
                    ]
                ),
                UserPromptPart(content="<runtime-context><agent-id>main</agent-id></runtime-context>"),
                UserPromptPart(content="<environment-context><file-system>data</file-system></environment-context>"),
            ]
        ),
    ]

    trimmed = _trim_history_for_compact(history, preserve_last_turn=True, injected_context_tags=_ALL_TAGS)

    # First turn: context stripped
    req0 = trimmed[0]
    assert isinstance(req0, ModelRequest)
    assert len(req0.parts) == 1

    # Last turn: all 3 parts preserved
    request = trimmed[2]
    assert isinstance(request, ModelRequest)
    assert len(request.parts) == 3
    part0 = request.parts[0]
    assert isinstance(part0, UserPromptPart)
    assert isinstance(part0.content, list)
    assert len(part0.content) == 2  # User text + project-guidance


def test_trim_strips_last_turn_by_default() -> None:
    """By default (preserve_last_turn=False), the last turn is also stripped."""
    history: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        "My question",
                        "<project-guidance name=AGENTS.md>\nProject info</project-guidance>",
                    ]
                ),
                UserPromptPart(content="<runtime-context><agent-id>main</agent-id></runtime-context>"),
            ]
        ),
    ]

    trimmed = _trim_history_for_compact(history, injected_context_tags=_ALL_TAGS)

    request = trimmed[0]
    assert isinstance(request, ModelRequest)
    # Context stripped: only user text remains, runtime-context part removed
    assert len(request.parts) == 1
    part0 = request.parts[0]
    assert isinstance(part0, UserPromptPart)
    assert isinstance(part0.content, list)
    assert len(part0.content) == 1
    assert part0.content[0] == "My question"


# =============================================================================
# Integration tests
# =============================================================================


async def test_create_agent_runs_auto_load_files_after_compact(tmp_path: Path) -> None:
    env = LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    )

    async with create_agent(
        model="test",
        env=env,
    ) as runtime:
        processors = _history_processors_from_agent(runtime)
        auto_load_indexes = [i for i, processor in enumerate(processors) if processor is process_auto_load_files]

        assert len(auto_load_indexes) == 2
        assert auto_load_indexes[-1] > auto_load_indexes[0]


async def test_create_agent_runs_runtime_instructions_after_compact(tmp_path: Path) -> None:
    env = LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    )

    async with create_agent(
        model="test",
        env=env,
    ) as runtime:
        processors = _history_processors_from_agent(runtime)
        runtime_indexes = [i for i, processor in enumerate(processors) if processor is inject_runtime_instructions]
        compact_indexes = [
            i for i, processor in enumerate(processors) if getattr(processor, "__name__", "") == "compact_filter"
        ]
        auto_load_indexes = [i for i, processor in enumerate(processors) if processor is process_auto_load_files]

        assert len(runtime_indexes) == 1
        assert compact_indexes
        assert auto_load_indexes
        assert runtime_indexes[0] > compact_indexes[0]
        assert runtime_indexes[0] > auto_load_indexes[-1]


def test_create_agent_uses_cache_friendly_compact_filter_by_default(tmp_path: Path, monkeypatch) -> None:
    env = LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    )
    cache_friendly_filter = MagicMock(__name__="cache_friendly_compact_filter")
    legacy_filter = MagicMock(__name__="legacy_compact_filter")
    create_cache_friendly = MagicMock(return_value=cache_friendly_filter)
    create_legacy = MagicMock(return_value=legacy_filter)
    monkeypatch.setattr(main_module, "create_cache_friendly_compact_filter", create_cache_friendly)
    monkeypatch.setattr(main_module, "create_compact_filter", create_legacy)

    runtime = create_agent(model="test", env=env)

    create_cache_friendly.assert_called_once()
    create_legacy.assert_not_called()
    assert cache_friendly_filter in _history_processors_from_agent(runtime)


def test_create_agent_can_use_legacy_compact_filter(tmp_path: Path, monkeypatch) -> None:
    env = LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    )
    cache_friendly_filter = MagicMock(__name__="cache_friendly_compact_filter")
    legacy_filter = MagicMock(__name__="legacy_compact_filter")
    create_cache_friendly = MagicMock(return_value=cache_friendly_filter)
    create_legacy = MagicMock(return_value=legacy_filter)
    compact_cfg = ModelConfig(context_window=123, compact_threshold=0.5)
    monkeypatch.setattr(main_module, "create_cache_friendly_compact_filter", create_cache_friendly)
    monkeypatch.setattr(main_module, "create_compact_filter", create_legacy)

    runtime = create_agent(
        model="test",
        env=env,
        compact_model="compact-model",
        compact_model_settings={"temperature": 0},
        compact_model_cfg=compact_cfg,
        use_cache_friendly_compact_filter=False,
    )

    create_cache_friendly.assert_not_called()
    create_legacy.assert_called_once_with(
        model="compact-model",
        model_settings={"temperature": 0},
        model_cfg=compact_cfg,
        main_model="test",
        main_model_settings=None,
    )
    assert legacy_filter in _history_processors_from_agent(runtime)


async def test_agent_context_resets_compact_depth_for_new_run_and_subagent(agent_context: AgentContext) -> None:
    agent_context._compact_depth = 2

    new_run_ctx = agent_context.prepare_new_run()
    subagent_ctx = agent_context.create_subagent_context("worker", agent_id="worker-1")

    assert new_run_ctx._compact_depth == 0
    assert subagent_ctx._compact_depth == 0


# =============================================================================
# _strip_incompatible_settings tests
# =============================================================================


def test_strip_incompatible_settings_no_incompatible_keys() -> None:
    """Settings without incompatible keys should be preserved."""
    settings = {"max_tokens": 4096, "temperature": 0.5}
    result = _strip_incompatible_settings(settings)
    assert result == {"max_tokens": 4096, "temperature": 0.5}


def test_strip_incompatible_settings_removes_cache_keys() -> None:
    """Anthropic cache keys should be stripped, other keys preserved."""
    settings = {
        "max_tokens": 4096,
        "anthropic_cache_instructions": True,
        "anthropic_cache_messages": True,
        "anthropic_cache": "1h",
        "anthropic_cache_tool_definitions": "5m",
    }
    result = _strip_incompatible_settings(settings)
    assert result == {"max_tokens": 4096}
    # Original should not be modified
    assert "anthropic_cache_instructions" in settings


def test_strip_incompatible_settings_removes_thinking_keys() -> None:
    """Thinking-related keys should be stripped for compact structured output."""
    settings = {
        "max_tokens": 4096,
        "thinking": "high",
        "anthropic_thinking": {"type": "enabled", "budget_tokens": 10000},
        "anthropic_effort": "high",
    }
    result = _strip_incompatible_settings(settings)
    assert result == {"max_tokens": 4096}


def test_strip_incompatible_settings_strips_interleaved_beta() -> None:
    """Interleaved thinking beta header should be stripped from extra_headers."""
    settings = {
        "max_tokens": 4096,
        "extra_headers": {
            "anthropic-beta": "context-1m-2025-08-07,interleaved-thinking-2025-05-14",
        },
    }
    result = _strip_incompatible_settings(settings)
    assert result == {
        "max_tokens": 4096,
        "extra_headers": {"anthropic-beta": "context-1m-2025-08-07"},
    }


def test_strip_incompatible_settings_removes_empty_beta_header() -> None:
    """If only incompatible betas exist, the entire header should be removed."""
    settings = {
        "max_tokens": 4096,
        "extra_headers": {
            "anthropic-beta": "interleaved-thinking-2025-05-14",
        },
    }
    result = _strip_incompatible_settings(settings)
    assert result == {"max_tokens": 4096}


def test_strip_incompatible_settings_preserves_non_beta_headers() -> None:
    """Non-beta extra headers should be preserved even when beta is removed."""
    settings = {
        "max_tokens": 4096,
        "extra_headers": {
            "anthropic-beta": "interleaved-thinking-2025-05-14",
            "x-custom": "value",
        },
    }
    result = _strip_incompatible_settings(settings)
    assert result == {
        "max_tokens": 4096,
        "extra_headers": {"x-custom": "value"},
    }


def test_strip_incompatible_settings_full_preset() -> None:
    """Should handle a full Anthropic preset with all settings."""
    settings = {
        "max_tokens": 32768,
        "thinking": "high",
        "anthropic_thinking": {"type": "adaptive"},
        "anthropic_effort": "high",
        "anthropic_cache_instructions": True,
        "anthropic_cache_messages": True,
        "anthropic_cache": True,
        "extra_headers": {
            "anthropic-beta": "context-1m-2025-08-07,interleaved-thinking-2025-05-14,context-management-2025-06-27",
        },
        "extra_body": {
            "context_management": {
                "edits": [
                    {"type": "clear_thinking_20251015", "keep": "all"},
                    {"type": "clear_tool_uses_20250919", "trigger": {"type": "input_tokens", "value": 100000}},
                ]
            }
        },
    }
    result = _strip_incompatible_settings(settings)
    assert result == {
        "max_tokens": 32768,
        "extra_headers": {"anthropic-beta": "context-1m-2025-08-07,context-management-2025-06-27"},
        "extra_body": {
            "context_management": {
                "edits": [
                    {"type": "clear_tool_uses_20250919", "trigger": {"type": "input_tokens", "value": 100000}},
                ]
            }
        },
    }


def test_strip_incompatible_settings_removes_clear_thinking_from_context_management() -> None:
    """clear_thinking edit should be removed from context_management."""
    settings = {
        "max_tokens": 4096,
        "extra_body": {
            "context_management": {
                "edits": [
                    {"type": "clear_thinking_20251015", "keep": {"type": "thinking_turns", "value": 3}},
                ]
            }
        },
    }
    result = _strip_incompatible_settings(settings)
    # Only clear_thinking edit -> context_management and extra_body removed entirely
    assert result == {"max_tokens": 4096}
    assert "extra_body" not in result


def test_strip_incompatible_settings_preserves_tool_use_clearing() -> None:
    """clear_tool_uses edit should be preserved when clear_thinking is stripped."""
    settings = {
        "max_tokens": 4096,
        "extra_body": {
            "context_management": {
                "edits": [
                    {"type": "clear_thinking_20251015", "keep": "all"},
                    {"type": "clear_tool_uses_20250919", "trigger": {"type": "input_tokens", "value": 50000}},
                ]
            }
        },
    }
    result = _strip_incompatible_settings(settings)
    assert result == {
        "max_tokens": 4096,
        "extra_body": {
            "context_management": {
                "edits": [
                    {"type": "clear_tool_uses_20250919", "trigger": {"type": "input_tokens", "value": 50000}},
                ]
            }
        },
    }


def test_strip_incompatible_settings_known_keys() -> None:
    """All expected keys should be in _COMPACT_STRIP_KEYS."""
    assert "anthropic_cache_tool_definitions" in _COMPACT_STRIP_KEYS
    assert "anthropic_cache_instructions" in _COMPACT_STRIP_KEYS
    assert "anthropic_cache_messages" in _COMPACT_STRIP_KEYS
    assert "anthropic_cache" in _COMPACT_STRIP_KEYS
    assert "thinking" in _COMPACT_STRIP_KEYS
    assert "anthropic_thinking" in _COMPACT_STRIP_KEYS
    assert "anthropic_effort" in _COMPACT_STRIP_KEYS


def test_strip_incompatible_settings_known_betas() -> None:
    """All expected betas should be in _INCOMPATIBLE_BETAS."""
    assert "interleaved-thinking-2025-05-14" in _INCOMPATIBLE_BETAS


def test_get_compact_agent_uses_tool_output_by_default() -> None:
    """Compact should prefer tool output for models that can force output tools."""
    agent = get_compact_agent(model="test")

    assert isinstance(agent.output_type, ToolOutput)
    assert agent.output_type.output is CondenseResult


# =============================================================================
# Keep tag: _build_compacted_messages tagging
# =============================================================================


def test_build_compacted_messages_tags_response_with_keep() -> None:
    """Compacted ModelResponse should have keep:compact metadata."""
    from ya_agent_sdk.agents.compact import _build_compacted_messages
    from ya_agent_sdk.filters._builders import KEEP_COMPACT, KEEP_TAG_KEY

    messages = _build_compacted_messages("Summary text", "Original prompt")
    # Message 0: initial request (no keep tag)
    assert messages[0].metadata is None or KEEP_TAG_KEY not in (messages[0].metadata or {})
    # Message 1: ModelResponse with summary
    assert isinstance(messages[1], ModelResponse)
    assert messages[1].metadata is not None
    assert messages[1].metadata[KEEP_TAG_KEY] == KEEP_COMPACT
    # Message 2: final request with original-request + context-restored
    assert isinstance(messages[2], ModelRequest)
    assert messages[2].metadata is not None
    assert messages[2].metadata[KEEP_TAG_KEY] == KEEP_COMPACT


def test_build_compacted_messages_uses_shared_builders() -> None:
    """Compacted messages should contain original-request, steering, and context-restored parts."""
    from ya_agent_sdk.agents.compact import _build_compacted_messages

    messages = _build_compacted_messages(
        "Summary text",
        "Build a CLI",
        steering_messages=["Use click"],
    )
    final_request = messages[2]
    assert isinstance(final_request, ModelRequest)
    user_parts = [p for p in final_request.parts if isinstance(p, UserPromptPart)]
    # Should have: original-request label, original prompt, steering label, steering msg, context-restored
    assert any("original-request" in p.content for p in user_parts)
    assert any(p.content == "Build a CLI" for p in user_parts)
    assert any("user-steering" in p.content for p in user_parts)
    assert any("[User Steering] Use click" in p.content for p in user_parts)
    assert any("context-restored" in p.content for p in user_parts)


# =============================================================================
# Keep tag: _trim_history_for_compact preserves tagged messages
# =============================================================================


def test_trim_history_preserves_keep_tagged_response() -> None:
    """Messages tagged with keep metadata should pass through trimming unchanged."""
    from ya_agent_sdk.agents.compact import _build_compacted_messages, _trim_history_for_compact

    # Simulate a history that includes a prior compact summary
    prior_compact = _build_compacted_messages("Prior session summary with lots of detail", "Original prompt")

    # Add normal conversation messages after the compact summary
    normal_messages: list[ModelMessage] = [
        ModelRequest(
            parts=[UserPromptPart(content="<runtime-context>big injected context</runtime-context>\nDo something")]
        ),
        ModelResponse(parts=[TextPart(content="OK, doing it")]),
        ModelRequest(parts=[UserPromptPart(content="Thanks")]),
    ]

    history = prior_compact + normal_messages
    trimmed = _trim_history_for_compact(history)

    # Prior compact messages (tagged) should be preserved exactly
    assert trimmed[1] is prior_compact[1]  # ModelResponse with summary - same object
    assert trimmed[2] is prior_compact[2]  # ModelRequest with context-restored - same object

    # Normal messages should still be processed (injected context stripped)
    trimmed_normal_request = trimmed[3]
    assert isinstance(trimmed_normal_request, ModelRequest)
    user_parts = [p for p in trimmed_normal_request.parts if isinstance(p, UserPromptPart)]
    # runtime-context should be stripped from normal messages
    assert all("runtime-context" not in p.content for p in user_parts)


def test_trim_history_preserves_keep_tagged_handoff() -> None:
    """Handoff-tagged messages should also be preserved during compact trimming."""
    from ya_agent_sdk.agents.compact import _trim_history_for_compact
    from ya_agent_sdk.filters._builders import KEEP_TAG_KEY
    from ya_agent_sdk.filters.handoff import _build_handoff_messages

    # Build handoff messages (tagged with keep:handoff)
    handoff_msgs = _build_handoff_messages(
        "Handoff summary with critical context " * 50,  # Long summary
        original_prompt="Build something",
        steering_messages=["Focus on tests"],
    )

    # Add more conversation after handoff
    extra: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="Continue work")]),
        ModelResponse(parts=[TextPart(content="Working on it")]),
    ]

    history = handoff_msgs + extra
    trimmed = _trim_history_for_compact(history)

    # Handoff-tagged messages should be preserved exactly (not truncated)
    for i, msg in enumerate(handoff_msgs):
        if msg.metadata and KEEP_TAG_KEY in msg.metadata:
            assert trimmed[i] is msg, f"Message {i} should be preserved by reference"


def test_trim_history_truncates_untagged_tool_return() -> None:
    """Untagged ToolReturnPart should still be truncated as before."""
    long_content = "x" * 1000

    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="Hello")]),
        ModelResponse(parts=[ToolCallPart(tool_name="test", args={}, tool_call_id="t1")]),
        ModelRequest(parts=[ToolReturnPart(tool_name="test", content=long_content, tool_call_id="t1")]),
    ]

    trimmed = _trim_history_for_compact(history)

    tool_returns = [
        p for msg in trimmed if isinstance(msg, ModelRequest) for p in msg.parts if isinstance(p, ToolReturnPart)
    ]
    assert len(tool_returns) == 1
    # Should be truncated (original was 1000 chars, max is 500)
    assert len(tool_returns[0].model_response_str()) < len(long_content)


class _FakeCompactResult:
    def __init__(self, output: CondenseResult, usage: RunUsage) -> None:
        self.output = output
        self._usage = usage

    def usage(self) -> RunUsage:
        return self._usage


class _FakeCompactStreamResult:
    def __init__(self, output: str, usage: RunUsage) -> None:
        self.output = output
        self._usage = usage

    async def get_output(self) -> str:
        return self.output

    def usage(self) -> RunUsage:
        return self._usage


class _FakeCompactStreamRun:
    def __init__(self, result: _FakeCompactStreamResult) -> None:
        self.result = result

    async def __aenter__(self) -> _FakeCompactStreamResult:
        return self.result

    async def __aexit__(self, exc_type, exc, tb) -> bool | None:
        return None


class _FakeCompactRun:
    def __init__(self, result: _FakeCompactResult) -> None:
        self.result = result
        self._yielded = False

    async def __aenter__(self) -> "_FakeCompactRun":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool | None:
        return None

    def __aiter__(self) -> "_FakeCompactRun":
        return self

    async def __anext__(self):
        if self._yielded:
            raise StopAsyncIteration
        self._yielded = True
        return object()


@pytest.mark.asyncio
async def test_cache_friendly_compact_filter_uses_current_agent_and_records_usage(
    agent_context: AgentContext, monkeypatch
):
    message_history = [
        ModelRequest(parts=[UserPromptPart(content="hello")]),
        ModelResponse(parts=[TextPart(content="world")]),
    ]
    agent_context.model_cfg = ModelConfig(context_window=10, compact_threshold=0.1)
    agent_context.user_prompts = "hello"
    object.__setattr__(agent_context, "_stream_queue_enabled", True)

    mock_run_ctx = MagicMock()
    mock_run_ctx.deps = agent_context
    mock_run_ctx.model = MagicMock(model_name="main-model")
    mock_run_ctx.agent = MagicMock()
    mock_run_ctx.agent._output_validators = [object()]

    stream_result = _FakeCompactStreamResult(
        output="## Condensed conversation summary\n\n### Analysis\n\nanalysis\n\n### Context\n\ncontext",
        usage=RunUsage(input_tokens=3, output_tokens=5, requests=1),
    )

    mock_run_ctx.agent.run.side_effect = AssertionError("parent agent should not run")
    mock_run_ctx.agent.run_stream.side_effect = AssertionError("parent agent should not run")
    compact_agent = MagicMock()
    compact_agent._output_validators = []
    compact_agent.run_stream.return_value = _FakeCompactStreamRun(stream_result)
    monkeypatch.setattr(compact_module.copy, "copy", lambda _agent: compact_agent)

    monkeypatch.setattr(compact_module, "_need_compact", lambda *_args, **_kwargs: True)

    compact_filter = create_cache_friendly_compact_filter(model_cfg=agent_context.model_cfg)
    compacted = await compact_filter(mock_run_ctx, message_history)

    mock_run_ctx.agent.run.assert_not_called()
    mock_run_ctx.agent.run_stream.assert_not_called()
    compact_agent.run_stream.assert_called_once()
    assert len(mock_run_ctx.agent._output_validators) == 1
    assert compact_agent._output_validators == []
    call_args = compact_agent.run_stream.call_args
    assert call_args.args[0] == (
        f"{compact_module.CACHE_FRIENDLY_COMPACT_INSTRUCTION}\n\n{compact_module.CACHE_FRIENDLY_COMPACT_PROMPT}"
    )
    assert call_args.kwargs["message_history"] == message_history
    assert call_args.kwargs["deps"] is agent_context
    assert call_args.kwargs["output_type"] is str
    assert "instructions" not in call_args.kwargs

    assert len(compacted) == 3
    assert agent_context.force_inject_instructions is True
    assert agent_context._compact_depth == 0
    assert len(agent_context.extra_usages) == 1
    assert agent_context.extra_usages[0].agent == "compact"
    assert agent_context.extra_usages[0].model_id == "main-model"

    queue = agent_context.agent_stream_queues[agent_context.agent_id]
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    assert [event.__class__.__name__ for event in events] == ["CompactStartEvent", "CompactCompleteEvent"]


@pytest.mark.asyncio
async def test_cache_friendly_compact_filter_skips_nested_compact(agent_context: AgentContext, monkeypatch):
    message_history = [ModelRequest(parts=[UserPromptPart(content="hello")])]
    agent_context._compact_depth = 1

    mock_run_ctx = MagicMock()
    mock_run_ctx.deps = agent_context
    mock_run_ctx.agent = MagicMock()

    monkeypatch.setattr(compact_module, "_need_compact", lambda *_args, **_kwargs: True)

    compact_filter = create_cache_friendly_compact_filter(model_cfg=agent_context.model_cfg)
    result = await compact_filter(mock_run_ctx, message_history)

    assert result is message_history
    mock_run_ctx.agent.run.assert_not_called()


@pytest.mark.asyncio
async def test_cache_friendly_compact_filter_uses_model_cfg_for_threshold(agent_context: AgentContext, monkeypatch):
    message_history = [ModelRequest(parts=[UserPromptPart(content="hello")])]
    agent_context.model_cfg = ModelConfig(context_window=None)
    override_cfg = ModelConfig(context_window=10, compact_threshold=0.1)
    seen_cfgs: list[ModelConfig] = []

    mock_run_ctx = MagicMock()
    mock_run_ctx.deps = agent_context
    mock_run_ctx.model = MagicMock(model_name="main-model")
    mock_run_ctx.agent = MagicMock()

    stream_result = _FakeCompactStreamResult(
        output="summary",
        usage=RunUsage(input_tokens=1, output_tokens=1, requests=1),
    )

    compact_agent = MagicMock()
    compact_agent.run_stream.return_value = _FakeCompactStreamRun(stream_result)
    monkeypatch.setattr(compact_module.copy, "copy", lambda _agent: compact_agent)

    def _fake_need_compact(ctx: AgentContext, _messages: list[ModelMessage]) -> bool:
        seen_cfgs.append(ctx.model_cfg)
        return True

    monkeypatch.setattr(compact_module, "_need_compact", _fake_need_compact)

    compact_filter = create_cache_friendly_compact_filter(model_cfg=override_cfg)
    await compact_filter(mock_run_ctx, message_history)

    assert seen_cfgs == [override_cfg]
    assert agent_context.model_cfg.context_window is None


@pytest.mark.asyncio
async def test_cache_friendly_compact_filter_returns_original_history_when_run_fails(
    agent_context: AgentContext, monkeypatch
):
    message_history = [ModelRequest(parts=[UserPromptPart(content="hello")])]
    agent_context.model_cfg = ModelConfig(context_window=10, compact_threshold=0.1)
    object.__setattr__(agent_context, "_stream_queue_enabled", True)

    mock_run_ctx = MagicMock()
    mock_run_ctx.deps = agent_context
    mock_run_ctx.agent = MagicMock()
    compact_agent = MagicMock()
    compact_agent.run_stream.side_effect = RuntimeError("run failed")
    monkeypatch.setattr(compact_module.copy, "copy", lambda _agent: compact_agent)

    monkeypatch.setattr(compact_module, "_need_compact", lambda *_args, **_kwargs: True)

    compact_filter = create_cache_friendly_compact_filter(model_cfg=agent_context.model_cfg)
    result = await compact_filter(mock_run_ctx, message_history)

    assert result == message_history
    assert agent_context._compact_depth == 0
    queue = agent_context.agent_stream_queues[agent_context.agent_id]
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    assert [event.__class__.__name__ for event in events] == ["CompactStartEvent", "CompactFailedEvent"]


@pytest.mark.asyncio
async def test_compact_filter_uses_agent_iter_and_records_usage(agent_context: AgentContext, monkeypatch):
    message_history = [
        ModelRequest(parts=[UserPromptPart(content="hello")]),
        ModelResponse(parts=[TextPart(content="world")]),
    ]
    agent_context.model_cfg = ModelConfig(context_window=10, compact_threshold=0.1)
    agent_context.user_prompts = "hello"
    object.__setattr__(agent_context, "_stream_queue_enabled", True)

    mock_run_ctx = MagicMock()
    mock_run_ctx.deps = agent_context

    condense_result = CondenseResult(
        analysis="analysis",
        context="context",
        original_prompt="hello",
        auto_load_files=["packages/ya-agent-sdk/README.md"],
    )
    run_result = _FakeCompactResult(
        output=condense_result,
        usage=RunUsage(input_tokens=3, output_tokens=5, requests=1),
    )

    fake_agent = MagicMock()
    fake_agent.model = MagicMock(model_name="test-model")
    fake_agent.iter.return_value = _FakeCompactRun(run_result)

    monkeypatch.setattr(compact_module, "get_compact_agent", lambda **kwargs: fake_agent)
    monkeypatch.setattr(compact_module, "_need_compact", lambda *_args, **_kwargs: True)

    compact_filter = create_compact_filter(model_cfg=agent_context.model_cfg)
    compacted = await compact_filter(mock_run_ctx, message_history)

    fake_agent.iter.assert_called_once()
    call_args = fake_agent.iter.call_args
    assert call_args.args[0] == compact_module.DEFAULT_COMPACT_INSTRUCTION
    assert call_args.kwargs["message_history"] == message_history
    compact_deps = call_args.kwargs["deps"]
    assert isinstance(compact_deps, AgentContext)
    assert compact_deps.env is agent_context.env
    assert compact_deps.model_cfg == agent_context.model_cfg

    assert len(compacted) == 3
    assert agent_context.auto_load_files == ["packages/ya-agent-sdk/README.md"]
    assert agent_context.force_inject_instructions is True
    assert len(agent_context.extra_usages) == 1
    assert agent_context.extra_usages[0].agent == "compact"
    assert agent_context.extra_usages[0].model_id == "test-model"

    queue = agent_context.agent_stream_queues[agent_context.agent_id]
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    assert [event.__class__.__name__ for event in events] == ["CompactStartEvent", "CompactCompleteEvent"]


@pytest.mark.asyncio
async def test_compact_filter_returns_original_history_when_iter_fails(agent_context: AgentContext, monkeypatch):
    message_history = [
        ModelRequest(parts=[UserPromptPart(content="hello")]),
        ModelResponse(parts=[TextPart(content="world")]),
    ]
    agent_context.model_cfg = ModelConfig(context_window=10, compact_threshold=0.1)
    object.__setattr__(agent_context, "_stream_queue_enabled", True)

    mock_run_ctx = MagicMock()
    mock_run_ctx.deps = agent_context

    fake_agent = MagicMock()
    fake_agent.model = MagicMock(model_name="test-model")

    @asynccontextmanager
    async def _failing_iter(*args, **kwargs):
        raise RuntimeError("iter failed")
        yield

    fake_agent.iter.side_effect = _failing_iter
    monkeypatch.setattr(compact_module, "get_compact_agent", lambda **kwargs: fake_agent)
    monkeypatch.setattr(compact_module, "_need_compact", lambda *_args, **_kwargs: True)

    compact_filter = create_compact_filter(model_cfg=agent_context.model_cfg)
    result = await compact_filter(mock_run_ctx, message_history)

    assert result == message_history
    queue = agent_context.agent_stream_queues[agent_context.agent_id]
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    assert [event.__class__.__name__ for event in events] == ["CompactStartEvent", "CompactFailedEvent"]
