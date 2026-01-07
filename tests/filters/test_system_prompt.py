"""Tests for pai_agent_sdk.filters.system_prompt module."""

from pathlib import Path
from unittest.mock import MagicMock

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)

from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.environment.local import LocalEnvironment
from pai_agent_sdk.filters.system_prompt import create_system_prompt_filter, fix_system_prompt


def test_fix_system_prompt_empty_history() -> None:
    """Should return empty history unchanged."""
    result = fix_system_prompt([], "New system prompt")
    assert result == []


def test_fix_system_prompt_no_existing_system_prompt() -> None:
    """Should inject system prompt when none exists."""
    request = ModelRequest(parts=[UserPromptPart(content="Hello")])
    history = [request]

    result = fix_system_prompt(history, "You are a helpful assistant.")

    assert len(result) == 1
    assert isinstance(result[0], ModelRequest)
    assert len(result[0].parts) == 2
    assert isinstance(result[0].parts[0], SystemPromptPart)
    assert result[0].parts[0].content == "You are a helpful assistant."
    assert isinstance(result[0].parts[1], UserPromptPart)
    assert result[0].parts[1].content == "Hello"


def test_fix_system_prompt_replaces_existing() -> None:
    """Should replace existing system prompt with new one."""
    request = ModelRequest(
        parts=[
            SystemPromptPart(content="Old system prompt"),
            UserPromptPart(content="Hello"),
        ]
    )
    history = [request]

    result = fix_system_prompt(history, "New system prompt")

    assert len(result) == 1
    assert isinstance(result[0], ModelRequest)
    assert len(result[0].parts) == 2
    assert isinstance(result[0].parts[0], SystemPromptPart)
    assert result[0].parts[0].content == "New system prompt"
    assert isinstance(result[0].parts[1], UserPromptPart)


def test_fix_system_prompt_removes_all_system_prompts() -> None:
    """Should remove all system prompts from all requests."""
    request1 = ModelRequest(
        parts=[
            SystemPromptPart(content="System prompt 1"),
            UserPromptPart(content="Hello"),
        ]
    )
    response = ModelResponse(parts=[TextPart(content="Hi there")])
    request2 = ModelRequest(
        parts=[
            SystemPromptPart(content="System prompt 2"),
            UserPromptPart(content="Follow up"),
        ]
    )
    history = [request1, response, request2]

    result = fix_system_prompt(history, "New system prompt")

    assert len(result) == 3

    # First request should have new system prompt
    assert isinstance(result[0], ModelRequest)
    assert len(result[0].parts) == 2
    assert isinstance(result[0].parts[0], SystemPromptPart)
    assert result[0].parts[0].content == "New system prompt"

    # Response should be unchanged
    assert isinstance(result[1], ModelResponse)

    # Second request should have system prompt removed
    assert isinstance(result[2], ModelRequest)
    assert len(result[2].parts) == 1
    assert isinstance(result[2].parts[0], UserPromptPart)


def test_fix_system_prompt_preserves_instructions() -> None:
    """Should preserve ModelRequest instructions field."""
    request = ModelRequest(
        parts=[SystemPromptPart(content="Old"), UserPromptPart(content="Hello")],
        instructions="Custom instructions",
    )
    history = [request]

    result = fix_system_prompt(history, "New system prompt")

    assert isinstance(result[0], ModelRequest)
    assert result[0].instructions == "Custom instructions"


def test_fix_system_prompt_history_starts_with_response() -> None:
    """Should not inject system prompt if first message is not ModelRequest."""
    response = ModelResponse(parts=[TextPart(content="Hi")])
    request = ModelRequest(parts=[UserPromptPart(content="Hello")])
    history = [response, request]

    result = fix_system_prompt(history, "New system prompt")

    # System prompt should not be injected since first message is not ModelRequest
    assert len(result) == 2
    assert isinstance(result[0], ModelResponse)
    assert isinstance(result[1], ModelRequest)
    # The request should not have a system prompt added
    assert len(result[1].parts) == 1
    assert isinstance(result[1].parts[0], UserPromptPart)


async def test_create_system_prompt_filter(tmp_path: Path) -> None:
    """Should create a working history processor."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(
            file_operator=env.file_operator,
            shell=env.shell,
        ) as ctx:
            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            processor = create_system_prompt_filter("You are a helpful assistant.")

            request = ModelRequest(
                parts=[
                    SystemPromptPart(content="Old prompt"),
                    UserPromptPart(content="Hello"),
                ]
            )
            history = [request]

            result = processor(mock_ctx, history)

            assert len(result) == 1
            assert isinstance(result[0], ModelRequest)
            assert len(result[0].parts) == 2
            assert isinstance(result[0].parts[0], SystemPromptPart)
            assert result[0].parts[0].content == "You are a helpful assistant."
