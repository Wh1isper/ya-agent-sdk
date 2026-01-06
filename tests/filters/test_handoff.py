"""Tests for pai_agent_sdk.filters.handoff module."""

from pathlib import Path
from unittest.mock import MagicMock

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.environment.local import LocalEnvironment
from pai_agent_sdk.filters.handoff import process_handoff_message


async def test_process_handoff_no_handoff_message(tmp_path: Path) -> None:
    """Should return unchanged history when no handoff message is set."""
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

            request = ModelRequest(parts=[UserPromptPart(content="Hello")])
            history = [request]

            result = process_handoff_message(mock_ctx, history)

            assert result == history
            assert len(request.parts) == 1


async def test_process_handoff_with_handoff_message(tmp_path: Path) -> None:
    """Should inject handoff summary and truncate history when handoff message is set."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(
            file_operator=env.file_operator,
            shell=env.shell,
        ) as ctx:
            ctx.handoff_message = "Previous context summary here"

            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            request = ModelRequest(parts=[UserPromptPart(content="Continue task")])
            history = [request]

            result = process_handoff_message(mock_ctx, history)

            # Should return truncated history with 3 messages
            assert len(result) == 3

            # First message should be the user request with handoff appended
            first_msg = result[0]
            assert isinstance(first_msg, ModelRequest)
            assert len(first_msg.parts) == 2
            assert isinstance(first_msg.parts[1], UserPromptPart)
            assert "<context-handoff>" in first_msg.parts[1].content
            assert "Previous context summary here" in first_msg.parts[1].content

            # Second message should be a ModelResponse with handoff tool call
            second_msg = result[1]
            assert isinstance(second_msg, ModelResponse)
            assert len(second_msg.parts) == 1
            assert isinstance(second_msg.parts[0], ToolCallPart)
            assert second_msg.parts[0].tool_name == "handoff"

            # Third message should be a ModelRequest with tool return
            third_msg = result[2]
            assert isinstance(third_msg, ModelRequest)
            assert len(third_msg.parts) == 1
            assert isinstance(third_msg.parts[0], ToolReturnPart)
            assert third_msg.parts[0].tool_name == "handoff"

            # Handoff message should be cleared
            assert ctx.handoff_message is None


async def test_process_handoff_empty_history(tmp_path: Path) -> None:
    """Should return unchanged empty history even with handoff message."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(
            file_operator=env.file_operator,
            shell=env.shell,
        ) as ctx:
            ctx.handoff_message = "Summary"

            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            result = process_handoff_message(mock_ctx, [])

            assert result == []
            # Handoff message should NOT be cleared since no injection happened
            assert ctx.handoff_message == "Summary"


async def test_process_handoff_finds_last_user_request(tmp_path: Path) -> None:
    """Should find the last true user request (not tool return) for injection."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(
            file_operator=env.file_operator,
            shell=env.shell,
        ) as ctx:
            ctx.handoff_message = "Context summary"

            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            # Create history with user request, response, and tool return
            user_request = ModelRequest(parts=[UserPromptPart(content="Do something")])
            response = ModelResponse(parts=[ToolCallPart(tool_call_id="tc1", tool_name="some_tool", args={})])
            tool_return = ModelRequest(
                parts=[ToolReturnPart(tool_call_id="tc1", tool_name="some_tool", content="result")]
            )
            final_user = ModelRequest(parts=[UserPromptPart(content="Next task")])

            history = [user_request, response, tool_return, final_user]

            result = process_handoff_message(mock_ctx, history)

            # Should inject into final_user (last true user request)
            assert len(result) == 3
            assert isinstance(result[0], ModelRequest)
            # The first part should be the original user prompt
            assert isinstance(result[0].parts[0], UserPromptPart)
            assert result[0].parts[0].content == "Next task"
            # The second part should be the handoff
            assert isinstance(result[0].parts[1], UserPromptPart)
            assert "<context-handoff>" in result[0].parts[1].content


async def test_process_handoff_skips_tool_return_only_request(tmp_path: Path) -> None:
    """Should skip ModelRequest that only contains ToolReturnPart."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(
            file_operator=env.file_operator,
            shell=env.shell,
        ) as ctx:
            ctx.handoff_message = "Summary"

            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            # History with only tool return request at the end
            user_request = ModelRequest(parts=[UserPromptPart(content="Start")])
            response = ModelResponse(parts=[TextPart(content="Response")])
            tool_return = ModelRequest(parts=[ToolReturnPart(tool_call_id="tc1", tool_name="tool", content="result")])

            history = [user_request, response, tool_return]

            result = process_handoff_message(mock_ctx, history)

            # Should inject into user_request (skipping tool_return)
            assert len(result) == 3
            first_msg = result[0]
            assert isinstance(first_msg, ModelRequest)
            assert isinstance(first_msg.parts[0], UserPromptPart)
            assert first_msg.parts[0].content == "Start"
            assert isinstance(first_msg.parts[1], UserPromptPart)
            assert "<context-handoff>" in first_msg.parts[1].content
