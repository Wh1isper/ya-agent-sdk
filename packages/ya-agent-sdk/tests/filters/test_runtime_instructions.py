"""Tests for ya_agent_sdk.filters.runtime_instructions module."""

from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolReturnPart, UserPromptPart
from ya_agent_sdk.context import AgentContext, ModelConfig
from ya_agent_sdk.environment.local import LocalEnvironment


@pytest.fixture
async def agent_context(tmp_path: Path) -> AgentContext:
    """Create an AgentContext for testing."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(
            env=env,
            model_cfg=ModelConfig(
                context_window=200000,
                proactive_context_management_threshold=0.5,
            ),
        ) as ctx:
            yield ctx


async def test_inject_runtime_instructions_empty_history(tmp_path: Path) -> None:
    """Should return unchanged history when no ModelRequest found."""
    # Create a mock RunContext
    from unittest.mock import MagicMock

    from ya_agent_sdk.filters.runtime_instructions import inject_runtime_instructions

    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(env=env) as ctx:
            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            result = await inject_runtime_instructions(mock_ctx, [])
            assert result == []


async def test_inject_runtime_instructions_inserts_before_user_prompt(tmp_path: Path) -> None:
    """Should insert runtime instructions before ordinary user prompt content."""
    from unittest.mock import MagicMock

    from ya_agent_sdk.filters.runtime_instructions import inject_runtime_instructions

    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(env=env) as ctx:
            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            # Create message history with a ModelRequest
            request = ModelRequest(parts=[UserPromptPart(content="Hello")])
            history = [request]

            result = await inject_runtime_instructions(mock_ctx, history)

            assert result == history
            # Should have added a part before the user prompt
            assert len(request.parts) == 2
            # The added part should be a UserPromptPart with runtime context
            added_part = request.parts[0]
            assert isinstance(added_part, UserPromptPart)
            assert "<runtime-context>" in added_part.content
            assert "<elapsed-time>" in added_part.content
            assert isinstance(request.parts[1], UserPromptPart)
            assert request.parts[1].content == "Hello"


async def test_inject_runtime_instructions_with_model_config(tmp_path: Path) -> None:
    """Should include model config in runtime instructions when set."""
    from unittest.mock import MagicMock

    from ya_agent_sdk.filters.runtime_instructions import inject_runtime_instructions

    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(
            env=env,
            model_cfg=ModelConfig(
                context_window=200000,
                proactive_context_management_threshold=0.5,
            ),
        ) as ctx:
            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            request = ModelRequest(parts=[UserPromptPart(content="Hello")])
            history = [request]

            await inject_runtime_instructions(mock_ctx, history)

            added_part = request.parts[0]
            assert isinstance(added_part, UserPromptPart)
            assert "<model-config>" in added_part.content
            assert "<context-window>200000</context-window>" in added_part.content


async def test_inject_runtime_instructions_finds_last_request(tmp_path: Path) -> None:
    """Should find and modify the last ModelRequest in history."""
    from unittest.mock import MagicMock

    from ya_agent_sdk.filters.runtime_instructions import inject_runtime_instructions

    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(env=env) as ctx:
            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            # Create history with multiple messages
            first_request = ModelRequest(parts=[UserPromptPart(content="First")])
            response = ModelResponse(parts=[TextPart(content="Response")])
            last_request = ModelRequest(parts=[UserPromptPart(content="Last")])
            history = [first_request, response, last_request]

            await inject_runtime_instructions(mock_ctx, history)

            # First request should be unchanged
            assert len(first_request.parts) == 1
            # Last request should have the runtime instructions
            assert len(last_request.parts) == 2
            assert isinstance(last_request.parts[0], UserPromptPart)
            assert "<runtime-context>" in last_request.parts[0].content
            assert isinstance(last_request.parts[1], UserPromptPart)
            assert last_request.parts[1].content == "Last"


async def test_inject_runtime_instructions_preserves_tool_response_order(tmp_path: Path) -> None:
    """Should insert runtime instructions after tool responses."""
    from unittest.mock import MagicMock

    from ya_agent_sdk.filters.runtime_instructions import inject_runtime_instructions

    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(env=env) as ctx:
            mock_ctx = MagicMock()
            mock_ctx.deps = ctx
            ctx.force_inject_instructions = True

            tool_return = ToolReturnPart(
                tool_name="test_tool",
                content="tool result",
                tool_call_id="call_123",
            )
            request = ModelRequest(parts=[tool_return, UserPromptPart(content="Continue")])
            history = [request]

            result = await inject_runtime_instructions(mock_ctx, history)

            assert result == history
            assert len(request.parts) == 3
            assert isinstance(request.parts[0], ToolReturnPart)
            assert isinstance(request.parts[1], UserPromptPart)
            assert "<runtime-context>" in request.parts[1].content
            assert isinstance(request.parts[2], UserPromptPart)
            assert request.parts[2].content == "Continue"


async def test_inject_runtime_instructions_only_response_in_history(tmp_path: Path) -> None:
    """Should return unchanged history when only ModelResponse present."""
    from unittest.mock import MagicMock

    from ya_agent_sdk.filters.runtime_instructions import inject_runtime_instructions

    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(env=env) as ctx:
            mock_ctx = MagicMock()
            mock_ctx.deps = ctx

            response = ModelResponse(parts=[TextPart(content="Response")])
            history = [response]

            result = await inject_runtime_instructions(mock_ctx, history)

            assert result == history
            assert len(response.parts) == 1
