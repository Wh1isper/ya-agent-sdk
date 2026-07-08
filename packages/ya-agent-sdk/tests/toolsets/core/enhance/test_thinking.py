"""Tests for ya_agent_sdk.toolsets.core.enhance.thinking module."""

from unittest.mock import MagicMock

from inline_snapshot import snapshot
from pydantic_ai import RunContext
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.toolsets.core.enhance.thinking import ThinkingTool


async def test_thinking_tool_attributes(agent_context: AgentContext) -> None:
    """Should have correct name, description and instruction."""
    assert ThinkingTool.name == "thinking"
    assert ThinkingTool.description == snapshot(
        "Think about something without obtaining new information or making changes."
    )
    # Test get_instruction with a mock context
    tool = ThinkingTool()
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = agent_context
    assert await tool.get_instruction(mock_run_ctx) == snapshot(
        """\
<thinking-guidelines>
Use for complex multi-step reasoning or durable intermediate observations.
Do not use for simple operations or as a substitute for visible task tracking.
Write thoughts in the user's language.
</thinking-guidelines>
"""
    )


def test_thinking_tool_initialization(agent_context: AgentContext) -> None:
    """Should initialize with context."""
    tool = ThinkingTool()
    assert tool.name == "thinking"


def test_thinking_tool_is_available(agent_context: AgentContext, mock_run_ctx) -> None:
    """Should be available by default."""
    tool = ThinkingTool()
    assert tool.is_available(mock_run_ctx) is True


async def test_thinking_tool_call_returns_thought(agent_context: AgentContext) -> None:
    """Should return the thought in a dictionary."""
    tool = ThinkingTool()

    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = agent_context

    result = await tool.call(mock_run_ctx, thought="This is a test thought")
    assert result == {"thought": "This is a test thought"}


async def test_thinking_tool_call_with_markdown(agent_context: AgentContext) -> None:
    """Should handle markdown formatted thoughts."""
    tool = ThinkingTool()

    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = agent_context

    markdown_thought = """
## Analysis
- Point 1
- Point 2
```python
code_example()
```
"""
    result = await tool.call(mock_run_ctx, thought=markdown_thought)
    assert result == {"thought": markdown_thought}


async def test_thinking_tool_call_with_empty_thought(agent_context: AgentContext) -> None:
    """Should handle empty thought string."""
    tool = ThinkingTool()

    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = agent_context

    result = await tool.call(mock_run_ctx, thought="")
    assert result == {"thought": ""}


async def test_thinking_tool_call_with_unicode(agent_context: AgentContext) -> None:
    """Should handle unicode characters."""
    tool = ThinkingTool()

    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = agent_context

    unicode_thought = "Thinking about: Hello World"
    result = await tool.call(mock_run_ctx, thought=unicode_thought)
    assert result == {"thought": unicode_thought}
