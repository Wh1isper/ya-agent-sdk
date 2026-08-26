"""Tests for ya_agent_sdk.toolsets.base module."""

from unittest.mock import MagicMock

import pytest
from pydantic_ai import ModelRetry, RunContext, UserError
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.tools import ToolApproved, ToolDenied
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.toolsets.base import (
    BaseTool,
    BaseToolset,
    InstructableToolset,
    UserInputPreprocessResult,
)
from ya_agent_sdk.toolsets.core.base import Toolset, UserInteraction

from .._instruction_helpers import instruction_text as _instruction_text


# --- UserInteraction tests ---
def test_user_interaction_approved() -> None:
    """Should create approved interaction."""
    interaction = UserInteraction(
        tool_call_id="test-id",
        approved=True,
        user_input={"key": "value"},
    )
    assert interaction.tool_call_id == "test-id"
    assert interaction.approved is True
    assert interaction.reason is None


def test_user_interaction_rejected() -> None:
    """Should create rejected interaction with reason."""
    interaction = UserInteraction(
        tool_call_id="test-id",
        approved=False,
        reason="Not allowed",
    )
    assert interaction.approved is False
    assert interaction.reason == "Not allowed"


# --- UserInputPreprocessResult tests ---
def test_user_input_preprocess_result_with_override_args() -> None:
    """Should store override args."""
    result = UserInputPreprocessResult(
        override_args={"path": "/new/path"},
        metadata={"source": "user"},
    )
    assert result.override_args == {"path": "/new/path"}
    assert result.metadata == {"source": "user"}


def test_user_input_preprocess_result_empty() -> None:
    """Should handle empty result."""
    result = UserInputPreprocessResult()
    assert result.override_args is None
    assert result.metadata is None


# --- Test tool classes ---
class DummyTool(BaseTool):
    """A simple test tool."""

    name = "dummy_tool"
    description = "A dummy tool for testing"

    async def get_instruction(self, ctx: RunContext[AgentContext]) -> str | None:
        return "Use this dummy tool for testing purposes."

    async def call(self, ctx: RunContext[AgentContext], message: str = "hello") -> str:
        return f"Dummy: {message}"


class UnavailableTool(BaseTool):
    """A tool that is not available."""

    name = "unavailable_tool"
    description = "An unavailable tool"

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        return False

    async def call(self, ctx: RunContext[AgentContext]) -> str:
        return "Should not be called"


class DelegationTool(BaseTool):
    """A tool that delegates to another agent."""

    name = "delegation_tool"
    description = "A delegation tool"
    tags = frozenset({"delegation"})

    async def call(self, ctx: RunContext[AgentContext]) -> str:
        return "Delegated"


class RetryTool(BaseTool):
    """A tool that asks the model to correct its call."""

    name = "retry_tool"
    description = "A retrying tool"

    async def call(self, ctx: RunContext[AgentContext]) -> str:
        raise ModelRetry("correct the arguments")


class MainAgentOnlyTool(BaseTool):
    """A host-facing tool that subagents must never receive."""

    name = "main_agent_only_tool"
    description = "A main-agent-only tool"
    main_agent_only = True

    async def call(self, ctx: RunContext[AgentContext]) -> str:
        return "main only"


# --- BaseTool tests ---
def test_base_tool_default_availability(agent_context: AgentContext) -> None:
    """Should be available by default."""
    from unittest.mock import MagicMock

    from pydantic_ai import RunContext

    tool = DummyTool()
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = agent_context
    assert tool.is_available(mock_run_ctx) is True


def test_base_tool_unavailable(agent_context: AgentContext) -> None:
    """Should report unavailability correctly."""
    from unittest.mock import MagicMock

    from pydantic_ai import RunContext

    tool = UnavailableTool()
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = agent_context
    assert tool.is_available(mock_run_ctx) is False


def test_base_tool_initialization(agent_context: AgentContext) -> None:
    """Should initialize without context."""
    tool = DummyTool()
    assert tool.name == "dummy_tool"
    assert tool.description == "A dummy tool for testing"


async def test_base_tool_process_user_input_returns_none(agent_context: AgentContext) -> None:
    """Should return None by default."""
    tool = DummyTool()
    result = await tool.process_user_input(agent_context, {"input": "data"})
    assert result is None


# --- BaseToolset tests ---
@pytest.mark.asyncio
async def test_base_toolset_get_instructions_returns_none() -> None:
    """Should return None by default."""

    class SimpleToolset(BaseToolset):
        @property
        def id(self) -> str | None:
            return None

        async def get_tools(self, ctx: RunContext) -> dict:
            return {}

        async def call_tool(
            self, name: str, tool_args: dict[str, object], ctx: RunContext[AgentContext], tool: object
        ) -> object:
            pass

    toolset = SimpleToolset()
    mock_ctx = MagicMock(spec=RunContext)
    assert await toolset.get_instructions(mock_ctx) is None


# --- Toolset tests ---
def test_toolset_initialization(agent_context: AgentContext) -> None:
    """Should initialize with tools and a bounded retry default."""
    toolset = Toolset(tools=[DummyTool])
    assert len(toolset._tool_classes) == 1
    assert "dummy_tool" in toolset._tool_classes
    assert toolset.max_retries == 5


async def test_toolset_retry_config_and_local_override(agent_context: AgentContext) -> None:
    run_ctx = MagicMock(spec=RunContext)
    run_ctx.deps = agent_context
    agent_context.retry_config.toolset = 7

    inherited = Toolset(tools=[DummyTool])
    inherited_tools = await inherited.get_tools(run_ctx)
    assert inherited_tools["dummy_tool"].max_retries == 7

    overridden = Toolset(tools=[DummyTool], max_retries=2)
    overridden_tools = await overridden.get_tools(run_ctx)
    assert overridden_tools["dummy_tool"].max_retries == 2

    overridden.max_retries = None
    inherited_again = await overridden.get_tools(run_ctx)
    assert inherited_again["dummy_tool"].max_retries == 7


async def test_toolset_propagates_model_retry(agent_context: AgentContext) -> None:
    """BaseTool ModelRetry must reach Pydantic AI's retry accounting."""
    toolset = Toolset(tools=[RetryTool])
    run_ctx = MagicMock(spec=RunContext)
    run_ctx.deps = agent_context
    run_ctx.tool_call_approved = False
    tools = await toolset.get_tools(run_ctx)

    with pytest.raises(ModelRetry, match="correct the arguments"):
        await toolset.call_tool("retry_tool", {}, run_ctx, tools["retry_tool"])


async def test_toolset_skip_unavailable_tools(agent_context: AgentContext) -> None:
    """Should skip unavailable tools when skip_unavailable=True in get_tools()."""
    from unittest.mock import MagicMock

    from pydantic_ai import RunContext

    toolset = Toolset(tools=[DummyTool, UnavailableTool], skip_unavailable=True)
    # All tools are registered in _tool_classes
    assert "dummy_tool" in toolset._tool_classes
    assert "unavailable_tool" in toolset._tool_classes
    # But unavailable tools are filtered out in get_tools()
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = agent_context
    tools = await toolset.get_tools(mock_run_ctx)
    assert "dummy_tool" in tools
    assert "unavailable_tool" not in tools


def test_toolset_duplicate_tool_name_raises(agent_context: AgentContext) -> None:
    """Should raise on duplicate tool names."""
    from pydantic_ai import UserError

    with pytest.raises(UserError, match="Duplicate tool name"):
        Toolset(tools=[DummyTool, DummyTool])


def test_toolset_id(agent_context: AgentContext) -> None:
    """Should store and return toolset ID."""
    toolset = Toolset(tools=[DummyTool], toolset_id="my-toolset")
    assert toolset.id == "my-toolset"


async def test_toolset_get_instructions(agent_context: AgentContext) -> None:
    """Should collect instructions from tools."""
    toolset = Toolset(tools=[DummyTool])
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = agent_context
    instructions = await toolset.get_instructions(mock_run_ctx)
    instruction_text = _instruction_text(instructions)
    assert instructions is not None
    assert "Use this dummy tool for testing purposes." in instruction_text


async def test_toolset_get_tools(agent_context: AgentContext) -> None:
    """Should return tool definitions."""
    toolset = Toolset(tools=[DummyTool])
    mock_run_ctx = MagicMock(spec=RunContext)
    mock_run_ctx.deps = agent_context
    tools = await toolset.get_tools(mock_run_ctx)
    assert "dummy_tool" in tools
    assert tools["dummy_tool"].tool_def.name == "dummy_tool"


async def test_toolset_process_hitl_call_approved(agent_context: AgentContext) -> None:
    """Should process approved HITL interactions."""
    toolset = Toolset(tools=[DummyTool])
    interactions = [
        UserInteraction(tool_call_id="call-1", approved=True),
    ]
    result = await toolset.process_hitl_call(agent_context, interactions, [])
    assert result is not None
    assert "call-1" in result.approvals
    assert isinstance(result.approvals["call-1"], ToolApproved)


async def test_toolset_process_hitl_call_rejected(agent_context: AgentContext) -> None:
    """Should process rejected HITL interactions."""
    toolset = Toolset(tools=[DummyTool])
    interactions = [
        UserInteraction(tool_call_id="call-1", approved=False, reason="Not safe"),
    ]
    result = await toolset.process_hitl_call(agent_context, interactions, [])
    assert result is not None
    assert "call-1" in result.approvals
    denied = result.approvals["call-1"]
    assert isinstance(denied, ToolDenied)
    assert denied.message == "Not safe"


async def test_toolset_process_hitl_call_none(agent_context: AgentContext) -> None:
    """Should return None when no interactions."""
    toolset = Toolset(tools=[DummyTool])
    result = await toolset.process_hitl_call(agent_context, None, [])
    assert result is None


async def test_toolset_process_hitl_with_user_input(agent_context: AgentContext) -> None:
    """Should process user input for approved interactions."""
    toolset = Toolset(tools=[DummyTool])
    tool_call = ToolCallPart(
        tool_name="dummy_tool",
        tool_call_id="call-1",
        args={},
    )
    message_history = [ModelResponse(parts=[tool_call])]
    interactions = [
        UserInteraction(
            tool_call_id="call-1",
            approved=True,
            user_input={"custom": "data"},
        ),
    ]
    result = await toolset.process_hitl_call(agent_context, interactions, message_history)
    assert result is not None
    assert isinstance(result.approvals["call-1"], ToolApproved)


# --- InstructableToolset protocol tests ---
def test_instructable_toolset_protocol_check(agent_context: AgentContext) -> None:
    """Should recognize conforming toolsets."""
    toolset = Toolset(tools=[DummyTool])
    assert isinstance(toolset, InstructableToolset)


# --- Toolset inspection and child-boundary tests ---
class AnotherTool(BaseTool):
    """Another test tool."""

    name = "another_tool"
    description = "Another tool"

    async def call(self, ctx: RunContext[AgentContext]) -> str:
        return "another"


def test_toolset_tool_names(agent_context: AgentContext) -> None:
    toolset = Toolset(tools=[DummyTool, AnotherTool])
    assert set(toolset.tool_names) == {"dummy_tool", "another_tool"}


async def test_main_agent_only_policy_cannot_be_disabled_by_skip_unavailable(
    agent_context: AgentContext,
) -> None:
    toolset = Toolset(
        tools=[DummyTool, MainAgentOnlyTool],
        skip_unavailable=False,
    )
    child_ctx = agent_context.create_subagent_context("helper")
    run_ctx = MagicMock(spec=RunContext)
    run_ctx.deps = child_ctx

    tools = await toolset.get_tools(run_ctx)

    assert "dummy_tool" in tools
    assert "main_agent_only_tool" not in tools
    assert await toolset.get_instructions(run_ctx) is not None
    assert toolset.is_tool_available("main_agent_only_tool", run_ctx) is False


async def test_main_agent_only_policy_rejects_stale_cached_tool_call(
    agent_context: AgentContext,
) -> None:
    toolset = Toolset(
        tools=[MainAgentOnlyTool],
        skip_unavailable=False,
    )
    main_run_ctx = MagicMock(spec=RunContext)
    main_run_ctx.deps = agent_context
    stale_tool = (await toolset.get_tools(main_run_ctx))["main_agent_only_tool"]

    child_run_ctx = MagicMock(spec=RunContext)
    child_run_ctx.deps = agent_context.create_subagent_context("helper")

    with pytest.raises(UserError, match="not available in this agent context"):
        await toolset.call_tool(
            "main_agent_only_tool",
            {},
            child_run_ctx,
            stale_tool,
        )
