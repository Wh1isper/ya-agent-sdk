"""Tests for subagent tool factory."""

import inspect

import pytest
from pydantic_ai import RunContext
from pydantic_ai.usage import RunUsage

from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.toolsets.core.base import Toolset
from pai_agent_sdk.toolsets.core.subagent import create_subagent_tool

# Test fixtures and mock functions


async def mock_search(
    ctx: AgentContext,  # Now receives subagent context directly
    query: str,
    max_results: int = 10,
) -> tuple[str, RunUsage]:
    """Mock search function that returns query info and usage."""
    usage = RunUsage(requests=1, input_tokens=10, output_tokens=20)
    return f"Results for '{query}' (max: {max_results})", usage


async def mock_analyze(
    ctx: AgentContext,  # Now receives subagent context directly
    content: str,
) -> tuple[dict, RunUsage]:
    """Mock analyze function that returns a dict."""
    usage = RunUsage(requests=1, input_tokens=5, output_tokens=15)
    return {"analysis": content, "score": 0.95}, usage


# Tests for create_subagent_tool function


def test_call_signature_is_correct():
    """Test that the call method has correct signature for pydantic-ai.

    The call method should have:
    - First parameter: ctx with type RunContext[AgentContext]
    - Remaining parameters: copied from call_func (without ctx)
    - Return type: str

    Note: 'self' is not in __signature__ because we assign a plain function
    to the class attribute, not a bound method. pydantic-ai Tool extracts
    parameters starting from 'ctx', which is correct.
    """
    SearchTool = create_subagent_tool(
        name="search",
        description="Search the web",
        call_func=mock_search,
    )

    # Get the signature from the class's call method
    sig = inspect.signature(SearchTool.call)
    params = list(sig.parameters.items())
    param_names = [name for name, _ in params]

    # The signature should contain: ctx, query, max_results
    # (self is handled separately by Python method binding, not in __signature__)
    assert param_names[0] == "ctx", f"First param should be ctx, got {param_names[0]}"
    assert "query" in param_names, "Should have query parameter from call_func"
    assert "max_results" in param_names, "Should have max_results parameter from call_func"

    # Check annotations - the key check is that ctx is RunContext[AgentContext], NOT AgentContext
    annotations = SearchTool.call.__annotations__
    assert annotations.get("ctx") == RunContext[AgentContext], (
        f"ctx should be RunContext[AgentContext], got {annotations.get('ctx')}"
    )
    assert annotations.get("query") is str, f"query should be str, got {annotations.get('query')}"
    assert annotations.get("max_results") is int, f"max_results should be int, got {annotations.get('max_results')}"
    assert annotations.get("return") is str, f"return should be str, got {annotations.get('return')}"

    # Check signature return annotation
    assert sig.return_annotation is str, f"Return annotation should be str, got {sig.return_annotation}"

    # Check parameter defaults
    params_dict = dict(params)
    if "max_results" in params_dict:
        max_results_param = params_dict["max_results"]
        assert max_results_param.default == 10, f"max_results default should be 10, got {max_results_param.default}"

    # Verify the signature parameter types match annotations
    ctx_param = params_dict["ctx"]
    assert ctx_param.annotation == RunContext[AgentContext], (
        f"ctx param annotation should be RunContext[AgentContext], got {ctx_param.annotation}"
    )


def test_creates_tool_class():
    """Test that create_subagent_tool returns a BaseTool subclass."""
    SearchTool = create_subagent_tool(
        name="search",
        description="Search the web",
        call_func=mock_search,
    )

    assert SearchTool.name == "search"
    assert SearchTool.description == "Search the web"
    assert SearchTool.__name__ == "SearchTool"


def test_pascal_case_naming():
    """Test that tool class names are converted to PascalCase."""
    Tool1 = create_subagent_tool(
        name="web_search",
        description="desc",
        call_func=mock_search,
    )
    assert Tool1.__name__ == "WebSearchTool"

    Tool2 = create_subagent_tool(
        name="analyze-content",
        description="desc",
        call_func=mock_search,
    )
    assert Tool2.__name__ == "AnalyzeContentTool"


@pytest.mark.asyncio
async def test_call_returns_string(agent_context: AgentContext):
    """Test that tool call returns string output."""
    AnalyzeTool = create_subagent_tool(
        name="analyze",
        description="Analyze content",
        call_func=mock_analyze,
    )

    tool = AnalyzeTool(agent_context)
    ctx = _create_mock_run_context(agent_context, tool_call_id="test-call-1")

    result = await tool.call(ctx, content="test content")

    # Dict should be converted to string
    assert isinstance(result, str)
    assert "analysis" in result
    assert "test content" in result


@pytest.mark.asyncio
async def test_usage_recorded(agent_context: AgentContext):
    """Test that usage is recorded in extra_usages."""
    SearchTool = create_subagent_tool(
        name="search",
        description="Search",
        call_func=mock_search,
    )

    tool = SearchTool(agent_context)
    tool_call_id = "test-call-123"
    ctx = _create_mock_run_context(agent_context, tool_call_id=tool_call_id)

    await tool.call(ctx, query="test query")

    # Check usage was recorded
    assert len(agent_context.extra_usages) == 1
    record = agent_context.extra_usages[0]
    assert record.uuid == tool_call_id
    assert record.agent == "search"
    assert record.usage.requests == 1
    assert record.usage.input_tokens == 10
    assert record.usage.output_tokens == 20


@pytest.mark.asyncio
async def test_usage_not_recorded_without_tool_call_id(agent_context: AgentContext):
    """Test that usage is not recorded when tool_call_id is None."""
    SearchTool = create_subagent_tool(
        name="search",
        description="Search",
        call_func=mock_search,
    )

    tool = SearchTool(agent_context)
    ctx = _create_mock_run_context(agent_context, tool_call_id=None)

    await tool.call(ctx, query="test query")

    # No usage should be recorded
    assert len(agent_context.extra_usages) == 0


def test_instruction_string(agent_context: AgentContext):
    """Test static instruction string."""
    SearchTool = create_subagent_tool(
        name="search",
        description="Search",
        call_func=mock_search,
        instruction="Use this to search the web.",
    )

    tool = SearchTool(agent_context)
    ctx = _create_mock_run_context(agent_context)

    assert tool.get_instruction(ctx) == "Use this to search the web."


def test_instruction_callable(agent_context: AgentContext):
    """Test dynamic instruction callable."""

    def dynamic_instruction(ctx: RunContext[AgentContext]) -> str:
        return f"Search with run_id: {ctx.deps.run_id}"

    SearchTool = create_subagent_tool(
        name="search",
        description="Search",
        call_func=mock_search,
        instruction=dynamic_instruction,
    )

    tool = SearchTool(agent_context)
    ctx = _create_mock_run_context(agent_context)

    instruction = tool.get_instruction(ctx)
    assert agent_context.run_id in instruction


def test_instruction_none(agent_context: AgentContext):
    """Test that no instruction returns None."""
    SearchTool = create_subagent_tool(
        name="search",
        description="Search",
        call_func=mock_search,
    )

    tool = SearchTool(agent_context)
    ctx = _create_mock_run_context(agent_context)

    assert tool.get_instruction(ctx) is None


@pytest.mark.asyncio
async def test_with_toolset(agent_context: AgentContext):
    """Test that created tool works with Toolset."""
    SearchTool = create_subagent_tool(
        name="search",
        description="Search the web",
        call_func=mock_search,
    )

    toolset = Toolset(agent_context, tools=[SearchTool])
    ctx = _create_mock_run_context(agent_context)

    tools = await toolset.get_tools(ctx)

    assert "search" in tools
    assert tools["search"].tool_def.name == "search"
    assert tools["search"].tool_def.description == "Search the web"


# Tests for subagent_stream_queues functionality


def test_stream_queues_default_dict(agent_context: AgentContext):
    """Test that stream_queues is a defaultdict creating queues on access."""
    # Access non-existent key should create a new queue
    queue = agent_context.subagent_stream_queues["test-tool-call-id"]
    assert queue is not None
    assert queue.empty()


@pytest.mark.asyncio
async def test_stream_queues_put_get(agent_context: AgentContext):
    """Test putting and getting events from stream queue."""
    tool_call_id = "test-tool-call-id"
    queue = agent_context.subagent_stream_queues[tool_call_id]

    # Put a custom event
    custom_event = {"event_kind": "custom", "data": "test"}
    await queue.put(custom_event)

    # Get the event
    event = await queue.get()
    assert event == custom_event


@pytest.mark.asyncio
async def test_stream_queues_multiple_tools(agent_context: AgentContext):
    """Test that different tool calls have separate queues."""
    queue1 = agent_context.subagent_stream_queues["tool-1"]
    queue2 = agent_context.subagent_stream_queues["tool-2"]

    await queue1.put("event1")
    await queue2.put("event2")

    assert await queue1.get() == "event1"
    assert await queue2.get() == "event2"


# Helper functions


def _create_mock_run_context(
    agent_context: AgentContext,
    tool_call_id: str | None = "mock-tool-call-id",
) -> RunContext[AgentContext]:
    """Create a mock RunContext for testing."""
    return RunContext[AgentContext](
        deps=agent_context,
        model=None,  # type: ignore[arg-type]
        usage=RunUsage(),
        prompt="test",
        messages=[],
        run_step=0,
        tool_call_id=tool_call_id,
    )
