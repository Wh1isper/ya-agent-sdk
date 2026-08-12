"""Tests for subagent factory with availability checking."""

from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessage, ModelRequest, RetryPromptPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo as FunctionAgentInfo
from pydantic_ai.models.function import DeltaToolCall, FunctionModel
from ya_agent_sdk.agents.guards import MessageBusGuardCapability
from ya_agent_sdk.context import AgentContext, BusMessage, ModelConfig, StreamRecoveryPolicy
from ya_agent_sdk.environment.local import LocalEnvironment
from ya_agent_sdk.events import SubagentCompleteEvent
from ya_agent_sdk.filters.bus_message import inject_bus_messages
from ya_agent_sdk.subagents import SubagentConfig, create_subagent_tool_from_config
from ya_agent_sdk.toolsets.core.base import BaseTool, Toolset
from ya_agent_sdk.toolsets.core.interaction import AskUserQuestionTool
from ya_agent_sdk.toolsets.core.subagent.factory import create_subagent_call_func


class GrepTool(BaseTool):
    """Test grep tool."""

    name = "grep"
    description = "Search file contents"

    async def call(self, ctx, pattern: str) -> str:
        return f"grep: {pattern}"


class ViewTool(BaseTool):
    """Test view tool."""

    name = "view"
    description = "View file contents"

    async def call(self, ctx, path: str) -> str:
        return f"view: {path}"


class UnavailableTool(BaseTool):
    """Test tool that is never available."""

    name = "unavailable_tool"
    description = "This tool is never available"

    def is_available(self, ctx) -> bool:
        return False

    async def call(self, ctx) -> str:
        return "should not be called"


class DynamicTool(BaseTool):
    """Test tool with dynamic availability."""

    name = "dynamic_tool"
    description = "This tool has dynamic availability"
    _available = True

    def is_available(self, ctx) -> bool:
        return DynamicTool._available

    async def call(self, ctx) -> str:
        return "dynamic"


class TestSubagentToolAvailability:
    """Tests for subagent tool availability checking."""

    def test_subagent_available_when_all_tools_exist(self, agent_context, mock_run_ctx) -> None:
        """Subagent should be available when all required tools exist and are available."""
        parent_toolset = Toolset(tools=[GrepTool, ViewTool])

        config = SubagentConfig(
            name="test_subagent",
            description="Test subagent",
            system_prompt="You are a test agent",
            tools=["grep", "view"],
        )

        tool_cls = create_subagent_tool_from_config(config, parent_toolset, model="test")
        tool_instance = tool_cls()

        assert tool_instance.is_available(mock_run_ctx) is True

    def test_subagent_unavailable_when_tool_missing(self, agent_context, mock_run_ctx) -> None:
        """Subagent should be unavailable when a required tool is missing."""
        parent_toolset = Toolset(tools=[GrepTool])  # ViewTool missing

        config = SubagentConfig(
            name="test_subagent",
            description="Test subagent",
            system_prompt="You are a test agent",
            tools=["grep", "view"],  # requires view which is missing
        )

        tool_cls = create_subagent_tool_from_config(config, parent_toolset, model="test")
        tool_instance = tool_cls()

        assert tool_instance.is_available(mock_run_ctx) is False

    def test_subagent_unavailable_when_tool_not_available(self, agent_context, mock_run_ctx) -> None:
        """Subagent should be unavailable when a required tool exists but is_available=False."""
        # UnavailableTool will be skipped by Toolset due to skip_unavailable=True
        parent_toolset = Toolset(tools=[GrepTool, UnavailableTool])

        config = SubagentConfig(
            name="test_subagent",
            description="Test subagent",
            system_prompt="You are a test agent",
            tools=["grep", "unavailable_tool"],
        )

        tool_cls = create_subagent_tool_from_config(config, parent_toolset, model="test")
        tool_instance = tool_cls()

        # unavailable_tool is not in parent_toolset because it was skipped
        assert tool_instance.is_available(mock_run_ctx) is False

    def test_subagent_available_when_tools_none(self, agent_context, mock_run_ctx) -> None:
        """Subagent should be available when tools=None (inherit all)."""
        parent_toolset = Toolset(tools=[GrepTool, ViewTool])

        config = SubagentConfig(
            name="test_subagent",
            description="Test subagent",
            system_prompt="You are a test agent",
            tools=None,  # inherit all
        )

        tool_cls = create_subagent_tool_from_config(config, parent_toolset, model="test")
        tool_instance = tool_cls()

        assert tool_instance.is_available(mock_run_ctx) is True

    def test_subagent_dynamic_availability(self, agent_context, mock_run_ctx) -> None:
        """Subagent availability should be checked dynamically."""
        # Start with dynamic tool available
        DynamicTool._available = True
        parent_toolset = Toolset(tools=[GrepTool, DynamicTool])

        config = SubagentConfig(
            name="test_subagent",
            description="Test subagent",
            system_prompt="You are a test agent",
            tools=["grep", "dynamic_tool"],
        )

        tool_cls = create_subagent_tool_from_config(config, parent_toolset, model="test")
        tool_instance = tool_cls()

        # Initially available
        assert tool_instance.is_available(mock_run_ctx) is True

        # Make dynamic tool unavailable
        DynamicTool._available = False

        # Now subagent should be unavailable (dynamic check)
        assert tool_instance.is_available(mock_run_ctx) is False

        # Restore
        DynamicTool._available = True
        assert tool_instance.is_available(mock_run_ctx) is True


class TestToolsetIsToolAvailable:
    """Tests for Toolset.is_tool_available method."""

    def test_is_tool_available_for_existing_tool(self, agent_context, mock_run_ctx) -> None:
        """Should return True for existing and available tool."""
        toolset = Toolset(tools=[GrepTool, ViewTool])

        assert toolset.is_tool_available("grep", mock_run_ctx) is True
        assert toolset.is_tool_available("view", mock_run_ctx) is True

    def test_is_tool_available_for_missing_tool(self, agent_context, mock_run_ctx) -> None:
        """Should return False for non-existent tool."""
        toolset = Toolset(tools=[GrepTool])

        assert toolset.is_tool_available("view", mock_run_ctx) is False
        assert toolset.is_tool_available("nonexistent", mock_run_ctx) is False

    def test_is_tool_available_for_unavailable_tool(self, agent_context, mock_run_ctx) -> None:
        """Should return False for tool that was skipped due to is_available=False."""
        # UnavailableTool is registered but is_available returns False
        toolset = Toolset(tools=[GrepTool, UnavailableTool])

        assert toolset.is_tool_available("grep", mock_run_ctx) is True
        assert toolset.is_tool_available("unavailable_tool", mock_run_ctx) is False

    def test_is_tool_available_dynamic(self, agent_context, mock_run_ctx) -> None:
        """Should dynamically check tool availability."""
        DynamicTool._available = True
        toolset = Toolset(tools=[DynamicTool])

        assert toolset.is_tool_available("dynamic_tool", mock_run_ctx) is True

        # Change availability
        DynamicTool._available = False
        assert toolset.is_tool_available("dynamic_tool", mock_run_ctx) is False

        # Restore
        DynamicTool._available = True


class TestOptionalTools:
    """Tests for optional_tools functionality."""

    def test_subagent_available_with_optional_tools_missing(self, agent_context, mock_run_ctx) -> None:
        """Subagent should be available even if optional tools are missing."""
        parent_toolset = Toolset(tools=[GrepTool, ViewTool])

        config = SubagentConfig(
            name="test_subagent",
            description="Test subagent",
            system_prompt="You are a test agent",
            tools=["grep"],  # required
            optional_tools=["nonexistent_tool"],  # optional, missing
        )

        tool_cls = create_subagent_tool_from_config(config, parent_toolset, model="test")
        tool_instance = tool_cls()

        # Should still be available because required tools exist
        assert tool_instance.is_available(mock_run_ctx) is True

    def test_subagent_unavailable_when_required_missing_but_optional_present(self, agent_context, mock_run_ctx) -> None:
        """Subagent should be unavailable if required tools are missing, even with optional present."""
        parent_toolset = Toolset(tools=[GrepTool, ViewTool])

        config = SubagentConfig(
            name="test_subagent",
            description="Test subagent",
            system_prompt="You are a test agent",
            tools=["nonexistent_tool"],  # required, missing
            optional_tools=["grep"],  # optional, present
        )

        tool_cls = create_subagent_tool_from_config(config, parent_toolset, model="test")
        tool_instance = tool_cls()

        # Should be unavailable because required tool is missing
        assert tool_instance.is_available(mock_run_ctx) is False

    def test_subagent_with_both_required_and_optional_tools(self, agent_context, mock_run_ctx) -> None:
        """Subagent should include both required and optional tools in subset."""
        parent_toolset = Toolset(tools=[GrepTool, ViewTool])

        config = SubagentConfig(
            name="test_subagent",
            description="Test subagent",
            system_prompt="You are a test agent",
            tools=["grep"],  # required
            optional_tools=["view"],  # optional
        )

        tool_cls = create_subagent_tool_from_config(config, parent_toolset, model="test")
        tool_instance = tool_cls()

        assert tool_instance.is_available(mock_run_ctx) is True

    def test_subagent_only_optional_tools_always_available(self, agent_context, mock_run_ctx) -> None:
        """Subagent with only optional_tools (no required) should always be available."""
        parent_toolset = Toolset(tools=[GrepTool, ViewTool])

        config = SubagentConfig(
            name="test_subagent",
            description="Test subagent",
            system_prompt="You are a test agent",
            tools=None,  # no required tools
            optional_tools=["grep", "nonexistent"],  # optional only
        )

        tool_cls = create_subagent_tool_from_config(config, parent_toolset, model="test")
        tool_instance = tool_cls()

        # Should be available because tools=None means inherit all (no required check)
        assert tool_instance.is_available(mock_run_ctx) is True


def test_subagent_toolset_excludes_ask_user_question_when_inheriting_all() -> None:
    from ya_agent_sdk.subagents.builder import _build_toolsets

    parent_toolset = Toolset(tools=[GrepTool, AskUserQuestionTool])
    config = SubagentConfig(
        name="worker",
        description="Worker",
        system_prompt="You are a worker.",
    )

    subagent_toolsets = _build_toolsets(config, parent_toolset)

    assert len(subagent_toolsets) == 1
    assert subagent_toolsets[0].tool_names == ["grep"]
    assert "ask_user_question" in parent_toolset.tool_names


def test_subagent_own_toolset_excludes_ask_user_question() -> None:
    from ya_agent_sdk.subagents.builder import _build_toolsets

    own_toolset = Toolset(tools=[ViewTool, AskUserQuestionTool])
    parent_toolset = Toolset(tools=[GrepTool])
    config = SubagentConfig(
        name="worker",
        description="Worker",
        system_prompt="You are a worker.",
        toolsets=[own_toolset],
    )

    subagent_toolsets = _build_toolsets(config, parent_toolset)

    assert subagent_toolsets[0].tool_names == ["view"]
    assert "ask_user_question" in own_toolset.tool_names


def test_subagent_requiring_ask_user_question_is_unavailable(mock_run_ctx) -> None:
    parent_toolset = Toolset(tools=[GrepTool, AskUserQuestionTool])
    config = SubagentConfig(
        name="worker",
        description="Worker",
        system_prompt="You are a worker.",
        tools=["ask_user_question"],
    )

    tool_cls = create_subagent_tool_from_config(config, parent_toolset, model="test")

    assert parent_toolset.is_tool_available("ask_user_question", mock_run_ctx) is True
    assert tool_cls().is_available(mock_run_ctx) is False


class TestModelCfgResolution:
    """Tests for model_cfg resolution in subagent creation."""

    def test_model_cfg_from_preset_string(self, agent_context) -> None:
        """Subagent should resolve model_cfg from preset string."""
        parent_toolset = Toolset(tools=[GrepTool])

        config = SubagentConfig(
            name="test_subagent",
            description="Test subagent",
            system_prompt="You are a test agent",
            model_cfg="claude_200k",  # preset string
        )

        # Just verify it doesn't raise - actual ModelConfig creation happens internally
        tool_cls = create_subagent_tool_from_config(config, parent_toolset, model="test")
        assert tool_cls is not None

    def test_model_cfg_from_dict(self, agent_context) -> None:
        """Subagent should accept model_cfg as dict."""
        parent_toolset = Toolset(tools=[GrepTool])

        config = SubagentConfig(
            name="test_subagent",
            description="Test subagent",
            system_prompt="You are a test agent",
            model_cfg={"context_window": 100000, "max_images": 5},
        )

        tool_cls = create_subagent_tool_from_config(config, parent_toolset, model="test")
        assert tool_cls is not None

    def test_model_cfg_inherit(self, agent_context) -> None:
        """Subagent should inherit model_cfg when set to 'inherit'."""
        parent_toolset = Toolset(tools=[GrepTool])

        config = SubagentConfig(
            name="test_subagent",
            description="Test subagent",
            system_prompt="You are a test agent",
            model_cfg="inherit",
        )

        tool_cls = create_subagent_tool_from_config(config, parent_toolset, model="test")
        assert tool_cls is not None

    def test_model_cfg_none_inherits(self, agent_context) -> None:
        """Subagent should inherit model_cfg when None (default)."""
        parent_toolset = Toolset(tools=[GrepTool])

        config = SubagentConfig(
            name="test_subagent",
            description="Test subagent",
            system_prompt="You are a test agent",
            model_cfg=None,  # default, inherit
        )

        tool_cls = create_subagent_tool_from_config(config, parent_toolset, model="test")
        assert tool_cls is not None


# =============================================================================
# Agent registry cleanup on failure tests
# =============================================================================


@pytest.fixture
async def async_agent_context(tmp_path):
    """Create an async AgentContext for tests that need create_subagent_context."""
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as env:
        async with AgentContext(env=env) as ctx:
            yield ctx


async def test_subagent_pending_bus_message_redirects_without_model_retry(
    async_agent_context: AgentContext,
) -> None:
    calls: list[list[ModelMessage]] = []

    async def stream_function(messages: list[ModelMessage], _agent_info: FunctionAgentInfo):
        calls.append(list(messages))
        if len(calls) == 1:
            agent_id = next(iter(async_agent_context.agent_registry))
            async_agent_context.send_message(
                BusMessage(content="Revise the subagent answer", source="main", target=agent_id)
            )
            yield "first subagent answer"
            return
        yield "revised subagent answer"

    agent: Agent[AgentContext, str] = Agent(
        model=FunctionModel(stream_function=stream_function),
        name="steered_agent",
        retries={"tools": 1, "output": 0},
        capabilities=[MessageBusGuardCapability(), ProcessHistory(inject_bus_messages)],
    )
    call_func = create_subagent_call_func(agent)
    mock_ctx = MagicMock(spec=RunContext)
    mock_ctx.deps = async_agent_context
    mock_ctx.tool_call_id = "test-tool-call"

    response = await call_func(MagicMock(spec=BaseTool), mock_ctx, "test prompt")

    assert len(calls) == 2
    assert "revised subagent answer" in response
    assert any(
        isinstance(part, UserPromptPart) and "Revise the subagent answer" in str(part.content)
        for message in calls[1]
        if isinstance(message, ModelRequest)
        for part in message.parts
    )
    assert not any(
        isinstance(part, RetryPromptPart)
        for message in async_agent_context.subagent_history[next(iter(async_agent_context.subagent_history))]
        if isinstance(message, ModelRequest)
        for part in message.parts
    )


async def test_subagent_retries_transient_provider_stream_failure(
    async_agent_context: AgentContext,
) -> None:
    calls: list[list[ModelMessage]] = []

    async def stream_function(messages: list[ModelMessage], _agent_info: FunctionAgentInfo):
        calls.append(list(messages))
        if len(calls) == 1:
            yield "partial subagent response"
            raise UnexpectedModelBehavior(
                "An error occurred while processing your request. You can retry your request."
            )
        yield "recovered subagent response"

    async_agent_context.model_cfg = ModelConfig(
        stream_resume_on_error=True,
        stream_resume_max_attempts=1,
        stream_transport_resume_max_attempts=2,
        stream_resume_prompt="Resume the subagent task.",
    )
    agent: Agent[AgentContext, str] = Agent(
        model=FunctionModel(stream_function=stream_function),
        name="retrying_agent",
    )
    call_func = create_subagent_call_func(agent)
    mock_ctx = MagicMock(spec=RunContext)
    mock_ctx.deps = async_agent_context
    mock_ctx.tool_call_id = "test-tool-call"

    response = await call_func(MagicMock(spec=BaseTool), mock_ctx, "test prompt")

    assert len(calls) == 2
    assert "recovered subagent response" in response
    resume_request = calls[1][-1]
    assert isinstance(resume_request, ModelRequest)
    assert any(
        isinstance(part, UserPromptPart) and part.content == "Resume the subagent task."
        for part in resume_request.parts
    )
    agent_id = next(iter(async_agent_context.agent_registry))
    usage_entry = async_agent_context.usage_snapshot_entries[f"usage:{agent_id}:test-tool-call"]
    assert usage_entry.usage.requests == 2
    cost_estimate = async_agent_context.build_usage_snapshot().total_cost_estimate
    assert cost_estimate is not None
    assert cost_estimate.total_requests == 2
    assert cost_estimate.unpriced_requests >= 1


async def test_subagent_transport_budget_resets_after_successful_model_request(
    async_agent_context: AgentContext,
) -> None:
    calls = 0
    tool_calls = 0
    resume_attempts: list[int] = []

    async def stream_function(_messages: list[ModelMessage], _agent_info: FunctionAgentInfo):
        nonlocal calls
        calls += 1
        if calls in {1, 3}:
            raise UnexpectedModelBehavior(
                "An error occurred while processing your request. You can retry your request."
            )
        if calls == 2:
            yield {0: DeltaToolCall(name="ping", json_args="{}", tool_call_id="ping-call")}
            return
        yield "recovered after transport reset"

    def resume_prompt_factory(
        _exc: BaseException,
        attempt_index: int,
        _history: Sequence[ModelMessage],
    ) -> str:
        resume_attempts.append(attempt_index)
        return f"Resume subagent attempt {attempt_index}."

    async_agent_context.stream_recovery_policy = StreamRecoveryPolicy(
        enabled=True,
        max_attempts=1,
        transport_max_attempts=2,
        resume_prompt="Unused static prompt.",
        resume_prompt_factory=resume_prompt_factory,
    )
    agent: Agent[AgentContext, str] = Agent(
        model=FunctionModel(stream_function=stream_function),
        name="retrying_agent",
    )

    @agent.tool_plain
    def ping() -> str:
        nonlocal tool_calls
        tool_calls += 1
        return "pong"

    call_func = create_subagent_call_func(agent)
    mock_ctx = MagicMock(spec=RunContext)
    mock_ctx.deps = async_agent_context
    mock_ctx.tool_call_id = "test-tool-call"

    response = await call_func(MagicMock(spec=BaseTool), mock_ctx, "test prompt")

    assert calls == 4
    assert tool_calls == 1
    assert resume_attempts == [1, 2]
    assert "recovered after transport reset" in response
    agent_id = next(iter(async_agent_context.agent_registry))
    usage_entry = async_agent_context.usage_snapshot_entries[f"usage:{agent_id}:test-tool-call"]
    assert usage_entry.usage.requests == 4
    cost_estimate = async_agent_context.build_usage_snapshot().total_cost_estimate
    assert cost_estimate is not None
    assert cost_estimate.total_requests == 4


async def test_subagent_does_not_transport_retry_permanent_model_behavior(
    async_agent_context: AgentContext,
) -> None:
    calls = 0

    async def stream_function(_messages: list[ModelMessage], _agent_info: FunctionAgentInfo):
        nonlocal calls
        calls += 1
        raise UnexpectedModelBehavior("Cannot apply a text delta to an existing tool call part")
        yield  # pragma: no cover

    async_agent_context.stream_recovery_policy = StreamRecoveryPolicy(
        enabled=True,
        max_attempts=1,
        transport_max_attempts=10,
        resume_prompt="Resume the subagent task.",
    )
    object.__setattr__(async_agent_context, "_stream_queue_enabled", True)
    agent: Agent[AgentContext, str] = Agent(
        model=FunctionModel(stream_function=stream_function),
        name="failing_agent",
    )
    call_func = create_subagent_call_func(agent)
    mock_ctx = MagicMock(spec=RunContext)
    mock_ctx.deps = async_agent_context
    mock_ctx.tool_call_id = "test-tool-call"

    with pytest.raises(UnexpectedModelBehavior, match="Cannot apply a text delta"):
        await call_func(MagicMock(spec=BaseTool), mock_ctx, "test prompt")

    assert calls == 1
    emitted_events = []
    for queue in async_agent_context.agent_stream_queues.values():
        while not queue.empty():
            emitted_events.append(queue.get_nowait())
    complete_event = next(event for event in emitted_events if isinstance(event, SubagentCompleteEvent))
    assert complete_event.success is False
    assert complete_event.request_count == 1


async def test_agent_registry_cleaned_up_on_new_agent_failure(async_agent_context: AgentContext) -> None:
    """Agent registry should not have ghost entries when a new subagent fails."""
    agent: Agent[AgentContext, str] = Agent(
        model="test",
        system_prompt="You are a test agent",
        name="failing_agent",
    )
    call_func = create_subagent_call_func(agent)

    mock_ctx = MagicMock(spec=RunContext)
    mock_ctx.deps = async_agent_context
    mock_ctx.tool_call_id = "test-tool-call"

    mock_self = MagicMock(spec=BaseTool)

    with patch(
        "ya_agent_sdk.toolsets.core.subagent.factory._run_subagent_iter",
        new_callable=AsyncMock,
        side_effect=RuntimeError("Model API failed"),
    ):
        with pytest.raises(RuntimeError, match="Model API failed"):
            await call_func(mock_self, mock_ctx, "test prompt")

    # agent_registry should NOT contain a ghost entry
    assert len(async_agent_context.agent_registry) == 0
    # subagent_history should also be empty
    assert len(async_agent_context.subagent_history) == 0


async def test_agent_registry_preserved_on_resume_agent_failure(async_agent_context: AgentContext) -> None:
    """Agent registry entry should be preserved when a resumed subagent fails.

    If the agent was already registered (e.g., from a previous successful call),
    the registry entry should not be removed even on failure, since the agent
    has valid history from before.
    """
    agent: Agent[AgentContext, str] = Agent(
        model="test",
        system_prompt="You are a test agent",
        name="resume_agent",
    )
    call_func = create_subagent_call_func(agent)

    mock_ctx = MagicMock(spec=RunContext)
    mock_ctx.deps = async_agent_context
    mock_ctx.tool_call_id = "test-tool-call"

    mock_self = MagicMock(spec=BaseTool)

    # Pre-populate agent_registry to simulate a previously successful call
    from ya_agent_sdk.context import AgentInfo

    agent_id = "resume_agent-abcd"
    async_agent_context.agent_registry[agent_id] = AgentInfo(
        agent_id=agent_id,
        agent_name="resume_agent",
        parent_agent_id=None,
    )
    # Also populate subagent_history to simulate prior success
    async_agent_context.subagent_history[agent_id] = []

    with patch(
        "ya_agent_sdk.toolsets.core.subagent.factory._run_subagent_iter",
        new_callable=AsyncMock,
        side_effect=RuntimeError("Model API failed on resume"),
    ):
        with pytest.raises(RuntimeError, match="Model API failed on resume"):
            await call_func(mock_self, mock_ctx, "continue work", agent_id)

    # agent_registry should still contain the entry (not cleaned up for resume)
    assert agent_id in async_agent_context.agent_registry
    assert async_agent_context.agent_registry[agent_id].agent_name == "resume_agent"


# =============================================================================
# Independent toolsets tests
# =============================================================================


class ImageGenTool(BaseTool):
    """Test tool representing an independent capability (e.g., image generation)."""

    name = "image_gen"
    description = "Generate images"

    async def call(self, ctx, prompt: str) -> str:
        return f"image_gen: {prompt}"


class DataProcessTool(BaseTool):
    """Test tool representing an independent capability (e.g., data processing)."""

    name = "data_process"
    description = "Process data"

    async def call(self, ctx, data: str) -> str:
        return f"data_process: {data}"


class AutoInheritTool(BaseTool):
    """Test tool that is automatically inherited by subagents."""

    name = "task_create"
    description = "Create a task"
    auto_inherit = True

    async def call(self, ctx) -> str:
        return "task_create"


def test_subagent_with_own_toolsets_only(agent_context, mock_run_ctx) -> None:
    """Subagent with own toolsets and no parent tools should always be available."""
    own_toolset = Toolset(tools=[ImageGenTool])
    parent_toolset = Toolset(tools=[GrepTool, ViewTool])

    config = SubagentConfig(
        name="designer",
        description="Design agent",
        system_prompt="You are a designer",
        toolsets=[own_toolset],
        tools=None,  # no parent tools required
    )

    tool_cls = create_subagent_tool_from_config(config, parent_toolset, model="test")
    tool_instance = tool_cls()

    # Always available since tools=None (no required parent tools)
    assert tool_instance.is_available(mock_run_ctx) is True


def test_subagent_with_own_toolsets_and_parent_tools(agent_context, mock_run_ctx) -> None:
    """Subagent with own toolsets + required parent tools: availability depends on parent."""
    own_toolset = Toolset(tools=[ImageGenTool])
    parent_toolset = Toolset(tools=[GrepTool, ViewTool])

    config = SubagentConfig(
        name="designer",
        description="Design agent with search",
        system_prompt="You are a designer with search capability",
        toolsets=[own_toolset],
        tools=["grep"],  # also needs grep from parent
    )

    tool_cls = create_subagent_tool_from_config(config, parent_toolset, model="test")
    tool_instance = tool_cls()

    # Available because grep exists in parent
    assert tool_instance.is_available(mock_run_ctx) is True


def test_subagent_with_own_toolsets_unavailable_when_parent_tools_missing(agent_context, mock_run_ctx) -> None:
    """Subagent with own toolsets + required parent tools: unavailable if parent tools missing."""
    own_toolset = Toolset(tools=[ImageGenTool])
    parent_toolset = Toolset(tools=[GrepTool])  # no ViewTool

    config = SubagentConfig(
        name="designer",
        description="Design agent",
        system_prompt="You are a designer",
        toolsets=[own_toolset],
        tools=["grep", "view"],  # requires view which is missing
    )

    tool_cls = create_subagent_tool_from_config(config, parent_toolset, model="test")
    tool_instance = tool_cls()

    # Unavailable because view is missing from parent
    assert tool_instance.is_available(mock_run_ctx) is False


def test_subagent_with_own_toolsets_gets_auto_inherit(agent_context, mock_run_ctx) -> None:
    """Subagent with own toolsets should still get auto_inherit tools from parent."""
    own_toolset = Toolset(tools=[ImageGenTool])
    parent_toolset = Toolset(tools=[GrepTool, AutoInheritTool])

    config = SubagentConfig(
        name="designer",
        description="Design agent",
        system_prompt="You are a designer",
        toolsets=[own_toolset],
        tools=None,  # no parent tools explicitly requested
    )

    # Should not raise - auto_inherit tools are included
    tool_cls = create_subagent_tool_from_config(config, parent_toolset, model="test")
    assert tool_cls is not None


def test_subagent_without_toolsets_backward_compatible(agent_context, mock_run_ctx) -> None:
    """Subagent without toolsets should behave identically to before."""
    parent_toolset = Toolset(tools=[GrepTool, ViewTool])

    config = SubagentConfig(
        name="test_subagent",
        description="Test subagent",
        system_prompt="You are a test agent",
        # toolsets=None (default)
        tools=["grep", "view"],
    )

    tool_cls = create_subagent_tool_from_config(config, parent_toolset, model="test")
    tool_instance = tool_cls()

    assert tool_instance.is_available(mock_run_ctx) is True


def test_subagent_config_toolsets_default_none() -> None:
    """SubagentConfig should default toolsets to None."""
    config = SubagentConfig(
        name="test",
        description="test",
        system_prompt="test",
    )
    assert config.toolsets is None


def test_subagent_with_empty_toolsets_does_not_inherit_all_parent_tools(agent_context, mock_run_ctx) -> None:
    """Empty toolsets=[] should NOT inherit all parent tools (only auto_inherit).

    This is the key edge case: [] is falsy in Python, but semantically it means
    'this subagent has its own toolsets (currently empty)', not 'inherit all parent tools'.
    """
    parent_toolset = Toolset(tools=[GrepTool, ViewTool])

    config = SubagentConfig(
        name="empty_toolset_agent",
        description="Agent with explicitly empty toolsets",
        system_prompt="You are a test agent",
        toolsets=[],  # explicitly empty, NOT None
        tools=None,
    )

    # Should not raise
    tool_cls = create_subagent_tool_from_config(config, parent_toolset, model="test")
    tool_instance = tool_cls()

    # Should be available (no required parent tools)
    assert tool_instance.is_available(mock_run_ctx) is True


def test_build_toolsets_empty_list_vs_none() -> None:
    """Verify _build_toolsets distinguishes between toolsets=[] and toolsets=None."""
    from ya_agent_sdk.subagents.builder import _build_toolsets

    parent_toolset = Toolset(tools=[GrepTool, ViewTool])

    # toolsets=None -> gets all parent tools (1 toolset = full parent subset)
    config_none = SubagentConfig(
        name="t",
        description="t",
        system_prompt="t",
        toolsets=None,
        tools=None,
    )
    result_none = _build_toolsets(config_none, parent_toolset)
    assert len(result_none) == 1  # just the parent subset (all tools)

    # toolsets=[] -> gets empty own + auto_inherit parent subset (1 toolset)
    config_empty = SubagentConfig(
        name="t",
        description="t",
        system_prompt="t",
        toolsets=[],
        tools=None,
    )
    result_empty = _build_toolsets(config_empty, parent_toolset)
    assert len(result_empty) == 1  # just the parent subset (auto_inherit only)

    # toolsets=[own] -> gets own + auto_inherit parent subset (2 toolsets)
    own_toolset = Toolset(tools=[ImageGenTool])
    config_set = SubagentConfig(
        name="t",
        description="t",
        system_prompt="t",
        toolsets=[own_toolset],
        tools=None,
    )
    result_set = _build_toolsets(config_set, parent_toolset)
    assert len(result_set) == 2  # own_toolset + parent auto_inherit subset

    # Key difference: None gives full parent, [] gives only auto_inherit
    # With no auto_inherit tools, the empty-subset has 0 tools, full has 2
    none_parent_subset = result_none[0]
    empty_parent_subset = result_empty[0]
    assert set(none_parent_subset.tool_names) == {"grep", "view"}
    assert set(empty_parent_subset.tool_names) == set()  # no auto_inherit tools in parent
