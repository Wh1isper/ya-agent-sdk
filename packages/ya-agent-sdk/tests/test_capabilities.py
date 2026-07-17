"""Tests for capabilities parameter in create_agent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, Hooks, ProcessHistory
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import AbstractToolset, CombinedToolset, FunctionToolset, WrapperToolset
from pydantic_ai.toolsets._dynamic import DynamicToolset
from ya_agent_sdk.agents.main import create_agent
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.environment.local import LocalEnvironment
from ya_agent_sdk.subagents.config import SubagentConfig
from ya_agent_sdk.toolsets.core.base import BaseTool, Toolset
from ya_agent_sdk.toolsets.core.interaction import AskUserQuestionTool
from ya_agent_sdk.toolsets.tool_search import ToolSearchToolSet


@pytest.fixture
async def env(tmp_path: Path):
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as e:
        yield e


@dataclass
class MathCapability(AbstractCapability[Any]):
    """Test capability that provides math tools."""

    def get_toolset(self):
        toolset = FunctionToolset()

        @toolset.tool_plain
        def add(a: float, b: float) -> float:
            """Add two numbers."""
            return a + b

        return toolset


@dataclass
class InstructionCapability(AbstractCapability[Any]):
    """Test capability that provides instructions."""

    text: str = "You are a helpful calculator."

    def get_instructions(self):
        return self.text


@dataclass
class RequestCounter(AbstractCapability[Any]):
    """Test capability that counts model requests per run."""

    count: int = 0

    async def for_run(self, ctx: RunContext[Any]) -> RequestCounter:
        return RequestCounter()

    async def before_model_request(self, ctx, request_context):
        self.count += 1
        return request_context


class ExtraOrdinaryTool(BaseTool):
    """Ordinary tool in an extra toolset."""

    name = "extra_ordinary_tool"
    description = "An ordinary tool"

    async def call(self, ctx: RunContext) -> str:
        return "ok"


class ExtraDelegationTool(BaseTool):
    """Delegation-tagged tool in an extra toolset."""

    name = "extra_delegation_tool"
    description = "A delegation tool"
    tags = frozenset({"delegation"})

    async def call(self, ctx: RunContext) -> str:
        return "delegated"


@dataclass
class MainAgentOnlyToolCapability(AbstractCapability[Any]):
    """Capability that contributes an SDK Toolset with a host-facing tool."""

    def get_toolset(self) -> Toolset[Any]:
        return Toolset(
            tools=[ExtraOrdinaryTool, AskUserQuestionTool],
            skip_unavailable=False,
        )


@dataclass
class MainAgentOnlyWrapperCapability(AbstractCapability[Any]):
    """Outermost capability that attempts to inject a host-facing tool."""

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(
            position="outermost",
            wraps=(AbstractCapability,),
        )

    def get_wrapper_toolset(self, toolset):
        return CombinedToolset([
            toolset,
            Toolset(tools=[AskUserQuestionTool], skip_unavailable=False),
        ])


@dataclass
class DynamicMainAgentOnlyCapability(AbstractCapability[Any]):
    """Capability whose dynamic factory attempts to inject a host-facing tool."""

    async_factory: bool
    per_run_step: bool

    def get_toolset(self):
        def build_toolset() -> Toolset[Any]:
            return Toolset(
                tools=[ExtraOrdinaryTool, AskUserQuestionTool],
                skip_unavailable=False,
            )

        if self.async_factory:

            async def factory(ctx: RunContext[Any]) -> Toolset[Any]:
                return build_toolset()

        else:

            def factory(ctx: RunContext[Any]) -> Toolset[Any]:
                return build_toolset()

        return DynamicToolset(factory, per_run_step=self.per_run_step)


@dataclass
class DynamicOpaqueToolSearchCapability(AbstractCapability[Any]):
    """Dynamic factory that hides an SDK Toolset inside ToolSearchToolSet."""

    def get_toolset(self):
        def factory(ctx: RunContext[Any]) -> ToolSearchToolSet:
            return ToolSearchToolSet([
                Toolset(
                    tools=[ExtraOrdinaryTool, AskUserQuestionTool],
                    skip_unavailable=False,
                )
            ])

        return factory


@dataclass
class OpaqueToolSearchWrapperCapability(AbstractCapability[Any]):
    """Capability wrapper that hides an SDK Toolset inside ToolSearchToolSet."""

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position="outermost", wraps=(AbstractCapability,))

    def get_wrapper_toolset(self, toolset):
        return CombinedToolset([
            toolset,
            ToolSearchToolSet([
                Toolset(
                    tools=[ExtraOrdinaryTool, AskUserQuestionTool],
                    skip_unavailable=False,
                )
            ]),
        ])


@dataclass
class StableDynamicToolsetCapability(AbstractCapability[Any]):
    """Dynamic capability that returns the same lifecycle-aware toolset."""

    toolset: WrapperToolset[Any]
    async_factory: bool

    def get_toolset(self):
        if self.async_factory:

            async def factory(ctx: RunContext[Any]) -> WrapperToolset[Any]:
                return self.toolset

        else:

            def factory(ctx: RunContext[Any]) -> WrapperToolset[Any]:
                return self.toolset

        return DynamicToolset(factory, per_run_step=True, id="stable-dynamic")


@dataclass
class LifecycleTrackingToolset(WrapperToolset[Any]):
    events: list[str]

    async def __aenter__(self):
        await self.wrapped.__aenter__()
        self.events.append("enter")
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        self.events.append("exit")
        return await self.wrapped.__aexit__(*args)


@dataclass
class OutermostCapability(AbstractCapability[Any]):
    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position="outermost")


@dataclass
class InnermostCapability(AbstractCapability[Any]):
    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position="innermost")


@dataclass
class RequiredCapability(AbstractCapability[Any]):
    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(wrapped_by=(WrappingCapability,))


@dataclass
class WrappingCapability(AbstractCapability[Any]):
    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(wraps=(RequiredCapability,), requires=(RequiredCapability,))


def _assembled_sdk_toolsets(agent: Any) -> list[Toolset[Any]]:
    toolsets: list[Toolset[Any]] = []
    root = agent._get_toolset(run_capability=agent.root_capability)
    root.apply(lambda toolset: toolsets.append(toolset) if isinstance(toolset, Toolset) else None)
    return toolsets


async def test_self_fork_filters_delegation_tags_from_extra_toolsets(env):
    """Self fork should hide delegation tools from every SDK Toolset."""
    config = SubagentConfig(name="helper", description="Helper", system_prompt="You are helper.")
    extra_toolset = Toolset(tools=[ExtraOrdinaryTool, ExtraDelegationTool])

    runtime = create_agent(
        TestModel(),
        env=env,
        subagent_configs=[config],
        unified_subagents=True,
        toolsets=[extra_toolset],
        defer_model_check=True,
    )

    assert runtime.ctx.self_fork_agent is not None
    tool_names = [name for toolset in runtime.ctx.self_fork_agent._user_toolsets for name in toolset.tool_names]
    assert "extra_ordinary_tool" in tool_names
    assert "extra_delegation_tool" not in tool_names


async def test_self_fork_excludes_ask_user_question_from_core_toolset(env) -> None:
    config = SubagentConfig(name="helper", description="Helper", system_prompt="You are helper.")

    runtime = create_agent(
        TestModel(),
        env=env,
        tools=[AskUserQuestionTool],
        subagent_configs=[config],
        unified_subagents=True,
        defer_model_check=True,
    )

    assert runtime.core_toolset is not None
    assert "ask_user_question" in runtime.core_toolset.tool_names
    assert runtime.ctx.self_fork_agent is not None
    self_fork_tool_names = [
        name for toolset in runtime.ctx.self_fork_agent._user_toolsets for name in toolset.tool_names
    ]
    assert "ask_user_question" not in self_fork_tool_names


async def test_subagent_capability_toolset_excludes_ask_user_question() -> None:
    from ya_agent_sdk.subagents.builder import _build_subagent_agent

    agent, _ = _build_subagent_agent(
        SubagentConfig(name="helper", description="Helper", system_prompt="You are helper."),
        Toolset(tools=[]),
        model="test",
        capabilities=[MainAgentOnlyToolCapability()],
    )

    capability_toolsets = _assembled_sdk_toolsets(agent)
    tool_names = {tool_name for toolset in capability_toolsets for tool_name in toolset.tool_names}
    assert "extra_ordinary_tool" in tool_names
    assert "ask_user_question" not in tool_names
    assert any(toolset._skip_unavailable is False for toolset in capability_toolsets)


async def test_subagent_capability_wrapper_cannot_inject_ask_user_question() -> None:
    from ya_agent_sdk.subagents.builder import _build_subagent_agent

    model = TestModel()
    agent, _ = _build_subagent_agent(
        SubagentConfig(name="helper", description="Helper", system_prompt="You are helper."),
        Toolset(tools=[]),
        model=model,
        capabilities=[MainAgentOnlyWrapperCapability()],
    )

    assembled_toolsets = _assembled_sdk_toolsets(agent)
    assert "ask_user_question" not in {tool_name for toolset in assembled_toolsets for tool_name in toolset.tool_names}
    assert any(toolset._skip_unavailable is False for toolset in assembled_toolsets)

    await agent.run("Check the final tool surface", deps=AgentContext())
    assert model.last_model_request_parameters is not None
    assert "ask_user_question" not in {tool.name for tool in model.last_model_request_parameters.function_tools}


@pytest.mark.parametrize("async_factory", [False, True])
@pytest.mark.parametrize("per_run_step", [False, True])
async def test_subagent_dynamic_capability_cannot_inject_ask_user_question(
    async_factory: bool,
    per_run_step: bool,
) -> None:
    from ya_agent_sdk.subagents.builder import _build_subagent_agent

    model = TestModel(call_tools=[])
    agent, _ = _build_subagent_agent(
        SubagentConfig(name="helper", description="Helper", system_prompt="You are helper."),
        Toolset(tools=[]),
        model=model,
        capabilities=[
            DynamicMainAgentOnlyCapability(
                async_factory=async_factory,
                per_run_step=per_run_step,
            )
        ],
    )

    await agent.run("Check the dynamic tool surface", deps=AgentContext())

    assert model.last_model_request_parameters is not None
    tool_names = {tool.name for tool in model.last_model_request_parameters.function_tools}
    assert "extra_ordinary_tool" in tool_names
    assert "ask_user_question" not in tool_names


@pytest.mark.parametrize(
    "capability",
    [
        DynamicOpaqueToolSearchCapability(),
        OpaqueToolSearchWrapperCapability(),
    ],
)
async def test_subagent_opaque_composite_cannot_expose_ask_user_question(
    capability: AbstractCapability[Any],
) -> None:
    from ya_agent_sdk.subagents.builder import _build_subagent_agent

    model = TestModel(call_tools=[])
    agent, _ = _build_subagent_agent(
        SubagentConfig(name="helper", description="Helper", system_prompt="You are helper."),
        Toolset(tools=[]),
        model=model,
        capabilities=[capability],
    )
    parent_ctx = AgentContext()
    subagent_ctx = parent_ctx.create_subagent_context("helper")
    subagent_ctx.tool_search_loaded_tools = ["extra_ordinary_tool", "ask_user_question"]

    await agent.run("Check the opaque tool surface", deps=subagent_ctx)

    assert model.last_model_request_parameters is not None
    tool_names = {tool.name for tool in model.last_model_request_parameters.function_tools}
    assert {"tool_search", "extra_ordinary_tool"} <= tool_names
    assert "ask_user_question" not in tool_names


async def test_dynamic_policy_retry_after_filter_failure_does_not_use_stale_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ya_agent_sdk.subagents import agent as subagent_agent_module

    raw_toolset = Toolset(
        tools=[ExtraOrdinaryTool, AskUserQuestionTool],
        skip_unavailable=False,
    )
    dynamic = DynamicToolset(lambda ctx: raw_toolset, per_run_step=True)
    sanitized_dynamic = subagent_agent_module._sanitize_dynamic_toolset(dynamic)
    original_filter = subagent_agent_module._filter_subagent_toolset_tree
    filter_calls = 0

    def flaky_filter(toolset: AbstractToolset[AgentContext]) -> AbstractToolset[AgentContext]:
        nonlocal filter_calls
        filter_calls += 1
        if filter_calls == 1:
            raise RuntimeError("filter failed")
        return original_filter(toolset)

    monkeypatch.setattr(subagent_agent_module, "_filter_subagent_toolset_tree", flaky_filter)
    run_ctx = MagicMock(spec=RunContext)
    run_ctx.deps = AgentContext()

    with pytest.raises(RuntimeError, match="filter failed"):
        await sanitized_dynamic.toolset_func(run_ctx)
    retried = await sanitized_dynamic.toolset_func(run_ctx)

    assert filter_calls == 2
    assert isinstance(retried, Toolset)
    assert "extra_ordinary_tool" in retried.tool_names
    assert "ask_user_question" not in retried.tool_names


@pytest.mark.parametrize("async_factory", [False, True])
async def test_subagent_dynamic_policy_preserves_stable_factory_lifecycle(
    async_factory: bool,
) -> None:
    from ya_agent_sdk.subagents.builder import _build_subagent_agent

    events: list[str] = []
    stable_toolset = LifecycleTrackingToolset(
        wrapped=Toolset(tools=[ExtraOrdinaryTool]),
        events=events,
    )
    model = TestModel(call_tools=["extra_ordinary_tool"])
    agent, _ = _build_subagent_agent(
        SubagentConfig(name="helper", description="Helper", system_prompt="You are helper."),
        Toolset(tools=[]),
        model=model,
        capabilities=[
            StableDynamicToolsetCapability(
                toolset=stable_toolset,
                async_factory=async_factory,
            )
        ],
    )
    assembled_leaves: list[AbstractToolset[Any]] = []
    agent._get_toolset(run_capability=agent.root_capability).apply(assembled_leaves.append)

    await agent.run("Use the ordinary tool", deps=AgentContext())

    assert any(isinstance(toolset, DynamicToolset) and toolset.id == "stable-dynamic" for toolset in assembled_leaves)
    assert events == ["enter", "exit"]


def test_subagent_capability_policy_preserves_ordering_constraints() -> None:
    from ya_agent_sdk.subagents.builder import _build_subagent_agent

    agent, _ = _build_subagent_agent(
        SubagentConfig(name="helper", description="Helper", system_prompt="You are helper."),
        Toolset(tools=[]),
        model="test",
        capabilities=[
            InnermostCapability(),
            RequiredCapability(),
            WrappingCapability(),
            OutermostCapability(),
        ],
    )
    leaves: list[AbstractCapability[Any]] = []
    agent.root_capability.apply(leaves.append)

    outer_index = next(index for index, capability in enumerate(leaves) if isinstance(capability, OutermostCapability))
    wrapper_index = next(index for index, capability in enumerate(leaves) if isinstance(capability, WrappingCapability))
    required_index = next(
        index for index, capability in enumerate(leaves) if isinstance(capability, RequiredCapability)
    )
    inner_index = next(index for index, capability in enumerate(leaves) if isinstance(capability, InnermostCapability))
    assert outer_index < wrapper_index < required_index < inner_index


async def test_self_fork_capability_toolset_excludes_ask_user_question(env) -> None:
    config = SubagentConfig(name="helper", description="Helper", system_prompt="You are helper.")
    runtime = create_agent(
        TestModel(),
        env=env,
        capabilities=[MainAgentOnlyToolCapability()],
        subagent_configs=[config],
        unified_subagents=True,
        defer_model_check=True,
    )

    main_tool_names = {
        tool_name for toolset in _assembled_sdk_toolsets(runtime.agent) for tool_name in toolset.tool_names
    }
    assert {"extra_ordinary_tool", "ask_user_question"} <= main_tool_names
    assert runtime.ctx.self_fork_agent is not None
    self_fork_capability_toolsets = _assembled_sdk_toolsets(runtime.ctx.self_fork_agent)
    self_fork_tool_names = {tool_name for toolset in self_fork_capability_toolsets for tool_name in toolset.tool_names}
    assert "extra_ordinary_tool" in self_fork_tool_names
    assert "ask_user_question" not in self_fork_tool_names
    assert any(toolset._skip_unavailable is False for toolset in self_fork_capability_toolsets)


async def test_capabilities_forwarded_to_agent(env):
    """Test that capabilities parameter is forwarded to the Agent."""
    math_cap = MathCapability()

    runtime = create_agent(
        TestModel(),
        env=env,
        capabilities=[math_cap],
        defer_model_check=True,
    )
    async with runtime:
        # Agent should have the capability's tools available
        result = await runtime.agent.run("What is 2 + 3?", deps=runtime.ctx)
        assert result.output is not None


async def test_capabilities_with_instructions(env):
    """Test that capability instructions are included."""
    cap = InstructionCapability(text="Always respond in JSON format.")

    runtime = create_agent(
        TestModel(),
        env=env,
        capabilities=[cap],
        defer_model_check=True,
    )
    async with runtime:
        result = await runtime.agent.run("Hello", deps=runtime.ctx)
        assert result.output is not None


async def test_capabilities_with_hooks(env):
    """Test that capability lifecycle hooks fire."""
    call_log: list[str] = []

    hooks = Hooks(
        before_run=lambda ctx: call_log.append("before_run"),
    )

    runtime = create_agent(
        TestModel(),
        env=env,
        capabilities=[hooks],
        defer_model_check=True,
    )
    async with runtime:
        await runtime.agent.run("Hello", deps=runtime.ctx)
        assert "before_run" in call_log


async def test_multiple_capabilities(env):
    """Test composing multiple capabilities."""
    math_cap = MathCapability()
    instr_cap = InstructionCapability(text="Be precise.")
    counter = RequestCounter()

    runtime = create_agent(
        TestModel(),
        env=env,
        capabilities=[math_cap, instr_cap, counter],
        defer_model_check=True,
    )
    async with runtime:
        await runtime.agent.run("Hello", deps=runtime.ctx)


async def test_capabilities_none_by_default(env):
    """Test that capabilities=None works (backward compat)."""
    runtime = create_agent(
        TestModel(),
        env=env,
        defer_model_check=True,
    )
    async with runtime:
        result = await runtime.agent.run("Hello", deps=runtime.ctx)
        assert result.output is not None


async def test_capabilities_empty_list(env):
    """Test that capabilities=[] works."""
    runtime = create_agent(
        TestModel(),
        env=env,
        capabilities=[],
        defer_model_check=True,
    )
    async with runtime:
        result = await runtime.agent.run("Hello", deps=runtime.ctx)
        assert result.output is not None


# =============================================================================
# Subagent Capabilities Inheritance Tests
# =============================================================================


async def test_subagent_inherits_capabilities(env):
    """Test that subagents inherit parent capabilities by default."""
    math_cap = MathCapability()

    config = SubagentConfig(
        name="helper",
        description="A helper subagent",
        system_prompt="You are a helper.",
    )

    runtime = create_agent(
        TestModel(),
        env=env,
        capabilities=[math_cap],
        subagent_configs=[config],
        defer_model_check=True,
    )
    async with runtime:
        # The subagent should have been built with inherited capabilities.
        # We verify by checking the agent was created successfully and
        # the core_toolset has subagent tools.
        assert runtime.core_toolset is not None
        assert "helper" in runtime.core_toolset._tool_classes


async def test_subagent_no_inherit_when_disabled(env):
    """Test inherit_capabilities=False prevents capability inheritance."""
    math_cap = MathCapability()

    config = SubagentConfig(
        name="helper",
        description="A helper subagent",
        system_prompt="You are a helper.",
    )

    runtime = create_agent(
        TestModel(),
        env=env,
        capabilities=[math_cap],
        inherit_capabilities=False,
        subagent_configs=[config],
        defer_model_check=True,
    )
    async with runtime:
        # Should still create the subagent, just without capabilities
        assert runtime.core_toolset is not None
        assert "helper" in runtime.core_toolset._tool_classes


async def test_subagent_config_capabilities_override(env):
    """Test that config.capabilities overrides parent capabilities."""
    parent_cap = InstructionCapability(text="parent instruction")
    child_cap = InstructionCapability(text="child instruction")

    config = SubagentConfig(
        name="helper",
        description="A helper subagent",
        system_prompt="You are a helper.",
        capabilities=[child_cap],
    )

    runtime = create_agent(
        TestModel(),
        env=env,
        capabilities=[parent_cap],
        subagent_configs=[config],
        defer_model_check=True,
    )
    async with runtime:
        assert runtime.core_toolset is not None
        assert "helper" in runtime.core_toolset._tool_classes


async def test_subagent_unified_inherits_capabilities(env):
    """Test unified subagent tool also receives inherited capabilities."""
    math_cap = MathCapability()

    configs = [
        SubagentConfig(
            name="helper1",
            description="Helper 1",
            system_prompt="You are helper 1.",
        ),
        SubagentConfig(
            name="helper2",
            description="Helper 2",
            system_prompt="You are helper 2.",
        ),
    ]

    runtime = create_agent(
        TestModel(),
        env=env,
        capabilities=[math_cap],
        subagent_configs=configs,
        unified_subagents=True,
        defer_model_check=True,
    )
    async with runtime:
        assert runtime.core_toolset is not None
        assert "delegate" in runtime.core_toolset._tool_classes
        delegate_tool = runtime.core_toolset._tool_classes["delegate"]
        assert delegate_tool._available_subagents == ("helper1", "helper2")


async def test_subagent_no_capabilities_when_parent_has_none(env):
    """Test subagents work fine when parent has no capabilities."""
    config = SubagentConfig(
        name="helper",
        description="A helper subagent",
        system_prompt="You are a helper.",
    )

    runtime = create_agent(
        TestModel(),
        env=env,
        subagent_configs=[config],
        defer_model_check=True,
    )
    async with runtime:
        assert runtime.core_toolset is not None
        assert "helper" in runtime.core_toolset._tool_classes


class CapturingTestModel(TestModel):
    """TestModel variant that records request messages."""

    last_messages: list[ModelMessage] | None = None

    async def request(self, messages, model_settings, model_request_parameters):
        self.last_messages = messages
        return await super().request(messages, model_settings, model_request_parameters)


async def test_process_history_capability_supported(env):
    """Test ProcessHistory capabilities can be passed directly to create_agent."""

    def add_marker(ctx: RunContext[Any], messages: list[ModelMessage]) -> list[ModelMessage]:
        for message in messages:
            if isinstance(message, ModelRequest):
                message.parts.append(UserPromptPart(content="capability-marker"))
                break
        return messages

    model = CapturingTestModel(custom_output_text="ok")
    runtime = create_agent(
        model,
        env=env,
        capabilities=[ProcessHistory(add_marker)],
        defer_model_check=True,
    )

    async with runtime:
        await runtime.agent.run("Hello", deps=runtime.ctx)

    assert model.last_messages is not None
    assert any(
        isinstance(message, ModelRequest)
        and any(isinstance(part, UserPromptPart) and part.content == "capability-marker" for part in message.parts)
        for message in model.last_messages
    )


async def test_create_agent_capability_phase_order(env):
    """Test pre_capabilities run before SDK history and capabilities run after it."""
    events: list[str] = []

    def marker_processor(marker: str):
        def process(ctx: RunContext[Any], messages: list[ModelMessage]) -> list[ModelMessage]:
            events.append(marker)
            for message in messages:
                if isinstance(message, ModelRequest):
                    message.parts.append(UserPromptPart(content=marker))
                    break
            return messages

        return process

    model = CapturingTestModel(custom_output_text="ok")
    runtime = create_agent(
        model,
        env=env,
        pre_capabilities=[ProcessHistory(marker_processor("pre-capability-marker"))],
        capabilities=[ProcessHistory(marker_processor("post-capability-marker"))],
        defer_model_check=True,
    )

    async with runtime:
        await runtime.agent.run("Hello", deps=runtime.ctx)

    assert model.last_messages is not None
    parts = [
        part.content
        for message in model.last_messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart)
    ]
    assert parts.index("pre-capability-marker") < parts.index("post-capability-marker")
    assert events.index("pre-capability-marker") < events.index("post-capability-marker")
