"""Tests for capabilities parameter in create_agent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability, Hooks, ProcessHistory
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset
from ya_agent_sdk.agents.main import create_agent
from ya_agent_sdk.environment.local import LocalEnvironment
from ya_agent_sdk.subagents.config import SubagentConfig


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


async def test_create_agent_history_processors_deprecated(env):
    """Test deprecated history_processors are converted to ProcessHistory capabilities."""

    def add_marker(ctx: RunContext[Any], messages: list[ModelMessage]) -> list[ModelMessage]:
        for message in messages:
            if isinstance(message, ModelRequest):
                message.parts.append(UserPromptPart(content="legacy-marker"))
                break
        return messages

    model = CapturingTestModel(custom_output_text="ok")

    with pytest.warns(DeprecationWarning, match="history_processors"):
        runtime = create_agent(
            model,
            env=env,
            history_processors=[add_marker],
            defer_model_check=True,
        )

    async with runtime:
        await runtime.agent.run("Hello", deps=runtime.ctx)

    assert model.last_messages is not None
    assert any(
        isinstance(message, ModelRequest)
        and any(isinstance(part, UserPromptPart) and part.content == "legacy-marker" for part in message.parts)
        for message in model.last_messages
    )


async def test_deprecated_history_processor_phase_order(env):
    """Test deprecated pre/post history processor phase ordering is preserved."""

    def marker_processor(marker: str):
        def process(ctx: RunContext[Any], messages: list[ModelMessage]) -> list[ModelMessage]:
            for message in messages:
                if isinstance(message, ModelRequest):
                    message.parts.append(UserPromptPart(content=marker))
                    break
            return messages

        return process

    model = CapturingTestModel(custom_output_text="ok")

    with pytest.warns(DeprecationWarning):
        runtime = create_agent(
            model,
            env=env,
            pre_history_processors=[marker_processor("pre-marker")],
            history_processors=[marker_processor("post-marker")],
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
    assert parts.index("pre-marker") < parts.index("post-marker")
    assert parts.index("post-marker") == len(parts) - 1


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


async def test_history_processors_migrate_to_capability_phases(env):
    """Test deprecated history processors match the new capability phase positions."""
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
    with pytest.warns(DeprecationWarning):
        runtime = create_agent(
            model,
            env=env,
            pre_history_processors=[marker_processor("pre-history-marker")],
            history_processors=[marker_processor("post-history-marker")],
            pre_capabilities=[ProcessHistory(marker_processor("pre-capability-marker"))],
            capabilities=[ProcessHistory(marker_processor("post-capability-marker"))],
            defer_model_check=True,
        )

    async with runtime:
        await runtime.agent.run("Hello", deps=runtime.ctx)

    assert events.index("pre-capability-marker") < events.index("pre-history-marker")
    assert events.index("pre-history-marker") < events.index("post-history-marker")
    assert events.index("post-history-marker") < events.index("post-capability-marker")
