"""Capability-first runtime composition tests."""

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


@pytest.fixture
async def env(tmp_path: Path):
    async with LocalEnvironment(
        allowed_paths=[tmp_path],
        default_path=tmp_path,
        tmp_base_dir=tmp_path,
    ) as environment:
        yield environment


@dataclass
class MathCapability(AbstractCapability[Any]):
    def get_toolset(self):
        toolset = FunctionToolset()

        @toolset.tool_plain
        def add(a: float, b: float) -> float:
            """Add two numbers."""
            return a + b

        return toolset


@dataclass
class InstructionCapability(AbstractCapability[Any]):
    text: str

    def get_instructions(self):
        return self.text


@dataclass
class RequestCounter(AbstractCapability[Any]):
    count: int = 0

    async def for_run(self, ctx: RunContext[Any]) -> RequestCounter:
        del ctx
        return RequestCounter()

    async def before_model_request(self, ctx, request_context):
        del ctx
        self.count += 1
        return request_context


async def test_capabilities_forwarded_to_agent(env) -> None:
    runtime = create_agent(
        TestModel(),
        env=env,
        capabilities=[MathCapability()],
        defer_model_check=True,
    )

    async with runtime:
        result = await runtime.agent.run("What is 2 + 3?", deps=runtime.ctx)

    assert result.output is not None


async def test_capabilities_compose_tools_instructions_and_hooks(env) -> None:
    call_log: list[str] = []
    runtime = create_agent(
        TestModel(),
        env=env,
        capabilities=[
            MathCapability(),
            InstructionCapability(text="Be precise."),
            RequestCounter(),
            Hooks(before_run=lambda ctx: call_log.append("before_run")),
        ],
        defer_model_check=True,
    )

    async with runtime:
        result = await runtime.agent.run("Hello", deps=runtime.ctx)

    assert result.output is not None
    assert call_log == ["before_run"]


async def test_runtime_accepts_no_explicit_capabilities(env) -> None:
    runtime = create_agent(TestModel(), env=env, defer_model_check=True)

    async with runtime:
        result = await runtime.agent.run("Hello", deps=runtime.ctx)

    assert result.output is not None


class CapturingTestModel(TestModel):
    last_messages: list[ModelMessage] | None = None

    async def request(self, messages, model_settings, model_request_parameters):
        self.last_messages = messages
        return await super().request(messages, model_settings, model_request_parameters)


async def test_process_history_capability_supported(env) -> None:
    def add_marker(
        ctx: RunContext[Any],
        messages: list[ModelMessage],
    ) -> list[ModelMessage]:
        del ctx
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
