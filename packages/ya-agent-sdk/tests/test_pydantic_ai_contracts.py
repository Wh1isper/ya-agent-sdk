"""Executable contracts for the Pydantic AI primitives used by YA Agent 2.0."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from pydantic_ai import Agent, AgentSpec, RunContext
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import UserPromptPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.test import TestModel
from pydantic_graph import End
from ya_agent_sdk.agents.main import create_agent


@dataclass
class _FirstCapability(AbstractCapability[None]):
    trace: list[str] = field(default_factory=list)

    async def wrap_model_request(
        self,
        ctx: RunContext[None],
        *,
        request_context: ModelRequestContext,
        handler,
    ):
        del ctx
        self.trace.append("first:enter")
        response = await handler(request_context)
        self.trace.append("first:exit")
        return response


@dataclass
class _BeforeLastCapability(AbstractCapability[None]):
    trace: list[str] = field(default_factory=list)

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(wraps=(_LastCapability,))

    async def wrap_model_request(
        self,
        ctx: RunContext[None],
        *,
        request_context: ModelRequestContext,
        handler,
    ):
        del ctx
        self.trace.append("before_last:enter")
        response = await handler(request_context)
        self.trace.append("before_last:exit")
        return response


@dataclass
class _LastCapability(AbstractCapability[None]):
    trace: list[str] = field(default_factory=list)

    async def wrap_model_request(
        self,
        ctx: RunContext[None],
        *,
        request_context: ModelRequestContext,
        handler,
    ):
        del ctx
        self.trace.append("last:enter")
        response = await handler(request_context)
        self.trace.append("last:exit")
        return response


@dataclass
class _CycleLeftCapability(AbstractCapability[None]):
    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(wraps=(_CycleRightCapability,))


@dataclass
class _CycleRightCapability(AbstractCapability[None]):
    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(wraps=(_CycleLeftCapability,))


@dataclass
class _GreetingCapability(AbstractCapability[None]):
    text: str = "hello"

    def get_instructions(self) -> str:
        return self.text


async def test_native_ordering_uses_ready_batch_source_order() -> None:
    trace: list[str] = []
    agent = Agent(
        TestModel(call_tools=[]),
        capabilities=[
            _LastCapability(trace),
            _FirstCapability(trace),
            _BeforeLastCapability(trace),
        ],
    )

    await agent.run("test")

    assert trace == [
        "first:enter",
        "before_last:enter",
        "last:enter",
        "last:exit",
        "before_last:exit",
        "first:exit",
    ]


def test_native_ordering_reports_cycles() -> None:
    with pytest.raises(UserError, match="Circular ordering constraints"):
        Agent(
            TestModel(call_tools=[]),
            capabilities=[_CycleLeftCapability(), _CycleRightCapability()],
        )


async def test_native_enqueue_applies_structured_input_before_terminal() -> None:
    agent = Agent(TestModel(call_tools=[]))

    async with agent.iter("first") as run:
        enqueue_id = run.enqueue("second", priority="asap")
        node = run.next_node
        while not isinstance(node, End):
            node = await run.next(node)

        user_content = [
            part.content for message in run.all_messages() for part in message.parts if isinstance(part, UserPromptPart)
        ]

    assert enqueue_id is not None
    assert user_content == ["first", "second"]


async def test_create_agent_preserves_native_spec_name_without_override() -> None:
    runtime = create_agent(
        TestModel(call_tools=[]),
        spec=AgentSpec(name="native-worker"),
    )

    async with runtime:
        assert runtime.agent.name == "native-worker"


async def test_create_agent_uses_model_from_spec() -> None:
    runtime = create_agent(spec=AgentSpec(model="test", name="native-worker"))

    async with runtime:
        assert runtime.agent.name == "native-worker"


async def test_create_agent_explicit_name_overrides_native_spec_name() -> None:
    runtime = create_agent(
        TestModel(call_tools=[]),
        spec=AgentSpec(name="native-worker"),
        agent_name="host-worker",
    )

    async with runtime:
        assert runtime.agent.name == "host-worker"


async def test_agent_spec_constructs_selected_custom_capability() -> None:
    model = TestModel(call_tools=[])
    agent = Agent.from_spec(
        {
            "model": "test",
            "capabilities": [{"_GreetingCapability": {"text": "from the declarative spec"}}],
        },
        model=model,
        custom_capability_types=[_GreetingCapability],
    )

    await agent.run("test")

    request_parameters = model.last_model_request_parameters
    assert request_parameters is not None
    assert [part.content for part in request_parameters.instruction_parts] == ["from the declarative spec"]
