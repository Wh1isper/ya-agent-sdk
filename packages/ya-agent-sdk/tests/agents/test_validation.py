from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import pytest
from pydantic_ai import Agent, AgentSpec
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, CombinedCapability
from pydantic_ai.models.test import TestModel
from ya_agent_sdk.agents import validate_agent_spec_capabilities
from ya_agent_sdk.agents.main import create_agent
from ya_agent_sdk.agents.validation import prepare_agent_spec_capabilities
from ya_agent_sdk.capabilities import ToolTimeoutCapability
from ya_agent_sdk.context import AgentContext


@dataclass
class _RequiredHostCapability(AbstractCapability[AgentContext]):
    pass


@dataclass
class _ValidatedCustomCapability(AbstractCapability[Any]):
    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError("value must be non-negative")

    @classmethod
    def get_serialization_name(cls) -> str:
        return "test.validated_custom"

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(requires=(_RequiredHostCapability,))


def _spec(value: int) -> AgentSpec:
    return AgentSpec.from_dict({"capabilities": [{"name": "test.validated_custom", "arguments": {"value": value}}]})


def test_validate_agent_spec_capabilities_constructs_declarative_and_host_plan() -> None:
    validate_agent_spec_capabilities(
        _spec(1),
        deps_type=AgentContext,
        custom_capability_types=(_ValidatedCustomCapability,),
        capabilities=(_RequiredHostCapability(),),
    )


def test_validate_agent_spec_capabilities_rejects_constructor_values() -> None:
    with pytest.raises(
        ValueError,
        match=r"Invalid test root plan.*value must be non-negative",
    ):
        validate_agent_spec_capabilities(
            _spec(-1),
            deps_type=AgentContext,
            custom_capability_types=(_ValidatedCustomCapability,),
            capabilities=(_RequiredHostCapability(),),
            error_context="test root plan",
        )


def test_validate_agent_spec_capabilities_checks_combined_ordering() -> None:
    with pytest.raises(ValueError, match="Invalid agent capability plan"):
        validate_agent_spec_capabilities(
            _spec(1),
            deps_type=AgentContext,
            custom_capability_types=(_ValidatedCustomCapability,),
        )


@pytest.mark.parametrize("nested", [False, True])
def test_validate_agent_spec_capabilities_rejects_host_collision(nested: bool) -> None:
    host = ToolTimeoutCapability(timeout=600)
    capabilities = [CombinedCapability([host])] if nested else [host]
    spec = AgentSpec(capabilities=[{"ToolTimeoutCapability": {"timeout": 5}}])
    with pytest.raises(ValueError, match=r"Capability id 'tool_timeout'.*multiple"):
        validate_agent_spec_capabilities(
            spec,
            deps_type=AgentContext,
            custom_capability_types=(ToolTimeoutCapability,),
            capabilities=capabilities,
        )


def test_validate_agent_spec_capabilities_rejects_declarative_duplicates() -> None:
    spec = AgentSpec(capabilities=[{"ToolTimeoutCapability": {"timeout": 5}}] * 2)
    with pytest.raises(ValueError, match=r"Capability id 'tool_timeout'.*multiple"):
        validate_agent_spec_capabilities(
            spec,
            deps_type=AgentContext,
            custom_capability_types=(ToolTimeoutCapability,),
        )


@dataclass
class _CountedCapability(AbstractCapability[AgentContext]):
    constructions: ClassVar[int] = 0

    def __post_init__(self) -> None:
        type(self).constructions += 1


def test_prepare_agent_spec_preserves_spec_and_constructs_once() -> None:
    _CountedCapability.constructions = 0
    spec = AgentSpec(capabilities=[{"_CountedCapability": {}}, {"_CountedCapability": {}}])
    before = spec.model_dump()
    host = _RequiredHostCapability()
    prepared, capabilities = prepare_agent_spec_capabilities(
        spec,
        deps_type=AgentContext,
        custom_capability_types=(_CountedCapability,),
        capabilities=(host,),
    )
    Agent.from_spec(prepared, model=TestModel(), deps_type=AgentContext, capabilities=capabilities)
    assert _CountedCapability.constructions == 2  # Anonymous capabilities remain distinct.
    assert len(capabilities) == 3
    assert capabilities[-1] is host
    assert prepared.capabilities == []
    assert spec.model_dump() == before


async def test_runtime_rejects_declarative_host_collision(agent_context: AgentContext) -> None:
    runtime = create_agent(
        TestModel(),
        spec=AgentSpec(capabilities=[{"ToolTimeoutCapability": {"timeout": 5}}]),
        custom_capability_types=(ToolTimeoutCapability,),
        capabilities=[ToolTimeoutCapability(timeout=600)],
        env=agent_context.env,
    )
    with pytest.raises(ValueError, match=r"Capability id 'tool_timeout'.*multiple"):
        async with runtime:
            pytest.fail("Conflicting capabilities must be rejected before runtime entry completes")
