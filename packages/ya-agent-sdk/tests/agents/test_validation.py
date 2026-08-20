from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from pydantic_ai import AgentSpec
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
from ya_agent_sdk.agents import validate_agent_spec_capabilities
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
