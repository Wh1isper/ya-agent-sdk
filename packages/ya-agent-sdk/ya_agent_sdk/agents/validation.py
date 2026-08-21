"""Fail-fast native AgentSpec construction validation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic_ai import Agent, AgentSpec
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models.test import TestModel


def validate_agent_spec_capabilities(
    spec: AgentSpec,
    *,
    deps_type: type[Any],
    custom_capability_types: Sequence[type[AbstractCapability[Any]]] = (),
    capabilities: Sequence[AbstractCapability[Any]] = (),
    error_context: str = "agent capability plan",
) -> None:
    """Construct a throwaway native agent to validate one static capability plan.

    This executes declarative capability ``from_spec()`` factories and validates their
    combined native ordering with the supplied programmatic capability instances. It
    does not enter capability lifecycles or contact a model provider.
    """
    try:
        Agent.from_spec(
            spec,
            deps_type=deps_type,
            custom_capability_types=custom_capability_types,
            model=TestModel(call_tools=[]),
            capabilities=capabilities,
        )
    except Exception as exc:
        raise ValueError(f"Invalid {error_context}: {exc}") from exc
