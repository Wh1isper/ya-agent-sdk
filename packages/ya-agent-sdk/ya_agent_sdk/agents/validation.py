"""Fail-fast native AgentSpec construction validation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic_ai import Agent, AgentSpec
from pydantic_ai.agent import _capabilities_from_spec, _validate_spec
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models.test import TestModel


def prepare_agent_spec_capabilities(
    spec: AgentSpec,
    *,
    deps_type: type[Any],
    custom_capability_types: Sequence[type[AbstractCapability[Any]]] = (),
    capabilities: Sequence[AbstractCapability[Any]] = (),
) -> tuple[AgentSpec, tuple[AbstractCapability[Any], ...]]:
    """Materialize native specs once and reject singleton collisions before merging.

    Pydantic AI 2.39 merges capabilities with class-default IDs during construction.
    YA requires exact, non-overlapping grants instead. Keep native registry/template
    handling in this adapter, then pass the same instances to native construction.
    """
    validated_spec, template_context = _validate_spec(spec, deps_type)
    combined = (
        *_capabilities_from_spec(validated_spec, custom_capability_types, template_context),
        *capabilities,
    )
    seen_ids: set[str] = set()

    def visit(capability: AbstractCapability[Any]) -> None:
        if capability.id is None:
            return
        if capability.id in seen_ids:
            raise ValueError(
                f"Capability id {capability.id!r} is used by multiple capabilities. "
                "Capability ids must be unique within a plan."
            )
        seen_ids.add(capability.id)

    for capability in combined:
        capability.apply(visit)
    return validated_spec.model_copy(update={"capabilities": []}), combined


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
        prepared_spec, prepared_capabilities = prepare_agent_spec_capabilities(
            spec,
            deps_type=deps_type,
            custom_capability_types=custom_capability_types,
            capabilities=capabilities,
        )
        Agent.from_spec(
            prepared_spec,
            deps_type=deps_type,
            custom_capability_types=custom_capability_types,
            model=TestModel(call_tools=[]),
            capabilities=prepared_capabilities,
        )
    except Exception as exc:
        raise ValueError(f"Invalid {error_context}: {exc}") from exc
