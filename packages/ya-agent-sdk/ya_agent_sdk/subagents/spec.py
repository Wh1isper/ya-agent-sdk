"""Portable declarative subagent domain models."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator
from pydantic_ai import AgentSpec
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import UserContent

_ModelT = TypeVar("_ModelT", bound=BaseModel)


class SubagentExecutionMode(StrEnum):
    """How a child execution returns control to its parent."""

    foreground = "foreground"
    background = "background"


class SubagentHistoryPolicy(StrEnum):
    """History granted to a newly spawned child."""

    isolated = "isolated"
    resumable = "resumable"
    parent_snapshot = "parent_snapshot"


class SubagentLinkagePolicy(StrEnum):
    """Logical ownership of a child execution."""

    child = "child"
    detached = "detached"


class SubagentDurability(StrEnum):
    """Minimum execution durability required by a plan."""

    process = "process"
    restart = "restart"


class AgentTemplateContext(BaseModel):
    """Immutable, serializable authority exposed to portable templates."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    template: dict[str, JsonValue] = Field(default_factory=dict)


class SubagentSpec(BaseModel):
    """Versioned YA envelope around one native Pydantic AI ``AgentSpec``."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, extra="forbid")

    schema_version: Literal[1] = 1
    route: str
    agent: AgentSpec
    history: SubagentHistoryPolicy = SubagentHistoryPolicy.isolated
    history_message_limit: int = Field(default=100, ge=1, le=1000)
    host_requirements: tuple[str, ...] = ()
    max_depth: int = Field(default=1, ge=1)
    spawn_targets: tuple[str, ...] = ()
    execution_modes: tuple[SubagentExecutionMode, ...] = (SubagentExecutionMode.foreground,)
    linkage: SubagentLinkagePolicy = SubagentLinkagePolicy.child
    durability: SubagentDurability = SubagentDurability.process

    @field_validator("route")
    @classmethod
    def _validate_route(cls, value: str) -> str:
        route = value.strip()
        if not route:
            raise ValueError("Subagent route cannot be empty")
        if any(char.isspace() for char in route):
            raise ValueError("Subagent route cannot contain whitespace")
        return route

    @field_validator("host_requirements", "spawn_targets")
    @classmethod
    def _deduplicate_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if any(not item for item in normalized):
            raise ValueError("Subagent policy names cannot be empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("Subagent policy names must be unique")
        return normalized

    @field_validator("execution_modes")
    @classmethod
    def _validate_modes(
        cls,
        value: tuple[SubagentExecutionMode, ...],
    ) -> tuple[SubagentExecutionMode, ...]:
        if not value:
            raise ValueError("Subagent execution_modes cannot be empty")
        if len(set(value)) != len(value):
            raise ValueError("Subagent execution_modes must be unique")
        return value

    @model_validator(mode="after")
    def _validate_native_identity(self) -> SubagentSpec:
        if self.agent.name is not None and self.agent.name != self.route:
            raise ValueError(f"Native AgentSpec name {self.agent.name!r} must match route {self.route!r}")
        return self


class SelfForkPolicy(BaseModel):
    """Declarative self-fork policy; never stores live parent capabilities."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, extra="forbid")

    schema_version: Literal[1] = 1
    agent: AgentSpec
    history_message_limit: int = Field(default=100, ge=1, le=1000)
    execution_modes: tuple[SubagentExecutionMode, ...] = (SubagentExecutionMode.foreground,)
    max_depth: int = Field(default=1, ge=1)


class CustomCapabilityAudit(BaseModel):
    """One custom capability type used while resolving a plan."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    serialization_name: str
    provenance: str


class SubagentPlanDescriptor(BaseModel):
    """Portable, content-addressed snapshot of one fully resolved child plan."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, extra="forbid")

    schema_version: Literal[1] = 1
    descriptor_id: str
    fingerprint: str
    spec: SubagentSpec
    normalized_agent_spec: AgentSpec
    template_context: AgentTemplateContext
    custom_capability_audit: tuple[CustomCapabilityAudit, ...] = ()
    injected_policy_ids: tuple[str, ...] = ()
    effective_output_schema: dict[str, Any] | None = None
    supports_deferred_output: bool = True
    restart_durable: bool = False
    initial_history: tuple[dict[str, Any], ...] = ()

    @classmethod
    def from_plan(cls, plan: ResolvedSubagentPlan) -> SubagentPlanDescriptor:
        """Freeze an independent canonical snapshot without live host objects."""
        return cls(
            descriptor_id=plan.descriptor_id,
            fingerprint=plan.fingerprint,
            spec=_clone_model(plan.spec, SubagentSpec),
            normalized_agent_spec=_clone_model(plan.normalized_agent_spec, AgentSpec),
            template_context=_clone_model(plan.template_context, AgentTemplateContext),
            custom_capability_audit=tuple(
                _clone_model(item, CustomCapabilityAudit) for item in plan.custom_capability_audit
            ),
            injected_policy_ids=tuple(plan.injected_policy_ids),
            effective_output_schema=_clone_json(plan.effective_output_schema),
            supports_deferred_output=plan.supports_deferred_output,
            restart_durable=plan.restart_durable,
            initial_history=tuple(_clone_json(item) for item in plan.initial_history),
        )


@dataclass(frozen=True, slots=True)
class ResolvedSubagentPlan:
    """Normalized, validated child plan ready for a host driver."""

    descriptor_id: str
    fingerprint: str
    spec: SubagentSpec
    normalized_agent_spec: AgentSpec
    template_context: AgentTemplateContext
    custom_capability_audit: tuple[CustomCapabilityAudit, ...]
    injected_policy_ids: tuple[str, ...]
    host_capabilities: tuple[AbstractCapability[Any], ...] = field(repr=False)
    effective_output_schema: dict[str, Any] | None = None
    supports_deferred_output: bool = True
    restart_durable: bool = False
    initial_history: tuple[dict[str, Any], ...] = ()

    def to_descriptor(self) -> SubagentPlanDescriptor:
        """Return the portable snapshot used by restart-durable drivers."""
        return SubagentPlanDescriptor.from_plan(self)


def clone_subagent_descriptor(
    descriptor: SubagentPlanDescriptor,
) -> SubagentPlanDescriptor:
    """Clone a descriptor through its canonical portable representation."""
    return _clone_model(descriptor, SubagentPlanDescriptor)


def clone_resolved_subagent_plan(
    plan: ResolvedSubagentPlan,
) -> ResolvedSubagentPlan:
    """Clone all portable plan values while retaining explicit host authorities."""
    descriptor = plan.to_descriptor()
    return ResolvedSubagentPlan(
        descriptor_id=descriptor.descriptor_id,
        fingerprint=descriptor.fingerprint,
        spec=descriptor.spec,
        normalized_agent_spec=descriptor.normalized_agent_spec,
        template_context=descriptor.template_context,
        custom_capability_audit=descriptor.custom_capability_audit,
        injected_policy_ids=descriptor.injected_policy_ids,
        host_capabilities=tuple(copy.deepcopy(plan.host_capabilities)),
        effective_output_schema=descriptor.effective_output_schema,
        supports_deferred_output=descriptor.supports_deferred_output,
        restart_durable=descriptor.restart_durable,
        initial_history=descriptor.initial_history,
    )


def _clone_model(value: BaseModel, model_type: type[_ModelT]) -> _ModelT:
    return model_type.model_validate(value.model_dump(mode="json", by_alias=True))


def _clone_json(value: Any) -> Any:
    if value is None:
        return None
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


class SubagentExecutionState(StrEnum):
    """Lifecycle state persisted for one execution."""

    pending = "pending"
    running = "running"
    suspended = "suspended"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"
    lost = "lost"


class SubagentInputState(StrEnum):
    """Disposition of the execution's initial input."""

    accepted = "accepted"
    applied = "applied"
    rejected = "rejected"


class SubagentDeliveryState(StrEnum):
    """Canonical parent-delivery state for a committed terminal result."""

    not_required = "not_required"
    pending = "pending"
    delivered = "delivered"


class SubagentExecutionRecord(BaseModel):
    """Host-neutral durable record for one scoped child execution."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    schema_version: Literal[3] = 3
    execution_id: str = Field(default_factory=lambda: str(uuid4()))
    root_execution_id: str
    owner_scope_id: str
    idempotency_key: str
    descriptor_id: str
    plan_fingerprint: str
    route: str
    mode: SubagentExecutionMode
    state: SubagentExecutionState = SubagentExecutionState.pending
    input_state: SubagentInputState = SubagentInputState.accepted
    delivery_state: SubagentDeliveryState = SubagentDeliveryState.not_required
    parent_agent_id: str
    parent_logical_run_id: str | None
    parent_runtime_descriptor_id: str | None = None
    child_logical_run_id: str = Field(default_factory=lambda: str(uuid4()))
    depth: int = Field(default=1, ge=1)
    prompt: str | list[UserContent]
    resumed_from: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    output: JsonValue = None
    error: str | None = None
    history: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, JsonValue] = Field(default_factory=dict)
    segment_index: int = Field(default=0, ge=0)
    deferred: dict[str, JsonValue] | None = None
    deferred_results: dict[str, JsonValue] | None = None
    resumable_state: dict[str, JsonValue] = Field(default_factory=dict)
    delivery_logical_run_id: str | None = None
    delivery_input_id: str | None = None

    @property
    def terminal(self) -> bool:
        return self.state in {
            SubagentExecutionState.succeeded,
            SubagentExecutionState.failed,
            SubagentExecutionState.cancelled,
            SubagentExecutionState.lost,
        }


@dataclass(frozen=True, slots=True)
class SubagentHandle:
    """Stable public handle for a child execution."""

    execution_id: str
    route: str
    mode: SubagentExecutionMode


@dataclass(frozen=True, slots=True)
class SubagentDriverOutcome:
    """One committed driver outcome before parent delivery."""

    state: SubagentExecutionState
    input_state: SubagentInputState
    output: JsonValue = None
    error: str | None = None
    history: tuple[dict[str, Any], ...] = ()
    usage: dict[str, JsonValue] = field(default_factory=dict)
    deferred: dict[str, JsonValue] | None = None
    resumable_state: dict[str, JsonValue] = field(default_factory=dict)
