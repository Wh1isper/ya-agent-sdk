"""Typed product records for YAACLI durable sessions."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator
from ya_agent_sdk.subagents import SubagentPlanDescriptor


def utc_now() -> datetime:
    """Return an aware UTC timestamp."""
    return datetime.now(UTC)


class SessionStatus(StrEnum):
    active = "active"
    tombstoned = "tombstoned"


class LogicalRunStatus(StrEnum):
    pending = "pending"
    running = "running"
    suspended = "suspended"
    cancelling = "cancelling"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"
    interrupted = "interrupted"


class InputState(StrEnum):
    accepted = "accepted"
    enqueued = "enqueued"
    applied = "applied"
    rejected = "rejected"


class InputPriority(StrEnum):
    asap = "asap"
    when_idle = "when_idle"


class ActionState(StrEnum):
    pending = "pending"
    resolved = "resolved"
    timed_out = "timed_out"
    cancelled = "cancelled"
    consumed = "consumed"


class SessionRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    workspace_ref: str
    status: SessionStatus
    head_revision_id: str | None = None
    created_at: datetime
    updated_at: datetime
    tombstoned_at: datetime | None = None


class SessionSummary(BaseModel):
    """Frontend-neutral projection of one active durable session."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    workspace_ref: str
    status: SessionStatus
    head_revision_id: str | None = None
    created_at: datetime
    updated_at: datetime
    input_preview: str | None = None
    output_preview: str | None = None
    message_count: int = 0
    display_event_count: int = 0
    model: str | None = None
    model_profile_id: str | None = None


class RevisionRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    revision_id: str
    session_id: str
    logical_run_id: str
    commit_kind: str
    parent_revision_id: str | None = None
    message_history: list[JsonValue] = Field(default_factory=list)
    resumable_state: dict[str, JsonValue] = Field(default_factory=dict)
    input_ledger: dict[str, JsonValue] = Field(default_factory=dict)
    display_projection: list[JsonValue] = Field(default_factory=list)
    usage: dict[str, JsonValue] = Field(default_factory=dict)
    terminal: dict[str, JsonValue] = Field(default_factory=dict)
    created_at: datetime


class LogicalRunRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    logical_run_id: str
    session_id: str
    execution_id: str
    expected_head_revision_id: str | None = None
    model: str | None = None
    model_profile_id: str | None = None
    status: LogicalRunStatus
    cancellation_reason: str | None = None
    pending_action_batch_id: str | None = None
    created_at: datetime
    updated_at: datetime

    @property
    def terminal(self) -> bool:
        return self.status in {
            LogicalRunStatus.completed,
            LogicalRunStatus.failed,
            LogicalRunStatus.cancelled,
            LogicalRunStatus.interrupted,
        }


class ExecutionRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    execution_id: str
    logical_run_id: str
    status: LogicalRunStatus
    created_at: datetime
    updated_at: datetime


class InputRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_id: str
    logical_run_id: str
    order_index: int
    idempotency_key: str
    origin: Literal["user", "feature"]
    priority: InputPriority
    content: list[JsonValue]
    state: InputState
    native_enqueue_id: str | None = None
    rejection_reason: str | None = None
    created_at: datetime
    updated_at: datetime


class EventRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_id: str
    session_id: str
    logical_run_id: str | None = None
    sequence: int
    event_type: str
    payload: dict[str, JsonValue]
    created_at: datetime


class ActionItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    action_item_id: str
    batch_id: str
    tool_call_id: str
    decision_kind: Literal["approval", "external_result"]
    request: dict[str, JsonValue]
    state: ActionState
    decision_id: str | None = None
    decision: dict[str, JsonValue] | None = None
    actor: str | None = None
    created_at: datetime
    decided_at: datetime | None = None
    consumed_at: datetime | None = None


class ActionBatch(BaseModel):
    model_config = ConfigDict(frozen=True)

    batch_id: str
    logical_run_id: str
    state: ActionState
    deadline_at: datetime | None = None
    items: tuple[ActionItem, ...] = ()
    created_at: datetime
    updated_at: datetime


class ChildPlanManifest(BaseModel):
    """Exact active child plans required to reconstruct one root runtime."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal[1] = 1
    active_routes: dict[str, str] = Field(default_factory=dict)
    descriptors: tuple[SubagentPlanDescriptor, ...] = ()

    @model_validator(mode="after")
    def _validate_active_descriptors(self) -> ChildPlanManifest:
        descriptors_by_id = {descriptor.descriptor_id: descriptor for descriptor in self.descriptors}
        if len(descriptors_by_id) != len(self.descriptors):
            raise ValueError("Child plan manifest descriptors must be unique")
        missing = sorted(set(self.active_routes.values()) - descriptors_by_id.keys())
        if missing:
            raise ValueError(f"Active child routes reference missing descriptors: {missing}")
        mismatched = sorted(
            route
            for route, descriptor_id in self.active_routes.items()
            if descriptors_by_id[descriptor_id].spec.route != route
        )
        if mismatched:
            raise ValueError(f"Active child routes reference descriptors for different routes: {mismatched}")
        return self


class StartRunRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    expected_head_revision_id: str | None = None
    idempotency_key: str
    initial_content: list[JsonValue]
    model: str | None = None
    model_profile_id: str | None = None


class RevisionPayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    message_history: list[JsonValue] = Field(default_factory=list)
    resumable_state: dict[str, JsonValue] = Field(default_factory=dict)
    input_ledger: dict[str, JsonValue] = Field(default_factory=dict)
    display_projection: list[JsonValue] = Field(default_factory=list)
    usage: dict[str, JsonValue] = Field(default_factory=dict)
    terminal: dict[str, JsonValue] = Field(default_factory=dict)


class ExecutionCheckpointRecord(BaseModel):
    """Latest canonical SDK boundary persisted without publishing session head."""

    model_config = ConfigDict(frozen=True)

    execution_id: str
    logical_run_id: str
    segment_index: int = Field(ge=0)
    segment_status: Literal["completed", "suspended"]
    payload: RevisionPayload
    deferred_requests: dict[str, JsonValue] | None = None
    created_at: datetime
    updated_at: datetime


JsonDict = dict[str, Any]
