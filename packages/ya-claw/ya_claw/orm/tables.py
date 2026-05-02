from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ya_claw.orm.base import Base

_ALLOWED_RUN_STATUSES = ("queued", "running", "completed", "failed", "cancelled")
_ALLOWED_SESSION_TYPES = ("conversation", "memory")
_ALLOWED_BRIDGE_EVENT_STATUSES = ("received", "queued", "submitted", "steered", "duplicate", "failed")
_ALLOWED_SCHEDULE_STATUSES = ("active", "paused", "completed", "deleted")
_ALLOWED_SCHEDULE_TRIGGER_KINDS = ("cron", "once")
_ALLOWED_SCHEDULE_EXECUTION_MODES = ("continue_session", "fork_session", "isolate_session")
_ALLOWED_SCHEDULE_ACTIVE_POLICIES = ("steer", "queue")
_ALLOWED_SCHEDULE_FIRE_STATUSES = ("pending", "submitted", "steered", "skipped", "failed")
_ALLOWED_HEARTBEAT_FIRE_STATUSES = ("pending", "submitted", "skipped", "failed")


def utc_now() -> datetime:
    return datetime.now(UTC)


class ProfileRecord(Base):
    __tablename__ = "profiles"

    name: Mapped[str] = mapped_column(String(255), primary_key=True)
    model: Mapped[str] = mapped_column(String(255), nullable=False)
    model_settings_preset: Mapped[str | None] = mapped_column(String(255), nullable=True)
    model_settings_override: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    model_config_preset: Mapped[str | None] = mapped_column(String(255), nullable=True)
    model_config_override: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    builtin_toolsets: Mapped[list[str]] = mapped_column(JSON, default=list)
    subagents: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    include_builtin_subagents: Mapped[bool] = mapped_column(Boolean, default=False)
    unified_subagents: Mapped[bool] = mapped_column(Boolean, default=False)
    need_user_approve_tools: Mapped[list[str]] = mapped_column(JSON, default=list)
    need_user_approve_mcps: Mapped[list[str]] = mapped_column(JSON, default=list)
    enabled_mcps: Mapped[list[str]] = mapped_column(JSON, default=list)
    disabled_mcps: Mapped[list[str]] = mapped_column(JSON, default=list)
    mcp_servers: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    workspace_backend_hint: Mapped[str | None] = mapped_column(String(32), nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    source_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source_version: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_checksum: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class SessionRecord(Base):
    __tablename__ = "sessions"
    __table_args__ = (
        CheckConstraint(
            f"session_type IN {_ALLOWED_SESSION_TYPES!s}",
            name="ck_sessions_session_type",
        ),
        Index("ix_sessions_session_type_updated", "session_type", "updated_at"),
        Index("ix_sessions_source_session", "source_session_id"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    parent_session_id: Mapped[str | None] = mapped_column(
        ForeignKey("sessions.id", ondelete="SET NULL"),
        nullable=True,
    )
    profile_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    session_type: Mapped[str] = mapped_column(String(32), default="conversation")
    source_session_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    session_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    head_run_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    head_success_run_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    active_run_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    parent: Mapped[SessionRecord | None] = relationship(remote_side=lambda: [SessionRecord.id])
    runs: Mapped[list[RunRecord]] = relationship(back_populates="session", cascade="all, delete-orphan")


class RunRecord(Base):
    __tablename__ = "runs"
    __table_args__ = (
        CheckConstraint(
            f"status IN {_ALLOWED_RUN_STATUSES!s}",
            name="ck_runs_status",
        ),
        Index("ix_runs_session_created_at", "session_id", "created_at"),
        Index("ix_runs_session_sequence_no", "session_id", "sequence_no", unique=True),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"), index=True)
    sequence_no: Mapped[int] = mapped_column(Integer, nullable=False)
    restore_from_run_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="queued")
    trigger_type: Mapped[str] = mapped_column(String(32), default="api")
    profile_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    input_parts: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    run_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    output_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    termination_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    committed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    claimed_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    session: Mapped[SessionRecord] = relationship(back_populates="runs")


class SessionMemoryStateRecord(Base):
    __tablename__ = "session_memory_states"

    source_session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"),
        primary_key=True,
    )
    memory_session_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    last_extracted_sequence_no: Mapped[int] = mapped_column(Integer, default=0)
    turns_since_extract: Mapped[int] = mapped_column(Integer, default=0)
    extract_count: Mapped[int] = mapped_column(Integer, default=0)
    extracts_since_summary: Mapped[int] = mapped_column(Integer, default=0)
    pending_extract: Mapped[bool] = mapped_column(Boolean, default=False)
    pending_summary: Mapped[bool] = mapped_column(Boolean, default=False)
    last_extract_run_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    last_summary_run_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    memory_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class ScheduleRecord(Base):
    __tablename__ = "schedules"
    __table_args__ = (
        CheckConstraint(
            f"status IN {_ALLOWED_SCHEDULE_STATUSES!s}",
            name="ck_schedules_status",
        ),
        CheckConstraint(
            f"trigger_kind IN {_ALLOWED_SCHEDULE_TRIGGER_KINDS!s}",
            name="ck_schedules_trigger_kind",
        ),
        CheckConstraint(
            f"execution_mode IN {_ALLOWED_SCHEDULE_EXECUTION_MODES!s}",
            name="ck_schedules_execution_mode",
        ),
        CheckConstraint(
            f"on_active IN {_ALLOWED_SCHEDULE_ACTIVE_POLICIES!s}",
            name="ck_schedules_on_active",
        ),
        Index("ix_schedules_due", "status", "next_fire_at"),
        Index("ix_schedules_trigger_kind", "trigger_kind"),
        Index("ix_schedules_owner_session", "owner_session_id"),
        Index("ix_schedules_target_session", "target_session_id"),
        Index("ix_schedules_source_session", "source_session_id"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="active")
    owner_kind: Mapped[str] = mapped_column(String(32), default="api")
    owner_session_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    owner_run_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    profile_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    trigger_kind: Mapped[str] = mapped_column(String(32), default="cron")
    cron_expr: Mapped[str | None] = mapped_column(String(255), nullable=True)
    run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    timezone: Mapped[str] = mapped_column(String(64), default="UTC")
    next_fire_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    execution_mode: Mapped[str] = mapped_column(String(32), default="isolate_session")
    target_session_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    source_session_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    on_active: Mapped[str] = mapped_column(String(32), default="queue")
    input_parts_template: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    schedule_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    last_fire_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_fire_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    last_session_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    last_run_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    fire_count: Mapped[int] = mapped_column(Integer, default=0)
    failure_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class ScheduleFireRecord(Base):
    __tablename__ = "schedule_fires"
    __table_args__ = (
        CheckConstraint(
            f"status IN {_ALLOWED_SCHEDULE_FIRE_STATUSES!s}",
            name="ck_schedule_fires_status",
        ),
        UniqueConstraint("schedule_id", "dedupe_key", name="uq_schedule_fires_dedupe"),
        Index("ix_schedule_fires_schedule_created", "schedule_id", "created_at"),
        Index("ix_schedule_fires_status_scheduled", "status", "scheduled_at"),
        Index("ix_schedule_fires_run", "run_id"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    schedule_id: Mapped[str] = mapped_column(ForeignKey("schedules.id", ondelete="CASCADE"), nullable=False)
    scheduled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    fired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    dedupe_key: Mapped[str] = mapped_column(String(255), nullable=False)
    target_session_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    source_session_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_session_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    run_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    active_run_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    input_parts: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    fire_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class HeartbeatFireRecord(Base):
    __tablename__ = "heartbeat_fires"
    __table_args__ = (
        CheckConstraint(
            f"status IN {_ALLOWED_HEARTBEAT_FIRE_STATUSES!s}",
            name="ck_heartbeat_fires_status",
        ),
        UniqueConstraint("dedupe_key", name="uq_heartbeat_fires_dedupe"),
        Index("ix_heartbeat_fires_status_scheduled", "status", "scheduled_at"),
        Index("ix_heartbeat_fires_run", "run_id"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    scheduled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    fired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    dedupe_key: Mapped[str] = mapped_column(String(255), nullable=False)
    session_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    run_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    fire_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class BridgeConversationRecord(Base):
    __tablename__ = "bridge_conversations"
    __table_args__ = (
        UniqueConstraint("adapter", "tenant_key", "external_chat_id", name="uq_bridge_conversations_chat"),
        Index("ix_bridge_conversations_session_id", "session_id"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    adapter: Mapped[str] = mapped_column(String(32), nullable=False)
    tenant_key: Mapped[str] = mapped_column(String(255), nullable=False, default="default")
    external_chat_id: Mapped[str] = mapped_column(String(255), nullable=False)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    profile_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    conversation_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)
    last_event_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class BridgeEventRecord(Base):
    __tablename__ = "bridge_events"
    __table_args__ = (
        CheckConstraint(
            f"status IN {_ALLOWED_BRIDGE_EVENT_STATUSES!s}",
            name="ck_bridge_events_status",
        ),
        UniqueConstraint("adapter", "tenant_key", "event_id", name="uq_bridge_events_event"),
        UniqueConstraint("adapter", "tenant_key", "external_message_id", name="uq_bridge_events_message"),
        Index("ix_bridge_events_chat_created_at", "external_chat_id", "created_at"),
        Index("ix_bridge_events_session_id", "session_id"),
        Index("ix_bridge_events_run_id", "run_id"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    adapter: Mapped[str] = mapped_column(String(32), nullable=False)
    tenant_key: Mapped[str] = mapped_column(String(255), nullable=False, default="default")
    event_id: Mapped[str] = mapped_column(String(255), nullable=False)
    external_message_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    external_chat_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    conversation_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    session_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    run_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    event_type: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="received")
    raw_event: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    normalized_event: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class RuntimeInstanceRecord(Base):
    __tablename__ = "runtime_instances"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    hostname: Mapped[str | None] = mapped_column(String(255), nullable=True)
    process_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="active")
    instance_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    heartbeat_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    stopped_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
