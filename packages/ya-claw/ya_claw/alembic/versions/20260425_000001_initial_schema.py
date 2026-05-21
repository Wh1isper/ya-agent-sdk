from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260425_000001"
down_revision = None
branch_labels = None
depends_on = None

_ALLOWED_RUN_STATUSES = ("queued", "running", "completed", "failed", "cancelled")
_ALLOWED_BRIDGE_EVENT_STATUSES = ("received", "queued", "submitted", "duplicate", "failed")


def upgrade() -> None:
    op.create_table(
        "profiles",
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("model", sa.String(length=255), nullable=False),
        sa.Column("model_settings_preset", sa.String(length=255), nullable=True),
        sa.Column("model_settings_override", sa.JSON(), nullable=True),
        sa.Column("model_config_preset", sa.String(length=255), nullable=True),
        sa.Column("model_config_override", sa.JSON(), nullable=True),
        sa.Column("system_prompt", sa.Text(), nullable=True),
        sa.Column("builtin_toolsets", sa.JSON(), nullable=False),
        sa.Column("subagents", sa.JSON(), nullable=False),
        sa.Column("include_builtin_subagents", sa.Boolean(), nullable=False),
        sa.Column("unified_subagents", sa.Boolean(), nullable=False),
        sa.Column("need_user_approve_tools", sa.JSON(), nullable=False),
        sa.Column("need_user_approve_mcps", sa.JSON(), nullable=False),
        sa.Column("enabled_mcps", sa.JSON(), nullable=False),
        sa.Column("disabled_mcps", sa.JSON(), nullable=False),
        sa.Column("mcp_servers", sa.JSON(), nullable=False),
        sa.Column("workspace_backend_hint", sa.String(length=32), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("source_type", sa.String(length=64), nullable=True),
        sa.Column("source_version", sa.String(length=255), nullable=True),
        sa.Column("source_checksum", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("name"),
    )

    op.create_table(
        "sessions",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("parent_session_id", sa.String(length=32), nullable=True),
        sa.Column("profile_name", sa.String(length=255), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("head_run_id", sa.String(length=32), nullable=True),
        sa.Column("head_success_run_id", sa.String(length=32), nullable=True),
        sa.Column("active_run_id", sa.String(length=32), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["parent_session_id"], ["sessions.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "runs",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("session_id", sa.String(length=32), nullable=False),
        sa.Column("sequence_no", sa.Integer(), nullable=False),
        sa.Column("restore_from_run_id", sa.String(length=32), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=True),
        sa.Column("trigger_type", sa.String(length=32), nullable=True),
        sa.Column("profile_name", sa.String(length=255), nullable=True),
        sa.Column("input_parts", sa.JSON(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("output_text", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("termination_reason", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("committed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("claimed_by", sa.String(length=255), nullable=True),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(f"status IN {_ALLOWED_RUN_STATUSES!s}", name="ck_runs_status"),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_runs_session_id", "runs", ["session_id"])
    op.create_index("ix_runs_session_created_at", "runs", ["session_id", "created_at"])
    op.create_index("ix_runs_session_sequence_no", "runs", ["session_id", "sequence_no"], unique=True)

    op.create_table(
        "bridge_conversations",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("adapter", sa.String(length=32), nullable=False),
        sa.Column("tenant_key", sa.String(length=255), nullable=False),
        sa.Column("external_chat_id", sa.String(length=255), nullable=False),
        sa.Column("session_id", sa.String(length=32), nullable=False),
        sa.Column("profile_name", sa.String(length=255), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_event_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("adapter", "tenant_key", "external_chat_id", name="uq_bridge_conversations_chat"),
    )
    op.create_index("ix_bridge_conversations_session_id", "bridge_conversations", ["session_id"])

    op.create_table(
        "bridge_events",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("adapter", sa.String(length=32), nullable=False),
        sa.Column("tenant_key", sa.String(length=255), nullable=False),
        sa.Column("event_id", sa.String(length=255), nullable=False),
        sa.Column("external_message_id", sa.String(length=255), nullable=True),
        sa.Column("external_chat_id", sa.String(length=255), nullable=True),
        sa.Column("conversation_id", sa.String(length=32), nullable=True),
        sa.Column("session_id", sa.String(length=32), nullable=True),
        sa.Column("run_id", sa.String(length=32), nullable=True),
        sa.Column("event_type", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=True),
        sa.Column("raw_event", sa.JSON(), nullable=False),
        sa.Column("normalized_event", sa.JSON(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(f"status IN {_ALLOWED_BRIDGE_EVENT_STATUSES!s}", name="ck_bridge_events_status"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("adapter", "tenant_key", "event_id", name="uq_bridge_events_event"),
        sa.UniqueConstraint("adapter", "tenant_key", "external_message_id", name="uq_bridge_events_message"),
    )
    op.create_index("ix_bridge_events_chat_created_at", "bridge_events", ["external_chat_id", "created_at"])
    op.create_index("ix_bridge_events_session_id", "bridge_events", ["session_id"])
    op.create_index("ix_bridge_events_run_id", "bridge_events", ["run_id"])

    op.create_table(
        "runtime_instances",
        sa.Column("id", sa.String(length=255), nullable=False),
        sa.Column("hostname", sa.String(length=255), nullable=True),
        sa.Column("process_id", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("stopped_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("runtime_instances")
    op.drop_index("ix_bridge_events_run_id", table_name="bridge_events")
    op.drop_index("ix_bridge_events_session_id", table_name="bridge_events")
    op.drop_index("ix_bridge_events_chat_created_at", table_name="bridge_events")
    op.drop_table("bridge_events")
    op.drop_index("ix_bridge_conversations_session_id", table_name="bridge_conversations")
    op.drop_table("bridge_conversations")
    op.drop_index("ix_runs_session_sequence_no", table_name="runs")
    op.drop_index("ix_runs_session_created_at", table_name="runs")
    op.drop_index("ix_runs_session_id", table_name="runs")
    op.drop_table("runs")
    op.drop_table("sessions")
    op.drop_table("profiles")
