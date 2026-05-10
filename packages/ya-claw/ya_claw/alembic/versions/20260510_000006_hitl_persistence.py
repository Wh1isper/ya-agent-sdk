from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260510_000006"
down_revision = "20260502_000005"
branch_labels = None
depends_on = None

_OLD_ALLOWED_BRIDGE_EVENT_STATUSES = ("received", "queued", "submitted", "steered", "duplicate", "failed")
_NEW_ALLOWED_BRIDGE_EVENT_STATUSES = ("received", "queued", "submitted", "steered", "deferred", "duplicate", "failed")
_ALLOWED_HITL_BATCH_STATUSES = ("pending", "completed", "cancelled")
_ALLOWED_HITL_INTERACTION_STATUSES = ("pending", "approved", "denied")
_ALLOWED_HITL_DEFERRED_INPUT_STATUSES = ("pending", "consumed", "discarded")
_ALLOWED_BRIDGE_HITL_MESSAGE_STATUSES = ("active", "completed", "failed")


def upgrade() -> None:
    _replace_bridge_event_status_constraint(_NEW_ALLOWED_BRIDGE_EVENT_STATUSES)

    op.create_table(
        "hitl_batches",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("session_id", sa.String(length=32), nullable=False),
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=True),
        sa.Column("current_interaction_id", sa.String(length=255), nullable=True),
        sa.Column("deferred_requests", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(f"status IN {_ALLOWED_HITL_BATCH_STATUSES!s}", name="ck_hitl_batches_status"),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_hitl_batches_run_status", "hitl_batches", ["run_id", "status"])
    op.create_index("ix_hitl_batches_session_status", "hitl_batches", ["session_id", "status"])

    op.create_table(
        "hitl_interactions",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("batch_id", sa.String(length=32), nullable=False),
        sa.Column("session_id", sa.String(length=32), nullable=False),
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column("interaction_id", sa.String(length=255), nullable=False),
        sa.Column("tool_call_id", sa.String(length=255), nullable=False),
        sa.Column("tool_name", sa.String(length=255), nullable=True),
        sa.Column("kind", sa.String(length=64), nullable=True),
        sa.Column("sequence_no", sa.Integer(), nullable=False),
        sa.Column("total_count", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("arguments_preview", sa.JSON(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("response", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(f"status IN {_ALLOWED_HITL_INTERACTION_STATUSES!s}", name="ck_hitl_interactions_status"),
        sa.ForeignKeyConstraint(["batch_id"], ["hitl_batches.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("batch_id", "interaction_id", name="uq_hitl_interactions_batch_interaction"),
    )
    op.create_index("ix_hitl_interactions_batch_sequence", "hitl_interactions", ["batch_id", "sequence_no"])
    op.create_index("ix_hitl_interactions_run_status", "hitl_interactions", ["run_id", "status"])

    op.create_table(
        "hitl_deferred_inputs",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("batch_id", sa.String(length=32), nullable=False),
        sa.Column("session_id", sa.String(length=32), nullable=False),
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column("conversation_id", sa.String(length=32), nullable=True),
        sa.Column("adapter", sa.String(length=32), nullable=False),
        sa.Column("tenant_key", sa.String(length=255), nullable=False),
        sa.Column("external_event_id", sa.String(length=255), nullable=False),
        sa.Column("external_message_id", sa.String(length=255), nullable=True),
        sa.Column("external_chat_id", sa.String(length=255), nullable=True),
        sa.Column("sequence_no", sa.Integer(), nullable=False),
        sa.Column("input_parts", sa.JSON(), nullable=False),
        sa.Column("source_metadata", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            f"status IN {_ALLOWED_HITL_DEFERRED_INPUT_STATUSES!s}",
            name="ck_hitl_deferred_inputs_status",
        ),
        sa.ForeignKeyConstraint(["batch_id"], ["hitl_batches.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("adapter", "tenant_key", "external_event_id", name="uq_hitl_deferred_inputs_event"),
        sa.UniqueConstraint("adapter", "tenant_key", "external_message_id", name="uq_hitl_deferred_inputs_message"),
    )
    op.create_index("ix_hitl_deferred_inputs_batch_sequence", "hitl_deferred_inputs", ["batch_id", "sequence_no"])
    op.create_index("ix_hitl_deferred_inputs_run_status", "hitl_deferred_inputs", ["run_id", "status"])

    op.create_table(
        "bridge_hitl_messages",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("adapter", sa.String(length=32), nullable=False),
        sa.Column("tenant_key", sa.String(length=255), nullable=False),
        sa.Column("external_chat_id", sa.String(length=255), nullable=False),
        sa.Column("external_message_id", sa.String(length=255), nullable=False),
        sa.Column("session_id", sa.String(length=32), nullable=False),
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column("batch_id", sa.String(length=32), nullable=True),
        sa.Column("interaction_id", sa.String(length=255), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            f"status IN {_ALLOWED_BRIDGE_HITL_MESSAGE_STATUSES!s}",
            name="ck_bridge_hitl_messages_status",
        ),
        sa.ForeignKeyConstraint(["batch_id"], ["hitl_batches.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("adapter", "tenant_key", "external_message_id", name="uq_bridge_hitl_messages_message"),
    )
    op.create_index("ix_bridge_hitl_messages_batch", "bridge_hitl_messages", ["batch_id"])
    op.create_index(
        "ix_bridge_hitl_messages_chat_status",
        "bridge_hitl_messages",
        ["adapter", "tenant_key", "external_chat_id", "status"],
    )
    op.create_index("ix_bridge_hitl_messages_run", "bridge_hitl_messages", ["run_id"])


def downgrade() -> None:
    op.drop_index("ix_bridge_hitl_messages_run", table_name="bridge_hitl_messages")
    op.drop_index("ix_bridge_hitl_messages_chat_status", table_name="bridge_hitl_messages")
    op.drop_index("ix_bridge_hitl_messages_batch", table_name="bridge_hitl_messages")
    op.drop_table("bridge_hitl_messages")

    op.drop_index("ix_hitl_deferred_inputs_run_status", table_name="hitl_deferred_inputs")
    op.drop_index("ix_hitl_deferred_inputs_batch_sequence", table_name="hitl_deferred_inputs")
    op.drop_table("hitl_deferred_inputs")

    op.drop_index("ix_hitl_interactions_run_status", table_name="hitl_interactions")
    op.drop_index("ix_hitl_interactions_batch_sequence", table_name="hitl_interactions")
    op.drop_table("hitl_interactions")

    op.drop_index("ix_hitl_batches_session_status", table_name="hitl_batches")
    op.drop_index("ix_hitl_batches_run_status", table_name="hitl_batches")
    op.drop_table("hitl_batches")

    op.execute("UPDATE bridge_events SET status = 'steered' WHERE status = 'deferred'")
    _replace_bridge_event_status_constraint(_OLD_ALLOWED_BRIDGE_EVENT_STATUSES)


def _replace_bridge_event_status_constraint(allowed_statuses: tuple[str, ...]) -> None:
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        with op.batch_alter_table("bridge_events") as batch_op:
            batch_op.drop_constraint("ck_bridge_events_status", type_="check")
            batch_op.create_check_constraint("ck_bridge_events_status", f"status IN {allowed_statuses!s}")
        return

    op.drop_constraint("ck_bridge_events_status", "bridge_events", type_="check")
    op.create_check_constraint(
        "ck_bridge_events_status",
        "bridge_events",
        sa.column("status").in_(allowed_statuses),
    )
