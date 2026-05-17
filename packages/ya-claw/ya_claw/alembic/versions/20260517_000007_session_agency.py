from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260517_000007"
down_revision = "20260510_000006"
branch_labels = None
depends_on = None

_ALLOWED_SESSION_TYPES = ("conversation", "memory", "agency")
_OLD_SESSION_TYPES = ("conversation", "memory")
_ALLOWED_AGENCY_SIGNAL_STATUSES = ("pending", "steered", "submitted", "consumed", "skipped", "failed")


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        with op.batch_alter_table("sessions") as batch_op:
            batch_op.drop_constraint("ck_sessions_session_type", type_="check")
            batch_op.create_check_constraint("ck_sessions_session_type", f"session_type IN {_ALLOWED_SESSION_TYPES!s}")
    else:
        op.drop_constraint("ck_sessions_session_type", "sessions", type_="check")
        op.create_check_constraint(
            "ck_sessions_session_type",
            "sessions",
            sa.column("session_type").in_(_ALLOWED_SESSION_TYPES),
        )

    op.create_table(
        "session_agency_states",
        sa.Column("source_session_id", sa.String(length=32), nullable=False),
        sa.Column("agency_session_id", sa.String(length=32), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("last_observed_sequence_no", sa.Integer(), nullable=False),
        sa.Column("episode_count", sa.Integer(), nullable=False),
        sa.Column("pending_signal_count", sa.Integer(), nullable=False),
        sa.Column("last_agency_run_id", sa.String(length=32), nullable=True),
        sa.Column("last_agency_reason", sa.String(length=64), nullable=True),
        sa.Column("last_action_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cooldown_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["source_session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("source_session_id"),
    )

    op.create_table(
        "agency_signals",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("source_session_id", sa.String(length=32), nullable=False),
        sa.Column("agency_session_id", sa.String(length=32), nullable=True),
        sa.Column("reason", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False),
        sa.Column("dedupe_key", sa.String(length=255), nullable=False),
        sa.Column("source_run_ids", sa.JSON(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("run_id", sa.String(length=32), nullable=True),
        sa.CheckConstraint(f"status IN {_ALLOWED_AGENCY_SIGNAL_STATUSES!s}", name="ck_agency_signals_status"),
        sa.ForeignKeyConstraint(["source_session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source_session_id", "dedupe_key", name="uq_agency_signals_source_dedupe"),
    )
    op.create_index("ix_agency_signals_status_created", "agency_signals", ["status", "created_at"])
    op.create_index("ix_agency_signals_source_status", "agency_signals", ["source_session_id", "status"])
    op.create_index("ix_agency_signals_run", "agency_signals", ["run_id"])
    op.create_index("ix_sessions_type_source_unique", "sessions", ["session_type", "source_session_id"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_sessions_type_source_unique", table_name="sessions")
    op.drop_index("ix_agency_signals_run", table_name="agency_signals")
    op.drop_index("ix_agency_signals_source_status", table_name="agency_signals")
    op.drop_index("ix_agency_signals_status_created", table_name="agency_signals")
    op.drop_table("agency_signals")
    op.drop_table("session_agency_states")

    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        with op.batch_alter_table("sessions") as batch_op:
            batch_op.drop_constraint("ck_sessions_session_type", type_="check")
            batch_op.create_check_constraint("ck_sessions_session_type", f"session_type IN {_OLD_SESSION_TYPES!s}")
    else:
        op.drop_constraint("ck_sessions_session_type", "sessions", type_="check")
        op.create_check_constraint(
            "ck_sessions_session_type",
            "sessions",
            sa.column("session_type").in_(_OLD_SESSION_TYPES),
        )
