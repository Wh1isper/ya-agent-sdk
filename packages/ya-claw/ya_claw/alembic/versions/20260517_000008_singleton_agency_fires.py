"""singleton_agency_fires

Revision ID: 20260517_000008
Revises: 20260517_000007
Create Date: 2026-05-17 22:45:33.565091

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "20260517_000008"
down_revision = "20260517_000007"
branch_labels = None
depends_on = None

_ALLOWED_AGENCY_FIRE_STATUSES = ("pending", "submitted", "steered", "merged", "consumed", "skipped", "failed")


def upgrade() -> None:
    op.drop_index("ix_agency_signals_run", table_name="agency_signals")
    op.drop_index("ix_agency_signals_source_status", table_name="agency_signals")
    op.drop_index("ix_agency_signals_status_created", table_name="agency_signals")
    op.drop_table("agency_signals")
    op.drop_table("session_agency_states")

    op.create_table(
        "agency_fires",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("scheduled_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("fired_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("dedupe_key", sa.String(length=255), nullable=False),
        sa.Column("source_session_id", sa.String(length=32), nullable=True),
        sa.Column("source_run_id", sa.String(length=32), nullable=True),
        sa.Column("agency_session_id", sa.String(length=32), nullable=True),
        sa.Column("run_id", sa.String(length=32), nullable=True),
        sa.Column("active_run_id", sa.String(length=32), nullable=True),
        sa.Column("priority", sa.Integer(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(f"status IN {_ALLOWED_AGENCY_FIRE_STATUSES!s}", name="ck_agency_fires_status"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("dedupe_key", name="uq_agency_fires_dedupe"),
    )
    op.create_index("ix_agency_fires_status_scheduled", "agency_fires", ["status", "scheduled_at"])
    op.create_index("ix_agency_fires_kind_created", "agency_fires", ["kind", "created_at"])
    op.create_index("ix_agency_fires_run", "agency_fires", ["run_id"])
    op.create_index("ix_agency_fires_source", "agency_fires", ["source_session_id"])


def downgrade() -> None:
    op.drop_index("ix_agency_fires_source", table_name="agency_fires")
    op.drop_index("ix_agency_fires_run", table_name="agency_fires")
    op.drop_index("ix_agency_fires_kind_created", table_name="agency_fires")
    op.drop_index("ix_agency_fires_status_scheduled", table_name="agency_fires")
    op.drop_table("agency_fires")

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
        sa.CheckConstraint(
            "status IN ('pending', 'steered', 'submitted', 'consumed', 'skipped', 'failed')",
            name="ck_agency_signals_status",
        ),
        sa.ForeignKeyConstraint(["source_session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source_session_id", "dedupe_key", name="uq_agency_signals_source_dedupe"),
    )
    op.create_index("ix_agency_signals_status_created", "agency_signals", ["status", "created_at"])
    op.create_index("ix_agency_signals_source_status", "agency_signals", ["source_session_id", "status"])
    op.create_index("ix_agency_signals_run", "agency_signals", ["run_id"])
