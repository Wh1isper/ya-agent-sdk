"""session list indexes

Revision ID: 20260712_000011
Revises: 20260602_000010
Create Date: 2026-07-12 07:18:00.000000

"""

from __future__ import annotations

from alembic import op

revision = "20260712_000011"
down_revision = "20260602_000010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "ix_sessions_type_updated_id",
        "sessions",
        ["session_type", "updated_at", "id"],
        unique=False,
    )
    op.drop_index("ix_sessions_session_type_updated", table_name="sessions")
    op.create_index(
        "ix_runs_session_status_sequence",
        "runs",
        ["session_id", "status", "sequence_no"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_runs_session_status_sequence", table_name="runs")
    op.create_index(
        "ix_sessions_session_type_updated",
        "sessions",
        ["session_type", "updated_at"],
        unique=False,
    )
    op.drop_index("ix_sessions_type_updated_id", table_name="sessions")
