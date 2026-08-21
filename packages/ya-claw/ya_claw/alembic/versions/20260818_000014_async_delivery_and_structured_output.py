"""make async completion delivery recoverable and preserve structured output

Revision ID: 20260818_000014
Revises: 20260818_000013
Create Date: 2026-08-18 10:30:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260818_000014"
down_revision = "20260818_000013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "runs",
        sa.Column("source_delivery_id", sa.String(length=255), nullable=True),
    )
    op.add_column("runs", sa.Column("output_json", sa.JSON(), nullable=True))
    op.create_index(
        "uq_runs_source_delivery",
        "runs",
        ["source_delivery_id"],
        unique=True,
    )
    op.add_column(
        "session_async_tasks",
        sa.Column("delivery_run_id", sa.String(length=32), nullable=True),
    )
    op.create_index(
        "ix_session_async_tasks_delivery_run",
        "session_async_tasks",
        ["delivery_run_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_session_async_tasks_delivery_run",
        table_name="session_async_tasks",
    )
    op.drop_column("session_async_tasks", "delivery_run_id")
    op.drop_index("uq_runs_source_delivery", table_name="runs")
    op.drop_column("runs", "output_json")
    op.drop_column("runs", "source_delivery_id")
