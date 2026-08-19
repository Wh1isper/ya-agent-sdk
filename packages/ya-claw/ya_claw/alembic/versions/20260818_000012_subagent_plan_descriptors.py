"""persist portable subagent plan descriptors

Revision ID: 20260818_000012
Revises: 20260712_000011
Create Date: 2026-08-18 09:00:00.000000

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260818_000012"
down_revision = "20260712_000011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("session_async_tasks", sa.Column("subagent_spec_version", sa.String(length=64), nullable=True))
    op.add_column("session_async_tasks", sa.Column("agent_spec_hash", sa.String(length=255), nullable=True))
    op.add_column("session_async_tasks", sa.Column("plan_fingerprint", sa.String(length=255), nullable=True))
    op.add_column("session_async_tasks", sa.Column("plan_descriptor_ref", sa.String(length=255), nullable=True))
    op.add_column("session_async_tasks", sa.Column("plan_descriptor", sa.JSON(), nullable=True))
    op.add_column("session_async_tasks", sa.Column("delivery_status", sa.String(length=32), nullable=True))
    op.add_column("session_async_tasks", sa.Column("delivery_id", sa.String(length=255), nullable=True))
    op.create_index(
        "ix_session_async_tasks_delivery",
        "session_async_tasks",
        ["delivery_status"],
        unique=False,
    )
    op.execute(
        sa.text(
            "UPDATE runs SET status = 'failed', "
            "error_message = 'Legacy async task has no immutable 2.0 plan descriptor.', "
            "finished_at = CURRENT_TIMESTAMP "
            "WHERE status IN ('queued', 'running') AND id IN ("
            "SELECT task_run_id FROM session_async_tasks WHERE task_run_id IS NOT NULL"
            ")"
        )
    )
    op.execute(
        sa.text(
            "UPDATE sessions SET active_run_id = NULL "
            "WHERE active_run_id IN ("
            "SELECT task_run_id FROM session_async_tasks WHERE task_run_id IS NOT NULL"
            ")"
        )
    )
    op.execute(
        sa.text(
            "UPDATE session_async_tasks SET status = 'failed', delivery_status = 'rejected', "
            "error_message = 'Legacy async task has no immutable 2.0 plan descriptor.', "
            "completed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP "
            "WHERE status IN ('queued', 'running')"
        )
    )


def downgrade() -> None:
    op.drop_index("ix_session_async_tasks_delivery", table_name="session_async_tasks")
    op.drop_column("session_async_tasks", "delivery_id")
    op.drop_column("session_async_tasks", "delivery_status")
    op.drop_column("session_async_tasks", "plan_descriptor")
    op.drop_column("session_async_tasks", "plan_descriptor_ref")
    op.drop_column("session_async_tasks", "plan_fingerprint")
    op.drop_column("session_async_tasks", "agent_spec_hash")
    op.drop_column("session_async_tasks", "subagent_spec_version")
