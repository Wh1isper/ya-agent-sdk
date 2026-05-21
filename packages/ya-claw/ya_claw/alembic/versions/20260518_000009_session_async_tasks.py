"""session_async_tasks

Revision ID: 20260518_000009
Revises: 20260517_000008
Create Date: 2026-05-18 11:30:00.000000

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260518_000009"
down_revision = "20260517_000008"
branch_labels = None
depends_on = None

_ALLOWED_SESSION_TYPES = ("conversation", "memory", "agency", "async_task")
_OLD_SESSION_TYPES = ("conversation", "memory", "agency")
_ALLOWED_ASYNC_TASK_STATUSES = ("queued", "running", "completed", "failed", "cancelled")
_ALLOWED_ASYNC_TASK_WAKE_POLICIES = ("steer_or_run", "record_only")


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
        "session_async_tasks",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("parent_session_id", sa.String(length=32), nullable=False),
        sa.Column("parent_run_id", sa.String(length=32), nullable=True),
        sa.Column("parent_agent_id", sa.String(length=255), nullable=False),
        sa.Column("task_session_id", sa.String(length=32), nullable=False),
        sa.Column("task_run_id", sa.String(length=32), nullable=True),
        sa.Column("subagent_name", sa.String(length=255), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("wake_policy", sa.String(length=32), nullable=False),
        sa.Column("input_parts", sa.JSON(), nullable=False),
        sa.Column("result_run_id", sa.String(length=32), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(f"status IN {_ALLOWED_ASYNC_TASK_STATUSES!s}", name="ck_session_async_tasks_status"),
        sa.CheckConstraint(
            f"wake_policy IN {_ALLOWED_ASYNC_TASK_WAKE_POLICIES!s}",
            name="ck_session_async_tasks_wake_policy",
        ),
        sa.ForeignKeyConstraint(["parent_session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["task_session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("parent_session_id", "name", name="uq_session_async_tasks_parent_name"),
    )
    op.create_index("ix_session_async_tasks_parent_status", "session_async_tasks", ["parent_session_id", "status"])
    op.create_index("ix_session_async_tasks_task_session", "session_async_tasks", ["task_session_id"])
    op.create_index("ix_session_async_tasks_task_run", "session_async_tasks", ["task_run_id"])
    op.create_index("ix_session_async_tasks_name", "session_async_tasks", ["parent_session_id", "name"])


def downgrade() -> None:
    op.drop_index("ix_session_async_tasks_name", table_name="session_async_tasks")
    op.drop_index("ix_session_async_tasks_task_run", table_name="session_async_tasks")
    op.drop_index("ix_session_async_tasks_task_session", table_name="session_async_tasks")
    op.drop_index("ix_session_async_tasks_parent_status", table_name="session_async_tasks")
    op.drop_table("session_async_tasks")

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
