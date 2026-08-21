"""add durable run input inbox

Revision ID: 20260818_000013
Revises: 20260818_000012
Create Date: 2026-08-18 09:45:00.000000

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260818_000013"
down_revision = "20260818_000012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "run_input_inbox",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column("delivery_key", sa.String(length=255), nullable=False),
        sa.Column("origin", sa.String(length=32), nullable=False, server_default="user"),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="accepted"),
        sa.Column("input_parts", sa.JSON(), nullable=False),
        sa.Column("sdk_input_id", sa.String(length=255), nullable=True),
        sa.Column("enqueue_id", sa.String(length=255), nullable=True),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("applied_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('accepted', 'enqueued', 'applied', 'rejected')",
            name="ck_run_input_inbox_status",
        ),
        sa.CheckConstraint(
            "origin IN ('user', 'feature')",
            name="ck_run_input_inbox_origin",
        ),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "delivery_key", name="uq_run_input_inbox_delivery"),
    )
    op.create_index(
        "ix_run_input_inbox_run_status",
        "run_input_inbox",
        ["run_id", "status", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_run_input_inbox_sdk_input",
        "run_input_inbox",
        ["run_id", "sdk_input_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_run_input_inbox_sdk_input", table_name="run_input_inbox")
    op.drop_index("ix_run_input_inbox_run_status", table_name="run_input_inbox")
    op.drop_table("run_input_inbox")
