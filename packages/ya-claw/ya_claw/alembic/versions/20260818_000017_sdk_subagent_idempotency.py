"""make SDK subagent spawn idempotent and persist input admission

Revision ID: 20260818_000017
Revises: 20260818_000016
Create Date: 2026-08-18 23:15:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260818_000017"
down_revision = "20260818_000016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "session_async_tasks",
        sa.Column("sdk_owner_scope_id", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "session_async_tasks",
        sa.Column("sdk_idempotency_key", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "session_async_tasks",
        sa.Column("sdk_intent_fingerprint", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "session_async_tasks",
        sa.Column("sdk_input_state", sa.String(length=32), nullable=True),
    )
    with op.batch_alter_table("session_async_tasks") as batch_op:
        batch_op.create_check_constraint(
            "ck_session_async_tasks_sdk_identity_complete",
            "(sdk_owner_scope_id IS NULL AND sdk_idempotency_key IS NULL "
            "AND sdk_intent_fingerprint IS NULL AND sdk_input_state IS NULL) OR "
            "(sdk_owner_scope_id IS NOT NULL AND sdk_idempotency_key IS NOT NULL "
            "AND sdk_intent_fingerprint IS NOT NULL AND sdk_input_state IS NOT NULL)",
        )
        batch_op.create_check_constraint(
            "ck_session_async_tasks_sdk_owner",
            "sdk_owner_scope_id IS NULL OR sdk_owner_scope_id = parent_session_id",
        )
        batch_op.create_check_constraint(
            "ck_session_async_tasks_sdk_input_state",
            "sdk_input_state IS NULL OR sdk_input_state IN ('accepted', 'applied', 'rejected')",
        )
        batch_op.create_unique_constraint(
            "uq_session_async_tasks_sdk_idempotency",
            ["sdk_owner_scope_id", "sdk_idempotency_key"],
        )


def downgrade() -> None:
    with op.batch_alter_table("session_async_tasks") as batch_op:
        batch_op.drop_constraint(
            "uq_session_async_tasks_sdk_idempotency",
            type_="unique",
        )
        batch_op.drop_constraint(
            "ck_session_async_tasks_sdk_input_state",
            type_="check",
        )
        batch_op.drop_constraint(
            "ck_session_async_tasks_sdk_owner",
            type_="check",
        )
        batch_op.drop_constraint(
            "ck_session_async_tasks_sdk_identity_complete",
            type_="check",
        )
    op.drop_column("session_async_tasks", "sdk_input_state")
    op.drop_column("session_async_tasks", "sdk_intent_fingerprint")
    op.drop_column("session_async_tasks", "sdk_idempotency_key")
    op.drop_column("session_async_tasks", "sdk_owner_scope_id")
