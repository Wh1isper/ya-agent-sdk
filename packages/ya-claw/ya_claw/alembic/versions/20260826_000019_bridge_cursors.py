"""add durable bridge polling cursors

Revision ID: 20260826_000019
Revises: 20260819_000018
Create Date: 2026-08-26 08:45:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260826_000019"
down_revision = "20260819_000018"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "bridge_cursors",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("adapter", sa.String(length=32), nullable=False),
        sa.Column("tenant_key", sa.String(length=255), nullable=False),
        sa.Column("cursor_key", sa.String(length=255), nullable=False),
        sa.Column("cursor_value", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("adapter", "tenant_key", "cursor_key", name="uq_bridge_cursors_scope"),
    )


def downgrade() -> None:
    op.drop_table("bridge_cursors")
