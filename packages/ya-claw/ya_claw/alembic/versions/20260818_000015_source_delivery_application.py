"""track model-visible application of continuation source deliveries

Revision ID: 20260818_000015
Revises: 20260818_000014
Create Date: 2026-08-18 11:50:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260818_000015"
down_revision = "20260818_000014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "runs",
        sa.Column("source_delivery_applied_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("runs", "source_delivery_applied_at")
