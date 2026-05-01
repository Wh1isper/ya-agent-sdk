from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260501_000004"
down_revision = "20260426_000003"
branch_labels = None
depends_on = None

_ALLOWED_SESSION_TYPES = ("conversation", "memory")


def upgrade() -> None:
    op.add_column("sessions", sa.Column("session_type", sa.String(length=32), nullable=True))
    op.add_column("sessions", sa.Column("source_session_id", sa.String(length=32), nullable=True))
    op.execute("UPDATE sessions SET session_type = 'conversation' WHERE session_type IS NULL")

    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        with op.batch_alter_table("sessions") as batch_op:
            batch_op.alter_column("session_type", existing_type=sa.String(length=32), nullable=False)
            batch_op.create_check_constraint("ck_sessions_session_type", f"session_type IN {_ALLOWED_SESSION_TYPES!s}")
    else:
        op.alter_column("sessions", "session_type", existing_type=sa.String(length=32), nullable=False)
        op.create_check_constraint(
            "ck_sessions_session_type",
            "sessions",
            sa.column("session_type").in_(_ALLOWED_SESSION_TYPES),
        )

    op.create_index("ix_sessions_session_type_updated", "sessions", ["session_type", "updated_at"])
    op.create_index("ix_sessions_source_session", "sessions", ["source_session_id"])

    op.create_table(
        "session_memory_states",
        sa.Column("source_session_id", sa.String(length=32), nullable=False),
        sa.Column("memory_session_id", sa.String(length=32), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("last_extracted_sequence_no", sa.Integer(), nullable=False),
        sa.Column("turns_since_extract", sa.Integer(), nullable=False),
        sa.Column("extract_count", sa.Integer(), nullable=False),
        sa.Column("extracts_since_summary", sa.Integer(), nullable=False),
        sa.Column("pending_extract", sa.Boolean(), nullable=False),
        sa.Column("pending_summary", sa.Boolean(), nullable=False),
        sa.Column("last_extract_run_id", sa.String(length=32), nullable=True),
        sa.Column("last_summary_run_id", sa.String(length=32), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["source_session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("source_session_id"),
    )


def downgrade() -> None:
    op.drop_table("session_memory_states")
    op.drop_index("ix_sessions_source_session", table_name="sessions")
    op.drop_index("ix_sessions_session_type_updated", table_name="sessions")

    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        with op.batch_alter_table("sessions") as batch_op:
            batch_op.drop_constraint("ck_sessions_session_type", type_="check")
            batch_op.drop_column("source_session_id")
            batch_op.drop_column("session_type")
    else:
        op.drop_constraint("ck_sessions_session_type", "sessions", type_="check")
        op.drop_column("sessions", "source_session_id")
        op.drop_column("sessions", "session_type")
