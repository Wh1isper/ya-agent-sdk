"""align runtime schema and add durable lifecycle projection state

Revision ID: 20260819_000018
Revises: 20260818_000017
Create Date: 2026-08-19 23:00:00.000000
"""

from __future__ import annotations

from typing import Any

import sqlalchemy as sa
from alembic import op

revision = "20260819_000018"
down_revision = "20260818_000017"
branch_labels = None
depends_on = None

_NOW = object()
_DT = sa.DateTime(timezone=True)
_I = sa.Integer()

_REQUIRED_COLUMNS: dict[str, tuple[tuple[str, sa.types.TypeEngine[Any], object], ...]] = {
    "agency_fires": (("created_at", _DT, _NOW), ("updated_at", _DT, _NOW)),
    "bridge_conversations": (("created_at", _DT, _NOW), ("updated_at", _DT, _NOW)),
    "bridge_events": (
        ("status", sa.String(32), "received"),
        ("created_at", _DT, _NOW),
        ("updated_at", _DT, _NOW),
    ),
    "bridge_hitl_messages": (
        ("status", sa.String(32), "active"),
        ("created_at", _DT, _NOW),
        ("updated_at", _DT, _NOW),
    ),
    "heartbeat_fires": (
        ("status", sa.String(32), "pending"),
        ("created_at", _DT, _NOW),
        ("updated_at", _DT, _NOW),
    ),
    "hitl_batches": (
        ("status", sa.String(32), "pending"),
        ("created_at", _DT, _NOW),
        ("updated_at", _DT, _NOW),
    ),
    "hitl_deferred_inputs": (
        ("status", sa.String(32), "pending"),
        ("created_at", _DT, _NOW),
        ("updated_at", _DT, _NOW),
    ),
    "hitl_interactions": (
        ("kind", sa.String(64), "approval"),
        ("status", sa.String(32), "pending"),
        ("created_at", _DT, _NOW),
        ("updated_at", _DT, _NOW),
    ),
    "profiles": (("created_at", _DT, _NOW), ("updated_at", _DT, _NOW)),
    "runs": (
        ("status", sa.String(32), "queued"),
        ("trigger_type", sa.String(32), "api"),
        ("created_at", _DT, _NOW),
    ),
    "runtime_instances": (
        ("status", sa.String(32), "active"),
        ("started_at", _DT, _NOW),
        ("heartbeat_at", _DT, _NOW),
    ),
    "schedule_fires": (
        ("status", sa.String(32), "pending"),
        ("created_at", _DT, _NOW),
        ("updated_at", _DT, _NOW),
    ),
    "schedules": (
        ("status", sa.String(32), "active"),
        ("owner_kind", sa.String(32), "api"),
        ("timezone", sa.String(64), "UTC"),
        ("execution_mode", sa.String(32), "isolate_session"),
        ("on_active", sa.String(32), "queue"),
        ("fire_count", _I, 0),
        ("failure_count", _I, 0),
        ("created_at", _DT, _NOW),
        ("updated_at", _DT, _NOW),
    ),
    "session_async_tasks": (("created_at", _DT, _NOW), ("updated_at", _DT, _NOW)),
    "session_memory_states": (("created_at", _DT, _NOW), ("updated_at", _DT, _NOW)),
    "sessions": (("created_at", _DT, _NOW), ("updated_at", _DT, _NOW)),
    "workflow_definitions": (
        ("status", sa.String(32), "active"),
        ("definition_version", _I, 1),
        ("schema_version", sa.String(64), "ya-claw.workflow.v1"),
        ("owner_kind", sa.String(32), "api"),
        ("scope", sa.String(32), "global"),
        ("created_at", _DT, _NOW),
        ("updated_at", _DT, _NOW),
    ),
    "workflow_events": (("source_kind", sa.String(32), "workflow"), ("created_at", _DT, _NOW)),
    "workflow_node_runs": (
        ("attempt_no", _I, 1),
        ("status", sa.String(32), "pending"),
        ("updated_at", _DT, _NOW),
    ),
    "workflow_runs": (
        ("status", sa.String(32), "queued"),
        ("trigger_kind", sa.String(32), "api"),
        ("created_at", _DT, _NOW),
        ("updated_at", _DT, _NOW),
    ),
}


def _backfill(table_name: str, column_name: str, value: object) -> None:
    column = sa.column(column_name)
    table = sa.table(table_name, column)
    replacement = sa.func.current_timestamp() if value is _NOW else value
    op.execute(table.update().where(column.is_(None)).values({column_name: replacement}))


def _set_required(*, nullable: bool) -> None:
    for table_name, columns in _REQUIRED_COLUMNS.items():
        if not nullable:
            for column_name, _column_type, default in columns:
                _backfill(table_name, column_name, default)
        with op.batch_alter_table(table_name) as batch_op:
            for column_name, column_type, _default in columns:
                batch_op.alter_column(
                    column_name,
                    existing_type=column_type,
                    nullable=nullable,
                )


def upgrade() -> None:
    op.add_column(
        "runs",
        sa.Column("lifecycle_projected_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.execute(
        sa.text(
            "UPDATE runs SET lifecycle_projected_at = "
            "COALESCE(committed_at, finished_at, created_at, CURRENT_TIMESTAMP) "
            "WHERE status = 'completed'"
        )
    )
    op.create_table(
        "memory_lifecycle_effects",
        sa.Column("effect_id", sa.String(255), primary_key=True),
        sa.Column(
            "source_session_id",
            sa.String(32),
            sa.ForeignKey("sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("projection_run_id", sa.String(32), nullable=False),
        sa.Column("effect_kind", sa.String(32), nullable=False),
        sa.Column("source_sequence_no", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_memory_lifecycle_effects_source",
        "memory_lifecycle_effects",
        ["source_session_id", "source_sequence_no"],
        unique=False,
    )
    _set_required(nullable=False)
    op.create_index(
        "ix_runs_lifecycle_projection",
        "runs",
        ["status", "lifecycle_projected_at", "committed_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_runs_lifecycle_projection", table_name="runs")
    _set_required(nullable=True)
    op.drop_index("ix_memory_lifecycle_effects_source", table_name="memory_lifecycle_effects")
    op.drop_table("memory_lifecycle_effects")
    op.drop_column("runs", "lifecycle_projected_at")
