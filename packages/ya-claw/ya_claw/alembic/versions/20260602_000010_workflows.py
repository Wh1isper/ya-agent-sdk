"""workflows

Revision ID: 20260602_000010
Revises: 20260518_000009
Create Date: 2026-06-02 03:30:00.000000

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260602_000010"
down_revision = "20260518_000009"
branch_labels = None
depends_on = None

_ALLOWED_SCHEDULE_EXECUTION_MODES = ("continue_session", "fork_session", "isolate_session", "workflow")
_OLD_SCHEDULE_EXECUTION_MODES = ("continue_session", "fork_session", "isolate_session")
_ALLOWED_WORKFLOW_DEFINITION_STATUSES = ("draft", "active", "archived")
_ALLOWED_WORKFLOW_SCOPES = ("global", "session")
_ALLOWED_WORKFLOW_RUN_STATUSES = ("queued", "running", "waiting", "completed", "failed", "cancelled")
_ALLOWED_WORKFLOW_TRIGGER_KINDS = ("web", "api", "agent", "schedule", "bridge", "system")
_ALLOWED_WORKFLOW_NODE_RUN_STATUSES = (
    "pending",
    "ready",
    "queued",
    "running",
    "waiting",
    "completed",
    "failed",
    "cancelled",
    "skipped",
)


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        with op.batch_alter_table("schedules") as batch_op:
            batch_op.drop_constraint("ck_schedules_execution_mode", type_="check")
            batch_op.add_column(sa.Column("workflow_id", sa.String(length=32), nullable=True))
            batch_op.add_column(sa.Column("workflow_inputs_template", sa.JSON(), nullable=True))
            batch_op.add_column(sa.Column("last_workflow_run_id", sa.String(length=32), nullable=True))
            batch_op.create_check_constraint(
                "ck_schedules_execution_mode",
                f"execution_mode IN {_ALLOWED_SCHEDULE_EXECUTION_MODES!s}",
            )
        with op.batch_alter_table("schedule_fires") as batch_op:
            batch_op.add_column(sa.Column("workflow_run_id", sa.String(length=32), nullable=True))
    else:
        op.drop_constraint("ck_schedules_execution_mode", "schedules", type_="check")
        op.add_column("schedules", sa.Column("workflow_id", sa.String(length=32), nullable=True))
        op.add_column("schedules", sa.Column("workflow_inputs_template", sa.JSON(), nullable=True))
        op.add_column("schedules", sa.Column("last_workflow_run_id", sa.String(length=32), nullable=True))
        op.create_check_constraint(
            "ck_schedules_execution_mode",
            "schedules",
            sa.column("execution_mode").in_(_ALLOWED_SCHEDULE_EXECUTION_MODES),
        )
        op.add_column("schedule_fires", sa.Column("workflow_run_id", sa.String(length=32), nullable=True))

    op.create_table(
        "workflow_definitions",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=True),
        sa.Column("definition_version", sa.Integer(), nullable=True),
        sa.Column("schema_version", sa.String(length=64), nullable=True),
        sa.Column("owner_kind", sa.String(length=32), nullable=True),
        sa.Column("owner_session_id", sa.String(length=32), nullable=True),
        sa.Column("owner_run_id", sa.String(length=32), nullable=True),
        sa.Column("scope", sa.String(length=32), nullable=True),
        sa.Column("tags", sa.JSON(), nullable=False),
        sa.Column("when_to_use", sa.Text(), nullable=True),
        sa.Column("argument_hint", sa.Text(), nullable=True),
        sa.Column("input_schema", sa.JSON(), nullable=False),
        sa.Column("definition", sa.JSON(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            f"status IN {_ALLOWED_WORKFLOW_DEFINITION_STATUSES!s}", name="ck_workflow_definitions_status"
        ),
        sa.CheckConstraint(f"scope IN {_ALLOWED_WORKFLOW_SCOPES!s}", name="ck_workflow_definitions_scope"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_workflow_definitions_status_updated", "workflow_definitions", ["status", "updated_at"])
    op.create_index("ix_workflow_definitions_owner_session", "workflow_definitions", ["owner_session_id"])
    op.create_index("ix_workflow_definitions_scope_status", "workflow_definitions", ["scope", "status"])

    op.create_table(
        "workflow_runs",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("workflow_id", sa.String(length=32), nullable=False),
        sa.Column("workflow_version", sa.Integer(), nullable=False),
        sa.Column("definition_snapshot", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=True),
        sa.Column("trigger_kind", sa.String(length=32), nullable=True),
        sa.Column("supervisor_session_id", sa.String(length=32), nullable=True),
        sa.Column("supervisor_run_id", sa.String(length=32), nullable=True),
        sa.Column("profile_name", sa.String(length=255), nullable=True),
        sa.Column("workspace", sa.JSON(), nullable=True),
        sa.Column("inputs", sa.JSON(), nullable=False),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("current_node_ids", sa.JSON(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(f"status IN {_ALLOWED_WORKFLOW_RUN_STATUSES!s}", name="ck_workflow_runs_status"),
        sa.CheckConstraint(
            f"trigger_kind IN {_ALLOWED_WORKFLOW_TRIGGER_KINDS!s}", name="ck_workflow_runs_trigger_kind"
        ),
        sa.ForeignKeyConstraint(["workflow_id"], ["workflow_definitions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_workflow_runs_workflow_created", "workflow_runs", ["workflow_id", "created_at"])
    op.create_index("ix_workflow_runs_status_updated", "workflow_runs", ["status", "updated_at"])
    op.create_index("ix_workflow_runs_supervisor_session", "workflow_runs", ["supervisor_session_id"])

    op.create_table(
        "workflow_node_runs",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("workflow_run_id", sa.String(length=32), nullable=False),
        sa.Column("node_id", sa.String(length=255), nullable=False),
        sa.Column("attempt_no", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=True),
        sa.Column("profile_name", sa.String(length=255), nullable=True),
        sa.Column("session_id", sa.String(length=32), nullable=True),
        sa.Column("run_id", sa.String(length=32), nullable=True),
        sa.Column("input_parts", sa.JSON(), nullable=False),
        sa.Column("output_text", sa.Text(), nullable=True),
        sa.Column("output_json", sa.JSON(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("needs", sa.JSON(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(f"status IN {_ALLOWED_WORKFLOW_NODE_RUN_STATUSES!s}", name="ck_workflow_node_runs_status"),
        sa.ForeignKeyConstraint(["workflow_run_id"], ["workflow_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_workflow_node_runs_workflow_node", "workflow_node_runs", ["workflow_run_id", "node_id"])
    op.create_index("ix_workflow_node_runs_run", "workflow_node_runs", ["run_id"])

    op.create_table(
        "workflow_events",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("workflow_run_id", sa.String(length=32), nullable=False),
        sa.Column("node_run_id", sa.String(length=32), nullable=True),
        sa.Column("source_kind", sa.String(length=32), nullable=True),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["workflow_run_id"], ["workflow_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_workflow_events_run_created", "workflow_events", ["workflow_run_id", "created_at"])
    op.create_index("ix_workflow_events_node", "workflow_events", ["node_run_id"])


def downgrade() -> None:
    op.drop_index("ix_workflow_events_node", table_name="workflow_events")
    op.drop_index("ix_workflow_events_run_created", table_name="workflow_events")
    op.drop_table("workflow_events")
    op.drop_index("ix_workflow_node_runs_run", table_name="workflow_node_runs")
    op.drop_index("ix_workflow_node_runs_workflow_node", table_name="workflow_node_runs")
    op.drop_table("workflow_node_runs")
    op.drop_index("ix_workflow_runs_supervisor_session", table_name="workflow_runs")
    op.drop_index("ix_workflow_runs_status_updated", table_name="workflow_runs")
    op.drop_index("ix_workflow_runs_workflow_created", table_name="workflow_runs")
    op.drop_table("workflow_runs")
    op.drop_index("ix_workflow_definitions_scope_status", table_name="workflow_definitions")
    op.drop_index("ix_workflow_definitions_owner_session", table_name="workflow_definitions")
    op.drop_index("ix_workflow_definitions_status_updated", table_name="workflow_definitions")
    op.drop_table("workflow_definitions")

    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        with op.batch_alter_table("schedule_fires") as batch_op:
            batch_op.drop_column("workflow_run_id")
        with op.batch_alter_table("schedules") as batch_op:
            batch_op.drop_constraint("ck_schedules_execution_mode", type_="check")
            batch_op.drop_column("last_workflow_run_id")
            batch_op.drop_column("workflow_inputs_template")
            batch_op.drop_column("workflow_id")
            batch_op.create_check_constraint(
                "ck_schedules_execution_mode",
                f"execution_mode IN {_OLD_SCHEDULE_EXECUTION_MODES!s}",
            )
    else:
        op.drop_column("schedule_fires", "workflow_run_id")
        op.drop_constraint("ck_schedules_execution_mode", "schedules", type_="check")
        op.drop_column("schedules", "last_workflow_run_id")
        op.drop_column("schedules", "workflow_inputs_template")
        op.drop_column("schedules", "workflow_id")
        op.create_check_constraint(
            "ck_schedules_execution_mode",
            "schedules",
            sa.column("execution_mode").in_(_OLD_SCHEDULE_EXECUTION_MODES),
        )
