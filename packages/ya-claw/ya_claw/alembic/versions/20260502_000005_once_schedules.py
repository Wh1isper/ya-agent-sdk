from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260502_000005"
down_revision = "20260501_000004"
branch_labels = None
depends_on = None

_ALLOWED_SCHEDULE_STATUSES = ("active", "paused", "completed", "deleted")
_ALLOWED_SCHEDULE_STATUSES_DOWNGRADE = ("active", "paused", "deleted")
_ALLOWED_SCHEDULE_TRIGGER_KINDS = ("cron", "once")


def upgrade() -> None:
    bind = op.get_bind()
    op.add_column("schedules", sa.Column("trigger_kind", sa.String(length=32), nullable=True))
    op.add_column("schedules", sa.Column("run_at", sa.DateTime(timezone=True), nullable=True))
    op.execute("UPDATE schedules SET trigger_kind = 'cron' WHERE trigger_kind IS NULL")

    if bind.dialect.name == "sqlite":
        with op.batch_alter_table("schedules") as batch_op:
            batch_op.drop_constraint("ck_schedules_status", type_="check")
            batch_op.alter_column("trigger_kind", existing_type=sa.String(length=32), nullable=False)
            batch_op.alter_column("cron_expr", existing_type=sa.String(length=255), nullable=True)
            batch_op.create_check_constraint("ck_schedules_status", f"status IN {_ALLOWED_SCHEDULE_STATUSES!s}")
            batch_op.create_check_constraint(
                "ck_schedules_trigger_kind",
                f"trigger_kind IN {_ALLOWED_SCHEDULE_TRIGGER_KINDS!s}",
            )
    else:
        op.drop_constraint("ck_schedules_status", "schedules", type_="check")
        op.alter_column("schedules", "trigger_kind", existing_type=sa.String(length=32), nullable=False)
        op.alter_column("schedules", "cron_expr", existing_type=sa.String(length=255), nullable=True)
        op.create_check_constraint(
            "ck_schedules_status",
            "schedules",
            sa.column("status").in_(_ALLOWED_SCHEDULE_STATUSES),
        )
        op.create_check_constraint(
            "ck_schedules_trigger_kind",
            "schedules",
            sa.column("trigger_kind").in_(_ALLOWED_SCHEDULE_TRIGGER_KINDS),
        )

    op.create_index("ix_schedules_trigger_kind", "schedules", ["trigger_kind"])


def downgrade() -> None:
    bind = op.get_bind()
    op.execute("UPDATE schedules SET status = 'paused' WHERE status = 'completed'")
    op.execute("UPDATE schedules SET cron_expr = '* * * * *' WHERE cron_expr IS NULL")
    op.drop_index("ix_schedules_trigger_kind", table_name="schedules")

    if bind.dialect.name == "sqlite":
        with op.batch_alter_table("schedules") as batch_op:
            batch_op.drop_constraint("ck_schedules_trigger_kind", type_="check")
            batch_op.drop_constraint("ck_schedules_status", type_="check")
            batch_op.alter_column("cron_expr", existing_type=sa.String(length=255), nullable=False)
            batch_op.create_check_constraint(
                "ck_schedules_status", f"status IN {_ALLOWED_SCHEDULE_STATUSES_DOWNGRADE!s}"
            )
            batch_op.drop_column("run_at")
            batch_op.drop_column("trigger_kind")
    else:
        op.drop_constraint("ck_schedules_trigger_kind", "schedules", type_="check")
        op.drop_constraint("ck_schedules_status", "schedules", type_="check")
        op.alter_column("schedules", "cron_expr", existing_type=sa.String(length=255), nullable=False)
        op.create_check_constraint(
            "ck_schedules_status",
            "schedules",
            sa.column("status").in_(_ALLOWED_SCHEDULE_STATUSES_DOWNGRADE),
        )
        op.drop_column("schedules", "run_at")
        op.drop_column("schedules", "trigger_kind")
