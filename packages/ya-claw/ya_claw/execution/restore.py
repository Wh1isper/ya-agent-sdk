from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from ya_claw.execution.store import RunStore
from ya_claw.orm.tables import RunRecord, SessionRecord
from ya_claw.session_lineage import require_restore_source


@dataclass(slots=True)
class ResolvedRestorePoint:
    run_id: str
    session_id: str
    status: str
    state: dict[str, Any] | None
    message: list[dict[str, Any]] | None


async def resolve_restore_run(
    db_session: AsyncSession,
    session: SessionRecord,
    *,
    explicit_run_id: str | None,
) -> RunRecord | None:
    restore_run_id = explicit_run_id or session.head_success_run_id
    if restore_run_id is None:
        return None

    return await require_restore_source(
        db_session,
        target_session=session,
        restore_run_id=restore_run_id,
    )


async def load_restore_point(
    db_session: AsyncSession,
    run_store: RunStore,
    session: SessionRecord,
    *,
    explicit_run_id: str | None,
) -> ResolvedRestorePoint | None:
    record = await resolve_restore_run(db_session, session, explicit_run_id=explicit_run_id)
    if record is None:
        return None
    return ResolvedRestorePoint(
        run_id=record.id,
        session_id=record.session_id,
        status=record.status,
        state=run_store.read_state(record.id),
        message=run_store.read_message(record.id),
    )
