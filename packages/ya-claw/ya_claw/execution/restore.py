from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ya_claw.execution.store import RunStore
from ya_claw.orm.tables import RunRecord, SessionRecord

_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})


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

    record = await db_session.get(RunRecord, restore_run_id)
    if not isinstance(record, RunRecord):
        raise HTTPException(status_code=404, detail=f"Run '{restore_run_id}' was not found.")
    if record.session_id != session.id:
        raise HTTPException(
            status_code=422,
            detail=f"Run '{restore_run_id}' does not belong to session '{session.id}'.",
        )
    if record.status not in _TERMINAL_STATUSES:
        raise HTTPException(
            status_code=422,
            detail=f"Run '{restore_run_id}' is not a terminal restore source for session '{session.id}'.",
        )
    return record


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
