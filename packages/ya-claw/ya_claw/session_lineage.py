from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ya_claw.orm.tables import RunRecord, SessionRecord

_TERMINAL_RUN_STATUSES = frozenset({"completed", "failed", "cancelled"})


async def require_restore_source(
    db_session: AsyncSession,
    *,
    target_session: SessionRecord,
    restore_run_id: str,
) -> RunRecord:
    """Require a terminal run from the target session or one of its ancestors."""
    restore_record = await db_session.get(RunRecord, restore_run_id)
    if not isinstance(restore_record, RunRecord):
        raise HTTPException(status_code=404, detail=f"Run '{restore_run_id}' was not found.")

    restore_session_id = restore_record.session_id
    ancestor_session_id: str | None = target_session.id
    visited: set[str] = set()
    while isinstance(ancestor_session_id, str) and ancestor_session_id not in visited:
        if ancestor_session_id == restore_session_id:
            break
        visited.add(ancestor_session_id)
        ancestor = await db_session.get(SessionRecord, ancestor_session_id)
        ancestor_session_id = ancestor.parent_session_id if isinstance(ancestor, SessionRecord) else None
    else:
        ancestor_session_id = None

    if ancestor_session_id != restore_session_id:
        raise HTTPException(
            status_code=422,
            detail=(f"Run '{restore_run_id}' is not a valid restore source for session '{target_session.id}'."),
        )
    if restore_record.status not in _TERMINAL_RUN_STATUSES:
        raise HTTPException(
            status_code=422,
            detail=(f"Run '{restore_run_id}' is not a terminal restore source for session '{target_session.id}'."),
        )
    return restore_record
