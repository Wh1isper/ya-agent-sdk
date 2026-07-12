from __future__ import annotations

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from ya_claw.orm.tables import SessionRecord

SESSION_PRUNE_CLAIM = "__prune_pending__"


class SessionPruneClaimedError(RuntimeError):
    def __init__(self, session_id: str) -> None:
        super().__init__(f"Session '{session_id}' is being pruned.")
        self.session_id = session_id


async def lock_session_reference(db_session: AsyncSession, session_id: str) -> SessionRecord | None:
    """Lock a referenced session and reject a concurrent persistent prune claim."""
    await db_session.execute(
        update(SessionRecord)
        .where(SessionRecord.id == session_id)
        .values(
            active_run_id=SessionRecord.active_run_id,
            updated_at=SessionRecord.updated_at,
        )
    )
    record = await db_session.get(
        SessionRecord,
        session_id,
        populate_existing=True,
    )
    if isinstance(record, SessionRecord) and record.active_run_id == SESSION_PRUNE_CLAIM:
        raise SessionPruneClaimedError(session_id)
    return record
