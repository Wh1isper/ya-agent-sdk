from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ya_claw.config import ClawSettings
from ya_claw.controller.models import (
    MemoryActionRequest,
    MemoryActionResponse,
    MemoryJobKind,
)
from ya_claw.memory.lifecycle import MemoryLifecycle
from ya_claw.orm.tables import SessionRecord
from ya_claw.runtime_state import InMemoryRuntimeState


class MemoryController:
    async def enqueue_extract(
        self,
        *,
        settings: ClawSettings,
        session_factory: async_sessionmaker[AsyncSession],
        runtime_state: InMemoryRuntimeState,
        source_session_id: str,
        request: MemoryActionRequest,
        submit_run,
    ) -> MemoryActionResponse:
        lifecycle = MemoryLifecycle(
            settings=settings,
            session_factory=session_factory,
            runtime_state=runtime_state,
            submit_run=submit_run,
        )
        try:
            run_id = await lifecycle.enqueue_manual_extract(
                source_session_id=source_session_id,
                reason=request.reason or "manual_extract",
                source_run_ids=list(request.run_ids),
            )
        except TypeError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return MemoryActionResponse(
            accepted=True,
            source_session_id=source_session_id,
            run_id=run_id,
            kind=MemoryJobKind.EXTRACT,
            reason=request.reason,
        )

    async def enqueue_summary(
        self,
        *,
        settings: ClawSettings,
        session_factory: async_sessionmaker[AsyncSession],
        runtime_state: InMemoryRuntimeState,
        source_session_id: str,
        request: MemoryActionRequest,
        submit_run,
    ) -> MemoryActionResponse:
        lifecycle = MemoryLifecycle(
            settings=settings,
            session_factory=session_factory,
            runtime_state=runtime_state,
            submit_run=submit_run,
        )
        try:
            run_id = await lifecycle.enqueue_manual_summary(
                source_session_id=source_session_id,
                reason=request.reason or "manual_summary",
            )
        except TypeError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return MemoryActionResponse(
            accepted=True,
            source_session_id=source_session_id,
            run_id=run_id,
            kind=MemoryJobKind.SUMMARY,
            reason=request.reason,
        )

    async def _require_source_session(self, db_session: AsyncSession, source_session_id: str) -> SessionRecord:
        source_session = await db_session.get(SessionRecord, source_session_id)
        if not isinstance(source_session, SessionRecord):
            raise HTTPException(status_code=404, detail=f"Session '{source_session_id}' was not found.")
        return source_session
