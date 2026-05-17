from __future__ import annotations

from collections.abc import Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ya_claw.agency.lifecycle import AgencyLifecycle, build_signal_response
from ya_claw.config import ClawSettings
from ya_claw.controller.models import (
    AgencyGetResponse,
    AgencySignalRequest,
    AgencySignalResponse,
    AgencySignalSummary,
    AgencyStateSummary,
    AgencyUpdateRequest,
    agency_signal_summary_from_record,
    agency_state_summary_from_record,
    run_summary_from_record,
    session_summary_from_record,
)
from ya_claw.orm.tables import AgencySignalRecord, RunRecord, SessionRecord
from ya_claw.runtime_state import InMemoryRuntimeState


class AgencyController:
    def __init__(self) -> None:
        pass

    async def get(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        runtime_state: InMemoryRuntimeState,
        source_session_id: str,
    ) -> AgencyGetResponse:
        lifecycle = AgencyLifecycle(settings=settings, runtime_state=runtime_state)
        state_record = await lifecycle.get_state(db_session, source_session_id, ensure=True)
        signals = await self._list_signals(db_session, source_session_id)
        agency_session_summary = None
        if isinstance(state_record.agency_session_id, str):
            agency_session = await db_session.get(SessionRecord, state_record.agency_session_id)
            if isinstance(agency_session, SessionRecord):
                agency_session_summary = await self._build_session_summary(db_session, agency_session)
        return AgencyGetResponse(
            state=agency_state_summary_from_record(state_record),
            signals=signals,
            agency_session=agency_session_summary,
        )

    async def update(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        runtime_state: InMemoryRuntimeState,
        source_session_id: str,
        request: AgencyUpdateRequest,
    ) -> AgencyStateSummary:
        lifecycle = AgencyLifecycle(settings=settings, runtime_state=runtime_state)
        return await lifecycle.update_state(
            db_session,
            source_session_id,
            enabled=request.enabled,
            metadata=request.metadata,
        )

    async def signal(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        runtime_state: InMemoryRuntimeState,
        source_session_id: str,
        request: AgencySignalRequest,
        *,
        submit_run: Callable[[str], bool] | None = None,
    ) -> AgencySignalResponse:
        lifecycle = AgencyLifecycle(settings=settings, runtime_state=runtime_state, submit_run=submit_run)
        delivery = await lifecycle.create_signal(db_session, source_session_id, request, dispatch=True)
        return build_signal_response(delivery)

    async def compact(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        runtime_state: InMemoryRuntimeState,
        source_session_id: str,
        request: AgencySignalRequest,
        *,
        submit_run: Callable[[str], bool] | None = None,
    ) -> AgencySignalResponse:
        payload = request.model_copy(update={"reason": "compact"})
        return await self.signal(
            db_session,
            settings,
            runtime_state,
            source_session_id,
            payload,
            submit_run=submit_run,
        )

    async def _list_signals(self, db_session: AsyncSession, source_session_id: str) -> list[AgencySignalSummary]:
        result = await db_session.execute(
            select(AgencySignalRecord)
            .where(AgencySignalRecord.source_session_id == source_session_id)
            .order_by(AgencySignalRecord.created_at.desc())
            .limit(50)
        )
        return [agency_signal_summary_from_record(record) for record in result.scalars().all()]

    async def _build_session_summary(self, db_session: AsyncSession, session_record: SessionRecord):
        result = await db_session.execute(
            select(RunRecord)
            .where(RunRecord.session_id == session_record.id)
            .order_by(RunRecord.sequence_no.desc(), RunRecord.id.desc())
        )
        runs = list(result.scalars().all())
        latest_run = run_summary_from_record(runs[0]) if runs else None
        return session_summary_from_record(
            session_record,
            run_count=len(runs),
            latest_run=latest_run,
        )
