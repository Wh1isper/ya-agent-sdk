from __future__ import annotations

from typing import Any, Literal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ya_claw.agency.lifecycle import AGENCY_SINGLETON_SCOPE_KEY, AgencyLifecycle
from ya_claw.config import ClawSettings
from ya_claw.controller.models import (
    AgencyClearResponse,
    AgencyConfigResponse,
    AgencyFireListResponse,
    AgencyFireSummary,
    AgencyRiskPolicy,
    AgencyStatusResponse,
    agency_fire_summary_from_record,
    run_summary_from_record,
    session_summary_from_record,
)
from ya_claw.memory.store import WorkspaceMemoryStore
from ya_claw.orm.tables import AgencyFireRecord, RunRecord, SessionRecord
from ya_claw.runtime_state import InMemoryRuntimeState
from ya_claw.workspace.models import WorkspaceBinding, WorkspaceMountBinding


class AgencyController:
    async def bootstrap(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        runtime_state: InMemoryRuntimeState,
    ) -> AgencyConfigResponse:
        lifecycle = AgencyLifecycle(settings=settings, runtime_state=runtime_state)
        await lifecycle.ensure_agency_session(db_session)
        await db_session.commit()
        return await self.config(db_session, settings, runtime_state)

    async def config(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        runtime_state: InMemoryRuntimeState,
    ) -> AgencyConfigResponse:
        lifecycle = AgencyLifecycle(settings=settings, runtime_state=runtime_state)
        agency_session = await lifecycle.ensure_agency_session(db_session)
        await db_session.commit()
        await db_session.refresh(agency_session)
        metadata = _agency_metadata(agency_session)
        return AgencyConfigResponse(
            enabled=settings.agency_enabled,
            profile_name=str(
                metadata.get("profile_name") or agency_session.profile_name or settings.resolved_agency_profile
            ),
            timer_interval_seconds=settings.agency_timer_interval_seconds,
            agency_session_id=agency_session.id,
            singleton_scope_key=AGENCY_SINGLETON_SCOPE_KEY,
            singleton_source_session_id=agency_session.source_session_id or "",
            risk_policy=_resolved_risk_policy(settings, metadata),
            memory_files={
                "index": "AGENCY.md",
                "action_log": "agency/ACTION_LOG.md",
            },
            next_fire_at=await lifecycle.next_timer_fire_at(db_session),
        )

    async def status(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        runtime_state: InMemoryRuntimeState,
    ) -> AgencyStatusResponse:
        lifecycle = AgencyLifecycle(settings=settings, runtime_state=runtime_state)
        agency_session = await lifecycle.ensure_agency_session(db_session)
        await db_session.commit()
        await db_session.refresh(agency_session)
        latest_run = await _latest_run(db_session, agency_session.id)
        active_run = await _active_run(db_session, agency_session)
        pending_fire_count = await _fire_count(db_session, "pending")
        return AgencyStatusResponse(
            enabled=settings.agency_enabled,
            agency_session_id=agency_session.id,
            state=_agency_state(agency_session, latest_run=latest_run),
            active_run=run_summary_from_record(active_run) if active_run is not None else None,
            latest_run=run_summary_from_record(latest_run) if latest_run is not None else None,
            active_run_id=active_run.id if active_run is not None else None,
            latest_run_id=latest_run.id if latest_run is not None else None,
            next_fire_at=await lifecycle.next_timer_fire_at(db_session),
            pending_fire_count=pending_fire_count,
            last_fire=await self.last_fire(db_session),
            agency_session=session_summary_from_record(
                agency_session,
                run_count=await _run_count(db_session, agency_session.id),
                latest_run=run_summary_from_record(latest_run) if latest_run is not None else None,
            ),
        )

    async def list_fires(self, db_session: AsyncSession, *, limit: int = 50) -> AgencyFireListResponse:
        normalized_limit = min(max(limit, 1), 200)
        result = await db_session.execute(
            select(AgencyFireRecord, RunRecord.status)
            .outerjoin(RunRecord, AgencyFireRecord.run_id == RunRecord.id)
            .order_by(AgencyFireRecord.created_at.desc())
            .limit(normalized_limit)
        )
        return AgencyFireListResponse(
            fires=[
                agency_fire_summary_from_record(record, run_status=run_status)
                for record, run_status in result.all()
                if isinstance(record, AgencyFireRecord)
            ]
        )

    async def clear(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        runtime_state: InMemoryRuntimeState,
    ) -> AgencyClearResponse:
        lifecycle = AgencyLifecycle(settings=settings, runtime_state=runtime_state)
        clear_result = await lifecycle.clear_agency_session(db_session)
        for run_id in clear_result.archived_run_ids:
            runtime_state.clear_run(run_id)
        workspace_dir = settings.resolved_workspace_dir
        WorkspaceMemoryStore(
            WorkspaceBinding(
                host_path=workspace_dir,
                virtual_path=workspace_dir,
                cwd=workspace_dir,
                readable_paths=[workspace_dir],
                writable_paths=[workspace_dir],
                mounts=[WorkspaceMountBinding(id="default", host_path=workspace_dir, virtual_path=workspace_dir)],
                fingerprint="agency-clear",
            )
        ).reset_agency()
        agency_session = await lifecycle.ensure_agency_session(db_session)
        await db_session.commit()
        await db_session.refresh(agency_session)
        return AgencyClearResponse(
            accepted=True,
            cleared_session_id=clear_result.cleared_session_id,
            new_agency_session_id=agency_session.id,
            archived_run_ids=clear_result.archived_run_ids,
            deleted_fire_count=clear_result.deleted_fire_count,
            cleared_at=clear_result.cleared_at,
            agency_session=session_summary_from_record(
                agency_session,
                run_count=await _run_count(db_session, agency_session.id),
                latest_run=None,
            ),
        )

    async def last_fire(self, db_session: AsyncSession) -> AgencyFireSummary | None:
        result = await db_session.execute(
            select(AgencyFireRecord, RunRecord.status)
            .outerjoin(RunRecord, AgencyFireRecord.run_id == RunRecord.id)
            .order_by(AgencyFireRecord.created_at.desc())
            .limit(1)
        )
        row = result.one_or_none()
        if row is None:
            return None
        record, run_status = row
        return (
            agency_fire_summary_from_record(record, run_status=run_status)
            if isinstance(record, AgencyFireRecord)
            else None
        )


def _agency_metadata(agency_session: SessionRecord) -> dict[str, Any]:
    metadata = agency_session.session_metadata if isinstance(agency_session.session_metadata, dict) else {}
    agency = metadata.get("agency") if isinstance(metadata.get("agency"), dict) else {}
    return dict(agency) if isinstance(agency, dict) else {}


def _resolved_risk_policy(settings: ClawSettings, agency_metadata: dict[str, Any]) -> AgencyRiskPolicy:
    threshold = settings.agency_unattended_shell_review_risk_threshold
    if threshold is None:
        threshold = settings.unattended_shell_review_risk_threshold
    return (
        AgencyRiskPolicy(max_auto_action_risk=threshold)
        if threshold in {"low", "medium", "high", "extra_high"}
        else AgencyRiskPolicy()
    )


def _agency_state(
    agency_session: SessionRecord,
    *,
    latest_run: RunRecord | None,
) -> Literal["idle", "queued", "running"]:
    if isinstance(agency_session.active_run_id, str):
        return "running"
    if latest_run is not None and latest_run.status == "queued":
        return "queued"
    return "idle"


async def _latest_run(db_session: AsyncSession, session_id: str) -> RunRecord | None:
    result = await db_session.execute(
        select(RunRecord)
        .where(RunRecord.session_id == session_id)
        .order_by(RunRecord.sequence_no.desc(), RunRecord.id.desc())
        .limit(1)
    )
    return result.scalars().first()


async def _active_run(db_session: AsyncSession, agency_session: SessionRecord) -> RunRecord | None:
    if isinstance(agency_session.active_run_id, str):
        record = await db_session.get(RunRecord, agency_session.active_run_id)
        if isinstance(record, RunRecord):
            return record
    if isinstance(agency_session.head_run_id, str):
        record = await db_session.get(RunRecord, agency_session.head_run_id)
        if isinstance(record, RunRecord) and record.status in {"queued", "running"}:
            return record
    return None


async def _run_count(db_session: AsyncSession, session_id: str) -> int:
    result = await db_session.execute(select(func.count()).where(RunRecord.session_id == session_id))
    return int(result.scalar_one_or_none() or 0)


async def _fire_count(db_session: AsyncSession, status: str) -> int:
    result = await db_session.execute(select(func.count()).where(AgencyFireRecord.status == status))
    return int(result.scalar_one_or_none() or 0)
