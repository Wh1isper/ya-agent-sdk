from __future__ import annotations

from typing import Any, Literal

from fastapi import HTTPException
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
    AgencySourceSessionSubmitRequest,
    AgencySourceSessionSubmitResponse,
    AgencyStatusResponse,
    DispatchMode,
    RunStatus,
    SessionSubmitRequest,
    TextPart,
    TriggerType,
    agency_fire_summary_from_record,
    run_summary_from_record,
    session_summary_from_record,
)
from ya_claw.controller.session import SessionController
from ya_claw.memory.store import WorkspaceMemoryStore
from ya_claw.orm.tables import AgencyFireRecord, RunRecord, SessionRecord
from ya_claw.runtime_state import InMemoryRuntimeState
from ya_claw.workspace.models import WorkspaceBinding, WorkspaceMountBinding


def _normalize_handoff_tags(tags: list[str]) -> list[str]:
    normalized: list[str] = []
    for tag in tags:
        clean = tag.strip()
        if clean and clean not in normalized:
            normalized.append(clean)
    if "agency-reminder" not in normalized:
        normalized.insert(0, "agency-reminder")
    return normalized


class AgencyController:
    def __init__(self) -> None:
        self._session_controller = SessionController()

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
            state=_agency_state(agency_session, active_run=active_run, latest_run=latest_run),
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

    async def submit_to_session(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        runtime_state: InMemoryRuntimeState,
        request: AgencySourceSessionSubmitRequest,
    ) -> AgencySourceSessionSubmitResponse:
        if not settings.agency_enabled:
            raise HTTPException(status_code=403, detail="Agency submit-to-session requires agency to be enabled.")
        if not request.prompt.strip():
            raise HTTPException(status_code=422, detail="prompt is required.")
        if not isinstance(request.agency_session_id, str) or not request.agency_session_id.strip():
            raise HTTPException(status_code=422, detail="agency_session_id is required.")
        if not isinstance(request.agency_run_id, str) or not request.agency_run_id.strip():
            raise HTTPException(status_code=422, detail="agency_run_id is required.")

        agency_session = await db_session.get(SessionRecord, request.agency_session_id)
        if not isinstance(agency_session, SessionRecord) or agency_session.session_type != "agency":
            raise HTTPException(status_code=403, detail="Agency submit-to-session requires an agency session.")
        agency_run = await db_session.get(RunRecord, request.agency_run_id)
        if (
            not isinstance(agency_run, RunRecord)
            or agency_run.session_id != agency_session.id
            or agency_run.trigger_type != TriggerType.AGENCY.value
            or agency_run.status != RunStatus.RUNNING.value
            or agency_session.active_run_id != agency_run.id
            or runtime_state.get_run_handle(agency_run.id) is None
        ):
            raise HTTPException(
                status_code=403, detail="Agency submit-to-session requires the active agency runtime run."
            )

        source_session = await db_session.get(SessionRecord, request.session_id)
        if not isinstance(source_session, SessionRecord):
            raise HTTPException(status_code=404, detail=f"Session '{request.session_id}' was not found.")
        if source_session.session_type != "conversation":
            raise HTTPException(status_code=422, detail="session_id must reference a conversation session.")

        handoff_kind = request.handoff_kind.strip() or "reminder"
        handoff_tags = _normalize_handoff_tags(request.handoff_tags)
        handoff = {
            "agency_session_id": agency_session.id,
            "agency_run_id": agency_run.id,
            "source_session_id": source_session.id,
            "kind": handoff_kind,
            "tags": handoff_tags,
            "metadata": dict(request.metadata),
        }
        agency_handoff_metadata = {"latest": handoff, "handoffs": [handoff]}
        response = await self._session_controller.submit_input(
            db_session,
            settings,
            runtime_state,
            source_session.id,
            SessionSubmitRequest(
                input_parts=[
                    TextPart(
                        type="text",
                        text=request.prompt,
                        metadata={
                            "source": "agency_handoff",
                            "handoff_kind": handoff_kind,
                            "handoff_tags": handoff_tags,
                            "agency_session_id": agency_session.id,
                            "agency_run_id": agency_run.id,
                        },
                    )
                ],
                metadata={"agency_handoff": agency_handoff_metadata},
                dispatch_mode=DispatchMode.ASYNC,
                trigger_type=TriggerType.AGENCY_HANDOFF,
            ),
        )
        return AgencySourceSessionSubmitResponse(
            source_session_id=source_session.id,
            delivery=response.delivery,
            run_id=response.run_id,
            status=response.status,
            session_submit=response,
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
    active_run: RunRecord | None,
    latest_run: RunRecord | None,
) -> Literal["idle", "queued", "running"]:
    if active_run is not None and active_run.status == RunStatus.RUNNING.value:
        return "running"
    if active_run is not None and active_run.status == RunStatus.QUEUED.value:
        return "queued"
    if latest_run is not None and latest_run.status == RunStatus.QUEUED.value:
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
        if isinstance(record, RunRecord) and record.status in {RunStatus.QUEUED.value, RunStatus.RUNNING.value}:
            return record
    if isinstance(agency_session.head_run_id, str):
        record = await db_session.get(RunRecord, agency_session.head_run_id)
        if isinstance(record, RunRecord) and record.status in {RunStatus.QUEUED.value, RunStatus.RUNNING.value}:
            return record
    return None


async def _run_count(db_session: AsyncSession, session_id: str) -> int:
    result = await db_session.execute(select(func.count()).where(RunRecord.session_id == session_id))
    return int(result.scalar_one_or_none() or 0)


async def _fire_count(db_session: AsyncSession, status: str) -> int:
    result = await db_session.execute(select(func.count()).where(AgencyFireRecord.status == status))
    return int(result.scalar_one_or_none() or 0)
