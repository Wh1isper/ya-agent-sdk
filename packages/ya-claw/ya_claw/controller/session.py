from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from fastapi import HTTPException
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy import Select, and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import load_only

from ya_claw.config import ClawSettings
from ya_claw.controller.models import (
    RunCreateRequest,
    RunDetail,
    RunStatus,
    RunSummary,
    SessionCreateRequest,
    SessionCreateResponse,
    SessionDetail,
    SessionForkRequest,
    SessionGetResponse,
    SessionListResponse,
    SessionRunCreateRequest,
    SessionSubmitRequest,
    SessionSubmitResponse,
    SessionSummary,
    SessionTurnsResponse,
    SteerRequest,
    active_interactions_from_run_record,
    memory_state_summary_from_record,
    run_detail_from_record,
    run_summary_from_record,
    session_summary_from_record,
    session_turn_from_record,
)
from ya_claw.controller.run import RunController
from ya_claw.controller.session_lifecycle import SessionPruneClaimedError, lock_session_reference
from ya_claw.controller.store import read_run_message_blob_if_exists, read_run_state_blob_if_exists
from ya_claw.controller.workspace_runtime import reconcile_session_sandbox_metadata
from ya_claw.orm.tables import RunRecord, SessionMemoryStateRecord, SessionRecord
from ya_claw.runtime_state import InMemoryRuntimeState
from ya_claw.workspace.models import metadata_with_workspace

_DEFAULT_SESSION_RUNS_LIMIT = 20
_MAX_SESSION_RUNS_LIMIT = 100


class _SessionRunPage(BaseModel):
    items: list[RunSummary] = Field(default_factory=list)
    limit: int
    has_more: bool = False
    next_before_sequence_no: int | None = None


class SessionController:
    def __init__(self) -> None:
        self._run_controller = RunController()

    async def create(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        runtime_state: InMemoryRuntimeState,
        request: SessionCreateRequest,
    ) -> SessionCreateResponse:
        session_id = uuid4().hex
        logger.debug(
            "Creating session session_id={} profile={} trigger_type={} initial_input_parts={} dispatch_mode={}",
            session_id,
            request.profile_name,
            request.trigger_type,
            len(request.input_parts),
            request.dispatch_mode,
        )
        session_metadata = metadata_with_workspace(request.metadata, request.workspace)
        record = SessionRecord(
            id=session_id,
            profile_name=request.profile_name,
            session_metadata=session_metadata,
            session_type="conversation",
        )
        db_session.add(record)
        await db_session.commit()
        await db_session.refresh(record)

        created_run = None
        if request.input_parts:
            created_run = await self._run_controller.create(
                db_session,
                settings,
                runtime_state,
                RunCreateRequest(
                    session_id=session_id,
                    profile_name=request.profile_name,
                    input_parts=request.input_parts,
                    trigger_type=request.trigger_type,
                    metadata={},
                    workspace=request.workspace,
                    dispatch_mode=request.dispatch_mode,
                ),
            )
            refreshed_record = await db_session.get(SessionRecord, session_id)
            if isinstance(refreshed_record, SessionRecord):
                record = refreshed_record

        summary = await self._build_summary(db_session, record)
        logger.info(
            "Session created session_id={} profile={} initial_run_id={}",
            session_id,
            record.profile_name,
            created_run.id if created_run is not None else None,
        )
        return SessionCreateResponse(session=summary, run=created_run)

    async def create_run(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        runtime_state: InMemoryRuntimeState,
        session_id: str,
        request: SessionRunCreateRequest,
    ) -> RunDetail:
        logger.debug(
            "Creating session run session_id={} reset_state={} restore_from_run_id={} dispatch_mode={}",
            session_id,
            request.reset_state,
            request.restore_from_run_id,
            request.dispatch_mode,
        )
        record = await db_session.get(SessionRecord, session_id)
        if not isinstance(record, SessionRecord):
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' was not found.")
        return await self._create_run_for_session(db_session, settings, runtime_state, record, request)

    async def submit_input(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        runtime_state: InMemoryRuntimeState,
        session_id: str,
        request: SessionSubmitRequest,
    ) -> SessionSubmitResponse:
        async with runtime_state.session_lock(session_id):
            return await self.submit_input_locked(
                db_session,
                settings,
                runtime_state,
                session_id,
                request,
            )

    async def submit_input_locked(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        runtime_state: InMemoryRuntimeState,
        session_id: str,
        request: SessionSubmitRequest,
    ) -> SessionSubmitResponse:
        if not request.input_parts:
            raise HTTPException(status_code=422, detail="input_parts must not be empty for session input submission.")
        record = await db_session.get(SessionRecord, session_id)
        if not isinstance(record, SessionRecord):
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' was not found.")
        active_run = await self._active_run_record(db_session, record)
        if isinstance(active_run, RunRecord):
            if active_run.status == RunStatus.QUEUED:
                input_payload = [part.model_dump(mode="json") for part in request.input_parts]
                active_run.input_parts = [*list(active_run.input_parts or []), *input_payload]
                active_run.run_metadata = _merge_submit_metadata(active_run.run_metadata, request.metadata)
                await db_session.commit()
                await db_session.refresh(active_run)
                return SessionSubmitResponse(
                    session_id=active_run.session_id,
                    run_id=active_run.id,
                    delivery="merged",
                    status=active_run.status,
                    run=run_detail_from_record(active_run),
                )
            input_payload = [part.model_dump(mode="json") for part in request.input_parts]
            active_run.input_parts = [*list(active_run.input_parts or []), *input_payload]
            if request.metadata:
                active_run.run_metadata = _merge_submit_metadata(active_run.run_metadata, request.metadata)
            await db_session.commit()
            await db_session.refresh(active_run)
            control = await self._run_controller.steer(
                db_session,
                runtime_state,
                active_run.id,
                SteerRequest(input_parts=request.input_parts),
            )
            return SessionSubmitResponse(
                session_id=control.session_id,
                run_id=control.run_id,
                delivery="steered",
                status=control.status,
            )
        run = await self._create_run_for_session(
            db_session,
            settings,
            runtime_state,
            record,
            SessionRunCreateRequest(
                restore_from_run_id=request.restore_from_run_id,
                reset_state=request.reset_state,
                input_parts=request.input_parts,
                metadata=request.metadata,
                workspace=request.workspace,
                dispatch_mode=request.dispatch_mode,
                trigger_type=request.trigger_type,
            ),
        )
        return SessionSubmitResponse(
            session_id=session_id,
            run_id=run.id,
            delivery="queued" if request.dispatch_mode == "queue" else "submitted",
            status=run.status,
            run=run,
        )

    async def _create_run_for_session(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        runtime_state: InMemoryRuntimeState,
        record: SessionRecord,
        request: SessionRunCreateRequest,
    ) -> RunDetail:
        run_metadata = dict(request.metadata)
        if request.reset_state:
            run_metadata["reset_state"] = True
        return await self._run_controller.create(
            db_session,
            settings,
            runtime_state,
            RunCreateRequest(
                session_id=record.id,
                restore_from_run_id=request.restore_from_run_id,
                reset_state=request.reset_state,
                profile_name=record.profile_name,
                input_parts=request.input_parts,
                trigger_type=request.trigger_type,
                metadata=run_metadata,
                workspace=request.workspace,
                dispatch_mode=request.dispatch_mode,
            ),
        )

    async def _active_run_record(self, db_session: AsyncSession, record: SessionRecord) -> RunRecord | None:
        if isinstance(record.active_run_id, str):
            active = await db_session.get(RunRecord, record.active_run_id)
            if isinstance(active, RunRecord) and active.status in {RunStatus.QUEUED, RunStatus.RUNNING}:
                return active
        if isinstance(record.head_run_id, str):
            head = await db_session.get(RunRecord, record.head_run_id)
            if isinstance(head, RunRecord) and head.status in {RunStatus.QUEUED, RunStatus.RUNNING}:
                return head
        return None

    async def list(
        self,
        db_session: AsyncSession,
        *,
        settings: ClawSettings | None = None,
        include_internal: bool = False,
        limit: int | None = None,
        before_updated_at: datetime | None = None,
        before_id: str | None = None,
        include_latest_output: bool = True,
        reconcile_workspace: bool = True,
    ) -> list[SessionSummary]:
        logger.debug(
            "Listing sessions include_internal={} limit={} before_updated_at={} before_id={} include_latest_output={} reconcile_workspace={}",
            include_internal,
            limit,
            before_updated_at,
            before_id,
            include_latest_output,
            reconcile_workspace,
        )
        statement: Select[tuple[SessionRecord]] = select(SessionRecord)
        if not include_internal:
            statement = statement.where(SessionRecord.session_type == "conversation")
        if isinstance(before_updated_at, datetime):
            if isinstance(before_id, str) and before_id:
                statement = statement.where(
                    or_(
                        SessionRecord.updated_at < before_updated_at,
                        and_(
                            SessionRecord.updated_at == before_updated_at,
                            SessionRecord.id < before_id,
                        ),
                    )
                )
            else:
                statement = statement.where(SessionRecord.updated_at < before_updated_at)
        statement = statement.order_by(SessionRecord.updated_at.desc(), SessionRecord.id.desc())
        if isinstance(limit, int):
            statement = statement.limit(min(max(limit, 1), 200))
        result = await db_session.execute(statement)
        records = list(result.scalars().all())
        if settings is not None and reconcile_workspace:
            await self._reconcile_workspace_states(db_session, settings=settings, records=records)
        return await self._build_summaries(
            db_session,
            records,
            include_latest_output=include_latest_output,
        )

    async def list_page(
        self,
        db_session: AsyncSession,
        *,
        settings: ClawSettings | None = None,
        include_internal: bool = False,
        limit: int = 50,
        before_updated_at: datetime | None = None,
        before_id: str | None = None,
        include_latest_output: bool = False,
    ) -> SessionListResponse:
        if (before_updated_at is None) != (before_id is None):
            raise HTTPException(
                status_code=422,
                detail="before_updated_at and before_id must be provided together.",
            )
        normalized_limit = min(max(limit, 1), 100)
        summaries = await self.list(
            db_session,
            settings=settings,
            include_internal=include_internal,
            limit=normalized_limit + 1,
            before_updated_at=before_updated_at,
            before_id=before_id,
            include_latest_output=include_latest_output,
            reconcile_workspace=False,
        )
        has_more = len(summaries) > normalized_limit
        page_summaries = summaries[:normalized_limit]
        count_statement = select(func.count(SessionRecord.id))
        if not include_internal:
            count_statement = count_statement.where(SessionRecord.session_type == "conversation")
        total = int((await db_session.scalar(count_statement)) or 0)
        next_anchor = page_summaries[-1] if has_more and page_summaries else None
        return SessionListResponse(
            sessions=page_summaries,
            total=total,
            limit=normalized_limit,
            has_more=has_more,
            next_before_updated_at=next_anchor.updated_at if next_anchor is not None else None,
            next_before_id=next_anchor.id if next_anchor is not None else None,
        )

    async def get(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        session_id: str,
        *,
        runs_limit: int = _DEFAULT_SESSION_RUNS_LIMIT,
        before_sequence_no: int | None = None,
        include_message: bool = False,
        include_input_parts: bool = False,
        include_head_payload: bool = True,
    ) -> SessionGetResponse:
        logger.debug(
            "Fetching session session_id={} runs_limit={} before_sequence_no={} include_message={} include_input_parts={} include_head_payload={}",
            session_id,
            runs_limit,
            before_sequence_no,
            include_message,
            include_input_parts,
            include_head_payload,
        )
        record = await db_session.get(SessionRecord, session_id)
        if not isinstance(record, SessionRecord):
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' was not found.")
        await self._reconcile_workspace_states(db_session, settings=settings, records=[record])

        summary = await self._build_summary(db_session, record)
        run_list = await self._list_runs(
            db_session,
            settings,
            session_id,
            limit=runs_limit,
            before_sequence_no=before_sequence_no,
            include_message=include_message,
            include_input_parts=include_input_parts,
        )
        state_payload = None
        message_payload = None
        if include_head_payload and isinstance(record.head_success_run_id, str):
            state_payload = read_run_state_blob_if_exists(settings, record.head_success_run_id)
            if include_message:
                message_payload = read_run_message_blob_if_exists(settings, record.head_success_run_id)

        return SessionGetResponse(
            session=SessionDetail(
                **summary.model_dump(),
                runs=run_list.items,
                runs_limit=run_list.limit,
                runs_has_more=run_list.has_more,
                runs_next_before_sequence_no=run_list.next_before_sequence_no,
            ),
            state=state_payload,
            message=message_payload,
        )

    async def _list_runs(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        session_id: str,
        *,
        limit: int = _DEFAULT_SESSION_RUNS_LIMIT,
        before_sequence_no: int | None = None,
        include_message: bool = False,
        include_input_parts: bool = False,
    ) -> _SessionRunPage:
        logger.debug(
            "Listing session runs session_id={} limit={} before_sequence_no={} include_message={} include_input_parts={}",
            session_id,
            limit,
            before_sequence_no,
            include_message,
            include_input_parts,
        )
        record = await db_session.get(SessionRecord, session_id)
        if not isinstance(record, SessionRecord):
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' was not found.")

        normalized_limit = min(max(limit, 1), _MAX_SESSION_RUNS_LIMIT)
        statement = select(RunRecord).where(RunRecord.session_id == session_id)
        if isinstance(before_sequence_no, int):
            statement = statement.where(RunRecord.sequence_no < before_sequence_no)
        statement = statement.order_by(RunRecord.sequence_no.desc(), RunRecord.id.desc()).limit(normalized_limit + 1)

        result = await db_session.execute(statement)
        run_records = list(result.scalars().all())
        has_more = len(run_records) > normalized_limit
        page_records = run_records[:normalized_limit]
        items = [
            self._run_controller.build_session_run_summary(
                settings,
                run_record,
                include_message=include_message,
                include_input_parts=include_input_parts,
            )
            for run_record in page_records
        ]
        next_before_sequence_no = page_records[-1].sequence_no if has_more and page_records else None
        return _SessionRunPage(
            items=items,
            limit=normalized_limit,
            has_more=has_more,
            next_before_sequence_no=next_before_sequence_no,
        )

    async def list_turns(
        self,
        db_session: AsyncSession,
        session_id: str,
        *,
        limit: int = _DEFAULT_SESSION_RUNS_LIMIT,
        before_sequence_no: int | None = None,
        cursor: str | None = None,
    ) -> SessionTurnsResponse:
        logger.debug(
            "Listing session turns session_id={} limit={} before_sequence_no={} cursor={}",
            session_id,
            limit,
            before_sequence_no,
            cursor,
        )
        record = await db_session.get(SessionRecord, session_id)
        if not isinstance(record, SessionRecord):
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' was not found.")

        normalized_limit = min(max(limit, 1), _MAX_SESSION_RUNS_LIMIT)
        statement = select(RunRecord).where(
            RunRecord.session_id == session_id,
            RunRecord.status == RunStatus.COMPLETED,
        )
        cursor_record: RunRecord | None = None
        if isinstance(cursor, str) and cursor.strip() != "":
            cursor_record = await db_session.get(RunRecord, cursor.strip())
            if not isinstance(cursor_record, RunRecord) or cursor_record.session_id != session_id:
                raise HTTPException(
                    status_code=404, detail=f"Run cursor '{cursor}' was not found in session '{session_id}'."
                )
            statement = statement.where(
                or_(
                    RunRecord.sequence_no < cursor_record.sequence_no,
                    (RunRecord.sequence_no == cursor_record.sequence_no) & (RunRecord.id < cursor_record.id),
                )
            )
        elif isinstance(before_sequence_no, int):
            statement = statement.where(RunRecord.sequence_no < before_sequence_no)
        statement = statement.order_by(RunRecord.sequence_no.desc(), RunRecord.id.desc()).limit(normalized_limit + 1)

        result = await db_session.execute(statement)
        run_records = list(result.scalars().all())
        has_more = len(run_records) > normalized_limit
        page_records = run_records[:normalized_limit]
        next_page_anchor = page_records[-1] if has_more and page_records else None
        return SessionTurnsResponse(
            session_id=session_id,
            limit=normalized_limit,
            has_more=has_more,
            next_cursor=next_page_anchor.id if isinstance(next_page_anchor, RunRecord) else None,
            next_before_sequence_no=next_page_anchor.sequence_no if isinstance(next_page_anchor, RunRecord) else None,
            turns=[session_turn_from_record(run_record) for run_record in page_records],
        )

    async def fork(self, db_session: AsyncSession, session_id: str, request: SessionForkRequest) -> SessionSummary:
        logger.debug(
            "Forking session source_session_id={} restore_from_run_id={} profile={}",
            session_id,
            request.restore_from_run_id,
            request.profile_name,
        )
        try:
            source_record = await lock_session_reference(db_session, session_id)
        except SessionPruneClaimedError as exc:
            raise HTTPException(
                status_code=409,
                detail=f"Session '{session_id}' is being pruned and cannot be forked.",
            ) from exc
        if not isinstance(source_record, SessionRecord):
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' was not found.")

        restore_from_run_id = request.restore_from_run_id or source_record.head_success_run_id
        if restore_from_run_id is None:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' does not have a forkable run.")

        restore_record = await db_session.get(RunRecord, restore_from_run_id)
        if not isinstance(restore_record, RunRecord):
            raise HTTPException(status_code=404, detail=f"Run '{restore_from_run_id}' was not found.")
        if restore_record.session_id != source_record.id:
            raise HTTPException(
                status_code=422,
                detail=f"Run '{restore_from_run_id}' does not belong to session '{session_id}'.",
            )

        source_metadata = (
            dict(source_record.session_metadata) if isinstance(source_record.session_metadata, dict) else {}
        )
        fork_metadata = {**source_metadata, **dict(request.metadata)}
        fork_metadata = metadata_with_workspace(fork_metadata, request.workspace)
        fork_record = SessionRecord(
            id=uuid4().hex,
            parent_session_id=source_record.id,
            profile_name=request.profile_name or source_record.profile_name,
            session_metadata=fork_metadata,
            session_type="conversation",
            head_run_id=restore_record.id,
            head_success_run_id=restore_record.id
            if restore_record.status == RunStatus.COMPLETED
            else source_record.head_success_run_id,
        )
        db_session.add(fork_record)
        await db_session.commit()
        await db_session.refresh(fork_record)
        logger.info(
            "Session forked source_session_id={} fork_session_id={} restore_from_run_id={}",
            session_id,
            fork_record.id,
            restore_record.id,
        )
        return await self._build_summary(db_session, fork_record)

    async def resolve_active_run_id(self, db_session: AsyncSession, session_id: str) -> str:
        logger.debug("Resolving active run session_id={}", session_id)
        record = await db_session.get(SessionRecord, session_id)
        if not isinstance(record, SessionRecord):
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' was not found.")
        if not isinstance(record.active_run_id, str):
            raise HTTPException(status_code=409, detail=f"Session '{session_id}' does not have an active run.")
        return record.active_run_id

    async def _reconcile_workspace_states(
        self,
        db_session: AsyncSession,
        *,
        settings: ClawSettings,
        records: list[SessionRecord],
    ) -> None:
        if settings.workspace_provider_backend != "docker":
            return
        for record in records:
            await reconcile_session_sandbox_metadata(
                settings=settings,
                db_session=db_session,
                session_record=record,
            )

    async def _build_summary(self, db_session: AsyncSession, record: SessionRecord) -> SessionSummary:
        summaries = await self._build_summaries(db_session, [record])
        return summaries[0]

    async def _build_summaries(
        self,
        db_session: AsyncSession,
        records: list[SessionRecord],
        *,
        include_latest_output: bool = True,
    ) -> list[SessionSummary]:
        if not records:
            return []

        session_ids = [record.id for record in records]
        run_stats = (
            select(
                RunRecord.session_id.label("session_id"),
                func.count(RunRecord.id).label("run_count"),
                func.max(RunRecord.sequence_no).label("latest_sequence_no"),
            )
            .where(RunRecord.session_id.in_(session_ids))
            .group_by(RunRecord.session_id)
            .subquery()
        )
        latest_run_statement = select(RunRecord, run_stats.c.run_count).join(
            run_stats,
            and_(
                RunRecord.session_id == run_stats.c.session_id,
                RunRecord.sequence_no == run_stats.c.latest_sequence_no,
            ),
        )
        if not include_latest_output:
            latest_run_statement = latest_run_statement.options(
                load_only(
                    RunRecord.id,
                    RunRecord.session_id,
                    RunRecord.sequence_no,
                    RunRecord.restore_from_run_id,
                    RunRecord.status,
                    RunRecord.trigger_type,
                    RunRecord.profile_name,
                    RunRecord.input_parts,
                    RunRecord.run_metadata,
                    RunRecord.error_message,
                    RunRecord.termination_reason,
                    RunRecord.created_at,
                    RunRecord.started_at,
                    RunRecord.finished_at,
                    RunRecord.committed_at,
                )
            )
        latest_run_result = await db_session.execute(latest_run_statement)
        latest_runs: dict[str, tuple[RunRecord, int]] = {
            run_record.session_id: (run_record, int(run_count)) for run_record, run_count in latest_run_result.all()
        }

        memory_result = await db_session.execute(
            select(SessionMemoryStateRecord).where(SessionMemoryStateRecord.source_session_id.in_(session_ids))
        )
        memory_states = {
            state.source_session_id: memory_state_summary_from_record(state) for state in memory_result.scalars().all()
        }
        summaries: list[SessionSummary] = []
        for record in records:
            latest_run_entry = latest_runs.get(record.id)
            latest_run_record = latest_run_entry[0] if latest_run_entry is not None else None
            run_count = latest_run_entry[1] if latest_run_entry is not None else 0
            latest_run = (
                run_summary_from_record(
                    latest_run_record,
                    include_output_text=include_latest_output,
                )
                if latest_run_record is not None
                else None
            )
            active_interactions = (
                active_interactions_from_run_record(latest_run_record) if latest_run_record is not None else None
            )
            summaries.append(
                session_summary_from_record(
                    record,
                    run_count=run_count,
                    latest_run=latest_run,
                    memory_state=memory_states.get(record.id),
                    active_interactions=active_interactions,
                )
            )

        return summaries


def _merge_submit_metadata(existing: dict[str, Any] | None, incoming: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(existing or {})
    for key, value in incoming.items():
        if key == "agency" and isinstance(value, dict):
            current = metadata.get("agency")
            merged = dict(current) if isinstance(current, dict) else {}
            for agency_key, agency_value in value.items():
                if agency_key == "sources" and isinstance(agency_value, list):
                    merged[agency_key] = _append_unique_sources(merged.get(agency_key), agency_value)
                elif isinstance(agency_value, list):
                    merged[agency_key] = _append_unique_values(merged.get(agency_key), agency_value)
                else:
                    merged[agency_key] = agency_value
            metadata["agency"] = merged
        elif key == "agency_handoff" and isinstance(value, dict):
            metadata["agency_handoff"] = _merge_agency_handoff_metadata(metadata.get("agency_handoff"), value)
        else:
            metadata[key] = value
    return metadata


def _append_unique_values(existing: object, values: list[object]) -> list[object]:
    result = list(existing) if isinstance(existing, list) else []
    seen = {repr(item) for item in result}
    for value in values:
        marker = repr(value)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(value)
    return result


def _merge_agency_handoff_metadata(existing: object, incoming: dict[str, Any]) -> dict[str, Any]:
    current = dict(existing) if isinstance(existing, dict) else {}
    incoming_copy = dict(incoming)
    existing_handoffs = current.get("handoffs")
    incoming_handoffs = incoming_copy.get("handoffs")
    merged_handoffs: list[object] = []
    if isinstance(existing_handoffs, list):
        merged_handoffs.extend(existing_handoffs)
    elif current:
        merged_handoffs.append(current.get("latest", current))
    if isinstance(incoming_handoffs, list):
        merged_handoffs.extend(incoming_handoffs)
    else:
        merged_handoffs.append(incoming_copy.get("latest", incoming_copy))
    return {
        **current,
        **incoming_copy,
        "latest": incoming_copy.get("latest", incoming_copy),
        "handoffs": _append_unique_values([], merged_handoffs),
    }


def _append_unique_sources(existing: object, values: list[object]) -> list[object]:
    result_dicts = [dict(item) for item in existing if isinstance(item, dict)] if isinstance(existing, list) else []
    seen = {item.get("fire_id") for item in result_dicts if isinstance(item.get("fire_id"), str)}
    for value in values:
        if not isinstance(value, dict):
            continue
        fire_id = value.get("fire_id")
        if isinstance(fire_id, str) and fire_id in seen:
            continue
        if isinstance(fire_id, str):
            seen.add(fire_id)
        result_dicts.append(dict(value))
    return list(result_dicts)
