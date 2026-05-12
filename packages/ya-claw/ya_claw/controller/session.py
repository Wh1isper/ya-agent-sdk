from __future__ import annotations

from uuid import uuid4

from fastapi import HTTPException
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy import Select, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

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
    SessionRunCreateRequest,
    SessionSummary,
    SessionTurnsResponse,
    active_interactions_from_run_record,
    memory_state_summary_from_record,
    run_summary_from_record,
    session_summary_from_record,
    session_turn_from_record,
)
from ya_claw.controller.run import RunController
from ya_claw.controller.store import read_run_message_blob_if_exists, read_run_state_blob_if_exists
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

        run_metadata = dict(request.metadata)
        if request.reset_state:
            run_metadata["reset_state"] = True

        return await self._run_controller.create(
            db_session,
            settings,
            runtime_state,
            RunCreateRequest(
                session_id=session_id,
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

    async def list(self, db_session: AsyncSession, *, include_internal: bool = False) -> list[SessionSummary]:
        logger.debug("Listing sessions include_internal={}", include_internal)
        statement: Select[tuple[SessionRecord]] = select(SessionRecord)
        if not include_internal:
            statement = statement.where(SessionRecord.session_type == "conversation")
        statement = statement.order_by(SessionRecord.updated_at.desc())
        result = await db_session.execute(statement)
        records = list(result.scalars().all())
        return await self._build_summaries(db_session, records)

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
    ) -> SessionGetResponse:
        logger.debug(
            "Fetching session session_id={} runs_limit={} before_sequence_no={} include_message={} include_input_parts={}",
            session_id,
            runs_limit,
            before_sequence_no,
            include_message,
            include_input_parts,
        )
        record = await db_session.get(SessionRecord, session_id)
        if not isinstance(record, SessionRecord):
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' was not found.")

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
        if isinstance(record.head_success_run_id, str):
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
        source_record = await db_session.get(SessionRecord, session_id)
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

    async def _build_summary(self, db_session: AsyncSession, record: SessionRecord) -> SessionSummary:
        summaries = await self._build_summaries(db_session, [record])
        return summaries[0]

    async def _build_summaries(self, db_session: AsyncSession, records: list[SessionRecord]) -> list[SessionSummary]:
        if not records:
            return []

        session_ids = [record.id for record in records]
        statement: Select[tuple[RunRecord]] = (
            select(RunRecord)
            .where(RunRecord.session_id.in_(session_ids))
            .order_by(RunRecord.session_id.asc(), RunRecord.sequence_no.desc(), RunRecord.id.desc())
        )
        result = await db_session.execute(statement)
        run_records = list(result.scalars().all())

        grouped_runs: dict[str, list[RunRecord]] = {session_id: [] for session_id in session_ids}
        for run_record in run_records:
            grouped_runs.setdefault(run_record.session_id, []).append(run_record)

        memory_result = await db_session.execute(
            select(SessionMemoryStateRecord).where(SessionMemoryStateRecord.source_session_id.in_(session_ids))
        )
        memory_states = {
            state.source_session_id: memory_state_summary_from_record(state) for state in memory_result.scalars().all()
        }

        summaries: list[SessionSummary] = []
        for record in records:
            runs = grouped_runs.get(record.id, [])
            latest_run_record = runs[0] if runs else None
            latest_run = run_summary_from_record(latest_run_record) if latest_run_record is not None else None
            active_interactions = (
                active_interactions_from_run_record(latest_run_record) if latest_run_record is not None else None
            )
            summaries.append(
                session_summary_from_record(
                    record,
                    run_count=len(runs),
                    latest_run=latest_run,
                    memory_state=memory_states.get(record.id),
                    active_interactions=active_interactions,
                )
            )

        return summaries
