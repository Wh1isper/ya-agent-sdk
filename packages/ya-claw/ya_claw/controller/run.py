from __future__ import annotations

from contextlib import suppress
from typing import Any, Literal
from uuid import uuid4

from fastapi import HTTPException
from loguru import logger
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from ya_agent_stream_protocol.sdk import AguiAdapterConfig, AguiEventAdapter

from ya_claw.config import ClawSettings
from ya_claw.controller.hitl import HitlController
from ya_claw.controller.models import (
    ControlResponse,
    InteractionRespondRequest,
    InteractionRespondResponse,
    RunCreateRequest,
    RunDetail,
    RunGetResponse,
    RunStatus,
    RunSummary,
    RunTraceItem,
    RunTraceResponse,
    SessionSummary,
    SteerRequest,
    TerminationReason,
    active_interactions_from_run_record,
    run_detail_from_record,
    run_summary_from_record,
    session_summary_from_record,
)
from ya_claw.controller.session_lifecycle import (
    SESSION_PRUNE_CLAIM,
    SessionPruneClaimedError,
    lock_session_reference,
)
from ya_claw.controller.store import (
    ensure_run_dir,
    read_run_message_blob_if_exists,
    read_run_state_blob_if_exists,
)
from ya_claw.execution.state_machine import cancel_run, interrupt_run, queue_run
from ya_claw.orm.tables import RunRecord, SessionRecord
from ya_claw.runtime_state import InMemoryRuntimeState
from ya_claw.workspace.models import metadata_with_workspace

_ACTIVE_RUN_STATUSES = frozenset({RunStatus.QUEUED, RunStatus.RUNNING})
CLAW_AGUI_ADAPTER_CONFIG = AguiAdapterConfig(run_event_prefix="ya_claw")


class RunController:
    def __init__(self) -> None:
        self._hitl_controller = HitlController()

    async def create(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        runtime_state: InMemoryRuntimeState,
        request: RunCreateRequest,
    ) -> RunDetail:
        session_id = request.session_id
        if session_id is None:
            if request.restore_from_run_id is not None:
                raise HTTPException(
                    status_code=422,
                    detail="session_id is required when restore_from_run_id is provided.",
                )
            session_id = uuid4().hex
            logger.debug("Creating session for standalone run session_id={}", session_id)
            session_record = SessionRecord(
                id=session_id,
                profile_name=request.profile_name,
                session_metadata=metadata_with_workspace({}, request.workspace),
                session_type="conversation",
            )
            db_session.add(session_record)
        else:
            await db_session.execute(
                update(SessionRecord)
                .where(
                    SessionRecord.id == session_id,
                    SessionRecord.active_run_id.is_(None),
                )
                .values(
                    active_run_id=None,
                    updated_at=SessionRecord.updated_at,
                )
            )
            session_record = await db_session.get(
                SessionRecord,
                session_id,
                populate_existing=True,
                with_for_update=True,
            )
            if not isinstance(session_record, SessionRecord):
                raise HTTPException(status_code=404, detail=f"Session '{session_id}' was not found.")
            if isinstance(session_record.active_run_id, str):
                if session_record.active_run_id == SESSION_PRUNE_CLAIM:
                    raise HTTPException(
                        status_code=409,
                        detail=f"Session '{session_id}' is being pruned and cannot accept a new run.",
                    )
                raise HTTPException(
                    status_code=409,
                    detail=f"Session '{session_id}' already has an active run '{session_record.active_run_id}'.",
                )

        if request.reset_state and request.restore_from_run_id is not None:
            raise HTTPException(
                status_code=422,
                detail="reset_state and restore_from_run_id cannot be used together.",
            )

        restore_from_run_id = None
        if not request.reset_state:
            restore_from_run_id = request.restore_from_run_id or session_record.head_success_run_id
        if restore_from_run_id is not None:
            await self._validate_restore_source(db_session, session_id, restore_from_run_id)

        logger.debug(
            "Creating run session_id={} profile={} dispatch_mode={} trigger_type={} restore_from_run_id={} reset_state={}",
            session_id,
            request.profile_name or session_record.profile_name,
            request.dispatch_mode,
            request.trigger_type,
            restore_from_run_id,
            request.reset_state,
        )
        run_metadata = metadata_with_workspace(request.metadata, request.workspace)
        if request.reset_state:
            run_metadata["restore_state"] = False

        sequence_no = await self._next_sequence_no(db_session, session_id)
        run_id = uuid4().hex
        run_record = RunRecord(
            id=run_id,
            session_id=session_id,
            sequence_no=sequence_no,
            restore_from_run_id=restore_from_run_id,
            status=RunStatus.QUEUED,
            trigger_type=request.trigger_type,
            profile_name=request.profile_name or session_record.profile_name,
            input_parts=[part.model_dump(mode="json") for part in request.input_parts],
            run_metadata=run_metadata,
        )
        db_session.add(run_record)
        queue_run(session_record, run_record)

        await db_session.commit()
        await db_session.refresh(run_record)

        ensure_run_dir(settings, run_id)
        runtime_state.register_run(session_id, run_id, dispatch_mode=request.dispatch_mode)
        agui_adapter = AguiEventAdapter(session_id=session_id, run_id=run_id, config=CLAW_AGUI_ADAPTER_CONFIG)
        await runtime_state.append_run_event(
            run_id,
            agui_adapter.build_run_custom_event(
                "run_queued",
                {
                    "run_id": run_id,
                    "session_id": session_id,
                    "status": run_record.status,
                    "sequence_no": sequence_no,
                    "dispatch_mode": request.dispatch_mode,
                },
            ),
        )

        logger.info(
            "Run queued run_id={} session_id={} sequence_no={} dispatch_mode={}",
            run_id,
            session_id,
            sequence_no,
            request.dispatch_mode,
        )
        return run_detail_from_record(run_record)

    async def get(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        run_id: str,
        *,
        include_state: bool,
        include_message: bool,
    ) -> RunGetResponse:
        logger.debug(
            "Fetching run detail run_id={} include_state={} include_message={}",
            run_id,
            include_state,
            include_message,
        )
        run_record = await db_session.get(RunRecord, run_id)
        if not isinstance(run_record, RunRecord):
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' was not found.")

        session_record = await db_session.get(SessionRecord, run_record.session_id)
        if not isinstance(session_record, SessionRecord):
            raise HTTPException(status_code=404, detail=f"Session '{run_record.session_id}' was not found.")

        state_payload = read_run_state_blob_if_exists(settings, run_id)
        message_events = read_run_message_blob_if_exists(settings, run_id)
        has_state = state_payload is not None
        has_message = message_events is not None

        return RunGetResponse(
            session=await self._build_session_summary(db_session, session_record),
            run=run_detail_from_record(run_record, has_state=has_state, has_message=has_message),
            state=state_payload if include_state else None,
            message=message_events if include_message else None,
        )

    def build_session_run_summary(
        self,
        settings: ClawSettings,
        run_record: RunRecord,
        *,
        include_message: bool,
        include_input_parts: bool = False,
    ) -> RunSummary:
        message_payload = read_run_message_blob_if_exists(settings, run_record.id) if include_message else None
        return run_summary_from_record(
            run_record,
            message=message_payload,
            include_input_parts=include_input_parts,
        )

    async def get_trace(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        run_id: str,
        *,
        max_item_chars: int = 4000,
        max_total_chars: int = 12000,
    ) -> RunTraceResponse:
        run_record = await db_session.get(RunRecord, run_id)
        if not isinstance(run_record, RunRecord):
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' was not found.")

        normalized_item_chars = min(max(max_item_chars, 256), 20000)
        normalized_total_chars = min(max(max_total_chars, normalized_item_chars), 100000)
        message_payload = read_run_message_blob_if_exists(settings, run_id) or []
        trace, truncated = _project_run_trace(
            message_payload,
            max_item_chars=normalized_item_chars,
            max_total_chars=normalized_total_chars,
        )
        logger.debug(
            "Built run trace run_id={} item_count={} truncated={}",
            run_id,
            len(trace),
            truncated,
        )
        return RunTraceResponse(
            run_id=run_record.id,
            session_id=run_record.session_id,
            item_count=len(trace),
            max_item_chars=normalized_item_chars,
            max_total_chars=normalized_total_chars,
            truncated=truncated,
            trace=trace,
        )

    async def cancel(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        runtime_state: InMemoryRuntimeState,
        run_id: str,
    ) -> RunDetail:
        return await self._stop_run(
            db_session,
            settings,
            runtime_state,
            run_id,
            event_type="run.cancelled",
            termination_reason=TerminationReason.CANCEL,
        )

    async def interrupt(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        runtime_state: InMemoryRuntimeState,
        run_id: str,
    ) -> RunDetail:
        return await self._stop_run(
            db_session,
            settings,
            runtime_state,
            run_id,
            event_type="run.interrupted",
            termination_reason=TerminationReason.INTERRUPT,
        )

    async def steer(
        self,
        db_session: AsyncSession,
        runtime_state: InMemoryRuntimeState,
        run_id: str,
        request: SteerRequest,
    ) -> ControlResponse:
        logger.debug("Steering run run_id={} input_parts={}", run_id, len(request.input_parts))
        run_record = await db_session.get(RunRecord, run_id)
        if not isinstance(run_record, RunRecord):
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' was not found.")
        if run_record.status not in _ACTIVE_RUN_STATUSES:
            raise HTTPException(status_code=409, detail=f"Run '{run_id}' is not active.")
        if not request.input_parts:
            raise HTTPException(status_code=422, detail="input_parts must not be empty for steer requests.")

        input_payload = [part.model_dump(mode="json") for part in request.input_parts]
        try:
            await runtime_state.record_steering(run_id, input_payload)
            agui_adapter = AguiEventAdapter(
                session_id=run_record.session_id, run_id=run_id, config=CLAW_AGUI_ADAPTER_CONFIG
            )
            await runtime_state.append_run_event(
                run_id,
                agui_adapter.build_run_custom_event(
                    "run_steered",
                    {
                        "run_id": run_id,
                        "session_id": run_record.session_id,
                        "input_parts": input_payload,
                    },
                ),
            )
        except KeyError as exc:
            raise HTTPException(status_code=409, detail=f"Run '{run_id}' is not active in runtime state.") from exc

        return ControlResponse(
            session_id=run_record.session_id,
            run_id=run_id,
            status=RunStatus(run_record.status),
        )

    async def respond_interaction(
        self,
        db_session: AsyncSession,
        runtime_state: InMemoryRuntimeState,
        run_id: str,
        interaction_id: str,
        request: InteractionRespondRequest,
    ) -> InteractionRespondResponse:
        response = await self._hitl_controller.respond_interaction(
            db_session,
            runtime_state,
            run_id,
            interaction_id,
            request,
        )
        await db_session.commit()
        return response

    async def _stop_run(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        runtime_state: InMemoryRuntimeState,
        run_id: str,
        *,
        event_type: str,
        termination_reason: TerminationReason,
    ) -> RunDetail:
        logger.info("Stopping run run_id={} termination_reason={}", run_id, termination_reason)
        run_record = await db_session.get(RunRecord, run_id)
        if not isinstance(run_record, RunRecord):
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' was not found.")

        session_record = await db_session.get(SessionRecord, run_record.session_id)
        if not isinstance(session_record, SessionRecord):
            raise HTTPException(status_code=404, detail=f"Session '{run_record.session_id}' was not found.")

        if run_record.status in _ACTIVE_RUN_STATUSES:
            with suppress(KeyError):
                await runtime_state.request_stop(run_id, termination_reason)
            if termination_reason == TerminationReason.INTERRUPT:
                interrupt_run(session_record, run_record)
            else:
                cancel_run(session_record, run_record)
            await db_session.commit()
            await db_session.refresh(run_record)
            logger.info(
                "Run stopped run_id={} session_id={} status={} termination_reason={}",
                run_id,
                run_record.session_id,
                run_record.status,
                run_record.termination_reason,
            )

        await self._emit_terminal_event(runtime_state, run_record, event_type)

        state_payload = read_run_state_blob_if_exists(settings, run_id)
        message_payload = read_run_message_blob_if_exists(settings, run_id)
        return run_detail_from_record(
            run_record,
            has_state=state_payload is not None,
            has_message=message_payload is not None,
        )

    async def _emit_terminal_event(
        self,
        runtime_state: InMemoryRuntimeState,
        run_record: RunRecord,
        event_type: str,
    ) -> None:
        agui_adapter = AguiEventAdapter(
            session_id=run_record.session_id, run_id=run_record.id, config=CLAW_AGUI_ADAPTER_CONFIG
        )
        event_payload = {
            "run_id": run_record.id,
            "session_id": run_record.session_id,
            "status": run_record.status,
            "termination_reason": run_record.termination_reason,
        }
        mapped_event = (
            agui_adapter.build_run_custom_event("run_interrupted", event_payload)
            if event_type == "run.interrupted"
            else agui_adapter.build_run_custom_event("run_cancelled", event_payload)
        )
        try:
            await runtime_state.append_run_event(
                run_record.id,
                mapped_event,
                terminal=True,
            )
        except KeyError:
            await runtime_state.close_run(run_record.id)

    async def _validate_restore_source(
        self,
        db_session: AsyncSession,
        session_id: str,
        restore_from_run_id: str,
    ) -> RunRecord:
        restore_record = await db_session.get(RunRecord, restore_from_run_id)
        if not isinstance(restore_record, RunRecord):
            raise HTTPException(status_code=404, detail=f"Run '{restore_from_run_id}' was not found.")
        restore_session_id = restore_record.session_id
        if restore_session_id == session_id:
            return restore_record

        session_record = await db_session.get(SessionRecord, session_id)
        ancestor_session_id = session_record.parent_session_id if isinstance(session_record, SessionRecord) else None
        while isinstance(ancestor_session_id, str) and restore_session_id != ancestor_session_id:
            ancestor_record = await db_session.get(SessionRecord, ancestor_session_id)
            ancestor_session_id = (
                ancestor_record.parent_session_id if isinstance(ancestor_record, SessionRecord) else None
            )
        if restore_session_id != ancestor_session_id:
            raise HTTPException(
                status_code=422,
                detail=f"Run '{restore_from_run_id}' is not a valid restore source for session '{session_id}'.",
            )

        try:
            restore_session = await lock_session_reference(db_session, restore_session_id)
        except SessionPruneClaimedError as exc:
            raise HTTPException(
                status_code=409,
                detail=f"Restore source session '{restore_session_id}' is being pruned.",
            ) from exc
        if not isinstance(restore_session, SessionRecord):
            raise HTTPException(
                status_code=409,
                detail=f"Restore source session '{restore_session_id}' is no longer available.",
            )
        locked_restore_record = await db_session.get(
            RunRecord,
            restore_from_run_id,
            populate_existing=True,
        )
        if not isinstance(locked_restore_record, RunRecord) or locked_restore_record.session_id != restore_session_id:
            raise HTTPException(
                status_code=409,
                detail=f"Restore source run '{restore_from_run_id}' is no longer available.",
            )
        return locked_restore_record

    async def _build_session_summary(
        self,
        db_session: AsyncSession,
        session_record: SessionRecord,
    ) -> SessionSummary:
        run_count_statement = select(func.count()).where(RunRecord.session_id == session_record.id)
        run_count_result = await db_session.execute(run_count_statement)
        run_count = run_count_result.scalar_one()

        latest_run_statement = (
            select(RunRecord)
            .where(RunRecord.session_id == session_record.id)
            .order_by(RunRecord.sequence_no.desc(), RunRecord.id.desc())
            .limit(1)
        )
        latest_run_result = await db_session.execute(latest_run_statement)
        latest_run_record = latest_run_result.scalar_one_or_none()
        latest_run = run_summary_from_record(latest_run_record) if isinstance(latest_run_record, RunRecord) else None
        active_interactions = (
            active_interactions_from_run_record(latest_run_record) if isinstance(latest_run_record, RunRecord) else None
        )
        return session_summary_from_record(
            session_record,
            run_count=run_count,
            latest_run=latest_run,
            active_interactions=active_interactions,
        )

    async def _next_sequence_no(self, db_session: AsyncSession, session_id: str) -> int:
        statement = select(func.max(RunRecord.sequence_no)).where(RunRecord.session_id == session_id)
        result = await db_session.execute(statement)
        max_value = result.scalar_one_or_none()
        if isinstance(max_value, int):
            return max_value + 1
        return 1


def _project_run_trace(
    events: list[dict[str, Any]],
    *,
    max_item_chars: int,
    max_total_chars: int,
) -> tuple[list[RunTraceItem], bool]:
    trace: list[RunTraceItem] = []
    consumed_chars = 0
    truncated = False

    for event in events:
        event_type = str(event.get("type", "")).strip()
        item_type = _trace_item_type(event_type)
        if item_type is None:
            continue

        raw_content = _trace_content(event, item_type)
        remaining_chars = max_total_chars - consumed_chars
        if remaining_chars <= 0:
            truncated = True
            break

        item_limit = min(max_item_chars, remaining_chars)
        content, item_truncated = _truncate_text(raw_content, item_limit)
        consumed_chars += len(content or "")
        truncated = truncated or item_truncated
        trace.append(
            RunTraceItem(
                sequence_no=len(trace) + 1,
                type=item_type,
                tool_call_id=_string_field(event, "toolCallId", "tool_call_id"),
                tool_name=_string_field(event, "toolCallName", "tool_call_name"),
                message_id=_string_field(event, "messageId", "message_id"),
                role=_string_field(event, "role"),
                content=content,
                truncated=item_truncated,
            )
        )
        if item_truncated and consumed_chars >= max_total_chars:
            truncated = True
            break

    return trace, truncated


def _trace_item_type(event_type: str) -> Literal["tool_call", "tool_response"] | None:
    if event_type == "TOOL_CALL_CHUNK":
        return "tool_call"
    if event_type == "TOOL_CALL_RESULT":
        return "tool_response"
    return None


def _trace_content(event: dict[str, Any], item_type: str) -> str | None:
    value = event.get("delta") if item_type == "tool_call" else event.get("content")
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _truncate_text(value: str | None, limit: int) -> tuple[str | None, bool]:
    if value is None:
        return None, False
    if len(value) <= limit:
        return value, False
    if limit <= 0:
        return "", True
    return value[:limit], True


def _string_field(event: dict[str, Any], *names: str) -> str | None:
    for name in names:
        value = event.get(name)
        if isinstance(value, str) and value.strip() != "":
            return value
    return None
