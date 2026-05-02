from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, Literal, cast
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import HTTPException
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import Select, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ya_claw.config import ClawSettings
from ya_claw.controller.models import (
    DispatchMode,
    RunCreateRequest,
    SteerRequest,
    TextPart,
    TriggerType,
    parse_input_parts,
)
from ya_claw.controller.run import RunController
from ya_claw.controller.session import SessionController
from ya_claw.execution.dispatcher import RunDispatcher
from ya_claw.orm.tables import RunRecord, ScheduleFireRecord, ScheduleRecord, SessionRecord, utc_now
from ya_claw.runtime_state import InMemoryRuntimeState

ScheduleStatus = Literal["active", "paused", "completed", "deleted"]
ScheduleTriggerKind = Literal["cron", "once"]
ScheduleExecutionMode = Literal["continue_session", "fork_session", "isolate_session"]
ScheduleActivePolicy = Literal["steer", "queue"]
ScheduleFireStatus = Literal["pending", "submitted", "steered", "skipped", "failed"]
ScheduleRunStatus = Literal["queued", "running", "completed", "failed", "cancelled"]


class ScheduleCreateRequest(BaseModel):
    name: str
    description: str | None = None
    prompt: str
    trigger_kind: ScheduleTriggerKind | None = None
    cron: str | None = None
    run_at: datetime | None = None
    timezone: str = "UTC"
    enabled: bool = True
    continue_current_session: bool = False
    start_from_current_session: bool = False
    steer_when_running: bool = False
    owner_kind: Literal["api", "user", "agent"] = "api"
    owner_session_id: str | None = None
    owner_run_id: str | None = None
    profile_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_prompt_and_trigger(self) -> ScheduleCreateRequest:
        if self.name.strip() == "":
            raise ValueError("name is required")
        if self.prompt.strip() == "":
            raise ValueError("prompt is required")
        trigger_kind = infer_trigger_kind(self.trigger_kind, self.cron, self.run_at)
        if trigger_kind == "cron" and (not isinstance(self.cron, str) or self.cron.strip() == ""):
            raise ValueError("cron is required for cron schedules")
        if trigger_kind == "once" and self.run_at is None:
            raise ValueError("run_at is required for one-time schedules")
        self.trigger_kind = trigger_kind
        return self


class ScheduleUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    prompt: str | None = None
    trigger_kind: ScheduleTriggerKind | None = None
    cron: str | None = None
    run_at: datetime | None = None
    timezone: str | None = None
    enabled: bool | None = None
    continue_current_session: bool | None = None
    start_from_current_session: bool | None = None
    steer_when_running: bool | None = None
    metadata: dict[str, Any] | None = None


class ScheduleManualTriggerRequest(BaseModel):
    prompt_override: str | None = None


class ScheduleFireSummary(BaseModel):
    id: str
    schedule_id: str
    scheduled_at: datetime
    fired_at: datetime | None = None
    status: ScheduleFireStatus
    target_session_id: str | None = None
    source_session_id: str | None = None
    created_session_id: str | None = None
    run_id: str | None = None
    active_run_id: str | None = None
    run_status: ScheduleRunStatus | None = None
    input_preview: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime


class ScheduleSummary(BaseModel):
    id: str
    name: str
    description: str | None = None
    enabled: bool
    status: ScheduleStatus
    prompt: str
    trigger: dict[str, Any]
    cron: dict[str, Any]
    mode: dict[str, bool]
    execution_mode: ScheduleExecutionMode
    owner_kind: str
    owner_session_id: str | None = None
    owner_run_id: str | None = None
    profile_name: str | None = None
    target_session_id: str | None = None
    source_session_id: str | None = None
    last_fire: ScheduleFireSummary | None = None
    fire_count: int = 0
    failure_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class ScheduleListResponse(BaseModel):
    schedules: list[ScheduleSummary] = Field(default_factory=list)


class ScheduleFireListResponse(BaseModel):
    fires: list[ScheduleFireSummary] = Field(default_factory=list)


class ScheduleController:
    def __init__(self) -> None:
        self._run_controller = RunController()
        self._session_controller = SessionController()

    async def list(
        self,
        db_session: AsyncSession,
        *,
        include_deleted: bool = False,
        owner_session_id: str | None = None,
        schedule_id: str | None = None,
        limit: int = 100,
        include_recent_runs: bool = True,
    ) -> ScheduleListResponse:
        normalized_limit = min(max(limit, 1), 500)
        statement: Select[tuple[ScheduleRecord]] = select(ScheduleRecord)
        if isinstance(schedule_id, str) and schedule_id.strip() != "":
            statement = statement.where(ScheduleRecord.id == schedule_id)
        if not include_deleted:
            statement = statement.where(ScheduleRecord.status != "deleted")
        if isinstance(owner_session_id, str) and owner_session_id.strip() != "":
            statement = statement.where(ScheduleRecord.owner_session_id == owner_session_id)
        statement = statement.order_by(ScheduleRecord.updated_at.desc()).limit(normalized_limit)
        result = await db_session.execute(statement)
        records = list(result.scalars().all())
        summaries = [
            await self._summary_from_record(db_session, record, include_recent_runs=include_recent_runs)
            for record in records
        ]
        return ScheduleListResponse(schedules=summaries)

    async def get(self, db_session: AsyncSession, schedule_id: str) -> ScheduleSummary:
        record = await self._get_schedule_record(db_session, schedule_id)
        return await self._summary_from_record(db_session, record, include_recent_runs=True)

    async def create(self, db_session: AsyncSession, request: ScheduleCreateRequest) -> ScheduleSummary:
        now = utc_now()
        execution_mode, target_session_id, source_session_id, on_active = _resolve_facade_mode(
            continue_current_session=request.continue_current_session,
            start_from_current_session=request.start_from_current_session,
            steer_when_running=request.steer_when_running,
            current_session_id=request.owner_session_id,
        )
        if execution_mode in {"continue_session", "fork_session"}:
            session_id = target_session_id if execution_mode == "continue_session" else source_session_id
            await self._require_session(db_session, session_id)

        trigger_kind = infer_trigger_kind(request.trigger_kind, request.cron, request.run_at)
        timezone = normalize_timezone(request.timezone)
        record = ScheduleRecord(
            id=uuid4().hex,
            name=request.name.strip(),
            description=_clean_optional(request.description),
            status="active" if request.enabled else "paused",
            owner_kind=request.owner_kind,
            owner_session_id=request.owner_session_id,
            owner_run_id=request.owner_run_id,
            profile_name=_clean_optional(request.profile_name),
            trigger_kind=trigger_kind,
            cron_expr=_clean_optional(request.cron) if trigger_kind == "cron" else None,
            run_at=_as_utc_aware(request.run_at) if trigger_kind == "once" and request.run_at is not None else None,
            timezone=timezone,
            next_fire_at=None,
            execution_mode=execution_mode,
            target_session_id=target_session_id,
            source_session_id=source_session_id,
            on_active=on_active,
            input_parts_template=[TextPart(type="text", text=request.prompt).model_dump(mode="json")],
            schedule_metadata=dict(request.metadata),
        )
        if request.enabled:
            record.next_fire_at = compute_schedule_next_fire_at(record, now=now)
        db_session.add(record)
        await db_session.commit()
        await db_session.refresh(record)
        return await self._summary_from_record(db_session, record, include_recent_runs=True)

    async def update(  # noqa: C901
        self,
        db_session: AsyncSession,
        schedule_id: str,
        request: ScheduleUpdateRequest,
    ) -> ScheduleSummary:
        record = await self._get_schedule_record(db_session, schedule_id)
        if isinstance(request.name, str):
            normalized_name = request.name.strip()
            if normalized_name == "":
                raise HTTPException(status_code=422, detail="name is required.")
            record.name = normalized_name
        if request.description is not None:
            record.description = _clean_optional(request.description)
        if isinstance(request.prompt, str):
            if request.prompt.strip() == "":
                raise HTTPException(status_code=422, detail="prompt is required.")
            record.input_parts_template = [TextPart(type="text", text=request.prompt).model_dump(mode="json")]
        if request.trigger_kind is not None:
            record.trigger_kind = request.trigger_kind
            if record.trigger_kind == "cron":
                record.run_at = None
            else:
                record.cron_expr = None
        if isinstance(request.cron, str):
            if request.cron.strip() == "":
                raise HTTPException(status_code=422, detail="cron is required.")
            record.cron_expr = request.cron.strip()
            if request.trigger_kind is None and request.run_at is None:
                record.trigger_kind = "cron"
        if request.run_at is not None:
            record.run_at = _as_utc_aware(request.run_at)
            if request.trigger_kind is None and request.cron is None:
                record.trigger_kind = "once"
        if isinstance(request.timezone, str):
            record.timezone = normalize_timezone(request.timezone)
        if request.metadata is not None:
            record.schedule_metadata = dict(request.metadata)
        if (
            request.continue_current_session is not None
            or request.start_from_current_session is not None
            or request.steer_when_running is not None
        ):
            current_continue, current_start, current_steer = _facade_flags_from_record(record)
            execution_mode, target_session_id, source_session_id, on_active = _resolve_facade_mode(
                continue_current_session=current_continue
                if request.continue_current_session is None
                else request.continue_current_session,
                start_from_current_session=current_start
                if request.start_from_current_session is None
                else request.start_from_current_session,
                steer_when_running=current_steer if request.steer_when_running is None else request.steer_when_running,
                current_session_id=record.owner_session_id,
            )
            if execution_mode in {"continue_session", "fork_session"}:
                session_id = target_session_id if execution_mode == "continue_session" else source_session_id
                await self._require_session(db_session, session_id)
            record.execution_mode = execution_mode
            record.target_session_id = target_session_id
            record.source_session_id = source_session_id
            record.on_active = on_active
        validate_schedule_trigger(record)
        if request.enabled is not None:
            record.status = "active" if request.enabled else "paused"
        elif record.status == "completed" and record.trigger_kind == "once":
            record.status = "active"
        if record.status == "active":
            record.next_fire_at = compute_schedule_next_fire_at(record, now=utc_now())
        else:
            record.next_fire_at = None
        record.updated_at = utc_now()
        await db_session.commit()
        await db_session.refresh(record)
        return await self._summary_from_record(db_session, record, include_recent_runs=True)

    async def delete(self, db_session: AsyncSession, schedule_id: str) -> ScheduleSummary:
        record = await self._get_schedule_record(db_session, schedule_id)
        record.status = "deleted"
        record.next_fire_at = None
        record.updated_at = utc_now()
        await db_session.commit()
        await db_session.refresh(record)
        return await self._summary_from_record(db_session, record, include_recent_runs=True)

    async def list_fires(
        self,
        db_session: AsyncSession,
        schedule_id: str,
        *,
        limit: int = 50,
    ) -> ScheduleFireListResponse:
        await self._get_schedule_record(db_session, schedule_id)
        normalized_limit = min(max(limit, 1), 200)
        statement = (
            select(ScheduleFireRecord, RunRecord.status)
            .outerjoin(RunRecord, ScheduleFireRecord.run_id == RunRecord.id)
            .where(ScheduleFireRecord.schedule_id == schedule_id)
            .order_by(ScheduleFireRecord.created_at.desc())
            .limit(normalized_limit)
        )
        result = await db_session.execute(statement)
        return ScheduleFireListResponse(
            fires=[
                fire_summary_from_record(record, run_status=run_status)
                for record, run_status in result.all()
                if isinstance(record, ScheduleFireRecord)
            ]
        )

    async def trigger(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        runtime_state: InMemoryRuntimeState,
        dispatcher: RunDispatcher,
        schedule_id: str,
        request: ScheduleManualTriggerRequest | None = None,
    ) -> ScheduleFireSummary:
        record = await self._get_schedule_record(db_session, schedule_id)
        fire_record = await self._create_fire(
            db_session,
            record,
            scheduled_at=utc_now(),
            manual=True,
            prompt_override=request.prompt_override if request is not None else None,
        )
        await self.dispatch_fire(db_session, settings, runtime_state, dispatcher, record, fire_record)
        await db_session.commit()
        await db_session.refresh(fire_record)
        return fire_summary_from_record(fire_record)

    async def dispatch_due(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        runtime_state: InMemoryRuntimeState,
        dispatcher: RunDispatcher,
        *,
        now: datetime | None = None,
        limit: int = 20,
    ) -> list[ScheduleFireSummary]:
        effective_now = _as_utc_aware(now or utc_now())
        statement = (
            select(ScheduleRecord)
            .where(
                ScheduleRecord.status == "active",
                ScheduleRecord.next_fire_at.is_not(None),
            )
            .order_by(ScheduleRecord.next_fire_at.asc(), ScheduleRecord.id.asc())
            .limit(max(limit, 1))
        )
        result = await db_session.execute(statement)
        fired: list[ScheduleFireSummary] = []
        for record in result.scalars().all():
            next_fire_at = _as_utc_aware(record.next_fire_at) if record.next_fire_at is not None else None
            if next_fire_at is None or next_fire_at > effective_now:
                continue
            scheduled_at = next_fire_at
            fire_record = await self._create_fire(db_session, record, scheduled_at=scheduled_at, manual=False)
            await self.dispatch_fire(db_session, settings, runtime_state, dispatcher, record, fire_record)
            self._advance_schedule_after_fire(record, fire_record, now=effective_now)
            record.updated_at = utc_now()
            fired.append(fire_summary_from_record(fire_record))
        await db_session.commit()
        return fired

    async def dispatch_pending(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        runtime_state: InMemoryRuntimeState,
        dispatcher: RunDispatcher,
        *,
        now: datetime | None = None,
        limit: int = 20,
    ) -> list[ScheduleFireSummary]:
        effective_now = _as_utc_aware(now or utc_now())
        statement = (
            select(ScheduleFireRecord)
            .where(ScheduleFireRecord.status == "pending")
            .order_by(ScheduleFireRecord.scheduled_at.asc(), ScheduleFireRecord.id.asc())
            .limit(max(limit, 1))
        )
        result = await db_session.execute(statement)
        fired: list[ScheduleFireSummary] = []
        for fire_record in result.scalars().all():
            if _as_utc_aware(fire_record.scheduled_at) > effective_now:
                continue
            record = await db_session.get(ScheduleRecord, fire_record.schedule_id)
            if not isinstance(record, ScheduleRecord) or record.status == "deleted":
                fire_record.status = "skipped"
                fire_record.error_message = "Schedule is no longer available."
                continue
            await self.dispatch_fire(db_session, settings, runtime_state, dispatcher, record, fire_record)
            if record.trigger_kind == "once" and fire_record.status != "pending":
                record.status = "completed"
                record.next_fire_at = None
                record.updated_at = utc_now()
            fired.append(fire_summary_from_record(fire_record))
        await db_session.commit()
        return fired

    async def dispatch_fire(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        runtime_state: InMemoryRuntimeState,
        dispatcher: RunDispatcher,
        record: ScheduleRecord,
        fire_record: ScheduleFireRecord,
    ) -> None:
        fire_record.fired_at = utc_now()
        fire_record.target_session_id = record.target_session_id
        fire_record.source_session_id = record.source_session_id
        try:
            if record.execution_mode == "continue_session":
                await self._dispatch_continue_session(
                    db_session, settings, runtime_state, dispatcher, record, fire_record
                )
            elif record.execution_mode == "fork_session":
                await self._dispatch_fork_session(db_session, settings, runtime_state, dispatcher, record, fire_record)
            else:
                await self._dispatch_isolate_session(
                    db_session, settings, runtime_state, dispatcher, record, fire_record
                )
        except HTTPException as exc:
            fire_record.status = "failed"
            fire_record.error_message = str(exc.detail)
            record.failure_count += 1
        except Exception as exc:
            fire_record.status = "failed"
            fire_record.error_message = str(exc)
            record.failure_count += 1
        record.last_fire_at = fire_record.fired_at
        record.last_fire_id = fire_record.id
        record.last_session_id = fire_record.created_session_id
        record.last_run_id = fire_record.run_id
        record.fire_count += 1
        record.updated_at = utc_now()

    async def _dispatch_continue_session(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        runtime_state: InMemoryRuntimeState,
        dispatcher: RunDispatcher,
        record: ScheduleRecord,
        fire_record: ScheduleFireRecord,
    ) -> None:
        if not isinstance(record.target_session_id, str):
            raise HTTPException(status_code=422, detail="target_session_id is required.")
        session = await self._require_session(db_session, record.target_session_id)
        if isinstance(session.active_run_id, str):
            if record.on_active == "steer":
                await self._run_controller.steer(
                    db_session,
                    runtime_state,
                    session.active_run_id,
                    SteerRequest(input_parts=parse_input_parts(list(fire_record.input_parts))),
                )
                fire_record.status = "steered"
                fire_record.active_run_id = session.active_run_id
                return
            fire_record.status = "pending"
            fire_record.error_message = "Target session is active; queued for later delivery."
            return
        run = await self._run_controller.create(
            db_session,
            settings,
            runtime_state,
            RunCreateRequest(
                session_id=session.id,
                reset_state=True,
                profile_name=record.profile_name or session.profile_name,
                input_parts=parse_input_parts(list(fire_record.input_parts)),
                trigger_type=TriggerType.SCHEDULE,
                metadata=_source_metadata(record, fire_record),
                dispatch_mode=DispatchMode.ASYNC,
            ),
        )
        self._apply_dispatch_result(fire_record, dispatcher.dispatch(run.id, DispatchMode.ASYNC))
        fire_record.created_session_id = run.session_id
        fire_record.run_id = run.id

    async def _dispatch_fork_session(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        runtime_state: InMemoryRuntimeState,
        dispatcher: RunDispatcher,
        record: ScheduleRecord,
        fire_record: ScheduleFireRecord,
    ) -> None:
        if not isinstance(record.source_session_id, str):
            raise HTTPException(status_code=422, detail="source_session_id is required.")
        source_session = await self._require_session(db_session, record.source_session_id)
        fork_session = SessionRecord(
            id=uuid4().hex,
            parent_session_id=source_session.id,
            profile_name=record.profile_name or source_session.profile_name,
            session_metadata={"source": "schedule", "schedule_id": record.id, "schedule_fire_id": fire_record.id},
        )
        db_session.add(fork_session)
        await db_session.flush()
        run = await self._run_controller.create(
            db_session,
            settings,
            runtime_state,
            RunCreateRequest(
                session_id=fork_session.id,
                reset_state=True,
                profile_name=fork_session.profile_name,
                input_parts=parse_input_parts(list(fire_record.input_parts)),
                trigger_type=TriggerType.SCHEDULE,
                metadata=_source_metadata(record, fire_record),
                dispatch_mode=DispatchMode.ASYNC,
            ),
        )
        self._apply_dispatch_result(fire_record, dispatcher.dispatch(run.id, DispatchMode.ASYNC))
        fire_record.created_session_id = fork_session.id
        fire_record.run_id = run.id

    async def _dispatch_isolate_session(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        runtime_state: InMemoryRuntimeState,
        dispatcher: RunDispatcher,
        record: ScheduleRecord,
        fire_record: ScheduleFireRecord,
    ) -> None:
        run = await self._run_controller.create(
            db_session,
            settings,
            runtime_state,
            RunCreateRequest(
                session_id=None,
                reset_state=True,
                profile_name=record.profile_name,
                input_parts=parse_input_parts(list(fire_record.input_parts)),
                trigger_type=TriggerType.SCHEDULE,
                metadata=_source_metadata(record, fire_record),
                dispatch_mode=DispatchMode.ASYNC,
            ),
        )
        self._apply_dispatch_result(fire_record, dispatcher.dispatch(run.id, DispatchMode.ASYNC))
        fire_record.created_session_id = run.session_id
        fire_record.run_id = run.id

    def _apply_dispatch_result(self, fire_record: ScheduleFireRecord, dispatch_result: object) -> None:
        if getattr(dispatch_result, "submitted", False):
            fire_record.status = "submitted"
            fire_record.error_message = None
            return
        fire_record.status = "pending"
        fire_record.error_message = f"Dispatch skipped: {getattr(dispatch_result, 'reason', None)}"

    async def _create_fire(
        self,
        db_session: AsyncSession,
        record: ScheduleRecord,
        *,
        scheduled_at: datetime,
        manual: bool,
        prompt_override: str | None = None,
    ) -> ScheduleFireRecord:
        input_parts = (
            [TextPart(type="text", text=prompt_override).model_dump(mode="json")]
            if prompt_override
            else list(record.input_parts_template)
        )
        fire_record = ScheduleFireRecord(
            id=uuid4().hex,
            schedule_id=record.id,
            scheduled_at=scheduled_at,
            status="pending",
            dedupe_key=f"{record.id}:{'manual' if manual else scheduled_at.isoformat()}:{uuid4().hex if manual else ''}",
            target_session_id=record.target_session_id,
            source_session_id=record.source_session_id,
            input_parts=input_parts,
            fire_metadata={"manual": manual},
        )
        db_session.add(fire_record)
        try:
            await db_session.flush()
        except IntegrityError:
            await db_session.rollback()
            result = await db_session.execute(
                select(ScheduleFireRecord).where(
                    ScheduleFireRecord.schedule_id == record.id,
                    ScheduleFireRecord.dedupe_key == fire_record.dedupe_key,
                )
            )
            existing = result.scalar_one_or_none()
            if isinstance(existing, ScheduleFireRecord):
                return existing
            raise
        return fire_record

    def _advance_schedule_after_fire(
        self,
        record: ScheduleRecord,
        fire_record: ScheduleFireRecord,
        *,
        now: datetime,
    ) -> None:
        if record.trigger_kind == "cron":
            record.next_fire_at = compute_schedule_next_fire_at(record, now=now)
            return
        if record.trigger_kind == "once" and fire_record.status == "pending":
            record.next_fire_at = _as_utc_aware(fire_record.scheduled_at)
            return
        record.status = "completed"
        record.next_fire_at = None

    async def _summary_from_record(
        self,
        db_session: AsyncSession,
        record: ScheduleRecord,
        *,
        include_recent_runs: bool,
    ) -> ScheduleSummary:
        last_fire = None
        if include_recent_runs:
            statement = (
                select(ScheduleFireRecord)
                .where(ScheduleFireRecord.schedule_id == record.id)
                .order_by(ScheduleFireRecord.created_at.desc())
                .limit(1)
            )
            result = await db_session.execute(statement)
            fire_record = result.scalar_one_or_none()
            last_fire = fire_summary_from_record(fire_record) if isinstance(fire_record, ScheduleFireRecord) else None
        prompt = _prompt_from_input_parts(record.input_parts_template)
        continue_current_session, start_from_current_session, steer_when_running = _facade_flags_from_record(record)
        trigger = trigger_summary_from_record(record)
        return ScheduleSummary(
            id=record.id,
            name=record.name,
            description=record.description,
            enabled=record.status == "active",
            status=cast(ScheduleStatus, record.status),
            prompt=prompt,
            trigger=trigger,
            cron={"expr": record.cron_expr, "timezone": record.timezone, "next_fire_at": record.next_fire_at},
            mode={
                "continue_current_session": continue_current_session,
                "start_from_current_session": start_from_current_session,
                "steer_when_running": steer_when_running,
            },
            execution_mode=cast(ScheduleExecutionMode, record.execution_mode),
            owner_kind=record.owner_kind,
            owner_session_id=record.owner_session_id,
            owner_run_id=record.owner_run_id,
            profile_name=record.profile_name,
            target_session_id=record.target_session_id,
            source_session_id=record.source_session_id,
            last_fire=last_fire,
            fire_count=record.fire_count,
            failure_count=record.failure_count,
            metadata=dict(record.schedule_metadata),
            created_at=record.created_at,
            updated_at=record.updated_at,
        )

    async def _get_schedule_record(self, db_session: AsyncSession, schedule_id: str) -> ScheduleRecord:
        record = await db_session.get(ScheduleRecord, schedule_id)
        if not isinstance(record, ScheduleRecord) or record.status == "deleted":
            raise HTTPException(status_code=404, detail=f"Schedule '{schedule_id}' was not found.")
        return record

    async def _require_session(self, db_session: AsyncSession, session_id: str | None) -> SessionRecord:
        if not isinstance(session_id, str) or session_id.strip() == "":
            raise HTTPException(status_code=422, detail="session_id is required.")
        record = await db_session.get(SessionRecord, session_id)
        if not isinstance(record, SessionRecord):
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' was not found.")
        return record


def _as_utc_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def fire_summary_from_record(
    record: ScheduleFireRecord,
    *,
    run_status: str | None = None,
) -> ScheduleFireSummary:
    return ScheduleFireSummary(
        id=record.id,
        schedule_id=record.schedule_id,
        scheduled_at=record.scheduled_at,
        fired_at=record.fired_at,
        status=cast(ScheduleFireStatus, record.status),
        target_session_id=record.target_session_id,
        source_session_id=record.source_session_id,
        created_session_id=record.created_session_id,
        run_id=record.run_id,
        active_run_id=record.active_run_id,
        run_status=cast(ScheduleRunStatus, run_status) if run_status is not None else None,
        input_preview=_prompt_from_input_parts(record.input_parts),
        error_message=record.error_message,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def infer_trigger_kind(
    trigger_kind: ScheduleTriggerKind | None,
    cron_expr: str | None,
    run_at: datetime | None,
) -> ScheduleTriggerKind:
    if trigger_kind is not None:
        return trigger_kind
    if run_at is not None and (cron_expr is None or cron_expr.strip() == ""):
        return "once"
    return "cron"


def validate_schedule_trigger(record: ScheduleRecord) -> None:
    if record.trigger_kind == "cron":
        if not isinstance(record.cron_expr, str) or record.cron_expr.strip() == "":
            raise HTTPException(status_code=422, detail="cron is required for cron schedules.")
        return
    if record.trigger_kind == "once":
        if record.run_at is None:
            raise HTTPException(status_code=422, detail="run_at is required for one-time schedules.")
        return
    raise HTTPException(status_code=422, detail=f"Unknown schedule trigger kind '{record.trigger_kind}'.")


def compute_schedule_next_fire_at(record: ScheduleRecord, *, now: datetime | None = None) -> datetime:
    validate_schedule_trigger(record)
    if record.trigger_kind == "cron":
        if not isinstance(record.cron_expr, str):
            raise HTTPException(status_code=422, detail="cron is required for cron schedules.")
        return compute_next_fire_at(record.cron_expr, record.timezone, now=now)
    if record.run_at is None:
        raise HTTPException(status_code=422, detail="run_at is required for one-time schedules.")
    return _as_utc_aware(record.run_at)


def trigger_summary_from_record(record: ScheduleRecord) -> dict[str, Any]:
    if record.trigger_kind == "once":
        return {
            "kind": "once",
            "run_at": record.run_at,
            "timezone": record.timezone,
            "next_fire_at": record.next_fire_at,
        }
    return {
        "kind": "cron",
        "cron": record.cron_expr,
        "timezone": record.timezone,
        "next_fire_at": record.next_fire_at,
    }


def compute_next_fire_at(cron_expr: str, timezone: str, *, now: datetime | None = None) -> datetime:
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        raise HTTPException(status_code=422, detail="cron must contain five fields.")
    minute_values = _parse_cron_field(fields[0], 0, 59)
    hour_values = _parse_cron_field(fields[1], 0, 23)
    dom_values = _parse_cron_field(fields[2], 1, 31)
    month_values = _parse_cron_field(fields[3], 1, 12)
    dow_values = _parse_cron_field(fields[4], 0, 7)
    tz = ZoneInfo(normalize_timezone(timezone))
    effective_now = now or utc_now()
    local_cursor = effective_now.astimezone(tz).replace(second=0, microsecond=0) + timedelta(minutes=1)
    for minute_offset in range(0, 366 * 24 * 60):
        candidate = local_cursor + timedelta(minutes=minute_offset)
        cron_dow = (candidate.weekday() + 1) % 7
        if (
            candidate.minute in minute_values
            and candidate.hour in hour_values
            and candidate.day in dom_values
            and candidate.month in month_values
            and (cron_dow in dow_values or (cron_dow == 0 and 7 in dow_values))
        ):
            return candidate.astimezone(UTC)
    raise HTTPException(status_code=422, detail="cron did not produce a fire time within one year.")


def normalize_timezone(value: str) -> str:
    normalized = value.strip() or "UTC"
    try:
        ZoneInfo(normalized)
    except ZoneInfoNotFoundError as exc:
        raise HTTPException(status_code=422, detail=f"Unknown timezone '{normalized}'.") from exc
    return normalized


def _parse_cron_field(raw_field: str, minimum: int, maximum: int) -> set[int]:  # noqa: C901
    values: set[int] = set()
    for raw_part in raw_field.split(","):
        part = raw_part.strip()
        if part == "*":
            values.update(range(minimum, maximum + 1))
            continue
        step = 1
        if "/" in part:
            base, _, raw_step = part.partition("/")
            if not raw_step.isdigit() or int(raw_step) <= 0:
                raise HTTPException(status_code=422, detail=f"Invalid cron step: {part}")
            step = int(raw_step)
            part = base or "*"
        if part == "*":
            values.update(range(minimum, maximum + 1, step))
            continue
        if "-" in part:
            raw_start, _, raw_end = part.partition("-")
            if not raw_start.isdigit() or not raw_end.isdigit():
                raise HTTPException(status_code=422, detail=f"Invalid cron range: {part}")
            start = int(raw_start)
            end = int(raw_end)
            if start > end or start < minimum or end > maximum:
                raise HTTPException(status_code=422, detail=f"Cron range out of bounds: {part}")
            values.update(range(start, end + 1, step))
            continue
        if not part.isdigit():
            raise HTTPException(status_code=422, detail=f"Invalid cron field: {raw_field}")
        value = int(part)
        if value < minimum or value > maximum:
            raise HTTPException(status_code=422, detail=f"Cron value out of bounds: {value}")
        if (value - minimum) % step == 0:
            values.add(value)
    return values


def _resolve_facade_mode(
    *,
    continue_current_session: bool,
    start_from_current_session: bool,
    steer_when_running: bool,
    current_session_id: str | None,
) -> tuple[str, str | None, str | None, str]:
    on_active = "steer" if steer_when_running else "queue"
    if continue_current_session:
        return "continue_session", current_session_id, None, on_active
    if start_from_current_session:
        return "fork_session", None, current_session_id, on_active
    return "isolate_session", None, None, on_active


def _facade_flags_from_record(record: ScheduleRecord) -> tuple[bool, bool, bool]:
    return (
        record.execution_mode == "continue_session",
        record.execution_mode == "fork_session",
        record.on_active == "steer",
    )


def _source_metadata(record: ScheduleRecord, fire_record: ScheduleFireRecord) -> dict[str, Any]:
    return {
        "source": "schedule",
        "schedule_id": record.id,
        "schedule_fire_id": fire_record.id,
        "execution_mode": record.execution_mode,
    }


def _prompt_from_input_parts(input_parts: list[dict[str, Any]]) -> str:
    parts = parse_input_parts(input_parts)
    for part in parts:
        if isinstance(part, TextPart):
            return part.text
    return ""


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None
