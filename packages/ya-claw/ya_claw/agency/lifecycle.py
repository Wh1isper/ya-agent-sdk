from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ya_claw.agui_adapter import AguiEventAdapter
from ya_claw.config import ClawSettings
from ya_claw.controller.models import (
    AgencyBudget,
    AgencySignalReason,
    AgencySignalRequest,
    AgencySignalResponse,
    AgencySignalStatus,
    AgencyStateSummary,
    CommandPart,
    DispatchMode,
    TriggerType,
    agency_signal_summary_from_record,
    agency_state_summary_from_record,
)
from ya_claw.orm.tables import AgencySignalRecord, RunRecord, SessionAgencyStateRecord, SessionRecord
from ya_claw.runtime_state import InMemoryRuntimeState

_SIGNAL_PRIORITIES: dict[str, int] = {
    AgencySignalReason.MANUAL.value: 10,
    AgencySignalReason.FAILED_RUN_FOLLOWUP.value: 20,
    AgencySignalReason.OPEN_INTENTION_DUE.value: 30,
    AgencySignalReason.SCHEDULE.value: 40,
    AgencySignalReason.MEMORY_COMMITTED.value: 50,
    AgencySignalReason.INACTIVITY.value: 60,
    AgencySignalReason.COMPACT.value: 70,
}


@dataclass(slots=True)
class AgencySignalDelivery:
    signal: AgencySignalRecord
    state: SessionAgencyStateRecord
    agency_session: SessionRecord
    run_id: str | None = None
    active_run_id: str | None = None
    delivery: str = "pending"


@dataclass(slots=True)
class AgencyTickResult:
    submitted_run_ids: list[str] = field(default_factory=list)
    steered_signal_ids: list[str] = field(default_factory=list)
    created_signal_ids: list[str] = field(default_factory=list)


class AgencyLifecycle:
    def __init__(
        self,
        *,
        settings: ClawSettings,
        runtime_state: InMemoryRuntimeState,
        submit_run: Callable[[str], bool] | None = None,
    ) -> None:
        self._settings = settings
        self._runtime_state = runtime_state
        self._submit_run = submit_run

    async def get_state(
        self,
        db_session: AsyncSession,
        source_session_id: str,
        *,
        ensure: bool = True,
    ) -> SessionAgencyStateRecord:
        source_session = await _load_source_session(db_session, source_session_id)
        if ensure:
            return await self.ensure_state(db_session, source_session)
        state = await db_session.get(SessionAgencyStateRecord, source_session_id)
        if isinstance(state, SessionAgencyStateRecord):
            return state
        return await self.ensure_state(db_session, source_session)

    async def ensure_state(
        self,
        db_session: AsyncSession,
        source_session: SessionRecord,
    ) -> SessionAgencyStateRecord:
        _validate_source_conversation_session(source_session)
        state = await db_session.get(SessionAgencyStateRecord, source_session.id)
        if isinstance(state, SessionAgencyStateRecord):
            if state.agency_session_id is None:
                agency_session = await self.ensure_agency_session(db_session, source_session)
                state.agency_session_id = agency_session.id
            state.pending_signal_count = await _count_pending_signals(db_session, source_session.id)
            return state
        agency_session = await self.ensure_agency_session(db_session, source_session)
        state = SessionAgencyStateRecord(
            source_session_id=source_session.id,
            agency_session_id=agency_session.id,
            enabled=_metadata_agency_enabled(source_session.session_metadata, default=self._settings.agency_enabled),
            agency_metadata={},
        )
        db_session.add(state)
        await db_session.flush()
        return state

    async def ensure_agency_session(
        self,
        db_session: AsyncSession,
        source_session: SessionRecord,
    ) -> SessionRecord:
        statement = select(SessionRecord).where(
            SessionRecord.session_type == "agency",
            SessionRecord.source_session_id == source_session.id,
        )
        result = await db_session.execute(statement)
        record = result.scalars().first()
        if isinstance(record, SessionRecord):
            return record
        record = SessionRecord(
            id=uuid4().hex,
            parent_session_id=source_session.id,
            profile_name=self._resolve_agency_profile(source_session),
            session_type="agency",
            source_session_id=source_session.id,
            session_metadata=_agency_session_metadata(source_session),
        )
        db_session.add(record)
        await db_session.flush()
        return record

    async def update_state(
        self,
        db_session: AsyncSession,
        source_session_id: str,
        *,
        enabled: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgencyStateSummary:
        source_session = await _load_source_session(db_session, source_session_id)
        state = await self.ensure_state(db_session, source_session)
        if isinstance(enabled, bool):
            state.enabled = enabled
        if metadata is not None:
            state.agency_metadata = {**dict(state.agency_metadata or {}), **metadata}
        state.pending_signal_count = await _count_pending_signals(db_session, source_session.id)
        await db_session.commit()
        await db_session.refresh(state)
        return agency_state_summary_from_record(state)

    async def create_signal(
        self,
        db_session: AsyncSession,
        source_session_id: str,
        request: AgencySignalRequest,
        *,
        dispatch: bool = True,
    ) -> AgencySignalDelivery:
        source_session = await _load_source_session(db_session, source_session_id)
        state = await self.ensure_state(db_session, source_session)
        agency_session = await self.ensure_agency_session(db_session, source_session)
        state.agency_session_id = agency_session.id
        if not state.enabled and request.reason != AgencySignalReason.MANUAL:
            raise HTTPException(status_code=409, detail=f"Agency is disabled for session '{source_session_id}'.")
        if request.reason == AgencySignalReason.MANUAL and not state.enabled:
            state.enabled = True

        signal = await self._insert_signal(db_session, state, agency_session, request)
        state.pending_signal_count = await _count_pending_signals(db_session, source_session.id)
        await db_session.commit()
        await db_session.refresh(signal)
        await db_session.refresh(state)
        await db_session.refresh(agency_session)

        delivery = AgencySignalDelivery(signal=signal, state=state, agency_session=agency_session, delivery="pending")
        if dispatch:
            delivery = await self.dispatch_source(db_session, source_session_id)
        return delivery

    async def dispatch_source(self, db_session: AsyncSession, source_session_id: str) -> AgencySignalDelivery:
        source_session = await _load_source_session(db_session, source_session_id)
        state = await self.ensure_state(db_session, source_session)
        agency_session = await self.ensure_agency_session(db_session, source_session)
        signals = await _load_pending_signals(
            db_session, source_session_id, limit=self._settings.agency_max_signals_per_tick
        )
        if not signals:
            state.pending_signal_count = 0
            await db_session.commit()
            return AgencySignalDelivery(
                signal=await _latest_signal(db_session, source_session_id),
                state=state,
                agency_session=agency_session,
                delivery="pending",
            )

        active_run = await _active_agency_run(db_session, agency_session)
        if isinstance(active_run, RunRecord):
            input_parts = [_signal_input_part(signal) for signal in signals]
            input_payload = [part.model_dump(mode="json") for part in input_parts]
            await self._runtime_state.record_steering(active_run.id, input_payload)
            agui_adapter = AguiEventAdapter(session_id=active_run.session_id, run_id=active_run.id)
            await self._runtime_state.append_run_event(
                active_run.id,
                agui_adapter.build_run_steered_event({
                    "run_id": active_run.id,
                    "session_id": active_run.session_id,
                    "input_parts": input_payload,
                }),
            )
            for signal in signals:
                signal.status = AgencySignalStatus.STEERED.value
                signal.run_id = active_run.id
                signal.agency_session_id = agency_session.id
            state.pending_signal_count = await _count_pending_signals(db_session, source_session_id)
            state.last_agency_run_id = active_run.id
            state.last_agency_reason = signals[0].reason
            await db_session.commit()
            await db_session.refresh(state)
            await db_session.refresh(signals[0])
            return AgencySignalDelivery(
                signal=signals[0],
                state=state,
                agency_session=agency_session,
                active_run_id=active_run.id,
                delivery="steered",
            )

        run_record = await self._create_agency_run(db_session, source_session, agency_session, state, signals)
        for signal in signals:
            signal.status = AgencySignalStatus.SUBMITTED.value
            signal.run_id = run_record.id
            signal.agency_session_id = agency_session.id
        state.pending_signal_count = await _count_pending_signals(db_session, source_session_id)
        state.last_agency_run_id = run_record.id
        state.last_agency_reason = signals[0].reason
        await db_session.commit()
        await db_session.refresh(state)
        await db_session.refresh(signals[0])
        await db_session.refresh(run_record)
        if self._submit_run is not None:
            self._submit_run(run_record.id)
        return AgencySignalDelivery(
            signal=signals[0],
            state=state,
            agency_session=agency_session,
            run_id=run_record.id,
            delivery="submitted",
        )

    async def tick(self, db_session: AsyncSession) -> AgencyTickResult:
        result = AgencyTickResult()
        await self._create_due_schedule_signals(db_session, result)
        source_ids = await _source_ids_with_pending_signals(
            db_session, limit=self._settings.agency_max_sessions_per_tick
        )
        for source_session_id in source_ids:
            delivery = await self.dispatch_source(db_session, source_session_id)
            if delivery.delivery == "submitted" and delivery.run_id is not None:
                result.submitted_run_ids.append(delivery.run_id)
            elif delivery.delivery == "steered":
                result.steered_signal_ids.append(delivery.signal.id)
        return result

    async def on_agency_run_committed(self, db_session: AsyncSession, run_record: RunRecord) -> None:
        agency = _agency_metadata(run_record.run_metadata)
        source_session_id = _string_or_none(agency.get("source_session_id"))
        if source_session_id is None:
            return
        state = await db_session.get(SessionAgencyStateRecord, source_session_id)
        if not isinstance(state, SessionAgencyStateRecord):
            return
        state.episode_count += 1
        state.last_agency_run_id = run_record.id
        reasons = agency.get("reasons")
        if isinstance(reasons, list) and reasons:
            state.last_agency_reason = str(reasons[0])
        state.last_action_at = run_record.committed_at or datetime.now(UTC)
        state.cooldown_until = (run_record.committed_at or datetime.now(UTC)) + timedelta(
            seconds=max(0, self._settings.agency_cooldown_seconds)
        )
        state.pending_signal_count = await _count_pending_signals(db_session, source_session_id)
        consumed_ids = _consumed_signal_ids(run_record)
        await _mark_run_signals_consumed(db_session, run_record.id, consumed_ids=consumed_ids)

    async def _insert_signal(
        self,
        db_session: AsyncSession,
        state: SessionAgencyStateRecord,
        agency_session: SessionRecord,
        request: AgencySignalRequest,
    ) -> AgencySignalRecord:
        signal_id = uuid4().hex
        reason = request.reason.value if isinstance(request.reason, AgencySignalReason) else str(request.reason)
        client_token = _string_or_none(request.client_token) or signal_id
        dedupe_key = _dedupe_key(
            source_session_id=state.source_session_id,
            reason=reason,
            client_token=client_token,
            source_run_ids=request.source_run_ids,
        )
        signal_metadata: dict[str, Any] = {
            "agency": {
                "kind": "session_agency_signal",
                "reason": reason,
                "source_session_id": state.source_session_id,
                "source_run_ids": list(request.source_run_ids),
                "client_token": client_token,
                "budget_override": request.budget.model_dump(mode="json") if request.budget is not None else None,
                "prompt_override": request.prompt_override,
            },
            **dict(request.metadata),
        }
        signal = AgencySignalRecord(
            id=signal_id,
            source_session_id=state.source_session_id,
            agency_session_id=agency_session.id,
            reason=reason,
            status=AgencySignalStatus.PENDING.value,
            priority=_SIGNAL_PRIORITIES.get(reason, 100),
            dedupe_key=dedupe_key,
            source_run_ids=list(request.source_run_ids),
            signal_metadata=signal_metadata,
        )
        db_session.add(signal)
        try:
            await db_session.flush()
            return signal
        except IntegrityError:
            await db_session.rollback()
            existing = await _load_signal_by_dedupe(db_session, state.source_session_id, dedupe_key)
            if existing is None:
                raise
            return existing

    async def _create_agency_run(
        self,
        db_session: AsyncSession,
        source_session: SessionRecord,
        agency_session: SessionRecord,
        state: SessionAgencyStateRecord,
        signals: list[AgencySignalRecord],
    ) -> RunRecord:
        sequence_no = await _next_sequence_no(db_session, agency_session.id)
        run_id = uuid4().hex
        current_sequence_no = await _current_source_sequence_no(db_session, source_session.id)
        budget = _budget_from_signals(signals)
        run_metadata = {
            "agency": {
                "kind": "session_agency_episode",
                "source_session_id": source_session.id,
                "agency_session_id": agency_session.id,
                "signal_ids": [signal.id for signal in signals],
                "reasons": [signal.reason for signal in signals],
                "source_run_ids": _unique_source_run_ids(signals),
                "last_observed_sequence_no": state.last_observed_sequence_no,
                "current_sequence_no": current_sequence_no,
                "episode_id": f"episode-{run_id}",
                "budget": budget.model_dump(mode="json"),
                "risk_policy": {
                    "max_auto_action_risk": _max_auto_action_risk(source_session.session_metadata),
                    "approval_required_for": [],
                },
            },
            "restore_state": True,
        }
        input_parts = [_signal_input_part(signal) for signal in signals]
        run_record = RunRecord(
            id=run_id,
            session_id=agency_session.id,
            sequence_no=sequence_no,
            restore_from_run_id=agency_session.head_success_run_id,
            status="queued",
            trigger_type=TriggerType.AGENCY.value,
            profile_name=agency_session.profile_name,
            input_parts=[part.model_dump(mode="json") for part in input_parts],
            run_metadata=run_metadata,
        )
        db_session.add(run_record)
        _queue_run(agency_session, run_record)
        self._runtime_state.register_run(agency_session.id, run_id, dispatch_mode=DispatchMode.ASYNC)
        return run_record

    async def _create_due_schedule_signals(self, db_session: AsyncSession, result: AgencyTickResult) -> None:
        now = datetime.now(UTC)
        await self._ensure_recent_conversation_states(db_session)
        statement = (
            select(SessionAgencyStateRecord)
            .where(SessionAgencyStateRecord.enabled.is_(True))
            .order_by(SessionAgencyStateRecord.updated_at.asc())
            .limit(self._settings.agency_max_sessions_per_tick)
        )
        rows = await db_session.execute(statement)
        states = list(rows.scalars().all())
        for state in states:
            if state.cooldown_until is not None and state.cooldown_until > now:
                continue
            source_session = await db_session.get(SessionRecord, state.source_session_id)
            if not isinstance(source_session, SessionRecord) or source_session.session_type != "conversation":
                continue
            current_sequence_no = await _current_source_sequence_no(db_session, source_session.id)
            if current_sequence_no <= 0:
                continue
            if source_session.updated_at + timedelta(seconds=max(0, self._settings.agency_idle_after_seconds)) > now:
                continue
            if current_sequence_no <= state.last_observed_sequence_no and state.last_action_at is not None:
                continue
            due_bucket = now.strftime("%Y%m%d%H%M")
            signal = AgencySignalRequest(
                reason=AgencySignalReason.SCHEDULE,
                client_token=f"tick-{due_bucket}-{current_sequence_no}",
                metadata={
                    "scheduled_at": now.isoformat(),
                    "current_sequence_no": current_sequence_no,
                },
            )
            created = await self._insert_signal(
                db_session, state, await self.ensure_agency_session(db_session, source_session), signal
            )
            result.created_signal_ids.append(created.id)
            state.last_observed_sequence_no = max(state.last_observed_sequence_no, current_sequence_no)
            state.pending_signal_count = await _count_pending_signals(db_session, source_session.id)
        await db_session.commit()

    async def _ensure_recent_conversation_states(self, db_session: AsyncSession) -> None:
        result = await db_session.execute(
            select(SessionRecord)
            .where(SessionRecord.session_type == "conversation")
            .order_by(SessionRecord.updated_at.desc())
            .limit(self._settings.agency_max_sessions_per_tick)
        )
        for source_session in result.scalars().all():
            if isinstance(source_session, SessionRecord):
                await self.ensure_state(db_session, source_session)
        await db_session.flush()

    def _resolve_agency_profile(self, source_session: SessionRecord) -> str:
        if isinstance(self._settings.agency_profile, str) and self._settings.agency_profile.strip() != "":
            return self._settings.agency_profile.strip()
        return source_session.profile_name or self._settings.resolved_agency_profile


def build_signal_response(delivery: AgencySignalDelivery) -> AgencySignalResponse:
    return AgencySignalResponse(
        accepted=True,
        source_session_id=delivery.state.source_session_id,
        agency_session_id=delivery.agency_session.id,
        signal=agency_signal_summary_from_record(delivery.signal),
        run_id=delivery.run_id,
        active_run_id=delivery.active_run_id,
        delivery=delivery.delivery,  # type: ignore[arg-type]
        state=agency_state_summary_from_record(delivery.state),
    )


def _queue_run(session: SessionRecord, run: RunRecord) -> None:
    effective_time = datetime.now(UTC)
    session.head_run_id = run.id
    session.profile_name = run.profile_name
    session.updated_at = effective_time
    run.status = "queued"


def _agency_session_metadata(source_session: SessionRecord) -> dict[str, Any]:
    metadata: dict[str, Any] = {"agency": {"source_session_id": source_session.id}}
    if isinstance(source_session.session_metadata, dict):
        for key in ("sandbox", "workspace"):
            value = source_session.session_metadata.get(key)
            if isinstance(value, dict):
                metadata[key] = dict(value)
    return metadata


def _metadata_agency_enabled(metadata: dict[str, Any], *, default: bool) -> bool:
    agency = metadata.get("agency") if isinstance(metadata, dict) else None
    if isinstance(agency, dict) and isinstance(agency.get("enabled"), bool):
        return bool(agency["enabled"])
    return default


def _max_auto_action_risk(metadata: dict[str, Any]) -> str:
    agency = metadata.get("agency") if isinstance(metadata, dict) else None
    value = agency.get("max_auto_action_risk") if isinstance(agency, dict) else None
    return value if value in {"low", "medium", "high", "extra_high"} else "low"


def _validate_source_conversation_session(source_session: SessionRecord) -> None:
    if source_session.session_type != "conversation":
        raise HTTPException(status_code=422, detail="Agency is supported for conversation sessions.")


async def _load_source_session(db_session: AsyncSession, source_session_id: str) -> SessionRecord:
    source_session = await db_session.get(SessionRecord, source_session_id)
    if not isinstance(source_session, SessionRecord):
        raise HTTPException(status_code=404, detail=f"Session '{source_session_id}' was not found.")
    _validate_source_conversation_session(source_session)
    return source_session


async def _count_pending_signals(db_session: AsyncSession, source_session_id: str) -> int:
    result = await db_session.execute(
        select(func.count()).where(
            AgencySignalRecord.source_session_id == source_session_id,
            AgencySignalRecord.status == AgencySignalStatus.PENDING.value,
        )
    )
    value = result.scalar_one_or_none()
    return int(value or 0)


async def _load_pending_signals(
    db_session: AsyncSession, source_session_id: str, *, limit: int
) -> list[AgencySignalRecord]:
    result = await db_session.execute(
        select(AgencySignalRecord)
        .where(
            AgencySignalRecord.source_session_id == source_session_id,
            AgencySignalRecord.status == AgencySignalStatus.PENDING.value,
        )
        .order_by(AgencySignalRecord.priority.asc(), AgencySignalRecord.created_at.asc())
        .limit(max(1, limit))
    )
    return list(result.scalars().all())


async def _latest_signal(db_session: AsyncSession, source_session_id: str) -> AgencySignalRecord:
    result = await db_session.execute(
        select(AgencySignalRecord)
        .where(AgencySignalRecord.source_session_id == source_session_id)
        .order_by(AgencySignalRecord.created_at.desc())
        .limit(1)
    )
    signal = result.scalars().first()
    if not isinstance(signal, AgencySignalRecord):
        raise HTTPException(status_code=404, detail="Agency signal was not found.")
    return signal


async def _active_agency_run(db_session: AsyncSession, agency_session: SessionRecord) -> RunRecord | None:
    if not isinstance(agency_session.active_run_id, str):
        return None
    record = await db_session.get(RunRecord, agency_session.active_run_id)
    if isinstance(record, RunRecord) and record.status in {"queued", "running"}:
        return record
    return None


async def _next_sequence_no(db_session: AsyncSession, session_id: str) -> int:
    result = await db_session.execute(select(func.max(RunRecord.sequence_no)).where(RunRecord.session_id == session_id))
    value = result.scalar_one_or_none()
    return value + 1 if isinstance(value, int) else 1


async def _current_source_sequence_no(db_session: AsyncSession, source_session_id: str) -> int:
    result = await db_session.execute(
        select(func.max(RunRecord.sequence_no)).where(RunRecord.session_id == source_session_id)
    )
    value = result.scalar_one_or_none()
    return value if isinstance(value, int) else 0


async def _source_ids_with_pending_signals(db_session: AsyncSession, *, limit: int) -> list[str]:
    result = await db_session.execute(
        select(AgencySignalRecord.source_session_id)
        .where(AgencySignalRecord.status == AgencySignalStatus.PENDING.value)
        .group_by(AgencySignalRecord.source_session_id)
        .order_by(func.min(AgencySignalRecord.priority).asc(), func.min(AgencySignalRecord.created_at).asc())
        .limit(max(1, limit))
    )
    return [source_id for source_id in result.scalars().all() if isinstance(source_id, str)]


async def _load_signal_by_dedupe(
    db_session: AsyncSession,
    source_session_id: str,
    dedupe_key: str,
) -> AgencySignalRecord | None:
    result = await db_session.execute(
        select(AgencySignalRecord).where(
            AgencySignalRecord.source_session_id == source_session_id,
            AgencySignalRecord.dedupe_key == dedupe_key,
        )
    )
    return result.scalars().first()


def _dedupe_key(*, source_session_id: str, reason: str, client_token: str, source_run_ids: list[str]) -> str:
    if reason == AgencySignalReason.MANUAL.value:
        return f"session:{source_session_id}:manual:{client_token}"
    if source_run_ids:
        return f"session:{source_session_id}:{reason}:{','.join(sorted(set(source_run_ids)))}"
    return f"session:{source_session_id}:{reason}:{client_token}"


def _signal_input_part(signal: AgencySignalRecord) -> CommandPart:
    return CommandPart(
        type="command",
        name="agency_signal",
        params={
            "signal_id": signal.id,
            "reason": signal.reason,
            "source_session_id": signal.source_session_id,
            "source_run_ids": list(signal.source_run_ids or []),
            "payload": dict(signal.signal_metadata or {}),
        },
    )


def _budget_from_signals(signals: list[AgencySignalRecord]) -> AgencyBudget:
    for signal in signals:
        agency = signal.signal_metadata.get("agency") if isinstance(signal.signal_metadata, dict) else None
        override = agency.get("budget_override") if isinstance(agency, dict) else None
        if isinstance(override, dict):
            return AgencyBudget.model_validate(override)
    return AgencyBudget()


def _unique_source_run_ids(signals: list[AgencySignalRecord]) -> list[str]:
    seen: set[str] = set()
    values: list[str] = []
    for signal in signals:
        for run_id in signal.source_run_ids or []:
            if isinstance(run_id, str) and run_id not in seen:
                seen.add(run_id)
                values.append(run_id)
    return values


def _agency_metadata(run_metadata: dict[str, Any]) -> dict[str, Any]:
    agency = run_metadata.get("agency") if isinstance(run_metadata, dict) else None
    return dict(agency) if isinstance(agency, dict) else {}


def _consumed_signal_ids(run_record: RunRecord) -> list[str]:
    metadata_ids = _agency_metadata(run_record.run_metadata).get("signal_ids")
    return [item for item in metadata_ids if isinstance(item, str)] if isinstance(metadata_ids, list) else []


async def _mark_run_signals_consumed(
    db_session: AsyncSession,
    run_id: str,
    *,
    consumed_ids: list[str],
) -> None:
    if not consumed_ids:
        return
    result = await db_session.execute(select(AgencySignalRecord).where(AgencySignalRecord.id.in_(consumed_ids)))
    now = datetime.now(UTC)
    for signal in result.scalars().all():
        signal.status = AgencySignalStatus.CONSUMED.value
        signal.consumed_at = now
        signal.run_id = run_id


def _string_or_none(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None
