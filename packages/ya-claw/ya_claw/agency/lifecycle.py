from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any, Literal
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ya_claw.agui_adapter import AguiEventAdapter
from ya_claw.config import ClawSettings
from ya_claw.controller.models import (
    AgencyFireKind,
    AgencyFireStatus,
    AgencyRiskPolicy,
    CommandPart,
    DispatchMode,
    TriggerType,
)
from ya_claw.orm.tables import AgencyFireRecord, RunRecord, SessionRecord
from ya_claw.runtime_state import InMemoryRuntimeState

AGENCY_SINGLETON_SCOPE_KEY = "agency:global"
AGENCY_SINGLETON_SOURCE_SESSION_ID = sha256(AGENCY_SINGLETON_SCOPE_KEY.encode("utf-8")).hexdigest()[:32]
_FIRE_PRIORITIES: dict[str, int] = {
    AgencyFireKind.MANUAL.value: 10,
    AgencyFireKind.MEMORY_COMMITTED.value: 30,
    AgencyFireKind.TIMER.value: 50,
    AgencyFireKind.COMPACT.value: 70,
}


@dataclass(slots=True)
class AgencyFireDelivery:
    fire: AgencyFireRecord
    agency_session: SessionRecord
    run_id: str | None = None
    active_run_id: str | None = None
    delivery: Literal["pending", "submitted", "steered", "merged", "duplicate", "skipped"] = "pending"


@dataclass(slots=True)
class AgencyFireInsertResult:
    fire: AgencyFireRecord
    created: bool


@dataclass(slots=True)
class AgencyTickResult:
    created_fire_ids: list[str] = field(default_factory=list)
    submitted_run_ids: list[str] = field(default_factory=list)
    steered_fire_ids: list[str] = field(default_factory=list)
    merged_fire_ids: list[str] = field(default_factory=list)
    skipped_fire_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AgencyClearResult:
    cleared_session_id: str | None
    archived_run_ids: list[str] = field(default_factory=list)
    deleted_fire_count: int = 0
    cleared_at: datetime = field(default_factory=lambda: datetime.now(UTC))


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

    async def load_agency_session(self, db_session: AsyncSession) -> SessionRecord | None:
        statement = select(SessionRecord).where(
            SessionRecord.session_type == "agency",
            SessionRecord.source_session_id == AGENCY_SINGLETON_SOURCE_SESSION_ID,
        )
        result = await db_session.execute(statement)
        return result.scalars().first()

    async def ensure_agency_session(self, db_session: AsyncSession) -> SessionRecord:
        profile_name = self._settings.resolved_agency_profile
        record = await self.load_agency_session(db_session)
        if isinstance(record, SessionRecord):
            record.profile_name = profile_name
            metadata = dict(record.session_metadata or {})
            agency = dict(metadata.get("agency") or {}) if isinstance(metadata.get("agency"), dict) else {}
            agency.update({
                "kind": "claw_agency_session",
                "scope": "global",
                "scope_key": AGENCY_SINGLETON_SCOPE_KEY,
                "version": 1,
                "profile_name": profile_name,
            })
            agency["risk_policy"] = _settings_default_risk_policy(self._settings).model_dump(mode="json")
            metadata["agency"] = agency
            record.session_metadata = metadata
            return record

        record = SessionRecord(
            id=uuid4().hex,
            parent_session_id=None,
            profile_name=profile_name,
            session_type="agency",
            source_session_id=AGENCY_SINGLETON_SOURCE_SESSION_ID,
            session_metadata=_agency_session_metadata(
                profile_name=profile_name,
                risk_policy=_settings_default_risk_policy(self._settings),
            ),
        )
        db_session.add(record)
        try:
            await db_session.flush()
            return record
        except IntegrityError:
            await db_session.rollback()
            existing = await self.load_agency_session(db_session)
            if isinstance(existing, SessionRecord):
                return existing
            raise

    async def clear_agency_session(self, db_session: AsyncSession) -> AgencyClearResult:
        record = await self.load_agency_session(db_session)
        cleared_at = datetime.now(UTC)
        delete_result = await db_session.execute(delete(AgencyFireRecord))
        deleted_fire_count = int(delete_result.rowcount or 0)
        if not isinstance(record, SessionRecord):
            await db_session.commit()
            return AgencyClearResult(
                cleared_session_id=None,
                deleted_fire_count=deleted_fire_count,
                cleared_at=cleared_at,
            )

        runs_result = await db_session.execute(select(RunRecord).where(RunRecord.session_id == record.id))
        archived_runs = [run for run in runs_result.scalars().all() if isinstance(run, RunRecord)]
        archived_run_ids = [run.id for run in archived_runs]
        for run in archived_runs:
            if run.status in {"queued", "running"}:
                with contextlib.suppress(KeyError):
                    await self._runtime_state.request_stop(run.id, "clear_agency")
                run.status = "cancelled"
                run.termination_reason = "clear_agency"
                run.finished_at = cleared_at
        metadata = dict(record.session_metadata or {})
        agency = dict(metadata.get("agency") or {}) if isinstance(metadata.get("agency"), dict) else {}
        agency.update({
            "archived": True,
            "cleared_at": cleared_at.isoformat(),
            "cleared_from_source_session_id": record.source_session_id,
        })
        metadata["agency"] = agency
        record.session_metadata = metadata
        record.source_session_id = uuid4().hex
        record.head_run_id = None
        record.head_success_run_id = None
        record.active_run_id = None
        record.updated_at = cleared_at
        await db_session.commit()
        return AgencyClearResult(
            cleared_session_id=record.id,
            archived_run_ids=archived_run_ids,
            deleted_fire_count=deleted_fire_count,
            cleared_at=cleared_at,
        )

    async def create_fire(
        self,
        db_session: AsyncSession,
        *,
        kind: AgencyFireKind | str,
        source_session_id: str | None = None,
        source_run_id: str | None = None,
        client_token: str | None = None,
        prompt: str | None = None,
        payload: dict[str, Any] | None = None,
        scheduled_at: datetime | None = None,
        dedupe_key: str | None = None,
        dispatch: bool = True,
    ) -> AgencyFireDelivery:
        agency_session = await self.ensure_agency_session(db_session)
        fire_kind = kind.value if isinstance(kind, AgencyFireKind) else str(kind)
        if source_session_id is not None:
            await _load_source_conversation_session(db_session, source_session_id)
        if not self._settings.agency_enabled:
            if fire_kind == AgencyFireKind.MANUAL.value:
                raise HTTPException(status_code=409, detail="Agency is disabled.")
            fire = await self._insert_fire(
                db_session,
                agency_session=agency_session,
                kind=fire_kind,
                source_session_id=source_session_id,
                source_run_id=source_run_id,
                client_token=client_token,
                prompt=prompt,
                payload=payload,
                scheduled_at=scheduled_at,
                dedupe_key=dedupe_key,
                status=AgencyFireStatus.SKIPPED.value,
            )
            await db_session.commit()
            return AgencyFireDelivery(fire=fire.fire, agency_session=agency_session, delivery="skipped")

        insert_result = await self._insert_fire(
            db_session,
            agency_session=agency_session,
            kind=fire_kind,
            source_session_id=source_session_id,
            source_run_id=source_run_id,
            client_token=client_token,
            prompt=prompt,
            payload=payload,
            scheduled_at=scheduled_at,
            dedupe_key=dedupe_key,
            status=AgencyFireStatus.PENDING.value,
        )
        fire = insert_result.fire
        await db_session.commit()
        await db_session.refresh(fire)
        await db_session.refresh(agency_session)
        if not insert_result.created:
            return AgencyFireDelivery(
                fire=fire,
                agency_session=agency_session,
                run_id=fire.run_id
                if fire.status in {AgencyFireStatus.SUBMITTED.value, AgencyFireStatus.MERGED.value}
                else None,
                active_run_id=fire.active_run_id if fire.status == AgencyFireStatus.STEERED.value else None,
                delivery="duplicate",
            )
        if not dispatch:
            return AgencyFireDelivery(fire=fire, agency_session=agency_session, delivery="pending")
        return await self.dispatch_pending(db_session)

    async def dispatch_due(self, db_session: AsyncSession) -> AgencyFireDelivery | None:
        await self.ensure_agency_session(db_session)
        if not self._settings.agency_enabled:
            await db_session.commit()
            return None
        due_at = await self.next_timer_fire_at(db_session)
        if due_at is None or _as_utc_aware(due_at) > datetime.now(UTC):
            await db_session.commit()
            return None
        return await self.create_fire(
            db_session,
            kind=AgencyFireKind.TIMER,
            scheduled_at=due_at,
            dedupe_key=f"agency:timer:{_as_utc_aware(due_at).isoformat()}",
            payload={
                "reason": "scheduled_timer",
                "timer_interval_seconds": self._settings.agency_timer_interval_seconds,
            },
            dispatch=True,
        )

    async def next_timer_fire_at(self, db_session: AsyncSession) -> datetime | None:
        await self.ensure_agency_session(db_session)
        if not self._settings.agency_enabled:
            return None
        result = await db_session.execute(
            select(AgencyFireRecord)
            .where(AgencyFireRecord.kind == AgencyFireKind.TIMER.value)
            .order_by(AgencyFireRecord.scheduled_at.desc(), AgencyFireRecord.created_at.desc())
            .limit(1)
        )
        last_fire = result.scalars().first()
        if not isinstance(last_fire, AgencyFireRecord):
            return datetime.now(UTC)
        return _as_utc_aware(last_fire.scheduled_at) + timedelta(
            seconds=max(self._settings.agency_timer_interval_seconds, 1)
        )

    async def dispatch_pending(self, db_session: AsyncSession) -> AgencyFireDelivery:
        agency_session = await self.ensure_agency_session(db_session)
        fires = await _load_pending_fires(
            db_session,
            agency_session_id=agency_session.id,
            limit=self._settings.agency_fire_batch_limit,
        )
        if not fires:
            raise HTTPException(status_code=404, detail="Agency fire was not found.")

        active_run = await _blocking_agency_run(db_session, agency_session)
        if isinstance(active_run, RunRecord):
            input_payload = [_fire_input_part(fire).model_dump(mode="json") for fire in fires]
            _append_fires_to_run_metadata(active_run, fires, steered=active_run.status == "running")
            if active_run.status == "queued":
                active_run.input_parts = [*list(active_run.input_parts or []), *input_payload]
                for fire in fires:
                    fire.status = AgencyFireStatus.MERGED.value
                    fire.run_id = active_run.id
                    fire.active_run_id = None
                    fire.agency_session_id = agency_session.id
                delivery_kind: Literal["merged", "steered"] = "merged"
            else:
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
                for fire in fires:
                    fire.status = AgencyFireStatus.STEERED.value
                    fire.run_id = active_run.id
                    fire.active_run_id = active_run.id
                    fire.agency_session_id = agency_session.id
                delivery_kind = "steered"
            await db_session.commit()
            await db_session.refresh(fires[0])
            await db_session.refresh(agency_session)
            return AgencyFireDelivery(
                fire=fires[0],
                agency_session=agency_session,
                run_id=active_run.id if delivery_kind == "merged" else None,
                active_run_id=active_run.id if delivery_kind == "steered" else None,
                delivery=delivery_kind,
            )

        run_record = await self._create_agency_run(db_session, agency_session, fires)
        for fire in fires:
            fire.status = AgencyFireStatus.SUBMITTED.value
            fire.run_id = run_record.id
            fire.active_run_id = None
            fire.agency_session_id = agency_session.id
        try:
            await db_session.commit()
        except IntegrityError:
            await db_session.rollback()
            return await self.dispatch_pending(db_session)
        await db_session.refresh(fires[0])
        await db_session.refresh(agency_session)
        await db_session.refresh(run_record)
        if self._submit_run is not None:
            self._submit_run(run_record.id)
        return AgencyFireDelivery(
            fire=fires[0],
            agency_session=agency_session,
            run_id=run_record.id,
            delivery="submitted",
        )

    async def tick(self, db_session: AsyncSession) -> AgencyTickResult:
        result = AgencyTickResult()
        delivery = await self.dispatch_due(db_session)
        if delivery is None:
            return result
        result.created_fire_ids.append(delivery.fire.id)
        if delivery.delivery == "submitted" and delivery.run_id is not None:
            result.submitted_run_ids.append(delivery.run_id)
        elif delivery.delivery == "steered":
            result.steered_fire_ids.append(delivery.fire.id)
        elif delivery.delivery == "merged":
            result.merged_fire_ids.append(delivery.fire.id)
        elif delivery.delivery == "skipped":
            result.skipped_fire_ids.append(delivery.fire.id)
        return result

    async def on_agency_run_committed(self, db_session: AsyncSession, run_record: RunRecord) -> None:
        consumed_at = (
            _as_utc_aware(run_record.committed_at) if run_record.committed_at is not None else datetime.now(UTC)
        )
        consumed_ids = _consumed_fire_ids(run_record)
        if consumed_ids:
            await _mark_run_fires_consumed(
                db_session, run_record.id, consumed_ids=consumed_ids, consumed_at=consumed_at
            )

    async def on_agency_run_terminal(self, db_session: AsyncSession, run_record: RunRecord) -> None:
        await _mark_run_fires_failed(db_session, run_record.id)

    async def _insert_fire(
        self,
        db_session: AsyncSession,
        *,
        agency_session: SessionRecord,
        kind: str,
        source_session_id: str | None,
        source_run_id: str | None,
        client_token: str | None,
        prompt: str | None,
        payload: dict[str, Any] | None,
        scheduled_at: datetime | None,
        dedupe_key: str | None,
        status: str,
    ) -> AgencyFireInsertResult:
        fire_id = uuid4().hex
        effective_scheduled_at = scheduled_at or datetime.now(UTC)
        effective_payload = dict(payload or {})
        if prompt is not None:
            effective_payload["prompt"] = prompt
        if client_token is not None:
            effective_payload["client_token"] = client_token
        effective_dedupe_key = dedupe_key or _dedupe_key(
            kind=kind,
            source_session_id=source_session_id,
            source_run_id=source_run_id,
            client_token=client_token or fire_id,
            scheduled_at=effective_scheduled_at,
        )
        existing = await _load_fire_by_dedupe(db_session, effective_dedupe_key)
        if existing is not None:
            return AgencyFireInsertResult(fire=existing, created=False)
        record = AgencyFireRecord(
            id=fire_id,
            kind=kind,
            status=status,
            scheduled_at=effective_scheduled_at,
            fired_at=datetime.now(UTC),
            dedupe_key=effective_dedupe_key,
            source_session_id=source_session_id,
            source_run_id=source_run_id,
            agency_session_id=agency_session.id,
            run_id=None,
            active_run_id=None,
            priority=_FIRE_PRIORITIES.get(kind, 100),
            payload=effective_payload,
            error_message=None,
        )
        db_session.add(record)
        try:
            await db_session.flush()
            return AgencyFireInsertResult(fire=record, created=True)
        except IntegrityError:
            await db_session.rollback()
            existing = await _load_fire_by_dedupe(db_session, effective_dedupe_key)
            if existing is None:
                raise
            return AgencyFireInsertResult(fire=existing, created=False)

    async def _create_agency_run(
        self,
        db_session: AsyncSession,
        agency_session: SessionRecord,
        fires: list[AgencyFireRecord],
    ) -> RunRecord:
        sequence_no = await _next_sequence_no(db_session, agency_session.id)
        run_id = uuid4().hex
        risk_policy = _resolve_risk_policy(settings=self._settings, agency_session=agency_session)
        metadata = _episode_metadata(
            agency_session=agency_session,
            run_id=run_id,
            fires=fires,
            risk_policy=risk_policy,
        )
        run_record = RunRecord(
            id=run_id,
            session_id=agency_session.id,
            sequence_no=sequence_no,
            restore_from_run_id=agency_session.head_success_run_id,
            status="queued",
            trigger_type=TriggerType.AGENCY.value,
            profile_name=agency_session.profile_name,
            input_parts=[_fire_input_part(fire).model_dump(mode="json") for fire in fires],
            run_metadata=metadata,
        )
        db_session.add(run_record)
        _queue_run(agency_session, run_record)
        self._runtime_state.register_run(agency_session.id, run_id, dispatch_mode=DispatchMode.ASYNC)
        return run_record


def _queue_run(session: SessionRecord, run: RunRecord) -> None:
    effective_time = datetime.now(UTC)
    session.head_run_id = run.id
    session.profile_name = run.profile_name
    session.updated_at = effective_time
    run.status = "queued"


def _agency_session_metadata(*, profile_name: str, risk_policy: AgencyRiskPolicy) -> dict[str, Any]:
    return {
        "agency": {
            "kind": "claw_agency_session",
            "scope": "global",
            "scope_key": AGENCY_SINGLETON_SCOPE_KEY,
            "version": 1,
            "profile_name": profile_name,
            "risk_policy": risk_policy.model_dump(mode="json"),
        }
    }


def _settings_default_risk_policy(settings: ClawSettings) -> AgencyRiskPolicy:
    threshold = settings.agency_unattended_shell_review_risk_threshold
    if threshold is None:
        threshold = settings.unattended_shell_review_risk_threshold
    if threshold in {"low", "medium", "high", "extra_high"}:
        return AgencyRiskPolicy(max_auto_action_risk=threshold)
    return AgencyRiskPolicy()


def _resolve_risk_policy(*, settings: ClawSettings, agency_session: SessionRecord) -> AgencyRiskPolicy:
    return _settings_default_risk_policy(settings)


async def _load_source_conversation_session(db_session: AsyncSession, source_session_id: str) -> SessionRecord:
    source_session = await db_session.get(SessionRecord, source_session_id)
    if not isinstance(source_session, SessionRecord):
        raise HTTPException(status_code=404, detail=f"Session '{source_session_id}' was not found.")
    if source_session.session_type != "conversation":
        raise HTTPException(status_code=422, detail="Agency triggers can reference conversation sessions.")
    return source_session


async def _load_pending_fires(
    db_session: AsyncSession,
    *,
    agency_session_id: str,
    limit: int,
) -> list[AgencyFireRecord]:
    result = await db_session.execute(
        select(AgencyFireRecord)
        .where(
            AgencyFireRecord.agency_session_id == agency_session_id,
            AgencyFireRecord.status == AgencyFireStatus.PENDING.value,
        )
        .order_by(AgencyFireRecord.priority.asc(), AgencyFireRecord.created_at.asc())
        .limit(max(1, limit))
    )
    return list(result.scalars().all())


async def _blocking_agency_run(db_session: AsyncSession, agency_session: SessionRecord) -> RunRecord | None:
    for run_id in (agency_session.active_run_id, agency_session.head_run_id):
        if not isinstance(run_id, str):
            continue
        record = await db_session.get(RunRecord, run_id)
        if isinstance(record, RunRecord) and record.status in {"queued", "running"}:
            return record
    result = await db_session.execute(
        select(RunRecord)
        .where(RunRecord.session_id == agency_session.id, RunRecord.status.in_(["queued", "running"]))
        .order_by(RunRecord.sequence_no.asc())
        .limit(1)
    )
    return result.scalars().first()


async def _next_sequence_no(db_session: AsyncSession, session_id: str) -> int:
    result = await db_session.execute(select(func.max(RunRecord.sequence_no)).where(RunRecord.session_id == session_id))
    value = result.scalar_one_or_none()
    return value + 1 if isinstance(value, int) else 1


async def _load_fire_by_dedupe(db_session: AsyncSession, dedupe_key: str) -> AgencyFireRecord | None:
    result = await db_session.execute(select(AgencyFireRecord).where(AgencyFireRecord.dedupe_key == dedupe_key))
    return result.scalars().first()


def _dedupe_key(
    *,
    kind: str,
    source_session_id: str | None,
    source_run_id: str | None,
    client_token: str,
    scheduled_at: datetime,
) -> str:
    if kind == AgencyFireKind.TIMER.value:
        return f"agency:timer:{_as_utc_aware(scheduled_at).isoformat()}"
    if kind == AgencyFireKind.MEMORY_COMMITTED.value and isinstance(source_run_id, str):
        return f"agency:memory_committed:{source_run_id}"
    source_part = source_session_id or "global"
    return f"agency:{kind}:{source_part}:{client_token}"


def _fire_input_part(fire: AgencyFireRecord) -> CommandPart:
    return CommandPart(
        type="command",
        name="agency_fire",
        params={
            "fire_id": fire.id,
            "kind": fire.kind,
            "source_session_id": fire.source_session_id,
            "source_run_id": fire.source_run_id,
            "payload": dict(fire.payload or {}),
        },
    )


def _episode_metadata(
    *,
    agency_session: SessionRecord,
    run_id: str,
    fires: list[AgencyFireRecord],
    risk_policy: AgencyRiskPolicy,
) -> dict[str, Any]:
    source_session_ids = _unique_strings([fire.source_session_id for fire in fires])
    source_run_ids = _unique_strings([fire.source_run_id for fire in fires])
    return {
        "agency": {
            "kind": "episode",
            "agency_session_id": agency_session.id,
            "fire_ids": [fire.id for fire in fires],
            "trigger_kinds": _unique_strings([fire.kind for fire in fires]),
            "sources": [
                {
                    "fire_id": fire.id,
                    "source_session_id": fire.source_session_id,
                    "source_run_id": fire.source_run_id,
                    "kind": fire.kind,
                }
                for fire in fires
                if fire.source_session_id is not None or fire.source_run_id is not None
            ],
            "source_session_ids": source_session_ids,
            "primary_source_session_id": source_session_ids[0] if source_session_ids else None,
            "source_run_ids": source_run_ids,
            "episode_id": f"episode-{run_id}",
            "risk_policy": risk_policy.model_dump(mode="json"),
        },
        "restore_state": True,
    }


def _append_fires_to_run_metadata(run_record: RunRecord, fires: list[AgencyFireRecord], *, steered: bool) -> None:
    metadata = dict(run_record.run_metadata or {})
    agency = dict(metadata.get("agency") or {}) if isinstance(metadata.get("agency"), dict) else {}
    agency["fire_ids"] = _append_unique_strings(agency.get("fire_ids"), [fire.id for fire in fires])
    agency["trigger_kinds"] = _append_unique_strings(agency.get("trigger_kinds"), [fire.kind for fire in fires])
    agency["source_session_ids"] = _append_unique_strings(
        agency.get("source_session_ids"),
        [fire.source_session_id for fire in fires if isinstance(fire.source_session_id, str)],
    )
    agency["source_run_ids"] = _append_unique_strings(
        agency.get("source_run_ids"), [fire.source_run_id for fire in fires if isinstance(fire.source_run_id, str)]
    )
    sources = list(agency.get("sources") or []) if isinstance(agency.get("sources"), list) else []
    existing_fire_ids = {item.get("fire_id") for item in sources if isinstance(item, dict)}
    for fire in fires:
        if fire.id in existing_fire_ids:
            continue
        sources.append({
            "fire_id": fire.id,
            "source_session_id": fire.source_session_id,
            "source_run_id": fire.source_run_id,
            "kind": fire.kind,
        })
    agency["sources"] = sources
    if steered:
        agency["steered_fire_ids"] = _append_unique_strings(agency.get("steered_fire_ids"), [fire.id for fire in fires])
    metadata["agency"] = agency
    run_record.run_metadata = metadata


def _consumed_fire_ids(run_record: RunRecord) -> list[str]:
    agency = run_record.run_metadata.get("agency") if isinstance(run_record.run_metadata, dict) else None
    if not isinstance(agency, dict):
        return []
    explicit_ids = agency.get("consumed_fire_ids")
    fire_ids = agency.get("fire_ids")
    values = [item for item in fire_ids if isinstance(item, str)] if isinstance(fire_ids, list) else []
    if isinstance(explicit_ids, list):
        values = _append_unique_strings(values, [item for item in explicit_ids if isinstance(item, str)])
    return values


async def _mark_run_fires_consumed(
    db_session: AsyncSession,
    run_id: str,
    *,
    consumed_ids: list[str],
    consumed_at: datetime,
) -> None:
    result = await db_session.execute(
        select(AgencyFireRecord).where(
            AgencyFireRecord.id.in_(consumed_ids),
            AgencyFireRecord.run_id == run_id,
        )
    )
    for fire in result.scalars().all():
        fire.status = AgencyFireStatus.CONSUMED.value
        fire.consumed_at = consumed_at
        fire.run_id = run_id


async def _mark_run_fires_failed(db_session: AsyncSession, run_id: str) -> None:
    result = await db_session.execute(
        select(AgencyFireRecord).where(
            AgencyFireRecord.run_id == run_id,
            AgencyFireRecord.status.in_([
                AgencyFireStatus.SUBMITTED.value,
                AgencyFireStatus.STEERED.value,
                AgencyFireStatus.MERGED.value,
            ]),
        )
    )
    for fire in result.scalars().all():
        fire.status = AgencyFireStatus.FAILED.value


def _unique_strings(values: list[str | None]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if isinstance(value, str) and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _append_unique_strings(existing: object, values: list[str]) -> list[str]:
    result = [item for item in existing if isinstance(item, str)] if isinstance(existing, list) else []
    seen = set(result)
    for value in values:
        if isinstance(value, str) and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _as_utc_aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
