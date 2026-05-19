from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any, Literal
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ya_claw.config import ClawSettings
from ya_claw.controller.models import (
    AgencyFireKind,
    AgencyFireStatus,
    AgencyRiskPolicy,
    CommandPart,
    DispatchMode,
    SessionSubmitRequest,
    TriggerType,
)
from ya_claw.orm.tables import AgencyFireRecord, RunRecord, SessionRecord
from ya_claw.runtime_state import InMemoryRuntimeState

AGENCY_SINGLETON_SCOPE_KEY = "agency:global"
AGENCY_SINGLETON_SOURCE_SESSION_ID = sha256(AGENCY_SINGLETON_SCOPE_KEY.encode("utf-8")).hexdigest()[:32]
_FIRE_PRIORITIES: dict[str, int] = {
    AgencyFireKind.MESSAGE_OBSERVED.value: 20,
    AgencyFireKind.MEMORY_SESSION_COMPLETED.value: 30,
}


@dataclass(slots=True)
class AgencyFireDelivery:
    fire: AgencyFireRecord
    agency_session: SessionRecord
    run_id: str | None = None
    active_run_id: str | None = None
    delivery: Literal["pending", "submitted", "steered", "merged", "duplicate"] = "pending"


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
        fire_count_result = await db_session.execute(select(func.count()).select_from(AgencyFireRecord))
        deleted_fire_count = int(fire_count_result.scalar_one_or_none() or 0)
        await db_session.execute(delete(AgencyFireRecord))
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
        context_bundle: dict[str, Any] | None = None,
        scheduled_at: datetime | None = None,
        dedupe_key: str | None = None,
        dispatch: bool = True,
    ) -> AgencyFireDelivery:
        if not self._settings.agency_enabled:
            raise HTTPException(status_code=409, detail="Agency is disabled.")
        agency_session = await self.ensure_agency_session(db_session)
        fire_kind = kind.value if isinstance(kind, AgencyFireKind) else str(kind)
        if source_session_id is not None:
            await _load_source_conversation_session(db_session, source_session_id)
        insert_result = await self._insert_fire(
            db_session,
            agency_session=agency_session,
            kind=fire_kind,
            source_session_id=source_session_id,
            source_run_id=source_run_id,
            client_token=client_token,
            prompt=prompt,
            payload=_payload_with_context_bundle(payload, context_bundle),
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

    async def observe_message(
        self,
        db_session: AsyncSession,
        *,
        source_session_id: str,
        source_run_id: str | None,
        input_parts: list[Any],
        source_kind: str,
        client_token: str | None = None,
        metadata: dict[str, Any] | None = None,
        dispatch: bool = True,
    ) -> AgencyFireDelivery | None:
        source_session = await db_session.get(SessionRecord, source_session_id)
        if not isinstance(source_session, SessionRecord) or source_session.session_type != "conversation":
            return None
        return await self.create_fire(
            db_session,
            kind=AgencyFireKind.MESSAGE_OBSERVED,
            source_session_id=source_session_id,
            source_run_id=source_run_id,
            client_token=client_token,
            payload={
                "source_kind": source_kind,
                "source_session_id": source_session_id,
                "source_run_id": source_run_id,
                "input_parts": [_dump_input_part(part) for part in input_parts],
                "metadata": dict(metadata or {}),
            },
            dispatch=dispatch,
        )

    async def on_memory_session_completed(
        self,
        db_session: AsyncSession,
        *,
        source_session_id: str,
        memory_run_id: str,
        memory_session_id: str | None,
        memory_job_kind: str,
        output_text: str | None,
        output_summary: str | None,
        payload: dict[str, Any] | None = None,
        dispatch: bool = True,
    ) -> AgencyFireDelivery | None:
        source_session = await db_session.get(SessionRecord, source_session_id)
        if not isinstance(source_session, SessionRecord) or source_session.session_type != "conversation":
            return None
        return await self.create_fire(
            db_session,
            kind=AgencyFireKind.MEMORY_SESSION_COMPLETED,
            source_session_id=source_session_id,
            source_run_id=memory_run_id,
            client_token=memory_run_id,
            payload={
                "source_session_id": source_session_id,
                "memory_session_id": memory_session_id,
                "memory_run_id": memory_run_id,
                "memory_job_kind": memory_job_kind,
                "output_text": output_text,
                "output_summary": output_summary,
                "memory": dict(payload or {}),
            },
            dispatch=dispatch,
        )

    async def dispatch_due(self, db_session: AsyncSession) -> AgencyFireDelivery | None:
        await self.ensure_agency_session(db_session)
        await db_session.commit()
        return None

    async def next_timer_fire_at(self, db_session: AsyncSession) -> datetime | None:
        await self.ensure_agency_session(db_session)
        return None

    async def dispatch_pending(self, db_session: AsyncSession) -> AgencyFireDelivery:
        agency_session = await self.ensure_agency_session(db_session)
        async with self._runtime_state.session_lock(agency_session.id):
            fires = await _load_pending_fires(
                db_session,
                agency_session_id=agency_session.id,
                limit=self._settings.agency_fire_batch_limit,
            )
            if not fires:
                raise HTTPException(status_code=404, detail="Agency fire was not found.")

            from ya_claw.controller.session import SessionController

            response = await SessionController().submit_input_locked(
                db_session,
                self._settings,
                self._runtime_state,
                agency_session.id,
                SessionSubmitRequest(
                    input_parts=[_fire_input_part(fire) for fire in fires],
                    metadata=_episode_metadata(
                        agency_session=agency_session,
                        run_id=None,
                        fires=fires,
                        risk_policy=_resolve_risk_policy(settings=self._settings, agency_session=agency_session),
                    ),
                    dispatch_mode=DispatchMode.ASYNC,
                    trigger_type=TriggerType.AGENCY,
                ),
            )
            if response.delivery == "steered":
                _append_fires_to_run_metadata(await _require_run(db_session, response.run_id), fires, steered=True)
                for fire in fires:
                    fire.status = AgencyFireStatus.STEERED.value
                    fire.run_id = response.run_id
                    fire.active_run_id = response.run_id
                    fire.agency_session_id = agency_session.id
                delivery_kind: Literal["merged", "steered", "submitted"] = "steered"
            elif response.delivery == "merged":
                run_record = await _require_run(db_session, response.run_id)
                for fire in fires:
                    fire.status = AgencyFireStatus.MERGED.value
                    fire.run_id = run_record.id
                    fire.active_run_id = None
                    fire.agency_session_id = agency_session.id
                delivery_kind = "merged"
            else:
                run_record = await _require_run(db_session, response.run_id)
                for fire in fires:
                    fire.status = AgencyFireStatus.SUBMITTED.value
                    fire.run_id = run_record.id
                    fire.active_run_id = None
                    fire.agency_session_id = agency_session.id
                delivery_kind = "submitted"
            try:
                await db_session.commit()
            except IntegrityError:
                await db_session.rollback()
                return await self.dispatch_pending(db_session)
        await db_session.refresh(fires[0])
        await db_session.refresh(agency_session)
        if self._submit_run is not None and response.delivery == "submitted":
            self._submit_run(response.run_id)
        return AgencyFireDelivery(
            fire=fires[0],
            agency_session=agency_session,
            run_id=response.run_id if delivery_kind != "steered" else None,
            active_run_id=response.run_id if delivery_kind == "steered" else None,
            delivery=delivery_kind,
        )

    async def tick(self, db_session: AsyncSession) -> AgencyTickResult:
        result = AgencyTickResult()
        try:
            delivery = await self.dispatch_pending(db_session)
        except HTTPException as exc:
            if exc.status_code == 404:
                await db_session.commit()
                return result
            raise
        result.created_fire_ids.append(delivery.fire.id)
        if delivery.delivery == "submitted" and delivery.run_id is not None:
            result.submitted_run_ids.append(delivery.run_id)
        elif delivery.delivery == "steered":
            result.steered_fire_ids.append(delivery.fire.id)
        elif delivery.delivery == "merged":
            result.merged_fire_ids.append(delivery.fire.id)
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
    if kind == AgencyFireKind.MEMORY_SESSION_COMPLETED.value and isinstance(source_run_id, str):
        return f"agency:memory_session_completed:{source_run_id}"
    source_part = source_session_id or "global"
    return f"agency:{kind}:{source_part}:{client_token}"


def _fire_input_part(fire: AgencyFireRecord) -> CommandPart:
    payload = dict(fire.payload or {})
    return CommandPart(
        type="command",
        name="agency_fire",
        params={
            "fire_id": fire.id,
            "kind": fire.kind,
            "source": {
                "session_id": fire.source_session_id,
                "run_id": fire.source_run_id,
                "fire_id": fire.id,
                "kind": fire.kind,
            },
            "source_session_id": fire.source_session_id,
            "source_run_id": fire.source_run_id,
            "context_bundle": payload.get("context_bundle")
            if isinstance(payload.get("context_bundle"), dict)
            else None,
            "payload": payload,
        },
    )


def _episode_metadata(
    *,
    agency_session: SessionRecord,
    run_id: str | None,
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
            "episode_id": f"episode-{run_id}" if run_id is not None else None,
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


def _dump_input_part(part: Any) -> dict[str, Any]:
    if hasattr(part, "model_dump"):
        value = part.model_dump(mode="json")
        return dict(value) if isinstance(value, dict) else {"type": "value", "value": value}
    if isinstance(part, dict):
        return dict(part)
    return {"type": "value", "value": part}


async def _require_run(db_session: AsyncSession, run_id: str) -> RunRecord:
    record = await db_session.get(RunRecord, run_id)
    if not isinstance(record, RunRecord):
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' was not found.")
    return record


def _payload_with_context_bundle(
    payload: dict[str, Any] | None,
    context_bundle: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if context_bundle is None:
        return payload
    effective = dict(payload or {})
    effective["context_bundle"] = dict(context_bundle)
    return effective
