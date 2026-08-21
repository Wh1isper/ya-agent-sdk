"""Durable SQL inbox for structured input sent to active Claw runs."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from ya_agent_sdk.inputs import EnqueueReceipt, InputDisposition
from ya_agent_stream_protocol.sdk import AguiAdapterConfig, AguiEventAdapter

from ya_claw.orm.tables import RunInputInboxRecord, RunRecord, SessionAsyncTaskRecord
from ya_claw.runtime_state import InMemoryRuntimeState

RUN_INPUT_CLOSED_CODE = "run_input_closed"
_OPEN_INPUT_STATUSES = ("accepted", "enqueued")
_ACTIVE_RUN_STATUSES = ("queued", "running")
_AGUI_ADAPTER_CONFIG = AguiAdapterConfig(run_event_prefix="ya_claw")


class RunInputClosedError(RuntimeError):
    """The authoritative run state closed before input admission."""


@dataclass(frozen=True, slots=True)
class RunInputAdmission:
    record: RunInputInboxRecord
    created: bool


async def lock_run_record(
    db_session: AsyncSession,
    run_id: str,
) -> RunRecord | None:
    """Acquire the database serialization boundary shared by ingress and termination."""
    await db_session.execute(update(RunRecord).where(RunRecord.id == run_id).values(status=RunRecord.status))
    record = await db_session.get(
        RunRecord,
        run_id,
        populate_existing=True,
        with_for_update=True,
    )
    return record if isinstance(record, RunRecord) else None


async def admit_run_input(
    db_session: AsyncSession,
    run_record: RunRecord,
    input_parts: list[dict[str, Any]],
    *,
    delivery_key: str | None = None,
    origin: str = "user",
) -> RunInputAdmission:
    """Insert one idempotent input and report whether this call created it."""
    locked_run = await lock_run_record(db_session, run_record.id)
    if locked_run is None:
        raise RunInputClosedError(f"Run {run_record.id!r} no longer exists")
    normalized_key = delivery_key.strip() if isinstance(delivery_key, str) else ""
    input_id = uuid4().hex
    if not normalized_key:
        normalized_key = input_id
    result = await db_session.execute(
        select(RunInputInboxRecord)
        .where(
            RunInputInboxRecord.run_id == run_record.id,
            RunInputInboxRecord.delivery_key == normalized_key,
        )
        .limit(1)
    )
    existing = result.scalar_one_or_none()
    if isinstance(existing, RunInputInboxRecord):
        if existing.input_parts != input_parts or existing.origin != origin:
            raise ValueError(f"Run input delivery key {normalized_key!r} was reused with different content")
        return RunInputAdmission(record=existing, created=False)
    if locked_run.status not in _ACTIVE_RUN_STATUSES:
        raise RunInputClosedError(f"Run {run_record.id!r} is not active")

    record = RunInputInboxRecord(
        id=input_id,
        run_id=run_record.id,
        delivery_key=normalized_key,
        origin=origin,
        status="accepted",
        input_parts=input_parts,
    )
    try:
        async with db_session.begin_nested():
            db_session.add(record)
            await db_session.flush()
    except IntegrityError:
        result = await db_session.execute(
            select(RunInputInboxRecord)
            .where(
                RunInputInboxRecord.run_id == run_record.id,
                RunInputInboxRecord.delivery_key == normalized_key,
            )
            .limit(1)
        )
        concurrent = result.scalar_one_or_none()
        if not isinstance(concurrent, RunInputInboxRecord):
            raise
        if concurrent.input_parts != input_parts or concurrent.origin != origin:
            raise ValueError(f"Run input delivery key {normalized_key!r} was reused with different content") from None
        return RunInputAdmission(record=concurrent, created=False)
    return RunInputAdmission(record=record, created=True)


async def accept_run_input(
    db_session: AsyncSession,
    run_record: RunRecord,
    input_parts: list[dict[str, Any]],
    *,
    delivery_key: str | None = None,
    origin: str = "user",
) -> RunInputInboxRecord:
    """Insert one idempotent input before any process-local delivery attempt."""
    admission = await admit_run_input(
        db_session,
        run_record,
        input_parts,
        delivery_key=delivery_key,
        origin=origin,
    )
    return admission.record


async def deliver_accepted_run_inputs(
    db_session: AsyncSession,
    runtime_state: InMemoryRuntimeState,
    run_id: str,
) -> list[RunInputInboxRecord]:
    """Deliver accepted rows in order; unavailable ingress leaves them durable."""
    async with runtime_state.run_input_lock(run_id):
        run_record = await lock_run_record(db_session, run_id)
        if run_record is None:
            await db_session.rollback()
            return []
        if run_record.status not in _ACTIVE_RUN_STATUSES:
            await reject_open_run_inputs(
                db_session,
                run_id=run_id,
                reason=f"Run is already terminal with status {run_record.status}.",
                commit=False,
            )
            await db_session.commit()
            return []
        result = await db_session.execute(
            select(RunInputInboxRecord)
            .where(
                RunInputInboxRecord.run_id == run_id,
                RunInputInboxRecord.status == "accepted",
            )
            .order_by(RunInputInboxRecord.created_at.asc(), RunInputInboxRecord.id.asc())
        )
        records = list(result.scalars().all())
        delivered: list[RunInputInboxRecord] = []
        for record in records:
            try:
                receipt = await runtime_state.record_steering(
                    run_id,
                    record.id,
                    list(record.input_parts),
                )
                if not isinstance(receipt, EnqueueReceipt):
                    raise TypeError("Run input ingress must return an EnqueueReceipt")
            except (KeyError, RuntimeError):
                break
            except (ValueError, OSError, TypeError) as exc:
                record.attempt_count += 1
                record.status = "rejected"
                record.error_message = str(exc)[:4000]
                record.updated_at = datetime.now(UTC)
                delivered.append(record)
                continue
            record.attempt_count += 1
            record.sdk_input_id = receipt.input_id
            record.enqueue_id = receipt.enqueue_id
            record.error_message = None
            if receipt.disposition is InputDisposition.rejected:
                record.status = "rejected"
                record.error_message = "SDK logical input router rejected the input"
            elif receipt.disposition is InputDisposition.applied:
                record.status = "applied"
                record.applied_at = datetime.now(UTC)
            elif receipt.disposition is InputDisposition.enqueued:
                record.status = "enqueued"
            else:
                record.status = "accepted"
            record.updated_at = datetime.now(UTC)
            delivered.append(record)
        enqueued = [record for record in delivered if record.status == "enqueued"]
        await db_session.commit()
        await _publish_enqueued_run_input_events(
            runtime_state,
            run_record=run_record,
            records=enqueued,
        )
        return delivered


async def _publish_enqueued_run_input_events(
    runtime_state: InMemoryRuntimeState,
    *,
    run_record: RunRecord,
    records: list[RunInputInboxRecord],
) -> None:
    if not records:
        return
    agui_adapter = AguiEventAdapter(
        session_id=run_record.session_id,
        run_id=run_record.id,
        config=_AGUI_ADAPTER_CONFIG,
    )
    event = agui_adapter.build_run_custom_event(
        "input_enqueued",
        {"disposition": InputDisposition.enqueued.value},
    )
    for _ in records:
        with suppress(KeyError):
            await runtime_state.append_run_event(run_record.id, event)


async def mark_run_input_applied(
    db_session: AsyncSession,
    *,
    run_id: str,
    sdk_input_id: str | None,
    enqueue_id: str,
) -> RunInputInboxRecord | None:
    """Persist native application before execution advances beyond its event."""
    run_record = await lock_run_record(db_session, run_id)
    if run_record is None:
        await db_session.rollback()
        return None
    predicates = [RunInputInboxRecord.enqueue_id == enqueue_id]
    if isinstance(sdk_input_id, str):
        predicates.append(RunInputInboxRecord.sdk_input_id == sdk_input_id)
    result = await db_session.execute(
        select(RunInputInboxRecord)
        .where(
            RunInputInboxRecord.run_id == run_id,
            RunInputInboxRecord.status != "rejected",
            or_(*predicates),
        )
        .order_by(RunInputInboxRecord.created_at.asc())
        .with_for_update()
        .limit(1)
    )
    record = result.scalar_one_or_none()
    if not isinstance(record, RunInputInboxRecord):
        await db_session.commit()
        return None
    if record.status == "applied":
        await db_session.commit()
        return None
    if run_record.status not in _ACTIVE_RUN_STATUSES:
        record.status = "rejected"
        record.error_message = f"Run is already terminal with status {run_record.status}."
        record.updated_at = datetime.now(UTC)
        await db_session.commit()
        return None
    record.status = "applied"
    record.enqueue_id = enqueue_id
    record.updated_at = datetime.now(UTC)
    record.applied_at = record.updated_at
    task_result = await db_session.execute(
        select(SessionAsyncTaskRecord)
        .where(SessionAsyncTaskRecord.delivery_id == record.delivery_key)
        .with_for_update()
    )
    for task_record in task_result.scalars().all():
        task_record.delivery_status = "applied"
        task_record.updated_at = record.updated_at
    await db_session.commit()
    return record


async def reject_open_run_inputs(
    db_session: AsyncSession,
    *,
    run_id: str,
    reason: str,
    commit: bool = True,
) -> int:
    """Reject input that can no longer be applied to a terminal run."""
    result = await db_session.execute(
        select(RunInputInboxRecord)
        .where(
            RunInputInboxRecord.run_id == run_id,
            RunInputInboxRecord.status.in_(_OPEN_INPUT_STATUSES),
        )
        .with_for_update()
    )
    records = list(result.scalars().all())
    now = datetime.now(UTC)
    for record in records:
        record.status = "rejected"
        record.error_message = reason[:4000]
        record.updated_at = now
        if record.origin == "feature":
            task_result = await db_session.execute(
                select(SessionAsyncTaskRecord)
                .where(SessionAsyncTaskRecord.delivery_id == record.delivery_key)
                .with_for_update()
            )
            for task_record in task_result.scalars().all():
                if task_record.delivery_status != "applied":
                    task_record.delivery_status = "accepted"
                    task_record.delivery_run_id = None
                    task_record.updated_at = now
    if records:
        if commit:
            await db_session.commit()
        else:
            await db_session.flush()
    return len(records)
