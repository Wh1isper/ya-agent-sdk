from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from typing import Any, Literal, cast
from uuid import uuid4

from fastapi import HTTPException
from pydantic import BaseModel
from pydantic_ai import DeferredToolRequests
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ya_claw.bridge.models import BridgeAdapterType, BridgeInboundMessage
from ya_claw.controller.models import (
    ActiveInteraction,
    InteractionRespondRequest,
    InteractionRespondResponse,
    UserInteraction,
)
from ya_claw.execution.input_inbox import accept_run_input, lock_run_record
from ya_claw.orm.tables import (
    BridgeHitlMessageRecord,
    HitlBatchRecord,
    HitlDeferredInputRecord,
    HitlInteractionRecord,
    RunRecord,
)


@dataclass(frozen=True, slots=True)
class InteractionResponseResult:
    response: InteractionRespondResponse
    resolution: UserInteraction


class DeferredInputPayload(BaseModel):
    id: str
    batch_id: str
    session_id: str
    run_id: str
    sequence_no: int
    input_parts: list[dict[str, Any]]
    source_metadata: dict[str, Any]


class HitlBatchPayload(BaseModel):
    batch_id: str
    run_id: str
    session_id: str
    active_interactions: list[ActiveInteraction]
    current_interaction: ActiveInteraction | None = None


class HitlController:
    async def create_batch(
        self,
        db_session: AsyncSession,
        *,
        session_id: str,
        run_id: str,
        interactions: list[ActiveInteraction],
        deferred_requests: DeferredToolRequests | dict[str, Any] | None = None,
    ) -> HitlBatchPayload:
        if not interactions:
            raise ValueError("interactions must not be empty")
        existing = await self.get_pending_batch_for_run(db_session, run_id)
        if existing is not None:
            active = await self.get_active_interactions(
                db_session,
                run_id=run_id,
                batch_id=existing.id,
            )
            return HitlBatchPayload(
                batch_id=existing.id,
                run_id=run_id,
                session_id=session_id,
                active_interactions=active,
                current_interaction=_current_interaction(active),
            )

        batch = HitlBatchRecord(
            id=uuid4().hex,
            session_id=session_id,
            run_id=run_id,
            status="pending",
            current_interaction_id=interactions[0].interaction_id,
            deferred_requests=_serialize_deferred_requests(deferred_requests),
        )
        db_session.add(batch)
        for interaction in interactions:
            db_session.add(_interaction_record_from_model(batch.id, interaction))
        await db_session.flush()
        return HitlBatchPayload(
            batch_id=batch.id,
            run_id=run_id,
            session_id=session_id,
            active_interactions=interactions,
            current_interaction=interactions[0],
        )

    async def get_pending_batch_for_run(self, db_session: AsyncSession, run_id: str) -> HitlBatchRecord | None:
        result = await db_session.execute(
            select(HitlBatchRecord)
            .where(HitlBatchRecord.run_id == run_id, HitlBatchRecord.status == "pending")
            .order_by(HitlBatchRecord.created_at.desc())
            .limit(1)
        )
        record = result.scalar_one_or_none()
        return record if isinstance(record, HitlBatchRecord) else None

    async def get_pending_batch_for_session(self, db_session: AsyncSession, session_id: str) -> HitlBatchRecord | None:
        result = await db_session.execute(
            select(HitlBatchRecord)
            .where(HitlBatchRecord.session_id == session_id, HitlBatchRecord.status == "pending")
            .order_by(HitlBatchRecord.created_at.desc())
            .limit(1)
        )
        record = result.scalar_one_or_none()
        return record if isinstance(record, HitlBatchRecord) else None

    async def get_active_interactions(
        self,
        db_session: AsyncSession,
        *,
        run_id: str,
        batch_id: str | None = None,
    ) -> list[ActiveInteraction]:
        statement = select(HitlInteractionRecord).where(HitlInteractionRecord.run_id == run_id)
        if batch_id is not None:
            statement = statement.where(HitlInteractionRecord.batch_id == batch_id)
        result = await db_session.execute(statement.order_by(HitlInteractionRecord.sequence_no.asc()))
        return [_interaction_model_from_record(record) for record in result.scalars().all()]

    async def respond_interaction(
        self,
        db_session: AsyncSession,
        run_id: str,
        interaction_id: str,
        request: InteractionRespondRequest,
    ) -> InteractionResponseResult:
        run_record = await lock_run_record(db_session, run_id)
        if not isinstance(run_record, RunRecord):
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' was not found.")
        if run_record.status != "running":
            raise HTTPException(status_code=409, detail=f"Run '{run_id}' is not waiting for interaction.")

        active_batch_id = run_record.run_metadata.get("active_hitl_batch_id")
        batch = None
        if isinstance(active_batch_id, str):
            batch = await db_session.get(HitlBatchRecord, active_batch_id, with_for_update=True)
        if not isinstance(batch, HitlBatchRecord):
            batch_result = await db_session.execute(
                select(HitlBatchRecord)
                .where(HitlBatchRecord.run_id == run_id, HitlBatchRecord.status == "pending")
                .order_by(HitlBatchRecord.created_at.desc())
                .with_for_update()
                .limit(1)
            )
            batch = batch_result.scalar_one_or_none()
        if not isinstance(batch, HitlBatchRecord):
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' has no active HITL batch.")

        result = await db_session.execute(
            select(HitlInteractionRecord)
            .where(
                HitlInteractionRecord.batch_id == batch.id,
                HitlInteractionRecord.interaction_id == interaction_id,
            )
            .with_for_update()
        )
        interaction_record = result.scalar_one_or_none()
        if not isinstance(interaction_record, HitlInteractionRecord):
            raise HTTPException(status_code=404, detail=f"Interaction '{interaction_id}' was not found.")

        now = datetime.now(UTC)
        if interaction_record.status == "pending":
            if batch.status != "pending":
                raise HTTPException(status_code=409, detail=f"HITL batch '{batch.id}' is already {batch.status}.")
            interaction_record.status = "approved" if request.approved else "denied"
            interaction_record.response = {
                "approved": request.approved,
                "reason": request.reason,
                "user_input": request.user_input,
                "client_token": request.client_token,
            }
            interaction_record.resolved_at = now
            interaction_record.updated_at = now

        active_interactions = await self.get_active_interactions(
            db_session,
            run_id=run_id,
            batch_id=batch.id,
        )
        current = _current_interaction(active_interactions)
        remaining = sum(1 for interaction in active_interactions if interaction.status == "pending")
        batch.current_interaction_id = current.interaction_id if current is not None else None
        batch.updated_at = now

        metadata = dict(run_record.run_metadata)
        metadata["active_interactions"] = [interaction.model_dump(mode="json") for interaction in active_interactions]
        metadata["active_hitl_batch_id"] = batch.id
        run_record.run_metadata = metadata

        await db_session.flush()
        resolved = _interaction_model_from_record(interaction_record)
        stored_response = interaction_record.response if isinstance(interaction_record.response, dict) else {}
        return InteractionResponseResult(
            response=InteractionRespondResponse(
                session_id=run_record.session_id,
                run_id=run_id,
                interaction_id=resolved.interaction_id,
                tool_call_id=resolved.tool_call_id,
                status=resolved.status,
                remaining_interaction_count=remaining,
                current_interaction=current,
            ),
            resolution=UserInteraction(
                tool_call_id=resolved.tool_call_id,
                approved=resolved.status == "approved",
                reason=stored_response.get("reason"),
                user_input=stored_response.get("user_input"),
            ),
        )

    async def mark_batch_completed(self, db_session: AsyncSession, *, run_id: str) -> None:
        batch = await self.get_pending_batch_for_run(db_session, run_id)
        if batch is None:
            return
        now = datetime.now(UTC)
        batch.status = "completed"
        batch.current_interaction_id = None
        batch.completed_at = now
        batch.updated_at = now
        await db_session.flush()

    async def cancel_pending_batch(
        self,
        db_session: AsyncSession,
        *,
        run_id: str,
        reason: str,
    ) -> bool:
        """Close orphaned HITL state when its process-owned run is interrupted."""
        batch = await self.get_pending_batch_for_run(db_session, run_id)
        now = datetime.now(UTC)
        if batch is not None:
            batch.status = "cancelled"
            batch.current_interaction_id = None
            batch.completed_at = now
            batch.updated_at = now
            interaction_result = await db_session.execute(
                select(HitlInteractionRecord)
                .where(
                    HitlInteractionRecord.batch_id == batch.id,
                    HitlInteractionRecord.status == "pending",
                )
                .with_for_update()
            )
            for interaction in interaction_result.scalars().all():
                interaction.status = "denied"
                interaction.response = {"approved": False, "reason": reason, "user_input": None}
                interaction.resolved_at = now
                interaction.updated_at = now

        input_result = await db_session.execute(
            select(HitlDeferredInputRecord)
            .where(
                HitlDeferredInputRecord.run_id == run_id,
                HitlDeferredInputRecord.status == "pending",
            )
            .with_for_update()
        )
        deferred_inputs = list(input_result.scalars().all())
        for deferred_input in deferred_inputs:
            deferred_input.status = "discarded"
            deferred_input.updated_at = now
        if batch is not None or deferred_inputs:
            await db_session.flush()
            return True
        return False

    async def enqueue_deferred_input(
        self,
        db_session: AsyncSession,
        *,
        batch: HitlBatchRecord,
        message: BridgeInboundMessage,
        conversation_id: str | None,
        input_parts: list[dict[str, Any]],
        source_metadata: dict[str, Any] | None = None,
    ) -> int | None:
        run_record = await lock_run_record(db_session, batch.run_id)
        if not isinstance(run_record, RunRecord) or run_record.status != "running":
            return None
        locked_batch = await db_session.get(
            HitlBatchRecord,
            batch.id,
            populate_existing=True,
            with_for_update=True,
        )
        if (
            not isinstance(locked_batch, HitlBatchRecord)
            or locked_batch.run_id != run_record.id
            or locked_batch.status != "pending"
        ):
            return None
        existing = await self._find_existing_deferred_input(db_session, message)
        if existing is not None:
            return await self.count_pending_deferred_inputs(
                db_session,
                run_id=locked_batch.run_id,
                batch_id=locked_batch.id,
            )

        next_sequence_result = await db_session.execute(
            select(func.max(HitlDeferredInputRecord.sequence_no)).where(
                HitlDeferredInputRecord.batch_id == locked_batch.id
            )
        )
        next_sequence = (next_sequence_result.scalar_one_or_none() or 0) + 1
        try:
            async with db_session.begin_nested():
                db_session.add(
                    HitlDeferredInputRecord(
                        id=uuid4().hex,
                        batch_id=locked_batch.id,
                        session_id=locked_batch.session_id,
                        run_id=locked_batch.run_id,
                        conversation_id=conversation_id,
                        adapter=message.adapter,
                        tenant_key=message.tenant_key,
                        external_event_id=message.event_id,
                        external_message_id=message.message_id,
                        external_chat_id=message.chat_id,
                        sequence_no=next_sequence,
                        input_parts=input_parts,
                        source_metadata=dict(source_metadata or {}),
                        status="pending",
                    )
                )
                await db_session.flush()
        except IntegrityError:
            pass
        return await self.count_pending_deferred_inputs(
            db_session,
            run_id=locked_batch.run_id,
            batch_id=locked_batch.id,
        )

    async def count_pending_deferred_inputs(
        self,
        db_session: AsyncSession,
        *,
        run_id: str,
        batch_id: str | None = None,
    ) -> int:
        statement = (
            select(func.count())
            .select_from(HitlDeferredInputRecord)
            .where(
                HitlDeferredInputRecord.run_id == run_id,
                HitlDeferredInputRecord.status == "pending",
            )
        )
        if batch_id is not None:
            statement = statement.where(HitlDeferredInputRecord.batch_id == batch_id)
        result = await db_session.execute(statement)
        value = result.scalar_one()
        return int(value)

    async def stage_deferred_inputs(
        self,
        db_session: AsyncSession,
        *,
        run_id: str,
        batch_id: str,
    ) -> list[DeferredInputPayload]:
        """Atomically move HITL-gap inputs into the canonical run inbox."""
        run_record = await lock_run_record(db_session, run_id)
        if not isinstance(run_record, RunRecord):
            raise KeyError(run_id)
        result = await db_session.execute(
            select(HitlDeferredInputRecord)
            .where(
                HitlDeferredInputRecord.run_id == run_id,
                HitlDeferredInputRecord.batch_id == batch_id,
                HitlDeferredInputRecord.status == "pending",
            )
            .order_by(HitlDeferredInputRecord.sequence_no.asc(), HitlDeferredInputRecord.created_at.asc())
            .with_for_update()
        )
        records = [record for record in result.scalars().all() if isinstance(record, HitlDeferredInputRecord)]
        now = datetime.now(UTC)
        payloads: list[DeferredInputPayload] = []
        for record in records:
            await accept_run_input(
                db_session,
                run_record,
                list(record.input_parts),
                delivery_key=f"hitl-deferred:{record.id}",
                origin="user",
            )
            record.status = "consumed"
            record.consumed_at = now
            record.updated_at = now
            payloads.append(
                DeferredInputPayload(
                    id=record.id,
                    batch_id=record.batch_id,
                    session_id=record.session_id,
                    run_id=record.run_id,
                    sequence_no=record.sequence_no,
                    input_parts=list(record.input_parts),
                    source_metadata=dict(record.source_metadata),
                )
            )
        await db_session.flush()
        return payloads

    async def upsert_bridge_hitl_message(
        self,
        db_session: AsyncSession,
        *,
        adapter: BridgeAdapterType | str,
        tenant_key: str,
        external_chat_id: str,
        external_message_id: str,
        session_id: str,
        run_id: str,
        batch_id: str | None = None,
        interaction_id: str | None = None,
        status: str = "active",
    ) -> BridgeHitlMessageRecord:
        result = await db_session.execute(
            select(BridgeHitlMessageRecord).where(
                BridgeHitlMessageRecord.adapter == adapter,
                BridgeHitlMessageRecord.tenant_key == tenant_key,
                BridgeHitlMessageRecord.external_message_id == external_message_id,
            )
        )
        record = result.scalar_one_or_none()
        now = datetime.now(UTC)
        if isinstance(record, BridgeHitlMessageRecord):
            record.external_chat_id = external_chat_id
            record.session_id = session_id
            record.run_id = run_id
            record.batch_id = batch_id
            record.interaction_id = interaction_id
            record.status = status
            record.updated_at = now
            if status == "completed":
                record.completed_at = now
            await db_session.flush()
            return record

        record = BridgeHitlMessageRecord(
            id=uuid4().hex,
            adapter=str(adapter),
            tenant_key=tenant_key,
            external_chat_id=external_chat_id,
            external_message_id=external_message_id,
            session_id=session_id,
            run_id=run_id,
            batch_id=batch_id,
            interaction_id=interaction_id,
            status=status,
            completed_at=now if status == "completed" else None,
        )
        db_session.add(record)
        await db_session.flush()
        return record

    async def get_bridge_hitl_message(
        self,
        db_session: AsyncSession,
        *,
        adapter: BridgeAdapterType | str,
        tenant_key: str,
        run_id: str,
    ) -> BridgeHitlMessageRecord | None:
        result = await db_session.execute(
            select(BridgeHitlMessageRecord)
            .where(
                BridgeHitlMessageRecord.adapter == adapter,
                BridgeHitlMessageRecord.tenant_key == tenant_key,
                BridgeHitlMessageRecord.run_id == run_id,
            )
            .order_by(BridgeHitlMessageRecord.created_at.desc())
            .limit(1)
        )
        record = result.scalar_one_or_none()
        return record if isinstance(record, BridgeHitlMessageRecord) else None

    async def _find_existing_deferred_input(
        self,
        db_session: AsyncSession,
        message: BridgeInboundMessage,
    ) -> HitlDeferredInputRecord | None:
        result = await db_session.execute(
            select(HitlDeferredInputRecord).where(
                HitlDeferredInputRecord.adapter == message.adapter,
                HitlDeferredInputRecord.tenant_key == message.tenant_key,
                HitlDeferredInputRecord.external_event_id == message.event_id,
            )
        )
        record = result.scalar_one_or_none()
        if isinstance(record, HitlDeferredInputRecord):
            return record
        result = await db_session.execute(
            select(HitlDeferredInputRecord).where(
                HitlDeferredInputRecord.adapter == message.adapter,
                HitlDeferredInputRecord.tenant_key == message.tenant_key,
                HitlDeferredInputRecord.external_message_id == message.message_id,
            )
        )
        record = result.scalar_one_or_none()
        return record if isinstance(record, HitlDeferredInputRecord) else None


def _interaction_record_from_model(batch_id: str, interaction: ActiveInteraction) -> HitlInteractionRecord:
    return HitlInteractionRecord(
        id=uuid4().hex,
        batch_id=batch_id,
        session_id=interaction.session_id,
        run_id=interaction.run_id,
        interaction_id=interaction.interaction_id,
        tool_call_id=interaction.tool_call_id,
        tool_name=interaction.tool_name,
        kind=interaction.kind,
        sequence_no=interaction.sequence_no,
        total_count=interaction.total_count,
        status=interaction.status,
        title=interaction.title,
        description=interaction.description,
        arguments_preview=interaction.arguments_preview,
        interaction_metadata=dict(interaction.metadata),
        created_at=interaction.created_at,
        resolved_at=interaction.resolved_at,
    )


def _interaction_model_from_record(record: HitlInteractionRecord) -> ActiveInteraction:
    return ActiveInteraction(
        interaction_id=record.interaction_id,
        run_id=record.run_id,
        session_id=record.session_id,
        tool_call_id=record.tool_call_id,
        tool_name=record.tool_name,
        kind=record.kind,
        title=record.title,
        description=record.description,
        arguments_preview=record.arguments_preview,
        metadata=dict(record.interaction_metadata),
        status=_interaction_status(record.status),
        sequence_no=record.sequence_no,
        total_count=record.total_count,
        created_at=record.created_at,
        resolved_at=record.resolved_at,
    )


def _current_interaction(interactions: list[ActiveInteraction]) -> ActiveInteraction | None:
    for interaction in interactions:
        if interaction.status == "pending":
            return interaction
    return None


def _interaction_status(value: str) -> Literal["pending", "approved", "denied"]:
    if value == "approved" or value == "denied":
        return value
    return "pending"


def _serialize_deferred_requests(value: DeferredToolRequests | dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return dict(value)
    if is_dataclass(value) and not isinstance(value, type):
        dumped = asdict(value)
        return cast(dict[str, Any], dumped) if isinstance(dumped, dict) else {"value": dumped}
    return {"repr": repr(value)}


def user_interaction_from_record(record: HitlInteractionRecord) -> UserInteraction | None:
    if record.status == "pending":
        return None
    response = record.response if isinstance(record.response, dict) else {}
    return UserInteraction(
        tool_call_id=record.tool_call_id,
        approved=record.status == "approved",
        reason=response.get("reason") if isinstance(response.get("reason"), str) else None,
        user_input=response.get("user_input"),
    )
