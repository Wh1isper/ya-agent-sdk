from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

from fastapi import HTTPException
from loguru import logger
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from ya_agent_sdk.capabilities import CapabilityCatalog
from ya_agent_sdk.subagents import (
    ResolvedSubagentPlan,
    SubagentInputState,
    SubagentPlanDescriptor,
    SubagentSpec,
)

from ya_claw.config import ClawSettings
from ya_claw.controller.models import (
    AsyncTaskCancelRequest,
    AsyncTaskDetail,
    AsyncTaskListResponse,
    AsyncTaskResponse,
    AsyncTaskSpawnRequest,
    AsyncTaskStatus,
    AsyncTaskSteerRequest,
    AsyncTaskSummary,
    CommandPart,
    DispatchMode,
    InputPart,
    RunCreateRequest,
    SessionSummary,
    SteerRequest,
    TextPart,
    TriggerType,
    active_interactions_from_run_record,
    run_summary_from_record,
    session_summary_from_record,
)
from ya_claw.controller.run import RunController
from ya_claw.controller.session_lifecycle import lock_session_reference
from ya_claw.controller.store import read_run_message_blob_if_exists, read_run_state_blob_if_exists
from ya_claw.execution.input_inbox import (
    accept_run_input,
    deliver_accepted_run_inputs,
    lock_run_record,
)
from ya_claw.execution.subagents import (
    resolve_claw_subagent_plan,
    restore_claw_subagent_plan,
)
from ya_claw.orm.tables import RunInputInboxRecord, RunRecord, SessionAsyncTaskRecord, SessionRecord
from ya_claw.runtime_state import InMemoryRuntimeState

_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
_ACTIVE_STATUSES = {"queued", "running"}


@dataclass(frozen=True, slots=True)
class _WakeAction:
    run_record: RunRecord
    deliver_input: bool = False
    publish_created: bool = False


@dataclass(frozen=True, slots=True)
class _WakeResolution:
    delivery_run_id: str | None
    delivery_status: str
    action: _WakeAction | None = None


@runtime_checkable
class ResolvedSubagentProfile(Protocol):
    subagent_specs: tuple[SubagentSpec, ...]


@runtime_checkable
class ProfileResolverProtocol(Protocol):
    async def resolve(
        self,
        profile_name: str | None,
    ) -> ResolvedSubagentProfile: ...


class AsyncTaskController:
    def __init__(self) -> None:
        self._run_controller = RunController()

    async def _lock_parent_and_replay_sdk(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        *,
        parent_session_id: str,
        request: AsyncTaskSpawnRequest,
    ) -> tuple[SessionRecord, AsyncTaskResponse | None]:
        await self._load_parent_session(db_session, parent_session_id)
        parent_session = await lock_session_reference(db_session, parent_session_id)
        if not isinstance(parent_session, SessionRecord):
            raise HTTPException(
                status_code=404,
                detail=f"Parent session '{parent_session_id}' was not found.",
            )
        if request.sdk_idempotency_key is None:
            return parent_session, None
        if request.sdk_owner_scope_id != parent_session_id:
            raise HTTPException(
                status_code=409,
                detail="SDK subagent owner scope must match the parent session.",
            )
        existing = await self._load_task_by_sdk_idempotency(
            db_session,
            owner_scope_id=parent_session_id,
            idempotency_key=request.sdk_idempotency_key,
        )
        replay = (
            await self._sdk_replay_response(
                db_session,
                settings,
                existing,
                request,
            )
            if isinstance(existing, SessionAsyncTaskRecord)
            else None
        )
        return parent_session, replay

    async def spawn_delegate(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        runtime_state: InMemoryRuntimeState,
        *,
        parent_session_id: str,
        parent_run_id: str | None,
        parent_agent_id: str = "main",
        request: AsyncTaskSpawnRequest,
        profile_resolver: ProfileResolverProtocol | None = None,
    ) -> AsyncTaskResponse:
        parent_session, sdk_replay = await self._lock_parent_and_replay_sdk(
            db_session,
            settings,
            parent_session_id=parent_session_id,
            request=request,
        )
        if sdk_replay is not None:
            return sdk_replay
        sdk_request = request.sdk_idempotency_key is not None

        name = await self._resolve_name(
            db_session,
            parent_session_id=parent_session_id,
            subagent_name=request.subagent_name,
            requested_name=request.name,
            preserve_requested_name=sdk_request,
        )
        existing = await self._load_task_by_name(
            db_session,
            parent_session_id=parent_session_id,
            name=name,
        )
        if existing is None:
            subagent_plan, resume_task = await self._resolve_spawn_plan(
                db_session,
                parent_session=parent_session,
                request=request,
                profile_resolver=profile_resolver,
                settings=settings,
            )
            wake_policy = "record_only" if request.context.get("mode") == "foreground" else str(request.wake_policy)
            try:
                detail = await self._create_task(
                    db_session,
                    settings,
                    runtime_state,
                    parent_session=parent_session,
                    parent_run_id=parent_run_id,
                    parent_agent_id=parent_agent_id,
                    subagent_name=request.subagent_name,
                    subagent_plan=subagent_plan,
                    name=name,
                    prompt=request.prompt,
                    context=request.context,
                    wake_policy=wake_policy,
                    sdk_owner_scope_id=request.sdk_owner_scope_id,
                    sdk_idempotency_key=request.sdk_idempotency_key,
                    sdk_intent_fingerprint=request.sdk_intent_fingerprint,
                    resume_task=resume_task,
                )
            except IntegrityError:
                await db_session.rollback()
                existing = await self._load_spawn_integrity_conflict(
                    db_session,
                    parent_session_id=parent_session_id,
                    request=request,
                    name=name,
                )
                if not isinstance(existing, SessionAsyncTaskRecord):
                    raise
            else:
                detail.delivery = "submitted"
                return AsyncTaskResponse(task=detail)

        if sdk_request:
            if existing.sdk_idempotency_key != request.sdk_idempotency_key:
                raise HTTPException(
                    status_code=409,
                    detail=f"SDK subagent execution handle '{name}' already exists.",
                )
            return await self._sdk_replay_response(
                db_session,
                settings,
                existing,
                request,
            )
        if existing.subagent_name != request.subagent_name:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Async subagent name '{name}' belongs to route '{existing.subagent_name}', "
                    f"not '{request.subagent_name}'."
                ),
            )
        await self._refresh_task_status(db_session, existing)
        if existing.status in _ACTIVE_STATUSES:
            detail = await self._build_detail(db_session, settings, existing)
            detail.delivery = "existing_active"
            detail.instruction = (
                f"Subagent execution '{existing.name}' is {existing.status}. "
                "Use steer_subagent for additional input while it is running."
            )
            return AsyncTaskResponse(task=detail)

        detail = await self._resume_task(
            db_session,
            settings,
            runtime_state,
            task_record=existing,
            parent_run_id=parent_run_id,
            prompt=request.prompt,
            context=request.context,
            wake_policy=str(request.wake_policy),
        )
        detail.delivery = "resumed"
        return AsyncTaskResponse(task=detail)

    async def _load_spawn_integrity_conflict(
        self,
        db_session: AsyncSession,
        *,
        parent_session_id: str,
        request: AsyncTaskSpawnRequest,
        name: str,
    ) -> SessionAsyncTaskRecord | None:
        if request.sdk_idempotency_key is not None:
            existing = await self._load_task_by_sdk_idempotency(
                db_session,
                owner_scope_id=parent_session_id,
                idempotency_key=request.sdk_idempotency_key,
            )
            if existing is not None:
                return existing
        return await self._load_task_by_name(
            db_session,
            parent_session_id=parent_session_id,
            name=name,
        )

    async def list_tasks(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        *,
        parent_session_id: str,
        include_terminal: bool = True,
    ) -> AsyncTaskListResponse:
        await self._load_parent_session(db_session, parent_session_id)
        statement = select(SessionAsyncTaskRecord).where(SessionAsyncTaskRecord.parent_session_id == parent_session_id)
        if not include_terminal:
            statement = statement.where(SessionAsyncTaskRecord.status.in_(list(_ACTIVE_STATUSES)))
        statement = statement.order_by(SessionAsyncTaskRecord.updated_at.desc())
        result = await db_session.execute(statement)
        records = list(result.scalars().all())
        for record in records:
            await self._refresh_task_status(db_session, record)
        await db_session.commit()
        return AsyncTaskListResponse(
            parent_session_id=parent_session_id,
            subagents=[_list_summary_from_record(record) for record in records],
        )

    async def get_task(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        *,
        parent_session_id: str,
        task_id_or_name: str,
    ) -> AsyncTaskResponse:
        record = await self._load_task(db_session, parent_session_id=parent_session_id, task_id_or_name=task_id_or_name)
        await self._refresh_task_status(db_session, record)
        await db_session.commit()
        return AsyncTaskResponse(task=await self._build_detail(db_session, settings, record))

    async def steer_task(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        runtime_state: InMemoryRuntimeState,
        *,
        parent_session_id: str,
        task_id_or_name: str,
        request: AsyncTaskSteerRequest,
    ) -> AsyncTaskResponse:
        record = await self._load_task(db_session, parent_session_id=parent_session_id, task_id_or_name=task_id_or_name)
        await self._refresh_task_status(db_session, record)
        if record.status == AsyncTaskStatus.QUEUED.value:
            raise HTTPException(
                status_code=409,
                detail=f"Async subagent '{record.name}' is queued and is not accepting steering input.",
            )
        if record.status in _TERMINAL_STATUSES:
            raise HTTPException(
                status_code=409,
                detail=f"Async subagent '{record.name}' is {record.status} and is not accepting steering input.",
            )
        run_id = record.task_run_id
        child_session = await db_session.get(SessionRecord, record.task_session_id)
        if isinstance(child_session, SessionRecord) and isinstance(child_session.active_run_id, str):
            run_id = child_session.active_run_id
        if not isinstance(run_id, str):
            raise HTTPException(
                status_code=409,
                detail=f"Async subagent '{record.name}' has no active child run and is not accepting steering input.",
            )

        input_parts = list(request.input_parts)
        if request.prompt is not None and request.prompt.strip():
            input_parts.append(TextPart(type="text", text=request.prompt.strip()))
        if not input_parts:
            raise HTTPException(
                status_code=422, detail="prompt or input_parts is required for async subagent steering."
            )
        receipt = await self._run_controller.steer(
            db_session,
            runtime_state,
            run_id,
            SteerRequest(
                input_parts=input_parts,
                idempotency_key=request.idempotency_key,
            ),
        )
        if receipt.input_id is None or receipt.input_disposition is None:
            raise RuntimeError("Run steering completed without a durable input receipt")
        detail = await self._build_detail(db_session, settings, record)
        detail.delivery = "steered"
        detail.input_id = receipt.input_id
        detail.input_delivery_key = receipt.input_delivery_key
        detail.input_disposition = receipt.input_disposition
        detail.input_sdk_id = receipt.input_sdk_id
        detail.input_enqueue_id = receipt.input_enqueue_id
        detail.instruction = f"Steering input for async subagent '{record.name}' is {receipt.input_disposition}."
        return AsyncTaskResponse(task=detail)

    async def cancel_task(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        runtime_state: InMemoryRuntimeState,
        *,
        parent_session_id: str,
        task_id_or_name: str,
        request: AsyncTaskCancelRequest | None = None,
        submit_run: Callable[[str], object] | None = None,
    ) -> AsyncTaskResponse:
        record = await self._load_task(
            db_session,
            parent_session_id=parent_session_id,
            task_id_or_name=task_id_or_name,
        )
        await self._refresh_task_status(db_session, record)
        run_record = (
            await db_session.get(RunRecord, record.task_run_id) if isinstance(record.task_run_id, str) else None
        )
        if isinstance(run_record, RunRecord) and run_record.status in _ACTIVE_STATUSES:
            await self._run_controller.cancel(
                db_session,
                settings,
                runtime_state,
                run_record.id,
            )
            await db_session.refresh(run_record)
        if isinstance(run_record, RunRecord):
            if request is not None and request.reason:
                run_record.error_message = request.reason[:4000]
                await db_session.commit()
            response = await self.on_run_terminal(
                db_session,
                settings,
                runtime_state,
                run_record=run_record,
                submit_run=submit_run,
            )
            if response is not None:
                response.task.delivery = "cancelled"
                return response

        record.status = AsyncTaskStatus.CANCELLED.value
        cancelled_at = datetime.now(UTC)
        record.completed_at = cancelled_at
        record.updated_at = cancelled_at
        if request is not None and request.reason:
            record.error_message = request.reason[:4000]
        if record.wake_policy == "steer_or_run" and record.delivery_status != "applied":
            record.delivery_id = record.delivery_id or f"async-task:{record.id}:cancelled"
            if record.delivery_status not in {"accepted", "enqueued"}:
                record.delivery_status = "accepted"
        await db_session.commit()
        await self._dispatch_completion_delivery(
            db_session,
            settings,
            runtime_state,
            task_id=record.id,
            submit_run=submit_run,
        )
        await db_session.refresh(record)
        detail = await self._build_detail(db_session, settings, record)
        detail.delivery = "cancelled"
        return AsyncTaskResponse(task=detail)

    async def on_run_terminal(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        runtime_state: InMemoryRuntimeState,
        *,
        run_record: RunRecord,
        submit_run: Callable[[str], object] | None = None,
    ) -> AsyncTaskResponse | None:
        result = await db_session.execute(
            select(SessionAsyncTaskRecord)
            .where(SessionAsyncTaskRecord.task_run_id == run_record.id)
            .with_for_update()
            .limit(1)
        )
        record = result.scalar_one_or_none()
        if not isinstance(record, SessionAsyncTaskRecord):
            return None

        status = run_record.status if run_record.status in _TERMINAL_STATUSES else AsyncTaskStatus.FAILED.value
        now = datetime.now(UTC)
        record.status = status
        record.result_run_id = run_record.id
        record.error_message = run_record.error_message or record.error_message
        record.completed_at = record.completed_at or now
        record.updated_at = now
        if record.wake_policy == "steer_or_run" and record.delivery_status != "applied":
            record.delivery_id = record.delivery_id or f"async-task:{record.id}:{run_record.id}"
            if record.delivery_status not in {"accepted", "enqueued"}:
                record.delivery_status = "accepted"
        await db_session.commit()

        await self._dispatch_completion_delivery(
            db_session,
            settings,
            runtime_state,
            task_id=record.id,
            submit_run=submit_run,
        )
        await db_session.refresh(record)
        logger.info(
            "Async subagent terminal task_id={} name={} status={} child_run_id={} delivery_status={}",
            record.id,
            record.name,
            record.status,
            run_record.id,
            record.delivery_status,
        )
        return AsyncTaskResponse(task=await self._build_detail(db_session, settings, record))

    async def recover_pending_deliveries(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        runtime_state: InMemoryRuntimeState,
        *,
        submit_run: Callable[[str], object] | None = None,
        parent_session_id: str | None = None,
    ) -> list[str]:
        statement = select(SessionAsyncTaskRecord.id).where(
            SessionAsyncTaskRecord.status.in_(list(_TERMINAL_STATUSES)),
            SessionAsyncTaskRecord.wake_policy == "steer_or_run",
            SessionAsyncTaskRecord.delivery_status.in_(["accepted", "enqueued"]),
        )
        if isinstance(parent_session_id, str):
            statement = statement.where(SessionAsyncTaskRecord.parent_session_id == parent_session_id)
        result = await db_session.execute(statement.order_by(SessionAsyncTaskRecord.updated_at.asc()))
        task_ids = list(result.scalars().all())
        for task_id in task_ids:
            await self._dispatch_completion_delivery(
                db_session,
                settings,
                runtime_state,
                task_id=task_id,
                submit_run=submit_run,
            )
        return task_ids

    async def _dispatch_completion_delivery(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        runtime_state: InMemoryRuntimeState,
        *,
        task_id: str,
        submit_run: Callable[[str], object] | None,
    ) -> None:
        record = await db_session.get(SessionAsyncTaskRecord, task_id)
        if (
            not isinstance(record, SessionAsyncTaskRecord)
            or record.status not in _TERMINAL_STATUSES
            or record.wake_policy != "steer_or_run"
            or record.delivery_status == "applied"
        ):
            return
        action = await self._wake_parent(db_session, settings=settings, task_id=task_id)
        await db_session.commit()
        if action is None:
            return

        if action.publish_created:
            await self._run_controller.publish_created(
                settings,
                runtime_state,
                action.run_record,
                dispatch_mode=DispatchMode.ASYNC.value,
            )
        if action.deliver_input:
            await deliver_accepted_run_inputs(
                db_session,
                runtime_state,
                action.run_record.id,
            )
            await self._sync_delivery_status(db_session, task_id)
        if submit_run is not None and action.run_record.status == "queued":
            submit_run(action.run_record.id)

    async def _sync_delivery_status(
        self,
        db_session: AsyncSession,
        task_id: str,
    ) -> None:
        snapshot = await db_session.get(
            SessionAsyncTaskRecord,
            task_id,
            populate_existing=True,
        )
        if (
            not isinstance(snapshot, SessionAsyncTaskRecord)
            or not isinstance(snapshot.delivery_id, str)
            or not isinstance(snapshot.delivery_run_id, str)
        ):
            return
        parent_session = await db_session.get(
            SessionRecord,
            snapshot.parent_session_id,
            populate_existing=True,
            with_for_update=True,
        )
        if not isinstance(parent_session, SessionRecord):
            return
        run_record = await lock_run_record(db_session, snapshot.delivery_run_id)
        if not isinstance(run_record, RunRecord):
            return
        input_result = await db_session.execute(
            select(RunInputInboxRecord)
            .where(
                RunInputInboxRecord.run_id == run_record.id,
                RunInputInboxRecord.delivery_key == snapshot.delivery_id,
            )
            .order_by(RunInputInboxRecord.created_at.desc())
            .with_for_update()
            .limit(1)
        )
        input_record = input_result.scalar_one_or_none()
        record = await db_session.get(
            SessionAsyncTaskRecord,
            task_id,
            populate_existing=True,
            with_for_update=True,
        )
        if (
            not isinstance(record, SessionAsyncTaskRecord)
            or record.delivery_id != snapshot.delivery_id
            or record.delivery_run_id != snapshot.delivery_run_id
        ):
            return
        if isinstance(input_record, RunInputInboxRecord):
            record.delivery_status = input_record.status
            if input_record.status == "rejected":
                record.delivery_run_id = None
                record.delivery_status = "accepted"
            await db_session.commit()

    async def _create_task(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        runtime_state: InMemoryRuntimeState,
        *,
        parent_session: SessionRecord,
        parent_run_id: str | None,
        parent_agent_id: str,
        subagent_name: str,
        subagent_plan: ResolvedSubagentPlan,
        name: str,
        prompt: str,
        context: dict[str, Any],
        wake_policy: str,
        sdk_owner_scope_id: str | None = None,
        sdk_idempotency_key: str | None = None,
        sdk_intent_fingerprint: str | None = None,
        resume_task: SessionAsyncTaskRecord | None = None,
    ) -> AsyncTaskDetail:
        task_id = uuid4().hex
        child_session_id = (
            resume_task.task_session_id if isinstance(resume_task, SessionAsyncTaskRecord) else uuid4().hex
        )
        task_metadata = _task_metadata(
            task_id=task_id,
            parent_session_id=parent_session.id,
            parent_run_id=parent_run_id,
            subagent_name=subagent_name,
            subagent_plan=subagent_plan,
            name=name,
            profile_source=parent_session.profile_name,
            context=context,
        )
        if isinstance(resume_task, SessionAsyncTaskRecord):
            child_session = await db_session.get(
                SessionRecord,
                child_session_id,
                populate_existing=True,
                with_for_update=True,
            )
            if not isinstance(child_session, SessionRecord):
                raise HTTPException(
                    status_code=409,
                    detail="Prior subagent child session was not found.",
                )
            child_session.session_metadata = {
                **dict(child_session.session_metadata or {}),
                "async_task": {"task_id": task_id},
            }
        else:
            child_session = SessionRecord(
                id=child_session_id,
                parent_session_id=parent_session.id,
                profile_name=parent_session.profile_name,
                session_type="async_task",
                session_metadata={"async_task": {"task_id": task_id}},
            )
            db_session.add(child_session)
        record = SessionAsyncTaskRecord(
            id=task_id,
            parent_session_id=parent_session.id,
            parent_run_id=parent_run_id,
            parent_agent_id=parent_agent_id,
            task_session_id=child_session_id,
            task_run_id=None,
            subagent_name=subagent_name,
            name=name,
            status=AsyncTaskStatus.QUEUED.value,
            wake_policy=wake_policy,
            input_parts=[part.model_dump(mode="json") for part in _input_parts(prompt, context)],
            task_metadata=task_metadata,
            subagent_spec_version=str(subagent_plan.spec.schema_version),
            agent_spec_hash=str(task_metadata["agent_spec_hash"]),
            plan_fingerprint=subagent_plan.fingerprint,
            plan_descriptor_ref=subagent_plan.descriptor_id,
            plan_descriptor=subagent_plan.to_descriptor().model_dump(mode="json"),
            sdk_owner_scope_id=sdk_owner_scope_id,
            sdk_idempotency_key=sdk_idempotency_key,
            sdk_intent_fingerprint=sdk_intent_fingerprint,
            sdk_input_state=(SubagentInputState.accepted.value if sdk_owner_scope_id is not None else None),
            delivery_status="pending",
        )
        db_session.add(record)
        await db_session.flush()
        run_record = await self._run_controller.create_record(
            db_session,
            RunCreateRequest(
                session_id=child_session_id,
                restore_from_run_id=(
                    child_session.head_success_run_id if isinstance(resume_task, SessionAsyncTaskRecord) else None
                ),
                profile_name=parent_session.profile_name,
                input_parts=_input_parts(prompt, context),
                trigger_type=TriggerType.ASYNC_TASK,
                metadata={"async_task": {"task_id": task_id}},
                dispatch_mode=DispatchMode.ASYNC,
            ),
            trusted_metadata=True,
            capability_plugins=settings.resolved_capability_plugins,
        )
        record.task_run_id = run_record.id
        if record.sdk_owner_scope_id is not None:
            record.sdk_input_state = SubagentInputState.applied.value
        await db_session.commit()
        await db_session.refresh(record)
        await self._run_controller.publish_created(
            settings,
            runtime_state,
            run_record,
            dispatch_mode=DispatchMode.ASYNC.value,
        )
        return await self._build_detail(db_session, settings, record)

    async def _resume_task(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        runtime_state: InMemoryRuntimeState,
        *,
        task_record: SessionAsyncTaskRecord,
        parent_run_id: str | None,
        prompt: str,
        context: dict[str, Any],
        wake_policy: str,
    ) -> AsyncTaskDetail:
        self._restore_task_plan(
            task_record,
            capability_catalog=settings.resolved_capability_plugins.catalog,
        )
        child_session = await db_session.get(
            SessionRecord,
            task_record.task_session_id,
            populate_existing=True,
            with_for_update=True,
        )
        if not isinstance(child_session, SessionRecord):
            raise HTTPException(status_code=404, detail=f"Child session '{task_record.task_session_id}' was not found.")
        task_metadata = dict(task_record.task_metadata or {})
        task_metadata["parent_run_id"] = parent_run_id
        if context:
            task_metadata["context"] = dict(context)
        run_record = await self._run_controller.create_record(
            db_session,
            RunCreateRequest(
                session_id=child_session.id,
                restore_from_run_id=child_session.head_success_run_id,
                profile_name=child_session.profile_name,
                input_parts=_input_parts(prompt, context),
                trigger_type=TriggerType.ASYNC_TASK,
                metadata={"async_task": {"task_id": task_record.id}},
                dispatch_mode=DispatchMode.ASYNC,
            ),
            trusted_metadata=True,
            capability_plugins=settings.resolved_capability_plugins,
        )
        task_record.parent_run_id = parent_run_id
        task_record.task_run_id = run_record.id
        task_record.status = AsyncTaskStatus.QUEUED.value
        task_record.wake_policy = wake_policy
        task_record.input_parts = [part.model_dump(mode="json") for part in _input_parts(prompt, context)]
        task_record.result_run_id = None
        task_record.error_message = None
        task_record.completed_at = None
        task_record.delivery_status = "pending"
        task_record.delivery_id = None
        task_record.task_metadata = task_metadata
        await db_session.commit()
        await db_session.refresh(task_record)
        await self._run_controller.publish_created(
            settings,
            runtime_state,
            run_record,
            dispatch_mode=DispatchMode.ASYNC.value,
        )
        return await self._build_detail(db_session, settings, task_record)

    async def _wake_parent(
        self,
        db_session: AsyncSession,
        *,
        settings: ClawSettings,
        task_id: str,
    ) -> _WakeAction | None:
        snapshot = await db_session.get(
            SessionAsyncTaskRecord,
            task_id,
            populate_existing=True,
        )
        if not isinstance(snapshot, SessionAsyncTaskRecord):
            return None
        if not isinstance(snapshot.delivery_id, str):
            raise TypeError("Async task wake has no stable delivery identity")
        delivery_id = snapshot.delivery_id

        parent_session = await db_session.get(
            SessionRecord,
            snapshot.parent_session_id,
            populate_existing=True,
            with_for_update=True,
        )
        if not isinstance(parent_session, SessionRecord):
            record = await db_session.get(
                SessionAsyncTaskRecord,
                task_id,
                populate_existing=True,
                with_for_update=True,
            )
            if isinstance(record, SessionAsyncTaskRecord) and record.delivery_status != "applied":
                record.delivery_status = "rejected"
            return None

        snapshot = await db_session.get(
            SessionAsyncTaskRecord,
            task_id,
            populate_existing=True,
        )
        if (
            not isinstance(snapshot, SessionAsyncTaskRecord)
            or snapshot.status not in _TERMINAL_STATUSES
            or snapshot.wake_policy != "steer_or_run"
            or snapshot.delivery_status == "applied"
            or snapshot.delivery_id != delivery_id
        ):
            return None

        wake_part = CommandPart(
            type="command",
            name="async_task_completed",
            params={
                "task_id": snapshot.id,
                "task_session_id": snapshot.task_session_id,
                "task_run_id": snapshot.task_run_id,
                "subagent_name": snapshot.subagent_name,
                "name": snapshot.name,
                "status": snapshot.status,
                "result_available": snapshot.result_run_id is not None,
            },
        )
        resolution = await self._resolve_existing_wake(
            db_session,
            snapshot=snapshot,
            delivery_id=delivery_id,
        )
        if resolution is None:
            resolution = await self._schedule_wake(
                db_session,
                parent_session=parent_session,
                delivery_id=delivery_id,
                wake_part=wake_part,
                settings=settings,
            )
        stored = await self._store_wake_resolution(
            db_session,
            task_id=task_id,
            delivery_id=delivery_id,
            resolution=resolution,
        )
        if not stored:
            return None
        action = resolution.action
        if action is not None:
            logger.info(
                "Async subagent wake scheduled parent_session_id={} parent_run_id={} task_id={} mode={}",
                parent_session.id,
                action.run_record.id,
                snapshot.id,
                "input" if action.deliver_input else "continuation",
            )
        return action

    async def _resolve_existing_wake(
        self,
        db_session: AsyncSession,
        *,
        snapshot: SessionAsyncTaskRecord,
        delivery_id: str,
    ) -> _WakeResolution | None:
        target_run = (
            await lock_run_record(db_session, snapshot.delivery_run_id)
            if isinstance(snapshot.delivery_run_id, str)
            else None
        )
        if isinstance(target_run, RunRecord):
            if target_run.source_delivery_id == delivery_id:
                return await self._resolve_source_wake(db_session, target_run)
            resolution = await self._resolve_inbox_wake(
                db_session,
                target_run=target_run,
                delivery_id=delivery_id,
            )
            if resolution is not None:
                return resolution

        source_result = await db_session.execute(
            select(RunRecord).where(RunRecord.source_delivery_id == delivery_id).limit(1)
        )
        source_candidate = source_result.scalar_one_or_none()
        if not isinstance(source_candidate, RunRecord):
            return None
        source_run = await lock_run_record(db_session, source_candidate.id)
        if not isinstance(source_run, RunRecord):
            return None
        return await self._resolve_source_wake(db_session, source_run)

    async def _resolve_inbox_wake(
        self,
        db_session: AsyncSession,
        *,
        target_run: RunRecord,
        delivery_id: str,
    ) -> _WakeResolution | None:
        input_result = await db_session.execute(
            select(RunInputInboxRecord)
            .where(
                RunInputInboxRecord.run_id == target_run.id,
                RunInputInboxRecord.delivery_key == delivery_id,
            )
            .with_for_update()
            .limit(1)
        )
        input_record = input_result.scalar_one_or_none()
        if not isinstance(input_record, RunInputInboxRecord):
            return None
        if input_record.status == "applied":
            return _WakeResolution(target_run.id, "applied")
        if input_record.status not in {"accepted", "enqueued"}:
            return None
        if target_run.status in _ACTIVE_STATUSES:
            return _WakeResolution(
                target_run.id,
                input_record.status,
                _WakeAction(run_record=target_run, deliver_input=True),
            )
        input_record.status = "rejected"
        input_record.error_message = "Parent run became terminal before async completion delivery."
        input_record.updated_at = datetime.now(UTC)
        return None

    async def _resolve_source_wake(
        self,
        db_session: AsyncSession,
        source_run: RunRecord,
    ) -> _WakeResolution | None:
        if source_run.source_delivery_applied_at is not None:
            return _WakeResolution(source_run.id, "applied")
        if source_run.status == "queued":
            return _WakeResolution(
                source_run.id,
                "enqueued",
                _WakeAction(run_record=source_run),
            )
        if source_run.status == "running":
            return _WakeResolution(source_run.id, "enqueued")
        # A terminal continuation without the explicit model-visible marker did
        # not apply its command. Release the key so recovery can retarget it.
        source_run.source_delivery_id = None
        await db_session.flush()
        return None

    async def _schedule_wake(
        self,
        db_session: AsyncSession,
        *,
        parent_session: SessionRecord,
        delivery_id: str,
        wake_part: CommandPart,
        settings: ClawSettings,
    ) -> _WakeResolution:
        if isinstance(parent_session.active_run_id, str):
            parent_run = await lock_run_record(db_session, parent_session.active_run_id)
            if isinstance(parent_run, RunRecord) and parent_run.status in _ACTIVE_STATUSES:
                input_record = await accept_run_input(
                    db_session,
                    parent_run,
                    [wake_part.model_dump(mode="json")],
                    delivery_key=delivery_id,
                    origin="feature",
                )
                return _WakeResolution(
                    parent_run.id,
                    input_record.status,
                    _WakeAction(run_record=parent_run, deliver_input=True),
                )
        return await self._create_wake_run(
            db_session,
            parent_session=parent_session,
            delivery_id=delivery_id,
            wake_part=wake_part,
            settings=settings,
        )

    async def _create_wake_run(
        self,
        db_session: AsyncSession,
        *,
        parent_session: SessionRecord,
        delivery_id: str,
        wake_part: CommandPart,
        settings: ClawSettings,
    ) -> _WakeResolution:
        try:
            async with db_session.begin_nested():
                run_record = await self._run_controller.create_record(
                    db_session,
                    RunCreateRequest(
                        session_id=parent_session.id,
                        restore_from_run_id=parent_session.head_success_run_id,
                        profile_name=parent_session.profile_name,
                        input_parts=[wake_part],
                        trigger_type=TriggerType.ASYNC_TASK,
                        metadata={"async_task_wake": wake_part.params or {}},
                        dispatch_mode=DispatchMode.ASYNC,
                    ),
                    trusted_metadata=True,
                    source_delivery_id=delivery_id,
                    capability_plugins=settings.resolved_capability_plugins,
                )
        except IntegrityError:
            existing_result = await db_session.execute(
                select(RunRecord).where(RunRecord.source_delivery_id == delivery_id).limit(1)
            )
            existing_run = existing_result.scalar_one_or_none()
            if not isinstance(existing_run, RunRecord):
                raise
            locked_existing = await lock_run_record(db_session, existing_run.id)
            if not isinstance(locked_existing, RunRecord):
                raise TypeError("Async completion continuation disappeared") from None
            run_record = locked_existing
            publish_created = False
        else:
            publish_created = True
        if run_record.source_delivery_applied_at is not None:
            return _WakeResolution(run_record.id, "applied")
        if run_record.status == "running":
            return _WakeResolution(run_record.id, "enqueued")
        if run_record.status != "queued":
            run_record.source_delivery_id = None
            await db_session.flush()
            return await self._create_wake_run(
                db_session,
                parent_session=parent_session,
                delivery_id=delivery_id,
                wake_part=wake_part,
                settings=settings,
            )
        return _WakeResolution(
            run_record.id,
            "enqueued",
            _WakeAction(
                run_record=run_record,
                publish_created=publish_created,
            ),
        )

    async def _store_wake_resolution(
        self,
        db_session: AsyncSession,
        *,
        task_id: str,
        delivery_id: str,
        resolution: _WakeResolution,
    ) -> bool:
        record = await db_session.get(
            SessionAsyncTaskRecord,
            task_id,
            populate_existing=True,
            with_for_update=True,
        )
        if (
            not isinstance(record, SessionAsyncTaskRecord)
            or record.delivery_id != delivery_id
            or record.status not in _TERMINAL_STATUSES
            or record.wake_policy != "steer_or_run"
            or record.delivery_status == "applied"
        ):
            return False
        record.delivery_run_id = resolution.delivery_run_id
        record.delivery_status = resolution.delivery_status
        return True

    async def _refresh_task_status(self, db_session: AsyncSession, record: SessionAsyncTaskRecord) -> None:
        if not isinstance(record.task_run_id, str):
            return
        run_record = await db_session.get(RunRecord, record.task_run_id)
        if not isinstance(run_record, RunRecord):
            return
        if run_record.status == record.status:
            return
        if run_record.status in {"queued", "running", "completed", "failed", "cancelled"}:
            record.status = run_record.status
            record.updated_at = datetime.now(UTC)
            if run_record.status in _TERMINAL_STATUSES:
                record.completed_at = run_record.finished_at or record.updated_at
                record.result_run_id = run_record.id
                record.error_message = run_record.error_message

    async def _build_detail(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        record: SessionAsyncTaskRecord,
    ) -> AsyncTaskDetail:
        child_session_summary: SessionSummary | None = None
        latest_run: RunRecord | None = None
        child_session = await db_session.get(SessionRecord, record.task_session_id)
        if isinstance(child_session, SessionRecord):
            child_session_summary = await _session_summary(db_session, child_session)
            latest_run = await _latest_run(db_session, child_session.id)
        run_summary = (
            run_summary_from_record(latest_run, include_input_parts=True) if isinstance(latest_run, RunRecord) else None
        )
        run_payload = run_summary.model_dump(mode="json") if run_summary is not None else None
        state_payload = (
            read_run_state_blob_if_exists(settings, record.result_run_id or "") if record.result_run_id else None
        )
        message_payload = (
            read_run_message_blob_if_exists(settings, record.result_run_id or "") if record.result_run_id else None
        )
        summary = _detail_summary_from_record(record)
        payload = summary.model_dump(
            exclude={
                "child_session",
                "latest_run",
                "output_text",
                "output_json",
                "trace_ref",
            }
        )
        return AsyncTaskDetail(
            **payload,
            child_session=child_session_summary.model_dump(mode="json") if child_session_summary is not None else None,
            latest_run=run_payload,
            output_text=latest_run.output_text if isinstance(latest_run, RunRecord) else None,
            output_json=latest_run.output_json if isinstance(latest_run, RunRecord) else None,
            trace_ref={
                "run_id": record.result_run_id,
                "trace_path": f"/api/v1/runs/{record.result_run_id}/trace",
                "has_state": state_payload is not None,
                "has_message": message_payload is not None,
            }
            if isinstance(record.result_run_id, str)
            else None,
        )

    async def _load_parent_session(self, db_session: AsyncSession, parent_session_id: str) -> SessionRecord:
        parent_session = await db_session.get(SessionRecord, parent_session_id)
        if not isinstance(parent_session, SessionRecord):
            raise HTTPException(status_code=404, detail=f"Parent session '{parent_session_id}' was not found.")
        return parent_session

    async def _load_task(
        self,
        db_session: AsyncSession,
        *,
        parent_session_id: str,
        task_id_or_name: str,
    ) -> SessionAsyncTaskRecord:
        value = task_id_or_name.strip()
        statement = select(SessionAsyncTaskRecord).where(SessionAsyncTaskRecord.parent_session_id == parent_session_id)
        statement = statement.where((SessionAsyncTaskRecord.id == value) | (SessionAsyncTaskRecord.name == value))
        result = await db_session.execute(statement.limit(1))
        record = result.scalar_one_or_none()
        if not isinstance(record, SessionAsyncTaskRecord):
            raise HTTPException(status_code=404, detail=f"Async subagent '{task_id_or_name}' was not found.")
        return record

    async def _load_task_by_name(
        self,
        db_session: AsyncSession,
        *,
        parent_session_id: str,
        name: str,
    ) -> SessionAsyncTaskRecord | None:
        result = await db_session.execute(
            select(SessionAsyncTaskRecord)
            .where(SessionAsyncTaskRecord.parent_session_id == parent_session_id, SessionAsyncTaskRecord.name == name)
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def _load_task_by_sdk_idempotency(
        self,
        db_session: AsyncSession,
        *,
        owner_scope_id: str,
        idempotency_key: str | None,
    ) -> SessionAsyncTaskRecord | None:
        if idempotency_key is None:
            return None
        result = await db_session.execute(
            select(SessionAsyncTaskRecord)
            .where(
                SessionAsyncTaskRecord.sdk_owner_scope_id == owner_scope_id,
                SessionAsyncTaskRecord.sdk_idempotency_key == idempotency_key,
            )
            .limit(1)
        )
        task = result.scalar_one_or_none()
        return task if isinstance(task, SessionAsyncTaskRecord) else None

    async def _sdk_replay_response(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        task: SessionAsyncTaskRecord,
        request: AsyncTaskSpawnRequest,
    ) -> AsyncTaskResponse:
        if (
            task.sdk_owner_scope_id != request.sdk_owner_scope_id
            or task.sdk_idempotency_key != request.sdk_idempotency_key
            or task.sdk_intent_fingerprint != request.sdk_intent_fingerprint
        ):
            raise HTTPException(
                status_code=409,
                detail="SDK subagent idempotency key was reused with different intent.",
            )
        if task.subagent_name != request.subagent_name:
            raise HTTPException(
                status_code=409,
                detail="SDK subagent idempotency key belongs to a different route.",
            )
        await self._refresh_task_status(db_session, task)
        detail = await self._build_detail(db_session, settings, task)
        detail.delivery = "existing_active" if task.status in _ACTIVE_STATUSES else "recorded"
        detail.instruction = (
            f"Subagent execution '{task.name}' is {task.status}; "
            "the SDK idempotency key resolves to this committed execution."
        )
        return AsyncTaskResponse(task=detail)

    async def _resolve_name(
        self,
        db_session: AsyncSession,
        *,
        parent_session_id: str,
        subagent_name: str,
        requested_name: str | None,
        preserve_requested_name: bool,
    ) -> str:
        if preserve_requested_name:
            if not isinstance(requested_name, str) or not requested_name:
                raise HTTPException(status_code=422, detail="SDK subagent execution handle is required.")
            if len(requested_name) > 255:
                raise HTTPException(
                    status_code=422,
                    detail="SDK subagent execution handle cannot exceed 255 characters.",
                )
            return requested_name
        if isinstance(requested_name, str) and requested_name.strip():
            return _normalize_name(requested_name)
        base = _normalize_name(subagent_name)
        existing_result = await db_session.execute(
            select(SessionAsyncTaskRecord.name).where(SessionAsyncTaskRecord.parent_session_id == parent_session_id)
        )
        existing = {name for name in existing_result.scalars().all() if isinstance(name, str)}
        if base not in existing:
            return base
        index = 2
        while f"{base}-{index}" in existing:
            index += 1
        return f"{base}-{index}"

    async def _resolve_spawn_plan(
        self,
        db_session: AsyncSession,
        *,
        parent_session: SessionRecord,
        request: AsyncTaskSpawnRequest,
        profile_resolver: ProfileResolverProtocol | None,
        settings: ClawSettings,
    ) -> tuple[ResolvedSubagentPlan, SessionAsyncTaskRecord | None]:
        resumed_from = request.context.get("_sdk_resumed_from")
        resume_task = (
            await self._load_task_by_name(
                db_session,
                parent_session_id=parent_session.id,
                name=resumed_from,
            )
            if isinstance(resumed_from, str) and resumed_from.strip()
            else None
        )
        if isinstance(resumed_from, str) and resume_task is None:
            raise HTTPException(
                status_code=409,
                detail=f"Prior subagent execution {resumed_from!r} was not found.",
            )
        if isinstance(resume_task, SessionAsyncTaskRecord):
            await self._refresh_task_status(db_session, resume_task)
            if resume_task.status not in _TERMINAL_STATUSES:
                raise HTTPException(
                    status_code=409,
                    detail=f"Prior subagent execution {resumed_from!r} is not terminal.",
                )
            plan = self._restore_task_plan(
                resume_task,
                capability_catalog=settings.resolved_capability_plugins.catalog,
            )
            if plan.spec.route != request.subagent_name:
                raise HTTPException(
                    status_code=409,
                    detail="A resumed execution must use the prior immutable subagent route.",
                )
        else:
            plan = await self._validate_subagent(
                profile_resolver,
                profile_name=parent_session.profile_name,
                subagent_name=request.subagent_name,
                capability_catalog=settings.resolved_capability_plugins.catalog,
            )
        advertised_fingerprint = request.context.get("plan_fingerprint")
        if isinstance(advertised_fingerprint, str) and advertised_fingerprint != plan.fingerprint:
            raise HTTPException(
                status_code=409,
                detail=(
                    "The parent runtime subagent plan is stale and does not match "
                    "the immutable plan selected for this execution."
                ),
            )
        return plan, resume_task

    async def _validate_subagent(
        self,
        profile_resolver: ProfileResolverProtocol | None,
        *,
        profile_name: str | None,
        subagent_name: str,
        capability_catalog: CapabilityCatalog,
    ) -> ResolvedSubagentPlan:
        if profile_resolver is None:
            raise RuntimeError("Profile resolver is required for durable subagent resolution")
        profile = await profile_resolver.resolve(profile_name)
        for spec in profile.subagent_specs:
            if isinstance(spec, SubagentSpec) and spec.route == subagent_name:
                return resolve_claw_subagent_plan(
                    spec,
                    capability_catalog=capability_catalog,
                )
        raise HTTPException(status_code=404, detail=f"Subagent '{subagent_name}' is not configured for this profile.")

    def _restore_task_plan(
        self,
        task_record: SessionAsyncTaskRecord,
        *,
        capability_catalog: CapabilityCatalog,
    ) -> ResolvedSubagentPlan:
        if not isinstance(task_record.plan_descriptor, dict):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Async subagent '{task_record.name}' has no immutable 2.0 plan descriptor and cannot be resumed."
                ),
            )
        descriptor = SubagentPlanDescriptor.model_validate(task_record.plan_descriptor)
        if descriptor.descriptor_id != task_record.plan_descriptor_ref:
            raise HTTPException(status_code=409, detail="Async subagent descriptor identity is inconsistent.")
        if descriptor.fingerprint != task_record.plan_fingerprint:
            raise HTTPException(status_code=409, detail="Async subagent descriptor fingerprint is inconsistent.")
        if descriptor.spec.route != task_record.subagent_name:
            raise HTTPException(status_code=409, detail="Async subagent descriptor route is inconsistent.")
        try:
            return restore_claw_subagent_plan(
                descriptor,
                capability_catalog=capability_catalog,
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=409,
                detail=f"Async subagent descriptor is not executable: {exc}",
            ) from exc


def _input_parts(prompt: str, context: dict[str, Any]) -> list[InputPart]:
    parts: list[InputPart] = []
    public_context = {
        key: value
        for key, value in context.items()
        if not key.startswith("_") and key not in {"mode", "plan_fingerprint"}
    }
    if public_context:
        parts.append(
            CommandPart(
                type="command",
                name="async_subagent_context",
                params=public_context,
            )
        )
    parts.append(TextPart(type="text", text=prompt))
    return parts


def _task_metadata(
    *,
    task_id: str,
    parent_session_id: str,
    parent_run_id: str | None,
    subagent_name: str,
    subagent_plan: ResolvedSubagentPlan,
    name: str,
    profile_source: str | None,
    context: dict[str, Any],
) -> dict[str, Any]:
    serialized_agent = subagent_plan.normalized_agent_spec.model_dump(
        mode="json",
        by_alias=True,
        exclude_defaults=True,
    )
    agent_spec_hash = _stable_hash(serialized_agent)
    payload: dict[str, Any] = {
        "task_id": task_id,
        "kind": "subagent",
        "parent_session_id": parent_session_id,
        "parent_run_id": parent_run_id,
        "subagent_name": subagent_name,
        "name": name,
        "profile_source": profile_source,
        "subagent_spec_version": str(subagent_plan.spec.schema_version),
        "agent_spec_hash": agent_spec_hash,
        "plan_fingerprint": subagent_plan.fingerprint,
        "plan_descriptor_ref": subagent_plan.descriptor_id,
    }
    if context:
        payload["context"] = dict(context)
    return payload


def _list_summary_from_record(record: SessionAsyncTaskRecord) -> AsyncTaskSummary:
    return AsyncTaskSummary(
        name=record.name,
        task_session_id=record.task_session_id,
        status=record.status,
    )


def _detail_summary_from_record(record: SessionAsyncTaskRecord) -> AsyncTaskDetail:
    return AsyncTaskDetail(
        name=record.name,
        task_session_id=record.task_session_id,
        status=record.status,
        task_id=record.id,
        parent_session_id=record.parent_session_id,
        parent_run_id=record.parent_run_id,
        parent_agent_id=record.parent_agent_id,
        task_run_id=record.task_run_id,
        subagent_name=record.subagent_name,
        wake_policy=record.wake_policy,
        result_run_id=record.result_run_id,
        error_message=record.error_message,
        metadata=dict(record.task_metadata or {}),
        created_at=record.created_at,
        updated_at=record.updated_at,
        completed_at=record.completed_at,
        sdk_input_state=(
            SubagentInputState(record.sdk_input_state).value if isinstance(record.sdk_input_state, str) else None
        ),
    )


async def _session_summary(db_session: AsyncSession, session_record: SessionRecord) -> SessionSummary:
    run_count_result = await db_session.execute(select(func.count()).where(RunRecord.session_id == session_record.id))
    run_count = run_count_result.scalar_one()
    latest_run = await _latest_run(db_session, session_record.id)
    latest_summary = run_summary_from_record(latest_run) if isinstance(latest_run, RunRecord) else None
    active_interactions = active_interactions_from_run_record(latest_run) if isinstance(latest_run, RunRecord) else None
    return session_summary_from_record(
        session_record,
        run_count=run_count,
        latest_run=latest_summary,
        memory_state=None,
        active_interactions=active_interactions,
    )


async def _latest_run(db_session: AsyncSession, session_id: str) -> RunRecord | None:
    result = await db_session.execute(
        select(RunRecord)
        .where(RunRecord.session_id == session_id)
        .order_by(RunRecord.sequence_no.desc(), RunRecord.id.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


def _stable_hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _normalize_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-._").lower()
    if normalized == "":
        raise HTTPException(status_code=422, detail="Async subagent name must contain letters or numbers.")
    return normalized[:255]
