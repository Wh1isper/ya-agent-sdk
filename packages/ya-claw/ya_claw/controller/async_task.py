from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

from fastapi import HTTPException
from loguru import logger
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from ya_agent_sdk.subagents import get_builtin_subagent_configs

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
from ya_claw.controller.store import read_run_message_blob_if_exists, read_run_state_blob_if_exists
from ya_claw.orm.tables import RunRecord, SessionAsyncTaskRecord, SessionRecord
from ya_claw.runtime_state import InMemoryRuntimeState

_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
_ACTIVE_STATUSES = {"queued", "running"}
_RECENT_RESULT_LIMIT = 10


@runtime_checkable
class ProfileResolverProtocol(Protocol):
    async def resolve(self, profile_name: str | None) -> Any: ...


class AsyncTaskController:
    def __init__(self) -> None:
        self._run_controller = RunController()

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
        parent_session = await self._load_parent_session(db_session, parent_session_id)
        await self._validate_subagent(
            profile_resolver,
            profile_name=parent_session.profile_name,
            subagent_name=request.subagent_name,
        )
        name = await self._resolve_name(
            db_session,
            parent_session_id=parent_session_id,
            subagent_name=request.subagent_name,
            requested_name=request.name,
        )
        existing = await self._load_task_by_name(db_session, parent_session_id=parent_session_id, name=name)
        if existing is None:
            detail = await self._create_task(
                db_session,
                settings,
                runtime_state,
                parent_session=parent_session,
                parent_run_id=parent_run_id,
                parent_agent_id=parent_agent_id,
                subagent_name=request.subagent_name,
                name=name,
                prompt=request.prompt,
                context=request.context,
                wake_policy=str(request.wake_policy),
            )
            detail.delivery = "submitted"
            return AsyncTaskResponse(task=detail)

        await self._refresh_task_status(db_session, existing)
        if existing.status in _ACTIVE_STATUSES:
            detail = await self._build_detail(db_session, settings, existing)
            detail.delivery = "existing_active"
            detail.instruction = (
                f"Async subagent '{existing.name}' is {existing.status}. "
                "Use steer_async_subagent for additional input while it is running."
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
            detail = await self._build_detail(db_session, settings, record)
            detail.delivery = "idle"
            detail.instruction = (
                f"Async subagent '{record.name}' is queued. Steering can be sent after it starts running."
            )
            return AsyncTaskResponse(task=detail)
        if record.status in _TERMINAL_STATUSES:
            detail = await self._build_detail(db_session, settings, record)
            detail.delivery = "idle"
            detail.instruction = (
                f"Async subagent '{record.name}' is {record.status}. "
                f"Use spawn_delegate with name='{record.name}' to continue it."
            )
            return AsyncTaskResponse(task=detail)
        run_id = record.task_run_id
        child_session = await db_session.get(SessionRecord, record.task_session_id)
        if isinstance(child_session, SessionRecord) and isinstance(child_session.active_run_id, str):
            run_id = child_session.active_run_id
        if not isinstance(run_id, str):
            detail = await self._build_detail(db_session, settings, record)
            detail.delivery = "idle"
            detail.instruction = f"Async subagent '{record.name}' has no active child run."
            return AsyncTaskResponse(task=detail)

        input_parts = list(request.input_parts)
        if request.prompt is not None and request.prompt.strip():
            input_parts.append(TextPart(type="text", text=request.prompt.strip()))
        if not input_parts:
            raise HTTPException(
                status_code=422, detail="prompt or input_parts is required for async subagent steering."
            )
        await self._run_controller.steer(db_session, runtime_state, run_id, SteerRequest(input_parts=input_parts))
        detail = await self._build_detail(db_session, settings, record)
        detail.delivery = "steered"
        detail.instruction = f"Steering input sent to async subagent '{record.name}'."
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
    ) -> AsyncTaskResponse:
        record = await self._load_task(db_session, parent_session_id=parent_session_id, task_id_or_name=task_id_or_name)
        await self._refresh_task_status(db_session, record)
        if record.status in _ACTIVE_STATUSES and isinstance(record.task_run_id, str):
            run_record = await db_session.get(RunRecord, record.task_run_id)
            if isinstance(run_record, RunRecord) and run_record.status in _ACTIVE_STATUSES:
                await self._run_controller.cancel(db_session, settings, runtime_state, run_record.id)
        record.status = AsyncTaskStatus.CANCELLED.value
        cancelled_at = datetime.now(UTC)
        record.completed_at = cancelled_at
        record.updated_at = cancelled_at
        if request is not None and request.reason:
            record.error_message = request.reason[:4000]
        await db_session.commit()
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
        submit_run: Callable[[str], bool] | None = None,
    ) -> AsyncTaskResponse | None:
        result = await db_session.execute(
            select(SessionAsyncTaskRecord).where(SessionAsyncTaskRecord.task_run_id == run_record.id).limit(1)
        )
        record = result.scalar_one_or_none()
        if not isinstance(record, SessionAsyncTaskRecord):
            return None

        status = run_record.status if run_record.status in _TERMINAL_STATUSES else AsyncTaskStatus.FAILED.value
        now = datetime.now(UTC)
        record.status = status
        record.result_run_id = run_record.id
        record.result_summary = run_record.output_summary
        record.error_message = run_record.error_message
        record.completed_at = now
        record.updated_at = now
        await db_session.flush()

        if record.wake_policy == "steer_or_run":
            await self._wake_parent(db_session, settings, runtime_state, record=record, submit_run=submit_run)
        await db_session.commit()
        await db_session.refresh(record)
        logger.info(
            "Async subagent terminal task_id={} name={} status={} child_run_id={}",
            record.id,
            record.name,
            record.status,
            run_record.id,
        )
        return AsyncTaskResponse(task=await self._build_detail(db_session, settings, record))

    async def build_injected_context(
        self,
        db_session: AsyncSession,
        *,
        parent_session_id: str,
        limit: int = _RECENT_RESULT_LIMIT,
    ) -> str | None:
        statement = (
            select(SessionAsyncTaskRecord)
            .where(SessionAsyncTaskRecord.parent_session_id == parent_session_id)
            .order_by(SessionAsyncTaskRecord.updated_at.desc())
            .limit(max(limit, 1))
        )
        result = await db_session.execute(statement)
        records = list(result.scalars().all())
        if not records:
            return None
        lines = [f'<async-subagents session-id="{parent_session_id}">']
        for record in records:
            attrs = {
                "id": record.id,
                "name": record.name,
                "subagent-name": record.subagent_name,
                "status": record.status,
                "session-id": record.task_session_id,
            }
            if isinstance(record.task_run_id, str):
                attrs["run-id"] = record.task_run_id
            if record.status in _TERMINAL_STATUSES:
                attrs["result"] = "available" if record.result_run_id else "unavailable"
            attr_text = " ".join(f'{key}="{_xml_escape(value)}"' for key, value in attrs.items())
            lines.append(f"  <subagent {attr_text} />")
        lines.append("</async-subagents>")
        return "\n".join(lines)

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
        name: str,
        prompt: str,
        context: dict[str, Any],
        wake_policy: str,
    ) -> AsyncTaskDetail:
        task_id = uuid4().hex
        child_session_id = uuid4().hex
        task_metadata = _task_metadata(
            task_id=task_id,
            parent_session_id=parent_session.id,
            parent_run_id=parent_run_id,
            subagent_name=subagent_name,
            name=name,
            profile_source=parent_session.profile_name,
            context=context,
        )
        child_session = SessionRecord(
            id=child_session_id,
            parent_session_id=parent_session.id,
            profile_name=parent_session.profile_name,
            session_type="async_task",
            session_metadata={"async_task": task_metadata},
        )
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
        )
        db_session.add(child_session)
        db_session.add(record)
        await db_session.commit()
        await db_session.refresh(record)

        run = await self._run_controller.create(
            db_session,
            settings,
            runtime_state,
            RunCreateRequest(
                session_id=child_session_id,
                profile_name=parent_session.profile_name,
                input_parts=_input_parts(prompt, context),
                trigger_type=TriggerType.ASYNC_TASK,
                metadata={"async_task": task_metadata},
                dispatch_mode=DispatchMode.ASYNC,
            ),
        )
        refreshed = await db_session.get(SessionAsyncTaskRecord, task_id)
        if not isinstance(refreshed, SessionAsyncTaskRecord):
            raise TypeError(f"Async task '{task_id}' disappeared after child run creation.")
        refreshed.task_run_id = run.id
        refreshed.status = AsyncTaskStatus.QUEUED.value
        await db_session.commit()
        await db_session.refresh(refreshed)
        return await self._build_detail(db_session, settings, refreshed)

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
        child_session = await db_session.get(SessionRecord, task_record.task_session_id)
        if not isinstance(child_session, SessionRecord):
            raise HTTPException(status_code=404, detail=f"Child session '{task_record.task_session_id}' was not found.")
        task_metadata = dict(task_record.task_metadata or {})
        task_metadata["parent_run_id"] = parent_run_id
        if context:
            task_metadata["context"] = dict(context)
        run = await self._run_controller.create(
            db_session,
            settings,
            runtime_state,
            RunCreateRequest(
                session_id=child_session.id,
                restore_from_run_id=child_session.head_run_id or child_session.head_success_run_id,
                profile_name=child_session.profile_name,
                input_parts=_input_parts(prompt, context),
                trigger_type=TriggerType.ASYNC_TASK,
                metadata={"async_task": task_metadata},
                dispatch_mode=DispatchMode.ASYNC,
            ),
        )
        refreshed = await db_session.get(SessionAsyncTaskRecord, task_record.id)
        if not isinstance(refreshed, SessionAsyncTaskRecord):
            raise TypeError(f"Async task '{task_record.id}' disappeared after child run creation.")
        refreshed.parent_run_id = parent_run_id
        refreshed.task_run_id = run.id
        refreshed.status = AsyncTaskStatus.QUEUED.value
        refreshed.wake_policy = wake_policy
        refreshed.input_parts = [part.model_dump(mode="json") for part in _input_parts(prompt, context)]
        refreshed.result_run_id = None
        refreshed.result_summary = None
        refreshed.error_message = None
        refreshed.completed_at = None
        refreshed.task_metadata = task_metadata
        await db_session.commit()
        await db_session.refresh(refreshed)
        return await self._build_detail(db_session, settings, refreshed)

    async def _wake_parent(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        runtime_state: InMemoryRuntimeState,
        *,
        record: SessionAsyncTaskRecord,
        submit_run: Callable[[str], bool] | None,
    ) -> None:
        parent_session = await db_session.get(SessionRecord, record.parent_session_id)
        if not isinstance(parent_session, SessionRecord):
            return
        wake_part = CommandPart(
            type="command",
            name="async_task_completed",
            params={
                "task_id": record.id,
                "task_session_id": record.task_session_id,
                "task_run_id": record.task_run_id,
                "subagent_name": record.subagent_name,
                "name": record.name,
                "status": record.status,
                "output_summary": record.result_summary,
                "result_available": record.result_run_id is not None,
            },
        )
        if isinstance(parent_session.active_run_id, str):
            try:
                await runtime_state.record_steering(parent_session.active_run_id, [wake_part.model_dump(mode="json")])
                logger.info(
                    "Async subagent wake steered parent_session_id={} parent_run_id={} task_id={}",
                    parent_session.id,
                    parent_session.active_run_id,
                    record.id,
                )
                return
            except KeyError:
                logger.debug(
                    "Async subagent parent active run missing runtime handle parent_session_id={} parent_run_id={}",
                    parent_session.id,
                    parent_session.active_run_id,
                )
        run = await self._run_controller.create(
            db_session,
            settings,
            runtime_state,
            RunCreateRequest(
                session_id=parent_session.id,
                restore_from_run_id=parent_session.head_run_id or parent_session.head_success_run_id,
                profile_name=parent_session.profile_name,
                input_parts=[wake_part],
                trigger_type=TriggerType.ASYNC_TASK,
                metadata={"async_task_wake": wake_part.params or {}},
                dispatch_mode=DispatchMode.ASYNC,
            ),
        )
        await db_session.flush()
        if submit_run is not None:
            submit_run(run.id)
        logger.info(
            "Async subagent wake submitted parent_session_id={} parent_run_id={} task_id={}",
            parent_session.id,
            run.id,
            record.id,
        )

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
                record.result_summary = run_record.output_summary
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
            exclude={"child_session", "latest_run", "output_text", "output_summary", "trace_ref"}
        )
        return AsyncTaskDetail(
            **payload,
            child_session=child_session_summary.model_dump(mode="json") if child_session_summary is not None else None,
            latest_run=run_payload,
            output_text=latest_run.output_text if isinstance(latest_run, RunRecord) else None,
            output_summary=latest_run.output_summary if isinstance(latest_run, RunRecord) else None,
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

    async def _resolve_name(
        self,
        db_session: AsyncSession,
        *,
        parent_session_id: str,
        subagent_name: str,
        requested_name: str | None,
    ) -> str:
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

    async def _validate_subagent(
        self,
        profile_resolver: ProfileResolverProtocol | None,
        *,
        profile_name: str | None,
        subagent_name: str,
    ) -> None:
        if profile_resolver is None:
            return
        profile = await profile_resolver.resolve(profile_name)
        configs = getattr(profile, "subagent_configs", [])
        if any(getattr(config, "name", None) == subagent_name for config in configs):
            return
        if getattr(profile, "include_builtin_subagents", False) and subagent_name in get_builtin_subagent_configs():
            return
        raise HTTPException(status_code=404, detail=f"Subagent '{subagent_name}' is not configured for this profile.")


def _input_parts(prompt: str, context: dict[str, Any]) -> list[InputPart]:
    parts: list[InputPart] = []
    if context:
        parts.append(CommandPart(type="command", name="async_subagent_context", params=dict(context)))
    parts.append(TextPart(type="text", text=prompt))
    return parts


def _task_metadata(
    *,
    task_id: str,
    parent_session_id: str,
    parent_run_id: str | None,
    subagent_name: str,
    name: str,
    profile_source: str | None,
    context: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "task_id": task_id,
        "kind": "subagent",
        "parent_session_id": parent_session_id,
        "parent_run_id": parent_run_id,
        "subagent_name": subagent_name,
        "name": name,
        "profile_source": profile_source,
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
        result_summary=record.result_summary,
        error_message=record.error_message,
        metadata=dict(record.task_metadata or {}),
        created_at=record.created_at,
        updated_at=record.updated_at,
        completed_at=record.completed_at,
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


def _normalize_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-._").lower()
    if normalized == "":
        raise HTTPException(status_code=422, detail="Async subagent name must contain letters or numbers.")
    return normalized[:255]


def _xml_escape(value: object) -> str:
    return str(value).replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
