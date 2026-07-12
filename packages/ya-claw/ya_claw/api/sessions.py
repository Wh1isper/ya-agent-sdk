from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from fastapi import APIRouter, Header, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sse_starlette.sse import EventSourceResponse

from ya_claw.agency.lifecycle import AgencyLifecycle
from ya_claw.config import ClawSettings
from ya_claw.controller.async_task import AsyncTaskController, ProfileResolverProtocol
from ya_claw.controller.memory import MemoryController
from ya_claw.controller.models import (
    AsyncTaskCancelRequest,
    AsyncTaskListResponse,
    AsyncTaskResponse,
    AsyncTaskSpawnRequest,
    AsyncTaskSteerRequest,
    ControlResponse,
    DispatchMode,
    MemoryActionRequest,
    MemoryActionResponse,
    RunDetail,
    SessionCreateRequest,
    SessionCreateResponse,
    SessionForkRequest,
    SessionGetResponse,
    SessionListResponse,
    SessionRunCreateRequest,
    SessionSubmitRequest,
    SessionSubmitResponse,
    SessionSummary,
    SessionTurnsResponse,
    SteerRequest,
)
from ya_claw.controller.run import RunController
from ya_claw.controller.session import SessionController
from ya_claw.execution.coordinator import ExecutionSupervisor
from ya_claw.execution.dispatcher import RunDispatcher
from ya_claw.notifications import NotificationHub
from ya_claw.runtime_state import InMemoryRuntimeState

router = APIRouter(prefix="/sessions", tags=["sessions"])
session_controller = SessionController()
run_controller = RunController()
memory_controller = MemoryController()
async_task_controller = AsyncTaskController()


@router.post("", response_model=SessionCreateResponse, status_code=201)
async def create_session(request: Request, payload: SessionCreateRequest) -> SessionCreateResponse:
    settings = _get_settings(request)
    runtime_state = _get_runtime_state(request)
    session_factory = _get_session_factory(request)
    payload.dispatch_mode = DispatchMode.ASYNC
    async with session_factory() as db_session:
        response = await session_controller.create(db_session, settings, runtime_state, payload)
        if response.run is not None and payload.input_parts:
            await _observe_session_message(
                db_session,
                settings,
                runtime_state,
                session_id=response.session.id,
                run_id=response.run.id,
                input_parts=payload.input_parts,
                source_kind=payload.trigger_type.value
                if hasattr(payload.trigger_type, "value")
                else str(payload.trigger_type),
                metadata=payload.metadata,
                submit_run=lambda run_id: _dispatch_run(request, run_id, DispatchMode.ASYNC, require_submission=False),
            )

    await _publish_session_notification(request, "session.created", response.session)
    if response.run is not None:
        await _publish_run_notification(request, "run.created", response.run)
        _dispatch_run(request, response.run.id, payload.dispatch_mode, require_submission=False)
    return response


@router.post(":stream")
async def create_session_stream(request: Request, payload: SessionCreateRequest) -> EventSourceResponse:
    runtime_state = _get_runtime_state(request)
    settings = _get_settings(request)
    session_factory = _get_session_factory(request)
    payload.dispatch_mode = DispatchMode.STREAM
    async with session_factory() as db_session:
        response = await session_controller.create(db_session, settings, runtime_state, payload)
        if response.run is not None and payload.input_parts:
            await _observe_session_message(
                db_session,
                settings,
                runtime_state,
                session_id=response.session.id,
                run_id=response.run.id,
                input_parts=payload.input_parts,
                source_kind=payload.trigger_type.value
                if hasattr(payload.trigger_type, "value")
                else str(payload.trigger_type),
                metadata=payload.metadata,
                submit_run=lambda run_id: _dispatch_run(request, run_id, DispatchMode.ASYNC, require_submission=False),
            )

    await _publish_session_notification(request, "session.created", response.session)
    if response.run is None:
        raise HTTPException(status_code=422, detail="input_parts are required for streamed session creation.")
    await _publish_run_notification(request, "run.created", response.run)
    _dispatch_run(request, response.run.id, payload.dispatch_mode, require_submission=True)
    return EventSourceResponse(runtime_state.stream_run_events(response.run.id))


@router.get("", response_model=list[SessionSummary])
async def list_sessions(
    request: Request,
    include_internal: bool = False,
    limit: int | None = None,
    before_updated_at: datetime | None = None,
    before_id: str | None = None,
    include_latest_output: bool = True,
) -> list[SessionSummary]:
    settings = _get_settings(request)
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        return await session_controller.list(
            db_session,
            settings=settings,
            include_internal=include_internal,
            limit=limit,
            before_updated_at=before_updated_at,
            before_id=before_id,
            include_latest_output=include_latest_output,
        )


@router.get("/page", response_model=SessionListResponse)
async def list_sessions_page(
    request: Request,
    include_internal: bool = False,
    limit: int = 50,
    before_updated_at: datetime | None = None,
    before_id: str | None = None,
    include_latest_output: bool = False,
) -> SessionListResponse:
    settings = _get_settings(request)
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        return await session_controller.list_page(
            db_session,
            settings=settings,
            include_internal=include_internal,
            limit=limit,
            before_updated_at=before_updated_at,
            before_id=before_id,
            include_latest_output=include_latest_output,
        )


@router.get("/{session_id}", response_model=SessionGetResponse)
async def get_session(
    request: Request,
    session_id: str,
    runs_limit: int = 20,
    before_sequence_no: int | None = None,
    include_message: bool = False,
    include_input_parts: bool = False,
    include_head_payload: bool = True,
) -> SessionGetResponse:
    settings = _get_settings(request)
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        return await session_controller.get(
            db_session,
            settings,
            session_id,
            runs_limit=runs_limit,
            before_sequence_no=before_sequence_no,
            include_message=include_message,
            include_input_parts=include_input_parts,
            include_head_payload=include_head_payload,
        )


@router.post("/{session_id}/memory:extract", response_model=MemoryActionResponse, status_code=202)
async def extract_session_memory(
    request: Request,
    session_id: str,
    payload: MemoryActionRequest,
) -> MemoryActionResponse:
    return await memory_controller.enqueue_extract(
        settings=_get_settings(request),
        session_factory=_get_session_factory(request),
        runtime_state=_get_runtime_state(request),
        source_session_id=session_id,
        request=payload,
        submit_run=lambda run_id: _dispatch_run(request, run_id, DispatchMode.ASYNC, require_submission=False),
    )


@router.post("/{session_id}/memory:summarize", response_model=MemoryActionResponse, status_code=202)
async def summarize_session_memory(
    request: Request,
    session_id: str,
    payload: MemoryActionRequest,
) -> MemoryActionResponse:
    return await memory_controller.enqueue_summary(
        settings=_get_settings(request),
        session_factory=_get_session_factory(request),
        runtime_state=_get_runtime_state(request),
        source_session_id=session_id,
        request=payload,
        submit_run=lambda run_id: _dispatch_run(request, run_id, DispatchMode.ASYNC, require_submission=False),
    )


@router.get("/{session_id}/async-tasks", response_model=AsyncTaskListResponse)
async def list_session_async_tasks(
    request: Request,
    session_id: str,
    include_terminal: bool = True,
) -> AsyncTaskListResponse:
    settings = _get_settings(request)
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        return await async_task_controller.list_tasks(
            db_session,
            settings,
            parent_session_id=session_id,
            include_terminal=include_terminal,
        )


@router.post("/{session_id}/async-tasks:spawn", response_model=AsyncTaskResponse, status_code=202)
async def spawn_session_async_task(
    request: Request,
    session_id: str,
    payload: AsyncTaskSpawnRequest,
) -> AsyncTaskResponse:
    settings = _get_settings(request)
    runtime_state = _get_runtime_state(request)
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        response = await async_task_controller.spawn_delegate(
            db_session,
            settings,
            runtime_state,
            parent_session_id=session_id,
            parent_run_id=payload.parent_run_id,
            parent_agent_id=payload.parent_agent_id,
            request=payload,
            profile_resolver=_get_profile_resolver(request),
        )
    if isinstance(response.task.task_run_id, str) and response.task.delivery in {"submitted", "resumed"}:
        _dispatch_run(request, response.task.task_run_id, DispatchMode.ASYNC, require_submission=False)
    return response


@router.get("/{session_id}/async-tasks/{task_id_or_name}", response_model=AsyncTaskResponse)
async def get_session_async_task(request: Request, session_id: str, task_id_or_name: str) -> AsyncTaskResponse:
    settings = _get_settings(request)
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        return await async_task_controller.get_task(
            db_session,
            settings,
            parent_session_id=session_id,
            task_id_or_name=task_id_or_name,
        )


@router.post("/{session_id}/async-tasks/{task_id_or_name}:steer", response_model=AsyncTaskResponse)
async def steer_session_async_task(
    request: Request,
    session_id: str,
    task_id_or_name: str,
    payload: AsyncTaskSteerRequest,
) -> AsyncTaskResponse:
    settings = _get_settings(request)
    runtime_state = _get_runtime_state(request)
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        return await async_task_controller.steer_task(
            db_session,
            settings,
            runtime_state,
            parent_session_id=session_id,
            task_id_or_name=task_id_or_name,
            request=payload,
        )


@router.post("/{session_id}/async-tasks/{task_id_or_name}:cancel", response_model=AsyncTaskResponse)
async def cancel_session_async_task(
    request: Request,
    session_id: str,
    task_id_or_name: str,
    payload: AsyncTaskCancelRequest,
) -> AsyncTaskResponse:
    settings = _get_settings(request)
    runtime_state = _get_runtime_state(request)
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        return await async_task_controller.cancel_task(
            db_session,
            settings,
            runtime_state,
            parent_session_id=session_id,
            task_id_or_name=task_id_or_name,
            request=payload,
        )


@router.get("/{session_id}/turns", response_model=SessionTurnsResponse)
async def list_session_turns(
    request: Request,
    session_id: str,
    limit: int = 20,
    before_sequence_no: int | None = None,
    cursor: str | None = None,
) -> SessionTurnsResponse:
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        return await session_controller.list_turns(
            db_session,
            session_id,
            limit=limit,
            before_sequence_no=before_sequence_no,
            cursor=cursor,
        )


@router.post("/{session_id}/submit", response_model=SessionSubmitResponse, status_code=202)
async def submit_session_input(
    request: Request, session_id: str, payload: SessionSubmitRequest
) -> SessionSubmitResponse:
    settings = _get_settings(request)
    runtime_state = _get_runtime_state(request)
    session_factory = _get_session_factory(request)
    payload.dispatch_mode = DispatchMode.ASYNC
    async with session_factory() as db_session:
        response = await session_controller.submit_input(db_session, settings, runtime_state, session_id, payload)
        await _observe_session_message(
            db_session,
            settings,
            runtime_state,
            session_id=session_id,
            run_id=response.run_id,
            input_parts=payload.input_parts,
            source_kind=payload.trigger_type.value
            if hasattr(payload.trigger_type, "value")
            else str(payload.trigger_type),
            metadata=payload.metadata,
            submit_run=lambda run_id: _dispatch_run(request, run_id, DispatchMode.ASYNC, require_submission=False),
        )
    if response.run is not None:
        await _publish_run_notification(request, "run.created", response.run)
        _dispatch_run(request, response.run.id, payload.dispatch_mode, require_submission=False)
    return response


@router.post("/{session_id}/runs", response_model=RunDetail, status_code=201)
async def create_session_run(request: Request, session_id: str, payload: SessionRunCreateRequest) -> RunDetail:
    settings = _get_settings(request)
    runtime_state = _get_runtime_state(request)
    session_factory = _get_session_factory(request)
    payload.dispatch_mode = DispatchMode.ASYNC
    async with session_factory() as db_session:
        run = await session_controller.create_run(db_session, settings, runtime_state, session_id, payload)
        await _observe_session_message(
            db_session,
            settings,
            runtime_state,
            session_id=session_id,
            run_id=run.id,
            input_parts=payload.input_parts,
            source_kind=payload.trigger_type.value
            if hasattr(payload.trigger_type, "value")
            else str(payload.trigger_type),
            metadata=payload.metadata,
            submit_run=lambda run_id: _dispatch_run(request, run_id, DispatchMode.ASYNC, require_submission=False),
        )
    await _publish_run_notification(request, "run.created", run)
    _dispatch_run(request, run.id, payload.dispatch_mode, require_submission=False)
    return run


@router.post("/{session_id}/runs:stream")
async def create_session_run_stream(
    request: Request, session_id: str, payload: SessionRunCreateRequest
) -> EventSourceResponse:
    settings = _get_settings(request)
    runtime_state = _get_runtime_state(request)
    session_factory = _get_session_factory(request)
    payload.dispatch_mode = DispatchMode.STREAM
    async with session_factory() as db_session:
        run = await session_controller.create_run(db_session, settings, runtime_state, session_id, payload)
        await _observe_session_message(
            db_session,
            settings,
            runtime_state,
            session_id=session_id,
            run_id=run.id,
            input_parts=payload.input_parts,
            source_kind=payload.trigger_type.value
            if hasattr(payload.trigger_type, "value")
            else str(payload.trigger_type),
            metadata=payload.metadata,
            submit_run=lambda run_id: _dispatch_run(request, run_id, DispatchMode.ASYNC, require_submission=False),
        )
    await _publish_run_notification(request, "run.created", run)
    _dispatch_run(request, run.id, payload.dispatch_mode, require_submission=True)
    return EventSourceResponse(runtime_state.stream_run_events(run.id))


@router.post("/{session_id}/steer", response_model=ControlResponse)
async def steer_session(request: Request, session_id: str, payload: SteerRequest) -> ControlResponse:
    runtime_state = _get_runtime_state(request)
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        run_id = await session_controller.resolve_active_run_id(db_session, session_id)
        return await run_controller.steer(db_session, runtime_state, run_id, payload)


@router.post("/{session_id}/interrupt", response_model=RunDetail)
async def interrupt_session(request: Request, session_id: str) -> RunDetail:
    settings = _get_settings(request)
    runtime_state = _get_runtime_state(request)
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        run_id = await session_controller.resolve_active_run_id(db_session, session_id)
        run = await run_controller.interrupt(db_session, settings, runtime_state, run_id)
    await _publish_run_notification(request, "run.updated", run)
    return run


@router.post("/{session_id}/cancel", response_model=RunDetail)
async def cancel_session(request: Request, session_id: str) -> RunDetail:
    settings = _get_settings(request)
    runtime_state = _get_runtime_state(request)
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        run_id = await session_controller.resolve_active_run_id(db_session, session_id)
        run = await run_controller.cancel(db_session, settings, runtime_state, run_id)
    await _publish_run_notification(request, "run.updated", run)
    return run


@router.post("/{session_id}/fork", response_model=SessionSummary, status_code=201)
async def fork_session(request: Request, session_id: str, payload: SessionForkRequest) -> SessionSummary:
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        forked_session = await session_controller.fork(db_session, session_id, payload)
    await _publish_session_notification(request, "session.created", forked_session)
    return forked_session


@router.get("/{session_id}/events")
async def stream_session_events(
    request: Request,
    session_id: str,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
) -> EventSourceResponse:
    runtime_state = _get_runtime_state(request)
    if runtime_state.get_session_run_handle(session_id) is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' does not have an active event buffer.")
    return EventSourceResponse(runtime_state.stream_session_events(session_id, last_event_id=last_event_id))


async def _publish_session_notification(request: Request, event_type: str, session: SessionSummary) -> None:
    notification_hub = _get_notification_hub(request)
    await notification_hub.publish(
        event_type,
        {
            "session_id": session.id,
            "status": session.status,
            "status_reason": session.status_reason,
            "status_detail": session.status_detail,
            "profile_name": session.profile_name,
            "run_count": session.run_count,
            "head_run_id": session.head_run_id,
            "head_success_run_id": session.head_success_run_id,
            "active_run_id": session.active_run_id,
        },
    )


async def _publish_run_notification(request: Request, event_type: str, run: RunDetail) -> None:
    notification_hub = _get_notification_hub(request)
    session_status_reason = _session_status_reason_from_run(run)
    session_status_detail = _session_status_detail_from_run(run)
    await notification_hub.publish(
        event_type,
        {
            "session_id": run.session_id,
            "run_id": run.id,
            "status": run.status,
            "sequence_no": run.sequence_no,
            "profile_name": run.profile_name,
            "termination_reason": run.termination_reason,
            "error_message": run.error_message,
            "session_status": run.status,
            "session_status_reason": session_status_reason,
            "session_status_detail": session_status_detail,
        },
    )
    if event_type != "session.updated":
        await notification_hub.publish(
            "session.updated",
            {
                "session_id": run.session_id,
                "status": run.status,
                "status_reason": session_status_reason,
                "status_detail": session_status_detail,
                "profile_name": run.profile_name,
                "head_run_id": run.id,
                "active_run_id": run.id if run.status in {"queued", "running"} else None,
                "latest_run_id": run.id,
                "latest_run_sequence_no": run.sequence_no,
                "latest_run_status": run.status,
                "termination_reason": run.termination_reason,
                "error_message": run.error_message,
                "updated_at": (run.finished_at or run.started_at or run.created_at).isoformat(),
            },
        )


def _session_status_reason_from_run(run: RunDetail) -> str:
    if run.status == "queued":
        return "run_queued"
    if run.status == "running":
        active_interactions = run.metadata.get("active_interactions")
        if isinstance(active_interactions, list) and active_interactions:
            return "hitl_pending"
        return "run_running"
    if run.status == "completed":
        return "run_completed"
    if run.status == "failed":
        return "run_failed"
    if run.status == "cancelled":
        return "run_cancelled"
    return "idle"


def _session_status_detail_from_run(run: RunDetail) -> dict[str, object]:
    detail: dict[str, object] = {
        "run_id": run.id,
        "sequence_no": run.sequence_no,
        "trigger_type": run.trigger_type,
    }
    if run.termination_reason is not None:
        detail["termination_reason"] = run.termination_reason
    if run.error_message is not None:
        detail["error_message"] = run.error_message
    active_interactions = run.metadata.get("active_interactions")
    if isinstance(active_interactions, list) and active_interactions:
        detail["active_interactions"] = active_interactions
        detail["active_interaction_count"] = len(active_interactions)
    return detail


async def _observe_session_message(
    db_session: AsyncSession,
    settings: ClawSettings,
    runtime_state: InMemoryRuntimeState,
    *,
    session_id: str,
    run_id: str | None,
    input_parts: Sequence[object],
    source_kind: str,
    metadata: dict[str, object],
    submit_run,
) -> None:
    if source_kind == "agency_handoff":
        return
    if not settings.agency_enabled:
        return
    lifecycle = AgencyLifecycle(settings=settings, runtime_state=runtime_state, submit_run=submit_run)
    try:
        await lifecycle.observe_message(
            db_session,
            source_session_id=session_id,
            source_run_id=run_id,
            input_parts=list(input_parts),
            source_kind=source_kind,
            client_token=None,
            metadata=metadata,
        )
    except HTTPException as exc:
        if exc.status_code != 409:
            raise


def _dispatch_run(request: Request, run_id: str, mode: DispatchMode, *, require_submission: bool) -> bool:
    dispatcher = RunDispatcher(_get_execution_supervisor(request))
    result = dispatcher.dispatch(run_id, mode)
    if require_submission and not result.submitted:
        raise HTTPException(status_code=503, detail="Execution supervisor is unavailable.")
    return result.submitted


def _get_settings(request: Request) -> ClawSettings:
    settings = request.app.state.settings
    if not isinstance(settings, ClawSettings):
        raise TypeError("Application settings are unavailable.")
    return settings


def _get_runtime_state(request: Request) -> InMemoryRuntimeState:
    runtime_state = request.app.state.runtime_state
    if not isinstance(runtime_state, InMemoryRuntimeState):
        raise TypeError("Runtime state is unavailable.")
    return runtime_state


def _get_notification_hub(request: Request) -> NotificationHub:
    notification_hub = request.app.state.notification_hub
    if not isinstance(notification_hub, NotificationHub):
        raise TypeError("Notification hub is unavailable.")
    return notification_hub


def _get_execution_supervisor(request: Request) -> ExecutionSupervisor | None:
    supervisor = request.app.state.execution_supervisor
    if supervisor is None or isinstance(supervisor, ExecutionSupervisor):
        return supervisor
    raise TypeError("Execution supervisor is unavailable.")


def _get_profile_resolver(request: Request) -> ProfileResolverProtocol | None:
    resolver = getattr(request.app.state, "profile_resolver", None)
    if resolver is None or isinstance(resolver, ProfileResolverProtocol):
        return resolver
    raise TypeError("Profile resolver is unavailable.")


def _get_session_factory(request: Request) -> async_sessionmaker[AsyncSession]:
    session_factory = request.app.state.db_session_factory
    if not isinstance(session_factory, async_sessionmaker):
        raise TypeError("Database session factory is unavailable.")
    return session_factory
