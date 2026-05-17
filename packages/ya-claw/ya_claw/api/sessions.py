from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sse_starlette.sse import EventSourceResponse

from ya_claw.config import ClawSettings
from ya_claw.controller.memory import MemoryController
from ya_claw.controller.models import (
    ControlResponse,
    DispatchMode,
    MemoryActionRequest,
    MemoryActionResponse,
    RunDetail,
    SessionCreateRequest,
    SessionCreateResponse,
    SessionForkRequest,
    SessionGetResponse,
    SessionRunCreateRequest,
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


@router.post("", response_model=SessionCreateResponse, status_code=201)
async def create_session(request: Request, payload: SessionCreateRequest) -> SessionCreateResponse:
    settings = _get_settings(request)
    runtime_state = _get_runtime_state(request)
    session_factory = _get_session_factory(request)
    payload.dispatch_mode = DispatchMode.ASYNC
    async with session_factory() as db_session:
        response = await session_controller.create(db_session, settings, runtime_state, payload)

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

    await _publish_session_notification(request, "session.created", response.session)
    if response.run is None:
        raise HTTPException(status_code=422, detail="input_parts are required for streamed session creation.")
    await _publish_run_notification(request, "run.created", response.run)
    _dispatch_run(request, response.run.id, payload.dispatch_mode, require_submission=True)
    return EventSourceResponse(runtime_state.stream_run_events(response.run.id))


@router.get("", response_model=list[SessionSummary])
async def list_sessions(request: Request, include_internal: bool = False) -> list[SessionSummary]:
    settings = _get_settings(request)
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        return await session_controller.list(db_session, settings=settings, include_internal=include_internal)


@router.get("/{session_id}", response_model=SessionGetResponse)
async def get_session(
    request: Request,
    session_id: str,
    runs_limit: int = 20,
    before_sequence_no: int | None = None,
    include_message: bool = False,
    include_input_parts: bool = False,
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


@router.post("/{session_id}/runs", response_model=RunDetail, status_code=201)
async def create_session_run(request: Request, session_id: str, payload: SessionRunCreateRequest) -> RunDetail:
    settings = _get_settings(request)
    runtime_state = _get_runtime_state(request)
    session_factory = _get_session_factory(request)
    payload.dispatch_mode = DispatchMode.ASYNC
    async with session_factory() as db_session:
        run = await session_controller.create_run(db_session, settings, runtime_state, session_id, payload)

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


def _get_session_factory(request: Request) -> async_sessionmaker[AsyncSession]:
    session_factory = request.app.state.db_session_factory
    if not isinstance(session_factory, async_sessionmaker):
        raise TypeError("Database session factory is unavailable.")
    return session_factory
