from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ya_claw.config import ClawSettings
from ya_claw.controller.agency import AgencyController
from ya_claw.controller.models import (
    AgencyClearResponse,
    AgencyConfigResponse,
    AgencyFireListResponse,
    AgencySourceSessionSubmitRequest,
    AgencySourceSessionSubmitResponse,
    AgencyStatusResponse,
    DispatchMode,
)
from ya_claw.execution.coordinator import ExecutionSupervisor
from ya_claw.execution.dispatcher import RunDispatcher
from ya_claw.notifications import NotificationHub
from ya_claw.runtime_state import InMemoryRuntimeState

router = APIRouter(prefix="/agency", tags=["agency"])
controller = AgencyController()


@router.get("/config", response_model=AgencyConfigResponse)
async def get_agency_config(request: Request) -> AgencyConfigResponse:
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        return await controller.config(db_session, _get_settings(request), _get_runtime_state(request))


@router.get("/status", response_model=AgencyStatusResponse)
async def get_agency_status(request: Request) -> AgencyStatusResponse:
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        return await controller.status(db_session, _get_settings(request), _get_runtime_state(request))


@router.get("/fires", response_model=AgencyFireListResponse)
async def list_agency_fires(request: Request, limit: int = 50) -> AgencyFireListResponse:
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        return await controller.list_fires(db_session, limit=limit)


@router.post(":bootstrap", response_model=AgencyConfigResponse, status_code=202)
async def bootstrap_agency(request: Request) -> AgencyConfigResponse:
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        response = await controller.bootstrap(db_session, _get_settings(request), _get_runtime_state(request))
    await _get_notification_hub(request).publish("agency.config.updated", response.model_dump(mode="json"))
    return response


@router.post("/source-session:submit", response_model=AgencySourceSessionSubmitResponse, status_code=202)
async def submit_agency_handoff_to_source_session(
    request: Request,
    payload: AgencySourceSessionSubmitRequest,
) -> AgencySourceSessionSubmitResponse:
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        response = await controller.submit_to_source_session(
            db_session,
            _get_settings(request),
            _get_runtime_state(request),
            payload,
        )
    if response.session_submit.run is not None:
        _dispatch_run(request, response.session_submit.run.id, DispatchMode.ASYNC, require_submission=False)
    await _get_notification_hub(request).publish(
        "agency.source_session.submitted",
        {
            **response.model_dump(mode="json"),
            "session_id": response.source_session_id,
            "run_id": response.run_id,
        },
    )
    return response


@router.post(":clear", response_model=AgencyClearResponse, status_code=202)
async def clear_agency(request: Request) -> AgencyClearResponse:
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        response = await controller.clear(db_session, _get_settings(request), _get_runtime_state(request))
    await _get_notification_hub(request).publish(
        "agency.cleared",
        {
            "cleared_session_id": response.cleared_session_id,
            "new_agency_session_id": response.new_agency_session_id,
            "deleted_fire_count": response.deleted_fire_count,
            "cleared_at": response.cleared_at.isoformat(),
        },
    )
    return response


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


def _dispatch_run(request: Request, run_id: str, mode: DispatchMode, *, require_submission: bool) -> bool:
    dispatcher = RunDispatcher(_get_execution_supervisor(request))
    result = dispatcher.dispatch(run_id, mode)
    if require_submission and not result.submitted:
        raise HTTPException(status_code=503, detail="Execution supervisor is unavailable.")
    return result.submitted


def _get_execution_supervisor(request: Request) -> ExecutionSupervisor | None:
    supervisor = request.app.state.execution_supervisor
    if supervisor is None or isinstance(supervisor, ExecutionSupervisor):
        return supervisor
    raise TypeError("Execution supervisor is unavailable.")


def _get_session_factory(request: Request) -> async_sessionmaker[AsyncSession]:
    session_factory = request.app.state.db_session_factory
    if not isinstance(session_factory, async_sessionmaker):
        raise HTTPException(status_code=503, detail="Database session factory is unavailable.")
    return session_factory
