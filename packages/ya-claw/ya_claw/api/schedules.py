from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ya_claw.config import ClawSettings
from ya_claw.controller.schedule import (
    ScheduleController,
    ScheduleCreateRequest,
    ScheduleFireListResponse,
    ScheduleFireSummary,
    ScheduleListResponse,
    ScheduleManualTriggerRequest,
    ScheduleSummary,
    ScheduleUpdateRequest,
)
from ya_claw.execution.coordinator import ExecutionSupervisor
from ya_claw.execution.dispatcher import RunDispatcher
from ya_claw.notifications import NotificationHub
from ya_claw.runtime_state import InMemoryRuntimeState

router = APIRouter(prefix="/schedules", tags=["schedules"])
controller = ScheduleController()


@router.get("", response_model=ScheduleListResponse)
async def list_schedules(
    request: Request,
    include_deleted: bool = False,
    owner_session_id: str | None = None,
    schedule_id: str | None = None,
    limit: int = 100,
    include_recent_runs: bool = True,
) -> ScheduleListResponse:
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        return await controller.list(
            db_session,
            include_deleted=include_deleted,
            owner_session_id=owner_session_id,
            schedule_id=schedule_id,
            limit=limit,
            include_recent_runs=include_recent_runs,
        )


@router.post("", response_model=ScheduleSummary, status_code=201)
async def create_schedule(request: Request, payload: ScheduleCreateRequest) -> ScheduleSummary:
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        schedule = await controller.create(db_session, payload)
    await _publish_schedule_notification(request, "schedule.created", schedule)
    return schedule


@router.get("/{schedule_id}", response_model=ScheduleSummary)
async def get_schedule(request: Request, schedule_id: str) -> ScheduleSummary:
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        return await controller.get(db_session, schedule_id)


@router.patch("/{schedule_id}", response_model=ScheduleSummary)
async def update_schedule(request: Request, schedule_id: str, payload: ScheduleUpdateRequest) -> ScheduleSummary:
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        schedule = await controller.update(db_session, schedule_id, payload)
    await _publish_schedule_notification(request, "schedule.updated", schedule)
    return schedule


@router.delete("/{schedule_id}", response_model=ScheduleSummary)
async def delete_schedule(request: Request, schedule_id: str) -> ScheduleSummary:
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        schedule = await controller.delete(db_session, schedule_id)
    await _publish_schedule_notification(request, "schedule.deleted", schedule)
    return schedule


@router.post("/{schedule_id}:pause", response_model=ScheduleSummary)
async def pause_schedule(request: Request, schedule_id: str) -> ScheduleSummary:
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        schedule = await controller.update(db_session, schedule_id, ScheduleUpdateRequest(enabled=False))
    await _publish_schedule_notification(request, "schedule.updated", schedule)
    return schedule


@router.post("/{schedule_id}:resume", response_model=ScheduleSummary)
async def resume_schedule(request: Request, schedule_id: str) -> ScheduleSummary:
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        schedule = await controller.update(db_session, schedule_id, ScheduleUpdateRequest(enabled=True))
    await _publish_schedule_notification(request, "schedule.updated", schedule)
    return schedule


@router.post("/{schedule_id}:trigger", response_model=ScheduleFireSummary, status_code=201)
async def trigger_schedule(
    request: Request,
    schedule_id: str,
    payload: ScheduleManualTriggerRequest | None = None,
) -> ScheduleFireSummary:
    settings = _get_settings(request)
    runtime_state = _get_runtime_state(request)
    session_factory = _get_session_factory(request)
    dispatcher = RunDispatcher(_get_execution_supervisor(request))
    async with session_factory() as db_session:
        fire = await controller.trigger(db_session, settings, runtime_state, dispatcher, schedule_id, payload)
    await _publish_fire_notification(request, "schedule.fire.created", fire)
    return fire


@router.get("/{schedule_id}/fires", response_model=ScheduleFireListResponse)
async def list_schedule_fires(request: Request, schedule_id: str, limit: int = 50) -> ScheduleFireListResponse:
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        return await controller.list_fires(db_session, schedule_id, limit=limit)


async def _publish_schedule_notification(request: Request, event_type: str, schedule: ScheduleSummary) -> None:
    notification_hub = _get_notification_hub(request)
    await notification_hub.publish(
        event_type,
        {
            "schedule_id": schedule.id,
            "status": schedule.status,
            "enabled": schedule.enabled,
            "next_fire_at": schedule.trigger.get("next_fire_at"),
            "trigger_kind": schedule.trigger.get("kind"),
            "run_at": schedule.trigger.get("run_at"),
            "execution_mode": schedule.execution_mode,
        },
    )


async def _publish_fire_notification(request: Request, event_type: str, fire: ScheduleFireSummary) -> None:
    notification_hub = _get_notification_hub(request)
    await notification_hub.publish(
        event_type,
        {
            "schedule_id": fire.schedule_id,
            "schedule_fire_id": fire.id,
            "status": fire.status,
            "session_id": fire.created_session_id,
            "run_id": fire.run_id,
            "active_run_id": fire.active_run_id,
        },
    )


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
        raise HTTPException(status_code=503, detail="Database session factory is unavailable.")
    return session_factory
