from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ya_claw.config import ClawSettings
from ya_claw.controller.workflow import (
    WorkflowActorContext,
    WorkflowCancelRequest,
    WorkflowController,
    WorkflowDefinitionCreateRequest,
    WorkflowDefinitionDetail,
    WorkflowDefinitionListResponse,
    WorkflowDefinitionUpdateRequest,
    WorkflowEventListResponse,
    WorkflowNodeSteerRequest,
    WorkflowRunDetail,
    WorkflowRunListResponse,
    WorkflowTriggerRequest,
)
from ya_claw.execution.coordinator import ExecutionSupervisor
from ya_claw.execution.dispatcher import RunDispatcher
from ya_claw.notifications import NotificationHub
from ya_claw.runtime_state import InMemoryRuntimeState

router = APIRouter(tags=["workflows"])
controller = WorkflowController()


@router.get("/workflows", response_model=WorkflowDefinitionListResponse)
async def list_workflows(
    request: Request,
    query: str | None = None,
    tags: Annotated[list[str] | None, Query()] = None,
    status: str | None = None,
    scope: str | None = None,
    owner_kind: str | None = None,
    owner_session_id: str | None = None,
    supervisor_session_id: str | None = None,
    trigger_kind: str | None = None,
    created_by_current_session: bool = False,
    supervised_by_current_session: bool = False,
    touched_by_current_session: bool = False,
    only_current_session: bool = False,
    include_archived: bool = False,
    current_session_id: str | None = None,
    limit: int = 100,
) -> WorkflowDefinitionListResponse:
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        return await controller.list_definitions(
            db_session,
            query=query,
            tags=tags,
            status=status,
            scope=scope,
            owner_kind=owner_kind,
            owner_session_id=owner_session_id,
            supervisor_session_id=supervisor_session_id,
            trigger_kind=trigger_kind,
            created_by_current_session=created_by_current_session,
            supervised_by_current_session=supervised_by_current_session,
            touched_by_current_session=touched_by_current_session,
            only_current_session=only_current_session,
            include_archived=include_archived,
            current_session_id=current_session_id,
            limit=limit,
        )


@router.post("/workflows", response_model=WorkflowDefinitionDetail, status_code=201)
async def create_workflow(request: Request, payload: WorkflowDefinitionCreateRequest) -> WorkflowDefinitionDetail:
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        workflow = await controller.create_definition(db_session, payload)
    await _publish(request, "workflow.created", {"workflow_id": workflow.id, "status": workflow.status})
    return workflow


@router.get("/workflows/{workflow_id}", response_model=WorkflowDefinitionDetail)
async def get_workflow(request: Request, workflow_id: str) -> WorkflowDefinitionDetail:
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        return await controller.get_definition(db_session, workflow_id)


@router.patch("/workflows/{workflow_id}", response_model=WorkflowDefinitionDetail)
async def update_workflow(
    request: Request,
    workflow_id: str,
    payload: WorkflowDefinitionUpdateRequest,
) -> WorkflowDefinitionDetail:
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        workflow = await controller.update_definition(db_session, workflow_id, payload)
    await _publish(request, "workflow.updated", {"workflow_id": workflow.id, "status": workflow.status})
    return workflow


@router.post("/workflows/{workflow_id}:archive", response_model=WorkflowDefinitionDetail)
async def archive_workflow(request: Request, workflow_id: str) -> WorkflowDefinitionDetail:
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        workflow = await controller.archive_definition(db_session, workflow_id)
    await _publish(request, "workflow.archived", {"workflow_id": workflow.id, "status": workflow.status})
    return workflow


@router.post("/workflows/{workflow_id}:trigger", response_model=WorkflowRunDetail, status_code=201)
async def trigger_workflow(
    request: Request,
    workflow_id: str,
    payload: WorkflowTriggerRequest,
) -> WorkflowRunDetail:
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        run = await controller.trigger(db_session, workflow_id, payload)
    await _publish(request, "workflow.run.created", {"workflow_id": workflow_id, "workflow_run_id": run.id})
    return run


@router.get("/workflow-runs", response_model=WorkflowRunListResponse)
async def list_workflow_runs(
    request: Request,
    workflow_id: str | None = None,
    status: str | None = None,
    trigger_kind: str | None = None,
    supervisor_session_id: str | None = None,
    only_current_session: bool = False,
    only_supervised_by_current_session: bool = False,
    only_touched_by_current_session: bool = False,
    include_completed: bool = True,
    current_session_id: str | None = None,
    limit: int = 100,
) -> WorkflowRunListResponse:
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        return await controller.list_runs(
            db_session,
            workflow_id=workflow_id,
            status=status,
            trigger_kind=trigger_kind,
            supervisor_session_id=supervisor_session_id,
            only_current_session=only_current_session,
            only_supervised_by_current_session=only_supervised_by_current_session,
            only_touched_by_current_session=only_touched_by_current_session,
            include_completed=include_completed,
            current_session_id=current_session_id,
            limit=limit,
        )


@router.get("/workflow-runs/{workflow_run_id}", response_model=WorkflowRunDetail)
async def get_workflow_run(request: Request, workflow_run_id: str) -> WorkflowRunDetail:
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        return await controller.get_run(db_session, workflow_run_id)


@router.get("/workflow-runs/{workflow_run_id}/events", response_model=WorkflowEventListResponse)
async def list_workflow_events(
    request: Request,
    workflow_run_id: str,
    after_event_id: str | None = None,
    limit: int = 200,
) -> WorkflowEventListResponse:
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        return await controller.list_events(db_session, workflow_run_id, after_event_id=after_event_id, limit=limit)


@router.post("/workflow-runs/{workflow_run_id}/cancel", response_model=WorkflowRunDetail)
async def cancel_workflow_run(
    request: Request,
    workflow_run_id: str,
    payload: WorkflowCancelRequest | None = None,
) -> WorkflowRunDetail:
    settings = _get_settings(request)
    runtime_state = _get_runtime_state(request)
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        run = await controller.cancel_run(db_session, settings, runtime_state, workflow_run_id, payload)
    await _publish(request, "workflow.run.cancelled", {"workflow_run_id": run.id, "status": run.status})
    return run


@router.post("/workflow-runs/{workflow_run_id}/nodes/{node_id}/steer", response_model=WorkflowRunDetail)
async def steer_workflow_node(
    request: Request,
    workflow_run_id: str,
    node_id: str,
    payload: WorkflowNodeSteerRequest,
) -> WorkflowRunDetail:
    runtime_state = _get_runtime_state(request)
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        run = await controller.steer_node(db_session, runtime_state, workflow_run_id, node_id, payload)
    await _publish(request, "workflow.node.steered", {"workflow_run_id": run.id, "node_id": node_id})
    return run


@router.post("/agent/workflows", response_model=WorkflowDefinitionDetail, status_code=201, include_in_schema=False)
async def create_agent_workflow(request: Request, payload: WorkflowDefinitionCreateRequest) -> WorkflowDefinitionDetail:
    session_factory = _get_session_factory(request)
    actor = _actor_from_headers(request)
    async with session_factory() as db_session:
        workflow = await controller.create_definition(db_session, payload, actor=actor)
    await _publish(request, "workflow.created", {"workflow_id": workflow.id, "status": workflow.status})
    return workflow


@router.post(
    "/agent/workflows/{workflow_id}:trigger", response_model=WorkflowRunDetail, status_code=201, include_in_schema=False
)
async def trigger_agent_workflow(
    request: Request,
    workflow_id: str,
    payload: WorkflowTriggerRequest,
) -> WorkflowRunDetail:
    session_factory = _get_session_factory(request)
    actor = _actor_from_headers(request)
    async with session_factory() as db_session:
        run = await controller.trigger(db_session, workflow_id, payload, actor=actor)
    await _publish(request, "workflow.run.created", {"workflow_id": workflow_id, "workflow_run_id": run.id})
    return run


def _actor_from_headers(request: Request) -> WorkflowActorContext:
    return WorkflowActorContext(
        actor_kind="agent",
        current_session_id=request.headers.get("X-YA-Claw-Session-Id"),
        current_run_id=request.headers.get("X-YA-Claw-Run-Id"),
        profile_name=request.headers.get("X-YA-Claw-Profile-Name"),
    )


async def _publish(request: Request, event_type: str, payload: dict[str, object]) -> None:
    notification_hub = _get_notification_hub(request)
    await notification_hub.publish(event_type, payload)


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


def _get_execution_supervisor(request: Request) -> ExecutionSupervisor | None:
    supervisor = request.app.state.execution_supervisor
    if supervisor is None or isinstance(supervisor, ExecutionSupervisor):
        return supervisor
    raise TypeError("Execution supervisor is unavailable.")


def _get_run_dispatcher(request: Request) -> RunDispatcher:
    return RunDispatcher(_get_execution_supervisor(request))


def _get_notification_hub(request: Request) -> NotificationHub:
    notification_hub = request.app.state.notification_hub
    if not isinstance(notification_hub, NotificationHub):
        raise TypeError("Notification hub is unavailable.")
    return notification_hub


def _get_session_factory(request: Request) -> async_sessionmaker[AsyncSession]:
    session_factory = request.app.state.db_session_factory
    if not isinstance(session_factory, async_sessionmaker):
        raise HTTPException(status_code=503, detail="Database session factory is unavailable.")
    return session_factory
