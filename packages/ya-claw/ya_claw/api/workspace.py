from __future__ import annotations

from fastapi import APIRouter, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ya_claw.config import ClawSettings
from ya_claw.controller.workspace_runtime import WorkspaceRuntimeController
from ya_claw.notifications import NotificationHub
from ya_claw.runtime_state import InMemoryRuntimeState
from ya_claw.workspace import EnvironmentFactory, WorkspaceProvider
from ya_claw.workspace.runtime_models import (
    SessionSandboxState,
    SessionWorkspaceState,
    WorkspaceResolveRequest,
    WorkspaceResolveResponse,
    WorkspaceRuntimeStatus,
)

router = APIRouter(tags=["workspace"])
workspace_controller = WorkspaceRuntimeController()


@router.get("/workspace/runtime", response_model=WorkspaceRuntimeStatus)
async def get_workspace_runtime(request: Request) -> WorkspaceRuntimeStatus:
    return workspace_controller.get_runtime_status(_get_settings(request))


@router.post("/workspace:resolve", response_model=WorkspaceResolveResponse)
async def resolve_workspace(request: Request, payload: WorkspaceResolveRequest) -> WorkspaceResolveResponse:
    return workspace_controller.resolve_workspace(
        workspace_provider=_get_workspace_provider(request),
        metadata=payload.metadata,
    )


@router.get("/sessions/{session_id}/workspace", response_model=SessionWorkspaceState)
async def get_session_workspace(request: Request, session_id: str) -> SessionWorkspaceState:
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        return await workspace_controller.get_session_workspace(
            settings=_get_settings(request),
            db_session=db_session,
            workspace_provider=_get_workspace_provider(request),
            session_id=session_id,
        )


@router.get("/sessions/{session_id}/sandbox", response_model=SessionSandboxState)
async def get_session_sandbox(request: Request, session_id: str) -> SessionSandboxState:
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        return await workspace_controller.get_session_sandbox(
            settings=_get_settings(request),
            db_session=db_session,
            session_id=session_id,
        )


@router.post("/sessions/{session_id}/sandbox:prepare", response_model=SessionSandboxState)
async def prepare_session_sandbox(request: Request, session_id: str) -> SessionSandboxState:
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        return await workspace_controller.prepare_session_sandbox(
            settings=_get_settings(request),
            db_session=db_session,
            workspace_provider=_get_workspace_provider(request),
            environment_factory=_get_environment_factory(request),
            notification_hub=_get_notification_hub(request),
            runtime_state=_get_runtime_state(request),
            session_id=session_id,
        )


@router.post("/sessions/{session_id}/sandbox:stop", response_model=SessionSandboxState)
async def stop_session_sandbox(request: Request, session_id: str) -> SessionSandboxState:
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        return await workspace_controller.stop_session_sandbox(
            settings=_get_settings(request),
            db_session=db_session,
            notification_hub=_get_notification_hub(request),
            runtime_state=_get_runtime_state(request),
            session_id=session_id,
        )


def _get_settings(request: Request) -> ClawSettings:
    settings = request.app.state.settings
    if not isinstance(settings, ClawSettings):
        raise TypeError("Application settings are unavailable.")
    return settings


def _get_session_factory(request: Request) -> async_sessionmaker[AsyncSession]:
    session_factory = request.app.state.db_session_factory
    if not isinstance(session_factory, async_sessionmaker):
        raise TypeError("Database session factory is unavailable.")
    return session_factory


def _get_workspace_provider(request: Request) -> WorkspaceProvider:
    workspace_provider = request.app.state.workspace_provider
    if not isinstance(workspace_provider, WorkspaceProvider):
        raise TypeError("Workspace provider is unavailable.")
    return workspace_provider


def _get_environment_factory(request: Request) -> EnvironmentFactory:
    environment_factory = request.app.state.environment_factory
    if not isinstance(environment_factory, EnvironmentFactory):
        raise TypeError("Environment factory is unavailable.")
    return environment_factory


def _get_notification_hub(request: Request) -> NotificationHub:
    notification_hub = request.app.state.notification_hub
    if not isinstance(notification_hub, NotificationHub):
        raise TypeError("Notification hub is unavailable.")
    return notification_hub


def _get_runtime_state(request: Request) -> InMemoryRuntimeState:
    runtime_state = request.app.state.runtime_state
    if not isinstance(runtime_state, InMemoryRuntimeState):
        raise TypeError("Runtime state is unavailable.")
    return runtime_state
