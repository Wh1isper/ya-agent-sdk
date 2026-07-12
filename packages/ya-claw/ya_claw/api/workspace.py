from __future__ import annotations

from collections.abc import Iterator
from urllib.parse import quote

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ya_claw.config import ClawSettings
from ya_claw.controller.workspace_files import WorkspaceDownload, WorkspaceFilesController
from ya_claw.controller.workspace_runtime import WorkspaceRuntimeController
from ya_claw.notifications import NotificationHub
from ya_claw.runtime_state import InMemoryRuntimeState
from ya_claw.workspace import EnvironmentFactory, WorkspaceProvider
from ya_claw.workspace.file_models import WorkspaceFileListResponse, WorkspaceTextFileResponse
from ya_claw.workspace.runtime_models import (
    SessionSandboxState,
    SessionWorkspaceState,
    WorkspaceResolveRequest,
    WorkspaceResolveResponse,
    WorkspaceRuntimeStatus,
)

router = APIRouter(tags=["workspace"])
workspace_controller = WorkspaceRuntimeController()
workspace_files_controller = WorkspaceFilesController()


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


@router.get("/sessions/{session_id}/workspace/files", response_model=WorkspaceFileListResponse)
async def list_session_workspace_files(
    request: Request,
    session_id: str,
    path: str | None = Query(default=None),
    include_hidden: bool = False,
    limit: int = Query(default=200, ge=1, le=1000),
    cursor: str | None = Query(default=None, max_length=4096),
    offset: int = Query(default=0, ge=0, le=100_000),
) -> WorkspaceFileListResponse:
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        return await workspace_files_controller.list_files(
            db_session=db_session,
            workspace_provider=_get_workspace_provider(request),
            session_id=session_id,
            path=path,
            include_hidden=include_hidden,
            limit=limit,
            offset=offset,
            cursor=cursor,
        )


@router.get("/sessions/{session_id}/workspace/file", response_model=WorkspaceTextFileResponse)
async def read_session_workspace_file(
    request: Request,
    session_id: str,
    path: str = Query(min_length=1),
) -> WorkspaceTextFileResponse:
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        return await workspace_files_controller.read_text_file(
            db_session=db_session,
            workspace_provider=_get_workspace_provider(request),
            session_id=session_id,
            path=path,
        )


@router.get("/sessions/{session_id}/workspace/file:download", response_model=None)
async def download_session_workspace_file(
    request: Request,
    session_id: str,
    path: str = Query(min_length=1),
) -> StreamingResponse:
    session_factory = _get_session_factory(request)
    async with session_factory() as db_session:
        download = await workspace_files_controller.open_download(
            db_session=db_session,
            workspace_provider=_get_workspace_provider(request),
            session_id=session_id,
            path=path,
            max_bytes=_get_settings(request).workspace_download_max_bytes,
        )
    return StreamingResponse(
        _stream_download(download),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": _attachment_header(download.filename),
            "X-Content-Type-Options": "nosniff",
        },
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


def _stream_download(download: WorkspaceDownload) -> Iterator[bytes]:
    bytes_sent = 0
    try:
        while True:
            bytes_remaining = download.max_bytes - bytes_sent
            chunk = download.file.read(min(64 * 1024, bytes_remaining + 1))
            if not chunk:
                return
            bytes_sent += len(chunk)
            if bytes_sent > download.max_bytes:
                raise RuntimeError("Workspace file exceeded the configured download limit while streaming.")
            yield chunk
    finally:
        download.file.close()


def _attachment_header(filename: str) -> str:
    quoted_filename = quote(filename, safe="")
    if quoted_filename == filename:
        return f'attachment; filename="{filename}"'
    return f"attachment; filename*=utf-8''{quoted_filename}"


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
