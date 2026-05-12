from __future__ import annotations

from fastapi import APIRouter, Header, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from ya_claw.config import ClawSettings
from ya_claw.notifications import NotificationHub
from ya_claw.runtime_state import InMemoryRuntimeState

router = APIRouter(prefix="/claw", tags=["claw"])


class ClawFeatures(BaseModel):
    session_events: bool = True
    run_events: bool = True
    notifications: bool = True
    notification_replay: bool = True
    session_status_reasons: bool = True
    hitl_status_reason: bool = True
    profiles: bool = True
    schedules: bool = True
    heartbeat: bool = True
    session_workspace_binding: bool = True
    run_workspace_override: bool = True
    multi_mount_workspaces: bool = True
    session_docker_sandbox: bool = True
    run_scoped_auto_task_sandbox: bool = True
    sandbox_idle_ttl: bool = True


class ClawInfo(BaseModel):
    name: str
    environment: str
    version: str
    service_version: str
    service_commit: str | None = None
    service_revision: str
    service_build: str | None = None
    service_image: str | None = None
    public_base_url: str
    instance_id: str
    auth: str = "bearer"
    surfaces: list[str] = Field(default_factory=list)
    workspace_provider_backend: str
    storage_model: str
    features: ClawFeatures = Field(default_factory=ClawFeatures)
    workspace_mount_modes: list[str] = Field(default_factory=lambda: ["rw", "ro"])
    sandbox_retention_policies: list[str] = Field(default_factory=lambda: ["stop_on_idle", "keep_warm"])
    limits: dict[str, int] = Field(default_factory=dict)


@router.get("/info", response_model=ClawInfo)
async def get_claw_info(request: Request) -> ClawInfo:
    settings = _get_settings(request)
    return ClawInfo(
        name=settings.app_name,
        environment=settings.environment,
        version=settings.resolved_service_revision,
        service_version=settings.resolved_service_version,
        service_commit=settings.resolved_service_commit,
        service_revision=settings.resolved_service_revision,
        service_build=settings.resolved_service_build,
        service_image=settings.resolved_service_image,
        public_base_url=settings.public_base_url,
        instance_id=settings.instance_id,
        surfaces=["profiles", "sessions", "runs", "schedules", "heartbeat", "notifications"],
        workspace_provider_backend=settings.workspace_provider_backend,
        storage_model=_storage_model(settings.resolved_database_url),
        limits={
            "max_workspace_mounts_per_session": 8,
            "default_sandbox_idle_ttl_seconds": settings.resolved_workspace_provider_docker_idle_ttl_seconds,
        },
    )


@router.get("/notifications")
async def stream_notifications(
    request: Request,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
) -> EventSourceResponse:
    notification_hub = _get_notification_hub(request)
    return EventSourceResponse(notification_hub.stream(last_event_id=last_event_id))


def _storage_model(database_url: str) -> str:
    if database_url.startswith("sqlite"):
        return "sqlite"
    if database_url.startswith("postgresql"):
        return "postgresql"
    return "external"


def _get_settings(request: Request) -> ClawSettings:
    settings = request.app.state.settings
    if not isinstance(settings, ClawSettings):
        raise TypeError("Application settings are unavailable.")
    return settings


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
