from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ya_claw.workspace.models import WorkspaceBinding

WorkspaceRuntimeBackend = Literal["local", "docker", "remote", "cloud", "unknown"]
WorkspaceRuntimeStatusValue = Literal["ready", "degraded", "unavailable", "checking"]
RuntimeCheckStatus = Literal["ready", "warning", "error", "checking", "skipped"]
SessionSandboxStatus = Literal["created", "mounted", "preparing", "ready", "failed", "stopped"]
SessionSandboxReadyState = Literal["not_started", "starting", "ready", "failed"]


class RuntimeCheck(BaseModel):
    id: str
    status: RuntimeCheckStatus
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class WorkspacePathStatus(BaseModel):
    service_path: str | None = None
    docker_host_path: str | None = None
    virtual_path: str | None = None
    exists: bool = False
    writable: bool = False


class WorkspaceRuntimeCapabilities(BaseModel):
    file_browse: bool = True
    shell_access: bool = Field(default=True, serialization_alias="shell", validation_alias="shell")
    sandbox_prepare: bool = False
    sandbox_stop: bool = False


class DockerDaemonStatus(BaseModel):
    status: RuntimeCheckStatus
    server_version: str | None = None
    error_message: str | None = None


class DockerImageStatus(BaseModel):
    ref: str
    present: bool = False
    digest: str | None = None
    error_message: str | None = None


class DockerWorkspaceUserStatus(BaseModel):
    uid: int | None = None
    gid: int | None = None
    exec_user: str | None = None


class DockerContainerCacheStatus(BaseModel):
    enabled: bool = False
    cache_dir: str | None = None


class DockerRuntimeStatus(BaseModel):
    daemon: DockerDaemonStatus
    image: DockerImageStatus
    workspace_user: DockerWorkspaceUserStatus
    container_cache: DockerContainerCacheStatus
    retention_policy: str | None = None
    idle_ttl_seconds: int | None = None


class WorkspaceRuntimeStatus(BaseModel):
    backend: WorkspaceRuntimeBackend
    status: WorkspaceRuntimeStatusValue
    execution_location: str
    workspace: WorkspacePathStatus
    capabilities: WorkspaceRuntimeCapabilities
    checks: list[RuntimeCheck] = Field(default_factory=list)
    docker: DockerRuntimeStatus | None = None
    updated_at: str


class WorkspaceMountView(BaseModel):
    id: str | None = None
    name: str | None = None
    host_path: str
    docker_host_path: str | None = None
    virtual_path: str
    mode: str = "rw"


class WorkspaceBindingView(BaseModel):
    provider: str
    backend_hint: str | None = None
    host_path: str
    docker_host_path: str | None = None
    virtual_path: str
    cwd: str
    readable_paths: list[str] = Field(default_factory=list)
    writable_paths: list[str] = Field(default_factory=list)
    fingerprint: str
    generation: int | None = None
    sandbox_scope: str | None = None
    mounts: list[WorkspaceMountView] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionSandboxState(BaseModel):
    backend: str | None = None
    scope: str | None = None
    status: SessionSandboxStatus = "created"
    ready_state: SessionSandboxReadyState = "not_started"
    container_ref: str | None = None
    container_id: str | None = None
    verified_container_id: str | None = None
    image: str | None = None
    image_digest: str | None = None
    work_dir: str | None = None
    retention_policy: str | None = None
    idle_ttl_seconds: int | None = None
    ttl_seconds_remaining: int | None = None
    expires_at: str | None = None
    last_used_at: str | None = None
    last_started_at: str | None = None
    error_message: str | None = None
    updated_at: str


class SessionWorkspaceState(BaseModel):
    binding: WorkspaceBindingView | None = None
    sandbox_state: SessionSandboxState | None = None


class WorkspaceResolveRequest(BaseModel):
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkspaceResolveResponse(BaseModel):
    binding: WorkspaceBindingView
    sandbox_state: SessionSandboxState | None = None


def build_session_workspace_state(metadata: dict[str, Any] | None) -> SessionWorkspaceState | None:
    sandbox_state = build_session_sandbox_state(metadata)
    if sandbox_state is None:
        return None
    return SessionWorkspaceState(sandbox_state=sandbox_state)


def build_workspace_binding_view(binding: WorkspaceBinding) -> WorkspaceBindingView:
    return WorkspaceBindingView(
        provider=str(binding.metadata.get("provider") or binding.backend_hint or "unknown"),
        backend_hint=binding.backend_hint,
        host_path=str(binding.host_path),
        docker_host_path=str(binding.docker_host_path) if binding.docker_host_path is not None else None,
        virtual_path=str(binding.virtual_path),
        cwd=str(binding.cwd),
        readable_paths=[str(path) for path in binding.readable_paths],
        writable_paths=[str(path) for path in binding.writable_paths],
        fingerprint=binding.fingerprint,
        generation=binding.generation,
        sandbox_scope=binding.sandbox_scope,
        mounts=[
            WorkspaceMountView(
                id=mount.id,
                name=mount.name,
                host_path=str(mount.host_path),
                docker_host_path=str(mount.docker_host_path) if mount.docker_host_path is not None else None,
                virtual_path=str(mount.virtual_path),
                mode=mount.mode,
            )
            for mount in binding.mounts
        ],
        metadata=dict(binding.metadata),
    )


def build_session_sandbox_state(
    metadata: dict[str, Any] | None, *, now: datetime | None = None
) -> SessionSandboxState | None:
    normalized_metadata = dict(metadata or {})
    sandbox = normalized_metadata.get("sandbox")
    if not isinstance(sandbox, dict):
        return None
    return build_session_sandbox_state_from_sandbox(sandbox, now=now)


def build_session_sandbox_state_from_sandbox(
    sandbox: dict[str, Any],
    *,
    now: datetime | None = None,
) -> SessionSandboxState:
    current_time = now or datetime.now(UTC)
    ready_state = _normalize_ready_state(sandbox.get("ready_state"))
    status = _normalize_sandbox_status(sandbox.get("status"), ready_state=ready_state)
    last_used_at = _normalize_optional_str(sandbox.get("last_used_at"))
    idle_ttl_seconds = _normalize_positive_int(sandbox.get("idle_ttl_seconds"))
    expires_at = None
    ttl_seconds_remaining = None
    parsed_last_used_at = _parse_datetime(last_used_at)
    if parsed_last_used_at is not None and idle_ttl_seconds is not None:
        expires_at_dt = parsed_last_used_at + timedelta(seconds=idle_ttl_seconds)
        expires_at = _isoformat_utc(expires_at_dt)
        ttl_seconds_remaining = max(0, int((expires_at_dt - current_time).total_seconds()))
        if status == "stopped":
            ttl_seconds_remaining = 0

    return SessionSandboxState(
        backend=_normalize_optional_str(sandbox.get("provider")),
        scope=_normalize_optional_str(sandbox.get("scope")),
        status=status,
        ready_state=ready_state,
        container_ref=_normalize_optional_str(sandbox.get("container_ref")),
        container_id=_normalize_optional_str(sandbox.get("container_id")),
        verified_container_id=_normalize_optional_str(sandbox.get("verified_container_id")),
        image=_normalize_optional_str(sandbox.get("image")),
        image_digest=_normalize_optional_str(sandbox.get("image_digest")),
        work_dir=_normalize_optional_str(sandbox.get("cwd")) or _normalize_optional_str(sandbox.get("work_dir")),
        retention_policy=_normalize_optional_str(sandbox.get("retention_policy")),
        idle_ttl_seconds=idle_ttl_seconds,
        ttl_seconds_remaining=ttl_seconds_remaining,
        expires_at=expires_at,
        last_used_at=last_used_at,
        last_started_at=_normalize_optional_str(sandbox.get("last_started_at")),
        error_message=_normalize_optional_str(sandbox.get("error_message")),
        updated_at=_isoformat_utc(current_time),
    )


def session_sandbox_event_payload(
    *,
    session_id: str,
    run_id: str | None = None,
    sandbox_state: SessionSandboxState,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "workspace.sandbox.updated",
        "session_id": session_id,
        "sandbox_state": sandbox_state.model_dump(mode="json"),
    }
    if run_id is not None:
        payload["run_id"] = run_id
    return payload


def _normalize_sandbox_status(value: Any, *, ready_state: SessionSandboxReadyState) -> SessionSandboxStatus:
    if value in {"created", "mounted", "preparing", "ready", "failed", "stopped"}:
        return value
    if ready_state == "ready":
        return "ready"
    if ready_state == "starting":
        return "preparing"
    if ready_state == "failed":
        return "failed"
    return "created"


def _normalize_ready_state(value: Any) -> SessionSandboxReadyState:
    if value in {"not_started", "starting", "ready", "failed"}:
        return value
    return "not_started"


def _normalize_optional_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _normalize_positive_int(value: Any) -> int | None:
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and value.strip().isdigit():
        normalized = int(value.strip())
        return normalized if normalized > 0 else None
    return None


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or value.strip() == "":
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _isoformat_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def path_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        test_file = path / ".ya-claw-write-test"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink()
        return True
    except Exception:
        return False
