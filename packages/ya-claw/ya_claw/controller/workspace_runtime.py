from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from ya_agent_sdk.environment import SandboxEnvironment

from ya_claw.config import ClawSettings
from ya_claw.execution.sandbox_ttl import _delete_cache_file, _stop_docker_container
from ya_claw.notifications import NotificationHub
from ya_claw.orm.tables import SessionRecord
from ya_claw.runtime_state import InMemoryRuntimeState
from ya_claw.workspace import EnvironmentFactory, WorkspaceProvider, build_workspace_sandbox_metadata
from ya_claw.workspace.runtime_models import (
    DockerContainerCacheStatus,
    DockerDaemonStatus,
    DockerImageStatus,
    DockerRuntimeStatus,
    DockerWorkspaceUserStatus,
    RuntimeCheck,
    SessionSandboxState,
    SessionWorkspaceState,
    WorkspacePathStatus,
    WorkspaceResolveResponse,
    WorkspaceRuntimeCapabilities,
    WorkspaceRuntimeStatus,
    WorkspaceRuntimeStatusValue,
    build_session_sandbox_state,
    build_session_sandbox_state_from_sandbox,
    build_workspace_binding_view,
    path_writable,
    session_sandbox_event_payload,
)


class WorkspaceRuntimeController:
    def get_runtime_status(self, settings: ClawSettings) -> WorkspaceRuntimeStatus:
        now = _utc_now_iso()
        workspace_path = settings.resolved_workspace_dir.expanduser()
        workspace_exists = workspace_path.exists()
        workspace_writable = path_writable(workspace_path)
        workspace_status = WorkspacePathStatus(
            service_path=str(workspace_path),
            docker_host_path=str(settings.resolved_workspace_provider_docker_host_workspace_dir)
            if settings.workspace_provider_backend == "docker"
            else None,
            virtual_path="/workspace" if settings.workspace_provider_backend == "docker" else str(workspace_path),
            exists=workspace_exists,
            writable=workspace_writable,
        )
        checks = [
            RuntimeCheck(
                id="workspace.path",
                status="ready" if workspace_exists and workspace_writable else "error",
                message="Workspace path is available." if workspace_writable else "Workspace path is unavailable.",
                details={"service_path": str(workspace_path)},
            )
        ]

        if settings.workspace_provider_backend == "docker":
            docker_status = _inspect_docker_runtime(settings)
            checks.extend(_docker_checks(docker_status))
            status = _runtime_status_from_checks(checks)
            return WorkspaceRuntimeStatus(
                backend="docker",
                status=status,
                execution_location="docker",
                workspace=workspace_status,
                capabilities=WorkspaceRuntimeCapabilities(
                    file_browse=True,
                    shell_access=True,
                    sandbox_prepare=docker_status.daemon.status == "ready",
                    sandbox_stop=docker_status.daemon.status == "ready",
                ),
                checks=checks,
                docker=docker_status,
                updated_at=now,
            )

        return WorkspaceRuntimeStatus(
            backend="local",
            status=_runtime_status_from_checks(checks),
            execution_location="local",
            workspace=workspace_status,
            capabilities=WorkspaceRuntimeCapabilities(
                file_browse=True,
                shell_access=True,
                sandbox_prepare=False,
                sandbox_stop=False,
            ),
            checks=checks,
            docker=None,
            updated_at=now,
        )

    def resolve_workspace(
        self,
        *,
        workspace_provider: WorkspaceProvider,
        metadata: dict[str, Any],
    ) -> WorkspaceResolveResponse:
        binding = workspace_provider.resolve(metadata)
        return WorkspaceResolveResponse(
            binding=build_workspace_binding_view(binding),
            sandbox_state=build_session_sandbox_state(binding.metadata),
        )

    async def get_session_workspace(
        self,
        *,
        db_session: AsyncSession,
        workspace_provider: WorkspaceProvider,
        session_id: str,
    ) -> SessionWorkspaceState:
        session_record = await _require_session(db_session, session_id)
        metadata = _session_metadata_with_id(session_record)
        binding = workspace_provider.resolve(metadata)
        return SessionWorkspaceState(
            binding=build_workspace_binding_view(binding),
            sandbox_state=build_session_sandbox_state(session_record.session_metadata),
        )

    async def get_session_sandbox(
        self,
        *,
        db_session: AsyncSession,
        session_id: str,
    ) -> SessionSandboxState:
        session_record = await _require_session(db_session, session_id)
        sandbox_state = build_session_sandbox_state(session_record.session_metadata)
        if sandbox_state is None:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' does not have sandbox state.")
        return sandbox_state

    async def prepare_session_sandbox(
        self,
        *,
        settings: ClawSettings,
        db_session: AsyncSession,
        workspace_provider: WorkspaceProvider,
        environment_factory: EnvironmentFactory,
        notification_hub: NotificationHub,
        runtime_state: InMemoryRuntimeState,
        session_id: str,
    ) -> SessionSandboxState:
        if settings.workspace_provider_backend != "docker":
            raise HTTPException(status_code=409, detail="Sandbox preparation requires a Docker workspace backend.")

        session_record = await _require_session(db_session, session_id)
        metadata = _session_metadata_with_id(session_record)
        binding = workspace_provider.resolve(metadata)
        environment = environment_factory.build(binding)
        preparing_state = await self._persist_sandbox_metadata(
            db_session=db_session,
            session_record=session_record,
            sandbox_metadata={
                **_binding_sandbox_metadata(binding.metadata),
                "provider": "docker",
                "scope": "session",
                "status": "preparing",
                "ready_state": "starting",
                "last_used_at": _utc_now_iso(),
                "retention_policy": settings.resolved_workspace_provider_docker_retention_policy,
                "idle_ttl_seconds": settings.resolved_workspace_provider_docker_idle_ttl_seconds,
            },
        )
        await _publish_sandbox_update(
            notification_hub=notification_hub,
            runtime_state=runtime_state,
            session_id=session_id,
            sandbox_state=preparing_state,
        )

        try:
            async with environment:
                if isinstance(environment, SandboxEnvironment):
                    await environment.ensure_ready_shell()
                sandbox_metadata = build_workspace_sandbox_metadata(binding=binding, environment=environment)
                if sandbox_metadata is None:
                    raise HTTPException(status_code=409, detail="Workspace environment does not expose sandbox state.")
                now = _utc_now_iso()
                sandbox_metadata["status"] = "ready"
                sandbox_metadata["ready_state"] = "ready"
                sandbox_metadata["last_used_at"] = now
                sandbox_metadata.setdefault("last_started_at", now)
                sandbox_state = await self._persist_sandbox_metadata(
                    db_session=db_session,
                    session_record=session_record,
                    sandbox_metadata=sandbox_metadata,
                )
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("Failed to prepare session sandbox session_id={} error={}", session_id, exc)
            failed_state = await self._persist_sandbox_metadata(
                db_session=db_session,
                session_record=session_record,
                sandbox_metadata={
                    **_binding_sandbox_metadata(binding.metadata),
                    "provider": "docker",
                    "scope": "session",
                    "status": "failed",
                    "ready_state": "failed",
                    "error_message": str(exc),
                    "last_used_at": _utc_now_iso(),
                    "retention_policy": settings.resolved_workspace_provider_docker_retention_policy,
                    "idle_ttl_seconds": settings.resolved_workspace_provider_docker_idle_ttl_seconds,
                },
            )
            await _publish_sandbox_update(
                notification_hub=notification_hub,
                runtime_state=runtime_state,
                session_id=session_id,
                sandbox_state=failed_state,
            )
            raise HTTPException(status_code=500, detail=f"Failed to prepare sandbox: {exc}") from exc

        await _publish_sandbox_update(
            notification_hub=notification_hub,
            runtime_state=runtime_state,
            session_id=session_id,
            sandbox_state=sandbox_state,
        )
        return sandbox_state

    async def stop_session_sandbox(
        self,
        *,
        settings: ClawSettings,
        db_session: AsyncSession,
        notification_hub: NotificationHub,
        runtime_state: InMemoryRuntimeState,
        session_id: str,
    ) -> SessionSandboxState:
        if settings.workspace_provider_backend != "docker":
            raise HTTPException(status_code=409, detail="Sandbox stop requires a Docker workspace backend.")

        session_record = await _require_session(db_session, session_id)
        metadata = session_record.session_metadata if isinstance(session_record.session_metadata, dict) else {}
        sandbox = metadata.get("sandbox")
        if not isinstance(sandbox, dict):
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' does not have sandbox state.")

        container_id = _normalize_string(sandbox.get("verified_container_id")) or _normalize_string(
            sandbox.get("container_id")
        )
        cache_path = _normalize_path(sandbox.get("cache_path"))
        if container_id is not None:
            await _stop_docker_container(container_id)
        await _delete_cache_file(cache_path)

        next_sandbox = {
            **sandbox,
            "status": "stopped",
            "ready_state": "not_started",
            "container_id": None,
            "verified_container_id": None,
            "last_used_at": _utc_now_iso(),
            "error_message": None,
        }
        sandbox_state = await self._persist_sandbox_metadata(
            db_session=db_session,
            session_record=session_record,
            sandbox_metadata=next_sandbox,
        )
        await _publish_sandbox_update(
            notification_hub=notification_hub,
            runtime_state=runtime_state,
            session_id=session_id,
            sandbox_state=sandbox_state,
        )
        return sandbox_state

    async def _persist_sandbox_metadata(
        self,
        *,
        db_session: AsyncSession,
        session_record: SessionRecord,
        sandbox_metadata: dict[str, Any],
    ) -> SessionSandboxState:
        metadata = dict(session_record.session_metadata) if isinstance(session_record.session_metadata, dict) else {}
        metadata["sandbox"] = sandbox_metadata
        session_record.session_metadata = metadata
        await db_session.commit()
        await db_session.refresh(session_record)
        return build_session_sandbox_state_from_sandbox(sandbox_metadata)


async def _require_session(db_session: AsyncSession, session_id: str) -> SessionRecord:
    session_record = await db_session.get(SessionRecord, session_id)
    if not isinstance(session_record, SessionRecord):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' was not found.")
    return session_record


def _session_metadata_with_id(session_record: SessionRecord) -> dict[str, Any]:
    metadata = dict(session_record.session_metadata) if isinstance(session_record.session_metadata, dict) else {}
    metadata.setdefault("session_id", session_record.id)
    return metadata


def _binding_sandbox_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    sandbox = metadata.get("sandbox")
    return dict(sandbox) if isinstance(sandbox, dict) else {}


async def _publish_sandbox_update(
    *,
    notification_hub: NotificationHub,
    runtime_state: InMemoryRuntimeState,
    session_id: str,
    sandbox_state: SessionSandboxState,
) -> None:
    handle = runtime_state.get_session_run_handle(session_id)
    run_id = handle.run_id if handle is not None else None
    payload = session_sandbox_event_payload(session_id=session_id, run_id=run_id, sandbox_state=sandbox_state)
    await notification_hub.publish("workspace.sandbox.updated", payload)
    if run_id is not None:
        try:
            await runtime_state.append_run_event(run_id, payload, replay=False)
        except KeyError:
            return


def _inspect_docker_runtime(settings: ClawSettings) -> DockerRuntimeStatus:
    daemon = DockerDaemonStatus(status="checking")
    image = DockerImageStatus(ref=settings.workspace_provider_docker_image)
    client: Any | None = None
    try:
        import docker

        client = docker.from_env()
        version_payload = client.version()
        server_version = version_payload.get("Version") if isinstance(version_payload, dict) else None
        daemon = DockerDaemonStatus(
            status="ready",
            server_version=server_version if isinstance(server_version, str) else None,
        )
        image_obj = client.images.get(settings.workspace_provider_docker_image)
        image = DockerImageStatus(
            ref=settings.workspace_provider_docker_image,
            present=True,
            digest=_image_digest(image_obj),
        )
    except Exception as exc:
        message = str(exc)
        if daemon.status != "ready":
            daemon = DockerDaemonStatus(status="error", error_message=message)
        else:
            image = DockerImageStatus(
                ref=settings.workspace_provider_docker_image, present=False, error_message=message
            )
    finally:
        if client is not None:
            client.close()

    return DockerRuntimeStatus(
        daemon=daemon,
        image=image,
        workspace_user=DockerWorkspaceUserStatus(
            uid=settings.resolved_workspace_provider_docker_uid,
            gid=settings.resolved_workspace_provider_docker_gid,
            exec_user=settings.resolved_workspace_provider_docker_exec_user,
        ),
        container_cache=DockerContainerCacheStatus(
            enabled=True,
            cache_dir=str(settings.resolved_workspace_provider_docker_container_cache_dir),
        ),
        retention_policy=settings.resolved_workspace_provider_docker_retention_policy,
        idle_ttl_seconds=settings.resolved_workspace_provider_docker_idle_ttl_seconds,
    )


def _docker_checks(docker_status: DockerRuntimeStatus) -> list[RuntimeCheck]:
    return [
        RuntimeCheck(
            id="docker.daemon",
            status=docker_status.daemon.status,
            message="Docker daemon is available."
            if docker_status.daemon.status == "ready"
            else "Docker daemon is unavailable.",
            details={
                "server_version": docker_status.daemon.server_version,
                "error": docker_status.daemon.error_message,
            },
        ),
        RuntimeCheck(
            id="docker.image",
            status="ready" if docker_status.image.present else "error",
            message="Docker workspace image is present."
            if docker_status.image.present
            else "Docker workspace image is unavailable.",
            details={
                "image": docker_status.image.ref,
                "digest": docker_status.image.digest,
                "error": docker_status.image.error_message,
            },
        ),
    ]


def _runtime_status_from_checks(checks: list[RuntimeCheck]) -> WorkspaceRuntimeStatusValue:
    if any(check.status == "error" for check in checks):
        return "unavailable"
    if any(check.status == "warning" for check in checks):
        return "degraded"
    if any(check.status == "checking" for check in checks):
        return "checking"
    return "ready"


def _image_digest(image_obj: Any) -> str | None:
    attrs = image_obj.attrs if isinstance(image_obj.attrs, dict) else {}
    repo_digests = attrs.get("RepoDigests")
    if isinstance(repo_digests, list):
        for digest in repo_digests:
            if isinstance(digest, str) and digest.strip():
                return digest.strip()
    image_id = image_obj.id if isinstance(image_obj.id, str) else None
    if image_id is not None and image_id.strip():
        return image_id.strip()
    short_id = image_obj.short_id if isinstance(image_obj.short_id, str) else None
    return short_id.strip() if short_id is not None and short_id.strip() else None


def _normalize_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _normalize_path(value: Any) -> Path | None:
    normalized = _normalize_string(value)
    return Path(normalized).expanduser() if normalized is not None else None


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
