from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ya_claw.config import ClawSettings
from ya_claw.orm.tables import SessionRecord
from ya_claw.workspace.models import SANDBOX_SCOPE_SESSION
from ya_claw.workspace.provider import get_docker_container_lock


class DockerSandboxTtlDispatcher:
    def __init__(
        self,
        *,
        settings: ClawSettings,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._settings = settings
        self._session_factory = session_factory
        self._task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()

    async def startup(self) -> None:
        if self._settings.workspace_provider_backend != "docker":
            logger.info("Docker sandbox TTL dispatcher disabled backend={}", self._settings.workspace_provider_backend)
            return
        if self._task is not None:
            return
        self._stopping.clear()
        self._task = asyncio.create_task(self._run_loop(), name="ya-claw-docker-sandbox-ttl-dispatcher")
        logger.info(
            "Docker sandbox TTL dispatcher started interval_seconds={}",
            self._interval_seconds,
        )

    async def shutdown(self) -> None:
        self._stopping.set()
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        logger.info("Docker sandbox TTL dispatcher stopped")

    async def cleanup_once(self) -> int:
        async with self._session_factory() as db_session:
            statement = select(SessionRecord)
            result = await db_session.execute(statement)
            stopped = 0
            for session_record in result.scalars().all():
                metadata = session_record.session_metadata if isinstance(session_record.session_metadata, dict) else {}
                sandbox = metadata.get("sandbox")
                if not isinstance(sandbox, dict):
                    continue
                if not _is_expired_stop_on_idle_sandbox(sandbox):
                    continue
                container_id = _normalize_string(sandbox.get("container_id"))
                cache_path = _normalize_path(sandbox.get("cache_path"))
                container_ref = _normalize_string(sandbox.get("container_ref")) or container_id or session_record.id
                lock = get_docker_container_lock(cache_path=cache_path, container_ref=container_ref)
                async with lock:
                    stop_succeeded = True
                    if container_id is not None:
                        stop_succeeded = await _stop_docker_container(container_id)
                    if stop_succeeded:
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
                        stopped += 1
                    else:
                        next_sandbox = {
                            **sandbox,
                            "status": sandbox.get("status") or "running",
                            "error_message": f"Failed to stop idle Docker workspace container: {container_id}",
                            "last_stop_attempt_at": _utc_now_iso(),
                        }
                    session_record.session_metadata = {**metadata, "sandbox": next_sandbox}
            await db_session.commit()
            return stopped

    async def _run_loop(self) -> None:
        while not self._stopping.is_set():
            try:
                stopped = await self.cleanup_once()
                if stopped:
                    logger.info("Docker sandbox TTL cleanup stopped_count={}", stopped)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Docker sandbox TTL cleanup tick failed error={}", exc)
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=self._interval_seconds)
            except TimeoutError:
                continue

    @property
    def _interval_seconds(self) -> int:
        ttl = self._settings.resolved_workspace_provider_docker_idle_ttl_seconds
        return max(60, min(600, ttl // 2))


def _is_expired_stop_on_idle_sandbox(sandbox: dict[str, Any]) -> bool:
    if sandbox.get("provider") != "docker":
        return False
    if sandbox.get("scope") != SANDBOX_SCOPE_SESSION:
        return False
    if sandbox.get("retention_policy") != "stop_on_idle":
        return False
    if _normalize_string(sandbox.get("container_id")) is None:
        return False
    last_used_at = _parse_datetime(sandbox.get("last_used_at"))
    if last_used_at is None:
        return False
    ttl_seconds = _normalize_positive_int(sandbox.get("idle_ttl_seconds")) or 3600
    return datetime.now(UTC) >= last_used_at + timedelta(seconds=ttl_seconds)


async def _delete_cache_file(cache_path: Path | None) -> None:
    if cache_path is None:
        return

    def _delete() -> None:
        with contextlib.suppress(FileNotFoundError):
            cache_path.unlink()
        with contextlib.suppress(OSError):
            cache_path.parent.rmdir()

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _delete)


async def _stop_docker_container(container_id: str) -> bool:
    def _stop() -> bool:
        try:
            import docker

            client = docker.from_env()
            try:
                container = client.containers.get(container_id)
                container.stop(timeout=10)
                return True
            except Exception as exc:
                if exc.__class__.__name__ == "NotFound":
                    return True
                raise
            finally:
                client.close()
        except Exception as exc:
            logger.warning("Failed to stop idle Docker workspace container id={} error={}", container_id, exc)
            return False

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _stop)


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


def _normalize_path(value: Any) -> Path | None:
    normalized = _normalize_string(value)
    return Path(normalized).expanduser() if normalized is not None else None


def _normalize_string(value: Any) -> str | None:
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


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
