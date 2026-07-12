from __future__ import annotations

import asyncio

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ya_claw.config import ClawSettings
from ya_claw.controller.session_prune import SessionPruneController, SessionPruneResult


class SessionPruneDispatcher:
    def __init__(
        self,
        *,
        settings: ClawSettings,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._settings = settings
        self._session_factory = session_factory
        self._controller = SessionPruneController()
        self._task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()

    async def startup(self) -> None:
        if not (self._settings.session_prune_enabled and self._settings.session_prune_generated_sessions_enabled):
            async with self._session_factory() as db_session:
                await self._controller.release_all_prune_claims(db_session)
        if not self._should_run():
            logger.info("Session prune dispatcher disabled")
            return
        if self._task is not None:
            return
        self._stopping.clear()
        self._task = asyncio.create_task(self._run_loop(), name="ya-claw-session-prune-dispatcher")
        logger.info(
            "Session prune dispatcher started interval_seconds={} startup_delay_seconds={} session_prune_enabled={} "
            "once_schedule_hide_after_days={}",
            self._settings.session_prune_interval_seconds,
            self._settings.session_prune_startup_delay_seconds,
            self._settings.session_prune_enabled,
            self._settings.session_prune_once_schedules_hide_after_days,
        )

    async def shutdown(self) -> None:
        self._stopping.set()
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        logger.info("Session prune dispatcher stopped")

    async def prune_once(self) -> SessionPruneResult:
        async with self._session_factory() as db_session:
            return await self._controller.prune_once(db_session, self._settings)

    def _should_run(self) -> bool:
        return self._settings.session_prune_enabled or self._settings.session_prune_once_schedules_hide_after_days > 0

    async def _run_loop(self) -> None:
        try:
            await asyncio.wait_for(
                self._stopping.wait(),
                timeout=max(self._settings.session_prune_startup_delay_seconds, 0),
            )
            return
        except TimeoutError:
            pass

        while not self._stopping.is_set():
            try:
                result = await self.prune_once()
                logger.info(
                    "Session prune completed pruned_run_store_dirs={} deleted_runs={} deleted_sessions={} "
                    "deleted_orphan_run_dirs={} deleted_schedule_fires={} deleted_heartbeat_fires={} "
                    "hidden_once_schedules={} reclaimed_bytes={} failed_paths={} failed_docker_sandboxes={}",
                    result.pruned_run_store_dirs,
                    result.deleted_runs,
                    result.deleted_sessions,
                    result.deleted_orphan_run_dirs,
                    result.deleted_schedule_fires,
                    result.deleted_heartbeat_fires,
                    result.hidden_once_schedules,
                    result.reclaimed_bytes,
                    len(result.failed_run_store_paths),
                    len(result.failed_docker_sandbox_session_ids),
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Session prune tick failed error={}", exc)
            try:
                await asyncio.wait_for(
                    self._stopping.wait(),
                    timeout=max(self._settings.session_prune_interval_seconds, 1),
                )
            except TimeoutError:
                continue
