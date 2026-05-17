from __future__ import annotations

import asyncio

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ya_claw.agency.lifecycle import AgencyLifecycle, AgencyTickResult
from ya_claw.config import ClawSettings
from ya_claw.controller.models import DispatchMode
from ya_claw.execution.dispatcher import RunDispatcher
from ya_claw.notifications import NotificationHub
from ya_claw.runtime_state import InMemoryRuntimeState


class AgencyDispatcher:
    def __init__(
        self,
        *,
        settings: ClawSettings,
        session_factory: async_sessionmaker[AsyncSession],
        runtime_state: InMemoryRuntimeState,
        run_dispatcher: RunDispatcher,
        notification_hub: NotificationHub | None = None,
    ) -> None:
        self._settings = settings
        self._session_factory = session_factory
        self._runtime_state = runtime_state
        self._run_dispatcher = run_dispatcher
        self._notification_hub = notification_hub
        self._task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()

    async def startup(self) -> None:
        if not self._settings.agency_enabled:
            logger.info("Agency dispatcher disabled")
            return
        await self.bootstrap()
        if self._task is not None:
            return
        self._stopping.clear()
        self._task = asyncio.create_task(self._run_loop(), name="ya-claw-agency-dispatcher")
        logger.info("Agency dispatcher started")

    async def bootstrap(self) -> None:
        lifecycle = AgencyLifecycle(settings=self._settings, runtime_state=self._runtime_state)
        async with self._session_factory() as db_session:
            agency_session = await lifecycle.ensure_agency_session(db_session)
            await db_session.commit()
        logger.info("Agency default coordinator ready agency_session_id={}", agency_session.id)

    async def shutdown(self) -> None:
        self._stopping.set()
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        logger.info("Agency dispatcher stopped")

    async def dispatch_once(self) -> AgencyTickResult:
        lifecycle = AgencyLifecycle(
            settings=self._settings,
            runtime_state=self._runtime_state,
            submit_run=lambda run_id: self._run_dispatcher.dispatch(run_id, DispatchMode.ASYNC).submitted,
        )
        async with self._session_factory() as db_session:
            result = await lifecycle.tick(db_session)
        if self._notification_hub is not None:
            for fire_id in result.created_fire_ids:
                await self._notification_hub.publish("agency.fire.updated", {"agency_fire_id": fire_id})
            for run_id in result.submitted_run_ids:
                await self._notification_hub.publish("agency.episode.updated", {"run_id": run_id})
        return result

    async def _run_loop(self) -> None:
        while not self._stopping.is_set():
            try:
                result = await self.dispatch_once()
                total = (
                    len(result.created_fire_ids)
                    + len(result.submitted_run_ids)
                    + len(result.steered_fire_ids)
                    + len(result.merged_fire_ids)
                )
                if total:
                    logger.info(
                        "Agency dispatcher tick processed created_fires={} submitted_runs={} steered_fires={} merged_fires={}",
                        len(result.created_fire_ids),
                        len(result.submitted_run_ids),
                        len(result.steered_fire_ids),
                        len(result.merged_fire_ids),
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Agency dispatcher tick failed error={}", exc)
            try:
                await asyncio.wait_for(
                    self._stopping.wait(),
                    timeout=max(self._settings.agency_tick_seconds, 1),
                )
            except TimeoutError:
                continue
