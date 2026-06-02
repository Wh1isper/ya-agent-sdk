from __future__ import annotations

import asyncio

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ya_claw.config import ClawSettings
from ya_claw.controller.workflow import WorkflowController
from ya_claw.execution.dispatcher import RunDispatcher
from ya_claw.notifications import NotificationHub
from ya_claw.orm.tables import WorkflowRunRecord
from ya_claw.runtime_state import InMemoryRuntimeState


class WorkflowExecutor:
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
        self._controller = WorkflowController()
        self._task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()

    async def startup(self) -> None:
        if not self._settings.workflow_dispatch_enabled:
            logger.info("Workflow executor disabled")
            return
        if self._task is not None:
            return
        self._stopping.clear()
        self._task = asyncio.create_task(self._run_loop(), name="ya-claw-workflow-executor")
        logger.info("Workflow executor started")

    async def shutdown(self) -> None:
        self._stopping.set()
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        logger.info("Workflow executor stopped")

    async def dispatch_once(self) -> int:
        async with self._session_factory() as db_session:
            statement = (
                select(WorkflowRunRecord)
                .where(WorkflowRunRecord.status.in_(("queued", "running", "waiting")))
                .order_by(WorkflowRunRecord.created_at.asc(), WorkflowRunRecord.id.asc())
                .limit(max(self._settings.workflow_max_runs_per_tick, 1))
            )
            try:
                result = await db_session.execute(statement)
            except Exception as exc:
                if "workflow_runs" in str(exc) and "no such table" in str(exc):
                    logger.warning("Workflow tables are unavailable; skipping workflow executor tick.")
                    return 0
                raise
            records = list(result.scalars().all())
            processed = 0
            for record in records:
                try:
                    await self._controller.execute_once(
                        db_session,
                        self._settings,
                        self._runtime_state,
                        self._run_dispatcher,
                        record,
                    )
                    processed += 1
                except Exception as exc:
                    logger.exception("Workflow execution tick failed workflow_run_id={} error={}", record.id, exc)
                    record.status = "failed"
                    record.error_message = str(exc)
                await db_session.commit()
                await self._publish_run(record)
            return processed

    async def _run_loop(self) -> None:
        while not self._stopping.is_set():
            try:
                count = await self.dispatch_once()
                if count:
                    logger.info("Workflow executor processed runs count={}", count)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Workflow executor tick failed error={}", exc)
            try:
                await asyncio.wait_for(
                    self._stopping.wait(),
                    timeout=max(self._settings.workflow_tick_seconds, 1),
                )
            except TimeoutError:
                continue

    async def _publish_run(self, record: WorkflowRunRecord) -> None:
        if self._notification_hub is None:
            return
        await self._notification_hub.publish(
            "workflow.run.updated",
            {
                "workflow_id": record.workflow_id,
                "workflow_run_id": record.id,
                "status": record.status,
                "current_node_ids": list(record.current_node_ids or []),
                "error_message": record.error_message,
            },
        )
