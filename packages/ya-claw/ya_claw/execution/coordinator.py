from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

from loguru import logger
from pydantic import BaseModel
from pydantic_ai import DeferredToolRequests, DeferredToolResults
from pydantic_ai.messages import EnqueuedMessagesEvent, ModelMessage, ModelMessagesTypeAdapter
from pydantic_ai.tools import ToolDenied
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from ya_agent_environment import Environment
from ya_agent_sdk.agents.main import AgentInterrupted, AgentRuntime
from ya_agent_sdk.context import ResumableState, StreamEvent
from ya_agent_sdk.environment import SandboxEnvironment
from ya_agent_sdk.events import ModelRequestCompleteEvent, ModelRequestStartEvent, UsageSnapshotEvent
from ya_agent_sdk.execution import AgentExecutionHarness, AgentSegment, AgentSegmentRequest
from ya_agent_sdk.inputs import InputDisposition, InputOrigin
from ya_agent_sdk.interactions import (
    DeferredApprovalResolution,
    DeferredCallResolution,
    DeferredInteractionResolution,
    DeferredInteractionResolver,
)
from ya_agent_sdk.subagents import SubagentPlanDescriptor
from ya_agent_sdk.usage import UsageSnapshot, combine_usage_snapshots
from ya_agent_stream_protocol.sdk import AguiAdapterConfig, AguiEventAdapter

from ya_claw.agency.lifecycle import AgencyLifecycle
from ya_claw.config import ClawSettings
from ya_claw.context import ClawAgentContext
from ya_claw.controller.async_task import AsyncTaskController
from ya_claw.controller.hitl import HitlController
from ya_claw.controller.models import (
    ActiveInteraction,
    InputPart,
    RunStatus,
    SessionStatusReason,
    TriggerType,
    parse_input_parts,
)
from ya_claw.controller.session_lifecycle import lock_session_reference
from ya_claw.controller.store import extract_usage_snapshot_from_state, with_usage_snapshot_metadata
from ya_claw.execution.checkpoint import build_message_checkpoint, commit_run_artifacts, write_message_checkpoint
from ya_claw.execution.input import InputMappingResult, map_input_parts
from ya_claw.execution.input_inbox import (
    deliver_accepted_run_inputs,
    lock_run_record,
    mark_run_input_applied,
    reject_open_run_inputs,
)
from ya_claw.execution.lifecycle_projection import CompletedRunProjector
from ya_claw.execution.profile import (
    PROFILE_SNAPSHOT_METADATA_KEY,
    ProfileResolver,
    ResolvedProfile,
    resolved_profile_from_descriptor,
    resolved_profile_from_subagent_plan,
)
from ya_claw.execution.restore import ResolvedRestorePoint, load_restore_point
from ya_claw.execution.resumable_state import load_resumable_state
from ya_claw.execution.runtime import ClawRuntimeBuilder
from ya_claw.execution.state_machine import complete_run, fail_run, interrupt_run, mark_run_running
from ya_claw.execution.store import RunStore
from ya_claw.execution.subagents import restore_claw_subagent_plan
from ya_claw.hitl import build_active_interactions
from ya_claw.json_types import JsonValue
from ya_claw.memory.lifecycle import MemoryLifecycle
from ya_claw.notifications import NotificationHub
from ya_claw.orm.tables import RunInputInboxRecord, RunRecord, SessionAsyncTaskRecord, SessionRecord
from ya_claw.runtime_state import InMemoryRuntimeState
from ya_claw.toolsets.session import CLAW_SELF_CLIENT_KEY, ClawSelfClient
from ya_claw.workspace import (
    EnvironmentFactory,
    WorkspaceBinding,
    WorkspaceProvider,
    build_workspace_sandbox_metadata,
)
from ya_claw.workspace.docker_lifecycle import build_docker_container_ref
from ya_claw.workspace.models import (
    SANDBOX_SCOPE_RUN,
    SANDBOX_SCOPE_SESSION,
    WORKSPACE_SNAPSHOT_METADATA_KEY,
    SandboxScopeLiteral,
    merge_workspace_metadata,
    workspace_snapshot,
)
from ya_claw.workspace.runtime_models import build_session_sandbox_state_from_sandbox, session_sandbox_event_payload

CLAW_AGUI_ADAPTER_CONFIG = AguiAdapterConfig(run_event_prefix="ya_claw")
_ASYNC_DELIVERY_RECOVERY_INTERVAL_SECONDS = 1.0


@dataclass(slots=True)
class ExecutionBuffers:
    latest_state_payload: dict[str, Any] | None = None
    latest_message_payload: dict[str, Any] | None = None
    latest_usage_snapshot: UsageSnapshot | None = None
    terminal_event: dict[str, Any] | None = None
    output_text: str | None = None
    output_json: JsonValue = None
    claw_metadata: dict[str, Any] = field(default_factory=dict)
    success_committed: bool = False
    terminal_event_emitted: bool = False
    clear_runtime_handle: bool = False


class _AgentRunMessages(Protocol):
    def all_messages(self) -> list[ModelMessage]: ...


_TERMINAL_RUN_STATUSES = frozenset({"completed", "failed", "cancelled"})
_ACTIVE_ASYNC_TASK_STATUSES = frozenset({"queued", "running"})


def _index_run_usage_snapshot(run_record: RunRecord, buffers: ExecutionBuffers) -> None:
    usage_snapshot = buffers.latest_usage_snapshot
    if usage_snapshot is None:
        usage_snapshot = extract_usage_snapshot_from_state(buffers.latest_state_payload)
    if usage_snapshot is not None:
        run_record.run_metadata = with_usage_snapshot_metadata(run_record.run_metadata, usage_snapshot)


def _with_cumulative_usage_snapshot(
    stream_event: StreamEvent,
    *,
    segment_base: UsageSnapshot | None,
    run_id: str,
) -> tuple[StreamEvent, UsageSnapshot | None]:
    event = stream_event.event
    if not isinstance(event, UsageSnapshotEvent) or event.snapshot is None:
        return stream_event, None
    cumulative = combine_usage_snapshots(segment_base, event.snapshot, run_id=run_id)
    return replace(stream_event, event=replace(event, snapshot=cumulative)), cumulative


class ExecutionSupervisor:
    def __init__(
        self,
        *,
        settings: ClawSettings,
        session_factory: async_sessionmaker[AsyncSession],
        runtime_state: InMemoryRuntimeState,
        workspace_provider: WorkspaceProvider,
        environment_factory: EnvironmentFactory,
        profile_resolver: ProfileResolver,
        runtime_builder: ClawRuntimeBuilder,
        notification_hub: NotificationHub | None = None,
        execution_harness: AgentExecutionHarness | None = None,
    ) -> None:
        self._settings = settings
        self._session_factory = session_factory
        self._runtime_state = runtime_state
        self._workspace_provider = workspace_provider
        self._environment_factory = environment_factory
        self._profile_resolver = profile_resolver
        self._runtime_builder = runtime_builder
        self._notification_hub = notification_hub
        self._execution_harness = execution_harness or AgentExecutionHarness()
        self._run_store = RunStore(settings)
        self._hitl_controller = HitlController()
        self._accepting_submissions = True
        self._delivery_recovery_task: asyncio.Task[None] | None = None

    @property
    def accepting_submissions(self) -> bool:
        return self._accepting_submissions

    def get_background_task(self, run_id: str) -> asyncio.Task[None] | None:
        return self._runtime_state.get_background_task(run_id)

    def submit_run(self, run_id: str) -> bool:
        if not self._accepting_submissions:
            logger.info("Run submission skipped run_id={} reason=supervisor_shutting_down", run_id)
            return False
        if self._runtime_state.get_background_task(run_id) is not None:
            logger.debug("Run submission skipped run_id={} reason=background_task_exists", run_id)
            return False
        logger.info("Submitting run to execution supervisor run_id={}", run_id)
        task = asyncio.create_task(self._claim_and_execute(run_id), name=f"ya-claw-supervisor-{run_id}")
        self._runtime_state.register_background_task(run_id, task)
        return True

    def schedule_run(self, run_id: str) -> bool:
        return self.submit_run(run_id)

    async def startup(self) -> dict[str, list[str]]:
        """Recover durable work and start supervised async-delivery retries."""
        recovery = await self.startup_recover()
        if self._delivery_recovery_task is None or self._delivery_recovery_task.done():
            self._delivery_recovery_task = asyncio.create_task(
                self._delivery_recovery_loop(),
                name="ya-claw-async-delivery-recovery",
            )
        return recovery

    async def _delivery_recovery_loop(self) -> None:
        while True:
            await asyncio.sleep(_ASYNC_DELIVERY_RECOVERY_INTERVAL_SECONDS)
            try:
                recovered = await self.recover_pending_async_deliveries()
                if recovered:
                    logger.info(
                        "Periodic async delivery recovery completed run_ids={}",
                        recovered,
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Periodic async delivery recovery failed; pending deliveries remain retryable")
            try:
                projected = await self.recover_pending_lifecycle_projections()
                if projected:
                    logger.info("Periodic lifecycle projection recovery completed run_ids={}", projected)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Periodic lifecycle projection recovery failed; completed runs remain retryable")

    async def shutdown(self) -> None:
        self._accepting_submissions = False
        delivery_recovery_task = self._delivery_recovery_task
        self._delivery_recovery_task = None
        if delivery_recovery_task is not None:
            delivery_recovery_task.cancel()
            with suppress(asyncio.CancelledError):
                await delivery_recovery_task
        active_tasks = self._active_background_tasks()
        if not active_tasks:
            logger.info("Execution supervisor stopped")
            return

        logger.info(
            "Waiting for active run tasks before shutdown count={} run_ids={}",
            len(active_tasks),
            sorted(active_tasks),
        )
        timeout_seconds = self._settings.shutdown_timeout_seconds
        try:
            if timeout_seconds is None:
                await asyncio.gather(*active_tasks.values(), return_exceptions=True)
            else:
                done, pending = await asyncio.wait(set(active_tasks.values()), timeout=timeout_seconds)
                if done:
                    await asyncio.gather(*done, return_exceptions=True)
                if pending:
                    stale_tasks = self._active_background_tasks()
                    logger.warning(
                        "Cancelling active run tasks after shutdown timeout timeout_seconds={} count={} run_ids={}",
                        timeout_seconds,
                        len(stale_tasks),
                        sorted(stale_tasks),
                    )
                    for task in pending:
                        task.cancel()
                    cancelled_done, still_pending = await asyncio.wait(pending, timeout=1)
                    if cancelled_done:
                        await asyncio.gather(*cancelled_done, return_exceptions=True)
                    if still_pending:
                        logger.warning(
                            "Run tasks remained pending after cancellation count={} task_names={}",
                            len(still_pending),
                            sorted(task.get_name() for task in still_pending),
                        )
        finally:
            for run_id in list(self._runtime_state.background_tasks):
                self._runtime_state.clear_background_task(run_id)
            logger.info("Execution supervisor stopped")

    def _active_background_tasks(self) -> dict[str, asyncio.Task[None]]:
        return {run_id: task for run_id, task in self._runtime_state.background_tasks.items() if not task.done()}

    async def recover_queued_runs(self) -> list[str]:
        async with self._session_factory() as db_session:
            statement = select(RunRecord).where(RunRecord.status == "queued").order_by(RunRecord.created_at.asc())
            result = await db_session.execute(statement)
            run_ids = [record.id for record in result.scalars().all()]
        logger.info("Recovering queued runs count={} run_ids={}", len(run_ids), run_ids)
        for run_id in run_ids:
            self.submit_run(run_id)
        return run_ids

    async def cancel_orphaned_running_runs(self) -> list[str]:
        async with self._session_factory() as db_session:
            statement = select(RunRecord).where(RunRecord.status == "running")
            result = await db_session.execute(statement)
            records = list(result.scalars().all())
            cancelled_ids: list[str] = []
            for candidate in records:
                session_record, run_record = await _lock_run_scope(
                    db_session,
                    session_id=candidate.session_id,
                    run_id=candidate.id,
                )
                if run_record.status != "running":
                    continue
                recovery_reason = "Run was marked interrupted during YA Claw startup recovery."
                run_record.error_message = recovery_reason
                interrupt_run(session_record, run_record)
                await self._hitl_controller.cancel_pending_batch(
                    db_session,
                    run_id=run_record.id,
                    reason=recovery_reason,
                )
                metadata = dict(run_record.run_metadata)
                metadata.pop("active_interactions", None)
                metadata.pop("active_hitl_batch_id", None)
                run_record.run_metadata = metadata
                await reject_open_run_inputs(
                    db_session,
                    run_id=run_record.id,
                    reason="Run was interrupted during startup recovery before input could be applied.",
                    commit=False,
                )
                cancelled_ids.append(run_record.id)
            await db_session.flush()
            await self._process_recovered_async_task_runs(db_session, records)
            await db_session.commit()
            logger.info("Cancelled orphaned running runs count={} run_ids={}", len(cancelled_ids), cancelled_ids)
            return cancelled_ids

    async def _process_recovered_async_task_runs(
        self,
        db_session: AsyncSession,
        records: list[RunRecord],
    ) -> list[str]:
        async_task_controller = AsyncTaskController()
        recovered_ids: list[str] = []
        for run_record in records:
            if run_record.status not in _TERMINAL_RUN_STATUSES:
                continue
            result = await db_session.execute(
                select(SessionAsyncTaskRecord.id)
                .where(
                    SessionAsyncTaskRecord.task_run_id == run_record.id,
                    SessionAsyncTaskRecord.status.in_(list(_ACTIVE_ASYNC_TASK_STATUSES)),
                )
                .limit(1)
            )
            if result.scalar_one_or_none() is None:
                continue
            await async_task_controller.on_run_terminal(
                db_session,
                self._settings,
                self._runtime_state,
                run_record=run_record,
                submit_run=self.submit_run,
            )
            recovered_ids.append(run_record.id)
        return recovered_ids

    async def recover_stale_async_task_runs(self) -> list[str]:
        async with self._session_factory() as db_session:
            statement = (
                select(RunRecord)
                .join(SessionAsyncTaskRecord, SessionAsyncTaskRecord.task_run_id == RunRecord.id)
                .where(
                    SessionAsyncTaskRecord.status.in_(list(_ACTIVE_ASYNC_TASK_STATUSES)),
                    RunRecord.status.in_(list(_TERMINAL_RUN_STATUSES)),
                )
                .order_by(RunRecord.finished_at.asc(), RunRecord.created_at.asc())
            )
            result = await db_session.execute(statement)
            records = list(result.scalars().all())
            recovered_ids = await self._process_recovered_async_task_runs(db_session, records)
            await db_session.commit()
        logger.info("Recovered stale async task runs count={} run_ids={}", len(recovered_ids), recovered_ids)
        return recovered_ids

    async def recover_pending_async_deliveries(self) -> list[str]:
        async with self._session_factory() as db_session:
            return await AsyncTaskController().recover_pending_deliveries(
                db_session,
                self._settings,
                self._runtime_state,
                submit_run=self.submit_run,
            )

    async def recover_pending_lifecycle_projections(self) -> list[str]:
        return await CompletedRunProjector(
            settings=self._settings,
            session_factory=self._session_factory,
            runtime_state=self._runtime_state,
            run_store=self._run_store,
            submit_run=self.submit_run,
            agency_submit_run=self.submit_run,
        ).recover_pending()

    async def startup_recover(self) -> dict[str, list[str]]:
        empty = {
            "cancelled_running": [],
            "recovered_async_tasks": [],
            "recovered_async_deliveries": [],
            "recovered_lifecycle_projections": [],
            "submitted_queued": [],
        }
        try:
            cancelled_running = await self.cancel_orphaned_running_runs()
        except SQLAlchemyError:
            logger.warning("Run tables are unavailable; skipping startup recovery.")
            return empty

        try:
            recovered_async_tasks = await self.recover_stale_async_task_runs()
        except Exception:
            logger.exception("Stale async task recovery failed; other durable recovery will continue")
            recovered_async_tasks = []
        try:
            recovered_async_deliveries = await self.recover_pending_async_deliveries()
        except Exception:
            logger.exception("Async delivery recovery failed; other durable recovery will continue")
            recovered_async_deliveries = []
        try:
            recovered_lifecycle_projections = await self.recover_pending_lifecycle_projections()
        except Exception:
            logger.exception("Lifecycle projection recovery failed; queued run recovery will continue")
            recovered_lifecycle_projections = []
        try:
            submitted_queued = await self.recover_queued_runs()
        except SQLAlchemyError:
            logger.warning("Run tables became unavailable during queued run recovery.")
            submitted_queued = []
        return {
            "cancelled_running": cancelled_running,
            "recovered_async_tasks": recovered_async_tasks,
            "recovered_async_deliveries": recovered_async_deliveries,
            "recovered_lifecycle_projections": recovered_lifecycle_projections,
            "submitted_queued": submitted_queued,
        }

    async def _claim_and_execute(self, run_id: str) -> None:
        try:
            claimed = await self._claim_run(run_id)
            if not claimed:
                return
            coordinator = RunCoordinator(
                settings=self._settings,
                session_factory=self._session_factory,
                runtime_state=self._runtime_state,
                workspace_provider=self._workspace_provider,
                environment_factory=self._environment_factory,
                profile_resolver=self._profile_resolver,
                runtime_builder=self._runtime_builder,
                run_store=self._run_store,
                notification_hub=self._notification_hub,
                execution_harness=self._execution_harness,
                submit_run=self.submit_run,
            )
            await coordinator.execute(run_id)
        finally:
            self._runtime_state.clear_background_task(run_id)

    async def _claim_run(self, run_id: str) -> bool:
        if not self._accepting_submissions:
            logger.info("Run claim skipped run_id={} reason=supervisor_shutting_down", run_id)
            return False

        async with self._session_factory() as db_session:
            loaded_session, loaded_run = await _load_run_scope(db_session, run_id)
            session_record, run_record = await _lock_run_scope(
                db_session,
                session_id=loaded_session.id,
                run_id=loaded_run.id,
            )
            if run_record.status != "queued":
                logger.debug("Run claim skipped run_id={} status={}", run_id, run_record.status)
                return False
            if not self._accepting_submissions:
                logger.info("Run claim skipped run_id={} reason=supervisor_shutting_down", run_id)
                return False

            dispatch_mode = self._resolve_dispatch_mode(run_id)
            if self._runtime_state.get_run_handle(run_id) is None:
                self._runtime_state.register_run(session_record.id, run_id, dispatch_mode=dispatch_mode)

            logger.info(
                "Claiming run run_id={} session_id={} dispatch_mode={} instance_id={}",
                run_id,
                session_record.id,
                dispatch_mode,
                self._settings.instance_id,
            )
            mark_run_running(session_record, run_record, claimed_by=self._settings.instance_id)
            await db_session.commit()
            await db_session.refresh(run_record)
            await _publish_run_status_notification(
                self._notification_hub,
                "run.updated",
                run_record,
            )

            agui_adapter = AguiEventAdapter(
                session_id=session_record.id, run_id=run_id, config=CLAW_AGUI_ADAPTER_CONFIG
            )
            await self._runtime_state.append_run_event(
                run_id,
                agui_adapter.build_run_started_event(input_parts=list(run_record.input_parts)),
            )
            logger.info("Run claimed run_id={} session_id={}", run_id, session_record.id)
            return True

    def _resolve_dispatch_mode(self, run_id: str) -> str:
        handle = self._runtime_state.get_run_handle(run_id)
        if handle is None:
            return "async"
        return handle.dispatch_mode


class RunCoordinator:
    def __init__(
        self,
        *,
        settings: ClawSettings,
        session_factory: async_sessionmaker[AsyncSession],
        runtime_state: InMemoryRuntimeState,
        workspace_provider: WorkspaceProvider,
        environment_factory: EnvironmentFactory,
        profile_resolver: ProfileResolver,
        runtime_builder: ClawRuntimeBuilder,
        run_store: RunStore | None = None,
        notification_hub: NotificationHub | None = None,
        execution_harness: AgentExecutionHarness | None = None,
        submit_run: Callable[[str], bool] | None = None,
    ) -> None:
        self._settings = settings
        self._session_factory = session_factory
        self._runtime_state = runtime_state
        self._workspace_provider = workspace_provider
        self._environment_factory = environment_factory
        self._profile_resolver = profile_resolver
        self._runtime_builder = runtime_builder
        self._notification_hub = notification_hub
        self._execution_harness = execution_harness or AgentExecutionHarness()
        self._run_store = run_store or RunStore(settings)
        self._submit_run = submit_run
        self._hitl_controller = HitlController()

    def _completed_run_projector(self) -> CompletedRunProjector:
        return CompletedRunProjector(
            settings=self._settings,
            session_factory=self._session_factory,
            runtime_state=self._runtime_state,
            run_store=self._run_store,
            submit_run=self._submit_memory_run,
            agency_submit_run=self._submit_background_run,
        )

    async def execute(self, run_id: str) -> None:  # noqa: C901
        buffers = ExecutionBuffers()
        logger.info("Executing run run_id={}", run_id)
        try:
            async with self._session_factory() as db_session:
                session_record, run_record = await _load_run_scope(db_session, run_id)
                if run_record.status != "running":
                    logger.debug("Run execution skipped run_id={} status={}", run_id, run_record.status)
                    return
                if self._runtime_state.get_termination_requested(run_id) is not None:
                    logger.debug("Run execution skipped run_id={} reason=termination_requested", run_id)
                    return

                profile = await self._resolve_run_profile(
                    db_session,
                    session_record,
                    run_record,
                )
                workspace_binding = await self._resolve_workspace_binding(
                    db_session,
                    run_record,
                    session_record,
                    profile,
                )
                self._persist_run_workspace_snapshot(run_record, workspace_binding)
                await db_session.commit()
                restore_point = None
                if _run_restores_state(run_record):
                    restore_point = await load_restore_point(
                        db_session,
                        self._run_store,
                        session_record,
                        explicit_run_id=run_record.restore_from_run_id,
                    )
                dispatch_mode = self._resolve_dispatch_mode(run_id)
                logger.debug(
                    "Run execution prepared run_id={} session_id={} profile={} dispatch_mode={} restore_from_run_id={}",
                    run_id,
                    session_record.id,
                    profile.name,
                    dispatch_mode,
                    run_record.restore_from_run_id,
                )

            await self._execute_agent_run(
                run_id=run_id,
                session_id=session_record.id,
                dispatch_mode=dispatch_mode,
                workspace_binding=workspace_binding,
                restore_point=restore_point,
                input_parts=parse_input_parts(list(run_record.input_parts)),
                profile=profile,
                profile_name=run_record.profile_name,
                trigger_type=run_record.trigger_type,
                run_metadata=dict(run_record.run_metadata),
                buffers=buffers,
            )

            if not buffers.success_committed:
                async with self._runtime_state.session_lock(session_record.id):
                    await self._commit_successful_run(
                        run_id=run_id,
                        session_id=session_record.id,
                        dispatch_mode=dispatch_mode,
                        buffers=buffers,
                    )

        except AgentInterrupted:
            logger.info("Run interrupted by agent runtime run_id={}", run_id)
            async with self._session_factory() as db_session:
                session_record, run_record = await _load_run_scope(db_session, run_id)
                _index_run_usage_snapshot(run_record, buffers)
                if buffers.latest_message_payload is not None:
                    checkpoint = build_message_checkpoint(
                        run_id=run_record.id,
                        session_id=session_record.id,
                        checkpoint_kind=f"run_{run_record.termination_reason or 'interrupt'}",
                        message=self._runtime_state.get_replay_events(run_id),
                    )
                    write_message_checkpoint(self._run_store, checkpoint)
                await db_session.commit()
                await db_session.refresh(run_record)
                async_task_controller = AsyncTaskController()
                await async_task_controller.on_run_terminal(
                    db_session,
                    self._settings,
                    self._runtime_state,
                    run_record=run_record,
                    submit_run=self._submit_memory_run,
                )
                await async_task_controller.recover_pending_deliveries(
                    db_session,
                    self._settings,
                    self._runtime_state,
                    submit_run=self._submit_memory_run,
                    parent_session_id=session_record.id,
                )
        except Exception as exc:
            logger.exception("YA Claw run execution failed run_id={}", run_id)
            async with self._session_factory() as db_session:
                session_candidate, run_candidate = await _load_run_scope(
                    db_session,
                    run_id,
                )
                session_record, run_record = await _lock_run_scope(
                    db_session,
                    session_id=session_candidate.id,
                    run_id=run_candidate.id,
                )
                _index_run_usage_snapshot(run_record, buffers)
                if buffers.latest_message_payload is not None:
                    checkpoint = build_message_checkpoint(
                        run_id=run_record.id,
                        session_id=session_record.id,
                        checkpoint_kind="run_failed",
                        message=self._runtime_state.get_replay_events(run_id),
                    )
                    write_message_checkpoint(self._run_store, checkpoint)
                termination_requested = self._runtime_state.get_termination_requested(run_id)
                if run_record.status in _TERMINAL_RUN_STATUSES or termination_requested is not None:
                    await db_session.commit()
                    if run_record.status == "completed":
                        buffers.success_committed = True
                    logger.error(
                        "Post-commit run processing failed without changing terminal state run_id={} status={}",
                        run_id,
                        run_record.status,
                    )
                    try:
                        async_task_controller = AsyncTaskController()
                        await async_task_controller.on_run_terminal(
                            db_session,
                            self._settings,
                            self._runtime_state,
                            run_record=run_record,
                            submit_run=self._submit_memory_run,
                        )
                        await async_task_controller.recover_pending_deliveries(
                            db_session,
                            self._settings,
                            self._runtime_state,
                            submit_run=self._submit_memory_run,
                            parent_session_id=session_record.id,
                        )
                    except Exception:
                        logger.exception(
                            "Terminal run recovery hook failed run_id={} status={}",
                            run_id,
                            run_record.status,
                        )
                    if run_record.status == "completed":
                        try:
                            await self._completed_run_projector().project(run_id)
                        except Exception:
                            logger.exception("Terminal run lifecycle projection failed run_id={}", run_id)
                    return

                fail_run(session_record, run_record)
                run_record.error_message = self._stringify_error(exc)
                run_record.output_text = buffers.output_text
                run_record.output_json = buffers.output_json
                await reject_open_run_inputs(
                    db_session,
                    run_id=run_record.id,
                    reason="Run failed before input could be applied.",
                    commit=False,
                )
                await db_session.commit()
                await db_session.refresh(run_record)
                async_task_controller = AsyncTaskController()
                await async_task_controller.on_run_terminal(
                    db_session,
                    self._settings,
                    self._runtime_state,
                    run_record=run_record,
                    submit_run=self._submit_memory_run,
                )
                await async_task_controller.recover_pending_deliveries(
                    db_session,
                    self._settings,
                    self._runtime_state,
                    submit_run=self._submit_memory_run,
                    parent_session_id=session_record.id,
                )
                if session_record.session_type == "memory":
                    lifecycle = MemoryLifecycle(
                        settings=self._settings,
                        session_factory=self._session_factory,
                        runtime_state=self._runtime_state,
                        submit_run=self._submit_memory_run,
                    )
                    await lifecycle.on_memory_run_terminal(memory_run_id=run_record.id)
                elif session_record.session_type == "agency":
                    agency_lifecycle = AgencyLifecycle(
                        settings=self._settings,
                        runtime_state=self._runtime_state,
                        submit_run=self._submit_background_run,
                    )
                    await agency_lifecycle.on_agency_run_terminal(db_session, run_record)
                    await db_session.commit()
                    await db_session.refresh(run_record)
                await _publish_run_status_notification(
                    self._notification_hub,
                    "run.updated",
                    run_record,
                )
                agui_adapter = AguiEventAdapter(
                    session_id=session_record.id, run_id=run_id, config=CLAW_AGUI_ADAPTER_CONFIG
                )
                logger.info(
                    "Run failed run_id={} session_id={} error={}",
                    run_id,
                    session_record.id,
                    run_record.error_message,
                )
                await self._runtime_state.append_run_event(
                    run_id,
                    agui_adapter.build_run_error_event(
                        message=run_record.error_message or "YA Claw run failed.",
                        code=run_record.termination_reason,
                    ),
                    terminal=True,
                )
                buffers.terminal_event_emitted = True
                buffers.clear_runtime_handle = self._resolve_dispatch_mode(run_id) != "stream"
                if not buffers.clear_runtime_handle:
                    self._runtime_state.schedule_run_cleanup(run_id)
        finally:
            if not buffers.terminal_event_emitted:
                await self._runtime_state.close_run(run_id)
                buffers.clear_runtime_handle = self._resolve_dispatch_mode(run_id) != "stream"
                if not buffers.clear_runtime_handle:
                    self._runtime_state.schedule_run_cleanup(run_id)
                logger.debug(
                    "Run runtime state closed run_id={} terminal_event_emitted={}",
                    run_id,
                    buffers.terminal_event_emitted,
                )
            if buffers.clear_runtime_handle:
                self._runtime_state.clear_run(run_id)

    async def _commit_successful_run(
        self,
        *,
        run_id: str,
        session_id: str,
        dispatch_mode: str,
        buffers: ExecutionBuffers,
    ) -> None:
        async with self._session_factory() as db_session:
            session_candidate, run_candidate = await _load_run_scope(
                db_session,
                run_id,
            )
            session_record, run_record = await _lock_run_scope(
                db_session,
                session_id=session_candidate.id,
                run_id=run_candidate.id,
            )
            if run_record.status in _TERMINAL_RUN_STATUSES:
                _index_run_usage_snapshot(run_record, buffers)
                await db_session.commit()
                if run_record.status == "completed":
                    buffers.success_committed = True
                return

            effective_message_payload = buffers.latest_message_payload or {
                "events": self._runtime_state.get_replay_events(run_id),
                "message_history": [],
                "messages": [],
                "message_count": 0,
            }
            effective_state_payload = buffers.latest_state_payload or {
                "container_id": None,
                "context_state": {},
                "resumable_state": {},
                "message_history": list(effective_message_payload["message_history"]),
                "message_count": effective_message_payload["message_count"],
                "version": 3,
            }
            complete_run(session_record, run_record)
            run_record.output_text = buffers.output_text
            run_record.output_json = buffers.output_json
            _index_run_usage_snapshot(run_record, buffers)
            await reject_open_run_inputs(
                db_session,
                run_id=run_record.id,
                reason="Run completed before input could be applied.",
                commit=False,
            )
            agui_adapter = AguiEventAdapter(
                session_id=session_record.id, run_id=run_id, config=CLAW_AGUI_ADAPTER_CONFIG
            )
            buffers.terminal_event = agui_adapter.build_run_finished_event(
                result={
                    "termination_reason": run_record.termination_reason,
                    "committed_at": run_record.committed_at.isoformat() if run_record.committed_at else None,
                    "output_text": run_record.output_text,
                    "output_json": run_record.output_json,
                }
            )
            commit_run_artifacts(
                self._run_store,
                run_id=run_record.id,
                session_id=session_record.id,
                state=effective_state_payload,
                message=self._extract_replay_events(
                    effective_message_payload,
                    terminal_event=buffers.terminal_event,
                ),
            )
            await db_session.commit()
            buffers.success_committed = True
            buffers.clear_runtime_handle = dispatch_mode != "stream"
            await db_session.refresh(run_record)
            try:
                await _publish_run_status_notification(
                    self._notification_hub,
                    "run.updated",
                    run_record,
                )
            except Exception:
                logger.exception("Run completion notification failed run_id={}", run_id)

            try:
                await self._runtime_state.append_run_event(
                    run_id,
                    buffers.terminal_event,
                    terminal=True,
                )
            except Exception:
                logger.exception("Run completion event delivery failed run_id={}", run_id)
            else:
                buffers.terminal_event_emitted = True
            if not buffers.clear_runtime_handle:
                self._runtime_state.schedule_run_cleanup(run_id)
            try:
                async_task_controller = AsyncTaskController()
                await async_task_controller.on_run_terminal(
                    db_session,
                    self._settings,
                    self._runtime_state,
                    run_record=run_record,
                    submit_run=self._submit_memory_run,
                )
                await async_task_controller.recover_pending_deliveries(
                    db_session,
                    self._settings,
                    self._runtime_state,
                    submit_run=self._submit_memory_run,
                    parent_session_id=session_record.id,
                )
            except Exception:
                logger.exception("Completed run async task projection failed run_id={}", run_id)
            try:
                await self._completed_run_projector().project(run_id)
            except Exception:
                logger.exception("Completed run lifecycle projection failed run_id={}", run_id)
            logger.info(
                "Run completed run_id={} session_id={} output_text_chars={}",
                run_id,
                session_id,
                len(run_record.output_text or ""),
            )

    async def _build_steering_prompt_from_batches(
        self,
        batches: list[list[dict[str, Any]]],
        *,
        runtime: AgentRuntime[ClawAgentContext, Any, Environment],
    ) -> str | list[Any]:
        prompts: list[str | list[Any]] = []
        for raw_batch in batches:
            logger.debug("Forwarding terminal-gate steering input input_parts={}", len(raw_batch))
            parts = parse_input_parts(list(raw_batch))
            mapping = await map_input_parts(parts, file_operator=runtime.ctx.file_operator)
            prompts.append(self._build_user_prompt(mapping))
        if len(prompts) == 1:
            return prompts[0]
        text_items: list[str] = []
        mixed_items: list[Any] = []
        has_mixed = False
        for prompt in prompts:
            if isinstance(prompt, str):
                text_items.append(prompt)
                mixed_items.append(prompt)
            else:
                has_mixed = True
                mixed_items.extend(prompt)
        if has_mixed:
            return mixed_items
        return "\n\n".join(text_items)

    async def _execute_agent_run(  # noqa: C901
        self,
        *,
        run_id: str,
        session_id: str,
        dispatch_mode: str,
        workspace_binding: WorkspaceBinding,
        restore_point: ResolvedRestorePoint | None,
        input_parts: list[InputPart],
        profile: ResolvedProfile,
        profile_name: str | None,
        trigger_type: str,
        run_metadata: dict[str, Any],
        buffers: ExecutionBuffers,
    ) -> None:
        logger.info(
            "Starting agent run run_id={} session_id={} profile={} dispatch_mode={} workspace_provider={} cwd={}",
            run_id,
            session_id,
            profile.name,
            dispatch_mode,
            workspace_binding.metadata.get("provider"),
            workspace_binding.cwd,
        )
        environment = self._environment_factory.build(workspace_binding, profile=profile)
        memory_metadata = run_metadata.get("memory") if isinstance(run_metadata, dict) else None
        agency_metadata = run_metadata.get("agency") if isinstance(run_metadata, dict) else None
        self_client_session_id = _memory_source_session_id(memory_metadata) or session_id
        environment.resources.set(
            CLAW_SELF_CLIENT_KEY,
            ClawSelfClient(
                base_url=self._settings.public_base_url,
                api_token=self._settings.require_api_token(),
                session_id=self_client_session_id,
                run_id=run_id,
                profile_name=profile.name,
            ),
        )

        restored_state = self._extract_resumable_state(restore_point)
        source_metadata = _runtime_source_metadata(
            trigger_type=trigger_type,
            run_metadata=run_metadata,
            memory_metadata=memory_metadata,
            agency_metadata=agency_metadata,
        )
        runtime = self._runtime_builder.build(
            profile=profile,
            binding=workspace_binding,
            environment=environment,
            restore_state=restored_state,
            session_id=session_id,
            run_id=run_id,
            restore_from_run_id=restore_point.run_id if restore_point is not None else None,
            dispatch_mode=dispatch_mode,
            source_kind=trigger_type,
            source_metadata=source_metadata,
            claw_metadata={
                "profile": profile.metadata,
                "trigger_type": trigger_type,
                "run_metadata": run_metadata,
            },
        )
        message_history = self._extract_message_history(restore_point)
        agui_adapter = AguiEventAdapter(session_id=session_id, run_id=run_id, config=CLAW_AGUI_ADAPTER_CONFIG)
        deferred_tool_results: DeferredToolResults | None = None
        cumulative_usage_snapshot: UsageSnapshot | None = None
        use_initial_prompt = True
        refresh_task: asyncio.Task[None] | None = None

        try:
            async with runtime:
                try:
                    runtime.ctx.container_id = self._extract_environment_container_id(environment)
                    await self._persist_workspace_sandbox(session_id, workspace_binding, environment)
                    refresh_task = self._start_workspace_sandbox_refresh(
                        session_id=session_id,
                        workspace_binding=workspace_binding,
                        environment=environment,
                    )
                    logger.debug(
                        "Agent runtime entered run_id={} session_id={} container_id={}",
                        run_id,
                        session_id,
                        runtime.ctx.container_id,
                    )
                    next_user_prompt: str | list[Any] | None = None
                    while True:
                        current_user_prompt = next_user_prompt
                        next_user_prompt = None
                        segment_base_usage_snapshot = cumulative_usage_snapshot
                        segment_request: AgentSegmentRequest[ClawAgentContext, Any, Environment] = AgentSegmentRequest(
                            user_prompt=current_user_prompt,
                            user_prompt_factory=(
                                (lambda runtime_obj: self._build_initial_prompt(runtime_obj, input_parts))
                                if use_initial_prompt and current_user_prompt is None
                                else None
                            ),
                            message_history=message_history,
                            deferred_tool_results=deferred_tool_results,
                            resume_on_error=self._settings.agent_stream_resume_on_error,
                            resume_max_attempts=self._settings.agent_stream_resume_max_attempts,
                            transport_resume_max_attempts=self._settings.agent_stream_transport_resume_max_attempts,
                            resume_prompt=self._settings.agent_stream_resume_prompt,
                        )
                        async with self._execution_harness.stream_segment(
                            runtime,
                            segment_request,
                        ) as streamer:
                            self._runtime_state.bind_input_ingress(
                                run_id,
                                lambda input_id, raw_batch: self._enqueue_runtime_input(
                                    input_id,
                                    raw_batch,
                                    runtime=runtime,
                                    streamer=streamer,
                                ),
                            )
                            await self._deliver_accepted_run_inputs(run_id)
                            steering_task = asyncio.create_task(
                                self._forward_runtime_signals(run_id=run_id, streamer=streamer),
                                name=f"ya-claw-run-{run_id}-signals",
                            )
                            try:
                                async for stream_event in streamer:
                                    effective_stream_event, updated_usage_snapshot = _with_cumulative_usage_snapshot(
                                        stream_event,
                                        segment_base=segment_base_usage_snapshot,
                                        run_id=run_id,
                                    )
                                    if updated_usage_snapshot is not None:
                                        cumulative_usage_snapshot = updated_usage_snapshot
                                        buffers.latest_usage_snapshot = updated_usage_snapshot
                                    if isinstance(
                                        effective_stream_event.event,
                                        EnqueuedMessagesEvent,
                                    ):
                                        input_applied = await self._mark_run_input_applied(
                                            run_id,
                                            effective_stream_event.event,
                                            runtime=runtime,
                                        )
                                        if input_applied:
                                            await self._runtime_state.append_run_event(
                                                run_id,
                                                agui_adapter.build_run_custom_event(
                                                    "input_applied",
                                                    {"disposition": InputDisposition.applied.value},
                                                ),
                                            )
                                    if isinstance(
                                        effective_stream_event.event,
                                        ModelRequestStartEvent,
                                    ):
                                        await self._mark_source_delivery_applied(run_id)
                                    for agui_event in agui_adapter.adapt_stream_event(effective_stream_event):
                                        await self._runtime_state.append_run_event(run_id, agui_event)
                                    if streamer.run is not None:
                                        output = streamer.run.result.output if streamer.run.result else None
                                        buffers.output_text, buffers.output_json = self._normalize_output(output)
                                        if isinstance(
                                            effective_stream_event.event,
                                            (ModelRequestStartEvent, ModelRequestCompleteEvent, UsageSnapshotEvent),
                                        ):
                                            checkpoint = build_message_checkpoint(
                                                run_id=run_id,
                                                session_id=session_id,
                                                checkpoint_kind=type(effective_stream_event.event).__name__,
                                                message=self._runtime_state.get_replay_events(run_id),
                                            )
                                            write_message_checkpoint(self._run_store, checkpoint)
                                streamer.raise_if_exception()
                                logger.debug("Agent stream completed run_id={} session_id={}", run_id, session_id)
                            finally:
                                self._runtime_state.clear_input_ingress(run_id)
                                steering_task.cancel()
                                await asyncio.gather(steering_task, return_exceptions=True)

                            if streamer.run is None:
                                if self._runtime_state.get_termination_requested(run_id) is not None:
                                    raise AgentInterrupted()
                                raise RuntimeError("Stream agent completed without run context.")

                            segment_outcome = streamer.outcome()
                            buffers.latest_message_payload = self._build_message_payload(
                                streamer.run,
                                replay_events=self._runtime_state.get_replay_events(run_id),
                                recoverable_messages=list(segment_outcome.checkpoint.messages),
                            )
                            runtime.ctx.container_id = self._extract_environment_container_id(environment)
                            buffers.claw_metadata = dict(runtime.ctx.claw_metadata)
                            buffers.latest_state_payload = self._build_state_payload(
                                runtime.ctx,
                                workspace_binding=workspace_binding,
                                restore_point=restore_point,
                                profile=profile,
                                trigger_type=trigger_type,
                                message_payload=buffers.latest_message_payload,
                                usage_snapshot=cumulative_usage_snapshot,
                            )
                            output = segment_outcome.output
                            if isinstance(output, DeferredToolRequests):
                                message_history = list(segment_outcome.checkpoint.messages)
                                self._persist_deferred_segment_checkpoint(
                                    run_id=run_id,
                                    session_id=session_id,
                                    buffers=buffers,
                                )
                                deferred_tool_results = await self._handle_deferred_tool_requests(
                                    run_id=run_id,
                                    session_id=session_id,
                                    deferred_requests=output,
                                    agui_adapter=agui_adapter,
                                    run_metadata=run_metadata,
                                )
                                use_initial_prompt = False
                                continue

                            buffers.output_text, buffers.output_json = self._normalize_output(output)
                            logger.debug(
                                "Agent run artifacts prepared run_id={} message_count={} output_text_chars={}",
                                run_id,
                                buffers.latest_message_payload.get("message_count")
                                if isinstance(buffers.latest_message_payload, dict)
                                else None,
                                len(buffers.output_text or ""),
                            )
                            async with self._runtime_state.session_lock(session_id):
                                await self._commit_successful_run(
                                    run_id=run_id,
                                    session_id=session_id,
                                    dispatch_mode=dispatch_mode,
                                    buffers=buffers,
                                )
                            break

                finally:
                    if refresh_task is not None:
                        refresh_task.cancel()
                        await asyncio.gather(refresh_task, return_exceptions=True)
        finally:
            runtime.ctx.container_id = self._extract_environment_container_id(environment)
            await self._persist_workspace_sandbox(session_id, workspace_binding, environment, final=True)

    def _persist_deferred_segment_checkpoint(
        self,
        *,
        run_id: str,
        session_id: str,
        buffers: ExecutionBuffers,
    ) -> None:
        """Persist the stable native segment before exposing its HITL batch."""
        if buffers.latest_state_payload is None or buffers.latest_message_payload is None:
            raise RuntimeError("Deferred segment completed without stable checkpoint payloads")
        checkpointed_at = datetime.now(UTC).isoformat()
        self._run_store.write_state(
            run_id,
            {
                **buffers.latest_state_payload,
                "run_id": run_id,
                "session_id": session_id,
                "checkpoint_kind": "deferred_segment",
                "checkpointed_at": checkpointed_at,
            },
        )
        self._run_store.write_message(
            run_id,
            self._extract_replay_events(buffers.latest_message_payload),
        )

    async def _handle_deferred_tool_requests(
        self,
        *,
        run_id: str,
        session_id: str,
        deferred_requests: DeferredToolRequests,
        agui_adapter: AguiEventAdapter,
        run_metadata: dict[str, Any],
    ) -> DeferredToolResults | None:
        if _trigger_type_from_run_metadata(run_metadata) == TriggerType.AGENCY.value:
            return _deny_deferred_tool_requests(
                deferred_requests,
                reason="Agency runs use unattended deny mode for approval-required tool calls.",
            )

        interactions = build_active_interactions(deferred_requests, run_id=run_id, session_id=session_id)
        bridge_metadata = run_metadata.get("bridge") if isinstance(run_metadata.get("bridge"), dict) else None
        if bridge_metadata is not None:
            for interaction in interactions:
                interaction.metadata = {**interaction.metadata, "bridge": dict(bridge_metadata)}
        if not interactions:
            return DeferredToolResults()
        batch_id = await self._enter_hitl_pending(
            run_id, interactions, agui_adapter, deferred_requests=deferred_requests
        )
        inputs_staged = False
        try:
            user_interactions = await self._runtime_state.wait_hitl_batch(run_id)
            external_call_ids = {request.tool_call_id for request in deferred_requests.calls}
            resolutions: list[DeferredInteractionResolution] = []
            for interaction in user_interactions:
                if interaction.tool_call_id in external_call_ids:
                    resolutions.append(
                        DeferredCallResolution(
                            tool_call_id=interaction.tool_call_id,
                            result=(
                                interaction.user_input
                                if interaction.approved
                                else ToolDenied(message=interaction.reason or "External call denied")
                            ),
                        )
                    )
                else:
                    resolutions.append(
                        DeferredApprovalResolution(
                            tool_call_id=interaction.tool_call_id,
                            approved=interaction.approved,
                            reason=interaction.reason,
                            override_args=(
                                interaction.user_input if isinstance(interaction.user_input, dict) else None
                            ),
                        )
                    )
            results = DeferredInteractionResolver().resolve(deferred_requests, resolutions)
            await self._forward_deferred_hitl_inputs(run_id=run_id, batch_id=batch_id)
            inputs_staged = True
            return results
        finally:
            await self._exit_hitl_pending(
                run_id,
                agui_adapter,
                inputs_staged=inputs_staged,
            )

    async def _enter_hitl_pending(
        self,
        run_id: str,
        interactions: list[ActiveInteraction],
        agui_adapter: AguiEventAdapter,
        *,
        deferred_requests: DeferredToolRequests,
    ) -> str:
        if not interactions:
            raise ValueError("interactions must not be empty")
        session_id = interactions[0].session_id
        async with self._session_factory() as db_session:
            _, run_record = await _lock_run_scope(
                db_session,
                session_id=session_id,
                run_id=run_id,
            )
            batch = await self._hitl_controller.create_batch(
                db_session,
                session_id=session_id,
                run_id=run_id,
                interactions=interactions,
                deferred_requests=deferred_requests,
            )
            active_payload = [interaction.model_dump(mode="json") for interaction in batch.active_interactions]
            metadata = dict(run_record.run_metadata)
            metadata["active_interactions"] = active_payload
            metadata["active_hitl_batch_id"] = batch.batch_id
            run_record.run_metadata = metadata
            self._runtime_state.set_hitl_pending(run_id, session_id, interactions)
            await db_session.commit()
            await db_session.refresh(run_record)
            await _publish_run_status_notification(self._notification_hub, "run.updated", run_record)
        await self._runtime_state.append_run_event(
            run_id,
            agui_adapter.build_run_custom_event(
                "hitl_pending",
                {
                    "run_id": run_id,
                    "session_id": session_id,
                    "batch_id": batch.batch_id,
                    "active_interactions": active_payload,
                    "active_interaction_count": len(active_payload),
                },
            ),
        )
        return batch.batch_id

    async def _exit_hitl_pending(
        self,
        run_id: str,
        agui_adapter: AguiEventAdapter,
        *,
        inputs_staged: bool,
    ) -> None:
        self._runtime_state.clear_hitl(run_id)
        async with self._session_factory() as db_session:
            run_reference = await db_session.get(RunRecord, run_id)
            if not isinstance(run_reference, RunRecord):
                raise TypeError(f"Run '{run_id}' was not found.")
            session_record, run_record = await _lock_run_scope(
                db_session,
                session_id=run_reference.session_id,
                run_id=run_id,
            )
            if not inputs_staged:
                await self._hitl_controller.cancel_pending_batch(
                    db_session,
                    run_id=run_id,
                    reason="HITL continuation ended before deferred inputs were staged.",
                )
            metadata = dict(run_record.run_metadata)
            metadata.pop("active_interactions", None)
            metadata.pop("active_hitl_batch_id", None)
            run_record.run_metadata = metadata
            await db_session.commit()
            await db_session.refresh(run_record)
            await _publish_run_status_notification(self._notification_hub, "run.updated", run_record)
        await self._runtime_state.append_run_event(
            run_id,
            agui_adapter.build_run_custom_event(
                "hitl_resolved" if inputs_staged else "hitl_cancelled",
                {
                    "run_id": run_id,
                    "session_id": session_record.id,
                },
            ),
        )

    async def _forward_deferred_hitl_inputs(
        self,
        *,
        run_id: str,
        batch_id: str,
    ) -> None:
        async with self._session_factory() as db_session:
            deferred_inputs = await self._hitl_controller.stage_deferred_inputs(
                db_session,
                run_id=run_id,
                batch_id=batch_id,
            )
            await self._hitl_controller.mark_batch_completed(db_session, run_id=run_id)
            await db_session.commit()
        for deferred_input in deferred_inputs:
            logger.debug(
                "Staged deferred HITL input in canonical inbox run_id={} batch_id={} sequence_no={}",
                run_id,
                batch_id,
                deferred_input.sequence_no,
            )

    async def _build_initial_prompt(
        self,
        runtime_obj: AgentRuntime[ClawAgentContext, Any, Environment],
        input_parts: list[InputPart],
    ) -> str | list[Any]:
        mapping = await map_input_parts(input_parts, file_operator=runtime_obj.ctx.file_operator)
        return self._build_user_prompt(mapping)

    async def _enqueue_runtime_input(
        self,
        input_id: str,
        raw_batch: list[dict[str, Any]],
        *,
        runtime: AgentRuntime[ClawAgentContext, Any, Environment],
        streamer: AgentSegment[ClawAgentContext, Any, Environment],
    ) -> Any:
        async with self._session_factory() as db_session:
            input_record = await db_session.get(RunInputInboxRecord, input_id)
            if not isinstance(input_record, RunInputInboxRecord):
                raise TypeError(f"Run input {input_id!r} no longer exists")
            origin = InputOrigin(input_record.origin)
        parts = parse_input_parts(list(raw_batch))
        mapping = await map_input_parts(parts, file_operator=runtime.ctx.file_operator)
        content = self._build_user_prompt(mapping)
        if isinstance(content, str):
            return await streamer.enqueue(
                content,
                origin=origin,
                input_id=input_id,
            )
        return await streamer.enqueue(
            *content,
            origin=origin,
            input_id=input_id,
        )

    async def _deliver_accepted_run_inputs(self, run_id: str) -> None:
        async with self._session_factory() as db_session:
            await deliver_accepted_run_inputs(
                db_session,
                self._runtime_state,
                run_id,
            )

    async def _mark_run_input_applied(
        self,
        run_id: str,
        event: EnqueuedMessagesEvent,
        *,
        runtime: AgentRuntime[ClawAgentContext, Any, Environment],
    ) -> bool:
        router = runtime.ctx.input_router
        sdk_input_id = router.observe_event(event) if router is not None else None
        async with self._session_factory() as db_session:
            record = await mark_run_input_applied(
                db_session,
                run_id=run_id,
                sdk_input_id=sdk_input_id,
                enqueue_id=event.enqueue_id,
            )
        return record is not None

    async def _mark_source_delivery_applied(self, run_id: str) -> None:
        """Persist that a continuation command reached a native model request."""
        async with self._session_factory() as db_session:
            run_record = await lock_run_record(db_session, run_id)
            if not isinstance(run_record, RunRecord):
                return
            if run_record.source_delivery_id is None or run_record.source_delivery_applied_at is not None:
                return
            run_record.source_delivery_applied_at = datetime.now(UTC)
            await db_session.commit()

    async def _forward_runtime_signals(
        self,
        *,
        run_id: str,
        streamer: AgentSegment[ClawAgentContext, Any, Environment],
    ) -> None:
        while True:
            termination_reason = self._runtime_state.get_termination_requested(run_id)
            if isinstance(termination_reason, str):
                streamer.interrupt()
                return
            await asyncio.sleep(0.1)

    async def _resolve_run_profile(
        self,
        db_session: AsyncSession,
        session_record: SessionRecord,
        run_record: RunRecord,
    ) -> ResolvedProfile:
        if session_record.session_type != "async_task":
            descriptor = run_record.run_metadata.get(PROFILE_SNAPSHOT_METADATA_KEY)
            if not isinstance(descriptor, dict):
                raise RuntimeError(f"Run {run_record.id!r} has no exact execution profile descriptor")
            return resolved_profile_from_descriptor(descriptor)

        result = await db_session.execute(
            select(SessionAsyncTaskRecord)
            .where(
                SessionAsyncTaskRecord.task_session_id == session_record.id,
                SessionAsyncTaskRecord.task_run_id == run_record.id,
            )
            .limit(1)
        )
        task_record = result.scalar_one_or_none()
        if not isinstance(task_record, SessionAsyncTaskRecord):
            raise TypeError(f"Async task run '{run_record.id}' has no server-owned task linkage.")
        session_metadata = session_record.session_metadata if isinstance(session_record.session_metadata, dict) else {}
        session_task = session_metadata.get("async_task")
        session_task_id = session_task.get("task_id") if isinstance(session_task, dict) else None
        run_metadata = run_record.run_metadata if isinstance(run_record.run_metadata, dict) else {}
        run_task = run_metadata.get("async_task")
        run_task_id = run_task.get("task_id") if isinstance(run_task, dict) else None
        if session_task_id != task_record.id or run_task_id != task_record.id:
            raise RuntimeError(f"Async task run '{run_record.id}' has inconsistent session/run linkage.")
        if not isinstance(task_record.plan_descriptor, dict):
            raise TypeError(f"Async task '{task_record.id}' has no immutable 2.0 plan descriptor.")
        descriptor = SubagentPlanDescriptor.model_validate(task_record.plan_descriptor)
        if descriptor.spec.route != task_record.subagent_name:
            raise ValueError("Async task descriptor route does not match the task record")
        if descriptor.descriptor_id != task_record.plan_descriptor_ref:
            raise ValueError("Async task descriptor identity does not match the task record")
        if descriptor.fingerprint != task_record.plan_fingerprint:
            raise ValueError("Async task descriptor fingerprint does not match the task record")
        plan = restore_claw_subagent_plan(
            descriptor,
            capability_catalog=self._settings.resolved_capability_plugins.catalog,
        )
        return resolved_profile_from_subagent_plan(plan)

    async def _resolve_workspace_binding(
        self,
        db_session: AsyncSession,
        run_record: RunRecord,
        session_record: SessionRecord,
        profile: ResolvedProfile,
    ) -> WorkspaceBinding:
        session_metadata = session_record.session_metadata if isinstance(session_record.session_metadata, dict) else {}
        workspace_authority_metadata = session_metadata
        if session_record.session_type == "async_task" and isinstance(session_record.parent_session_id, str):
            parent_session = await db_session.get(
                SessionRecord,
                session_record.parent_session_id,
            )
            if not isinstance(parent_session, SessionRecord):
                raise RuntimeError(f"Async task session {session_record.id!r} has no authoritative parent session")
            workspace_authority_metadata = (
                parent_session.session_metadata if isinstance(parent_session.session_metadata, dict) else {}
            )
        run_metadata = run_record.run_metadata if isinstance(run_record.run_metadata, dict) else {}
        trigger_type = str(run_record.trigger_type)
        sandbox_scope = _sandbox_scope_for_trigger(trigger_type)
        metadata: dict[str, Any] = {
            "run_id": run_record.id,
            "session_id": session_record.id,
            "profile_name": profile.name,
            "trigger_type": trigger_type,
        }
        workspace_metadata = merge_workspace_metadata(
            session_metadata=workspace_authority_metadata,
            run_metadata=run_metadata,
        )
        if workspace_metadata is not None:
            metadata["workspace"] = workspace_metadata
        if sandbox_scope == SANDBOX_SCOPE_SESSION:
            sandbox = session_metadata.get("sandbox")
            if isinstance(sandbox, dict):
                metadata["sandbox"] = dict(sandbox)
        else:
            sandbox = run_metadata.get("sandbox")
            if isinstance(sandbox, dict):
                metadata["sandbox"] = dict(sandbox)
        binding = self._workspace_provider.resolve(metadata)
        self._apply_workspace_sandbox_lifecycle(binding, run_record=run_record, session_record=session_record)
        if isinstance(profile.workspace_backend_hint, str) and profile.workspace_backend_hint.strip() != "":
            binding.backend_hint = profile.workspace_backend_hint
            binding.metadata["workspace_backend_hint"] = profile.workspace_backend_hint
        logger.debug(
            "Workspace binding resolved run_id={} session_id={} provider={} backend_hint={} scope={} generation={} fingerprint={} host_path={} virtual_path={} docker_host_path={}",
            run_record.id,
            session_record.id,
            binding.metadata.get("provider"),
            binding.backend_hint,
            binding.sandbox_scope,
            binding.generation,
            binding.fingerprint,
            binding.host_path,
            binding.virtual_path,
            binding.docker_host_path,
        )
        return binding

    def _apply_workspace_sandbox_lifecycle(
        self,
        binding: WorkspaceBinding,
        *,
        run_record: RunRecord,
        session_record: SessionRecord,
    ) -> None:
        if (binding.backend_hint or binding.metadata.get("provider")) != "docker":
            return
        scope = _sandbox_scope_for_trigger(str(run_record.trigger_type))
        existing = _existing_sandbox_for_scope(scope, run_record=run_record, session_record=session_record)
        generation = _next_sandbox_generation(existing=existing, fingerprint=binding.fingerprint, scope=scope)
        container_ref = _build_scoped_container_ref(
            scope=scope,
            session_id=session_record.id,
            run_id=run_record.id,
            generation=generation,
        )
        sandbox_metadata = {
            **existing,
            "provider": "docker",
            "scope": scope,
            "generation": generation,
            "workspace_fingerprint": binding.fingerprint,
            "container_ref": container_ref,
            "image": binding.metadata.get("docker_image"),
            "session_id": session_record.id,
            "run_id": run_record.id,
            "retention_policy": self._settings.resolved_workspace_provider_docker_retention_policy,
            "idle_ttl_seconds": self._settings.resolved_workspace_provider_docker_idle_ttl_seconds,
        }
        if scope == SANDBOX_SCOPE_RUN:
            sandbox_metadata["cleanup_on_exit"] = True
            sandbox_metadata.pop("container_id", None)
        if _sandbox_fingerprint(existing) != binding.fingerprint:
            sandbox_metadata.pop("container_id", None)
            sandbox_metadata.pop("image_digest", None)
            sandbox_metadata["status"] = "created"
        binding.generation = generation
        binding.sandbox_scope = scope
        binding.metadata["sandbox"] = sandbox_metadata

    def _persist_run_workspace_snapshot(self, run_record: RunRecord, binding: WorkspaceBinding) -> None:
        metadata = dict(run_record.run_metadata)
        metadata[WORKSPACE_SNAPSHOT_METADATA_KEY] = workspace_snapshot(binding)
        run_record.run_metadata = metadata

    def _extract_environment_container_id(self, environment: Environment) -> str | None:
        if isinstance(environment, SandboxEnvironment):
            return environment.ready_container_id
        return None

    def _start_workspace_sandbox_refresh(
        self,
        *,
        session_id: str,
        workspace_binding: WorkspaceBinding,
        environment: Environment,
    ) -> asyncio.Task[None] | None:
        if workspace_binding.sandbox_scope != SANDBOX_SCOPE_SESSION:
            return None
        if not isinstance(environment, SandboxEnvironment):
            return None
        interval_seconds = max(30, min(300, self._settings.resolved_workspace_provider_docker_idle_ttl_seconds // 3))
        return asyncio.create_task(
            self._refresh_workspace_sandbox_loop(
                session_id=session_id,
                workspace_binding=workspace_binding,
                environment=environment,
                interval_seconds=interval_seconds,
            ),
            name=f"ya-claw-sandbox-refresh-{session_id}",
        )

    async def _refresh_workspace_sandbox_loop(
        self,
        *,
        session_id: str,
        workspace_binding: WorkspaceBinding,
        environment: Environment,
        interval_seconds: int,
    ) -> None:
        try:
            while True:
                await asyncio.sleep(interval_seconds)
                await self._persist_workspace_sandbox(session_id, workspace_binding, environment)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Workspace sandbox last_used_at refresh failed session_id={}", session_id)

    async def _persist_workspace_sandbox(
        self,
        session_id: str,
        workspace_binding: WorkspaceBinding,
        environment: Environment,
        *,
        final: bool = False,
    ) -> None:
        sandbox_metadata = build_workspace_sandbox_metadata(binding=workspace_binding, environment=environment)
        if sandbox_metadata is None:
            return
        now = _utc_now_iso()
        sandbox_metadata["last_used_at"] = now
        if sandbox_metadata.get("status") == "running" and sandbox_metadata.get("container_id") is not None:
            sandbox_metadata.setdefault("last_started_at", now)
        scope = str(sandbox_metadata.get("scope") or workspace_binding.sandbox_scope or SANDBOX_SCOPE_SESSION)
        if final and scope == SANDBOX_SCOPE_RUN:
            sandbox_metadata["status"] = "stopped"
            sandbox_metadata["container_id"] = None

        async with self._session_factory() as db_session:
            session_record = await db_session.get(SessionRecord, session_id)
            if not isinstance(session_record, SessionRecord):
                return
            if scope == SANDBOX_SCOPE_SESSION:
                session_metadata = dict(session_record.session_metadata)
                session_metadata["sandbox"] = sandbox_metadata
                session_record.session_metadata = session_metadata
            run_id = _normalize_metadata_string(sandbox_metadata.get("run_id"))
            if run_id is not None:
                run_record = await db_session.get(RunRecord, run_id)
                if isinstance(run_record, RunRecord):
                    run_metadata = dict(run_record.run_metadata)
                    run_metadata["sandbox"] = sandbox_metadata
                    run_metadata[WORKSPACE_SNAPSHOT_METADATA_KEY] = workspace_snapshot(workspace_binding)
                    run_record.run_metadata = run_metadata
            await db_session.commit()
            logger.debug(
                "Persisted workspace sandbox metadata session_id={} scope={} generation={} container_id={} container_ref={}",
                session_id,
                scope,
                sandbox_metadata.get("generation"),
                sandbox_metadata.get("container_id"),
                sandbox_metadata.get("container_ref"),
            )
        if scope == SANDBOX_SCOPE_SESSION:
            await self._publish_workspace_sandbox_update(session_id=session_id, sandbox_metadata=sandbox_metadata)

    async def _publish_workspace_sandbox_update(
        self,
        *,
        session_id: str,
        sandbox_metadata: dict[str, Any],
    ) -> None:
        if self._notification_hub is None:
            return
        sandbox_state = build_session_sandbox_state_from_sandbox(sandbox_metadata)
        handle = self._runtime_state.get_session_run_handle(session_id)
        run_id = handle.run_id if handle is not None else None
        payload = session_sandbox_event_payload(
            session_id=session_id,
            run_id=run_id,
            sandbox_state=sandbox_state,
        )
        await self._notification_hub.publish("workspace.sandbox.updated", payload)
        if run_id is not None:
            try:
                await self._runtime_state.append_run_event(run_id, payload, replay=False)
            except KeyError:
                return

    def _resolve_dispatch_mode(self, run_id: str) -> str:
        handle = self._runtime_state.get_run_handle(run_id)
        if handle is None:
            return "async"
        return handle.dispatch_mode

    def _build_user_prompt(self, mapping: InputMappingResult) -> str | list[Any]:
        if len(mapping.user_prompt) == 1 and isinstance(mapping.user_prompt[0], str):
            return mapping.user_prompt[0]
        return list(mapping.user_prompt)

    def _extract_resumable_state(self, restore_point: ResolvedRestorePoint | None) -> ResumableState | None:
        if restore_point is None or restore_point.state is None:
            return None
        raw_state = restore_point.state.get("context_state")
        if not isinstance(raw_state, dict):
            raw_state = restore_point.state.get("resumable_state")
        if not isinstance(raw_state, dict):
            raw_state = restore_point.state.get("exported_state")
        if not isinstance(raw_state, dict):
            return None
        return load_resumable_state(raw_state)

    def _extract_message_history(self, restore_point: ResolvedRestorePoint | None) -> list[ModelMessage] | None:
        if restore_point is None:
            return None

        raw_messages: list[Any] | None = None
        if isinstance(restore_point.state, dict):
            state_messages = restore_point.state.get("message_history")
            if isinstance(state_messages, list):
                raw_messages = state_messages
        if not isinstance(raw_messages, list) and isinstance(restore_point.message, list):
            message_events = restore_point.message
            message_history = self._message_history_from_replay_events(message_events)
            if isinstance(message_history, list):
                raw_messages = message_history

        if not isinstance(raw_messages, list):
            return None
        # Run store/state payloads contain JSON-compatible ModelMessagesTypeAdapter.dump_python(..., mode="json") output.
        # validate_json restores nested BinaryContent in ToolReturnPart.content via val_json_bytes="base64".
        return cast(list[ModelMessage], ModelMessagesTypeAdapter.validate_json(json.dumps(raw_messages).encode()))

    def _build_state_payload(
        self,
        ctx: ClawAgentContext,
        *,
        workspace_binding: WorkspaceBinding,
        restore_point: ResolvedRestorePoint | None,
        profile: ResolvedProfile,
        trigger_type: str,
        message_payload: dict[str, Any] | None,
        usage_snapshot: UsageSnapshot | None,
    ) -> dict[str, Any]:
        exported_state = ctx.export_state().model_dump(mode="json")
        message_history = message_payload.get("message_history") if isinstance(message_payload, dict) else None
        if not isinstance(message_history, list):
            message_history = []
        message_count = message_payload.get("message_count") if isinstance(message_payload, dict) else None
        return {
            "container_id": ctx.container_id,
            "context_state": exported_state,
            "usage_snapshot": usage_snapshot.model_dump(mode="json") if usage_snapshot is not None else None,
            "message_history": message_history,
            "message_count": message_count,
            "resumable_state": exported_state,
            "restore": {
                "run_id": restore_point.run_id,
                "status": restore_point.status,
            }
            if restore_point is not None
            else None,
            "workspace": {
                "virtual_path": str(workspace_binding.virtual_path),
                "cwd": str(workspace_binding.cwd),
                "readable_paths": [str(path) for path in workspace_binding.readable_paths],
                "writable_paths": [str(path) for path in workspace_binding.writable_paths],
                "metadata": self._serialize_value(workspace_binding.metadata),
            },
            "profile": {
                "name": profile.name,
                "metadata": self._serialize_value(profile.metadata),
            },
            "context": {
                "session_id": ctx.session_id,
                "claw_run_id": ctx.claw_run_id,
                "profile_name": ctx.profile_name,
                "restore_from_run_id": ctx.restore_from_run_id,
                "dispatch_mode": ctx.dispatch_mode,
                "source_kind": ctx.source_kind,
                "source_metadata": self._serialize_value(ctx.source_metadata),
                "claw_metadata": self._serialize_value(ctx.claw_metadata),
                "workspace_binding": self._serialize_value(
                    ctx.workspace_binding.model_dump(mode="json") if ctx.workspace_binding is not None else None
                ),
            },
            "trigger_type": trigger_type,
            "version": 5,
        }

    def _extract_replay_events(
        self,
        message_payload: dict[str, Any] | None,
        *,
        terminal_event: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        raw_events = message_payload.get("events") if isinstance(message_payload, dict) else None
        events = [event for event in raw_events if isinstance(event, dict)] if isinstance(raw_events, list) else []
        if terminal_event is not None:
            events.append(terminal_event)
        return events

    def _build_message_payload(
        self,
        run: _AgentRunMessages,
        *,
        replay_events: list[dict[str, Any]],
        recoverable_messages: list[ModelMessage] | None = None,
    ) -> dict[str, Any]:
        source_messages = recoverable_messages if recoverable_messages is not None else run.all_messages()
        messages = ModelMessagesTypeAdapter.dump_python(source_messages, mode="json")
        events = list(replay_events)
        return {
            "events": events,
            "message_history": messages,
            "messages": events,
            "message_count": len(messages) if isinstance(messages, list) else None,
        }

    def _message_history_from_replay_events(self, events: list[dict[str, Any]]) -> list[Any] | None:
        if len(events) == 1:
            payload = events[0]
            message_history = payload.get("message_history")
            if isinstance(message_history, list):
                return message_history
        return None

    def _build_message_payload_from_messages(
        self,
        messages_source: list[ModelMessage],
        *,
        replay_events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        messages = ModelMessagesTypeAdapter.dump_python(messages_source, mode="json")
        events = list(replay_events)
        return {
            "events": events,
            "message_history": messages,
            "messages": events,
            "message_count": len(messages) if isinstance(messages, list) else None,
        }

    def _normalize_output(
        self,
        output: object,
    ) -> tuple[str | None, JsonValue]:
        if output is None:
            return None, None
        if isinstance(output, str):
            value = output.strip()
            return value or None, None
        if isinstance(output, BaseModel):
            json_value: JsonValue = output.model_dump(mode="json")
        elif is_dataclass(output) and not isinstance(output, type):
            json_value = cast(JsonValue, asdict(output))
        elif isinstance(output, dict | list | int | float | bool):
            json_value = cast(JsonValue, output)
        else:
            return str(output), None
        return (
            json.dumps(
                json_value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            json_value,
        )

    def _stringify_error(self, exc: Exception) -> str:
        try:
            value = str(exc)
        except Exception:
            value = repr(exc)
        return value[:4000]

    def _submit_background_run(self, run_id: str) -> bool:
        if self._submit_run is None:
            logger.info("Run submission skipped run_id={} reason=no_submission_gateway", run_id)
            return False
        return self._submit_run(run_id)

    def _submit_memory_run(self, run_id: str) -> bool:
        return self._submit_background_run(run_id)

    def _serialize_value(self, value: object) -> JsonValue:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, datetime):
            return value.astimezone(UTC).isoformat()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if is_dataclass(value) and not isinstance(value, type):
            return self._serialize_value(asdict(value))
        if isinstance(value, dict):
            return {str(key): self._serialize_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._serialize_value(item) for item in value]
        return str(value)


ExecutionCoordinator = RunCoordinator


def _run_restores_state(run_record: RunRecord) -> bool:
    run_metadata = run_record.run_metadata if isinstance(run_record.run_metadata, dict) else {}
    return run_metadata.get("restore_state") is not False


def _runtime_source_metadata(
    *,
    trigger_type: str,
    run_metadata: dict[str, Any],
    memory_metadata: object,
    agency_metadata: object = None,
) -> dict[str, Any]:
    metadata = {"trigger_type": trigger_type, **run_metadata}
    if isinstance(memory_metadata, dict):
        metadata["memory"] = memory_metadata
    if isinstance(agency_metadata, dict):
        metadata["agency"] = agency_metadata
    return metadata


def _memory_source_session_id(memory_metadata: object) -> str | None:
    if not isinstance(memory_metadata, dict):
        return None
    value = memory_metadata.get("source_session_id")
    return value if isinstance(value, str) and value.strip() else None


def _agency_source_session_ids(agency_metadata: object) -> list[str]:
    if not isinstance(agency_metadata, dict):
        return []
    values = agency_metadata.get("source_session_ids")
    if not isinstance(values, list):
        return []
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if isinstance(value, str) and value.strip() and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _agency_primary_source_session_id(agency_metadata: object) -> str | None:
    if not isinstance(agency_metadata, dict):
        return None
    value = agency_metadata.get("primary_source_session_id")
    if isinstance(value, str) and value.strip():
        return value
    source_session_ids = _agency_source_session_ids(agency_metadata)
    return source_session_ids[0] if source_session_ids else None


def _trigger_type_from_run_metadata(run_metadata: dict[str, Any]) -> str | None:
    value = run_metadata.get("trigger_type")
    return value if isinstance(value, str) else None


def _deny_deferred_tool_requests(deferred_requests: DeferredToolRequests, *, reason: str) -> DeferredToolResults:
    results = DeferredToolResults()
    for request in deferred_requests.approvals:
        results.approvals[request.tool_call_id] = ToolDenied(message=reason)
    for request in deferred_requests.calls:
        results.calls[request.tool_call_id] = ToolDenied(message=reason)
    return results


async def _publish_run_status_notification(
    notification_hub: NotificationHub | None,
    event_type: str,
    run_record: RunRecord,
) -> None:
    if notification_hub is None:
        return
    memory_metadata = run_record.run_metadata.get("memory") if isinstance(run_record.run_metadata, dict) else None
    agency_metadata = run_record.run_metadata.get("agency") if isinstance(run_record.run_metadata, dict) else None
    source_session_id = _memory_source_session_id(memory_metadata) or _agency_primary_source_session_id(agency_metadata)
    active_interactions = _active_interactions_from_run(run_record)
    session_status_reason = _session_status_reason_from_run(run_record, active_interactions=active_interactions)
    session_status_detail = _session_status_detail_from_run(run_record, active_interactions=active_interactions)
    payload = {
        "session_id": run_record.session_id,
        "source_session_id": source_session_id,
        "run_id": run_record.id,
        "status": run_record.status,
        "sequence_no": run_record.sequence_no,
        "profile_name": run_record.profile_name,
        "termination_reason": run_record.termination_reason,
        "error_message": run_record.error_message,
        "session_status": run_record.status,
        "session_status_reason": session_status_reason,
        "session_status_detail": session_status_detail,
    }
    await notification_hub.publish(event_type, payload)
    if event_type != "session.updated":
        await notification_hub.publish(
            "session.updated",
            {
                "session_id": run_record.session_id,
                "source_session_id": source_session_id,
                "status": run_record.status,
                "status_reason": session_status_reason,
                "status_detail": session_status_detail,
                "profile_name": run_record.profile_name,
                "head_run_id": run_record.id,
                "active_run_id": run_record.id if run_record.status in {RunStatus.QUEUED, RunStatus.RUNNING} else None,
                "latest_run_id": run_record.id,
                "latest_run_sequence_no": run_record.sequence_no,
                "latest_run_status": run_record.status,
                "termination_reason": run_record.termination_reason,
                "error_message": run_record.error_message,
                "updated_at": (run_record.finished_at or run_record.started_at or run_record.created_at).isoformat(),
            },
        )


def _active_interactions_from_run(run_record: RunRecord) -> list[dict[str, Any]]:
    if not isinstance(run_record.run_metadata, dict):
        return []
    interactions = run_record.run_metadata.get("active_interactions")
    if not isinstance(interactions, list):
        return []
    return [interaction for interaction in interactions if isinstance(interaction, dict)]


def _session_status_reason_from_run(
    run_record: RunRecord,
    *,
    active_interactions: list[dict[str, Any]] | None = None,
) -> str:
    if run_record.status == RunStatus.QUEUED:
        return SessionStatusReason.RUN_QUEUED
    if run_record.status == RunStatus.RUNNING:
        if active_interactions:
            return SessionStatusReason.HITL_PENDING
        return SessionStatusReason.RUN_RUNNING
    if run_record.status == RunStatus.COMPLETED:
        return SessionStatusReason.RUN_COMPLETED
    if run_record.status == RunStatus.FAILED:
        return SessionStatusReason.RUN_FAILED
    if run_record.status == RunStatus.CANCELLED:
        return SessionStatusReason.RUN_CANCELLED
    return SessionStatusReason.IDLE


def _session_status_detail_from_run(
    run_record: RunRecord,
    *,
    active_interactions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    detail: dict[str, Any] = {
        "run_id": run_record.id,
        "sequence_no": run_record.sequence_no,
        "trigger_type": run_record.trigger_type,
    }
    if isinstance(run_record.termination_reason, str):
        detail["termination_reason"] = run_record.termination_reason
    if isinstance(run_record.error_message, str):
        detail["error_message"] = run_record.error_message
    if active_interactions:
        detail["active_interactions"] = active_interactions
        detail["active_interaction_count"] = len(active_interactions)
    return detail


def _sandbox_scope_for_trigger(trigger_type: str) -> SandboxScopeLiteral:
    if trigger_type in {TriggerType.SCHEDULE.value, TriggerType.WORKFLOW.value, TriggerType.HEARTBEAT.value}:
        return SANDBOX_SCOPE_RUN
    return SANDBOX_SCOPE_SESSION


def _existing_sandbox_for_scope(
    scope: SandboxScopeLiteral,
    *,
    run_record: RunRecord,
    session_record: SessionRecord,
) -> dict[str, Any]:
    if scope == SANDBOX_SCOPE_RUN:
        metadata = run_record.run_metadata if isinstance(run_record.run_metadata, dict) else {}
    else:
        metadata = session_record.session_metadata if isinstance(session_record.session_metadata, dict) else {}
    sandbox = metadata.get("sandbox")
    return dict(sandbox) if isinstance(sandbox, dict) else {}


def _sandbox_fingerprint(sandbox: dict[str, Any]) -> str | None:
    value = sandbox.get("workspace_fingerprint")
    return value if isinstance(value, str) and value.strip() else None


def _next_sandbox_generation(*, existing: dict[str, Any], fingerprint: str, scope: SandboxScopeLiteral) -> int:
    if scope == SANDBOX_SCOPE_RUN:
        return 1
    current_generation = existing.get("generation")
    generation = current_generation if isinstance(current_generation, int) and current_generation > 0 else 0
    if generation > 0 and _sandbox_fingerprint(existing) == fingerprint:
        return generation
    return generation + 1


def _build_scoped_container_ref(*, scope: SandboxScopeLiteral, session_id: str, run_id: str, generation: int) -> str:
    return build_docker_container_ref(
        scope=scope,
        session_id=session_id,
        run_id=run_id,
        generation=generation,
    )


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _normalize_metadata_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


async def _lock_run_scope(
    db_session: AsyncSession,
    *,
    session_id: str,
    run_id: str,
) -> tuple[SessionRecord, RunRecord]:
    """Lock session then run for all serialized run state transitions."""
    session_record = await lock_session_reference(db_session, session_id)
    if not isinstance(session_record, SessionRecord):
        raise TypeError(f"Session '{session_id}' was not found.")
    run_record = await lock_run_record(db_session, run_id)
    if not isinstance(run_record, RunRecord):
        raise TypeError(f"Run '{run_id}' was not found.")
    if run_record.session_id != session_record.id:
        raise RuntimeError(f"Run '{run_id}' does not belong to session '{session_id}'.")
    return session_record, run_record


async def _load_run_scope(db_session: AsyncSession, run_id: str) -> tuple[SessionRecord, RunRecord]:
    run_record = await db_session.get(RunRecord, run_id)
    if not isinstance(run_record, RunRecord):
        raise TypeError(f"Run '{run_id}' was not found.")
    session_record = await db_session.get(SessionRecord, run_record.session_id)
    if not isinstance(session_record, SessionRecord):
        raise TypeError(f"Session '{run_record.session_id}' was not found.")
    return session_record, run_record
