from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from loguru import logger
from pydantic import BaseModel
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from y_agent_environment import Environment
from ya_agent_sdk.agents.main import AgentInterrupted, AgentRuntime, stream_agent
from ya_agent_sdk.context import BusMessage, ResumableState
from ya_agent_sdk.environment import SandboxEnvironment
from ya_agent_sdk.events import ModelRequestCompleteEvent, ModelRequestStartEvent

from ya_claw.agui_adapter import AguiEventAdapter
from ya_claw.config import ClawSettings
from ya_claw.context import ClawAgentContext
from ya_claw.controller.models import InputPart, RunStatus, SessionStatusReason, parse_input_parts
from ya_claw.execution.background import BACKGROUND_MONITOR_KEY, BackgroundMonitor
from ya_claw.execution.checkpoint import build_message_checkpoint, commit_run_artifacts, write_message_checkpoint
from ya_claw.execution.input import InputMappingResult, map_input_parts
from ya_claw.execution.profile import ProfileResolver, ResolvedProfile
from ya_claw.execution.restore import ResolvedRestorePoint, load_restore_point
from ya_claw.execution.runtime import ClawRuntimeBuilder
from ya_claw.execution.state_machine import complete_run, fail_run, interrupt_run, mark_run_running
from ya_claw.execution.store import RunStore
from ya_claw.memory.lifecycle import MemoryLifecycle
from ya_claw.notifications import NotificationHub
from ya_claw.orm.tables import RunRecord, SessionRecord
from ya_claw.runtime_state import InMemoryRuntimeState
from ya_claw.toolsets.session import CLAW_SELF_CLIENT_KEY, ClawSelfClient
from ya_claw.workspace import (
    EnvironmentFactory,
    WorkspaceBinding,
    WorkspaceProvider,
    build_workspace_sandbox_metadata,
)


@dataclass(slots=True)
class ExecutionBuffers:
    latest_state_payload: dict[str, Any] | None = None
    latest_message_payload: dict[str, Any] | None = None
    terminal_event: dict[str, Any] | None = None
    output_text: str | None = None
    output_summary: str | None = None
    claw_metadata: dict[str, Any] = field(default_factory=dict)


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
    ) -> None:
        self._settings = settings
        self._session_factory = session_factory
        self._runtime_state = runtime_state
        self._workspace_provider = workspace_provider
        self._environment_factory = environment_factory
        self._profile_resolver = profile_resolver
        self._runtime_builder = runtime_builder
        self._notification_hub = notification_hub
        self._run_store = RunStore(settings)
        self._accepting_submissions = True

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

    async def shutdown(self) -> None:
        self._accepting_submissions = False
        while True:
            active_tasks = {
                run_id: task for run_id, task in self._runtime_state.background_tasks.items() if not task.done()
            }
            if not active_tasks:
                logger.info("Execution supervisor stopped")
                return
            logger.info(
                "Waiting for active run tasks before shutdown count={} run_ids={}",
                len(active_tasks),
                sorted(active_tasks),
            )
            await asyncio.gather(*active_tasks.values(), return_exceptions=True)

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
            for run_record in records:
                session_record = await db_session.get(SessionRecord, run_record.session_id)
                if not isinstance(session_record, SessionRecord):
                    continue
                run_record.error_message = "Run was marked interrupted during YA Claw startup recovery."
                interrupt_run(session_record, run_record)
                cancelled_ids.append(run_record.id)
            await db_session.commit()
            logger.info("Cancelled orphaned running runs count={} run_ids={}", len(cancelled_ids), cancelled_ids)
            return cancelled_ids

    async def startup_recover(self) -> dict[str, list[str]]:
        try:
            cancelled_running = await self.cancel_orphaned_running_runs()
            submitted_queued = await self.recover_queued_runs()
        except SQLAlchemyError:
            logger.warning("Run tables are unavailable; skipping startup recovery.")
            return {"cancelled_running": [], "submitted_queued": []}
        return {
            "cancelled_running": cancelled_running,
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
            )
            await coordinator.execute(run_id)
        finally:
            self._runtime_state.clear_background_task(run_id)

    async def _claim_run(self, run_id: str) -> bool:
        async with self._session_factory() as db_session:
            session_record, run_record = await _load_run_scope(db_session, run_id)
            if run_record.status != "queued":
                logger.debug("Run claim skipped run_id={} status={}", run_id, run_record.status)
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

            agui_adapter = AguiEventAdapter(session_id=session_record.id, run_id=run_id)
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
    ) -> None:
        self._settings = settings
        self._session_factory = session_factory
        self._runtime_state = runtime_state
        self._workspace_provider = workspace_provider
        self._environment_factory = environment_factory
        self._profile_resolver = profile_resolver
        self._runtime_builder = runtime_builder
        self._notification_hub = notification_hub
        self._run_store = run_store or RunStore(settings)

    async def execute(self, run_id: str) -> None:  # noqa: C901
        buffers = ExecutionBuffers()
        terminal_event_emitted = False

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

                profile = await self._profile_resolver.resolve(run_record.profile_name or session_record.profile_name)
                workspace_binding = self._resolve_workspace_binding(run_record, session_record, profile)
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

            async with self._session_factory() as db_session:
                session_record, run_record = await _load_run_scope(db_session, run_id)
                if run_record.status == "cancelled":
                    await db_session.commit()
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
                run_record.output_summary = buffers.output_summary
                agui_adapter = AguiEventAdapter(session_id=session_record.id, run_id=run_id)
                buffers.terminal_event = agui_adapter.build_run_finished_event(
                    result={
                        "termination_reason": run_record.termination_reason,
                        "committed_at": run_record.committed_at.isoformat() if run_record.committed_at else None,
                        "output_summary": run_record.output_summary,
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
                await db_session.refresh(run_record)
                await _publish_run_status_notification(
                    self._notification_hub,
                    "run.updated",
                    run_record,
                )

                await self._runtime_state.append_run_event(
                    run_id,
                    buffers.terminal_event,
                    terminal=True,
                )
                terminal_event_emitted = True
                lifecycle = MemoryLifecycle(
                    settings=self._settings,
                    session_factory=self._session_factory,
                    runtime_state=self._runtime_state,
                    submit_run=self._submit_memory_run,
                )
                if session_record.session_type == "memory":
                    await lifecycle.on_memory_run_committed(memory_run_id=run_record.id)
                else:
                    await lifecycle.on_run_committed(
                        source_session_id=session_record.id,
                        source_run_id=run_record.id,
                        source_sequence_no=run_record.sequence_no,
                        profile_name=run_record.profile_name,
                        claw_metadata=buffers.claw_metadata,
                    )
                logger.info(
                    "Run completed run_id={} session_id={} output_summary_chars={}",
                    run_id,
                    session_record.id,
                    len(run_record.output_summary or ""),
                )
        except AgentInterrupted:
            logger.info("Run interrupted by agent runtime run_id={}", run_id)
            async with self._session_factory() as db_session:
                session_record, run_record = await _load_run_scope(db_session, run_id)
                if buffers.latest_message_payload is not None:
                    checkpoint = build_message_checkpoint(
                        run_id=run_record.id,
                        session_id=session_record.id,
                        checkpoint_kind=f"run_{run_record.termination_reason or 'interrupt'}",
                        message=self._runtime_state.get_replay_events(run_id),
                    )
                    write_message_checkpoint(self._run_store, checkpoint)
                await db_session.commit()
        except Exception as exc:
            logger.exception("YA Claw run execution failed run_id={}", run_id)
            async with self._session_factory() as db_session:
                session_record, run_record = await _load_run_scope(db_session, run_id)
                if buffers.latest_message_payload is not None:
                    checkpoint = build_message_checkpoint(
                        run_id=run_record.id,
                        session_id=session_record.id,
                        checkpoint_kind="run_failed",
                        message=self._runtime_state.get_replay_events(run_id),
                    )
                    write_message_checkpoint(self._run_store, checkpoint)
                termination_requested = self._runtime_state.get_termination_requested(run_id)
                if run_record.status == "cancelled" or termination_requested is not None:
                    await db_session.commit()
                    return

                fail_run(session_record, run_record)
                run_record.error_message = self._stringify_error(exc)
                run_record.output_text = buffers.output_text
                run_record.output_summary = buffers.output_summary
                await db_session.commit()
                await db_session.refresh(run_record)
                if session_record.session_type == "memory":
                    lifecycle = MemoryLifecycle(
                        settings=self._settings,
                        session_factory=self._session_factory,
                        runtime_state=self._runtime_state,
                        submit_run=self._submit_memory_run,
                    )
                    await lifecycle.on_memory_run_terminal(memory_run_id=run_record.id)
                await _publish_run_status_notification(
                    self._notification_hub,
                    "run.updated",
                    run_record,
                )
                agui_adapter = AguiEventAdapter(session_id=session_record.id, run_id=run_id)
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
                terminal_event_emitted = True
        finally:
            if not terminal_event_emitted:
                await self._runtime_state.close_run(run_id)
                logger.debug(
                    "Run runtime state closed run_id={} terminal_event_emitted={}", run_id, terminal_event_emitted
                )

    async def _execute_agent_run(
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
        environment = self._environment_factory.build(workspace_binding)
        background_monitor = BackgroundMonitor(run_id=run_id, runtime_state=self._runtime_state)
        environment.resources.set(BACKGROUND_MONITOR_KEY, background_monitor)
        memory_metadata = run_metadata.get("memory") if isinstance(run_metadata, dict) else None
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
        restored_messages = self._extract_message_history(restore_point)
        agui_adapter = AguiEventAdapter(session_id=session_id, run_id=run_id)

        async with runtime:
            background_monitor.set_core_toolset(runtime.core_toolset)

            if isinstance(environment, Environment):
                runtime.ctx.container_id = self._extract_environment_container_id(environment)
            await self._persist_workspace_sandbox(session_id, workspace_binding, environment)
            logger.debug(
                "Agent runtime entered run_id={} session_id={} container_id={}",
                run_id,
                session_id,
                runtime.ctx.container_id,
            )
            async with stream_agent(
                runtime,
                user_prompt_factory=lambda runtime_obj: self._build_initial_prompt(runtime_obj, input_parts),
                message_history=restored_messages,
                resume_on_error=self._settings.agent_stream_resume_on_error,
                resume_max_attempts=self._settings.agent_stream_resume_max_attempts,
                resume_prompt=self._settings.agent_stream_resume_prompt,
            ) as streamer:
                steering_task = asyncio.create_task(
                    self._forward_runtime_signals(
                        run_id=run_id,
                        runtime=runtime,
                        streamer=streamer,
                    ),
                    name=f"ya-claw-run-{run_id}-signals",
                )
                try:
                    async for stream_event in streamer:
                        for agui_event in agui_adapter.adapt_stream_event(stream_event):
                            await self._runtime_state.append_run_event(run_id, agui_event)
                        if streamer.run is not None:
                            buffers.latest_message_payload = self._build_message_payload(
                                streamer.run,
                                replay_events=self._runtime_state.get_replay_events(run_id),
                            )
                            output = streamer.run.result.output if streamer.run.result else None
                            buffers.output_text = self._stringify_output(output)
                            buffers.output_summary = self._summarize_output(output)
                            if isinstance(stream_event.event, (ModelRequestStartEvent, ModelRequestCompleteEvent)):
                                checkpoint = build_message_checkpoint(
                                    run_id=run_id,
                                    session_id=session_id,
                                    checkpoint_kind=type(stream_event.event).__name__,
                                    message=self._runtime_state.get_replay_events(run_id),
                                )
                                write_message_checkpoint(self._run_store, checkpoint)
                    streamer.raise_if_exception()
                    logger.debug("Agent stream completed run_id={} session_id={}", run_id, session_id)
                finally:
                    steering_task.cancel()
                    await asyncio.gather(steering_task, return_exceptions=True)

                drained_background = await background_monitor.drain_or_cancel(timeout=10.0)
                if not drained_background:
                    logger.warning("YA Claw background subagents cancelled after drain timeout run_id={}", run_id)

                if streamer.run is None:
                    if self._runtime_state.get_termination_requested(run_id) is not None:
                        raise AgentInterrupted()
                    raise RuntimeError("Stream agent completed without run context.")

                buffers.latest_message_payload = self._build_message_payload(
                    streamer.run,
                    replay_events=self._runtime_state.get_replay_events(run_id),
                )
                buffers.claw_metadata = dict(runtime.ctx.claw_metadata)
                buffers.latest_state_payload = self._build_state_payload(
                    runtime.ctx,
                    workspace_binding=workspace_binding,
                    restore_point=restore_point,
                    profile=profile,
                    trigger_type=trigger_type,
                    message_payload=buffers.latest_message_payload,
                )
                output = streamer.run.result.output if streamer.run.result else None
                buffers.output_text = self._stringify_output(output)
                buffers.output_summary = self._summarize_output(output)
                logger.debug(
                    "Agent run artifacts prepared run_id={} message_count={} output_summary_chars={}",
                    run_id,
                    buffers.latest_message_payload.get("message_count")
                    if isinstance(buffers.latest_message_payload, dict)
                    else None,
                    len(buffers.output_summary or ""),
                )

    async def _build_initial_prompt(
        self,
        runtime_obj: AgentRuntime[ClawAgentContext, Any, Environment],
        input_parts: list[InputPart],
    ) -> str | list[Any]:
        mapping = await map_input_parts(input_parts, file_operator=runtime_obj.ctx.file_operator)
        return self._build_user_prompt(mapping)

    async def _forward_runtime_signals(
        self,
        *,
        run_id: str,
        runtime: AgentRuntime[ClawAgentContext, Any, Environment],
        streamer: Any,
    ) -> None:
        while True:
            termination_reason = self._runtime_state.get_termination_requested(run_id)
            if isinstance(termination_reason, str):
                streamer.interrupt()
                return

            steering_batches = self._runtime_state.consume_steering_inputs(run_id)
            for raw_batch in steering_batches:
                logger.debug("Forwarding steering input run_id={} input_parts={}", run_id, len(raw_batch))
                parts = parse_input_parts(list(raw_batch))
                mapping = await map_input_parts(parts, file_operator=runtime.ctx.file_operator)
                content = self._build_user_prompt(mapping)
                runtime.ctx.send_message(BusMessage(content=content, source="user", target="main"))

            await asyncio.sleep(0.1)

    def _resolve_workspace_binding(
        self,
        run_record: RunRecord,
        session_record: SessionRecord,
        profile: ResolvedProfile,
    ) -> WorkspaceBinding:
        metadata: dict[str, Any] = {
            "run_id": run_record.id,
            "session_id": session_record.id,
            "profile_name": profile.name,
            "trigger_type": run_record.trigger_type,
        }
        if isinstance(session_record.session_metadata, dict):
            sandbox = session_record.session_metadata.get("sandbox")
            if isinstance(sandbox, dict):
                metadata["sandbox"] = dict(sandbox)
        binding = self._workspace_provider.resolve(metadata)
        if isinstance(profile.workspace_backend_hint, str) and profile.workspace_backend_hint.strip() != "":
            binding.backend_hint = profile.workspace_backend_hint
            binding.metadata["workspace_backend_hint"] = profile.workspace_backend_hint
        logger.debug(
            "Workspace binding resolved run_id={} session_id={} provider={} backend_hint={} host_path={} virtual_path={} docker_host_path={}",
            run_record.id,
            session_record.id,
            binding.metadata.get("provider"),
            binding.backend_hint,
            binding.host_path,
            binding.virtual_path,
            binding.docker_host_path,
        )
        return binding

    def _extract_environment_container_id(self, environment: Environment) -> str | None:
        if isinstance(environment, SandboxEnvironment):
            value = environment.container_id
            if isinstance(value, str) and value.strip() != "":
                return value.strip()
        return None

    async def _persist_workspace_sandbox(
        self,
        session_id: str,
        workspace_binding: WorkspaceBinding,
        environment: Environment,
    ) -> None:
        sandbox_metadata = build_workspace_sandbox_metadata(binding=workspace_binding, environment=environment)
        if sandbox_metadata is None:
            return

        async with self._session_factory() as db_session:
            session_record = await db_session.get(SessionRecord, session_id)
            if not isinstance(session_record, SessionRecord):
                return
            session_metadata = dict(session_record.session_metadata)
            if session_metadata.get("sandbox") == sandbox_metadata:
                return
            session_metadata["sandbox"] = sandbox_metadata
            session_record.session_metadata = session_metadata
            await db_session.commit()
            logger.debug(
                "Persisted workspace sandbox metadata session_id={} container_id={} container_ref={}",
                session_id,
                sandbox_metadata.get("container_id"),
                sandbox_metadata.get("container_ref"),
            )

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
        return ResumableState.model_validate(raw_state)

    def _extract_message_history(self, restore_point: ResolvedRestorePoint | None) -> list[ModelMessage] | None:
        if restore_point is None:
            return None

        raw_messages: list[Any] | None = None
        if isinstance(restore_point.state, dict):
            state_messages = restore_point.state.get("message_history")
            if isinstance(state_messages, list):
                raw_messages = state_messages

        if not isinstance(raw_messages, list):
            return None
        return cast(list[ModelMessage], ModelMessagesTypeAdapter.validate_python(raw_messages))

    def _build_state_payload(
        self,
        ctx: ClawAgentContext,
        *,
        workspace_binding: WorkspaceBinding,
        restore_point: ResolvedRestorePoint | None,
        profile: ResolvedProfile,
        trigger_type: str,
        message_payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        exported_state = ctx.export_state().model_dump(mode="json")
        message_history = message_payload.get("message_history") if isinstance(message_payload, dict) else None
        if not isinstance(message_history, list):
            message_history = []
        message_count = message_payload.get("message_count") if isinstance(message_payload, dict) else None
        return {
            "container_id": ctx.container_id,
            "context_state": exported_state,
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
            "version": 4,
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

    def _build_message_payload(self, run: Any, *, replay_events: list[dict[str, Any]]) -> dict[str, Any]:
        messages = ModelMessagesTypeAdapter.dump_python(run.all_messages(), mode="json")
        return {
            "events": list(replay_events),
            "message_history": messages,
            "messages": list(replay_events),
            "message_count": len(messages) if isinstance(messages, list) else None,
        }

    def _stringify_output(self, output: Any) -> str | None:
        if output is None:
            return None
        if isinstance(output, str):
            value = output.strip()
            return value or None
        return str(output)

    def _summarize_output(self, output: Any) -> str | None:
        value = self._stringify_output(output)
        if value is None:
            return None
        return value[:4000]

    def _stringify_error(self, exc: Exception) -> str:
        try:
            value = str(exc)
        except Exception:
            value = repr(exc)
        return value[:4000]

    def _submit_memory_run(self, run_id: str) -> bool:
        supervisor = ExecutionSupervisor(
            settings=self._settings,
            session_factory=self._session_factory,
            runtime_state=self._runtime_state,
            workspace_provider=self._workspace_provider,
            environment_factory=self._environment_factory,
            profile_resolver=self._profile_resolver,
            runtime_builder=self._runtime_builder,
            notification_hub=self._notification_hub,
        )
        return supervisor.submit_run(run_id)

    def _serialize_value(self, value: Any) -> Any:
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
    memory_metadata: Any,
) -> dict[str, Any]:
    metadata = {"trigger_type": trigger_type, **run_metadata}
    if isinstance(memory_metadata, dict):
        metadata["memory"] = memory_metadata
    return metadata


def _memory_source_session_id(memory_metadata: Any) -> str | None:
    if not isinstance(memory_metadata, dict):
        return None
    value = memory_metadata.get("source_session_id")
    return value if isinstance(value, str) and value.strip() else None


async def _publish_run_status_notification(
    notification_hub: NotificationHub | None,
    event_type: str,
    run_record: RunRecord,
) -> None:
    if notification_hub is None:
        return
    memory_metadata = run_record.run_metadata.get("memory") if isinstance(run_record.run_metadata, dict) else None
    source_session_id = _memory_source_session_id(memory_metadata)
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
        "output_summary": run_record.output_summary,
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


async def _load_run_scope(db_session: AsyncSession, run_id: str) -> tuple[SessionRecord, RunRecord]:
    run_record = await db_session.get(RunRecord, run_id)
    if not isinstance(run_record, RunRecord):
        raise TypeError(f"Run '{run_id}' was not found.")
    session_record = await db_session.get(SessionRecord, run_record.session_id)
    if not isinstance(session_record, SessionRecord):
        raise TypeError(f"Session '{run_record.session_id}' was not found.")
    return session_record, run_record
