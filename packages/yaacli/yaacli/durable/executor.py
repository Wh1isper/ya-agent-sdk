"""Process-local YAACLI execution coordinator backed by product SQLite state."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import JsonValue, TypeAdapter
from pydantic_ai import (
    DeferredToolRequests,
    DeferredToolResults,
    EnqueuedMessagesEvent,
    ToolDenied,
    UsageLimits,
    UserContent,
)
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, ModelRequest, RetryPromptPart, UserPromptPart
from pydantic_ai.usage import RunUsage
from pydantic_core import to_jsonable_python
from ya_agent_sdk.agents.main import AgentRuntime, RuntimeReadyContext
from ya_agent_sdk.context import NoteManager, ResumableState, StreamEvent, TaskManager, ToolProxyState
from ya_agent_sdk.execution import AgentExecutionHarness, AgentSegmentRequest, AgentSegmentStatus
from ya_agent_sdk.inputs import ActiveRunRegistry, InputDisposition, InputOrigin, RunInputLedger
from ya_agent_sdk.subagents import DelegationCapability

from yaacli.durable.bindings import runtime_bindings
from yaacli.durable.capabilities import DurableInboxPumpCapability
from yaacli.durable.models import (
    ActionBatch,
    ActionState,
    ExecutionCheckpointRecord,
    InputRecord,
    InputState,
    LogicalRunRecord,
    LogicalRunStatus,
    RevisionPayload,
    RuntimeDescriptor,
    utc_now,
)
from yaacli.durable.projections import DURABLE_STEERING_EVENT_NAMES, durable_steering_display_events
from yaacli.durable.restoration import restore_resumable_state_safely
from yaacli.durable.store import SessionStore
from yaacli.environment import TUIEnvironment
from yaacli.session import TUIContext

_USER_CONTENT = TypeAdapter(list[UserContent])
RuntimeFactory = Callable[
    [RuntimeDescriptor, str],
    AgentRuntime[TUIContext, Any, TUIEnvironment],
]
ExecutionEventSink = Callable[[StreamEvent], Awaitable[None]]
DisplayProjectionProvider = Callable[[], Sequence[JsonValue]]
HeadlessHITLPolicy = Literal["wait", "deny"]

logger = logging.getLogger(__name__)


class RuntimePlanUnavailableError(RuntimeError):
    """A persisted execution references a plan this worker cannot execute exactly."""


@dataclass(frozen=True, slots=True)
class RegisteredRuntimePlan:
    """One entered runtime bound to its immutable descriptor and authority."""

    descriptor: RuntimeDescriptor
    runtime: AgentRuntime[TUIContext, Any, TUIEnvironment]
    binding_ref: str
    binding_context: TUIContext


class RuntimePlanRegistry:
    """Exact descriptor dispatch with one independently selected active plan."""

    def __init__(self, active_descriptor_id: str) -> None:
        self._active_descriptor_id = active_descriptor_id
        self._plans: dict[str, RegisteredRuntimePlan] = {}

    @property
    def active(self) -> RegisteredRuntimePlan:
        return self.get(self._active_descriptor_id)

    def register(self, plan: RegisteredRuntimePlan) -> None:
        plan.descriptor.assert_integrity()
        descriptor_id = plan.descriptor.descriptor_id
        existing = self._plans.get(descriptor_id)
        if existing is not None and existing is not plan:
            raise ValueError(f"Runtime descriptor {descriptor_id!r} is already registered")
        self._plans[descriptor_id] = plan

    def get(self, descriptor_id: str) -> RegisteredRuntimePlan:
        try:
            return self._plans[descriptor_id]
        except KeyError as exc:
            raise RuntimePlanUnavailableError(
                f"Runtime descriptor {descriptor_id!r} is not registered by this worker"
            ) from exc

    def activate(self, descriptor_id: str) -> RegisteredRuntimePlan:
        plan = self.get(descriptor_id)
        self._active_descriptor_id = descriptor_id
        return plan

    def list(self) -> tuple[RegisteredRuntimePlan, ...]:
        return tuple(self._plans.values())


class LocalExecutionCoordinator:
    """Coordinate product commands and process-local SDK segment tasks."""

    def __init__(
        self,
        *,
        store: SessionStore,
        runtime_registry: RuntimePlanRegistry,
        execution_harness: AgentExecutionHarness | None = None,
        event_sink: ExecutionEventSink | None = None,
        display_projection_provider: DisplayProjectionProvider | None = None,
    ) -> None:
        self.store = store
        self.runtime_registry = runtime_registry
        self.execution_harness = execution_harness or AgentExecutionHarness()
        self.event_sink = event_sink
        self.display_projection_provider = display_projection_provider
        self._outbox_lock = asyncio.Lock()
        self._tasks: dict[str, asyncio.Task[dict[str, Any]]] = {}
        self._action_events: dict[str, asyncio.Event] = {}
        self._runtime_locks: dict[str, asyncio.Lock] = {}
        self._shutting_down = False

    async def dispatch_outbox(self) -> int:
        """Apply every currently available product command at least once."""
        delivered = 0
        async with self._outbox_lock:
            while commands := self.store.claim_outbox(limit=1):
                for command in commands:
                    try:
                        if command.command_kind == "start_execution":
                            self._start_execution(_required_string(command.payload, "execution_id"))
                        elif command.command_kind == "notify_input":
                            # The product row is the wake authority. PersistedInboxCapability
                            # drains it at the next native graph boundary.
                            pass
                        elif command.command_kind == "notify_action":
                            execution_id = command.aggregate_id
                            self._action_events.setdefault(execution_id, asyncio.Event()).set()
                        elif command.command_kind == "cancel_execution":
                            await self._cancel_execution(command.aggregate_id)
                        elif command.command_kind == "cancel_subagent_execution":
                            await self._cancel_subagent_execution(
                                _required_string(command.payload, "execution_id"),
                                _required_string(command.payload, "owner_scope_id"),
                            )
                        else:
                            raise ValueError(f"Unknown execution command kind {command.command_kind!r}")
                    except Exception as exc:
                        self.store.retry_outbox(command.command_id, str(exc) or repr(exc))
                        raise
                    else:
                        self.store.complete_outbox(command.command_id)
                        delivered += 1
        return delivered

    def _start_execution(self, execution_id: str) -> None:
        existing = self._tasks.get(execution_id)
        if existing is not None:
            return
        execution = self.store.get_execution(execution_id)
        if execution is None:
            raise KeyError(execution_id)
        run = self.store.get_run(execution.logical_run_id)
        if run is None:
            raise KeyError(execution.logical_run_id)
        if run.terminal:
            return
        if run.status is LogicalRunStatus.pending:
            self.store.set_run_status(run.logical_run_id, LogicalRunStatus.running)
        self._tasks[execution_id] = asyncio.create_task(
            self._execute(execution_id),
            name=f"yaacli-execution-{execution_id}",
        )

    async def _cancel_execution(self, execution_id: str) -> None:
        task = self._tasks.get(execution_id)
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

        execution = self.store.get_execution(execution_id)
        if execution is None:
            raise KeyError(execution_id)
        run = self.store.get_run(execution.logical_run_id)
        if run is None:
            raise KeyError(execution.logical_run_id)
        if not run.terminal and run.status is LogicalRunStatus.cancelling:
            await self._commit_cancelled(run)
        if task is not None and task.done():
            self._tasks.pop(execution_id, None)

    async def wait(self, logical_run_id: str) -> LogicalRunRecord:
        """Drive durable commands until the exact logical run is terminal."""
        while True:
            run = self.store.get_run(logical_run_id)
            if run is None:
                raise KeyError(logical_run_id)
            if run.terminal:
                return run

            try:
                await self.dispatch_outbox()
            except Exception:
                logger.warning(
                    "Durable command dispatch failed while waiting for logical run %s",
                    logical_run_id,
                    exc_info=True,
                )

            task = self._tasks.get(run.execution_id)
            if task is None:
                await asyncio.sleep(0.05)
                continue
            done, _pending = await asyncio.wait({task}, timeout=0.05)
            if done:
                await asyncio.shield(task)

    def accept_cancel(self, logical_run_id: str, reason: str) -> None:
        """Durably accept cancellation before attempting process-local dispatch."""
        run = self.store.get_run(logical_run_id)
        if run is None:
            raise KeyError(logical_run_id)
        if run.terminal:
            return
        self.store.set_run_status(
            logical_run_id,
            LogicalRunStatus.cancelling,
            cancellation_reason=reason,
        )
        self.store.enqueue_command(
            "cancel_execution",
            run.execution_id,
            {
                "execution_id": run.execution_id,
                "logical_run_id": logical_run_id,
                "reason": reason,
            },
            command_id=f"cancel:{run.execution_id}",
        )

    async def cancel(self, logical_run_id: str, reason: str) -> None:
        self.accept_cancel(logical_run_id, reason)
        await self.dispatch_outbox()

    async def recover_orphaned_executions(self) -> tuple[str, ...]:
        """Finalize process-owned work that cannot survive a process restart."""
        recovered: list[str] = []
        for session in self.store.list_sessions(limit=1_000_000):
            execution_id = session.active_execution_id
            if execution_id is None:
                continue
            execution = self.store.get_execution(execution_id)
            if execution is None:
                continue
            run = self.store.get_run(execution.logical_run_id)
            if run is None or run.terminal or run.status is LogicalRunStatus.pending:
                continue
            if run.status is LogicalRunStatus.cancelling:
                await self._commit_cancelled(run)
            else:
                await self._commit_interrupted(
                    run,
                    reason="Execution was interrupted by process restart before its active segment completed.",
                )
            recovered.append(run.logical_run_id)
        return tuple(recovered)

    async def shutdown(self) -> None:
        self._shutting_down = True
        active = [task for task in self._tasks.values() if not task.done()]
        for task in active:
            task.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)

    async def _cancel_subagent_execution(self, execution_id: str, owner_scope_id: str) -> None:
        record = None
        for plan in self.runtime_registry.list():
            for capability in plan.runtime.capabilities:
                if not isinstance(capability, DelegationCapability):
                    continue
                try:
                    record = await capability.service.admin_get(execution_id)
                except KeyError:
                    continue
                break
            if record is not None:
                break
        if record is None:
            raise KeyError(execution_id)
        if record.owner_scope_id != owner_scope_id:
            raise PermissionError(f"Subagent execution {execution_id!r} is not owned by scope {owner_scope_id!r}")
        descriptor_id = record.parent_runtime_descriptor_id
        if descriptor_id is None:
            raise RuntimePlanUnavailableError(f"Subagent execution {execution_id!r} has no owning runtime descriptor")
        owner_plan = self.runtime_registry.get(descriptor_id)
        for capability in owner_plan.runtime.capabilities:
            if isinstance(capability, DelegationCapability):
                await capability.service.cancel(
                    execution_id,
                    caller_scope_id=owner_scope_id,
                )
                return
        raise RuntimePlanUnavailableError(f"Runtime descriptor {descriptor_id!r} does not own a delegation service")

    async def _execute(self, execution_id: str) -> dict[str, Any]:
        execution = self.store.get_execution(execution_id)
        if execution is None:
            raise KeyError(execution_id)
        run = self.store.get_run(execution.logical_run_id)
        if run is None:
            raise KeyError(execution.logical_run_id)
        if run.terminal:
            return await self._replay_terminal(run)
        try:
            plan = self._validate_execution_plan(run, execution)
        except BaseException as exc:
            await self._commit_plan_failure(run, exc)
            raise

        if run.status is LogicalRunStatus.pending:
            run = self.store.set_run_status(run.logical_run_id, LogicalRunStatus.running)
        elif run.status is LogicalRunStatus.cancelling:
            return await self._commit_cancelled(run)

        initial = self._initial_input(run.logical_run_id)
        prompt = _USER_CONTENT.validate_python(initial.content)
        if initial.state is not InputState.applied:
            self.store.transition_input(initial.input_id, initial.state, InputState.applied)

        runtime_lock = self._runtime_locks.setdefault(plan.descriptor.descriptor_id, asyncio.Lock())
        execution_binding_ref = f"{plan.binding_ref}:execution:{execution_id}"
        latest_history: list[ModelMessage] | None = None
        latest_context = plan.runtime.ctx
        resume_state: ResumableState | None = None
        cumulative_usage = RunUsage()
        segment_index = 0
        deferred_results: DeferredToolResults | None = None
        current_prompt: str | Sequence[UserContent] | None = (
            prompt[0] if len(prompt) == 1 and isinstance(prompt[0], str) else prompt
        )
        try:
            while True:
                request_limit = plan.descriptor.main_plan_manifest.request_limit
                remaining_requests = request_limit - cumulative_usage.requests
                if remaining_requests <= 0:
                    raise RuntimeError(f"Execution exhausted the cumulative model request limit of {request_limit}.")

                async with runtime_lock:
                    context = self._new_execution_context(run, plan.runtime)
                    if segment_index == 0:
                        latest_history = self._restore_head(run, context)
                    elif resume_state is not None:
                        restore_resumable_state_safely(resume_state, context)
                        context.run_input_ledger.logical_run_id = run.logical_run_id
                    context.durable_binding_ref = execution_binding_ref
                    context.durable_logical_run_id = run.logical_run_id
                    context.delegation_scope_id = run.session_id
                    plan.runtime.ctx = context
                    latest_context = context
                    runtime_bindings.register(execution_binding_ref, context, self.store)
                    is_initial_segment = segment_index == 0

                    async def prepare_segment(
                        ready: RuntimeReadyContext[TUIContext, Any, TUIEnvironment],
                        *,
                        initial_segment: bool = is_initial_segment,
                    ) -> None:
                        ready.runtime.ctx.durable_logical_run_id = run.logical_run_id
                        ready.runtime.ctx.durable_binding_ref = execution_binding_ref
                        ready.runtime.ctx.delegation_scope_id = run.session_id
                        if initial_segment:
                            self._record_initial(ready.runtime.ctx, initial, prompt)

                    request: AgentSegmentRequest[TUIContext, Any, TUIEnvironment] = AgentSegmentRequest(
                        user_prompt=current_prompt,
                        message_history=latest_history,
                        deferred_tool_results=deferred_results,
                        usage_limits=UsageLimits(request_limit=remaining_requests),
                        on_runtime_ready=prepare_segment,
                    )
                    try:
                        async with self.execution_harness.stream_segment(plan.runtime, request) as segment:
                            async for event in segment:
                                if isinstance(event.event, EnqueuedMessagesEvent):
                                    # Native enqueue confirmation can precede the
                                    # capability's next node hook. Reconcile before
                                    # host projection and terminal failure fencing.
                                    DurableInboxPumpCapability.reconcile_applied_enqueue(
                                        plan.runtime.ctx,
                                        event.event.enqueue_id,
                                    )
                                if self.event_sink is not None:
                                    await self.event_sink(event)
                            segment.raise_if_exception()
                            outcome = segment.outcome()
                    finally:
                        runtime_bindings.unregister(execution_binding_ref, context)

                latest_context = plan.runtime.ctx
                latest_history = list(outcome.checkpoint.messages)
                resume_state = outcome.checkpoint.state
                cumulative_usage.incr(outcome.checkpoint.usage.total_usage)
                checkpoint_payload = self._revision_payload(
                    latest_context,
                    run,
                    latest_history,
                    cumulative_usage,
                    {},
                )
                deferred_payload = (
                    cast(dict[str, JsonValue], to_jsonable_python(outcome.deferred_requests))
                    if outcome.deferred_requests is not None
                    else None
                )
                now = utc_now()
                self.store.put_execution_checkpoint(
                    ExecutionCheckpointRecord(
                        execution_id=execution_id,
                        logical_run_id=run.logical_run_id,
                        segment_index=segment_index,
                        segment_status=outcome.status.value,
                        payload=checkpoint_payload,
                        deferred_requests=deferred_payload,
                        created_at=now,
                        updated_at=now,
                    )
                )
                if outcome.status is AgentSegmentStatus.completed:
                    await self._reconcile_subagent_deliveries(plan.runtime, latest_context)
                    return await self._commit_success(
                        run,
                        latest_context,
                        latest_history,
                        outcome.output,
                        cumulative_usage,
                    )

                requests = outcome.deferred_requests
                if requests is None or (not requests.approvals and not requests.calls):
                    raise RuntimeError("Agent returned an empty DeferredToolRequests payload")
                batch = self._persist_action_batch(run, segment_index, requests)
                batch = await self._resolve_action_batch(
                    run,
                    batch,
                    plan.descriptor.main_plan_manifest.hitl_policy,
                )
                deferred_results = self._build_deferred_results(requests, batch)
                self.store.set_run_status(run.logical_run_id, LogicalRunStatus.running)
                current_prompt = None
                segment_index += 1
        except asyncio.CancelledError:
            current = self.store.get_run(run.logical_run_id) or run
            if current.status is LogicalRunStatus.cancelling:
                return await self._commit_cancelled(current)
            return await self._commit_interrupted(
                current,
                reason=(
                    "Execution was interrupted during worker shutdown."
                    if self._shutting_down
                    else "Execution was interrupted before its active segment completed."
                ),
            )
        except BaseException as exc:
            await self._commit_failure(run, exc)
            raise
        finally:
            latest_context.durable_logical_run_id = None
            self._action_events.pop(execution_id, None)

    async def _reconcile_subagent_deliveries(
        self,
        runtime: AgentRuntime[TUIContext, Any, TUIEnvironment],
        context: TUIContext,
    ) -> None:
        for capability in runtime.capabilities:
            if isinstance(capability, DelegationCapability):
                await capability.service.deliver_pending(context)

    def _validate_execution_plan(self, run: LogicalRunRecord, execution: Any) -> RegisteredRuntimePlan:
        descriptor = self.store.get_descriptor(run.descriptor_id)
        if descriptor is None:
            raise RuntimePlanUnavailableError(f"Runtime descriptor {run.descriptor_id!r} is unavailable")
        if (
            descriptor.descriptor_id != execution.descriptor_id
            or descriptor.plan_fingerprint != execution.plan_fingerprint
            or descriptor.executable_version != execution.executable_version
        ):
            raise RuntimePlanUnavailableError(f"Execution {execution.execution_id!r} has inconsistent plan identity")
        plan = self.runtime_registry.get(descriptor.descriptor_id)
        if (
            plan.descriptor.plan_fingerprint != descriptor.plan_fingerprint
            or plan.descriptor.behavior_payload() != descriptor.behavior_payload()
        ):
            raise RuntimePlanUnavailableError(
                f"Runtime descriptor {descriptor.descriptor_id!r} does not match its registered worker plan"
            )
        return plan

    def _new_execution_context(
        self,
        run: LogicalRunRecord,
        runtime: AgentRuntime[TUIContext, Any, TUIEnvironment],
    ) -> TUIContext:
        context = runtime.ctx.prepare_new_run()
        if not isinstance(context, TUIContext):  # pragma: no cover - factory contract
            raise TypeError("YAACLI execution requires TUIContext")
        context.task_manager = TaskManager()
        context.note_manager = NoteManager()
        context.active_run_registry = ActiveRunRegistry()
        context.agent_stream_info = {}
        context.files_to_inspect = []
        context.tool_proxy = ToolProxyState()
        context.durable_logical_run_id = run.logical_run_id
        context.delegation_scope_id = run.session_id
        context.run_input_ledger = RunInputLedger(logical_run_id=run.logical_run_id)
        context.input_router = None
        return context

    async def _replay_terminal(self, run: LogicalRunRecord) -> dict[str, Any]:
        revision = self.store.get_revision_for_run(run.logical_run_id)
        if revision is None:
            raise RuntimeError(f"Terminal logical run {run.logical_run_id!r} has no revision")
        terminal = dict(revision.terminal)
        self.store.append_event(
            run.session_id,
            _terminal_event_type(run.status),
            {"revision_id": revision.revision_id, **terminal},
            event_id=f"terminal:{run.logical_run_id}",
            logical_run_id=run.logical_run_id,
        )
        return terminal

    def _restore_head(self, run: LogicalRunRecord, context: TUIContext) -> list[ModelMessage] | None:
        if run.expected_head_revision_id is None:
            return None
        revision = self.store.get_revision(run.expected_head_revision_id)
        if revision is None:
            raise RuntimeError(f"Expected revision {run.expected_head_revision_id!r} is unavailable")
        if revision.resumable_state:
            state = ResumableState.model_validate(revision.resumable_state)
            restore_resumable_state_safely(state, context)
            context.durable_logical_run_id = run.logical_run_id
            context.run_input_ledger = RunInputLedger(logical_run_id=run.logical_run_id)
        return ModelMessagesTypeAdapter.validate_python(revision.message_history)

    def _initial_input(self, logical_run_id: str) -> InputRecord:
        for item in self.store.list_inputs(logical_run_id):
            if item.order_index == 0:
                return item
        raise RuntimeError(f"Logical run {logical_run_id!r} has no initial input")

    @staticmethod
    def _record_initial(context: TUIContext, item: InputRecord, content: list[UserContent]) -> None:
        if context.run_input_ledger.find(item.input_id) is not None:
            return
        prompt_content: str | list[UserContent]
        prompt_content = content[0] if len(content) == 1 and isinstance(content[0], str) else content
        record = context.run_input_ledger.accept(
            [ModelRequest(parts=[UserPromptPart(content=prompt_content)])],
            origin=InputOrigin.user,
            priority="asap",
            input_id=item.input_id,
        )
        record.disposition = InputDisposition.applied

    def _persist_action_batch(
        self,
        run: LogicalRunRecord,
        segment_index: int,
        requests: DeferredToolRequests,
    ) -> ActionBatch:
        items: list[dict[str, Any]] = []
        for request in requests.approvals:
            items.append({
                "tool_call_id": request.tool_call_id,
                "decision_kind": "approval",
                "request": cast(dict[str, Any], to_jsonable_python(request)),
            })
        for request in requests.calls:
            items.append({
                "tool_call_id": request.tool_call_id,
                "decision_kind": "external_result",
                "request": cast(dict[str, Any], to_jsonable_python(request)),
            })
        return self.store.create_action_batch(
            run.logical_run_id,
            items,
            batch_id=f"{run.logical_run_id}:{segment_index}:actions",
        )

    async def _resolve_action_batch(
        self,
        run: LogicalRunRecord,
        batch: ActionBatch,
        hitl_policy: HeadlessHITLPolicy,
    ) -> ActionBatch:
        if hitl_policy == "deny":
            for item in batch.items:
                if item.state is not ActionState.pending:
                    continue
                decision: dict[str, Any]
                if item.decision_kind == "approval":
                    decision = {"approved": False, "message": "Headless mode denies HITL requests by policy."}
                else:
                    decision = {
                        "result": "Headless mode denies deferred calls by policy.",
                        "denied": True,
                    }
                batch = self.store.decide_action(
                    item.action_item_id,
                    decision_id=f"headless-deny:{item.action_item_id}",
                    decision=decision,
                    actor="headless-policy",
                )
            return batch

        event = self._action_events.setdefault(run.execution_id, asyncio.Event())
        while batch.state is ActionState.pending:
            await event.wait()
            event.clear()
            refreshed = self.store.get_action_batch(batch.batch_id)
            if refreshed is None:
                raise RuntimeError(f"Pending action batch {batch.batch_id!r} disappeared")
            batch = refreshed
            current = self.store.get_run(run.logical_run_id)
            if current is None or current.status is LogicalRunStatus.cancelling:
                raise asyncio.CancelledError
        return batch

    @staticmethod
    def _build_deferred_results(
        requests: DeferredToolRequests,
        batch: ActionBatch,
    ) -> DeferredToolResults:
        decisions = {item.tool_call_id: item for item in batch.items}
        results = DeferredToolResults()
        for request in requests.approvals:
            item = decisions[request.tool_call_id]
            decision = item.decision or {}
            if decision.get("approved") is True:
                results.approvals[request.tool_call_id] = True
            else:
                results.approvals[request.tool_call_id] = ToolDenied(
                    message=str(decision.get("message") or "Tool call denied")
                )
        for request in requests.calls:
            item = decisions[request.tool_call_id]
            decision = item.decision or {}
            value = decision.get("result")
            if not isinstance(value, str):
                value = json.dumps(value, ensure_ascii=False, sort_keys=True)
            results.calls[request.tool_call_id] = RetryPromptPart(
                content=value,
                tool_name=request.tool_name,
                tool_call_id=request.tool_call_id,
            )
        return results

    async def _commit_success(
        self,
        run: LogicalRunRecord,
        context: TUIContext,
        history: list[ModelMessage],
        output: Any,
        usage: RunUsage,
    ) -> dict[str, Any]:
        terminal = {
            "status": LogicalRunStatus.completed.value,
            "output": to_jsonable_python(output),
        }
        self.store.commit_terminal(
            run.logical_run_id,
            commit_kind="success",
            payload=self._revision_payload(context, run, history, usage, terminal),
            terminal_status=LogicalRunStatus.completed,
            event_type="RUN_FINISHED",
        )
        return terminal

    async def _commit_plan_failure(self, run: LogicalRunRecord, error: BaseException) -> dict[str, Any]:
        terminal: dict[str, JsonValue] = {
            "status": LogicalRunStatus.failed.value,
            "error_type": type(error).__name__,
            "error": str(error) or repr(error),
        }
        payload = self._stable_payload(run)
        self.store.commit_terminal(
            run.logical_run_id,
            commit_kind="failure",
            payload=payload.model_copy(update={"terminal": terminal}),
            terminal_status=LogicalRunStatus.failed,
            event_type="RUN_ERROR",
        )
        return terminal

    async def _commit_failure(
        self,
        run: LogicalRunRecord,
        error: BaseException,
    ) -> dict[str, Any]:
        terminal: dict[str, JsonValue] = {
            "status": LogicalRunStatus.failed.value,
            "error_type": type(error).__name__,
            "error": str(error) or repr(error),
        }
        payload = self._stable_payload(run)
        self.store.commit_terminal(
            run.logical_run_id,
            commit_kind="failure",
            payload=payload.model_copy(update={"terminal": terminal}),
            terminal_status=LogicalRunStatus.failed,
            event_type="RUN_ERROR",
        )
        return terminal

    async def _commit_cancelled(self, run: LogicalRunRecord) -> dict[str, Any]:
        terminal: dict[str, JsonValue] = {
            "status": LogicalRunStatus.cancelled.value,
            "reason": run.cancellation_reason or "cancelled",
        }
        payload = self._stable_payload(run)
        self.store.commit_terminal(
            run.logical_run_id,
            commit_kind="cancelled",
            payload=payload.model_copy(update={"terminal": terminal}),
            terminal_status=LogicalRunStatus.cancelled,
            event_type="run_cancelled",
        )
        return terminal

    async def _commit_interrupted(self, run: LogicalRunRecord, *, reason: str) -> dict[str, Any]:
        terminal: dict[str, JsonValue] = {
            "status": LogicalRunStatus.interrupted.value,
            "reason": reason,
        }
        payload = self._stable_payload(run)
        self.store.commit_terminal(
            run.logical_run_id,
            commit_kind="interrupted",
            payload=payload.model_copy(update={"terminal": terminal}),
            terminal_status=LogicalRunStatus.interrupted,
            event_type="run_interrupted",
        )
        return terminal

    def _stable_payload(self, run: LogicalRunRecord) -> RevisionPayload:
        checkpoint = self.store.get_execution_checkpoint(run.execution_id)
        if checkpoint is not None:
            payload = checkpoint.payload
        elif run.expected_head_revision_id is None:
            payload = RevisionPayload()
        else:
            revision = self.store.get_revision(run.expected_head_revision_id)
            if revision is None:
                raise RuntimeError(f"Expected revision {run.expected_head_revision_id!r} is unavailable")
            payload = RevisionPayload(
                message_history=revision.message_history,
                resumable_state=revision.resumable_state,
                input_ledger=revision.input_ledger,
                display_projection=revision.display_projection,
                usage=revision.usage,
            )
        return payload.model_copy(
            update={
                "display_projection": self._stable_terminal_display_projection(
                    run,
                    payload.display_projection,
                )
            }
        )

    def _stable_terminal_display_projection(
        self,
        run: LogicalRunRecord,
        stable: Sequence[JsonValue],
    ) -> list[JsonValue]:
        merged = list(stable)
        if self.display_projection_provider is not None:
            merged = _merge_durable_display_projection(
                merged,
                self.display_projection_provider(),
            )
        return self._with_durable_steering_projection(run, merged)

    def _current_terminal_display_projection(self, run: LogicalRunRecord) -> list[JsonValue]:
        current = list(self.display_projection_provider()) if self.display_projection_provider is not None else []
        return self._with_durable_steering_projection(run, current)

    def _with_durable_steering_projection(
        self,
        run: LogicalRunRecord,
        projection: Sequence[JsonValue],
    ) -> list[JsonValue]:
        return _merge_durable_display_projection(
            projection,
            durable_steering_display_events(
                run.session_id,
                self.store.list_inputs(run.logical_run_id),
            ),
        )

    def _revision_payload(
        self,
        context: TUIContext,
        run: LogicalRunRecord,
        history: list[ModelMessage],
        usage: Any,
        terminal: dict[str, Any],
    ) -> RevisionPayload:
        product_inputs = [item.model_dump(mode="json") for item in self.store.list_inputs(run.logical_run_id)]
        return RevisionPayload(
            message_history=cast(list[Any], ModelMessagesTypeAdapter.dump_python(history, mode="json")),
            resumable_state=cast(
                dict[str, Any],
                context.export_state(include_usage_ledger=False).model_dump(mode="json"),
            ),
            input_ledger=cast(
                dict[str, JsonValue],
                {
                    "product_records": product_inputs,
                    "native": context.run_input_ledger.model_dump(mode="json"),
                },
            ),
            display_projection=self._current_terminal_display_projection(run),
            usage=cast(dict[str, Any], to_jsonable_python(usage)),
            terminal=terminal,
        )


class LocalExecutionWorker:
    """Own entered runtime plans and their process-local execution coordinator."""

    def __init__(
        self,
        *,
        store: SessionStore,
        runtime_registry: RuntimePlanRegistry,
        coordinator: LocalExecutionCoordinator,
    ) -> None:
        self.store = store
        self.runtime_registry = runtime_registry
        self.coordinator = coordinator
        self._closed = False
        self._dispatcher_task: asyncio.Task[None] | None = None

    @property
    def runtime(self) -> AgentRuntime[TUIContext, Any, TUIEnvironment]:
        return self.runtime_registry.active.runtime

    @property
    def binding_ref(self) -> str:
        return self.runtime_registry.active.binding_ref

    @property
    def descriptor(self) -> RuntimeDescriptor:
        return self.runtime_registry.active.descriptor

    def activate(self, descriptor_id: str) -> RegisteredRuntimePlan:
        return self.runtime_registry.activate(descriptor_id)

    @classmethod
    async def create(
        cls,
        *,
        store: SessionStore,
        state_path: Path,
        active_descriptor: RuntimeDescriptor,
        runtime_factory: RuntimeFactory,
        available_descriptors: Sequence[RuntimeDescriptor] = (),
        executable_version: str | None = None,
        event_sink: ExecutionEventSink | None = None,
        display_projection_provider: DisplayProjectionProvider | None = None,
    ) -> LocalExecutionWorker:
        state_path = state_path.expanduser().resolve()
        if executable_version is None:
            from yaacli.runtime_identity import runtime_executable_version

            executable_version = runtime_executable_version()
        base_binding_ref = f"yaacli-runtime:{uuid.uuid5(uuid.NAMESPACE_URL, str(state_path))}"
        registry = RuntimePlanRegistry(active_descriptor.descriptor_id)
        coordinator = LocalExecutionCoordinator(
            store=store,
            runtime_registry=registry,
            event_sink=event_sink,
            display_projection_provider=display_projection_provider,
        )
        await coordinator.recover_orphaned_executions()
        from yaacli.durable.sqlite import SQLiteSessionStore
        from yaacli.durable.subagents import SQLiteSubagentExecutionStore

        if isinstance(store, SQLiteSessionStore):
            child_store = SQLiteSubagentExecutionStore(store.path)
            try:
                child_store.recover_orphaned_executions()
            finally:
                child_store.close_sync()

        descriptors: dict[str, RuntimeDescriptor] = {}
        for descriptor in (*available_descriptors, *store.list_nonterminal_descriptors(), active_descriptor):
            descriptor.assert_integrity()
            existing = descriptors.get(descriptor.descriptor_id)
            if existing is not None and existing.behavior_payload() != descriptor.behavior_payload():
                raise RuntimePlanUnavailableError(
                    f"Runtime descriptor {descriptor.descriptor_id!r} has conflicting persisted content"
                )
            descriptors[descriptor.descriptor_id] = descriptor

        entered: list[RegisteredRuntimePlan] = []
        try:
            for descriptor in descriptors.values():
                if descriptor.executable_version != executable_version:
                    raise RuntimePlanUnavailableError(
                        f"Runtime descriptor {descriptor.descriptor_id!r} requires unavailable executable "
                        f"{descriptor.executable_version!r}"
                    )
                binding_ref = f"{base_binding_ref}:plan:{descriptor.plan_fingerprint}"
                try:
                    runtime = runtime_factory(descriptor, binding_ref)
                except BaseException as exc:
                    raise RuntimePlanUnavailableError(
                        f"Runtime descriptor {descriptor.descriptor_id!r} cannot be reconstructed: {exc}"
                    ) from exc
                await runtime.__aenter__()
                runtime.ctx.runtime_descriptor_id = descriptor.descriptor_id
                plan = RegisteredRuntimePlan(
                    descriptor=descriptor,
                    runtime=runtime,
                    binding_ref=binding_ref,
                    binding_context=runtime.ctx,
                )
                entered.append(plan)
                runtime_bindings.register(binding_ref, runtime.ctx, store)
                registry.register(plan)
        except BaseException:
            for plan in reversed(entered):
                runtime_bindings.unregister(plan.binding_ref, plan.binding_context)
                await plan.runtime.__aexit__(None, None, None)
            raise

        worker = cls(store=store, runtime_registry=registry, coordinator=coordinator)
        store.recover_outbox()
        try:
            await coordinator.dispatch_outbox()
        except Exception:
            logger.exception("Initial execution outbox reconciliation failed; retrying in background")
        worker._dispatcher_task = asyncio.create_task(
            worker._dispatch_loop(),
            name="yaacli-execution-outbox",
        )
        return worker

    async def _dispatch_loop(self) -> None:
        while True:
            try:
                await self.coordinator.dispatch_outbox()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Execution outbox dispatch failed; command remains retryable")
            await asyncio.sleep(0.25)

    async def __aenter__(self) -> LocalExecutionWorker:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._dispatcher_task is not None:
            self._dispatcher_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._dispatcher_task
            self._dispatcher_task = None
        await self.coordinator.shutdown()
        for plan in reversed(self.runtime_registry.list()):
            await plan.runtime.__aexit__(None, None, None)
            runtime_bindings.unregister(plan.binding_ref, plan.binding_context)


def _merge_durable_display_projection(
    stable: Sequence[JsonValue],
    live: Sequence[JsonValue],
) -> list[JsonValue]:
    """Merge only durable-input UI facts into a stable terminal projection."""
    merged = list(stable)
    identities = {identity for event in merged if (identity := _durable_display_event_identity(event)) is not None}
    for event in live:
        identity = _durable_display_event_identity(event)
        if identity is None or identity in identities:
            continue
        merged.append(event)
        identities.add(identity)
    return merged


def _durable_display_event_identity(event: JsonValue) -> tuple[str, str] | None:
    if not isinstance(event, dict) or event.get("type") != "CUSTOM":
        return None
    name = event.get("name")
    value = event.get("value")
    if not isinstance(name, str) or name not in DURABLE_STEERING_EVENT_NAMES or not isinstance(value, dict):
        return None
    projection_key = value.get("projection_key")
    if not isinstance(projection_key, str) or not projection_key:
        return None
    return name, projection_key


def _terminal_event_type(status: LogicalRunStatus) -> str:
    if status is LogicalRunStatus.completed:
        return "RUN_FINISHED"
    if status is LogicalRunStatus.failed:
        return "RUN_ERROR"
    if status is LogicalRunStatus.cancelled:
        return "run_cancelled"
    if status is LogicalRunStatus.interrupted:
        return "run_interrupted"
    raise ValueError(f"Logical run status {status.value!r} is not terminal")


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Outbox payload field {key!r} must be a non-empty string")
    return value
