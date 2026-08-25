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
from ya_agent_sdk.execution import AgentExecutionHarness, AgentSegment, AgentSegmentRequest, AgentSegmentStatus
from ya_agent_sdk.inputs import ActiveRunRegistry, InputDisposition, InputOrigin, RunInputLedger
from ya_agent_sdk.subagents import DelegationCapability

from yaacli.display_replay import BoundedDisplayReplay
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
    utc_now,
)
from yaacli.durable.projections import (
    DURABLE_STEERING_EVENT_NAMES,
    STEERING_ACCEPTED_EVENT_NAME,
    STEERING_APPLIED_EVENT_NAME,
    durable_steering_display_events,
)
from yaacli.durable.restoration import restore_resumable_state_safely
from yaacli.durable.store import SessionStore
from yaacli.environment import TUIEnvironment
from yaacli.session import TUIContext

_USER_CONTENT = TypeAdapter(list[UserContent])
RuntimeBuilder = Callable[
    [str],
    AgentRuntime[TUIContext, Any, TUIEnvironment],
]
ExecutionEventSink = Callable[[StreamEvent], Awaitable[None]]
DisplayProjectionProvider = Callable[[], Sequence[JsonValue]]
HeadlessHITLPolicy = Literal["wait", "deny"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LocalRuntimeSpec:
    """Process-local instructions for building one selectable runtime."""

    runtime_id: str
    build: RuntimeBuilder
    request_limit: int
    hitl_policy: HeadlessHITLPolicy


@dataclass(frozen=True, slots=True)
class LocalRuntime:
    """One entered runtime owned exclusively by the current process."""

    runtime_id: str
    runtime: AgentRuntime[TUIContext, Any, TUIEnvironment]
    binding_ref: str
    binding_context: TUIContext
    request_limit: int
    hitl_policy: HeadlessHITLPolicy


class LocalExecutionCoordinator:
    """Coordinate product commands and process-local SDK segment tasks."""

    def __init__(
        self,
        *,
        store: SessionStore,
        runtimes: dict[str, LocalRuntime],
        active_runtime_id: str,
        execution_harness: AgentExecutionHarness | None = None,
        event_sink: ExecutionEventSink | None = None,
        display_projection_provider: DisplayProjectionProvider | None = None,
    ) -> None:
        self.store = store
        self._runtimes = runtimes
        self._active_runtime_id = active_runtime_id
        self.execution_harness = execution_harness or AgentExecutionHarness()
        self.event_sink = event_sink
        self.display_projection_provider = display_projection_provider
        self._tasks: dict[str, asyncio.Task[dict[str, Any]]] = {}
        self._execution_runtimes: dict[str, LocalRuntime] = {}
        self._action_events: dict[str, asyncio.Event] = {}
        self._runtime_locks: dict[str, asyncio.Lock] = {}
        self._shutting_down = False

    @property
    def active_runtime(self) -> LocalRuntime:
        return self._runtimes[self._active_runtime_id]

    def activate(self, runtime_id: str) -> LocalRuntime:
        plan = self._runtimes[runtime_id]
        self._active_runtime_id = runtime_id
        return plan

    def start(self, execution_id: str) -> None:
        """Start one execution with the runtime selected by this process."""
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
        self._execution_runtimes[execution_id] = self.active_runtime
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

    def notify_input(self, logical_run_id: str) -> None:
        """Confirm that newly persisted input belongs to a process-local run."""
        run = self.store.get_run(logical_run_id)
        if run is None:
            raise KeyError(logical_run_id)
        if run.execution_id not in self._tasks:
            raise RuntimeError(f"Logical run {logical_run_id!r} is not owned by this process")

    def notify_action(self, logical_run_id: str) -> None:
        """Wake one process-local run after its action decision is persisted."""
        run = self.store.get_run(logical_run_id)
        if run is None:
            raise KeyError(logical_run_id)
        if run.execution_id not in self._tasks:
            raise RuntimeError(f"Logical run {logical_run_id!r} is not owned by this process")
        self._action_events.setdefault(run.execution_id, asyncio.Event()).set()

    async def wait(self, logical_run_id: str) -> LogicalRunRecord:
        """Wait for one execution started by this process."""
        run = self.store.get_run(logical_run_id)
        if run is None:
            raise KeyError(logical_run_id)
        if run.terminal:
            return run
        task = self._tasks.get(run.execution_id)
        if task is None:
            raise RuntimeError(f"Logical run {logical_run_id!r} was not started by this process")
        await asyncio.shield(task)
        terminal = self.store.get_run(logical_run_id)
        if terminal is None:
            raise KeyError(logical_run_id)
        return terminal

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

    async def cancel(self, logical_run_id: str, reason: str) -> None:
        self.accept_cancel(logical_run_id, reason)
        run = self.store.get_run(logical_run_id)
        if run is None:
            raise KeyError(logical_run_id)
        await self._cancel_execution(run.execution_id)

    async def shutdown(self) -> None:
        self._shutting_down = True
        active = [task for task in self._tasks.values() if not task.done()]
        for task in active:
            task.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)

    async def _execute(self, execution_id: str) -> dict[str, Any]:
        execution = self.store.get_execution(execution_id)
        if execution is None:
            raise KeyError(execution_id)
        run = self.store.get_run(execution.logical_run_id)
        if run is None:
            raise KeyError(execution.logical_run_id)
        if run.terminal:
            return await self._replay_terminal(run)
        plan = self._execution_runtimes.get(execution_id)
        if plan is None:
            raise RuntimeError(f"Execution {execution_id!r} is not owned by this process")

        if run.status is LogicalRunStatus.pending:
            run = self.store.set_run_status(run.logical_run_id, LogicalRunStatus.running)
        elif run.status is LogicalRunStatus.cancelling:
            return await self._commit_cancelled(run)

        initial = self._initial_input(run.logical_run_id)
        prompt = _USER_CONTENT.validate_python(initial.content)
        if initial.state is not InputState.applied:
            self.store.transition_input(initial.input_id, initial.state, InputState.applied)

        runtime_lock = self._runtime_locks.setdefault(plan.runtime_id, asyncio.Lock())
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
        terminal_recovery: RevisionPayload | None = None
        try:
            while True:
                request_limit = plan.request_limit
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
                            try:
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
                            except BaseException:
                                terminal_recovery = self._capture_terminal_recovery(
                                    run,
                                    context,
                                    segment,
                                    stable_history=latest_history or [],
                                    current_prompt=current_prompt,
                                    cumulative_usage=cumulative_usage,
                                )
                                raise
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
                    plan.hitl_policy,
                )
                deferred_results = self._build_deferred_results(requests, batch)
                self.store.set_run_status(run.logical_run_id, LogicalRunStatus.running)
                current_prompt = None
                segment_index += 1
        except asyncio.CancelledError:
            current = self.store.get_run(run.logical_run_id) or run
            if current.status is LogicalRunStatus.cancelling:
                return await self._commit_cancelled(current, recovery=terminal_recovery)
            return await self._commit_interrupted(
                current,
                reason=(
                    "Execution was interrupted during worker shutdown."
                    if self._shutting_down
                    else "Execution was interrupted before its active segment completed."
                ),
                recovery=terminal_recovery,
            )
        except BaseException as exc:
            await self._commit_failure(run, exc, recovery=terminal_recovery)
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
        *,
        recovery: RevisionPayload | None = None,
    ) -> dict[str, Any]:
        terminal: dict[str, JsonValue] = {
            "status": LogicalRunStatus.failed.value,
            "error_type": type(error).__name__,
            "error": str(error) or repr(error),
        }
        payload = recovery or self._stable_payload(run)
        self.store.commit_terminal(
            run.logical_run_id,
            commit_kind="failure",
            payload=payload.model_copy(update={"terminal": terminal}),
            terminal_status=LogicalRunStatus.failed,
            event_type="RUN_ERROR",
        )
        return terminal

    async def _commit_cancelled(
        self,
        run: LogicalRunRecord,
        *,
        recovery: RevisionPayload | None = None,
    ) -> dict[str, Any]:
        terminal: dict[str, JsonValue] = {
            "status": LogicalRunStatus.cancelled.value,
            "reason": run.cancellation_reason or "cancelled",
        }
        payload = recovery or self._stable_payload(run)
        self.store.commit_terminal(
            run.logical_run_id,
            commit_kind="cancelled",
            payload=payload.model_copy(update={"terminal": terminal}),
            terminal_status=LogicalRunStatus.cancelled,
            event_type="run_cancelled",
        )
        return terminal

    async def _commit_interrupted(
        self,
        run: LogicalRunRecord,
        *,
        reason: str,
        recovery: RevisionPayload | None = None,
    ) -> dict[str, Any]:
        terminal: dict[str, JsonValue] = {
            "status": LogicalRunStatus.interrupted.value,
            "reason": reason,
        }
        payload = recovery or self._stable_payload(run)
        self.store.commit_terminal(
            run.logical_run_id,
            commit_kind="interrupted",
            payload=payload.model_copy(update={"terminal": terminal}),
            terminal_status=LogicalRunStatus.interrupted,
            event_type="run_interrupted",
        )
        return terminal

    def _capture_terminal_recovery(
        self,
        run: LogicalRunRecord,
        context: TUIContext,
        segment: AgentSegment[TUIContext, Any, TUIEnvironment],
        *,
        stable_history: Sequence[ModelMessage],
        current_prompt: str | Sequence[UserContent] | None,
        cumulative_usage: RunUsage,
    ) -> RevisionPayload | None:
        """Capture safe process-local state for one controlled terminal exit."""
        try:
            history = _merge_recoverable_history(
                stable_history,
                segment.recoverable_messages(),
                current_prompt=current_prompt,
            )
            usage = RunUsage()
            usage.incr(cumulative_usage)
            if segment.run is not None:
                usage.incr(segment.run.usage)
            payload = self._revision_payload(context, run, history, usage, {})
            return self._sanitize_terminal_recovery(run, payload)
        except Exception:
            logger.exception(
                "Failed to capture terminal recovery state for logical run %s; using stable checkpoint",
                run.logical_run_id,
            )
            return None

    def _sanitize_terminal_recovery(
        self,
        run: LogicalRunRecord,
        payload: RevisionPayload,
    ) -> RevisionPayload:
        """Fence unresolved input while preserving applied recovery state."""
        state = ResumableState.model_validate(payload.resumable_state)
        ledger = state.run_input_ledger.model_copy(deep=True)
        for item in self.store.list_inputs(run.logical_run_id):
            if item.state is InputState.applied:
                continue
            record = ledger.find(item.input_id)
            if record is not None and record.disposition is not InputDisposition.applied:
                ledger.reject(item.input_id, "run terminated before input application")
        state = state.model_copy(update={"run_input_ledger": ledger})
        return payload.model_copy(
            update={
                "resumable_state": cast(
                    dict[str, JsonValue],
                    state.model_dump(mode="json"),
                ),
                "input_ledger": cast(
                    dict[str, JsonValue],
                    {
                        **payload.input_ledger,
                        "native": ledger.model_dump(mode="json"),
                    },
                ),
                "display_projection": _safe_recovery_display_projection(payload.display_projection),
            }
        )

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
        projection = list(stable)
        if self.display_projection_provider is not None:
            live = list(self.display_projection_provider())
            if live:
                # The provider is the complete bounded TUI projection, not a
                # delta. Prefer it so cancellation between native segment
                # checkpoints cannot roll visible tool calls back to the last
                # stable checkpoint.
                projection = live
        return self._with_durable_steering_projection(run, projection)

    def _current_terminal_display_projection(self, run: LogicalRunRecord) -> list[JsonValue]:
        current = list(self.display_projection_provider()) if self.display_projection_provider is not None else []
        return self._with_durable_steering_projection(run, current)

    def _with_durable_steering_projection(
        self,
        run: LogicalRunRecord,
        projection: Sequence[JsonValue],
    ) -> list[JsonValue]:
        durable_events = durable_steering_display_events(
            run.session_id,
            self.store.list_inputs(run.logical_run_id),
        )
        merged = _merge_durable_display_projection(projection, durable_events)
        bounded = _bound_display_projection(merged)
        return _drop_orphaned_current_applied_events(bounded, durable_events)

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
    """Lazily own process-local runtimes and runs created by this process."""

    def __init__(
        self,
        *,
        store: SessionStore,
        runtime_specs: dict[str, LocalRuntimeSpec],
        runtimes: dict[str, LocalRuntime],
        base_binding_ref: str,
        coordinator: LocalExecutionCoordinator,
    ) -> None:
        self.store = store
        self._runtime_specs = runtime_specs
        self._runtimes = runtimes
        self._base_binding_ref = base_binding_ref
        self.coordinator = coordinator
        self._activation_lock = asyncio.Lock()
        self._closed = False

    @property
    def active(self) -> LocalRuntime:
        return self.coordinator.active_runtime

    @property
    def runtime(self) -> AgentRuntime[TUIContext, Any, TUIEnvironment]:
        return self.active.runtime

    @property
    def binding_ref(self) -> str:
        return self.active.binding_ref

    @property
    def runtime_id(self) -> str:
        return self.active.runtime_id

    async def activate(self, runtime_id: str) -> LocalRuntime:
        """Enter and cache one runtime before making it active."""
        if self._closed:
            raise RuntimeError("Local execution worker is closed")
        try:
            spec = self._runtime_specs[runtime_id]
        except KeyError:
            raise KeyError(runtime_id) from None

        async with self._activation_lock:
            if self._closed:
                raise RuntimeError("Local execution worker is closed")
            local_runtime = self._runtimes.get(runtime_id)
            if local_runtime is None:
                local_runtime = await self._enter_runtime(spec)
                self._runtimes[runtime_id] = local_runtime
            return self.coordinator.activate(runtime_id)

    async def _enter_runtime(self, spec: LocalRuntimeSpec) -> LocalRuntime:
        binding_ref = f"{self._base_binding_ref}:local:{spec.runtime_id}"
        runtime = spec.build(binding_ref)
        entered = False
        try:
            await runtime.__aenter__()
            entered = True
            runtime.ctx.runtime_descriptor_id = None
            local_runtime = LocalRuntime(
                runtime_id=spec.runtime_id,
                runtime=runtime,
                binding_ref=binding_ref,
                binding_context=runtime.ctx,
                request_limit=spec.request_limit,
                hitl_policy=spec.hitl_policy,
            )
            runtime_bindings.register(binding_ref, runtime.ctx, self.store)
            return local_runtime
        except BaseException as error:
            runtime_bindings.unregister(binding_ref, runtime.ctx)
            if entered:
                try:
                    await runtime.__aexit__(None, None, None)
                except BaseException as cleanup_error:
                    raise BaseExceptionGroup(
                        f"Runtime {spec.runtime_id!r} failed to enter and close",
                        [error, cleanup_error],
                    ) from None
            raise

    @classmethod
    async def create(
        cls,
        *,
        store: SessionStore,
        state_path: Path,
        active_runtime_id: str,
        runtime_specs: Sequence[LocalRuntimeSpec],
        event_sink: ExecutionEventSink | None = None,
        display_projection_provider: DisplayProjectionProvider | None = None,
    ) -> LocalExecutionWorker:
        state_path = state_path.expanduser().resolve()
        specs_by_id: dict[str, LocalRuntimeSpec] = {}
        for spec in runtime_specs:
            if spec.runtime_id in specs_by_id:
                raise ValueError(f"Duplicate local runtime ID {spec.runtime_id!r}")
            specs_by_id[spec.runtime_id] = spec
        if active_runtime_id not in specs_by_id:
            raise KeyError(active_runtime_id)

        base_binding_ref = f"yaacli-runtime:{uuid.uuid5(uuid.NAMESPACE_URL, str(state_path))}"
        runtimes: dict[str, LocalRuntime] = {}
        coordinator = LocalExecutionCoordinator(
            store=store,
            runtimes=runtimes,
            active_runtime_id=active_runtime_id,
            event_sink=event_sink,
            display_projection_provider=display_projection_provider,
        )
        worker = cls(
            store=store,
            runtime_specs=specs_by_id,
            runtimes=runtimes,
            base_binding_ref=base_binding_ref,
            coordinator=coordinator,
        )
        try:
            active_runtime = await worker._enter_runtime(specs_by_id[active_runtime_id])
        except BaseException:
            worker._closed = True
            raise
        runtimes[active_runtime_id] = active_runtime
        return worker

    async def __aenter__(self) -> LocalExecutionWorker:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()

    async def close(self) -> None:
        async with self._activation_lock:
            if self._closed:
                return
            self._closed = True
            await self.coordinator.shutdown()
            errors: list[BaseException] = []
            for local_runtime in reversed(tuple(self._runtimes.values())):
                try:
                    await local_runtime.runtime.__aexit__(None, None, None)
                except BaseException as error:
                    errors.append(error)
                finally:
                    runtime_bindings.unregister(local_runtime.binding_ref, local_runtime.binding_context)
            if len(errors) == 1:
                raise errors[0]
            if errors:
                raise BaseExceptionGroup("Multiple local runtimes failed to close", errors)


def _merge_recoverable_history(
    stable: Sequence[ModelMessage],
    recovered: Sequence[ModelMessage],
    *,
    current_prompt: str | Sequence[UserContent] | None,
) -> list[ModelMessage]:
    """Merge one segment's safe partial history onto its stable boundary."""
    stable_messages = list(stable)
    recovered_messages = list(recovered)
    if recovered_messages[: len(stable_messages)] == stable_messages:
        merged = recovered_messages
    elif stable_messages[: len(recovered_messages)] == recovered_messages:
        merged = stable_messages
    else:
        overlap = next(
            (
                size
                for size in range(min(len(stable_messages), len(recovered_messages)), 0, -1)
                if stable_messages[-size:] == recovered_messages[:size]
            ),
            0,
        )
        merged = [*stable_messages, *recovered_messages[overlap:]]

    if current_prompt is None:
        return merged
    prompt_content: str | list[UserContent]
    prompt_content = current_prompt if isinstance(current_prompt, str) else list(current_prompt)
    current_messages = merged[len(stable_messages) :]
    if any(
        isinstance(message, ModelRequest)
        and any(isinstance(part, UserPromptPart) and part.content == prompt_content for part in message.parts)
        for message in current_messages
    ):
        return merged
    merged.insert(
        len(stable_messages),
        ModelRequest(parts=[UserPromptPart(content=prompt_content)]),
    )
    return merged


def _safe_recovery_display_projection(events: Sequence[JsonValue]) -> list[JsonValue]:
    """Retain observed UI facts while bounding the non-authoritative replay."""
    return _bound_display_projection(events)


def _bound_display_projection(events: Sequence[JsonValue]) -> list[JsonValue]:
    replay = BoundedDisplayReplay()
    replay.extend_snapshot([dict(event) for event in events if isinstance(event, dict)])
    return cast(list[JsonValue], replay.snapshot())


def _drop_orphaned_current_applied_events(
    bounded: Sequence[JsonValue],
    durable_events: Sequence[JsonValue],
) -> list[JsonValue]:
    current_keys = {
        identity[1] for event in durable_events if (identity := _durable_display_event_identity(event)) is not None
    }
    retained_accepted_keys = {
        identity[1]
        for event in bounded
        if (identity := _durable_display_event_identity(event)) is not None
        and identity[0] == STEERING_ACCEPTED_EVENT_NAME
    }
    retained: list[JsonValue] = []
    for event in bounded:
        identity = _durable_display_event_identity(event)
        if (
            identity is not None
            and identity[0] == STEERING_APPLIED_EVENT_NAME
            and identity[1] in current_keys
            and identity[1] not in retained_accepted_keys
        ):
            continue
        retained.append(event)
    return retained


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
