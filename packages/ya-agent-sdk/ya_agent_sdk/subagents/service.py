"""Portable subagent registry, store, driver, and execution service."""

from __future__ import annotations

import asyncio
import inspect
import json
import threading
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any, Protocol, cast, runtime_checkable
from uuid import uuid4

from pydantic import TypeAdapter
from pydantic_ai import (
    Agent,
    DeferredToolRequests,
    DeferredToolResults,
    StructuredDict,
    UsageLimits,
)
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    UserContent,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.usage import RunUsage
from pydantic_core import to_jsonable_python

from ya_agent_sdk.agents.driver import drive_streamed_run
from ya_agent_sdk.agents.models import infer_model
from ya_agent_sdk.context import AgentContext, ResumableState, ToolProxyState
from ya_agent_sdk.events import SubagentCompleteEvent, SubagentStartEvent
from ya_agent_sdk.inputs import (
    EnqueueReceipt,
    InputDisposition,
    InputOrigin,
    LogicalRunInputRouter,
    RunInputLedger,
)
from ya_agent_sdk.subagents.resolver import validate_resolved_subagent_plan_integrity
from ya_agent_sdk.subagents.spec import (
    ResolvedSubagentPlan,
    SubagentDeliveryState,
    SubagentDriverOutcome,
    SubagentExecutionMode,
    SubagentExecutionRecord,
    SubagentExecutionState,
    SubagentHandle,
    SubagentInputState,
    clone_resolved_subagent_plan,
)

_DEFERRED_REQUESTS = TypeAdapter(DeferredToolRequests)
_DEFERRED_RESULTS = TypeAdapter(DeferredToolResults)


class SubagentRegistry:
    """Thread-safe registry with active routes and retained plan versions."""

    def __init__(self, plans: Sequence[ResolvedSubagentPlan] = ()) -> None:
        self._lock = threading.RLock()
        self._active_by_route: dict[str, ResolvedSubagentPlan] = {}
        self._by_descriptor_id: dict[str, ResolvedSubagentPlan] = {}
        for plan in plans:
            self.register(plan)

    def register(self, plan: ResolvedSubagentPlan) -> None:
        """Register the active plan used for new executions of one route."""
        stored = self._validated_clone(plan)
        with self._lock:
            existing = self._active_by_route.get(stored.spec.route)
            if existing is not None:
                raise ValueError(
                    f"Subagent route {stored.spec.route!r} is already registered as {existing.descriptor_id!r}"
                )
            self._register_descriptor(stored)
            self._active_by_route[stored.spec.route] = stored

    def register_retained(self, plan: ResolvedSubagentPlan) -> None:
        """Retain an exact historical plan without changing the active route."""
        stored = self._validated_clone(plan)
        with self._lock:
            self._register_descriptor(stored)

    def get(self, route: str) -> ResolvedSubagentPlan:
        """Return the active plan for a new execution."""
        with self._lock:
            try:
                return clone_resolved_subagent_plan(self._active_by_route[route])
            except KeyError as exc:
                raise KeyError(f"Unknown subagent route {route!r}") from exc

    def get_descriptor(self, descriptor_id: str) -> ResolvedSubagentPlan:
        """Return the exact immutable plan version for an existing execution."""
        with self._lock:
            try:
                return clone_resolved_subagent_plan(self._by_descriptor_id[descriptor_id])
            except KeyError as exc:
                raise KeyError(f"Unknown subagent descriptor {descriptor_id!r}") from exc

    def list(self) -> tuple[ResolvedSubagentPlan, ...]:
        """List only active routes for model-facing rosters and new spawns."""
        with self._lock:
            return tuple(
                clone_resolved_subagent_plan(self._active_by_route[name]) for name in sorted(self._active_by_route)
            )

    def list_registered(self) -> tuple[ResolvedSubagentPlan, ...]:
        """List every executable plan version, including retained descriptors."""
        with self._lock:
            return tuple(
                clone_resolved_subagent_plan(self._by_descriptor_id[descriptor_id])
                for descriptor_id in sorted(self._by_descriptor_id)
            )

    def __contains__(self, route: object) -> bool:
        with self._lock:
            return route in self._active_by_route

    @staticmethod
    def _validated_clone(plan: ResolvedSubagentPlan) -> ResolvedSubagentPlan:
        stored = clone_resolved_subagent_plan(plan)
        validate_resolved_subagent_plan_integrity(stored)
        return stored

    def _register_descriptor(self, stored: ResolvedSubagentPlan) -> None:
        if stored.descriptor_id not in self._by_descriptor_id:
            self._by_descriptor_id[stored.descriptor_id] = stored


@runtime_checkable
class SubagentExecutionStore(Protocol):
    """Persistence boundary for execution records."""

    async def close(self) -> None: ...

    async def create(self, record: SubagentExecutionRecord) -> SubagentExecutionRecord: ...

    async def save(self, record: SubagentExecutionRecord) -> SubagentExecutionRecord: ...

    async def get(
        self,
        execution_id: str,
        *,
        owner_scope_id: str | None = None,
    ) -> SubagentExecutionRecord | None: ...

    async def get_by_idempotency_key(
        self,
        idempotency_key: str,
        *,
        owner_scope_id: str,
    ) -> SubagentExecutionRecord | None: ...

    async def list(
        self,
        *,
        owner_scope_id: str | None = None,
    ) -> tuple[SubagentExecutionRecord, ...]: ...


class InMemorySubagentExecutionStore:
    """Process-local store that makes no restart-durability claim."""

    restart_durable = False

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._records: dict[str, SubagentExecutionRecord] = {}
        self._idempotency: dict[tuple[str, str], str] = {}

    async def close(self) -> None:
        """Release the process-local store."""

    async def create(
        self,
        record: SubagentExecutionRecord,
    ) -> SubagentExecutionRecord:
        async with self._lock:
            if record.execution_id in self._records:
                raise ValueError(f"Subagent execution {record.execution_id!r} already exists")
            existing_id = self._idempotency.get((record.owner_scope_id, record.idempotency_key))
            if existing_id is not None:
                return self._records[existing_id].model_copy(deep=True)
            stored = record.model_copy(deep=True)
            self._records[record.execution_id] = stored
            self._idempotency[(record.owner_scope_id, record.idempotency_key)] = record.execution_id
            return stored.model_copy(deep=True)

    async def save(
        self,
        record: SubagentExecutionRecord,
    ) -> SubagentExecutionRecord:
        async with self._lock:
            if record.execution_id not in self._records:
                raise KeyError(record.execution_id)
            stored = record.model_copy(deep=True)
            self._records[record.execution_id] = stored
            self._idempotency[(record.owner_scope_id, record.idempotency_key)] = record.execution_id
            return stored.model_copy(deep=True)

    async def get(
        self,
        execution_id: str,
        *,
        owner_scope_id: str | None = None,
    ) -> SubagentExecutionRecord | None:
        async with self._lock:
            record = self._records.get(execution_id)
            if record is None or (owner_scope_id is not None and record.owner_scope_id != owner_scope_id):
                return None
            return record.model_copy(deep=True)

    async def get_by_idempotency_key(
        self,
        idempotency_key: str,
        *,
        owner_scope_id: str,
    ) -> SubagentExecutionRecord | None:
        async with self._lock:
            execution_id = self._idempotency.get((owner_scope_id, idempotency_key))
            if execution_id is None:
                return None
            return self._records[execution_id].model_copy(deep=True)

    async def list(
        self,
        *,
        owner_scope_id: str | None = None,
    ) -> tuple[SubagentExecutionRecord, ...]:
        async with self._lock:
            records = (
                record
                for record in self._records.values()
                if owner_scope_id is None or record.owner_scope_id == owner_scope_id
            )
            return tuple(record.model_copy(deep=True) for record in sorted(records, key=lambda item: item.created_at))


@runtime_checkable
class SubagentDriver(Protocol):
    """Host execution boundary for one already-resolved child plan."""

    restart_durable: bool

    async def run(
        self,
        plan: ResolvedSubagentPlan,
        record: SubagentExecutionRecord,
        parent_ctx: AgentContext,
    ) -> SubagentDriverOutcome: ...

    async def cancel(self, record: SubagentExecutionRecord) -> None: ...


@runtime_checkable
class SubagentSteeringDriver(Protocol):
    """Optional durable-driver boundary for externally hosted child input."""

    async def steer(
        self,
        record: SubagentExecutionRecord,
        *content: Any,
        origin: InputOrigin,
        idempotency_key: str | None,
    ) -> Any: ...


@runtime_checkable
class RetainedSubagentPlanProvider(Protocol):
    """Optional host boundary for lazily restoring an exact historical plan."""

    async def load_retained_plan(
        self,
        record: SubagentExecutionRecord,
    ) -> ResolvedSubagentPlan | None: ...


ModelResolver = Callable[[str], str | Model]
ChildContextConfigurer = Callable[
    [ResolvedSubagentPlan, SubagentExecutionRecord, AgentContext, AgentContext],
    Awaitable[None] | None,
]


class InProcessSubagentDriver:
    """Standalone in-process adapter over Pydantic AI's native graph driver."""

    restart_durable = False

    def __init__(
        self,
        *,
        custom_capability_types: Sequence[type[Any]] = (),
        model_resolver: ModelResolver | None = None,
        request_limit: int | None = None,
        child_context_configurer: ChildContextConfigurer | None = None,
    ) -> None:
        if request_limit is not None and request_limit <= 0:
            raise ValueError("request_limit must be positive")
        self.custom_capability_types = tuple(custom_capability_types)
        self.model_resolver = model_resolver or infer_model
        self.request_limit = request_limit
        self.child_context_configurer = child_context_configurer

    async def cancel(self, record: SubagentExecutionRecord) -> None:
        """Process-local cancellation is owned by the service task."""
        del record

    async def _resolve_model(
        self,
        plan: ResolvedSubagentPlan,
        record: SubagentExecutionRecord,
        parent_ctx: AgentContext,
    ) -> str | Model:
        model_name = plan.normalized_agent_spec.model
        if model_name is None:  # pragma: no cover - resolver guarantees this
            raise ValueError(f"Resolved subagent {plan.spec.route!r} has no model")
        model = self.model_resolver(model_name)
        if isinstance(model, str):
            model = infer_model(model)
        if parent_ctx.model_wrapper is None or not isinstance(model, Model):
            return model
        wrapped = parent_ctx.model_wrapper(
            model,
            plan.spec.route,
            {
                **parent_ctx.get_wrapper_metadata(),
                "agent_id": record.execution_id,
                "parent_run_id": parent_ctx.run_id,
            },
        )
        if inspect.isawaitable(wrapped):
            wrapped = await wrapped
        if not isinstance(wrapped, Model):
            raise TypeError("model_wrapper must return a Pydantic AI Model")
        return wrapped

    def _usage_limits(self, record: SubagentExecutionRecord) -> UsageLimits | None:
        if self.request_limit is None:
            return None
        raw_used_requests = record.usage.get("requests", 0)
        if not isinstance(raw_used_requests, int):
            raise TypeError("Subagent usage.requests must be an integer")
        remaining_requests = self.request_limit - raw_used_requests
        if remaining_requests <= 0:
            raise RuntimeError(
                f"Subagent continuation exhausted the cumulative model request limit of {self.request_limit}."
            )
        return UsageLimits(request_limit=remaining_requests)

    async def run(
        self,
        plan: ResolvedSubagentPlan,
        record: SubagentExecutionRecord,
        parent_ctx: AgentContext,
    ) -> SubagentDriverOutcome:
        model = await self._resolve_model(plan, record, parent_ctx)

        child_ctx = parent_ctx.create_subagent_context(
            plan.spec.route,
            record.execution_id,
            parent_run_id=parent_ctx.run_id,
            run_input_ledger=RunInputLedger(logical_run_id=record.child_logical_run_id),
            input_router=None,
            usage_snapshot_entries={},
            deferred_tool_metadata={},
            tool_proxy=ToolProxyState(),
        )
        if record.resumable_state:
            ResumableState.model_validate(record.resumable_state).restore(child_ctx)
        child_ctx.run_input_ledger.logical_run_id = record.child_logical_run_id
        child_ctx.subagent_depth = record.depth
        child_ctx.subagent_target = plan.spec.route
        child_ctx.subagent_descriptor_id = plan.descriptor_id
        child_ctx.delegation_scope_id = record.owner_scope_id
        if self.child_context_configurer is not None:
            configured = self.child_context_configurer(plan, record, parent_ctx, child_ctx)
            if inspect.isawaitable(configured):
                await configured

        history = _deserialize_history(record.history or list(plan.initial_history))
        deferred_results = (
            _DEFERRED_RESULTS.validate_python(record.deferred_results) if record.deferred_results is not None else None
        )
        current_prompt: str | list[UserContent] | None = None if deferred_results is not None else record.prompt
        usage_limits = self._usage_limits(record)
        agent = Agent.from_spec(
            plan.normalized_agent_spec,
            deps_type=type(child_ctx),
            custom_capability_types=self.custom_capability_types,
            model=model,
            output_type=resolve_subagent_output_type(plan),
            capabilities=plan.host_capabilities,
        )
        router = LogicalRunInputRouter(child_ctx.run_input_ledger)
        registration = parent_ctx.active_run_registry.register(router)
        child_ctx.input_router = router
        run = None
        deferred_suspension = False
        input_applied = record.input_state is SubagentInputState.applied
        try:
            async with child_ctx, agent:
                if deferred_results is None:
                    child_ctx.run_input_ledger.record_initial(
                        [ModelRequest(parts=[UserPromptPart(content=record.prompt)])],
                        origin=InputOrigin.user,
                    )
                    input_applied = True
                    record.input_state = SubagentInputState.applied
                async with agent.iter(
                    current_prompt,
                    deps=child_ctx,
                    message_history=history,
                    deferred_tool_results=deferred_results,
                    usage_limits=usage_limits,
                    run_id=(f"{record.child_logical_run_id}:{record.segment_index}"),
                ) as run:
                    attempt_id = str(uuid4())
                    await router.bind(run, native_attempt_id=attempt_id)
                    try:
                        await drive_streamed_run(
                            run,
                            lambda node, current_run: _stream_child_node(
                                node,
                                current_run,
                                child_ctx,
                                router,
                            ),
                        )
                    finally:
                        router.unbind(native_attempt_id=attempt_id)

                result = run.result
                if result is None:  # pragma: no cover - native graph invariant
                    raise RuntimeError("Subagent graph completed without a result")
                serialized_history = tuple(
                    ModelMessagesTypeAdapter.dump_python(
                        run.all_messages(),
                        mode="json",
                    )
                )
                usage = to_jsonable_python(_combine_usage(record.usage, result.usage))
                if isinstance(result.output, DeferredToolRequests):
                    deferred_suspension = True
                    return SubagentDriverOutcome(
                        state=SubagentExecutionState.suspended,
                        input_state=SubagentInputState.applied,
                        history=serialized_history,
                        usage=cast(dict[str, Any], usage),
                        deferred=cast(
                            dict[str, Any],
                            to_jsonable_python(result.output),
                        ),
                        resumable_state=_export_context_state(child_ctx),
                    )
                return SubagentDriverOutcome(
                    state=SubagentExecutionState.succeeded,
                    input_state=SubagentInputState.applied,
                    output=to_jsonable_python(result.output),
                    history=serialized_history,
                    usage=cast(dict[str, Any], usage),
                    resumable_state=_export_context_state(child_ctx),
                )
        except Exception as exc:
            return SubagentDriverOutcome(
                state=SubagentExecutionState.failed,
                input_state=(SubagentInputState.applied if input_applied else SubagentInputState.rejected),
                error=str(exc) or repr(exc),
                history=tuple(record.history),
                usage=dict(record.usage),
                resumable_state=dict(record.resumable_state),
            )
        finally:
            router.close(
                reason="subagent execution closed",
                reject_unresolved=not deferred_suspension,
            )
            parent_ctx.active_run_registry.unregister(registration)
            child_ctx.input_router = None


async def _stream_child_node(
    node: Any,
    run: Any,
    child_ctx: AgentContext,
    router: LogicalRunInputRouter,
) -> None:
    if Agent.is_user_prompt_node(node) or Agent.is_end_node(node):
        return
    if not (Agent.is_model_request_node(node) or Agent.is_call_tools_node(node)):
        return
    async with node.stream(run.ctx) as stream:
        async for event in stream:
            router.observe_event(event)
            await child_ctx.emit_event(child_ctx.tool_id_wrapper.wrap_event(event))


def resolve_subagent_output_type(plan: ResolvedSubagentPlan) -> Any:
    """Build the exact native output contract, including durable suspension."""
    output_type = StructuredDict(plan.effective_output_schema) if plan.effective_output_schema is not None else str
    if plan.supports_deferred_output:
        return [output_type, DeferredToolRequests]
    return output_type


def _deserialize_history(values: Sequence[dict[str, Any]]) -> list[ModelMessage] | None:
    if not values:
        return None
    return ModelMessagesTypeAdapter.validate_json(json.dumps(list(values)).encode())


def _combine_usage(
    previous: dict[str, Any],
    current: RunUsage,
) -> RunUsage:
    cumulative = RunUsage(**previous) if previous else RunUsage()
    cumulative.incr(current)
    return cumulative


@runtime_checkable
class SubagentCompletionDelivery(Protocol):
    """Host boundary for persisting and observing canonical parent input."""

    async def deliver(
        self,
        record: SubagentExecutionRecord,
        parent_ctx: AgentContext,
        message: str,
    ) -> EnqueueReceipt | None: ...


@runtime_checkable
class SubagentDeferredResolver(Protocol):
    """Host boundary for resolving a child's native deferred interaction."""

    async def resolve(
        self,
        record: SubagentExecutionRecord,
        requests: DeferredToolRequests,
    ) -> DeferredToolResults: ...


class SubagentExecutionService:
    """Application service shared by foreground and background delegation."""

    def __init__(
        self,
        registry: SubagentRegistry,
        store: SubagentExecutionStore,
        driver: SubagentDriver,
        *,
        completion_delivery: SubagentCompletionDelivery | None = None,
        deferred_resolver: SubagentDeferredResolver | None = None,
        retained_plan_provider: RetainedSubagentPlanProvider | None = None,
    ) -> None:
        if bool(getattr(driver, "restart_durable", False)) != bool(getattr(store, "restart_durable", False)):
            raise ValueError("Subagent driver and store must agree on restart durability")
        self.registry = registry
        self.store = store
        self.driver = driver
        self.completion_delivery = completion_delivery
        self.deferred_resolver = deferred_resolver
        self.retained_plan_provider = retained_plan_provider
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._active_run_registries: dict[str, Any] = {}
        self._task_lock = asyncio.Lock()
        self._delivery_locks: dict[str, asyncio.Lock] = {}
        self._closed = False

    async def spawn(
        self,
        route: str,
        prompt: str | Sequence[UserContent],
        parent_ctx: AgentContext,
        *,
        mode: SubagentExecutionMode = SubagentExecutionMode.foreground,
        idempotency_key: str | None = None,
        history: Sequence[dict[str, Any]] = (),
    ) -> SubagentHandle:
        """Create an execution from the active route plan and start it exactly once."""
        return await self._spawn(
            self.registry.get(route),
            prompt,
            parent_ctx,
            mode=mode,
            idempotency_key=idempotency_key,
            initial_history=history,
        )

    async def _spawn(
        self,
        plan: ResolvedSubagentPlan,
        prompt: str | Sequence[UserContent],
        parent_ctx: AgentContext,
        *,
        mode: SubagentExecutionMode,
        idempotency_key: str | None,
        resumed_record: SubagentExecutionRecord | None = None,
        initial_history: Sequence[dict[str, Any]] = (),
    ) -> SubagentHandle:
        """Create an execution from one exact immutable plan."""
        if self._closed:
            raise RuntimeError("Subagent execution service is closed")
        route = plan.spec.route
        if mode not in plan.spec.execution_modes:
            raise ValueError(f"Subagent {route!r} does not allow {mode.value!r} execution")
        parent_depth = parent_ctx.subagent_depth
        depth = parent_depth + 1
        if depth > plan.spec.max_depth:
            raise ValueError(f"Subagent {route!r} maximum depth {plan.spec.max_depth} would be exceeded")
        _authorize_nested_spawn(
            self.registry,
            parent_ctx=parent_ctx,
            route=route,
        )

        owner_scope_id = _scope_id(parent_ctx)
        key = idempotency_key or str(uuid4())
        existing = await self.store.get_by_idempotency_key(
            key,
            owner_scope_id=owner_scope_id,
        )
        if existing is not None:
            _validate_spawn_replay(
                existing,
                plan=plan,
                prompt=prompt,
                mode=mode,
                resumed_record=resumed_record,
            )
            if not existing.terminal and (
                existing.state is not SubagentExecutionState.suspended or self.deferred_resolver is not None
            ):
                await self._schedule_execution(
                    await self._plan_for_record(existing),
                    existing,
                    parent_ctx,
                )
            return _handle(existing)

        execution_id = str(uuid4())
        child_logical_run_id = str(uuid4())
        parent_logical_run_id = (
            parent_ctx.input_router.logical_run_id
            if parent_ctx.input_router is not None
            else parent_ctx.run_input_ledger.logical_run_id
        )
        record = SubagentExecutionRecord(
            execution_id=execution_id,
            root_execution_id=(resumed_record.root_execution_id if resumed_record is not None else execution_id),
            owner_scope_id=owner_scope_id,
            idempotency_key=key,
            descriptor_id=plan.descriptor_id,
            plan_fingerprint=plan.fingerprint,
            route=route,
            mode=mode,
            delivery_state=(
                SubagentDeliveryState.pending
                if mode is SubagentExecutionMode.background and plan.spec.linkage.value == "child"
                else SubagentDeliveryState.not_required
            ),
            parent_agent_id=parent_ctx.agent_id,
            parent_logical_run_id=parent_logical_run_id,
            parent_runtime_descriptor_id=parent_ctx.runtime_descriptor_id,
            child_logical_run_id=child_logical_run_id,
            depth=depth,
            prompt=(list(prompt) if not isinstance(prompt, str) else prompt),
            resumed_from=(resumed_record.execution_id if resumed_record is not None else None),
            history=(list(resumed_record.history) if resumed_record is not None else list(initial_history)),
            resumable_state=_initial_child_state(
                parent_ctx,
                logical_run_id=child_logical_run_id,
            ),
        )
        created = await self.store.create(record)
        if created.execution_id != execution_id:
            return _handle(created)

        await self._schedule_execution(plan, created, parent_ctx)
        return _handle(created)

    async def resume(
        self,
        execution_id: str,
        prompt: str | Sequence[UserContent],
        parent_ctx: AgentContext,
        *,
        mode: SubagentExecutionMode | None = None,
        idempotency_key: str | None = None,
    ) -> SubagentHandle:
        """Start a linked execution from a committed prior history snapshot."""
        previous = await self._required_record(
            execution_id,
            owner_scope_id=_scope_id(parent_ctx),
        )
        if not previous.terminal:
            raise ValueError(f"Subagent execution {execution_id!r} has not reached a terminal state")
        plan = await self._plan_for_record(previous)
        return await self._spawn(
            plan,
            prompt,
            parent_ctx,
            mode=mode or previous.mode,
            idempotency_key=idempotency_key,
            resumed_record=previous,
        )

    async def continue_deferred(
        self,
        execution_id: str,
        results: DeferredToolResults,
        parent_ctx: AgentContext,
    ) -> SubagentHandle:
        """Persist host decisions and continue the same suspended execution."""
        owner_scope_id = _scope_id(parent_ctx)
        record = await self._required_record(
            execution_id,
            owner_scope_id=owner_scope_id,
        )
        if record.state is not SubagentExecutionState.suspended or record.deferred is None:
            raise ValueError(f"Subagent execution {execution_id!r} has no suspended deferred request")
        saved = await self._accept_deferred_results(record, results)
        await self._schedule_execution(
            await self._plan_for_record(saved),
            saved,
            parent_ctx,
        )
        return _handle(saved)

    async def steer(
        self,
        execution_id: str,
        *content: Any,
        caller_scope_id: str,
        origin: InputOrigin = InputOrigin.user,
        idempotency_key: str | None = None,
    ) -> Any:
        """Target structured input at one currently accepting child run."""
        record = await self._required_record(
            execution_id,
            owner_scope_id=caller_scope_id,
        )
        async with self._task_lock:
            active_registry = self._active_run_registries.get(execution_id)
        if active_registry is not None:
            router = active_registry.get(record.child_logical_run_id)
            if router is not None:
                return await router.enqueue(
                    *content,
                    origin=origin,
                    input_id=idempotency_key,
                )
        if isinstance(self.driver, SubagentSteeringDriver):
            return await self.driver.steer(
                record,
                *content,
                origin=origin,
                idempotency_key=idempotency_key,
            )
        raise RuntimeError(f"Subagent execution {execution_id!r} is not accepting input")

    async def cancel(
        self,
        execution_id: str,
        *,
        caller_scope_id: str,
    ) -> SubagentExecutionRecord:
        """Cancel one active execution owned by the caller scope."""
        record = await self._required_record(
            execution_id,
            owner_scope_id=caller_scope_id,
        )
        await self.driver.cancel(record)
        async with self._task_lock:
            task = self._tasks.get(execution_id)
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        return await self._required_record(
            execution_id,
            owner_scope_id=caller_scope_id,
        )

    async def wait(
        self,
        execution_id: str,
        *,
        caller_scope_id: str,
        timeout: float | None = None,
    ) -> SubagentExecutionRecord:
        """Wait for current process work and return the latest committed record."""
        async with self._task_lock:
            task = self._tasks.get(execution_id)
        if task is not None:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        return await self._required_record(
            execution_id,
            owner_scope_id=caller_scope_id,
        )

    async def get(
        self,
        execution_id: str,
        *,
        caller_scope_id: str,
    ) -> SubagentExecutionRecord:
        return await self._required_record(
            execution_id,
            owner_scope_id=caller_scope_id,
        )

    async def list(
        self,
        *,
        caller_scope_id: str,
    ) -> tuple[SubagentExecutionRecord, ...]:
        return await self.store.list(owner_scope_id=caller_scope_id)

    async def admin_get(self, execution_id: str) -> SubagentExecutionRecord:
        """Privileged host inspection outside model-facing authorization."""
        return await self._required_record(execution_id)

    async def admin_list(self) -> tuple[SubagentExecutionRecord, ...]:
        """Privileged host listing outside model-facing authorization."""
        return await self.store.list()

    async def deliver_pending(self, parent_ctx: AgentContext) -> int:
        """Recover unfinished durable children, then deliver terminal results once."""
        owner_scope_id = _scope_id(parent_ctx)
        scoped_records = await self.store.list(owner_scope_id=owner_scope_id)
        if bool(getattr(self.driver, "restart_durable", False)):
            for record in scoped_records:
                if record.terminal or (
                    record.state is SubagentExecutionState.suspended and self.deferred_resolver is None
                ):
                    continue
                await self._schedule_execution(
                    await self._plan_for_record(record),
                    record,
                    parent_ctx,
                )
        delivered = 0
        for record in scoped_records:
            if (
                record.delivery_state is not SubagentDeliveryState.pending
                or not record.terminal
                or record.parent_logical_run_id is None
            ):
                continue
            lock = self._delivery_locks.setdefault(
                record.execution_id,
                asyncio.Lock(),
            )
            async with lock:
                delivered += await self._deliver_record(
                    record.execution_id,
                    owner_scope_id=owner_scope_id,
                    parent_ctx=parent_ctx,
                )
        return delivered

    async def _deliver_record(
        self,
        execution_id: str,
        *,
        owner_scope_id: str,
        parent_ctx: AgentContext,
    ) -> int:
        current = await self._required_record(
            execution_id,
            owner_scope_id=owner_scope_id,
        )
        if current.delivery_state is not SubagentDeliveryState.pending:
            return 0
        message = _completion_message(current)
        receipt = await self._deliver_completion(current, parent_ctx, message)
        if receipt is None:
            return 0
        if receipt.disposition is InputDisposition.rejected:
            current.delivery_logical_run_id = None
            current.delivery_input_id = None
            await self.store.save(current)
            receipt = await self._deliver_completion(current, parent_ctx, message)
            if receipt is None or receipt.disposition is InputDisposition.rejected:
                return 0
        current.delivery_logical_run_id = receipt.logical_run_id
        current.delivery_input_id = receipt.input_id
        if receipt.disposition is InputDisposition.applied:
            current.delivery_state = SubagentDeliveryState.delivered
        await self.store.save(current)
        return int(receipt.disposition is InputDisposition.applied)

    async def _deliver_completion(
        self,
        record: SubagentExecutionRecord,
        parent_ctx: AgentContext,
        message: str,
    ) -> EnqueueReceipt | None:
        if self.completion_delivery is not None:
            return await self.completion_delivery.deliver(
                record,
                parent_ctx,
                message,
            )
        logical_run_id = record.delivery_logical_run_id or record.parent_logical_run_id
        if logical_run_id is None:
            return None
        router = parent_ctx.active_run_registry.get(logical_run_id)
        if router is None:
            return None
        input_id = record.delivery_input_id or f"subagent-completion:{record.execution_id}"
        existing = router.ledger.find(input_id)
        if existing is not None:
            return EnqueueReceipt(
                logical_run_id=logical_run_id,
                input_id=input_id,
                disposition=existing.disposition,
                enqueue_id=existing.latest_enqueue_id,
            )
        return await router.enqueue(
            message,
            origin=InputOrigin.feature,
            input_id=input_id,
        )

    async def close(self) -> None:
        """Release adapter work without invalidating restart-durable executions."""
        if self._closed:
            return
        self._closed = True
        async with self._task_lock:
            tasks = tuple(self._tasks.items())
        for _execution_id, task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(
                *(task for _, task in tasks),
                return_exceptions=True,
            )
        if not bool(getattr(self.driver, "restart_durable", False)):
            for execution_id, _task in tasks:
                record = await self.store.get(execution_id)
                if record is not None and record.state in {
                    SubagentExecutionState.pending,
                    SubagentExecutionState.running,
                }:
                    record.state = SubagentExecutionState.lost
                    record.error = "Process-local subagent execution ended with its host process"
                    record.completed_at = datetime.now(UTC)
                    await self.store.save(record)
        await self.store.close()

    async def _schedule_execution(
        self,
        plan: ResolvedSubagentPlan,
        record: SubagentExecutionRecord,
        parent_ctx: AgentContext,
    ) -> None:
        async with self._task_lock:
            current = self._tasks.get(record.execution_id)
            if current is not None and not current.done():
                return
            task = asyncio.create_task(
                self._execute(plan, record, parent_ctx),
                name=f"subagent:{record.route}:{record.execution_id}",
            )
            self._tasks[record.execution_id] = task
            self._active_run_registries[record.execution_id] = parent_ctx.active_run_registry
        task.add_done_callback(
            lambda _task, current_id=record.execution_id: asyncio.create_task(self._forget_task(current_id))
        )

    async def _execute(
        self,
        plan: ResolvedSubagentPlan,
        record: SubagentExecutionRecord,
        parent_ctx: AgentContext,
    ) -> None:
        first_segment = record.started_at is None
        if first_segment:
            record.started_at = datetime.now(UTC)
        if first_segment:
            await parent_ctx.emit_event(
                SubagentStartEvent(
                    event_id=record.execution_id,
                    execution_id=record.execution_id,
                    mode=record.mode.value,
                    parent_logical_run_id=record.parent_logical_run_id,
                    agent_id=record.execution_id,
                    agent_name=record.route,
                    prompt_preview=_prompt_preview(record.prompt),
                )
            )
        try:
            terminal = await self._run_segments(plan, record, parent_ctx)
        except asyncio.CancelledError:
            if self._closed and bool(getattr(self.driver, "restart_durable", False)):
                raise
            record.state = SubagentExecutionState.cancelled
            if record.input_state is SubagentInputState.accepted:
                record.input_state = SubagentInputState.rejected
            record.error = "Subagent execution was cancelled"
            record.completed_at = datetime.now(UTC)
            await self.store.save(record)
            await self._emit_complete(parent_ctx, record)
            raise
        except BaseException as exc:
            record.state = SubagentExecutionState.failed
            if record.input_state is SubagentInputState.accepted:
                record.input_state = SubagentInputState.rejected
            record.error = str(exc) or repr(exc)
            record.completed_at = datetime.now(UTC)
            await self.store.save(record)
            await self._emit_complete(parent_ctx, record)
            return

        if not terminal:
            return
        await self._emit_complete(parent_ctx, record)
        if record.delivery_state is SubagentDeliveryState.pending and record.terminal:
            await self.deliver_pending(parent_ctx)

    async def _run_segments(
        self,
        plan: ResolvedSubagentPlan,
        record: SubagentExecutionRecord,
        parent_ctx: AgentContext,
    ) -> bool:
        """Run and persist segments until terminal or externally suspended."""
        while True:
            if record.state is SubagentExecutionState.suspended:
                if self.deferred_resolver is None or record.deferred is None:
                    return False
                requests = _DEFERRED_REQUESTS.validate_python(record.deferred)
                results = await self.deferred_resolver.resolve(
                    record.model_copy(deep=True),
                    requests,
                )
                await self._accept_deferred_results(record, results)

            record.state = SubagentExecutionState.running
            await self.store.save(record)
            outcome = await self.driver.run(plan, record, parent_ctx)
            _apply_driver_input_state(record, outcome)
            record.state = outcome.state
            record.output = outcome.output
            record.error = outcome.error
            record.history = list(outcome.history)
            record.usage = outcome.usage
            record.deferred = outcome.deferred
            record.deferred_results = None
            record.resumable_state = outcome.resumable_state
            if outcome.state is not SubagentExecutionState.suspended:
                record.completed_at = datetime.now(UTC)
            await self.store.save(record)
            if outcome.state is not SubagentExecutionState.suspended:
                return True
            if self.deferred_resolver is None:
                return False

    async def _accept_deferred_results(
        self,
        record: SubagentExecutionRecord,
        results: DeferredToolResults,
    ) -> SubagentExecutionRecord:
        if record.state is not SubagentExecutionState.suspended or record.deferred is None:
            raise ValueError(f"Subagent execution {record.execution_id!r} has no suspended deferred request")
        requests = _DEFERRED_REQUESTS.validate_python(record.deferred)
        approval_ids = {item.tool_call_id for item in requests.approvals}
        call_ids = {item.tool_call_id for item in requests.calls}
        if set(results.approvals) != approval_ids or set(results.calls) != call_ids:
            raise ValueError("Deferred results must resolve every pending child approval and call exactly once")
        record.deferred_results = cast(
            dict[str, Any],
            to_jsonable_python(results),
        )
        record.segment_index += 1
        record.state = SubagentExecutionState.pending
        record.error = None
        record.completed_at = None
        await self.store.save(record)
        return record

    async def _plan_for_record(
        self,
        record: SubagentExecutionRecord,
    ) -> ResolvedSubagentPlan:
        try:
            plan = self.registry.get_descriptor(record.descriptor_id)
        except KeyError:
            if self.retained_plan_provider is None:
                raise
            retained = await self.retained_plan_provider.load_retained_plan(record)
            if retained is None:
                raise KeyError(f"Unknown subagent descriptor {record.descriptor_id!r}") from None
            self.registry.register_retained(retained)
            plan = self.registry.get_descriptor(record.descriptor_id)
        if (
            plan.descriptor_id != record.descriptor_id
            or plan.spec.route != record.route
            or plan.fingerprint != record.plan_fingerprint
        ):
            raise RuntimeError(f"Subagent plan identity is invalid for execution {record.execution_id!r}")
        return plan

    async def _emit_complete(
        self,
        parent_ctx: AgentContext,
        record: SubagentExecutionRecord,
    ) -> None:
        duration = 0.0
        if record.started_at is not None:
            end = record.completed_at or datetime.now(UTC)
            duration = (end - record.started_at).total_seconds()
        requests = record.usage.get("requests", 0)
        await parent_ctx.emit_event(
            SubagentCompleteEvent(
                event_id=record.execution_id,
                execution_id=record.execution_id,
                mode=record.mode.value,
                parent_logical_run_id=record.parent_logical_run_id,
                agent_id=record.execution_id,
                agent_name=record.route,
                success=record.state is SubagentExecutionState.succeeded,
                request_count=requests if isinstance(requests, int) else 0,
                result_preview=(str(record.output)[:200] if record.output is not None else ""),
                error=record.error or "",
                duration_seconds=duration,
            )
        )

    async def _required_record(
        self,
        execution_id: str | None,
        *,
        owner_scope_id: str | None = None,
    ) -> SubagentExecutionRecord:
        if execution_id is None:
            raise KeyError("Subagent execution id is required")
        record = await self.store.get(
            execution_id,
            owner_scope_id=owner_scope_id,
        )
        if record is None:
            raise KeyError(f"Unknown subagent execution {execution_id!r}")
        return record

    async def _forget_task(self, execution_id: str) -> None:
        async with self._task_lock:
            self._tasks.pop(execution_id, None)
            self._active_run_registries.pop(execution_id, None)


def _apply_driver_input_state(
    record: SubagentExecutionRecord,
    outcome: SubagentDriverOutcome,
) -> None:
    if outcome.input_state is SubagentInputState.accepted:
        raise RuntimeError("A subagent driver outcome must resolve the initial input state")
    if record.input_state is SubagentInputState.applied and outcome.input_state is not SubagentInputState.applied:
        raise RuntimeError("A subagent driver cannot revoke an applied initial input")
    if outcome.input_state is SubagentInputState.rejected and outcome.state not in {
        SubagentExecutionState.failed,
        SubagentExecutionState.cancelled,
        SubagentExecutionState.lost,
    }:
        raise RuntimeError("A rejected initial input cannot produce a successful or suspended execution")
    record.input_state = outcome.input_state


def _authorize_nested_spawn(
    registry: SubagentRegistry,
    *,
    parent_ctx: AgentContext,
    route: str,
) -> None:
    parent_route = parent_ctx.subagent_target
    if parent_route is None:
        return
    parent_descriptor_id = parent_ctx.subagent_descriptor_id
    if parent_descriptor_id is None:
        raise RuntimeError(f"Subagent {parent_route!r} has no exact descriptor identity for nested delegation")
    parent_plan = registry.get_descriptor(parent_descriptor_id)
    if parent_plan.spec.route != parent_route:
        raise RuntimeError(f"Subagent descriptor {parent_descriptor_id!r} does not identify route {parent_route!r}")
    if route not in parent_plan.spec.spawn_targets:
        raise ValueError(f"Subagent {parent_route!r} is not allowed to spawn {route!r}")


def _validate_spawn_replay(
    existing: SubagentExecutionRecord,
    *,
    plan: ResolvedSubagentPlan,
    prompt: str | Sequence[UserContent],
    mode: SubagentExecutionMode,
    resumed_record: SubagentExecutionRecord | None,
) -> None:
    expected_prompt = prompt if isinstance(prompt, str) else list(prompt)
    expected_resumed_from = resumed_record.execution_id if resumed_record is not None else None
    if (
        existing.route != plan.spec.route
        or existing.mode is not mode
        or to_jsonable_python(existing.prompt) != to_jsonable_python(expected_prompt)
        or existing.resumed_from != expected_resumed_from
    ):
        raise ValueError("Subagent idempotency key was reused with different intent")


def _scope_id(context: AgentContext) -> str:
    scope_id = context.delegation_scope_id
    if not isinstance(scope_id, str) or not scope_id:
        raise RuntimeError("Subagent operations require a stable delegation scope")
    return scope_id


def _handle(record: SubagentExecutionRecord) -> SubagentHandle:
    return SubagentHandle(
        execution_id=record.execution_id,
        route=record.route,
        mode=record.mode,
    )


def _prompt_preview(prompt: str | list[UserContent]) -> str:
    if isinstance(prompt, str):
        return prompt[:200]
    return str(prompt)[:200]


def _initial_child_state(
    parent_ctx: AgentContext,
    *,
    logical_run_id: str,
) -> dict[str, Any]:
    """Freeze an independent child state snapshot at the delegation boundary."""
    state = parent_ctx.export_state(include_usage_ledger=False)
    state.run_input_ledger = RunInputLedger(logical_run_id=logical_run_id)
    state.usage_snapshot_entries = {}
    state.user_prompts = None
    state.previous_assistant_response_reference = None
    state.handoff_message = None
    state.deferred_tool_metadata = {}
    state.files_to_inspect = []
    state.tool_proxy = ToolProxyState()
    return cast(dict[str, Any], to_jsonable_python(state))


def _export_context_state(context: AgentContext) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        to_jsonable_python(context.export_state(include_usage_ledger=True)),
    )


def _completion_message(record: SubagentExecutionRecord) -> str:
    payload = {
        "execution_id": record.execution_id,
        "route": record.route,
        "state": record.state.value,
        "output": record.output,
        "error": record.error,
    }
    return (
        "<subagent-completion>\n"
        + json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n</subagent-completion>"
    )
