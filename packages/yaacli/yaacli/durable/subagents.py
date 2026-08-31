"""YAACLI runtime adapters for file-backed process-local SDK subagents."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Sequence
from dataclasses import dataclass, replace
from typing import Any, ClassVar
from uuid import uuid4

from pydantic import TypeAdapter
from pydantic_ai import (
    DeferredToolRequests,
    DeferredToolResults,
    EnqueuedMessagesEvent,
    RunContext,
    UserContent,
)
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelRequest,
    UserPromptPart,
)
from pydantic_graph import End
from ya_agent_sdk.agents.retry_recovery import DEFAULT_STREAM_RESUME_PROMPT
from ya_agent_sdk.context import (
    AgentContext,
    ModelConfig,
    StreamRecoveryPolicy,
)
from ya_agent_sdk.inputs import (
    EnqueueReceipt,
    InputDisposition,
    InputOrigin,
    LogicalRunInputRouter,
)
from ya_agent_sdk.subagents import (
    AsyncioSubagentExecutionHost,
    InProcessSubagentDriver,
    SubagentPlanResolver,
)
from ya_agent_sdk.subagents.spec import (
    ResolvedSubagentPlan,
    SubagentDriverOutcome,
    SubagentExecutionRecord,
)

from yaacli.durable.bindings import runtime_bindings
from yaacli.durable.file_subagents import FileSubagentExecutionStore
from yaacli.durable.models import (
    InputPriority,
    InputRecord,
    InputState,
)
from yaacli.durable.store import (
    InvalidTransitionError,
    TombstonedSessionError,
)
from yaacli.session import TUIContext
from yaacli.subagent_config import model_cfg_from_agent_spec

_DEFERRED_RESULTS = TypeAdapter(DeferredToolResults)
_USER_CONTENT = TypeAdapter(list[UserContent])


@dataclass(frozen=True, slots=True)
class FileRetainedSubagentPlanProvider:
    """Lazily restore an exact historical child plan from its execution file."""

    store: FileSubagentExecutionStore
    resolver: SubagentPlanResolver

    async def load_retained_plan(
        self,
        record: SubagentExecutionRecord,
    ) -> ResolvedSubagentPlan | None:
        descriptor = self.store.get_descriptor(record.descriptor_id)
        if descriptor is None:
            return None
        return self.resolver.restore(descriptor)


class DurableSubagentCompletionDelivery:
    """Persist child completion into the active compatible session run."""

    def __init__(self, binding_ref: str) -> None:
        self.binding_ref = binding_ref

    async def deliver(
        self,
        record: SubagentExecutionRecord,
        parent_ctx: AgentContext,
        message: str,
    ) -> EnqueueReceipt | None:
        if not isinstance(parent_ctx, TUIContext):
            return None
        parent_logical_run_id = record.parent_logical_run_id
        if parent_logical_run_id is None:
            return None
        store = runtime_bindings.get(self.binding_ref).store
        source_run = store.get_run(parent_logical_run_id)
        if source_run is None:
            return None

        if record.delivery_logical_run_id is not None and record.delivery_input_id is not None:
            target_run = store.get_run(record.delivery_logical_run_id)
            if target_run is None or target_run.session_id != source_run.session_id:
                return EnqueueReceipt(
                    logical_run_id=record.delivery_logical_run_id,
                    input_id=record.delivery_input_id,
                    disposition=InputDisposition.rejected,
                )
            existing = next(
                (
                    item
                    for item in store.list_inputs(record.delivery_logical_run_id)
                    if item.input_id == record.delivery_input_id
                ),
                None,
            )
            if existing is None:
                return EnqueueReceipt(
                    logical_run_id=record.delivery_logical_run_id,
                    input_id=record.delivery_input_id,
                    disposition=InputDisposition.rejected,
                )
            return EnqueueReceipt(
                logical_run_id=existing.logical_run_id,
                input_id=existing.input_id,
                disposition=InputDisposition(existing.state.value),
                enqueue_id=existing.native_enqueue_id,
            )

        current_logical_run_id = parent_ctx.durable_logical_run_id
        if current_logical_run_id is None:
            return None
        current_run = store.get_run(current_logical_run_id)
        if current_run is None or current_run.session_id != source_run.session_id:
            return None
        try:
            accepted = store.accept_input(
                current_logical_run_id,
                [message],
                idempotency_key=f"subagent-completion:{record.execution_id}",
                priority=InputPriority.asap,
                origin="feature",
            )
        except (InvalidTransitionError, TombstonedSessionError):
            return None
        return EnqueueReceipt(
            logical_run_id=accepted.logical_run_id,
            input_id=accepted.input_id,
            disposition=InputDisposition(accepted.state.value),
            enqueue_id=accepted.native_enqueue_id,
        )


@dataclass(kw_only=True)
class DurableSubagentInboxCapability(AbstractCapability[TUIContext]):
    """Apply persisted process-local child steering at native graph boundaries."""

    store: FileSubagentExecutionStore
    id: str | None = "yaacli_durable_subagent_inbox_v2"

    _safe_at_runtime: ClassVar[bool] = False

    @classmethod
    def get_serialization_name(cls) -> None:
        return None

    async def after_node_run(
        self,
        ctx: RunContext[TUIContext],
        *,
        node: Any,
        result: Any,
    ) -> Any:
        del node
        execution_id = ctx.deps.agent_id
        self._sync_applied_inputs(ctx.deps)
        if isinstance(result, End) and isinstance(result.data.output, DeferredToolRequests):
            return result
        pending = (
            self.store.close_and_list_inputs(execution_id)
            if isinstance(result, End)
            else self.store.list_inputs(
                execution_id,
                states=(InputState.accepted, InputState.enqueued),
            )
        )
        self._enqueue_pending(ctx, pending)
        return result

    def _enqueue_pending(
        self,
        ctx: RunContext[TUIContext],
        pending: tuple[InputRecord, ...],
    ) -> None:
        for item in pending:
            content = _USER_CONTENT.validate_python(item.content)
            prompt_content: str | list[UserContent]
            prompt_content = content[0] if len(content) == 1 and isinstance(content[0], str) else content
            ledger_record = ctx.deps.run_input_ledger.accept(
                [ModelRequest(parts=[UserPromptPart(content=prompt_content)])],
                origin=(InputOrigin.user if item.origin == "user" else InputOrigin.feature),
                priority=item.priority.value,
                input_id=item.input_id,
            )
            if ledger_record.disposition is InputDisposition.applied:
                self.store.transition_input(item.input_id, item.state, InputState.applied)
                continue
            if ledger_record.disposition is InputDisposition.rejected:
                continue
            native_attempt_id = self._current_native_attempt_id(ctx)
            if native_attempt_id is None:
                continue
            current_attempt = next(
                (
                    attempt
                    for attempt in ledger_record.enqueue_attempts
                    if attempt.native_attempt_id == native_attempt_id
                ),
                None,
            )
            if current_attempt is not None:
                self.store.transition_input(
                    item.input_id,
                    item.state,
                    InputState.enqueued,
                    native_enqueue_id=current_attempt.enqueue_id,
                )
                continue
            enqueue_id = ctx.enqueue(*content, priority=item.priority.value)
            if enqueue_id is None:  # pragma: no cover - non-empty store invariant
                continue
            ctx.deps.run_input_ledger.mark_enqueued(
                ledger_record.input_id,
                native_attempt_id=native_attempt_id,
                enqueue_id=enqueue_id,
            )
            self.store.transition_input(
                item.input_id,
                item.state,
                InputState.enqueued,
                native_enqueue_id=enqueue_id,
            )

    @staticmethod
    def _current_native_attempt_id(ctx: RunContext[TUIContext]) -> str | None:
        router = ctx.deps.input_router
        if isinstance(router, LogicalRunInputRouter):
            return router.current_native_attempt_id
        return ctx.run_id or ctx.deps.run_id

    async def wrap_run_event_stream(
        self,
        ctx: RunContext[TUIContext],
        *,
        stream: AsyncIterable[Any],
    ) -> AsyncIterator[Any]:
        async for event in stream:
            if isinstance(event, EnqueuedMessagesEvent):
                self._mark_applied(ctx.deps, event.enqueue_id)
            yield event

    def _mark_applied(self, deps: TUIContext, enqueue_id: str) -> None:
        deps.run_input_ledger.mark_applied_by_enqueue_id(enqueue_id)
        self._sync_applied_inputs(deps)

    def _sync_applied_inputs(self, deps: TUIContext) -> None:
        for item in self.store.list_inputs(
            deps.agent_id,
            states=(InputState.accepted, InputState.enqueued),
        ):
            ledger_record = deps.run_input_ledger.find(item.input_id)
            if ledger_record is not None and ledger_record.disposition is InputDisposition.applied:
                self.store.transition_input(
                    item.input_id,
                    item.state,
                    InputState.applied,
                )


class LocalProcessorSubagentExecutionHost(AsyncioSubagentExecutionHost):
    """Own fully asynchronous child tasks at the YAACLI processor boundary."""


class LocalSubagentDriver:
    """Persist host steering while composing SDK in-process child execution."""

    restart_durable = False

    def __init__(
        self,
        *,
        store: FileSubagentExecutionStore,
        request_limit: int,
        default_model_cfg: ModelConfig,
        custom_capability_types: Sequence[type[Any]] = (),
        runtime_capabilities: Sequence[AbstractCapability[Any]] = (),
    ) -> None:
        self.store = store
        self.default_model_cfg = default_model_cfg.model_copy(deep=True)
        self.runtime_capabilities = tuple(runtime_capabilities)
        self._driver = InProcessSubagentDriver(
            custom_capability_types=custom_capability_types,
            request_limit=request_limit,
            child_context_configurer=self._configure_child_context,
        )

    def _configure_child_context(
        self,
        plan: ResolvedSubagentPlan,
        record: SubagentExecutionRecord,
        parent_ctx: AgentContext,
        child_ctx: AgentContext,
    ) -> None:
        del parent_ctx
        if not isinstance(child_ctx, TUIContext):
            raise TypeError("YAACLI subagents require TUIContext")
        child_model_cfg = model_cfg_from_agent_spec(plan.normalized_agent_spec)
        child_ctx.model_cfg = (
            child_model_cfg if child_model_cfg is not None else self.default_model_cfg.model_copy(deep=True)
        )
        inherited_policy = child_ctx.stream_recovery_policy
        child_ctx.stream_recovery_policy = StreamRecoveryPolicy(
            enabled=child_ctx.model_cfg.stream_resume_on_error,
            max_attempts=child_ctx.model_cfg.stream_resume_max_attempts,
            transport_max_attempts=child_ctx.model_cfg.stream_transport_resume_max_attempts,
            resume_prompt=(
                child_ctx.model_cfg.stream_resume_prompt
                or (inherited_policy.resume_prompt if inherited_policy is not None else DEFAULT_STREAM_RESUME_PROMPT)
            ),
            resume_prompt_factory=(inherited_policy.resume_prompt_factory if inherited_policy is not None else None),
        )
        child_ctx.parent_run_id = record.parent_logical_run_id
        child_ctx.provider_session_id = record.execution_id
        child_ctx.provider_thread_id = record.root_execution_id
        child_ctx.runtime_descriptor_id = record.parent_runtime_descriptor_id
        child_ctx.durable_logical_run_id = record.child_logical_run_id
        child_ctx.model_profile_instructions = None
        child_ctx.shell_env = {}
        child_ctx.files_to_inspect = []
        child_ctx.goal_task = None
        child_ctx.goal_iteration = 0
        child_ctx.goal_needs_post_restore_audit = False
        child_ctx.goal_last_context_handoff_source = None

    async def run(
        self,
        plan: ResolvedSubagentPlan,
        record: SubagentExecutionRecord,
        parent_ctx: AgentContext,
    ) -> SubagentDriverOutcome:
        self.store.put_descriptor(plan)
        self.store.require_executable(record.execution_id)
        execution_plan = replace(
            plan,
            host_capabilities=(*plan.host_capabilities, *self.runtime_capabilities),
        )
        return await self._driver.run(execution_plan, record, parent_ctx)

    async def steer(
        self,
        record: SubagentExecutionRecord,
        *content: Any,
        origin: InputOrigin,
        idempotency_key: str | None,
    ) -> EnqueueReceipt:
        input_key = idempotency_key or str(uuid4())
        try:
            accepted = self.store.accept_input(
                record.execution_id,
                content,
                idempotency_key=input_key,
                origin=origin,
            )
        except (InvalidTransitionError, TombstonedSessionError):
            return EnqueueReceipt(
                logical_run_id=record.child_logical_run_id,
                input_id=input_key,
                disposition=InputDisposition.rejected,
            )
        return EnqueueReceipt(
            logical_run_id=record.child_logical_run_id,
            input_id=accepted.input_id,
            disposition=InputDisposition(accepted.state.value),
            enqueue_id=accepted.native_enqueue_id,
        )

    async def cancel(self, record: SubagentExecutionRecord) -> None:
        await self._driver.cancel(record)
