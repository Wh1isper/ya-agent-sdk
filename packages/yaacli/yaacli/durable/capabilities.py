"""Capabilities that connect native Pydantic AI runs to the durable inbox."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass
from typing import Any

from pydantic import TypeAdapter
from pydantic_ai import DeferredToolRequests, EnqueuedMessagesEvent, RunContext, UserContent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_graph import End
from ya_agent_sdk.inputs import InputDisposition, InputOrigin

from yaacli.durable.bindings import runtime_bindings
from yaacli.durable.models import InputRecord, InputState
from yaacli.session import TUIContext

_USER_CONTENT = TypeAdapter(list[UserContent])


@dataclass(kw_only=True)
class DurableInboxPumpCapability(AbstractCapability[TUIContext]):
    """Drain persisted product input at native graph boundaries."""

    id: str | None = "yaacli_durable_inbox_v2"

    async def after_node_run(
        self,
        ctx: RunContext[TUIContext],
        *,
        node: Any,
        result: Any,
    ) -> Any:
        deps = ctx.deps
        binding_ref = deps.durable_binding_ref
        logical_run_id = deps.durable_logical_run_id
        if binding_ref is None or logical_run_id is None:
            return result
        if isinstance(result, End) and isinstance(result.data.output, DeferredToolRequests):
            return result

        store = runtime_bindings.get(binding_ref).store
        self._sync_applied_inputs(deps)
        if isinstance(result, End):
            pending = store.close_and_list_inputs(logical_run_id)
        else:
            pending = store.list_inputs(
                logical_run_id,
                states=(InputState.accepted, InputState.enqueued),
            )
        self._enqueue_pending(ctx, pending)
        return result

    def _enqueue_pending(
        self,
        ctx: RunContext[TUIContext],
        pending: tuple[InputRecord, ...],
    ) -> None:
        deps = ctx.deps
        if deps.durable_binding_ref is None:
            return
        store = runtime_bindings.get(deps.durable_binding_ref).store
        for item in pending:
            content = _USER_CONTENT.validate_python(item.content)
            prompt_content: str | list[UserContent]
            prompt_content = content[0] if len(content) == 1 and isinstance(content[0], str) else content
            ledger_record = deps.run_input_ledger.accept(
                [ModelRequest(parts=[UserPromptPart(content=prompt_content)])],
                origin=(InputOrigin.user if item.origin == "user" else InputOrigin.feature),
                priority=item.priority.value,
                input_id=item.input_id,
            )
            native_attempt_id = ctx.run_id or deps.run_id
            if ledger_record.disposition in (InputDisposition.applied, InputDisposition.rejected) or any(
                attempt.native_attempt_id == native_attempt_id for attempt in ledger_record.enqueue_attempts
            ):
                continue
            enqueue_id = ctx.enqueue(*content, priority=item.priority.value)
            if enqueue_id is None:  # pragma: no cover - non-empty store invariant
                continue
            deps.run_input_ledger.mark_enqueued(
                ledger_record.input_id,
                native_attempt_id=native_attempt_id,
                enqueue_id=enqueue_id,
            )
            store.transition_input(
                item.input_id,
                item.state,
                InputState.enqueued,
                native_enqueue_id=enqueue_id,
            )

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

    @staticmethod
    def _mark_applied(deps: TUIContext, enqueue_id: str) -> None:
        deps.run_input_ledger.mark_applied_by_enqueue_id(enqueue_id)
        DurableInboxPumpCapability._sync_applied_inputs(deps)

    @staticmethod
    def _sync_applied_inputs(deps: TUIContext) -> None:
        binding_ref = deps.durable_binding_ref
        logical_run_id = deps.durable_logical_run_id
        if binding_ref is None or logical_run_id is None:
            return
        store = runtime_bindings.get(binding_ref).store
        for item in store.list_inputs(
            logical_run_id,
            states=(InputState.enqueued,),
        ):
            ledger_record = deps.run_input_ledger.find(item.input_id)
            if ledger_record is not None and ledger_record.disposition is InputDisposition.applied:
                store.transition_input(
                    item.input_id,
                    InputState.enqueued,
                    InputState.applied,
                )
