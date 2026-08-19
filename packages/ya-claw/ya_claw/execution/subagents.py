"""Shared Claw policy for resolving and restoring portable subagent plans."""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

from pydantic_ai import AgentSpec
from pydantic_ai.capabilities import AbstractCapability
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from ya_agent_sdk.capabilities import (
    RuntimeFoundationCapability,
    ToolApprovalCapability,
    ToolObservationCapability,
    ToolRetryCapability,
    ToolTimeoutCapability,
    build_default_capability_catalog,
)
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.inputs import EnqueueReceipt, InputDisposition, InputOrigin
from ya_agent_sdk.subagents import (
    ResolvedSubagentPlan,
    SubagentDeliveryState,
    SubagentDriverOutcome,
    SubagentExecutionMode,
    SubagentExecutionRecord,
    SubagentExecutionState,
    SubagentInputState,
    SubagentPlanDescriptor,
    SubagentPlanResolver,
    SubagentRegistry,
    SubagentSpec,
)

from ya_claw.controller.store import read_run_state_blob_if_exists
from ya_claw.execution.capabilities import ClawToolsCapability
from ya_claw.orm.tables import RunInputInboxRecord, RunRecord, SessionAsyncTaskRecord
from ya_claw.profile_spec import ClawProfileHostConfig


def build_claw_host_capabilities(
    *,
    groups: tuple[str, ...] = (),
    allowlist: frozenset[str] | None = None,
    approval_tools: frozenset[str] = frozenset(),
    approval_mcps: frozenset[str] = frozenset(),
) -> tuple[AbstractCapability[AgentContext], ...]:
    """Build the complete ordered Claw runtime and final-policy grant set."""
    return (
        RuntimeFoundationCapability(),
        ClawToolsCapability(groups=groups, allowlist=allowlist),
        ToolApprovalCapability(tools=approval_tools, toolset_ids=approval_mcps),
        ToolObservationCapability(),
        ToolRetryCapability(),
        ToolTimeoutCapability(),
    )


def build_claw_subagent_plan_resolver(
    spec: AgentSpec | None = None,
) -> SubagentPlanResolver:
    """Build one resolver whose injected grants exactly match child execution."""
    groups, allowlist, approval_tools, approval_mcps = _child_host_policy(spec)
    return SubagentPlanResolver(
        build_default_capability_catalog(),
        host_capabilities=build_claw_host_capabilities(
            groups=groups,
            allowlist=allowlist,
            approval_tools=approval_tools,
            approval_mcps=approval_mcps,
        ),
        restart_durable=True,
    )


def resolve_claw_subagent_plan(spec: SubagentSpec) -> ResolvedSubagentPlan:
    """Resolve a child against the exact Claw grants it will execute with."""
    return build_claw_subagent_plan_resolver(spec.agent).resolve(spec)


def restore_claw_subagent_plan(
    descriptor: SubagentPlanDescriptor,
) -> ResolvedSubagentPlan:
    """Restore a descriptor against grants derived only from its immutable spec."""
    return build_claw_subagent_plan_resolver(descriptor.normalized_agent_spec).restore(descriptor)


@runtime_checkable
class ClawSubagentClient(Protocol):
    """Internal client boundary used by the durable SDK adapters."""

    async def spawn_delegate(
        self,
        *,
        subagent_name: str,
        prompt: str,
        name: str | None,
        context: dict[str, Any] | None,
        sdk_owner_scope_id: str,
        sdk_idempotency_key: str,
        sdk_intent_fingerprint: str,
    ) -> dict[str, Any]: ...

    async def get_async_subagent(
        self,
        *,
        name_or_task_id: str,
    ) -> dict[str, Any]: ...

    async def steer_async_subagent(
        self,
        *,
        name_or_task_id: str,
        prompt: str | None,
        input_parts: list[dict[str, Any]] | None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]: ...

    async def cancel_async_subagent(
        self,
        *,
        name_or_task_id: str,
        reason: str | None,
    ) -> dict[str, Any]: ...


class ClawSubagentExecutionStore:
    """SQL-backed SDK execution records scoped to one parent Claw session."""

    restart_durable = True
    _RECORD_KEY = "sdk_execution_record"

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        parent_session_id: str,
        client: ClawSubagentClient,
    ) -> None:
        self._session_factory = session_factory
        self._parent_session_id = parent_session_id
        self._client = client

    async def close(self) -> None:
        """The application owns the SQL engine and internal HTTP client."""

    async def load_retained_plan(
        self,
        record: SubagentExecutionRecord,
    ) -> ResolvedSubagentPlan | None:
        """Restore the exact server-owned plan persisted with an execution."""
        self._require_owner_scope(record)
        async with self._session_factory() as db_session:
            task = await self._task_by_execution_id(db_session, record.execution_id)
            if task is None:
                return None
            if not isinstance(task.plan_descriptor, dict):
                raise TypeError(f"Subagent execution {record.execution_id!r} has no immutable plan descriptor")
            descriptor = SubagentPlanDescriptor.model_validate(task.plan_descriptor)
        if (
            descriptor.descriptor_id != record.descriptor_id
            or descriptor.descriptor_id != task.plan_descriptor_ref
            or descriptor.fingerprint != record.plan_fingerprint
            or descriptor.fingerprint != task.plan_fingerprint
            or descriptor.spec.route != record.route
            or descriptor.spec.route != task.subagent_name
        ):
            raise RuntimeError(f"Subagent plan identity is invalid for execution {record.execution_id!r}")
        return restore_claw_subagent_plan(descriptor)

    async def create(
        self,
        record: SubagentExecutionRecord,
    ) -> SubagentExecutionRecord:
        self._require_owner_scope(record)
        existing = await self.get_by_idempotency_key(
            record.idempotency_key,
            owner_scope_id=record.owner_scope_id,
        )
        if existing is not None:
            if _spawn_intent_fingerprint(existing) != _spawn_intent_fingerprint(record):
                raise ValueError("Claw SDK idempotency key was reused with different intent")
            return existing
        context = {
            "_sdk_execution_record": record.model_dump(mode="json"),
            "mode": record.mode.value,
            "plan_fingerprint": record.plan_fingerprint,
        }
        if record.resumed_from is not None:
            context["_sdk_resumed_from"] = record.resumed_from
        intent_fingerprint = _spawn_intent_fingerprint(record)
        await self._client.spawn_delegate(
            subagent_name=record.route,
            prompt=_prompt_text(record.prompt),
            name=record.execution_id,
            context=context,
            sdk_owner_scope_id=record.owner_scope_id,
            sdk_idempotency_key=record.idempotency_key,
            sdk_intent_fingerprint=intent_fingerprint,
        )
        created = await self.get_by_idempotency_key(
            record.idempotency_key,
            owner_scope_id=record.owner_scope_id,
        )
        if created is None:
            raise RuntimeError("Committed Claw subagent task is unavailable after spawn")
        return await self.save(created)

    async def save(
        self,
        record: SubagentExecutionRecord,
    ) -> SubagentExecutionRecord:
        self._require_owner_scope(record)
        async with self._session_factory() as db_session:
            task = await self._task_by_execution_id(
                db_session,
                record.execution_id,
                for_update=True,
            )
            if task is None:
                raise KeyError(record.execution_id)
            synchronized = await self._synchronize(db_session, task, record)
            metadata = dict(task.task_metadata or {})
            metadata[self._RECORD_KEY] = synchronized.model_dump(mode="json")
            task.task_metadata = metadata
            await db_session.commit()
            return synchronized.model_copy(deep=True)

    async def get(
        self,
        execution_id: str,
        *,
        owner_scope_id: str | None = None,
    ) -> SubagentExecutionRecord | None:
        if owner_scope_id is not None and owner_scope_id != self._parent_session_id:
            return None
        async with self._session_factory() as db_session:
            task = await self._task_by_execution_id(db_session, execution_id)
            if task is None:
                return None
            record = self._record_from_task(task)
            if record is None or record.owner_scope_id != self._parent_session_id:
                return None
            return await self._synchronize(db_session, task, record)

    async def get_by_idempotency_key(
        self,
        idempotency_key: str,
        *,
        owner_scope_id: str,
    ) -> SubagentExecutionRecord | None:
        if owner_scope_id != self._parent_session_id:
            return None
        async with self._session_factory() as db_session:
            result = await db_session.execute(
                select(SessionAsyncTaskRecord)
                .where(
                    SessionAsyncTaskRecord.sdk_owner_scope_id == owner_scope_id,
                    SessionAsyncTaskRecord.sdk_idempotency_key == idempotency_key,
                )
                .limit(1)
            )
            task = result.scalar_one_or_none()
            if not isinstance(task, SessionAsyncTaskRecord):
                return None
            record = self._record_from_task(task)
            if record is None or record.owner_scope_id != owner_scope_id:
                raise RuntimeError("Claw SDK idempotency row has no matching execution record")
            return await self._synchronize(db_session, task, record)

    async def list(
        self,
        *,
        owner_scope_id: str | None = None,
    ) -> tuple[SubagentExecutionRecord, ...]:
        if owner_scope_id is not None and owner_scope_id != self._parent_session_id:
            return ()
        async with self._session_factory() as db_session:
            result = await db_session.execute(
                select(SessionAsyncTaskRecord)
                .where(SessionAsyncTaskRecord.parent_session_id == self._parent_session_id)
                .order_by(SessionAsyncTaskRecord.created_at.asc())
            )
            records: list[SubagentExecutionRecord] = []
            for task in result.scalars().all():
                record = self._record_from_task(task)
                if record is not None and record.owner_scope_id == self._parent_session_id:
                    records.append(await self._synchronize(db_session, task, record))
            return tuple(records)

    def _require_owner_scope(self, record: SubagentExecutionRecord) -> None:
        if record.owner_scope_id != self._parent_session_id:
            raise ValueError("Claw subagent execution owner scope must match its parent session")

    async def _task_by_execution_id(
        self,
        db_session: AsyncSession,
        execution_id: str,
        *,
        for_update: bool = False,
    ) -> SessionAsyncTaskRecord | None:
        statement = (
            select(SessionAsyncTaskRecord)
            .where(
                SessionAsyncTaskRecord.parent_session_id == self._parent_session_id,
                SessionAsyncTaskRecord.name == execution_id,
            )
            .limit(1)
        )
        if for_update:
            statement = statement.with_for_update()
        result = await db_session.execute(statement)
        task = result.scalar_one_or_none()
        return task if isinstance(task, SessionAsyncTaskRecord) else None

    def _record_from_task(
        self,
        task: SessionAsyncTaskRecord,
    ) -> SubagentExecutionRecord | None:
        metadata = task.task_metadata
        raw = metadata.get(self._RECORD_KEY) if isinstance(metadata, dict) else None
        if not isinstance(raw, dict):
            context = metadata.get("context") if isinstance(metadata, dict) else None
            raw = context.get("_sdk_execution_record") if isinstance(context, dict) else None
        return SubagentExecutionRecord.model_validate(raw) if isinstance(raw, dict) else None

    async def _synchronize(
        self,
        db_session: AsyncSession,
        task: SessionAsyncTaskRecord,
        record: SubagentExecutionRecord,
    ) -> SubagentExecutionRecord:
        synchronized = record.model_copy(deep=True)
        if isinstance(task.sdk_input_state, str):
            persisted_input_state = SubagentInputState(task.sdk_input_state)
            if (
                synchronized.input_state is SubagentInputState.applied
                and persisted_input_state is not SubagentInputState.applied
            ):
                raise RuntimeError("Claw task cannot revoke an applied SDK initial input")
            synchronized.input_state = persisted_input_state
        if task.status == "running" and synchronized.state is SubagentExecutionState.pending:
            synchronized.state = SubagentExecutionState.running
            synchronized.started_at = synchronized.started_at or task.updated_at
        terminal_states = {
            "completed": SubagentExecutionState.succeeded,
            "failed": SubagentExecutionState.failed,
            "cancelled": SubagentExecutionState.cancelled,
        }
        state = terminal_states.get(task.status)
        if state is not None:
            synchronized.state = state
            synchronized.completed_at = task.completed_at or task.updated_at
            synchronized.error = task.error_message
            run_id = task.result_run_id or task.task_run_id
            run = await db_session.get(RunRecord, run_id) if isinstance(run_id, str) else None
            if isinstance(run, RunRecord):
                synchronized.output = run.output_json if run.output_json is not None else run.output_text
        if synchronized.mode is SubagentExecutionMode.background:
            if task.delivery_status == "applied":
                synchronized.delivery_state = SubagentDeliveryState.delivered
            elif synchronized.terminal:
                synchronized.delivery_state = SubagentDeliveryState.pending
        return synchronized


class ClawSubagentCompletionDelivery:
    """Observe the canonical SQL-owned parent completion delivery."""

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        parent_session_id: str,
    ) -> None:
        self._session_factory = session_factory
        self._parent_session_id = parent_session_id

    async def deliver(
        self,
        record: SubagentExecutionRecord,
        parent_ctx: AgentContext,
        message: str,
    ) -> EnqueueReceipt | None:
        del parent_ctx, message
        if record.owner_scope_id != self._parent_session_id:
            raise ValueError("Claw completion owner scope must match its parent session")
        async with self._session_factory() as db_session:
            result = await db_session.execute(
                select(SessionAsyncTaskRecord)
                .where(
                    SessionAsyncTaskRecord.parent_session_id == self._parent_session_id,
                    SessionAsyncTaskRecord.name == record.execution_id,
                )
                .limit(1)
            )
            task = result.scalar_one_or_none()
            if (
                not isinstance(task, SessionAsyncTaskRecord)
                or not isinstance(task.delivery_id, str)
                or not isinstance(task.delivery_status, str)
            ):
                return None
            delivery_id = task.delivery_id
            delivery_run_id = task.delivery_run_id
            status = task.delivery_status
            input_id = delivery_id
            enqueue_id: str | None = delivery_run_id
            if isinstance(delivery_run_id, str):
                inbox_result = await db_session.execute(
                    select(RunInputInboxRecord)
                    .where(
                        RunInputInboxRecord.run_id == delivery_run_id,
                        RunInputInboxRecord.delivery_key == delivery_id,
                    )
                    .order_by(RunInputInboxRecord.created_at.desc())
                    .limit(1)
                )
                inbox = inbox_result.scalar_one_or_none()
                if isinstance(inbox, RunInputInboxRecord):
                    status = inbox.status
                    input_id = inbox.sdk_input_id or inbox.id
                    enqueue_id = inbox.enqueue_id
                else:
                    delivery_run = await db_session.get(RunRecord, delivery_run_id)
                    if isinstance(delivery_run, RunRecord) and delivery_run.source_delivery_id == delivery_id:
                        if delivery_run.source_delivery_applied_at is not None:
                            status = InputDisposition.applied.value
                        elif delivery_run.status in {"queued", "running"}:
                            status = InputDisposition.enqueued.value
                        else:
                            status = InputDisposition.rejected.value
            try:
                disposition = InputDisposition(status)
            except ValueError as exc:
                raise RuntimeError(f"Unknown Claw completion delivery status {status!r}") from exc
        logical_run_id = record.parent_logical_run_id or delivery_run_id or self._parent_session_id
        return EnqueueReceipt(
            logical_run_id=logical_run_id,
            input_id=input_id,
            disposition=disposition,
            enqueue_id=enqueue_id,
        )


class ClawSubagentDriver:
    """Restart-durable driver over Claw child sessions and supervised runs."""

    restart_durable = True

    def __init__(
        self,
        *,
        client: ClawSubagentClient,
        data_settings: Any,
        poll_interval_seconds: float = 0.2,
    ) -> None:
        self._client = client
        self._settings = data_settings
        self._poll_interval_seconds = poll_interval_seconds

    async def run(
        self,
        plan: ResolvedSubagentPlan,
        record: SubagentExecutionRecord,
        parent_ctx: AgentContext,
    ) -> SubagentDriverOutcome:
        del plan, parent_ctx
        while True:
            payload = await self._client.get_async_subagent(name_or_task_id=record.execution_id)
            task = _task_payload(payload)
            status = task.get("status")
            if status in {"completed", "failed", "cancelled"}:
                break
            await asyncio.sleep(self._poll_interval_seconds)
        state = {
            "completed": SubagentExecutionState.succeeded,
            "failed": SubagentExecutionState.failed,
            "cancelled": SubagentExecutionState.cancelled,
        }[str(status)]
        output = task.get("output_json")
        if output is None:
            output = task.get("output_text")
        history = _history_from_task(self._settings, task)
        raw_input_state = task.get("sdk_input_state")
        if not isinstance(raw_input_state, str):
            raise TypeError("Claw task has no persisted SDK initial-input admission state")
        input_state = SubagentInputState(raw_input_state)
        if input_state is SubagentInputState.accepted:
            raise RuntimeError("Terminal Claw task still has an unresolved SDK initial input")
        return SubagentDriverOutcome(
            state=state,
            input_state=input_state,
            output=output,
            error=(str(task.get("error_message")) if task.get("error_message") is not None else None),
            history=history,
        )

    async def steer(
        self,
        record: SubagentExecutionRecord,
        *content: Any,
        origin: InputOrigin,
        idempotency_key: str | None,
    ) -> EnqueueReceipt:
        del origin
        prompt = "\n".join(str(item) for item in content)
        key = idempotency_key or f"subagent-steer:{record.execution_id}:{uuid4()}"
        payload = await self._client.steer_async_subagent(
            name_or_task_id=record.execution_id,
            prompt=prompt,
            input_parts=None,
            idempotency_key=key,
        )
        task = _task_payload(payload)
        input_id = task.get("input_id")
        disposition = task.get("input_disposition")
        if not isinstance(input_id, str) or not isinstance(disposition, str):
            raise TypeError("Claw steering response has no durable input receipt")
        sdk_input_id = task.get("input_sdk_id")
        enqueue_id = task.get("input_enqueue_id")
        return EnqueueReceipt(
            logical_run_id=record.child_logical_run_id,
            input_id=(sdk_input_id if isinstance(sdk_input_id, str) else input_id),
            disposition=InputDisposition(disposition),
            enqueue_id=(enqueue_id if isinstance(enqueue_id, str) else None),
        )

    async def cancel(self, record: SubagentExecutionRecord) -> None:
        await self._client.cancel_async_subagent(
            name_or_task_id=record.execution_id,
            reason="Cancelled through the portable subagent service",
        )
        while True:
            payload = await self._client.get_async_subagent(name_or_task_id=record.execution_id)
            if _task_payload(payload).get("status") in {
                "completed",
                "failed",
                "cancelled",
            }:
                return
            await asyncio.sleep(self._poll_interval_seconds)


def build_claw_delegation_service(
    *,
    registry: SubagentRegistry,
    session_factory: async_sessionmaker[AsyncSession],
    parent_session_id: str,
    client: ClawSubagentClient,
    settings: Any,
) -> Any:
    """Construct the SDK service over Claw's durable SQL/run adapters."""
    from ya_agent_sdk.subagents import SubagentExecutionService

    store = ClawSubagentExecutionStore(
        session_factory=session_factory,
        parent_session_id=parent_session_id,
        client=client,
    )
    driver = ClawSubagentDriver(client=client, data_settings=settings)
    return SubagentExecutionService(
        registry,
        store,
        driver,
        completion_delivery=ClawSubagentCompletionDelivery(
            session_factory=session_factory,
            parent_session_id=parent_session_id,
        ),
        retained_plan_provider=store,
    )


def _task_payload(payload: dict[str, Any]) -> dict[str, Any]:
    task = payload.get("task")
    if not isinstance(task, dict):
        raise TypeError("Claw durable subagent response has no task payload")
    return task


def _prompt_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    return json.dumps(prompt, ensure_ascii=False, sort_keys=True, default=str)


def _spawn_intent_fingerprint(record: SubagentExecutionRecord) -> str:
    payload = record.model_dump(
        mode="json",
        include={
            "owner_scope_id",
            "idempotency_key",
            "descriptor_id",
            "plan_fingerprint",
            "route",
            "mode",
            "parent_agent_id",
            "parent_logical_run_id",
            "depth",
            "prompt",
            "resumed_from",
        },
    )
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _history_from_task(settings: Any, task: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    run_id = task.get("result_run_id") or task.get("task_run_id")
    if not isinstance(run_id, str):
        return ()
    payload = read_run_state_blob_if_exists(settings, run_id)
    if not isinstance(payload, dict):
        return ()
    history = payload.get("message_history")
    if not isinstance(history, list):
        return ()
    return tuple(dict(item) for item in history if isinstance(item, dict))


def _child_host_policy(
    spec: AgentSpec | None,
) -> tuple[
    tuple[str, ...],
    frozenset[str] | None,
    frozenset[str],
    frozenset[str],
]:
    metadata = spec.metadata if isinstance(spec, AgentSpec) else None
    claw = metadata.get("claw") if isinstance(metadata, dict) else None
    if not isinstance(claw, dict):
        return (), None, frozenset(), frozenset()
    host = ClawProfileHostConfig.model_validate(claw)
    return (
        host.tool_groups,
        None,
        frozenset(host.need_user_approve_tools),
        frozenset(host.need_user_approve_mcps),
    )
