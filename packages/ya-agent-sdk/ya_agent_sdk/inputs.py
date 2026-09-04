"""Logical-run input identity, retention, and native enqueue routing."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field
from pydantic_ai import AgentRun
from pydantic_ai._enqueue import EnqueueContent, PendingMessage, PendingMessagePriority
from pydantic_ai.messages import EnqueuedMessagesEvent, ModelMessage, ModelMessagesTypeAdapter


class InputDisposition(StrEnum):
    """Durable disposition of one accepted logical input."""

    accepted = "accepted"
    enqueued = "enqueued"
    applied = "applied"
    rejected = "rejected"


class InputOrigin(StrEnum):
    user = "user"
    feature = "feature"


class EnqueueAttempt(BaseModel):
    native_attempt_id: str
    enqueue_id: str


class RunInputRecord(BaseModel):
    input_id: str = Field(default_factory=lambda: str(uuid4()))
    origin: InputOrigin
    messages: list[ModelMessage]
    priority: PendingMessagePriority = "asap"
    disposition: InputDisposition = InputDisposition.accepted
    enqueue_attempts: list[EnqueueAttempt] = Field(default_factory=list)
    rejection_reason: str | None = None

    @property
    def latest_enqueue_id(self) -> str | None:
        return self.enqueue_attempts[-1].enqueue_id if self.enqueue_attempts else None


class RunInputLedger(BaseModel):
    """Versioned retention truth for structured logical-run input."""

    schema_version: Literal[1] = 1
    logical_run_id: str = Field(default_factory=lambda: str(uuid4()))
    records: list[RunInputRecord] = Field(default_factory=list)

    def accept(
        self,
        messages: Sequence[ModelMessage],
        *,
        origin: InputOrigin,
        priority: PendingMessagePriority,
        input_id: str | None = None,
    ) -> RunInputRecord:
        """Accept one input or return its identical existing durable record."""
        if input_id is not None:
            if not input_id.strip():
                raise ValueError("Logical input identity must not be empty")
            existing = self.find(input_id)
            if existing is not None:
                if (
                    existing.origin is not origin
                    or existing.priority != priority
                    or _message_identity(existing.messages) != _message_identity(messages)
                ):
                    raise ValueError(f"Logical input identity {input_id!r} was reused with different content")
                return existing
        record = RunInputRecord(
            input_id=input_id or str(uuid4()),
            origin=origin,
            messages=list(messages),
            priority=priority,
        )
        self.records.append(record)
        return record

    def record_initial(
        self,
        messages: Sequence[ModelMessage],
        *,
        origin: InputOrigin = InputOrigin.user,
    ) -> RunInputRecord:
        record = self.accept(messages, origin=origin, priority="asap")
        record.disposition = InputDisposition.applied
        return record

    def mark_enqueued(
        self,
        input_id: str,
        *,
        native_attempt_id: str,
        enqueue_id: str,
    ) -> None:
        record = self.get(input_id)
        if record.disposition in (InputDisposition.applied, InputDisposition.rejected):
            return
        if any(attempt.native_attempt_id == native_attempt_id for attempt in record.enqueue_attempts):
            return
        record.enqueue_attempts.append(
            EnqueueAttempt(
                native_attempt_id=native_attempt_id,
                enqueue_id=enqueue_id,
            )
        )
        record.disposition = InputDisposition.enqueued

    def mark_applied_by_enqueue_id(self, enqueue_id: str) -> str | None:
        for record in self.records:
            if any(item.enqueue_id == enqueue_id for item in record.enqueue_attempts):
                if record.disposition is not InputDisposition.rejected:
                    record.disposition = InputDisposition.applied
                return record.input_id
        return None

    def reject(self, input_id: str, reason: str) -> None:
        record = self.get(input_id)
        if record.disposition is InputDisposition.applied:
            return
        record.disposition = InputDisposition.rejected
        record.rejection_reason = reason

    def find(self, input_id: str) -> RunInputRecord | None:
        for record in self.records:
            if record.input_id == input_id:
                return record
        return None

    def get(self, input_id: str) -> RunInputRecord:
        record = self.find(input_id)
        if record is not None:
            return record
        raise KeyError(input_id)

    def unresolved(self) -> tuple[RunInputRecord, ...]:
        return tuple(
            record
            for record in self.records
            if record.disposition in (InputDisposition.accepted, InputDisposition.enqueued)
        )

    def retained_user_messages(
        self,
        *,
        delivered_input_ids: Collection[str] = (),
    ) -> tuple[ModelMessage, ...]:
        """Return user input that destructive history reduction must retain.

        Native pending-message drain runs before downstream history capabilities,
        while its ``EnqueuedMessagesEvent`` is observed only after request streaming
        starts. ``delivered_input_ids`` closes that narrow ordering window without
        prematurely changing the durable input disposition.
        """
        delivered = frozenset(delivered_input_ids)
        return tuple(
            message
            for record in self.records
            if record.origin is InputOrigin.user
            and (
                record.disposition is InputDisposition.applied
                or (record.disposition is InputDisposition.enqueued and record.input_id in delivered)
            )
            for message in record.messages
        )

    def applied_user_messages(self) -> tuple[ModelMessage, ...]:
        return self.retained_user_messages()


def _message_identity(messages: Sequence[ModelMessage]) -> bytes:
    """Return semantic input identity without constructor-generated timestamps."""
    payload = ModelMessagesTypeAdapter.dump_python(list(messages), mode="json")

    def normalize(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items() if key != "timestamp"}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    return json.dumps(
        normalize(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


class LogicalInputClosedError(RuntimeError):
    """The logical input router closed before admission could be linearized."""


@dataclass(frozen=True, slots=True)
class EnqueueReceipt:
    logical_run_id: str
    input_id: str
    disposition: InputDisposition
    enqueue_id: str | None = None


@dataclass(slots=True)
class _NativeBinding:
    run: AgentRun[Any, Any]
    loop: asyncio.AbstractEventLoop
    native_attempt_id: str
    pending_messages: list[PendingMessage]


class LogicalRunInputRouter:
    """Route one logical run's accepted input across native run attempts."""

    def __init__(
        self,
        ledger: RunInputLedger,
        *,
        max_pending_count: int = 128,
        max_pending_bytes: int = 2 * 1024 * 1024,
    ) -> None:
        if max_pending_count <= 0 or max_pending_bytes <= 0:
            raise ValueError("Logical input limits must be positive")
        self.ledger = ledger
        self.max_pending_count = max_pending_count
        self.max_pending_bytes = max_pending_bytes
        self._lock = threading.RLock()
        self._binding: _NativeBinding | None = None
        self._closed = False
        self._inflight = 0

    @property
    def logical_run_id(self) -> str:
        return self.ledger.logical_run_id

    @property
    def accepting(self) -> bool:
        with self._lock:
            return not self._closed

    @property
    def current_native_attempt_id(self) -> str | None:
        """Return the identity shared by the active native run binding."""
        with self._lock:
            return self._binding.native_attempt_id if self._binding is not None else None

    def request_delivered_input_ids(
        self,
        pending_messages: list[PendingMessage] | None,
    ) -> frozenset[str]:
        """Identify current-attempt input already drained into this native request.

        Pydantic AI removes a pending message before downstream history capabilities
        run, but emits its application event only when request streaming begins. The
        queue identity check excludes nested agent runs that share the same deps and
        logical router, such as the compact summary run.
        """
        with self._lock:
            binding = self._binding
            if binding is None or pending_messages is None or pending_messages is not binding.pending_messages:
                return frozenset()
            still_pending = {item.enqueue_id for item in pending_messages}
            return frozenset(
                record.input_id
                for record in self.ledger.unresolved()
                for attempt in record.enqueue_attempts
                if attempt.native_attempt_id == binding.native_attempt_id and attempt.enqueue_id not in still_pending
            )

    async def enqueue(
        self,
        *content: EnqueueContent,
        priority: PendingMessagePriority = "asap",
        origin: InputOrigin = InputOrigin.user,
        input_id: str | None = None,
    ) -> EnqueueReceipt:
        pending = PendingMessage.from_content(*content, priority=priority)
        if pending is None:
            raise ValueError("Logical input cannot be empty")
        with self._lock:
            existing = self.ledger.find(input_id) if input_id is not None else None
            if existing is not None:
                record = self.ledger.accept(
                    pending.messages,
                    origin=origin,
                    priority=priority,
                    input_id=input_id,
                )
                return self._receipt(record)
            if self._closed:
                raise LogicalInputClosedError("Logical run input is closed")
            self._check_capacity(pending.messages)
            record = self.ledger.accept(
                pending.messages,
                origin=origin,
                priority=priority,
                input_id=input_id,
            )
            binding = self._binding

        if binding is not None:
            await self._enqueue_record(record, binding)
        return self._receipt(record)

    async def bind(
        self,
        run: AgentRun[Any, Any],
        *,
        native_attempt_id: str,
    ) -> None:
        binding = _NativeBinding(
            run=run,
            loop=asyncio.get_running_loop(),
            native_attempt_id=native_attempt_id,
            pending_messages=run.ctx.state.pending_messages,
        )
        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot bind a closed logical input router")
            self._binding = binding
            unresolved = self.ledger.unresolved()
        for record in unresolved:
            if not any(item.native_attempt_id == native_attempt_id for item in record.enqueue_attempts):
                await self._enqueue_record(record, binding)

    def unbind(self, *, native_attempt_id: str) -> None:
        with self._lock:
            if self._binding is not None and self._binding.native_attempt_id == native_attempt_id:
                self._binding = None

    def observe_event(self, event: object) -> str | None:
        if not isinstance(event, EnqueuedMessagesEvent):
            return None
        with self._lock:
            return self.ledger.mark_applied_by_enqueue_id(event.enqueue_id)

    def close(
        self,
        *,
        reason: str = "logical run closed",
        reject_unresolved: bool = True,
    ) -> None:
        """Seal ingress and optionally reject input that cannot continue.

        Deferred suspension seals the attempt without rejecting unresolved
        records; the next continuation creates a router over the same ledger.
        """
        with self._lock:
            if self._inflight:
                raise RuntimeError("Cannot close logical input while native enqueue is in flight")
            self._closed = True
            self._binding = None
            if reject_unresolved:
                for record in self.ledger.unresolved():
                    self.ledger.reject(record.input_id, reason)

    def _receipt(self, record: RunInputRecord) -> EnqueueReceipt:
        return EnqueueReceipt(
            logical_run_id=self.logical_run_id,
            input_id=record.input_id,
            disposition=record.disposition,
            enqueue_id=record.latest_enqueue_id,
        )

    async def _enqueue_record(
        self,
        record: RunInputRecord,
        binding: _NativeBinding,
    ) -> str:
        with self._lock:
            self._inflight += 1
        try:
            enqueue_id = await _call_on_loop(
                binding.loop,
                binding.run.enqueue,
                *record.messages,
                priority=record.priority,
            )
            if enqueue_id is None:
                raise RuntimeError("Native enqueue unexpectedly returned no identity")
            with self._lock:
                self.ledger.mark_enqueued(
                    record.input_id,
                    native_attempt_id=binding.native_attempt_id,
                    enqueue_id=enqueue_id,
                )
            return enqueue_id
        finally:
            with self._lock:
                self._inflight -= 1

    def _check_capacity(self, messages: Sequence[ModelMessage]) -> None:
        unresolved = self.ledger.unresolved()
        if len(unresolved) >= self.max_pending_count:
            raise OverflowError("Logical input count limit exceeded")
        current_bytes = sum(len(ModelMessagesTypeAdapter.dump_json(record.messages)) for record in unresolved)
        incoming_bytes = len(ModelMessagesTypeAdapter.dump_json(list(messages)))
        if current_bytes + incoming_bytes > self.max_pending_bytes:
            raise OverflowError("Logical input byte limit exceeded")


def retained_user_messages_for_request(
    ledger: RunInputLedger,
    router: LogicalRunInputRouter | None,
    pending_messages: list[PendingMessage] | None,
) -> tuple[ModelMessage, ...]:
    """Return applied and current-request-delivered user input for reduction."""
    delivered_input_ids = (
        router.request_delivered_input_ids(pending_messages)
        if isinstance(router, LogicalRunInputRouter)
        else frozenset()
    )
    return ledger.retained_user_messages(delivered_input_ids=delivered_input_ids)


@dataclass(frozen=True, slots=True)
class ActiveRunRegistration:
    logical_run_id: str
    generation: str


class ActiveRunRegistry:
    """Root service for locating currently accepting logical-run routers."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._routers: dict[str, tuple[str, LogicalRunInputRouter]] = {}

    def register(self, router: LogicalRunInputRouter) -> ActiveRunRegistration:
        generation = str(uuid4())
        with self._lock:
            if router.logical_run_id in self._routers:
                raise ValueError(f"Logical run {router.logical_run_id!r} is already registered")
            self._routers[router.logical_run_id] = (generation, router)
        return ActiveRunRegistration(router.logical_run_id, generation)

    def get(self, logical_run_id: str) -> LogicalRunInputRouter | None:
        with self._lock:
            registered = self._routers.get(logical_run_id)
            return registered[1] if registered is not None else None

    def unregister(self, registration: ActiveRunRegistration) -> bool:
        with self._lock:
            registered = self._routers.get(registration.logical_run_id)
            if registered is None or registered[0] != registration.generation:
                return False
            del self._routers[registration.logical_run_id]
            return True


async def _call_on_loop(
    loop: asyncio.AbstractEventLoop,
    func,
    *args: Any,
    **kwargs: Any,
) -> Any:
    current_loop = asyncio.get_running_loop()
    if current_loop is loop:
        return func(*args, **kwargs)

    future: asyncio.Future[Any] = current_loop.create_future()

    def invoke() -> None:
        try:
            result = func(*args, **kwargs)
        except BaseException as exc:
            current_loop.call_soon_threadsafe(future.set_exception, exc)
        else:
            current_loop.call_soon_threadsafe(future.set_result, result)

    loop.call_soon_threadsafe(invoke)
    return await future
