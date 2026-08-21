from __future__ import annotations

from typing import Any

import pytest
from pydantic_ai.messages import EnqueuedMessagesEvent
from ya_agent_sdk.inputs import (
    ActiveRunRegistry,
    InputDisposition,
    InputOrigin,
    LogicalInputClosedError,
    LogicalRunInputRouter,
    RunInputLedger,
)


class FakeAgentRun:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], str]] = []

    def enqueue(self, *content: Any, priority: str = "asap") -> str:
        self.calls.append((content, priority))
        return f"enqueue-{len(self.calls)}"


def test_run_input_ledger_round_trips_structured_messages() -> None:
    ledger = RunInputLedger(logical_run_id="logical-1")
    record = ledger.accept(
        [],
        origin=InputOrigin.user,
        priority="when_idle",
    )
    record.disposition = InputDisposition.rejected
    record.rejection_reason = "test"

    restored = RunInputLedger.model_validate(ledger.model_dump(mode="json"))

    assert restored == ledger


async def test_router_buffers_before_bind_and_correlates_application() -> None:
    ledger = RunInputLedger(logical_run_id="logical-1")
    router = LogicalRunInputRouter(ledger)
    receipt = await router.enqueue("hello", priority="when_idle")
    run = FakeAgentRun()

    assert receipt.disposition is InputDisposition.accepted
    assert receipt.enqueue_id is None
    assert router.current_native_attempt_id is None

    await router.bind(run, native_attempt_id="attempt-1")  # type: ignore[arg-type]

    record = ledger.get(receipt.input_id)
    assert router.current_native_attempt_id == "attempt-1"
    assert record.disposition is InputDisposition.enqueued
    assert record.latest_enqueue_id == "enqueue-1"
    assert run.calls[0][1] == "when_idle"

    applied_input_id = router.observe_event(
        EnqueuedMessagesEvent(
            enqueue_id="enqueue-1",
            messages=tuple(record.messages),
        )
    )

    assert applied_input_id == receipt.input_id
    assert record.disposition is InputDisposition.applied


async def test_router_reuses_stable_input_identity_without_native_duplicate() -> None:
    ledger = RunInputLedger(logical_run_id="logical-1")
    router = LogicalRunInputRouter(ledger)
    run = FakeAgentRun()
    await router.bind(run, native_attempt_id="attempt-1")  # type: ignore[arg-type]

    first = await router.enqueue(
        "hello",
        input_id="sql-input-1",
        origin=InputOrigin.feature,
    )
    repeated = await router.enqueue(
        "hello",
        input_id="sql-input-1",
        origin=InputOrigin.feature,
    )

    assert first == repeated
    assert first.input_id == "sql-input-1"
    assert len(run.calls) == 1
    assert len(ledger.records) == 1


async def test_router_rejects_stable_input_identity_reuse_with_different_payload() -> None:
    router = LogicalRunInputRouter(RunInputLedger(logical_run_id="logical-1"))
    await router.enqueue("first", input_id="sql-input-1")

    with pytest.raises(ValueError, match="reused with different content"):
        await router.enqueue("second", input_id="sql-input-1")


async def test_router_closed_admission_is_typed_but_existing_identity_remains_idempotent() -> None:
    router = LogicalRunInputRouter(RunInputLedger(logical_run_id="logical-1"))
    first = await router.enqueue("first", input_id="sql-input-1")
    router.close(reason="finished")

    repeated = await router.enqueue("first", input_id="sql-input-1")
    assert repeated.input_id == first.input_id
    assert repeated.disposition is InputDisposition.rejected

    with pytest.raises(LogicalInputClosedError, match="input is closed"):
        await router.enqueue("second", input_id="sql-input-2")


async def test_router_rebinds_unapplied_input_once_per_native_attempt() -> None:
    ledger = RunInputLedger(logical_run_id="logical-1")
    router = LogicalRunInputRouter(ledger)
    receipt = await router.enqueue("hello")
    first = FakeAgentRun()
    second = FakeAgentRun()

    await router.bind(first, native_attempt_id="attempt-1")  # type: ignore[arg-type]
    await router.bind(first, native_attempt_id="attempt-1")  # type: ignore[arg-type]
    router.unbind(native_attempt_id="attempt-1")
    await router.bind(second, native_attempt_id="attempt-2")  # type: ignore[arg-type]

    record = ledger.get(receipt.input_id)
    assert len(first.calls) == 1
    assert len(second.calls) == 1
    assert [item.native_attempt_id for item in record.enqueue_attempts] == [
        "attempt-1",
        "attempt-2",
    ]
    router.unbind(native_attempt_id="attempt-2")
    assert router.current_native_attempt_id is None


async def test_router_enforces_one_pending_budget() -> None:
    router = LogicalRunInputRouter(
        RunInputLedger(logical_run_id="logical-1"),
        max_pending_count=1,
    )

    await router.enqueue("first")

    try:
        await router.enqueue("second")
    except OverflowError as exc:
        assert str(exc) == "Logical input count limit exceeded"
    else:
        raise AssertionError("Expected pending input admission to fail")


def test_registry_generation_cannot_unregister_newer_owner() -> None:
    registry = ActiveRunRegistry()
    router = LogicalRunInputRouter(RunInputLedger(logical_run_id="logical-1"))
    registration = registry.register(router)

    assert registry.get("logical-1") is router
    assert registry.unregister(registration)
    assert registry.get("logical-1") is None
    assert not registry.unregister(registration)


def test_router_close_rejects_unapplied_input() -> None:
    ledger = RunInputLedger(logical_run_id="logical-1")
    router = LogicalRunInputRouter(ledger)
    record = ledger.accept([], origin=InputOrigin.feature, priority="asap")

    router.close(reason="cancelled")

    assert record.disposition is InputDisposition.rejected
    assert record.rejection_reason == "cancelled"
