from pathlib import Path
from unittest.mock import MagicMock

from ya_agent_sdk.inputs import InputDisposition, InputOrigin, LogicalRunInputRouter
from yaacli.durable.bindings import runtime_bindings
from yaacli.durable.capabilities import DurableInboxPumpCapability, _model_content
from yaacli.durable.models import (
    InputPriority,
    InputState,
    LogicalRunStatus,
    StartRunRequest,
)
from yaacli.durable.sqlite import SQLiteSessionStore
from yaacli.session import TUIContext


def test_active_run_user_input_uses_legacy_steering_envelope(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "product.sqlite3")
    session = store.create_session(str(tmp_path), session_id="session-test")
    run = store.start_run(
        StartRunRequest(
            session_id=session.session_id,
            idempotency_key="turn-test",
            initial_content=["initial"],
        )
    )
    initial = store.list_inputs(run.logical_run_id)[0]
    steering = store.accept_input(
        run.logical_run_id,
        ["change direction"],
        idempotency_key="steer-test",
        priority=InputPriority.asap,
        origin="user",
    )
    feature = store.accept_input(
        run.logical_run_id,
        ["background result"],
        idempotency_key="feature-test",
        priority=InputPriority.asap,
        origin="feature",
    )

    try:
        assert _model_content(initial) == ["initial"]
        assert _model_content(feature) == ["background result"]
        assert _model_content(steering) == [
            '<bus-message source="user">\n'
            "<steering>\n"
            "change direction\n"
            "</steering>\n\n"
            "<system-reminder>\n"
            "The user has provided additional guidance during task execution.\n"
            "Review the <steering> content carefully, consider how it affects your current approach,\n"
            "and adjust your work accordingly while continuing toward the goal.\n"
            "</system-reminder>\n"
            "</bus-message>"
        ]
        persisted = next(item for item in store.list_inputs(run.logical_run_id) if item.input_id == steering.input_id)
        assert persisted.content == ["change direction"]
    finally:
        store.close()


async def test_durable_inbox_reuses_router_enqueue_once_per_native_attempt(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "product.sqlite3")
    session = store.create_session(str(tmp_path), session_id="session-test")
    run = store.start_run(
        StartRunRequest(
            session_id=session.session_id,
            idempotency_key="turn-test",
            initial_content=["initial"],
        )
    )
    store.set_run_status(run.logical_run_id, LogicalRunStatus.running)
    initial = store.list_inputs(run.logical_run_id)[0]
    store.transition_input(initial.input_id, InputState.accepted, InputState.applied)
    accepted = store.accept_input(
        run.logical_run_id,
        ["background shell completed"],
        idempotency_key="shell-completion",
        priority=InputPriority.asap,
        origin="feature",
    )

    binding_ref = "test-binding"
    deps = TUIContext(
        durable_binding_ref=binding_ref,
        durable_logical_run_id=run.logical_run_id,
    )
    deps.run_input_ledger.logical_run_id = run.logical_run_id
    ctx = MagicMock()
    ctx.deps = deps
    ctx.run_id = "native-run-1"
    ctx.enqueue.side_effect = ["enqueue-1", "unexpected-duplicate"]
    capability = DurableInboxPumpCapability()
    runtime_bindings.register(binding_ref, deps, store)
    try:
        capability._enqueue_pending(ctx, (accepted,))
        enqueued = store.list_inputs(
            run.logical_run_id,
            states=(InputState.enqueued,),
        )
        capability._enqueue_pending(ctx, enqueued)

        persisted = store.list_inputs(run.logical_run_id)[1]
        ledger_record = deps.run_input_ledger.get(accepted.input_id)
        assert ctx.enqueue.call_count == 1
        assert persisted.state is InputState.enqueued
        assert persisted.native_enqueue_id == "enqueue-1"
        assert ledger_record.disposition is InputDisposition.enqueued
        assert len(ledger_record.enqueue_attempts) == 1

        router = LogicalRunInputRouter(deps.run_input_ledger)
        deps.input_router = router
        native_run = MagicMock()
        native_run.enqueue.side_effect = ["enqueue-2", "enqueue-3"]
        await router.bind(native_run, native_attempt_id="native-attempt-2")

        ctx.run_id = "different-pydantic-run-id"
        capability._enqueue_pending(ctx, (persisted,))
        persisted = store.list_inputs(run.logical_run_id)[1]
        assert ctx.enqueue.call_count == 1
        native_run.enqueue.assert_called_once()
        assert persisted.native_enqueue_id == "enqueue-2"
        assert [attempt.native_attempt_id for attempt in ledger_record.enqueue_attempts] == [
            "native-run-1",
            "native-attempt-2",
        ]

        capability.reconcile_applied_enqueue(deps, "enqueue-1")
        assert store.list_inputs(run.logical_run_id)[1].state is InputState.applied

        accepted_before_hook = store.accept_input(
            run.logical_run_id,
            ["applied before capability hook"],
            idempotency_key="applied-before-hook",
            priority=InputPriority.asap,
            origin="feature",
        )
        receipt = await router.enqueue(
            "applied before capability hook",
            input_id=accepted_before_hook.input_id,
            origin=InputOrigin.feature,
        )
        assert receipt.enqueue_id == "enqueue-3"
        deps.run_input_ledger.mark_applied_by_enqueue_id("enqueue-3")
        capability._sync_applied_inputs(deps)
        assert store.list_inputs(run.logical_run_id)[2].state is InputState.applied
    finally:
        runtime_bindings.unregister(binding_ref, deps)
        store.close()
