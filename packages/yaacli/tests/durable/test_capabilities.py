from pathlib import Path
from unittest.mock import MagicMock

from ya_agent_sdk.inputs import InputDisposition
from yaacli.durable.application import build_runtime_descriptor
from yaacli.durable.bindings import runtime_bindings
from yaacli.durable.capabilities import DurableInboxPumpCapability
from yaacli.durable.models import (
    InputPriority,
    InputState,
    LogicalRunStatus,
    MainRuntimeManifest,
    StartRunRequest,
)
from yaacli.durable.sqlite import SQLiteSessionStore
from yaacli.session import TUIContext


def test_durable_inbox_enqueues_each_input_once_per_native_run(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "product.sqlite3")
    descriptor = build_runtime_descriptor(
        agent_spec={"name": "yaacli_main_v2", "model": "test"},
        main_plan_manifest=MainRuntimeManifest(),
    )
    session = store.create_session(str(tmp_path), session_id="session-test")
    run = store.start_run(
        StartRunRequest(
            session_id=session.session_id,
            idempotency_key="turn-test",
            descriptor=descriptor,
            initial_content=["initial"],
            plan_fingerprint=descriptor.plan_fingerprint,
            executable_version=descriptor.executable_version,
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
    ctx.enqueue.side_effect = ["enqueue-1", "enqueue-2"]
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

        ctx.run_id = "native-run-2"
        capability._enqueue_pending(ctx, (persisted,))
        persisted = store.list_inputs(run.logical_run_id)[1]
        assert ctx.enqueue.call_count == 2
        assert persisted.native_enqueue_id == "enqueue-2"
        assert len(ledger_record.enqueue_attempts) == 2

        capability._mark_applied(deps, "enqueue-1")
        assert store.list_inputs(run.logical_run_id)[1].state is InputState.applied
    finally:
        runtime_bindings.unregister(binding_ref, deps)
        store.close()
