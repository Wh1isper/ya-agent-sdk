from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import pytest
from pydantic_ai import DeferredToolRequests, Tool
from pydantic_ai.capabilities import Toolset as NativeToolsetCapability
from pydantic_ai.messages import ModelMessage, ModelRequest, RetryPromptPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset
from ya_agent_sdk.agents.main import create_agent
from ya_agent_sdk.execution import AgentSegment
from yaacli.durable.application import SessionApplicationService
from yaacli.durable.bindings import runtime_bindings
from yaacli.durable.capabilities import DurableInboxPumpCapability
from yaacli.durable.executor import LocalExecutionWorker, LocalRuntimeSpec, _merge_recoverable_history
from yaacli.durable.models import InputState, LogicalRunStatus
from yaacli.durable.sqlite import SQLiteSessionStore
from yaacli.environment import TUIEnvironment
from yaacli.session import TUIContext


def _runtime_spec(
    tmp_path: Path,
    model: Any,
    *,
    runtime_id: str = "test",
    capabilities: Sequence[Any] = (),
    output_type: Any = str,
    request_limit: int = 1000,
    hitl_policy: Literal["wait", "deny"] = "deny",
) -> LocalRuntimeSpec:
    def build(binding_ref: str):
        return create_agent(
            model,
            capabilities=[*capabilities, DurableInboxPumpCapability()],
            output_type=output_type,
            context_type=TUIContext,
            context_kwargs={"durable_binding_ref": binding_ref},
            env=TUIEnvironment,
            env_kwargs={"allowed_paths": [tmp_path], "default_path": tmp_path},
            agent_name="yaacli_main_v2",
        )

    return LocalRuntimeSpec(
        runtime_id=runtime_id,
        build=build,
        request_limit=request_limit,
        hitl_policy=hitl_policy,
    )


async def _create_worker(
    store: SQLiteSessionStore,
    tmp_path: Path,
    *specs: LocalRuntimeSpec,
    active_runtime_id: str | None = None,
) -> LocalExecutionWorker:
    if not specs:
        raise ValueError("At least one runtime spec is required")
    return await LocalExecutionWorker.create(
        store=store,
        state_path=tmp_path / "coordinator.state",
        active_runtime_id=active_runtime_id or specs[0].runtime_id,
        runtime_specs=specs,
    )


def test_recoverable_history_merges_partial_stable_overlap_without_duplication() -> None:
    first = ModelRequest(parts=[UserPromptPart(content="first")])
    overlap = ModelRequest(parts=[UserPromptPart(content="overlap")])
    recovered = ModelRequest(parts=[UserPromptPart(content="recovered")])

    assert _merge_recoverable_history(
        [first, overlap],
        [overlap, recovered],
        current_prompt=None,
    ) == [first, overlap, recovered]


async def test_turn_runs_through_local_coordinator_and_commits_revision(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "product.sqlite3")
    worker = await _create_worker(
        store,
        tmp_path,
        _runtime_spec(tmp_path, TestModel(custom_output_text="local answer")),
    )
    try:
        service = SessionApplicationService(store, worker.coordinator)
        session = service.create_session(str(tmp_path), session_id="session-test")

        revision = await service.run_turn(
            session.session_id,
            ["hello"],
            model="test:model",
            model_profile_id="test-profile",
            idempotency_key="turn-test",
        )

        assert revision.terminal == {"output": "local answer", "status": "completed"}
        run = store.get_run(revision.logical_run_id)
        assert run is not None
        assert run.status is LogicalRunStatus.completed
        assert run.model == "test:model"
        assert run.model_profile_id == "test-profile"
        assert store.list_inputs(run.logical_run_id)[0].state is InputState.applied
        assert store.get_execution_checkpoint(run.execution_id) is None
        assert store.read_events(session.session_id)[-1].event_type == "RUN_FINISHED"
        persisted = store.get_session(session.session_id)
        assert persisted is not None
        assert persisted.head_revision_id == revision.revision_id
    finally:
        await worker.close()
        store.close()


async def test_terminal_fallback_prefers_live_tool_projection_over_stable_checkpoint(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "product.sqlite3")
    live_projection = [
        {"type": "RUN_STARTED", "runId": "run-live"},
        {
            "type": "TOOL_CALL_CHUNK",
            "toolCallId": "visible-tool",
            "toolCallName": "shell_exec",
            "delta": '{"command":"pytest"}',
        },
        {
            "type": "TOOL_CALL_RESULT",
            "toolCallId": "visible-tool",
            "toolCallName": "shell_exec",
            "content": "tests running",
        },
    ]
    worker = await LocalExecutionWorker.create(
        store=store,
        state_path=tmp_path / "coordinator.state",
        active_runtime_id="test",
        runtime_specs=[_runtime_spec(tmp_path, TestModel())],
        display_projection_provider=lambda: live_projection,
    )
    try:
        service = SessionApplicationService(store, worker.coordinator)
        session = service.create_session(str(tmp_path), session_id="tool-projection-session")
        run = service.accept_turn(session.session_id, ["run tools"], idempotency_key="tool-turn")

        payload = worker.coordinator._stable_payload(run)

        assert any(event.get("toolCallId") == "visible-tool" for event in payload.display_projection)
    finally:
        await worker.close()
        store.close()


async def test_cancel_acceptance_freezes_display_projection_before_worker_settlement(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "product.sqlite3")
    live_projection = [
        {"type": "RUN_STARTED", "runId": "run-live"},
        {
            "type": "TOOL_CALL_CHUNK",
            "toolCallId": "visible-tool",
            "toolCallName": "shell_exec",
            "delta": '{"command":"pytest"}',
        },
    ]
    worker = await LocalExecutionWorker.create(
        store=store,
        state_path=tmp_path / "coordinator.state",
        active_runtime_id="test",
        runtime_specs=[_runtime_spec(tmp_path, TestModel())],
        display_projection_provider=lambda: live_projection,
    )
    try:
        service = SessionApplicationService(store, worker.coordinator)
        session = service.create_session(str(tmp_path), session_id="cancel-projection-session")
        run = service.accept_turn(session.session_id, ["run tools"], idempotency_key="tool-turn")

        worker.coordinator.accept_cancel(run.logical_run_id, "user_interrupted")
        live_projection.clear()
        payload = worker.coordinator._stable_payload(run)

        assert any(event.get("toolCallId") == "visible-tool" for event in payload.display_projection)
        cancelling = store.get_run(run.logical_run_id)
        assert cancelling is not None
        await worker.coordinator._commit_cancelled(cancelling)
        revision = store.get_revision_for_run(run.logical_run_id)
        assert revision is not None
        assert any(event.get("toolCallId") == "visible-tool" for event in revision.display_projection)
        assert run.logical_run_id not in worker.coordinator._terminal_display_projection_snapshots
    finally:
        await worker.close()
        store.close()


async def test_cancel_without_display_provider_preserves_stable_projection(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "product.sqlite3")
    worker = await _create_worker(store, tmp_path, _runtime_spec(tmp_path, TestModel()))
    try:
        service = SessionApplicationService(store, worker.coordinator)
        session = service.create_session(str(tmp_path), session_id="cancel-without-provider")
        run = service.accept_turn(session.session_id, ["run tools"], idempotency_key="tool-turn")
        stable_projection = [
            {"type": "RUN_STARTED", "runId": "stable-run"},
            {
                "type": "TOOL_CALL_START",
                "toolCallId": "stable-tool",
                "toolCallName": "shell_exec",
            },
        ]

        worker.coordinator.accept_cancel(run.logical_run_id, "user_interrupted")

        assert worker.coordinator._stable_terminal_display_projection(run, stable_projection) == stable_projection
        assert run.logical_run_id not in worker.coordinator._terminal_display_projection_snapshots
    finally:
        await worker.close()
        store.close()


async def test_active_feature_input_is_applied_once_across_tool_nodes(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "product.sqlite3")
    marker = "unique-background-shell-completion"
    first_model_started = asyncio.Event()
    release_first_model = asyncio.Event()
    final_messages: list[ModelMessage] = []
    model_calls = 0
    toolset = FunctionToolset[TUIContext](id="steps")

    async def step(value: str) -> str:
        return value

    toolset.add_tool(Tool(step))

    async def stream_response(
        messages: list[ModelMessage],
        _info: Any,
    ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            first_model_started.set()
            await release_first_model.wait()
        if model_calls <= 2:
            yield {
                0: DeltaToolCall(
                    name="step",
                    json_args=f'{{"value":"step-{model_calls}"}}',
                    tool_call_id=f"step-{model_calls}",
                )
            }
            return
        final_messages.extend(messages)
        yield "completed after feature input"

    worker = await _create_worker(
        store,
        tmp_path,
        _runtime_spec(
            tmp_path,
            FunctionModel(stream_function=stream_response),
            capabilities=[NativeToolsetCapability(toolset, id="steps")],
        ),
    )
    try:
        service = SessionApplicationService(store, worker.coordinator)
        session = service.create_session(str(tmp_path), session_id="feature-session")
        run = await service.start_turn(
            session.session_id,
            ["start tool sequence"],
            idempotency_key="feature-turn",
        )
        await asyncio.wait_for(first_model_started.wait(), timeout=5)
        accepted = await service.submit_input(
            run.logical_run_id,
            [marker],
            idempotency_key="shell-completion",
            origin="feature",
        )
        release_first_model.set()

        await asyncio.wait_for(service.wait(run.logical_run_id), timeout=5)
        persisted = store.list_inputs(run.logical_run_id)

        assert model_calls == 3
        assert str(final_messages).count(marker) == 1
        assert next(item for item in persisted if item.input_id == accepted.input_id).state is InputState.applied
    finally:
        release_first_model.set()
        await worker.close()
        store.close()


async def test_cancel_before_start_commits_terminal_revision(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "product.sqlite3")
    worker = await _create_worker(store, tmp_path, _runtime_spec(tmp_path, TestModel()))
    try:
        service = SessionApplicationService(store, worker.coordinator)
        session = service.create_session(str(tmp_path), session_id="cancel-before-start")
        run = service.accept_turn(
            session.session_id,
            ["cancel me"],
            idempotency_key="cancelled-turn",
        )
        service.accept_cancel(run.logical_run_id, reason="cancelled before start")
        service.start(run.logical_run_id)

        cancelled = await service.wait(run.logical_run_id)
        revision = store.get_revision_for_run(run.logical_run_id)
        assert cancelled.status is LogicalRunStatus.cancelled
        assert revision is not None
        assert revision.terminal == {
            "reason": "cancelled before start",
            "status": "cancelled",
        }
    finally:
        await worker.close()
        store.close()


async def test_worker_uses_runtime_active_when_each_run_starts(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "product.sqlite3")
    worker = await _create_worker(
        store,
        tmp_path,
        _runtime_spec(
            tmp_path,
            TestModel(custom_output_text="first runtime"),
            runtime_id="first",
        ),
        _runtime_spec(
            tmp_path,
            TestModel(custom_output_text="second runtime"),
            runtime_id="second",
        ),
        active_runtime_id="first",
    )
    try:
        service = SessionApplicationService(store, worker.coordinator)
        session = service.create_session(str(tmp_path), session_id="runtime-selection")

        first = await service.run_turn(session.session_id, ["first turn"])
        await worker.activate("second")
        second = await service.run_turn(session.session_id, ["second turn"])

        assert first.terminal["output"] == "first runtime"
        assert second.terminal["output"] == "second runtime"
        assert worker.runtime_id == "second"
    finally:
        await worker.close()
        store.close()


async def test_worker_lazily_enters_profiles_and_closes_only_cached_runtimes(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "product.sqlite3")
    builds: list[str] = []
    enters: list[str] = []
    exits: list[str] = []

    class FakeRuntime:
        def __init__(self, runtime_id: str) -> None:
            self.runtime_id = runtime_id
            self.ctx = SimpleNamespace(runtime_descriptor_id="initial")

        async def __aenter__(self) -> FakeRuntime:
            enters.append(self.runtime_id)
            return self

        async def __aexit__(self, *_args: Any) -> None:
            exits.append(self.runtime_id)

    def spec(runtime_id: str) -> LocalRuntimeSpec:
        def build(_binding_ref: str) -> Any:
            builds.append(runtime_id)
            return FakeRuntime(runtime_id)

        return LocalRuntimeSpec(
            runtime_id=runtime_id,
            build=build,
            request_limit=10,
            hitl_policy="deny",
        )

    worker = await _create_worker(
        store,
        tmp_path,
        spec("first"),
        spec("second"),
        spec("unused"),
        active_runtime_id="first",
    )
    try:
        assert builds == ["first"]
        assert enters == ["first"]

        first_activation = await worker.activate("second")
        second_activation = await worker.activate("second")

        assert first_activation is second_activation
        assert builds == ["first", "second"]
        assert enters == ["first", "second"]
        assert worker.runtime_id == "second"
    finally:
        await worker.close()
        store.close()

    assert exits == ["second", "first"]
    assert "unused" not in builds


async def test_runtime_activation_cleans_up_when_binding_registration_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteSessionStore(tmp_path / "product.sqlite3")
    exits: list[str] = []

    class FakeRuntime:
        def __init__(self, runtime_id: str) -> None:
            self.runtime_id = runtime_id
            self.ctx = SimpleNamespace(runtime_descriptor_id="initial")

        async def __aenter__(self) -> FakeRuntime:
            return self

        async def __aexit__(self, *_args: Any) -> None:
            exits.append(self.runtime_id)

    def spec(runtime_id: str) -> LocalRuntimeSpec:
        return LocalRuntimeSpec(
            runtime_id=runtime_id,
            build=lambda _binding_ref: FakeRuntime(runtime_id),
            request_limit=10,
            hitl_policy="deny",
        )

    worker = await _create_worker(store, tmp_path, spec("first"), spec("second"))
    original_register = runtime_bindings.register

    def register(binding_ref: str, context: object, bound_store: SQLiteSessionStore) -> None:
        if binding_ref.endswith(":local:second"):
            raise ValueError("injected binding conflict")
        original_register(binding_ref, context, bound_store)

    monkeypatch.setattr(runtime_bindings, "register", register)
    try:
        with pytest.raises(ValueError, match="injected binding conflict"):
            await worker.activate("second")
        assert exits == ["second"]
        assert worker.runtime_id == "first"
    finally:
        await worker.close()
        store.close()

    assert exits == ["second", "first"]


async def test_worker_close_fences_new_runtime_activation(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "product.sqlite3")
    close_started = asyncio.Event()
    release_close = asyncio.Event()
    builds: list[str] = []

    class FakeRuntime:
        def __init__(self, runtime_id: str) -> None:
            self.runtime_id = runtime_id
            self.ctx = SimpleNamespace(runtime_descriptor_id="initial")

        async def __aenter__(self) -> FakeRuntime:
            return self

        async def __aexit__(self, *_args: Any) -> None:
            if self.runtime_id == "first":
                close_started.set()
                await release_close.wait()

    def spec(runtime_id: str) -> LocalRuntimeSpec:
        def build(_binding_ref: str) -> FakeRuntime:
            builds.append(runtime_id)
            return FakeRuntime(runtime_id)

        return LocalRuntimeSpec(
            runtime_id=runtime_id,
            build=build,
            request_limit=10,
            hitl_policy="deny",
        )

    worker = await _create_worker(store, tmp_path, spec("first"), spec("second"))
    close_task = asyncio.create_task(worker.close())
    try:
        await asyncio.wait_for(close_started.wait(), timeout=1)
        with pytest.raises(RuntimeError, match="worker is closed"):
            await worker.activate("second")
    finally:
        release_close.set()
        await close_task
        store.close()

    assert builds == ["first"]


async def test_cumulative_request_limit_spans_deferred_segments(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "product.sqlite3")
    effects: list[str] = []

    async def guarded_effect(value: str) -> str:
        effects.append(value)
        return value

    approval_toolset = FunctionToolset[TUIContext](
        [Tool(guarded_effect, requires_approval=True)],
        id="approval",
    )
    worker = await _create_worker(
        store,
        tmp_path,
        _runtime_spec(
            tmp_path,
            TestModel(call_tools=["guarded_effect"], custom_output_text="unreachable"),
            capabilities=[NativeToolsetCapability(approval_toolset, id="approval")],
            output_type=[str, DeferredToolRequests],
            request_limit=1,
        ),
    )
    try:
        service = SessionApplicationService(store, worker.coordinator)
        session = service.create_session(str(tmp_path), session_id="session-budget")
        run = await service.start_turn(
            session.session_id,
            ["request the guarded effect"],
            idempotency_key="budget-turn",
        )

        with pytest.raises(RuntimeError, match="cumulative model request limit of 1"):
            await service.wait(run.logical_run_id)

        persisted_run = store.get_run(run.logical_run_id)
        revision = store.get_revision_for_run(run.logical_run_id)
        assert persisted_run is not None
        assert persisted_run.status is LogicalRunStatus.failed
        assert revision is not None
        assert revision.terminal == {
            "status": "failed",
            "error_type": "RuntimeError",
            "error": "Execution exhausted the cumulative model request limit of 1.",
        }
        assert revision.usage["requests"] == 1
        assert effects == []
    finally:
        await worker.close()
        store.close()


async def test_suspended_run_does_not_block_another_session(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "product.sqlite3")
    approval_toolset = FunctionToolset[TUIContext](id="approval")

    async def guarded_effect(value: str) -> str:
        return value

    approval_toolset.add_tool(Tool(guarded_effect, requires_approval=True))

    async def stream_response(
        messages: list[ModelMessage],
        _info: Any,
    ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        has_tool_result = any(
            isinstance(message, ModelRequest)
            and any(isinstance(part, ToolReturnPart | RetryPromptPart) for part in message.parts)
            for message in messages
        )
        if "suspend this run" in str(messages) and not has_tool_result:
            yield {
                0: DeltaToolCall(
                    name="guarded_effect",
                    json_args='{"value":"one"}',
                    tool_call_id="suspended-call",
                )
            }
        else:
            yield "completed independently"

    worker = await _create_worker(
        store,
        tmp_path,
        _runtime_spec(
            tmp_path,
            FunctionModel(stream_function=stream_response),
            capabilities=[NativeToolsetCapability(approval_toolset, id="approval")],
            output_type=[str, DeferredToolRequests],
            hitl_policy="wait",
        ),
    )
    try:
        service = SessionApplicationService(store, worker.coordinator)
        first_session = service.create_session(str(tmp_path), session_id="suspended-session")
        second_session = service.create_session(str(tmp_path), session_id="independent-session")
        first_run = await service.start_turn(first_session.session_id, ["suspend this run"])
        for _ in range(100):
            persisted = store.get_run(first_run.logical_run_id)
            if persisted is not None and persisted.status is LogicalRunStatus.suspended:
                break
            await asyncio.sleep(0.02)
        else:
            pytest.fail("first run did not suspend")

        second_revision = await asyncio.wait_for(
            service.run_turn(second_session.session_id, ["complete independently"]),
            timeout=5,
        )
        assert second_revision.terminal["output"] == "completed independently"

        persisted = store.get_run(first_run.logical_run_id)
        assert persisted is not None
        assert persisted.pending_action_batch_id is not None
        batch = store.get_action_batch(persisted.pending_action_batch_id)
        assert batch is not None
        await service.decide_action(
            batch.items[0].action_item_id,
            {"approved": False, "message": "test completed"},
        )
        completed = await service.wait(first_run.logical_run_id)
        assert completed.status is LogicalRunStatus.completed
        revision = store.get_revision_for_run(first_run.logical_run_id)
        assert revision is not None
        assert revision.usage["requests"] == 2
    finally:
        await worker.close()
        store.close()


async def test_worker_shutdown_interrupts_active_process_owned_run(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "product.sqlite3")
    model_started = asyncio.Event()
    release_model = asyncio.Event()

    async def stream_response(
        _messages: list[ModelMessage],
        _info: Any,
    ) -> AsyncIterator[str]:
        model_started.set()
        await release_model.wait()
        yield "late output"

    worker = await _create_worker(
        store,
        tmp_path,
        _runtime_spec(tmp_path, FunctionModel(stream_function=stream_response)),
    )
    service = SessionApplicationService(store, worker.coordinator)
    session = service.create_session(str(tmp_path), session_id="shutdown-session")
    run = await service.start_turn(session.session_id, ["wait"])
    await asyncio.wait_for(model_started.wait(), timeout=5)

    try:
        await worker.close()

        interrupted = store.get_run(run.logical_run_id)
        revision = store.get_revision_for_run(run.logical_run_id)
        assert interrupted is not None
        assert interrupted.status is LogicalRunStatus.interrupted
        assert revision is not None
        assert revision.terminal["status"] == "interrupted"
        assert "worker shutdown" in str(revision.terminal["reason"])
        assert str(revision.message_history).count("wait") == 1
    finally:
        release_model.set()
        await worker.close()
        store.close()


async def test_controlled_interruption_of_continuation_segment_keeps_safe_partial_history(
    tmp_path: Path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "product.sqlite3")
    continuation_started = asyncio.Event()
    release_continuation = asyncio.Event()
    model_calls = 0
    approval_toolset = FunctionToolset[TUIContext](id="approval")

    async def guarded_effect(value: str) -> str:
        return value

    approval_toolset.add_tool(Tool(guarded_effect, requires_approval=True))

    async def stream_response(
        _messages: list[ModelMessage],
        _info: Any,
    ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            yield {
                0: DeltaToolCall(
                    name="guarded_effect",
                    json_args='{"value":"approved"}',
                    tool_call_id="approval-call",
                )
            }
            return
        yield "continuation partial output"
        continuation_started.set()
        await release_continuation.wait()

    worker = await _create_worker(
        store,
        tmp_path,
        _runtime_spec(
            tmp_path,
            FunctionModel(stream_function=stream_response),
            capabilities=[NativeToolsetCapability(approval_toolset, id="approval")],
            output_type=[str, DeferredToolRequests],
            hitl_policy="wait",
        ),
    )
    try:
        service = SessionApplicationService(store, worker.coordinator)
        session = service.create_session(str(tmp_path), session_id="continuation-recovery")
        run = await service.start_turn(session.session_id, ["start approved continuation"])
        for _ in range(100):
            persisted = store.get_run(run.logical_run_id)
            if persisted is not None and persisted.status is LogicalRunStatus.suspended:
                break
            await asyncio.sleep(0.02)
        else:
            pytest.fail("run did not suspend")
        assert persisted is not None
        assert persisted.pending_action_batch_id is not None
        batch = store.get_action_batch(persisted.pending_action_batch_id)
        assert batch is not None
        await service.decide_action(batch.items[0].action_item_id, {"approved": True})
        await asyncio.wait_for(continuation_started.wait(), timeout=5)

        await service.cancel(run.logical_run_id, reason="stop continuation")

        revision = store.get_revision_for_run(run.logical_run_id)
        assert revision is not None
        assert revision.terminal == {"status": "cancelled", "reason": "stop continuation"}
        history = str(revision.message_history)
        assert model_calls == 2
        assert history.count("start approved continuation") == 1
        assert history.count("continuation partial output") == 1
    finally:
        release_continuation.set()
        await worker.close()
        store.close()


async def test_terminal_recovery_capture_failure_uses_last_stable_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteSessionStore(tmp_path / "product.sqlite3")
    second_started = asyncio.Event()
    release_failure = asyncio.Event()
    model_calls = 0

    async def stream_response(
        _messages: list[ModelMessage],
        _info: Any,
    ) -> AsyncIterator[str]:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            yield "stable first answer"
            return
        yield "discarded partial answer"
        second_started.set()
        await release_failure.wait()
        raise RuntimeError("second turn failed")

    worker = await _create_worker(
        store,
        tmp_path,
        _runtime_spec(tmp_path, FunctionModel(stream_function=stream_response)),
    )
    try:
        service = SessionApplicationService(store, worker.coordinator)
        session = service.create_session(str(tmp_path), session_id="capture-fallback")
        stable = await service.run_turn(session.session_id, ["first input"])
        run = await service.start_turn(session.session_id, ["second input"])
        await asyncio.wait_for(second_started.wait(), timeout=5)

        def fail_recovery(_segment: AgentSegment[Any, Any, Any]) -> list[ModelMessage]:
            raise RuntimeError("capture failed")

        monkeypatch.setattr(AgentSegment, "recoverable_messages", fail_recovery)
        release_failure.set()
        with pytest.raises(RuntimeError, match="second turn failed"):
            await service.wait(run.logical_run_id)

        revision = store.get_revision_for_run(run.logical_run_id)
        assert revision is not None
        assert revision.terminal["status"] == "failed"
        assert revision.message_history == stable.message_history
        assert "second input" not in str(revision.message_history)
        assert "discarded partial answer" not in str(revision.message_history)
    finally:
        release_failure.set()
        await worker.close()
        store.close()
