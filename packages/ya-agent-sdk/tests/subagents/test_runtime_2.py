from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import ClassVar

import pytest
from pydantic import TypeAdapter
from pydantic_ai import (
    Agent,
    AgentSpec,
    DeferredToolRequests,
    DeferredToolResults,
    Tool,
    ToolDenied,
)
from pydantic_ai._spec import CapabilitySpec
from pydantic_ai.agent.spec import load_capability_from_nested_spec
from pydantic_ai.capabilities import AbstractCapability, CombinedCapability, PrefixTools, WrapperCapability
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo as FunctionAgentInfo
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset
from ya_agent_sdk.capabilities import (
    SupportsDeferredOutput,
    ToolApprovalCapability,
    build_capability_catalog,
    build_default_capability_catalog,
)
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.events import SubagentCompleteEvent, SubagentStartEvent
from ya_agent_sdk.inputs import (
    EnqueueReceipt,
    InputDisposition,
    InputOrigin,
    LogicalRunInputRouter,
    RunInputLedger,
)
from ya_agent_sdk.subagents import (
    AgentTemplateContext,
    AsyncioSubagentExecutionHost,
    DelegationCapability,
    InMemorySubagentExecutionStore,
    InProcessSubagentDriver,
    ResolvedSubagentPlan,
    SelfForkPolicy,
    SubagentDeliveryState,
    SubagentDriverOutcome,
    SubagentExecutionMode,
    SubagentExecutionRecord,
    SubagentExecutionService,
    SubagentExecutionState,
    SubagentHistoryPolicy,
    SubagentInputState,
    SubagentPlanResolver,
    SubagentRegistry,
    SubagentSpec,
    resolve_subagent_output_type,
)


@dataclass
class AuditCapability(AbstractCapability[AgentContext]):
    text: str = "audit"

    def get_instructions(self) -> str:
        return self.text


@dataclass
class DeferredApprovalCapability(SupportsDeferredOutput, AbstractCapability[AgentContext]):
    id: str | None = "deferred-approval"

    effects: ClassVar[list[str]] = []

    def get_toolset(self) -> FunctionToolset[AgentContext]:
        async def guarded_effect(value: str) -> str:
            self.effects.append(value)
            return value

        return FunctionToolset(
            [Tool(guarded_effect, requires_approval=True)],
            id="approval",
        )


@dataclass(init=False)
class PositionalWrapperCapability(WrapperCapability[AgentContext]):
    @classmethod
    def get_serialization_name(cls) -> str:
        return "PositionalWrapperCapability"

    @classmethod
    def from_spec(cls, capability: CapabilitySpec) -> PositionalWrapperCapability:
        return cls(wrapped=load_capability_from_nested_spec(capability))


@dataclass(init=False)
class OptionalWrapperCapability(WrapperCapability[AgentContext]):
    @classmethod
    def get_serialization_name(cls) -> str:
        return "OptionalWrapperCapability"

    @classmethod
    def from_spec(cls, *, capability: CapabilitySpec | None = None) -> OptionalWrapperCapability:
        if capability is None:
            raise ValueError("capability is required")
        return cls(wrapped=load_capability_from_nested_spec(capability))


def test_resolver_normalizes_templates_and_has_stable_fingerprint() -> None:
    catalog = build_default_capability_catalog(explicit_types=[AuditCapability])
    resolver = SubagentPlanResolver(catalog, default_model="test")
    spec = SubagentSpec(
        route="researcher",
        agent=AgentSpec.from_dict({
            "description": "Research for {{template.project}}",
            "instructions": "Work on {{template.project}}",
            "capabilities": [{"AuditCapability": {"text": "{{template.policy}}"}}],
        }),
    )
    template = AgentTemplateContext(template={"project": "YA", "policy": "strict"})

    first = resolver.resolve(spec, template_context=template)
    second = resolver.resolve(spec, template_context=template)

    assert first.fingerprint == second.fingerprint
    assert first.descriptor_id == second.descriptor_id
    assert first.descriptor_id == f"researcher:{first.fingerprint}"
    assert len(first.fingerprint) == 64
    assert str(first.normalized_agent_spec.instructions) == "Work on YA"
    assert str(first.normalized_agent_spec.description) == "Research for YA"
    assert first.custom_capability_audit[0].serialization_name == "AuditCapability"
    assert first.custom_capability_audit[0].provenance.startswith("explicit:")


def test_resolver_derives_deferred_output_from_selected_capabilities() -> None:
    resolver = SubagentPlanResolver(build_default_capability_catalog(), default_model="test")
    ordinary = resolver.resolve(SubagentSpec(route="ordinary", agent=AgentSpec()))
    guarded = resolver.resolve(
        SubagentSpec(
            route="guarded",
            agent=AgentSpec.from_dict({"capabilities": ["ToolApprovalCapability"]}),
        )
    )
    host_resolver = SubagentPlanResolver(
        build_default_capability_catalog(),
        default_model="test",
        host_capabilities=(ToolApprovalCapability(),),
    )
    host_guarded = host_resolver.resolve(SubagentSpec(route="host-guarded", agent=AgentSpec()))
    combined_host_resolver = SubagentPlanResolver(
        build_default_capability_catalog(),
        default_model="test",
        host_capabilities=(CombinedCapability([ToolApprovalCapability()]),),
    )
    combined_host_guarded = combined_host_resolver.resolve(
        SubagentSpec(route="combined-host-guarded", agent=AgentSpec())
    )
    wrapped_host_resolver = SubagentPlanResolver(
        build_default_capability_catalog(),
        default_model="test",
        host_capabilities=(PrefixTools(wrapped=ToolApprovalCapability(), prefix="host"),),
    )
    wrapped_host_guarded = wrapped_host_resolver.resolve(SubagentSpec(route="wrapped-host-guarded", agent=AgentSpec()))
    nested = resolver.resolve(
        SubagentSpec(
            route="nested",
            agent=AgentSpec.from_dict({
                "capabilities": [
                    {
                        "PrefixTools": {
                            "prefix": "guarded",
                            "capability": "ToolApprovalCapability",
                        }
                    }
                ]
            }),
        )
    )

    assert ordinary.supports_deferred_output is False
    assert resolve_subagent_output_type(ordinary) is str
    for plan in (guarded, nested, host_guarded, combined_host_guarded, wrapped_host_guarded):
        assert plan.supports_deferred_output is True
        assert resolve_subagent_output_type(plan) == [str, DeferredToolRequests]
    assert resolver.restore(guarded.to_descriptor()).supports_deferred_output is True
    assert resolver.restore(nested.to_descriptor()).supports_deferred_output is True
    assert host_resolver.restore(host_guarded.to_descriptor()).supports_deferred_output is True
    assert combined_host_resolver.restore(combined_host_guarded.to_descriptor()).supports_deferred_output is True
    assert wrapped_host_resolver.restore(wrapped_host_guarded.to_descriptor()).supports_deferred_output is True


@pytest.mark.parametrize(
    ("wrapper_type", "serialized_capability"),
    [
        (
            PositionalWrapperCapability,
            {"PositionalWrapperCapability": "ToolApprovalCapability"},
        ),
        (
            OptionalWrapperCapability,
            {
                "OptionalWrapperCapability": {
                    "capability": "ToolApprovalCapability",
                }
            },
        ),
    ],
)
def test_resolver_traverses_typed_custom_wrapper_specs(
    wrapper_type: type[AbstractCapability[AgentContext]],
    serialized_capability: dict[str, object],
) -> None:
    resolver = SubagentPlanResolver(
        build_default_capability_catalog(explicit_types=[wrapper_type]),
        default_model="test",
    )
    plan = resolver.resolve(
        SubagentSpec(
            route="wrapped",
            agent=AgentSpec.from_dict({"capabilities": [serialized_capability]}),
        )
    )

    audit_names = {audit.serialization_name for audit in plan.custom_capability_audit}
    assert plan.supports_deferred_output is True
    assert wrapper_type.__name__ in audit_names
    assert "ToolApprovalCapability" in audit_names
    assert resolver.restore(plan.to_descriptor()).supports_deferred_output is True


def test_resolver_does_not_treat_arbitrary_capability_arguments_as_nested_specs() -> None:
    resolver = SubagentPlanResolver(build_default_capability_catalog(), default_model="test")
    plan = resolver.resolve(
        SubagentSpec(
            route="metadata",
            agent=AgentSpec.from_dict({
                "capabilities": [
                    {
                        "SetToolMetadata": {
                            "note": "ToolApprovalCapability",
                            "payload": {"name": []},
                        }
                    }
                ]
            }),
        )
    )

    assert plan.supports_deferred_output is False
    assert all(audit.serialization_name != "ToolApprovalCapability" for audit in plan.custom_capability_audit)


def test_resolver_rejects_descriptor_with_broadened_deferred_output() -> None:
    resolver = SubagentPlanResolver(build_default_capability_catalog(), default_model="test")
    plan = resolver.resolve(SubagentSpec(route="ordinary", agent=AgentSpec()))
    descriptor = plan.to_descriptor().model_copy(update={"supports_deferred_output": True})

    with pytest.raises(ValueError, match="deferred output contract"):
        resolver.restore(descriptor)


def test_packaging_provenance_does_not_change_plan_identity_or_restore() -> None:
    explicit_catalog = build_default_capability_catalog(explicit_types=[AuditCapability])
    explicit_resolver = SubagentPlanResolver(explicit_catalog, default_model="test")
    sdk_resolver = SubagentPlanResolver(
        build_capability_catalog(sdk_types=explicit_catalog.custom_capability_types),
        default_model="test",
    )
    spec = SubagentSpec(
        route="worker",
        agent=AgentSpec.from_dict({"capabilities": [{"AuditCapability": {"text": "same"}}]}),
    )

    explicit_plan = explicit_resolver.resolve(spec)
    sdk_plan = sdk_resolver.resolve(spec)
    restored = sdk_resolver.restore(explicit_plan.to_descriptor())

    assert explicit_plan.custom_capability_audit != sdk_plan.custom_capability_audit
    assert explicit_plan.fingerprint == sdk_plan.fingerprint
    assert restored.fingerprint == explicit_plan.fingerprint
    assert restored.custom_capability_audit == explicit_plan.custom_capability_audit

    explicit_first = SubagentRegistry([explicit_plan])
    explicit_first.register_retained(sdk_plan)
    explicit_first.register_retained(restored)
    sdk_first = SubagentRegistry([sdk_plan])
    sdk_first.register_retained(explicit_plan)
    assert len(explicit_first.list_registered()) == 1
    assert len(sdk_first.list_registered()) == 1


def test_resolver_injects_final_visibility_and_builds_bounded_self_fork() -> None:
    resolver = SubagentPlanResolver(
        build_default_capability_catalog(),
        default_model="test",
    )
    history = tuple({"index": index} for index in range(5))

    plan = resolver.resolve_self(
        SelfForkPolicy(
            agent=AgentSpec(description="Fork the current agent"),
            history_message_limit=2,
            execution_modes=(
                SubagentExecutionMode.foreground,
                SubagentExecutionMode.background,
            ),
        ),
        history=history,
    )

    assert plan.spec.route == "self"
    assert plan.spec.history is SubagentHistoryPolicy.parent_snapshot
    assert plan.spec.history_message_limit == 2
    assert plan.initial_history == history[-2:]
    assert any(capability.name == "ToolVisibilityCapability" for capability in plan.normalized_agent_spec.capabilities)


def test_resolver_clones_portable_values_across_plan_and_registry_boundaries() -> None:
    resolver = SubagentPlanResolver(
        build_default_capability_catalog(),
        default_model="test",
    )
    spec = SubagentSpec(
        route="worker",
        agent=AgentSpec(
            metadata={"nested": {"values": ["original"]}},
            instructions="Work on {{template.project}}",
        ),
    )
    template = AgentTemplateContext(
        template={"project": "YA", "nested": {"values": ["template"]}},
    )

    plan = resolver.resolve(spec, template_context=template)
    descriptor = plan.to_descriptor()
    registry = SubagentRegistry([plan])

    assert spec.agent.metadata is not None
    spec.agent.metadata["nested"]["values"].append("mutated")
    template.template["nested"]["values"].append("mutated")
    assert plan.normalized_agent_spec.metadata == {"nested": {"values": ["original"]}}
    assert plan.template_context.template == {
        "project": "YA",
        "nested": {"values": ["template"]},
    }

    assert plan.normalized_agent_spec.metadata is not None
    plan.normalized_agent_spec.metadata["nested"]["values"].append("plan")
    assert descriptor.normalized_agent_spec.metadata == {"nested": {"values": ["original"]}}

    registry_copy = registry.get("worker")
    assert registry_copy.normalized_agent_spec.metadata is not None
    registry_copy.normalized_agent_spec.metadata["nested"]["values"].append("copy")
    assert registry.get("worker").normalized_agent_spec.metadata == {"nested": {"values": ["original"]}}


def test_registry_rejects_a_plan_mutated_after_resolution() -> None:
    resolver = SubagentPlanResolver(
        build_default_capability_catalog(),
        default_model="test",
    )
    plan = resolver.resolve(
        SubagentSpec(
            route="worker",
            agent=AgentSpec(metadata={"nested": {"values": ["original"]}}),
        )
    )
    assert plan.normalized_agent_spec.metadata is not None
    plan.normalized_agent_spec.metadata["nested"]["values"].append("mutated")

    with pytest.raises(ValueError, match="fingerprint"):
        SubagentRegistry([plan])


def test_registry_rejects_mutated_source_agent_snapshot() -> None:
    resolver = SubagentPlanResolver(
        build_default_capability_catalog(),
        default_model="test",
    )
    plan = resolver.resolve(
        SubagentSpec(
            route="worker",
            agent=AgentSpec(metadata={"nested": {"values": ["original"]}}),
        )
    )
    assert plan.spec.agent.metadata is not None
    plan.spec.agent.metadata["nested"]["values"].append("mutated")

    with pytest.raises(ValueError, match="fingerprint"):
        SubagentRegistry([plan])


def test_resolver_and_registry_clone_host_capability_configuration() -> None:
    source_capability = AuditCapability(text="host")
    resolver = SubagentPlanResolver(
        build_default_capability_catalog(explicit_types=[AuditCapability]),
        default_model="test",
        host_capabilities=(source_capability,),
    )
    plan = resolver.resolve(SubagentSpec(route="worker", agent=AgentSpec()))
    registry = SubagentRegistry([plan])

    source_capability.text = "source-mutated"
    plan.host_capabilities[0].text = "plan-mutated"
    registry_copy = registry.get("worker")
    registry_copy.host_capabilities[0].text = "copy-mutated"

    stored_capability = registry.get("worker").host_capabilities[0]
    assert isinstance(stored_capability, AuditCapability)
    assert stored_capability.text == "host"


def test_resolver_renders_only_native_template_capable_fields() -> None:
    resolver = SubagentPlanResolver(
        build_default_capability_catalog(),
        default_model="test",
    )
    plan = resolver.resolve(
        SubagentSpec(
            route="worker",
            agent=AgentSpec(
                instructions="Work on {{template.project}}",
                metadata={"literal": "{{template.project}}"},
            ),
        ),
        template_context={"project": "YA"},
    )

    assert str(plan.normalized_agent_spec.instructions) == "Work on YA"
    assert plan.normalized_agent_spec.metadata == {"literal": "{{template.project}}"}


def test_resolver_rejects_nonportable_template_authority() -> None:
    resolver = SubagentPlanResolver(
        build_default_capability_catalog(),
        default_model="test",
    )
    spec = SubagentSpec(
        route="unsafe",
        agent=AgentSpec.from_dict({"instructions": "{{env.file_operator}}"}),
    )

    with pytest.raises(Exception, match="env"):
        resolver.resolve(spec)


def test_resolver_requires_explicit_host_features_and_durability() -> None:
    resolver = SubagentPlanResolver(build_default_capability_catalog())

    with pytest.raises(ValueError, match="unavailable host features"):
        resolver.resolve(
            SubagentSpec(
                route="hosted",
                host_requirements=("sql",),
                agent=AgentSpec(model="test"),
            )
        )

    with pytest.raises(ValueError, match="restart-durable"):
        resolver.resolve(
            SubagentSpec.model_validate({
                "route": "durable",
                "durability": "restart",
                "agent": {"model": "test"},
            })
        )


def test_process_durability_is_a_minimum_across_host_drivers() -> None:
    catalog = build_default_capability_catalog()
    spec = SubagentSpec(
        route="portable",
        durability="process",
        agent=AgentSpec(model="test"),
    )

    process_plan = SubagentPlanResolver(catalog, restart_durable=False).resolve(spec)
    restart_plan = SubagentPlanResolver(catalog, restart_durable=True).resolve(spec)

    assert process_plan.restart_durable is False
    assert restart_plan.restart_durable is True


async def test_default_service_runs_foreground_inline_with_readable_identity() -> None:
    catalog = build_default_capability_catalog()
    plan = SubagentPlanResolver(catalog).resolve(
        SubagentSpec(
            route="worker",
            execution_modes=(SubagentExecutionMode.foreground, SubagentExecutionMode.background),
            agent=AgentSpec(model="test"),
        )
    )
    store = InMemorySubagentExecutionStore()
    service = SubagentExecutionService(
        SubagentRegistry([plan]),
        store,
        InProcessSubagentDriver(
            custom_capability_types=catalog.custom_capability_types,
            model_resolver=lambda _name: TestModel(call_tools=[]),
        ),
    )
    parent = AgentContext(delegation_scope_id="session")
    object.__setattr__(parent, "_stream_queue_enabled", True)

    handle = await service.spawn("worker", "inline work", parent)
    committed = await store.get(handle.execution_id, owner_scope_id="session")
    lifecycle_events = []
    queue = parent.agent_stream_queues[handle.execution_id]
    while not queue.empty():
        lifecycle_events.append(queue.get_nowait())

    assert re.fullmatch(r"worker-[0-9a-f]{4}", handle.execution_id)
    lifecycle_types = [
        type(event) for event in lifecycle_events if isinstance(event, SubagentStartEvent | SubagentCompleteEvent)
    ]
    assert lifecycle_types == [SubagentStartEvent, SubagentCompleteEvent]
    assert parent.agent_stream_info[handle.execution_id].agent_name == "worker"
    assert committed is not None
    assert committed.state is SubagentExecutionState.succeeded
    with pytest.raises(ValueError, match="does not support 'background'"):
        await service.spawn(
            "worker",
            "must be host-owned",
            parent,
            mode=SubagentExecutionMode.background,
        )
    await service.close()


async def test_async_host_retains_completed_failure_until_wait_observes_it() -> None:
    host = AsyncioSubagentExecutionHost()

    async def fail() -> None:
        raise RuntimeError("host operation failed")

    await host.start(
        "worker-bg-fail",
        SubagentExecutionMode.background,
        fail,
        task_name="test-host-failure",
    )
    await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="host operation failed"):
        await host.wait("worker-bg-fail")
    await host.close()


async def test_in_process_service_runs_same_plan_in_foreground_and_background() -> None:
    catalog = build_default_capability_catalog()
    resolver = SubagentPlanResolver(catalog)
    plan = resolver.resolve(
        SubagentSpec(
            route="worker",
            execution_modes=(
                SubagentExecutionMode.foreground,
                SubagentExecutionMode.background,
            ),
            agent=AgentSpec(model="test", description="A test worker"),
        )
    )
    registry = SubagentRegistry([plan])
    store = InMemorySubagentExecutionStore()
    driver = InProcessSubagentDriver(
        custom_capability_types=catalog.custom_capability_types,
        model_resolver=lambda _name: TestModel(call_tools=[]),
    )
    service = SubagentExecutionService(
        registry,
        store,
        driver,
        execution_host=AsyncioSubagentExecutionHost(),
    )
    parent = AgentContext(run_input_ledger=RunInputLedger())

    foreground = await service.spawn(
        "worker",
        "foreground task",
        parent,
        mode=SubagentExecutionMode.foreground,
    )
    assert parent.delegation_scope_id is not None
    foreground_record = await service.wait(
        foreground.execution_id,
        caller_scope_id=parent.delegation_scope_id,
    )

    parent_agent = Agent(TestModel(call_tools=[]), deps_type=AgentContext)
    parent_router = LogicalRunInputRouter(parent.run_input_ledger)
    registration = parent.active_run_registry.register(parent_router)
    parent.input_router = parent_router
    try:
        async with parent_agent.iter("parent", deps=parent) as run:
            await parent_router.bind(run, native_attempt_id="parent-attempt")
            background = await service.spawn(
                "worker",
                "background task",
                parent,
                mode=SubagentExecutionMode.background,
            )
            background_record = await service.wait(
                background.execution_id,
                caller_scope_id=parent.delegation_scope_id,
            )
            assert await service.deliver_pending(parent) == 0
            parent_router.unbind(native_attempt_id="parent-attempt")
    finally:
        parent_router.close()
        parent.active_run_registry.unregister(registration)
        parent.input_router = None
        await service.close()

    assert foreground_record.state.value == "succeeded"
    assert background_record.state.value == "succeeded"
    assert foreground_record.output == "success (no tool calls)"
    assert re.fullmatch(r"worker-[0-9a-f]{4}", foreground.execution_id)
    assert re.fullmatch(r"worker-bg-[0-9a-f]{4}", background.execution_id)
    assert background_record.delivery_state.value == "pending"
    feature_records = [record for record in parent.run_input_ledger.records if record.origin is InputOrigin.feature]
    assert len(feature_records) == 1
    assert feature_records[0].disposition.value == "rejected"
    assert "background task" not in str(feature_records[0].messages)
    assert background.execution_id in str(feature_records[0].messages)


async def test_service_idempotency_returns_one_execution() -> None:
    catalog = build_default_capability_catalog()
    plan = SubagentPlanResolver(catalog).resolve(SubagentSpec(route="worker", agent=AgentSpec(model="test")))
    service = SubagentExecutionService(
        SubagentRegistry([plan]),
        InMemorySubagentExecutionStore(),
        InProcessSubagentDriver(
            custom_capability_types=catalog.custom_capability_types,
            model_resolver=lambda _name: TestModel(call_tools=[]),
        ),
    )
    parent = AgentContext()

    first, second = await asyncio.gather(
        service.spawn("worker", "one", parent, idempotency_key="same"),
        service.spawn("worker", "one", parent, idempotency_key="same"),
    )
    with pytest.raises(ValueError, match="different intent"):
        await service.spawn("worker", "two", parent, idempotency_key="same")
    assert parent.delegation_scope_id is not None
    await service.wait(
        first.execution_id,
        caller_scope_id=parent.delegation_scope_id,
    )
    records = await service.list(
        caller_scope_id=parent.delegation_scope_id,
    )
    await service.close()

    assert first.execution_id == second.execution_id
    assert len(records) == 1
    assert records[0].input_state is SubagentInputState.applied


async def test_two_services_retry_a_store_level_execution_id_collision() -> None:
    class RacingStore(InMemorySubagentExecutionStore):
        def __init__(self) -> None:
            super().__init__()
            self.fixed_arrivals = 0
            self.both_fixed = asyncio.Event()

        async def create(self, record: SubagentExecutionRecord) -> SubagentExecutionRecord:
            if record.execution_id == "worker-fixed":
                self.fixed_arrivals += 1
                if self.fixed_arrivals == 2:
                    self.both_fixed.set()
                await self.both_fixed.wait()
            return await super().create(record)

    catalog = build_default_capability_catalog()
    plan = SubagentPlanResolver(catalog).resolve(SubagentSpec(route="worker", agent=AgentSpec(model="test")))
    store = RacingStore()

    def build_service(retry_id: str) -> SubagentExecutionService:
        candidates = iter(("worker-fixed", retry_id))
        return SubagentExecutionService(
            SubagentRegistry([plan]),
            store,
            InProcessSubagentDriver(
                custom_capability_types=catalog.custom_capability_types,
                model_resolver=lambda _name: TestModel(call_tools=[]),
            ),
            execution_id_factory=lambda _route, _mode: next(candidates),
        )

    first_service = build_service("worker-retry-a")
    second_service = build_service("worker-retry-b")
    first, second = await asyncio.gather(
        first_service.spawn(
            "worker",
            "first",
            AgentContext(delegation_scope_id="first-owner"),
        ),
        second_service.spawn(
            "worker",
            "second",
            AgentContext(delegation_scope_id="second-owner"),
        ),
    )

    assert "worker-fixed" in {first.execution_id, second.execution_id}
    assert first.execution_id != second.execution_id
    await first_service.close()
    await second_service.close()


async def test_initial_input_is_rejected_when_driver_fails_before_application() -> None:
    catalog = build_default_capability_catalog()
    plan = SubagentPlanResolver(catalog).resolve(SubagentSpec(route="worker", agent=AgentSpec(model="test")))

    def reject_model_resolution(_name: str) -> str:
        raise RuntimeError("model resolution failed")

    service = SubagentExecutionService(
        SubagentRegistry([plan]),
        InMemorySubagentExecutionStore(),
        InProcessSubagentDriver(
            custom_capability_types=catalog.custom_capability_types,
            model_resolver=reject_model_resolution,
        ),
    )
    parent = AgentContext(delegation_scope_id="session")

    handle = await service.spawn("worker", "never applied", parent)
    record = await service.wait(handle.execution_id, caller_scope_id="session")
    await service.close()

    assert record.state is SubagentExecutionState.failed
    assert record.input_state is SubagentInputState.rejected
    assert record.error == "model resolution failed"


async def test_initial_input_stays_applied_when_model_fails_after_graph_admission() -> None:
    catalog = build_default_capability_catalog()
    plan = SubagentPlanResolver(catalog).resolve(SubagentSpec(route="worker", agent=AgentSpec(model="test")))

    async def fail_after_admission(
        _messages: list[ModelMessage],
        _info: FunctionAgentInfo,
    ):
        raise RuntimeError("model request failed")
        yield "unreachable"  # pragma: no cover

    service = SubagentExecutionService(
        SubagentRegistry([plan]),
        InMemorySubagentExecutionStore(),
        InProcessSubagentDriver(
            custom_capability_types=catalog.custom_capability_types,
            model_resolver=lambda _name: FunctionModel(stream_function=fail_after_admission),
        ),
    )
    parent = AgentContext(delegation_scope_id="session")

    handle = await service.spawn("worker", "applied before model failure", parent)
    record = await service.wait(handle.execution_id, caller_scope_id="session")
    await service.close()

    assert record.state is SubagentExecutionState.failed
    assert record.input_state is SubagentInputState.applied
    assert record.error == "model request failed"


async def test_initial_input_stays_applied_when_cancelled_after_graph_admission() -> None:
    catalog = build_default_capability_catalog()
    plan = SubagentPlanResolver(catalog).resolve(
        SubagentSpec(
            route="worker",
            execution_modes=(SubagentExecutionMode.background,),
            agent=AgentSpec(model="test"),
        )
    )
    admitted = asyncio.Event()
    never = asyncio.Event()

    async def block_after_admission(
        _messages: list[ModelMessage],
        _info: FunctionAgentInfo,
    ):
        admitted.set()
        await never.wait()
        yield "unreachable"  # pragma: no cover

    service = SubagentExecutionService(
        SubagentRegistry([plan]),
        InMemorySubagentExecutionStore(),
        InProcessSubagentDriver(
            custom_capability_types=catalog.custom_capability_types,
            model_resolver=lambda _name: FunctionModel(stream_function=block_after_admission),
        ),
        execution_host=AsyncioSubagentExecutionHost(),
    )
    parent = AgentContext(delegation_scope_id="session")

    handle = await service.spawn(
        "worker",
        "applied before cancellation",
        parent,
        mode=SubagentExecutionMode.background,
    )
    await asyncio.wait_for(admitted.wait(), timeout=2)
    record = await service.cancel(handle.execution_id, caller_scope_id="session")
    await service.close()

    assert record.state is SubagentExecutionState.cancelled
    assert record.input_state is SubagentInputState.applied


async def test_async_host_close_marks_active_process_local_execution_lost() -> None:
    catalog = build_default_capability_catalog()
    plan = SubagentPlanResolver(catalog).resolve(
        SubagentSpec(
            route="worker",
            execution_modes=(SubagentExecutionMode.background,),
            agent=AgentSpec(model="test"),
        )
    )
    admitted = asyncio.Event()
    never = asyncio.Event()

    async def block_after_admission(
        _messages: list[ModelMessage],
        _info: FunctionAgentInfo,
    ):
        admitted.set()
        await never.wait()
        yield "unreachable"  # pragma: no cover

    store = InMemorySubagentExecutionStore()
    service = SubagentExecutionService(
        SubagentRegistry([plan]),
        store,
        InProcessSubagentDriver(
            custom_capability_types=catalog.custom_capability_types,
            model_resolver=lambda _name: FunctionModel(stream_function=block_after_admission),
        ),
        execution_host=AsyncioSubagentExecutionHost(),
    )
    parent = AgentContext(delegation_scope_id="session")

    handle = await service.spawn(
        "worker",
        "interrupted by host shutdown",
        parent,
        mode=SubagentExecutionMode.background,
    )
    await asyncio.wait_for(admitted.wait(), timeout=2)
    await service.close()
    record = await store.get(handle.execution_id, owner_scope_id="session")

    assert record is not None
    assert record.state is SubagentExecutionState.lost
    assert record.input_state is SubagentInputState.rejected
    assert record.error == "Process-local subagent execution ended with its host process"


async def test_close_waits_for_spawn_admission_before_closing_host_and_store() -> None:
    class BlockingCreateStore(InMemorySubagentExecutionStore):
        def __init__(self) -> None:
            super().__init__()
            self.create_started = asyncio.Event()
            self.release_create = asyncio.Event()

        async def create(self, record: SubagentExecutionRecord) -> SubagentExecutionRecord:
            self.create_started.set()
            await self.release_create.wait()
            return await super().create(record)

    catalog = build_default_capability_catalog()
    plan = SubagentPlanResolver(catalog).resolve(
        SubagentSpec(
            route="worker",
            execution_modes=(SubagentExecutionMode.background,),
            agent=AgentSpec(model="test"),
        )
    )
    store = BlockingCreateStore()
    service = SubagentExecutionService(
        SubagentRegistry([plan]),
        store,
        InProcessSubagentDriver(
            custom_capability_types=catalog.custom_capability_types,
            model_resolver=lambda _name: TestModel(call_tools=[]),
        ),
        execution_host=AsyncioSubagentExecutionHost(),
    )
    spawn_task = asyncio.create_task(
        service.spawn(
            "worker",
            "admitted before close",
            AgentContext(delegation_scope_id="session"),
            mode=SubagentExecutionMode.background,
        )
    )
    await store.create_started.wait()
    close_task = asyncio.create_task(service.close())
    await asyncio.sleep(0)

    assert not close_task.done()
    store.release_create.set()
    handle = await spawn_task
    await close_task
    record = await store.get(handle.execution_id, owner_scope_id="session")

    assert record is not None
    assert record.terminal


async def test_service_scopes_every_execution_operation_to_its_owner() -> None:
    class WaitTrackingHost(AsyncioSubagentExecutionHost):
        def __init__(self) -> None:
            super().__init__()
            self.wait_calls: list[str] = []

        async def wait(self, execution_id: str, *, timeout: float | None = None) -> None:
            self.wait_calls.append(execution_id)
            await super().wait(execution_id, timeout=timeout)

    catalog = build_default_capability_catalog()
    plan = SubagentPlanResolver(catalog).resolve(SubagentSpec(route="worker", agent=AgentSpec(model="test")))
    host = WaitTrackingHost()
    service = SubagentExecutionService(
        SubagentRegistry([plan]),
        InMemorySubagentExecutionStore(),
        InProcessSubagentDriver(
            custom_capability_types=catalog.custom_capability_types,
            model_resolver=lambda _name: TestModel(call_tools=[]),
        ),
        execution_host=host,
    )
    owner = AgentContext(delegation_scope_id="session-owner")
    stranger = AgentContext(delegation_scope_id="session-stranger")
    handle = await service.spawn("worker", "bounded work", owner)
    with pytest.raises(KeyError, match="Unknown subagent execution"):
        await service.wait(
            handle.execution_id,
            caller_scope_id="session-stranger",
            timeout=1,
        )
    assert host.wait_calls == []
    record = await service.wait(
        handle.execution_id,
        caller_scope_id="session-owner",
    )

    assert host.wait_calls == [handle.execution_id]
    assert record.owner_scope_id == "session-owner"
    assert await service.list(caller_scope_id="session-stranger") == ()
    with pytest.raises(KeyError, match="Unknown subagent execution"):
        await service.get(
            handle.execution_id,
            caller_scope_id="session-stranger",
        )
    with pytest.raises(KeyError, match="Unknown subagent execution"):
        await service.resume(handle.execution_id, "continue", stranger)
    await service.close()


async def test_suspended_execution_continues_in_place_with_cumulative_usage() -> None:
    DeferredApprovalCapability.effects = []
    catalog = build_default_capability_catalog()
    plan = SubagentPlanResolver(
        catalog,
        host_capabilities=(DeferredApprovalCapability(),),
    ).resolve(SubagentSpec(route="worker", agent=AgentSpec(model="test")))
    service = SubagentExecutionService(
        SubagentRegistry([plan]),
        InMemorySubagentExecutionStore(),
        InProcessSubagentDriver(
            custom_capability_types=catalog.custom_capability_types,
            model_resolver=lambda _name: TestModel(
                call_tools=["guarded_effect"],
                custom_output_text="continued",
            ),
        ),
    )
    parent = AgentContext(delegation_scope_id="session")
    handle = await service.spawn("worker", "use the guarded effect", parent)
    suspended = await service.wait(
        handle.execution_id,
        caller_scope_id="session",
    )

    assert suspended.state.value == "suspended", suspended.error
    assert suspended.deferred is not None
    requests = TypeAdapter(DeferredToolRequests).validate_python(suspended.deferred)
    tool_call_id = requests.approvals[0].tool_call_id
    continued = await service.continue_deferred(
        handle.execution_id,
        DeferredToolResults(approvals={tool_call_id: ToolDenied(message="Denied by the host test")}),
        parent,
    )
    completed = await service.wait(
        continued.execution_id,
        caller_scope_id="session",
    )
    await service.close()

    assert continued.execution_id == handle.execution_id
    assert completed.state.value == "succeeded"
    assert completed.segment_index == 1
    assert completed.output == "continued"
    assert completed.usage["requests"] == 2
    assert completed.deferred_results is None
    assert DeferredApprovalCapability.effects == []


def test_plan_descriptor_round_trips_and_rejects_tampering() -> None:
    catalog = build_default_capability_catalog(explicit_types=[AuditCapability])
    resolver = SubagentPlanResolver(
        catalog,
        host_capabilities=(AuditCapability(text="host"),),
        restart_durable=True,
    )
    plan = resolver.resolve(
        SubagentSpec(
            route="worker",
            agent=AgentSpec(
                model="test",
                name="worker",
                output_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
                capabilities=[{"AuditCapability": {"text": "child"}}],
            ),
            durability="restart",
        )
    )

    payload = plan.to_descriptor().model_dump(mode="json")
    restored = resolver.restore(type(plan.to_descriptor()).model_validate(payload))

    assert restored.fingerprint == plan.fingerprint
    assert restored.normalized_agent_spec.output_schema == plan.normalized_agent_spec.output_schema
    assert restored.host_capabilities == plan.host_capabilities

    payload["normalized_agent_spec"]["model"] = "changed"
    with pytest.raises(ValueError, match="fingerprint"):
        resolver.restore(type(plan.to_descriptor()).model_validate(payload))


def test_registry_keeps_active_route_and_retained_descriptor_versions() -> None:
    resolver = SubagentPlanResolver(
        build_default_capability_catalog(),
        default_model="test",
    )
    retained = resolver.resolve(
        SubagentSpec(
            route="worker",
            agent=AgentSpec(instructions="retained plan"),
        )
    )
    active = resolver.resolve(
        SubagentSpec(
            route="worker",
            agent=AgentSpec(instructions="active plan"),
        )
    )

    registry = SubagentRegistry([active])
    registry.register_retained(retained)

    assert registry.get("worker").descriptor_id == active.descriptor_id
    assert registry.get_descriptor(retained.descriptor_id).fingerprint == retained.fingerprint
    assert [plan.descriptor_id for plan in registry.list()] == [active.descriptor_id]
    assert {plan.descriptor_id for plan in registry.list_registered()} == {
        active.descriptor_id,
        retained.descriptor_id,
    }


async def test_nested_spawn_uses_parent_exact_descriptor_when_active_policy_removes_grant() -> None:
    catalog = build_default_capability_catalog()
    resolver = SubagentPlanResolver(
        catalog,
        default_model="test",
    )
    retained_parent = resolver.resolve(
        SubagentSpec(
            route="parent",
            agent=AgentSpec(instructions="retained parent"),
            spawn_targets=("worker",),
        )
    )
    active_parent = resolver.resolve(
        SubagentSpec(
            route="parent",
            agent=AgentSpec(instructions="active parent"),
        )
    )
    worker = resolver.resolve(
        SubagentSpec(
            route="worker",
            agent=AgentSpec(instructions="worker"),
            max_depth=2,
        )
    )
    registry = SubagentRegistry([active_parent, worker])
    registry.register_retained(retained_parent)
    service = SubagentExecutionService(
        registry,
        InMemorySubagentExecutionStore(),
        InProcessSubagentDriver(
            custom_capability_types=catalog.custom_capability_types,
            model_resolver=lambda _name: TestModel(custom_output_text="done"),
        ),
    )
    parent_ctx = AgentContext(
        delegation_scope_id="session",
        subagent_depth=1,
        subagent_target="parent",
        subagent_descriptor_id=retained_parent.descriptor_id,
    )

    handle = await service.spawn("worker", "work", parent_ctx)
    completed = await service.wait(handle.execution_id, caller_scope_id="session")
    await service.close()

    assert completed.state is SubagentExecutionState.succeeded


async def test_nested_spawn_uses_retained_parent_policy_after_active_route_deletion() -> None:
    catalog = build_default_capability_catalog()
    resolver = SubagentPlanResolver(
        catalog,
        default_model="test",
    )
    retained_parent = resolver.resolve(
        SubagentSpec(
            route="parent",
            agent=AgentSpec(instructions="deleted parent"),
            spawn_targets=("worker",),
        )
    )
    worker = resolver.resolve(
        SubagentSpec(
            route="worker",
            agent=AgentSpec(instructions="worker"),
            max_depth=2,
        )
    )
    registry = SubagentRegistry([worker])
    registry.register_retained(retained_parent)
    service = SubagentExecutionService(
        registry,
        InMemorySubagentExecutionStore(),
        InProcessSubagentDriver(
            custom_capability_types=catalog.custom_capability_types,
            model_resolver=lambda _name: TestModel(custom_output_text="done"),
        ),
    )
    parent_ctx = AgentContext(
        delegation_scope_id="session",
        subagent_depth=1,
        subagent_target="parent",
        subagent_descriptor_id=retained_parent.descriptor_id,
    )

    handle = await service.spawn("worker", "work", parent_ctx)
    completed = await service.wait(handle.execution_id, caller_scope_id="session")
    await service.close()

    assert completed.state is SubagentExecutionState.succeeded


async def test_nested_spawn_rejects_grant_added_only_to_active_parent_policy() -> None:
    catalog = build_default_capability_catalog()
    resolver = SubagentPlanResolver(
        catalog,
        default_model="test",
    )
    retained_parent = resolver.resolve(
        SubagentSpec(
            route="parent",
            agent=AgentSpec(instructions="retained parent"),
        )
    )
    active_parent = resolver.resolve(
        SubagentSpec(
            route="parent",
            agent=AgentSpec(instructions="active parent"),
            spawn_targets=("worker",),
        )
    )
    worker = resolver.resolve(
        SubagentSpec(
            route="worker",
            agent=AgentSpec(instructions="worker"),
            max_depth=2,
        )
    )
    registry = SubagentRegistry([active_parent, worker])
    registry.register_retained(retained_parent)
    service = SubagentExecutionService(
        registry,
        InMemorySubagentExecutionStore(),
        InProcessSubagentDriver(model_resolver=lambda _name: TestModel()),
    )
    parent_ctx = AgentContext(
        delegation_scope_id="session",
        subagent_depth=1,
        subagent_target="parent",
        subagent_descriptor_id=retained_parent.descriptor_id,
    )

    with pytest.raises(ValueError, match="is not allowed to spawn"):
        await service.spawn("worker", "work", parent_ctx)
    await service.close()


async def test_host_deferred_resolver_continues_same_execution_until_terminal() -> None:
    DeferredApprovalCapability.effects = []
    catalog = build_default_capability_catalog()
    plan = SubagentPlanResolver(
        catalog,
        host_capabilities=(DeferredApprovalCapability(),),
    ).resolve(SubagentSpec(route="worker", agent=AgentSpec(model="test")))

    class DenyResolver:
        def __init__(self) -> None:
            self.records: list[SubagentExecutionRecord] = []

        async def resolve(
            self,
            record: SubagentExecutionRecord,
            requests: DeferredToolRequests,
        ) -> DeferredToolResults:
            self.records.append(record)
            return DeferredToolResults(
                approvals={request.tool_call_id: ToolDenied(message="Denied by host") for request in requests.approvals}
            )

    resolver = DenyResolver()
    service = SubagentExecutionService(
        SubagentRegistry([plan]),
        InMemorySubagentExecutionStore(),
        InProcessSubagentDriver(
            custom_capability_types=catalog.custom_capability_types,
            model_resolver=lambda _name: TestModel(
                call_tools=["guarded_effect"],
                custom_output_text="continued",
            ),
        ),
        deferred_resolver=resolver,
    )
    parent = AgentContext(delegation_scope_id="session")

    handle = await service.spawn("worker", "use the guarded effect", parent)
    completed = await service.wait(
        handle.execution_id,
        caller_scope_id="session",
    )
    await service.close()

    assert completed.execution_id == handle.execution_id
    assert completed.state.value == "succeeded"
    assert completed.segment_index == 1
    assert completed.output == "continued"
    assert completed.usage["requests"] == 2
    assert len(resolver.records) == 1
    assert resolver.records[0].state.value == "suspended"
    assert DeferredApprovalCapability.effects == []


async def test_idempotent_recovery_executes_existing_descriptor_not_active_route() -> None:
    resolver = SubagentPlanResolver(
        build_default_capability_catalog(),
        default_model="test",
    )
    retained = resolver.resolve(SubagentSpec(route="worker", agent=AgentSpec(instructions="retained")))
    active = resolver.resolve(SubagentSpec(route="worker", agent=AgentSpec(instructions="active")))
    registry = SubagentRegistry([active])
    registry.register_retained(retained)
    store = InMemorySubagentExecutionStore()
    await store.create(
        SubagentExecutionRecord(
            execution_id="retained-execution",
            root_execution_id="retained-execution",
            owner_scope_id="session",
            idempotency_key="same",
            descriptor_id=retained.descriptor_id,
            plan_fingerprint=retained.fingerprint,
            route="worker",
            mode=SubagentExecutionMode.foreground,
            parent_agent_id="main",
            parent_logical_run_id="parent-run",
            prompt="retained work",
        )
    )

    class CapturingDriver:
        restart_durable = False

        def __init__(self) -> None:
            self.descriptor_ids: list[str] = []

        async def run(
            self,
            plan: ResolvedSubagentPlan,
            record: SubagentExecutionRecord,
            parent_ctx: AgentContext,
        ) -> SubagentDriverOutcome:
            del record, parent_ctx
            self.descriptor_ids.append(plan.descriptor_id)
            return SubagentDriverOutcome(
                state=SubagentExecutionState.succeeded,
                input_state=SubagentInputState.applied,
                output="done",
            )

        async def cancel(self, record: SubagentExecutionRecord) -> None:
            del record

    driver = CapturingDriver()
    service = SubagentExecutionService(registry, store, driver)
    parent = AgentContext(delegation_scope_id="session")

    handle = await service.spawn(
        "worker",
        "retained work",
        parent,
        idempotency_key="same",
    )
    completed = await service.wait(
        handle.execution_id,
        caller_scope_id="session",
    )
    await service.close()

    assert completed.output == "done"
    assert driver.descriptor_ids == [retained.descriptor_id]


async def test_resume_lazily_restores_prior_descriptor_instead_of_active_route() -> None:
    resolver = SubagentPlanResolver(
        build_default_capability_catalog(),
        default_model="test",
    )
    prior = resolver.resolve(SubagentSpec(route="worker", agent=AgentSpec(instructions="prior")))
    active = resolver.resolve(SubagentSpec(route="worker", agent=AgentSpec(instructions="active")))
    store = InMemorySubagentExecutionStore()
    await store.create(
        SubagentExecutionRecord(
            execution_id="prior-execution",
            root_execution_id="prior-execution",
            owner_scope_id="session",
            idempotency_key="prior",
            descriptor_id=prior.descriptor_id,
            plan_fingerprint=prior.fingerprint,
            route="worker",
            mode=SubagentExecutionMode.foreground,
            state=SubagentExecutionState.succeeded,
            input_state="applied",
            parent_agent_id="main",
            parent_logical_run_id="parent-run",
            prompt="prior work",
            history=[{"kind": "request", "parts": []}],
        )
    )

    class Provider:
        async def load_retained_plan(
            self,
            record: SubagentExecutionRecord,
        ) -> ResolvedSubagentPlan | None:
            assert record.descriptor_id == prior.descriptor_id
            return prior

    class CapturingDriver:
        restart_durable = False

        def __init__(self) -> None:
            self.plans: list[ResolvedSubagentPlan] = []

        async def run(
            self,
            plan: ResolvedSubagentPlan,
            record: SubagentExecutionRecord,
            parent_ctx: AgentContext,
        ) -> SubagentDriverOutcome:
            del record, parent_ctx
            self.plans.append(plan)
            return SubagentDriverOutcome(
                state=SubagentExecutionState.succeeded,
                input_state=SubagentInputState.applied,
                output="resumed",
            )

        async def cancel(self, record: SubagentExecutionRecord) -> None:
            del record

    driver = CapturingDriver()
    registry = SubagentRegistry([active])
    service = SubagentExecutionService(
        registry,
        store,
        driver,
        retained_plan_provider=Provider(),
    )
    parent = AgentContext(delegation_scope_id="session")

    handle = await service.resume(
        "prior-execution",
        "continue",
        parent,
        idempotency_key="resume",
    )
    completed = await service.wait(
        handle.execution_id,
        caller_scope_id="session",
    )
    await service.close()

    assert completed.descriptor_id == prior.descriptor_id
    assert completed.plan_fingerprint == prior.fingerprint
    assert completed.resumed_from == "prior-execution"
    assert driver.plans[0].descriptor_id == prior.descriptor_id
    assert registry.get("worker").descriptor_id == active.descriptor_id


def test_fixed_delegation_mode_rejects_incompatible_visible_routes() -> None:
    catalog = build_default_capability_catalog()
    plan = SubagentPlanResolver(catalog).resolve(SubagentSpec(route="worker", agent=AgentSpec(model="test")))
    registry = SubagentRegistry([plan])
    service = SubagentExecutionService(
        registry,
        InMemorySubagentExecutionStore(),
        InProcessSubagentDriver(
            custom_capability_types=catalog.custom_capability_types,
            model_resolver=lambda _name: TestModel(call_tools=[]),
        ),
        execution_host=AsyncioSubagentExecutionHost(),
    )

    with pytest.raises(ValueError, match="not allowed by subagent routes: 'worker'"):
        DelegationCapability(
            registry=registry,
            service=service,
            default_mode=SubagentExecutionMode.background,
        )


async def test_delegation_tool_fixes_inline_mode_and_hides_mode_argument_by_default() -> None:
    catalog = build_default_capability_catalog()
    plan = SubagentPlanResolver(catalog).resolve(SubagentSpec(route="worker", agent=AgentSpec(model="test")))
    registry = SubagentRegistry([plan])
    service = SubagentExecutionService(
        registry,
        InMemorySubagentExecutionStore(),
        InProcessSubagentDriver(
            custom_capability_types=catalog.custom_capability_types,
            model_resolver=lambda _name: TestModel(call_tools=[]),
        ),
    )
    capability = DelegationCapability(registry=registry, service=service)
    observed_delegate_schema: dict[str, object] = {}

    def parent_model(
        messages: list[ModelMessage],
        info: FunctionAgentInfo,
    ) -> ModelResponse:
        delegate_tool = next(tool for tool in info.function_tools if tool.name == "delegate")
        observed_delegate_schema.update(delegate_tool.parameters_json_schema)
        last = messages[-1]
        if isinstance(last, ModelRequest) and any(isinstance(part, ToolReturnPart) for part in last.parts):
            return ModelResponse(parts=[TextPart(content="done")])
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="delegate",
                    tool_call_id="fixed-inline-call",
                    args={"subagent_name": "worker", "prompt": "bounded work"},
                )
            ]
        )

    parent = AgentContext(
        delegation_scope_id="session",
        run_input_ledger=RunInputLedger(logical_run_id="parent-logical-run"),
    )
    agent = Agent(
        FunctionModel(function=parent_model),
        deps_type=AgentContext,
        capabilities=[capability],
    )

    result = await agent.run("delegate inline", deps=parent)
    records = await service.list(caller_scope_id="session")
    await service.close()

    assert result.output == "done"
    properties = observed_delegate_schema["properties"]
    assert isinstance(properties, dict)
    assert "mode" not in properties
    assert len(records) == 1
    assert records[0].mode is SubagentExecutionMode.foreground
    assert re.fullmatch(r"worker-[0-9a-f]{4}", records[0].execution_id)


async def test_fixed_background_delegation_returns_readable_handle_without_waiting() -> None:
    catalog = build_default_capability_catalog()
    plan = SubagentPlanResolver(catalog).resolve(
        SubagentSpec(
            route="worker",
            execution_modes=(SubagentExecutionMode.background,),
            agent=AgentSpec(model="test"),
        )
    )
    started = asyncio.Event()
    never = asyncio.Event()

    async def blocked_child(
        _messages: list[ModelMessage],
        _info: FunctionAgentInfo,
    ):
        started.set()
        await never.wait()
        yield "unreachable"  # pragma: no cover

    registry = SubagentRegistry([plan])
    service = SubagentExecutionService(
        registry,
        InMemorySubagentExecutionStore(),
        InProcessSubagentDriver(
            custom_capability_types=catalog.custom_capability_types,
            model_resolver=lambda _name: FunctionModel(stream_function=blocked_child),
        ),
        execution_host=AsyncioSubagentExecutionHost(),
    )
    capability = DelegationCapability(
        registry=registry,
        service=service,
        default_mode=SubagentExecutionMode.background,
    )
    observed_delegate_schema: dict[str, object] = {}
    tool_result: dict[str, object] = {}
    public_record: dict[str, object] = {}
    steering_result: dict[str, object] = {}

    def parent_model(
        messages: list[ModelMessage],
        info: FunctionAgentInfo,
    ) -> ModelResponse:
        delegate_tool = next(tool for tool in info.function_tools if tool.name == "delegate")
        observed_delegate_schema.update(delegate_tool.parameters_json_schema)
        last = messages[-1]
        if isinstance(last, ModelRequest):
            returned = next((part for part in last.parts if isinstance(part, ToolReturnPart)), None)
            if returned is not None and returned.tool_name == "delegate":
                tool_result.update(json.loads(str(returned.content)))
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="subagent_info",
                            tool_call_id="inspect-background-call",
                            args={"execution_id": tool_result["execution_id"]},
                        )
                    ]
                )
            if returned is not None and returned.tool_name == "subagent_info":
                public_record.update(json.loads(str(returned.content)))
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="steer_subagent",
                            tool_call_id="steer-background-call",
                            args={
                                "execution_id": tool_result["execution_id"],
                                "message": "additional guidance",
                            },
                        )
                    ]
                )
            if returned is not None:
                steering_result.update(json.loads(str(returned.content)))
                return ModelResponse(parts=[TextPart(content="delegated")])
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="delegate",
                    tool_call_id="fixed-background-call",
                    args={"subagent_name": "worker", "prompt": "independent work"},
                )
            ]
        )

    parent = AgentContext(
        delegation_scope_id="session",
        run_input_ledger=RunInputLedger(logical_run_id="parent-logical-run"),
    )
    agent = Agent(
        FunctionModel(function=parent_model),
        deps_type=AgentContext,
        capabilities=[capability],
    )

    result = await agent.run("delegate asynchronously", deps=parent)
    await asyncio.wait_for(started.wait(), timeout=2)
    records = await service.list(caller_scope_id="session")

    assert result.output == "delegated"
    properties = observed_delegate_schema["properties"]
    assert isinstance(properties, dict)
    assert "mode" not in properties
    assert len(records) == 1
    assert records[0].state is SubagentExecutionState.running
    assert records[0].mode is SubagentExecutionMode.background
    assert tool_result == {
        "execution_id": records[0].execution_id,
        "mode": "background",
        "route": "worker",
    }
    assert re.fullmatch(r"worker-bg-[0-9a-f]{4}", records[0].execution_id)
    assert public_record["execution_id"] == records[0].execution_id
    assert "child_logical_run_id" not in public_record
    assert "owner_scope_id" not in public_record
    assert "resumable_state" not in public_record
    assert steering_result == {
        "disposition": "enqueued",
        "execution_id": records[0].execution_id,
    }
    await service.close()


async def test_delegation_tool_replay_uses_native_tool_call_identity() -> None:
    catalog = build_default_capability_catalog()
    plan = SubagentPlanResolver(catalog).resolve(SubagentSpec(route="worker", agent=AgentSpec(model="test")))
    registry = SubagentRegistry([plan])
    store = InMemorySubagentExecutionStore()
    service = SubagentExecutionService(
        registry,
        store,
        InProcessSubagentDriver(
            custom_capability_types=catalog.custom_capability_types,
            model_resolver=lambda _name: TestModel(call_tools=[]),
        ),
    )
    capability = DelegationCapability(
        registry=registry,
        service=service,
        allow_mode_override=True,
    )

    def parent_model(
        messages: list[ModelMessage],
        _info: FunctionAgentInfo,
    ) -> ModelResponse:
        last = messages[-1]
        if isinstance(last, ModelRequest) and any(isinstance(part, ToolReturnPart) for part in last.parts):
            return ModelResponse(parts=[TextPart(content="done")])
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="delegate",
                    tool_call_id="stable-delegate-call",
                    args={
                        "subagent_name": "worker",
                        "prompt": "bounded work",
                        "mode": "foreground",
                    },
                )
            ]
        )

    parent = AgentContext(
        delegation_scope_id="session",
        run_input_ledger=RunInputLedger(logical_run_id="parent-logical-run"),
    )
    agent = Agent(
        FunctionModel(function=parent_model),
        deps_type=AgentContext,
        capabilities=[capability],
    )

    first = await agent.run("delegate once", deps=parent)
    second = await agent.run("replay the durable tool step", deps=parent)
    records = await service.list(caller_scope_id="session")
    await service.close()

    assert first.output == "done"
    assert second.output == "done"
    assert len(records) == 1
    assert records[0].idempotency_key == ("delegation:parent-logical-run:stable-delegate-call:spawn:worker")


async def test_live_child_steering_reuses_stable_operation_identity() -> None:
    plan = SubagentPlanResolver(
        build_default_capability_catalog(),
        default_model="test",
    ).resolve(SubagentSpec(route="worker", agent=AgentSpec()))
    store = InMemorySubagentExecutionStore()
    record = SubagentExecutionRecord(
        execution_id="child",
        root_execution_id="child",
        owner_scope_id="session",
        idempotency_key="spawn-once",
        descriptor_id=plan.descriptor_id,
        plan_fingerprint=plan.fingerprint,
        route="worker",
        mode=SubagentExecutionMode.background,
        state=SubagentExecutionState.running,
        parent_agent_id="main",
        parent_logical_run_id="parent-run",
        child_logical_run_id="child-run",
        prompt="work",
    )
    await store.create(record)
    service = SubagentExecutionService(
        SubagentRegistry([plan]),
        store,
        InProcessSubagentDriver(model_resolver=lambda _name: TestModel(call_tools=[])),
    )
    active_registry = AgentContext().active_run_registry
    router = LogicalRunInputRouter(RunInputLedger(logical_run_id="child-run"))
    registration = active_registry.register(router)
    service._active_run_registries[record.execution_id] = active_registry

    try:
        first = await service.steer(
            record.execution_id,
            "focus once",
            caller_scope_id="session",
            idempotency_key="stable-steer-call",
        )
        second = await service.steer(
            record.execution_id,
            "focus once",
            caller_scope_id="session",
            idempotency_key="stable-steer-call",
        )
    finally:
        router.close()
        active_registry.unregister(registration)
        await service.close()

    assert first.input_id == "stable-steer-call"
    assert second.input_id == first.input_id
    assert len(router.ledger.records) == 1


async def test_completion_delivery_waits_for_application_and_retargets_rejection() -> None:
    plan = SubagentPlanResolver(
        build_default_capability_catalog(),
        default_model="test",
    ).resolve(
        SubagentSpec(
            route="worker",
            execution_modes=(SubagentExecutionMode.background,),
            agent=AgentSpec(),
        )
    )
    store = InMemorySubagentExecutionStore()
    record = SubagentExecutionRecord(
        execution_id="child",
        root_execution_id="child",
        owner_scope_id="session",
        idempotency_key="child-once",
        descriptor_id=plan.descriptor_id,
        plan_fingerprint=plan.fingerprint,
        route="worker",
        mode=SubagentExecutionMode.background,
        state=SubagentExecutionState.succeeded,
        delivery_state=SubagentDeliveryState.pending,
        parent_agent_id="main",
        parent_logical_run_id="parent-run",
        prompt="work",
    )
    await store.create(record)

    class CompletionDelivery:
        def __init__(self) -> None:
            self.calls = 0

        async def deliver(
            self,
            current: SubagentExecutionRecord,
            parent_ctx: AgentContext,
            message: str,
        ) -> EnqueueReceipt | None:
            del parent_ctx, message
            self.calls += 1
            if current.delivery_input_id == "input-one":
                return EnqueueReceipt(
                    logical_run_id="run-one",
                    input_id="input-one",
                    disposition=InputDisposition.rejected,
                )
            if self.calls == 1:
                return EnqueueReceipt(
                    logical_run_id="run-one",
                    input_id="input-one",
                    disposition=InputDisposition.accepted,
                )
            return EnqueueReceipt(
                logical_run_id="run-two",
                input_id="input-two",
                disposition=InputDisposition.applied,
            )

    delivery = CompletionDelivery()
    service = SubagentExecutionService(
        SubagentRegistry([plan]),
        store,
        InProcessSubagentDriver(model_resolver=lambda _name: TestModel(call_tools=[])),
        completion_delivery=delivery,
    )
    parent = AgentContext(delegation_scope_id="session")

    assert await service.deliver_pending(parent) == 0
    accepted = await service.get("child", caller_scope_id="session")
    assert accepted.delivery_state is SubagentDeliveryState.pending
    assert accepted.delivery_input_id == "input-one"

    assert await service.deliver_pending(parent) == 1
    applied = await service.get("child", caller_scope_id="session")
    await service.close()

    assert applied.delivery_state is SubagentDeliveryState.delivered
    assert applied.delivery_logical_run_id == "run-two"
    assert applied.delivery_input_id == "input-two"
