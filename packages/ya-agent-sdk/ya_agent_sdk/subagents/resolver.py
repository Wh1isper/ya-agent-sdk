"""Resolution and fingerprinting for portable subagent plans."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import fields, is_dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic_ai import Agent, AgentSpec, TemplateStr
from pydantic_ai._spec import CapabilitySpec
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel

from ya_agent_sdk.capabilities import CapabilityCatalog
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.subagents.spec import (
    AgentTemplateContext,
    CustomCapabilityAudit,
    ResolvedSubagentPlan,
    SelfForkPolicy,
    SubagentDurability,
    SubagentHistoryPolicy,
    SubagentPlanDescriptor,
    SubagentSpec,
    clone_subagent_descriptor,
)

ModelPolicy = Callable[[str], bool]


def validate_resolved_subagent_plan_integrity(plan: ResolvedSubagentPlan) -> None:
    """Reject a resolved plan whose mutable values no longer match its identity."""
    policy_ids = tuple(_capability_identity(capability) for capability in plan.host_capabilities)
    if policy_ids != plan.injected_policy_ids:
        raise ValueError("Resolved subagent plan host capability grants were mutated")
    fingerprint = _fingerprint(
        plan.spec,
        plan.normalized_agent_spec,
        plan.template_context,
        plan.custom_capability_audit,
        plan.injected_policy_ids,
        effective_output_schema=plan.effective_output_schema,
        supports_deferred_output=plan.supports_deferred_output,
        restart_durable=plan.restart_durable,
        initial_history=plan.initial_history,
    )
    if fingerprint != plan.fingerprint:
        raise ValueError("Resolved subagent plan fingerprint is invalid")
    if plan.descriptor_id != f"{plan.spec.route}:{fingerprint}":
        raise ValueError("Resolved subagent plan identity is invalid")


class SubagentPlanResolver:
    """Validate native specs and produce immutable exact-grant child plans."""

    def __init__(
        self,
        catalog: CapabilityCatalog,
        *,
        available_host_requirements: Iterable[str] = (),
        host_capabilities: Sequence[AbstractCapability[Any]] = (),
        default_model: str | None = None,
        model_policy: ModelPolicy | None = None,
        restart_durable: bool = False,
    ) -> None:
        self.catalog = catalog
        self.available_host_requirements = frozenset(available_host_requirements)
        self.host_capabilities = tuple(copy.deepcopy(tuple(host_capabilities)))
        self.default_model = default_model
        self.model_policy = model_policy
        self.restart_durable = restart_durable

    def resolve(
        self,
        spec: SubagentSpec,
        *,
        template_context: AgentTemplateContext | Mapping[str, Any] | None = None,
    ) -> ResolvedSubagentPlan:
        """Resolve one named plan without inheriting any root feature grants."""
        spec = SubagentSpec.model_validate(spec.model_dump(mode="json", by_alias=True))
        context = _coerce_template_context(template_context)
        context = AgentTemplateContext.model_validate(context.model_dump(mode="json"))
        missing_requirements = sorted(set(spec.host_requirements) - self.available_host_requirements)
        if missing_requirements:
            raise ValueError(f"Subagent {spec.route!r} requires unavailable host features: {missing_requirements!r}")
        if spec.durability is SubagentDurability.restart and not self.restart_durable:
            raise ValueError(
                f"Subagent {spec.route!r} requires restart-durable execution, but the selected driver is process-local"
            )

        normalized = self._normalize_agent_spec(spec, context)
        model_name = normalized.model
        if model_name is None:  # pragma: no cover - guarded by normalization
            raise ValueError(f"Subagent {spec.route!r} has no model")
        if self.model_policy is not None and not self.model_policy(model_name):
            raise ValueError(f"Subagent {spec.route!r} model {model_name!r} is rejected by host policy")

        audit = self._custom_capability_audit(normalized)
        self._validate_native_plan(normalized)
        injected_policy_ids = tuple(_capability_identity(capability) for capability in self.host_capabilities)
        effective_output_schema = dict(normalized.output_schema) if normalized.output_schema is not None else None
        fingerprint = _fingerprint(
            spec,
            normalized,
            context,
            audit,
            injected_policy_ids,
            effective_output_schema=effective_output_schema,
            supports_deferred_output=True,
            restart_durable=self.restart_durable,
            initial_history=(),
        )
        return ResolvedSubagentPlan(
            descriptor_id=f"{spec.route}:{fingerprint}",
            fingerprint=fingerprint,
            spec=spec,
            normalized_agent_spec=normalized,
            template_context=context,
            custom_capability_audit=audit,
            injected_policy_ids=injected_policy_ids,
            host_capabilities=tuple(copy.deepcopy(self.host_capabilities)),
            effective_output_schema=effective_output_schema,
            supports_deferred_output=True,
            restart_durable=self.restart_durable,
        )

    def resolve_self(
        self,
        policy: SelfForkPolicy,
        *,
        history: Sequence[dict[str, Any]] = (),
        template_context: AgentTemplateContext | Mapping[str, Any] | None = None,
    ) -> ResolvedSubagentPlan:
        """Resolve ``self`` from catalog references and one bounded history snapshot."""
        spec = SubagentSpec(
            route="self",
            agent=policy.agent,
            history=SubagentHistoryPolicy.parent_snapshot,
            history_message_limit=policy.history_message_limit,
            max_depth=policy.max_depth,
            execution_modes=policy.execution_modes,
        )
        resolved = self.resolve(spec, template_context=template_context)
        bounded_history = tuple(
            json.loads(json.dumps(item, ensure_ascii=False, sort_keys=True))
            for item in history[-policy.history_message_limit :]
        )
        fingerprint = _fingerprint(
            resolved.spec,
            resolved.normalized_agent_spec,
            resolved.template_context,
            resolved.custom_capability_audit,
            resolved.injected_policy_ids,
            effective_output_schema=resolved.effective_output_schema,
            supports_deferred_output=resolved.supports_deferred_output,
            restart_durable=resolved.restart_durable,
            initial_history=bounded_history,
        )
        return replace(
            resolved,
            descriptor_id=f"{resolved.spec.route}:{fingerprint}",
            fingerprint=fingerprint,
            initial_history=bounded_history,
        )

    def restore(self, descriptor: SubagentPlanDescriptor) -> ResolvedSubagentPlan:
        """Validate and restore a portable descriptor without consulting mutable config."""
        descriptor = clone_subagent_descriptor(descriptor)
        expected_policy_ids = tuple(_capability_identity(capability) for capability in self.host_capabilities)
        if descriptor.injected_policy_ids != expected_policy_ids:
            raise ValueError("Subagent descriptor host capability grants do not match this driver")
        missing_requirements = sorted(set(descriptor.spec.host_requirements) - self.available_host_requirements)
        if missing_requirements:
            raise ValueError(
                f"Subagent {descriptor.spec.route!r} requires unavailable host features: {missing_requirements!r}"
            )
        if descriptor.restart_durable and not self.restart_durable:
            raise ValueError(
                f"Subagent {descriptor.spec.route!r} requires restart-durable execution, "
                "but the selected driver is process-local"
            )
        if descriptor.spec.route != descriptor.normalized_agent_spec.name:
            raise ValueError("Subagent descriptor route does not match normalized AgentSpec name")
        model_name = descriptor.normalized_agent_spec.model
        if model_name is None:
            raise ValueError(f"Subagent {descriptor.spec.route!r} has no model")
        if self.model_policy is not None and not self.model_policy(model_name):
            raise ValueError(f"Subagent {descriptor.spec.route!r} model {model_name!r} is rejected by host policy")
        audit = self._custom_capability_audit(descriptor.normalized_agent_spec)
        if audit != descriptor.custom_capability_audit:
            raise ValueError("Subagent descriptor capability provenance does not match the active catalog")
        self._validate_native_plan(descriptor.normalized_agent_spec)
        fingerprint = _fingerprint(
            descriptor.spec,
            descriptor.normalized_agent_spec,
            descriptor.template_context,
            descriptor.custom_capability_audit,
            descriptor.injected_policy_ids,
            effective_output_schema=descriptor.effective_output_schema,
            supports_deferred_output=descriptor.supports_deferred_output,
            restart_durable=descriptor.restart_durable,
            initial_history=descriptor.initial_history,
        )
        if descriptor.fingerprint != fingerprint:
            raise ValueError("Subagent descriptor fingerprint is invalid")
        expected_descriptor_id = f"{descriptor.spec.route}:{fingerprint}"
        if descriptor.descriptor_id != expected_descriptor_id:
            raise ValueError("Subagent descriptor identity is invalid")
        return ResolvedSubagentPlan(
            descriptor_id=descriptor.descriptor_id,
            fingerprint=descriptor.fingerprint,
            spec=descriptor.spec,
            normalized_agent_spec=descriptor.normalized_agent_spec,
            template_context=descriptor.template_context,
            custom_capability_audit=descriptor.custom_capability_audit,
            injected_policy_ids=descriptor.injected_policy_ids,
            host_capabilities=tuple(copy.deepcopy(self.host_capabilities)),
            effective_output_schema=descriptor.effective_output_schema,
            supports_deferred_output=descriptor.supports_deferred_output,
            restart_durable=descriptor.restart_durable,
            initial_history=descriptor.initial_history,
        )

    def _normalize_agent_spec(
        self,
        spec: SubagentSpec,
        template_context: AgentTemplateContext,
    ) -> AgentSpec:
        raw = spec.agent.model_dump(
            mode="json",
            by_alias=True,
            exclude_defaults=True,
        )
        raw["name"] = spec.route
        if raw.get("model") is None:
            if self.default_model is None:
                raise ValueError(f"Subagent {spec.route!r} must define a model or use a resolver default")
            raw["model"] = self.default_model
        for field_name in ("description", "instructions"):
            if field_name in raw:
                raw[field_name] = _render_template_field(
                    raw[field_name],
                    template_context,
                )
        capabilities = list(raw.get("capabilities") or [])
        if not any(
            isinstance(capability, Mapping) and capability.get("name") == "ToolVisibilityCapability"
            for capability in capabilities
        ):
            capabilities.append({"name": "ToolVisibilityCapability", "arguments": {}})
        raw["capabilities"] = capabilities
        return AgentSpec.model_validate(raw)

    def _custom_capability_audit(
        self,
        spec: AgentSpec,
    ) -> tuple[CustomCapabilityAudit, ...]:
        used_names = {
            capability_spec.name
            for capability_spec in _walk_capability_specs(spec.capabilities)
            if capability_spec.name in self.catalog
        }
        return tuple(
            CustomCapabilityAudit(
                serialization_name=name,
                provenance=self.catalog.provenance(name),
            )
            for name in sorted(used_names)
        )

    def _validate_native_plan(self, spec: AgentSpec) -> None:
        """Use the public native constructor as the ordering/dependency validator."""
        try:
            Agent.from_spec(
                spec,
                deps_type=AgentContext,
                custom_capability_types=self.catalog.custom_capability_types,
                model=TestModel(call_tools=[]),
                capabilities=self.host_capabilities,
            )
        except UserError as exc:
            raise ValueError(f"Invalid subagent capability plan: {exc}") from exc


def _coerce_template_context(
    value: AgentTemplateContext | Mapping[str, Any] | None,
) -> AgentTemplateContext:
    if value is None:
        return AgentTemplateContext()
    if isinstance(value, AgentTemplateContext):
        return value
    return AgentTemplateContext(template=dict(value))


def _render_template_field(value: Any, context: AgentTemplateContext) -> Any:
    """Render only native AgentSpec fields that explicitly support TemplateStr."""
    if isinstance(value, str) and "{{" in value:
        template = TemplateStr(value, deps_type=AgentTemplateContext)
        return template.render(context)
    if isinstance(value, list):
        return [_render_template_field(item, context) for item in value]
    return value


def _walk_capability_specs(values: Sequence[CapabilitySpec]) -> Iterable[CapabilitySpec]:
    for value in values:
        yield value
        yield from _walk_nested_capability_specs(value.arguments)


def _walk_nested_capability_specs(value: Any) -> Iterable[CapabilitySpec]:
    if isinstance(value, CapabilitySpec):
        yield value
        yield from _walk_nested_capability_specs(value.arguments)
    elif isinstance(value, Mapping):
        for nested in value.values():
            yield from _walk_nested_capability_specs(nested)
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for nested in value:
            yield from _walk_nested_capability_specs(nested)


def _capability_identity(capability: AbstractCapability[Any]) -> str:
    serialization_name = type(capability).get_serialization_name()
    type_name = serialization_name or (f"{type(capability).__module__}.{type(capability).__qualname__}")
    config = _canonical_capability_value(capability)
    encoded = json.dumps(
        config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    config_hash = hashlib.sha256(encoded).hexdigest()
    capability_id = f":{capability.id}" if capability.id is not None else ""
    return f"{type_name}{capability_id}:{config_hash}"


def _canonical_capability_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_capability_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_canonical_capability_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_canonical_capability_value(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True))
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonical_capability_value(getattr(value, field.name))
            for field in fields(value)
            if field.init and not field.name.startswith("_")
        }
    raise TypeError(
        f"Host-injected capability configuration is not portable: {type(value).__module__}.{type(value).__qualname__}"
    )


def _fingerprint(
    spec: SubagentSpec,
    normalized_agent_spec: AgentSpec,
    template_context: AgentTemplateContext,
    custom_capability_audit: tuple[CustomCapabilityAudit, ...],
    injected_policy_ids: tuple[str, ...],
    *,
    effective_output_schema: dict[str, Any] | None,
    supports_deferred_output: bool,
    restart_durable: bool,
    initial_history: tuple[dict[str, Any], ...],
) -> str:
    payload = {
        "schema_version": spec.schema_version,
        "route": spec.route,
        "source_agent": spec.agent.model_dump(
            mode="json",
            by_alias=True,
            exclude_defaults=True,
        ),
        "agent": normalized_agent_spec.model_dump(
            mode="json",
            by_alias=True,
            exclude_defaults=True,
        ),
        "history": spec.history.value,
        "history_message_limit": spec.history_message_limit,
        "host_requirements": list(spec.host_requirements),
        "max_depth": spec.max_depth,
        "spawn_targets": list(spec.spawn_targets),
        "execution_modes": [mode.value for mode in spec.execution_modes],
        "linkage": spec.linkage.value,
        "durability": spec.durability.value,
        "template": template_context.model_dump(mode="json"),
        "custom_capability_audit": [item.model_dump(mode="json") for item in custom_capability_audit],
        "injected_policy_ids": list(injected_policy_ids),
        "effective_output_schema": effective_output_schema,
        "supports_deferred_output": supports_deferred_output,
        "restart_durable": restart_durable,
        "initial_history": list(initial_history),
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()
