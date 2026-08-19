from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

import yaml
from loguru import logger
from pydantic import BaseModel, ConfigDict, model_validator
from pydantic_ai import AgentSpec
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from ya_agent_sdk.context import ShellReviewConfig, ShellReviewRiskLevel
from ya_agent_sdk.environment import ShellSandboxConfig
from ya_agent_sdk.presets import resolve_model_cfg
from ya_agent_sdk.subagents import ResolvedSubagentPlan, SubagentSpec

from ya_claw.config import ClawSettings
from ya_claw.mcp import normalize_profile_mcp_servers
from ya_claw.orm.tables import ProfileRecord
from ya_claw.profile_spec import (
    CLAW_PROFILE_SCHEMA_VERSION,
    ClawProfileHostConfig,
    ClawProfileSeedDefinition,
)

_DEFAULT_PROFILE_NAME = "default"
PROFILE_SNAPSHOT_METADATA_KEY = "execution_profile_snapshot"
_PROFILE_SNAPSHOT_SCHEMA_VERSION = 1


class ExecutionProfileDescriptor(BaseModel):
    """Content-addressed native profile captured when a run is accepted."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    descriptor_id: str
    name: str
    agent_spec: dict[str, Any]
    host_config: dict[str, Any]
    subagent_specs: tuple[dict[str, Any], ...] = ()
    source_type: str | None = None
    source_version: str | None = None
    source_checksum: str | None = None

    def behavior_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"descriptor_id"})

    @model_validator(mode="after")
    def _validate_fingerprint(self) -> ExecutionProfileDescriptor:
        fingerprint = _profile_descriptor_fingerprint(self.behavior_payload())
        if self.descriptor_id != f"sha256:{fingerprint}":
            raise ValueError("Execution profile descriptor content does not match its fingerprint")
        return self


class ClawShellReviewConfig(ShellReviewConfig):
    unattended_risk_threshold: ShellReviewRiskLevel | None = None


@dataclass(frozen=True, slots=True)
class ResolvedProfile:
    """One native agent definition plus Claw host policy."""

    name: str
    agent_spec: AgentSpec
    model_config: dict[str, Any] | None = None
    host_tool_groups: tuple[str, ...] = ()
    host_tool_allowlist: frozenset[str] | None = None
    subagent_specs: tuple[SubagentSpec, ...] = ()
    approval_tools: frozenset[str] = frozenset()
    approval_mcps: frozenset[str] = frozenset()
    shell_review: ClawShellReviewConfig | None = None
    shell_sandbox: ShellSandboxConfig | None = None
    enabled_mcps: frozenset[str] = frozenset()
    disabled_mcps: frozenset[str] = frozenset()
    mcp_servers: dict[str, Any] = field(default_factory=dict)
    workspace_backend_hint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def subagent(self, route: str) -> SubagentSpec:
        for spec in self.subagent_specs:
            if spec.route == route:
                return spec
        raise KeyError(route)


async def capture_execution_profile_descriptor(
    db_session: AsyncSession,
    profile_name: str,
) -> ExecutionProfileDescriptor:
    """Capture one enabled profile in the run-admission transaction."""
    record = await db_session.get(ProfileRecord, profile_name)
    if not isinstance(record, ProfileRecord) or not record.enabled:
        raise ValueError(f"Execution profile {profile_name!r} could not be resolved at run admission")
    core: dict[str, Any] = {
        "schema_version": _PROFILE_SNAPSHOT_SCHEMA_VERSION,
        "name": record.name,
        "agent_spec": dict(record.agent_spec),
        "host_config": dict(record.host_config),
        "subagent_specs": tuple(dict(item) for item in record.subagent_specs or []),
        "source_type": record.source_type,
        "source_version": record.source_version,
        "source_checksum": record.source_checksum,
    }
    fingerprint = _profile_descriptor_fingerprint(core)
    return ExecutionProfileDescriptor.model_validate({**core, "descriptor_id": f"sha256:{fingerprint}"})


def resolved_profile_from_descriptor(
    descriptor: ExecutionProfileDescriptor | dict[str, Any],
) -> ResolvedProfile:
    """Restore exact runtime behavior without reading the mutable profile catalog."""
    exact = (
        descriptor
        if isinstance(descriptor, ExecutionProfileDescriptor)
        else ExecutionProfileDescriptor.model_validate(descriptor)
    )
    agent_spec = AgentSpec.model_validate(exact.agent_spec)
    host = ClawProfileHostConfig.model_validate(exact.host_config)
    model_config = _resolve_model_config(host)
    return ResolvedProfile(
        name=exact.name,
        agent_spec=agent_spec,
        model_config=model_config,
        host_tool_groups=host.tool_groups,
        subagent_specs=tuple(SubagentSpec.model_validate(item) for item in exact.subagent_specs),
        approval_tools=frozenset(host.need_user_approve_tools),
        approval_mcps=frozenset(host.need_user_approve_mcps),
        shell_review=_resolve_shell_review(model_config),
        shell_sandbox=_resolve_shell_sandbox(model_config),
        enabled_mcps=frozenset(host.enabled_mcps),
        disabled_mcps=frozenset(host.disabled_mcps),
        mcp_servers=normalize_profile_mcp_servers(host.mcp_servers),
        workspace_backend_hint=host.workspace_backend_hint,
        metadata={
            "source_type": exact.source_type,
            "source_version": exact.source_version,
            "source_checksum": exact.source_checksum,
            "profile_descriptor_id": exact.descriptor_id,
        },
    )


def _profile_descriptor_fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def resolved_profile_from_subagent_plan(plan: ResolvedSubagentPlan) -> ResolvedProfile:
    """Build a child profile solely from its immutable resolved descriptor."""
    host = _host_from_agent_metadata(plan.normalized_agent_spec)
    model_config = _resolve_model_config(host)
    return ResolvedProfile(
        name=plan.spec.route,
        agent_spec=plan.normalized_agent_spec,
        model_config=model_config,
        host_tool_groups=host.tool_groups,
        subagent_specs=(),
        approval_tools=frozenset(host.need_user_approve_tools),
        approval_mcps=frozenset(host.need_user_approve_mcps),
        shell_review=_resolve_shell_review(model_config),
        shell_sandbox=_resolve_shell_sandbox(model_config),
        enabled_mcps=frozenset(host.enabled_mcps),
        disabled_mcps=frozenset(host.disabled_mcps),
        mcp_servers=normalize_profile_mcp_servers(host.mcp_servers),
        workspace_backend_hint=host.workspace_backend_hint,
        metadata={
            "subagent_route": plan.spec.route,
            "plan_descriptor_id": plan.descriptor_id,
            "plan_fingerprint": plan.fingerprint,
        },
    )


class ProfileResolver:
    def __init__(
        self,
        *,
        settings: ClawSettings,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._settings = settings
        self._session_factory = session_factory

    async def resolve(self, profile_name: str | None) -> ResolvedProfile:
        resolved_name = profile_name or self._settings.default_profile
        logger.debug(
            "Resolving execution profile requested={} resolved={}",
            profile_name,
            resolved_name,
        )
        if isinstance(resolved_name, str) and resolved_name.strip():
            record = await self._load_profile_record(resolved_name)
            if isinstance(record, ProfileRecord):
                logger.info(
                    "Execution profile resolved name={} source_type={}",
                    record.name,
                    record.source_type,
                )
                return self._resolved_from_record(record)
        profile_value = resolved_name or _DEFAULT_PROFILE_NAME
        raise ValueError(f"Execution profile '{profile_value}' could not be resolved.")

    async def seed_profiles(self, *, prune_missing: bool = False) -> list[str]:
        seed_file = self._settings.resolved_profile_seed_file
        if seed_file is None or not seed_file.exists():
            logger.debug("Profile seed skipped seed_file={}", seed_file)
            return []
        logger.info(
            "Seeding execution profiles seed_file={} prune_missing={}",
            seed_file,
            prune_missing,
        )
        seed_content = seed_file.read_text(encoding="utf-8")
        definitions = _load_seed_definitions(seed_content)
        source_checksum = hashlib.sha256(seed_content.encode("utf-8")).hexdigest()
        async with self._session_factory() as db_session:
            result = await db_session.execute(select(ProfileRecord))
            existing = {record.name: record for record in result.scalars().all()}
            seeded_names: list[str] = []
            for definition in definitions:
                seeded_names.append(definition.name)
                record = existing.get(definition.name)
                if record is None:
                    record = ProfileRecord(name=definition.name)
                    db_session.add(record)
                _apply_seed_definition(
                    record,
                    definition,
                    source_checksum=source_checksum,
                )
            if prune_missing:
                for name, record in existing.items():
                    if record.source_type == "seed" and name not in seeded_names:
                        await db_session.delete(record)
            await db_session.commit()
        logger.info(
            "Execution profiles seeded count={} names={}",
            len(seeded_names),
            seeded_names,
        )
        return seeded_names

    async def list_enabled_models(self) -> list[str]:
        async with self._session_factory() as db_session:
            result = await db_session.execute(select(ProfileRecord).where(ProfileRecord.enabled.is_(True)))
            models: list[str] = []
            for record in result.scalars().all():
                model = AgentSpec.model_validate(record.agent_spec).model
                if isinstance(model, str) and model.strip():
                    models.append(model)
            return models

    async def _load_profile_record(self, profile_name: str) -> ProfileRecord | None:
        async with self._session_factory() as db_session:
            record = await db_session.get(ProfileRecord, profile_name)
            if isinstance(record, ProfileRecord) and record.enabled:
                return record
        return None

    def _resolved_from_record(self, record: ProfileRecord) -> ResolvedProfile:
        agent_spec = AgentSpec.model_validate(record.agent_spec)
        host = ClawProfileHostConfig.model_validate(record.host_config)
        model_config = _resolve_model_config(host)
        subagent_specs = tuple(SubagentSpec.model_validate(item) for item in record.subagent_specs or [])
        return ResolvedProfile(
            name=record.name,
            agent_spec=agent_spec,
            model_config=model_config,
            host_tool_groups=host.tool_groups,
            subagent_specs=subagent_specs,
            approval_tools=frozenset(host.need_user_approve_tools),
            approval_mcps=frozenset(host.need_user_approve_mcps),
            shell_review=_resolve_shell_review(model_config),
            shell_sandbox=_resolve_shell_sandbox(model_config),
            enabled_mcps=frozenset(host.enabled_mcps),
            disabled_mcps=frozenset(host.disabled_mcps),
            mcp_servers=normalize_profile_mcp_servers(host.mcp_servers),
            workspace_backend_hint=host.workspace_backend_hint,
            metadata={
                "source_type": record.source_type,
                "source_version": record.source_version,
                "source_checksum": record.source_checksum,
            },
        )


def _host_from_agent_metadata(agent_spec: AgentSpec) -> ClawProfileHostConfig:
    metadata = agent_spec.metadata
    raw = metadata.get("claw") if isinstance(metadata, dict) else None
    if not isinstance(raw, dict):
        return ClawProfileHostConfig()
    return ClawProfileHostConfig.model_validate(raw)


def _resolve_model_config(host: ClawProfileHostConfig) -> dict[str, Any] | None:
    base = resolve_model_cfg(host.model_config_preset)
    override = host.model_config_override
    if base is None and override is None:
        return None
    merged: dict[str, Any] = {}
    if isinstance(base, dict):
        merged.update(base)
    if isinstance(override, dict):
        merged.update(override)
    return merged


def _resolve_shell_review(model_config: dict[str, Any] | None) -> ClawShellReviewConfig | None:
    if not isinstance(model_config, dict):
        return None
    security = model_config.get("security")
    if isinstance(security, dict) and isinstance(security.get("shell_review"), dict):
        return _resolve_claw_shell_review_config(security["shell_review"])
    return None


def _resolve_shell_sandbox(model_config: dict[str, Any] | None) -> ShellSandboxConfig | None:
    if not isinstance(model_config, dict):
        return None
    security = model_config.get("security")
    if isinstance(security, dict) and isinstance(security.get("shell_sandbox"), dict):
        return ShellSandboxConfig.model_validate(security["shell_sandbox"])
    return None


def _resolve_claw_shell_review_config(raw: dict[str, Any]) -> ClawShellReviewConfig:
    config = dict(raw)
    config.setdefault("risk_threshold", "extra_high")
    return ClawShellReviewConfig.model_validate(config)


def _load_seed_definitions(seed_content: str) -> tuple[ClawProfileSeedDefinition, ...]:
    payload = yaml.safe_load(seed_content)
    if not isinstance(payload, dict):
        raise TypeError("Profile seed must be a versioned mapping")
    if payload.get("version") != CLAW_PROFILE_SCHEMA_VERSION:
        raise ValueError(
            f"Profile seed version must be {CLAW_PROFILE_SCHEMA_VERSION}; legacy profile documents are not accepted"
        )
    rows = payload.get("profiles")
    if not isinstance(rows, list):
        raise TypeError("Profile seed 'profiles' must be a list")
    definitions = tuple(ClawProfileSeedDefinition.model_validate(row) for row in rows)
    names = [definition.name for definition in definitions]
    if len(names) != len(set(names)):
        raise ValueError("Profile seed names must be unique")
    return definitions


def _apply_seed_definition(
    record: ProfileRecord,
    definition: ClawProfileSeedDefinition,
    *,
    source_checksum: str,
) -> None:
    host = definition.host.model_copy(
        update={"mcp_servers": normalize_profile_mcp_servers(definition.host.mcp_servers)}
    )
    record.agent_spec = definition.agent.model_dump(mode="json", by_alias=True)
    record.host_config = host.model_dump(mode="json")
    record.subagent_specs = [spec.model_dump(mode="json", by_alias=True) for spec in definition.subagents]
    record.enabled = definition.enabled
    record.source_type = definition.source_type or "seed"
    record.source_version = definition.source_version
    record.source_checksum = definition.source_checksum or source_checksum
