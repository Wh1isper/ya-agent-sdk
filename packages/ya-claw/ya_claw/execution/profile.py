from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import yaml
from loguru import logger
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from ya_agent_sdk.context import ShellReviewConfig, ShellReviewRiskLevel
from ya_agent_sdk.presets import INHERIT, resolve_model_cfg, resolve_model_settings
from ya_agent_sdk.subagents.config import SubagentConfig

from ya_claw.config import ClawSettings
from ya_claw.mcp import normalize_profile_mcp_servers
from ya_claw.orm.tables import ProfileRecord

_DEFAULT_BUILTIN_TOOLSETS = ["core"]
_DEFAULT_PROFILE_NAME = "default"


class ClawShellReviewConfig(ShellReviewConfig):
    unattended_risk_threshold: ShellReviewRiskLevel | None = None


class InlineSubagentDefinition(BaseModel):
    name: str
    description: str
    system_prompt: str
    tools: list[str] | None = None
    optional_tools: list[str] | None = None
    model: str | None = None
    model_settings_preset: str | None = None
    model_settings_override: dict[str, Any] | None = None
    model_config_preset: str | None = None
    model_config_override: dict[str, Any] | None = None


@dataclass(slots=True)
class ResolvedProfile:
    name: str
    model: str
    model_settings: dict[str, Any] | None
    model_config: dict[str, Any] | None
    system_prompt: str | None = None
    builtin_toolsets: list[str] = field(default_factory=lambda: list(_DEFAULT_BUILTIN_TOOLSETS))
    builtin_tool_allowlist: list[str] | None = None
    subagent_configs: list[SubagentConfig] = field(default_factory=list)
    include_builtin_subagents: bool = False
    unified_subagents: bool = False
    need_user_approve_tools: list[str] = field(default_factory=list)
    need_user_approve_mcps: list[str] = field(default_factory=list)
    shell_review: ClawShellReviewConfig | None = None
    enabled_mcps: list[str] = field(default_factory=list)
    disabled_mcps: list[str] = field(default_factory=list)
    mcp_servers: dict[str, Any] = field(default_factory=dict)
    workspace_backend_hint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


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
        logger.debug("Resolving execution profile requested={} resolved={}", profile_name, resolved_name)
        if isinstance(resolved_name, str) and resolved_name.strip() != "":
            record = await self._load_profile_record(resolved_name)
            if isinstance(record, ProfileRecord):
                logger.info("Execution profile resolved name={} source_type={}", record.name, record.source_type)
                return self._resolved_from_record(record)
        logger.warning("Execution profile missing requested={}", resolved_name)
        return self._bootstrap_profile(requested_name=resolved_name)

    async def seed_profiles(self, *, prune_missing: bool = False) -> list[str]:
        seed_file = self._settings.resolved_profile_seed_file
        if seed_file is None or not seed_file.exists():
            logger.debug("Profile seed skipped seed_file={}", seed_file)
            return []
        logger.info("Seeding execution profiles seed_file={} prune_missing={}", seed_file, prune_missing)
        seed_content = seed_file.read_text(encoding="utf-8")
        rows = _load_seed_rows(seed_content)
        source_checksum = hashlib.sha256(seed_content.encode("utf-8")).hexdigest()
        async with self._session_factory() as db_session:
            existing_result = await db_session.execute(select(ProfileRecord))
            existing_records = {record.name: record for record in existing_result.scalars().all()}
            seeded_names: list[str] = []
            for row in rows:
                name = str(row.get("name", "")).strip()
                if name == "":
                    continue
                seeded_names.append(name)
                record = existing_records.get(name)
                if record is None:
                    record = ProfileRecord(name=name)
                    db_session.add(record)
                _apply_seed_row(record, row, source_checksum=source_checksum)
            if prune_missing:
                for name, record in existing_records.items():
                    if record.source_type == "seed" and name not in seeded_names:
                        await db_session.delete(record)
            await db_session.commit()
        logger.info("Execution profiles seeded count={} names={}", len(seeded_names), seeded_names)
        return seeded_names

    async def list_enabled_models(self) -> list[str]:
        async with self._session_factory() as db_session:
            result = await db_session.execute(select(ProfileRecord.model).where(ProfileRecord.enabled.is_(True)))
            return [model for model in result.scalars().all() if isinstance(model, str) and model.strip()]

    async def _load_profile_record(self, profile_name: str) -> ProfileRecord | None:
        async with self._session_factory() as db_session:
            record = await db_session.get(ProfileRecord, profile_name)
            if isinstance(record, ProfileRecord) and record.enabled:
                return record
        return None

    def _resolved_from_record(self, record: ProfileRecord) -> ResolvedProfile:
        return ResolvedProfile(
            name=record.name,
            model=record.model,
            model_settings=_merge_dicts(
                resolve_model_settings(record.model_settings_preset),
                record.model_settings_override,
            ),
            model_config=_merge_dicts(
                resolve_model_cfg(record.model_config_preset),
                record.model_config_override,
            ),
            system_prompt=record.system_prompt,
            builtin_toolsets=_resolve_builtin_toolsets(record.builtin_toolsets),
            subagent_configs=self._resolve_subagent_configs(record.subagents),
            include_builtin_subagents=bool(record.include_builtin_subagents),
            unified_subagents=bool(record.unified_subagents),
            need_user_approve_tools=list(record.need_user_approve_tools or []),
            need_user_approve_mcps=list(record.need_user_approve_mcps or []),
            shell_review=_resolve_shell_review(record.model_config_override),
            enabled_mcps=list(record.enabled_mcps or []),
            disabled_mcps=list(record.disabled_mcps or []),
            mcp_servers=normalize_profile_mcp_servers(record.mcp_servers),
            workspace_backend_hint=record.workspace_backend_hint,
            metadata={
                "source_type": record.source_type,
                "source_version": record.source_version,
                "source_checksum": record.source_checksum,
            },
        )

    def _bootstrap_profile(self, *, requested_name: str | None) -> ResolvedProfile:
        profile_value = requested_name or self._settings.default_profile or _DEFAULT_PROFILE_NAME
        raise ValueError(f"Execution profile '{profile_value}' could not be resolved.")

    def _resolve_subagent_configs(self, raw_subagents: list[dict[str, Any]] | None) -> list[SubagentConfig]:
        resolved_configs: list[SubagentConfig] = []
        for raw_subagent in raw_subagents or []:
            inline = InlineSubagentDefinition.model_validate(raw_subagent)
            resolved_configs.append(
                SubagentConfig(
                    name=inline.name,
                    description=inline.description,
                    system_prompt=inline.system_prompt,
                    model=inline.model,
                    model_settings=_merge_dicts(
                        _resolve_inheritable_model_settings(inline.model_settings_preset),
                        inline.model_settings_override,
                    ),
                    model_cfg=_merge_dicts(
                        resolve_model_cfg(inline.model_config_preset),
                        inline.model_config_override,
                    ),
                    tools=inline.tools,
                    optional_tools=inline.optional_tools,
                )
            )
        return resolved_configs


def _resolve_shell_review(model_config_override: dict[str, Any] | None) -> ClawShellReviewConfig | None:
    if not isinstance(model_config_override, dict):
        return None
    raw_security = model_config_override.get("security")
    if isinstance(raw_security, dict) and isinstance(raw_security.get("shell_review"), dict):
        return _resolve_claw_shell_review_config(raw_security["shell_review"])
    raw_legacy = model_config_override.get("shell_review")
    if isinstance(raw_legacy, dict):
        return _resolve_claw_shell_review_config(raw_legacy)
    return None


def _resolve_claw_shell_review_config(raw: dict[str, Any]) -> ClawShellReviewConfig:
    config = dict(raw)
    config.setdefault("risk_threshold", "extra_high")
    return ClawShellReviewConfig.model_validate(config)


def _resolve_inheritable_model_settings(preset_or_dict: str | dict[str, Any] | None) -> dict[str, Any] | None:
    if preset_or_dict is None:
        return None
    if preset_or_dict == INHERIT:
        return None
    return resolve_model_settings(preset_or_dict)


def _merge_dicts(base: dict[str, Any] | None, override: dict[str, Any] | None) -> dict[str, Any] | None:
    if base is None and override is None:
        return None
    merged: dict[str, Any] = {}
    if isinstance(base, dict):
        merged.update(base)
    if isinstance(override, dict):
        merged.update(override)
    return merged


def _load_seed_rows(seed_content: str) -> list[dict[str, Any]]:
    payload = yaml.safe_load(seed_content)
    if payload is None:
        return []
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        candidate = payload.get("profiles")
        rows = candidate if isinstance(candidate, list) else []
    else:
        rows = []
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            normalized_rows.append(dict(row))
    return normalized_rows


def _apply_seed_row(record: ProfileRecord, row: dict[str, Any], *, source_checksum: str | None) -> None:
    record.model = str(row.get("model", record.model or "")).strip()
    record.model_settings_preset = _normalize_optional_str(row.get("model_settings_preset"))
    record.model_settings_override = _normalize_optional_dict(row.get("model_settings_override"))
    record.model_config_preset = _normalize_optional_str(row.get("model_config_preset"))
    record.model_config_override = _resolve_seed_model_config_override(row)
    record.system_prompt = _normalize_optional_str(row.get("system_prompt"))
    record.builtin_toolsets = _resolve_seed_builtin_toolsets(row)
    record.subagents = _normalize_optional_dict_list(row.get("subagents")) or []
    record.include_builtin_subagents = bool(row.get("include_builtin_subagents", False))
    record.unified_subagents = bool(row.get("unified_subagents", False))
    record.need_user_approve_tools = _normalize_optional_str_list(row.get("need_user_approve_tools")) or []
    record.need_user_approve_mcps = _normalize_optional_str_list(row.get("need_user_approve_mcps")) or []
    record.enabled_mcps = _normalize_optional_str_list(row.get("enabled_mcps")) or []
    record.disabled_mcps = _normalize_optional_str_list(row.get("disabled_mcps")) or []
    record.mcp_servers = normalize_profile_mcp_servers(_normalize_optional_dict(row.get("mcp_servers")))
    record.workspace_backend_hint = _normalize_optional_str(row.get("workspace_backend_hint"))
    record.enabled = bool(row.get("enabled", True))
    record.source_type = _normalize_optional_str(row.get("source_type")) or "seed"
    record.source_version = _normalize_optional_str(row.get("source_version"))
    record.source_checksum = _normalize_optional_str(row.get("source_checksum")) or source_checksum


def _resolve_builtin_toolsets(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip() != ""]
    return list(_DEFAULT_BUILTIN_TOOLSETS)


def _resolve_seed_model_config_override(row: dict[str, Any]) -> dict[str, Any] | None:
    override = _normalize_optional_dict(row.get("model_config_override")) or {}
    security = _normalize_optional_dict(row.get("security"))
    if security is not None:
        merged_security = dict(override.get("security") or {}) if isinstance(override.get("security"), dict) else {}
        merged_security.update(security)
        override["security"] = merged_security
    return override or None


def _resolve_seed_builtin_toolsets(row: dict[str, Any]) -> list[str]:
    if "builtin_toolsets" in row:
        return _normalize_optional_str_list(row.get("builtin_toolsets"), allow_empty=True) or []
    if "toolsets" in row:
        return _normalize_optional_str_list(row.get("toolsets"), allow_empty=True) or []
    return list(_DEFAULT_BUILTIN_TOOLSETS)


def _normalize_optional_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _normalize_optional_dict(value: object) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return dict(value)
    return None


def _normalize_optional_str_list(value: object, *, allow_empty: bool = False) -> list[str] | None:
    if not isinstance(value, list):
        return None
    values = [str(item).strip() for item in value if str(item).strip() != ""]
    if values or allow_empty:
        return values
    return None


def _normalize_optional_dict_list(value: object) -> list[dict[str, Any]] | None:
    if not isinstance(value, list):
        return None
    rows = [dict(item) for item in value if isinstance(item, dict)]
    return rows or None
