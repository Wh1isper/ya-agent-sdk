"""store native AgentSpec profiles and remove the legacy profile compiler schema

Revision ID: 20260818_000016
Revises: 20260818_000015
Create Date: 2026-08-18 16:30:00.000000
"""

from __future__ import annotations

from typing import Any

import sqlalchemy as sa
from alembic import op
from pydantic_core import to_jsonable_python
from sqlalchemy.engine import Connection
from sqlalchemy.sql.elements import ColumnClause
from ya_agent_sdk.presets import resolve_model_cfg, resolve_model_settings

revision = "20260818_000016"
down_revision = "20260818_000015"
branch_labels = None
depends_on = None

_FEATURE_CAPABILITIES: dict[str, tuple[str, ...]] = {
    "content": ("MediaReadCapability",),
    "filesystem": ("FilesystemCapability",),
    "shell": ("ShellCapability",),
    "web": ("WebSearchCapability", "WebContentCapability"),
    "document": ("DocumentConversionCapability",),
}
_CORE_GROUPS = (
    "content",
    "filesystem",
    "shell",
    "web",
    "document",
    "session",
    "schedule",
    "workflow",
    "agency",
)
_HOST_GROUPS = frozenset({"session", "schedule", "workflow", "agency"})
_NEW_COLUMNS: dict[str, sa.JSON] = {
    "agent_spec": sa.JSON(),
    "host_config": sa.JSON(),
    "subagent_specs": sa.JSON(),
}
_OLD_COLUMNS = (
    "model",
    "model_settings_preset",
    "model_settings_override",
    "model_config_preset",
    "model_config_override",
    "system_prompt",
    "builtin_toolsets",
    "subagents",
    "include_builtin_subagents",
    "unified_subagents",
    "need_user_approve_tools",
    "need_user_approve_mcps",
    "enabled_mcps",
    "disabled_mcps",
    "mcp_servers",
    "workspace_backend_hint",
)


def upgrade() -> None:
    bind = op.get_bind()
    existing_columns = _profile_columns(bind)
    old_columns = set(_OLD_COLUMNS)
    if old_columns.isdisjoint(existing_columns):
        _validate_completed_native_schema(bind, existing_columns)
        return
    missing_old_columns = sorted(old_columns - existing_columns.keys())
    if missing_old_columns:
        raise RuntimeError(
            "Cannot resume native profile migration from an incomplete legacy schema; "
            f"missing columns: {', '.join(missing_old_columns)}"
        )
    for name, column_type in _NEW_COLUMNS.items():
        if name not in existing_columns:
            op.add_column("profiles", sa.Column(name, column_type, nullable=True))

    profiles = sa.table(
        "profiles",
        sa.column("name", sa.String()),
        *_legacy_profile_columns(),
        sa.column("agent_spec", sa.JSON()),
        sa.column("host_config", sa.JSON()),
        sa.column("subagent_specs", sa.JSON()),
    )
    rows = bind.execute(sa.select(profiles)).mappings().all()
    for row in rows:
        agent_spec, host_config, subagent_specs = _migrate_profile(dict(row))
        bind.execute(
            profiles
            .update()
            .where(profiles.c.name == row["name"])
            .values(
                agent_spec=_json_mapping(agent_spec),
                host_config=_json_mapping(host_config),
                subagent_specs=[_json_mapping(item) for item in subagent_specs],
            )
        )

    with op.batch_alter_table("profiles") as batch_op:
        batch_op.alter_column("agent_spec", existing_type=sa.JSON(), nullable=False)
        batch_op.alter_column("host_config", existing_type=sa.JSON(), nullable=False)
        batch_op.alter_column("subagent_specs", existing_type=sa.JSON(), nullable=False)
        for name in _OLD_COLUMNS:
            batch_op.drop_column(name)


def downgrade() -> None:
    with op.batch_alter_table("profiles") as batch_op:
        batch_op.add_column(sa.Column("model", sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column("model_settings_preset", sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column("model_settings_override", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("model_config_preset", sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column("model_config_override", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("system_prompt", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("builtin_toolsets", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("subagents", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("include_builtin_subagents", sa.Boolean(), nullable=True))
        batch_op.add_column(sa.Column("unified_subagents", sa.Boolean(), nullable=True))
        batch_op.add_column(sa.Column("need_user_approve_tools", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("need_user_approve_mcps", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("enabled_mcps", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("disabled_mcps", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("mcp_servers", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("workspace_backend_hint", sa.String(length=32), nullable=True))

    bind = op.get_bind()
    table = sa.table(
        "profiles",
        sa.column("name", sa.String()),
        sa.column("agent_spec", sa.JSON()),
        sa.column("host_config", sa.JSON()),
        sa.column("subagent_specs", sa.JSON()),
        *_legacy_profile_columns(),
    )
    for row in bind.execute(sa.select(table)).mappings().all():
        agent = dict(row["agent_spec"] or {})
        host = dict(row["host_config"] or {})
        bind.execute(
            table
            .update()
            .where(table.c.name == row["name"])
            .values(
                model=str(agent.get("model") or ""),
                model_settings_override=agent.get("model_settings"),
                model_config_preset=host.get("model_config_preset"),
                model_config_override=host.get("model_config_override"),
                system_prompt=agent.get("instructions"),
                builtin_toolsets=list(host.get("tool_groups") or []),
                subagents=[],
                include_builtin_subagents=False,
                unified_subagents=True,
                need_user_approve_tools=list(host.get("need_user_approve_tools") or []),
                need_user_approve_mcps=list(host.get("need_user_approve_mcps") or []),
                enabled_mcps=list(host.get("enabled_mcps") or []),
                disabled_mcps=list(host.get("disabled_mcps") or []),
                mcp_servers=dict(host.get("mcp_servers") or {}),
                workspace_backend_hint=host.get("workspace_backend_hint"),
            )
        )
    with op.batch_alter_table("profiles") as batch_op:
        batch_op.alter_column("model", existing_type=sa.String(length=255), nullable=False)
        for name in (
            "builtin_toolsets",
            "subagents",
            "include_builtin_subagents",
            "unified_subagents",
            "need_user_approve_tools",
            "need_user_approve_mcps",
            "enabled_mcps",
            "disabled_mcps",
            "mcp_servers",
        ):
            batch_op.alter_column(name, nullable=False)
        batch_op.drop_column("subagent_specs")
        batch_op.drop_column("host_config")
        batch_op.drop_column("agent_spec")


def _profile_columns(bind: Connection) -> dict[str, dict[str, Any]]:
    return {str(column["name"]): dict(column) for column in sa.inspect(bind).get_columns("profiles")}


def _validate_completed_native_schema(bind: Connection, columns: dict[str, dict[str, Any]]) -> None:
    missing = sorted(set(_NEW_COLUMNS) - columns.keys())
    if missing:
        raise RuntimeError(
            "Cannot resume native profile migration because the legacy profile columns "
            f"are gone and native columns are missing: {', '.join(missing)}"
        )
    nullable = sorted(name for name in _NEW_COLUMNS if columns[name].get("nullable") is not False)
    if nullable:
        raise RuntimeError(
            "Cannot resume native profile migration because the legacy profile columns "
            f"are gone and native columns are still nullable: {', '.join(nullable)}"
        )
    invalid_types = sorted(name for name in _NEW_COLUMNS if not isinstance(columns[name].get("type"), sa.JSON))
    if invalid_types:
        raise RuntimeError(
            "Cannot resume native profile migration because native columns do not use "
            f"the expected JSON type: {', '.join(invalid_types)}"
        )
    profiles = sa.table(
        "profiles",
        *(sa.column(name, sa.JSON()) for name in _NEW_COLUMNS),
    )
    contains_null = bind.execute(
        sa.select(sa.exists().where(sa.or_(*(profiles.c[name].is_(None) for name in _NEW_COLUMNS))))
    ).scalar_one()
    if contains_null:
        raise RuntimeError("Cannot resume native profile migration because native profile data contains NULL")


def _json_mapping(value: dict[str, Any]) -> dict[str, Any]:
    normalized = to_jsonable_python(value)
    if not isinstance(normalized, dict):
        raise TypeError("Expected profile migration output to be a JSON object")
    return normalized


def _legacy_profile_columns() -> tuple[ColumnClause[Any], ...]:
    return (
        sa.column("model", sa.String()),
        sa.column("model_settings_preset", sa.String()),
        sa.column("model_settings_override", sa.JSON()),
        sa.column("model_config_preset", sa.String()),
        sa.column("model_config_override", sa.JSON()),
        sa.column("system_prompt", sa.Text()),
        sa.column("builtin_toolsets", sa.JSON()),
        sa.column("subagents", sa.JSON()),
        sa.column("include_builtin_subagents", sa.Boolean()),
        sa.column("unified_subagents", sa.Boolean()),
        sa.column("need_user_approve_tools", sa.JSON()),
        sa.column("need_user_approve_mcps", sa.JSON()),
        sa.column("enabled_mcps", sa.JSON()),
        sa.column("disabled_mcps", sa.JSON()),
        sa.column("mcp_servers", sa.JSON()),
        sa.column("workspace_backend_hint", sa.String()),
    )


def _migrate_profile(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    profile_name = str(row.get("name") or "<unknown>")
    if row.get("include_builtin_subagents") is True:
        raise RuntimeError(
            f"Profile {profile_name!r} enables legacy builtin subagents. "
            "Before upgrading, materialize the selected builtin agents in the "
            "profile's explicit subagents list and set include_builtin_subagents=false."
        )
    if row.get("unified_subagents") is not True:
        raise RuntimeError(
            f"Profile {profile_name!r} uses legacy per-subagent generated tools. "
            "YA Claw 2.0 exposes one DelegationCapability, so explicitly convert the "
            "profile to unified_subagents=true before upgrading."
        )
    groups = _expand_groups(row.get("builtin_toolsets"))
    capabilities = _feature_capabilities(groups)
    model_settings = _merge(
        resolve_model_settings(_preset(row.get("model_settings_preset"))),
        _mapping(row.get("model_settings_override")),
    )
    agent_spec = {
        "model": str(row.get("model") or ""),
        "name": str(row.get("name") or ""),
        "instructions": row.get("system_prompt"),
        "model_settings": model_settings,
        "capabilities": capabilities,
        "metadata": {"claw_profile": str(row.get("name") or "")},
    }
    host_config = {
        "model_config_preset": _preset(row.get("model_config_preset")),
        "model_config_override": _mapping(row.get("model_config_override")),
        "tool_groups": [group for group in groups if group in _HOST_GROUPS],
        "need_user_approve_tools": list(row.get("need_user_approve_tools") or []),
        "need_user_approve_mcps": list(row.get("need_user_approve_mcps") or []),
        "enabled_mcps": list(row.get("enabled_mcps") or []),
        "disabled_mcps": list(row.get("disabled_mcps") or []),
        "mcp_servers": dict(row.get("mcp_servers") or {}),
        "workspace_backend_hint": row.get("workspace_backend_hint"),
    }
    subagent_specs = [
        _migrate_subagent(
            item,
            parent_agent=agent_spec,
            parent_host=host_config,
            feature_capabilities=capabilities,
        )
        for item in row.get("subagents") or []
        if isinstance(item, dict)
    ]
    return agent_spec, host_config, subagent_specs


def _migrate_subagent(
    raw: dict[str, Any],
    *,
    parent_agent: dict[str, Any],
    parent_host: dict[str, Any],
    feature_capabilities: list[str],
) -> dict[str, Any]:
    route = str(raw.get("name") or "").strip()
    model = raw.get("model")
    if not isinstance(model, str) or not model.strip() or model == "inherit":
        model = parent_agent["model"]
    settings_preset = _preset(raw.get("model_settings_preset"))
    settings = _merge(
        resolve_model_settings(settings_preset),
        _mapping(raw.get("model_settings_override")),
    )
    if settings is None:
        settings = parent_agent.get("model_settings")
    config_preset = _preset(raw.get("model_config_preset"))
    model_config = _merge(
        resolve_model_cfg(config_preset),
        _mapping(raw.get("model_config_override")),
    )
    if model_config is None:
        model_config = _merge(
            resolve_model_cfg(parent_host.get("model_config_preset")),
            _mapping(parent_host.get("model_config_override")),
        )
    capabilities: list[Any] = list(feature_capabilities)
    selected = list(dict.fromkeys([*(raw.get("tools") or []), *(raw.get("optional_tools") or [])]))
    if selected:
        capabilities.append({"ToolVisibilityCapability": {"allow": selected}})
    child_host = dict(parent_host)
    child_host["model_config_preset"] = None
    child_host["model_config_override"] = model_config
    return {
        "schema_version": 1,
        "route": route,
        "agent": {
            "model": model,
            "name": route,
            "description": raw.get("description") or route,
            "instructions": raw.get("system_prompt"),
            "model_settings": settings,
            "capabilities": capabilities,
            "metadata": {"claw": child_host},
        },
        "execution_modes": ["foreground", "background"],
        "durability": "restart",
    }


def _expand_groups(value: Any) -> tuple[str, ...]:
    source = value if isinstance(value, list) else ["core"]
    expanded: list[str] = []
    for item in source:
        name = str(item)
        values = _CORE_GROUPS if name == "core" else (name,)
        for group in values:
            if group != "background" and group not in expanded:
                expanded.append(group)
    return tuple(expanded)


def _feature_capabilities(groups: tuple[str, ...]) -> list[str]:
    result: list[str] = []
    for group in groups:
        for capability in _FEATURE_CAPABILITIES.get(group, ()):
            if capability not in result:
                result.append(capability)
    return result


def _preset(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() and value != "inherit" else None


def _mapping(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, dict) else None


def _merge(base: dict[str, Any] | None, override: dict[str, Any] | None) -> dict[str, Any] | None:
    if base is None and override is None:
        return None
    return {**(base or {}), **(override or {})}
