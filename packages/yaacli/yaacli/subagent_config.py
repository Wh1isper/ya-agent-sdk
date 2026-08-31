"""YAACLI subagent configuration loading and normalization."""

from __future__ import annotations

import copy
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, field_validator
from pydantic_ai import AgentSpec
from ya_agent_sdk.context import ModelConfig
from ya_agent_sdk.presets import resolve_model_settings
from ya_agent_sdk.subagents import SubagentExecutionMode, SubagentSpec

from yaacli.model_profiles import resolve_profile_model_cfg

YAACLI_MODEL_CFG_METADATA_KEY = "yaacli_model_cfg"
YAACLI_INHERIT_MODEL_SETTINGS_METADATA_KEY = "yaacli_inherit_model_settings"
YAACLI_INHERIT_MODEL_CFG_METADATA_KEY = "yaacli_inherit_model_cfg"
_SUBAGENT_SPEC_SUFFIXES = frozenset({".yaml", ".yml", ".json"})
_SUBAGENT_CONFIG_SUFFIXES = frozenset({*_SUBAGENT_SPEC_SUFFIXES, ".md"})
_MARKDOWN_FRONTMATTER_PATTERN = re.compile(
    r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)(.*)\Z",
    re.DOTALL,
)


class _MarkdownSubagentFrontmatter(BaseModel):
    """Supported generic Markdown subagent frontmatter."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    instruction: str | None = None
    tools: list[str] | None = None
    optional_tools: list[str] | None = None
    model: str | None = None
    model_settings: str | dict[str, Any] | None = None
    model_cfg: str | dict[str, Any] | None = None

    @field_validator("name", "description")
    @classmethod
    def _reject_empty_required_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value cannot be empty")
        return normalized

    @field_validator("instruction")
    @classmethod
    def _normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("tools", "optional_tools", mode="before")
    @classmethod
    def _normalize_tool_names(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.split(",")
        if not isinstance(value, list):
            return value
        normalized: list[str] = []
        for item in value:
            if not isinstance(item, str):
                return value
            name = item.strip()
            if not name:
                raise ValueError("tool names cannot be empty")
            if name not in normalized:
                normalized.append(name)
        return normalized


def load_subagent_specs(
    directory: Path,
    *,
    markdown_capabilities: Sequence[dict[str, Any]],
) -> dict[str, SubagentSpec]:
    """Normalize native documents and generic Markdown definitions into portable specs."""
    specs: dict[str, SubagentSpec] = {}
    sources: dict[str, Path] = {}
    if not directory.is_dir():
        return specs

    markdown_paths = sorted(directory.glob("*.md"))
    markdown_stems = {path.stem for path in markdown_paths}
    native_by_stem: dict[str, Path] = {}
    native_paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in _SUBAGENT_SPEC_SUFFIXES and path.stem not in markdown_stems
    )
    for path in native_paths:
        previous_path = native_by_stem.get(path.stem)
        if previous_path is not None:
            raise ValueError(
                f"Duplicate subagent basename {path.stem!r} in {previous_path.name} and {path.name} under {directory}"
            )
        spec = _load_native_subagent_spec(path)
        _add_subagent_spec(specs, sources, spec, path, directory=directory)
        native_by_stem[path.stem] = path

    for path in markdown_paths:
        spec = _load_markdown_subagent_spec(path, capabilities=markdown_capabilities)
        _add_subagent_spec(specs, sources, spec, path, directory=directory)
    return specs


def has_subagent_definition(directory: Path, stem: str) -> bool:
    """Return whether any supported configuration format defines this basename."""
    return any((directory / f"{stem}{suffix}").is_file() for suffix in _SUBAGENT_CONFIG_SUFFIXES)


def model_cfg_from_agent_spec(spec: AgentSpec) -> ModelConfig | None:
    """Read an optional YAACLI child context configuration from native metadata."""
    value = (spec.metadata or {}).get(YAACLI_MODEL_CFG_METADATA_KEY)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError(f"AgentSpec metadata {YAACLI_MODEL_CFG_METADATA_KEY!r} must be a mapping")
    return resolve_profile_model_cfg(value)


def materialize_subagent_model_configuration(
    spec: SubagentSpec,
    *,
    inherited_model_settings: dict[str, Any] | None,
    inherited_model_cfg: ModelConfig,
) -> SubagentSpec:
    """Resolve Markdown inheritance markers into one exact portable child spec."""
    metadata = dict(spec.agent.metadata or {})
    settings_marker = metadata.pop(YAACLI_INHERIT_MODEL_SETTINGS_METADATA_KEY, None)
    model_cfg_marker = metadata.pop(YAACLI_INHERIT_MODEL_CFG_METADATA_KEY, None)
    if settings_marker is None and model_cfg_marker is None:
        return spec
    if settings_marker is not None and settings_marker is not True:
        raise ValueError(f"AgentSpec metadata {YAACLI_INHERIT_MODEL_SETTINGS_METADATA_KEY!r} must be true")
    if model_cfg_marker is not None and model_cfg_marker is not True:
        raise ValueError(f"AgentSpec metadata {YAACLI_INHERIT_MODEL_CFG_METADATA_KEY!r} must be true")

    agent_payload = spec.agent.model_dump(mode="python", by_alias=True)
    if settings_marker is True:
        agent_payload["model_settings"] = (
            dict(inherited_model_settings) if inherited_model_settings is not None else None
        )
    if model_cfg_marker is True:
        metadata[YAACLI_MODEL_CFG_METADATA_KEY] = inherited_model_cfg.model_dump(mode="json")
    agent_payload["metadata"] = metadata or None
    return spec.model_copy(update={"agent": AgentSpec.model_validate(agent_payload)})


def _load_native_subagent_spec(path: Path) -> SubagentSpec:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("document root must be a mapping")
        return SubagentSpec.model_validate(payload)
    except Exception as exc:
        raise ValueError(f"Invalid subagent spec {path}: {exc}") from exc


def _load_markdown_subagent_spec(
    path: Path,
    *,
    capabilities: Sequence[dict[str, Any]],
) -> SubagentSpec:
    try:
        frontmatter, body = _parse_markdown_subagent(path.read_text(encoding="utf-8"))
        return _markdown_to_subagent_spec(frontmatter, body, capabilities=capabilities)
    except Exception as exc:
        raise ValueError(f"Invalid Markdown subagent {path}: {exc}") from exc


def _parse_markdown_subagent(content: str) -> tuple[_MarkdownSubagentFrontmatter, str]:
    match = _MARKDOWN_FRONTMATTER_PATTERN.match(content.strip())
    if match is None:
        raise ValueError("expected YAML frontmatter delimited by '---'")
    payload = yaml.safe_load(match.group(1))
    if not isinstance(payload, dict):
        raise TypeError("YAML frontmatter must be a mapping")
    return _MarkdownSubagentFrontmatter.model_validate(payload), match.group(2).strip()


def _markdown_to_subagent_spec(
    frontmatter: _MarkdownSubagentFrontmatter,
    body: str,
    *,
    capabilities: Sequence[dict[str, Any]],
) -> SubagentSpec:
    materialized_capabilities = copy.deepcopy(list(capabilities))
    visible_tools = _visible_markdown_tools(frontmatter)
    if visible_tools is not None:
        materialized_capabilities = _narrow_capability_visibility(materialized_capabilities, visible_tools)

    metadata: dict[str, Any] = {}
    inherit_model_cfg = frontmatter.model_cfg in (None, "inherit")
    if inherit_model_cfg:
        metadata[YAACLI_INHERIT_MODEL_CFG_METADATA_KEY] = True
    else:
        metadata[YAACLI_MODEL_CFG_METADATA_KEY] = resolve_profile_model_cfg(frontmatter.model_cfg).model_dump(
            mode="json"
        )

    model = None if frontmatter.model in (None, "inherit") else frontmatter.model
    inherit_model_settings = frontmatter.model_settings in (None, "inherit")
    model_settings_input = None if inherit_model_settings else frontmatter.model_settings
    if inherit_model_settings:
        metadata[YAACLI_INHERIT_MODEL_SETTINGS_METADATA_KEY] = True
    roster_description = "\n\n".join(
        part for part in (frontmatter.description, frontmatter.instruction) if part is not None and part.strip()
    )
    agent_spec = AgentSpec.from_dict({
        "name": frontmatter.name,
        "description": roster_description,
        "instructions": body or None,
        "model": model,
        "model_settings": resolve_model_settings(model_settings_input),
        "metadata": metadata or None,
        "capabilities": materialized_capabilities,
    })
    return SubagentSpec(
        route=frontmatter.name,
        agent=agent_spec,
        execution_modes=(
            SubagentExecutionMode.foreground,
            SubagentExecutionMode.background,
        ),
    )


def _visible_markdown_tools(frontmatter: _MarkdownSubagentFrontmatter) -> list[str] | None:
    if frontmatter.tools is None and frontmatter.optional_tools is None:
        return None
    return list(dict.fromkeys([*(frontmatter.tools or []), *(frontmatter.optional_tools or [])]))


def _narrow_capability_visibility(
    capabilities: list[dict[str, Any]],
    visible_tools: list[str],
) -> list[dict[str, Any]]:
    narrowed: list[dict[str, Any]] = []
    found_visibility = False
    for capability in capabilities:
        if _capability_name(capability) != "ToolVisibilityCapability":
            narrowed.append(capability)
            continue
        found_visibility = True
        visibility = copy.deepcopy(capability)
        arguments = visibility.get("arguments")
        if not isinstance(arguments, dict):
            raise TypeError("ToolVisibilityCapability arguments must be a mapping")
        baseline_allow = arguments.get("allow")
        if baseline_allow is None:
            effective_allow = list(visible_tools)
        else:
            if not isinstance(baseline_allow, list) or not all(isinstance(item, str) for item in baseline_allow):
                raise TypeError("ToolVisibilityCapability allow must be a list of tool names")
            baseline_names = set(baseline_allow)
            effective_allow = [name for name in visible_tools if name in baseline_names]
        arguments["allow"] = effective_allow
        narrowed.append(visibility)
    if not found_visibility:
        narrowed.append({
            "name": "ToolVisibilityCapability",
            "arguments": {"allow": list(visible_tools)},
        })
    return narrowed


def _capability_name(value: object) -> str | None:
    if not isinstance(value, dict):
        return None
    name = value.get("name")
    if isinstance(name, str):
        return name
    if len(value) == 1:
        candidate = next(iter(value))
        return candidate if isinstance(candidate, str) else None
    return None


def _add_subagent_spec(
    specs: dict[str, SubagentSpec],
    sources: dict[str, Path],
    spec: SubagentSpec,
    path: Path,
    *,
    directory: Path,
) -> None:
    if spec.route in specs:
        raise ValueError(
            f"Duplicate subagent route {spec.route!r} in {sources[spec.route].name} and {path.name} under {directory}"
        )
    specs[spec.route] = spec
    sources[spec.route] = path
