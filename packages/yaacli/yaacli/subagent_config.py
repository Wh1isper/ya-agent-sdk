"""Strict loading for native portable YAACLI subagent specs."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic_ai import AgentSpec
from ya_agent_sdk.context import ModelConfig
from ya_agent_sdk.subagents import SubagentSpec

YAACLI_MODEL_CFG_METADATA_KEY = "yaacli_model_cfg"
_SUBAGENT_SPEC_SUFFIXES = frozenset({".yaml", ".yml", ".json"})


def load_subagent_specs(directory: Path) -> dict[str, SubagentSpec]:
    """Load versioned ``SubagentSpec`` documents and reject duplicate routes."""
    specs: dict[str, SubagentSpec] = {}
    if not directory.is_dir():
        return specs
    legacy_paths = sorted(directory.glob("*.md"))
    if legacy_paths:
        names = ", ".join(path.name for path in legacy_paths)
        raise ValueError(
            "Legacy Markdown subagent configuration is unsupported; replace it with "
            f"versioned SubagentSpec YAML or JSON: {names}"
        )
    paths = sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in _SUBAGENT_SPEC_SUFFIXES
    )
    for path in paths:
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise TypeError("document root must be a mapping")
            spec = SubagentSpec.model_validate(payload)
        except Exception as exc:
            raise ValueError(f"Invalid subagent spec {path}: {exc}") from exc
        if spec.route in specs:
            raise ValueError(f"Duplicate subagent route {spec.route!r} in {directory}")
        specs[spec.route] = spec
    return specs


def model_cfg_from_agent_spec(spec: AgentSpec) -> ModelConfig | None:
    """Read an optional YAACLI child context configuration from native metadata."""
    value = (spec.metadata or {}).get(YAACLI_MODEL_CFG_METADATA_KEY)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError(f"AgentSpec metadata {YAACLI_MODEL_CFG_METADATA_KEY!r} must be a mapping")
    return ModelConfig.model_validate(value)
