"""Strict native subagent configuration tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from yaacli.subagent_config import load_subagent_specs


def _payload(route: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "route": route,
        "agent": {
            "name": route,
            "description": "Native helper",
            "instructions": "Help with the requested task.",
            "capabilities": [
                {"name": "RuntimeFoundationCapability", "arguments": {}},
                {"name": "FilesystemCapability", "arguments": {}},
            ],
        },
        "execution_modes": ["foreground", "background"],
        "durability": "restart",
    }


def test_loads_versioned_native_subagent_spec(tmp_path: Path) -> None:
    (tmp_path / "helper.yaml").write_text(
        yaml.safe_dump(_payload("helper"), sort_keys=False),
        encoding="utf-8",
    )

    specs = load_subagent_specs(tmp_path)

    assert tuple(specs) == ("helper",)
    spec = specs["helper"]
    assert spec.schema_version == 1
    assert spec.agent.name == "helper"
    assert [item.name for item in spec.agent.capabilities] == [
        "RuntimeFoundationCapability",
        "FilesystemCapability",
    ]


def test_rejects_legacy_markdown_subagent_configuration(tmp_path: Path) -> None:
    (tmp_path / "helper.md").write_text(
        "---\nname: helper\ndescription: old\ntools: [view]\n---\nOld prompt",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Legacy Markdown"):
        load_subagent_specs(tmp_path)


def test_rejects_old_tool_fields_in_native_document(tmp_path: Path) -> None:
    payload = _payload("helper")
    payload["tools"] = ["view"]
    (tmp_path / "helper.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="extra_forbidden"):
        load_subagent_specs(tmp_path)


def test_rejects_duplicate_routes_across_documents(tmp_path: Path) -> None:
    for filename in ("one.yaml", "two.json"):
        (tmp_path / filename).write_text(
            yaml.safe_dump(_payload("helper"), sort_keys=False),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="Duplicate subagent route"):
        load_subagent_specs(tmp_path)
