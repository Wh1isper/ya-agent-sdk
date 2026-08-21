"""YAACLI subagent configuration tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from ya_agent_sdk.capabilities import build_default_capability_catalog
from ya_agent_sdk.context import ModelConfig
from ya_agent_sdk.subagents import SubagentPlanResolver, SubagentSpec
from yaacli.runtime import _standard_child_capability_specs
from yaacli.subagent_config import (
    has_subagent_definition,
    load_subagent_specs,
    materialize_subagent_model_configuration,
)


def _payload(route: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "route": route,
        "agent": {
            "name": route,
            "description": "Native helper",
            "instructions": "Help with the requested task.",
            "model": "test",
            "capabilities": [
                {"name": "RuntimeFoundationCapability", "arguments": {}},
                {"name": "FilesystemCapability", "arguments": {}},
            ],
        },
        "execution_modes": ["foreground", "background"],
        "durability": "restart",
    }


def _load_specs(directory: Path, *, enable_codeact: bool = True) -> dict[str, SubagentSpec]:
    return load_subagent_specs(
        directory,
        markdown_capabilities=_standard_child_capability_specs(enable_codeact=enable_codeact),
    )


def _markdown(
    *,
    name: str = "helper",
    extra_frontmatter: str = "",
    body: str = "You are a focused helper.",
) -> str:
    return f"""---
name: {name}
description: Generic Markdown helper
instruction: Use this helper for bounded work.
{extra_frontmatter}---

{body}
"""


def test_loads_versioned_native_subagent_spec(tmp_path: Path) -> None:
    (tmp_path / "helper.yaml").write_text(
        yaml.safe_dump(_payload("helper"), sort_keys=False),
        encoding="utf-8",
    )

    specs = _load_specs(tmp_path)

    assert tuple(specs) == ("helper",)
    spec = specs["helper"]
    assert spec.schema_version == 1
    assert spec.agent.name == "helper"
    assert [item.name for item in spec.agent.capabilities] == [
        "RuntimeFoundationCapability",
        "FilesystemCapability",
    ]


def test_loads_generic_markdown_as_portable_subagent_spec(tmp_path: Path) -> None:
    (tmp_path / "helper.md").write_text(
        _markdown(extra_frontmatter="model: inherit\nmodel_settings: inherit\nmodel_cfg: inherit\n"),
        encoding="utf-8",
    )

    specs = _load_specs(tmp_path)

    spec = specs["helper"]
    assert spec.agent.name == "helper"
    assert spec.agent.description == ("Generic Markdown helper\n\nUse this helper for bounded work.")
    assert spec.agent.instructions == "You are a focused helper."
    assert spec.agent.model is None
    assert spec.agent.model_settings is None
    assert spec.agent.metadata == {
        "yaacli_inherit_model_settings": True,
        "yaacli_inherit_model_cfg": True,
    }
    assert [item.name for item in spec.agent.capabilities] == [
        "RuntimeFoundationCapability",
        "MediaReadCapability",
        "DocumentConversionCapability",
        "FilesystemCapability",
        "ShellCapability",
        "WebSearchCapability",
        "WebContentCapability",
        "TaskCapability",
        "NoteCapability",
        "CodeActCapability",
    ]
    assert [mode.value for mode in spec.execution_modes] == ["foreground", "background"]
    assert spec.durability.value == "process"

    materialized = materialize_subagent_model_configuration(
        spec,
        inherited_model_settings={"max_tokens": 2048},
        inherited_model_cfg=ModelConfig(context_window=200_000),
    )
    assert materialized.agent.model_settings == {"max_tokens": 2048}
    assert materialized.agent.metadata is not None
    assert materialized.agent.metadata["yaacli_model_cfg"]["context_window"] == 200_000
    assert "yaacli_inherit_model_settings" not in materialized.agent.metadata
    assert "yaacli_inherit_model_cfg" not in materialized.agent.metadata


def test_markdown_resolves_model_configuration_and_tool_visibility(tmp_path: Path) -> None:
    (tmp_path / "helper.md").write_text(
        _markdown(
            extra_frontmatter=(
                "model: openai-chat:gpt-4o\n"
                "model_settings:\n"
                "  temperature: 0.25\n"
                "model_cfg:\n"
                "  context_window: 100000\n"
                "tools: view, grep\n"
                "optional_tools:\n"
                "  - glob\n"
                "  - view\n"
            )
        ),
        encoding="utf-8",
    )

    spec = _load_specs(tmp_path)["helper"]

    assert spec.agent.model == "openai-chat:gpt-4o"
    assert spec.agent.model_settings == {"temperature": 0.25}
    assert spec.agent.metadata is not None
    model_cfg = spec.agent.metadata["yaacli_model_cfg"]
    assert isinstance(model_cfg, dict)
    assert model_cfg["context_window"] == 100_000
    visibility = next(item for item in spec.agent.capabilities if item.name == "ToolVisibilityCapability")
    assert visibility.arguments == {"allow": ["view", "grep", "glob"]}
    plan = SubagentPlanResolver(
        build_default_capability_catalog(),
        default_model="test",
    ).resolve(spec)
    assert plan.spec.route == "helper"


def test_markdown_ignores_same_basename_native_capability_snapshot(tmp_path: Path) -> None:
    payload = _payload("helper")
    agent = payload["agent"]
    assert isinstance(agent, dict)
    capabilities = agent["capabilities"]
    assert isinstance(capabilities, list)
    capabilities.extend([
        {"name": "ThinkingCapability", "arguments": {}},
        {"name": "TodoCapability", "arguments": {}},
    ])
    (tmp_path / "helper.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    (tmp_path / "helper.md").write_text(
        _markdown(extra_frontmatter="model: inherit\nmodel_settings: inherit\nmodel_cfg: inherit\n"),
        encoding="utf-8",
    )

    spec = _load_specs(tmp_path)["helper"]
    capability_names = [item.name for item in spec.agent.capabilities]

    assert spec.agent.description == ("Generic Markdown helper\n\nUse this helper for bounded work.")
    assert spec.agent.model is None
    assert "TaskCapability" in capability_names
    assert "ThinkingCapability" not in capability_names
    assert "TodoCapability" not in capability_names
    assert spec.durability.value == "process"
    plan = SubagentPlanResolver(
        build_default_capability_catalog(),
        default_model="test",
    ).resolve(spec)
    assert plan.spec.route == "helper"


def test_markdown_ignores_malformed_same_basename_native_document(tmp_path: Path) -> None:
    (tmp_path / "helper.yaml").write_text("agent: [", encoding="utf-8")
    (tmp_path / "helper.md").write_text(_markdown(), encoding="utf-8")

    spec = _load_specs(tmp_path)["helper"]

    assert spec.agent.name == "helper"
    assert spec.durability.value == "process"


def test_markdown_tool_list_narrows_standard_capabilities_not_native_sidecar(tmp_path: Path) -> None:
    payload = _payload("helper")
    agent = payload["agent"]
    assert isinstance(agent, dict)
    capabilities = agent["capabilities"]
    assert isinstance(capabilities, list)
    capabilities.append({
        "name": "ToolVisibilityCapability",
        "arguments": {
            "allow": ["view"],
            "deny": ["shell_exec"],
        },
    })
    (tmp_path / "helper.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    (tmp_path / "helper.md").write_text(
        _markdown(extra_frontmatter="tools: [view, shell_exec]\n"),
        encoding="utf-8",
    )

    spec = _load_specs(tmp_path)["helper"]

    visibility = next(item for item in spec.agent.capabilities if item.name == "ToolVisibilityCapability")
    assert visibility.arguments == {"allow": ["view", "shell_exec"]}


def test_markdown_standard_capabilities_follow_codeact_setting(tmp_path: Path) -> None:
    (tmp_path / "helper.md").write_text(_markdown(), encoding="utf-8")

    spec = _load_specs(tmp_path, enable_codeact=False)["helper"]

    assert "TaskCapability" in {item.name for item in spec.agent.capabilities}
    assert "CodeActCapability" not in {item.name for item in spec.agent.capabilities}


def test_same_basename_markdown_can_replace_native_route_identity(tmp_path: Path) -> None:
    (tmp_path / "helper.yaml").write_text(
        yaml.safe_dump(_payload("original"), sort_keys=False),
        encoding="utf-8",
    )
    (tmp_path / "helper.md").write_text(
        _markdown(name="renamed"),
        encoding="utf-8",
    )

    specs = _load_specs(tmp_path)

    assert tuple(specs) == ("renamed",)
    spec = specs["renamed"]
    assert spec.agent.name == "renamed"
    assert "TaskCapability" in {item.name for item in spec.agent.capabilities}
    assert spec.durability.value == "process"


def test_same_basename_route_renames_validate_final_precedence_set(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text(
        yaml.safe_dump(_payload("original-a"), sort_keys=False),
        encoding="utf-8",
    )
    (tmp_path / "b.yaml").write_text(
        yaml.safe_dump(_payload("target"), sort_keys=False),
        encoding="utf-8",
    )
    (tmp_path / "a.md").write_text(_markdown(name="target"), encoding="utf-8")
    (tmp_path / "b.md").write_text(_markdown(name="renamed-b"), encoding="utf-8")

    specs = _load_specs(tmp_path)

    assert tuple(specs) == ("target", "renamed-b")
    assert specs["target"].durability.value == "process"
    assert specs["renamed-b"].durability.value == "process"


def test_rejects_invalid_markdown_subagent_configuration(tmp_path: Path) -> None:
    (tmp_path / "helper.md").write_text("No frontmatter", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Invalid Markdown subagent.*expected YAML frontmatter"):
        _load_specs(tmp_path)


def test_rejects_unknown_markdown_frontmatter_fields(tmp_path: Path) -> None:
    (tmp_path / "helper.md").write_text(
        _markdown(extra_frontmatter="unknown_policy: true\n"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"(?s)Invalid Markdown subagent.*extra_forbidden"):
        _load_specs(tmp_path)


def test_rejects_old_tool_fields_in_native_document(tmp_path: Path) -> None:
    payload = _payload("helper")
    payload["tools"] = ["view"]
    (tmp_path / "helper.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="extra_forbidden"):
        _load_specs(tmp_path)


def test_rejects_duplicate_routes_across_native_documents(tmp_path: Path) -> None:
    for filename in ("one.yaml", "two.json"):
        (tmp_path / filename).write_text(
            yaml.safe_dump(_payload("helper"), sort_keys=False),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="Duplicate subagent route"):
        _load_specs(tmp_path)


def test_rejects_ambiguous_native_documents_with_same_basename(tmp_path: Path) -> None:
    (tmp_path / "helper.json").write_text(
        yaml.safe_dump(_payload("first"), sort_keys=False),
        encoding="utf-8",
    )
    (tmp_path / "helper.yaml").write_text(
        yaml.safe_dump(_payload("second"), sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"Duplicate subagent basename.*helper.json.*helper.yaml"):
        _load_specs(tmp_path)


def test_rejects_duplicate_route_from_different_markdown_basename(tmp_path: Path) -> None:
    (tmp_path / "one.yaml").write_text(
        yaml.safe_dump(_payload("helper"), sort_keys=False),
        encoding="utf-8",
    )
    (tmp_path / "two.md").write_text(_markdown(), encoding="utf-8")

    with pytest.raises(ValueError, match=r"Duplicate subagent route.*one.yaml.*two.md"):
        _load_specs(tmp_path)


def test_detects_any_supported_subagent_definition_format(tmp_path: Path) -> None:
    assert has_subagent_definition(tmp_path, "helper") is False

    (tmp_path / "helper.md").write_text(_markdown(), encoding="utf-8")

    assert has_subagent_definition(tmp_path, "helper") is True
    assert has_subagent_definition(tmp_path, "other") is False
