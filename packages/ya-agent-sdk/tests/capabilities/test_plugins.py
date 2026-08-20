from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic_ai import AgentSpec
from pydantic_ai.capabilities import AbstractCapability, CombinedCapability
from pydantic_ai.models.test import TestModel
from ya_agent_sdk.agents.main import create_agent
from ya_agent_sdk.capabilities import (
    CAPABILITY_PLUGIN_SCHEMA_VERSION,
    CapabilityPluginManifest,
    ResolvedCapabilityPlugins,
    build_default_capability_catalog,
    load_capability_plugin_manifest,
    load_capability_plugins,
    resolve_capability_plugins,
)
from ya_agent_sdk.subagents import SubagentPlanResolver, SubagentSpec


@dataclass
class ExamplePluginCapability(AbstractCapability[Any]):
    label: str = "default"
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def get_serialization_name(cls) -> str:
        return "example.plugin"


@dataclass
class ExplicitCapability(AbstractCapability[Any]):
    pass


class FakeEntryPoint:
    name = "example.plugin"
    value = "example_plugin:ExamplePluginCapability"
    dist = SimpleNamespace(name="example-plugin", version="1.2.3")

    def __init__(self) -> None:
        self.load_count = 0

    def load(self) -> object:
        self.load_count += 1
        return ExamplePluginCapability


def test_load_capability_plugins_resolves_catalog_and_ordered_root_grants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry_point = FakeEntryPoint()
    monkeypatch.setattr(
        "ya_agent_sdk.capabilities.catalog.importlib.metadata.entry_points",
        lambda **kwargs: [entry_point],
    )
    manifest_path = tmp_path / "plugins.toml"
    manifest_path.write_text(
        """\
schema_version = 1
entry_points = ["example.plugin"]

[[capabilities]]
name = "example.plugin"
arguments = { label = "first" }

[[capabilities]]
name = "example.plugin"
arguments = { label = "second" }
""",
        encoding="utf-8",
    )

    plugins = load_capability_plugins(manifest_path, explicit_types=[ExplicitCapability])

    assert CAPABILITY_PLUGIN_SCHEMA_VERSION == 1
    assert entry_point.load_count == 1
    assert plugins.catalog["example.plugin"] is ExamplePluginCapability
    assert plugins.catalog["ExplicitCapability"] is ExplicitCapability
    assert [capability.name for capability in plugins.root_agent_spec.capabilities] == [
        "example.plugin",
        "example.plugin",
    ]
    assert [capability.arguments for capability in plugins.root_agent_spec.capabilities] == [
        {"label": "first"},
        {"label": "second"},
    ]
    assert plugins.custom_capability_types == plugins.catalog.custom_capability_types


def test_resolved_plugins_append_root_grants_without_mutating_agent_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry_point = FakeEntryPoint()
    monkeypatch.setattr(
        "ya_agent_sdk.capabilities.catalog.importlib.metadata.entry_points",
        lambda **kwargs: [entry_point],
    )
    manifest = CapabilityPluginManifest.model_validate({
        "schema_version": 1,
        "entry_points": ["example.plugin"],
        "capabilities": [
            {
                "name": "example.plugin",
                "arguments": {"label": "plugin", "options": {"values": ["original"]}},
            }
        ],
    })
    plugins = resolve_capability_plugins(manifest)
    base = AgentSpec.from_dict({"capabilities": ["ToolSearch"]})

    merged = plugins.apply_to_root_agent_spec(base)

    assert [capability.name for capability in base.capabilities] == ["ToolSearch"]
    assert [capability.name for capability in merged.capabilities] == ["ToolSearch", "example.plugin"]

    manifest.capabilities[0].arguments["label"] = "mutated"
    options = manifest.capabilities[0].arguments["options"]
    assert isinstance(options, dict)
    values = options["values"]
    assert isinstance(values, list)
    values.append("mutated")
    assert plugins.root_agent_spec.capabilities[0].arguments == {
        "label": "plugin",
        "options": {"values": ["original"]},
    }


def test_resolved_plugins_expose_isolated_manifest_and_root_spec_copies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry_point = FakeEntryPoint()
    monkeypatch.setattr(
        "ya_agent_sdk.capabilities.catalog.importlib.metadata.entry_points",
        lambda **kwargs: [entry_point],
    )
    plugins = resolve_capability_plugins(
        CapabilityPluginManifest.model_validate({
            "schema_version": 1,
            "entry_points": ["example.plugin"],
            "capabilities": [
                {
                    "name": "example.plugin",
                    "arguments": {"label": "canonical", "options": {"values": ["original"]}},
                }
            ],
        })
    )

    exposed_manifest = plugins.manifest
    exposed_manifest.capabilities[0].arguments["label"] = "mutated"
    exposed_root_spec = plugins.root_agent_spec
    exposed_root_spec.capabilities[0].arguments["label"] = "mutated"

    canonical_manifest = plugins.manifest
    canonical_root_spec = plugins.root_agent_spec
    assert canonical_manifest.capabilities[0].arguments == {
        "label": "canonical",
        "options": {"values": ["original"]},
    }
    assert canonical_root_spec.capabilities[0].arguments == canonical_manifest.capabilities[0].arguments


def test_resolve_rejects_grant_arguments_that_cannot_call_plugin_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry_point = FakeEntryPoint()
    monkeypatch.setattr(
        "ya_agent_sdk.capabilities.catalog.importlib.metadata.entry_points",
        lambda **kwargs: [entry_point],
    )
    manifest = CapabilityPluginManifest.model_validate({
        "schema_version": 1,
        "entry_points": ["example.plugin"],
        "capabilities": [{"name": "example.plugin", "arguments": {"unexpected": True}}],
    })

    with pytest.raises(ValueError, match=r"grant 0.*invalid arguments.*unexpected"):
        resolve_capability_plugins(manifest)


def test_selected_but_ungranted_plugin_is_available_only_in_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry_point = FakeEntryPoint()
    monkeypatch.setattr(
        "ya_agent_sdk.capabilities.catalog.importlib.metadata.entry_points",
        lambda **kwargs: [entry_point],
    )
    manifest = CapabilityPluginManifest(schema_version=1, entry_points=("example.plugin",))

    plugins = resolve_capability_plugins(manifest)

    assert plugins.catalog["example.plugin"] is ExamplePluginCapability
    assert plugins.root_agent_spec.capabilities == []


@pytest.mark.asyncio
async def test_resolved_plugins_construct_root_runtime_without_granting_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry_point = FakeEntryPoint()
    monkeypatch.setattr(
        "ya_agent_sdk.capabilities.catalog.importlib.metadata.entry_points",
        lambda **kwargs: [entry_point],
    )
    manifest = CapabilityPluginManifest.model_validate({
        "schema_version": 1,
        "entry_points": ["example.plugin"],
        "capabilities": [{"name": "example.plugin", "arguments": {"label": "root"}}],
    })
    plugins = resolve_capability_plugins(manifest)
    child_plan = SubagentPlanResolver(plugins.catalog, default_model="test").resolve(
        SubagentSpec(route="child", agent=AgentSpec())
    )
    runtime = create_agent(
        TestModel(),
        spec=plugins.root_agent_spec,
        custom_capability_types=plugins.custom_capability_types,
    )

    async with runtime:
        root_capability = runtime.agent.root_capability

    assert isinstance(root_capability, CombinedCapability)
    configured = [item for item in root_capability.capabilities if isinstance(item, ExamplePluginCapability)]
    assert len(configured) == 1
    assert configured[0].label == "root"
    assert all(item.name != "example.plugin" for item in child_plan.normalized_agent_spec.capabilities)


def test_empty_selection_does_not_scan_entry_point_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject_scan(**_kwargs: object) -> object:
        raise AssertionError("empty plugin selection must not scan installed entry points")

    monkeypatch.setattr(
        "ya_agent_sdk.capabilities.catalog.importlib.metadata.entry_points",
        reject_scan,
    )

    plugins = resolve_capability_plugins(CapabilityPluginManifest(schema_version=1))

    assert plugins.root_agent_spec.capabilities == []


def test_resolved_plugins_reject_inconsistent_manifest_and_catalog() -> None:
    manifest = CapabilityPluginManifest(schema_version=1, entry_points=("example.plugin",))

    with pytest.raises(ValueError, match="different selected entry points"):
        ResolvedCapabilityPlugins(
            manifest=manifest,
            catalog=build_default_capability_catalog(),
        )


@pytest.mark.parametrize(
    "argument_name",
    [
        "accessToken",
        "refreshToken",
        "clientSecret",
        "apiKey",
        "APIKey",
        "privateKey",
        "passwordHash",
        "passphrase",
        "db_passwd",
        "pwd",
        "sshPassphrase",
        "bearer",
        "jwt",
        "github_pat",
        "service_account_key",
        "serviceAccountKey",
        "oauth_code",
        "oauthCode",
        "authorizationHeader",
        "auth",
        "client.secret",
        "private/key",
    ],
)
def test_manifest_rejects_sensitive_argument_naming_variants(argument_name: str) -> None:
    with pytest.raises(ValueError, match="must not contain secrets"):
        CapabilityPluginManifest.model_validate({
            "schema_version": 1,
            "entry_points": ["example.plugin"],
            "capabilities": [
                {
                    "name": "example.plugin",
                    "arguments": {"nested": [{argument_name: "secret"}]},
                }
            ],
        })


def test_resolve_revalidates_mutated_manifest_before_entry_point_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = CapabilityPluginManifest.model_validate({
        "schema_version": 1,
        "entry_points": ["example.plugin"],
        "capabilities": [{"name": "example.plugin", "arguments": {"label": "safe"}}],
    })
    manifest.capabilities[0].arguments["clientSecret"] = "late-mutation"

    def reject_scan(**_kwargs: object) -> object:
        raise AssertionError("an invalid mutated manifest must fail before discovery")

    monkeypatch.setattr(
        "ya_agent_sdk.capabilities.catalog.importlib.metadata.entry_points",
        reject_scan,
    )

    with pytest.raises(ValueError, match="must not contain secrets"):
        resolve_capability_plugins(manifest)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("schema_version = 2\n", "schema_version"),
        ("schema_version = true\n", "must be integer 1"),
        (
            'schema_version = 1\nentry_points = ["example.plugin"]\ncapabilities = [{ name = "example.plugin", arguments = { value = nan } }]\n',
            "finite number",
        ),
        (
            'schema_version = 1\nentry_points = ["example.plugin"]\ncapabilities = [{ name = "example.plugin", arguments = { value = inf } }]\n',
            "finite number",
        ),
        (
            'schema_version = 1\nentry_points = ["example.plugin"]\ncapabilities = [{ name = "example.plugin", arguments = { value = -inf } }]\n',
            "finite number",
        ),
        (
            'schema_version = 1\nentry_points = ["example.plugin"]\ncapabilities = [{ name = "example.plugin", arguments = { auth = { access_token = "secret" } } }]\n',
            "must not contain secrets",
        ),
        (
            'schema_version = 1\nentry_points = ["example.plugin"]\ncapabilities = [{ name = "example.plugin", arguments = { api-key = "secret" } }]\n',
            "must not contain secrets",
        ),
        ("schema_version = 1\nunknown = true\n", "unknown"),
        (
            'schema_version = 1\nentry_points = ["example.plugin", "example.plugin"]\n',
            "must be unique",
        ),
        (
            """\
schema_version = 1
entry_points = []
[[capabilities]]
name = "example.plugin"
""",
            "must reference selected entry points",
        ),
        (
            """\
schema_version = 1
entry_points = ["example.plugin"]
capabilities = [{ "example.plugin" = {} }]
""",
            "name",
        ),
    ],
)
def test_manifest_rejects_invalid_contract(
    tmp_path: Path,
    content: str,
    message: str,
) -> None:
    manifest_path = tmp_path / "plugins.toml"
    manifest_path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_capability_plugin_manifest(manifest_path)


def test_manifest_reports_invalid_toml_with_path(tmp_path: Any) -> None:
    manifest_path = tmp_path / "plugins.toml"
    manifest_path.write_text("schema_version = [", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Invalid TOML.*plugins\.toml"):
        load_capability_plugin_manifest(manifest_path)
