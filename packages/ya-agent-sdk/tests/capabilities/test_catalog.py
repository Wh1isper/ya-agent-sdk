from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic_ai.capabilities import AbstractCapability
from ya_agent_sdk.capabilities import (
    CapabilityTypeProvenance,
    build_capability_catalog,
    build_default_capability_catalog,
    discover_capability_types,
)


@dataclass
class AlphaCapability(AbstractCapability[Any]):
    value: str = "alpha"


@dataclass
class ZetaCapability(AbstractCapability[Any]):
    value: str = "zeta"


@dataclass
class DuplicateAlphaCapability(AbstractCapability[Any]):
    @classmethod
    def get_serialization_name(cls) -> str:
        return "AlphaCapability"


class NotDeclaredAsDataclassCapability(AbstractCapability[Any]):
    pass


@dataclass
class NativeNameCollisionCapability(AbstractCapability[Any]):
    @classmethod
    def get_serialization_name(cls) -> str:
        return "ToolSearch"


@dataclass
class InvalidSchemaCapability(AbstractCapability[Any]):
    @classmethod
    def from_spec(cls, value: MissingSchemaType) -> InvalidSchemaCapability:  # type: ignore[name-defined]  # noqa: F821
        return cls()


def test_default_catalog_exposes_tasks_without_the_removed_todo_capability() -> None:
    catalog = build_default_capability_catalog()

    assert "TaskCapability" in catalog
    assert "ThinkingCapability" in catalog
    assert "TodoCapability" not in catalog


class FakeEntryPoint:
    def __init__(
        self,
        name: str,
        target: object,
        distribution: str,
        *,
        version: str = "1.0",
        value: str | None = None,
        load_error: Exception | None = None,
    ) -> None:
        self.name = name
        self.value = value or f"tests.plugins:{name}"
        self.dist = SimpleNamespace(name=distribution, version=version)
        self._target = target
        self._load_error = load_error
        self.load_count = 0

    def load(self) -> object:
        self.load_count += 1
        if self._load_error is not None:
            raise self._load_error
        return self._target


def test_catalog_is_sorted_for_lookup_without_mutating_runtime_order() -> None:
    catalog = build_capability_catalog(explicit_types=[ZetaCapability, AlphaCapability])

    assert tuple(catalog) == ("AlphaCapability", "ZetaCapability")
    assert catalog.custom_capability_types == (AlphaCapability, ZetaCapability)
    assert catalog.provenance("AlphaCapability") == CapabilityTypeProvenance(
        serialization_name="AlphaCapability",
        source_kind="explicit",
        class_module=AlphaCapability.__module__,
        class_qualname="AlphaCapability",
    )


def test_catalog_rejects_serialization_name_collision() -> None:
    with pytest.raises(ValueError, match="provided by both"):
        build_capability_catalog(explicit_types=[AlphaCapability, DuplicateAlphaCapability])


def test_catalog_requires_custom_class_dataclass_declaration() -> None:
    with pytest.raises(TypeError, match="declared as a dataclass"):
        build_capability_catalog(explicit_types=[NotDeclaredAsDataclassCapability])


def test_catalog_rejects_native_serialization_name_collision() -> None:
    with pytest.raises(ValueError, match="conflicts with a native Pydantic AI capability"):
        build_capability_catalog(explicit_types=[NativeNameCollisionCapability])


def test_catalog_uses_native_schema_validation_with_registration_provenance() -> None:
    with pytest.raises(ValueError, match="InvalidSchemaCapability") as exc_info:
        build_capability_catalog(explicit_types=[InvalidSchemaCapability])

    assert isinstance(exc_info.value.__cause__, NameError)
    assert "MissingSchemaType" in str(exc_info.value)


def test_discovery_returns_sorted_metadata_without_loading_plugins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alpha = FakeEntryPoint(
        "AlphaCapability",
        AlphaCapability,
        "alpha-package",
        version="2.1",
        value="alpha_plugin:AlphaCapability",
    )
    zeta = FakeEntryPoint("ZetaCapability", ZetaCapability, "zeta-package")
    monkeypatch.setattr(
        "ya_agent_sdk.capabilities.catalog.importlib.metadata.entry_points",
        lambda **kwargs: [zeta, alpha],
    )

    references = discover_capability_types()

    assert [reference.entry_point_name for reference in references] == ["AlphaCapability", "ZetaCapability"]
    assert references[0].import_target == "alpha_plugin:AlphaCapability"
    assert references[0].distribution_name == "alpha-package"
    assert references[0].distribution_version == "2.1"
    assert alpha.load_count == 0
    assert zeta.load_count == 0


def test_catalog_loads_only_selected_entry_points(monkeypatch: pytest.MonkeyPatch) -> None:
    selected = FakeEntryPoint("AlphaCapability", AlphaCapability, "selected-package")
    unselected = FakeEntryPoint("ZetaCapability", ZetaCapability, "unselected-package")
    monkeypatch.setattr(
        "ya_agent_sdk.capabilities.catalog.importlib.metadata.entry_points",
        lambda **kwargs: [selected, unselected],
    )

    catalog = build_capability_catalog(selected_entry_points=["AlphaCapability"])

    assert catalog["AlphaCapability"] is AlphaCapability
    provenance = catalog.provenance("AlphaCapability")
    assert provenance.source_kind == "entry_point"
    assert provenance.entry_point is not None
    assert provenance.entry_point.distribution_name == "selected-package"
    assert provenance.display_name == ("entry-point:selected-package@1.0:AlphaCapability=tests.plugins:AlphaCapability")
    assert selected.load_count == 1
    assert unselected.load_count == 0


def test_catalog_reports_entry_point_provenance_on_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry_point = FakeEntryPoint(
        "AlphaCapability",
        AlphaCapability,
        "broken-package",
        version="3.2",
        load_error=RuntimeError("broken import"),
    )
    monkeypatch.setattr(
        "ya_agent_sdk.capabilities.catalog.importlib.metadata.entry_points",
        lambda **kwargs: [entry_point],
    )

    with pytest.raises(ImportError, match="broken-package"):
        build_capability_catalog(selected_entry_points=["AlphaCapability"])


def test_catalog_reports_entry_point_provenance_for_invalid_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry_point = FakeEntryPoint(
        "AlphaCapability",
        object(),
        "invalid-package",
        version="4.1",
    )
    monkeypatch.setattr(
        "ya_agent_sdk.capabilities.catalog.importlib.metadata.entry_points",
        lambda **kwargs: [entry_point],
    )

    with pytest.raises(TypeError, match="invalid-package"):
        build_capability_catalog(selected_entry_points=["AlphaCapability"])


def test_catalog_rejects_entry_point_name_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry_point = FakeEntryPoint("ConfiguredName", AlphaCapability, "package")
    monkeypatch.setattr(
        "ya_agent_sdk.capabilities.catalog.importlib.metadata.entry_points",
        lambda **kwargs: [entry_point],
    )

    with pytest.raises(ValueError, match="must equal serialization name"):
        build_capability_catalog(selected_entry_points=["ConfiguredName"])


def test_catalog_rejects_missing_selected_entry_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ya_agent_sdk.capabilities.catalog.importlib.metadata.entry_points",
        lambda **kwargs: [],
    )

    with pytest.raises(ValueError, match="were not found"):
        build_capability_catalog(selected_entry_points=["MissingCapability"])


def test_catalog_rejects_duplicate_selected_entry_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = FakeEntryPoint("AlphaCapability", AlphaCapability, "first-package")
    second = FakeEntryPoint("AlphaCapability", AlphaCapability, "second-package")
    monkeypatch.setattr(
        "ya_agent_sdk.capabilities.catalog.importlib.metadata.entry_points",
        lambda **kwargs: [first, second],
    )

    with pytest.raises(ValueError, match="is provided more than once"):
        build_capability_catalog(selected_entry_points=["AlphaCapability"])

    assert first.load_count == 0
    assert second.load_count == 0
