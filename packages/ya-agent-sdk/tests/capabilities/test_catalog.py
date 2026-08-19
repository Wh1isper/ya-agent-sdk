from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic_ai.capabilities import AbstractCapability
from ya_agent_sdk.capabilities import build_capability_catalog


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


class FakeEntryPoint:
    def __init__(self, name: str, value: object, distribution: str) -> None:
        self.name = name
        self.dist = SimpleNamespace(name=distribution)
        self._value = value
        self.load_count = 0

    def load(self) -> object:
        self.load_count += 1
        return self._value


def test_catalog_is_sorted_for_lookup_without_mutating_runtime_order() -> None:
    catalog = build_capability_catalog(explicit_types=[ZetaCapability, AlphaCapability])

    assert tuple(catalog) == ("AlphaCapability", "ZetaCapability")
    assert catalog.custom_capability_types == (AlphaCapability, ZetaCapability)


def test_catalog_rejects_serialization_name_collision() -> None:
    with pytest.raises(ValueError, match="provided by both"):
        build_capability_catalog(explicit_types=[AlphaCapability, DuplicateAlphaCapability])


def test_catalog_requires_custom_class_dataclass_declaration() -> None:
    with pytest.raises(TypeError, match="declared as a dataclass"):
        build_capability_catalog(explicit_types=[NotDeclaredAsDataclassCapability])


def test_catalog_loads_only_selected_entry_points(monkeypatch: pytest.MonkeyPatch) -> None:
    selected = FakeEntryPoint("AlphaCapability", AlphaCapability, "selected-package")
    unselected = FakeEntryPoint("ZetaCapability", ZetaCapability, "unselected-package")
    monkeypatch.setattr(
        "ya_agent_sdk.capabilities.catalog.importlib.metadata.entry_points",
        lambda **kwargs: [selected, unselected],
    )

    catalog = build_capability_catalog(selected_entry_points=["AlphaCapability"])

    assert catalog["AlphaCapability"] is AlphaCapability
    assert catalog.provenance("AlphaCapability") == ("entry-point:selected-package:AlphaCapability")
    assert selected.load_count == 1
    assert unselected.load_count == 0


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
