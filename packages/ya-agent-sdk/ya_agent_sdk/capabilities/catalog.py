"""Immutable capability type catalog used by declarative agent plans."""

from __future__ import annotations

import dataclasses
import importlib.metadata
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

from pydantic_ai.capabilities import AbstractCapability

ENTRY_POINT_GROUP = "ya_agent_sdk.capabilities"

CapabilityType = type[AbstractCapability[Any]]


@dataclass(frozen=True, slots=True)
class CapabilityRegistration:
    """A validated capability type and its construction-time provenance."""

    serialization_name: str
    capability_type: CapabilityType
    source: str


class CapabilityCatalog(Mapping[str, CapabilityType]):
    """Immutable lookup for the custom capability types available to AgentSpec."""

    __slots__ = ("_registrations", "_types")

    def __init__(self, registrations: Sequence[CapabilityRegistration]) -> None:
        ordered = sorted(registrations, key=lambda item: item.serialization_name)
        self._registrations = tuple(ordered)
        self._types = MappingProxyType({item.serialization_name: item.capability_type for item in ordered})

    def __getitem__(self, key: str) -> CapabilityType:
        return self._types[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._types)

    def __len__(self) -> int:
        return len(self._types)

    @property
    def registrations(self) -> tuple[CapabilityRegistration, ...]:
        return self._registrations

    @property
    def custom_capability_types(self) -> tuple[CapabilityType, ...]:
        return tuple(item.capability_type for item in self._registrations)

    def provenance(self, serialization_name: str) -> str:
        for item in self._registrations:
            if item.serialization_name == serialization_name:
                return item.source
        raise KeyError(serialization_name)


def build_capability_catalog(
    *,
    sdk_types: Iterable[CapabilityType] = (),
    explicit_types: Iterable[CapabilityType] = (),
    selected_entry_points: Iterable[str] = (),
) -> CapabilityCatalog:
    """Build one process catalog without importing unselected entry points."""

    registrations: dict[str, CapabilityRegistration] = {}
    for capability_type in sdk_types:
        _register(
            registrations,
            capability_type,
            source="sdk",
        )
    for capability_type in explicit_types:
        _register(
            registrations,
            capability_type,
            source=f"explicit:{capability_type.__module__}.{capability_type.__qualname__}",
        )

    for registration in _load_selected_entry_points(selected_entry_points):
        _register(
            registrations,
            registration.capability_type,
            source=registration.source,
        )

    return CapabilityCatalog(tuple(registrations.values()))


def _load_selected_entry_points(
    selected_entry_points: Iterable[str],
) -> tuple[CapabilityRegistration, ...]:
    selected = tuple(dict.fromkeys(selected_entry_points))
    if not selected:
        return ()

    discovered: dict[str, list[importlib.metadata.EntryPoint]] = {}
    for entry_point in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
        if entry_point.name in selected:
            discovered.setdefault(entry_point.name, []).append(entry_point)

    missing = [name for name in selected if name not in discovered]
    if missing:
        raise ValueError(f"Selected capability entry points were not found: {sorted(missing)!r}")

    registrations: list[CapabilityRegistration] = []
    for name in selected:
        matches = discovered[name]
        if len(matches) != 1:
            sources = sorted(_entry_point_source(item) for item in matches)
            raise ValueError(f"Capability entry point {name!r} is provided more than once: {sources!r}")
        entry_point = matches[0]
        loaded = entry_point.load()
        if not isinstance(loaded, type) or not issubclass(loaded, AbstractCapability):
            raise TypeError(f"Capability entry point {name!r} must load one AbstractCapability class")
        capability_type = cast(CapabilityType, loaded)
        serialization_name = _serialization_name(capability_type)
        if name != serialization_name:
            raise ValueError(
                f"Capability entry point name {name!r} must equal serialization name {serialization_name!r}"
            )
        registrations.append(
            CapabilityRegistration(
                serialization_name=serialization_name,
                capability_type=capability_type,
                source=_entry_point_source(entry_point),
            )
        )
    return tuple(registrations)


def _register(
    registrations: dict[str, CapabilityRegistration],
    capability_type: CapabilityType,
    *,
    source: str,
) -> None:
    serialization_name = _serialization_name(capability_type)
    existing = registrations.get(serialization_name)
    if existing is not None:
        if existing.capability_type is capability_type:
            return
        raise ValueError(
            f"Capability serialization name {serialization_name!r} is provided by both "
            f"{existing.source!r} and {source!r}"
        )
    registrations[serialization_name] = CapabilityRegistration(
        serialization_name=serialization_name,
        capability_type=capability_type,
        source=source,
    )


def _serialization_name(capability_type: CapabilityType) -> str:
    if not isinstance(capability_type, type) or not issubclass(capability_type, AbstractCapability):
        raise TypeError("Capability catalog entries must be AbstractCapability classes")
    if not dataclasses.is_dataclass(capability_type) or "__dataclass_fields__" not in capability_type.__dict__:
        raise TypeError(
            f"Capability class {capability_type.__module__}.{capability_type.__qualname__} "
            "must be declared as a dataclass"
        )
    name = capability_type.get_serialization_name()
    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            f"Capability class {capability_type.__module__}.{capability_type.__qualname__} "
            "must define a non-empty serialization name"
        )
    return name


def _entry_point_source(entry_point: importlib.metadata.EntryPoint) -> str:
    distribution = entry_point.dist
    distribution_name = distribution.name if distribution is not None else "unknown"
    return f"entry-point:{distribution_name}:{entry_point.name}"
