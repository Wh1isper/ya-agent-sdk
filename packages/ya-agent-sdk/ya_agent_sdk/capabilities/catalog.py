"""Immutable capability type catalog used by declarative agent plans."""

from __future__ import annotations

import dataclasses
import importlib.metadata
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, cast

from pydantic_ai import AgentSpec
from pydantic_ai.capabilities import CAPABILITY_TYPES, AbstractCapability

ENTRY_POINT_GROUP = "ya_agent_sdk.capabilities"

CapabilityType = type[AbstractCapability[Any]]
CapabilitySourceKind = Literal["sdk", "explicit", "entry_point"]


@dataclass(frozen=True, slots=True)
class CapabilityTypeReference:
    """Metadata for one capability entry point, without importing its target."""

    entry_point_name: str
    import_target: str
    distribution_name: str | None
    distribution_version: str | None


@dataclass(frozen=True, slots=True)
class CapabilityTypeProvenance:
    """Structured origin of one validated capability type."""

    serialization_name: str
    source_kind: CapabilitySourceKind
    class_module: str
    class_qualname: str
    entry_point: CapabilityTypeReference | None = None

    @property
    def display_name(self) -> str:
        """Return a concise stable value for logs and persisted audit records."""
        class_name = f"{self.class_module}.{self.class_qualname}"
        if self.entry_point is None:
            return f"{self.source_kind}:{class_name}"
        distribution = self.entry_point.distribution_name or "unknown"
        if self.entry_point.distribution_version:
            distribution = f"{distribution}@{self.entry_point.distribution_version}"
        return f"entry-point:{distribution}:{self.entry_point.entry_point_name}={self.entry_point.import_target}"


@dataclass(frozen=True, slots=True)
class CapabilityRegistration:
    """A validated capability type and its construction-time provenance."""

    serialization_name: str
    capability_type: CapabilityType
    provenance: CapabilityTypeProvenance


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

    def provenance(self, serialization_name: str) -> CapabilityTypeProvenance:
        for item in self._registrations:
            if item.serialization_name == serialization_name:
                return item.provenance
        raise KeyError(serialization_name)


def discover_capability_types() -> tuple[CapabilityTypeReference, ...]:
    """Discover capability plugin metadata without importing plugin code."""

    references = (_entry_point_reference(entry_point) for entry_point in _entry_points())
    return tuple(
        sorted(
            references,
            key=lambda item: (
                item.entry_point_name,
                item.distribution_name or "",
                item.distribution_version or "",
                item.import_target,
            ),
        )
    )


def build_capability_catalog(
    *,
    sdk_types: Iterable[CapabilityType] = (),
    explicit_types: Iterable[CapabilityType] = (),
    selected_entry_points: Iterable[str] = (),
) -> CapabilityCatalog:
    """Build one process catalog without importing unselected entry points."""

    registrations: dict[str, CapabilityRegistration] = {}
    for capability_type in sdk_types:
        _register(registrations, capability_type, source_kind="sdk")
    for capability_type in explicit_types:
        _register(registrations, capability_type, source_kind="explicit")

    for registration in _load_selected_entry_points(selected_entry_points):
        _register(
            registrations,
            registration.capability_type,
            source_kind="entry_point",
            entry_point=registration.provenance.entry_point,
        )

    catalog = CapabilityCatalog(tuple(registrations.values()))
    try:
        AgentSpec.model_json_schema_with_capabilities(catalog.custom_capability_types)
    except Exception as exc:
        origins = [registration.provenance.display_name for registration in catalog.registrations]
        raise ValueError(f"Capability catalog schema validation failed for {origins!r}: {exc}") from exc
    return catalog


def _load_selected_entry_points(
    selected_entry_points: Iterable[str],
) -> tuple[CapabilityRegistration, ...]:
    selected = tuple(dict.fromkeys(selected_entry_points))
    if not selected:
        return ()

    discovered: dict[str, list[importlib.metadata.EntryPoint]] = {}
    for entry_point in _entry_points():
        if entry_point.name in selected:
            discovered.setdefault(entry_point.name, []).append(entry_point)

    missing = [name for name in selected if name not in discovered]
    if missing:
        raise ValueError(f"Selected capability entry points were not found: {sorted(missing)!r}")

    registrations = [_load_entry_point(_select_entry_point(name, discovered[name])) for name in selected]
    return tuple(registrations)


def _select_entry_point(
    name: str,
    matches: Sequence[importlib.metadata.EntryPoint],
) -> importlib.metadata.EntryPoint:
    if len(matches) == 1:
        return matches[0]
    references = sorted(
        (_entry_point_reference(item) for item in matches),
        key=lambda item: (
            item.distribution_name or "",
            item.distribution_version or "",
            item.import_target,
        ),
    )
    raise ValueError(f"Capability entry point {name!r} is provided more than once: {references!r}")


def _load_entry_point(entry_point: importlib.metadata.EntryPoint) -> CapabilityRegistration:
    reference = _entry_point_reference(entry_point)
    try:
        loaded = entry_point.load()
    except Exception as exc:
        raise ImportError(f"Failed to load capability entry point {reference!r}") from exc
    if not isinstance(loaded, type) or not issubclass(loaded, AbstractCapability):
        raise TypeError(f"Capability entry point {reference!r} must load one AbstractCapability class")
    capability_type = cast(CapabilityType, loaded)
    try:
        serialization_name = _serialization_name(capability_type)
    except TypeError as exc:
        raise TypeError(f"Invalid capability entry point {reference!r}: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"Invalid capability entry point {reference!r}: {exc}") from exc
    if reference.entry_point_name != serialization_name:
        raise ValueError(
            f"Capability entry point {reference!r} name must equal serialization name {serialization_name!r}"
        )
    return CapabilityRegistration(
        serialization_name=serialization_name,
        capability_type=capability_type,
        provenance=_provenance(
            serialization_name,
            capability_type,
            source_kind="entry_point",
            entry_point=reference,
        ),
    )


def _register(
    registrations: dict[str, CapabilityRegistration],
    capability_type: CapabilityType,
    *,
    source_kind: CapabilitySourceKind,
    entry_point: CapabilityTypeReference | None = None,
) -> None:
    serialization_name = _serialization_name(capability_type)
    provenance = _provenance(
        serialization_name,
        capability_type,
        source_kind=source_kind,
        entry_point=entry_point,
    )
    existing = registrations.get(serialization_name)
    if existing is not None:
        if existing.capability_type is capability_type and existing.provenance == provenance:
            return
        raise ValueError(
            f"Capability serialization name {serialization_name!r} is provided by both "
            f"{existing.provenance.display_name!r} and {provenance.display_name!r}"
        )
    registrations[serialization_name] = CapabilityRegistration(
        serialization_name=serialization_name,
        capability_type=capability_type,
        provenance=provenance,
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
    if name in CAPABILITY_TYPES:
        raise ValueError(f"Capability serialization name {name!r} conflicts with a native Pydantic AI capability")
    return name


def _provenance(
    serialization_name: str,
    capability_type: CapabilityType,
    *,
    source_kind: CapabilitySourceKind,
    entry_point: CapabilityTypeReference | None = None,
) -> CapabilityTypeProvenance:
    return CapabilityTypeProvenance(
        serialization_name=serialization_name,
        source_kind=source_kind,
        class_module=capability_type.__module__,
        class_qualname=capability_type.__qualname__,
        entry_point=entry_point,
    )


def _entry_points() -> Sequence[importlib.metadata.EntryPoint]:
    return tuple(importlib.metadata.entry_points(group=ENTRY_POINT_GROUP))


def _entry_point_reference(entry_point: importlib.metadata.EntryPoint) -> CapabilityTypeReference:
    distribution = entry_point.dist
    return CapabilityTypeReference(
        entry_point_name=entry_point.name,
        import_target=entry_point.value,
        distribution_name=distribution.name if distribution is not None else None,
        distribution_version=distribution.version if distribution is not None else None,
    )
