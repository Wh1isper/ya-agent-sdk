"""Strict file configuration for selected capability plugins."""

from __future__ import annotations

import copy
import inspect
import math
import os
import re
import tomllib
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, field_validator, model_validator
from pydantic_ai import AgentSpec
from pydantic_ai.capabilities import AbstractCapability

from ya_agent_sdk.capabilities.catalog import CapabilityCatalog, CapabilityType
from ya_agent_sdk.capabilities.defaults import build_default_capability_catalog

CAPABILITY_PLUGIN_SCHEMA_VERSION = 1
_SENSITIVE_ARGUMENT_TOKENS = frozenset({
    "auth",
    "authorization",
    "bearer",
    "cookie",
    "credential",
    "credentials",
    "jwt",
    "passcode",
    "passphrase",
    "passwd",
    "password",
    "pat",
    "pwd",
    "secret",
    "token",
})
_SENSITIVE_KEY_QUALIFIERS = frozenset({
    "access",
    "api",
    "client",
    "encryption",
    "private",
    "secret",
    "signing",
})
_SENSITIVE_COMPACT_NAMES = frozenset({
    "accesskey",
    "accesstoken",
    "apikey",
    "apisecret",
    "appsecret",
    "authheader",
    "authtoken",
    "authorizationheader",
    "bearertoken",
    "clientsecret",
    "cookieheader",
    "encryptionkey",
    "hashedpassword",
    "idtoken",
    "passworddigest",
    "passwordhash",
    "privatekey",
    "refreshtoken",
    "secretkey",
    "signingkey",
})


def _find_non_finite_number_path(value: JsonValue, path: tuple[str | int, ...] = ()) -> tuple[str | int, ...] | None:
    if isinstance(value, float) and not math.isfinite(value):
        return path
    if isinstance(value, dict):
        for key, item in value.items():
            nested = _find_non_finite_number_path(item, (*path, key))
            if nested is not None:
                return nested
    elif isinstance(value, list):
        for index, item in enumerate(value):
            nested = _find_non_finite_number_path(item, (*path, index))
            if nested is not None:
                return nested
    return None


def _canonical_argument_name(name: str) -> tuple[str, ...]:
    with_camel_boundaries = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    with_acronym_boundaries = re.sub(
        r"(?<=[A-Z])(?=[A-Z][a-z])",
        "_",
        with_camel_boundaries,
    )
    return tuple(part for part in re.split(r"[^a-z0-9]+", with_acronym_boundaries.lower()) if part)


def _is_sensitive_argument_name(name: str) -> bool:
    tokens = _canonical_argument_name(name)
    token_set = set(tokens)
    if _SENSITIVE_ARGUMENT_TOKENS.intersection(token_set):
        return True
    if "key" in token_set and _SENSITIVE_KEY_QUALIFIERS.intersection(token_set):
        return True
    if {"service", "account", "key"} <= token_set or {"oauth", "code"} <= token_set:
        return True
    compact = "".join(tokens)
    return any(compact.startswith(candidate) or compact.endswith(candidate) for candidate in _SENSITIVE_COMPACT_NAMES)


def _find_sensitive_argument_path(value: JsonValue, path: tuple[str | int, ...] = ()) -> tuple[str | int, ...] | None:
    if isinstance(value, dict):
        for key, item in value.items():
            item_path = (*path, key)
            if _is_sensitive_argument_name(key):
                return item_path
            nested = _find_sensitive_argument_path(item, item_path)
            if nested is not None:
                return nested
    elif isinstance(value, list):
        for index, item in enumerate(value):
            nested = _find_sensitive_argument_path(item, (*path, index))
            if nested is not None:
                return nested
    return None


class CapabilityPluginGrant(BaseModel):
    """One ordered root-agent capability grant from a plugin manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    name: str = Field(min_length=1)
    arguments: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("capability grant names must not contain surrounding whitespace")
        return value

    @field_validator("arguments")
    @classmethod
    def _reject_secret_arguments(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        non_finite_path = _find_non_finite_number_path(value)
        if non_finite_path is not None:
            raise ValueError(f"capability plugin arguments require a finite number at {non_finite_path!r}")
        sensitive_path = _find_sensitive_argument_path(value)
        if sensitive_path is not None:
            raise ValueError(f"capability plugin arguments must not contain secrets or credentials: {sensitive_path!r}")
        return value


class CapabilityPluginManifest(BaseModel):
    """Versioned selection and root-grant policy for installed capability plugins."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    entry_points: tuple[str, ...] = ()
    capabilities: tuple[CapabilityPluginGrant, ...] = ()

    @field_validator("schema_version", mode="before")
    @classmethod
    def _validate_schema_version(cls, value: Any) -> Any:
        if not isinstance(value, int) or isinstance(value, bool) or value != CAPABILITY_PLUGIN_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be integer {CAPABILITY_PLUGIN_SCHEMA_VERSION}")
        return value

    @field_validator("entry_points")
    @classmethod
    def _validate_entry_points(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        invalid = [value for value in values if not value or value != value.strip()]
        if invalid:
            raise ValueError("entry-point names must be non-empty and contain no surrounding whitespace")
        duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
        if duplicates:
            raise ValueError(f"entry-point names must be unique: {duplicates!r}")
        return values

    @model_validator(mode="after")
    def _validate_root_grants_are_selected(self) -> Self:
        selected = set(self.entry_points)
        unselected = sorted({grant.name for grant in self.capabilities if grant.name not in selected})
        if unselected:
            raise ValueError(f"root capability grants must reference selected entry points: {unselected!r}")
        return self


def _validated_manifest_copy(manifest: CapabilityPluginManifest) -> CapabilityPluginManifest:
    return CapabilityPluginManifest.model_validate(copy.deepcopy(manifest.model_dump(mode="python")))


def _validate_grant_argument_shapes(
    manifest: CapabilityPluginManifest,
    catalog: CapabilityCatalog,
) -> None:
    default_from_spec = AbstractCapability.from_spec.__func__
    for index, grant in enumerate(manifest.capabilities):
        capability_type = catalog[grant.name]
        from_spec = capability_type.from_spec
        from_spec_function = getattr(from_spec, "__func__", from_spec)
        factory = capability_type if from_spec_function is default_from_spec else from_spec
        try:
            inspect.signature(factory).bind(**grant.arguments)
        except TypeError as exc:
            raise ValueError(
                f"Capability plugin grant {index} for {grant.name!r} has invalid arguments: {exc}"
            ) from exc


@dataclass(frozen=True, slots=True, init=False)
class ResolvedCapabilityPlugins:
    """One immutable catalog snapshot and its ordered root-agent grants."""

    _manifest: CapabilityPluginManifest = field(repr=False)
    catalog: CapabilityCatalog
    _root_agent_spec: AgentSpec = field(init=False, repr=False, compare=False)

    def __init__(
        self,
        manifest: CapabilityPluginManifest,
        catalog: CapabilityCatalog,
    ) -> None:
        validated_manifest = _validated_manifest_copy(manifest)
        object.__setattr__(self, "_manifest", validated_manifest)
        object.__setattr__(self, "catalog", catalog)
        selected = set(validated_manifest.entry_points)
        resolved_entry_points = {
            registration.serialization_name
            for registration in catalog.registrations
            if registration.provenance.source_kind == "entry_point"
        }
        if selected != resolved_entry_points:
            raise ValueError("Capability plugin manifest and catalog contain different selected entry points")
        _validate_grant_argument_shapes(validated_manifest, catalog)
        root_agent_spec = AgentSpec.model_validate({
            "capabilities": [grant.model_dump(mode="python") for grant in validated_manifest.capabilities]
        })
        object.__setattr__(self, "_root_agent_spec", root_agent_spec)

    @property
    def manifest(self) -> CapabilityPluginManifest:
        """Return an isolated copy of the canonical validated manifest."""
        return _validated_manifest_copy(self._manifest)

    @property
    def custom_capability_types(self) -> tuple[CapabilityType, ...]:
        """Return the exact custom-type tuple for native agent construction."""
        return self.catalog.custom_capability_types

    @property
    def root_agent_spec(self) -> AgentSpec:
        """Return a new native spec containing only configured root grants."""
        return self._root_agent_spec.model_copy(deep=True)

    def apply_to_root_agent_spec(self, agent_spec: AgentSpec | None = None) -> AgentSpec:
        """Append configured root grants to a copy of a native root-agent spec."""
        raw = (agent_spec or AgentSpec()).model_dump(mode="python")
        raw["capabilities"] = [
            *raw["capabilities"],
            *self._root_agent_spec.model_dump(mode="python")["capabilities"],
        ]
        return AgentSpec.model_validate(raw)


def load_capability_plugin_manifest(
    path: str | os.PathLike[str],
) -> CapabilityPluginManifest:
    """Read and strictly validate one versioned TOML plugin manifest."""
    manifest_path = Path(path)
    try:
        with manifest_path.open("rb") as manifest_file:
            raw = tomllib.load(manifest_file)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid TOML in capability plugin manifest {manifest_path}: {exc}") from exc

    try:
        return CapabilityPluginManifest.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"Invalid capability plugin manifest {manifest_path}: {exc}") from exc


def resolve_capability_plugins(
    manifest: CapabilityPluginManifest,
    *,
    explicit_types: Iterable[CapabilityType] = (),
) -> ResolvedCapabilityPlugins:
    """Resolve one validated manifest into the SDK's default capability catalog."""
    validated_manifest = _validated_manifest_copy(manifest)
    catalog = build_default_capability_catalog(
        explicit_types=explicit_types,
        selected_entry_points=validated_manifest.entry_points,
    )
    return ResolvedCapabilityPlugins(manifest=validated_manifest, catalog=catalog)


def load_capability_plugins(
    path: str | os.PathLike[str],
    *,
    explicit_types: Iterable[CapabilityType] = (),
) -> ResolvedCapabilityPlugins:
    """Load a TOML manifest and resolve its explicitly selected plugin types."""
    return resolve_capability_plugins(
        load_capability_plugin_manifest(path),
        explicit_types=explicit_types,
    )
