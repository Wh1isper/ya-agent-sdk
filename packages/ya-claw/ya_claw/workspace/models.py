from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path, PurePath
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from ya_agent_sdk.environment.virtual_path import (
    VirtualPath,
    VirtualPathLike,
    is_virtual_path_relative_to,
)
from ya_agent_sdk.environment.virtual_path import (
    normalize_virtual_path as normalize_agent_virtual_path,
)
from ya_agent_sdk.environment.virtual_path import (
    relative_virtual_path as agent_relative_virtual_path,
)

WORKSPACE_METADATA_KEY = "workspace"
WORKSPACE_SNAPSHOT_METADATA_KEY = "workspace_snapshot"
SANDBOX_METADATA_KEY = "sandbox"
SANDBOX_SCOPE_SESSION = "session"
SANDBOX_SCOPE_RUN = "run"
DEFAULT_WORKSPACE_MOUNT_LIMIT = 8

WorkspaceMountMode = Literal["rw", "ro"]
SandboxRetentionPolicy = Literal["stop_on_idle", "keep_warm"]
SandboxScopeLiteral = Literal["session", "run"]
_MOUNT_ID_PATTERN = re.compile(r"[^a-zA-Z0-9_.-]+")


class SandboxScope(StrEnum):
    SESSION = SANDBOX_SCOPE_SESSION
    RUN = SANDBOX_SCOPE_RUN


class WorkspaceMountSpec(BaseModel):
    id: str | None = None
    name: str | None = None
    host_path: str
    virtual_path: str
    mode: WorkspaceMountMode = "rw"
    docker_host_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "name", "host_path", "virtual_path", "docker_host_path", mode="before")
    @classmethod
    def _strip_optional_string(cls, value: object) -> str | None:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return None

    @field_validator("host_path", "virtual_path")
    @classmethod
    def _require_string(cls, value: str | None) -> str:
        if value is None or value.strip() == "":
            raise ValueError("workspace mount paths must be non-empty strings")
        return value.strip()


class WorkspaceBindingSpec(BaseModel):
    mounts: list[WorkspaceMountSpec]
    default_mount_id: str | None = None
    cwd: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("default_mount_id", "cwd", mode="before")
    @classmethod
    def _strip_optional_string(cls, value: object) -> str | None:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return None

    @model_validator(mode="after")
    def _validate_binding(self) -> WorkspaceBindingSpec:  # noqa: C901
        if len(self.mounts) == 0:
            raise ValueError("workspace must declare at least one mount")
        if len(self.mounts) > DEFAULT_WORKSPACE_MOUNT_LIMIT:
            raise ValueError(f"workspace mounts exceed limit {DEFAULT_WORKSPACE_MOUNT_LIMIT}")

        normalized_mounts: list[WorkspaceMountSpec] = []
        seen_ids: set[str] = set()
        seen_virtual_paths: set[str] = set()
        for mount in self.mounts:
            virtual_path = normalize_virtual_path(mount.virtual_path)
            mount_id = mount.id or derive_mount_id(virtual_path)
            if mount_id in seen_ids:
                raise ValueError(f"workspace mount id '{mount_id}' is duplicated")
            if virtual_path in seen_virtual_paths:
                raise ValueError(f"workspace virtual path '{virtual_path}' is duplicated")
            seen_ids.add(mount_id)
            seen_virtual_paths.add(virtual_path)
            normalized_mounts.append(mount.model_copy(update={"id": mount_id, "virtual_path": virtual_path}))

        default_mount_id = self.default_mount_id
        if default_mount_id is None and len(normalized_mounts) == 1:
            default_mount_id = normalized_mounts[0].id
        if default_mount_id is None:
            raise ValueError("workspace.default_mount_id is required when multiple mounts are declared")
        if default_mount_id not in seen_ids:
            raise ValueError(f"workspace.default_mount_id '{default_mount_id}' does not match a declared mount")

        cwd = normalize_virtual_path(self.cwd) if self.cwd is not None else None
        if cwd is None:
            default_mount = next(mount for mount in normalized_mounts if mount.id == default_mount_id)
            cwd = default_mount.virtual_path
        if not any(virtual_path_contains(mount.virtual_path, cwd) for mount in normalized_mounts):
            raise ValueError("workspace.cwd must be within a declared virtual mount")

        self.mounts = normalized_mounts
        self.default_mount_id = default_mount_id
        self.cwd = cwd
        return self


class SandboxState(BaseModel):
    provider: Literal["docker"] = "docker"
    scope: SandboxScopeLiteral = SANDBOX_SCOPE_SESSION
    generation: int = 1
    workspace_fingerprint: str
    container_ref: str
    container_id: str | None = None
    image: str | None = None
    status: str = "created"
    retention_policy: SandboxRetentionPolicy = "stop_on_idle"
    idle_ttl_seconds: int = 3600
    cache_path: str | None = None
    last_started_at: str | None = None
    last_used_at: str | None = None


@dataclass(frozen=True, slots=True, init=False)
class WorkspaceMountBinding:
    id: str
    host_path: Path
    virtual_path: VirtualPath
    mode: WorkspaceMountMode = "rw"
    docker_host_path: Path | None = None
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        *,
        id: str,  # noqa: A002
        host_path: Path,
        virtual_path: VirtualPathLike,
        mode: WorkspaceMountMode = "rw",
        docker_host_path: Path | None = None,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "host_path", host_path)
        object.__setattr__(self, "virtual_path", normalize_agent_virtual_path(virtual_path))
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "docker_host_path", docker_host_path)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "metadata", dict(metadata or {}))


@dataclass(slots=True, init=False)
class WorkspaceBinding:
    host_path: Path
    virtual_path: VirtualPath
    cwd: VirtualPath
    readable_paths: list[VirtualPath]
    writable_paths: list[VirtualPath]
    mounts: list[WorkspaceMountBinding]
    fingerprint: str
    generation: int | None = None
    sandbox_scope: SandboxScopeLiteral | None = None
    docker_host_path: Path | None = None
    environment_overrides: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    backend_hint: str | None = None

    def __init__(
        self,
        *,
        host_path: Path,
        virtual_path: VirtualPathLike,
        cwd: VirtualPathLike,
        readable_paths: Sequence[VirtualPathLike],
        writable_paths: Sequence[VirtualPathLike],
        mounts: list[WorkspaceMountBinding],
        fingerprint: str,
        generation: int | None = None,
        sandbox_scope: SandboxScopeLiteral | None = None,
        docker_host_path: Path | None = None,
        environment_overrides: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
        backend_hint: str | None = None,
    ) -> None:
        self.host_path = host_path
        self.virtual_path = normalize_agent_virtual_path(virtual_path)
        self.cwd = normalize_agent_virtual_path(cwd)
        self.readable_paths = [normalize_agent_virtual_path(path) for path in readable_paths]
        self.writable_paths = [normalize_agent_virtual_path(path) for path in writable_paths]
        self.mounts = list(mounts)
        self.fingerprint = fingerprint
        self.generation = generation
        self.sandbox_scope = sandbox_scope
        self.docker_host_path = docker_host_path
        self.environment_overrides = dict(environment_overrides or {})
        self.metadata = dict(metadata or {})
        self.backend_hint = backend_hint

    @property
    def default_mount(self) -> WorkspaceMountBinding:
        for mount in self.mounts:
            if mount.host_path == self.host_path and mount.virtual_path == self.virtual_path:
                return mount
        return self.mounts[0]


def normalize_workspace_spec(value: WorkspaceBindingSpec | dict[str, Any] | None) -> WorkspaceBindingSpec | None:
    if value is None:
        return None
    if isinstance(value, WorkspaceBindingSpec):
        return value
    return WorkspaceBindingSpec.model_validate(value)


def workspace_metadata_payload(workspace: WorkspaceBindingSpec | dict[str, Any] | None) -> dict[str, Any] | None:
    spec = normalize_workspace_spec(workspace)
    if spec is None:
        return None
    return spec.model_dump(mode="json")


def metadata_with_workspace(
    metadata: dict[str, Any] | None,
    workspace: WorkspaceBindingSpec | dict[str, Any] | None,
) -> dict[str, Any]:
    normalized = dict(metadata or {})
    payload = workspace_metadata_payload(workspace)
    if payload is not None:
        normalized[WORKSPACE_METADATA_KEY] = payload
    return normalized


def extract_workspace_metadata(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(metadata, dict):
        return None
    raw_workspace = metadata.get(WORKSPACE_METADATA_KEY)
    if not isinstance(raw_workspace, dict):
        return None
    return workspace_metadata_payload(raw_workspace)


def merge_workspace_metadata(
    *,
    session_metadata: dict[str, Any] | None,
    run_metadata: dict[str, Any] | None,
) -> dict[str, Any] | None:
    session_workspace = extract_workspace_metadata(session_metadata)
    run_workspace = extract_workspace_metadata(run_metadata)
    return run_workspace or session_workspace


def derive_mount_id(virtual_path: str) -> str:
    leaf = virtual_path.rstrip("/").rsplit("/", 1)[-1] or "workspace"
    normalized = _MOUNT_ID_PATTERN.sub("-", leaf).strip("-._")
    return normalized or "workspace"


def normalize_virtual_path(value: str | None) -> str:
    if value is None or value.strip() == "":
        raise ValueError("virtual path must be a non-empty absolute path")
    normalized = normalize_agent_virtual_path(value.strip())
    if not normalized.is_absolute():
        raise ValueError("virtual path must be absolute")
    return normalized.as_posix()


def virtual_path_contains(parent: str | PurePath, child: str | PurePath) -> bool:
    return is_virtual_path_relative_to(child, parent)


def relative_virtual_path(parent: str | PurePath, child: str | PurePath) -> VirtualPath:
    parent_value = normalize_virtual_path(str(parent))
    child_value = normalize_virtual_path(str(child))
    if not virtual_path_contains(parent_value, child_value):
        raise ValueError(f"virtual path '{child_value}' is outside '{parent_value}'")
    return agent_relative_virtual_path(child_value, parent_value)


def workspace_fingerprint_payload(
    *,
    provider: str,
    workspace: WorkspaceBindingSpec,
    docker_image: str | None = None,
    workspace_uid: int | None = None,
    workspace_gid: int | None = None,
    extra_mounts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "provider": provider,
        "docker_image": docker_image,
        "workspace_uid": workspace_uid,
        "workspace_gid": workspace_gid,
        "extra_mounts": list(extra_mounts or []),
        "mounts": [
            {
                "id": mount.id,
                "host_path": str(Path(mount.host_path).expanduser()),
                "docker_host_path": str(Path(mount.docker_host_path).expanduser())
                if mount.docker_host_path is not None
                else None,
                "virtual_path": mount.virtual_path,
                "mode": mount.mode,
            }
            for mount in workspace.mounts
        ],
        "default_mount_id": workspace.default_mount_id,
        "cwd": workspace.cwd,
    }


def compute_workspace_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def workspace_snapshot(binding: WorkspaceBinding) -> dict[str, Any]:
    return {
        "fingerprint": binding.fingerprint,
        "generation": binding.generation,
        "sandbox_scope": binding.sandbox_scope,
        "backend": binding.backend_hint,
        "mounts": [
            {
                "id": mount.id,
                "name": mount.name,
                "host_path": str(mount.host_path),
                "docker_host_path": str(mount.docker_host_path) if mount.docker_host_path is not None else None,
                "virtual_path": str(mount.virtual_path),
                "mode": mount.mode,
                "metadata": dict(mount.metadata),
            }
            for mount in binding.mounts
        ],
        "default_mount_id": binding.default_mount.id,
        "cwd": str(binding.cwd),
        "readable_paths": [str(path) for path in binding.readable_paths],
        "writable_paths": [str(path) for path in binding.writable_paths],
        "metadata": dict(binding.metadata),
    }
