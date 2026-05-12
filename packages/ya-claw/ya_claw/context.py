from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from ya_agent_sdk.context import AgentContext

from ya_claw.workspace import WorkspaceBinding


class ClawWorkspaceMountSnapshot(BaseModel):
    id: str
    name: str | None = None
    virtual_path: str
    mode: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClawWorkspaceBindingSnapshot(BaseModel):
    virtual_path: str
    cwd: str
    readable_paths: list[str] = Field(default_factory=list)
    writable_paths: list[str] = Field(default_factory=list)
    mounts: list[ClawWorkspaceMountSnapshot] = Field(default_factory=list)
    default_mount_id: str | None = None
    fingerprint: str | None = None
    generation: int | None = None
    sandbox_scope: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    backend_hint: str | None = None

    @classmethod
    def from_binding(cls, binding: WorkspaceBinding) -> ClawWorkspaceBindingSnapshot:
        return cls(
            virtual_path=str(binding.virtual_path),
            cwd=str(binding.cwd),
            readable_paths=[str(path) for path in binding.readable_paths],
            writable_paths=[str(path) for path in binding.writable_paths],
            mounts=[
                ClawWorkspaceMountSnapshot(
                    id=mount.id,
                    name=mount.name,
                    virtual_path=str(mount.virtual_path),
                    mode=mount.mode,
                    metadata=dict(mount.metadata),
                )
                for mount in binding.mounts
            ],
            default_mount_id=binding.default_mount.id if binding.mounts else None,
            fingerprint=binding.fingerprint,
            generation=binding.generation,
            sandbox_scope=binding.sandbox_scope,
            metadata=dict(binding.metadata),
            backend_hint=binding.backend_hint,
        )


class ClawAgentContext(AgentContext):
    session_id: str | None = None
    claw_run_id: str | None = None
    profile_name: str | None = None
    restore_from_run_id: str | None = None
    dispatch_mode: str | None = None
    container_id: str | None = None
    workspace_binding: ClawWorkspaceBindingSnapshot | None = None
    source_kind: str | None = None
    source_metadata: dict[str, Any] = Field(default_factory=dict)
    claw_metadata: dict[str, Any] = Field(default_factory=dict)

    def get_wrapper_metadata(self) -> dict[str, Any]:
        return {
            **super().get_wrapper_metadata(),
            "session_id": self.session_id,
            "claw_run_id": self.claw_run_id,
            "profile_name": self.profile_name,
            "container_id": self.container_id,
        }
