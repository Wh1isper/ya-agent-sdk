from __future__ import annotations

from typing import Any, Protocol, runtime_checkable
from xml.etree.ElementTree import Element, SubElement, tostring

from loguru import logger
from pydantic import BaseModel, Field
from pydantic_ai.tools import RunContext
from ya_agent_sdk.context import ENVIRONMENT_CONTEXT_TAG, RUNTIME_CONTEXT_TAG, AgentContext

from ya_claw.workspace import WorkspaceBinding

ASYNC_SUBAGENTS_CONTEXT_TAG = "async-subagents"
CLAW_INJECTED_CONTEXT_TAGS = (
    RUNTIME_CONTEXT_TAG,
    ENVIRONMENT_CONTEXT_TAG,
    ASYNC_SUBAGENTS_CONTEXT_TAG,
)
_CLAW_SELF_CLIENT_KEY = "claw_self_client"


@runtime_checkable
class AsyncSubagentListClient(Protocol):
    async def list_async_subagents(self, *, include_terminal: bool) -> dict[str, Any]: ...


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
    is_async_subagent: bool = False
    profile_name: str | None = None
    restore_from_run_id: str | None = None
    dispatch_mode: str | None = None
    container_id: str | None = None
    workspace_binding: ClawWorkspaceBindingSnapshot | None = None
    source_kind: str | None = None
    source_metadata: dict[str, Any] = Field(default_factory=dict)
    claw_metadata: dict[str, Any] = Field(default_factory=dict)

    async def get_context_instructions(
        self,
        run_context: RunContext[AgentContext] | None = None,
        *,
        is_user_prompt: bool = True,
    ) -> str:
        instructions = await super().get_context_instructions(run_context, is_user_prompt=is_user_prompt)
        if not is_user_prompt or self.is_async_subagent:
            return instructions
        async_subagents_context = await self._fetch_async_subagents_context()
        if async_subagents_context is None:
            return instructions
        return f"{instructions}\n\n{async_subagents_context}" if instructions else async_subagents_context

    async def _fetch_async_subagents_context(self) -> str | None:
        if self.resources is None:
            return None
        resource = self.resources.get(_CLAW_SELF_CLIENT_KEY)
        if not isinstance(resource, AsyncSubagentListClient):
            return None
        try:
            payload = await resource.list_async_subagents(include_terminal=True)
        except Exception as exc:
            logger.debug("Failed to fetch async subagents context session_id={} error={}", self.session_id, exc)
            return None
        return _async_subagents_context_from_payload(payload, session_id=self.session_id)

    def get_wrapper_metadata(self) -> dict[str, Any]:
        return {
            **super().get_wrapper_metadata(),
            "session_id": self.session_id,
            "claw_run_id": self.claw_run_id,
            "profile_name": self.profile_name,
            "container_id": self.container_id,
            "is_async_subagent": self.is_async_subagent,
        }


def _async_subagents_context_from_payload(payload: dict[str, Any], *, session_id: str | None) -> str | None:
    raw_items = payload.get("subagents")
    if not isinstance(raw_items, list) or not raw_items:
        return None

    root = Element(ASYNC_SUBAGENTS_CONTEXT_TAG)
    if isinstance(session_id, str) and session_id.strip() != "":
        root.set("session-id", session_id)
    root.set(
        "hint",
        "Use list_async_subagents or get_async_subagent for details; use steer_async_subagent for active child runs.",
    )
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        child = SubElement(root, "subagent")
        for source_key, attr_name in (
            ("name", "name"),
            ("task_session_id", "session-id"),
            ("status", "status"),
        ):
            value = item.get(source_key)
            if value is None:
                continue
            text = str(value).strip()
            if text != "":
                child.set(attr_name, text[:1000])
    if len(root) == 0:
        return None
    return tostring(root, encoding="unicode")
