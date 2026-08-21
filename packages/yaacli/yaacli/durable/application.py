"""Frontend-neutral application service for durable YAACLI sessions."""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from typing import Protocol

from pydantic import JsonValue

from yaacli.durable.models import (
    ActionBatch,
    InputPriority,
    InputRecord,
    LogicalRunRecord,
    RevisionPayload,
    RevisionRecord,
    SessionRecord,
    SessionStatus,
    SessionSummary,
    StartRunRequest,
)
from yaacli.durable.store import SessionStore


class SessionExecutionCoordinator(Protocol):
    def start(self, execution_id: str) -> None: ...

    def notify_input(self, logical_run_id: str) -> None: ...

    def notify_action(self, logical_run_id: str) -> None: ...

    async def wait(self, logical_run_id: str) -> LogicalRunRecord: ...

    def accept_cancel(self, logical_run_id: str, reason: str) -> None: ...

    async def cancel(self, logical_run_id: str, reason: str) -> None: ...


class SessionApplicationService:
    """Single product boundary used by TUI and headless frontends."""

    def __init__(
        self,
        store: SessionStore,
        coordinator: SessionExecutionCoordinator | None = None,
    ) -> None:
        self.store = store
        self.coordinator = coordinator

    def create_session(
        self,
        workspace_ref: str,
        *,
        session_id: str | None = None,
    ) -> SessionRecord:
        return self.store.create_session(workspace_ref, session_id=session_id)

    def get_session(self, session_id: str) -> SessionRecord:
        session = self.store.get_session(session_id)
        if session is None or session.status is not SessionStatus.active:
            raise KeyError(session_id)
        return session

    def resolve_session(self, session_id_or_prefix: str) -> SessionRecord:
        """Resolve an active durable session by exact ID or unambiguous prefix."""
        exact = self.store.get_session(session_id_or_prefix)
        if exact is not None and exact.status is SessionStatus.active:
            return exact
        matches = [
            session
            for session in self.store.list_sessions(limit=1000)
            if session.session_id.startswith(session_id_or_prefix)
        ]
        if not matches:
            raise ValueError(f"Session not found: {session_id_or_prefix}")
        if len(matches) > 1:
            choices = ", ".join(session.session_id for session in matches[:10])
            raise ValueError(f"Ambiguous session prefix: {choices}")
        return matches[0]

    def list_session_summaries(self, *, limit: int = 100) -> tuple[SessionSummary, ...]:
        """Return bounded durable projections for CLI and TUI session browsers."""
        return tuple(self._summarize(session) for session in self.store.list_sessions(limit=limit))

    def get_session_summary(self, session_id_or_prefix: str) -> SessionSummary:
        return self._summarize(self.resolve_session(session_id_or_prefix))

    def delete_session(self, session_id_or_prefix: str) -> SessionSummary:
        """Tombstone one inactive session and return its final active projection."""
        session = self.resolve_session(session_id_or_prefix)
        summary = self._summarize(session)
        self.store.tombstone_session(session.session_id)
        return summary

    def _summarize(self, session: SessionRecord) -> SessionSummary:
        revision = self.store.get_revision(session.head_revision_id) if session.head_revision_id is not None else None
        input_preview: str | None = None
        output_preview: str | None = None
        model: str | None = None
        model_profile_id: str | None = None
        if revision is not None:
            run = self.store.get_run(revision.logical_run_id)
            if run is not None:
                inputs = self.store.list_inputs(run.logical_run_id)
                if inputs:
                    input_preview = _preview_json_values(inputs[0].content)
                model = run.model
                model_profile_id = run.model_profile_id
            output_preview = _preview_json_value(revision.terminal.get("output"))
        return SessionSummary(
            session_id=session.session_id,
            workspace_ref=session.workspace_ref,
            status=session.status,
            head_revision_id=session.head_revision_id,
            created_at=session.created_at,
            updated_at=session.updated_at,
            input_preview=input_preview,
            output_preview=output_preview,
            message_count=len(revision.message_history) if revision is not None else 0,
            display_event_count=len(revision.display_projection) if revision is not None else 0,
            model=model,
            model_profile_id=model_profile_id,
        )

    def _require_coordinator(self) -> SessionExecutionCoordinator:
        if self.coordinator is None:
            raise RuntimeError("This session service has no execution coordinator")
        return self.coordinator

    def import_snapshot(
        self,
        session_id: str,
        *,
        payload: RevisionPayload,
        source: str,
        model: str | None = None,
        model_profile_id: str | None = None,
    ) -> RevisionRecord:
        self.get_session(session_id)
        return self.store.import_revision(
            session_id,
            payload=payload,
            source=source,
            model=model,
            model_profile_id=model_profile_id,
        )

    def accept_turn(
        self,
        session_id: str,
        content: Sequence[JsonValue],
        *,
        model: str | None = None,
        model_profile_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> LogicalRunRecord:
        """Durably accept one turn without coupling acknowledgement to dispatch."""
        session = self.get_session(session_id)
        return self.store.start_run(
            StartRunRequest(
                session_id=session_id,
                expected_head_revision_id=session.head_revision_id,
                idempotency_key=idempotency_key or str(uuid.uuid4()),
                initial_content=list(content),
                model=model,
                model_profile_id=model_profile_id,
            )
        )

    async def start_turn(
        self,
        session_id: str,
        content: Sequence[JsonValue],
        *,
        model: str | None = None,
        model_profile_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> LogicalRunRecord:
        run = self.accept_turn(
            session_id,
            content,
            model=model,
            model_profile_id=model_profile_id,
            idempotency_key=idempotency_key,
        )
        self._require_coordinator().start(run.execution_id)
        return run

    def start(self, logical_run_id: str) -> None:
        """Start one already accepted run in the current process."""
        run = self.store.get_run(logical_run_id)
        if run is None:
            raise KeyError(logical_run_id)
        self._require_coordinator().start(run.execution_id)

    async def wait(self, logical_run_id: str) -> LogicalRunRecord:
        """Wait for one logical run through the configured execution backend."""
        return await self._require_coordinator().wait(logical_run_id)

    async def run_turn(
        self,
        session_id: str,
        content: Sequence[JsonValue],
        *,
        model: str | None = None,
        model_profile_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> RevisionRecord:
        run = self.accept_turn(
            session_id,
            content,
            model=model,
            model_profile_id=model_profile_id,
            idempotency_key=idempotency_key,
        )
        coordinator = self._require_coordinator()
        coordinator.start(run.execution_id)
        await coordinator.wait(run.logical_run_id)
        revision = self.store.get_revision_for_run(run.logical_run_id)
        if revision is None:
            raise RuntimeError(f"Logical run {run.logical_run_id!r} terminated without publishing a revision")
        return revision

    def accept_input(
        self,
        logical_run_id: str,
        content: Sequence[JsonValue],
        *,
        idempotency_key: str | None = None,
        priority: InputPriority = InputPriority.asap,
        origin: str = "user",
    ) -> InputRecord:
        """Durably accept active-run input before the caller acknowledges it."""
        return self.store.accept_input(
            logical_run_id,
            content,
            idempotency_key=idempotency_key or str(uuid.uuid4()),
            priority=priority,
            origin=origin,
        )

    def notify_input(self, logical_run_id: str) -> None:
        self._require_coordinator().notify_input(logical_run_id)

    async def submit_input(
        self,
        logical_run_id: str,
        content: Sequence[JsonValue],
        *,
        idempotency_key: str | None = None,
        priority: InputPriority = InputPriority.asap,
        origin: str = "user",
    ) -> InputRecord:
        record = self.accept_input(
            logical_run_id,
            content,
            idempotency_key=idempotency_key,
            priority=priority,
            origin=origin,
        )
        self._require_coordinator().notify_input(logical_run_id)
        return record

    def accept_action(
        self,
        action_item_id: str,
        decision: dict[str, JsonValue],
        *,
        decision_id: str | None = None,
        actor: str | None = None,
    ) -> ActionBatch:
        """Durably accept one action decision without coupling it to dispatch."""
        return self.store.decide_action(
            action_item_id,
            decision_id=decision_id or str(uuid.uuid4()),
            decision=decision,
            actor=actor,
        )

    def notify_action(self, logical_run_id: str) -> None:
        self._require_coordinator().notify_action(logical_run_id)

    async def decide_action(
        self,
        action_item_id: str,
        decision: dict[str, JsonValue],
        *,
        decision_id: str | None = None,
        actor: str | None = None,
    ) -> ActionBatch:
        batch = self.accept_action(
            action_item_id,
            decision,
            decision_id=decision_id,
            actor=actor,
        )
        self._require_coordinator().notify_action(batch.logical_run_id)
        return batch

    def accept_cancel(self, logical_run_id: str, *, reason: str = "cancelled") -> None:
        self._require_coordinator().accept_cancel(logical_run_id, reason)

    async def cancel(self, logical_run_id: str, *, reason: str = "cancelled") -> None:
        await self._require_coordinator().cancel(logical_run_id, reason)


def _preview_json_values(values: Sequence[JsonValue]) -> str | None:
    if len(values) == 1:
        return _preview_json_value(values[0])
    return _bounded_preview(json.dumps(list(values), ensure_ascii=False))


def _preview_json_value(value: JsonValue | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return _bounded_preview(value)
    return _bounded_preview(json.dumps(value, ensure_ascii=False))


def _bounded_preview(value: str, *, limit: int = 2000) -> str:
    normalized = value.strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."
