from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from loguru import logger
from pydantic_ai.messages import ModelMessagesTypeAdapter
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from ya_agent_environment import Environment
from ya_agent_sdk.agents.lifecycle import BaseLifecycleExtension, ContextHandoffCompleteContext, ContextHandoffSource
from ya_agent_sdk.agents.main import RuntimeReadyContext

from ya_claw.config import ClawSettings
from ya_claw.context import ClawAgentContext
from ya_claw.controller.models import DispatchMode, MemoryJobKind, TriggerType
from ya_claw.execution.state_machine import queue_run
from ya_claw.orm.tables import RunRecord, SessionMemoryStateRecord, SessionRecord
from ya_claw.runtime_state import InMemoryRuntimeState

MEMORY_CONTEXT_TAG = "memory-context"
MEMORY_MD_CONTEXT_TAG = "memory-md-context"
MEMORY_FILE_INDEX_TAG = "memory-file-index"
AUTO_TASK_CONTEXT_TAGS = ("heartbeat-guidance", "schedule-context", "heartbeat-context")
AGENCY_CONTEXT_TAGS = ("agency-context", "agency-index-context", "agency-action-log-context", "agency-file-index")
_MEMORY_EXCLUDED_TRIGGER_TYPES = {
    TriggerType.HEARTBEAT.value,
    TriggerType.SCHEDULE.value,
    TriggerType.MEMORY.value,
    TriggerType.AGENCY.value,
}
_MEMORY_TRIGGER_KEY = "memory_triggers"
_PENDING_MEMORY_REQUESTS_KEY = "pending_requests"


@dataclass(slots=True)
class MemoryRunRequest:
    source_session_id: str
    kind: MemoryJobKind
    reason: str
    source_run_ids: list[str] = field(default_factory=list)
    source_sequence_start: int | None = None
    source_sequence_end: int | None = None
    trigger_payload: dict[str, Any] | None = None
    source_identity: dict[str, Any] | None = None


class ClawMemoryExtension(BaseLifecycleExtension[ClawAgentContext, Environment]):
    """SDK lifecycle extension for workspace-native YA Claw memory."""

    name = "ya_claw_memory"

    def __init__(
        self,
        *,
        settings: ClawSettings,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
    ) -> None:
        self._settings = settings
        self._session_factory = session_factory

    async def on_runtime_ready(self, ctx: RuntimeReadyContext[ClawAgentContext, Any, Environment]) -> None:
        try:
            await self._on_runtime_ready(ctx)
        except Exception:
            logger.exception("YA Claw memory context injection failed")

    async def on_context_handoff_complete(self, ctx: ContextHandoffCompleteContext[ClawAgentContext]) -> None:
        try:
            self._on_context_handoff_complete(ctx)
        except Exception:
            logger.exception("YA Claw memory context handoff capture failed")

    async def _on_runtime_ready(self, ctx: RuntimeReadyContext[ClawAgentContext, Any, Environment]) -> None:
        runtime_ctx = getattr(ctx.runtime, "ctx", None)
        if runtime_ctx is None:
            return
        source_kind = getattr(runtime_ctx, "source_kind", None)
        existing_tags = runtime_ctx.injected_context_tags
        if source_kind in {TriggerType.HEARTBEAT.value, TriggerType.SCHEDULE.value}:
            missing_auto_tags = [tag for tag in AUTO_TASK_CONTEXT_TAGS if tag not in existing_tags]
            if missing_auto_tags:
                runtime_ctx.injected_context_tags = (*existing_tags, *missing_auto_tags)
            return
        if source_kind == TriggerType.AGENCY.value:
            missing_agency_tags = [
                tag
                for tag in (*AGENCY_CONTEXT_TAGS, MEMORY_MD_CONTEXT_TAG, MEMORY_FILE_INDEX_TAG)
                if tag not in existing_tags
            ]
            if missing_agency_tags:
                runtime_ctx.injected_context_tags = (*existing_tags, *missing_agency_tags)
            return
        if not self._settings.memory_enabled or not self._settings.memory_inject_enabled:
            return
        if source_kind == TriggerType.MEMORY.value:
            return
        missing_tags = [
            tag
            for tag in (MEMORY_CONTEXT_TAG, MEMORY_MD_CONTEXT_TAG, MEMORY_FILE_INDEX_TAG, *AUTO_TASK_CONTEXT_TAGS)
            if tag not in runtime_ctx.injected_context_tags
        ]
        if missing_tags:
            runtime_ctx.injected_context_tags = (*runtime_ctx.injected_context_tags, *missing_tags)

    def _on_context_handoff_complete(self, ctx: ContextHandoffCompleteContext[ClawAgentContext]) -> None:
        if not self._settings.memory_enabled:
            return
        deps = ctx.deps
        if getattr(deps, "source_kind", None) in _MEMORY_EXCLUDED_TRIGGER_TYPES:
            return
        if ctx.source == ContextHandoffSource.COMPACT and not self._settings.memory_extract_on_compact:
            return
        if ctx.source == ContextHandoffSource.SUMMARIZE_TOOL and not self._settings.memory_extract_on_summarize:
            return
        claw_metadata = getattr(deps, "claw_metadata", None)
        source_session_id = getattr(deps, "session_id", None)
        source_run_id = getattr(deps, "claw_run_id", None)
        if not isinstance(claw_metadata, dict) or not isinstance(source_session_id, str):
            return
        trigger_payload = {
            "reason": f"{ctx.source.value}_handoff",
            "event_id": ctx.event_id,
            "source": ctx.source.value,
            "source_session_id": source_session_id,
            "source_run_ids": [source_run_id] if isinstance(source_run_id, str) else [],
            "summary_markdown": ctx.summary_markdown,
            "trimmed_messages": ModelMessagesTypeAdapter.dump_python(ctx.trimmed_messages, mode="json"),
            "handoff_messages": ModelMessagesTypeAdapter.dump_python(ctx.handoff_messages, mode="json"),
        }
        triggers = claw_metadata.setdefault(_MEMORY_TRIGGER_KEY, [])
        if isinstance(triggers, list):
            triggers.append(trigger_payload)


class MemoryLifecycle:
    """Coordinates workspace memory extract and summary background runs."""

    def __init__(
        self,
        *,
        settings: ClawSettings,
        session_factory: async_sessionmaker[AsyncSession],
        runtime_state: InMemoryRuntimeState,
        submit_run: Callable[[str], bool] | None = None,
    ) -> None:
        self._settings = settings
        self._session_factory = session_factory
        self._runtime_state = runtime_state
        self._submit_run = submit_run

    async def on_run_committed(
        self,
        *,
        source_session_id: str,
        source_run_id: str,
        source_sequence_no: int,
        profile_name: str | None,
        claw_metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        if (
            not self._settings.memory_enabled
            or _trigger_type_from_metadata(claw_metadata) in _MEMORY_EXCLUDED_TRIGGER_TYPES
        ):
            return []
        queued_run_ids: list[str] = []
        async with self._session_factory() as db_session:
            source_session = await db_session.get(SessionRecord, source_session_id)
            if not isinstance(source_session, SessionRecord) or source_session.session_type != "conversation":
                return []
            state = await self._ensure_state(db_session, source_session)
            if not state.enabled:
                await db_session.commit()
                return []
            if source_sequence_no > state.last_extracted_sequence_no:
                state.turns_since_extract += 1

            extract_triggered = False
            for trigger in _extract_memory_triggers(claw_metadata):
                run_id = await self._enqueue_or_mark_pending(
                    db_session,
                    state,
                    source_session,
                    MemoryRunRequest(
                        source_session_id=source_session_id,
                        kind=MemoryJobKind.EXTRACT,
                        reason=str(trigger.get("reason") or "context_handoff"),
                        source_run_ids=[item for item in trigger.get("source_run_ids", []) if isinstance(item, str)]
                        or [source_run_id],
                        source_sequence_start=source_sequence_no,
                        source_sequence_end=source_sequence_no,
                        trigger_payload=trigger,
                    ),
                )
                state.turns_since_extract = 0
                extract_triggered = True
                if run_id is not None:
                    queued_run_ids.append(run_id)

            if not extract_triggered and state.turns_since_extract >= max(1, self._settings.memory_extract_every_turns):
                run_id = await self._enqueue_or_mark_pending(
                    db_session,
                    state,
                    source_session,
                    MemoryRunRequest(
                        source_session_id=source_session_id,
                        kind=MemoryJobKind.EXTRACT,
                        reason="turn_threshold",
                        source_run_ids=[source_run_id],
                        source_sequence_start=state.last_extracted_sequence_no + 1,
                        source_sequence_end=source_sequence_no,
                    ),
                )
                state.turns_since_extract = 0
                if run_id is not None:
                    queued_run_ids.append(run_id)
            await db_session.commit()

        self._submit_all(queued_run_ids)
        return queued_run_ids

    async def on_memory_run_committed(self, *, memory_run_id: str) -> list[str]:
        if not self._settings.memory_enabled:
            return []
        queued_run_ids: list[str] = []
        async with self._session_factory() as db_session:
            scope = await self._load_memory_run_scope(db_session, memory_run_id)
            if scope is None:
                return []
            _run, source_session, state, kind, memory = scope
            if kind == MemoryJobKind.EXTRACT:
                state.last_extracted_sequence_no = max(
                    state.last_extracted_sequence_no,
                    _int_or_zero(memory.get("source_sequence_end")),
                )
                state.turns_since_extract = 0
                state.extract_count += 1
                state.extracts_since_summary += 1
                state.pending_extract = False
                state.last_extract_run_id = memory_run_id
                if state.extracts_since_summary >= max(1, self._settings.memory_summary_every_extracts):
                    run_id = await self._enqueue_or_mark_pending(
                        db_session,
                        state,
                        source_session,
                        MemoryRunRequest(
                            source_session_id=source_session.id,
                            kind=MemoryJobKind.SUMMARY,
                            reason="extract_threshold",
                        ),
                    )
                    if run_id is not None:
                        queued_run_ids.append(run_id)
            elif kind == MemoryJobKind.SUMMARY:
                state.extracts_since_summary = 0
                state.pending_summary = False
                state.last_summary_run_id = memory_run_id

            run_id = await self._enqueue_next_pending(db_session, state, source_session)
            if run_id is not None:
                queued_run_ids.append(run_id)

            await db_session.commit()

        self._submit_all(queued_run_ids)
        return queued_run_ids

    async def on_memory_run_terminal(self, *, memory_run_id: str) -> list[str]:
        if not self._settings.memory_enabled:
            return []
        queued_run_ids: list[str] = []
        async with self._session_factory() as db_session:
            scope = await self._load_memory_run_scope(db_session, memory_run_id)
            if scope is None:
                return []
            _run, source_session, state, _kind, _memory = scope
            run_id = await self._enqueue_next_pending(db_session, state, source_session)
            if run_id is not None:
                queued_run_ids.append(run_id)
            await db_session.commit()
        self._submit_all(queued_run_ids)
        return queued_run_ids

    async def enqueue_manual_extract(
        self,
        *,
        source_session_id: str,
        reason: str = "manual_extract",
        source_run_ids: list[str] | None = None,
    ) -> str | None:
        if not self._settings.memory_enabled:
            raise ValueError("Session memory is disabled.")
        async with self._session_factory() as db_session:
            source_session = await db_session.get(SessionRecord, source_session_id)
            if not isinstance(source_session, SessionRecord):
                raise TypeError(f"Session '{source_session_id}' was not found.")
            _validate_source_conversation_session(source_session)
            state = await self._ensure_state(db_session, source_session)
            if source_run_ids:
                matched_run_ids, sequence_start, sequence_end = await _matched_run_ids_and_sequence_range(
                    db_session, source_session_id, source_run_ids
                )
                if len(matched_run_ids) != len(set(source_run_ids)):
                    missing = sorted(set(source_run_ids) - set(matched_run_ids))
                    raise ValueError(f"Requested run_ids did not match completed source runs: {missing}")
                request_run_ids = matched_run_ids
            else:
                sequence_start, sequence_end = await _sequence_range_for_run_ids(db_session, source_session_id, None)
                request_run_ids = []
            run_id = await self._enqueue_or_mark_pending(
                db_session,
                state,
                source_session,
                MemoryRunRequest(
                    source_session_id=source_session_id,
                    kind=MemoryJobKind.EXTRACT,
                    reason=reason,
                    source_run_ids=request_run_ids,
                    source_sequence_start=sequence_start,
                    source_sequence_end=sequence_end,
                ),
            )
            state.turns_since_extract = 0
            await db_session.commit()
        if run_id is not None:
            self._submit_all([run_id])
        return run_id

    async def enqueue_manual_summary(self, *, source_session_id: str, reason: str = "manual_summary") -> str | None:
        if not self._settings.memory_enabled:
            raise ValueError("Session memory is disabled.")
        async with self._session_factory() as db_session:
            source_session = await db_session.get(SessionRecord, source_session_id)
            if not isinstance(source_session, SessionRecord):
                raise TypeError(f"Session '{source_session_id}' was not found.")
            _validate_source_conversation_session(source_session)
            state = await self._ensure_state(db_session, source_session)
            run_id = await self._enqueue_or_mark_pending(
                db_session,
                state,
                source_session,
                MemoryRunRequest(source_session_id=source_session_id, kind=MemoryJobKind.SUMMARY, reason=reason),
            )
            await db_session.commit()
        if run_id is not None:
            self._submit_all([run_id])
        return run_id

    async def _load_memory_run_scope(
        self, db_session: AsyncSession, memory_run_id: str
    ) -> tuple[RunRecord, SessionRecord, SessionMemoryStateRecord, MemoryJobKind, dict[str, Any]] | None:
        run = await db_session.get(RunRecord, memory_run_id)
        if not isinstance(run, RunRecord) or run.trigger_type != TriggerType.MEMORY.value:
            return None
        memory = _memory_metadata(run.run_metadata)
        source_session_id = _string_or_none(memory.get("source_session_id"))
        kind = _memory_kind(memory.get("kind"))
        if source_session_id is None or kind is None:
            return None
        source_session = await db_session.get(SessionRecord, source_session_id)
        if not isinstance(source_session, SessionRecord):
            return None
        state = await self._ensure_state(db_session, source_session)
        return run, source_session, state, kind, memory

    async def _ensure_state(self, db_session: AsyncSession, source_session: SessionRecord) -> SessionMemoryStateRecord:
        state = await db_session.get(SessionMemoryStateRecord, source_session.id)
        if isinstance(state, SessionMemoryStateRecord):
            if state.memory_session_id is None:
                memory_session = await self._ensure_memory_session(db_session, source_session)
                state.memory_session_id = memory_session.id
            return state
        memory_session = await self._ensure_memory_session(db_session, source_session)
        state = SessionMemoryStateRecord(
            source_session_id=source_session.id,
            memory_session_id=memory_session.id,
            enabled=_metadata_memory_enabled(source_session.session_metadata, default=True),
        )
        db_session.add(state)
        await db_session.flush()
        return state

    async def _ensure_memory_session(self, db_session: AsyncSession, source_session: SessionRecord) -> SessionRecord:
        statement = select(SessionRecord).where(
            SessionRecord.session_type == "memory",
            SessionRecord.source_session_id == source_session.id,
        )
        result = await db_session.execute(statement)
        record = result.scalars().first()
        if isinstance(record, SessionRecord):
            return record
        record = SessionRecord(
            id=uuid4().hex,
            parent_session_id=source_session.id,
            profile_name=self._settings.memory_profile or source_session.profile_name or self._settings.default_profile,
            session_type="memory",
            source_session_id=source_session.id,
            session_metadata=_memory_session_metadata(source_session),
        )
        db_session.add(record)
        await db_session.flush()
        return record

    async def _enqueue_or_mark_pending(
        self,
        db_session: AsyncSession,
        state: SessionMemoryStateRecord,
        source_session: SessionRecord,
        request: MemoryRunRequest,
    ) -> str | None:
        memory_session = await self._ensure_memory_session(db_session, source_session)
        state.memory_session_id = memory_session.id
        if request.source_identity is None:
            request.source_identity = await _build_source_identity(db_session, source_session, request)
        if await _memory_session_busy(db_session, memory_session.id):
            _mark_pending_request(state, request)
            return None
        run = await _create_memory_run(
            db_session,
            memory_session,
            request,
            memory_metadata=_request_metadata(request, memory_session_id=memory_session.id),
            input_text=await _build_memory_prompt(db_session, request),
        )
        self._runtime_state.register_run(memory_session.id, run.id, dispatch_mode=DispatchMode.ASYNC)
        if request.kind == MemoryJobKind.EXTRACT:
            state.pending_extract = False
            state.last_extract_run_id = run.id
        else:
            state.pending_summary = False
            state.last_summary_run_id = run.id
        return run.id

    async def _enqueue_next_pending(
        self,
        db_session: AsyncSession,
        state: SessionMemoryStateRecord,
        source_session: SessionRecord,
    ) -> str | None:
        request = _pop_next_pending_request(state)
        if request is None:
            return None
        return await self._enqueue_or_mark_pending(db_session, state, source_session, request)

    def _submit_all(self, run_ids: list[str]) -> None:
        if self._submit_run is None:
            return
        for run_id in run_ids:
            self._submit_run(run_id)


def _metadata_memory_enabled(metadata: dict[str, Any], *, default: bool) -> bool:
    memory = metadata.get("memory") if isinstance(metadata, dict) else None
    if isinstance(memory, dict) and isinstance(memory.get("enabled"), bool):
        return bool(memory["enabled"])
    return default


def _memory_session_metadata(source_session: SessionRecord) -> dict[str, Any]:
    metadata: dict[str, Any] = {"memory": {"source_session_id": source_session.id}}
    if isinstance(source_session.session_metadata, dict):
        sandbox = source_session.session_metadata.get("sandbox")
        if isinstance(sandbox, dict):
            metadata["sandbox"] = dict(sandbox)
    return metadata


def _validate_source_conversation_session(source_session: SessionRecord) -> None:
    if source_session.session_type != "conversation":
        raise ValueError(f"Memory actions are only supported for conversation sessions: {source_session.id}")


def _mark_pending_request(state: SessionMemoryStateRecord, request: MemoryRunRequest) -> None:
    pending_requests = _pending_requests(state)
    pending_requests.append(_request_metadata(request, memory_session_id=state.memory_session_id or ""))
    state.memory_metadata = {**dict(state.memory_metadata or {}), _PENDING_MEMORY_REQUESTS_KEY: pending_requests[-10:]}
    if request.kind == MemoryJobKind.EXTRACT:
        state.pending_extract = True
    else:
        state.pending_summary = True


def _pop_next_pending_request(state: SessionMemoryStateRecord) -> MemoryRunRequest | None:
    pending_requests = _pending_requests(state)
    if not pending_requests:
        state.pending_extract = False
        state.pending_summary = False
        return None

    request_data = pending_requests.pop(0)
    state.memory_metadata = {**dict(state.memory_metadata or {}), _PENDING_MEMORY_REQUESTS_KEY: pending_requests}
    state.pending_extract = any(item.get("kind") == MemoryJobKind.EXTRACT.value for item in pending_requests)
    state.pending_summary = any(item.get("kind") == MemoryJobKind.SUMMARY.value for item in pending_requests)
    kind = _memory_kind(request_data.get("kind"))
    if kind is None:
        return _pop_next_pending_request(state)
    trigger_payload = request_data.get("context_handoff")
    source_identity = request_data.get("source_identity")
    return MemoryRunRequest(
        source_session_id=str(request_data.get("source_session_id") or state.source_session_id),
        kind=kind,
        reason=str(request_data.get("reason") or "pending"),
        source_run_ids=[item for item in request_data.get("source_run_ids", []) if isinstance(item, str)]
        if isinstance(request_data.get("source_run_ids"), list)
        else [],
        source_sequence_start=request_data.get("source_sequence_start")
        if isinstance(request_data.get("source_sequence_start"), int)
        else None,
        source_sequence_end=request_data.get("source_sequence_end")
        if isinstance(request_data.get("source_sequence_end"), int)
        else None,
        trigger_payload=dict(trigger_payload) if isinstance(trigger_payload, dict) else None,
        source_identity=dict(source_identity) if isinstance(source_identity, dict) else None,
    )


def _pending_requests(state: SessionMemoryStateRecord) -> list[dict[str, Any]]:
    metadata = dict(state.memory_metadata or {})
    value = metadata.get(_PENDING_MEMORY_REQUESTS_KEY)
    return [dict(item) for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _extract_memory_triggers(claw_metadata: dict[str, Any] | None) -> list[dict[str, Any]]:
    value = claw_metadata.get(_MEMORY_TRIGGER_KEY) if isinstance(claw_metadata, dict) else None
    return [dict(item) for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _trigger_type_from_metadata(claw_metadata: dict[str, Any] | None) -> str | None:
    if not isinstance(claw_metadata, dict):
        return None
    trigger_type = claw_metadata.get("trigger_type")
    if isinstance(trigger_type, str) and trigger_type.strip():
        return trigger_type
    run_metadata = claw_metadata.get("run_metadata")
    if isinstance(run_metadata, dict):
        trigger_type = run_metadata.get("trigger_type")
        if isinstance(trigger_type, str) and trigger_type.strip():
            return trigger_type
    return None


def _request_metadata(request: MemoryRunRequest, *, memory_session_id: str) -> dict[str, Any]:
    metadata = {
        "kind": request.kind.value,
        "source_session_id": request.source_session_id,
        "memory_session_id": memory_session_id,
        "source_run_ids": list(request.source_run_ids),
        "source_sequence_start": request.source_sequence_start,
        "source_sequence_end": request.source_sequence_end,
        "reason": request.reason,
    }
    if request.trigger_payload is not None:
        metadata["context_handoff"] = request.trigger_payload
    if request.source_identity is not None:
        metadata["source_identity"] = request.source_identity
    return metadata


async def _build_memory_prompt(db_session: AsyncSession, request: MemoryRunRequest) -> str:
    source_identity = request.source_identity or await _build_source_identity(db_session, None, request)
    payload = {
        "kind": request.kind.value,
        "reason": request.reason,
        "source_session_id": request.source_session_id,
        "source_run_ids": request.source_run_ids,
        "source_sequence_start": request.source_sequence_start,
        "source_sequence_end": request.source_sequence_end,
        "source_identity": source_identity,
        "context_handoff": request.trigger_payload,
        "source_runs": await _source_run_material(db_session, request) if _should_embed_source_runs(request) else [],
    }
    if request.kind == MemoryJobKind.EXTRACT:
        instruction = (
            "Run workspace memory extraction. Use context_handoff trimmed messages when present. "
            "For threshold or manual extracts, inspect the referenced source session with session tools. "
            "Keep memory/MEMORY.md as a compact durable brief. Write details to event notes named memory/YYYYMMDD-event.md. "
            "Update memory/CHANGELOG.md after changing memory files."
        )
    else:
        instruction = (
            "Run workspace memory summary. Review memory/MEMORY.md, memory/CHANGELOG.md, and event notes matching memory/YYYYMMDD-event.md. "
            "Reorganize, merge, and rewrite memory files while preserving useful provenance. "
            "Keep MEMORY.md compact and move detailed chronology into event notes or CHANGELOG.md."
        )
    return "\n\n".join([
        instruction,
        "Memory file protocol: memory/MEMORY.md is the compact durable brief loaded for the primary agent. memory/CHANGELOG.md records memory updates. Event files use memory/YYYYMMDD-event.md filenames with YAML frontmatter containing name and description, and their frontmatter is the discovery surface for detailed memory.",
        "Memory job payload:",
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
    ])


def _should_embed_source_runs(request: MemoryRunRequest) -> bool:
    return request.kind == MemoryJobKind.EXTRACT and request.trigger_payload is not None


def _memory_metadata(run_metadata: dict[str, Any]) -> dict[str, Any]:
    value = run_metadata.get("memory") if isinstance(run_metadata, dict) else None
    return dict(value) if isinstance(value, dict) else {}


def _memory_kind(value: object) -> MemoryJobKind | None:
    try:
        return MemoryJobKind(value)
    except Exception:
        return None


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _int_or_zero(value: object) -> int:
    return value if isinstance(value, int) else 0


async def _create_memory_run(
    db_session: AsyncSession,
    memory_session: SessionRecord,
    request: MemoryRunRequest,
    *,
    memory_metadata: dict[str, Any],
    input_text: str,
) -> RunRecord:
    sequence_no = await _next_sequence_no(db_session, memory_session.id)
    run = RunRecord(
        id=uuid4().hex,
        session_id=memory_session.id,
        sequence_no=sequence_no,
        restore_from_run_id=None,
        status="queued",
        trigger_type=TriggerType.MEMORY.value,
        profile_name=memory_session.profile_name,
        input_parts=[{"type": "text", "text": input_text}],
        run_metadata={"memory": memory_metadata},
    )
    db_session.add(run)
    queue_run(memory_session, run)
    await db_session.flush()
    return run


async def _next_sequence_no(db_session: AsyncSession, session_id: str) -> int:
    result = await db_session.execute(select(func.max(RunRecord.sequence_no)).where(RunRecord.session_id == session_id))
    value = result.scalar_one_or_none()
    return value + 1 if isinstance(value, int) else 1


async def _memory_session_busy(db_session: AsyncSession, memory_session_id: str) -> bool:
    statement = select(func.count()).where(
        RunRecord.session_id == memory_session_id,
        RunRecord.status.in_(["queued", "running"]),
    )
    result = await db_session.execute(statement)
    return bool(result.scalar_one())


_BRIDGE_IDENTITY_KEYS = (
    "adapter",
    "tenant_key",
    "chat_id",
    "chat_type",
    "event_id",
    "message_id",
    "root_id",
    "parent_id",
    "thread_id",
    "sender_id",
    "sender_type",
    "message_type",
    "create_time",
)


async def _build_source_identity(
    db_session: AsyncSession,
    source_session: SessionRecord | None,
    request: MemoryRunRequest,
) -> dict[str, Any]:
    if source_session is None:
        loaded_session = await db_session.get(SessionRecord, request.source_session_id)
        source_session = loaded_session if isinstance(loaded_session, SessionRecord) else None

    identity: dict[str, Any] = {
        "source_session": {
            "session_id": request.source_session_id,
        },
    }
    if isinstance(source_session, SessionRecord):
        identity["source_session"] = {
            "session_id": source_session.id,
            "profile_name": source_session.profile_name,
            "session_type": source_session.session_type,
        }
        conversation_identity = _bridge_identity_from_container(source_session.session_metadata)
        if conversation_identity:
            identity["bridge"] = {"conversation": conversation_identity}

    source_runs = await _source_run_identities(db_session, request)
    if source_runs:
        identity["source_runs"] = source_runs
        latest_bridge_message = next(
            (item["bridge"] for item in reversed(source_runs) if isinstance(item.get("bridge"), dict)),
            None,
        )
        if isinstance(latest_bridge_message, dict):
            bridge_identity = dict(identity.get("bridge") or {})
            bridge_identity["latest_message"] = latest_bridge_message
            identity["bridge"] = bridge_identity
    return identity


async def _source_run_identities(db_session: AsyncSession, request: MemoryRunRequest) -> list[dict[str, Any]]:
    identities: list[dict[str, Any]] = []
    for record in await _source_run_records(db_session, request):
        item: dict[str, Any] = {
            "run_id": record.id,
            "sequence_no": record.sequence_no,
            "trigger_type": record.trigger_type,
        }
        bridge_identity = _bridge_identity_from_container(record.run_metadata)
        if bridge_identity:
            item["bridge"] = bridge_identity
        identities.append(item)
    return identities


def _bridge_identity_from_container(container: object) -> dict[str, Any]:
    if not isinstance(container, dict):
        return {}
    candidate = container.get("bridge")
    bridge = candidate if isinstance(candidate, dict) else container
    identity: dict[str, Any] = {}
    for key in _BRIDGE_IDENTITY_KEYS:
        value = bridge.get(key)
        if isinstance(value, str | int | float | bool):
            identity[key] = value
    return identity


async def _source_run_records(db_session: AsyncSession, request: MemoryRunRequest) -> list[RunRecord]:
    statement = select(RunRecord).where(
        RunRecord.session_id == request.source_session_id,
        RunRecord.status == "completed",
    )
    if request.source_run_ids:
        statement = statement.where(RunRecord.id.in_(request.source_run_ids))
    else:
        if isinstance(request.source_sequence_start, int):
            statement = statement.where(RunRecord.sequence_no >= request.source_sequence_start)
        if isinstance(request.source_sequence_end, int):
            statement = statement.where(RunRecord.sequence_no <= request.source_sequence_end)
    statement = statement.order_by(RunRecord.sequence_no.asc()).limit(50)
    result = await db_session.execute(statement)
    return list(result.scalars().all())


async def _source_run_material(db_session: AsyncSession, request: MemoryRunRequest) -> list[dict[str, Any]]:
    return [
        {
            "run_id": record.id,
            "sequence_no": record.sequence_no,
            "trigger_type": record.trigger_type,
            "source_identity": {
                "bridge": _bridge_identity_from_container(record.run_metadata),
            },
            "input_parts": record.input_parts,
            "output_text": record.output_text,
            "output_summary": record.output_summary,
            "committed_at": record.committed_at.isoformat() if record.committed_at is not None else None,
        }
        for record in await _source_run_records(db_session, request)
    ]


async def _max_completed_sequence_no(db_session: AsyncSession, source_session_id: str) -> int | None:
    result = await db_session.execute(
        select(func.max(RunRecord.sequence_no)).where(
            RunRecord.session_id == source_session_id,
            RunRecord.status == "completed",
        )
    )
    value = result.scalar_one_or_none()
    return value if isinstance(value, int) else None


async def _sequence_range_for_run_ids(
    db_session: AsyncSession,
    source_session_id: str,
    source_run_ids: list[str] | None,
) -> tuple[int | None, int | None]:
    if not source_run_ids:
        return 1, await _max_completed_sequence_no(db_session, source_session_id)
    result = await db_session.execute(
        select(func.min(RunRecord.sequence_no), func.max(RunRecord.sequence_no)).where(
            RunRecord.session_id == source_session_id,
            RunRecord.id.in_(source_run_ids),
            RunRecord.status == "completed",
        )
    )
    row = result.one()
    return row[0], row[1]


async def _matched_run_ids_and_sequence_range(
    db_session: AsyncSession,
    source_session_id: str,
    source_run_ids: list[str],
) -> tuple[list[str], int | None, int | None]:
    result = await db_session.execute(
        select(RunRecord.id, RunRecord.sequence_no)
        .where(
            RunRecord.session_id == source_session_id,
            RunRecord.id.in_(source_run_ids),
            RunRecord.status == "completed",
        )
        .order_by(RunRecord.sequence_no.asc())
    )
    rows = result.all()
    matched_ids = [row[0] for row in rows]
    sequence_values = [row[1] for row in rows if isinstance(row[1], int)]
    if not sequence_values:
        return matched_ids, None, None
    return matched_ids, min(sequence_values), max(sequence_values)
