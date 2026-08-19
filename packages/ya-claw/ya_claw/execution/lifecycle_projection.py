from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ya_claw.agency.lifecycle import AgencyLifecycle
from ya_claw.config import ClawSettings
from ya_claw.execution.store import RunStore
from ya_claw.memory.lifecycle import MemoryLifecycle
from ya_claw.orm.tables import RunRecord, SessionRecord
from ya_claw.runtime_state import InMemoryRuntimeState


@dataclass(slots=True)
class AgencyMemorySource:
    source_session_id: str
    source_run_id: str | None
    source_sequence_no: int


class CompletedRunProjector:
    """Project committed run facts into durable Memory and Agency lifecycle state."""

    def __init__(
        self,
        *,
        settings: ClawSettings,
        session_factory: async_sessionmaker[AsyncSession],
        runtime_state: InMemoryRuntimeState,
        run_store: RunStore,
        submit_run: Callable[[str], bool] | None = None,
        agency_submit_run: Callable[[str], bool] | None = None,
    ) -> None:
        self._settings = settings
        self._session_factory = session_factory
        self._runtime_state = runtime_state
        self._run_store = run_store
        self._submit_run = submit_run
        self._agency_submit_run = agency_submit_run

    async def recover_pending(self) -> list[str]:
        async with self._session_factory() as db_session:
            result = await db_session.execute(
                select(RunRecord.id)
                .where(
                    RunRecord.status == "completed",
                    RunRecord.lifecycle_projected_at.is_(None),
                )
                .order_by(RunRecord.committed_at.asc(), RunRecord.created_at.asc(), RunRecord.id.asc())
            )
            run_ids = list(result.scalars().all())

        projected: list[str] = []
        for run_id in run_ids:
            try:
                if await self.project(run_id):
                    projected.append(run_id)
            except Exception:
                logger.exception("Completed run lifecycle projection failed run_id={}", run_id)
        return projected

    async def project(self, run_id: str) -> bool:
        loaded = await self._load_source(run_id)
        if loaded is None:
            return False
        source, agency_sources = loaded
        claw_metadata = _claw_metadata(self._run_store.read_state(run_id))
        await self._project_source(source, agency_sources, claw_metadata)
        await self._mark_projected(run_id)
        return True

    async def _load_source(
        self,
        run_id: str,
    ) -> tuple[_RunProjectionSource, list[AgencyMemorySource]] | None:
        async with self._session_factory() as db_session:
            run_record = await db_session.get(RunRecord, run_id)
            if (
                not isinstance(run_record, RunRecord)
                or run_record.status != "completed"
                or run_record.lifecycle_projected_at is not None
            ):
                return None
            session_record = await db_session.get(SessionRecord, run_record.session_id)
            if not isinstance(session_record, SessionRecord):
                raise TypeError(f"Completed run {run_id!r} has no session")
            if session_record.session_type == "conversation":
                earlier = await db_session.execute(
                    select(RunRecord.id)
                    .where(
                        RunRecord.session_id == session_record.id,
                        RunRecord.status == "completed",
                        RunRecord.lifecycle_projected_at.is_(None),
                        RunRecord.sequence_no < run_record.sequence_no,
                    )
                    .limit(1)
                )
                if earlier.scalar_one_or_none() is not None:
                    return None

            agency_sources: list[AgencyMemorySource] = []
            if session_record.session_type == "agency":
                await self._agency_lifecycle().on_agency_run_committed(db_session, run_record)
                if self._settings.agency_memory_capture_enabled:
                    agency_metadata = (
                        run_record.run_metadata.get("agency") if isinstance(run_record.run_metadata, dict) else None
                    )
                    agency_sources = await _agency_memory_sources(db_session, agency_metadata)
                await db_session.commit()
            return (
                _RunProjectionSource(
                    run_id=run_record.id,
                    session_id=session_record.id,
                    session_type=session_record.session_type,
                    sequence_no=run_record.sequence_no,
                    profile_name=run_record.profile_name,
                    trigger_type=run_record.trigger_type,
                    output_text=run_record.output_text,
                    termination_reason=run_record.termination_reason,
                    head_success_run_id=session_record.head_success_run_id,
                ),
                agency_sources,
            )

    async def _project_source(
        self,
        source: _RunProjectionSource,
        agency_sources: list[AgencyMemorySource],
        claw_metadata: dict[str, Any],
    ) -> None:
        memory_lifecycle = self._memory_lifecycle()
        if source.session_type == "memory":
            await memory_lifecycle.on_memory_run_committed(memory_run_id=source.run_id)
            return
        if source.session_type == "async_task":
            return
        if source.session_type == "agency":
            for agency_source in agency_sources:
                await memory_lifecycle.on_run_committed(
                    source_session_id=agency_source.source_session_id,
                    source_run_id=agency_source.source_run_id or source.run_id,
                    source_sequence_no=agency_source.source_sequence_no,
                    profile_name=source.profile_name,
                    claw_metadata=claw_metadata,
                    effect_id=(
                        f"agency:{source.run_id}:{agency_source.source_session_id}:{agency_source.source_sequence_no}"
                    ),
                    effect_kind="agency_capture",
                    projection_run_id=source.run_id,
                )
            return

        await memory_lifecycle.on_run_committed(
            source_session_id=source.session_id,
            source_run_id=source.run_id,
            source_sequence_no=source.sequence_no,
            profile_name=source.profile_name,
            claw_metadata=claw_metadata,
            effect_id=f"conversation:{source.run_id}",
            effect_kind="conversation_run",
            projection_run_id=source.run_id,
        )
        if self._settings.agency_enabled and source.trigger_type != "agency_handoff":
            async with self._session_factory() as db_session:
                await self._agency_lifecycle().observe_run_output(
                    db_session,
                    source_session_id=source.session_id,
                    source_run_id=source.run_id,
                    source_sequence_no=source.sequence_no,
                    trigger_type=source.trigger_type,
                    source_kind=source.trigger_type,
                    output_text=source.output_text,
                    metadata={
                        "profile_name": source.profile_name,
                        "termination_reason": source.termination_reason,
                        "head_success_run_id": source.head_success_run_id,
                    },
                )

    async def _mark_projected(self, run_id: str) -> None:
        async with self._session_factory() as db_session:
            run_record = await db_session.get(RunRecord, run_id)
            if not isinstance(run_record, RunRecord) or run_record.status != "completed":
                raise RuntimeError(f"Completed run {run_id!r} is no longer available")
            if run_record.lifecycle_projected_at is None:
                run_record.lifecycle_projected_at = datetime.now(UTC)
                await db_session.commit()

    def _memory_lifecycle(self) -> MemoryLifecycle:
        return MemoryLifecycle(
            settings=self._settings,
            session_factory=self._session_factory,
            runtime_state=self._runtime_state,
            submit_run=self._submit_run,
            agency_submit_run=self._agency_submit_run,
        )

    def _agency_lifecycle(self) -> AgencyLifecycle:
        return AgencyLifecycle(
            settings=self._settings,
            runtime_state=self._runtime_state,
            submit_run=self._agency_submit_run,
        )


@dataclass(slots=True)
class _RunProjectionSource:
    run_id: str
    session_id: str
    session_type: str
    sequence_no: int
    profile_name: str | None
    trigger_type: str
    output_text: str | None
    termination_reason: str | None
    head_success_run_id: str | None


def _claw_metadata(state: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(state, dict):
        return {}
    context = state.get("context")
    if not isinstance(context, dict):
        return {}
    value = context.get("claw_metadata")
    return dict(value) if isinstance(value, dict) else {}


async def _agency_memory_sources(
    db_session: AsyncSession,
    agency_metadata: object,
) -> list[AgencyMemorySource]:
    if not isinstance(agency_metadata, dict):
        return []
    raw_sources = agency_metadata.get("sources")
    if isinstance(raw_sources, list):
        sources = [dict(item) for item in raw_sources if isinstance(item, dict)]
    else:
        values = agency_metadata.get("source_session_ids")
        sources = (
            [
                {"source_session_id": value, "source_run_id": None}
                for value in values
                if isinstance(value, str) and value.strip()
            ]
            if isinstance(values, list)
            else []
        )

    result: list[AgencyMemorySource] = []
    seen: set[tuple[str, str | None]] = set()
    for source in sources:
        source_session_id = source.get("source_session_id")
        source_run_id = source.get("source_run_id")
        if not isinstance(source_session_id, str) or not source_session_id.strip():
            continue
        if not isinstance(source_run_id, str):
            source_run_id = None
        identity = (source_session_id, source_run_id)
        if identity in seen:
            continue
        seen.add(identity)
        sequence_no = await _source_sequence_no(
            db_session,
            source_session_id=source_session_id,
            source_run_id=source_run_id,
        )
        if sequence_no is not None:
            result.append(
                AgencyMemorySource(
                    source_session_id=source_session_id,
                    source_run_id=source_run_id,
                    source_sequence_no=sequence_no,
                )
            )
    return result


async def _source_sequence_no(
    db_session: AsyncSession,
    *,
    source_session_id: str,
    source_run_id: str | None,
) -> int | None:
    if source_run_id is not None:
        source_run = await db_session.get(RunRecord, source_run_id)
        if isinstance(source_run, RunRecord) and source_run.session_id == source_session_id:
            return source_run.sequence_no
    result = await db_session.execute(
        select(RunRecord.sequence_no)
        .where(RunRecord.session_id == source_session_id)
        .order_by(RunRecord.sequence_no.desc())
        .limit(1)
    )
    value = result.scalar_one_or_none()
    return value if isinstance(value, int) else None
