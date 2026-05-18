from __future__ import annotations

import shutil
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ya_claw.config import ClawSettings
from ya_claw.orm.tables import (
    HeartbeatFireRecord,
    RunRecord,
    ScheduleFireRecord,
    ScheduleRecord,
    SessionRecord,
    utc_now,
)

_ACTIVE_RUN_STATUSES = frozenset({"queued", "running"})


class SessionPruneResult(BaseModel):
    pruned_run_store_dirs: int = 0
    deleted_runs: int = 0
    deleted_sessions: int = 0
    deleted_orphan_run_dirs: int = 0
    deleted_schedule_fires: int = 0
    deleted_heartbeat_fires: int = 0
    hidden_once_schedules: int = 0
    reclaimed_bytes: int = 0
    failed_run_store_paths: list[str] = Field(default_factory=list)

    def merge(self, other: SessionPruneResult) -> None:
        self.pruned_run_store_dirs += other.pruned_run_store_dirs
        self.deleted_runs += other.deleted_runs
        self.deleted_sessions += other.deleted_sessions
        self.deleted_orphan_run_dirs += other.deleted_orphan_run_dirs
        self.deleted_schedule_fires += other.deleted_schedule_fires
        self.deleted_heartbeat_fires += other.deleted_heartbeat_fires
        self.hidden_once_schedules += other.hidden_once_schedules
        self.reclaimed_bytes += other.reclaimed_bytes
        self.failed_run_store_paths.extend(other.failed_run_store_paths)


class SessionPruneController:
    async def prune_once(self, db_session: AsyncSession, settings: ClawSettings) -> SessionPruneResult:
        result = SessionPruneResult()
        if settings.session_prune_enabled:
            if settings.session_prune_generated_sessions_enabled:
                generated_session_ids = await self._select_generated_session_ids(db_session, settings)
                result.merge(await self._delete_sessions(db_session, settings, generated_session_ids))

            run_ids = await self._select_prunable_run_ids(db_session, settings)
            result.merge(await self._prune_run_store_dirs(settings, run_ids))

            if settings.session_prune_fire_records_older_than_days > 0:
                fire_result = await self._prune_fire_records(db_session, settings)
                result.deleted_schedule_fires += fire_result.deleted_schedule_fires
                result.deleted_heartbeat_fires += fire_result.deleted_heartbeat_fires

            if settings.session_prune_orphans_enabled:
                result.merge(await self._prune_orphan_run_store_dirs(db_session, settings))

        result.merge(await self._hide_expired_once_schedules(db_session, settings))

        return result

    async def _select_prunable_run_ids(self, db_session: AsyncSession, settings: ClawSettings) -> list[str]:
        keep_recent = max(settings.session_prune_run_keep_recent, 1)
        batch_size = max(settings.session_prune_batch_size, 1)
        cutoff = _cutoff_from_days(settings.session_prune_run_older_than_days)
        sessions = await self._load_sessions(db_session)
        runs = await self._load_runs(db_session)
        runs_by_session, runs_by_id = _index_runs(runs)
        protected_run_ids = _protected_run_ids(sessions, runs_by_session, runs_by_id, keep_recent=keep_recent)
        return _prunable_run_ids(
            runs,
            protected_run_ids,
            cutoff=cutoff,
            batch_size=batch_size,
        )

    async def _load_sessions(self, db_session: AsyncSession) -> list[SessionRecord]:
        result = await db_session.execute(select(SessionRecord))
        return list(result.scalars().all())

    async def _load_runs(self, db_session: AsyncSession) -> list[RunRecord]:
        result = await db_session.execute(
            select(RunRecord).order_by(RunRecord.session_id.asc(), RunRecord.sequence_no.desc(), RunRecord.id.desc())
        )
        return list(result.scalars().all())

    async def _select_generated_session_ids(self, db_session: AsyncSession, settings: ClawSettings) -> list[str]:
        batch_size = max(settings.session_prune_batch_size, 1)
        session_ids: list[str] = []
        session_ids.extend(await self._select_heartbeat_session_ids(db_session, settings, batch_size=batch_size))
        if len(session_ids) >= batch_size:
            return session_ids[:batch_size]
        remaining = batch_size - len(session_ids)
        session_ids.extend(await self._select_schedule_session_ids(db_session, settings, batch_size=remaining))
        return _dedupe(session_ids)[:batch_size]

    async def _select_heartbeat_session_ids(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        *,
        batch_size: int,
    ) -> list[str]:
        keep_recent = max(settings.session_prune_heartbeat_keep_recent, 1)
        cutoff = _cutoff_from_days(settings.session_prune_heartbeat_older_than_days)
        result = await db_session.execute(
            select(HeartbeatFireRecord)
            .where(HeartbeatFireRecord.session_id.is_not(None))
            .order_by(
                HeartbeatFireRecord.scheduled_at.desc(),
                HeartbeatFireRecord.created_at.desc(),
                HeartbeatFireRecord.id.desc(),
            )
        )
        seen: set[str] = set()
        ordered_records: list[HeartbeatFireRecord] = []
        for record in result.scalars().all():
            if not isinstance(record.session_id, str) or record.session_id in seen:
                continue
            seen.add(record.session_id)
            ordered_records.append(record)

        candidates: list[str] = []
        for record in ordered_records[keep_recent:]:
            if cutoff is not None and not _is_older_than(record.scheduled_at, cutoff):
                continue
            if isinstance(record.session_id, str):
                candidates.append(record.session_id)
            if len(candidates) >= batch_size:
                break
        return candidates

    async def _select_schedule_session_ids(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        *,
        batch_size: int,
    ) -> list[str]:
        keep_recent = max(settings.session_prune_schedule_keep_recent, 1)
        cutoff = _cutoff_from_days(settings.session_prune_schedule_older_than_days)
        result = await db_session.execute(
            select(ScheduleFireRecord)
            .where(ScheduleFireRecord.created_session_id.is_not(None))
            .order_by(
                ScheduleFireRecord.schedule_id.asc(),
                ScheduleFireRecord.scheduled_at.desc(),
                ScheduleFireRecord.created_at.desc(),
                ScheduleFireRecord.id.desc(),
            )
        )
        records_by_schedule: dict[str, list[ScheduleFireRecord]] = defaultdict(list)
        seen_session_ids_by_schedule: dict[str, set[str]] = defaultdict(set)
        for record in result.scalars().all():
            if not _is_generated_schedule_session(record):
                continue
            created_session_id = record.created_session_id
            if not isinstance(created_session_id, str):
                continue
            seen = seen_session_ids_by_schedule[record.schedule_id]
            if created_session_id in seen:
                continue
            seen.add(created_session_id)
            records_by_schedule[record.schedule_id].append(record)

        candidates: list[str] = []
        for schedule_id in sorted(records_by_schedule):
            for record in records_by_schedule[schedule_id][keep_recent:]:
                if cutoff is not None and not _is_older_than(record.scheduled_at, cutoff):
                    continue
                if isinstance(record.created_session_id, str):
                    candidates.append(record.created_session_id)
                if len(candidates) >= batch_size:
                    return candidates
        return candidates

    async def _prune_run_store_dirs(
        self,
        settings: ClawSettings,
        run_ids: Iterable[str],
    ) -> SessionPruneResult:
        normalized_run_ids = _dedupe([run_id for run_id in run_ids if run_id.strip() != ""])
        if not normalized_run_ids:
            return SessionPruneResult()

        run_paths = [settings.run_store_dir / run_id for run_id in normalized_run_ids]
        existing_paths = [path for path in run_paths if path.exists()]
        reclaimed_bytes = sum(_path_size(path) for path in existing_paths)
        file_result = self._delete_run_store_paths(existing_paths)
        return SessionPruneResult(
            pruned_run_store_dirs=len(existing_paths) - len(file_result.failed_run_store_paths),
            reclaimed_bytes=reclaimed_bytes,
            failed_run_store_paths=file_result.failed_run_store_paths,
        )

    async def _delete_sessions(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        session_ids: Iterable[str],
    ) -> SessionPruneResult:
        normalized_session_ids = await self._filter_deletable_session_ids(db_session, _dedupe(session_ids))
        if not normalized_session_ids:
            return SessionPruneResult()

        runs_result = await db_session.execute(
            select(RunRecord.id).where(RunRecord.session_id.in_(normalized_session_ids))
        )
        run_ids = [run_id for run_id in runs_result.scalars().all() if isinstance(run_id, str)]
        run_paths = [settings.run_store_dir / run_id for run_id in run_ids]
        reclaimed_bytes = sum(_path_size(path) for path in run_paths)

        await db_session.execute(
            update(SessionRecord)
            .where(SessionRecord.parent_session_id.in_(normalized_session_ids))
            .values(parent_session_id=None)
        )
        if run_ids:
            await db_session.execute(delete(RunRecord).where(RunRecord.id.in_(run_ids)))
        await db_session.execute(delete(SessionRecord).where(SessionRecord.id.in_(normalized_session_ids)))
        await db_session.commit()

        file_result = self._delete_run_store_paths(run_paths)
        return SessionPruneResult(
            deleted_runs=len(run_ids),
            deleted_sessions=len(normalized_session_ids),
            reclaimed_bytes=reclaimed_bytes,
            failed_run_store_paths=file_result.failed_run_store_paths,
        )

    async def _filter_deletable_session_ids(self, db_session: AsyncSession, session_ids: list[str]) -> list[str]:
        if not session_ids:
            return []
        protected_session_ids = await self._protected_session_ids(db_session)
        records_result = await db_session.execute(select(SessionRecord).where(SessionRecord.id.in_(session_ids)))
        records = [record for record in records_result.scalars().all() if isinstance(record, SessionRecord)]
        candidate_ids: list[str] = []
        for record in records:
            if record.id in protected_session_ids:
                continue
            if isinstance(record.active_run_id, str) and record.active_run_id.strip() != "":
                continue
            candidate_ids.append(record.id)
        if not candidate_ids:
            return []

        active_runs_result = await db_session.execute(
            select(RunRecord.session_id)
            .where(RunRecord.session_id.in_(candidate_ids), RunRecord.status.in_(tuple(_ACTIVE_RUN_STATUSES)))
            .distinct()
        )
        active_session_ids = {
            session_id for session_id in active_runs_result.scalars().all() if isinstance(session_id, str)
        }
        candidate_ids = [session_id for session_id in candidate_ids if session_id not in active_session_ids]
        if not candidate_ids:
            return []

        run_ids_result = await db_session.execute(select(RunRecord.id).where(RunRecord.session_id.in_(candidate_ids)))
        run_ids = [run_id for run_id in run_ids_result.scalars().all() if isinstance(run_id, str)]
        if not run_ids:
            return candidate_ids
        external_refs_result = await db_session.execute(
            select(RunRecord.restore_from_run_id)
            .where(RunRecord.restore_from_run_id.in_(run_ids), RunRecord.session_id.not_in(candidate_ids))
            .distinct()
        )
        externally_referenced_run_ids = {
            run_id for run_id in external_refs_result.scalars().all() if isinstance(run_id, str)
        }
        if not externally_referenced_run_ids:
            return candidate_ids
        referenced_session_result = await db_session.execute(
            select(RunRecord.session_id).where(RunRecord.id.in_(externally_referenced_run_ids)).distinct()
        )
        externally_referenced_session_ids = {
            session_id for session_id in referenced_session_result.scalars().all() if isinstance(session_id, str)
        }
        return [session_id for session_id in candidate_ids if session_id not in externally_referenced_session_ids]

    async def _protected_session_ids(self, db_session: AsyncSession) -> set[str]:
        protected_session_ids: set[str] = set()
        schedule_result = await db_session.execute(
            select(ScheduleRecord.target_session_id, ScheduleRecord.source_session_id).where(
                ScheduleRecord.status != "deleted"
            )
        )
        for target_session_id, source_session_id in schedule_result.all():
            for session_id in (target_session_id, source_session_id):
                if isinstance(session_id, str) and session_id.strip() != "":
                    protected_session_ids.add(session_id)
        parent_result = await db_session.execute(
            select(SessionRecord.parent_session_id).where(SessionRecord.parent_session_id.is_not(None)).distinct()
        )
        for session_id in parent_result.scalars().all():
            if isinstance(session_id, str) and session_id.strip() != "":
                protected_session_ids.add(session_id)
        return protected_session_ids

    async def _prune_fire_records(self, db_session: AsyncSession, settings: ClawSettings) -> SessionPruneResult:
        cutoff = _cutoff_from_days(settings.session_prune_fire_records_older_than_days)
        if cutoff is None:
            return SessionPruneResult()
        schedule_fire_ids = await self._select_prunable_schedule_fire_ids(db_session, cutoff)
        heartbeat_fire_ids = await self._select_prunable_heartbeat_fire_ids(db_session, cutoff)
        if schedule_fire_ids:
            await db_session.execute(delete(ScheduleFireRecord).where(ScheduleFireRecord.id.in_(schedule_fire_ids)))
        if heartbeat_fire_ids:
            await db_session.execute(delete(HeartbeatFireRecord).where(HeartbeatFireRecord.id.in_(heartbeat_fire_ids)))
        if schedule_fire_ids or heartbeat_fire_ids:
            await db_session.commit()
        return SessionPruneResult(
            deleted_schedule_fires=len(schedule_fire_ids),
            deleted_heartbeat_fires=len(heartbeat_fire_ids),
        )

    async def _select_prunable_schedule_fire_ids(self, db_session: AsyncSession, cutoff: datetime) -> list[str]:
        result = await db_session.execute(
            select(ScheduleFireRecord).order_by(
                ScheduleFireRecord.schedule_id.asc(),
                ScheduleFireRecord.created_at.desc(),
                ScheduleFireRecord.id.desc(),
            )
        )
        latest_fire_by_schedule: set[str] = set()
        prunable_ids: list[str] = []
        for record in result.scalars().all():
            if record.schedule_id not in latest_fire_by_schedule:
                latest_fire_by_schedule.add(record.schedule_id)
                continue
            if record.status == "pending":
                continue
            if not _is_older_than(record.created_at, cutoff):
                continue
            prunable_ids.append(record.id)
        return prunable_ids

    async def _select_prunable_heartbeat_fire_ids(self, db_session: AsyncSession, cutoff: datetime) -> list[str]:
        result = await db_session.execute(
            select(HeartbeatFireRecord).order_by(HeartbeatFireRecord.created_at.desc(), HeartbeatFireRecord.id.desc())
        )
        latest_seen = False
        prunable_ids: list[str] = []
        for record in result.scalars().all():
            if not latest_seen:
                latest_seen = True
                continue
            if record.status == "pending":
                continue
            if not _is_older_than(record.created_at, cutoff):
                continue
            prunable_ids.append(record.id)
        return prunable_ids

    async def _hide_expired_once_schedules(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
    ) -> SessionPruneResult:
        cutoff = _cutoff_from_days(settings.session_prune_once_schedules_hide_after_days)
        if cutoff is None:
            return SessionPruneResult()
        schedule_ids = await self._select_expired_once_schedule_ids(db_session, settings, cutoff)
        if not schedule_ids:
            return SessionPruneResult()
        now = utc_now()
        result = await db_session.execute(select(ScheduleRecord).where(ScheduleRecord.id.in_(schedule_ids)))
        records = [record for record in result.scalars().all() if isinstance(record, ScheduleRecord)]
        for record in records:
            metadata = dict(record.schedule_metadata or {})
            metadata["auto_hidden"] = True
            metadata["auto_hidden_reason"] = "expired_once_schedule"
            metadata["auto_hidden_at"] = now.isoformat()
            record.status = "deleted"
            record.next_fire_at = None
            record.schedule_metadata = metadata
            record.updated_at = now
        await db_session.commit()
        return SessionPruneResult(hidden_once_schedules=len(records))

    async def _select_expired_once_schedule_ids(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        cutoff: datetime,
    ) -> list[str]:
        batch_size = max(settings.session_prune_batch_size, 1)
        result = await db_session.execute(
            select(ScheduleRecord).where(ScheduleRecord.trigger_kind == "once", ScheduleRecord.status == "completed")
        )
        records = [record for record in result.scalars().all() if isinstance(record, ScheduleRecord)]
        ordered_candidates = sorted(
            (record for record in records if _is_older_than(_schedule_finished_at(record), cutoff)),
            key=lambda item: (_as_utc_aware(_schedule_finished_at(item)), item.id),
        )
        if not ordered_candidates:
            return []
        candidate_ids = [record.id for record in ordered_candidates]
        protected_ids = await self._protected_once_schedule_ids(db_session, candidate_ids)
        return [schedule_id for schedule_id in candidate_ids if schedule_id not in protected_ids][:batch_size]

    async def _protected_once_schedule_ids(self, db_session: AsyncSession, schedule_ids: list[str]) -> set[str]:
        if not schedule_ids:
            return set()
        protected_ids: set[str] = set()
        pending_result = await db_session.execute(
            select(ScheduleFireRecord.schedule_id)
            .where(ScheduleFireRecord.schedule_id.in_(schedule_ids), ScheduleFireRecord.status == "pending")
            .distinct()
        )
        protected_ids.update(
            schedule_id for schedule_id in pending_result.scalars().all() if isinstance(schedule_id, str)
        )

        active_result = await db_session.execute(
            select(ScheduleFireRecord.schedule_id)
            .join(RunRecord, ScheduleFireRecord.run_id == RunRecord.id)
            .where(ScheduleFireRecord.schedule_id.in_(schedule_ids), RunRecord.status.in_(tuple(_ACTIVE_RUN_STATUSES)))
            .distinct()
        )
        protected_ids.update(
            schedule_id for schedule_id in active_result.scalars().all() if isinstance(schedule_id, str)
        )
        return protected_ids

    async def _prune_orphan_run_store_dirs(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
    ) -> SessionPruneResult:
        run_store_dir = settings.run_store_dir
        if not run_store_dir.exists():
            return SessionPruneResult()
        candidate_paths = [path for path in run_store_dir.iterdir() if path.is_dir()]
        if not candidate_paths:
            return SessionPruneResult()
        candidate_ids = [path.name for path in candidate_paths]
        result = await db_session.execute(select(RunRecord.id).where(RunRecord.id.in_(candidate_ids)))
        existing_ids = {run_id for run_id in result.scalars().all() if isinstance(run_id, str)}
        orphan_paths = [path for path in candidate_paths if path.name not in existing_ids]
        reclaimed_bytes = sum(_path_size(path) for path in orphan_paths)
        file_result = self._delete_run_store_paths(orphan_paths)
        return SessionPruneResult(
            deleted_orphan_run_dirs=len(orphan_paths) - len(file_result.failed_run_store_paths),
            reclaimed_bytes=reclaimed_bytes,
            failed_run_store_paths=file_result.failed_run_store_paths,
        )

    def _delete_run_store_paths(self, paths: Iterable[Path]) -> SessionPruneResult:
        result = SessionPruneResult()
        for path in paths:
            if not path.exists():
                continue
            try:
                shutil.rmtree(path)
            except OSError as exc:
                logger.warning("Failed to delete run store path path={} error={}", path, exc)
                result.failed_run_store_paths.append(str(path))
        return result


def _is_generated_schedule_session(record: ScheduleFireRecord) -> bool:
    created_session_id = record.created_session_id
    if not isinstance(created_session_id, str) or created_session_id.strip() == "":
        return False
    if isinstance(record.target_session_id, str) and created_session_id == record.target_session_id:
        return False
    return not (isinstance(record.source_session_id, str) and created_session_id == record.source_session_id)


def _index_runs(runs: list[RunRecord]) -> tuple[dict[str, list[RunRecord]], dict[str, RunRecord]]:
    runs_by_session: dict[str, list[RunRecord]] = defaultdict(list)
    runs_by_id: dict[str, RunRecord] = {}
    for run in runs:
        runs_by_session[run.session_id].append(run)
        runs_by_id[run.id] = run
    return runs_by_session, runs_by_id


def _protected_run_ids(
    sessions: list[SessionRecord],
    runs_by_session: dict[str, list[RunRecord]],
    runs_by_id: dict[str, RunRecord],
    *,
    keep_recent: int,
) -> set[str]:
    protected_run_ids: set[str] = set()
    for session in sessions:
        session_runs = runs_by_session.get(session.id, [])
        protected_run_ids.update(run.id for run in session_runs[:keep_recent])
        protected_run_ids.update(_session_head_run_ids(session))
        protected_run_ids.update(_active_run_restore_ids(session_runs))
    for run_id in list(protected_run_ids):
        run = runs_by_id.get(run_id)
        if (
            isinstance(run, RunRecord)
            and isinstance(run.restore_from_run_id, str)
            and run.restore_from_run_id.strip() != ""
        ):
            protected_run_ids.add(run.restore_from_run_id)
    return protected_run_ids


def _session_head_run_ids(session: SessionRecord) -> set[str]:
    return {
        run_id
        for run_id in (session.head_run_id, session.head_success_run_id, session.active_run_id)
        if isinstance(run_id, str) and run_id.strip() != ""
    }


def _active_run_restore_ids(runs: list[RunRecord]) -> set[str]:
    protected_run_ids: set[str] = set()
    for run in runs:
        if run.status in _ACTIVE_RUN_STATUSES:
            protected_run_ids.add(run.id)
            if isinstance(run.restore_from_run_id, str) and run.restore_from_run_id.strip() != "":
                protected_run_ids.add(run.restore_from_run_id)
    return protected_run_ids


def _prunable_run_ids(
    runs: list[RunRecord],
    protected_run_ids: set[str],
    *,
    cutoff: datetime | None,
    batch_size: int,
) -> list[str]:
    prunable_run_ids: list[str] = []
    for run in sorted(runs, key=lambda item: (item.created_at, item.session_id, item.sequence_no, item.id)):
        if run.id in protected_run_ids:
            continue
        if cutoff is not None and not _is_older_than(run.created_at, cutoff):
            continue
        prunable_run_ids.append(run.id)
        if len(prunable_run_ids) >= batch_size:
            break
    return prunable_run_ids


def _cutoff_from_days(days: int) -> datetime | None:
    if days <= 0:
        return None
    return utc_now() - timedelta(days=days)


def _is_older_than(value: datetime, cutoff: datetime) -> bool:
    return _as_utc_aware(value) < cutoff


def _schedule_finished_at(record: ScheduleRecord) -> datetime:
    for value in (record.last_fire_at, record.run_at, record.updated_at, record.created_at):
        if isinstance(value, datetime):
            return value
    return utc_now()


def _as_utc_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total_size = 0
    for child in path.rglob("*"):
        if child.is_file():
            total_size += child.stat().st_size
    return total_size


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped
