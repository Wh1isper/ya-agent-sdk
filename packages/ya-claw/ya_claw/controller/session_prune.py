from __future__ import annotations

import shutil
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy import delete, inspect, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ya_claw.config import ClawSettings
from ya_claw.controller.session_lifecycle import SESSION_PRUNE_CLAIM
from ya_claw.orm.tables import (
    HeartbeatFireRecord,
    RunRecord,
    ScheduleFireRecord,
    ScheduleRecord,
    SessionRecord,
    utc_now,
)
from ya_claw.workspace.docker_lifecycle import (
    build_docker_container_labels,
    build_docker_container_ref,
    delete_docker_container_cache,
    remove_docker_container,
)
from ya_claw.workspace.provider import get_docker_container_lock

_ACTIVE_RUN_STATUSES = frozenset({"queued", "running"})


@dataclass(frozen=True, slots=True)
class _DockerSandboxAsset:
    container_ref: str
    cache_path: Path
    expected_labels: dict[str, str]


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
    failed_docker_sandbox_session_ids: list[str] = Field(default_factory=list)

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
        self.failed_docker_sandbox_session_ids.extend(other.failed_docker_sandbox_session_ids)


class SessionPruneController:
    async def release_all_prune_claims(self, db_session: AsyncSession) -> None:
        connection = await db_session.connection()
        has_sessions_table = await connection.run_sync(
            lambda sync_connection: inspect(sync_connection).has_table(SessionRecord.__tablename__)
        )
        if not has_sessions_table:
            return
        await db_session.execute(
            update(SessionRecord).where(SessionRecord.active_run_id == SESSION_PRUNE_CLAIM).values(active_run_id=None)
        )
        await db_session.commit()

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
        claim_retry_limit = max(batch_size // 2, 1)
        claimed_session_ids = await self._select_claimed_session_ids(
            db_session,
            batch_size=claim_retry_limit,
        )
        # Keep at least one fresh-candidate slot even when batch_size is one so
        # permanently failing claims cannot stop the queue from advancing.
        fresh_budget = max(batch_size - len(claimed_session_ids), 1)
        excluded_session_ids = set(claimed_session_ids)
        fresh_session_ids = await self._select_heartbeat_session_ids(
            db_session,
            settings,
            batch_size=fresh_budget,
            excluded_session_ids=excluded_session_ids,
        )
        excluded_session_ids.update(fresh_session_ids)
        remaining = fresh_budget - len(fresh_session_ids)
        if remaining > 0:
            fresh_session_ids.extend(
                await self._select_schedule_session_ids(
                    db_session,
                    settings,
                    batch_size=remaining,
                    excluded_session_ids=excluded_session_ids,
                )
            )
        return _dedupe([*claimed_session_ids, *fresh_session_ids])

    async def _select_claimed_session_ids(self, db_session: AsyncSession, *, batch_size: int) -> list[str]:
        result = await db_session.execute(
            select(SessionRecord.id)
            .where(SessionRecord.active_run_id == SESSION_PRUNE_CLAIM)
            .order_by(SessionRecord.updated_at.asc(), SessionRecord.id.asc())
            .limit(batch_size)
        )
        return [session_id for session_id in result.scalars().all() if isinstance(session_id, str)]

    async def _select_heartbeat_session_ids(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        *,
        batch_size: int,
        excluded_session_ids: set[str] | None = None,
    ) -> list[str]:
        excluded = excluded_session_ids or set()
        keep_recent = max(settings.session_prune_heartbeat_keep_recent, 1)
        cutoff = _cutoff_from_days(settings.session_prune_heartbeat_older_than_days)
        claimed_session_ids = select(SessionRecord.id).where(SessionRecord.active_run_id == SESSION_PRUNE_CLAIM)
        result = await db_session.execute(
            select(HeartbeatFireRecord)
            .where(
                HeartbeatFireRecord.session_id.is_not(None),
                HeartbeatFireRecord.session_id.not_in(claimed_session_ids),
            )
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
            if isinstance(record.session_id, str) and record.session_id not in excluded:
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
        excluded_session_ids: set[str] | None = None,
    ) -> list[str]:
        excluded = excluded_session_ids or set()
        keep_recent = max(settings.session_prune_schedule_keep_recent, 1)
        cutoff = _cutoff_from_days(settings.session_prune_schedule_older_than_days)
        claimed_session_ids = select(SessionRecord.id).where(SessionRecord.active_run_id == SESSION_PRUNE_CLAIM)
        result = await db_session.execute(
            select(ScheduleFireRecord)
            .where(
                ScheduleFireRecord.created_session_id.is_not(None),
                ScheduleFireRecord.created_session_id.not_in(claimed_session_ids),
            )
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
                if isinstance(record.created_session_id, str) and record.created_session_id not in excluded:
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
        requested_session_ids = _dedupe(session_ids)
        candidates = await self._filter_deletable_session_ids(db_session, requested_session_ids)
        rejected_session_ids = [session_id for session_id in requested_session_ids if session_id not in candidates]
        if rejected_session_ids:
            await self._release_prune_claims(db_session, rejected_session_ids)
        claimed_session_ids = await self._claim_sessions_for_prune(db_session, candidates)
        if not claimed_session_ids:
            return SessionPruneResult()

        cleaned_session_ids, failed_docker_sandbox_session_ids = await self._cleanup_docker_sandboxes(
            db_session,
            settings,
            claimed_session_ids,
        )
        if not cleaned_session_ids:
            return SessionPruneResult(
                failed_docker_sandbox_session_ids=failed_docker_sandbox_session_ids,
            )

        claimed_result = await db_session.execute(
            select(SessionRecord.id).where(
                SessionRecord.id.in_(cleaned_session_ids),
                SessionRecord.active_run_id == SESSION_PRUNE_CLAIM,
            )
        )
        deletable_session_ids = [
            session_id for session_id in claimed_result.scalars().all() if isinstance(session_id, str)
        ]
        if not deletable_session_ids:
            return SessionPruneResult(
                failed_docker_sandbox_session_ids=failed_docker_sandbox_session_ids,
            )

        runs_result = await db_session.execute(
            select(RunRecord.id).where(RunRecord.session_id.in_(deletable_session_ids))
        )
        run_ids = [run_id for run_id in runs_result.scalars().all() if isinstance(run_id, str)]
        run_paths = [settings.run_store_dir / run_id for run_id in run_ids]
        reclaimed_bytes = sum(_path_size(path) for path in run_paths)

        await db_session.execute(
            update(SessionRecord)
            .where(SessionRecord.parent_session_id.in_(deletable_session_ids))
            .values(parent_session_id=None)
        )
        if run_ids:
            await db_session.execute(delete(RunRecord).where(RunRecord.id.in_(run_ids)))
        await db_session.execute(
            delete(SessionRecord).where(
                SessionRecord.id.in_(deletable_session_ids),
                SessionRecord.active_run_id == SESSION_PRUNE_CLAIM,
            )
        )
        await db_session.commit()

        file_result = self._delete_run_store_paths(run_paths)
        return SessionPruneResult(
            deleted_runs=len(run_ids),
            deleted_sessions=len(deletable_session_ids),
            reclaimed_bytes=reclaimed_bytes,
            failed_run_store_paths=file_result.failed_run_store_paths,
            failed_docker_sandbox_session_ids=failed_docker_sandbox_session_ids,
        )

    async def _claim_sessions_for_prune(
        self,
        db_session: AsyncSession,
        session_ids: list[str],
    ) -> list[str]:
        if not session_ids:
            return []
        claimable_active_run = or_(
            SessionRecord.active_run_id.is_(None),
            SessionRecord.active_run_id == SESSION_PRUNE_CLAIM,
        )
        # Acquire a write lock before the active-run recheck. This is a no-op update so
        # SQLite serializes writers too; the follow-up SELECT gets a fresh PostgreSQL
        # READ COMMITTED snapshot after any concurrent run-creation transaction commits.
        await db_session.execute(
            update(SessionRecord)
            .where(
                SessionRecord.id.in_(session_ids),
                claimable_active_run,
            )
            .values(
                active_run_id=SessionRecord.active_run_id,
                updated_at=SessionRecord.updated_at,
            )
        )
        eligible_result = await db_session.execute(
            select(SessionRecord.id).where(
                SessionRecord.id.in_(session_ids),
                claimable_active_run,
            )
        )
        eligible_session_ids = [
            session_id for session_id in eligible_result.scalars().all() if isinstance(session_id, str)
        ]
        # Reference-creation paths lock the referenced session row before commit.
        # Rechecking every deletability condition while those row locks are held
        # closes races with new runs, forks, schedules, and restore references.
        claimable_session_ids = await self._filter_deletable_session_ids(db_session, eligible_session_ids)
        if claimable_session_ids:
            await db_session.execute(
                update(SessionRecord)
                .where(
                    SessionRecord.id.in_(claimable_session_ids),
                    SessionRecord.active_run_id.is_(None),
                )
                .values(active_run_id=SESSION_PRUNE_CLAIM)
            )
        await db_session.commit()
        result = await db_session.execute(
            select(SessionRecord.id).where(
                SessionRecord.id.in_(session_ids),
                SessionRecord.active_run_id == SESSION_PRUNE_CLAIM,
            )
        )
        return [session_id for session_id in result.scalars().all() if isinstance(session_id, str)]

    async def _release_prune_claims(self, db_session: AsyncSession, session_ids: list[str]) -> None:
        if not session_ids:
            return
        await db_session.execute(
            update(SessionRecord)
            .where(
                SessionRecord.id.in_(session_ids),
                SessionRecord.active_run_id == SESSION_PRUNE_CLAIM,
            )
            .values(active_run_id=None)
        )
        await db_session.commit()

    async def _cleanup_docker_sandboxes(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        session_ids: list[str],
    ) -> tuple[list[str], list[str]]:
        records_by_id, assets_by_session_id, invalid_session_ids = await _load_docker_sandbox_assets(
            db_session,
            settings,
            session_ids,
        )
        cleaned_session_ids: list[str] = []
        failed_session_ids: list[str] = []
        for session_id in session_ids:
            if not isinstance(records_by_id.get(session_id), SessionRecord):
                continue
            if session_id in invalid_session_ids:
                failed_session_ids.append(session_id)
                logger.warning(
                    "Session prune retained Docker-backed session with incomplete sandbox identity session_id={}",
                    session_id,
                )
                continue
            assets = _dedupe_docker_assets(assets_by_session_id.get(session_id, []))
            failed_asset = await _cleanup_docker_sandbox_assets(assets)
            if failed_asset is None:
                cleaned_session_ids.append(session_id)
                continue

            failed_session_ids.append(session_id)
            logger.warning(
                "Session prune retained Docker-backed session after sandbox cleanup failure session_id={} "
                "container_ref={} cache_path={}",
                session_id,
                failed_asset.container_ref,
                failed_asset.cache_path,
            )
        return cleaned_session_ids, failed_session_ids

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
            if (
                isinstance(record.active_run_id, str)
                and record.active_run_id.strip() != ""
                and record.active_run_id != SESSION_PRUNE_CLAIM
            ):
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


async def _load_docker_sandbox_assets(
    db_session: AsyncSession,
    settings: ClawSettings,
    session_ids: list[str],
) -> tuple[dict[str, SessionRecord], dict[str, list[_DockerSandboxAsset]], set[str]]:
    records_result = await db_session.execute(select(SessionRecord).where(SessionRecord.id.in_(session_ids)))
    records_by_id = {
        record.id: record for record in records_result.scalars().all() if isinstance(record, SessionRecord)
    }
    runs_result = await db_session.execute(select(RunRecord).where(RunRecord.session_id.in_(session_ids)))
    assets_by_session_id: dict[str, list[_DockerSandboxAsset]] = defaultdict(list)
    invalid_session_ids: set[str] = set()
    for record in records_by_id.values():
        _record_docker_sandbox_asset(
            settings,
            session_id=record.id,
            run_id=None,
            metadata=record.session_metadata,
            assets_by_session_id=assets_by_session_id,
            invalid_session_ids=invalid_session_ids,
        )
    for run_record in runs_result.scalars().all():
        if isinstance(run_record, RunRecord):
            _record_docker_sandbox_asset(
                settings,
                session_id=run_record.session_id,
                run_id=run_record.id,
                metadata=run_record.run_metadata,
                assets_by_session_id=assets_by_session_id,
                invalid_session_ids=invalid_session_ids,
            )
    return records_by_id, assets_by_session_id, invalid_session_ids


def _record_docker_sandbox_asset(
    settings: ClawSettings,
    *,
    session_id: str,
    run_id: str | None,
    metadata: object,
    assets_by_session_id: dict[str, list[_DockerSandboxAsset]],
    invalid_session_ids: set[str],
) -> None:
    asset, valid = _docker_sandbox_asset(
        settings,
        session_id=session_id,
        run_id=run_id,
        metadata=metadata,
    )
    if not valid:
        invalid_session_ids.add(session_id)
    elif asset is not None:
        assets_by_session_id[session_id].append(asset)


async def _cleanup_docker_sandbox_assets(
    assets: list[_DockerSandboxAsset],
) -> _DockerSandboxAsset | None:
    for asset in assets:
        lock = get_docker_container_lock(
            cache_path=asset.cache_path,
            container_ref=asset.container_ref,
        )
        async with lock:
            container_removed = await remove_docker_container(
                asset.container_ref,
                expected_labels=asset.expected_labels,
            )
            cache_deleted = container_removed and await delete_docker_container_cache(asset.cache_path)
        if not container_removed or not cache_deleted:
            return asset
    return None


def _docker_sandbox_asset(
    settings: ClawSettings,
    *,
    session_id: str,
    run_id: str | None,
    metadata: object,
) -> tuple[_DockerSandboxAsset | None, bool]:
    if not isinstance(metadata, dict):
        return None, True
    sandbox = metadata.get("sandbox")
    if not isinstance(sandbox, dict) or sandbox.get("provider") != "docker":
        return None, True

    scope = sandbox.get("scope")
    generation = sandbox.get("generation")
    metadata_session_id = sandbox.get("session_id")
    metadata_run_id = sandbox.get("run_id")
    if (
        scope not in {"session", "run"}
        or not isinstance(generation, int)
        or isinstance(generation, bool)
        or generation < 1
        or metadata_session_id != session_id
        or not _is_safe_identity_component(session_id)
    ):
        return None, False
    if scope == "run" and (run_id is None or metadata_run_id != run_id or not _is_safe_identity_component(run_id)):
        return None, False
    expected_run_id = run_id if scope == "run" else None
    try:
        container_ref = build_docker_container_ref(
            scope=scope,
            session_id=session_id,
            run_id=expected_run_id,
            generation=generation,
        )
        expected_labels = build_docker_container_labels(
            scope=scope,
            session_id=session_id,
            run_id=expected_run_id,
            generation=generation,
        )
    except ValueError:
        return None, False

    cache_root = settings.resolved_workspace_provider_docker_container_cache_dir
    if scope == "run" and run_id is not None:
        cache_path = cache_root / "runs" / run_id / "workspace.json"
    else:
        cache_path = cache_root / "sessions" / session_id / "workspace.json"
    return (
        _DockerSandboxAsset(
            container_ref=container_ref,
            cache_path=cache_path,
            expected_labels=expected_labels,
        ),
        True,
    )


def _dedupe_docker_assets(assets: list[_DockerSandboxAsset]) -> list[_DockerSandboxAsset]:
    seen: set[tuple[str, Path, tuple[tuple[str, str], ...]]] = set()
    deduped: list[_DockerSandboxAsset] = []
    for asset in assets:
        key = (asset.container_ref, asset.cache_path, tuple(sorted(asset.expected_labels.items())))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(asset)
    return deduped


def _is_safe_identity_component(value: str) -> bool:
    return value.strip() == value and value not in {"", ".", ".."} and "/" not in value and "\\" not in value


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped
