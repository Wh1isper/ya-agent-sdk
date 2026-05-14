from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from ya_claw.config import ClawSettings
from ya_claw.controller.workspace_runtime import WorkspaceRuntimeController, reconcile_session_sandbox_metadata
from ya_claw.db.engine import create_engine, create_session_factory
from ya_claw.orm.base import Base
from ya_claw.orm.tables import SessionRecord


async def _create_session_factory(tmp_path: Path):
    engine = create_engine(f"sqlite+aiosqlite:///{(tmp_path / 'runtime-reconcile.sqlite3').resolve()}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    return engine, create_session_factory(engine)


def _docker_settings(tmp_path: Path) -> ClawSettings:
    data_dir = tmp_path / "runtime-data"
    workspace_dir = tmp_path / "workspace"
    data_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    return ClawSettings(
        api_token="test-token",  # noqa: S106
        data_dir=data_dir,
        workspace_dir=workspace_dir,
        workspace_provider_backend="docker",
        _env_file=None,
    )


@pytest.mark.asyncio
async def test_reconcile_marks_stopped_snapshot_running_when_container_is_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_inspect(container_ref: str) -> dict[str, str | None]:
        assert container_ref == "container-running"
        return {"container_id": "container-running", "status": "running"}

    monkeypatch.setattr("ya_claw.controller.workspace_runtime._inspect_docker_container", fake_inspect)
    engine, session_factory = await _create_session_factory(tmp_path)
    try:
        async with session_factory() as db_session:
            session_record = SessionRecord(
                id="session-reconcile-running",
                profile_name="default",
                session_metadata={
                    "sandbox": {
                        "provider": "docker",
                        "scope": "session",
                        "status": "stopped",
                        "ready_state": "not_started",
                        "container_ref": "workspace-ref",
                        "container_id": "container-running",
                        "verified_container_id": None,
                        "retention_policy": "stop_on_idle",
                        "idle_ttl_seconds": 3600,
                        "last_used_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    }
                },
            )
            db_session.add(session_record)
            await db_session.commit()

            await reconcile_session_sandbox_metadata(
                settings=_docker_settings(tmp_path),
                db_session=db_session,
                session_record=session_record,
            )

            sandbox = session_record.session_metadata["sandbox"]
            assert sandbox["status"] == "running"
            assert sandbox["ready_state"] == "ready"
            assert sandbox["container_id"] == "container-running"
            assert sandbox["verified_container_id"] == "container-running"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_session_sandbox_api_reconciles_before_returning_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_inspect(container_ref: str) -> dict[str, str | None]:
        return {"container_id": "container-ready", "status": "running"}

    monkeypatch.setattr("ya_claw.controller.workspace_runtime._inspect_docker_container", fake_inspect)
    engine, session_factory = await _create_session_factory(tmp_path)
    try:
        async with session_factory() as db_session:
            session_record = SessionRecord(
                id="session-api-reconcile",
                profile_name="default",
                session_metadata={
                    "sandbox": {
                        "provider": "docker",
                        "scope": "session",
                        "status": "stopped",
                        "ready_state": "not_started",
                        "container_ref": "workspace-ref",
                        "container_id": "container-ready",
                        "retention_policy": "stop_on_idle",
                        "idle_ttl_seconds": 3600,
                        "last_used_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    }
                },
            )
            db_session.add(session_record)
            await db_session.commit()

            state = await WorkspaceRuntimeController().get_session_sandbox(
                settings=_docker_settings(tmp_path),
                db_session=db_session,
                session_id=session_record.id,
            )

            assert state.status == "ready"
            assert state.ready_state == "ready"
            assert state.container_id == "container-ready"
            assert state.verified_container_id == "container-ready"
    finally:
        await engine.dispose()
