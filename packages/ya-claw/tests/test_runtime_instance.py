from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine
from ya_claw.config import ClawSettings
from ya_claw.db.engine import create_engine, create_session_factory
from ya_claw.execution.instance import RuntimeInstanceManager
from ya_claw.orm.tables import RuntimeInstanceRecord


@pytest.fixture
async def db_engine(tmp_path: Path, initialize_sqlite_database: Callable[[str], None]) -> AsyncEngine:
    database_url = f"sqlite+aiosqlite:///{(tmp_path / 'instance.sqlite3').resolve()}"
    initialize_sqlite_database(database_url)
    engine = create_engine(database_url)
    try:
        yield engine
    finally:
        await engine.dispose()


def _settings(tmp_path: Path) -> ClawSettings:
    return ClawSettings(
        api_token="test-token",  # noqa: S106
        data_dir=tmp_path / "runtime-data",
        workspace_dir=tmp_path / "workspace",
        instance_id="instance-test",
    )


async def test_runtime_instance_manager_registers_heartbeats_and_marks_stopped(
    tmp_path: Path,
    db_engine: AsyncEngine,
) -> None:
    session_factory = create_session_factory(db_engine)
    manager = RuntimeInstanceManager(settings=_settings(tmp_path), session_factory=session_factory)

    registered = await manager.register(metadata={"environment": "test"})
    heartbeat = await manager.heartbeat()
    stopped = await manager.mark_stopped()

    async with session_factory() as db_session:
        record = await db_session.get(RuntimeInstanceRecord, "instance-test")

    assert registered is True
    assert heartbeat is True
    assert stopped is True
    assert isinstance(record, RuntimeInstanceRecord)
    assert record.status == "stopped"
    assert record.instance_metadata == {"environment": "test"}
    assert record.started_at is not None
    assert record.heartbeat_at is not None
    assert record.stopped_at is not None


async def test_runtime_instance_manager_tolerates_missing_table(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite+aiosqlite:///{(tmp_path / 'missing-table.sqlite3').resolve()}")
    session_factory = create_session_factory(engine)
    manager = RuntimeInstanceManager(settings=_settings(tmp_path), session_factory=session_factory)
    try:
        assert await manager.register() is False
        assert await manager.heartbeat() is False
        assert await manager.mark_stopped() is False
    finally:
        await engine.dispose()
