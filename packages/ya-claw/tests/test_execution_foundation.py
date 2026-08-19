from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from ya_claw.config import ClawSettings
from ya_claw.controller.models import CommandPart, ModePart, TextPart, UrlPart
from ya_claw.db.engine import create_engine, create_session_factory
from ya_claw.execution.checkpoint import build_message_checkpoint, commit_run_artifacts, write_message_checkpoint
from ya_claw.execution.input import map_input_parts, split_input_parts
from ya_claw.execution.restore import load_restore_point, resolve_restore_run
from ya_claw.execution.state_machine import complete_run, fail_run, interrupt_run, mark_run_running, queue_run
from ya_claw.execution.store import RunStore
from ya_claw.orm.tables import RunRecord, SessionRecord


@pytest.fixture
async def db_engine(tmp_path: Path, initialize_sqlite_database: Callable[[str], None]) -> AsyncEngine:
    database_url = f"sqlite+aiosqlite:///{(tmp_path / 'execution.sqlite3').resolve()}"
    initialize_sqlite_database(database_url)
    engine = create_engine(database_url)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture
async def db_session(db_engine: AsyncEngine) -> AsyncSession:
    session_factory = create_session_factory(db_engine)
    async with session_factory() as session:
        yield session


@pytest.fixture
def settings(tmp_path: Path) -> ClawSettings:
    data_dir = tmp_path / "runtime-data"
    workspace_dir = tmp_path / "workspace"
    data_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    return ClawSettings(api_token="test-token", data_dir=data_dir, workspace_dir=workspace_dir)  # noqa: S106


async def test_split_and_map_input_parts() -> None:
    parts = [
        ModePart(type="mode", mode="plan"),
        CommandPart(type="command", name="loop"),
        TextPart(type="text", text="hello"),
        UrlPart(type="url", url="https://example.com/a.png", kind="image"),
    ]

    split_result = split_input_parts(parts)
    mapped = await map_input_parts(parts)

    assert [part.mode for part in split_result.mode_parts] == ["plan"]
    assert [part.name for part in split_result.command_parts] == ["loop"]
    assert len(split_result.content_parts) == 2
    assert mapped.input_preview == "hello"
    assert mapped.user_prompt[0] == "hello"


async def test_run_store_checkpoint_and_commit(settings: ClawSettings) -> None:
    run_store = RunStore(settings)
    checkpoint = build_message_checkpoint(
        run_id="run-1",
        session_id="session-1",
        checkpoint_kind="model_response_end",
        message=[{"role": "assistant", "content": "done"}],
        created_at=datetime.now(UTC),
    )

    write_message_checkpoint(run_store, checkpoint)
    commit_run_artifacts(
        run_store,
        run_id="run-1",
        session_id="session-1",
        state={"exported_state": {"ok": True}},
        message=[{"role": "assistant", "content": "done"}],
        committed_at=datetime.now(UTC),
    )

    assert run_store.has_state("run-1") is True
    assert run_store.has_message("run-1") is True
    assert run_store.read_state("run-1") is not None
    assert run_store.read_message("run-1") == [{"role": "assistant", "content": "done"}]


async def test_run_store_rejects_non_array_message_payload(settings: ClawSettings) -> None:
    run_store = RunStore(settings)
    run_store.message_path("run-invalid").write_text('{"events": []}', encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid AGUI replay list"):
        run_store.read_message("run-invalid")


async def test_state_machine_and_restore_loader(
    db_session: AsyncSession,
    settings: ClawSettings,
) -> None:
    session = SessionRecord(id="session-1", profile_name="general", session_metadata={})
    run = RunRecord(
        id="run-1",
        session_id="session-1",
        sequence_no=1,
        restore_from_run_id=None,
        status="queued",
        trigger_type="api",
        profile_name="general",
        input_parts=[{"type": "text", "text": "hello"}],
        run_metadata={},
    )
    db_session.add(session)
    db_session.add(run)
    await db_session.commit()

    queue_run(session, run)
    assert session.active_run_id is None

    mark_run_running(session, run)
    assert session.active_run_id == run.id

    complete_run(session, run)
    await db_session.commit()

    run_store = RunStore(settings)
    commit_run_artifacts(
        run_store,
        run_id="run-1",
        session_id="session-1",
        state={"exported_state": {"ok": True}},
        message=[],
    )

    restore_run = await resolve_restore_run(db_session, session, explicit_run_id=None)
    restore_point = await load_restore_point(db_session, run_store, session, explicit_run_id=None)

    assert isinstance(restore_run, RunRecord)
    assert restore_run.id == "run-1"
    assert restore_point is not None
    assert restore_point.run_id == "run-1"
    assert restore_point.state is not None
    assert restore_point.message is not None

    failed_run = RunRecord(
        id="run-2",
        session_id="session-1",
        sequence_no=2,
        restore_from_run_id="run-1",
        status="running",
        trigger_type="api",
        profile_name="general",
        input_parts=[{"type": "text", "text": "retry"}],
        run_metadata={},
    )
    db_session.add(failed_run)
    await db_session.commit()

    mark_run_running(session, failed_run)
    fail_run(session, failed_run)
    await db_session.commit()
    explicit_failed = await resolve_restore_run(db_session, session, explicit_run_id="run-2")
    assert isinstance(explicit_failed, RunRecord)
    assert explicit_failed.id == "run-2"

    interrupted_run = RunRecord(
        id="run-3",
        session_id="session-1",
        sequence_no=3,
        restore_from_run_id="run-2",
        status="running",
        trigger_type="api",
        profile_name="general",
        input_parts=[{"type": "text", "text": "retry again"}],
        run_metadata={},
    )
    db_session.add(interrupted_run)
    await db_session.commit()

    mark_run_running(session, interrupted_run)
    interrupt_run(session, interrupted_run)
    await db_session.commit()
    explicit_interrupted = await resolve_restore_run(db_session, session, explicit_run_id="run-3")
    assert isinstance(explicit_interrupted, RunRecord)
    assert explicit_interrupted.id == "run-3"


async def test_restore_loader_accepts_ancestor_and_rejects_sibling(
    db_session: AsyncSession,
) -> None:
    parent = SessionRecord(id="parent", profile_name="general", session_metadata={})
    child = SessionRecord(id="child", parent_session_id="parent", profile_name="general", session_metadata={})
    sibling = SessionRecord(id="sibling", parent_session_id="parent", profile_name="general", session_metadata={})
    parent_run = RunRecord(
        id="parent-run",
        session_id="parent",
        sequence_no=1,
        status="completed",
        trigger_type="api",
        input_parts=[],
        run_metadata={},
    )
    sibling_run = RunRecord(
        id="sibling-run",
        session_id="sibling",
        sequence_no=1,
        status="completed",
        trigger_type="api",
        input_parts=[],
        run_metadata={},
    )
    db_session.add_all([parent, child, sibling, parent_run, sibling_run])
    await db_session.commit()

    restored = await resolve_restore_run(db_session, child, explicit_run_id=parent_run.id)
    assert restored is parent_run
    with pytest.raises(HTTPException) as exc_info:
        await resolve_restore_run(db_session, child, explicit_run_id=sibling_run.id)
    assert exc_info.value.status_code == 422
