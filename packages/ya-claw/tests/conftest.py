from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine as create_sync_engine
from sqlalchemy import insert
from sqlalchemy.engine import make_url
from ya_agent_sdk.inputs import EnqueueReceipt, InputDisposition
from ya_claw.cli import ClawCliApplication
from ya_claw.orm.tables import ProfileRecord
from ya_claw.runtime_state import InMemoryRuntimeState


@pytest.fixture(scope="session")
def initialize_sqlite_database(tmp_path_factory: pytest.TempPathFactory) -> Callable[[str], None]:
    """Initialize isolated SQLite databases from one schema-only template per worker."""
    template_path = tmp_path_factory.mktemp("sqlite-schema") / "template.sqlite3"
    database_url = f"sqlite+aiosqlite:///{template_path}"
    previous_url = os.environ.get("YA_CLAW_DATABASE_URL")
    os.environ["YA_CLAW_DATABASE_URL"] = database_url
    try:
        ClawCliApplication().upgrade_database()
    finally:
        if previous_url is None:
            os.environ.pop("YA_CLAW_DATABASE_URL", None)
        else:
            os.environ["YA_CLAW_DATABASE_URL"] = previous_url

    def initialize(database_url: str, *, profile_names: tuple[str, ...] = ()) -> None:
        url = make_url(database_url)
        if not url.drivername.startswith("sqlite") or url.database in (None, ":memory:"):
            raise ValueError("test database must use a file-backed SQLite URL")
        database_path = Path(url.database)
        database_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(template_path, database_path)
        unique_profile_names = tuple(dict.fromkeys(profile_names))
        if not unique_profile_names:
            return
        seed_engine = create_sync_engine(f"sqlite:///{database_path}")
        try:
            with seed_engine.begin() as connection:
                connection.execute(
                    insert(ProfileRecord),
                    [
                        {
                            "name": name,
                            "agent_spec": {"model": "test"},
                            "host_config": {},
                            "subagent_specs": [],
                            "enabled": True,
                            "source_type": "test",
                            "source_version": "1",
                        }
                        for name in unique_profile_names
                    ],
                )
        finally:
            seed_engine.dispose()

    return initialize


@pytest.fixture
def bind_recording_input_ingress() -> Callable[[InMemoryRuntimeState, str], list[list[dict[str, Any]]]]:
    def bind(runtime_state: InMemoryRuntimeState, run_id: str) -> list[list[dict[str, Any]]]:
        batches: list[list[dict[str, Any]]] = []

        async def ingress(input_id: str, input_parts: list[dict[str, Any]]) -> EnqueueReceipt:
            batches.append(input_parts)
            index = len(batches)
            return EnqueueReceipt(
                logical_run_id=run_id,
                input_id=f"sdk-{input_id}",
                disposition=InputDisposition.enqueued,
                enqueue_id=f"enqueue-{index}",
            )

        runtime_state.bind_input_ingress(run_id, ingress)
        return batches

    return bind
