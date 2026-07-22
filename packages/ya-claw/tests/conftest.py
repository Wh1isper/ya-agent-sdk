from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
import ya_claw.orm.tables  # noqa: F401
from sqlalchemy import create_engine as create_sync_engine
from sqlalchemy.engine import make_url
from ya_claw.orm.base import Base


@pytest.fixture(scope="session")
def initialize_sqlite_database(tmp_path_factory: pytest.TempPathFactory) -> Callable[[str], None]:
    """Initialize isolated SQLite databases from one schema-only template per worker."""
    template_path = tmp_path_factory.mktemp("sqlite-schema") / "template.sqlite3"
    template_engine = create_sync_engine(f"sqlite:///{template_path}")
    try:
        Base.metadata.create_all(template_engine)
    finally:
        template_engine.dispose()

    def initialize(database_url: str) -> None:
        url = make_url(database_url)
        if not url.drivername.startswith("sqlite") or url.database in (None, ":memory:"):
            raise ValueError("test database must use a file-backed SQLite URL")
        database_path = Path(url.database)
        database_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(template_path, database_path)

    return initialize
