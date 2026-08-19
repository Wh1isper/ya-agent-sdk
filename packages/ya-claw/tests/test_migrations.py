from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from ya_claw.cli import ClawCliApplication


def test_alembic_head_matches_orm_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'schema.sqlite3'}"
    monkeypatch.setenv("YA_CLAW_DATABASE_URL", database_url)
    application = ClawCliApplication()

    application.upgrade_database()
    command.check(application.alembic_config())
