from pathlib import Path

from sqlalchemy import text
from ya_claw.db.engine import create_engine


async def test_file_sqlite_engine_uses_wal_and_extended_busy_timeout(tmp_path: Path) -> None:
    database_path = (tmp_path / "engine.sqlite3").resolve()
    engine = create_engine(f"sqlite+aiosqlite:///{database_path}")

    try:
        async with engine.connect() as connection:
            journal_mode = await connection.scalar(text("PRAGMA journal_mode"))
            busy_timeout = await connection.scalar(text("PRAGMA busy_timeout"))
    finally:
        await engine.dispose()

    assert journal_mode == "wal"
    assert busy_timeout == 30_000
