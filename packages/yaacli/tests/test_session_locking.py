from __future__ import annotations

from pathlib import Path

from yaacli.sessions import local_file_lock


def test_local_file_lock_creates_lock_file(tmp_path: Path) -> None:
    lock_path = tmp_path / "nested" / ".session.lock"

    with local_file_lock(lock_path):
        assert lock_path.exists()

    assert lock_path.exists()
