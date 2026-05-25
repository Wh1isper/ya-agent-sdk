"""Tests for portable filesystem search helpers."""

from pathlib import Path

from ya_agent_sdk.environment.local import LocalEnvironment
from ya_agent_sdk.toolsets.core.filesystem import _ripgrep_core
from ya_agent_sdk.toolsets.core.filesystem._search import collect_walk_files, match_glob


def test_match_glob_bare_pattern_matches_recursively() -> None:
    """Bare glob patterns should match file names at any depth."""
    assert match_glob("main.py", "*.py")
    assert match_glob("src/app/main.py", "*.py")
    assert not match_glob("src/app/main.txt", "*.py")


def test_match_glob_recursive_pattern_matches_root_and_nested() -> None:
    """Recursive glob patterns should include root-level files."""
    assert match_glob("main.py", "**/*.py")
    assert match_glob("src/app/main.py", "**/*.py")


def test_match_glob_leading_slash_anchors_to_root() -> None:
    """Leading slash should anchor a glob to the FileOperator root."""
    assert match_glob("main.py", "/*.py")
    assert not match_glob("src/main.py", "/*.py")


def test_ripgrep_core_disable_env_forces_python_backend(monkeypatch) -> None:
    """YA_RIPGREP_CORE_DISABLE should disable the native extension adapter."""
    monkeypatch.setenv("YA_RIPGREP_CORE_DISABLE", "1")
    _ripgrep_core._native.cache_clear()
    try:
        assert _ripgrep_core.is_disabled()
        assert not _ripgrep_core.is_available()
        assert _ripgrep_core.match_glob("src/main.py", "*.py") is None
        assert match_glob("src/main.py", "*.py")
    finally:
        monkeypatch.delenv("YA_RIPGREP_CORE_DISABLE", raising=False)
        _ripgrep_core._native.cache_clear()


async def test_collect_walk_files_honors_root(tmp_path: Path) -> None:
    """collect_walk_files should traverse from the requested logical root."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('app')")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_app.py").write_text("print('test')")

    async with LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path) as env:
        file_operator = env.file_operator
        assert file_operator is not None
        candidates = await collect_walk_files(file_operator, root="src")

    paths = {candidate.path for candidate in candidates}
    assert paths == {"src/app.py"}
