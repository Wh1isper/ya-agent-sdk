"""Tests for portable filesystem search helpers."""

import re
from pathlib import Path

from ya_agent_sdk.environment.local import LocalEnvironment
from ya_agent_sdk.toolsets.core.filesystem import _ripgrep_core
from ya_agent_sdk.toolsets.core.filesystem._search import collect_walk_entries, collect_walk_files, match_glob


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
    (tmp_path / "src" / "nested").mkdir()
    (tmp_path / "src" / "nested" / "child.py").write_text("print('child')")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_app.py").write_text("print('test')")

    async with LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path) as env:
        file_operator = env.file_operator
        assert file_operator is not None
        candidates = await collect_walk_files(file_operator, root="src")

    paths = {candidate.path for candidate in candidates}
    assert paths == {"src/app.py", "src/nested/child.py"}


async def test_collect_walk_entries_includes_directories(tmp_path: Path) -> None:
    """collect_walk_entries should preserve directory candidates for glob."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('app')")

    async with LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path) as env:
        file_operator = env.file_operator
        assert file_operator is not None
        candidates = await collect_walk_entries(file_operator, root=".")

    paths = {candidate.path for candidate in candidates}
    assert "src" in paths
    assert "src/app.py" in paths


def test_native_glob_error_falls_back_to_python(monkeypatch) -> None:
    """Invalid native glob patterns should use the Python matcher fallback."""

    def raise_native(_path: str, _pattern: str) -> bool:
        raise ValueError("native glob error")

    monkeypatch.setattr(_ripgrep_core, "_native", lambda: type("Native", (), {"match_glob": raise_native})())
    assert match_glob("src/app.py", "*.py")


async def test_streaming_search_falls_back_when_native_regex_rejects_python_pattern(tmp_path: Path) -> None:
    """Python-valid regex syntax should still work when Rust regex rejects it."""
    from ya_agent_sdk.toolsets.core.filesystem._line_search import search_file_streaming

    (tmp_path / "app.py").write_text("foo\nfoobar\n", encoding="utf-8")
    async with LocalEnvironment(allowed_paths=[tmp_path], default_path=tmp_path, tmp_base_dir=tmp_path) as env:
        file_operator = env.file_operator
        assert file_operator is not None
        result = await search_file_streaming(
            file_operator,
            "app.py",
            re.compile(r"foo(?=bar)"),
            context_lines=0,
            max_matches_per_file=-1,
        )

    assert list(result.matches) == ["app.py:2"]
