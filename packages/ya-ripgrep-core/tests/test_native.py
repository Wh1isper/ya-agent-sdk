"""Tests for the native ripgrep core extension."""

import ya_ripgrep_core


def test_match_glob_bare_pattern_matches_recursively() -> None:
    assert ya_ripgrep_core.match_glob("main.py", "*.py")
    assert ya_ripgrep_core.match_glob("src/main.py", "*.py")
    assert not ya_ripgrep_core.match_glob("src/main.txt", "*.py")


def test_match_glob_recursive_pattern_matches_root_and_nested() -> None:
    assert ya_ripgrep_core.match_glob("main.py", "**/*.py")
    assert ya_ripgrep_core.match_glob("src/main.py", "**/*.py")


def test_match_glob_leading_slash_anchors_to_root() -> None:
    assert ya_ripgrep_core.match_glob("main.py", "/*.py")
    assert not ya_ripgrep_core.match_glob("src/main.py", "/*.py")


def test_rust_glob_matches_many_paths() -> None:
    matcher = ya_ripgrep_core.RustGlob("*.py")
    assert matcher.match_many(["main.py", "src/main.py", "src/main.txt"]) == [True, True, False]
    assert ya_ripgrep_core.match_globs(["main.py", "src/main.py", "src/main.txt"], "*.py") == [True, True, False]


def test_rust_regex_matches_lines() -> None:
    matcher = ya_ripgrep_core.RustRegex(r"def \w+")
    assert matcher.is_match("def hello():")
    assert not matcher.is_match("class Hello:")


def test_rust_regex_searches_bytes_with_context_and_limit() -> None:
    matcher = ya_ripgrep_core.RustRegex(r"TODO|FIXME")
    data = b"before\nTODO one\nafter\nFIXME two\nend\n"
    assert matcher.search_bytes(data, context_lines=1, max_matches=1) == [
        (2, "TODO one", "before\nTODO one\nafter\n", 1)
    ]
    assert matcher.search_bytes(data, context_lines=0, max_matches=0) == [
        (2, "TODO one", "TODO one\n", 2),
        (4, "FIXME two", "FIXME two\n", 4),
    ]
    assert ya_ripgrep_core.regex_search_bytes(r"FIXME", data, context_lines=1, max_matches=-1) == [
        (4, "FIXME two", "after\nFIXME two\nend\n", 3)
    ]


def test_rust_regex_searches_utf8_chinese_bytes() -> None:
    matcher = ya_ripgrep_core.RustRegex("性能优化|中文_TOKEN")
    data = "开头\n这里需要性能优化\n普通行\n中文_TOKEN 命中\n".encode()
    assert matcher.search_bytes(data, context_lines=0, max_matches=0) == [
        (2, "这里需要性能优化", "这里需要性能优化\n", 2),
        (4, "中文_TOKEN 命中", "中文_TOKEN 命中\n", 4),
    ]
    assert matcher.search_bytes(data, context_lines=1, max_matches=1) == [
        (2, "这里需要性能优化", "开头\n这里需要性能优化\n普通行\n", 1)
    ]
