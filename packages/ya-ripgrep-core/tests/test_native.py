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


def test_rust_regex_matches_lines() -> None:
    matcher = ya_ripgrep_core.RustRegex(r"def \w+")
    assert matcher.is_match("def hello():")
    assert not matcher.is_match("class Hello:")
