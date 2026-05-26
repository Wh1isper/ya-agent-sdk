"""Portable fileops-first search helpers."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from ya_agent_environment import FileOperator

from ya_agent_sdk.toolsets.core.filesystem import _ripgrep_core
from ya_agent_sdk.toolsets.core.filesystem._gitignore import GitignoreFilterResult, filter_gitignored

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True)
class SearchCandidate:
    """Logical filesystem candidate for glob/grep tools."""

    path: str
    size: int | None = None
    mtime: float | None = None


def normalize_logical_path(path: str) -> str:
    """Normalize a FileOperator logical path for matching and output."""
    normalized = path.replace("\\", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized or "."


def is_hidden_logical_path(path: str) -> bool:
    """Return True when any path segment is hidden."""
    return any(part.startswith(".") and part not in {".", ".."} for part in normalize_logical_path(path).split("/"))


def match_glob(path: str, pattern: str) -> bool:
    """Match a logical path with agent/ripgrep-style glob semantics.

    Bare file globs such as ``*.py`` match recursively. A leading slash anchors
    the pattern at the FileOperator root.
    """
    native_match = _ripgrep_core.match_glob(path, pattern)
    if native_match is not None:
        return native_match

    normalized_path = normalize_logical_path(path)
    normalized_pattern = pattern.replace("\\", "/") or "**/*"
    if normalized_pattern.startswith("./"):
        normalized_pattern = normalized_pattern[2:]

    anchored = normalized_pattern.startswith("/")
    if anchored:
        normalized_pattern = normalized_pattern.lstrip("/") or "*"
        if "/" not in normalized_pattern and "/" in normalized_path:
            return False
        return fnmatch.fnmatchcase(normalized_path, normalized_pattern)

    if normalized_pattern in {"**", "**/*"}:
        return True

    if fnmatch.fnmatchcase(normalized_path, normalized_pattern):
        return True

    if normalized_pattern.startswith("**/"):
        without_recursive_prefix = normalized_pattern[3:]
        if fnmatch.fnmatchcase(normalized_path, without_recursive_prefix):
            return True

    if "/" not in normalized_pattern:
        return fnmatch.fnmatchcase(PurePosixPath(normalized_path).name, normalized_pattern)

    return False


def walk_max_depth_for_glob(pattern: str) -> int | None:
    """Return a safe walk depth limit for anchored non-recursive glob patterns."""
    normalized_pattern = pattern.replace("\\", "/") or "**/*"
    if normalized_pattern.startswith("./"):
        normalized_pattern = normalized_pattern[2:]
    if not normalized_pattern.startswith("/"):
        return None

    anchored_pattern = normalized_pattern.lstrip("/")
    if "**" in anchored_pattern:
        return None
    if anchored_pattern == "":
        return 0
    return anchored_pattern.count("/")


async def collect_walk_entries(
    file_operator: FileOperator,
    *,
    root: str = ".",
    include_hidden: bool = False,
    max_depth: int | None = None,
) -> list[SearchCandidate]:
    """Collect file and directory candidates through FileOperator.walk_files."""
    candidates: list[SearchCandidate] = []
    async for entry in file_operator.walk_files(root, include_hidden=include_hidden, max_depth=max_depth):
        path = normalize_logical_path(entry["path"])
        if not include_hidden and is_hidden_logical_path(path):
            continue
        candidates.append(SearchCandidate(path=path, size=entry.get("size"), mtime=entry.get("mtime")))
    return candidates


async def collect_walk_files(
    file_operator: FileOperator,
    *,
    root: str = ".",
    include_hidden: bool = False,
    max_depth: int | None = None,
) -> list[SearchCandidate]:
    """Collect regular file candidates through FileOperator.walk_files."""
    candidates: list[SearchCandidate] = []
    async for entry in file_operator.walk_files(root, include_hidden=include_hidden, max_depth=max_depth):
        if not entry.get("is_file", False):
            continue
        path = normalize_logical_path(entry["path"])
        if not include_hidden and is_hidden_logical_path(path):
            continue
        candidates.append(SearchCandidate(path=path, size=entry.get("size"), mtime=entry.get("mtime")))
    return candidates


def filter_candidates_by_glob(candidates: Iterable[SearchCandidate], pattern: str) -> list[SearchCandidate]:
    """Filter candidates with match_glob."""
    candidates_list = list(candidates)
    if not candidates_list:
        return []
    native_matches = _ripgrep_core.match_globs([candidate.path for candidate in candidates_list], pattern)
    if native_matches is not None:
        return [candidate for candidate, matched in zip(candidates_list, native_matches, strict=True) if matched]
    return [candidate for candidate in candidates_list if match_glob(candidate.path, pattern)]


async def filter_candidates_ignored(
    candidates: list[SearchCandidate],
    file_operator: FileOperator,
) -> tuple[list[SearchCandidate], GitignoreFilterResult]:
    """Filter candidates through the existing gitignore helper."""
    paths = [candidate.path for candidate in candidates]
    filter_result = await filter_gitignored(paths, file_operator)
    kept_set = set(filter_result.kept)
    return [candidate for candidate in candidates if candidate.path in kept_set], filter_result


def sort_candidates_by_mtime(candidates: list[SearchCandidate]) -> list[SearchCandidate]:
    """Sort candidates by modification time descending."""
    return sorted(candidates, key=lambda candidate: candidate.mtime or 0.0, reverse=True)


async def collect_glob_candidates(
    file_operator: FileOperator,
    pattern: str,
    *,
    root: str = ".",
    include_ignored: bool = False,
    include_hidden: bool = False,
) -> tuple[list[SearchCandidate], GitignoreFilterResult | None]:
    """Collect glob candidates through walk_files, glob matching, and ignore filtering."""
    candidates = await collect_walk_entries(
        file_operator,
        root=root,
        include_hidden=include_hidden,
        max_depth=walk_max_depth_for_glob(pattern),
    )
    candidates = filter_candidates_by_glob(candidates, pattern)
    filter_result: GitignoreFilterResult | None = None
    if not include_ignored:
        candidates, filter_result = await filter_candidates_ignored(candidates, file_operator)
    return sort_candidates_by_mtime(candidates), filter_result


__all__ = [
    "SearchCandidate",
    "collect_glob_candidates",
    "collect_walk_entries",
    "collect_walk_files",
    "filter_candidates_by_glob",
    "filter_candidates_ignored",
    "is_hidden_logical_path",
    "match_glob",
    "normalize_logical_path",
    "sort_candidates_by_mtime",
    "walk_max_depth_for_glob",
]
