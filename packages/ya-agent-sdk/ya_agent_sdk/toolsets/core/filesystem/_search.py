"""Portable fileops-first search helpers."""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

import anyio
import pathspec
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


def _ignored_paths_matching_glob(paths: list[str], pattern: str) -> list[str]:
    """Return ignored paths relevant to the requested glob pattern."""
    return [path for path in paths if match_glob(path, pattern) or match_glob(path.rstrip("/"), pattern)]


def _ignored_dir_marker(path: str) -> str:
    """Normalize an ignored directory marker for summary grouping."""
    return path if path.endswith("/") else f"{path}/"


def _get_local_walk_paths(file_operator: FileOperator, root: str) -> tuple[Path, Path] | None:
    """Return local default/root paths when the operator supports local fast walking."""
    try:
        from ya_agent_sdk.environment.local import LocalFileOperator
    except Exception:
        return None

    if not isinstance(file_operator, LocalFileOperator):
        return None

    default_path = file_operator._local_default_path()
    if default_path is None:
        return None

    try:
        resolved_root = file_operator._resolve_path(root)
    except Exception:
        return None

    return default_path, resolved_root


async def _load_top_level_gitignore(file_operator: FileOperator) -> pathspec.PathSpec | None:
    """Load the top-level .gitignore used by the existing filter implementation."""
    try:
        if not await file_operator.exists(".gitignore"):
            return None
        content = await file_operator.read_file(".gitignore")
    except Exception:
        return None
    return pathspec.PathSpec.from_lines("gitignore", content.splitlines())


def _normalized_negation_pattern(pattern_text: str) -> str:
    """Normalize a gitignore negation pattern for conservative prefix checks."""
    normalized = pattern_text[1:] if pattern_text.startswith("!") else pattern_text
    normalized = normalized.lstrip("/")
    return normalized.rstrip("/")


def _contains_glob_wildcard(pattern_text: str) -> bool:
    """Return True when a gitignore pattern contains glob wildcards."""
    return any(char in pattern_text for char in "*?[")


def _negation_may_reinclude_dir(spec: pathspec.PathSpec, logical_dir: str, host_dir: Path) -> bool:
    """Return True when a negation pattern may re-include content under a directory.

    Path-specific negations protect their ancestor directories from pruning.
    Slashless wildcard negations are conservative because they can match names in
    many directories. Slashless literal negations only protect ignored dirs that
    actually contain that basename; this preserves common ``ignored/`` +
    ``!keep.txt`` semantics without letting unrelated literals such as ``!apps/``
    disable pruning for every ignored directory in large repositories.
    """
    logical_dir = logical_dir.rstrip("/")
    dir_name = logical_dir.rsplit("/", 1)[-1]
    for pattern in spec.patterns:
        if getattr(pattern, "include", None) is not False:
            continue
        negation = _normalized_negation_pattern(str(getattr(pattern, "pattern", "")))
        if not negation:
            continue
        if "/" not in negation:
            if _contains_glob_wildcard(negation):
                return True
            if negation in (logical_dir, dir_name):
                return True
            if (host_dir / negation).exists():
                return True
            continue
        if negation == logical_dir or negation.startswith(f"{logical_dir}/"):
            return True
    return False


def _gitignore_matches_dir(spec: pathspec.PathSpec, logical_path: str, host_path: Path) -> bool:
    """Return True when a logical directory path can be safely pruned."""
    logical_dir = logical_path.rstrip("/")
    if not (spec.match_file(logical_dir) or spec.match_file(_ignored_dir_marker(logical_dir))):
        return False
    return not _negation_may_reinclude_dir(spec, logical_dir, host_path)


async def _collect_local_gitignore_filtered(  # noqa: C901
    file_operator: FileOperator,
    *,
    root: str,
    include_hidden: bool,
    max_depth: int | None,
    files_only: bool,
) -> tuple[list[SearchCandidate], GitignoreFilterResult] | None:
    """Collect local candidates while pruning gitignored directories during os.walk.

    This fast path preserves the portable FileOperator path for remote and virtual
    filesystems, but avoids walking large ignored local directories such as
    node_modules, .venv, and build outputs before filtering.
    """
    local_paths = _get_local_walk_paths(file_operator, root)
    if local_paths is None:
        return None
    gitignore_spec = await _load_top_level_gitignore(file_operator)
    if gitignore_spec is None:
        return None

    default_path, resolved_root = local_paths
    if not await anyio.Path(resolved_root).exists():
        return [], GitignoreFilterResult(kept=[], ignored=[])

    def _walk() -> tuple[list[SearchCandidate], list[str]]:  # noqa: C901
        candidates: list[SearchCandidate] = []
        pruned_ignored: list[str] = []

        if resolved_root.is_file():
            try:
                rel = resolved_root.relative_to(default_path).as_posix()
                if (include_hidden or not is_hidden_logical_path(rel)) and not gitignore_spec.match_file(rel):
                    stat = resolved_root.lstat()
                    candidates.append(SearchCandidate(path=rel, size=stat.st_size, mtime=stat.st_mtime))
                elif gitignore_spec.match_file(rel):
                    pruned_ignored.append(rel)
            except (OSError, ValueError):
                pass
            return candidates, pruned_ignored

        root_depth = len(resolved_root.parts)
        for current, dirnames, filenames in os.walk(resolved_root, followlinks=False):
            current_path = Path(current)
            depth = len(current_path.parts) - root_depth
            if max_depth is not None and depth >= max_depth:
                dirnames[:] = []
            if not include_hidden:
                dirnames[:] = [name for name in dirnames if not name.startswith(".")]
                filenames = [name for name in filenames if not name.startswith(".")]

            kept_dirnames: list[str] = []
            for name in dirnames:
                path = current_path / name
                try:
                    rel = path.relative_to(default_path).as_posix()
                except ValueError:
                    continue
                if _gitignore_matches_dir(gitignore_spec, rel, path):
                    pruned_ignored.append(_ignored_dir_marker(rel))
                    continue
                kept_dirnames.append(name)
            dirnames[:] = kept_dirnames

            if not files_only:
                for name in sorted(dirnames):
                    path = current_path / name
                    try:
                        stat = path.lstat()
                        rel = path.relative_to(default_path).as_posix()
                    except (OSError, ValueError):
                        continue
                    candidates.append(SearchCandidate(path=rel, size=stat.st_size, mtime=stat.st_mtime))

            for name in sorted(filenames):
                path = current_path / name
                try:
                    stat = path.lstat()
                    rel = path.relative_to(default_path).as_posix()
                except (OSError, ValueError):
                    continue
                candidates.append(SearchCandidate(path=rel, size=stat.st_size, mtime=stat.st_mtime))

        return candidates, pruned_ignored

    candidates, pruned_ignored = await anyio.to_thread.run_sync(_walk)  # type: ignore[reportAttributeAccessIssue]
    candidates, filter_result = await filter_candidates_ignored(candidates, file_operator)
    if pruned_ignored:
        filter_result.ignored = [*pruned_ignored, *filter_result.ignored]
    return candidates, filter_result


async def collect_walk_entries_gitignore_filtered(
    file_operator: FileOperator,
    *,
    root: str = ".",
    include_hidden: bool = False,
    max_depth: int | None = None,
) -> tuple[list[SearchCandidate], GitignoreFilterResult] | None:
    """Collect entries with local gitignore directory pruning when available."""
    return await _collect_local_gitignore_filtered(
        file_operator,
        root=root,
        include_hidden=include_hidden,
        max_depth=max_depth,
        files_only=False,
    )


async def collect_walk_files_gitignore_filtered(
    file_operator: FileOperator,
    *,
    root: str = ".",
    include_hidden: bool = False,
    max_depth: int | None = None,
) -> tuple[list[SearchCandidate], GitignoreFilterResult] | None:
    """Collect files with local gitignore directory pruning when available."""
    return await _collect_local_gitignore_filtered(
        file_operator,
        root=root,
        include_hidden=include_hidden,
        max_depth=max_depth,
        files_only=True,
    )


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
    max_depth = walk_max_depth_for_glob(pattern)
    filter_result: GitignoreFilterResult | None = None

    if not include_ignored:
        filtered = await collect_walk_entries_gitignore_filtered(
            file_operator,
            root=root,
            include_hidden=include_hidden,
            max_depth=max_depth,
        )
        if filtered is not None:
            candidates, filter_result = filtered
            candidates = filter_candidates_by_glob(candidates, pattern)
            filter_result.ignored = _ignored_paths_matching_glob(filter_result.ignored, pattern)
            return sort_candidates_by_mtime(candidates), filter_result

    candidates = await collect_walk_entries(
        file_operator,
        root=root,
        include_hidden=include_hidden,
        max_depth=max_depth,
    )
    candidates = filter_candidates_by_glob(candidates, pattern)
    if not include_ignored:
        candidates, filter_result = await filter_candidates_ignored(candidates, file_operator)
    return sort_candidates_by_mtime(candidates), filter_result


__all__ = [
    "SearchCandidate",
    "collect_glob_candidates",
    "collect_walk_entries",
    "collect_walk_entries_gitignore_filtered",
    "collect_walk_files",
    "collect_walk_files_gitignore_filtered",
    "filter_candidates_by_glob",
    "filter_candidates_ignored",
    "is_hidden_logical_path",
    "match_glob",
    "normalize_logical_path",
    "sort_candidates_by_mtime",
    "walk_max_depth_for_glob",
]
