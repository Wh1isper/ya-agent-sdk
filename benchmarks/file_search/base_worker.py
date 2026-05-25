#!/usr/bin/env python3
"""Portable baseline search benchmark worker for target branches.

This script intentionally avoids importing private search helpers introduced by
this PR. It benchmarks the public FileOperator surface available on the target
branch so CI can compare PR head performance against the merge-base/base branch.
"""

from __future__ import annotations

import argparse
import asyncio
import fnmatch
import json
import logging
import platform
import re
import resource
import statistics
import sys
import time
import tracemalloc
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol, runtime_checkable

from ya_agent_sdk.environment.local import LocalEnvironment
from ya_agent_sdk.toolsets.core.filesystem._gitignore import filter_gitignored

Operation = Literal["glob", "grep"]
FULL_CASES: tuple[str, ...] = ("small", "medium", "large-files", "many-small", "ignored-heavy", "binary-mixed")
logger = logging.getLogger(__name__)


@runtime_checkable
class _GlobFileOperator(Protocol):
    """Public FileOperator surface available on older target branches."""

    async def glob(self, pattern: str) -> list[str]: ...


@runtime_checkable
class _WalkFileOperator(Protocol):
    """Public FileOperator surface available on current target branches."""

    def walk_files(
        self,
        root: str = ".",
        *,
        max_depth: int | None = None,
        include_hidden: bool = False,
        follow_symlinks: bool = False,
    ) -> AsyncIterator[dict[str, Any]]: ...


@dataclass(frozen=True)
class Query:
    """Benchmark query."""

    name: str
    operation: Operation
    pattern: str
    include: str = "**/*"
    root: str = "."
    context_lines: int = 0
    max_results: int = -1
    max_matches_per_file: int = -1
    max_files: int = -1
    include_ignored: bool = False
    include_hidden: bool = False


@dataclass(frozen=True)
class SearchCandidate:
    """Logical file candidate."""

    path: str
    size: int | None = None
    mtime: float | None = None


@dataclass(frozen=True)
class RawEntry:
    """Candidate entry collected from public FileOperator APIs."""

    path: str
    is_file: bool | None
    size: int | None
    mtime: float | None


@dataclass
class BenchmarkResult:
    """Single benchmark result row."""

    variant: str
    backend_available: bool
    case: str
    operation: str
    query: str
    pattern: str
    include: str
    root: str
    duration_ms: float
    cpu_user_ms: float
    cpu_system_ms: float
    peak_rss_mb: float
    tracemalloc_peak_mb: float
    files_seen: int
    files_matched: int
    files_searched: int
    bytes_read: int
    matches: int
    result_size_bytes: int
    python: str
    platform: str


QUERIES: tuple[Query, ...] = (
    Query(name="glob_broad", operation="glob", pattern="**/*"),
    Query(name="glob_selective", operation="glob", pattern="*.py"),
    Query(name="glob_anchored", operation="glob", pattern="/*.py"),
    Query(name="grep_rare", operation="grep", pattern="UNIQUE_TOKEN_777", include="*.py"),
    Query(name="grep_common", operation="grep", pattern="TODO|FIXME", include="*.py", max_results=500),
    Query(
        name="grep_context", operation="grep", pattern=r"def func_\d+", include="*.py", context_lines=2, max_results=500
    ),
    Query(
        name="grep_limited",
        operation="grep",
        pattern="TODO|FIXME",
        include="*.py",
        max_results=100,
        max_matches_per_file=10,
    ),
    Query(
        name="grep_ignored",
        operation="grep",
        pattern="IGNORED_TOKEN",
        include="**/*",
        include_ignored=True,
        max_results=500,
    ),
    Query(
        name="grep_hidden",
        operation="grep",
        pattern="HIDDEN_TOKEN",
        include="**/*",
        include_hidden=True,
        max_results=500,
    ),
)


def _csv(value: str | None) -> list[str] | None:
    if value is None or value == "":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _case_names(case: str) -> tuple[str, ...]:
    return FULL_CASES if case == "full" else (case,)


def _case_dataset_root(dataset_base: Path, requested_case: str, case_name: str) -> Path:
    return dataset_base / case_name if requested_case == "full" else dataset_base


def _selected_queries(names: Sequence[str] | None, operations: Sequence[str] | None) -> list[Query]:
    selected = list(QUERIES)
    if names:
        wanted = set(names)
        selected = [query for query in selected if query.name in wanted]
    if operations:
        wanted_ops = set(operations)
        selected = [query for query in selected if query.operation in wanted_ops]
    if not selected:
        raise SystemExit("No queries selected")
    return selected


def _normalize_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized or "."


def _is_hidden_path(path: str) -> bool:
    return any(part.startswith(".") and part not in {".", ".."} for part in _normalize_path(path).split("/"))


def _match_glob(path: str, pattern: str) -> bool:
    normalized_path = _normalize_path(path)
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
    if normalized_pattern.startswith("**/") and fnmatch.fnmatchcase(normalized_path, normalized_pattern[3:]):
        return True
    if "/" not in normalized_pattern:
        return fnmatch.fnmatchcase(PurePosixPath(normalized_path).name, normalized_pattern)
    return False


async def _collect_raw_entries(file_operator: Any, root: str, *, include_hidden: bool) -> list[RawEntry]:
    if isinstance(file_operator, _WalkFileOperator):
        entries: list[RawEntry] = []
        async for entry in file_operator.walk_files(root, include_hidden=include_hidden):
            entries.append(
                RawEntry(
                    path=str(entry["path"]),
                    is_file=bool(entry.get("is_file", False)),
                    size=entry.get("size"),
                    mtime=entry.get("mtime"),
                )
            )
        return entries

    if isinstance(file_operator, _GlobFileOperator):
        broad_pattern = "**/*" if root == "." else f"{root}/**/*"
        return [
            RawEntry(path=path, is_file=None, size=None, mtime=None) for path in await file_operator.glob(broad_pattern)
        ]

    raise TypeError("FileOperator does not provide glob or walk_files")


async def _stat_entry(file_operator: Any, path: str) -> RawEntry:
    try:
        stat = await file_operator.stat(path)
    except Exception:
        logger.debug("Failed to stat benchmark candidate %s", path, exc_info=True)
        return RawEntry(path=path, is_file=None, size=None, mtime=None)
    return RawEntry(path=path, is_file=bool(stat.get("is_file", True)), size=stat.get("size"), mtime=stat.get("mtime"))


async def _collect_candidates(
    file_operator: Any, query: Query, pattern: str, *, files_only: bool
) -> list[SearchCandidate]:
    root = "." if query.root in {"", "."} else query.root.rstrip("/")
    candidates: list[SearchCandidate] = []
    for entry in await _collect_raw_entries(file_operator, root, include_hidden=query.include_hidden):
        path = _normalize_path(entry.path)
        if not query.include_hidden and _is_hidden_path(path):
            continue
        if not _match_glob(path, pattern):
            continue
        if entry.is_file is None or entry.size is None or entry.mtime is None:
            entry = await _stat_entry(file_operator, path)
        if files_only and not bool(entry.is_file):
            continue
        candidates.append(SearchCandidate(path=path, size=entry.size, mtime=entry.mtime))
    return candidates


def _sort_candidates(candidates: list[SearchCandidate]) -> list[SearchCandidate]:
    return sorted(candidates, key=lambda candidate: candidate.mtime or 0.0, reverse=True)


async def _run_glob(file_operator: Any, query: Query) -> dict[str, int]:
    all_candidates = await _collect_candidates(file_operator, query, query.pattern, files_only=False)
    candidates = all_candidates
    if not query.include_ignored:
        ignored = await filter_gitignored([candidate.path for candidate in candidates], file_operator)
        kept = set(ignored.kept)
        candidates = [candidate for candidate in candidates if candidate.path in kept]
    candidates = _sort_candidates(candidates)
    if query.max_results > 0:
        candidates = candidates[: query.max_results]
    result_paths = [candidate.path for candidate in candidates]
    return {
        "files_seen": len(all_candidates),
        "files_matched": len(candidates),
        "files_searched": 0,
        "bytes_read": 0,
        "matches": len(result_paths),
        "result_size_bytes": len(json.dumps(result_paths)),
    }


async def _run_grep(file_operator: Any, query: Query) -> dict[str, int]:
    all_candidates = await _collect_candidates(file_operator, query, query.include, files_only=True)
    candidates = all_candidates
    if not query.include_ignored:
        ignored = await filter_gitignored([candidate.path for candidate in candidates], file_operator)
        kept = set(ignored.kept)
        candidates = [candidate for candidate in candidates if candidate.path in kept]
    candidates = _sort_candidates(candidates)
    if query.max_files > 0:
        candidates = candidates[: query.max_files]

    regex = re.compile(query.pattern, re.UNICODE)
    total_matches = 0
    result_size = 2
    bytes_read = 0
    searched = 0
    for candidate in candidates:
        if query.max_results > 0 and total_matches >= query.max_results:
            break
        remaining = query.max_results - total_matches if query.max_results > 0 else -1
        per_file_match_limit = _effective_per_file_match_limit(query.max_matches_per_file, remaining_results=remaining)
        if candidate.size is not None:
            bytes_read += candidate.size
        try:
            content = await file_operator.read_file(candidate.path)
        except Exception:
            logger.debug("Failed to read benchmark candidate %s", candidate.path, exc_info=True)
            continue
        searched += 1
        matches = _search_text(candidate.path, content, regex, query.context_lines, per_file_match_limit)
        total_matches += len(matches)
        result_size += len(json.dumps(matches, default=dict))
        if query.max_results > 0 and total_matches >= query.max_results:
            total_matches = query.max_results
            break
    return {
        "files_seen": len(all_candidates),
        "files_matched": len(candidates),
        "files_searched": searched,
        "bytes_read": bytes_read,
        "matches": total_matches,
        "result_size_bytes": result_size,
    }


def _effective_per_file_match_limit(max_matches_per_file: int, *, remaining_results: int) -> int:
    if remaining_results > 0 and max_matches_per_file > 0:
        return min(max_matches_per_file, remaining_results)
    if remaining_results > 0:
        return remaining_results
    return max_matches_per_file


def _search_text(
    file_path: str,
    content: str,
    regex: re.Pattern[str],
    context_lines: int,
    max_matches: int,
) -> dict[str, dict[str, Any]]:
    lines = content.splitlines(keepends=True)
    matches: dict[str, dict[str, Any]] = {}
    before_context: deque[tuple[int, str]] = deque(maxlen=max(0, context_lines))
    file_matches = 0
    for index, line in enumerate(lines, start=1):
        if regex.search(line) is not None and (max_matches <= 0 or file_matches < max_matches):
            start = max(1, index - context_lines)
            context_before = [value for _, value in before_context]
            context_after = lines[index : min(len(lines), index + context_lines)]
            context = "".join([*context_before, line, *context_after])
            matches[f"{file_path}:{index}"] = {
                "file_path": file_path,
                "line_number": index,
                "matching_line": line.rstrip("\n"),
                "context": context,
                "context_start_line": start,
            }
            file_matches += 1
        before_context.append((index, line))
    return matches


async def _measure(dataset: Path, case_name: str, query: Query) -> BenchmarkResult:
    before = resource.getrusage(resource.RUSAGE_SELF)
    tracemalloc.start()
    started = time.perf_counter_ns()
    async with LocalEnvironment(allowed_paths=[dataset], default_path=dataset, tmp_base_dir=dataset) as env:
        file_operator = env.file_operator
        if file_operator is None:
            raise RuntimeError("LocalEnvironment did not provide a file_operator")
        stats = (
            await _run_glob(file_operator, query)
            if query.operation == "glob"
            else await _run_grep(file_operator, query)
        )
    duration_ms = (time.perf_counter_ns() - started) / 1_000_000
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    after = resource.getrusage(resource.RUSAGE_SELF)
    return BenchmarkResult(
        variant="base",
        backend_available=False,
        case=case_name,
        operation=query.operation,
        query=query.name,
        pattern=query.pattern,
        include=query.include,
        root=query.root,
        duration_ms=duration_ms,
        cpu_user_ms=(after.ru_utime - before.ru_utime) * 1000,
        cpu_system_ms=(after.ru_stime - before.ru_stime) * 1000,
        peak_rss_mb=_rss_to_mb(after.ru_maxrss),
        tracemalloc_peak_mb=peak / (1024 * 1024),
        files_seen=stats["files_seen"],
        files_matched=stats["files_matched"],
        files_searched=stats["files_searched"],
        bytes_read=stats["bytes_read"],
        matches=stats["matches"],
        result_size_bytes=stats["result_size_bytes"],
        python=platform.python_version(),
        platform=platform.platform(),
    )


def _rss_to_mb(value: int) -> float:
    if sys.platform == "darwin":
        return value / (1024 * 1024)
    return value / 1024


def _format_progress(row: dict[str, Any]) -> str:
    return (
        f"{row['case']} {row['query']} {row['variant']}: "
        f"{row['duration_ms']:.2f} ms, rss={row['peak_rss_mb']:.1f} MB, matches={row['matches']}"
    )


def run_parent(args: argparse.Namespace) -> None:
    queries = _selected_queries(args.queries, args.operations)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.append:
        output.unlink()

    dataset_base = Path(args.dataset)
    rows: list[dict[str, Any]] = []
    for case_name in _case_names(args.case):
        dataset = _case_dataset_root(dataset_base, args.case, case_name)
        for query in queries:
            for repeat in range(args.repeat):
                result = asyncio.run(_measure(dataset, case_name, query))
                row = asdict(result)
                row["repeat_index"] = repeat
                rows.append(row)
                with output.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                print(_format_progress(row), flush=True)

    if args.summary:
        summarize_file(output, Path(args.summary))


def summarize_file(input_path: Path, output_path: Path | None = None) -> str:
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    markdown = summarize_rows(rows)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
    return markdown


def summarize_rows(rows: Sequence[dict[str, Any]]) -> str:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["case"], row["operation"], row["query"], row["variant"])].append(row)
    lines = [
        "# Base File Search Benchmark Summary",
        "",
        "| case | op | query | variant | p50 ms | p95 ms | peak RSS MB | matches | files searched |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in sorted(grouped):
        case, operation, query, variant = key
        items = grouped[key]
        durations = sorted(float(item["duration_ms"]) for item in items)
        p50 = statistics.median(durations)
        p95 = durations[min(len(durations) - 1, int(len(durations) * 0.95))]
        peak_rss = max(float(item["peak_rss_mb"]) for item in items)
        matches = int(statistics.median(int(item["matches"]) for item in items))
        files_searched = int(statistics.median(int(item["files_searched"]) for item in items))
        lines.append(
            f"| {case} | {operation} | {query} | {variant} | {p50:.2f} | {p95:.2f} | "
            f"{peak_rss:.1f} | {matches} | {files_searched} |"
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--operations", type=_csv, default=None)
    parser.add_argument("--queries", type=_csv, default=None)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary")
    parser.add_argument("--append", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    run_parent(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
