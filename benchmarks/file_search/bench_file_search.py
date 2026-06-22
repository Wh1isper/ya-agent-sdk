#!/usr/bin/env python3
"""Benchmark FileOperator-first file search backends.

The harness compares two implementations that share the same public search
interface:

- python-native: pure Python glob/regex fallback with FileOperator.walk_files
- ripgrep-core: ya-ripgrep-core accelerated glob/regex with FileOperator.walk_files

Results are emitted as JSONL and can be summarized as Markdown.
"""

from __future__ import annotations

import argparse
import asyncio
import fnmatch
import json
import os
import platform
import re
import resource
import shutil
import statistics
import subprocess
import sys
import time
import tracemalloc
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from ya_agent_sdk.environment.local import LocalEnvironment
from ya_agent_sdk.toolsets.core.filesystem import _ripgrep_core
from ya_agent_sdk.toolsets.core.filesystem._line_search import search_file_streaming
from ya_agent_sdk.toolsets.core.filesystem._search import (
    SearchCandidate,
    collect_walk_entries,
    collect_walk_entries_gitignore_filtered,
    collect_walk_files,
    collect_walk_files_gitignore_filtered,
    filter_candidates_by_glob,
    filter_candidates_ignored,
    sort_candidates_by_mtime,
    walk_max_depth_for_glob,
)
from ya_agent_sdk.toolsets.core.filesystem._utils import is_binary_file

Variant = str
Operation = Literal["glob", "grep"]
DEFAULT_GREP_MAX_FILE_SIZE = 10 * 1024 * 1024
BINARY_PROBE_BYTES = 8192


@dataclass(frozen=True)
class DatasetCase:
    """Synthetic dataset shape."""

    name: str
    text_files: int
    dirs: int
    lines_per_file: int
    line_width: int
    hidden_files: int = 0
    ignored_files: int = 0
    binary_files: int = 0
    large_files: int = 0
    large_file_mb: int = 1


FULL_CASES: tuple[str, ...] = ("small", "medium", "large-files", "many-small", "ignored-heavy", "binary-mixed")

CASES: dict[str, DatasetCase] = {
    "quick": DatasetCase(
        "quick",
        text_files=120,
        dirs=8,
        lines_per_file=80,
        line_width=96,
        hidden_files=8,
        ignored_files=12,
        binary_files=4,
    ),
    "small": DatasetCase(
        "small",
        text_files=500,
        dirs=32,
        lines_per_file=200,
        line_width=200,
        hidden_files=50,
        ignored_files=120,
        binary_files=20,
    ),
    "medium": DatasetCase(
        "medium",
        text_files=1_500,
        dirs=96,
        lines_per_file=400,
        line_width=220,
        hidden_files=120,
        ignored_files=400,
        binary_files=80,
    ),
    "large-files": DatasetCase(
        "large-files", text_files=80, dirs=16, lines_per_file=160, line_width=160, large_files=8, large_file_mb=16
    ),
    "many-small": DatasetCase(
        "many-small",
        text_files=8_000,
        dirs=192,
        lines_per_file=12,
        line_width=96,
        hidden_files=180,
        ignored_files=500,
    ),
    "ignored-heavy": DatasetCase(
        "ignored-heavy",
        text_files=2_000,
        dirs=64,
        lines_per_file=120,
        line_width=180,
        hidden_files=120,
        ignored_files=1_500,
        binary_files=80,
    ),
    "binary-mixed": DatasetCase(
        "binary-mixed", text_files=1_200, dirs=64, lines_per_file=120, line_width=180, binary_files=1_500
    ),
}


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


QUERIES: tuple[Query, ...] = (
    Query(name="glob_broad", operation="glob", pattern="**/*"),
    Query(name="glob_selective", operation="glob", pattern="*.py"),
    Query(name="glob_anchored", operation="glob", pattern="/*.py"),
    Query(name="grep_rare", operation="grep", pattern="UNIQUE_TOKEN_777", include="*.py"),
    Query(name="grep_unicode", operation="grep", pattern="性能优化|中文_TOKEN", include="*.py", max_results=500),
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


def _iter_case_files(case: DatasetCase) -> Iterator[tuple[Path, str | bytes]]:
    """Yield deterministic dataset file paths and contents."""
    extensions = ("py", "txt", "md", "json", "ts")
    for index in range(case.text_files):
        directory = Path(f"src/module_{index % max(1, case.dirs):03d}")
        extension = extensions[index % len(extensions)]
        path = directory / f"file_{index:06d}.{extension}"
        yield path, _text_content(index, case.lines_per_file, case.line_width, ignored=False, hidden=False)

    for index in range(case.ignored_files):
        path = Path("build/cache") / f"ignored_{index:06d}.py"
        yield path, _text_content(index, case.lines_per_file, case.line_width, ignored=True, hidden=False)

    for index in range(case.hidden_files):
        path = Path(".hidden") / f"hidden_{index:06d}.py"
        yield path, _text_content(index, max(4, case.lines_per_file // 4), case.line_width, ignored=False, hidden=True)

    for index in range(case.binary_files):
        chunk = bytes((index + offset) % 256 for offset in range(4096))
        yield Path("assets/bin") / f"blob_{index:06d}.bin", chunk

    for index in range(case.large_files):
        yield Path("large") / f"large_{index:04d}.py", _large_text_content(index, case.large_file_mb)


def _text_content(index: int, lines: int, width: int, *, ignored: bool, hidden: bool) -> str:
    """Build deterministic text content."""
    output: list[str] = []
    for line_number in range(lines):
        marker = "plain"
        if index % 97 == 0 and line_number == 3:
            marker = "UNIQUE_TOKEN_777"
        elif line_number % 31 == 0:
            marker = "TODO"
        elif line_number % 47 == 0:
            marker = "FIXME"
        if ignored and line_number == 1:
            marker = "IGNORED_TOKEN"
        if hidden and line_number == 1:
            marker = "HIDDEN_TOKEN"
        if index % 89 == 0 and line_number == 5:
            marker = "性能优化"
        if index % 137 == 0 and line_number == 7:
            marker = "中文_TOKEN"
        prefix = f"def func_{index}_{line_number}():" if line_number % 29 == 0 else f"line {line_number:04d}"
        body = f" {marker} dataset={index:06d} value={(index * 131 + line_number) % 100000:05d} "
        output.append((prefix + body + "x" * width)[:width])
    return "\n".join(output) + "\n"


def _large_text_content(index: int, size_mb: int) -> str:
    """Build large deterministic text content."""
    line = f"large file {index:04d} TODO UNIQUE_TOKEN_777 " + "y" * 200 + "\n"
    repeats = max(1, (size_mb * 1024 * 1024) // len(line))
    return line * repeats


def generate_dataset(case_name: str, output: Path, *, force: bool = False) -> None:
    """Generate a deterministic benchmark dataset."""
    case = _case(case_name)
    if output.exists():
        if not force:
            raise SystemExit(f"Dataset already exists: {output}. Use --force to regenerate.")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    (output / ".gitignore").write_text("build/\nassets/bin/\n", encoding="utf-8")

    manifest = {"case": asdict(case), "files": 0, "bytes": 0}
    for relative_path, content in _iter_case_files(case):
        path = output / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
            manifest["bytes"] += len(content)
        else:
            path.write_text(content, encoding="utf-8")
            manifest["bytes"] += len(content.encode("utf-8"))
        manifest["files"] += 1

    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _case(name: str) -> DatasetCase:
    try:
        return CASES[name]
    except KeyError as exc:
        available = ", ".join(sorted([*CASES, "full"]))
        raise SystemExit(f"Unknown case: {name}. Available cases: {available}") from exc


def _case_names(name: str) -> tuple[str, ...]:
    """Expand a benchmark case selector."""
    if name == "full":
        return FULL_CASES
    _case(name)
    return (name,)


def _case_dataset_root(base: Path, requested_case: str, case_name: str) -> Path:
    """Return dataset path for a requested case/case member pair."""
    if requested_case == "full":
        return base / case_name
    return base


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


def _backend_env(variant: str) -> dict[str, str]:
    env = os.environ.copy()
    if variant.endswith("python-native"):
        env["YA_RIPGREP_CORE_DISABLE"] = "1"
    elif variant.endswith("ripgrep-core"):
        env.pop("YA_RIPGREP_CORE_DISABLE", None)
    else:
        raise SystemExit(f"Unknown variant: {variant}")
    return env


def run_parent(args: argparse.Namespace) -> None:
    """Run benchmarks in child processes and write JSONL."""
    dataset_base = Path(args.dataset)
    case_names = _case_names(args.case)
    for case_name in case_names:
        dataset = _case_dataset_root(dataset_base, args.case, case_name)
        if not dataset.exists():
            generate_dataset(case_name, dataset, force=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.append:
        output.unlink()

    variants = list(args.variants)
    queries = _selected_queries(args.queries, args.operations)
    rows: list[dict[str, Any]] = []
    for case_name in case_names:
        dataset = _case_dataset_root(dataset_base, args.case, case_name)
        for query in queries:
            for variant in variants:
                for repeat in range(args.repeat):
                    command = [
                        sys.executable,
                        __file__,
                        "worker",
                        "--dataset",
                        str(dataset),
                        "--case",
                        case_name,
                        "--variant",
                        variant,
                        "--query",
                        query.name,
                        "--repeat-index",
                        str(repeat),
                    ]
                    completed = subprocess.run(  # noqa: S603
                        command,
                        check=True,
                        capture_output=True,
                        text=True,
                        env=_backend_env(variant),
                    )
                    row = json.loads(completed.stdout)
                    rows.append(row)
                    with output.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(row, sort_keys=True) + "\n")
                    print(_format_progress(row), flush=True)

    if args.summary:
        summarize_file(output, Path(args.summary))


def _format_progress(row: dict[str, Any]) -> str:
    return (
        f"{row['case']} {row['query']} {row['variant']}: "
        f"{row['duration_ms']:.2f} ms, rss={row['peak_rss_mb']:.1f} MB, matches={row['matches']}"
    )


def run_worker(args: argparse.Namespace) -> None:
    """Run a single benchmark query in the current process."""
    query = next((item for item in QUERIES if item.name == args.query), None)
    if query is None:
        raise SystemExit(f"Unknown query: {args.query}")
    result = asyncio.run(_measure(Path(args.dataset), args.case, args.variant, query))
    row = asdict(result)
    row["repeat_index"] = args.repeat_index
    print(json.dumps(row, sort_keys=True))


async def _measure(dataset: Path, case_name: str, variant: str, query: Query) -> BenchmarkResult:
    """Measure one query."""
    before = resource.getrusage(resource.RUSAGE_SELF)
    tracemalloc.start()
    started = time.perf_counter_ns()
    async with LocalEnvironment(allowed_paths=[dataset], default_path=dataset, tmp_base_dir=dataset) as env:
        file_operator = env.file_operator
        if file_operator is None:
            raise RuntimeError("LocalEnvironment did not provide a file_operator")
        if query.operation == "glob":
            stats = await _run_glob(file_operator, query)
        else:
            stats = await _run_grep(file_operator, query)
    duration_ms = (time.perf_counter_ns() - started) / 1_000_000
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    after = resource.getrusage(resource.RUSAGE_SELF)
    return BenchmarkResult(
        variant=variant,
        backend_available=_ripgrep_core.is_available(),
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


async def _run_glob(file_operator: Any, query: Query) -> dict[str, int]:
    candidates, already_filtered = await _collect_glob_walk_candidates(file_operator, query)
    files_seen = len(candidates)
    candidates = filter_candidates_by_glob(candidates, query.pattern)
    if not query.include_ignored and not already_filtered:
        candidates, _ = await filter_candidates_ignored(candidates, file_operator)
    candidates = sort_candidates_by_mtime(candidates)
    if query.max_results > 0:
        candidates = candidates[: query.max_results]
    result_paths = [candidate.path for candidate in candidates]
    return {
        "files_seen": files_seen,
        "files_matched": len(candidates),
        "files_searched": 0,
        "bytes_read": 0,
        "matches": len(result_paths),
        "result_size_bytes": len(json.dumps(result_paths)),
    }


async def _collect_glob_walk_candidates(file_operator: Any, query: Query) -> tuple[list[SearchCandidate], bool]:
    max_depth = walk_max_depth_for_glob(query.pattern)
    if not query.include_ignored:
        filtered = await collect_walk_entries_gitignore_filtered(
            file_operator,
            root=query.root,
            include_hidden=query.include_hidden,
            max_depth=max_depth,
        )
        if filtered is not None:
            return filtered[0], True
    candidates = await collect_walk_entries(
        file_operator,
        root=query.root,
        include_hidden=query.include_hidden,
        max_depth=max_depth,
    )
    return candidates, False


async def _run_grep(file_operator: Any, query: Query) -> dict[str, int]:
    all_candidates, already_filtered = await _collect_grep_walk_candidates(file_operator, query)
    candidates = filter_candidates_by_glob(all_candidates, query.include)
    if not query.include_ignored and not already_filtered:
        candidates, _ = await filter_candidates_ignored(candidates, file_operator)
    candidates = sort_candidates_by_mtime(candidates)
    if query.max_files > 0:
        candidates = candidates[: query.max_files]

    regex = re.compile(query.pattern, re.UNICODE)
    native_regex = _ripgrep_core.NativeRegex(query.pattern) if _ripgrep_core.is_available() else None
    native_searches_bytes = native_regex is not None and native_regex.supports_search_bytes
    total_matches = 0
    result_size = 2
    bytes_read = 0
    searched = 0
    for candidate in candidates:
        if query.max_results > 0 and total_matches >= query.max_results:
            break
        searchable, probe_bytes = await _check_searchable_candidate(
            file_operator,
            candidate,
            check_binary=not native_searches_bytes,
        )
        bytes_read += probe_bytes
        if not searchable:
            continue
        per_file_match_limit = _effective_per_file_match_limit(
            query.max_matches_per_file,
            remaining_results=query.max_results - total_matches if query.max_results > 0 else -1,
        )
        if candidate.size is not None:
            bytes_read += candidate.size
        try:
            result = await search_file_streaming(
                file_operator,
                candidate.path,
                regex,
                context_lines=query.context_lines,
                max_matches_per_file=per_file_match_limit,
                native_regex=native_regex,
            )
        except UnicodeError:
            continue
        searched += 1
        total_matches += len(result.matches)
        result_size += len(json.dumps(result.matches, default=dict))
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


async def _collect_grep_walk_candidates(file_operator: Any, query: Query) -> tuple[list[SearchCandidate], bool]:
    max_depth = walk_max_depth_for_glob(query.include)
    if query.include_ignored:
        return (
            await collect_walk_files(
                file_operator,
                root=query.root,
                include_hidden=query.include_hidden,
                max_depth=max_depth,
            ),
            False,
        )
    filtered = await collect_walk_files_gitignore_filtered(
        file_operator,
        root=query.root,
        include_hidden=query.include_hidden,
        max_depth=max_depth,
    )
    if filtered is not None:
        return filtered[0], True
    return (
        await collect_walk_files(
            file_operator,
            root=query.root,
            include_hidden=query.include_hidden,
            max_depth=max_depth,
        ),
        False,
    )


async def _check_searchable_candidate(
    file_operator: Any,
    candidate: SearchCandidate,
    *,
    check_binary: bool,
) -> tuple[bool, int]:
    if DEFAULT_GREP_MAX_FILE_SIZE > 0 and candidate.size is not None and candidate.size > DEFAULT_GREP_MAX_FILE_SIZE:
        return False, 0
    if not check_binary:
        return True, 0
    try:
        if await is_binary_file(file_operator, candidate.path):
            return False, min(candidate.size or BINARY_PROBE_BYTES, BINARY_PROBE_BYTES)
    except Exception:
        return True, 0
    return True, min(candidate.size or BINARY_PROBE_BYTES, BINARY_PROBE_BYTES)


def _effective_per_file_match_limit(max_matches_per_file: int, *, remaining_results: int) -> int:
    """Return the match limit for the next file after applying the global limit."""
    if remaining_results > 0 and max_matches_per_file > 0:
        return min(max_matches_per_file, remaining_results)
    if remaining_results > 0:
        return remaining_results
    return max_matches_per_file


def _python_match_glob(path: str, pattern: str) -> bool:
    """Pure Python matcher used for direct benchmark sanity checks."""
    pattern = pattern.replace("\\", "/") or "**/*"
    path = path.replace("\\", "/")
    if pattern.startswith("/"):
        stripped = pattern.lstrip("/") or "*"
        if "/" not in stripped and "/" in path:
            return False
        return fnmatch.fnmatchcase(path, stripped)
    if pattern in {"**", "**/*"}:
        return True
    if fnmatch.fnmatchcase(path, pattern):
        return True
    if pattern.startswith("**/") and fnmatch.fnmatchcase(path, pattern[3:]):
        return True
    return "/" not in pattern and fnmatch.fnmatchcase(Path(path).name, pattern)


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
        key = (row["case"], row["operation"], row["query"], row["variant"])
        grouped[key].append(row)

    aggregates = _aggregate_summary_rows(grouped)
    query_groups = _group_aggregates_by_query(aggregates)
    cases = sorted({case for case, _, _ in query_groups})
    repeats = _repeat_count(rows)
    row_count = len(rows)
    query_count = len(query_groups)
    variant_names = sorted({variant for variants in aggregates.values() for variant in variants})

    lines = [
        "# File Search Benchmark Summary",
        "",
        _format_overview_line(cases=cases, repeats=repeats, row_count=row_count, query_count=query_count),
        "",
        "## Quick read",
        "",
        *(_format_quick_read_lines(query_groups) or ["No comparable variant pairs were found."]),
        "",
        "## Direct comparisons",
        "",
    ]

    comparison_lines = _format_comparison_tables(query_groups)
    if comparison_lines:
        lines.extend(comparison_lines)
    else:
        lines.append("No direct comparisons are available for the collected variants.")

    lines.extend([
        "",
        "<details>",
        "<summary>Raw per-variant data</summary>",
        "",
        "| case | op | query | variant | backend | p50 | p95 | peak RSS | Python peak | matches | files |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for key in sorted(aggregates):
        case, operation, query = key
        for variant in sorted(aggregates[key]):
            item = aggregates[key][variant]
            lines.append(
                f"| {case} | {operation} | `{query}` | `{variant}` | {_backend_label(item)} | "
                f"{item['p50']:.2f} ms | {item['p95']:.2f} ms | {item['rss']:.1f} MB | "
                f"{item['py']:.1f} MB | {int(item['matches'])} | {int(item['files'])} |"
            )
    lines.extend(["", "</details>", "", _format_footer(variant_names)])
    return "\n".join(lines) + "\n"


def _aggregate_summary_rows(
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]],
) -> dict[tuple[str, str, str], dict[str, dict[str, float]]]:
    aggregates: dict[tuple[str, str, str], dict[str, dict[str, float]]] = defaultdict(dict)
    for key in sorted(grouped):
        case, operation, query, variant = key
        items = grouped[key]
        durations = sorted(float(item["duration_ms"]) for item in items)
        p50 = statistics.median(durations)
        p95 = durations[min(len(durations) - 1, int(len(durations) * 0.95))]
        peak_rss = max(float(item["peak_rss_mb"]) for item in items)
        python_peak = max(float(item["tracemalloc_peak_mb"]) for item in items)
        matches = int(statistics.median(int(item["matches"]) for item in items))
        files_searched = int(statistics.median(int(item["files_searched"]) for item in items))
        backend_available = any(bool(item["backend_available"]) for item in items)
        aggregates[(case, operation, query)][variant] = {
            "p50": p50,
            "p95": p95,
            "rss": peak_rss,
            "py": python_peak,
            "matches": float(matches),
            "files": float(files_searched),
            "backend": 1.0 if backend_available else 0.0,
        }
    return aggregates


def _group_aggregates_by_query(
    aggregates: dict[tuple[str, str, str], dict[str, dict[str, float]]],
) -> dict[tuple[str, str, str], dict[str, dict[str, float]]]:
    return dict(sorted(aggregates.items()))


def _repeat_count(rows: Sequence[dict[str, Any]]) -> int:
    repeat_indexes = {int(row.get("repeat_index", 0)) for row in rows}
    return len(repeat_indexes) if repeat_indexes else 0


def _format_overview_line(*, cases: Sequence[str], repeats: int, row_count: int, query_count: int) -> str:
    case_label = ", ".join(f"`{case}`" for case in cases) if cases else "unknown"
    return f"**Cases:** {case_label} · **Queries:** {query_count} · **Repeats:** {repeats} · **Rows:** {row_count}"


def _format_quick_read_lines(
    query_groups: dict[tuple[str, str, str], dict[str, dict[str, float]]],
) -> list[str]:
    rows: list[str] = []
    preferred_pairs = [
        ("head-ripgrep-core", "base-ripgrep-core", "New Rust vs old Rust"),
        ("head-python-native", "base-python-native", "New Python vs old Python"),
        ("ripgrep-core", "python-native", "Rust core vs Python"),
        ("head-ripgrep-core", "head-python-native", "Head Rust vs head Python"),
    ]
    for left, right, label in preferred_pairs:
        comparisons = [
            _compare_variants(case, operation, query, variants, left, right, label)
            for (case, operation, query), variants in query_groups.items()
        ]
        comparisons = [comparison for comparison in comparisons if comparison is not None]
        if not comparisons:
            continue
        faster = sum(1 for comparison in comparisons if comparison["left_wins"])
        left_ratios = [float(comparison["left_ratio"]) for comparison in comparisons]
        average_ratio = statistics.mean(left_ratios)
        best = max(comparisons, key=lambda comparison: float(comparison["left_ratio"]))
        lowest = min(comparisons, key=lambda comparison: float(comparison["left_ratio"]))
        rows.append(
            f"| {label} | {faster}/{len(comparisons)} | {average_ratio:.2f}x | "
            f"`{best['query']}` {best['left_ratio']:.2f}x | `{lowest['query']}` {lowest['left_ratio']:.2f}x |"
        )
    if not rows:
        return []
    return [
        "| comparison | wins | avg speed | best query | lowest query |",
        "| --- | ---: | ---: | --- | --- |",
        *rows,
    ]


def _format_comparison_tables(
    query_groups: dict[tuple[str, str, str], dict[str, dict[str, float]]],
) -> list[str]:
    sections = [
        (
            "PR change",
            [
                ("head-ripgrep-core", "base-ripgrep-core", "Rust"),
                ("head-python-native", "base-python-native", "Python"),
            ],
        ),
        (
            "Runtime backend",
            [
                ("ripgrep-core", "python-native", "Rust core vs Python"),
                ("head-ripgrep-core", "head-python-native", "Head Rust vs Python"),
                ("base-ripgrep-core", "base-python-native", "Base Rust vs Python"),
            ],
        ),
        ("Legacy head/base", [("ripgrep-core", "base", "Rust vs base"), ("python-native", "base", "Python vs base")]),
    ]
    lines: list[str] = []
    for title, pairs in sections:
        section_rows: list[dict[str, Any]] = []
        for left, right, label in pairs:
            for (case, operation, query), variants in query_groups.items():
                comparison = _compare_variants(case, operation, query, variants, left, right, label)
                if comparison is not None:
                    section_rows.append(comparison)
        if not section_rows:
            continue
        if lines:
            lines.append("")
        lines.extend([
            f"### {title}",
            "",
            "| query | comparison | winner | speedup | winner p50 | other p50 | saved | RSS winner/other |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in sorted(section_rows, key=lambda item: (str(item["query"]), str(item["label"]))):
            lines.append(
                f"| `{row['query']}` | {row['label']} | `{row['winner']}` | {row['ratio']:.2f}x | "
                f"{row['winner_p50']:.2f} ms | {row['other_p50']:.2f} ms | {row['saved_ms']:.2f} ms | "
                f"{row['winner_rss']:.1f}/{row['other_rss']:.1f} MB |"
            )
    return lines


def _compare_variants(
    case: str,
    operation: str,
    query: str,
    variants: dict[str, dict[str, float]],
    left_name: str,
    right_name: str,
    label: str,
) -> dict[str, Any] | None:
    if left_name not in variants or right_name not in variants:
        return None
    left = variants[left_name]
    right = variants[right_name]
    left_p50 = left["p50"]
    right_p50 = right["p50"]
    left_ratio = right_p50 / left_p50 if left_p50 else 0.0
    left_wins = left_p50 <= right_p50
    if left_wins:
        winner_name = left_name
        other_name = right_name
        winner = left
        other = right
        winner_p50 = left_p50
        other_p50 = right_p50
        ratio = left_ratio
    else:
        winner_name = right_name
        other_name = left_name
        winner = right
        other = left
        winner_p50 = right_p50
        other_p50 = left_p50
        ratio = left_p50 / right_p50 if right_p50 else 0.0
    return {
        "case": case,
        "operation": operation,
        "query": query,
        "label": label,
        "winner": winner_name,
        "other": other_name,
        "ratio": ratio,
        "left_ratio": left_ratio,
        "left_wins": left_wins,
        "winner_p50": winner_p50,
        "other_p50": other_p50,
        "saved_ms": other_p50 - winner_p50,
        "winner_rss": winner["rss"],
        "other_rss": other["rss"],
    }


def _backend_label(item: dict[str, float]) -> str:
    return "yes" if bool(item["backend"]) else "no"


def _format_footer(variant_names: Sequence[str]) -> str:
    variants = ", ".join(f"`{variant}`" for variant in variant_names) if variant_names else "unknown"
    return f"Variants: {variants}. Times are p50/p95 wall-clock milliseconds; lower is faster."


def _csv(value: str | None) -> list[str] | None:
    if value is None or value == "":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    case_choices = sorted([*CASES, "full"])

    generate = subparsers.add_parser("generate", help="Generate a deterministic benchmark dataset")
    generate.add_argument("--case", choices=case_choices, default="full")
    generate.add_argument("--output", type=Path, default=Path(".bench/file-search-full"))
    generate.add_argument("--force", action="store_true")

    run = subparsers.add_parser("run", help="Run benchmark queries")
    run.add_argument("--case", choices=case_choices, default="full")
    run.add_argument("--dataset", default=".bench/file-search-full")
    run.add_argument("--variants", nargs="+", default=["python-native", "ripgrep-core"])
    run.add_argument("--operations", type=_csv, default=None, help="Comma-separated operation filter: glob,grep")
    run.add_argument("--queries", type=_csv, default=None, help="Comma-separated query names")
    run.add_argument("--repeat", type=int, default=3)
    run.add_argument("--output", default=".bench/results/file-search.jsonl")
    run.add_argument("--summary", default=".bench/results/file-search-summary.md")
    run.add_argument("--append", action="store_true", help="Append to an existing JSONL output file")

    worker = subparsers.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("--dataset", required=True)
    worker.add_argument("--case", required=True)
    worker.add_argument("--variant", required=True)
    worker.add_argument("--query", required=True)
    worker.add_argument("--repeat-index", type=int, default=0)

    summarize = subparsers.add_parser("summarize", help="Summarize JSONL benchmark results")
    summarize.add_argument("input", type=Path)
    summarize.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        for case_name in _case_names(args.case):
            generate_dataset(case_name, _case_dataset_root(args.output, args.case, case_name), force=args.force)
    elif args.command == "run":
        run_parent(args)
    elif args.command == "worker":
        run_worker(args)
    elif args.command == "summarize":
        print(summarize_file(args.input, args.output))


if __name__ == "__main__":
    main()
