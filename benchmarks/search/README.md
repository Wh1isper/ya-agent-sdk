# Search Benchmarks

This benchmark suite measures the FileOperator-first filesystem search stack with two backends:

- `ripgrep-core`: default accelerated backend using `ya-ripgrep-core` for glob and regex matching.
- `python-native`: pure Python fallback backend using the same `walk_files` search interface with `YA_RIPGREP_CORE_DISABLE=1`.

The benchmark focuses on end-to-end tool-layer cost: traversal, ignore filtering, glob matching, streaming grep, result construction, CPU time, and memory usage.

## Default full run

```bash
make bench-search
```

This generates `.bench/search-full`, runs all representative cases with the full query matrix, writes raw JSONL to `.bench/results/search.jsonl`, and writes a Markdown summary to `.bench/results/search-summary.md`.

## Quick smoke run

```bash
make bench-search-quick
```

This generates `.bench/search-quick`, runs the quick case with the full query matrix, writes raw JSONL to `.bench/results/search-quick.jsonl`, and writes a Markdown summary to `.bench/results/search-quick-summary.md`.

## Manual run

```bash
uv run python benchmarks/search/bench_search.py generate \
  --case small \
  --output .bench/search-small \
  --force

uv run python benchmarks/search/bench_search.py run \
  --case full \
  --dataset .bench/search-full \
  --variants python-native ripgrep-core \
  --repeat 5 \
  --output .bench/results/search-small.jsonl \
  --summary .bench/results/search-small-summary.md
```

## Cases

| case            | purpose                            |
| --------------- | ---------------------------------- |
| `full`          | all representative cases           |
| `quick`         | fast local smoke benchmark         |
| `small`         | normal project-sized dataset       |
| `medium`        | large repository-sized dataset     |
| `large-files`   | streaming grep memory pressure     |
| `many-small`    | traversal and stat pressure        |
| `ignored-heavy` | `.gitignore` filtering pressure    |
| `binary-mixed`  | mixed text/binary repository shape |

## Metrics

Each JSONL row includes:

- `duration_ms`: wall-clock elapsed time.
- `cpu_user_ms` and `cpu_system_ms`: CPU usage for the worker process.
- `peak_rss_mb`: process peak resident memory.
- `tracemalloc_peak_mb`: Python heap allocation peak.
- `files_seen`, `files_matched`, `files_searched`: traversal and filter counts.
- `bytes_read`: candidate bytes read by grep.
- `matches`: result match count.
- `result_size_bytes`: serialized result payload size estimate.

## CI

The `Search Benchmarks` workflow runs automatically on pull requests that touch filesystem search, environment file traversal, the ripgrep core package, benchmark files, or dependency metadata. The default PR run uses the `quick` suite with the full query matrix and 3 repeats so PR feedback stays fast. It uploads raw JSONL and Markdown summary artifacts, and posts the Markdown summary table as a sticky PR comment.

Manual runs support full cases and query filters:

```bash
gh workflow run search-benchmarks.yml -f case=full -f repeat=3
gh workflow run search-benchmarks.yml -f case=large-files -f repeat=3 -f queries=grep_rare,grep_common
```

## Optimization notes

`ripgrep-core` accelerates three hot paths:

1. Batch glob matching for candidate lists, reducing per-path Python/Rust calls.
2. Whole-file byte search, moving line scanning, UTF-8 lossy decoding, context assembly, and per-file match limits into Rust.
3. Native regex compilation reused across files for each grep query.

The FileOperator boundary remains intact: Python still controls traversal and reads, while Rust handles matching-heavy work. Further benchmark-guided optimization should focus on global result-limit pushdown, FileOperator read path cost, candidate sorting, and ignore filtering.
