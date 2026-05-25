# Search Benchmarks

This benchmark suite measures the FileOperator-first filesystem search stack with two backends:

- `ripgrep-core`: default accelerated backend using `ya-ripgrep-core` for glob and regex matching.
- `python-native`: pure Python fallback backend using the same `walk_files` search interface with `YA_RIPGREP_CORE_DISABLE=1`.

The benchmark focuses on end-to-end tool-layer cost: traversal, ignore filtering, glob matching, streaming grep, result construction, CPU time, and memory usage.

## Quick run

```bash
make bench-search-quick
```

This generates `.bench/search-quick`, runs a small query matrix, writes raw JSONL to `.bench/results/search.jsonl`, and writes a Markdown summary to `.bench/results/search-summary.md`.

## Manual run

```bash
uv run python benchmarks/search/bench_search.py generate \
  --case small \
  --output .bench/search-small \
  --force

uv run python benchmarks/search/bench_search.py run \
  --case small \
  --dataset .bench/search-small \
  --variants python-native ripgrep-core \
  --repeat 5 \
  --output .bench/results/search-small.jsonl \
  --summary .bench/results/search-small-summary.md
```

## Cases

| case            | purpose                            |
| --------------- | ---------------------------------- |
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

The `Search Benchmarks` workflow is manually triggered with `workflow_dispatch`. It uploads raw JSONL and Markdown summary artifacts for review.

```bash
gh workflow run search-benchmarks.yml -f case=quick -f repeat=3
```
