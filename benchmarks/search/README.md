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

The `Search Benchmarks` workflow runs automatically on pull requests that touch filesystem search, environment file traversal, the ripgrep core package, benchmark files, or dependency metadata. The default PR run uses the `full` suite with the full query matrix and 3 repeats. It uploads raw JSONL and Markdown summary artifacts, and posts the Markdown summary table as a sticky PR comment.

Manual runs support larger cases and query filters:

```bash
gh workflow run search-benchmarks.yml -f case=full -f repeat=3
gh workflow run search-benchmarks.yml -f case=large-files -f repeat=3 -f queries=grep_rare,grep_common
```

## Optimization roadmap

Current `ripgrep-core` acceleration covers glob matching and per-line regex matching while the FileOperator boundary, candidate filtering, streaming decode, context buffering, and result construction stay in Python. Benchmark data should guide the next Rust boundary.

Recommended sequence:

1. Move line scanning plus context buffering into `ya-ripgrep-core` with a function that accepts bytes/chunks and returns compact match records.
2. Add a byte-oriented streaming API between Python and Rust so Python controls FileOperator reads while Rust handles decoding, matching, context windows, and match limits.
3. Batch glob matching in Rust for candidate lists to reduce per-path Python/Rust calls.
4. Add benchmark variants for each optimization stage and keep the PR comment table as the regression signal.
