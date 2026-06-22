# File Search Benchmarks

This benchmark suite measures the FileOperator-first filesystem file search stack with four comparable PR variants:

- `base-python-native`: the target branch implementation with `YA_RIPGREP_CORE_DISABLE=1`.
- `base-ripgrep-core`: the target branch implementation with `ya-ripgrep-core` enabled when the base branch supports it.
- `head-python-native`: the PR head implementation with `YA_RIPGREP_CORE_DISABLE=1`.
- `head-ripgrep-core`: the PR head implementation with `ya-ripgrep-core` enabled.

Local head-only runs can still use `python-native` and `ripgrep-core`.

The benchmark focuses on end-to-end tool-layer cost: traversal, ignore filtering, glob matching, grep binary/size guards, streaming grep, result construction, CPU time, and memory usage.

## Default full run

```bash
make bench-file-search
```

This generates `.bench/file-search-full`, runs all representative cases with the full query matrix, writes raw JSONL to `.bench/results/file-search.jsonl`, and writes a Markdown summary to `.bench/results/file-search-summary.md`.

`make bench-search` remains an alias for existing local workflows.

## Quick smoke run

```bash
make bench-file-search-quick
```

This generates `.bench/file-search-quick`, runs the quick case with the full query matrix, writes raw JSONL to `.bench/results/file-search-quick.jsonl`, and writes a Markdown summary to `.bench/results/file-search-quick-summary.md`.

`make bench-search-quick` remains an alias for existing local workflows.

## Manual run

```bash
uv run python benchmarks/file_search/bench_file_search.py generate \
  --case small \
  --output .bench/file-search-small \
  --force

uv run python benchmarks/file_search/bench_file_search.py run \
  --case small \
  --dataset .bench/file-search-small \
  --variants python-native ripgrep-core \
  --repeat 5 \
  --output .bench/results/file-search-small.jsonl \
  --summary .bench/results/file-search-small-summary.md
```

## PR/base comparison

CI checks out both the PR head and the PR base SHA. The head checkout generates the synthetic dataset, the base checkout runs `../head/benchmarks/file_search/base_worker.py` against the base implementation, then the head checkout appends its rows into the same JSONL file.

The summary includes a per-query table plus ratios for:

- `head-python-native` versus `base-python-native`
- `head-ripgrep-core` versus `base-ripgrep-core`
- `head-ripgrep-core` versus `head-python-native`
- `base-ripgrep-core` versus `base-python-native`

This keeps future PRs comparable after this branch merges, because each pull request uses its own target branch SHA as `base`.

## Cases

| case            | purpose                                        |
| --------------- | ---------------------------------------------- |
| `full`          | all representative cases                       |
| `quick`         | fast CI verification and local smoke benchmark |
| `small`         | normal project-sized dataset                   |
| `medium`        | large repository-sized dataset                 |
| `large-files`   | streaming grep memory pressure                 |
| `many-small`    | traversal and stat pressure                    |
| `ignored-heavy` | `.gitignore` filtering pressure                |
| `binary-mixed`  | mixed text/binary repository shape             |

## Metrics

Each JSONL row includes:

- `duration_ms`: wall-clock elapsed time.
- `cpu_user_ms` and `cpu_system_ms`: CPU usage for the worker process.
- `peak_rss_mb`: process peak resident memory.
- `tracemalloc_peak_mb`: Python heap allocation peak.
- `files_seen`, `files_matched`, `files_searched`: traversal and filter counts.
- `bytes_read`: candidate bytes read by grep, including binary probe bytes when applicable.
- `matches`: result match count.
- `result_size_bytes`: serialized result payload size estimate.

## CI

The `File Search Benchmarks` workflow runs automatically on pull requests that touch filesystem file search, environment file traversal, the ripgrep core package, benchmark files, or dependency metadata. The default PR run uses the `quick` suite with the full query matrix, including `grep_unicode`, and 3 repeats so PR feedback stays fast. It uploads raw JSONL and Markdown summary artifacts, and posts the Markdown summary table as a sticky PR comment.

Manual runs support full cases, query filters, and an explicit base ref:

```bash
gh workflow run file-search-benchmarks.yml -f case=full -f repeat=3
gh workflow run file-search-benchmarks.yml -f case=large-files -f repeat=3 -f queries=grep_rare,grep_common
gh workflow run file-search-benchmarks.yml -f case=quick -f base_ref=origin/main
```

## Optimization notes

`ripgrep-core` accelerates three hot paths:

1. Batch glob matching for candidate lists, reducing per-path Python/Rust calls.
2. Whole-file byte search, moving line scanning, UTF-8 lossy decoding, context assembly, and per-file match limits into Rust.
3. Native regex compilation reused across files for each grep query.

The FileOperator boundary remains intact: Python controls traversal and reads, while Rust handles matching-heavy work. Further benchmark-guided optimization should focus on atomic FileOperator primitives such as bounded reads, streaming reads, batched metadata access, and traversal filtering.
