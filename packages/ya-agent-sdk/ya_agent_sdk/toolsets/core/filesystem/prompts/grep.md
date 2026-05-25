<grep-tool>
FileOperator-backed content search with ripgrep-backed regex and glob semantics.
Returns matching lines with optional surrounding context.

<semantics>
- Traversal uses FileOperator.walk_files, so search stays inside the active environment boundary.
- Regex validation and matching use the native ripgrep core when available, with Python fallback for portability.
- File selection uses ripgrep-style glob semantics shared with the glob tool.
- Bare include patterns like `*.py` match recursively at any depth.
- `**/*.py` matches root-level and nested Python files.
- A leading slash anchors the include pattern to the FileOperator root, e.g. `/*.py` searches only root-level Python files.
- Hidden dot paths and gitignored paths are excluded by default.
- Directories, binary files, and files above the configured size limit are skipped.
</semantics>

<parameters>
- `pattern`: ripgrep-style regular expression pattern to search for.
- `include`: ripgrep-style glob used to select files, default `**/*`.
- `root`: logical root to traverse from, default `.`.
- `context_lines`: lines before and after each match, default `2`.
- `max_results`: maximum total matches; use `-1` for unlimited when scope is narrow.
- `max_matches_per_file`: maximum matches per file; use `-1` for unlimited.
- `max_files`: maximum files to search after filtering; use `-1` for unlimited.
- `include_hidden`: include hidden dot paths such as `.git`, `.venv`, and `.env`.
- `include_ignored`: include paths excluded by `.gitignore` and nested ignore files.
</parameters>

<best-practices>
- Use a specific `include` pattern or `root` for faster and cleaner results.
- Use glob first when you need to inspect candidate file names.
- Keep `context_lines` low for broad scans and raise it for targeted inspection.
- Set `include_hidden=true` for dotfiles and hidden directories.
- Set `include_ignored=true` for generated, dependency, cache, and build outputs.
- Increase `max_files`, `max_results`, or `max_matches_per_file` deliberately after narrowing scope.
</best-practices>
</grep-tool>
