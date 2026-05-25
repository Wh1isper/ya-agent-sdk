<glob-tool>
Fast FileOperator-backed file discovery with ripgrep-style glob semantics.
Results are sorted by modification time (newest first) and limited to 500 by default.

<semantics>
- Traversal uses FileOperator.walk_files, so it works across local, sandboxed, and remote environments.
- Pattern matching uses the native ripgrep core when available, with Python fallback for portability.
- Bare file patterns like `*.py` match recursively at any depth.
- `**/*.py` matches root-level and nested Python files.
- A leading slash anchors the pattern to the FileOperator root, e.g. `/*.py` matches only root-level Python files.
- Hidden dot paths and gitignored paths are excluded by default.
</semantics>

<parameters>
- `pattern`: ripgrep-style glob pattern to match files and directories.
- `root`: logical root to traverse from, default `.`.
- `include_hidden`: include hidden dot paths such as `.git`, `.venv`, and `.env`.
- `include_ignored`: include paths excluded by `.gitignore` and nested ignore files.
- `max_results`: maximum result count; use `-1` for unlimited when the pattern is narrow.
</parameters>

<patterns>
- `*.py` - Python files at any depth
- `src/**/*.ts` - TypeScript files under `src/`
- `/*.json` - JSON files directly under the FileOperator root
- `**/*.ts` - TypeScript files at root or under nested directories
</patterns>

<best-practices>
- Use specific patterns to narrow results before reading file contents.
- Use `root` to limit traversal to a subdirectory when the search scope is known.
- Prefer glob before grep when you need to inspect candidate file names first.
- Set `include_hidden=true` for dotfiles and hidden directories.
- Set `include_ignored=true` for generated, dependency, cache, and build outputs.
- Very large results are saved to a temp file with `output_file_path`; use view to read it.
</best-practices>
</glob-tool>
