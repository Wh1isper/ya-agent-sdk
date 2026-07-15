<grep-tool>
<best-practices>
- Use a specific `include` pattern or `root` for faster and cleaner results.
- Use glob first when you need to inspect candidate file names.
- Keep `context_lines` low for broad scans and raise it for targeted inspection.
- Use anchored include patterns when only root-level files should be searched; unanchored includes can match deeply.
- The search root's `.agents/` is exempt from hidden-path filtering so workspace Skill entrypoints remain discoverable; it still follows the FileOperator root `.gitignore`. Use `include_hidden=true` for other hidden paths.
- Include ignored paths only when the target is likely there.
- Increase result limits deliberately after narrowing scope.
</best-practices>
</grep-tool>
