<grep-tool>
<best-practices>
- Use a specific `include` pattern or `root` for faster and cleaner results.
- Use glob first when you need to inspect candidate file names.
- Keep `context_lines` low for broad scans and raise it for targeted inspection.
- Use anchored include patterns when only root-level files should be searched; unanchored includes can match deeply.
- Include hidden or ignored paths only when the target is likely there.
- Increase result limits deliberately after narrowing scope.
</best-practices>
</grep-tool>
