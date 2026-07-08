<glob-tool>
<best-practices>
- Use specific patterns to narrow results before reading file contents.
- Use `root` to limit traversal to a subdirectory when the search scope is known.
- Prefer glob before grep when you need to inspect candidate file names first.
- Use anchored patterns when only root-level matches are intended; unanchored patterns can match deeply.
- Include hidden or ignored paths only when the target is likely there.
- Treat unlimited results as deliberate only after narrowing scope.
</best-practices>
</glob-tool>
