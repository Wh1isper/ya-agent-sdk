<glob-tool>
<best-practices>
- Use specific patterns to narrow results before reading file contents.
- Use `root` to limit traversal to a subdirectory when the search scope is known.
- Prefer glob before grep when you need to inspect candidate file names first.
- Use anchored patterns when only root-level matches are intended; unanchored patterns can match deeply.
- The search root's `.agents/` is exempt from hidden-path filtering so workspace Skill entrypoints remain discoverable; it still follows the FileOperator root `.gitignore`. Use `include_hidden=true` for other hidden paths.
- Include ignored paths only when the target is likely there.
- Treat unlimited results as deliberate only after narrowing scope.
</best-practices>
</glob-tool>
