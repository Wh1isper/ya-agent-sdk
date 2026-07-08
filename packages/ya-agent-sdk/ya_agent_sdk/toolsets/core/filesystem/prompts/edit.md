<edit-tool>
<best-practices>
- Read the target snippet immediately before editing when context may be stale.
- old_string must match file content EXACTLY (including whitespace/indentation)
- Preserve exact indentation from view output (ignore line number prefixes)
- Include 3-5 lines of context to ensure unique matches
- Use multi_edit instead of multiple edit calls when changing the same file, especially when changes could otherwise be issued concurrently
</best-practices>
</edit-tool>
