<multi-edit-tool>
<best-practices>
- Prefer multi_edit over multiple single edits for the same file
- When making multiple changes to the same file, including changes planned in parallel, do not issue concurrent edit calls; combine them into one multi_edit call
- Each old_string must be unique (or use replace_all=true)
- Edits are applied sequentially - ensure earlier edits don't affect later ones
- Avoid overlapping snippets; one failed edit prevents the file from being written.
</best-practices>
</multi-edit-tool>
