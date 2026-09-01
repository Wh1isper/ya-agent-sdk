<note-guidelines>
<when-to-use>
- User states a preference that should be remembered for this session
- Important facts or decisions that you need to recall later
- Context that would be lost after summarize/compact
- Intermediate results worth preserving
</when-to-use>

<best-practices>
- Use descriptive, stable keys
- Keep values concise and delete entries when they are stale
- Runtime context contains a bounded note-key index; use `note_get` without a key when you need to discover omitted notes
- Use `note_get` with a relevant key when its value is needed for the current task
- Notes persist independently across compact; summaries should reference relevant keys instead of copying note values for retention
- Store large data in files and keep only the file path or index in notes
</best-practices>
</note-guidelines>
