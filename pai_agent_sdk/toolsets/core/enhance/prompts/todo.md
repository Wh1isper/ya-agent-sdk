<todo-guidelines>

<when-to-use>
Use `to_do_write`/`to_do_read` to track multi-step or multi-request tasks, or when user asks for a plan.
</when-to-use>

<workflow>
- Create entries with `{id, content, status, priority}` using uppercase slug prefix (e.g., TASK-1, TASK-2)
- Keep numbering stable within the session
- Update statuses immediately after finishing an item
- Rely on the last `to_do_write` state; call `to_do_read` only when truly needed
- Once every item is completed or cancelled, stop issuing further to-do tool calls
</workflow>

<language>
Use user's language when writing task content.
</language>

</todo-guidelines>
