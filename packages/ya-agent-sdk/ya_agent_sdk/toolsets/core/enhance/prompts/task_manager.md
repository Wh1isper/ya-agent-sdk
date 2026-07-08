<task-manager-guidelines>
<overview>
Use task management only when a visible checklist improves execution, coordination, or continuity.
</overview>

<when-to-use>
Create tasks for meaningful multi-step work with phases, dependencies, owners, blockers, approvals, handoffs, or parallelizable tracks.
</when-to-use>

<when-not-to-use>
- A single small action or direct answer
- A simple file read, search, edit, command, explanation, or verification
- Creating exactly one task just to mark it in_progress/completed; do the work directly instead
</when-not-to-use>

<task-design>
- Use the user's language for task subjects, descriptions, and active_form. If the user writes Chinese, create Chinese tasks; if the user writes English, create English tasks.
- Split work into tasks only when each task is a meaningful step with independent value, dependency, owner, or delegation target.
- Prefer 2-6 well-scoped tasks for complex work; avoid both one vague umbrella task and excessive micro-tasks.
- Descriptions should state the concrete outcome or acceptance criteria.
- Do not track routine tool-by-tool mechanics as separate tasks.
</task-design>

<workflow>
- Set in_progress when starting a tracked task.
- Set completed immediately after finishing it.
- Keep task status current; do not leave stale in_progress tasks.
</workflow>

<dependencies>
- Add blocked-by or blocks relationships only when they affect execution order, ownership, or parallelization.
- Do not model incidental ordering or obvious linear flow as dependencies.
</dependencies>

<subagent-coordination>
- Do not create tasks merely because a subagent is available.
- Create tasks for meaningful delegated or parallel work items when progress needs to be visible.
- The parent agent owns planning, integration, and final user-facing synthesis.
- If a task is assigned to a subagent, include the task ID, expected outcome, constraints, and reporting expectations in the delegation prompt.
- If no task list is needed, delegate directly without asking the subagent to create or claim a task.
- Update task status when delegated work starts and when results are integrated or the work is blocked.
</subagent-coordination>
</task-manager-guidelines>
