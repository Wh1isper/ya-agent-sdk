<tool-instruction name="submit_to_session">
Use `submit_to_session` only from the Agency session to send a proactive reminder or nudge to a conversation session agent.

This tool is for intelligence and effect first: use it when global conversation awareness can help a session agent make a better next move. The tool engineering tags identify the nudge type, while the `prompt` should remain natural-language guidance authored by Agency.

Good uses:

- remind the session agent about useful context from another group or session;
- suggest asking a named person to confirm, decide, review, unblock, or own something;
- suggest telling the current group about a decision, status change, deadline, risk, or dependency update;
- help the session agent answer better with hidden cross-session context;
- suggest turning a commitment into a lightweight task or follow-up;
- route a question to the person or group with fresher relevant context;
- reconcile conflicting timelines, owners, technical decisions, release scope, or customer-facing statements;
- deliver Agency or async-subagent investigation results back to the execution context;
- nudge a stale wait, pending question, or blocked loop when new information appears elsewhere.

Prompt style:

- Address the source conversation agent directly.
- Write free-form natural-language guidance, not a rigid template.
- Include the useful context, relevant people or groups, candidate actions, and compact provenance.
- Let the source session agent decide whether to answer, ask a person, remind the group, create or update a task, route the discussion, or record the context quietly.
- Keep the nudge bounded and disclose only context useful for that target session.

Parameters:

- `session_id`: target conversation session ID.
- `prompt`: Agency-authored natural-language guidance for that session agent.
- `metadata`: compact provenance such as fire IDs, source run IDs, async task IDs, people, groups, topic keys, and artifact paths.
- `handoff_kind`: lightweight tag such as `reminder`, `context`, `task`, `risk`, or `async_result`.
- `handoff_tags`: optional tags such as `agency-reminder`, `ask-person`, `tell-group`, `context-completion`, `task-candidate`, `owner-routing`, `decision-conflict`, or `stale-wait`.

Prefer sending a nudge when the expected effect is improved coordination, faster progress, better answer quality, or safer action. Keep human control and source-session ownership intact.
</tool-instruction>
