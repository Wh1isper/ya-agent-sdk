<tool-instruction name="submit_to_session">
Use this only when global conversation awareness can help a source session make a better next move, exchange relevant context, or keep useful background information available.

Good uses:

- exchange useful context from another group, session, run, memory output, or async result;
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
- Treat the handoff as advisory reference material; the target session may stay silent when a user-facing response adds little value.
- Keep the nudge bounded and disclose only context useful for that target session.

Prefer sending a nudge when the expected effect is improved coordination, faster progress, better answer quality, safer action, or useful context exchange. Keep human control and source-session ownership intact.
</tool-instruction>
