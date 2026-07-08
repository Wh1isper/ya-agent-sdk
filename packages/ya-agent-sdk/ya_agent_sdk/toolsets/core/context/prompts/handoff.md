<summarize-guidelines>
<overview>
Summarize when continuity would improve by compacting progress or switching focus.
</overview>

<communication>
Explain the transition naturally. Do not use technical jargon like "context reset", "context window", or "token limit" with the user.
</communication>

<when-to-summarize>
- System reminder indicates approaching context limit
- Conversation has accumulated a lot of back-and-forth that is no longer relevant
- User asks to work on a different topic or task
- Major task phase completed, moving to the next phase
- User explicitly asks to summarize and continue
</when-to-summarize>

<when-not-to-summarize>
- A `<context-restored>` block or completed-handoff reminder is already present; continue from restored context instead
- Current task is a direct continuation with all context still relevant
- Simple follow-up questions or minor adjustments
</when-not-to-summarize>

<before-summarizing>
- Capture meaningful remaining work as tasks when task tracking is active.
- Refresh or remove stale notes before they carry forward.
- Identify key files, decisions, constraints, user preferences, and immediate next steps.
</before-summarizing>

<content-structure>
```
## User Intent
[What the user is trying to accomplish]

## Current State
[What has been done, current progress]

## Key Decisions
- [Decision 1]: [Rationale]
- [Decision 2]: [Rationale]

## Past Interactions
A concise log of key interactions that already occurred, to prevent repetition after summary:
- [I asked user about X; user chose Y]
- [I edited file Z; build succeeded]
- [I proposed approach A; user rejected, prefers B]
- [I explained concept C to the user (do not repeat)]
Focus on interactions that would be wasteful or annoying to repeat.

## Next Step
[Immediate action to take after summary]
```
Keep content concise but complete. Do not duplicate task details unless extra explanation is needed.
</content-structure>

<auto-load-files>
Auto-load only files needed immediately after summary. Avoid large files, files already described in content, and temporary files.
</auto-load-files>
</summarize-guidelines>
