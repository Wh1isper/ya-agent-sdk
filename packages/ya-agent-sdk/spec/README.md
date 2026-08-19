# YA Agent SDK Spec

This spec set defines SDK-level execution primitives and runtime extension contracts.

## Section Map

| Section | Document | Topic |
| ------- | -------- | ----- |
| 01 | [01-lifecycle-extensions.md](01-lifecycle-extensions.md) | lifecycle extension objects, stream hooks, and compact/handoff callbacks |
| 02 | [02-oauth-codex-provider.md](02-oauth-codex-provider.md) | OAuth-backed Codex provider architecture, token store, provider headers, and SDK integration |
| 03 | [03-codeact-programs.md](03-codeact-programs.md) | inline CodeAct, reusable workspace programs, tool eligibility, replay semantics, and sandboxed tool dispatch |
| 04 | [04-runtime-host-contracts.md](04-runtime-host-contracts.md) | segment harness, stream recovery budgets, skill routing/catalog state, host interaction, and CodeAct boundaries |
| 05 | [05-capability-first-runtime.md](05-capability-first-runtime.md) | capability-first composition, native steering, YAACLI durable sessions, YA Claw active input, and native `AgentSpec`-based portable subagents |
| 06 | [06-capability-plugins/](06-capability-plugins/README.md) | SDK-owned entry-point and explicit-import discovery for Pydantic AI custom capability types |
