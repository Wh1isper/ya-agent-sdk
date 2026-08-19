# agent-builder

Canonical source for the `agent-builder` skill used by YAACLI.

## Purpose

This directory is the source of truth for the skill bundle that YAACLI ships as `building-agents`.

The sync flow is:

1. edit files in `skills/agent-builder/`
2. run `scripts/sync-skills.sh`
3. bundled files are copied into `packages/yaacli/yaacli/skills/building-agents/`

## Contents

| File                                               | Purpose                             |
| -------------------------------------------------- | ----------------------------------- |
| [`SKILL.md`](SKILL.md)                             | Main skill instructions             |
| [`context.md`](context.md)                         | Agent context and session reference |
| [`streaming.md`](streaming.md)                     | Streaming and lifecycle hooks       |
| [`events.md`](events.md)                           | Event model and sideband events     |
| [`toolset.md`](toolset.md)                         | Toolset architecture                |
| [`user-input.md`](user-input.md)                   | Structured deferred user input      |
| [`tool-search.md`](tool-search.md)                 | Native deferred tool search         |
| [`subagent.md`](subagent.md)                       | Portable subagent architecture      |
| [`skills.md`](skills.md)                           | SDK skill system internals          |
| [`environment.md`](environment.md)                 | Environment abstractions            |
| [`resumable-resources.md`](resumable-resources.md) | Resumable resources                 |
| [`model.md`](model.md)                             | Model configuration                 |
| [`logging.md`](logging.md)                         | Logging configuration               |
| [`media.md`](media.md)                             | Media upload                        |
| [`tool-proxy.md`](tool-proxy.md)                   | Tool proxy                          |

## Examples

Runnable examples stay in the repository root `examples/` directory and are copied into the bundled skill during sync.
