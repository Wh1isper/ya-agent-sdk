# 12. Feature Coverage Plan

## Goal

Keep YA Desktop product planning aligned across the visible UI, Tauri Desktop Core, and Claw runtime backend. This plan tracks every displayed capability, every backend capability waiting for Desktop UI integration, and every product capability that still needs a Desktop Core or Claw API contract.

## Capability States

| State                | Meaning                                                           | Product action                                                               |
| -------------------- | ----------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| UI mock              | The app shows the capability with prototype data.                 | Replace mock data with Desktop Core or Claw data.                            |
| UI local             | The app shows the capability from browser or Desktop-local state. | Promote to Desktop Core storage when the state becomes durable product data. |
| Desktop Core ready   | Tauri commands or native services exist.                          | Wire UI and React Query hooks.                                               |
| Claw backend ready   | Claw HTTP/SSE API exists.                                         | Add Desktop Claw client methods, hooks, and UI mapping.                      |
| API contract needed  | Product behavior is clear enough to specify an API.               | Update Desktop and Claw specs, then implement backend.                       |
| Product model needed | The product object and lifecycle need design.                     | Define ownership, states, permissions, and UX before implementation.         |

## Coverage Matrix

| Area          | Capability                           | User surface                    | Current UI                | Desktop Core                 | Claw backend                 | Next step                                                                   | Priority |
| ------------- | ------------------------------------ | ------------------------------- | ------------------------- | ---------------------------- | ---------------------------- | --------------------------------------------------------------------------- | -------- |
| Shell         | Top navigation                       | Sidebar                         | UI local                  | N/A                          | N/A                          | Keep nav focused on Home, Chats, Board, Spaces, Inbox, Settings.            | P0       |
| Shell         | Focus layout                         | Sidebar, top bar, right context | UI local                  | N/A                          | N/A                          | Persist collapsed state locally and later move to Desktop settings storage. | P0       |
| Runtime       | Local Claw lifecycle                 | Settings, Home                  | UI wired                  | Desktop Core ready           | N/A                          | Add stronger diagnostics and first-run guidance.                            | P0       |
| Runtime       | Runtime install/update/repair/remove | Settings                        | UI wired                  | Desktop Core ready           | N/A                          | Add stronger diagnostics and first-run guidance.                            | P1       |
| Connection    | Active local connection              | Home, Chats, top bar            | UI wired                  | Desktop Core ready           | API contract documented      | Move bearer token exposure behind keychain-backed token references.         | P0       |
| Connection    | Remote Claw connection               | Settings, Spaces                | UI direction              | API contract needed          | Claw HTTP ready in principle | Implement connection registry and token storage.                            | P2       |
| Connection    | Cloud Claw connection                | Settings, Spaces                | UI direction              | Product model needed         | Product model needed         | Define account, org, project, and auth model.                               | P3       |
| Home          | Command-first start                  | Home                            | UI wired                  | N/A                          | Claw backend ready           | Add profile and workspace selection before session creation.                | P1       |
| Home          | Recent chats                         | Home                            | UI wired                  | N/A                          | Claw backend ready           | Refresh from stream settlements and global notifications.                   | P0       |
| Home          | Active runs                          | Home, right context             | UI mock                   | N/A                          | Claw backend ready           | Derive from session/run status and notifications.                           | P1       |
| Home          | Current space summary                | Home                            | UI mock                   | Product model needed         | Workspace binding ready      | Add local folder registry and default workspace.                            | P1       |
| Chats         | Chat list                            | Chats                           | UI wired                  | N/A                          | Claw backend ready           | Add search, filtering, and stable chat titles.                              | P0       |
| Chats         | Chat detail                          | Chats                           | UI wired                  | N/A                          | Claw backend ready           | Add message replay and selected run drilldown.                              | P0       |
| Chats         | Turns preview                        | Chats                           | UI wired                  | N/A                          | Claw backend ready           | Add pagination and richer input part rendering.                             | P0       |
| Chats         | Run trace preview                    | Chats                           | UI wired                  | N/A                          | Claw backend ready           | Add trace item expansion and run-specific filters.                          | P0       |
| Chats         | Streaming run output                 | Chats, Home                     | UI partial                | N/A                          | Claw backend ready           | Reuse Home stream client for Chats continuation and message replay.         | P1       |
| Chats         | Continue chat                        | Chats                           | UI direction              | N/A                          | Claw backend ready           | Send input parts to session run stream endpoint.                            | P1       |
| Chats         | Rerun failed/interrupted work        | Chats, Inbox                    | UI direction              | N/A                          | Claw backend ready           | Add rerun action with `restore_from_run_id`.                                | P1       |
| Chats         | Cancel active run                    | Chats, Inbox                    | UI direction              | N/A                          | API contract needed          | Confirm cancel endpoint and state transition contract.                      | P1       |
| Board         | Kanban grouped by run/session status | Board                           | UI mock                   | N/A                          | Claw backend ready           | Build Board from chat read model.                                           | P1       |
| Board         | Group by priority/workspace          | Board                           | UI mock                   | Product model needed         | Metadata available           | Define Desktop priority metadata and workspace grouping.                    | P2       |
| Spaces        | Local folder registry                | Spaces                          | UI mock                   | API contract needed          | N/A                          | Store folders, trust, defaults, and mount presets in Desktop Core.          | P1       |
| Spaces        | Workspace trust                      | Spaces, right context           | UI mock                   | Product model needed         | Workspace validation ready   | Define trust levels and execution restrictions.                             | P1       |
| Spaces        | Session workspace binding            | Spaces, Chats                   | UI direction              | Product model needed         | Claw backend ready           | Attach selected mount set to new sessions.                                  | P1       |
| Spaces        | Workspace file tree                  | Chats, Spaces                   | UI direction              | N/A                          | Claw backend ready           | Add file browser against workspace filetree APIs.                           | P2       |
| Inbox         | Pending approvals                    | Inbox, right context            | UI mock                   | Native notifications pending | API contract documented      | Confirm approval list/response API and wire cards.                          | P1       |
| Inbox         | Failed background work               | Inbox, Board                    | UI mock                   | N/A                          | Claw backend ready           | Query failed runs and surface recovery actions.                             | P1       |
| Inbox         | Alerts and decisions                 | Inbox, tray                     | UI mock                   | Product model needed         | Notifications ready          | Normalize notification types into Inbox items.                              | P2       |
| Right context | Live connection/session/run context  | Right panel                     | UI mock                   | N/A                          | Claw backend ready           | Populate from active connection, selected session, and latest run.          | P1       |
| Notifications | Global SSE events                    | App shell, Inbox, tray          | UI direction              | N/A                          | Claw backend ready           | Add connection-level SSE client and cache updates.                          | P1       |
| Notifications | Native OS notifications              | OS shell                        | UI direction              | Tauri plugin path            | Claw notifications ready     | Map high-value run and HITL events to native notifications.                 | P2       |
| Tray          | Quick status/actions                 | OS tray                         | UI direction              | Tauri tray available         | Claw backend ready           | Add active run, start/stop runtime, and open Inbox actions.                 | P2       |
| Settings      | Runtime Manager                      | Settings                        | UI wired                  | Desktop Core ready           | N/A                          | Keep improving install/update diagnostics.                                  | P0       |
| Settings      | Preferences                          | Settings                        | UI direction              | Product model needed         | N/A                          | Define Desktop settings schema and storage migration.                       | P2       |
| Settings      | Keychain tokens                      | Settings                        | UI direction              | API contract needed          | Auth requirements documented | Add secure token storage abstraction.                                       | P2       |
| Settings      | Hotkeys                              | Settings, command palette       | UI direction              | Tauri plugin dependency      | N/A                          | Add route/action registry and global command palette.                       | P2       |
| Settings      | Logs and diagnostics                 | Settings                        | UI wired for runtime logs | Desktop Core partial         | Claw backend ready           | Combine app logs, runtime logs, health, and environment checks.             | P2       |
| Memory        | Session memory state                 | Chats, right context            | UI direction              | N/A                          | Claw backend ready           | Show memory state and manual extract/summarize actions.                     | P2       |
| Memory        | Memory files                         | Chats, Spaces                   | UI direction              | N/A                          | Workspace files ready        | Browse `memory/MEMORY.md`, changelog, and event files.                      | P3       |
| Profiles      | Runtime profiles                     | Settings, new chat              | UI direction              | N/A                          | Claw backend ready           | Fetch profiles and select profile for new chats.                            | P1       |
| Bridges       | Lark bridge sessions                 | Chats, Inbox                    | UI direction              | N/A                          | Claw backend ready           | Filter bridge-triggered sessions and show origin metadata.                  | P3       |
| Schedules     | Scheduled work                       | Board, Inbox, Settings          | UI direction              | N/A                          | Claw backend ready           | Define schedule UI and notification mapping.                                | P3       |
| Relay         | Local RPC tools                      | Spaces, Settings                | UI direction              | Product model needed         | Relay protocol ready         | Design Desktop relay provider UX and trust boundaries.                      | P3       |
| Computer use  | Host computer use                    | Chats, Inbox                    | Spec direction            | Product model needed         | Protocol needed              | Implement after first-party host computer use safety spec matures.          | P3       |
| Voice         | Desktop voice entry                  | Home                            | Product direction         | Product model needed         | N/A                          | Define capture, transcription, and command routing.                         | P3       |
| Release       | Unsigned draft releases              | GitHub releases                 | N/A                       | Workflow ready               | N/A                          | Keep release/desktop branch path working.                                   | P0       |
| Release       | Signing and app auto-update          | Settings, release channel       | UI direction              | API contract needed          | N/A                          | Ship signing and updater metadata together.                                 | P3       |

## P0 Implementation Slice

This slice turns the prototype into a read-only live Desktop client for local Claw. The first pass is implemented.

Deliverables:

1. Add `apps/ya-desktop/src/claw/` with types, HTTP client, active connection derivation, and React Query hooks. Done.
2. Derive the active local connection from `getLocalClawStatus()` when Local Claw is running and exposes `baseUrl`. Done.
3. Read Claw health/info when available. Done.
4. Read sessions from the active connection and map them to Desktop chat rows. Done.
5. Read selected session details, turns, and compact run traces. Done.
6. Replace Home recent work and Chats mock data with live data plus loading, offline, empty, and error states. Done.
7. Keep Board, Spaces, Inbox, and Live Context prototype data until their P1 slices. Done.
8. Move local bearer token exposure behind keychain-backed token references. Next.

## P1 Implementation Slice

P1 makes Desktop useful for day-to-day agent work.

Deliverables:

1. Home command creates a new session and streams the first run. Done.
2. Chats can continue an existing session, show streaming output, rerun recoverable work, and display run state transitions.
3. Board renders from the shared Desktop chat read model.
4. Spaces stores local folders, trust, default workspace, and mount presets.
5. Inbox surfaces failed runs and pending approval items.
6. Global SSE updates React Query caches and right context state.
7. Profiles load from Claw and can be selected for new chats.

## P2 Implementation Slice

P2 adds native Desktop depth.

Deliverables:

1. Connection registry for local and remote Claw targets.
2. Secure token storage through OS keychain.
3. Native notifications for run completion, failures, and approvals.
4. Tray quick status and common actions.
5. Diagnostics center combining Desktop Core, local runtime, and Claw health.
6. Workspace file browser and memory state UI.
7. Hotkey-backed command palette.

## P3 Implementation Slice

P3 expands YA Desktop beyond local runtime operation.

Deliverables:

1. Cloud Claw account, org, project, and workspace model.
2. Remote agent with local RPC tool transport.
3. Bridge-first workflows for Lark-originated sessions.
4. Scheduled work management.
5. Host computer use and voice entry.
6. Signed release channels and Desktop app auto-update.

## Review Cadence

Update this plan after every Desktop PR that changes product behavior, visible surfaces, Desktop Core commands, or Claw API usage. Each new feature should either complete an existing row or add a row with a clear state, next step, and priority.
