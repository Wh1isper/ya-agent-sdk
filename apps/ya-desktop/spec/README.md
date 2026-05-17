# YA Desktop Spec

YA Desktop is a native agent workspace for Claw-based runtimes. It gives users an OS-native command home, conversation-first work management, a kanban board, workspace folders, approval inbox, tray presence, local workspace access, multi-connection runtime selection, and future voice interactions.

The same desktop client can use:

- an app-managed local Claw runtime installed through `uv`
- a self-hosted remote Claw server
- a cloud Claw workspace
- a remote agent runtime with local file and shell RPC tools

## Design Direction

- Desktop is an independent product surface optimized for daily agent work, OS context, notifications, and local runtime control.
- Claw is the runtime: sessions, runs, profiles, workspace providers, memory, event replay, shell execution, bridges, schedules, and durable storage.
- `WorkspaceProvider` remains the core execution boundary for local, Docker, cloud, and remote RPC environments.
- A single Claw API client powers local, remote, and cloud connections.
- Desktop keeps multiple saved connection profiles and manages the local runtime lifecycle.
- Local execution uses controlled file operations plus a sandboxed shell by default.
- Desktop owns the richest HITL interaction surface through native notifications, approval cards, command previews, and Claw approval response APIs.
- Chats are the primary work objects in Desktop; Claw sessions and runs remain the runtime backing.
- Spaces represent workspace folders or cloud workspaces plus runtime connection, trust, execution location, and folder mount sets.
- Board is the kanban view over chats, grouped by status, priority, or workspace.
- Advanced runtime management remains available under Settings for profiles, schedules, bridges, heartbeat, logs, diagnostics, and connection internals.

## Section Map

| Section | Document                                                                         | Topic                                                                              |
| ------- | -------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| 00      | [00-overview.md](00-overview.md)                                                 | product surfaces, high-level architecture, and design principles                   |
| 01      | [01-local-sidecar-packaging.md](01-local-sidecar-packaging.md)                   | app-managed local Claw runtime lifecycle, user data layout, and packaging strategy |
| 02      | [02-connection-model.md](02-connection-model.md)                                 | local, remote, and cloud connection registry plus unified Claw client              |
| 03      | [03-cloud-and-rpc-workspaces.md](03-cloud-and-rpc-workspaces.md)                 | cloud workspace mode and remote agent with local RPC tools                         |
| 04      | [04-desktop-api-requirements.md](04-desktop-api-requirements.md)                 | Claw API additions needed by desktop clients                                       |
| 05      | [05-desktop-app-structure.md](05-desktop-app-structure.md)                       | Tauri app structure, system integrations, and implementation phases                |
| 06      | [06-sandboxed-workspace-provider.md](06-sandboxed-workspace-provider.md)         | local workspace provider with sandboxed shell for Linux and macOS                  |
| 07      | [07-websocket-notifications-and-hitl.md](07-websocket-notifications-and-hitl.md) | SSE notifications, session state transfer, and desktop HITL                        |
| 08      | [08-ui-technology-decision.md](08-ui-technology-decision.md)                     | desktop UI technology decision                                                     |
| 09      | [09-desktop-relay.md](09-desktop-relay.md)                                       | YA Environment Relay client integration for Desktop local capabilities             |
| 10      | [10-runtime-manager-and-updates.md](10-runtime-manager-and-updates.md)           | Runtime Manager, bundled `uv`, latest-first Claw updates, and rollback policy      |
| 11      | [11-signing-and-release.md](11-signing-and-release.md)                           | platform signing, notarization, updater signing, and release defaults              |
| 12      | [12-feature-coverage-plan.md](12-feature-coverage-plan.md)                       | feature coverage across visible UI, Desktop Core, and Claw backend                 |
| CU      | [computer-use/README.md](computer-use/README.md)                                 | first-party Host Computer Use architecture, protocol, safety, and UX               |

## Near-Term Decisions

- Use Tauri 2 + TypeScript UI + Rust Core for YA Desktop.
- Use app-managed `uv` to install and launch local Claw as the default runtime.
- Keep development and pull request desktop artifacts unsigned by default; pushes to `release/desktop/**` create unsigned `desktop-v*` GitHub draft releases for human review.
- Ship signing and Desktop app auto-update together in the release-channel phase.
- Provide a local packaging/install Makefile path so users can clone the repository, install YA Desktop locally, and rely on Claw runtime auto-update for runtime updates.
- Treat Desktop as a Native Agent Workspace with Home, Chats, Board, Spaces, Inbox, and Settings surfaces.
- Manage user work by conversations first; Claw sessions and runs are runtime backing objects.
- Use Board as the kanban view over conversations.
- Use Spaces for workspace folders, cloud workspaces, runtime location, trust, local runtime status, and mount-set presets.
- Use Claw HTTP/SSE APIs as the desktop MVP runtime contract, with WebSocket reserved for future RPC workspace transport.
- Add connection registry and folder registry from the beginning.
- Add a global default workspace directory for new chats.
- Bind each chat/session to a workspace mount set with one default folder and optional extra folders.
- Make remote Claw and cloud Claw first-class connection types.
- Treat remote agent with local RPC tools as a future workspace provider.
- Keep voice in the desktop interaction layer.
