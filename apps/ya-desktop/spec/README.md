# YA Desktop Spec

YA Desktop is a desktop entry point for Claw-based agent runtimes. It gives users a Raycast-style quick launcher, a full chat window, tray presence, local workspace access, multi-connection runtime selection, and future voice interactions.

The same desktop client can use:

- a bundled local Claw daemon
- a self-hosted remote Claw server
- a cloud Claw workspace
- a remote agent runtime with local file and shell RPC tools

## Design Direction

- Desktop is the interaction shell: launcher, chat UI, tray, hotkeys, clipboard, screenshots, voice, notifications, and connection management.
- Claw is the runtime: sessions, runs, profiles, workspace providers, memory, event replay, shell execution, bridges, and durable storage.
- `WorkspaceProvider` remains the core execution boundary for local, Docker, cloud, and remote RPC environments.
- A single Claw API client powers local, remote, and cloud connections.
- Desktop keeps multiple saved connection profiles and manages the local sidecar lifecycle.
- Local execution uses controlled file operations plus a sandboxed shell by default.
- Desktop should own the richest HITL interaction surface through Claw notifications and approval response APIs.

## Section Map

| Section | Document                                                                         | Topic                                                                        |
| ------- | -------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| 00      | [00-overview.md](00-overview.md)                                                 | product surfaces, high-level architecture, and design principles             |
| 01      | [01-local-sidecar-packaging.md](01-local-sidecar-packaging.md)                   | local `ya-clawd` sidecar lifecycle, user data layout, and packaging strategy |
| 02      | [02-connection-model.md](02-connection-model.md)                                 | local, remote, and cloud connection registry plus unified Claw client        |
| 03      | [03-cloud-and-rpc-workspaces.md](03-cloud-and-rpc-workspaces.md)                 | cloud workspace mode and remote agent with local RPC tools                   |
| 04      | [04-desktop-api-requirements.md](04-desktop-api-requirements.md)                 | Claw API additions needed by desktop clients                                 |
| 05      | [05-desktop-app-structure.md](05-desktop-app-structure.md)                       | Tauri app structure, system integrations, and implementation phases          |
| 06      | [06-sandboxed-workspace-provider.md](06-sandboxed-workspace-provider.md)         | local workspace provider with sandboxed shell for Linux and macOS            |
| 07      | [07-websocket-notifications-and-hitl.md](07-websocket-notifications-and-hitl.md) | SSE notifications, session state transfer, and desktop HITL                  |

## Near-Term Decisions

- Use Tauri + React for YA Desktop.
- Use local `ya-clawd` sidecar for default runtime.
- Use Claw HTTP/SSE APIs as the desktop MVP runtime contract, with WebSocket reserved for future RPC workspace transport.
- Add connection registry from the beginning.
- Make remote Claw and cloud Claw first-class connection types.
- Treat remote agent with local RPC tools as a future workspace provider.
- Keep voice in the desktop interaction layer.
