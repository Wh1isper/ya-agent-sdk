# 05. Desktop App Structure

## App Package

YA Desktop should live under `apps/ya-desktop` as an independent desktop application.

Suggested structure:

```text
apps/ya-desktop/
  package.json
  index.html
  vite.config.ts
  tsconfig.json
  src/
    app/
      App.tsx
    claw/
      ClawClient.ts
      ClawRealtimeClient.ts
      ConnectionRegistry.ts
      LocalDaemonManager.ts
      RemoteConnectionManager.ts
    hitl/
      ApprovalCenter.tsx
      ApprovalCard.tsx
      approvalStore.ts
    launcher/
      QuickLauncher.tsx
    chat/
      ChatWindow.tsx
      RunTimeline.tsx
      ShellStatusCard.tsx
    settings/
      ConnectionsSettings.tsx
      LocalDaemonSettings.tsx
      ProfilesSettings.tsx
  src-tauri/
    tauri.conf.json
    Cargo.toml
    src/
      main.rs
      daemon.rs
      keychain.rs
      hotkey.rs
      tray.rs
      system_context.rs
  spec/
    README.md
    00-overview.md
    01-local-sidecar-packaging.md
    02-connection-model.md
    03-cloud-and-rpc-workspaces.md
    04-desktop-api-requirements.md
    05-desktop-app-structure.md
    06-sandboxed-workspace-provider.md
    07-websocket-notifications-and-hitl.md
```

## Frontend Modules

### `app/`

Owns root routing, layout, and shared providers.

Responsibilities:

- active connection provider
- active workspace provider
- theme and settings provider
- global run notification state
- pending interaction state and routing
- window mode routing between launcher and full chat

### `claw/`

Owns API clients and connection orchestration.

Responsibilities:

- `ClawClient` HTTP client and run-stream SSE client
- `ClawRealtimeClient` global SSE client for notifications, reconnect replay, and session read-model updates
- connection registry
- local daemon lifecycle state
- remote connection health checks
- cloud auth context
- capability caching
- per-connection notification cursor storage

### `hitl/`

Owns human-in-the-loop approval surfaces.

Responsibilities:

- pending approval queue
- tool approval cards
- command, diff, and workspace context previews
- approve, reject, and user-input response actions
- tray notification click routing
- audit metadata display

### `launcher/`

Owns quick prompt UI.

Responsibilities:

- compact prompt
- selected text and clipboard preview
- active workspace selection
- command suggestions
- create or continue session
- submit normal run `input_parts`
- show compact approval prompts when a pending interaction targets the active connection

### `chat/`

Owns full chat and run inspection.

Responsibilities:

- session list
- run stream display
- AGUI event replay
- tool-call timeline
- shell output viewer
- file diff viewer
- artifact cards
- shell status cards
- HITL approval cards inline with run context

### `settings/`

Owns desktop and runtime settings.

Responsibilities:

- connection settings
- local daemon settings
- hotkey settings
- profile and model settings
- keychain-backed token setup
- log and diagnostics export

## Tauri Commands

Tauri should expose system integration commands to the frontend:

```rust
start_local_claw()
stop_local_claw()
restart_local_claw()
get_local_claw_status()
read_keychain_secret()
write_keychain_secret()
delete_keychain_secret()
register_global_hotkey()
unregister_global_hotkey()
capture_active_window_context()
read_clipboard()
write_clipboard()
show_tray_notification()
```

## Sidecar Manager

`daemon.rs` should own local Claw sidecar lifecycle.

Responsibilities:

- resolve sidecar binary path
- initialize app data directories
- generate and load local Claw token
- allocate or discover port
- spawn `ya-clawd serve`
- parse JSON ready line
- poll `/health`
- collect logs
- restart daemon in always-on mode
- terminate daemon on app shutdown according to user setting

## System Context Capture

`system_context.rs` should collect context for quick launcher input.

Initial context fields:

```ts
type DesktopContextDraft = {
  source: "global_hotkey" | "chat_window" | "tray_action";
  activeApp?: {
    name?: string;
    windowTitle?: string;
  };
  selection?: {
    text?: string;
  };
  clipboard?: {
    mime: string;
    text?: string;
  };
  screenshots?: Array<{
    mime: string;
    dataRef: string;
  }>;
  workspaceHint?: string;
};
```

## Implementation Phases

### Phase 1: Local Desktop MVP

1. Add `ya-clawd` CLI entrypoint to `packages/ya-claw`.
2. Add `GET /health` to Claw.
3. Build a Tauri dev app that starts local Claw through `uv run`.
4. Add `ClawClient` and stream a session run from the desktop UI.
5. Add quick launcher with global hotkey.
6. Add tray status and local daemon lifecycle controls.
7. Add local workspace selection.
8. Add sandboxed shell status and setup guidance for local workspaces.
9. Connect to global notifications through SSE and update the session read model.

### Phase 2: Multi-Connection Desktop

1. Add connection registry.
2. Add remote Claw URL + token setup.
3. Add active connection switcher.
4. Add `GET /api/v1/capabilities` to Claw.
5. Gate UI features by capabilities.
6. Add OS keychain storage for tokens.
7. Add remote session and workspace browsing.
8. Add reconnect replay and replay-gap refresh handling for global SSE notifications.

### Phase 3: Packaged Local Sidecar

1. Add PyInstaller config for `ya-clawd`.
2. Produce per-platform `ya-clawd` artifacts.
3. Bundle sidecar into Tauri app.
4. Add local DB migration and profile seeding on daemon startup.
5. Add log export and diagnostics.
6. Add update and rollback metadata.

### Phase 4: Cloud Workspace Mode

1. Add cloud connection type.
2. Add OAuth or hosted auth flow.
3. Add cloud workspace provider on remote Claw.
4. Add artifact and file browser UX for cloud workspaces.
5. Add team/org/project selection.

### Phase 5: Remote Runtime with Local RPC Tools

1. Add edge registration from Desktop to remote Claw.
2. Add remote RPC workspace provider.
3. Add RPC file and shell tools.
4. Add stdout/stderr streaming over WebSocket.
5. Add cancellation and reconnect semantics.
6. Add local and remote audit logs.

### Phase 6: Desktop HITL

1. Add pending interaction storage and notification handling to Claw.
2. Add `session.updated` notifications with `status_reason="hitl_pending"` and `interaction.requested` notifications.
3. Add HTTP response paths for approvals.
4. Add Desktop approval center, approval cards, and native notifications.
5. Record decision audit metadata in Claw and Desktop diagnostics.
