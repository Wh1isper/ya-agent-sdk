# 05. Desktop App Structure

## App Package

YA Desktop lives under `apps/ya-desktop` as an independent desktop application.

Current frontend structure:

```text
apps/ya-desktop/
  package.json
  index.html
  vite.config.ts
  tsconfig.json
  src/
    app/
      App.tsx
      Shell.tsx
      routes.tsx
      constants.ts
      storage.ts
      types.ts
      ui.tsx
      utils.ts
      chat/
        ConversationPanels.tsx
      pages/
        HomePage.tsx
        ChatsPage.tsx
        BoardPage.tsx
        SpacesPage.tsx
        InboxPage.tsx
        SettingsPage.tsx
    claw/
      client.ts
      connection.ts
      index.ts
      queries.ts
      streamEvents.ts
      streamEvents.test.ts
      types.ts
    lib/
      index.ts
    main.tsx
  src-tauri/
    tauri.conf.json
    Cargo.toml
    src/
      main.rs
      lib.rs
      daemon.rs
      keychain.rs
      hotkey.rs
      tray.rs
      system_context.rs
      notifications.rs
      diagnostics.rs
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
    08-ui-technology-decision.md
```

Target domain structure as the product grows:

```text
src/
  app/
    App.tsx
    Shell.tsx
    routes.tsx
    providers.tsx
    stores.ts
    layout/
      Sidebar.tsx
      TopBar.tsx
      DetailPanel.tsx
    ui/
      IconButton.tsx
      PanelCard.tsx
      StatusPill.tsx
  claw/
    ClawClient.ts
    ClawRealtimeClient.ts
    ConnectionRegistry.ts
    types.ts
  home/
    HomePage.tsx
    CommandBox.tsx
    RecentChats.tsx
    CurrentSpaceCard.tsx
  chats/
    ChatsPage.tsx
    ChatList.tsx
    ChatSurface.tsx
    MessageStream.tsx
    RunTimeline.tsx
    ToolTimeline.tsx
    ShellOutput.tsx
    DiffViewer.tsx
    ArtifactCards.tsx
  board/
    BoardPage.tsx
    BoardColumn.tsx
    BoardCard.tsx
  spaces/
    SpacesPage.tsx
    SpaceCard.tsx
    SpaceSwitcher.tsx
    FolderRegistry.tsx
    MountSetEditor.tsx
    DefaultWorkspacePicker.tsx
    SpaceTrustCard.tsx
    RuntimeLocationCard.tsx
  inbox/
    InboxPage.tsx
    ApprovalCard.tsx
    AlertCard.tsx
    inboxStore.ts
  settings/
    SettingsPage.tsx
    DesktopSettings.tsx
    DefaultWorkspaceSettings.tsx
    HotkeySettings.tsx
    VoiceSettings.tsx
    AdvancedRuntime.tsx
    DiagnosticsPanel.tsx
```

## Frontend Modules

### `app/`

Owns root routing, layout, shared providers, and the current product shell.

Current responsibilities:

- `App.tsx` wraps application providers.
- `Shell.tsx` owns the desktop shell, left navigation, top bar, Details panel, and layout preference persistence.
- `routes.tsx` exports `AppRouteOutlet`, a component switch over Home, Chats, Board, Spaces, Inbox, and Settings.
- `pages/*` contains the current page-level implementations.
- `chat/ConversationPanels.tsx` contains chat-detail and run-detail panels shared by the Chats page.
- `constants.ts`, `storage.ts`, `types.ts`, and `utils.ts` hold app-wide data definitions and helpers.
- `ui.tsx` holds shared UI primitives until the UI surface is large enough to split into `app/ui/*`.

Current `claw/` responsibilities:

- `client.ts` owns the HTTP client and endpoint mapping.
- `connection.ts` owns local connection discovery and active connection state helpers.
- `queries.ts` owns React Query hooks for sessions, runs, approvals, Spaces data, runtime metadata, and notifications.
- `streamEvents.ts` owns SSE/AGUI stream parsing helpers with focused tests in `streamEvents.test.ts`.
- `types.ts` owns frontend-facing Claw API types.

Product responsibilities:

- active connection provider
- active space provider
- active workspace mount-set provider
- theme and settings provider
- global run notification state
- pending interaction state and routing
- top-level navigation between Home, Chats, Board, Spaces, Inbox, and Settings
- calm ChatGPT-like shell with a quiet collapsible left navigation, dominant main work area, and optional Details panel
- persistent layout preferences for collapsed navigation and Details panel state when settings storage is available

### `claw/`

Owns API clients and connection orchestration.

Responsibilities:

- `ClawClient` HTTP client and run-stream SSE client
- `ClawRealtimeClient` global SSE client for notifications, reconnect replay, and chat read-model updates
- connection registry integration
- local daemon lifecycle state
- remote connection health checks
- cloud auth context
- capability caching
- per-connection notification cursor storage

### `home/`

Owns the command-first default surface.

Responsibilities:

- centered command input for starting conversations
- compact profile, current space, runtime, default workspace, and selected mount-set chips inside the composer
- selected text and clipboard preview
- screenshot and active app context preview
- recent chats and active runs below the composer
- pending approval summary
- shortcuts into Chats, Board, Spaces, Inbox, and diagnostics

### `chats/`

Owns conversation-first work management.

Responsibilities:

- chat list grouped by space and status
- selected chat detail surface optimized for reading and continuing one conversation
- per-chat workspace mount-set display and edit flow before creating the next run
- message stream and AGUI replay as the default focus
- run timeline, run controls, and compact traces in progressive details
- tool-call timeline
- shell output viewer
- file diff viewer
- artifact cards
- HITL approval cards inline with chat context

### `board/`

Owns kanban-style organization over chats.

Responsibilities:

- columns for Active, Waiting, Done, Failed, Scheduled, or custom views
- board cards backed by chat/session/run state
- filters by space, profile, status, trigger type, and runtime location
- card metadata for latest output summary, approvals, artifacts, and active run

### `spaces/`

Owns workspace folders and runtime locations.

Responsibilities:

- local workspace folder cards
- remote and cloud workspace cards
- folder registry for recent, trusted, and pinned folders
- global default workspace picker
- mount-set editor for one default folder plus optional extra folders
- active connection and runtime location
- workspace trust level
- default profile and model
- local sidecar status, logs, and diagnostics shortcuts
- file browsing entry points and memory summary

### `inbox/`

Owns human-in-the-loop and alert surfaces.

Responsibilities:

- pending approval queue
- command approval cards
- file diff approval cards
- workspace trust approval cards
- failed background run alerts
- bridge and schedule event alerts
- approve, reject, and user-input response actions
- tray notification click routing
- audit metadata display

### `settings/`

Owns desktop preferences and advanced runtime controls.

Responsibilities:

- desktop appearance and behavior
- global default workspace directory
- hotkey settings
- notification preferences
- voice preferences
- keychain-backed token setup
- autostart and always-on behavior
- advanced runtime: profiles, schedules, bridges, heartbeat, runtime instances, storage, logs, and diagnostics

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
capture_screenshot()
select_workspace_folder()
show_tray_notification()
export_diagnostics()
```

## Runtime Manager

`runtime_manager.rs` should own local Claw runtime installation, selection, and daemon lifecycle.

Responsibilities:

- manage app data runtime directories
- copy and use app-managed `uv`
- install Python and Claw runtime virtualenvs
- write and read `runtimes/claw/active.json`
- generate and load local Claw token
- allocate or discover port
- spawn active `ya-clawd serve`
- parse JSON ready line
- poll `/health`
- collect logs
- restart daemon in always-on mode
- terminate daemon on app shutdown according to user setting

## System Context Capture

`system_context.rs` should collect context for Home command input and global hotkey input.

Initial context fields:

```ts
type DesktopContextDraft = {
  source: 'global_hotkey' | 'home' | 'chat' | 'tray_action'
  activeApp?: {
    name?: string
    windowTitle?: string
  }
  selection?: {
    text?: string
  }
  clipboard?: {
    mime: string
    text?: string
  }
  screenshots?: Array<{
    mime: string
    dataRef: string
  }>
  spaceHint?: string
}
```

## Implementation Phases

### Phase 1: Conversation-First Desktop Shell

01. Add `ya-clawd` CLI entrypoint to `packages/ya-claw`.
02. Add `GET /health` to Claw.
03. Build a Tauri dev app that starts local Claw through `uv run`.
04. Add connection registry and local runtime status.
05. Add Home with command input and recent chat placeholders.
06. Add Chats with a quiet conversation rail, focused chat detail surface, and progressive run details.
07. Add Board with kanban columns over chats.
08. Add Spaces with workspace folder cards, runtime location, trust, and local runtime status.
09. Add Inbox with approval and alert placeholders.
10. Add Settings with preferences and advanced runtime entry.
11. Add tray status and local daemon lifecycle controls.
12. Add sandboxed shell status and setup guidance for local spaces.
13. Connect to global notifications through SSE and update chat/board/inbox read models.

### Phase 2: Multi-Connection Desktop

1. Add remote Claw URL + token setup.
2. Add active connection switcher through Spaces.
3. Add `GET /api/v1/capabilities` to Claw.
4. Gate UI features by capabilities.
5. Add OS keychain storage for tokens.
6. Add remote session and workspace browsing.
7. Add reconnect replay and replay-gap refresh handling for global SSE notifications.

### Phase 3: App-Managed Local Runtime

1. Bundle latest stable `uv` as a Desktop app resource.
2. Add Runtime Manager install/status/select/repair commands.
3. Install Claw runtime into app data on first launch.
4. Add latest-first Claw runtime update checks and compatibility verification.
5. Add local DB migration and profile seeding on daemon startup.
6. Add log export and diagnostics.
7. Add update and rollback metadata.

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
4. Add Desktop Inbox approval cards and native notifications.
5. Record decision audit metadata in Claw and Desktop diagnostics.
