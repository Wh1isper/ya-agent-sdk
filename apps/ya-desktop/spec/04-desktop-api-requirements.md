# 04. Claw API Requirements for Desktop

## Goal

YA Desktop uses Claw as a runtime endpoint. Local embedded Claw, self-hosted Claw, and cloud Claw should expose a consistent runtime API surface for session execution, run inspection, workspace browsing, capability discovery, and workspace execution.

YA Desktop owns desktop-native concerns such as connection config, global hotkeys, tray, keychain, clipboard, screenshots, and voice. Claw owns runtime concerns such as sessions, runs, streaming, workspaces, profiles, memory, and execution environments.

## Health

```http
GET /healthz
```

The current Desktop P0 client uses `/healthz` for local runtime status after deriving an active connection from the Tauri sidecar status.

Future public health may also expose:

```http
GET /health
```

It should return version, instance ID, uptime, and readiness.

Example response:

```json
{
  "status": "ready",
  "version": "0.4.0",
  "instance_id": "rt_abc",
  "uptime_seconds": 128
}
```

## Claw Info and Capabilities

The current Desktop P0 client reads runtime information from:

```http
GET /api/v1/claw/info
```

A future capability-specific endpoint can expose a normalized feature matrix:

```http
GET /api/v1/capabilities
```

Desktop uses Claw info and capabilities to adapt local, remote, and cloud UI.

Example response:

```json
{
  "server": {
    "name": "ya-claw",
    "version": "0.4.0",
    "instance_id": "rt_abc"
  },
  "features": {
    "sessions": true,
    "streaming": true,
    "workspace_filetree": true,
    "workspace_shell": true,
    "memory": true,
    "bridges": true,
    "notifications": true,
    "notification_replay": true,
    "session_status_reasons": true,
    "hitl": true,
    "hitl_status_reason": true,
    "sandboxed_shell": true,
    "session_workspace_binding": true,
    "run_workspace_override": true,
    "multi_mount_workspaces": true,
    "remote_rpc_environment": false
  },
  "auth": {
    "schemes": ["bearer"]
  },
  "workspace_providers": ["local", "docker", "cloud"],
  "local_shell_runtimes": ["linux_bubblewrap"],
  "workspace_mount_modes": ["rw", "ro"],
  "profiles": ["default", "code", "research"],
  "limits": {
    "max_upload_bytes": 104857600,
    "max_sse_duration_seconds": 3600,
    "max_workspace_mounts_per_session": 8
  }
}
```

## Workspace Registry

YA Desktop should keep its own connection registry. Claw can expose a workspace registry for runtime-owned workspaces, especially remote and cloud runtimes.

YA Desktop also keeps a local folder registry for user-selected folders, trust state, pinned spaces, recent folders, and the global default workspace directory. The Desktop folder registry is local desktop state. Claw receives the selected folders through the session/run `workspace` field at execution time.

```http
GET /api/v1/workspaces
POST /api/v1/workspaces
PATCH /api/v1/workspaces/{workspace_id}
POST /api/v1/workspaces/{workspace_id}:set-active
```

Workspace registry entries should support local, Docker, cloud, and future remote RPC workspaces.

Example local workspace:

```json
{
  "id": "local:ya-mono",
  "kind": "local",
  "name": "ya-mono",
  "root": "/home/wh1isper/code/oss/ya-mono",
  "provider": "local",
  "file_operator": "local_file_operator",
  "shell": {
    "kind": "sandboxed_shell",
    "runtime": "linux_bubblewrap",
    "workspace_mounts": [
      {
        "id": "main",
        "host_path": "/home/wh1isper/code/oss/ya-mono",
        "virtual_path": "/workspace/main",
        "mode": "rw"
      }
    ],
    "cwd": "/workspace/main"
  },
  "trust_level": "trusted",
  "profiles": ["default", "code", "research"]
}
```

Example cloud workspace:

```json
{
  "id": "cloud:org:project:repo",
  "kind": "cloud",
  "name": "repo",
  "uri": "cloud://org/project/repo",
  "provider": "cloud",
  "trust_level": "trusted",
  "profiles": ["coding-prod"]
}
```

## Session Workspace Binding

Desktop creates each chat with a workspace binding. This binding maps user-selected folders into Claw virtual workspace paths.

Type shape:

```ts
type WorkspaceMount = {
  id: string;
  name?: string;
  host_path: string;
  virtual_path: string;
  mode: "rw" | "ro";
  docker_host_path?: string;
  metadata?: Record<string, unknown>;
};

type WorkspaceBinding = {
  mounts: WorkspaceMount[];
  default_mount_id: string;
  cwd: string;
  metadata?: Record<string, unknown>;
};
```

Session create example:

```json
{
  "profile_name": "default",
  "input_parts": [
    {"type": "text", "text": "Review this repository and propose next steps."}
  ],
  "workspace": {
    "mounts": [
      {
        "id": "main",
        "name": "ya-mono",
        "host_path": "/Users/jizhongsheng/code/yet-another-agents/ya-mono",
        "virtual_path": "/workspace/main",
        "mode": "rw"
      },
      {
        "id": "docs",
        "name": "product-docs",
        "host_path": "/Users/jizhongsheng/docs/product",
        "virtual_path": "/workspace/docs",
        "mode": "ro"
      }
    ],
    "default_mount_id": "main",
    "cwd": "/workspace/main"
  }
}
```

Desktop behavior:

- New chats use the global default workspace directory as the first mount.
- Users can add extra folders to the chat before creating the session.
- A chat can show and edit its mount set before the next run.
- Session runs inherit the session workspace binding.
- Advanced run creation can send a workspace override for one run.

Claw behavior:

- Session create stores `workspace` in session metadata.
- Session run create inherits the session workspace by default.
- Run-level `workspace` replaces the session binding for that run.
- Claw validates host paths, virtual paths, cwd, and read/write modes before execution.

## Session and Run Streaming

Desktop should reuse the existing high-level session and low-level run surface:

```http
POST /api/v1/runs:stream
POST /api/v1/sessions:stream
POST /api/v1/sessions/{session_id}/runs:stream
GET /api/v1/sessions
GET /api/v1/sessions/{session_id}
GET /api/v1/sessions/{session_id}/turns
GET /api/v1/runs/{run_id}
GET /api/v1/runs/{run_id}/trace
```

The current Desktop client uses sessions, session detail, session turns, run trace, health, Claw info, profiles, global notifications, streamed session creation, streamed session run creation, and session cancellation. Home submits typed commands to `POST /api/v1/sessions:stream` with `input_parts: [{"type":"text","text":"..."}]`, stores a Desktop source marker in session metadata, sends selected workspace bindings, renders AGUI `TEXT_MESSAGE_CHUNK` events as inline output, and refreshes session lists when the stream settles.

Chats continuation uses `POST /api/v1/sessions/{session_id}/runs:stream` with the same event handling model. Desktop depends on AGUI-aligned replay events for stream rendering and run replay.

## Global Notifications

Desktop needs a connection-level realtime channel for session and run state movement outside the currently streamed run. Claw already exposes a global SSE notification stream:

```http
GET /api/v1/claw/notifications
```

Desktop should use SSE as the primary connection-level realtime channel. Future WebSocket support belongs to remote RPC workspace transport and richer bidirectional control-plane features.

Desktop uses notifications to update session lists, selected chat details, session turns, profile lists, pending interaction badges, and active run cards immediately.

Key notification payloads:

```json
{
  "id": "42",
  "type": "session.updated",
  "created_at": "2026-05-09T15:00:00Z",
  "payload": {
    "session_id": "session_123",
    "status": "running",
    "status_reason": "hitl_pending",
    "status_detail": {
      "run_id": "run_456",
      "sequence_no": 4,
      "active_interaction_count": 1
    },
    "active_run_id": "run_456"
  }
}
```

Desktop tracks `Last-Event-ID` per active connection and refreshes HTTP read models when session, run, HITL, or profile events arrive.

Detailed design lives in [07-websocket-notifications-and-hitl.md](07-websocket-notifications-and-hitl.md).

## Run Control and HITL

Desktop needs cancellation, retry-oriented lifecycle controls, steering, and approval response endpoints.

```http
POST /api/v1/runs/{run_id}:cancel
POST /api/v1/runs/{run_id}:pause
POST /api/v1/runs/{run_id}:resume
POST /api/v1/runs/{run_id}/interactions/{interaction_id}:respond
```

Existing Claw control routes use slash-style actions for run and session control:

```http
POST /api/v1/runs/{run_id}/cancel
POST /api/v1/runs/{run_id}/interrupt
POST /api/v1/runs/{run_id}/steer
POST /api/v1/sessions/{session_id}/cancel
POST /api/v1/sessions/{session_id}/interrupt
POST /api/v1/sessions/{session_id}/steer
```

Desktop renders HITL through session status reasons: `status="running"` plus `status_reason="hitl_pending"`. `status_detail.active_interactions` provides compact prompt metadata for badges and approval cards. The current Inbox surfaces HITL-pending metadata and failed/interrupted work; response actions remain mapped to run interaction endpoints as the approval card UI matures.

HITL response shape should align with the SDK `UserInteraction` model:

```json
{
  "responses": [
    {
      "tool_call_id": "call_abc",
      "approved": true,
      "reason": null,
      "user_input": null
    }
  ]
}
```

## Input Context

A previous design included a dedicated API for uploading captured OS context such as selected text, clipboard, active app metadata, and screenshots, then referencing that bundle from run input. This helps when the context is large, binary, or shared across multiple runs.

For the desktop MVP, YA Desktop can pass selected text, clipboard text, and user prompt directly through existing `input_parts`. Screenshots and larger attachments can use the existing upload or artifact path when needed.

Example MVP input:

```json
{
  "input_parts": [
    {"type": "text", "text": "Explain this selected code:\n\n..."}
  ],
  "workspace": {
    "mounts": [
      {
        "id": "main",
        "host_path": "/Users/jizhongsheng/code/yet-another-agents/ya-mono",
        "virtual_path": "/workspace/main",
        "mode": "rw"
      }
    ],
    "default_mount_id": "main",
    "cwd": "/workspace/main"
  },
  "profile_name": "default"
}
```

A generic context bundle API can be revisited when browser extension, mobile share sheet, voice transcripts, or screenshot-heavy workflows become product priorities.

## Sandboxed Shell Workspace Provider

Claw should support local workspace execution with path-bounded file operations and `SandboxedShell`. The host workspace remains the source of truth. Shell execution receives a bind mount or path allowlist for the selected workspace, and file operations continue through `LocalFileOperator`.

The default local shell should use:

- `linux_bubblewrap` on Linux
- `macos_seatbelt` on macOS
- bind mount or path allowlist for the selected workspace

Detailed design lives in [06-sandboxed-workspace-provider.md](06-sandboxed-workspace-provider.md).

## Remote RPC Groundwork

Remote runtime with local RPC tools remains a later workspace-provider direction.

Future components:

- `RemoteRpcWorkspaceProvider`
- `RpcFileOperator`
- `RpcShell`
- WebSocket edge gateway
- Device registration
- Workspace registration
- Tool request queue
- Heartbeat and reconnect
- Cancellation propagation
- Binary artifact transfer
