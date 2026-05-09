# 04. Claw API Requirements for Desktop

## Goal

YA Desktop uses Claw as a runtime endpoint. Local embedded Claw, self-hosted Claw, and cloud Claw should expose a consistent runtime API surface for session execution, run inspection, workspace browsing, capability discovery, and workspace execution.

YA Desktop owns desktop-native concerns such as connection config, global hotkeys, tray, keychain, clipboard, screenshots, and voice. Claw owns runtime concerns such as sessions, runs, streaming, workspaces, profiles, memory, and execution environments.

## Health

```http
GET /health
```

Should return version, instance ID, uptime, and readiness.

Example response:

```json
{
  "status": "ready",
  "version": "0.4.0",
  "instance_id": "rt_abc",
  "uptime_seconds": 128
}
```

## Capabilities

```http
GET /api/v1/capabilities
```

Used by Desktop to adapt local, remote, and cloud UI.

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
    "notification_websocket": true,
    "hitl": true,
    "sandboxed_shell": true,
    "remote_rpc_environment": false
  },
  "auth": {
    "schemes": ["bearer"]
  },
  "workspace_providers": ["local", "docker", "cloud"],
  "local_shell_runtimes": ["linux_bubblewrap"],
  "workspace_mount_modes": ["bind_mount"],
  "profiles": ["default", "code", "research"],
  "limits": {
    "max_upload_bytes": 104857600,
    "max_sse_duration_seconds": 3600
  }
}
```

## Workspace Registry

YA Desktop should keep its own connection registry. Claw can expose a workspace registry for runtime-owned workspaces, especially remote and cloud runtimes.

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
    "workspace_mount": {
      "host_path": "/home/wh1isper/code/oss/ya-mono",
      "sandbox_path": "/workspace",
      "writable": true
    }
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

## Session and Run Streaming

Desktop should reuse the existing high-level session and low-level run surface:

```http
POST /api/v1/runs:stream
POST /api/v1/sessions:stream
POST /api/v1/sessions/{session_id}/runs:stream
GET /api/v1/sessions
GET /api/v1/sessions/{session_id}
GET /api/v1/runs/{run_id}
GET /api/v1/runs/{run_id}/trace
```

Desktop depends on AGUI-aligned replay events for stream rendering and run replay.

## Global Notifications

Desktop needs a connection-level realtime channel for session and run state movement outside the currently streamed run. Claw already exposes a global SSE notification stream:

```http
GET /api/v1/claw/notifications
```

Desktop should prefer a WebSocket surface when the server advertises `notification_websocket=true`:

```http
GET /api/v1/claw/ws
```

The WebSocket should carry the same notification payloads as SSE, plus subscription commands, heartbeat, and HITL responses. Desktop uses notifications to update session lists, tray state, pending interaction badges, and active run cards immediately.

Key notification payloads:

```json
{
  "id": "42",
  "type": "run.updated",
  "created_at": "2026-05-09T15:00:00Z",
  "payload": {
    "session_id": "session_123",
    "run_id": "run_456",
    "status": "running",
    "sequence_no": 4
  }
}
```

Desktop should track `Last-Event-ID` or `last_notification_id` per connection and refresh HTTP read models when replay gaps occur.

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

Desktop should support `waiting_for_user` as a visible run/session state when Claw exposes it, and an `active_interactions` overlay when the server keeps status as `running`.

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
  "workspace_id": "local:ya-mono",
  "profile": "default"
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
