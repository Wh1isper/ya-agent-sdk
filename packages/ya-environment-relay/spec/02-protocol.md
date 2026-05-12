# 02. ya-environment-relay.v1 Protocol

## Transport

`ya-environment-relay.v1` uses WebSocket as its primary transport. The protocol requires bidirectional request/response, streaming output, cancellation, heartbeat, provider events, and artifact coordination.

Connection endpoint shape for a host runtime:

```http
GET /api/v1/relay/connect
Authorization: Bearer <relay-token>
Upgrade: websocket
```

Products can expose a different path, but the frame format should remain stable.

## Frame Types

```ts
type RelayFrame =
  | HelloFrame
  | EventFrame
  | RequestFrame
  | ResponseFrame
  | StreamFrame
  | CancelFrame
  | PingFrame
  | PongFrame;
```

All frames are JSON text frames unless a method explicitly negotiates binary transfer. Large artifacts should use artifact upload methods instead of raw WebSocket binary in the first version.

## Hello

The relay client sends `hello` immediately after connection:

```json
{
  "type": "hello",
  "protocol": "ya-environment-relay.v1",
  "client_id": "client_macbook_123",
  "client_kind": "ya_desktop",
  "client_version": "0.1.0",
  "capabilities": {
    "fileops": {
      "enabled": true,
      "roots": [
        {
          "root_id": "main",
          "label": "ya-mono",
          "virtual_path": "/workspace/main",
          "mode": "rw"
        }
      ]
    },
    "shell": {
      "enabled": true,
      "runtime": "sandboxed_local_shell",
      "interactive": true
    },
    "tools": {
      "enabled": true,
      "tool_count": 2
    },
    "computer": {
      "enabled": true,
      "platform": "macos",
      "screenshots": true,
      "accessibility_tree": true,
      "semantic_actions": true,
      "coordinate_input": true
    }
  }
}
```

The relay server responds with an event:

```json
{
  "type": "event",
  "event": "relay.accepted",
  "payload": {
    "connection_id": "relay_conn_123",
    "accepted_capabilities": ["fileops", "shell", "tools", "computer"],
    "server_time": "2026-05-11T15:40:00Z"
  }
}
```

## Request

Either side may send requests when authorized. The most common direction is runtime to provider.

```json
{
  "type": "request",
  "id": "req_123",
  "method": "file.read",
  "params": {
    "root_id": "main",
    "path": "README.md",
    "encoding": "utf-8"
  },
  "context": {
    "session_id": "session_abc",
    "run_id": "run_def",
    "tool_call_id": "call_ghi"
  }
}
```

`id` must be unique within a live connection until the request reaches terminal response or cancellation.

## Response

Successful response:

```json
{
  "type": "response",
  "id": "req_123",
  "result": {
    "content": "# Project\n...",
    "encoding": "utf-8"
  }
}
```

Error response:

```json
{
  "type": "response",
  "id": "req_123",
  "error": {
    "code": "permission_denied",
    "message": "The path is outside the selected roots.",
    "recoverable": true,
    "details": {
      "root_id": "main"
    }
  }
}
```

Common error codes:

```text
invalid_request
unknown_method
capability_unavailable
permission_denied
approval_required
policy_blocked
not_found
timeout
cancelled
provider_error
relay_disconnected
artifact_error
```

## Stream

Long-running requests can emit stream frames before the terminal response.

```json
{
  "type": "stream",
  "id": "req_shell_1",
  "event": "stdout",
  "data": "Running tests...\n"
}
```

Common stream event names:

```text
stdout
stderr
progress
artifact
status
log
```

Terminal completion uses `response`:

```json
{
  "type": "response",
  "id": "req_shell_1",
  "result": {
    "exit_code": 0,
    "duration_ms": 1234
  }
}
```

## Cancel

Either side can cancel an active request:

```json
{
  "type": "cancel",
  "id": "req_shell_1",
  "reason": "user_cancelled"
}
```

The receiver should attempt cancellation and then send a terminal response:

```json
{
  "type": "response",
  "id": "req_shell_1",
  "error": {
    "code": "cancelled",
    "message": "Request cancelled by user.",
    "recoverable": true
  }
}
```

## Heartbeat

Heartbeat frames keep the connection alive and measure latency.

```json
{
  "type": "ping",
  "id": "ping_1",
  "ts": "2026-05-11T15:45:00Z"
}
```

```json
{
  "type": "pong",
  "id": "ping_1",
  "ts": "2026-05-11T15:45:00Z"
}
```

## Events

Events are one-way notifications outside request/response.

```json
{
  "type": "event",
  "event": "capability.updated",
  "payload": {
    "capability": "computer",
    "state": "permission_required"
  }
}
```

Common events:

```text
relay.accepted
capability.updated
provider.paused
provider.resumed
artifact.uploaded
tool.registered
tool.unregistered
resource.updated
```

## Method Namespaces

```text
file.read
file.write
file.list
file.stat
file.mkdir
file.delete
file.search
file.watch

shell.start
shell.input
shell.resize
shell.signal
shell.cancel
shell.status

tool.list
tool.register
tool.unregister
tool.call
tool.cancel

resource.list
resource.get
resource.create
resource.dispose
resource.export_state
resource.restore_state

computer.status
computer.see
computer.act
computer.pause
computer.resume
computer.takeover
computer.release

artifact.reserve
artifact.upload
artifact.complete
artifact.abort
```

## Artifact Coordination

Large artifacts should be transferred out-of-band through runtime-owned storage.

Reserve request:

```json
{
  "type": "request",
  "id": "req_artifact_1",
  "method": "artifact.reserve",
  "params": {
    "run_id": "run_123",
    "kind": "screenshot",
    "mime_type": "image/png",
    "metadata": {
      "source": "computer.see"
    }
  }
}
```

Reserve response:

```json
{
  "type": "response",
  "id": "req_artifact_1",
  "result": {
    "artifact_id": "art_123",
    "upload_url": "https://runtime.example/upload/art_123",
    "headers": {
      "Authorization": "Bearer upload-token"
    }
  }
}
```

Complete request:

```json
{
  "type": "request",
  "id": "req_artifact_2",
  "method": "artifact.complete",
  "params": {
    "artifact_id": "art_123",
    "size_bytes": 203456,
    "sha256": "..."
  }
}
```

## Versioning

The protocol string is `ya-environment-relay.v1`. Breaking changes should use `ya-environment-relay.v2`. Additive method fields can be negotiated through the `hello.capabilities` payload.
