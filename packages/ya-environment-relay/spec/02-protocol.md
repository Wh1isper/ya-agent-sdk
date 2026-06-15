# 02. ya-environment-protocol.v1 Protocol

## Message Model

`ya-environment-protocol.v1` uses JSON-RPC 2.0 semantics with protocol-specific methods, notifications, context, and error codes. The message model is independent of transport.

Request:

```json
{
  "jsonrpc": "2.0",
  "id": "req_123",
  "method": "file.read",
  "params": {
    "mount_id": "main",
    "path": "README.md",
    "encoding": "utf-8"
  },
  "context": {
    "session_id": "session_abc",
    "run_id": "run_def",
    "tool_call_id": "call_ghi",
    "workspace_id": "workspace_xyz"
  }
}
```

Response:

```json
{
  "jsonrpc": "2.0",
  "id": "req_123",
  "result": {
    "content": "# Project\n",
    "encoding": "utf-8"
  }
}
```

Notification:

```json
{
  "jsonrpc": "2.0",
  "method": "$/event",
  "params": {
    "event": "provider.online",
    "payload": {
      "provider_id": "desktop-macbook",
      "instance_id": "envd_20260615_01"
    }
  }
}
```

`id` must be unique within a live logical connection until a terminal response is delivered. Notifications have no `id`.

## Transport Profiles

The protocol defines message semantics only. Transport profiles define framing, connection setup, and authentication carrier.

### stdio

stdio is the MVP transport for launching a provider as a child process:

```text
SDK/Claw process <-> stdio JSON-RPC <-> ya-envd
```

Framing uses LSP-style headers:

```text
Content-Length: 123

{...json...}
```

The parent process owns child process supervision. stdio does not provide daemon durability by itself; if the child process exits, provider-scoped shell/process state is lost unless the provider delegates state to a durable backend.

### Local Daemon Socket

Unix domain socket and Windows named pipe profiles are used for long-lived local daemons:

```text
App <-> socket/named-pipe JSON-RPC <-> ya-envd service
```

This profile supports reconnecting to a daemon that remains alive outside a single agent run.

### WebSocket

WebSocket is one remote transport profile:

```http
GET /api/v1/environment/connect
Authorization: Bearer <provider-token>
Upgrade: websocket
```

Each WebSocket text frame carries one JSON-RPC message. WebSocket is useful for Desktop-to-Claw and cross-machine providers but is not the primary protocol shape.

### Future Transports

Future transports can include TCP with TLS, QUIC, or embedded in-process channels. They must preserve JSON-RPC message semantics and event ordering within one logical connection.

## Handshake

The provider sends `initialize` after a transport is established. Either side may initiate the request depending on the transport profile, but the first completed handshake must establish protocol version, provider identity, capabilities, and accepted policy.

Initialize request:

```json
{
  "jsonrpc": "2.0",
  "id": "init_1",
  "method": "initialize",
  "params": {
    "protocol": "ya-environment-protocol.v1",
    "provider": {
      "provider_id": "desktop-macbook",
      "provider_kind": "ya_envd",
      "provider_version": "0.1.0",
      "instance_id": "envd_20260615_01",
      "environment_id": "space_main",
      "display_name": "Jizhong's MacBook"
    },
    "capabilities": {
      "fileops": {
        "enabled": true,
        "mounts": [
          {
            "mount_id": "main",
            "label": "ya-mono",
            "virtual_path": "/workspace/main",
            "mode": "rw",
            "state": "online",
            "generation": 7,
            "follow_symlinks": false,
            "watch_supported": true
          }
        ]
      },
      "shell": {
        "enabled": true,
        "targets": [
          {
            "target_id": "local-default",
            "platform": "darwin",
            "shell_kind": "zsh",
            "default_cwd": "/workspace/main",
            "allowed_mount_ids": ["main"],
            "supports_exec": true,
            "supports_processes": true,
            "supports_pty": true,
            "supports_signals": true,
            "process_persistence": "provider",
            "session_persistence": "provider"
          }
        ]
      },
      "tools": {
        "enabled": true
      },
      "resources": {
        "enabled": true
      },
      "artifacts": {
        "enabled": true
      },
      "computer": {
        "enabled": false
      }
    }
  }
}
```

Initialize response:

```json
{
  "jsonrpc": "2.0",
  "id": "init_1",
  "result": {
    "accepted_protocol": "ya-environment-protocol.v1",
    "connection_id": "conn_123",
    "accepted_capabilities": ["fileops", "shell", "process", "shell_session", "resources"],
    "server_time": "2026-06-15T12:00:00Z",
    "policy": {
      "grant_id": "grant_abc",
      "accepted_mount_ids": ["main"],
      "accepted_target_ids": ["local-default"]
    }
  }
}
```

## Context

Runtime-to-provider calls should include context when available:

```ts
type EnvironmentRequestContext = {
  session_id?: string;
  run_id?: string;
  tool_call_id?: string;
  workspace_id?: string;
  profile_name?: string;
  source_kind?: "api" | "agency" | "schedule" | "bridge" | "subagent" | "user";
  source_metadata?: Record<string, unknown>;
  user_visible_reason?: string;
};
```

Providers use context for local approvals, audit, artifact upload, and diagnostics. Providers must not require every field for low-level protocol correctness.

## Error Codes

Protocol errors use JSON-RPC `error` with stable protocol codes:

```json
{
  "jsonrpc": "2.0",
  "id": "req_123",
  "error": {
    "code": -32040,
    "message": "The path is outside the selected mount.",
    "data": {
      "name": "permission_denied",
      "recoverable": true,
      "mount_id": "main"
    }
  }
}
```

Common symbolic names:

```text
invalid_request
unknown_method
unsupported_protocol
capability_unavailable
provider_offline
provider_paused
permission_denied
approval_required
policy_blocked
not_found
conflict
timeout
cancelled
resource_lost
process_lost
session_lost
artifact_error
provider_error
transport_error
```

## Cancellation

Cancellation is a notification:

```json
{
  "jsonrpc": "2.0",
  "method": "$/cancelRequest",
  "params": {
    "id": "req_shell_1",
    "reason": "user_cancelled"
  }
}
```

The receiver should attempt cancellation and then emit a terminal response for the original request.

## Streaming

Long-running requests and process/session operations stream through notifications:

```json
{
  "jsonrpc": "2.0",
  "method": "$/stream",
  "params": {
    "id": "req_shell_1",
    "event": "stdout",
    "sequence": 12,
    "data": "Running tests...\n"
  }
}
```

Common stream events:

```text
stdout
stderr
status
progress
log
artifact
pty_output
```

`sequence` is monotonic per request or resource stream when provided. Providers should bound retained stream buffers and report truncation when limits are hit.

## Heartbeat

Heartbeat is transport independent:

```json
{
  "jsonrpc": "2.0",
  "id": "ping_1",
  "method": "$/ping",
  "params": {
    "ts": "2026-06-15T12:00:00Z"
  }
}
```

Response:

```json
{
  "jsonrpc": "2.0",
  "id": "ping_1",
  "result": {
    "ts": "2026-06-15T12:00:00Z"
  }
}
```

## Events

Events are notifications sent through `$/event`.

Common provider events:

```text
provider.starting
provider.online
provider.offline
provider.paused
provider.resumed
provider.degraded
provider.revoked
provider.error
capability.updated
mount.changed
mount.offline
mount.online
process.exited
process.lost
shell_session.available
shell_session.lost
resource.updated
resource.lost
tool.registered
tool.unregistered
artifact.uploaded
```

Events report state. They do not start agent runs. Applications decide whether to activate an agent.

## Method Namespaces

```text
initialize
shutdown

provider.status
provider.capabilities

file.read
file.write
file.append
file.list
file.stat
file.mkdir
file.delete
file.move
file.copy
file.search
file.watch

shell.exec

process.start
process.input
process.close_stdin
process.signal
process.wait
process.kill
process.status
process.list

shell_session.open
shell_session.exec
shell_session.input
shell_session.resize
shell_session.status
shell_session.list
shell_session.snapshot
shell_session.close

tool.list
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

## File Methods

`file.*` paths are relative to a `mount_id` unless a method explicitly accepts an absolute virtual path.

Read:

```json
{
  "method": "file.read",
  "params": {
    "mount_id": "main",
    "path": "README.md",
    "mode": "text",
    "encoding": "utf-8",
    "offset": 0,
    "length": 60000
  }
}
```

List:

```json
{
  "method": "file.list",
  "params": {
    "mount_id": "main",
    "path": ".",
    "include_hidden": false
  }
}
```

Watch:

```json
{
  "method": "file.watch",
  "params": {
    "mount_id": "main",
    "path": ".",
    "recursive": true
  }
}
```

`file.watch` returns a watch resource ID. Changes are delivered through `mount.changed` or resource stream events.

## Shell Methods

`shell.exec` is stateless:

```json
{
  "method": "shell.exec",
  "params": {
    "target_id": "local-default",
    "command": "pwd && pytest",
    "cwd": "/workspace/main",
    "env": {
      "PYTHONUNBUFFERED": "1"
    },
    "timeout_ms": 180000
  }
}
```

It returns:

```json
{
  "exit_code": 0,
  "stdout": "...",
  "stderr": "",
  "duration_ms": 1234,
  "truncated": false
}
```

`shell.exec` must not preserve `cd`, exported variables, aliases, functions, activated virtualenvs, or shell history across calls.

## Process Methods

`process.start` creates a daemon-tracked background process:

```json
{
  "method": "process.start",
  "params": {
    "target_id": "local-default",
    "command": "make web-dev",
    "cwd": "/workspace/main",
    "env": {},
    "output_limit_bytes": 1048576
  }
}
```

Result:

```json
{
  "process_id": "proc_123",
  "persistence": "provider",
  "started_at": "2026-06-15T12:01:00Z"
}
```

Output is delivered through stream notifications. `process.wait` can poll and drain output. If the daemon restarts and the process cannot be reattached, the provider emits `process.lost`.

## Shell Session Methods

`shell_session.open` creates an explicit stateful shell resource:

```json
{
  "method": "shell_session.open",
  "params": {
    "target_id": "local-default",
    "cwd": "/workspace/main",
    "env": {},
    "pty": {
      "cols": 120,
      "rows": 30
    },
    "persistence": "provider"
  }
}
```

Result:

```json
{
  "session_id": "shs_123",
  "resource_id": "res_shs_123",
  "persistence": "provider",
  "started_at": "2026-06-15T12:02:00Z"
}
```

Session persistence levels:

```text
none        provider does not support sessions
connection  session is valid only while the logical connection is alive
provider    session can be reattached while the provider daemon instance is alive
durable     session can survive provider restart through a durable backend
```

Providers must not advertise `durable` unless they can actually preserve or recover the underlying terminal/process state.

## Artifact Coordination

Large artifacts should move out-of-band through runtime-owned storage when they belong to a run.

Reserve:

```json
{
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

Result:

```json
{
  "artifact_id": "art_123",
  "upload_url": "https://runtime.example/upload/art_123",
  "headers": {
    "Authorization": "Bearer upload-token"
  }
}
```

Complete:

```json
{
  "method": "artifact.complete",
  "params": {
    "artifact_id": "art_123",
    "size_bytes": 203456,
    "sha256": "..."
  }
}
```

## Versioning

Breaking changes use a new protocol string such as `ya-environment-protocol.v2`. Additive fields and methods can be negotiated through `initialize.params.capabilities`.

Unknown fields must be ignored unless a method schema marks them invalid. Unknown methods return `unknown_method`. Unknown capabilities are ignored unless explicitly required by the peer.
