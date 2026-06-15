# 05. Implementation Plan

## Direction

The recommended implementation path is:

1. Define `ya-environment-protocol.v1` as a transport-independent JSON-RPC protocol.
2. Build `ya-envd` as the official Rust daemon implementation.
3. Add Python SDK adapters that map the daemon protocol into `Environment`, `FileOperator`, `Shell`, resources, and toolsets.
4. Integrate Claw and Desktop at the application layer for provider lifecycle, grants, binding, audit, and activation decisions.

## Repository Layout

Recommended Rust workspace:

```text
crates/
  ya-env-protocol/
    src/
      lib.rs
      jsonrpc.rs
      capabilities.rs
      methods.rs
      events.rs
      schema.rs
  ya-envd-core/
    src/
      provider.rs
      mounts.rs
      fileops.rs
      shell.rs
      process.rs
      shell_session.rs
      resources.rs
      policy.rs
      audit.rs
  ya-envd/
    src/
      main.rs
      transports/
        stdio.rs
        socket.rs
        websocket.rs
  ya-envd-client/
    src/
      lib.rs
```

Recommended Python package shape:

```text
packages/ya-environment-relay/
  ya_environment_relay/
    protocol.py
    errors.py
    capabilities.py
    client.py
    environment.py
    transports/
      stdio.py
      socket.py
      websocket.py
```

## Phase 1: Spec and Schemas

Deliverables:

- complete spec documents.
- JSON Schema definitions for request params, results, events, errors, capabilities, mounts, targets, resources, and grants.
- Rust `serde` models in `ya-env-protocol`.
- Python Pydantic models in `ya_environment_relay.protocol`.
- conformance fixtures for initialize, fileops, shell exec, process, shell session, events, and errors.
- method namespace constants.

Exit criteria:

- Rust and Python models serialize the same fixtures.
- unknown-field and unknown-method behavior is specified and tested.
- protocol string is `ya-environment-protocol.v1`.

## Phase 2: Rust stdio Daemon MVP

Deliverables:

- `ya-envd` binary.
- stdio JSON-RPC transport using `Content-Length` framing.
- initialize/shutdown.
- provider.status/provider.capabilities.
- local mount registry.
- file read/write/append/list/stat/mkdir/delete/move/copy.
- path normalization and mount boundary enforcement.
- basic audit log to stderr or a local file path selected by launch config.

Exit criteria:

- parent process can launch `ya-envd` and complete request/response roundtrips.
- fileops work against a temp mount on macOS and Linux.
- path traversal and symlink escape tests fail closed.

## Phase 3: Python DaemonEnvironment Fileops

Deliverables:

- stdio RPC client.
- `DaemonEnvironment`.
- `DaemonFileOperator`.
- environment instructions for mounts and provider readiness.
- integration tests against a fake transport and the real `ya-envd` binary when available.

Exit criteria:

- SDK filesystem tool tests can run against `DaemonFileOperator`.
- file reads, writes, list, stat, stream, move/copy, and error mapping are covered.

## Phase 4: Shell Exec and Background Process

Rust deliverables:

- `shell.exec` using provider-local shell policy.
- `process.start`.
- `process.input`.
- `process.close_stdin`.
- `process.wait`.
- `process.kill`.
- `process.signal` where supported.
- bounded stdout/stderr buffers.
- `process.exited` and `process.lost` events.

Python deliverables:

- `DaemonShell.execute`.
- SDK `Shell.start/write_stdin/wait_process/kill_process/send_signal` mapping.
- completed background result injection compatibility.

Exit criteria:

- existing SDK background shell behavior can be reproduced through daemon transport.
- stateless `shell.exec` does not preserve cwd/env across calls.
- lost process behavior is explicit.

## Phase 5: Stateful Shell Sessions

Deliverables:

- Rust PTY backend, preferably using a portable PTY crate.
- `shell_session.open/input/resize/status/list/snapshot/close`.
- provider-scoped session reattach while the daemon instance remains alive.
- session persistence advertisement.
- Python resource/tool surface for shell sessions.

Exit criteria:

- a UI or tool can open a shell session, send input, resize, detach, list, and reattach while daemon is alive.
- daemon restart marks in-memory sessions as lost unless a durable backend is configured.
- no SDK `Shell.execute` path depends on shell session state.

## Phase 6: Local Daemon Socket

Deliverables:

- Unix domain socket transport.
- Windows named pipe transport.
- auth token or OS ACL enforcement.
- reconnect to already-running daemon.
- provider online/offline/resumed events.

Exit criteria:

- Desktop or another app can keep `ya-envd` alive independently from a single agent run.
- Claw/SDK can reconnect and validate handle survival.

## Phase 7: Tools, Resources, and Artifacts

Deliverables:

- `resource.*` lifecycle.
- file watch resources.
- `tool.list/tool.call/tool.cancel`.
- artifact reserve/upload/complete/abort client support.
- runtime-owned artifact upload integration.

Exit criteria:

- custom tool descriptors become SDK tools.
- file watch events can be consumed without starting an agent automatically.
- artifacts are traceable to runs when context includes `run_id`.

## Phase 8: Claw Integration

Deliverables:

- provider registry.
- grant and binding persistence.
- `DaemonWorkspaceProvider`.
- session binding to provider/mounts/targets.
- provider readiness API.
- run trace projections for daemon calls.
- activation policy hooks for provider events.

Exit criteria:

- a Claw session can run on an agent-owned daemon provider.
- a Claw session can bind to a user-mounted provider when online.
- offline user-mounted providers do not block detached execution unless policy requires them.

## Phase 9: Desktop Integration

Deliverables:

- Desktop-managed `ya-envd` lifecycle.
- Space to provider environment mapping.
- local grants and approval UX.
- diagnostics UI.
- local audit UI.
- provider reconnect and activation signal handling.

Exit criteria:

- Desktop can expose a selected folder as a mount.
- Desktop can expose a sandboxed local shell target.
- user pause/revoke is reflected in Claw.
- provider resume can trigger an application-level activation decision.

## Phase 10: Computer Use

Deliverables:

- `computer.status`.
- `computer.see`.
- `computer.act`.
- `computer.pause/resume/takeover/release`.
- screenshot and accessibility artifact upload.
- local safety controls.

Exit criteria:

- computer use works as a specialized capability with standard pause/takeover semantics.

## Test Matrix

| Area           | Test                                                               |
| -------------- | ------------------------------------------------------------------ |
| protocol       | JSON-RPC parse/serialize fixtures, unknown fields, unknown methods |
| transport      | stdio framing, socket reconnect, WebSocket message ordering        |
| initialize     | protocol negotiation, capability acceptance, grant filtering       |
| fileops        | path normalization, symlink escape, ro/rw mode, streaming          |
| shell.exec     | stateless cwd/env behavior, timeout, output truncation             |
| process        | start, stdin, wait, signal, kill, exited/lost events               |
| shell_session  | open, input, resize, detach, reattach, lost on daemon restart      |
| readiness      | provider online/offline/paused/resumed and app activation boundary |
| tools          | JSON Schema descriptor to SDK tool, policy filtering               |
| resources      | create/list/export/restore/dispose                                 |
| artifacts      | reserve/upload/complete/abort failure modes                        |
| security       | token scope, grants, mount policy, shell policy, audit             |
| cross-platform | macOS, Linux, Windows path and shell behavior                      |

## MVP Cut

Recommended first production slice:

1. `ya-environment-protocol.v1` schemas and fixtures.
2. Rust `ya-envd` stdio binary.
3. Python stdio client.
4. `DaemonEnvironment` and `DaemonFileOperator`.
5. `shell.exec` and `process.start/wait/kill`.
6. provider readiness events.
7. Claw agent-owned daemon binding.

Desktop user-mounted providers and stateful shell sessions should follow once the daemon and adapter base are stable.
