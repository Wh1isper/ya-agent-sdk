# 05. Implementation Plan

## Phase 1: Spec and Models

Deliverables:

- spec folder.
- Python protocol models.
- TypeScript protocol model references for product clients.
- frame validation fixtures.
- method namespace constants.

Suggested Python package layout:

```text
packages/ya-relay/
  pyproject.toml
  ya_relay/
    __init__.py
    protocol.py
    errors.py
    capabilities.py
```

## Phase 2: In-Process Mock Transport

Build an in-process transport before WebSocket integration.

Deliverables:

- mock relay server.
- mock relay client.
- request/response roundtrip.
- stream and cancel tests.
- fake fileops provider.
- fake custom tool provider.

This validates Environment mapping without product-specific networking.

## Phase 3: SDK Environment Integration

Deliverables:

- `RelayEnvironment`.
- `RelayFileOperator`.
- `RelayShell`.
- `RelayToolset`.
- resource registry draft.

Tests:

- file read/list/write against fake provider.
- shell streaming result.
- generated tool call through `tool.call`.
- cancellation behavior.

## Phase 4: WebSocket Transport

Deliverables:

- relay server transport.
- relay client transport.
- heartbeat.
- reconnect state.
- request timeout handling.
- connection registry.

The server transport can first live in Claw, then move common pieces into `ya-relay` after the API stabilizes.

## Phase 5: Claw Integration

Deliverables:

- Claw relay connection endpoint.
- relay provider registry.
- `RelayWorkspaceProvider`.
- session workspace binding support.
- run trace projections for relay calls.
- artifact reserve/upload/complete endpoints or adapters.

## Phase 6: Desktop Integration

Deliverables:

- Desktop relay client.
- Space-level relay grants.
- local fileops provider.
- local shell provider.
- custom tool registration.
- relay diagnostics UI.

## Phase 7: Computer Use Capability

Deliverables:

- `RelayComputerProvider`.
- `computer.status`.
- `computer.see`.
- `computer.act`.
- artifact upload for screenshots and UI trees.
- Desktop Host Computer Bridge integration.

## Test Matrix

| Area              | Test                                                 |
| ----------------- | ---------------------------------------------------- |
| protocol          | frame parse/serialize fixtures                       |
| request lifecycle | success, error, timeout, cancel                      |
| streaming         | stdout/stderr/progress ordering                      |
| fileops           | path normalization and root boundary                 |
| shell             | output stream and signal handling                    |
| tools             | JSON Schema descriptor to SDK tool                   |
| artifacts         | reserve/upload/complete failure modes                |
| reconnect         | pending request failure and provider re-registration |
| security          | grant filtering and capability acceptance            |

## MVP Cut

Recommended first production slice:

1. `ya-relay.v1` protocol models.
2. Claw WebSocket endpoint.
3. Desktop relay client.
4. file read/list/stat.
5. shell start with stream and cancel.
6. custom tool list/call.
7. artifact reserve/complete.

Computer use becomes the first high-value specialized capability after the base relay environment is proven.
