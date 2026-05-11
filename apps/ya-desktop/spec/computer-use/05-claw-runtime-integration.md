# 05. Claw Runtime Integration

## Goal

Claw should treat Host Computer Use as a runtime capability exposed through profiles and tools. The provider implementation stays behind YA Desktop's bridge, while Claw remains responsible for model execution, approvals, run trace, and durable artifacts.

## Profile Shape

Profiles should opt into computer tools explicitly:

```yaml
- name: mac-host-computer
  model: openai:gpt-4.1
  builtin_toolsets:
    - core
    - computer
  workspace_backend_hint: local
  model_config_override:
    capabilities:
      - vision
    computer_use:
      enabled: true
      provider: host-macos
      require_approval: true
      max_actions_per_run: 80
```

The `computer` builtin toolset should resolve to a provider proxy at runtime. The proxy can target local embedded Desktop, remote Desktop RPC, or future sandbox providers.

## Runtime Capability Registration

Desktop should register provider endpoints with local `ya-clawd`:

```http
POST /api/v1/runtime/computer-providers
Authorization: Bearer <local-token>
Content-Type: application/json
```

Example request:

```json
{
  "provider_id": "host-macos",
  "kind": "host",
  "platform": "macos",
  "transport": {
    "kind": "loopback_http",
    "base_url": "http://127.0.0.1:49152",
    "token_ref": "desktop-keychain:computer-bridge-token"
  },
  "capabilities": {
    "screenshots": true,
    "accessibility_tree": true,
    "semantic_actions": true,
    "coordinate_input": true,
    "live_monitor": true
  },
  "policy": {
    "requires_desktop_permission_host": true
  }
}
```

Claw stores active provider registration in process memory. Desktop re-registers after sidecar restart.

## Tool Proxy

The Claw computer toolset should be provider-neutral:

```python
class ComputerUseToolset(Toolset):
    async def computer_see(...): ...
    async def computer_act(...): ...
    async def computer_wait(...): ...
    async def computer_status(...): ...
```

At call time the toolset:

1. resolves the provider from profile and session workspace binding.
2. applies Claw profile policy.
3. creates HITL interactions when required.
4. calls the provider bridge.
5. stores returned artifacts in the run-store.
6. emits AGUI-aligned events and compact trace projections.

## Session Binding

Session workspace binding can include a computer provider hint:

```json
{
  "metadata": {
    "computer_use": {
      "enabled": true,
      "provider_id": "host-macos",
      "trust_scope_id": "space_123:local:host-macos",
      "permission_host": "ya_desktop"
    }
  }
}
```

A session should only receive computer tools when both the profile and the Space trust scope enable the provider.

## Run Events

Computer tool calls should appear in run event streams as normal tool calls plus compact computer metadata:

```json
{
  "type": "tool_call.completed",
  "payload": {
    "tool_call_id": "call_123",
    "tool_name": "computer_act",
    "status": "succeeded",
    "computer": {
      "provider_id": "host-macos",
      "action_kind": "click",
      "strategy": "accessibility",
      "app_name": "Safari",
      "window_title": "GitHub",
      "snapshot_id": "snap_456",
      "screenshot_artifact_id": "art_789"
    }
  }
}
```

`message.json` should store compact replay entries. Large artifacts stay in the run-store.

## Artifact Storage

Recommended run-store layout:

```text
run-store/{run_id}/
  state.json
  message.json
  artifacts/
    computer/
      snap_0001.png
      snap_0001.tree.json
      action_0001.json
      redaction_0001.json
```

Artifact metadata should include provider, app/window, dimensions, redaction status, and retention class.

## HITL Integration

Computer tool policy should use existing Claw HITL primitives:

```yaml
need_user_approve_tools:
  - computer_act
```

Policy can narrow approvals by action category:

```yaml
computer_use_policy:
  require_approval_for:
    - credential_field
    - destructive_action
    - external_communication
  allow_without_approval:
    - screen_read
    - click
    - scroll
```

Desktop receives `interaction.requested` through global SSE and renders approval cards in Inbox.

## Remote RPC Mode

Remote Claw can call a Desktop provider through a Desktop-authenticated tunnel. The Desktop app decides which remote connections can use host computer use.

```mermaid
flowchart LR
    RemoteClaw[Remote Claw] --> RPC[Computer RPC Session]
    RPC --> Desktop[YA Desktop]
    Desktop --> Bridge[Host Computer Bridge]
    Bridge --> Provider[macOS Provider]
```

Remote RPC requirements:

- explicit user enablement per remote connection.
- short-lived capability token.
- provider-side pause and revoke controls.
- artifact upload confirmation policy for screenshots.
- clear UI showing the remote runtime identity.

## API Additions

Claw API additions:

```http
GET  /api/v1/computer-providers
POST /api/v1/runtime/computer-providers
DELETE /api/v1/runtime/computer-providers/{provider_id}
GET  /api/v1/sessions/{session_id}/computer/status
POST /api/v1/sessions/{session_id}/computer:pause
POST /api/v1/sessions/{session_id}/computer:resume
POST /api/v1/sessions/{session_id}/computer:takeover
POST /api/v1/sessions/{session_id}/computer:release
GET  /api/v1/runs/{run_id}/computer-trace
GET  /api/v1/runs/{run_id}/artifacts/{artifact_id}
```

The provider bridge API can stay private to Desktop and local Claw.

## SDK Placement

Provider-neutral models and toolset should live in the SDK:

```text
packages/ya-agent-sdk/ya_agent_sdk/toolsets/computer_use/
packages/ya-agent-sdk/ya_agent_sdk/computer/
```

Claw-specific provider registry and artifact storage should live in Claw:

```text
packages/ya-claw/ya_claw/computer/
packages/ya-claw/ya_claw/toolsets/computer.py
packages/ya-claw/ya_claw/api/computer.py
```

Desktop provider implementation should live in Desktop:

```text
apps/ya-desktop/src-tauri/src/computer/
apps/ya-desktop/src/features/computer-use/
```
