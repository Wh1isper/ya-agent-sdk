# 02. Connection Model

## Goal

YA Desktop should treat every runtime target as a Claw connection. The same UI should work with local embedded Claw, self-hosted Claw, and cloud Claw.

The local embedded connection adds daemon lifecycle management. Remote and cloud connections add authentication and server capability discovery.

## Current Local Connection Implementation

The current P0 Desktop implementation derives an active local connection directly from `getLocalClawStatus()`:

```ts
type DesktopClawConnection = {
  id: 'local'
  kind: 'local_embedded'
  name: 'Local Claw'
  baseUrl: string
  apiToken?: string | null
  dataDir?: string | null
  workspaceDir?: string | null
}
```

When Local Claw is running and exposes `baseUrl`, the React UI creates a `ClawHttpClient` for read-only P0 data. The local bearer token is still backed by the Desktop-managed `.env` file created by Rust Core. A future keychain slice will move this secret behind a secure token reference and leave only `tokenRef` in frontend-visible connection config.

## Connection Registry

YA Desktop maintains a connection registry in desktop app config.

```ts
type ClawConnection =
  | LocalEmbeddedClawConnection
  | RemoteClawConnection
  | CloudClawConnection

type BaseClawConnection = {
  id: string
  name: string
  baseUrl: string
  auth: ClawAuth
  capabilities?: ClawCapabilities
  lastHealth?: ClawHealth
}

type LocalEmbeddedClawConnection = BaseClawConnection & {
  kind: 'local_embedded'
  managed: true
  dataDir: string
  binaryPath: string
}

type RemoteClawConnection = BaseClawConnection & {
  kind: 'remote'
  managed: false
}

type CloudClawConnection = BaseClawConnection & {
  kind: 'cloud'
  managed: false
  orgId?: string
}
```

Example config:

```json
{
  "activeConnectionId": "local",
  "connections": [
    {
      "id": "local",
      "kind": "local_embedded",
      "name": "Local Computer",
      "baseUrl": "http://127.0.0.1:49321",
      "managed": true,
      "dataDir": "...",
      "auth": {
        "type": "bearer",
        "tokenRef": "keychain:local-claw-token"
      }
    },
    {
      "id": "prod-claw",
      "kind": "remote",
      "name": "Production Claw",
      "baseUrl": "https://claw.example.com",
      "auth": {
        "type": "bearer",
        "tokenRef": "keychain:prod-claw-token"
      }
    },
    {
      "id": "ya-cloud",
      "kind": "cloud",
      "name": "YA Cloud",
      "baseUrl": "https://api.ya.example.com/claw",
      "auth": {
        "type": "oauth",
        "accountId": "acct_123"
      }
    }
  ]
}
```

## Secret Storage

Connection config should store secret references. Secret values should live in the OS keychain:

- macOS Keychain
- Windows Credential Manager
- Linux Secret Service / libsecret

Local embedded Claw should generate its own bearer token during first setup and store it through the same keychain abstraction.

## Unified Claw Client

The UI should use one API client interface for local, remote, and cloud Claw.

```ts
export interface ClawClient {
  connectionId: string

  health(): Promise<ClawHealth>
  info(): Promise<ClawInfo>
  capabilities(): Promise<ClawCapabilities>

  listWorkspaces(): Promise<Workspace[]>
  listSessions(params?: ListSessionsParams): Promise<SessionList>
  getSession(sessionId: string): Promise<SessionDetail>
  listSessionTurns(sessionId: string): Promise<SessionTurnsResponse>
  getRunTrace(runId: string): Promise<RunTraceResponse>

  streamSessionRun(
    sessionId: string,
    input: CreateRunInput,
    handlers: StreamHandlers,
  ): Promise<RunHandle>

  cancelRun(runId: string): Promise<void>
}
```

Connection-specific behavior lives below the client:

- Local embedded connection manages daemon lifecycle.
- Remote connection manages base URL, token, and TLS.
- Cloud connection manages OAuth and org/project context.

## Capability Discovery

Desktop should fetch capabilities after connecting.

```http
GET /api/v1/capabilities
```

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
    "sandboxed_shell": true,
    "remote_rpc_environment": false
  },
  "auth": {
    "schemes": ["bearer"]
  },
  "workspace_providers": ["local", "docker", "cloud"],
  "local_shell_runtimes": ["linux_bwrap_seccomp"],
  "workspace_mount_modes": ["bind_mount"],
  "profiles": ["default", "code", "research"],
  "limits": {
    "max_upload_bytes": 104857600,
    "max_sse_duration_seconds": 3600
  }
}
```

Desktop uses capabilities to decide which UI features are enabled for the active connection:

- file tree
- shell output
- sandboxed shell state
- remote RPC tools
- cloud workspace controls
- bridge controls
- memory controls

## Connection Switching

Connection switching should preserve UI clarity:

```text
Connection: Local Computer
Workspace: /Users/you/code/ya-mono
Profile: default
```

```text
Connection: Team Cloud
Workspace: cloud://org/project/repo
Profile: coding-prod
```

On switch, Desktop should refresh:

- sessions
- workspaces
- profiles
- server capabilities
- active run streams
- running run state

## Authentication Models

Initial supported auth schemes:

- bearer token for local and self-hosted Claw
- OAuth or hosted account session for cloud Claw

Future auth additions can include:

- device code auth
- mTLS
- SSO-backed cloud tokens
- short-lived scoped run tokens

## Health State

Desktop should track health per connection:

```ts
type ClawHealth = {
  status: 'ready' | 'starting' | 'degraded' | 'unreachable'
  version?: string
  instanceId?: string
  uptimeSeconds?: number
  checkedAt: string
  error?: string
}
```

Health state should drive connection badges, reconnect prompts, daemon restart actions, and offline display states.
