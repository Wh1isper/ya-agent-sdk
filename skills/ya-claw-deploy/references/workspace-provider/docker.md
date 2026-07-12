# Docker Workspace Provider

`DockerWorkspaceProvider` is the backend used when `YA_CLAW_WORKSPACE_PROVIDER_BACKEND=docker`. It gives agents a virtual `/workspace` namespace and runs shell commands in Docker workspace containers.

Read these shape-specific guides first:

- [`service-local-docker-shell.md`](service-local-docker-shell.md) for a host YA Claw service with Docker shell execution
- [`service-docker-docker-shell.md`](service-docker-docker-shell.md) for a Dockerized YA Claw service with Docker shell execution
- [`overview.md`](overview.md) for the full workspace provider matrix

## Core Configuration

```env
YA_CLAW_WORKSPACE_PROVIDER_BACKEND=docker
YA_CLAW_WORKSPACE_DIR=/var/lib/ya-claw/workspace
YA_CLAW_WORKSPACE_PROVIDER_DOCKER_IMAGE=ghcr.io/wh1isper/ya-claw-workspace:latest
YA_CLAW_WORKSPACE_PROVIDER_DOCKER_CONTAINER_CACHE_DIR=/var/lib/ya-claw/data/docker-workspace-containers
YA_CLAW_WORKSPACE_PROVIDER_DOCKER_RETENTION_POLICY=stop_on_idle
YA_CLAW_WORKSPACE_PROVIDER_DOCKER_IDLE_TTL_SECONDS=3600
```

Set this when the service process path and Docker daemon path differ:

```env
YA_CLAW_WORKSPACE_PROVIDER_DOCKER_HOST_WORKSPACE_DIR=/srv/ya-claw/workspace
```

Mount provider support directories into workspace containers with comma-separated `host_path:container_path[:mode]` entries. Supported modes are `rw` and `ro`.

```env
YA_CLAW_WORKSPACE_PROVIDER_DOCKER_EXTRA_MOUNTS=/srv/ya-claw/home:/home/claw:rw,/srv/ya-claw/cache:/cache:ro
```

## Workspace Binding Semantics

YA Claw has two mount layers:

1. Logical workspace mounts come from API/session/run `workspace.mounts` and define the project folders, default cwd, guidance root, memory root, file browsing root, and runtime prompt workspace list.
2. Provider extra mounts come from `YA_CLAW_WORKSPACE_PROVIDER_DOCKER_EXTRA_MOUNTS` and add support paths such as home, cache, credentials, or shared tool directories to the concrete Docker container.

When API clients omit `workspace`, the configured service workspace becomes the fallback logical binding:

| Field              | Fallback value                                                                                                    |
| ------------------ | ----------------------------------------------------------------------------------------------------------------- |
| `host_path`        | service-visible workspace path from `YA_CLAW_WORKSPACE_DIR`                                                       |
| `docker_host_path` | Docker daemon-visible path from `YA_CLAW_WORKSPACE_PROVIDER_DOCKER_HOST_WORKSPACE_DIR`, defaulting to `host_path` |
| `virtual_path`     | `/workspace`                                                                                                      |
| `cwd`              | `/workspace`                                                                                                      |
| `backend_hint`     | `docker`                                                                                                          |

Request-level logical mounts support multiple folders:

```json
{
  "workspace": {
    "mounts": [
      {
        "id": "main",
        "host_path": "/srv/projects/app",
        "docker_host_path": "/srv/projects/app",
        "virtual_path": "/workspace/main",
        "mode": "rw"
      },
      {
        "id": "docs",
        "host_path": "/srv/projects/docs",
        "docker_host_path": "/srv/projects/docs",
        "virtual_path": "/workspace/docs",
        "mode": "ro"
      }
    ],
    "default_mount_id": "main",
    "cwd": "/workspace/main"
  }
}
```

YA Claw validates virtual path conflicts between logical mounts and provider extra mounts. Extra mounts are included in the workspace fingerprint and stay outside the logical workspace association.

## Sandbox Scopes and Generation

Docker sandbox scope follows the run source:

| Source                                  | Scope     | Container ref format                    | Cleanup behavior                          |
| --------------------------------------- | --------- | --------------------------------------- | ----------------------------------------- |
| API, bridge, memory, foreground session | `session` | `ya-claw-session-{session_id_short}-gN` | Reused until idle TTL or explicit cleanup |
| schedule, workflow, and heartbeat       | `run`     | `ya-claw-run-{run_id_short}`            | Stopped when the run exits                |

Session-scoped Docker stores one active generation in session metadata. The generation increments when the workspace fingerprint changes. Fingerprint inputs include logical mounts, provider extra mounts, Docker image, UID, GID, cwd, and backend.

Session-scoped cache path:

```text
${YA_CLAW_WORKSPACE_PROVIDER_DOCKER_CONTAINER_CACHE_DIR}/sessions/{session_id}/workspace.json
```

Run-scoped cache path:

```text
${YA_CLAW_WORKSPACE_PROVIDER_DOCKER_CONTAINER_CACHE_DIR}/runs/{run_id}/workspace.json
```

The cache file stores the current container ID, generation, fingerprint, container ref, Docker image, image digest, and lifecycle metadata. New containers carry YA Claw ownership labels for the full session ID, sandbox scope, generation, and run ID when run-scoped. Session pruning derives container names and cache paths from trusted database IDs and settings, then verifies all ownership labels before removal; metadata-provided container IDs, refs, and cache paths are not deletion authority.

Containers created by releases before ownership labels were introduced are intentionally not removed automatically. After upgrading, if generated-session pruning reports repeated Docker sandbox cleanup failures, inspect the deterministic `ya-claw-session-*` or `ya-claw-run-*` container, remove that legacy container manually after confirming it belongs to the deployment, and let the next run recreate it with ownership labels. This fail-closed behavior prevents an unlabelled name collision from deleting an unrelated container.

On each run, YA Claw resolves the current Docker image digest before trusting the cached container. A changed image digest takes precedence over container existence: YA Claw removes the stale container and starts a new container from the current image. YA Claw also starts stopped containers, checks Docker health when available, recreates failed containers, and writes refreshed metadata.

## Idle TTL

Session-scoped Docker sandboxes refresh `last_used_at` while a run is active and once more when the run exits after agent cleanup. The TTL cleaner scans session sandbox metadata, uses the same cache-path lock as run startup, stops idle containers, and deletes the session `workspace.json` cache file when:

```text
last_used_at + idle_ttl_seconds <= now
```

Retention policies:

| Policy         | Behavior                                                                                    |
| -------------- | ------------------------------------------------------------------------------------------- |
| `stop_on_idle` | Stop the session container after the idle TTL. Next run restarts it.                        |
| `keep_warm`    | Keep the session container running across runs and service restarts until explicit cleanup. |

Run-scoped schedule, workflow, and heartbeat sandboxes always use terminal cleanup.

## Docker Permission

The service process must access Docker Engine. Host deployments usually use group membership. Dockerized service deployments usually mount the Docker socket.

```yaml
volumes:
  - /var/run/docker.sock:/var/run/docker.sock
```

## Workspace Container Environment

Auto-started workspace containers receive:

```env
YA_CLAW_WORKSPACE_STARTUP_DIR=/workspace
YA_CLAW_WORKSPACE_UID=<configured uid>
YA_CLAW_WORKSPACE_GID=<configured gid>
YA_CLAW_HOST_UID=<configured uid>
YA_CLAW_HOST_GID=<configured gid>
```

Workspace environment values are injected when configured. Built-in Lark aliases are available for `lark-cli`, and additional process env values are forwarded by name:

```env
LARK_APP_ID=cli_xxx
LARK_APP_SECRET=replace-with-secret
MY_TOOL_API_KEY=replace-with-tool-key
YA_CLAW_WORKSPACE_ENV_VARS=MY_TOOL_API_KEY
```

## UID/GID Alignment

The service image can drop privileges through:

```env
YA_CLAW_RUN_UID=1000
YA_CLAW_RUN_GID=1000
```

The workspace container user and Docker exec identity can be set through:

```env
YA_CLAW_WORKSPACE_PROVIDER_DOCKER_UID=1000
YA_CLAW_WORKSPACE_PROVIDER_DOCKER_GID=1000
YA_CLAW_WORKSPACE_PROVIDER_DOCKER_EXEC_USER=auto
YA_CLAW_WORKSPACE_PROVIDER_DOCKER_HOME=/home/claw
```

`auto` resolves Docker exec to the configured workspace UID:GID. Use `root` for maintenance sessions that should execute commands as root.

## Workspace Image Contents

The official workspace image contains:

- Debian stable
- Python, `pip`, and `venv`
- Node.js and Corepack
- Git, OpenSSH, curl, wget, jq, unzip, zip, and shell utilities
- `lark-cli`
- bundled Lark and skill-creator skills copied into `/workspace/.agents/skills/`

## Verification

List active containers:

```bash
docker ps --filter 'name=ya-claw-session'
docker ps --filter 'name=ya-claw-run'
```

Inspect a session container:

```bash
docker inspect ya-claw-session-<session-short>-g<generation> --format '{{ json .Mounts }}'
docker exec -it ya-claw-session-<session-short>-g<generation> pwd
docker exec -it ya-claw-session-<session-short>-g<generation> ls -la /workspace
docker exec -it ya-claw-session-<session-short>-g<generation> lark-cli --version
```

Reset a session-scoped workspace sandbox after image, UID/GID, or mount changes:

```bash
docker rm -f ya-claw-session-<session-short>-g<generation>
rm -f /var/lib/ya-claw/data/docker-workspace-containers/sessions/<session-id>/workspace.json
```

Reset run-scoped diagnostics cache:

```bash
rm -rf /var/lib/ya-claw/data/docker-workspace-containers/runs/<run-id>
```
