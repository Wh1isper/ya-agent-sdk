# Docker Workspace Provider

`DockerWorkspaceProvider` is the backend used when `YA_CLAW_WORKSPACE_PROVIDER_BACKEND=docker`. It gives agents a `/workspace` path and runs shell commands in a reusable Docker workspace container.

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
```

Set this when the service process path and Docker daemon path differ:

```env
YA_CLAW_WORKSPACE_PROVIDER_DOCKER_HOST_WORKSPACE_DIR=/srv/ya-claw/workspace
```

Mount additional host directories into the reusable workspace container with comma-separated `host_path:container_path[:mode]` entries. Supported modes are `rw` and `ro`.

```env
YA_CLAW_WORKSPACE_PROVIDER_DOCKER_EXTRA_MOUNTS=/srv/ya-claw/home:/home/claw:rw,/srv/ya-claw/cache:/cache:ro
```

## Binding Semantics

`DockerWorkspaceProvider` returns a binding with:

| Field              | Meaning                                                                                                           |
| ------------------ | ----------------------------------------------------------------------------------------------------------------- |
| `host_path`        | service-visible workspace path from `YA_CLAW_WORKSPACE_DIR`                                                       |
| `docker_host_path` | Docker daemon-visible path from `YA_CLAW_WORKSPACE_PROVIDER_DOCKER_HOST_WORKSPACE_DIR`, defaulting to `host_path` |
| `virtual_path`     | `/workspace`                                                                                                      |
| `cwd`              | `/workspace`                                                                                                      |
| `backend_hint`     | `docker`                                                                                                          |

Extra mounts are recorded in binding metadata as `extra_mounts` and passed to Docker container creation. The workspace container sees the configured `container_path` values directly.

`DockerEnvironmentFactory` creates a `ReusableSandboxEnvironment`. It uses `host_path` for virtual file operations and `docker_host_path` as the bind mount source when creating the workspace container.

## Container Reuse

YA Claw builds a stable workspace container name from `docker_host_path` and image:

```text
ya-claw-workspace-<fingerprint>
```

The selected container ID is cached at:

```text
${YA_CLAW_WORKSPACE_PROVIDER_DOCKER_CONTAINER_CACHE_DIR}/workspace.json
```

On each run, YA Claw reads the cache, verifies the container, starts stopped containers, checks Docker health when available, recreates failed containers, and writes the refreshed cache. Recreate the workspace container after changing extra mount configuration.

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

```bash
docker ps --filter 'name=ya-claw-workspace'
docker inspect ya-claw-workspace-<fingerprint> --format '{{ json .Mounts }}'
docker exec -it ya-claw-workspace-<fingerprint> pwd
docker exec -it ya-claw-workspace-<fingerprint> ls -la /workspace
docker exec -it ya-claw-workspace-<fingerprint> lark-cli --version
```

Clear the workspace container after image or mount changes:

```bash
docker rm -f ya-claw-workspace-<fingerprint>
rm -f /var/lib/ya-claw/data/docker-workspace-containers/workspace.json
```
