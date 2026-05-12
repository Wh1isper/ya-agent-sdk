# Service Docker + Docker Shell

Use this shape when the YA Claw server runs in the `Dockerfile.ya-claw` service image and agent shell execution runs in a sibling Docker workspace container.

## Runtime Shape

```mermaid
flowchart LR
    CLIENT["Client"] --> SERVICE["YA Claw service container"]
    SERVICE --> SERVICE_WS["Service path: /var/lib/ya-claw/workspace"]
    SERVICE --> SOCK["Docker socket"]
    SOCK --> DAEMON["Docker daemon"]
    DAEMON --> HOST_WS["Host path: /srv/ya-claw/workspace"]
    DAEMON --> WSC["Workspace container"]
    HOST_WS <--> WSC
    WSC --> VPATH["/workspace"]
```

## Configuration

```env
YA_CLAW_WORKSPACE_PROVIDER_BACKEND=docker
YA_CLAW_WORKSPACE_DIR=/var/lib/ya-claw/workspace
YA_CLAW_WORKSPACE_PROVIDER_DOCKER_HOST_WORKSPACE_DIR=/srv/ya-claw/workspace
YA_CLAW_WORKSPACE_PROVIDER_DOCKER_IMAGE=ghcr.io/wh1isper/ya-claw-workspace:latest
YA_CLAW_WORKSPACE_PROVIDER_DOCKER_CONTAINER_CACHE_DIR=/var/lib/ya-claw/data/docker-workspace-containers
YA_CLAW_WORKSPACE_PROVIDER_DOCKER_EXTRA_MOUNTS=/srv/ya-claw/home:/home/claw:rw
YA_CLAW_WORKSPACE_PROVIDER_DOCKER_RETENTION_POLICY=stop_on_idle
YA_CLAW_WORKSPACE_PROVIDER_DOCKER_IDLE_TTL_SECONDS=3600
```

`YA_CLAW_WORKSPACE_DIR` is the service-container path. `YA_CLAW_WORKSPACE_PROVIDER_DOCKER_HOST_WORKSPACE_DIR` is the Docker daemon-visible host path used for the workspace container bind mount. Extra mount host paths are also Docker daemon-visible host paths.

## Compose Shape

```yaml
services:
  ya-claw:
    image: ghcr.io/wh1isper/ya-claw:latest
    restart: unless-stopped
    ports:
      - "9042:9042"
    env_file:
      - .env
    environment:
      YA_CLAW_DATA_DIR: /var/lib/ya-claw/data
      YA_CLAW_WORKSPACE_DIR: /var/lib/ya-claw/workspace
      YA_CLAW_WORKSPACE_PROVIDER_BACKEND: docker
      YA_CLAW_WORKSPACE_PROVIDER_DOCKER_HOST_WORKSPACE_DIR: /srv/ya-claw/workspace
      YA_CLAW_WORKSPACE_PROVIDER_DOCKER_EXTRA_MOUNTS: /srv/ya-claw/home:/home/claw:rw
    volumes:
      - /srv/ya-claw:/var/lib/ya-claw
      - ./profiles.yaml:/etc/ya-claw/profiles.yaml:ro
      - /var/run/docker.sock:/var/run/docker.sock
```

## Path Semantics

| Binding field                            | Value                        |
| ---------------------------------------- | ---------------------------- |
| service-visible `host_path`              | `/var/lib/ya-claw/workspace` |
| Docker daemon-visible `docker_host_path` | `/srv/ya-claw/workspace`     |
| agent-visible `virtual_path`             | `/workspace`                 |
| agent cwd                                | `/workspace`                 |

File operations map `/var/lib/ya-claw/workspace` to `/workspace` inside the service process. Docker shell execution mounts `/srv/ya-claw/workspace` into the workspace container as `/workspace`.

## Docker Permission

The service container needs Docker Engine access:

```yaml
volumes:
  - /var/run/docker.sock:/var/run/docker.sock
```

The socket grants the service enough authority to create, inspect, start, and remove workspace containers.

## UID/GID Alignment

The service image can drop privileges:

```env
YA_CLAW_RUN_UID=1000
YA_CLAW_RUN_GID=1000
```

The workspace container user and Docker exec identity can match that identity:

```env
YA_CLAW_WORKSPACE_PROVIDER_DOCKER_UID=1000
YA_CLAW_WORKSPACE_PROVIDER_DOCKER_GID=1000
YA_CLAW_WORKSPACE_PROVIDER_DOCKER_EXEC_USER=auto
YA_CLAW_WORKSPACE_PROVIDER_DOCKER_HOME=/home/claw
```

Use matching ownership for `/srv/ya-claw`, including `/srv/ya-claw/data` and `/srv/ya-claw/workspace`.

## Verification

```bash
docker compose logs -f ya-claw
docker ps --filter 'name=ya-claw-session'
docker inspect ya-claw-session-<session-short>-g<generation> --format '{{ json .Mounts }}'
docker exec -it ya-claw-session-<session-short>-g<generation> pwd
docker exec -it ya-claw-session-<session-short>-g<generation> ls -la /workspace
```

The workspace container mount source should show `/srv/ya-claw/workspace` and target `/workspace`.
