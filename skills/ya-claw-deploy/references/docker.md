# Docker Deployment

This deployment runs the YA Claw server in the `Dockerfile.ya-claw` service image. The recommended workspace shape for this server deployment is service Docker + Docker shell.

Read [`workspace-provider/service-docker-docker-shell.md`](workspace-provider/service-docker-docker-shell.md) for the path mapping details.

## Runtime Shape

```mermaid
flowchart TB
    RP["Reverse Proxy or Direct Client"] --> SVC["YA Claw Server Container"]
    SVC --> DB[("SQLite file or PostgreSQL")]
    SVC --> DATA["/var/lib/ya-claw/data"]
    SVC --> SERVICE_WS["/var/lib/ya-claw/workspace"]
    SVC --> SOCK["/var/run/docker.sock"]
    SOCK --> WSC["Reusable ya-claw-workspace Container"]
    HOST_WS["/srv/ya-claw/workspace"] <--> WSC
```

## Images

- `Dockerfile.ya-claw` builds the server image.
- `Dockerfile.ya-claw-workspace` builds the workspace image used by the Docker workspace provider.

Build locally:

```bash
make docker-build-claw
make docker-build-claw-workspace
```

Equivalent commands:

```bash
docker build -f Dockerfile.ya-claw -t ya-claw:dev .
docker build -f Dockerfile.ya-claw-workspace -t ya-claw-workspace:dev .
```

## Server Startup Contract

The server image runs:

```text
ya-claw start
```

The `start` command:

1. requires `YA_CLAW_API_TOKEN`
2. runs database migrations when `YA_CLAW_AUTO_MIGRATE=true`
3. seeds profiles when `YA_CLAW_AUTO_SEED_PROFILES=true`
4. starts the FastAPI service with bundled web assets

The image sets these defaults:

```env
YA_CLAW_ENVIRONMENT=production
YA_CLAW_HOST=0.0.0.0
YA_CLAW_PORT=9042
YA_CLAW_AUTO_MIGRATE=true
YA_CLAW_WEB_DIST_DIR=/srv/ya-claw/web-dist
MALLOC_ARENA_MAX=2
MALLOC_TRIM_THRESHOLD_=131072
PYTHONMALLOC=malloc
```

The workspace image also sets the same allocator defaults, so Python tools and CLIs in reusable workspace containers inherit the tuning.

## Environment

Minimal server env with SQLite:

```env
YA_CLAW_ENVIRONMENT=production
YA_CLAW_HOST=0.0.0.0
YA_CLAW_PORT=9042
YA_CLAW_PUBLIC_BASE_URL=https://claw.example.com
YA_CLAW_API_TOKEN=replace-with-a-long-random-token
YA_CLAW_AUTO_MIGRATE=true
YA_CLAW_DATA_DIR=/var/lib/ya-claw/data
YA_CLAW_WORKSPACE_DIR=/var/lib/ya-claw/workspace
YA_CLAW_WORKSPACE_PROVIDER_BACKEND=docker
YA_CLAW_WORKSPACE_PROVIDER_DOCKER_HOST_WORKSPACE_DIR=/srv/ya-claw/workspace
YA_CLAW_WORKSPACE_PROVIDER_DOCKER_IMAGE=ghcr.io/wh1isper/ya-claw-workspace:latest
YA_CLAW_WORKSPACE_PROVIDER_DOCKER_UID=1000
YA_CLAW_WORKSPACE_PROVIDER_DOCKER_GID=1000
YA_CLAW_WORKSPACE_PROVIDER_DOCKER_EXEC_USER=auto
YA_CLAW_WORKSPACE_PROVIDER_DOCKER_HOME=/home/claw
YA_CLAW_PROFILE_SEED_FILE=/etc/ya-claw/profiles.yaml
YA_CLAW_AUTO_SEED_PROFILES=true
MALLOC_ARENA_MAX=2
MALLOC_TRIM_THRESHOLD_=131072
PYTHONMALLOC=malloc
GATEWAY_API_KEY=replace-with-provider-key
GATEWAY_BASE_URL=https://gateway.example.com
```

For PostgreSQL, add:

```env
YA_CLAW_DATABASE_URL=postgresql+psycopg://ya_claw:ya_claw@postgres:5432/ya_claw
```

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
    volumes:
      - /srv/ya-claw:/var/lib/ya-claw
      - ./profiles.yaml:/etc/ya-claw/profiles.yaml:ro
      - /var/run/docker.sock:/var/run/docker.sock
```

The service sees `/var/lib/ya-claw/workspace`. Docker Engine sees `/srv/ya-claw/workspace`. The workspace container receives `/srv/ya-claw/workspace` mounted as `/workspace`.

## Persistent Paths

| Path                           | Purpose                            |
| ------------------------------ | ---------------------------------- |
| Host path                      | Service path                       |
| ---                            | ---                                |
| `/srv/ya-claw/data`            | `/var/lib/ya-claw/data`            |
| `/srv/ya-claw/workspace`       | `/var/lib/ya-claw/workspace`       |
| `/srv/ya-claw/ya_claw.sqlite3` | `/var/lib/ya-claw/ya_claw.sqlite3` |

With this parent mount, set `YA_CLAW_WORKSPACE_PROVIDER_DOCKER_HOST_WORKSPACE_DIR=/srv/ya-claw/workspace`.

## Start and Verify

```bash
mkdir -p /srv/ya-claw/data /srv/ya-claw/workspace
cp packages/ya-claw/profiles.yaml ./profiles.yaml
docker compose up -d
curl http://127.0.0.1:9042/healthz
curl \
  -H "Authorization: Bearer ${YA_CLAW_API_TOKEN}" \
  http://127.0.0.1:9042/api/v1/claw/info
```

After the first run that needs workspace execution:

```bash
docker ps --filter 'name=ya-claw-workspace'
docker inspect ya-claw-workspace-<fingerprint> --format '{{ json .Mounts }}'
```
