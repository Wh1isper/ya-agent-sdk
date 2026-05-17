# Operations

Use these checks for production deployment verification, upgrades, backup, restore, and troubleshooting.

## Health Checks

Unauthenticated service health:

```bash
curl http://127.0.0.1:9042/healthz
```

Authenticated service info:

```bash
curl -sS \
  -H "Authorization: Bearer ${YA_CLAW_API_TOKEN}" \
  http://127.0.0.1:9042/api/v1/claw/info
```

The info endpoint reports service capabilities, storage model, workspace provider backend, and auth mode.

## Runtime Logs

Docker compose:

```bash
docker compose logs -f ya-claw
```

systemd:

```bash
journalctl -u ya-claw -f
```

Look for:

- database migration status
- seeded profile names
- execution supervisor startup
- runtime instance registration
- agency dispatcher startup when session agency is enabled
- bridge supervisor startup when embedded bridge is enabled
- workspace container creation or reuse errors

## Agency Checks

Session agency uses paired internal agency sessions and durable agency signals. Use the dedicated operations guide for configuration, API checks, backup, and troubleshooting: [`agency.md`](agency.md).

Quick manual signal check:

```bash
curl -sS -X POST \
  -H "Authorization: Bearer ${YA_CLAW_API_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"reason":"manual","client_token":"ops-check-1","prompt_override":"Run a short agency deployment check."}' \
  http://127.0.0.1:9042/api/v1/sessions/${SESSION_ID}/agency:signal
```

Expected delivery is `submitted`, `steered`, or `duplicate`. Agency runs appear with `trigger_type="agency"`, and agency workspace state lives under `memory/AGENCY.md` and `memory/agency/**`.

## Bridge Checks

For embedded bridges, use the bridge-specific operations guide: [`bridge/operations.md`](bridge/operations.md).

Check service logs for bridge supervisor startup, adapter task creation, inbound dedupe results, conversation IDs, session IDs, and run IDs.

For Lark deployments, verify service ingress credentials and workspace reply credentials:

```env
YA_CLAW_BRIDGE_DISPATCH_MODE=embedded
YA_CLAW_BRIDGE_ENABLED_ADAPTERS=lark
YA_CLAW_BRIDGE_LARK_APP_ID=cli_xxx
YA_CLAW_BRIDGE_LARK_APP_SECRET=replace-with-app-secret
```

## Workspace Checks

For Docker shell shapes:

```bash
docker ps --filter 'name=ya-claw-session'
docker ps --filter 'name=ya-claw-run'
docker logs ya-claw-session-<session-short>-g<generation>
docker exec -it ya-claw-session-<session-short>-g<generation> pwd
docker exec -it ya-claw-session-<session-short>-g<generation> ls -la /workspace
docker exec -it ya-claw-session-<session-short>-g<generation> lark-cli --version
```

For service local + local shell:

```bash
sudo -u ya-claw sh -lc 'cd /var/lib/ya-claw/workspace && pwd && ls -la'
```

Remove a stale session workspace container and cache after changing the workspace image or mount contract:

```bash
docker rm -f ya-claw-session-<session-short>-g<generation>
rm -f /var/lib/ya-claw/data/docker-workspace-containers/sessions/<session-id>/workspace.json
```

## API Smoke Test

```bash
curl -sS \
  -H "Authorization: Bearer ${YA_CLAW_API_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"profile_name":"default","input_parts":[{"type":"text","text":"Run a deployment smoke test from the workspace."}]}' \
  http://127.0.0.1:9042/api/v1/sessions
```

Then list sessions:

```bash
curl -sS \
  -H "Authorization: Bearer ${YA_CLAW_API_TOKEN}" \
  http://127.0.0.1:9042/api/v1/sessions
```

## Upgrade

Docker compose baseline:

```bash
docker compose pull
docker compose up -d
curl http://127.0.0.1:9042/healthz
```

systemd + uv-managed tool baseline:

```bash
sudo env \
  UV_TOOL_DIR=/opt/ya-claw/tools \
  UV_TOOL_BIN_DIR=/opt/ya-claw/bin \
  uv tool upgrade ya-claw
sudo systemctl restart ya-claw
curl http://127.0.0.1:9042/healthz
```

Pin a specific version:

```bash
YA_CLAW_VERSION=replace-with-release-version
sudo env \
  UV_TOOL_DIR=/opt/ya-claw/tools \
  UV_TOOL_BIN_DIR=/opt/ya-claw/bin \
  uv tool install "ya-claw==${YA_CLAW_VERSION}" --python 3.13 --force
sudo systemctl restart ya-claw
```

Local source image baseline:

```bash
git pull
make docker-build-claw
make docker-build-claw-workspace
docker compose up -d --build
```

The `ya-claw start` command applies migrations when `YA_CLAW_AUTO_MIGRATE=true`.

## Session and Run Pruning

YA Claw can run a background prune job to control runtime data growth. The safe production mode prunes disk artifacts while retaining database metadata.

Recommended safe configuration:

```env
YA_CLAW_SESSION_PRUNE_ENABLED=true
YA_CLAW_SESSION_PRUNE_RUN_KEEP_RECENT=10
YA_CLAW_SESSION_PRUNE_RUN_OLDER_THAN_DAYS=30
YA_CLAW_SESSION_PRUNE_GENERATED_SESSIONS_ENABLED=false
YA_CLAW_SESSION_PRUNE_FIRE_RECORDS_OLDER_THAN_DAYS=0
YA_CLAW_SESSION_PRUNE_ORPHANS_ENABLED=true
```

Safe mode behavior:

- Deletes old `run-store/{run_id}` directories for prunable runs
- Keeps `sessions` and `runs` database rows
- Keeps `input_parts`, status, trigger type, metadata, `output_text`, and `output_summary`
- Protects each session's latest runs, `head_run_id`, `head_success_run_id`, `active_run_id`, active runs, and direct restore sources
- Deletes orphan `run-store/*` directories that have no matching `RunRecord.id`

The web UI displays a replay-artifacts-pruned notice when a selected run has database metadata but no replay files on disk.

Generated session database pruning is an explicit retention policy:

```env
YA_CLAW_SESSION_PRUNE_GENERATED_SESSIONS_ENABLED=true
YA_CLAW_SESSION_PRUNE_SCHEDULE_KEEP_RECENT=10
YA_CLAW_SESSION_PRUNE_SCHEDULE_OLDER_THAN_DAYS=30
YA_CLAW_SESSION_PRUNE_HEARTBEAT_KEEP_RECENT=10
YA_CLAW_SESSION_PRUNE_HEARTBEAT_OLDER_THAN_DAYS=7
```

This mode deletes `sessions` and `runs` rows for old heartbeat sessions and schedule isolate/fork generated sessions. It protects active schedule source/target sessions, parent sessions, active sessions, active run sessions, and sessions referenced by external `restore_from_run_id` links.

Fire-record database retention is separately enabled:

```env
YA_CLAW_SESSION_PRUNE_FIRE_RECORDS_OLDER_THAN_DAYS=30
```

This removes old `schedule_fires` and `heartbeat_fires` rows while preserving pending fires and latest fire records.

## Backup

Back up both database and run store. For SQLite compose deployments with a runtime volume mounted at `/var/lib/ya-claw`:

```bash
docker compose stop ya-claw
docker run --rm \
  -v ya-claw-runtime:/var/lib/ya-claw \
  -v "$PWD/backups:/backup" \
  alpine sh -lc 'cp /var/lib/ya-claw/ya_claw.sqlite3 /backup/ya_claw.sqlite3 && tar -czf /backup/data.tgz -C /var/lib/ya-claw data workspace'
docker compose start ya-claw
```

For PostgreSQL, combine `pg_dump` with a data/workspace archive.

## Troubleshooting

### Missing API Token

Startup fails when `YA_CLAW_API_TOKEN` is empty. Generate a long token and restart.

### Run Execution Stays Pending

Check service logs, profile model configuration, model provider credentials, and runtime supervisor startup messages.

### Workspace Container Startup Fails

Check Docker access from the service container or service user:

```bash
docker ps
```

For compose, confirm `/var/run/docker.sock` is mounted. For systemd, confirm the service user has Docker access.

For service Docker + Docker shell, confirm `YA_CLAW_WORKSPACE_PROVIDER_DOCKER_HOST_WORKSPACE_DIR` points to a Docker daemon-visible host path and that the service has the same workspace content mounted at `YA_CLAW_WORKSPACE_DIR`.

### Permission Errors in Workspace

Align service and workspace IDs for Docker shell shapes:

```env
YA_CLAW_RUN_UID=1000
YA_CLAW_RUN_GID=1000
YA_CLAW_WORKSPACE_PROVIDER_DOCKER_UID=1000
YA_CLAW_WORKSPACE_PROVIDER_DOCKER_GID=1000
YA_CLAW_WORKSPACE_PROVIDER_DOCKER_EXEC_USER=auto
YA_CLAW_WORKSPACE_PROVIDER_DOCKER_HOME=/home/claw
```

Then repair ownership on mounted paths.

### Profile Missing

Enable seed or seed manually:

```env
YA_CLAW_PROFILE_SEED_FILE=/etc/ya-claw/profiles.yaml
YA_CLAW_AUTO_SEED_PROFILES=true
```

```bash
ya-claw profiles seed --seed-file /etc/ya-claw/profiles.yaml
```

### Agency Signal Dispatch Errors

Read [`agency.md`](agency.md) for agency-specific checks, including dispatcher logs, profile availability, pending signal state, and workspace permissions.
