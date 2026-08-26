# Bridge Operations

Use these checks when enabling embedded bridges, verifying GitHub or Lark ingress, or troubleshooting bridge-triggered runs.

## Startup Checks

Health endpoint:

```bash
curl http://127.0.0.1:9042/healthz
```

Authenticated service info:

```bash
curl -sS \
  -H "Authorization: Bearer ${YA_CLAW_API_TOKEN}" \
  http://127.0.0.1:9042/api/v1/claw/info
```

Expected embedded bridge settings:

```env
YA_CLAW_BRIDGE_DISPATCH_MODE=embedded
YA_CLAW_BRIDGE_ENABLED_ADAPTERS=github,lark
YA_CLAW_BRIDGE_GITHUB_TOKEN=replace-with-classic-pat
YA_CLAW_BRIDGE_GITHUB_ALLOWED_SENDERS=alice,bob
YA_CLAW_BRIDGE_LARK_APP_ID=cli_xxx
YA_CLAW_BRIDGE_LARK_APP_SECRET=replace-with-app-secret
```

Inspect service logs for `BridgeSupervisor` startup, adapter task creation, GitHub polling or Lark websocket connection messages, inbound event handling, dedupe results, conversation IDs, session IDs, and run IDs.

## GitHub Credential and Polling Checks

The service process needs an ordinary account classic PAT and a non-empty sender policy:

```env
YA_CLAW_BRIDGE_GITHUB_TOKEN=replace-with-classic-pat
YA_CLAW_BRIDGE_GITHUB_ALLOWED_SENDERS=alice,bob
```

Use `notifications` scope for the notification inbox and `repo` when private repositories must be inspected or modified. Fine-grained PATs are not supported by GitHub's Notifications REST endpoints. `*` accepts every sender; explicit login entries reject notifications whose actor cannot be reliably attributed.

The same token is injected into workspaces as `GH_TOKEN`. Check the workspace container:

```bash
docker exec -it ya-claw-session-<session-short>-g<generation> gh --version
docker exec -it ya-claw-session-<session-short>-g<generation> sh -lc 'test -n "$GH_TOKEN"'
docker exec -it ya-claw-session-<session-short>-g<generation> gh auth status
```

Confirm outbound HTTPS access to `YA_CLAW_BRIDGE_GITHUB_API_URL`. Logs should identify the authenticated account, tenant, sender policy, and effective polling interval. The adapter scans all notification reasons but routes only Issue and Pull Request subjects.

## Lark Credential Checks

The service process needs Lark bridge ingress credentials:

```env
YA_CLAW_BRIDGE_LARK_APP_ID=cli_xxx
YA_CLAW_BRIDGE_LARK_APP_SECRET=replace-with-app-secret
```

The workspace needs `lark-cli` reply credentials. YA Claw injects built-in `LARK_APP_ID` and `LARK_APP_SECRET` aliases into workspace environments from explicit process env values or from the Lark bridge settings. Additional process env values are forwarded by listing names in `YA_CLAW_WORKSPACE_ENV_VARS`.

For Docker shell shapes, these values are passed when the session workspace container is created. Check the workspace container:

```bash
docker exec -it ya-claw-session-<session-short>-g<generation> lark-cli --version
docker exec -it ya-claw-session-<session-short>-g<generation> sh -lc 'test -n "$LARK_APP_ID" && test -n "$LARK_APP_SECRET"'
```

After credential changes, recreate the active session workspace container so Docker receives the new environment:

```bash
docker rm -f ya-claw-session-<session-short>-g<generation>
rm -f /var/lib/ya-claw/data/docker-workspace-containers/sessions/<session-id>/workspace.json
```

## Event Subscription Checks

Confirm the Lark/Feishu app subscribes to the event types configured in YA Claw:

```env
YA_CLAW_BRIDGE_LARK_EVENT_TYPES=im.chat.member.bot.added_v1,im.chat.member.user.added_v1,im.message.receive_v1,drive.notice.comment_add_v1,card.action.trigger
```

Align the Lark app subscription list with the YA Claw allowlist so each intended event type reaches the adapter handler.

## Profile Checks

Bridge-created sessions require a valid profile:

```env
YA_CLAW_BRIDGE_GITHUB_DEFAULT_PROFILE=default
YA_CLAW_BRIDGE_LARK_DEFAULT_PROFILE=default
YA_CLAW_PROFILE_SEED_FILE=/etc/ya-claw/profiles.yaml
YA_CLAW_AUTO_SEED_PROFILES=true
```

Check startup logs for seeded profile names. Use the profiles command when seeding manually:

```bash
uv run --package ya-claw ya-claw profiles seed --seed-file /etc/ya-claw/profiles.yaml
```

## Event Dedupe Checks

Bridge dedupe uses these keys:

1. `(adapter, tenant_key, event_id)`
2. `(adapter, tenant_key, external_message_id)`

Repeated Lark delivery or an overlapping GitHub notification scan should reuse the existing bridge event result. GitHub uses notification thread ID plus `updated_at` as both event and message ID, so later versions of one thread remain distinct. Inspect logs and database rows for `duplicate`, `submitted`, `queued`, `steered`, `deferred`, and `failed` statuses.

## Conversation Checks

GitHub maps repository ID plus Issue/PR kind and number to one conversation. Lark message events map chat conversations by `(adapter, tenant_key, chat_id)`; Drive and generic events use stable fallback keys for payloads that carry Drive tokens or event IDs.

A new conversation creates one bridge conversation row and one YA Claw session. Later events with the same key create runs under the same session, steer an active run, or defer input when the active run is waiting on HITL. GitHub sessions omit a custom workspace binding and inherit the configured default workspace.

## Troubleshooting

### Bridge Supervisor Startup

Confirm embedded dispatch and enabled adapters:

```env
YA_CLAW_BRIDGE_DISPATCH_MODE=embedded
YA_CLAW_BRIDGE_ENABLED_ADAPTERS=github,lark
```

Restart the service and inspect logs for bridge lifecycle messages.

### GitHub Adapter Cannot Authenticate or Poll

Confirm `YA_CLAW_BRIDGE_GITHUB_TOKEN` is a classic PAT, not a fine-grained PAT, and that it has `notifications` or `repo` scope. Check outbound access to `YA_CLAW_BRIDGE_GITHUB_API_URL`. Bootstrap and poll failures are retried on the configured interval.

### GitHub Notification Does Not Create a Run

Confirm the notification subject is an Issue or Pull Request. With an explicit sender list, confirm the latest source has an attributable `user.login` that matches exactly after case folding. Subject-only assignment, state-change, or review-request notifications may be unattributable; use `*` only when accepting every sender is intended. Check `/api/v1/bridges/events?adapter=github` and service logs.

### GitHub Agent Action Fails

Run `gh auth status` inside the session workspace container. Confirm PAT repository access and scopes, then recreate the container if the token changed. The agent prompt tells the agent to inspect the current Issue/PR before acting because notifications may coalesce updates.

### Lark Adapter Fails on Startup

Set both Lark bridge credentials:

```env
YA_CLAW_BRIDGE_LARK_APP_ID=cli_xxx
YA_CLAW_BRIDGE_LARK_APP_SECRET=replace-with-app-secret
```

Confirm outbound network access from the YA Claw service to the configured Lark domain:

```env
YA_CLAW_BRIDGE_LARK_DOMAIN=https://open.feishu.cn
```

### Events Arrive but Runs Fail

Check profile configuration, model provider credentials, execution supervisor startup, and workspace provider health. Bridge ingestion creates the run; normal run execution handles model calls and tools.

### Agent Reply Fails

Check `lark-cli` availability and credentials in the workspace container or local workspace environment. Confirm the Lark app has reply permissions for the chat/message type and that the agent uses the message ID and idempotency key provided in the bridge prompt.

### HITL Card Does Not Appear

Confirm `card.action.trigger` is included in `YA_CLAW_BRIDGE_LARK_EVENT_TYPES` and subscribed in the Lark app. Check that the run notification has `session_status_reason=hitl_pending` and `active_interactions` in status detail. Shell review cards require profile shell review `on_needs_approval=defer`; generic tool/MCP approval cards require profile `need_user_approve_tools` or `need_user_approve_mcps` on an interactive run.

### HITL Card Button Does Not Resolve

Confirm Lark card action events reach the adapter and normalize to a `BridgeInboundAction`. Embedded bridge calls the shared controller directly; manual bridge workers should post normalized actions to `POST /api/v1/bridges/inbound/actions`. Successful responses publish `run.hitl.responded` and patch the existing card to the next interaction or completed state.

### Messages During HITL

Bridge messages received while a run is HITL pending create durable deferred input rows. After all interactions resolve, the coordinator consumes them in sequence order through the logical-run input inbox and native Pydantic AI enqueue path.

### Unattended Schedule and Heartbeat Runs

Schedule and heartbeat runs use unattended approval behavior. Shell review `defer` becomes `deny`, and generic tool/MCP approval lists are cleared for that run. Configure profile-level `unattended_risk_threshold` for agent-specific background behavior. Use reduced built-in toolsets and limited MCP access for background profiles.

### Manual Command Status

Manual bridge CLI commands are placeholders for separated worker flows. Use the bridge inbound HTTP endpoints for normalized external events in manual mode.

## References

- Bridge overview: [`overview.md`](overview.md)
- GitHub bridge: [`github.md`](github.md)
- Lark bridge: [`lark.md`](lark.md)
- General operations: [`../operations.md`](../operations.md)
