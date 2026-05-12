# Lark Bridge

The built-in `lark` bridge adapter accepts Lark/Feishu events through the Lark websocket client, normalizes accepted events, and submits bridge-triggered YA Claw runs. Agents reply or act from the workspace with `lark-cli`.

## Embedded Deployment

Use embedded dispatch for current Lark bridge deployments:

```env
YA_CLAW_BRIDGE_DISPATCH_MODE=embedded
YA_CLAW_BRIDGE_ENABLED_ADAPTERS=lark
YA_CLAW_BRIDGE_LARK_APP_ID=cli_xxx
YA_CLAW_BRIDGE_LARK_APP_SECRET=replace-with-app-secret
YA_CLAW_BRIDGE_LARK_DEFAULT_PROFILE=default
YA_CLAW_BRIDGE_LARK_EVENT_TYPES=im.chat.member.bot.added_v1,im.chat.member.user.added_v1,im.message.receive_v1,drive.notice.comment_add_v1,card.action.trigger
YA_CLAW_BRIDGE_LARK_REPLY_IDENTITY=bot
YA_CLAW_BRIDGE_LARK_DOMAIN=https://open.feishu.cn
```

`BridgeSupervisor` starts `LarkBridgeAdapter` with the HTTP server. The adapter requires `YA_CLAW_BRIDGE_LARK_APP_ID` and `YA_CLAW_BRIDGE_LARK_APP_SECRET`, creates a Lark websocket client with `auto_reconnect=True`, registers every event type in `YA_CLAW_BRIDGE_LARK_EVENT_TYPES`, and passes normalized events to `BridgeController`.

## Lark App Requirements

Configure the Lark/Feishu app for websocket event delivery and subscribe to the event types enabled in YA Claw. The default allowlist is:

- `im.chat.member.bot.added_v1`
- `im.chat.member.user.added_v1`
- `im.message.receive_v1`
- `drive.notice.comment_add_v1`
- `card.action.trigger`

Grant the app permissions needed for the selected event subscriptions and for replies/actions performed by `lark-cli` from the agent workspace. `card.action.trigger` is required for Lark approval card buttons.

## Conversation Mapping

Lark events become `BridgeInboundMessage` records:

| Event shape                  | Conversation key                                                                    |
| ---------------------------- | ----------------------------------------------------------------------------------- |
| `im.message.receive_v1`      | message `chat_id`                                                                   |
| Generic chat event           | `chat_id` or `open_chat_id` from the event payload                                  |
| Drive event                  | `drive/{file_token}` or another stable Drive token as the fallback conversation key |
| Other accepted generic event | `event/{event_type}/{event_id}`                                                     |

The database maps `(adapter, tenant_key, external_chat_id)` to a YA Claw session. `tenant_key` comes from the Lark event header and falls back to `default`.

## Event and Message Dedupe

YA Claw dedupes inbound bridge traffic before creating runs:

1. `(adapter, tenant_key, event_id)`
2. `(adapter, tenant_key, external_message_id)`

Duplicate events return the existing session/run identifiers when available. Failed event processing records the error on the bridge event row.

## Profile Selection

Lark-triggered conversations use this profile resolution:

1. `YA_CLAW_BRIDGE_LARK_DEFAULT_PROFILE` when set
2. `YA_CLAW_DEFAULT_PROFILE`
3. `default`

Seed the selected profile at startup for deploys that rely on bundled profile configuration:

```env
YA_CLAW_PROFILE_SEED_FILE=/etc/ya-claw/profiles.yaml
YA_CLAW_AUTO_SEED_PROFILES=true
```

## Workspace Reply Credentials

Agents reply from the workspace with `lark-cli`. YA Claw builds built-in workspace reply credential aliases from the service process environment in this order:

1. `LARK_APP_ID` and `LARK_APP_SECRET`
2. `YA_CLAW_BRIDGE_LARK_APP_ID` and `YA_CLAW_BRIDGE_LARK_APP_SECRET`

Use explicit `LARK_*` variables when the workspace should use a separate Lark identity:

```env
LARK_APP_ID=cli_xxx
LARK_APP_SECRET=replace-with-app-secret
```

Forward additional process env values into the workspace with `YA_CLAW_WORKSPACE_ENV_VARS`:

```env
MY_TOOL_API_KEY=replace-with-tool-key
YA_CLAW_WORKSPACE_ENV_VARS=MY_TOOL_API_KEY
```

For Docker shell shapes, `DefaultEnvironmentFactory` passes these values into `DockerEnvironmentFactory`; `ReusableSandboxEnvironment` then creates the workspace container with Docker SDK `containers.run(environment=...)`. The variables become container-level environment values available to `lark-cli`.

Session workspace containers keep the environment from container creation time. After changing Lark credentials, remove the active session container and its cache so YA Claw creates a container with the new environment:

```bash
docker rm -f ya-claw-session-<session-short>-g<generation>
rm -f /var/lib/ya-claw/data/docker-workspace-containers/sessions/<session-id>/workspace.json
```

The official Docker workspace image includes `lark-cli` and copies Lark-related skills into `/workspace/.agents/skills/` at container startup.

## HITL Approval Cards

When a bridge-triggered run enters HITL, the Lark adapter sends one interactive approval card to the same `chat_id` that produced the inbound event. Private chat events receive the card in the private chat; group chat events receive the card in the group. The adapter stores the Lark card message ID in `bridge_hitl_messages` and patches that same card as the current interaction advances.

Card behavior:

- one active approval card per run
- shell review renders command, risk, cwd, and reason fields
- generic tool/MCP approval renders tool name and arguments
- command and arguments use plain text blocks for reliable Lark rendering
- approve/deny buttons emit `card.action.trigger` with an interaction token
- when the batch completes, the card is patched to the completed state

Messages sent to the chat while an approval card is pending are stored in `hitl_deferred_inputs`. After the user approves or denies every pending interaction, YA Claw injects the queued messages back into the same running agent in receive order.

## Agent Reply Contract

The bridge-created run prompt includes the source message ID and an idempotency key. The recommended reply shape is:

```bash
lark-cli im +messages-reply \
  --message-id <message_id> \
  --as bot \
  --text '<reply>' \
  --idempotency-key bridge-lark-<event_id>
```

Set `YA_CLAW_BRIDGE_LARK_REPLY_IDENTITY=bot` for the default bot reply identity. Use app permissions and workspace credentials that match the reply identity and action surface.

## Manual Mode

`YA_CLAW_BRIDGE_DISPATCH_MODE=manual` starts the HTTP server and leaves `BridgeSupervisor` outside the server lifespan. External bridge workers can post normalized payloads to the service:

```bash
POST /api/v1/bridges/inbound/messages
POST /api/v1/bridges/inbound/actions
```

These endpoints route through the same controller used by embedded mode, so dedupe, conversation mapping, HITL approval responses, and deferred bridge input behavior stay consistent. Current CLI bridge commands remain operator-facing placeholders:

```bash
uv run --package ya-claw ya-claw bridge ls
uv run --package ya-claw ya-claw bridge run lark
uv run --package ya-claw ya-claw bridge serve lark
```

Embedded mode owns active Lark websocket ingestion inside the YA Claw service process.

## References

- Bridge overview: [`overview.md`](overview.md)
- Bridge operations: [`operations.md`](operations.md)
- Environment settings: [`../environment.md`](../environment.md)
