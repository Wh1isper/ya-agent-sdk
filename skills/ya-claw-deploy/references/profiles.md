Profiles define reusable agent runtime behavior. They live in the database and can be seeded from YAML.

## Default Profile

`YA_CLAW_DEFAULT_PROFILE` defaults to `default`. Set it when a deployment uses another profile name as the request fallback.

```env
YA_CLAW_DEFAULT_PROFILE=default
```

## Seed Profiles on Startup

Production baseline:

```env
YA_CLAW_PROFILE_SEED_FILE=/etc/ya-claw/profiles.yaml
YA_CLAW_AUTO_SEED_PROFILES=true
```

Seeded profiles use create/update semantics. Every startup refreshes matching database profiles from the YAML file, including subagent configuration. Database profiles absent from the YAML file remain available.

Manual seed:

```bash
ya-claw profiles seed --seed-file /etc/ya-claw/profiles.yaml
```

API seed:

```bash
curl -X POST \
  -H "Authorization: Bearer ${YA_CLAW_API_TOKEN}" \
  http://127.0.0.1:9042/api/v1/profiles/seed
```

## Profile Contents

Profiles can define:

- model
- system prompt
- model settings and config presets
- built-in tool groups
- subagents
- generic approval review policy
- tool and MCP approval policy
- MCP server definitions
- enabled and disabled MCP namespaces
- workspace backend hint

Important built-in toolsets:

- `session`: read-only current-session inspection tools
- `schedule`: agent-owned schedule management tools

## Subscription-backed Codex Profiles

YA Claw can use a ChatGPT/Codex subscription account through the OAuth model provider. Create the host credential store before starting or seeding the service:

```bash
uvx ya-oauth login codex
```

Then seed a profile with the OAuth Codex model string:

```yaml
profiles:
- name: codex-oauth
  model: oauth@codex:gpt-5.5
  model_settings: openai_responses_high
  model_cfg: gpt5_270k
```

The service process reads credentials from `~/.yaai/auth.json`. Docker deployments should mount a persistent host directory to the service user's `~/.yaai` so refresh tokens survive image upgrades and container replacement.

## Approval Review

Approval review is configured per profile under `security.approval_review` in seed YAML or stored AgentProfile `model_config_override`. The reviewer applies to protected tool permission boundaries including shell execution, workspace writes, destructive file operations, downloads, external mutations, and configured MCP tools.

```yaml
profiles:
- name: default
  model: gateway@openai-responses:gpt-5.5
  security:
    approval_review:
      enabled: true
      model: gateway@openai-responses:gpt-5.4-mini
      model_settings: openai_responses_low
      timeout_seconds: 30
      max_denials: 3
      include_recent_messages: 12
      mcp_permissions:
        github:
          default_decision: auto_review
          categories: [external_integration, network, write]
          scopes: [external_service]
```

`model` is required when approval review is enabled. `model_settings` accepts SDK preset names such as `openai_responses_low` or an inline settings object. `max_denials` controls how many reviewer denials can be surfaced before the protected call returns a closed denial.

Interactive API, stream, and bridge runs can enter HITL when an approval review denies a protected call or a user-approval policy defers a tool call. Schedule and heartbeat runs clear `need_user_approve_tools` and `need_user_approve_mcps` for that run; approval review remains profile-controlled.

## Tool and MCP Approval

Profiles can require HITL for specific tools or MCP servers with `need_user_approve_tools` and `need_user_approve_mcps`.

```yaml
profiles:
- name: lark-interactive
  model: gateway@openai-responses:gpt-5.5
  need_user_approve_tools:
    - file_write
  need_user_approve_mcps:
    - github
```

Interactive runs surface these approvals through the same HITL mechanism as approval review. Bridge-triggered Lark runs render one active approval card in the source chat and update it in place as each interaction resolves.

## Profile Patterns for Interactive and Background Runs

Use an interactive profile for bridge or API sessions that should ask users for approval:

```yaml
profiles:
- name: lark-interactive
  model: gateway@openai-responses:gpt-5.5
  security:
    approval_review:
      enabled: true
      model: gateway@openai-responses:gpt-5.4-mini
      model_settings: openai_responses_low
      timeout_seconds: 30
      max_denials: 3
  need_user_approve_tools:
    - file_write
```

Use a constrained profile for schedule and heartbeat jobs:

```yaml
profiles:
- name: scheduled-maintenance
  model: gateway@openai-responses:gpt-5.5
  builtin_toolsets: [filesystem, shell, session]
  security:
    approval_review:
      enabled: true
      model: gateway@openai-responses:gpt-5.4-mini
      model_settings: openai_responses_low
      timeout_seconds: 30
      max_denials: 1
```

Schedule and heartbeat share the same runtime assembly path as normal runs, with profile-level approval review and a narrowed user approval surface. Prefer dedicated profiles for background tasks so available toolsets, MCP namespaces, and reviewer model settings match automation risk.

## Test Run

```bash
curl -sS \
  -H "Authorization: Bearer ${YA_CLAW_API_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"profile_name":"default","input_parts":[{"type":"text","text":"Inspect this workspace and report the current directory."}]}' \
  http://127.0.0.1:9042/api/v1/sessions
```

Then inspect sessions:

```bash
curl -sS \
  -H "Authorization: Bearer ${YA_CLAW_API_TOKEN}" \
  http://127.0.0.1:9042/api/v1/sessions
```
