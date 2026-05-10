# Profiles

Profiles define reusable agent runtime behavior. They live in the database and can be seeded from YAML.

## Default Profile

`YA_CLAW_DEFAULT_PROFILE` defaults to `default`. Set it only when a deployment uses another profile name as the request fallback.

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
- tool approval policy
- shell command review policy
- MCP server definitions
- enabled and disabled MCP namespaces
- workspace backend hint

Important built-in toolsets:

- `session`: read-only current-session inspection tools
- `schedule`: agent-owned schedule management tools

## Shell Command Review

Shell command review is configured per profile under `security.shell_review` in the seed YAML or stored AgentProfile `model_config_override`.

```yaml
profiles:
- name: default
  model: gateway@openai-responses:gpt-5.5
  security:
    shell_review:
      enabled: true
      model: gateway@openai-responses:gpt-5.4-mini
      model_settings: openai_responses_low
      on_needs_approval: deny
      risk_threshold: extra_high
```

Supported `risk_threshold` values are `low`, `medium`, `high`, and `extra_high`. Commands with reviewer risk below the threshold execute directly. Commands at or above the threshold enter the configured action.

YA Claw uses `extra_high` as the profile shell review threshold default. Set `risk_threshold` explicitly when a deployment wants a stricter policy, for example `high` for remote code execution, broad destructive workspace changes, writes outside the workspace, sensitive file reads, sudo usage, or system-level changes.

`on_needs_approval` accepts `deny` and `defer` at the config boundary. YA Claw runtime runs agents in auto-pilot mode and coerces enabled `defer` shell review to `deny`, so threshold-triggering commands are blocked during unattended runs.

`model` is required when shell review is enabled. `model_settings` accepts SDK preset names such as `openai_responses_low` or an inline settings object.

The bundled `packages/ya-claw/profiles.yaml` keeps shell review disabled by default and includes the default review model and `risk_threshold: extra_high` fields as an operator-ready template.

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
