# Profiles

YA Claw profiles are strict version 2 documents stored in the database and optionally
seeded from YAML. Each profile has three boundaries:

- native Pydantic AI `AgentSpec` behavior;
- Claw-owned host policy; and
- native SDK `SubagentSpec` children.

The runtime and HTTP API accept only this schema. Alembic performs the one-time upgrade
of persisted 1.x rows.

## Default Profile

`YA_CLAW_DEFAULT_PROFILE` defaults to `default`. Set it only when a deployment uses
another profile name as the request fallback.

```env
YA_CLAW_DEFAULT_PROFILE=default
```

## Seed Profiles on Startup

Production baseline:

```env
YA_CLAW_PROFILE_SEED_FILE=/etc/ya-claw/profiles.yaml
YA_CLAW_AUTO_SEED_PROFILES=true
```

Seeded profiles use create/update semantics. Startup refreshes matching rows from the
YAML file. Rows absent from YAML remain available unless an operator explicitly requests
pruning.

Manual seed:

```bash
ya-claw profiles seed --seed-file /etc/ya-claw/profiles.yaml
```

Explicit destructive pruning:

```bash
ya-claw profiles seed --seed-file /etc/ya-claw/profiles.yaml --prune-missing
```

API seed:

```bash
curl -X POST \
  -H "Authorization: Bearer ${YA_CLAW_API_TOKEN}" \
  http://127.0.0.1:9042/api/v1/profiles/seed
```

## Native Profile Schema

```yaml
version: 2
profiles:
  - schema_version: 2
    name: default
    agent:
      model: gateway@openai-responses:gpt-5.5
      name: default
      instructions: You are a workspace-bound execution agent.
      model_settings:
        openai_reasoning_effort: high
      capabilities:
        - FilesystemCapability
        - ShellCapability
        - WebSearchCapability
        - WebContentCapability
    host:
      model_config_preset: gpt5_270k
      model_config_override:
        security:
          shell_review:
            enabled: false
          shell_sandbox:
            enabled: true
            profile: workspace_write
            backend_preference: auto
            network: full
            env_allowlist: ["*"]
            raw_shell_approval: requires_human
            audit_enabled: true
      tool_groups: [session, schedule, workflow, agency]
      need_user_approve_tools: []
      need_user_approve_mcps: []
      enabled_mcps: []
      disabled_mcps: []
      mcp_servers: {}
      workspace_backend_hint: docker
    subagents:
      - schema_version: 1
        route: explorer
        execution_modes: [foreground, background]
        durability: restart
        agent:
          model: gateway@anthropic:claude-sonnet-4-6
          name: explorer
          description: Inspect the bound workspace.
          instructions: Return evidence with file paths and line numbers.
          capabilities:
            - FilesystemCapability
          metadata:
            claw:
              tool_groups: [session]
              need_user_approve_tools: []
              need_user_approve_mcps: []
              enabled_mcps: []
              disabled_mcps: []
              mcp_servers: {}
    enabled: true
    source_type: seed
    source_version: "2"
```

Rules:

- `agent.name`, when present, equals the profile name.
- Every child `agent.name` equals its `route`.
- Native behavior grants live in `agent.capabilities`.
- External serialization names are valid only when selected by the process plugin
  manifest; profiles cannot import Python targets.
- Claw host groups are limited to `session`, `schedule`, `workflow`, and `agency`.
- Filesystem, shell, media, web, document conversion, code execution, skills, and
  delegation are capabilities, not host groups.
- Child behavior is explicit. Root grants from the process plugin manifest are not
  inherited; a child must declare a selected external name in its own capabilities.
  There is no parent inheritance compiler, Markdown front matter, required/optional
  tool list, or automatic capability copy.
- Unknown fields are rejected.

Plugin selection and package installation are deployment configuration, not profile
fields. See [`plugins.md`](plugins.md). A profile descriptor persists only its native
capability grants; YA Claw uses the same startup catalog snapshot for admission,
children, and restoration.

## Subscription-backed Codex Profiles

Create the host credential store before starting or seeding the service:

```bash
uvx ya-oauth login codex
```

Then use the OAuth model string in a native profile:

```yaml
version: 2
profiles:
  - schema_version: 2
    name: codex-oauth
    agent:
      model: oauth@codex:gpt-5.5
      name: codex-oauth
      model_settings:
        openai_reasoning_effort: high
      capabilities: [FilesystemCapability, ShellCapability]
    host:
      model_config_preset: gpt5_270k
      tool_groups: [session]
    subagents: []
```

The service process reads credentials from `~/.yaai/auth.json`. Docker deployments
should mount a persistent host directory to the service user's `~/.yaai` so refresh
tokens survive image upgrades and container replacement.

## Shell Command Review

Shell review is Claw execution policy under
`host.model_config_override.security.shell_review`:

```yaml
host:
  model_config_override:
    security:
      shell_review:
        enabled: true
        model: gateway@openai-responses:gpt-5.4-mini
        model_settings: openai_responses_low
        on_needs_approval: defer
        risk_threshold: high
        unattended_risk_threshold: extra_high
```

Supported risk values are `low`, `medium`, `high`, and `extra_high`. Commands below the
threshold execute directly. Commands at or above it use `on_needs_approval`, which is
`defer` or `deny`.

Interactive API, stream, and bridge runs can defer to HITL. Schedule, heartbeat,
workflow, and agency runs are unattended: deferred review becomes denial. The review
`model` is required when review is enabled.

Unattended threshold precedence is:

1. profile `unattended_risk_threshold`;
2. `YA_CLAW_UNATTENDED_SHELL_REVIEW_RISK_THRESHOLD`; and
3. profile `risk_threshold`.

## Shell Sandbox Policy

Sandbox policy is adjacent host runtime policy:

```yaml
host:
  model_config_override:
    security:
      shell_sandbox:
        enabled: true
        profile: workspace_write
        backend_preference: auto
        network: full
        env_allowlist: ["*"]
        masked_path_aliases: []
        masked_paths: []
        raw_shell_approval: requires_human
        audit_enabled: true
```

Supported profiles are `read_only`, `workspace_write`,
`mounted_workspace_write`, `network_proxy`, and `danger_full_access`. Supported backend
preferences are `auto`, `linux_bwrap_seccomp`, `macos_seatbelt`,
`windows_restricted_token`, `docker`, `podman`, `nsjail`, and `raw_host`.

Network policy is `blocked`, `restricted`, `proxy`, or `full`. The environment allowlist
controls variables copied into sandboxed subprocesses. Path masks are opt-in; the
`common_credentials` alias masks the recommended SSH, GnuPG, cloud, Docker, and
Kubernetes credential directories.

Raw host approval is `forbidden`, `requires_human`, or `allowed_for_profile`. Raw host
execution also requires service-level allowance through
`YA_CLAW_SHELL_SANDBOX_ALLOW_RAW_HOST=true` or explicit profile allowance.

## Tool and MCP Approval

Host approval and MCP selection stay outside the native agent definition:

```yaml
host:
  need_user_approve_tools: [write]
  need_user_approve_mcps: [github]
  enabled_mcps: [github]
  disabled_mcps: []
  mcp_servers: {}
```

Interactive runs surface these approvals through the same HITL mechanism as shell
review. Unattended runs clear interactive approval requirements. Use dedicated profiles
to narrow unattended capabilities and MCP namespaces.

## API Update Shape

The profile name is the URL identity and is not duplicated at the request root:

```bash
curl -X PUT \
  -H "Authorization: Bearer ${YA_CLAW_API_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "schema_version": 2,
    "agent": {
      "model": "gateway@openai-responses:gpt-5.5",
      "name": "minimal",
      "capabilities": ["FilesystemCapability"]
    },
    "host": {"tool_groups": ["session"]},
    "subagents": [],
    "enabled": true
  }' \
  http://127.0.0.1:9042/api/v1/profiles/minimal
```

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
