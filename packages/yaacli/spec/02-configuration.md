# Configuration

## Overview

YAACLI configuration is implemented by `yaacli.config.ConfigManager` and the
Pydantic models in `yaacli.config`. Configuration is split by responsibility:

- `config.toml` contains model, TUI, session, media, subagent, custom-command,
  process-environment, and security settings;
- `tools.toml` contains interactive tool availability, MCP exposure mode, and tool/MCP approval policy;
- `mcp.json` contains MCP server definitions;
- global `plugins.toml` explicitly selects installed capability entry points and grants
  configured instances to the root agent;
- `.env` files provide supported `YAACLI_*` overrides and provider or SDK
  environment variables;
- `state.json` stores local UI selection state such as the last model profile.

The packaged templates under `yaacli/templates/` are the canonical examples.
There is no configuration migration service or `yaacli config set/reset`
command surface. When no model is configured, normal CLI startup runs the setup
wizard and creates missing global assets without overwriting existing ones.

## Locations and Precedence

| Artifact | Global location | Project location | Selection rule |
| --- | --- | --- | --- |
| Main configuration | `~/.yaacli/config.toml` | `.yaacli/config.toml` | Project file replaces the global file as a whole |
| Tool policy | `~/.yaacli/tools.toml` | `.yaacli/tools.toml` | Project file replaces the global file as a whole |
| MCP configuration | `~/.yaacli/mcp.json` | `.yaacli/mcp.json` | Project file replaces the global file as a whole |
| Capability plugins | `~/.yaacli/plugins.toml` | None | Optional fixed global manifest loaded once at startup |
| Subagents | `~/.yaacli/subagents/*.{md,yaml,yml,json}` | None | Generic Markdown definitions or strict versioned `SubagentSpec` documents |
| Skills | `~/.yaacli/skills/` | `.yaacli/skills/` | Project skills have higher routing priority |
| Local state | `~/.yaacli/state.json` | None | Stores selected model-profile state |
| Saved sessions | `~/.yaacli/sessions/` by default | Configurable | Controlled by `session.session_dir` |

`ConfigManager.load()` applies these steps:

1. load project `config.toml` when present, otherwise global `config.toml`;
2. deep-merge the supported `YAACLI_*` environment overrides;
3. load project `tools.toml` when present, otherwise global `tools.toml`, and
   replace the `tools` section;
4. validate the result as `YaacliConfig`.

Global and project files at the same layer are not merged. Environment
overrides intentionally cover only explicitly supported TUI/runtime fields and
do not replace model configuration.

## `config.toml`

A compact current example is:

```toml
# Top-level shell isolation settings.
shell_env = { EXAMPLE_RUNTIME_VALUE = "value" }
include_os_env = true

[general]
model = "anthropic:claude-sonnet-4-5"
model_settings = "anthropic_adaptive_high"
model_cfg = "claude_200k"
max_requests = 1000
max_goal_iterations = 10
# system_prompt_file = "~/.yaacli/system_prompt.md"

[model_profiles.fast]
label = "Fast"
model = "openai-responses:gpt-5.6-luna"
model_settings = "openai_responses_luna"
model_cfg = "gpt5_270k"

[display]
code_theme = "auto"
max_tool_result_lines = 5
max_arg_length = 100
max_output_lines = 1000
max_output_blocks = 1000
max_output_bytes = 4194304
max_stream_render_bytes = 524288
max_prompt_history = 500
show_token_usage = true
show_elapsed_time = true

[session]
auto_restore = false
max_turns_per_session = 20
max_sessions = 100
# max_session_age_days = 90
# session_dir = "~/.yaacli/sessions"
# database_path = "~/.yaacli/sessions/sessions-v2.sqlite3"

[media]
max_pending_attachments = 8
max_pending_attachment_bytes = 20971520

[media.s3]
enabled = false
bucket = ""
region = "us-east-1"
url_mode = "presign"
presign_expires_seconds = 3600
force_path_style = false

[notifications]
bell_on_turn_complete = true
bell_on_user_action_required = true

[oauth_refresh]
enabled = true
interval_seconds = 1800
failure_retry_seconds = 60
refresh_on_startup = true

[subagents]
disabled = []

[subagents.overrides.explorer]
model = "openai-responses:gpt-5.6-luna"
model_settings = "openai_responses_luna"

[security.shell_review]
enabled = false
# model = "gateway@openai-responses:gpt-5.4-mini"
# model_settings = "openai_responses_low"
# on_needs_approval = "defer"
# risk_threshold = "high"

[env]
# OPENAI_API_KEY = "sk-..."

[commands.review]
description = "Review the current changes"
prompt = "Please perform a comprehensive review of the current changes."
```

The exact complete template is
`packages/yaacli/yaacli/templates/config.toml`. User-authored configuration is strict:
unknown, misspelled, and removed fields are rejected. Custom commands always use the
same agent execution semantics and contain only `description` plus `prompt`.

### General and model profiles

`general.model` is required for normal runtime startup. `model_settings` and
`model_cfg` accept either a preset name or an inline mapping. `/model` selects
from `model_profiles`; the selected profile is persisted separately in
`~/.yaacli/state.json` and does not rewrite `config.toml`.

Both `[general]` and `[model_profiles.*]` may define static `instructions` for the
active main-agent profile. YAACLI registers them through a dynamic Pydantic AI
`instructions` callback, so the active profile is evaluated for every later model
request, including restored or compacted histories. A `/model` switch therefore takes
effect without rebuilding prior history.

`general.max_requests` bounds cumulative model requests within one durable logical run,
including every native deferred-tool continuation segment. A later user turn starts a
new logical run and receives a fresh budget.

Removed `general.max_loop_iterations` input is rejected. `/goal` uses only the strict
`max_goal_iterations` field.

### Subagent documents

YAACLI accepts two configuration inputs under `~/.yaacli/subagents/`. Both compile to
the same portable `SubagentSpec` and current capability-first child runtime.

A native YAML or JSON document exposes the complete strict contract:

```yaml
schema_version: 1
route: explorer
agent:
  name: explorer
  description: Inspect an unfamiliar codebase.
  model: anthropic:claude-sonnet-4-5
  capabilities:
    - FilesystemCapability
    - ShellCapability
history: isolated
execution_modes: [foreground, background]
linkage: child
durability: process
```

The generic Markdown format keeps YAML frontmatter concise and uses the Markdown body
for child instructions:

```markdown
---
name: explorer
description: Inspect an unfamiliar codebase.
instruction: Use this agent for focused local codebase exploration.
model: inherit
model_settings: inherit
model_cfg: inherit
tools: [glob, grep, ls, view]
optional_tools: [shell_exec]
---

You are a codebase exploration specialist. Return concise findings with file paths.
```

`name` and `description` are required. `instruction`, `tools`, `optional_tools`,
`model`, `model_settings`, and `model_cfg` are optional. `tools` and `optional_tools`
accept either YAML lists or comma-separated strings. The loader combines `description`
and optional `instruction` into the parent-facing native description, while the body
becomes `AgentSpec.instructions`. `inherit` or an omitted model field uses the active
root default. A model-settings or model-config mapping/preset is resolved at the same
trusted configuration boundary used by native overrides. Every inherited value is
materialized before child plan fingerprinting, so retained descriptors never depend on
the active profile at restore time.

Markdown is an input adapter, not the removed 1.x execution architecture. For the
common inherited-tool behavior, YAACLI supplies its current standard safe child
capability template at the trusted normalization boundary. The template follows
configuration-dependent built-ins such as CodeAct, while excluding ambient MCP,
third-party root grants, delegation, host-only tools, and live capability instances.
The adapter materializes that template into the child's `AgentSpec` before resolution
and fingerprinting; there is no runtime tool inheritance.

A `tools`/`optional_tools` list adds a final `ToolVisibilityCapability` allowlist over
the materialized template. Both fields contribute to that allowlist; they neither
create missing tools nor gate route registration. Native documents remain the format
for exact capability grants, child policy, custom plugin types, nesting, or host
requirements.

When `name.md` and `name.yaml`/`name.yml`/`name.json` coexist, Markdown is the complete
authoritative definition and the same-basename native document is not parsed or merged.
This makes stale native presets copied during an earlier upgrade inert instead of
letting their historical capability snapshot override current generic semantics. Other
duplicate routes remain a configuration error. Future setup runs do not copy a native
preset when any supported same-basename definition already exists. To use exact native
policy for a route, keep only its YAML/JSON definition.

YAACLI child execution is process-local, so native configured specs must use `process`.
A `restart` requirement is rejected rather than silently weakened. The
`[subagents.overrides.<route>]` table may replace `model`, `model_settings`, or
`model_cfg` after either source format is normalized. Project configuration does not
inject a second subagent directory or silently merge child specifications.

### Capability plugins

A plugin distribution must be installed into the same Python environment as YAACLI. For
an isolated uv tool installation:

```bash
uv tool install 'yaacli[rs]' --with acme-agent-plugin
```

The optional fixed global manifest uses the SDK schema directly:

```toml
schema_version = 1
entry_points = ["acme.search"]

[[capabilities]]
name = "acme.search"
arguments = { result_limit = 10 }
```

`entry_points` is an ordered, unique exact-name selection. It controls which installed
entry-point targets may be imported and added to YAACLI's immutable capability catalog.
`capabilities` is an ordered root-only grant list; each grant must reference a selected
name. A selected but ungranted type may be declared in a native child `AgentSpec`.
Installation alone does not select or grant anything.

YAACLI does not create `plugins.toml`, consider a project-local file, auto-install
packages, or scan all installed entry points. A missing global file produces the empty
SDK catalog snapshot. If the file exists, invalid TOML, unsupported schema, unknown
fields, duplicate or missing entry points, import failures, and catalog collisions are
fatal. Arguments must be JSON-compatible finite non-secret configuration; secret-like
keys are rejected recursively. Live authority and credentials belong in typed runtime
dependencies or host APIs, not durable specs.

The TUI and headless frontend each load the file exactly once at bootstrap. The same
snapshot is captured by profile, named-child, self-fork, retained-plan, historical, and
restored runtime factories. Manifest grants are applied only to the main root agent;
named children and self forks receive only their own explicit native grants. Restart
YAACLI after changing the file or installed distribution.

### Display retention

The display section independently bounds lines, blocks, UTF-8 bytes, raw stream
render bytes, and prompt-history entries. `code_theme` accepts `auto`, `dark`,
or `light`.

### Media upload and terminal notifications

`media.s3.enabled` activates S3 media upload. `bucket` is then required by runtime
construction. `region` defaults to `us-east-1`; omitted credentials use the AWS default
credential chain. `endpoint_url` and `force_path_style` support compatible stores.
`prefix` scopes object keys. `url_mode="presign"` uses
`presign_expires_seconds`; `url_mode="cdn"` requires `cdn_base_url`.

`notifications.bell_on_turn_complete` rings after a successful interactive turn.
`notifications.bell_on_user_action_required` rings when an interactive turn enters a
user-action boundary. Both default to true and affect only terminal presentation.

### Durable session storage

The session section controls durable database placement and startup restore:

| Field | Default | Meaning |
| --- | ---: | --- |
| `session_dir` | `~/.yaacli/sessions` | Parent directory for the default product database |
| `database_path` | `<session_dir>/sessions-v2.sqlite3` | Optional product-store override |
| `auto_restore` | `false` | Restore the newest matching workspace session on TUI startup |
| `max_turns_per_session` | `20` | Retain the newest complete run bundles in each session |
| `max_sessions` | `100` | Retain at most this many active, quiescent sessions |
| `max_session_age_days` | unset | Optionally tombstone quiescent sessions older than this many days |

Retention is safety-first. Nonterminal main or child work is never selected for automatic
session tombstoning. Manual tombstoning refuses a nonterminal main run and atomically
records nonterminal children as cancelled. A maintenance pass physically purges only previously tombstoned,
now-quiescent sessions, then tombstones newly selected sessions; this two-pass boundary
preserves the write fence and terminal-state check before physical deletion.

The `sessions-v2.sqlite3` name denotes the YAACLI 2 product-store generation, not the
internal schema marker. SQLite stores metadata and small coordination records; revision,
checkpoint, and child state files are placed in per-session directories next to the
database. The known schema-v5 YAACLI database is intentionally reset at the same path
for this cutover, with no payload migration or v3 database. YAACLI does not migrate or
open the former default `sessions.sqlite3`; that file remains untouched. An explicit
`database_path` or `YAACLI_DATABASE_PATH` is authoritative. Unknown incompatible or
unmarked schemas remain strict errors.

Every successful, failed, or cancelled logical run commits a terminal durable revision.
The removed file-snapshot save and pruning switches do not control this invariant.
Session storage and restore contracts are defined in `05-session-persistence.md`.

### Process and shell environments

`[env]` values are loaded into the YAACLI process only when the variable is not
already present in `os.environ`. They are intended for provider credentials and
SDK settings.

`[shell_env]` is passed to agent shell execution. `include_os_env = true`
includes the parent process environment as the base layer; setting it to
`false` limits shell execution to explicitly configured shell values and
per-call overrides. These two sections are separate so deployments can avoid
leaking provider credentials to subprocesses.

### Shell review

When `security.shell_review.enabled = true`, `model` must be a non-empty model
string. Validation fails otherwise. Review policy is runtime security
configuration and is separate from the static `tools.toml` approval list.

### Custom slash commands

Each `[commands.<name>]` entry requires `prompt` and may define `description`.
User definitions are merged
with the built-in `init` command; a user definition with the same name overrides
the built-in entry. Custom commands start an agent turn and are therefore
idle-only; submitting one while foreground work is active preserves the draft
and reports that the command is unavailable instead of converting its `/name`
text into steering.

## `tools.toml`

Tool policy is isolated from the main configuration so a project can override
permissions without copying model credentials or global UI settings:

```toml
[tools]
enable_codeact = true
enable_user_input = true
user_input_timeout_seconds = 120
mcp_mode = "direct"
need_approval = ["shell_exec", "write"]
need_approval_mcps = ["production-filesystem"]
```

`enable_codeact` defaults to `true` and exposes the restricted `run_code` and
`run_program` tools. `run_program` reads source through the active Environment's
`FileOperator`. Shell tools remain an independent execution surface and are not
added to the CodeAct callable catalog. Setting `enable_codeact = false` removes
both CodeAct tools.

`enable_user_input` defaults to `true` and controls whether the interactive TUI
registers the deferred `ask_user_question` tool. Setting it to `false` removes
the tool. `user_input_timeout_seconds` must be positive and finite, and defaults
to 120 seconds. It applies separately to each structured question; when a question is
left unanswered, YAACLI rejects the whole deferred call with a
`RetryPromptPart` explaining that the user did not respond and directing the
agent to continue using its best judgment without requesting the same input
again, then continues the same agent turn. Headless mode leaves the tool disabled regardless because it cannot
collect interactive answers.

`mcp_mode` accepts `"direct"` or `"proxy"` and defaults to `"direct"`. Direct
mode registers each configured MCP server as a native toolset and exposes its
tools as `<server>_<tool>` by default, avoiding cross-server name collisions
without an intermediate proxy call. Each server may set `prefix` in `mcp.json`
to replace `<server>` with a custom value, or set `prefix` to `""` to expose
native tool names without a prefix. An omitted or `null` `prefix` preserves the
server-name default. Proxy mode wraps all MCP servers in a fixed
`mcp_search_tool`/`mcp_call_tool` pair, reducing the stable model-facing tool
surface when many MCP tools are configured. In both modes, a server with
`"required": false` is skipped with a warning when it cannot initialize or list
tools.

Project `tools.toml` replaces the global file as a whole rather than merging
individual fields. Consequently, a project file that omits `enable_codeact` or
`enable_user_input` uses the corresponding schema default `true`, even when the
global file sets it to `false`. Projects that must disable CodeAct or interactive
clarification need to repeat the relevant setting in their own file.

`need_approval` contains tool names. `need_approval_mcps` contains MCP server
names whose tools require approval. An empty list means no additional static
approval requirement from that field.

## `mcp.json`

MCP servers use the SDK JSON schema:

```json
{
  "servers": {
    "filesystem": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"],
      "env": {},
      "prefix": "files"
    },
    "local": {
      "transport": "stdio",
      "command": "local-mcp-server",
      "prefix": ""
    }
  }
}
```

`ConfigManager.load_mcp_config()` checks project `.yaacli/mcp.json` first and
falls back to `~/.yaacli/mcp.json`. MCP files are not merged.

## Environment Overrides

YAACLI loads `.env` from the package development location and then the current
working directory without overriding variables already present in the process.
`EnvSettings` uses the `YAACLI_` prefix and supports:

```text
YAACLI_CODE_THEME
YAACLI_SHOW_TOKEN_USAGE
YAACLI_SHOW_ELAPSED_TIME
YAACLI_SESSION_DIR
YAACLI_DATABASE_PATH
YAACLI_AUTO_RESTORE
YAACLI_OAUTH_REFRESH_ENABLED
YAACLI_OAUTH_REFRESH_INTERVAL_SECONDS
YAACLI_OAUTH_REFRESH_FAILURE_RETRY_SECONDS
YAACLI_OAUTH_REFRESH_ON_STARTUP
```

Provider, OAuth, search, and SDK variables can also be placed in `.env`; they
are consumed by their owning packages rather than mapped into `YaacliConfig`.
`YA_AGENT_TOOL_TIMEOUT_SECONDS` sets the generic tool-execution ceiling and defaults
to 600 seconds; tool-owned shorter deadlines still apply. The repository examples are
`packages/yaacli/.env.example` and `yaacli/templates/env.example`.

## Initialization and Operations

Normal `yaacli` startup performs the operational setup:

1. load `.env` files;
2. ensure global built-in assets exist;
3. load configuration;
4. if `general.model` is empty, run the interactive setup wizard;
5. create missing `config.toml`, `mcp.json`, native YAML subagent presets, and
   built-in skills without overwriting existing files;
6. reload configuration and apply `[env]` values; and
7. load the optional global capability plugin manifest once before compiling runtime
   profiles and plans.

Configuration is edited through the TOML/JSON files. The implemented top-level
CLI exposes runtime options and saved-session commands:

```bash
yaacli --help
yaacli --session <session-id>
yaacli --profile <profile-id>
yaacli -p "prompt" --worker
yaacli sessions list
yaacli sessions show <session-id>
yaacli sessions delete <session-id>
```

`--profile` is a run-scoped override and does not update `state.json`. Selecting a profile through the interactive `/model` UI persists that choice for later launches.

There are no `yaacli config show/set/reset/validate` commands in the current
CLI.

## Verification

Configuration behavior is covered by `tests/test_config.py`,
`tests/test_model_profiles.py`, `tests/test_cli_setup.py`,
`tests/test_config_persistence_policy.py`, `tests/test_cli_headless.py`, and
`tests/test_sessions_cli.py`. The environment examples and packaged
configuration templates must remain aligned whenever an override or setting is
added.
