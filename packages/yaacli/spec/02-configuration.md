# Configuration

## Overview

YAACLI configuration is implemented by `yaacli.config.ConfigManager` and the
Pydantic models in `yaacli.config`. Configuration is split by responsibility:

- `config.toml` contains model, TUI, session, media, subagent, custom-command,
  process-environment, and security settings;
- `tools.toml` contains interactive tool availability, MCP exposure mode, and tool/MCP approval policy;
- `mcp.json` contains MCP server definitions;
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
| Subagents | `~/.yaacli/subagents/*.md` | Project subagents are supplied through CLI/project conventions | Loaded by the runtime |
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
agent_stream_resume_on_error = true
# Non-transport execution recovery attempts.
agent_stream_resume_max_attempts = 3
# Independent transient model HTTP/WebSocket recovery attempts.
agent_stream_transport_resume_max_attempts = 20
agent_stream_resume_prompt = "Continue from recovered history without repeating completed work."
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
auto_save_history = true
auto_restore = false
max_turns_per_session = 20
max_sessions = 100
# session_dir = "~/.yaacli/sessions"
# max_session_age_days = 90

[media]
max_pending_attachments = 8
max_pending_attachment_bytes = 20971520

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
`packages/yaacli/yaacli/templates/config.toml`. Custom commands always use the
same agent execution semantics; the former command-level ACT/PLAN mode is deprecated
and ignored.

### General and model profiles

`general.model` is required for normal runtime startup. `model_settings` and
`model_cfg` accept either a preset name or an inline mapping. `/model` selects
from `model_profiles`; the selected profile is persisted separately in
`~/.yaacli/state.json` and does not rewrite `config.toml`.

`general.max_loop_iterations` is accepted as a compatibility input only when
`max_goal_iterations` is absent. The normalized runtime field is
`max_goal_iterations`.

### Display retention

The display section independently bounds lines, blocks, UTF-8 bytes, raw stream
render bytes, and prompt-history entries. `code_theme` accepts `auto`, `dark`,
or `light`.

### Session retention

The session section controls automatic save/restore and durable retention:

| Field | Default | Meaning |
| --- | ---: | --- |
| `session_dir` | `~/.yaacli/sessions` | Optional saved-session directory override |
| `auto_save_history` | `true` | Save successful, cancelled, and failed recoverable turns from the interactive TUI |
| `auto_restore` | `false` | Restore the newest matching workspace session on TUI startup |
| `max_turns_per_session` | `20` | Maximum retained turn snapshots per session |
| `max_sessions` | `100` | Maximum retained sessions globally |
| `max_session_age_days` | unset | Optional age-based pruning threshold |

Positive limits are validated by Pydantic. `auto_save_history` controls only
interactive TUI persistence. Headless success is a durable protocol operation
and always saves a turn; headless failure or cancellation emits its terminal
event without saving a recovery snapshot. Session storage and restore contracts
are defined in `05-session-persistence.md`.

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
The former `mode` field is deprecated and ignored. User definitions are merged
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
YAACLI_AUTO_SAVE_HISTORY
YAACLI_AUTO_RESTORE
YAACLI_MAX_TURNS_PER_SESSION
YAACLI_MAX_SESSIONS
YAACLI_MAX_SESSION_AGE_DAYS
YAACLI_AGENT_STREAM_RESUME_ON_ERROR
YAACLI_AGENT_STREAM_RESUME_MAX_ATTEMPTS
YAACLI_AGENT_STREAM_RESUME_PROMPT
YAACLI_OAUTH_REFRESH_ENABLED
YAACLI_OAUTH_REFRESH_INTERVAL_SECONDS
YAACLI_OAUTH_REFRESH_FAILURE_RETRY_SECONDS
YAACLI_OAUTH_REFRESH_ON_STARTUP
```

Provider, OAuth, search, and SDK variables can also be placed in `.env`; they
are consumed by their owning packages rather than mapped into `YaacliConfig`.
The repository examples are `packages/yaacli/.env.example` and
`yaacli/templates/env.example`.

## Initialization and Operations

Normal `yaacli` startup performs the operational setup:

1. load `.env` files;
2. ensure global built-in assets exist;
3. load configuration;
4. if `general.model` is empty, run the interactive setup wizard;
5. create missing `config.toml`, `mcp.json`, subagent presets, and built-in
   skills without overwriting existing files;
6. reload configuration and apply `[env]` values.

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
`tests/test_model_profiles.py`, `tests/test_cli.py`, `tests/test_cli_headless.py`,
and `tests/test_sessions_cli.py`. The environment examples and packaged
configuration templates must remain aligned whenever an override or setting is
added.
