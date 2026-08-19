---
name: cli-config
description: Guide for configuring YAACLI CLI. Use this skill when users want to configure models, tools, subagents, custom commands, or other CLI settings. Covers both global and project-level configuration.
---

# YAACLI CLI Configuration

Configuration is loaded from multiple locations with project-level priority (no merging between levels).

## Configuration Locations

| Level | Location | Priority |
|-------|----------|----------|
| Global | `~/.yaacli/` | Default |
| Project | `.yaacli/` | Overrides global |

## Configuration Files

### config.toml (Global)

Main configuration file for model, display, and subagents.

```toml
[general]
# Model configuration (required)
# Format: "provider:model_name"
model = "anthropic:claude-sonnet-4-5"

# Model settings preset or custom dict
# Presets: anthropic, anthropic_adaptive_high, anthropic_adaptive_xhigh, openai_default, gemini_thinking_level_default
# `anthropic` resolves to adaptive thinking by default. `anthropic_adaptive_xhigh` is intended for Claude Opus 4.7.
model_settings = "anthropic"

# Model config for context management
# Presets: claude_200k, claude_1m, gpt5_270k, gpt5_1m, gemini_1m
model_cfg = "claude_200k"

# Optional static instructions for the default model profile
instructions = """
Prefer concise, evidence-based answers.
"""

# Cumulative model-request limit for one durable logical run
max_requests = 1000

[model_profiles.fast]
label = "Fast"
model = "openai-responses:gpt-5-mini"
model_settings = "openai_responses_low"
model_cfg = "gpt5_270k"
instructions = """
Optimize for speed. Avoid broad exploration unless necessary.
"""

[env]
# Environment variable overrides for API keys
# ANTHROPIC_API_KEY = "sk-ant-..."

[media]
max_pending_attachments = 8
max_pending_attachment_bytes = 20971520

[display]
code_theme = "auto"           # "auto", "dark", or "light"
max_tool_result_lines = 5
max_arg_length = 100
max_output_lines = 1000
max_output_blocks = 1000
max_output_bytes = 4194304
max_stream_render_bytes = 524288
max_prompt_history = 500
show_token_usage = true
show_elapsed_time = true

[subagents]
disabled = []                 # Subagents to disable by name
# [subagents.overrides.explorer]
# model = "openai-chat:gpt-4o"
```

### tools.toml (Project)

Project-level tool permissions in `.yaacli/tools.toml`:

```toml
[tools]
# Enable restricted run_code and run_program orchestration (default: true)
enable_codeact = true

# Enable structured clarifying questions in the interactive TUI (default: true)
enable_user_input = true
# Seconds to wait for each structured answer before rejecting the call (default: 120)
user_input_timeout_seconds = 120

# MCP tool exposure: "direct" (default) or "proxy"
mcp_mode = "direct"

# Tools requiring user approval before execution
need_approval = ["shell_exec", "write"]
```

Set `enable_codeact = false` to remove both `run_code` and `run_program`. Shell remains a separate execution surface and is not callable from CodeAct.

Set `enable_user_input = false` when the project should not expose the interactive `ask_user_question` tool. `user_input_timeout_seconds` must be positive and finite; an unanswered question rejects the deferred call after this interval so the agent can continue. Headless mode never exposes it.

Use `mcp_mode = "direct"` to expose each MCP server's native tools as namespaced `<server>_<tool>` names by default. In `mcp.json`, set a server's `prefix` to a custom string to replace `<server>`, or set `prefix` to `""` to expose native tool names without a prefix. Omit `prefix` or set it to `null` to preserve the server-name default. Use `mcp_mode = "proxy"` to expose only `mcp_search_tool` and `mcp_call_tool`, which is useful when many MCP tools would otherwise enlarge or destabilize the model-facing tool list. Servers marked `"required": false` remain optional in both modes.

Common patterns:
- `[]` - No additional static approval requirement
- `["shell_exec"]` - Approve foreground shell commands
- `["shell_exec", "write", "edit"]` - Approve shell and code modifications

### mcp.json

MCP server configurations:

```json
{
  "servers": {
    "my-server": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@some/mcp-server"],
      "env": {},
      "prefix": "custom"
    }
  }
}
```

## Custom Slash Commands

Define custom commands in `config.toml`:

```toml
[commands.deploy]
description = "Deploy to production"
prompt = """
Please help me deploy to production...
"""
```

Every custom command uses the same agent execution semantics. `/init` is provided by
default, while commands such as `/commit` and `/review` come from configuration.

`instructions` adds a static model-instruction segment only while its profile is active. It applies to every main-agent model request in TUI and headless runs, including restored and compacted histories; `/model` changes it for later requests in the current session.

## Subagent Configuration

Create strict versioned YAML or JSON `SubagentSpec` files in
`~/.yaacli/subagents/`:

```yaml
schema_version: 1
route: my-subagent
agent:
  name: my-subagent
  description: Brief description shown in the delegation roster.
  model: anthropic:claude-sonnet-4-5
  instructions: |
    You are a specialist in this domain. Return a bounded, evidence-based result.
  capabilities:
    - FilesystemCapability
    - ShellCapability
history: isolated
history_message_limit: 100
max_depth: 1
spawn_targets: []
execution_modes: [foreground, background]
linkage: child
durability: process
```

YAACLI uses a process-local child driver, so `durability: restart` is rejected rather
than silently weakened. An omitted `agent.model` uses the runtime's explicit
default-model resolver. Do not use
`inherit`. Child capabilities are explicit serialization names; YAACLI does not derive
them from tool names or copy the parent's final tool surface.

The `[subagents.overrides.<route>]` table may replace `model`, `model_settings`, or
`model_cfg`. `disabled` removes routes after loading. Markdown front matter, `tools`,
`optional_tools`, generated delegate definitions, and duplicate routes are rejected.

### Builtin Presets

| Preset | Purpose |
|--------|---------|
| `explorer` | Codebase navigation and evidence gathering |
| `executor` | Independent bounded task execution |
| `code-reviewer` | Architecture, correctness, security, and maintainability review |

## Skills Directory

Skills are loaded from (highest priority last):

1. Built-in: `yaacli/skills/` (package bundled)
2. Global: `~/.yaacli/skills/`
3. Project: `.yaacli/skills/`

## Environment Variables

TUI and durable-session settings can be overridden via `YAACLI_*` environment variables:

- `YAACLI_CODE_THEME`
- `YAACLI_SHOW_TOKEN_USAGE`
- `YAACLI_SHOW_ELAPSED_TIME`
- `YAACLI_SESSION_DIR`
- `YAACLI_DATABASE_PATH`
- `YAACLI_AUTO_RESTORE`
- `YAACLI_OAUTH_REFRESH_ENABLED`
- `YAACLI_OAUTH_REFRESH_INTERVAL_SECONDS`
- `YAACLI_OAUTH_REFRESH_FAILURE_RETRY_SECONDS`
- `YAACLI_OAUTH_REFRESH_ON_STARTUP`

## Quick Setup

Run `yaacli`. When `general.model` is empty, the first-run wizard initializes the
global configuration and built-in assets before starting the TUI.
