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

# Maximum requests per session
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
# Enable structured clarifying questions in the interactive TUI (default: true)
enable_user_input = true

# Tools requiring user approval before execution
need_approval = ["shell", "write"]
```

Set `enable_user_input = false` when the project should not expose the interactive `ask_user_question` tool. Headless mode never exposes it.

Common patterns:
- `[]` - No approval needed (trust all tools)
- `["shell"]` - Approve shell commands only
- `["shell", "write", "edit"]` - Approve all code modifications

### mcp.json

MCP server configurations:

```json
{
  "servers": {
    "my-server": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@some/mcp-server"],
      "env": {}
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

The former custom-command `mode` field is deprecated and ignored; every custom
command uses the same agent execution semantics. `/init` is provided by
default, while commands such as `/commit` and `/review` come from configuration.

`instructions` adds a static model-instruction segment only while its profile is active. It applies to every main-agent model request in TUI and headless runs, including restored and compacted histories; `/model` changes it for later requests in the current session.

## Subagent Configuration

Create markdown files in `~/.yaacli/subagents/`:

```markdown
---
name: my-subagent
description: Brief description shown when selecting tools
instruction: |
  When to use this subagent and what to provide
tools:
  - grep
  - view
optional_tools:
  - shell
model: inherit
---

You are a specialist in [domain].

## Process
1. Step one
2. Step two
```

### Configuration Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Unique identifier, becomes the tool name |
| `description` | Yes | Shown to model when selecting tools |
| `instruction` | No | Injected into parent's system prompt |
| `tools` | No | Required tools - ALL must be available |
| `optional_tools` | No | Optional tools - included if available |
| `model` | No | `"inherit"` or model name |

### Tool Availability Rules

- **Required tools** (`tools`): Subagent disabled if ANY unavailable
- **Optional tools** (`optional_tools`): Included only if available
- **No tools specified**: Inherits all parent tools

### Builtin Presets

| Preset | Purpose | Required Tools |
|--------|---------|----------------|
| `debugger` | Root cause analysis | glob, grep, view, ls |
| `explorer` | Codebase navigation | glob, grep, view, ls |
| `code-reviewer` | Code quality review | glob, grep, view, ls |
| `searcher` | Web research | search |

## Skills Directory

Skills are loaded from (highest priority last):

1. Built-in: `yaacli/skills/` (package bundled)
2. Global: `~/.yaacli/skills/`
3. Project: `.yaacli/skills/`

## Environment Variables

TUI settings can be overridden via `YAACLI_*` environment variables:

- `YAACLI_CODE_THEME`
- `YAACLI_SHOW_TOKEN_USAGE`
- `YAACLI_SESSION_DIR`

## Quick Setup

Run `yaacli setup` to initialize global configuration with defaults.
