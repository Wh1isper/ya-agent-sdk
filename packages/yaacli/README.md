# YAACLI CLI

TUI reference implementation for [ya-agent-sdk](https://github.com/wh1isper/ya-mono/tree/main/packages/ya-agent-sdk).

## Usage

Run with uvx:

```bash
uvx --from 'yaacli[rs]' yaacli
```

Install with uv:

```bash
uv tool install 'yaacli[rs]'
yaacli
```

`[rs]` installs the native Rust filesystem search binding. The equivalent extra-dependency form is:

```bash
uv tool install yaacli --with ya-ripgrep-core
```

`ya-ripgrep-core` is a library dependency, so `--with` is the matching uv form; `--with-executables-from` applies to companion packages that also expose CLI executables.

Update with uv:

```bash
uv tool upgrade yaacli
```

Install with pip:

```bash
pip install 'yaacli[rs]'
yaacli
```

Run as a module:

```bash
python -m yaacli
```

## Headless and Saved Sessions

Run one prompt without the TUI:

```bash
yaacli -p "Fix the failing tests"
yaacli -p "Continue" --session <session-id> --profile <profile-id>
yaacli -p "Run an isolated worker task" --worker
```

Headless stdout is an NDJSON event stream. Human-readable diagnostics, fatal details, and resume hints are written to stderr so scripts can parse every stdout line as JSON. A successful headless run always saves its durable session turn, independent of `session.auto_save_history`; a failed or cancelled run emits its terminal event but does not save a recovery snapshot. `--worker` requires `--prompt` and disables synchronous delegate subagents for that run. `--profile` applies only to the current invocation; selecting a profile through `/model` persists it for future launches.

Inspect or delete durable sessions without starting the TUI:

```bash
yaacli sessions list
yaacli sessions show <session-id>
yaacli sessions delete <session-id>
```

Session IDs may be supplied by unique prefix. The configured session directory and retention controls live under `[session]` in `config.toml`; see [`spec/05-session-persistence.md`](spec/05-session-persistence.md).

## TUI Interaction

The output viewport has priority over auxiliary UI. The task pane is hidden when empty, uses one summary row by default, and expands with `F2`. The model selector is an overlay and does not permanently consume output rows.

- Enter submits while idle and sends guidance to the active run while an agent is running.
- Ordinary text submitted during an active agent run is steering. The status bar shows how many steering messages are still waiting for a model request, without exposing their content. Registered slash commands and `!shell` remain local control syntax: safe busy commands execute and idle-only commands are rejected without clearing the draft. Other slash-prefixed text, including absolute paths such as `/home/user/file`, remains ordinary user input.
- `/cancel` or `Ctrl+C` requests cancellation of cancellable foreground work. Once the TUI enters `SAVING`, persistence is allowed to finish and cannot be cancelled; Ctrl+C does not exit while the save is in progress.
- `/clear` clears only the visible transcript; `/new` starts a fresh conversation and session, terminating and discarding background subagent and shell work owned by the previous session while keeping the runtime environment reusable.
- Background results never modify or clear the compose area. When the foreground is idle, a deliverable async subagent or monitored-shell notification automatically starts a main-agent turn with a small provenance-aware system reminder; the notification is delivered through the message bus. During an active agent run, `/integrate` can deliver queued notifications to its next model request; otherwise, unread notifications automatically wake the main agent after that run finishes.
- `/agents` shows running and recently completed background subagents; `/process` shows active background shell processes.
- `/attachments` and `/remove-image` inspect or edit images queued for the next turn.
- `/tool <call-id>` shows the complete retained result for a tool call.
- `/session <id>` restores a saved session. The CLI equivalent is `yaacli --session <id>`.
- Use `/help` for the complete built-in and configured command list. Slash commands and available skills complete while typing.
- Prefix an idle prompt with one or more available skill names to request them explicitly: `/lark-cli /agent-builder Build an agent that replies in Lark`. Only the leading consecutive `/skill-name` tokens are selected. If the first token is also a built-in or configured command, command dispatch takes precedence.
- The interactive TUI enables `ask_user_question` by default. When the agent needs clarification, YAACLI renders one to four structured questions and accepts an option number, comma-separated numbers for multi-select questions, or free text.
- Long status text wraps to the available terminal width. Foreground elapsed time uses compact forms such as `42s`, `3m 05s`, and `1h 05m 09s`.

## Built-in Skills

YAACLI ships with `building-agents` from the repository canonical source `skills/agent-builder/`.

The YA Claw deployment skill lives in `skills/ya-claw-deploy/` and is published as `YA_CLAW_DEPLOY_SKILL.zip` during release.

The repository sync script keeps bundled skill files under `packages/yaacli/yaacli/skills/` aligned.

YAACLI refreshes and resolves `/skill-name` against the effective SDK skill catalog at submission time, including built-in, global, shared, and project skills after normal priority rules. The visible transcript and prompt history retain the original input; the model receives a catalog-grounded explicit-selection marker plus the remaining task. A slash prefix that matches neither a registered command nor an available skill is submitted as ordinary user input.

## CodeAct

YAACLI exposes `run_code` and `run_program` by default. They execute restricted Python for tool orchestration; `run_program` reads reviewed programs through the runtime's `FileOperator`. Shell remains a separate execution surface and is not made callable from CodeAct. Disable both CodeAct tools globally in `~/.yaacli/tools.toml` or per project in `.yaacli/tools.toml`:

```toml
[tools]
enable_codeact = false
```

## MCP Tool Exposure

MCP tools are exposed directly to the model by default. Configure the behavior globally in `~/.yaacli/tools.toml` or per project in `.yaacli/tools.toml`:

```toml
[tools]
mcp_mode = "direct" # "direct" (default) or "proxy"
```

Direct mode registers each configured MCP server as a native toolset and exposes namespaced `<server>_<tool>` names by default. A server's optional `prefix` field in `mcp.json` overrides `<server>`; set it to `""` to expose the server's native tool names without a prefix. Omitting `prefix` or setting it to `null` preserves the default server-name prefix. Proxy mode exposes the fixed `mcp_search_tool` and `mcp_call_tool` pair instead, which can improve prompt-cache stability when many MCP tools are configured. Servers marked `"required": false` remain optional in both modes. If an optional server cannot connect or becomes unavailable, the TUI reports the degraded MCP namespace once per status change and continues without that server.

For direct host-managed MCP tools, structured output remains the tool's Python return value even when the response also contains media. Completed MCP error responses are returned for model or CodeAct inspection without consuming Pydantic AI's `ModelRetry` budget; failures that produce no MCP result are terminal failed tool outcomes.

```json
{
  "servers": {
    "docs": {
      "transport": "streamable_http",
      "url": "https://example.com/mcp",
      "prefix": "reference"
    },
    "local": {
      "transport": "stdio",
      "command": "local-mcp-server",
      "prefix": ""
    }
  }
}
```

## Structured User Input

The interactive TUI opts into the SDK's deferred `ask_user_question` tool. Disable it globally in `~/.yaacli/tools.toml` or for one project in `.yaacli/tools.toml`:

```toml
[tools]
enable_user_input = true
user_input_timeout_seconds = 120
```

`enable_user_input` defaults to `true`. Each structured question waits up to `user_input_timeout_seconds`, which defaults to 120 seconds. If no answer arrives, YAACLI rejects the deferred call with an English model prompt that explains the timeout and directs the agent to continue using its best judgment rather than requesting the same input again. Set `enable_user_input = false` to remove the tool entirely. Headless mode does not expose this tool because it cannot collect interactive answers. The SDK also leaves it disabled unless a host explicitly registers it and implements deferred continuation. Project `tools.toml` replaces the global tool policy as a whole; if a project file exists, repeat any non-default user-input settings there rather than relying on global values.

## Development

This package lives in the [`ya-mono`](https://github.com/wh1isper/ya-mono) workspace.

```bash
git clone git@github.com:YOUR_NAME/ya-mono.git
cd ya-mono
uv sync --all-packages
cp packages/yaacli/.env.example packages/yaacli/.env
```

YAACLI loads `.env` from `packages/yaacli/.env` and the current working directory without replacing variables already present in the process. The package file is loaded first and therefore wins duplicate keys; the working-directory file supplies only keys that remain unset.
Provider API keys can live in that `.env` file or in `~/.yaacli/config.toml` under `[env]`.
SDK and tool variables such as `YA_AGENT_*` and search API keys can also live in that same `.env` file because YAACLI loads it into the process environment at startup.
Use [`packages/ya-agent-sdk/.env.example`](../ya-agent-sdk/.env.example) as the reference list for SDK and tool variables.

The TUI detects the terminal's light or dark background at startup. For recognized local terminals such as VS Code's integrated terminal, it performs a short OSC 11 query; active queries are skipped over SSH to avoid delayed terminal responses. Detection then falls back to `COLORFGBG`, followed by the dark theme. Override detection in `~/.yaacli/config.toml` when needed:

```toml
[display]
code_theme = "auto" # auto, dark, or light
```

The equivalent environment override is `YAACLI_CODE_THEME=auto`.

## Terminal Bells

YAACLI emits a terminal bell by default after a successful interactive agent turn and whenever a turn pauses for HITL user input. Either notification can be disabled independently:

```toml
[notifications]
bell_on_turn_complete = false
bell_on_user_action_required = false
```

A run's elapsed timer is paused for the entire HITL wait and resumes after all requested user actions have been answered. The bell is emitted by the terminal, so it passes through SSH and is handled by the local terminal client. In VS Code Remote, configure the local VS Code **User** settings (not Remote or workspace settings) to show the visual bell and always play its terminal-bell accessibility signal:

```json
{
  "terminal.integrated.enableVisualBell": true,
  "terminal.integrated.bellDuration": 1000,
  "accessibility.signals.terminalBell": {
    "sound": "on",
    "announcement": "off"
  }
}
```

Use `"sound": "on"` for unconditional playback; `"auto"` lets VS Code decide whether to play the accessibility signal. `announcement` controls screen-reader output and can remain `"auto"` when needed. VS Code plays its own accessibility sound through the local audio output; this is not an operating-system desktop notification.

To verify terminal support independently of YAACLI, run `printf '\a'` in the same integrated terminal. It should show VS Code's visual-bell indicator. Remote YAACLI processes cannot directly invoke the client machine's operating-system notification API.

Codex OAuth credentials can be created once and reused from YAACLI:

```bash
uvx ya-oauth login codex
```

Then set `model = "oauth@codex:gpt-5.5"` in a YAACLI model profile.

Model profiles are configured in `~/.yaacli/config.toml` and selected with `/model` inside the TUI:

```toml
[general]
model = "anthropic:claude-sonnet-4-5"
model_settings = "anthropic_adaptive_high"
model_cfg = "claude_200k"
instructions = """
Prefer careful analysis and concise answers.
"""

[model_profiles.fast]
label = "Fast"
model = "openai-responses:gpt-5.6-luna"
model_settings = "openai_responses_luna"
model_cfg = "gpt5_270k"
instructions = """
Optimize for speed. Avoid broad exploration unless necessary.
"""

[model_profiles.pro]
label = "GPT-5.6 Pro"
model = "openai-responses:gpt-5.6"
model_settings = "openai_responses_pro"
model_cfg = "gpt5_270k"

[model_profiles.sol]
label = "GPT-5.6 Sol"
model = "openai-responses:gpt-5.6-sol"
model_settings = "openai_responses_max"
model_cfg = "gpt5_270k"

[model_profiles.codex_oauth]
label = "Codex OAuth"
model = "oauth@codex:gpt-5.5"
model_settings = "openai_responses_high"
model_cfg = "gpt5_350k"
```

`instructions` is an optional static model-instruction segment for the active main-agent profile. YAACLI evaluates it for every model request, alongside the built-in prompt (or `general.system_prompt_file`), project guidance, and user rules. Switching profiles with `/model` replaces the active profile instructions for later requests in the same session, including restored or compacted histories.

`[general]` is the startup fallback profile. The last selected profile is remembered in `~/.yaacli/state.json` and restored on the next launch when that profile still exists.

Shell command review is configured in `~/.yaacli/config.toml` under `security.shell_review`:

```toml
[security.shell_review]
enabled = true
model = "gateway@openai-responses:gpt-5.4-mini"
model_settings = "openai_responses_low"
on_needs_approval = "defer"
risk_threshold = "high"
```

When enabled, `model` is required. `model_settings` accepts SDK preset names or an inline TOML table. `risk_threshold` defaults to `high` and controls when the configured action triggers.

Run CLI tests from the workspace root:

```bash
make test-cli
```

## Clipboard Image Paste

Plain terminal paste always inserts text into the input box.
Use `Ctrl+V` or `/paste-image` to attach an image from the system clipboard. During an active agent run the image remains queued for the next turn and is never converted into steering text. Generated attachment chips are removed before registered commands, explicit skills, `!` control syntax, or ordinary prompts are classified, so a visible chip cannot hide a command. If the user deleted the chip, its binary is removed before dispatch.
On macOS terminal apps over SSH, map `Command+Shift+V` to send `Ctrl+V` if you want a native-feeling shortcut.

YAACLI reads clipboard images through Pillow first on macOS and Windows.
macOS also reads Finder-copied image files through Cocoa pasteboard APIs via `pyobjc-framework-Cocoa`.
Linux image paste still relies on `wl-paste` on Wayland or `xclip` on X11.

## License

BSD 3-Clause License. See the [repository license](https://github.com/wh1isper/ya-mono/blob/main/LICENSE).
