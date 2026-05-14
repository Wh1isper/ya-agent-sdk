# YAACLI CLI

TUI reference implementation for [ya-agent-sdk](https://github.com/wh1isper/ya-mono/tree/main/packages/ya-agent-sdk).

## Usage

Run with uvx:

```bash
uvx yaacli
```

Install with uv:

```bash
uv tool install yaacli
yaacli
```

Update with uv:

```bash
uv tool upgrade yaacli
```

Install with pip:

```bash
pip install yaacli
yaacli
```

Run as a module:

```bash
python -m yaacli
```

## Built-in Skills

YAACLI ships with `building-agents` from the repository canonical source `skills/agent-builder/`.

The YA Claw deployment skill lives in `skills/ya-claw-deploy/` and is published as `YA_CLAW_DEPLOY_SKILL.zip` during release.

The repository sync script keeps bundled skill files under `packages/yaacli/yaacli/skills/` aligned.

## Development

This package lives in the [`ya-mono`](https://github.com/wh1isper/ya-mono) workspace.

```bash
git clone git@github.com:YOUR_NAME/ya-mono.git
cd ya-mono
uv sync --all-packages
cp packages/yaacli/.env.example packages/yaacli/.env
```

YAACLI loads `.env` from `packages/yaacli/.env` and the current working directory.
Provider API keys can live in that `.env` file or in `~/.yaacli/config.toml` under `[env]`.
SDK and tool variables such as `YA_AGENT_*` and search API keys can also live in that same `.env` file because YAACLI loads it into the process environment at startup.
Use [`packages/ya-agent-sdk/.env.example`](../ya-agent-sdk/.env.example) as the reference list for SDK and tool variables.

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

[model_profiles.fast]
label = "Fast"
model = "openai-responses:gpt-5-mini"
model_settings = "openai_responses_low"
model_cfg = "gpt5_270k"

[model_profiles.codex_oauth]
label = "Codex OAuth"
model = "oauth@codex:gpt-5.5"
model_settings = "openai_responses_high"
model_cfg = "gpt5_270k"
```

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
Use `Ctrl+V` or `/paste-image` to attach an image from the system clipboard.
On macOS terminal apps over SSH, map `Command+Shift+V` to send `Ctrl+V` if you want a native-feeling shortcut.

YAACLI reads clipboard images through Pillow first on macOS and Windows.
macOS also reads Finder-copied image files through Cocoa pasteboard APIs via `pyobjc-framework-Cocoa`.
Linux image paste still relies on `wl-paste` on Wayland or `xclip` on X11.

## License

BSD 3-Clause License. See the [repository license](https://github.com/wh1isper/ya-mono/blob/main/LICENSE).
