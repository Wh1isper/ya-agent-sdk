# 04 - Tool Permission Catalog

`ya-agent-sdk` should classify every built-in tool with explicit permission metadata.

This document is the implementation-facing catalog for the approval review system in [03-approval-review.md](03-approval-review.md). It defines default permission profiles, runtime refinement rules, MCP heuristics, and validation expectations for built-in SDK tools.

## Goals

- Give every SDK built-in tool a stable `ToolPermissionProfile`.
- Keep permission classification independent from capability tags.
- Make high-risk actions enter `auto_review` by default.
- Make read-only workspace inspection run directly by default.
- Give shell, network, external integration, credential, and destructive behavior clear categories.
- Provide a concrete catalog for tests and implementation reviews.

## Permission Vocabulary

The approval review system uses these dimensions:

| Dimension | Values | Purpose |
| --------- | ------ | ------- |
| Source | `builtin`, `mcp`, `subagent`, `skill`, `user` | Tool origin |
| Category | `read`, `write`, `execute`, `network`, `destructive`, `credential`, `external_integration`, `context_management`, `delegation` | Action type |
| Scope | `workspace`, `session`, `local_system`, `network`, `external_service` | Affected boundary |
| Decision | `allow`, `auto_review`, `deny` | Default policy action |

Default policy guidance:

| Pattern | Default decision | Rationale |
| ------- | ---------------- | --------- |
| Workspace read | `allow` | Bounded inspection is core agent behavior |
| Session state read/write | `allow` | Internal planning and memory state stays in the current run/session boundary |
| Workspace file write | `auto_review` | Writes alter user-visible workspace state |
| Destructive file operation | `auto_review` | Deletes and broad rewrites need reviewer judgment |
| Local command execution | `auto_review` | Shell commands can combine write, network, credential, and destructive behavior |
| Network read | `allow` for search/fetch/scrape, `auto_review` for file download | Network inspection is common; downloads persist external content |
| External service mutation | `auto_review` | External integrations can affect remote state |
| Credential access | `auto_review` | Secret-like paths and environment values need reviewer judgment |
| Context management | `allow` | Summarize and handoff are SDK control-plane operations |
| Subagent delegation | `allow` by default, `auto_review` when delegated tool surface expands scope | Delegation itself is coordination; delegated tools still enforce their own policy |

## Presets

Implementation should add classmethods or constants on `ToolPermissionProfile` for common cases.

```python
ToolPermissionProfile.read_workspace()
ToolPermissionProfile.write_workspace()
ToolPermissionProfile.destructive_workspace()
ToolPermissionProfile.execute_local_system()
ToolPermissionProfile.network_read()
ToolPermissionProfile.network_download()
ToolPermissionProfile.external_integration_read()
ToolPermissionProfile.external_integration_write()
ToolPermissionProfile.context_management()
ToolPermissionProfile.session_state()
ToolPermissionProfile.delegation()
```

Suggested preset definitions:

| Preset | Source | Categories | Scopes | Decision |
| ------ | ------ | ---------- | ------ | -------- |
| `read_workspace()` | `builtin` | `read` | `workspace` | `allow` |
| `write_workspace()` | `builtin` | `write` | `workspace` | `auto_review` |
| `destructive_workspace()` | `builtin` | `write`, `destructive` | `workspace` | `auto_review` |
| `execute_local_system()` | `builtin` | `execute` | `workspace`, `local_system` | `auto_review` |
| `network_read()` | `builtin` | `read`, `network` | `network` | `allow` |
| `network_download()` | `builtin` | `read`, `write`, `network` | `workspace`, `network` | `auto_review` |
| `external_integration_read()` | `builtin` | `read`, `external_integration`, `network` | `external_service` | `allow` |
| `external_integration_write()` | `builtin` | `write`, `external_integration`, `network` | `external_service` | `auto_review` |
| `context_management()` | `builtin` | `context_management` | `session` | `allow` |
| `session_state()` | `builtin` | `read`, `write` | `session` | `allow` |
| `delegation()` | `builtin` | `delegation` | `session` | `allow` |

## Built-in Tool Catalog

### Filesystem Tools

| Tool | Class | Categories | Scopes | Decision | Runtime refinements |
| ---- | ----- | ---------- | ------ | -------- | ------------------- |
| `view` | `ViewTool` | `read` | `workspace` | `allow` | add `credential` and `auto_review` for secret-like paths |
| `ls` | `ListTool` | `read` | `workspace` | `allow` | add `credential` and `auto_review` for secret directories |
| `glob` | `GlobTool` | `read` | `workspace` | `allow` | add `credential` and `auto_review` when matching secret paths broadly |
| `grep` | `GrepTool` | `read` | `workspace` | `allow` | add `credential` and `auto_review` when pattern targets secret material |
| `write` | `WriteTool` | `write` | `workspace` | `auto_review` | add `credential` for secret-like target paths; add `destructive` for overwrite mode |
| `edit` | `EditTool` | `write` | `workspace` | `auto_review` | add `credential` for secret-like target paths; add `destructive` when old string is empty and target exists |
| `multi_edit` | `MultiEditTool` | `write` | `workspace` | `auto_review` | add `destructive` for many edits, replace-all, or empty create behavior |
| `mkdir` | `MkdirTool` | `write` | `workspace` | `auto_review` | keep bounded directory creation medium-risk |
| `move` | `MoveTool` | `write` | `workspace` | `auto_review` | add `destructive` when destination overwrites or source moves broad directory |
| `copy` | `CopyTool` | `read`, `write` | `workspace` | `auto_review` | add `credential` when source or destination is secret-like |
| `delete` | `DeleteTool` | `write`, `destructive` | `workspace` | `auto_review` | classify broad delete patterns as high risk |

Filesystem classifier helpers:

- `is_secret_path(path)` identifies `.env`, credential stores, key files, token files, SSH config, cloud credential files, auth caches, and common secret naming patterns.
- `is_broad_path(path)` identifies repository roots, workspace roots, parent-directory traversal, glob-like deletion targets, and large generated directories.
- `is_generated_path(path)` identifies cache, build, dist, and temporary outputs with lower risk than source trees.
- `is_source_path(path)` identifies source, config, tests, scripts, docs, and repository metadata.

Runtime result examples:

| Call pattern | Added categories | Decision |
| ------------ | ---------------- | -------- |
| `view(file_path="README.md")` | none | `allow` |
| `view(file_path=".env")` | `credential` | `auto_review` |
| `write(file_path="src/app.py", mode="w")` | `write` | `auto_review` |
| `delete(path="build")` | `destructive` | `auto_review` |
| `delete(path=".")` | `destructive` | `auto_review` with high-risk rationale |

### Shell Tools

| Tool | Class | Categories | Scopes | Decision | Runtime refinements |
| ---- | ----- | ---------- | ------ | -------- | ------------------- |
| `shell_exec` | `ShellTool` | `execute` | `workspace`, `local_system` | `auto_review` | command classifier adds `write`, `network`, `credential`, `destructive` |
| `shell_wait` | `ShellWaitTool` | `read` | `session` | `allow` | add `auto_review` only when attached process metadata indicates protected action continuation |
| `shell_status` | `ShellStatusTool` | `read` | `session` | `allow` | read-only process status |
| `shell_input` | `ShellInputTool` | `write`, `execute` | `session`, `local_system` | `auto_review` | stdin can trigger pending process behavior |
| `shell_signal` | `ShellSignalTool` | `execute` | `session`, `local_system` | `auto_review` | signal can alter process behavior |
| `shell_kill` | `ShellKillTool` | `execute`, `destructive` | `session`, `local_system` | `auto_review` | process termination is destructive to the active task |

Shell command classifier signals:

| Signal | Categories |
| ------ | ---------- |
| package install, curl, wget, git clone, browser open, remote API call | `network` |
| rm, unlink, truncate, clean, reset, checkout force, overwrite redirects | `destructive` |
| chmod, chown, sudo, launchctl, systemctl, docker, mount | `execute`, `local_system` |
| cat/read of env, keys, tokens, auth files | `credential` |
| file redirect, tee, sed -i, python scripts writing files | `write` |
| tests, linters, build commands | `execute`, sometimes `write` for caches |

Shell execution review uses generic approval review. Shell-specific risk classification belongs in `ShellTool.resolve_permission(...)` and reviewer context.

### Web and Network Tools

| Tool | Class | Categories | Scopes | Decision | Runtime refinements |
| ---- | ----- | ---------- | ------ | -------- | ------------------- |
| `search` | `SearchTool` | `read`, `network` | `network` | `allow` | add `external_integration` when provider credentials are used |
| `search_stock_image` | `SearchStockImageTool` | `read`, `network` | `network` | `allow` | provider API read |
| `search_image` | `SearchImageTool` | `read`, `network` | `network` | `allow` | provider API read |
| `fetch` | `FetchTool` | `read`, `network` | `network` | `allow` | `head_only=true` stays read-only; large binary output uses truncation summary |
| `scrape` | `ScrapeTool` | `read`, `network` | `network` | `allow` | crawler/provider usage adds `external_integration` metadata |
| `download` | `DownloadTool` | `read`, `write`, `network` | `workspace`, `network` | `auto_review` | destination path secret-like adds `credential`; executable extension adds high-risk rationale |

Network tools should include URL host, scheme, and destination path in review metadata. Tool output truncation should apply to fetched pages and scrape output before model history storage.

### Document and Media Tools

| Tool | Class | Categories | Scopes | Decision | Runtime refinements |
| ---- | ----- | ---------- | ------ | -------- | ------------------- |
| `pdf_convert` | `PdfConvertTool` | `read`, `write` | `workspace` | `auto_review` | generated markdown/images write to workspace export directory |
| `office_to_markdown` | `OfficeConvertTool` | `read`, `write` | `workspace` | `auto_review` | generated markdown/images write to workspace export directory |
| `read_image` | `ReadImageTool` | `read` | `workspace` | `allow` | URL inputs add `network`; secret-like file paths add `credential` |
| `read_audio` | `ReadAudioTool` | `read` | `workspace` | `allow` | URL inputs add `network`; transcription output uses truncation |
| `read_video` | `ReadVideoTool` | `read` | `workspace` | `allow` | URL inputs add `network`; analysis output uses truncation |
| `load_media_url` | `LoadMediaUrlTool` | `read`, `network` | `network` | `allow` | downloaded or cached content adds `write` when persisted |

Document conversion writes generated artifacts. Default decision should be `auto_review` even when input read is safe.

### Context and Enhancement Tools

| Tool | Class | Categories | Scopes | Decision | Runtime refinements |
| ---- | ----- | ---------- | ------ | -------- | ------------------- |
| `summarize` | `HandoffTool` | `context_management` | `session` | `allow` | context compaction is SDK control-plane behavior |
| `thinking` | `ThinkingTool` | `context_management` | `session` | `allow` | internal reasoning aid |
| `to_do_read` | `TodoReadTool` | `read` | `session` | `allow` | session planning state |
| `to_do_write` | `TodoWriteTool` | `write` | `session` | `allow` | session planning state |
| `task_create` | `TaskCreateTool` | `write` | `session` | `allow` | task manager state |
| `task_get` | `TaskGetTool` | `read` | `session` | `allow` | task manager state |
| `task_update` | `TaskUpdateTool` | `write` | `session` | `allow` | task manager state |
| `task_list` | `TaskListTool` | `read` | `session` | `allow` | task manager state |
| `note` | `NoteTool` | `write` | `session` | `allow` | note state can persist in runtime context; products may elevate durable memory writes |
| `note_get` | `NoteGetTool` | `read` | `session` | `allow` | note state read |

Session-state tools stay direct by default because they alter agent control-plane state inside the current runtime boundary.

### Subagent Tools

| Tool | Class | Categories | Scopes | Decision | Runtime refinements |
| ---- | ----- | ---------- | ------ | -------- | ------------------- |
| dynamic subagent tools | `DynamicSubagentTool` | `delegation` | `session` | `allow` | include delegated agent name, allowed toolsets, and prompt summary in metadata |
| `delegate` | `UnifiedSubagentTool` | `delegation` | `session` | `allow` | include selected subagent and delegated prompt summary |
| `subagent_info` | `SubagentInfoTool` | `read`, `delegation` | `session` | `allow` | read-only subagent registry |

Delegated agents use their own tool execution path, so tool permission checks still apply when the subagent invokes tools. The delegation tool should include metadata that lets a product audit which subagent received work.

### Tool Search and Tool Proxy

| Tool | Categories | Scopes | Decision | Runtime refinements |
| ---- | ---------- | ------ | -------- | ------------------- |
| `search_tools` | `read` | `session` | `allow` | tool discovery only |
| `call_tool` | inherited from underlying tool | inherited from underlying tool | inherited from underlying tool | proxy adds underlying namespace and tool metadata |

`ToolProxyToolset._execute_call(...)` should rely on the underlying toolset for approval review and add output truncation around the proxied result.

## MCP Permission Catalog

MCP tools originate outside the SDK built-in catalog. The SDK should infer a permission profile from server metadata, transport, namespace, tool name, and arguments.

### Server Defaults

| Transport | Categories | Scopes | Decision |
| --------- | ---------- | ------ | -------- |
| `stdio` | `external_integration` | `local_system` | `auto_review` |
| `streamable_http` | `external_integration`, `network` | `external_service` | `auto_review` |

### Tool Name Heuristics

| Name signal | Categories | Decision refinement |
| ----------- | ---------- | ------------------- |
| `read`, `get`, `list`, `search`, `find`, `query`, `fetch`, `inspect` | `read` | keep server default or allow through override |
| `write`, `create`, `update`, `patch`, `set`, `send`, `post`, `upload` | `write` | `auto_review` |
| `delete`, `remove`, `drop`, `destroy`, `truncate`, `reset` | `write`, `destructive` | `auto_review` |
| `run`, `exec`, `execute`, `command`, `shell`, `script` | `execute` | `auto_review` |
| `token`, `secret`, `credential`, `auth`, `key` | `credential` | `auto_review` |
| `issue`, `pull`, `repo`, `branch`, `commit`, `deploy`, `release` | `external_integration` | `auto_review` for write-like names |

### Server Overrides

Profile or SDK config can override MCP defaults:

```yaml
mcp_permissions:
  filesystem:
    default_decision: auto_review
    categories: [read, write]
    scopes: [workspace]
    tool_overrides:
      read_file:
        default_decision: allow
        categories: [read]
        scopes: [workspace]
      delete_file:
        default_decision: auto_review
        categories: [write, destructive]
        scopes: [workspace]
```

Override resolution order:

1. exact server and tool override
2. server-level permission profile
3. transport default
4. tool name heuristics
5. argument-sensitive refinements

## Argument-sensitive Refinements

The SDK should provide helper functions for common refinements.

### Path Arguments

Fields named `path`, `file_path`, `directory`, `save_dir`, `source`, `destination`, `src`, `dst`, `cwd`, or `working_dir` should be inspected.

Refinements:

- secret-like path adds `credential`
- workspace root or broad path adds high-risk `destructive` rationale for write/delete tools
- path outside workspace adds `local_system` and `auto_review`
- generated output path lowers rationale severity while preserving write classification

### URL Arguments

Fields named `url`, `urls`, `endpoint`, `host`, or `base_url` should be inspected.

Refinements:

- URL adds `network`
- private network hosts add high-risk rationale
- file URLs add `local_system`
- OAuth or token-bearing URLs add `credential`

### Environment Arguments

Fields named `env`, `environment`, `headers`, `token`, `api_key`, `authorization`, or `secret` should be inspected.

Refinements:

- secret-bearing values add `credential`
- environment mutation adds `write`
- outbound request headers with auth add `external_integration`

### Command Arguments

Fields named `command`, `cmd`, `script`, or `code` should be inspected.

Refinements:

- shell metacharacters and redirection add `execute` and often `write`
- network binaries add `network`
- destructive commands add `destructive`
- credential file reads add `credential`

## Review Metadata

Every approval review request should include enough metadata for policy decisions and audit.

Recommended metadata:

```python
{
    "toolset_id": "filesystem",
    "underlying_tool_name": "write",
    "mcp_server": "github",
    "mcp_transport": "streamable_http",
    "path_summary": {
        "paths": ["src/app.py"],
        "secret_like": False,
        "outside_workspace": False,
        "broad": False,
    },
    "network_summary": {
        "hosts": ["api.github.com"],
        "schemes": ["https"],
    },
    "command_summary": {
        "program": "uv",
        "writes_workspace": True,
        "uses_network": False,
    },
}
```

Sensitive values should be redacted before metadata enters events, traces, or reviewer prompt context. The exact pending action JSON still goes to the reviewer with product-chosen redaction policy applied for display and persistence.

## Implementation Steps

1. Add permission presets and classifier helpers in `ya_agent_sdk.security.approval`.
2. Add `permission` class attributes to built-in tools.
3. Override `resolve_permission(...)` for shell, filesystem write/delete, document conversion, download, and media URL tools.
4. Add MCP inference helper and server override resolution.
5. Add tests that assert every built-in `BaseTool` subclass declares a permission profile.
6. Add tests for path, URL, env, and command refinement helpers.

## Test Matrix

| Area | Expected test |
| ---- | ------------- |
| Catalog coverage | every built-in `BaseTool` has `permission` |
| Filesystem read | `view README.md` resolves `allow` |
| Secret read | `.env` read resolves `credential` and `auto_review` |
| File write | `write src/app.py` resolves `auto_review` |
| Delete | `delete src` resolves `destructive` and `auto_review` |
| Shell test command | `uv run pytest` resolves `execute` and `auto_review` |
| Shell network command | `curl https://example.com` adds `network` |
| Download | `download` resolves `network`, `write`, `auto_review` |
| Document conversion | `pdf_convert` resolves `read`, `write`, `auto_review` |
| Session tools | task and note tools resolve `allow` |
| Delegation | `delegate` resolves `delegation`, `allow` |
| MCP read override | exact tool override can resolve `allow` |
| MCP write heuristic | create/update/delete names resolve `auto_review` |
