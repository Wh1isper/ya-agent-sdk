# YA Desktop

YA Desktop is a Tauri 2 + TypeScript native agent workspace for Claw-based runtimes.

The product direction is Desktop-first: Command Center, Workspace Home, Tasks, Chat Work Surface, Approvals Inbox, Connections, and Advanced Runtime.

## Development

```bash
corepack pnpm --dir apps/ya-desktop install
corepack pnpm --dir apps/ya-desktop dev
corepack pnpm --dir apps/ya-desktop tauri:dev
```

## Checks

```bash
corepack pnpm --dir apps/ya-desktop lint
corepack pnpm --dir apps/ya-desktop build
corepack pnpm --dir apps/ya-desktop test
cargo check --manifest-path apps/ya-desktop/src-tauri/Cargo.toml
```
