# YA Claw Web

Vite + React + TypeScript web console for YA Claw.

## Product areas

- conversations, run activity, and proactive-agent inspection
- schedules and durable workflows
- workspace files, memory, and artifacts
- agent profiles, integrations, and runtime settings

## Development

Start the YA Claw backend, then run:

```bash
cd apps/ya-claw-web
pnpm install
pnpm dev
```

The development server proxies `/api` and `/healthz` to
`http://127.0.0.1:9042`. Set `VITE_CLAW_PROXY_TARGET` to use a different
backend origin.

The API token is entered through the connection screen. By default it remains
in browser memory for the active page session. Users can explicitly select
**Remember this connection on this device** to store the token in browser
`localStorage` and restore the connection after reloads or browser restarts.
Use this option only on a trusted personal browser profile. **Disconnect**
removes the local copy of the token; it does not revoke the bearer token on the
server.
