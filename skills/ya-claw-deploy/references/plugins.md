# Capability Plugin Deployment

YA Claw supports installed third-party Pydantic AI capability types through the
SDK-owned versioned manifest. Package installation, type selection, and root grants are
three separate operator decisions.

## Trust Boundary

A selected entry point imports Python into the YA Claw service process with full process
authority. It is not a workspace-container extension or sandbox. Install only trusted
distributions, keep the manifest administrator-controlled and read-only, and select
exact entry-point names. YA Claw never installs packages or loads every installed entry
point automatically.

## Install into the Service Environment

For a uv tool deployment:

```bash
uv tool install 'ya-claw[rs]' --with acme-agent-plugin
```

The plugin must be present in the exact environment that runs `ya-claw start`.
Installing it only in the reusable `ya-claw-workspace` image does not make its entry
point visible to the service.

## Configure the Manifest

```toml
schema_version = 1
entry_points = ["acme.search"]

[[capabilities]]
name = "acme.search"
arguments = { result_limit = 10 }
```

Set an absolute service-visible path:

```env
YA_CLAW_CAPABILITY_PLUGIN_MANIFEST=/etc/ya-claw/plugins.toml
```

Omitting the setting disables external plugins. An explicitly configured missing file,
invalid TOML, unsupported schema, unknown field, missing or duplicate entry point,
import failure, or catalog collision fails service startup.

`entry_points` selects the installed types available to profile resolution.
`capabilities` grants ordered instances to every root profile before admission
fingerprinting. Selected-but-ungranted types may be used by explicit child specs. Root
grants never enter named children or self forks implicitly.

Arguments are durable non-secret configuration. Secret-like keys are rejected
recursively, but this name-based guard cannot detect a secret under a neutral key.
Provide credentials and live authority through typed runtime dependencies or host APIs
rather than the manifest or profile specs.

## Derived Service Image

The official service image intentionally contains no third-party plugins. Build a
derived image:

```dockerfile
FROM ghcr.io/wh1isper/ya-claw:latest
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN uv pip install --python /opt/venv/bin/python acme-agent-plugin
```

Mount only the manifest at runtime:

```yaml
services:
  ya-claw:
    build: .
    environment:
      YA_CLAW_CAPABILITY_PLUGIN_MANIFEST: /etc/ya-claw/plugins.toml
    volumes:
      - ./plugins.toml:/etc/ya-claw/plugins.toml:ro
```

Do not bind-mount arbitrary Python package source into a running production service as a
substitute for an immutable image.

## Lifecycle and Verification

YA Claw resolves the manifest once before routes and execution lifecycles start. The
same catalog snapshot is used for profile admission, child descriptors, async resume,
retained-plan restoration, memory runs, and runtime construction. Package or manifest
changes require a service restart.

Verify deployment in this order:

1. import the plugin with the service interpreter;
2. start YA Claw and confirm `/healthz`;
3. submit a root run and verify the granted capability is present;
4. test any child profile that explicitly grants a selected type; and
5. restart with an intentionally invalid manifest in staging to confirm fail-closed
   startup.

For the schema and application contract, see
`packages/ya-agent-sdk/spec/06-capability-plugins/03-file-configuration.md` in the
repository.
