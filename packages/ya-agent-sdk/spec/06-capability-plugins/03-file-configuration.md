# Versioned File Configuration

## 1. Purpose

YA Agent SDK defines one host-neutral TOML manifest for applications that want operators
to select installed capability entry points and grant configured instances to a root
agent without writing Python bootstrap code for each plugin.

The manifest does not install packages, discover an import target from user data, mutate
a running catalog, or define another runtime. It feeds the existing
`CapabilityCatalog` and native Pydantic AI `AgentSpec` contracts.

## 2. Schema Version 1

```toml
schema_version = 1
entry_points = ["acme.search", "acme.audit"]

[[capabilities]]
name = "acme.search"
arguments = { result_limit = 10 }

[[capabilities]]
name = "acme.audit"
arguments = { policy = "strict" }
```

The top-level fields are:

- `schema_version`: required and exactly integer `1`;
- `entry_points`: an ordered, unique list of exact installed entry-point names; and
- `capabilities`: an ordered list of root-agent grants in normalized native
  `CapabilitySpec` form (`name` plus `arguments`).

Unknown fields and non-finite numbers fail validation. Names must be non-empty and
contain no surrounding whitespace. Every root grant must name an entry point selected
by `entry_points`. Manifest arguments are durable declarative configuration and must not
contain secrets or credentials; sensitive argument names such as `api_key`,
`access_token`, `password`, `passphrase`, `jwt`, and `secret` fail validation recursively.
This is fail-closed name-based validation, not value inspection: it cannot identify a
secret hidden under a neutral key. Applications and operators remain responsible for
keeping every secret value out of the manifest. Plugins obtain live credentials through
typed runtime dependencies or host APIs instead. Multiple grants may use the same
selected type because distinct configured instances are a native
capability composition concern; native singleton and ordering rules still apply during
agent construction.

A selected but ungranted entry point is available in the catalog for a named child or
another explicitly declared `AgentSpec`. It does not enter the root agent. Conversely,
a grant cannot load an unselected entry point.

## 3. SDK API

Applications use the one-shot loader when they already have a manifest path:

```python
from ya_agent_sdk.capabilities import load_capability_plugins

plugins = load_capability_plugins("/etc/my-agent/plugins.toml")
root_spec = plugins.apply_to_root_agent_spec(existing_agent_spec)

runtime = create_agent(
    model,
    spec=root_spec,
    custom_capability_types=plugins.custom_capability_types,
    capabilities=host_capabilities,
)
```

`load_capability_plugins()` performs these steps synchronously at trusted process
bootstrap:

1. parse TOML with the Python standard library;
2. validate the strict versioned manifest;
3. load only the selected entry points;
4. build the default immutable SDK catalog; and
5. return `ResolvedCapabilityPlugins` with the manifest, catalog, exact custom-type
   tuple, a root-only `AgentSpec`, and `apply_to_root_agent_spec()`.

The split APIs support applications that validate configuration separately or construct
it programmatically:

```python
manifest = load_capability_plugin_manifest(path)
plugins = resolve_capability_plugins(
    manifest,
    explicit_types=[ApplicationCapability],
)
```

Explicit classes extend the catalog but do not satisfy or bypass `entry_points`.
Manifest grants remain restricted to explicitly selected installed names. Applications
can grant an explicitly imported class through their ordinary native `AgentSpec` or
programmatic `capabilities=` list.

`apply_to_root_agent_spec()` returns a new native spec and appends manifest grants after
any capabilities already present in the supplied root spec. Hosts must never call this
root-only helper for named-child or self-fork specs. The resulting native list is only
the initial caller order; Pydantic AI still applies `CapabilityOrdering` constraints.

Manifest validation checks the file shape, JSON-compatible argument values, selection,
and type catalog. Resolution also binds each grant's keyword names to the selected
class constructor or overridden `from_spec()` signature without executing the factory,
so missing or unexpected arguments fail at bootstrap. Plugin-specific value, template-context, dependency-type, and combined ordering
validation still occurs through native Pydantic AI agent construction. Durable hosts
call the shared helper with their exact static plan before fingerprinting or persistence:

```python
from ya_agent_sdk.agents import validate_agent_spec_capabilities

validate_agent_spec_capabilities(
    root_spec,
    deps_type=ApplicationAgentContext,
    custom_capability_types=plugins.custom_capability_types,
    capabilities=build_fresh_host_capabilities(),
)
```

The helper uses public `Agent.from_spec()` with a no-network test model, executes
capability factories, and validates combined native ordering without entering lifecycle
resources. Validation and real runtime construction must use fresh programmatic
capability instances. Runtime entry remains authoritative for dynamic Environment,
context, and resource contributions.

## 4. Host Lifetime

A host reads its chosen manifest path once at process or frontend bootstrap and retains
the resulting catalog snapshot. The same snapshot must be used for:

- root `AgentSpec` construction;
- named-child and self-fork plan validation;
- child execution drivers;
- retained plan restoration; and
- every active or historical runtime reconstructed in that host process.

A runtime factory must not reread the file or rediscover entry points for each session.
Persist declarative `AgentSpec` content and ordinary plan descriptors, not Python class
objects or the catalog itself. Restarting the host is the adoption boundary for a file
or installed-package change.

Root grants apply only to the root agent. Catalog availability does not imply grant or
inheritance, and named children or self forks receive a plugin capability only when
their own native specification declares it.

## 5. Host Policy and Trust

The SDK intentionally does not choose a path, treat a missing file as empty, or merge
multiple manifests. Each application owns those bootstrap policies. A host may make one
fixed default path optional by checking for existence before calling the loader, but an
explicitly configured missing path should fail.

Because selected entry points execute Python with process authority, durable hosts
should use an administrator-controlled path rather than project-local ambient
configuration. Invalid TOML, unsupported versions, unknown fields, missing or duplicate
entry points, import failures, and catalog collisions fail closed. The SDK performs no
automatic installation, package upgrade, fallback, hot reload, or partial catalog
construction.
