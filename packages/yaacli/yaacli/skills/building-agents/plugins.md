# Capability Plugins

Use capability plugins when an application must accept installed third-party Pydantic AI
capability types without adding an application-specific import for every distribution.
The SDK owns one entry-point contract, one immutable catalog, and one strict versioned
TOML manifest. It does not provide a second runtime or ambient package loading.

## Plugin Package Contract

A distribution exports one dataclass capability class per entry point:

```toml
[project.entry-points."ya_agent_sdk.capabilities"]
"acme.search" = "acme_agent.capabilities:SearchCapability"
```

The entry-point name must exactly equal
`SearchCapability.get_serialization_name()`. One distribution may declare multiple
entry points. Importing the distribution must not mutate global state.

Capabilities may contribute native tools, instructions, request or history hooks, model
settings, run-local state, and `CapabilityOrdering`. Keep credentials, live resources,
host services, and durable execution state outside serialized capability arguments.
Obtain them through typed runtime dependencies or host APIs.

See `../../examples/capability_plugin/` for a complete installable distribution.

## Versioned Manifest

```toml
schema_version = 1
entry_points = ["acme.search", "acme.audit"]

[[capabilities]]
name = "acme.search"
arguments = { result_limit = 10 }
```

`entry_points` is the explicit ordered selection of installed types that the process may
import. `capabilities` is the ordered set of instances appended to the root agent's
native `AgentSpec`. Selection is availability, not a grant: a selected-but-ungranted
type may be used by an explicitly configured child, while an unselected type cannot be
granted anywhere.

The manifest is strict. Unknown fields, duplicate selections, unselected grants,
non-finite numbers, and secret-like argument names fail validation. This recursive,
name-based guard cannot identify a secret hidden under a neutral key. Never put API
keys, tokens, passwords, cookies, credentials, or other secrets in the manifest or any
persisted `AgentSpec`.

## Application Bootstrap

Load once at trusted process bootstrap and capture the returned snapshot in every
runtime or plan factory:

```python
from pathlib import Path

from pydantic_ai import AgentSpec
from ya_agent_sdk.agents import validate_agent_spec_capabilities
from ya_agent_sdk.agents.main import create_agent
from ya_agent_sdk.capabilities import load_capability_plugins
from ya_agent_sdk.context import AgentContext

plugins = load_capability_plugins(Path("/etc/my-agent/plugins.toml"))
root_spec = plugins.apply_to_root_agent_spec(AgentSpec.from_dict({"name": "my-agent"}))
validate_agent_spec_capabilities(
    root_spec,
    deps_type=AgentContext,
    custom_capability_types=plugins.custom_capability_types,
    capabilities=build_host_capabilities(),
)

runtime = create_agent(
    "anthropic:claude-sonnet-4-5",
    spec=root_spec,
    custom_capability_types=plugins.custom_capability_types,
    capabilities=build_host_capabilities(),
)
```

Build fresh host capability instances for validation and runtime construction; do not
reuse stateful throwaway instances. The validator calls native `Agent.from_spec()` with
a no-network `TestModel`, executing declarative `from_spec()` factories and validating
their combined static ordering without entering capability lifecycles. Runtime entry
still validates Environment, context, resource, and other dynamic contributions.

The application chooses the path and missing-file policy. Do not make
`create_agent()` scan installed entry points. Do not reload a manifest per request,
session, or runtime. A restart is the adoption boundary for package or manifest
changes.

If the application validates configuration in phases or also supplies directly imported
classes, use the split API:

```python
from ya_agent_sdk.capabilities import (
    load_capability_plugin_manifest,
    resolve_capability_plugins,
)

manifest = load_capability_plugin_manifest(path)
plugins = resolve_capability_plugins(
    manifest,
    explicit_types=[ApplicationCapability],
)
```

Explicit types extend the catalog but do not satisfy manifest `entry_points`. Grant an
explicit type through an ordinary native `AgentSpec` or the programmatic
`capabilities=` list.

## Root and Child Semantics

`apply_to_root_agent_spec()` is root-only. Never call it while compiling a named child
or self fork. Resolve all root and child specs against `plugins.catalog`, and pass the
same `plugins.custom_capability_types` tuple to every native runtime construction.

- manifest grants are appended only to the root spec;
- named children receive only capabilities declared in their own `AgentSpec`;
- self forks do not copy root manifest grants;
- selected plugin types are available for explicit child declarations;
- retained or resumed plans use the same process catalog snapshot; and
- capability ordering remains native Pydantic AI `CapabilityOrdering` plus caller order.

Durable hosts persist declarative specs and plan descriptors, not Python class objects or
the catalog. They must validate final root and child plans before admission and retain
compatible installed code for resume.

## Trust and Operations

Entry-point loading imports Python with full host-process authority. Install only trusted
packages into the same Python environment as the application, select exact names in an
administrator-controlled manifest, and fail closed on parse, import, collision, or plan
validation errors. The SDK does not install, upgrade, sandbox, hot reload, or partially
accept plugins.

YAACLI uses the fixed optional path `~/.yaacli/plugins.toml`. YA Claw uses the optional
explicit `YA_CLAW_CAPABILITY_PLUGIN_MANIFEST` path; an explicitly configured missing file
is fatal. Both hosts load one snapshot at startup and require a restart after changes.
