# Custom Capability Type Discovery

## 1. Decision

YA Agent SDK 2.0 provides one small, host-neutral mechanism for making third-party
Pydantic AI custom capability types available to native `AgentSpec` parsing and agent
construction.

Each Python package entry point exports exactly one capability class. Explicit Python
imports pass the same class directly. The SDK validates both paths and returns one
immutable `CapabilityCatalog` containing custom capability classes and lightweight
source provenance.

There is no parallel plugin runtime, provider identity layer, lifecycle hook, global
registry, durability manifest, or plugin-specific rollout protocol. The SDK also
provides one strict host-neutral TOML manifest that maps explicit entry-point selection
to native root-agent capability grants.

## 2. Core Contract

The packaging group is:

```text
ya_agent_sdk.capabilities
```

The entry-point name is the capability type's canonical serialization name:

```toml
[project.entry-points."ya_agent_sdk.capabilities"]
"acme.search" = "acme_agent.capabilities:SearchCapability"
```

A distribution contributes multiple types by declaring multiple entry points. Direct
imports remain first class:

```python
from acme_agent.capabilities import SearchCapability
from ya_agent_sdk.capabilities import build_default_capability_catalog

catalog = build_default_capability_catalog(explicit_types=[SearchCapability])
```

The SDK owns metadata discovery, selected entry-point loading, class validation,
serialization-name collision checks, deterministic registration ordering, provenance,
the versioned plugin manifest, and the `custom_capability_types` tuple passed to
Pydantic AI. YAACLI, YA Claw, and other hosts choose a trusted manifest path and retain
one resolved catalog snapshot for their process lifetime.

Catalog availability is not an agent grant. A capability instance enters an agent only
through `capabilities=` or a native `AgentSpec.capabilities` entry.

## 3. Design Outcomes

1. One entry point maps to one Pydantic AI custom capability class.
2. The class serialization name is the only plugin-facing type identity.
3. Distribution name/version and import target are provenance, not a second identity
   or compatibility protocol.
4. SDK built-in custom types and accepted external types form one immutable custom-type
   tuple; native Pydantic AI types remain in its native registry.
5. Catalog order is deterministic registry output only. It never determines runtime
   capability behavior or hook order.
6. Runtime order uses native Pydantic AI local relationships plus caller order as its
   ready-node tiebreaker. YA adds no plugin priority or stage registry and reserves
   global `position` tiers for genuine framework-wide boundaries.
7. Metadata discovery does not import packages. Loading occurs only when a host asks the
   SDK to build a catalog from installed names or explicitly chooses all discovered
   names.
8. Importing a package never mutates process-global state.
9. Environment and context contributions provide runtime instances and dependencies;
   they do not modify the static type catalog.
10. Plugin discovery says nothing about durable safety. Durable hosts validate the final
    resolved plan under their existing durable execution contract and reject unsupported
    custom behavior.

## 4. Document Map

| Document | Topic |
| --- | --- |
| [01-contract-and-discovery.md](01-contract-and-discovery.md) | entry-point and direct-import contracts, validation, catalog shape, provenance, and Pydantic AI integration |
| [02-host-integration-and-validation.md](02-host-integration-and-validation.md) | SDK/YAACLI/Claw ownership, portable plans, durability boundary, migration, and tests |
| [03-file-configuration.md](03-file-configuration.md) | strict TOML schema, loading API, root grants, process lifetime, and trust policy |

## 5. Scope

This specification covers serializable Pydantic AI custom capability type discovery.
It does not define:

- capability instances or runtime behavior composition;
- package installation, dependency resolution, sandboxing, or hot unload;
- a generic plugin, callback, hook, model, store, UI, or Environment extension system;
- the host-specific choice of manifest path or missing-file policy;
- durable operation metadata or adapters; or
- compatibility and deployment policy for retained executable code.

The [capability-first runtime specification](../05-capability-first-runtime.md) remains
authoritative for capability composition, instances, dependencies, resolved plans, and
durable host behavior.
