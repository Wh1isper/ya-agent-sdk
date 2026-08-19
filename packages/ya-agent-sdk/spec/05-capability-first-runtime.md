# Capability-First Runtime Architecture

This design defines the capability-first composition model for `ya-agent-sdk`, the
native steering contract shared by SDK hosts, YAACLI durable session execution, and the
portable subagent runtime used by SDK, YAACLI, and YA Claw.

Pydantic AI capabilities are the only public unit for composing agent-run behavior.
SDK tools, private toolsets, history transformers, state stores, and host services have
separate implementation roles and do not form competing composition planes.

User steering is delivered through Pydantic AI `AgentRun.enqueue()`. YA Claw exposes
immediate steering and queue-until-idle delivery as two product modes over the native
Pydantic AI priorities.

## Architecture Outcomes

1. `capabilities=` is the single public behavior-composition surface.
2. Every SDK feature owns its tools, instructions, run-local state, and related hooks
   through a cohesive capability.
3. Presets are `CombinedCapability` compositions of independently useful leaves.
4. Pydantic AI owns capability ordering, tool management, Tool Search, MCP,
   pending-message delivery, retries, and enqueue events.
5. `Environment` continues to own filesystem, shell, resources, temporary storage, and
   lifecycle authority. Capabilities use those authorities through typed context
   dependencies.
6. Durable task, note, handoff, logical-run input retention, and host state remains
   outside capability instances.
7. Stable guidance, dynamic catalogs, transient request context, and canonical
   conversation content use distinct prompt channels.
8. MessageBus, the steering-only replay field, the completion guard, Claw polling, and
   late-steering terminal re-segmentation are removed. A context-owned run input ledger
   preserves every applied user input across compact and handoff without becoming a
   second delivery path.
9. YA Claw `mode="steer"` maps to `priority="asap"`; `mode="queue"` maps to
   `priority="when_idle"`.
10. Durable queued runs and active-run queued input remain separate concepts.
11. YAACLI product turns execute through a host-owned local coordinator and the SDK
    segment harness while `SessionStore` remains the source of session, history, input,
    HITL, checkpoint, event, and retention truth.
12. Native Pydantic AI `AgentSpec` cores define declarative main and child agents;
    thin YA `SubagentSpec` envelopes, resolved plans, a public registry, and one execution
    service add only delegation behavior and replace duplicate agent schemas, tool
    inheritance, hidden delegate backends, and copied host resolvers.
13. The SDK owns installed and explicitly imported custom capability type discovery and
    supplies one immutable type catalog to standalone SDK, YAACLI, YA Claw, main, and
    child plan resolution.

## Document Map

| Document | Topic |
| --- | --- |
| [01-architecture.md](05-capability-first-runtime/01-architecture.md) | composition boundaries, dependency direction, runtime foundation, state ownership, subagents, Environment, and host assembly |
| [02-capability-catalog.md](05-capability-first-runtime/02-capability-catalog.md) | target leaf and preset capabilities, existing toolset/filter mapping, and internal adapter contracts |
| [03-instructions-and-request-pipeline.md](05-capability-first-runtime/03-instructions-and-request-pipeline.md) | instruction ownership, dynamic catalogs, immutable history processing, and capability ordering |
| [04-native-steering-and-claw-queue.md](05-capability-first-runtime/04-native-steering-and-claw-queue.md) | `AgentRun.enqueue`, canonical history, active-run registry, Claw API/runtime/events, and terminal races |
| [05-migration-and-validation.md](05-capability-first-runtime/05-migration-and-validation.md) | breaking 2.0 cutover, end-to-end implementation blocks, release criteria, and test matrix |
| [06-yaacli-durable-sessions.md](05-capability-first-runtime/06-yaacli-durable-sessions.md) | SDK segment harness, host-owned coordinators, YAACLI `SessionStore`, steering/HITL persistence, and interruption semantics |
| [07-subagent-runtime.md](05-capability-first-runtime/07-subagent-runtime.md) | native `AgentSpec`-based portable subagents, plan resolution, registry, execution service, and SDK/YAACLI/Claw drivers |

## Scope

The design covers:

- SDK `create_agent()` composition;
- all SDK feature toolsets and `BaseTool` feature lists;
- all default and optional SDK history filters;
- Tool Search, Tool Proxy, Skills, MCP, CodeAct, and delegation;
- task/note managers and other run-local or durable state owners;
- main-agent, named-subagent, and self-fork composition;
- Environment and resource contributions;
- YAACLI and YA Claw runtime assembly;
- SDK steering and YA Claw immediate/queued active-run input;
- logical-run user input retention across compact, handoff, native recovery, and durable
  replay;
- YAACLI product sessions and host-coordinated segment execution; and
- portable subagent resolution and host-specific execution across SDK, YAACLI, and
  YA Claw.

The design does not:

- replace `Environment`, `FileOperator`, or `Shell`;
- move filesystem or shell authority into capabilities;
- reimplement Pydantic AI primitives;
- turn every helper or lifecycle extension into a capability;
- make an active Claw run resumable after process loss;
- replace Claw's SQL run scheduler with YAACLI's local coordinator;
- make a Pydantic AI durable backend the product `SessionStore`;
- add a generic workflow framework to the SDK; or
- make `DispatchMode.QUEUE` select active-run steering priority.

## Core Composition

```python
runtime = create_agent(
    model,
    capabilities=[
        RuntimeFoundationCapability(...),
        FilesystemCapability(...),
        ShellCapability(...),
        TaskCapability(...),
        SkillsCapability(...),
        DelegationCapability(...),
    ],
)
```

The reference runtime foundation is a public `CombinedCapability` explicitly present
in `capabilities=`, not an implicit list of filters. `create_agent()` does not inject
it. Advanced hosts may compose its leaf capabilities directly while preserving the
required runtime contracts.

Custom Pydantic AI capabilities pass through unchanged. Raw function tools use
`Capability(tools=[...])`; external toolsets use
`pydantic_ai.capabilities.Toolset(...)`. Feature-level SDK APIs do not expose raw
SDK toolsets.

## Naming Contract

“Capability” means a Pydantic AI capability unless explicitly qualified.

The existing non-composable SDK `ModelFeature` enum becomes `ModelFeature`.
`BaseTool.tags` remains tool metadata. `BaseLifecycleExtension` remains a runtime
orchestration extension and is not part of the Pydantic AI capability lifecycle.

## Relationship to Existing Specifications

- [01-lifecycle-extensions.md](01-lifecycle-extensions.md) retains its division between
  agent-run capabilities and host/runtime lifecycle extensions. The capability-first
  architecture replaces its separate `pre_capabilities` and `capabilities` lists with
  one ordered capability list.
- [03-codeact-programs.md](03-codeact-programs.md) retains its security and execution
  boundaries. CodeAct is composed through `CodeActCapability`.
- [06-capability-plugins](06-capability-plugins/README.md) defines SDK-owned per-type
  entry-point and explicit-import discovery, an immutable custom-type catalog, and the
  host consumer boundary. Type discovery does not create a second behavior-composition
  surface.
- The 2.0 native steering contract in
  [04-native-steering-and-claw-queue.md](05-capability-first-runtime/04-native-steering-and-claw-queue.md)
  supersedes the MessageBus steering section of
  [04-runtime-host-contracts.md](04-runtime-host-contracts.md).
