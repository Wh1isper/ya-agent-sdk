# 00 - Overview

## Definition

YA Claw is a single-node runtime web service for `ya-agent-sdk`.

It provides a durable local execution shell around SDK agent construction and streaming primitives with:

- reusable execution profiles
- one configured default workspace directory with optional session-scoped mount sets
- resolver-driven workspace binding construction
- explicit runtime assembly from binding to agent runtime
- resumable sessions and runs
- queued-run execution before active execution begins
- in-process async task coordination
- live event streaming for the current node
- committed session persistence
- a first-party web shell

## Goals

### Product Goals

- make local and self-hosted deployment the default operating model
- keep runtime file and shell access bounded by declared workspace bindings
- preserve SDK capabilities such as continuation, subagents, compact, and streaming
- keep the runtime small enough to understand and evolve quickly
- keep active execution management inside one process for the single-node target
- support API, schedule, heartbeat, and bridge ingress through one queued-run execution model
- run schedule dispatchers, heartbeat dispatchers, and bridge adapters as supervised runtime components in the single-node service shape

### Non-Goals

- hosted platform concerns
- organization-level control plane design
- runtime-managed project catalogs
- distributed runtime scheduling
- multi-node execution ownership

## Top-level Architecture

```mermaid
flowchart TB
    subgraph Clients
        WEB[YA Claw Web]
        API_CLIENT[API Clients]
        CLI[ya-claw CLI]
        IM[IM Platforms]
    end

    subgraph Runtime[YA Claw Service]
        API[HTTP API]
        SESS[Session Controller]
        RUNS[Run Controller]
        SUP[ExecutionSupervisor]
        REG[ExecutionRegistry]
        BUF[Runtime State and Event Buffers]
        PROF[ProfileResolver]
        PROJ[WorkspaceProvider]
        ENVF[EnvironmentFactory]
        BUILD[ClawRuntimeBuilder]
        COORD[RunCoordinator]
        SCHED[ScheduleDispatcher]
        HEART[HeartbeatDispatcher]
        BRIDGE[BridgeSupervisor]
        BADAPT[Bridge Adapters]
    end

    subgraph SDK[ya-agent-sdk]
        AGENT[create_agent]
        STREAM[stream_agent]
        CTX[ClawAgentContext]
        ENV[Environment]
    end

    subgraph Storage[Runtime Storage]
        SQL[(SQLite / PostgreSQL)]
        MEM[(In-Process Memory)]
        FS[(Local Filesystem)]
        YAML[Profile Seed YAML]
    end

    WEB --> API
    API_CLIENT --> API
    CLI --> API
    IM --> BADAPT
    BADAPT --> BRIDGE
    BRIDGE --> SESS
    BRIDGE --> RUNS

    API --> SESS
    API --> RUNS
    SESS --> SQL
    RUNS --> SQL
    RUNS --> SUP
    SESS --> SUP
    SCHED --> SESS
    SCHED --> RUNS
    SCHED --> SUP
    HEART --> SESS
    HEART --> RUNS
    HEART --> SUP
    BRIDGE --> SUP

    SUP --> REG
    SUP --> COORD
    COORD --> PROF
    PROF --> SQL
    PROF --> YAML
    COORD --> PROJ
    COORD --> ENVF
    COORD --> BUILD
    BUILD --> CTX
    BUILD --> AGENT
    COORD --> STREAM
    ENVF --> ENV

    COORD --> SQL
    COORD --> FS
    REG --> MEM
    BUF --> MEM
```

## Runtime Boundary

| Concern                       | Owner                 |
| ----------------------------- | --------------------- |
| Agent execution primitives    | `ya-agent-sdk`        |
| Workspace binding enforcement | YA Claw               |
| Profile resolution            | YA Claw               |
| Runtime assembly              | YA Claw               |
| Session persistence           | YA Claw               |
| Run orchestration             | YA Claw               |
| Active execution tracking     | YA Claw               |
| Event delivery                | YA Claw               |
| Committed session persistence | YA Claw               |
| Default workspace selection   | YA Claw configuration |
| Channel transport             | bridge adapter        |
| LLM provider interaction      | SDK + model provider  |

## Core Runtime Objects

The architecture revolves around a small set of runtime objects:

- **Execution Profile**: reusable runtime configuration for model, prompt, tools, approvals, and policy
- **Workspace Binding**: declarative workspace view for one run, including mount set, default host path, virtual paths, cwd, path policy, and metadata
- **Environment Factory**: runtime component that turns a workspace binding into a concrete SDK `Environment`
- **ClawAgentContext**: YA Claw-specific `AgentContext` subclass that carries run, session, profile, workspace, and source metadata
- **ClawRuntimeBuilder**: runtime component that assembles `Environment`, `ClawAgentContext`, toolsets, and agent configuration into one SDK runtime
- **Execution Supervisor**: in-process manager that claims queued runs, starts coordinators, and tracks active execution
- **Run Coordinator**: short-lived per-run executor from claim to terminal state
- **Session**: durable conversational continuity
- **Run**: one execution attempt inside a session
- **Schedule**: agent-manageable or user-manageable timer resource that creates or steers runs
- **Heartbeat**: runtime-owned timer resource that creates isolated operational runs from `HEARTBEAT.md` guidance
- **Execution Registry**: in-memory registry of active run tasks and control signals

## Design Principle

YA Claw owns durable execution intent, runtime assembly, and active execution management.
Applications own project identity, user-facing folder registries, and high-level ingress context.
