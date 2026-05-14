# Events

Custom events for sideband streaming and lifecycle tracking.

## Overview

The event system provides two categories of events:

1. **Lifecycle Events**: Automatic execution tracking emitted by `stream_agent`
2. **Sideband Events**: Tool/feature-specific events (compact, handoff, subagent, etc.)

All events inherit from `AgentEvent` and are delivered via `StreamEvent` wrapper:

```python
from ya_agent_sdk.context import StreamEvent
from ya_agent_sdk.events import AgentEvent

@dataclass
class StreamEvent:
    agent_id: str      # Unique agent identifier
    agent_name: str    # Human-readable agent name
    event: AgentEvent  # The actual event
```

## Lifecycle Events

Lifecycle events are automatically emitted by `stream_agent` when `emit_lifecycle_events=True` (default).

### Event Flow

```mermaid
flowchart TB
    A[AgentExecutionStartEvent] --> B[ModelRequestStartEvent]
    B --> |"thinking..."| C[ModelRequestCompleteEvent]
    C --> |"has tool calls"| D[ToolCallsStartEvent]
    D --> |"running tools..."| E[ToolCallsCompleteEvent]
    E --> |"more requests"| B
    C --> |"no tool calls"| F[AgentExecutionCompleteEvent]
    E --> |"done"| F
    A --> |"error"| G[AgentExecutionFailedEvent]
    B --> |"error"| G
    D --> |"error"| G
```

### Event Types

| Event                         | When Emitted                    | Key Fields                              |
| ----------------------------- | ------------------------------- | --------------------------------------- |
| `AgentExecutionStartEvent`    | Agent execution begins          | `user_prompt`, `message_history_count`  |
| `ModelRequestStartEvent`      | Model request starts (thinking) | `loop_index`, `message_count`           |
| `ModelRequestCompleteEvent`   | Model response received         | `loop_index`, `duration_seconds`        |
| `ToolCallsStartEvent`         | Tool execution starts           | `loop_index`                            |
| `ToolCallsCompleteEvent`      | Tool execution completes        | `loop_index`, `duration_seconds`        |
| `AgentExecutionCompleteEvent` | Agent execution completes       | `total_loops`, `total_duration_seconds` |
| `AgentExecutionFailedEvent`   | Agent execution fails           | `error`, `error_type`, `total_loops`    |

### Loop Index

The `loop_index` field (zero-based) tracks iteration count:

- Loop 0: First model request + optional tool calls
- Loop 1: Second model request + optional tool calls
- ...

All events within the same loop share the same `loop_index`.

### Usage Example

```python
from ya_agent_sdk.agents import create_agent, stream_agent
from ya_agent_sdk.events import (
    AgentExecutionStartEvent,
    AgentExecutionCompleteEvent,
    ModelRequestStartEvent,
    ModelRequestCompleteEvent,
    ToolCallsStartEvent,
    ToolCallsCompleteEvent,
)

runtime = create_agent("openai-chat:gpt-4o")

async with stream_agent(runtime, "Hello") as streamer:
    async for stream_event in streamer:
        event = stream_event.event

        if isinstance(event, AgentExecutionStartEvent):
            print(f"Starting execution with {event.message_history_count} history messages")

        elif isinstance(event, ModelRequestStartEvent):
            print(f"Loop {event.loop_index}: Thinking...")

        elif isinstance(event, ModelRequestCompleteEvent):
            print(f"  Response received in {event.duration_seconds:.2f}s")

        elif isinstance(event, ToolCallsStartEvent):
            print(f"  Running tools...")

        elif isinstance(event, ToolCallsCompleteEvent):
            print(f"  Tools completed in {event.duration_seconds:.2f}s")

        elif isinstance(event, AgentExecutionCompleteEvent):
            print(f"Completed: {event.total_loops} loops in {event.total_duration_seconds:.2f}s")
```

### Disabling Lifecycle Events

Set `emit_lifecycle_events=False` for custom tracking or cleaner output:

```python
async with stream_agent(
    runtime,
    "Hello",
    emit_lifecycle_events=False,
) as streamer:
    async for event in streamer:
        pass  # Only model events, no lifecycle events
```

## Sideband Events

Sideband events are emitted by specific tools or features to communicate status.

### Compact Events

Emitted during context compaction (summarizing message history):

| Event                  | Description                       |
| ---------------------- | --------------------------------- |
| `CompactStartEvent`    | Compaction started                |
| `CompactCompleteEvent` | Compaction succeeded with summary |
| `CompactFailedEvent`   | Compaction failed with error      |

```python
from ya_agent_sdk.events import CompactStartEvent, CompactCompleteEvent, CompactFailedEvent

if isinstance(event, CompactCompleteEvent):
    print(f"Compacted {event.original_message_count} -> {event.compacted_message_count} messages")
```

### Handoff Events

Emitted during context handoff (clearing context with summary):

| Event                  | Description       |
| ---------------------- | ----------------- |
| `HandoffStartEvent`    | Handoff started   |
| `HandoffCompleteEvent` | Handoff succeeded |
| `HandoffFailedEvent`   | Handoff failed    |

```python
from ya_agent_sdk.events import HandoffCompleteEvent

if isinstance(event, HandoffCompleteEvent):
    print(f"Handoff complete: {event.new_message_count} messages preserved")
```

### Subagent Events

Emitted when delegating to subagents:

| Event                   | Description                  |
| ----------------------- | ---------------------------- |
| `SubagentStartEvent`    | Subagent execution started   |
| `SubagentCompleteEvent` | Subagent execution completed |

```python
from ya_agent_sdk.events import SubagentStartEvent, SubagentCompleteEvent

if isinstance(event, SubagentStartEvent):
    print(f"Delegating to {event.agent_name}: {event.prompt_preview}")
```

### Usage Snapshot Events

`UsageSnapshotEvent` is emitted whenever the unified per-run usage ledger changes. It is the realtime usage surface for clients and includes main-agent usage plus reported internal model usage such as subagents, compaction, and shell review.

| Field                   | Description                             |
| ----------------------- | --------------------------------------- |
| `snapshot.run_id`       | Current run ID                          |
| `snapshot.total_usage`  | Cumulative `RunUsage` across all agents |
| `snapshot.agent_usages` | Usage grouped by agent ID               |
| `snapshot.model_usages` | Usage grouped by model ID               |
| `source`                | Component that triggered the snapshot   |

```python
from ya_agent_sdk.events import UsageSnapshotEvent

if isinstance(event, UsageSnapshotEvent) and event.snapshot:
    print(event.snapshot.total_usage)
```

### Tool Search Events

Emitted during `ToolSearchToolSet` initialization to report namespace (wrapped toolset) connection status. This event fires once on the first `get_tools()` call after all wrapped toolsets have been initialized.

| Event                 | Description                              | Key Fields         |
| --------------------- | ---------------------------------------- | ------------------ |
| `ToolSearchInitEvent` | Namespace initialization status reported | `namespace_status` |

The `namespace_status` field is a `dict[str, NamespaceStatus]` where each key is a namespace ID and the value is one of:

| Status      | Description                                               |
| ----------- | --------------------------------------------------------- |
| `connected` | Namespace initialized successfully and is available       |
| `skipped`   | Namespace failed to initialize but was optional (skipped) |
| `error`     | Namespace was connected but encountered a runtime error   |

Required namespaces that fail initialization raise during `__aenter__` and do not appear in this event.

```python
from ya_agent_sdk.events import ToolSearchInitEvent, NamespaceStatus

if isinstance(event, ToolSearchInitEvent):
    for ns, status in event.namespace_status.items():
        print(f"  {ns}: {status}")
```

### File Change Events

Emitted when files are created, modified, moved, or copied. One event per tool call, may contain multiple file changes. Only emitted on **successful** operations; failed operations do not produce events.

| Event             | Description                  | Key Fields             |
| ----------------- | ---------------------------- | ---------------------- |
| `FileChangeEvent` | File system changes occurred | `changes`, `tool_name` |

The `tool_name` field indicates which tool triggered the changes (e.g., `"edit"`, `"multi_edit"`, `"write"`, `"move"`, `"copy"`).

Each entry in `changes` is a `FileChange` dataclass:

| Field          | Type                    | Description                                              |
| -------------- | ----------------------- | -------------------------------------------------------- |
| `path`         | `str`                   | File path that changed (relative to working dir)         |
| `action`       | `FileChangeAction`      | Type of change: `created`, `modified`, `moved`, `copied` |
| `destination`  | `str \| None`           | Target path for move/copy operations                     |
| `replacements` | `list[TextReplacement]` | Structured text replacements for edit operations         |

`TextReplacement` contains `old_string` and `new_string` fields (empty `old_string` indicates new file creation).

```python
from ya_agent_sdk.events import FileChangeEvent

if isinstance(event, FileChangeEvent):
    print(f"[{event.tool_name}] {len(event.changes)} file(s) changed")
    for change in event.changes:
        if change.destination:
            print(f"  {change.action}: {change.path} -> {change.destination}")
        else:
            print(f"  {change.action}: {change.path}")
```

### Task Events

Emitted when task state changes (create, update, list). Each event contains a **full snapshot** of all tasks, enabling stateless rendering without tracking incremental changes:

| Event       | Description                      | Key Fields |
| ----------- | -------------------------------- | ---------- |
| `TaskEvent` | Task created, updated, or listed | `tasks`    |

The `tasks` field is a list of `TaskInfo` dataclasses:

| Field         | Type          | Description                                            |
| ------------- | ------------- | ------------------------------------------------------ |
| `id`          | `str`         | Task identifier                                        |
| `subject`     | `str`         | Task title                                             |
| `description` | `str`         | Task description                                       |
| `status`      | `str`         | Current status (`pending`, `in_progress`, `completed`) |
| `active_form` | `str \| None` | Present progressive form shown during `in_progress`    |
| `owner`       | `str \| None` | Task owner/assignee                                    |
| `blocked_by`  | `list[str]`   | Task IDs that block this task                          |
| `blocks`      | `list[str]`   | Task IDs that this task blocks                         |

```python
from ya_agent_sdk.events import TaskEvent

if isinstance(event, TaskEvent):
    for task in event.tasks:
        print(f"#{task.id} [{task.status}] {task.subject}")
```

### Note Events

Emitted when note state changes (set, delete). Each event contains a **full snapshot** of all note entries for stateless rendering:

| Event       | Description               | Key Fields |
| ----------- | ------------------------- | ---------- |
| `NoteEvent` | Note entry set or deleted | `entries`  |

The `entries` field is a `dict[str, str]` mapping keys to values.

```python
from ya_agent_sdk.events import NoteEvent

if isinstance(event, NoteEvent):
    for key, value in event.entries.items():
        print(f"{key}: {value}")
```

### Message Bus Events

Emitted when messages are received from the message bus:

| Event                  | Description                    |
| ---------------------- | ------------------------------ |
| `MessageReceivedEvent` | Messages injected into context |

```python
from ya_agent_sdk.events import MessageReceivedEvent

if isinstance(event, MessageReceivedEvent):
    for msg in event.messages:
        print(f"Received from {msg.source}: {msg.rendered_content}")
```

## Event Correlation

Events can be correlated using `event_id`:

```python
# All lifecycle events in a run share the same event_id (ctx.run_id)
start = AgentExecutionStartEvent(event_id="run-123", ...)
model_start = ModelRequestStartEvent(event_id="run-123", loop_index=0)
complete = AgentExecutionCompleteEvent(event_id="run-123", ...)

# Sideband event pairs also share event_id
compact_start = CompactStartEvent(event_id="compact-456", ...)
compact_complete = CompactCompleteEvent(event_id="compact-456", ...)
```

## Custom Events

Create custom events by subclassing `AgentEvent`:

```python
from dataclasses import dataclass
from ya_agent_sdk.events import AgentEvent

@dataclass
class MyCustomEvent(AgentEvent):
    """Custom event for my feature."""
    custom_field: str = ""

# Emit via context
await ctx.emit_event(MyCustomEvent(event_id="custom-001", custom_field="value"))
```

## Type Aliases

For convenience, a union type is provided for lifecycle events:

```python
from ya_agent_sdk.events import LifecycleEvent

# LifecycleEvent = (
#     AgentExecutionStartEvent
#     | AgentExecutionCompleteEvent
#     | AgentExecutionFailedEvent
#     | ModelRequestStartEvent
#     | ModelRequestCompleteEvent
#     | ToolCallsStartEvent
#     | ToolCallsCompleteEvent
# )

def handle_lifecycle(event: LifecycleEvent) -> None:
    match event:
        case AgentExecutionStartEvent():
            print("Started")
        case AgentExecutionCompleteEvent():
            print("Completed")
        # ...
```
