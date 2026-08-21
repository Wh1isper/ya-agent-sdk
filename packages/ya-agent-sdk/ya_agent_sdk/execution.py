"""Host-neutral execution harness for one native agent segment.

The harness deliberately owns no scheduler, queue, persistence, or retry policy.
Durable applications compose it with their own coordinator and store, while
in-process applications can execute a segment directly.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import Generic

from pydantic_ai import AgentRun, DeferredToolRequests, DeferredToolResults, UsageLimits
from pydantic_ai._enqueue import EnqueueContent, PendingMessagePriority
from pydantic_ai.messages import ModelMessage

from ya_agent_sdk.agents.main import (
    AgentCompleteHook,
    AgentRuntime,
    AgentStartHook,
    AgentStreamer,
    EventHook,
    NodeHook,
    OutputT,
    ResumePromptFactory,
    RuntimeReadyHook,
    UserPromptFactory,
    UserPromptT,
    stream_agent,
)
from ya_agent_sdk.context import ResumableState, StreamEvent
from ya_agent_sdk.inputs import EnqueueReceipt, InputOrigin, RunInputLedger
from ya_agent_sdk.usage import UsageSnapshot
from ya_agent_sdk.utils import AgentDepsT, EnvT

AgentEventSink = Callable[[StreamEvent], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class AgentSegmentRequest(Generic[AgentDepsT, OutputT, EnvT]):
    """Inputs and process-local policies for one native agent segment.

    A segment ends when Pydantic AI returns either the configured terminal
    output or ``DeferredToolRequests``. Continuing a deferred interaction is a
    new segment with the prior checkpoint messages and host-supplied results.
    """

    user_prompt: UserPromptT | None = None
    user_prompt_factory: UserPromptFactory[AgentDepsT, OutputT, EnvT] | None = None
    message_history: Sequence[ModelMessage] | None = None
    deferred_tool_results: DeferredToolResults | None = None
    usage_limits: UsageLimits | None = None
    on_runtime_ready: RuntimeReadyHook[AgentDepsT, OutputT, EnvT] | None = None
    on_agent_start: AgentStartHook[AgentDepsT, OutputT, EnvT] | None = None
    on_agent_complete: AgentCompleteHook[AgentDepsT, OutputT, EnvT] | None = None
    pre_node_hook: NodeHook[AgentDepsT, OutputT] | None = None
    post_node_hook: NodeHook[AgentDepsT, OutputT] | None = None
    pre_event_hook: EventHook[AgentDepsT, OutputT] | None = None
    post_event_hook: EventHook[AgentDepsT, OutputT] | None = None
    raise_on_error: bool = True
    resume_on_error: bool | None = None
    resume_max_attempts: int | None = None
    transport_resume_max_attempts: int | None = None
    resume_prompt: UserPromptT | None = None
    resume_prompt_factory: ResumePromptFactory | None = None
    emit_lifecycle_events: bool = True


@dataclass(frozen=True, slots=True)
class AgentExecutionCheckpoint:
    """Canonical SDK artifacts at a completed native segment boundary."""

    messages: tuple[ModelMessage, ...]
    state: ResumableState
    usage: UsageSnapshot

    @property
    def input_ledger(self) -> RunInputLedger:
        """Return a detached snapshot of accepted/applied logical input."""
        return self.state.run_input_ledger.model_copy(deep=True)


class AgentSegmentStatus(StrEnum):
    completed = "completed"
    suspended = "suspended"


@dataclass(frozen=True, slots=True)
class AgentSegmentOutcome(Generic[OutputT]):
    """Terminal or host-suspended result of one completed segment."""

    status: AgentSegmentStatus
    output: OutputT | DeferredToolRequests
    checkpoint: AgentExecutionCheckpoint

    @property
    def deferred_requests(self) -> DeferredToolRequests | None:
        if isinstance(self.output, DeferredToolRequests):
            return self.output
        return None


class AgentSegment(Generic[AgentDepsT, OutputT, EnvT]):
    """Streaming handle for a segment executed by ``AgentExecutionHarness``."""

    def __init__(
        self,
        runtime: AgentRuntime[AgentDepsT, OutputT, EnvT],
        streamer: AgentStreamer[AgentDepsT, OutputT],
    ) -> None:
        self._runtime = runtime
        self._streamer = streamer

    @property
    def streamer(self) -> AgentStreamer[AgentDepsT, OutputT]:
        """Underlying low-level stream for compatibility-sensitive adapters."""
        return self._streamer

    @property
    def run(self) -> AgentRun[AgentDepsT, OutputT] | None:
        return self._streamer.run

    @property
    def exception(self) -> BaseException | None:
        return self._streamer.exception

    def __aiter__(self) -> AgentSegment[AgentDepsT, OutputT, EnvT]:
        return self

    async def __anext__(self) -> StreamEvent:
        return await self._streamer.__anext__()

    async def enqueue(
        self,
        *content: EnqueueContent,
        priority: PendingMessagePriority = "asap",
        origin: InputOrigin = InputOrigin.user,
        input_id: str | None = None,
    ) -> EnqueueReceipt:
        return await self._streamer.enqueue(
            *content,
            priority=priority,
            origin=origin,
            input_id=input_id,
        )

    def interrupt(self) -> None:
        self._streamer.interrupt()

    def raise_if_exception(self) -> None:
        self._streamer.raise_if_exception()

    def recoverable_messages(self) -> list[ModelMessage]:
        return self._streamer.recoverable_messages()

    def checkpoint(self) -> AgentExecutionCheckpoint:
        """Capture stable artifacts after the native run produced a result."""
        run = self._streamer.run
        if run is None or run.result is None:
            raise RuntimeError("Agent segment has not reached a stable result boundary")
        ctx = self._runtime.ctx
        return AgentExecutionCheckpoint(
            messages=tuple(self._streamer.recoverable_messages()),
            state=ctx.export_state(include_usage_ledger=True),
            usage=ctx.build_usage_snapshot(),
        )

    def outcome(self) -> AgentSegmentOutcome[OutputT]:
        """Build the completed/suspended outcome at the segment boundary."""
        run = self._streamer.run
        if run is None or run.result is None:
            raise RuntimeError("Agent segment has not reached a stable result boundary")
        output = run.result.output
        status = (
            AgentSegmentStatus.suspended if isinstance(output, DeferredToolRequests) else AgentSegmentStatus.completed
        )
        return AgentSegmentOutcome(
            status=status,
            output=output,
            checkpoint=self.checkpoint(),
        )


class AgentExecutionHarness:
    """Stateless in-process harness for native Pydantic AI segments."""

    @asynccontextmanager
    async def stream_segment(
        self,
        runtime: AgentRuntime[AgentDepsT, OutputT, EnvT],
        request: AgentSegmentRequest[AgentDepsT, OutputT, EnvT],
    ) -> AsyncIterator[AgentSegment[AgentDepsT, OutputT, EnvT]]:
        async with stream_agent(
            runtime,
            user_prompt=request.user_prompt,
            user_prompt_factory=request.user_prompt_factory,
            message_history=request.message_history,
            deferred_tool_results=request.deferred_tool_results,
            usage_limits=request.usage_limits,
            on_runtime_ready=request.on_runtime_ready,
            on_agent_start=request.on_agent_start,
            on_agent_complete=request.on_agent_complete,
            pre_node_hook=request.pre_node_hook,
            post_node_hook=request.post_node_hook,
            pre_event_hook=request.pre_event_hook,
            post_event_hook=request.post_event_hook,
            raise_on_error=request.raise_on_error,
            resume_on_error=request.resume_on_error,
            resume_max_attempts=request.resume_max_attempts,
            transport_resume_max_attempts=request.transport_resume_max_attempts,
            resume_prompt=request.resume_prompt,
            resume_prompt_factory=request.resume_prompt_factory,
            emit_lifecycle_events=request.emit_lifecycle_events,
        ) as streamer:
            yield AgentSegment(runtime, streamer)

    async def execute_segment(
        self,
        runtime: AgentRuntime[AgentDepsT, OutputT, EnvT],
        request: AgentSegmentRequest[AgentDepsT, OutputT, EnvT],
        *,
        event_sink: AgentEventSink | None = None,
    ) -> AgentSegmentOutcome[OutputT]:
        """Consume one segment in process, optionally forwarding every event."""
        async with self.stream_segment(runtime, request) as segment:
            async for event in segment:
                if event_sink is not None:
                    await event_sink(event)
            segment.raise_if_exception()
            return segment.outcome()


__all__ = [
    "AgentEventSink",
    "AgentExecutionCheckpoint",
    "AgentExecutionHarness",
    "AgentSegment",
    "AgentSegmentOutcome",
    "AgentSegmentRequest",
    "AgentSegmentStatus",
]
