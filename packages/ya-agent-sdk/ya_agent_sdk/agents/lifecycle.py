"""Lifecycle extension primitives for agent runtimes."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Generic

from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import RunUsage

from ya_agent_sdk.utils import AgentDepsT, EnvT


class ContextHandoffSource(StrEnum):
    """Source that completed a context handoff."""

    COMPACT = "compact"
    SUMMARIZE_TOOL = "summarize_tool"


@dataclass
class ContextHandoffCompleteContext(Generic[AgentDepsT]):
    """Context passed when history is replaced by a compact or handoff summary."""

    event_id: str
    deps: AgentDepsT
    source: ContextHandoffSource
    original_messages: list[ModelMessage]
    trimmed_messages: list[ModelMessage]
    handoff_messages: list[ModelMessage]
    summary_markdown: str
    usage: RunUsage | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompactStartContext(Generic[AgentDepsT]):
    """Context passed when compaction starts."""

    event_id: str
    deps: AgentDepsT
    original_messages: list[ModelMessage]


@dataclass
class CompactCompleteContext(ContextHandoffCompleteContext[AgentDepsT]):
    """Context passed when automatic compaction completes."""

    compacted_messages: list[ModelMessage] = field(default_factory=list)
    condense_result: Any = None


@dataclass
class CompactFailedContext(Generic[AgentDepsT]):
    """Context passed when compaction fails."""

    event_id: str
    deps: AgentDepsT
    original_messages: list[ModelMessage]
    trimmed_messages: list[ModelMessage] | None
    error: BaseException


@dataclass
class AgentErrorContext(Generic[AgentDepsT, EnvT]):
    """Context passed when stream execution raises an exception."""

    runtime: Any
    agent_info: Any
    output_queue: Any
    error: BaseException


class BaseLifecycleExtension(Generic[AgentDepsT, EnvT]):
    """Base class for optional runtime lifecycle extension hooks."""

    name = "base"

    async def on_runtime_ready(self, ctx: Any) -> None:
        pass

    async def on_agent_start(self, ctx: Any) -> None:
        pass

    async def on_before_node(self, ctx: Any) -> None:
        pass

    async def on_after_node(self, ctx: Any) -> None:
        pass

    async def on_before_event(self, ctx: Any) -> None:
        pass

    async def on_after_event(self, ctx: Any) -> None:
        pass

    async def on_agent_complete(self, ctx: Any) -> None:
        pass

    async def on_agent_error(self, ctx: AgentErrorContext[AgentDepsT, EnvT]) -> None:
        pass

    async def on_context_handoff_complete(self, ctx: ContextHandoffCompleteContext[AgentDepsT]) -> None:
        pass

    async def on_compact_start(self, ctx: CompactStartContext[AgentDepsT]) -> None:
        pass

    async def on_compact_complete(self, ctx: CompactCompleteContext[AgentDepsT]) -> None:
        pass

    async def on_compact_failed(self, ctx: CompactFailedContext[AgentDepsT]) -> None:
        pass


LifecycleExtension = BaseLifecycleExtension[Any, Any]
CompactLifecycleCallback = Callable[[CompactCompleteContext[Any]], Awaitable[None]]


async def run_extension_method(
    extensions: Sequence[BaseLifecycleExtension[Any, Any]],
    method_name: str,
    ctx: Any,
    *,
    logger: Any | None = None,
) -> None:
    """Run a lifecycle method on each extension."""

    _ = logger
    for extension in extensions:
        method = getattr(extension, method_name, None)
        if method is not None:
            await method(ctx)


async def run_compact_complete_callbacks(
    callbacks: Sequence[CompactLifecycleCallback] | None,
    ctx: CompactCompleteContext[Any],
) -> None:
    """Run compact callbacks in order."""

    for callback in callbacks or ():
        await callback(ctx)
