"""Immutable model-request helpers shared by foundation capabilities."""

from __future__ import annotations

import copy
import inspect
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from pydantic_ai import RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import ModelRequestContext

DepsT = TypeVar("DepsT")
HistoryProcessor = Callable[
    [RunContext[DepsT], list[ModelMessage]],
    list[ModelMessage] | Awaitable[list[ModelMessage]],
]


def copy_request_context(
    request_context: ModelRequestContext,
    *,
    messages: list[ModelMessage] | None = None,
) -> ModelRequestContext:
    """Clone a request envelope while preserving upstream read-only metadata."""
    cloned = copy.copy(request_context)
    cloned.messages = messages if messages is not None else copy.deepcopy(request_context.messages)
    return cloned


async def apply_history_processor(
    processor: HistoryProcessor[DepsT],
    ctx: RunContext[DepsT],
    request_context: ModelRequestContext,
) -> ModelRequestContext:
    """Apply a legacy algorithm copy-on-write at the canonical request boundary."""
    messages = copy.deepcopy(request_context.messages)
    processed = processor(ctx, messages)
    if inspect.isawaitable(processed):
        processed = await processed
    if processed == request_context.messages:
        return request_context
    return copy_request_context(request_context, messages=processed)


async def persist_history_projection(
    processor: HistoryProcessor[DepsT],
    ctx: RunContext[DepsT],
    request_context: ModelRequestContext,
    handler: Callable[[ModelRequestContext], Awaitable[Any]],
) -> Any:
    """Apply a processor to canonical history so future requests replay the same prefix."""
    messages = request_context.messages
    processed = processor(ctx, messages)
    if inspect.isawaitable(processed):
        processed = await processed
    if processed is not messages:
        messages[:] = processed
    return await handler(request_context)


async def project_history(
    processor: HistoryProcessor[DepsT],
    ctx: RunContext[DepsT],
    request_context: ModelRequestContext,
    handler: Callable[[ModelRequestContext], Awaitable[Any]],
) -> Any:
    """Apply a processor to a request-only copy and invoke the next handler."""
    projected = await apply_history_processor(processor, ctx, request_context)
    return await handler(projected)
