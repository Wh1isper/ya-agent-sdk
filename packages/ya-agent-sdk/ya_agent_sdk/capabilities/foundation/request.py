"""Request-envelope capabilities for media and runtime context projection."""

from __future__ import annotations

import copy
import inspect
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models import ModelRequestContext

from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.filters.auto_load_files import _build_file_inspection_prompt
from ya_agent_sdk.filters.capability import filter_by_capability
from ya_agent_sdk.filters.environment_instructions import create_environment_instructions_filter
from ya_agent_sdk.filters.image import (
    compress_large_images,
    drop_extra_images,
    drop_extra_videos,
    drop_gif_images,
    split_large_images,
)
from ya_agent_sdk.filters.runtime_instructions import inject_runtime_instructions

from ._request import copy_request_context, project_history

ModelHandler = Callable[[ModelRequestContext], Awaitable[Any]]


async def _project_media(
    ctx: RunContext[AgentContext],
    messages: list[ModelMessage],
) -> list[ModelMessage]:
    processors = (
        split_large_images,
        compress_large_images,
        drop_extra_images,
        drop_gif_images,
        drop_extra_videos,
        filter_by_capability,
    )
    current = messages
    for processor in processors:
        result = processor(ctx, current)
        current = await result if inspect.isawaitable(result) else result
    return current


@dataclass(kw_only=True)
class MediaCompatibilityCapability(AbstractCapability[AgentContext]):
    """Project media into the active provider's request envelope only."""

    id: str | None = "media_compatibility"

    def get_ordering(self) -> CapabilityOrdering:
        from ya_agent_sdk.capabilities.features.shell import ShellCapability

        return CapabilityOrdering(
            wraps=(
                FileInspectionCapability,
                ShellCapability,
                EnvironmentContextCapability,
                RuntimeContextCapability,
            )
        )

    async def wrap_model_request(
        self,
        ctx: RunContext[AgentContext],
        *,
        request_context: ModelRequestContext,
        handler: ModelHandler,
    ) -> Any:
        return await project_history(_project_media, ctx, request_context, handler)


@dataclass(kw_only=True)
class FileInspectionCapability(AbstractCapability[AgentContext]):
    """Inject a one-shot file-inspection reminder and commit after success."""

    id: str | None = "file_inspection"

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(wraps=(EnvironmentContextCapability, RuntimeContextCapability))

    async def wrap_model_request(
        self,
        ctx: RunContext[AgentContext],
        *,
        request_context: ModelRequestContext,
        handler: ModelHandler,
    ) -> Any:
        pending = tuple(ctx.deps.auto_load_files)
        if not pending:
            return await handler(request_context)

        messages = copy.deepcopy(request_context.messages)
        last_request = next(
            (message for message in reversed(messages) if isinstance(message, ModelRequest)),
            None,
        )
        if last_request is None:
            return await handler(request_context)

        last_request.parts = [
            *last_request.parts,
            UserPromptPart(content=_build_file_inspection_prompt(list(pending))),
        ]
        response = await handler(copy_request_context(request_context, messages=messages))

        remaining = list(ctx.deps.auto_load_files)
        for path in pending:
            with suppress(ValueError):
                remaining.remove(path)
        ctx.deps.auto_load_files = remaining
        return response


@dataclass(kw_only=True)
class EnvironmentContextCapability(AbstractCapability[AgentContext]):
    """Decorate one model request with current Environment instructions."""

    id: str | None = "environment_context"

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(wraps=(RuntimeContextCapability,))

    async def wrap_model_request(
        self,
        ctx: RunContext[AgentContext],
        *,
        request_context: ModelRequestContext,
        handler: ModelHandler,
    ) -> Any:
        env = ctx.deps.env
        if env is None:
            return await handler(request_context)
        processor = create_environment_instructions_filter(env)
        return await project_history(processor, ctx, request_context, handler)


@dataclass(kw_only=True)
class RuntimeContextCapability(AbstractCapability[AgentContext]):
    """Decorate one model request with fresh runtime/session context."""

    id: str | None = "runtime_context"

    async def wrap_model_request(
        self,
        ctx: RunContext[AgentContext],
        *,
        request_context: ModelRequestContext,
        handler: ModelHandler,
    ) -> Any:
        return await project_history(inject_runtime_instructions, ctx, request_context, handler)
