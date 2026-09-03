"""Main agent factory for creating configured agents.

This module provides the create_agent function for building agents
with proper environment and context lifecycle management.
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import sys
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Generic, Protocol, cast, runtime_checkable

import jinja2
from pydantic_ai import (
    Agent,
    AgentRetries,
    AgentSpec,
    DeferredToolRequests,
    DeferredToolResults,
    ModelSettings,
    RunContext,
    UsageLimits,
    UserError,
)
from pydantic_ai._agent_graph import CallToolsNode, ModelRequestNode
from pydantic_ai._enqueue import EnqueueContent, PendingMessage, PendingMessagePriority
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    BaseToolCallPart,
    ModelMessage,
    ModelResponse,
    ModelResponsePart,
    NativeToolCallPart,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    UserContent,
)
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.output import OutputSpec
from pydantic_ai.run import AgentRun
from pydantic_ai.usage import RunUsage
from typing_extensions import TypeVar
from ya_agent_environment import Environment

from ya_agent_sdk._logger import get_logger
from ya_agent_sdk.agents.driver import drive_streamed_run
from ya_agent_sdk.agents.lifecycle import AgentErrorContext, BaseLifecycleExtension, run_extension_method
from ya_agent_sdk.agents.models import infer_model
from ya_agent_sdk.agents.retry_recovery import (
    DEFAULT_STREAM_RESUME_PROMPT,
    StreamRetryController,
    close_unreturned_tool_calls,
    extract_resume_history,
    history_has_unreturned_tool_calls,
    recover_retry_message_history,
)
from ya_agent_sdk.context import (
    AgentContext,
    AgentInfo,
    AgentStreamEvent,
    ModelConfig,
    ModelFeature,
    ModelWrapper,
    ResumableState,
    StreamEvent,
    StreamRecoveryPolicy,
    ToolIdWrapper,
)
from ya_agent_sdk.environment.local import LocalEnvironment
from ya_agent_sdk.events import (
    AgentExecutionCompleteEvent,
    AgentExecutionFailedEvent,
    AgentExecutionResumeEvent,
    AgentExecutionStartEvent,
    LifecycleEvent,
    ModelRequestCompleteEvent,
    ModelRequestStartEvent,
    ToolCallsCompleteEvent,
    ToolCallsStartEvent,
    UsageSnapshotEvent,
)
from ya_agent_sdk.inputs import EnqueueReceipt, InputOrigin, LogicalRunInputRouter
from ya_agent_sdk.usage import CostEstimate, coerce_run_usage, estimate_latest_model_message_cost
from ya_agent_sdk.utils import AgentDepsT, EnvT, get_latest_request_usage

if TYPE_CHECKING:
    pass


logger = get_logger(__name__)

_STREAM_CLEANUP_TIMEOUT_SECONDS = 5.0

# =============================================================================
# Exceptions
# =============================================================================


class AgentInterrupted(Exception):
    """Raised when agent execution is interrupted by user.

    This exception is raised when `AgentStreamer.interrupt()` is called,
    providing immediate cancellation of all running tasks.
    """

    pass


def _has_tool_call_parts(parts: Sequence[object]) -> bool:
    return any(isinstance(part, BaseToolCallPart) for part in parts)


_PREVIOUS_ASSISTANT_RESPONSE_REFERENCE_MAX_CHARS = 32000
_PREVIOUS_ASSISTANT_RESPONSE_REFERENCE_KEEP_HEAD = 24000
_PREVIOUS_ASSISTANT_RESPONSE_REFERENCE_KEEP_TAIL = 6000


def _truncate_previous_assistant_response_reference(text: str) -> str:
    """Bound previous assistant visible output used as compact restore reference.

    The cap is intentionally generous because numbered execution plans and
    option lists often live in the previous assistant response and may be needed
    to resolve terse follow-up prompts such as "do 1, 2, and 3".
    """
    stripped = text.strip()
    if len(stripped) <= _PREVIOUS_ASSISTANT_RESPONSE_REFERENCE_MAX_CHARS:
        return stripped

    head = stripped[:_PREVIOUS_ASSISTANT_RESPONSE_REFERENCE_KEEP_HEAD]
    tail = stripped[-_PREVIOUS_ASSISTANT_RESPONSE_REFERENCE_KEEP_TAIL:]
    truncated_count = (
        len(stripped)
        - _PREVIOUS_ASSISTANT_RESPONSE_REFERENCE_KEEP_HEAD
        - _PREVIOUS_ASSISTANT_RESPONSE_REFERENCE_KEEP_TAIL
    )
    return f"{head}\n[... {truncated_count} chars truncated from previous assistant response ...]\n{tail}"


def _extract_previous_assistant_response_reference(
    message_history: Sequence[ModelMessage] | None,
) -> str | None:
    """Extract visible text from the assistant response before the current user prompt.

    The reference is used only for compact restore to resolve references in the
    current prompt, e.g. "1,2,3", "the above", or "that option". It intentionally
    excludes thinking, tool calls, tool returns, and other non-visible content.
    """
    if not message_history:
        return None

    for message in reversed(message_history):
        if not isinstance(message, ModelResponse):
            continue
        chunks = [part.content for part in message.parts if isinstance(part, TextPart) and part.content.strip()]
        if chunks:
            return _truncate_previous_assistant_response_reference("\n\n".join(chunks))
    return None


def _suspend_current_task_cancellation() -> tuple[asyncio.Task[Any] | None, int]:
    """Temporarily clear cancellation requests on the current task.

    During Ctrl+C handling, the consumer task enters `stream_agent` cleanup
    while already marked as cancelling. Any await inside the cleanup path will
    immediately raise `CancelledError`, which abandons internal task teardown
    and leaves pydantic-ai cleanup running in orphaned tasks.

    We temporarily drain the cancellation counter, perform cleanup, then restore
    the same number of cancellation requests before returning to the caller.
    """
    current_task = asyncio.current_task()
    if current_task is None:
        return None, 0

    cleared = 0
    while current_task.cancelling():
        current_task.uncancel()
        cleared += 1
    return current_task, cleared


def _restore_task_cancellation(task: asyncio.Task[Any] | None, count: int) -> None:
    """Restore previously cleared cancellation requests to a task."""
    if task is None or count <= 0:
        return

    for _ in range(count):
        task.cancel()


# =============================================================================
# Type Variables
# =============================================================================

OutputT = TypeVar("OutputT")


# =============================================================================
# Lifecycle Tracking
# =============================================================================


@dataclass
class LifecycleTracker:
    """Tracks lifecycle state during agent execution."""

    loop_index: int = 0


# =============================================================================
# Agent Runtime
# =============================================================================


@runtime_checkable
class _RuntimeManagedCapability(Protocol):
    """Private lifecycle boundary for host-backed capability services."""

    async def close_runtime(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ResolvedCapabilitySource:
    """One ordered capability source retained for runtime diagnostics."""

    source_id: str
    capabilities: tuple[AbstractCapability[Any], ...]


@dataclass
class AgentRuntime(Generic[AgentDepsT, OutputT, EnvT]):
    """Unentered runtime plan that constructs the Agent after authorities exist."""

    env: EnvT
    ctx: AgentDepsT
    explicit_capabilities: tuple[AbstractCapability[AgentDepsT], ...]
    agent_builder: Callable[
        [AgentDepsT, tuple[AbstractCapability[AgentDepsT], ...]],
        Agent[AgentDepsT, OutputT],
    ] = field(repr=False)
    lifecycle_extensions: list[BaseLifecycleExtension[AgentDepsT, EnvT]] = field(default_factory=list)
    _agent: Agent[AgentDepsT, OutputT] | None = field(default=None, init=False, repr=False)
    _resolved_capabilities: tuple[AbstractCapability[AgentDepsT], ...] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _capability_sources: tuple[ResolvedCapabilitySource, ...] = field(
        default=(),
        init=False,
        repr=False,
    )
    _exit_stack: AsyncExitStack | None = field(default=None, init=False, repr=False)
    _enter_count: int = field(default=0, init=False, repr=False)
    _enter_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    @property
    def agent(self) -> Agent[AgentDepsT, OutputT]:
        """Return the entered Agent; construction before runtime entry is an error."""
        if self._agent is None:
            raise RuntimeError("AgentRuntime.agent is unavailable before runtime entry")
        return self._agent

    @property
    def capabilities(self) -> tuple[AbstractCapability[AgentDepsT], ...]:
        """Return resolved top-level capabilities in native source order after entry."""
        if self._resolved_capabilities is None:
            raise RuntimeError("AgentRuntime.capabilities are unavailable before runtime entry")
        return self._resolved_capabilities

    @property
    def capability_sources(self) -> tuple[ResolvedCapabilitySource, ...]:
        """Return provenance-preserving source groups after entry."""
        if self._resolved_capabilities is None:
            raise RuntimeError("Capability provenance is unavailable before runtime entry")
        return self._capability_sources

    async def __aenter__(self) -> AgentRuntime[AgentDepsT, OutputT, EnvT]:
        """Enter Environment, restore context authority, then construct the Agent."""
        async with self._enter_lock:
            self._enter_count += 1
            if self._enter_count > 1:
                return self

            stack = AsyncExitStack()
            await stack.__aenter__()
            try:
                if not self.env._entered:
                    await stack.enter_async_context(self.env)
                if not self.ctx._entered:
                    await stack.enter_async_context(self.ctx)

                sources = self._collect_capability_sources()
                capabilities = tuple(
                    cast(AbstractCapability[AgentDepsT], capability)
                    for source in sources
                    for capability in source.capabilities
                )
                _validate_capability_sources(sources)
                for capability in capabilities:
                    if isinstance(capability, _RuntimeManagedCapability):
                        stack.push_async_callback(capability.close_runtime)
                agent = self.agent_builder(self.ctx, capabilities)
                await stack.enter_async_context(agent)
            except BaseException:
                self._enter_count = 0
                await stack.__aexit__(*sys.exc_info())
                raise

            self._capability_sources = sources
            self._resolved_capabilities = capabilities
            self._agent = agent
            self._exit_stack = stack
            return self

    def _collect_capability_sources(self) -> tuple[ResolvedCapabilitySource, ...]:
        sources: list[ResolvedCapabilitySource] = []
        if self.explicit_capabilities:
            sources.append(
                ResolvedCapabilitySource(
                    source_id="explicit",
                    capabilities=cast(tuple[AbstractCapability[Any], ...], self.explicit_capabilities),
                )
            )
        context_capabilities = tuple(self.ctx.get_capabilities())
        if context_capabilities:
            sources.append(
                ResolvedCapabilitySource(
                    source_id="context",
                    capabilities=cast(tuple[AbstractCapability[Any], ...], context_capabilities),
                )
            )
        for contribution in self.env.get_agent_contributions():
            sources.append(
                ResolvedCapabilitySource(
                    source_id=contribution.source_id,
                    capabilities=tuple(contribution.capabilities),
                )
            )
        return tuple(sources)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        """Release the runtime on the final balanced exit."""
        async with self._enter_lock:
            if self._enter_count <= 0:
                logger.warning("AgentRuntime.__aexit__ called without a matching entry")
                return None
            self._enter_count -= 1
            if self._enter_count > 0:
                return None
            stack = self._exit_stack
            self._exit_stack = None
            self._agent = None
            self._resolved_capabilities = None
            self._capability_sources = ()
            if stack is not None:
                return await stack.__aexit__(exc_type, exc_val, exc_tb)
            return None


def _validate_capability_sources(sources: Sequence[ResolvedCapabilitySource]) -> None:
    """Validate contribution values and singleton IDs with source diagnostics."""
    seen_ids: dict[str, tuple[str, type[AbstractCapability[Any]]]] = {}
    for source in sources:
        for capability in source.capabilities:
            if not isinstance(capability, AbstractCapability):
                raise TypeError(
                    f"Capability source {source.source_id!r} contributed "
                    f"{type(capability).__name__}, expected AbstractCapability"
                )

            def visit(
                leaf: AbstractCapability[Any],
                source_id: str = source.source_id,
            ) -> None:
                if leaf.id is None:
                    return
                previous = seen_ids.get(leaf.id)
                if previous is not None:
                    previous_source, previous_type = previous
                    raise ValueError(
                        f"Duplicate capability id {leaf.id!r} from {previous_source!r} "
                        f"({previous_type.__name__}) and {source_id!r} "
                        f"({type(leaf).__name__})"
                    )
                seen_ids[leaf.id] = (source_id, type(leaf))

            capability.apply(visit)


# =============================================================================
# System Prompt Loading
# =============================================================================


_CONTEXT_HEADER_MODEL_PREFIXES = ("oauth@codex:", "openai-responses-rs:", "openai-responses-ws:")
_GATEWAY_CONTEXT_HEADER_UPSTREAM_PREFIXES = (
    "openai-responses:",
    "openai-responses-rs:",
    "openai-responses-ws:",
)
_PROVIDER_CONTEXT_HEADER_NAMES = frozenset({
    "session_id",
    "session-id",
    "x-session-id",
    "thread_id",
    "thread-id",
    "x-client-request-id",
})


def _model_uses_context_headers(model: Model | KnownModelName | str | None) -> bool:
    if not isinstance(model, str):
        return False
    if model.startswith(_CONTEXT_HEADER_MODEL_PREFIXES):
        return True
    gateway_name, separator, upstream_model = model.partition("@")
    return bool(gateway_name and separator and upstream_model.startswith(_GATEWAY_CONTEXT_HEADER_UPSTREAM_PREFIXES))


def _patch_provider_session_settings(
    model_cfg: ModelConfig,
    model_settings: Mapping[str, Any] | None,
    model_extra_headers: Mapping[str, str],
) -> ModelSettings:
    """Bind one model request to the provider session carried by its current context."""
    patched_settings: dict[str, Any] = dict(model_settings or {})
    configured_extra_headers = patched_settings.get("extra_headers")
    patched_extra_headers = (
        {
            name: value
            for name, value in configured_extra_headers.items()
            if not isinstance(name, str) or name.lower() not in _PROVIDER_CONTEXT_HEADER_NAMES
        }
        if isinstance(configured_extra_headers, Mapping)
        else {}
    )
    patched_extra_headers.update(model_extra_headers)
    patched_settings["extra_headers"] = patched_extra_headers

    if model_cfg.has_capability(ModelFeature.openai_prompt_cache_key):
        configured_extra_body = patched_settings.get("extra_body")
        if isinstance(configured_extra_body, Mapping):
            patched_settings["extra_body"] = {
                name: value for name, value in configured_extra_body.items() if name != "prompt_cache_key"
            }
        patched_settings["openai_prompt_cache_key"] = model_extra_headers["x-session-id"]
    return cast(ModelSettings, patched_settings)


def _patch_prompt_cache_key(
    model_cfg: ModelConfig,
    model_settings: ModelSettings | None,
    model_extra_headers: dict[str, str] | None,
) -> ModelSettings | None:
    """Bind capable model requests to the same session used by their headers."""
    if not model_cfg.has_capability(ModelFeature.openai_prompt_cache_key) or model_extra_headers is None:
        return model_settings
    return _patch_provider_session_settings(model_cfg, model_settings, model_extra_headers)


@dataclass(slots=True)
class _ProviderSessionSettingsCapability(AbstractCapability[AgentContext]):
    """Resolve provider routing headers from the active run context on every request."""

    def get_model_settings(self) -> Callable[[RunContext[AgentContext]], ModelSettings]:
        def settings(ctx: RunContext[AgentContext]) -> ModelSettings:
            return _patch_provider_session_settings(
                ctx.deps.model_cfg,
                ctx.model_settings,
                ctx.deps.get_model_extra_headers(),
            )

        return settings


def _load_system_prompt(
    template: str | None = None,
    template_vars: dict[str, Any] | None = None,
) -> str:
    """Load and render system prompt.

    Args:
        template: Template string. If None, loads from prompts/main.md.
        template_vars: Variables to pass to Jinja2 template.

    Returns:
        Rendered system prompt string, or empty string if template is empty/not found.
    """
    if template is None:
        prompt_path = Path(__file__).parent / "prompts" / "main.md"
        if not prompt_path.exists():
            return ""
        template = prompt_path.read_text()

    if not template.strip():
        return ""

    # Always render with Jinja2 to support default values in templates
    env = jinja2.Environment(autoescape=False)  # noqa: S701
    jinja_template = env.from_string(template)
    return jinja_template.render(**(template_vars or {}))


# =============================================================================
# Agent Factory
# =============================================================================


def create_agent(
    model: Model | KnownModelName | str | None = None,
    *,
    spec: AgentSpec | None = None,
    custom_capability_types: Sequence[type[AbstractCapability[Any]]] = (),
    capabilities: Sequence[AbstractCapability[AgentDepsT]] = (),
    model_settings: ModelSettings | None = None,
    model_wrapper: ModelWrapper | None = None,
    output_type: OutputSpec[OutputT] = str,  # type: ignore[assignment]
    context_type: type[AgentDepsT] = AgentContext,  # type: ignore[assignment]
    model_cfg: ModelConfig | None = None,
    context_kwargs: Mapping[str, Any] | None = None,
    state: ResumableState | None = None,
    env: EnvT | type[EnvT] = LocalEnvironment,  # type: ignore[assignment]
    env_kwargs: Mapping[str, Any] | None = None,
    agent_name: str | None = None,
    instructions: str | None = None,
    system_prompt: str | None = None,
    system_prompt_template_vars: Mapping[str, Any] | None = None,
    retries: int | AgentRetries | None = None,
    defer_model_check: bool = False,
    end_strategy: str | None = None,
    lifecycle_extensions: Sequence[BaseLifecycleExtension[AgentDepsT, EnvT]] = (),
) -> AgentRuntime[AgentDepsT, OutputT, EnvT]:
    """Create an unentered 2.0 runtime plan.

    ``capabilities`` is the sole public behavior-composition surface. The
    Pydantic AI Agent is intentionally unavailable until runtime entry, after
    the Environment has restored its resources and all provenance-preserving
    contribution groups can be collected.
    """
    if isinstance(retries, int) and retries < 0:
        raise UserError("retries must be non-negative or None")

    actual_env = env if isinstance(env, Environment) else env(**dict(env_kwargs or {}))
    extensions = list(lifecycle_extensions)
    effective_agent_name = agent_name or (spec.name if spec is not None else None) or "main"
    effective_context_kwargs = dict(context_kwargs or {})
    ctx = context_type(
        env=actual_env,
        model_cfg=model_cfg or ModelConfig(),
        model_wrapper=model_wrapper,
        **effective_context_kwargs,
    ).with_state(state)
    ctx.lifecycle_extensions = extensions
    effective_system_prompt = _load_system_prompt(
        system_prompt,
        dict(system_prompt_template_vars or {}),
    )

    def build_agent(
        runtime_ctx: AgentDepsT,
        resolved_capabilities: tuple[AbstractCapability[AgentDepsT], ...],
    ) -> Agent[AgentDepsT, OutputT]:
        selected_model = model if model is not None else (spec.model if spec is not None else None)
        model_extra_headers = (
            runtime_ctx.get_model_extra_headers() if _model_uses_context_headers(selected_model) else None
        )
        selected_model_settings = (
            model_settings
            if model_settings is not None
            else cast(ModelSettings | None, spec.model_settings if spec is not None else None)
        )
        effective_model_settings = _patch_prompt_cache_key(
            runtime_ctx.model_cfg,
            selected_model_settings,
            model_extra_headers,
        )
        effective_capabilities = (
            (*resolved_capabilities, _ProviderSessionSettingsCapability())
            if model_extra_headers is not None
            else resolved_capabilities
        )
        base_model = (
            infer_model(selected_model, extra_headers=model_extra_headers)
            if isinstance(selected_model, str)
            else selected_model
        )
        effective_model: Model | None = base_model
        if base_model is not None and runtime_ctx.model_wrapper is not None:
            wrapped = runtime_ctx.model_wrapper(
                base_model,
                effective_agent_name,
                runtime_ctx.get_wrapper_metadata(),
            )
            if inspect.isawaitable(wrapped):
                raise TypeError(
                    "Async model_wrapper is not supported during runtime construction; "
                    "use a synchronous wrapper or provide a wrapped Model instance"
                )
            effective_model = wrapped

        effective_retries = retries
        if effective_retries is None and (spec is None or spec.retries is None):
            effective_retries = AgentRetries(
                tools=runtime_ctx.retry_config.tools,
                output=runtime_ctx.retry_config.output,
            )

        if spec is not None:
            return cast(
                Agent[AgentDepsT, OutputT],
                Agent.from_spec(
                    spec,
                    deps_type=context_type,
                    custom_capability_types=custom_capability_types,
                    model=effective_model,
                    output_type=output_type,
                    instructions=instructions,
                    system_prompt=effective_system_prompt,
                    name=effective_agent_name,
                    model_settings=effective_model_settings,
                    retries=effective_retries,
                    defer_model_check=defer_model_check,
                    end_strategy=cast(Any, end_strategy),
                    capabilities=effective_capabilities,
                ),
            )
        return Agent(
            model=effective_model,
            instructions=instructions,
            system_prompt=effective_system_prompt,
            model_settings=effective_model_settings,
            deps_type=context_type,
            output_type=output_type,
            capabilities=effective_capabilities,
            retries=effective_retries,
            defer_model_check=defer_model_check,
            end_strategy=cast(Any, end_strategy or "graceful"),
            name=effective_agent_name,
        )

    return AgentRuntime(
        env=actual_env,
        ctx=ctx,
        explicit_capabilities=tuple(capabilities),
        agent_builder=build_agent,
        lifecycle_extensions=extensions,
    )


# =============================================================================
# Stream Hook Types
# =============================================================================


@dataclass
class RuntimeReadyContext(Generic[AgentDepsT, OutputT, EnvT]):
    """Context passed to runtime ready hook (after runtime enter, before agent.iter).

    This hook is called after the runtime (env, ctx, agent) has been entered but
    before agent.iter() starts. Use it to:
    - Initialize resources that depend on the environment being ready
    - Emit custom events to the output stream
    - Modify context state before agent execution
    - Modify user_prompt or deferred_tool_results to control agent input

    Attributes:
        runtime: The AgentRuntime containing env, ctx, and agent.
        agent_info: Metadata about the main agent.
        output_queue: Queue for emitting custom StreamEvent to the output stream.
        user_prompt: The user prompt to send to the agent. Can be modified by hook.
        deferred_tool_results: Results from deferred tool calls. Can be modified by hook.
    """

    runtime: AgentRuntime[AgentDepsT, OutputT, EnvT]
    agent_info: AgentInfo
    output_queue: asyncio.Queue[StreamEvent]
    user_prompt: str | Sequence[UserContent] | None
    deferred_tool_results: DeferredToolResults | None


@dataclass
class AgentStartContext(Generic[AgentDepsT, OutputT, EnvT]):
    """Context passed to agent start hook (after agent.iter starts, before first node).

    This hook is called after agent.iter() has started and the run object is available,
    but before any nodes are processed. Use it to:
    - Access the run object for initial state inspection
    - Log agent start with run metadata
    - Emit custom events at agent start

    Attributes:
        runtime: The AgentRuntime containing env, ctx, and agent.
        agent_info: Metadata about the main agent.
        output_queue: Queue for emitting custom StreamEvent to the output stream.
        run: The AgentRun instance from agent.iter().
    """

    runtime: AgentRuntime[AgentDepsT, OutputT, EnvT]
    agent_info: AgentInfo
    output_queue: asyncio.Queue[StreamEvent]
    run: AgentRun[AgentDepsT, OutputT]


@dataclass
class AgentCompleteContext(Generic[AgentDepsT, OutputT, EnvT]):
    """Context passed to agent complete hook (after all nodes processed, before agent.iter exits).

    This hook is called after all nodes have been processed but before the agent.iter()
    context manager exits. Use it to:
    - Access the final result and usage statistics
    - Log agent completion with full run data
    - Emit custom completion events

    Attributes:
        runtime: The AgentRuntime containing env, ctx, and agent.
        agent_info: Metadata about the main agent.
        output_queue: Queue for emitting custom StreamEvent to the output stream.
        run: The AgentRun instance with result available.
    """

    runtime: AgentRuntime[AgentDepsT, OutputT, EnvT]
    agent_info: AgentInfo
    output_queue: asyncio.Queue[StreamEvent]
    run: AgentRun[AgentDepsT, OutputT]


@dataclass
class NodeHookContext(Generic[AgentDepsT, OutputT]):
    """Context passed to node-level hooks (pre/post node.stream).

    Attributes:
        agent_info: Metadata about the current agent.
        node: The current graph node (ModelRequestNode or CallToolsNode).
        run: The AgentRun instance from agent.iter().
        output_queue: Queue for emitting custom StreamEvent to the output stream.
    """

    agent_info: AgentInfo
    node: ModelRequestNode[AgentDepsT, OutputT] | CallToolsNode[AgentDepsT, OutputT]
    run: AgentRun[AgentDepsT, OutputT]
    output_queue: asyncio.Queue[StreamEvent]


@dataclass
class EventHookContext(Generic[AgentDepsT, OutputT]):
    """Context passed to event-level hooks (pre/post each event yield).

    Attributes:
        agent_info: Metadata about the current agent.
        event: The stream event being yielded.
        node: The current graph node.
        run: The AgentRun instance from agent.iter().
        output_queue: Queue for emitting custom StreamEvent to the output stream.
    """

    agent_info: AgentInfo
    event: AgentStreamEvent
    node: ModelRequestNode[AgentDepsT, OutputT] | CallToolsNode[AgentDepsT, OutputT]
    run: AgentRun[AgentDepsT, OutputT]
    output_queue: asyncio.Queue[StreamEvent]


# User prompt type alias
UserPromptT = str | Sequence[UserContent]

# Hook type aliases
RuntimeReadyHook = Callable[[RuntimeReadyContext[AgentDepsT, OutputT, EnvT]], Awaitable[None]]
AgentStartHook = Callable[[AgentStartContext[AgentDepsT, OutputT, EnvT]], Awaitable[None]]
AgentCompleteHook = Callable[[AgentCompleteContext[AgentDepsT, OutputT, EnvT]], Awaitable[None]]
NodeHook = Callable[[NodeHookContext[AgentDepsT, OutputT]], Awaitable[None]]
EventHook = Callable[[EventHookContext[AgentDepsT, OutputT]], Awaitable[None]]
UserPromptFactory = Callable[[AgentRuntime[AgentDepsT, OutputT, EnvT]], Awaitable[UserPromptT]]
ResumePromptFactory = Callable[[BaseException, int, Sequence[ModelMessage]], UserPromptT | Awaitable[UserPromptT]]


# =============================================================================
# Agent Streamer
# =============================================================================


@dataclass
class PartialTextAccumulator:
    """Accumulates recoverable streamed response parts for interrupted history.

    Text can be recovered while still partial. Thinking and tool-call parts are
    recovered after their closing PartEndEvent, so persisted history contains
    whole thinking blocks and whole tool-call arguments.
    """

    _parts: dict[int, ModelResponsePart] = field(default_factory=dict)
    _in_progress_parts: dict[
        int, ThinkingPart | ToolCallPart | NativeToolCallPart | ThinkingPartDelta | ToolCallPartDelta
    ] = field(default_factory=dict)

    def reset(self) -> None:
        """Start tracking a new model response."""
        self._parts.clear()
        self._in_progress_parts.clear()

    def observe(self, event: AgentStreamEvent) -> None:
        """Record a stream event for interrupted response recovery."""
        if isinstance(event, PartStartEvent):
            self._observe_part_start(event)
        elif isinstance(event, PartDeltaEvent):
            self._observe_part_delta(event)
        elif isinstance(event, PartEndEvent):
            self._observe_part_end(event)

    def _observe_part_start(self, event: PartStartEvent) -> None:
        if isinstance(event.part, TextPart):
            self._parts[event.index] = event.part
            self._in_progress_parts.pop(event.index, None)
        elif isinstance(event.part, ThinkingPart | BaseToolCallPart):
            self._parts.pop(event.index, None)
            self._in_progress_parts[event.index] = event.part
        else:
            self._parts.pop(event.index, None)
            self._in_progress_parts.pop(event.index, None)

    def _observe_part_delta(self, event: PartDeltaEvent) -> None:
        if isinstance(event.delta, TextPartDelta):
            part = self._parts.get(event.index)
            if isinstance(part, TextPart):
                self._parts[event.index] = event.delta.apply(part)
            else:
                self._parts[event.index] = TextPart(
                    content=event.delta.content_delta,
                    provider_name=event.delta.provider_name,
                    provider_details=event.delta.provider_details,
                )
            self._in_progress_parts.pop(event.index, None)
        elif isinstance(event.delta, ThinkingPartDelta):
            self._observe_thinking_delta(event.index, event.delta)
        elif isinstance(event.delta, ToolCallPartDelta):
            self._observe_tool_call_delta(event.index, event.delta)
        else:
            self._parts.pop(event.index, None)
            self._in_progress_parts.pop(event.index, None)

    def _observe_thinking_delta(self, index: int, delta: ThinkingPartDelta) -> None:
        part = self._in_progress_parts.get(index)
        if isinstance(part, (ThinkingPart, ThinkingPartDelta)):
            self._in_progress_parts[index] = delta.apply(part)
        else:
            self._in_progress_parts[index] = delta
        self._parts.pop(index, None)

    def _observe_tool_call_delta(self, index: int, delta: ToolCallPartDelta) -> None:
        part = self._in_progress_parts.get(index)
        if isinstance(part, (ToolCallPart, NativeToolCallPart, ToolCallPartDelta)):
            self._in_progress_parts[index] = delta.apply(part)
        else:
            self._in_progress_parts[index] = delta.as_part() or delta
        self._parts.pop(index, None)

    def _observe_part_end(self, event: PartEndEvent) -> None:
        if isinstance(event.part, (TextPart, ThinkingPart, BaseToolCallPart)):
            self._parts[event.index] = event.part
        else:
            self._parts.pop(event.index, None)
        self._in_progress_parts.pop(event.index, None)

    def build_response(self) -> ModelResponse | None:
        """Build a recoverable partial ModelResponse from stream progress."""
        parts = [self._parts[index] for index in sorted(self._parts) if self._is_non_empty_part(self._parts[index])]
        if not parts:
            return None
        return ModelResponse(
            parts=parts,
            metadata={"ya_agent_sdk": {"partial": True, "reason": "stream_interrupted"}},
        )

    @staticmethod
    def _is_non_empty_part(part: ModelResponsePart) -> bool:
        if isinstance(part, TextPart | ThinkingPart):
            return bool(part.content)
        if isinstance(part, BaseToolCallPart):
            return bool(part.tool_name)
        return True


@dataclass
class AgentStreamer(Generic[AgentDepsT, OutputT]):
    """Async iterator for streaming agent events with interrupt capability.

    This is a class-based async iterator (not an async generator) to avoid
    the "asynchronous generator is already running" error that occurs when
    Python's GC finalizer tries to aclose() an async generator that was
    interrupted by CancelledError during iteration.

    Attributes:
        run: The AgentRun instance. None until agent.iter() starts, available during
            and after streaming. Use to access messages, usage, and result.
        exception: The exception captured during streaming, if any. Available after
            streaming completes. Check this or call raise_if_exception() after iteration.

    Example::

        async with stream_agent(agent, "Hello", ctx=ctx) as streamer:
            async for event in streamer:
                print(f"[{event.agent_name}] {event.event}")
                if streamer.run:
                    print(f"Messages so far: {len(streamer.run.all_messages())}")
                if should_stop:
                    streamer.interrupt()
                    break
            # After streaming, check for exceptions manually
            streamer.raise_if_exception()
            # Access final result and usage
            if streamer.run:
                print(f"Usage: {streamer.run.usage}")
    """

    _output_queue: asyncio.Queue[StreamEvent]
    _main_task: asyncio.Task[None]
    _poll_done: asyncio.Event
    _tasks: list[asyncio.Task[None]] = field(default_factory=list)
    _partial_text: PartialTextAccumulator = field(default_factory=PartialTextAccumulator)
    _tool_id_wrapper: ToolIdWrapper | None = None
    _input_router: LogicalRunInputRouter | None = None
    run: AgentRun[AgentDepsT, OutputT] | None = None
    exception: BaseException | None = None
    _interrupted: bool = False

    def recoverable_messages(self) -> list[ModelMessage]:
        """Return run history plus safe text-only partial output.

        Completed runs return `run.all_messages()` unchanged. Interrupted or failed
        streams may have emitted assistant text before pydantic-ai finalized the
        ModelResponse; when the emitted response contains text parts only, that
        text is appended as a partial ModelResponse so callers can persist it.
        """
        messages: list[ModelMessage] = []
        if self.run is not None:
            messages = list(self.run.all_messages())
            if self._tool_id_wrapper is not None:
                messages = self._tool_id_wrapper.wrap_messages(None, messages)
            if messages and isinstance(messages[-1], ModelResponse):
                self._mark_interrupted_response(messages[-1])
                return messages
        partial_response = self._partial_text.build_response()
        if partial_response is not None:
            messages.append(partial_response)
            if self._tool_id_wrapper is not None:
                messages = self._tool_id_wrapper.wrap_messages(None, messages)
        return messages

    @staticmethod
    def _mark_interrupted_response(response: ModelResponse) -> None:
        """Annotate pydantic-ai v2 interrupted partial responses with SDK metadata."""
        if response.state != "interrupted":
            return
        metadata = dict(response.metadata or {})
        sdk_metadata = dict(metadata.get("ya_agent_sdk") or {})
        sdk_metadata.setdefault("partial", True)
        sdk_metadata.setdefault("reason", "stream_interrupted")
        metadata["ya_agent_sdk"] = sdk_metadata
        response.metadata = metadata

    async def enqueue(
        self,
        *content: EnqueueContent,
        priority: PendingMessagePriority = "asap",
        origin: InputOrigin = InputOrigin.user,
        input_id: str | None = None,
    ) -> EnqueueReceipt:
        """Accept structured input for this logical run through native enqueue."""
        if self._input_router is None:
            raise RuntimeError("This stream has no active logical input router")
        return await self._input_router.enqueue(
            *content,
            priority=priority,
            origin=origin,
            input_id=input_id,
        )

    def interrupt(self) -> None:
        """Interrupt the stream immediately, cancelling all running tasks.

        This method provides hard cancellation - all running tasks are cancelled
        immediately via asyncio.Task.cancel(). When the context manager exits,
        AgentInterrupted will be raised.
        """
        self._interrupted = True
        self.exception = AgentInterrupted("Agent execution was interrupted")
        for task in self._tasks:
            if not task.done():
                task.cancel()

    def raise_if_exception(self) -> None:
        """Raise the captured exception if any occurred during streaming.

        Call this after iteration completes to propagate any errors from
        the main agent or subagent tasks.

        Raises:
            AgentInterrupted: If interrupt() was called.
            BaseException: Additional exception that occurred during streaming.
        """
        # Check stored exception first
        if self.exception is not None:
            raise self.exception

        # Also check tasks for exceptions (in case called before context manager exits)
        for task in self._tasks:
            if task.done() and not task.cancelled():
                exc = task.exception()
                if exc is not None:
                    raise exc

    def __aiter__(self) -> AgentStreamer[AgentDepsT, OutputT]:
        return self

    def _check_task_exceptions(self) -> None:
        """Check all tasks for exceptions and raise the first one found."""
        for task in self._tasks:
            if task.done() and not task.cancelled():
                exc = task.exception()
                if exc is not None:
                    raise exc

    def _all_tasks_done(self) -> bool:
        """Check if all producer tasks have finished (done/cancelled)."""
        return all(task.done() for task in self._tasks)

    async def __anext__(self) -> StreamEvent:
        """Get next event from the output queue.

        Monitors all tasks for exceptions and propagates them immediately.
        Returns StopAsyncIteration when all producers are done and queue is empty.
        """
        while True:
            # Drain queued events before surfacing producer exceptions. A task may
            # enqueue lifecycle events immediately before raising, and callers should
            # be able to observe those terminal events.
            try:
                return self._output_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass

            # Check if any task failed after queued events have been consumed.
            self._check_task_exceptions()

            # Check exit condition: poll done and output queue empty
            if self._poll_done.is_set() and self._output_queue.empty():
                # Final check for task exceptions before stopping
                self._check_task_exceptions()
                raise StopAsyncIteration

            # Fallback: if all tasks are done and queue is empty, stop iteration
            # even if _poll_done was never set (e.g. tasks cancelled before finally block ran)
            if self._all_tasks_done() and self._output_queue.empty():
                self._check_task_exceptions()
                raise StopAsyncIteration

            try:
                event = await asyncio.wait_for(self._output_queue.get(), timeout=0.1)
                return event
            except TimeoutError:
                continue


# =============================================================================
# Stream Agent
# =============================================================================


@asynccontextmanager
async def stream_agent(  # noqa: C901
    runtime: AgentRuntime[AgentDepsT, OutputT, EnvT],
    user_prompt: UserPromptT | None = None,
    *,
    user_prompt_factory: UserPromptFactory[AgentDepsT, OutputT, EnvT] | None = None,
    message_history: Sequence[ModelMessage] | None = None,
    deferred_tool_results: DeferredToolResults | None = None,
    usage_limits: UsageLimits | None = None,
    # Hooks
    on_runtime_ready: RuntimeReadyHook[AgentDepsT, OutputT, EnvT] | None = None,
    on_agent_start: AgentStartHook[AgentDepsT, OutputT, EnvT] | None = None,
    on_agent_complete: AgentCompleteHook[AgentDepsT, OutputT, EnvT] | None = None,
    pre_node_hook: NodeHook[AgentDepsT, OutputT] | None = None,
    post_node_hook: NodeHook[AgentDepsT, OutputT] | None = None,
    pre_event_hook: EventHook[AgentDepsT, OutputT] | None = None,
    post_event_hook: EventHook[AgentDepsT, OutputT] | None = None,
    # Error handling
    raise_on_error: bool = True,
    resume_on_error: bool | None = None,
    resume_max_attempts: int | None = None,
    transport_resume_max_attempts: int | None = None,
    resume_prompt: UserPromptT | None = None,
    resume_prompt_factory: ResumePromptFactory | None = None,
    # Lifecycle events
    emit_lifecycle_events: bool = True,
) -> AsyncGenerator[AgentStreamer[AgentDepsT, OutputT], None]:
    """Stream agent execution with subagent event aggregation.

    This context manager runs the agent and yields a streamer that merges
    events from the main agent and all subagents into a single stream.

    Lifecycle Management:
        This function automatically manages the runtime lifecycle internally.
        When called, it will:
        1. Enter the runtime (env -> ctx -> agent) if not already entered
        2. Execute the agent with the given prompt
        3. Exit the runtime when streaming completes

        The runtime uses `_entered` flags to avoid double-entering components
        that are already active. This means you can safely call stream_agent
        without manually entering the runtime first.

        Note: Manual runtime lifecycle management is not recommended.
        Let stream_agent handle it automatically for proper resource cleanup.

    Args:
        runtime: The AgentRuntime containing agent and context.
        user_prompt: The prompt to send to the agent. Can be string or
            sequence of UserContent for multimodal input.
        user_prompt_factory: Async callable that receives AgentRuntime and returns
            a user prompt. Called after runtime enters, before on_runtime_ready.
            Use when prompt generation requires runtime resources (e.g., reading files).
            If both user_prompt and user_prompt_factory are provided, factory takes precedence.
        message_history: Optional conversation history.
        deferred_tool_results: Results from deferred tool calls.
        on_runtime_ready: Called after runtime enters but before agent.iter() starts.
            Use to initialize resources, emit events, or modify context state.
        on_agent_start: Called after agent.iter() starts, before first node.
            Use to access run object for initial state inspection.
        on_agent_complete: Called after all nodes processed, before agent.iter() exits.
            Use to access final result and usage statistics.
        pre_node_hook: Called before node.stream() starts.
        post_node_hook: Called after node.stream() completes.
        pre_event_hook: Called before each event is yielded.
        post_event_hook: Called after each event is yielded.
        raise_on_error: If True (default), exceptions during streaming are re-raised
            immediately. If False, exceptions are captured in streamer.exception
            and can be checked after iteration via raise_if_exception().
        resume_on_error: If True, retry failed stream attempts inside the same stream.
            If None, uses ctx.model_cfg.stream_resume_on_error.
        resume_max_attempts: Maximum total attempts for non-transport stream errors
            when resume_on_error is enabled. If None, uses
            ctx.model_cfg.stream_resume_max_attempts.
        transport_resume_max_attempts: Independent maximum attempts for each
            consecutive transient model transport failure streak. These attempts do
            not consume the non-transport resume budget, and a successful model request
            resets the streak. If None, uses ctx.model_cfg.stream_transport_resume_max_attempts.
        resume_prompt: Prompt sent after a recoverable stream failure. If unset,
            uses ctx.model_cfg.stream_resume_prompt, then the built-in default.
        resume_prompt_factory: Callable that builds a resume prompt from the exception,
            next attempt index, and recovered message history.
        emit_lifecycle_events: If True (default), emit built-in lifecycle events
            (AgentExecutionStartEvent, LoopStartEvent, NodeStartEvent, etc.) to the
            stream. Set to False to disable these events for cleaner output or
            when implementing custom event handling via hooks.

    Yields:
        AgentStreamer that can be iterated for StreamEvent objects.
        Each event contains agent_id, agent_name, and the raw event.

    Example::

        # Recommended: Let stream_agent manage the runtime lifecycle
        runtime = create_agent("openai-chat:gpt-4")
        async with stream_agent(
            runtime,
            "Search for Python tutorials",
        ) as streamer:
            async for event in streamer:
                if event.agent_name == "main":
                    # Handle main agent events
                    pass
                else:
                    # Handle subagent events
                    pass
    """
    # Validate mutually exclusive parameters
    if user_prompt is not None and user_prompt_factory is not None:
        msg = "Cannot specify both 'user_prompt' and 'user_prompt_factory'. Use one or the other."
        raise UserError(msg)

    # Enter an unentered runtime in the caller task so Environment restoration
    # and capability resolution complete before the Agent is accessed. The
    # producer task owns only its fresh per-run context.
    entered_runtime_here = runtime._enter_count == 0
    if entered_runtime_here:
        await runtime.__aenter__()
    agent = runtime.agent
    extensions = list(runtime.lifecycle_extensions)

    # Create a fresh context for this run to isolate per-run state.
    # This prevents ContextVar tokens and other run-specific state from
    # leaking between consecutive runs, which causes "was created in a
    # different Context" errors when pydantic-ai's cleanup runs in the
    # wrong contextvars.Context (see pydantic-ai issue #674).
    fresh_ctx = runtime.ctx.prepare_new_run(
        resume_logical_run=deferred_tool_results is not None,
    )
    runtime.ctx = fresh_ctx
    ctx = fresh_ctx
    ctx.lifecycle_extensions = extensions
    input_router = LogicalRunInputRouter(ctx.run_input_ledger)
    ctx.input_router = input_router
    input_registration = ctx.active_run_registry.register(input_router)

    def configure_stream_recovery_policy() -> StreamRecoveryPolicy:
        effective_resume_on_error = ctx.model_cfg.stream_resume_on_error if resume_on_error is None else resume_on_error
        effective_resume_max_attempts = (
            ctx.model_cfg.stream_resume_max_attempts if resume_max_attempts is None else resume_max_attempts
        )
        effective_transport_resume_max_attempts = (
            ctx.model_cfg.stream_transport_resume_max_attempts
            if transport_resume_max_attempts is None
            else transport_resume_max_attempts
        )
        effective_resume_prompt = (
            resume_prompt
            if resume_prompt is not None
            else (
                ctx.model_cfg.stream_resume_prompt
                if ctx.model_cfg.stream_resume_prompt is not None
                else DEFAULT_STREAM_RESUME_PROMPT
            )
        )
        policy = StreamRecoveryPolicy(
            enabled=effective_resume_on_error,
            max_attempts=max(1, effective_resume_max_attempts),
            transport_max_attempts=max(1, effective_transport_resume_max_attempts),
            resume_prompt=effective_resume_prompt,
            resume_prompt_factory=resume_prompt_factory,
        )
        ctx.stream_recovery_policy = policy
        return policy

    # Publish the effective policy before hooks run, then refresh it after
    # on_runtime_ready in case the hook changed model configuration.
    configure_stream_recovery_policy()

    # Enable streaming for emit_event.
    # Must use object.__setattr__ because Pydantic v2 silently ignores
    # normal attribute assignment on private attrs of model_copy() instances.
    object.__setattr__(ctx, "_stream_queue_enabled", True)

    # A caller may already own the runtime lifecycle. In that case run_main
    # must only enter the fresh per-run context it created above, rather than
    # re-entering AgentRuntime from its child task. If cleanup reaches its
    # deadline, the child can then finish its own context cleanup later without
    # becoming the task that closes the caller-owned env and agent resources.
    runtime_was_entered = runtime._enter_count > 0

    output_queue: asyncio.Queue[StreamEvent] = asyncio.Queue()
    main_done = asyncio.Event()
    poll_done = asyncio.Event()
    partial_text = PartialTextAccumulator()
    attempt_failed_during_model_request = False
    successful_model_request_count = 0

    logger.debug(
        "Starting stream_agent with user_prompt=%s",
        user_prompt[:100] if isinstance(user_prompt, str) else type(user_prompt),
    )

    # Build main agent info
    main_agent_info = AgentInfo(agent_id="main", agent_name=agent.name or "main")

    def suppress_benign_stream_cleanup_error(exc: BaseException) -> bool:
        if isinstance(exc, TypeError) and "cannot create weak reference to 'NoneType'" in str(exc):
            logger.debug("Suppressed anyio CancelScope cleanup error from streaming: %s", exc)
            return True
        if isinstance(exc, ValueError) and "was created in a different Context" in str(exc):
            logger.debug("Suppressed ContextVar cleanup error from pydantic-ai streaming: %s", exc)
            return True
        return False

    async def process_node_event(
        event: AgentStreamEvent,
        node: ModelRequestNode[AgentDepsT, OutputT] | CallToolsNode[AgentDepsT, OutputT],
        run: AgentRun[AgentDepsT, OutputT],
    ) -> None:
        """Run host-side event hooks and publish one model/tool event."""
        event_ctx = EventHookContext(
            agent_info=main_agent_info, event=event, node=node, run=run, output_queue=output_queue
        )
        await run_extension_method(extensions, "on_before_event", event_ctx, logger=logger)
        if pre_event_hook:
            await pre_event_hook(event_ctx)

        input_router.observe_event(event)
        wrapped_event = ctx.tool_id_wrapper.wrap_event(event)
        partial_text.observe(wrapped_event)
        await output_queue.put(
            StreamEvent(
                agent_id=main_agent_info.agent_id,
                agent_name=main_agent_info.agent_name,
                event=wrapped_event,
            )
        )

        await run_extension_method(extensions, "on_after_event", event_ctx, logger=logger)
        if post_event_hook:
            await post_event_hook(event_ctx)

    async def process_node(
        node: ModelRequestNode[AgentDepsT, OutputT] | CallToolsNode[AgentDepsT, OutputT],
        run: AgentRun[AgentDepsT, OutputT],
    ) -> None:
        """Process a single node with hooks."""
        nonlocal attempt_failed_during_model_request, successful_model_request_count

        # PRE NODE HOOK
        logger.debug("Processing node: %s", type(node).__name__)
        node_ctx = NodeHookContext(agent_info=main_agent_info, node=node, run=run, output_queue=output_queue)
        await run_extension_method(extensions, "on_before_node", node_ctx, logger=logger)
        if pre_node_hook:
            await pre_node_hook(node_ctx)

        event_processing_failed = False
        try:
            async with node.stream(run.ctx) as request_stream:
                async for event in request_stream:
                    try:
                        await process_node_event(event, node, run)
                    except BaseException:
                        event_processing_failed = True
                        raise
        except BaseException as exc:
            # Suppress known benign anyio/ContextVar cleanup failures only when
            # they came from stream cleanup. Host-side event processing errors
            # always use the non-transport execution budget.
            if event_processing_failed or not suppress_benign_stream_cleanup_error(exc):
                attempt_failed_during_model_request = isinstance(node, ModelRequestNode) and not event_processing_failed
                raise

        if isinstance(node, ModelRequestNode):
            successful_model_request_count += 1

        # POST NODE HOOK
        logger.debug("Node completed: %s", type(node).__name__)
        await run_extension_method(extensions, "on_after_node", node_ctx, logger=logger)
        if post_node_hook:
            await post_node_hook(node_ctx)

    # Lifecycle tracker for loop counting.
    # loop_index is set at the start of each ModelRequest and used by both
    # ModelRequest and ToolCalls events within the same loop iteration.
    tracker = LifecycleTracker()

    async def emit_lifecycle_event(event: LifecycleEvent) -> None:
        """Emit a lifecycle event if enabled."""
        if emit_lifecycle_events:
            await output_queue.put(
                StreamEvent(
                    agent_id=main_agent_info.agent_id,
                    agent_name=main_agent_info.agent_name,
                    event=event,
                )
            )

    async def handle_model_request_node(
        node: ModelRequestNode[AgentDepsT, OutputT],
        run: AgentRun[AgentDepsT, OutputT],
        node_start_time: float,
        *,
        attempt_index: int,
    ) -> None:
        """Handle model_request node with lifecycle events.

        Each ModelRequestNode marks the start of a new loop iteration.
        The loop_index is incremented here before processing.
        """
        current_loop = tracker.loop_index
        tracker.loop_index += 1  # Increment for next loop

        await emit_lifecycle_event(
            ModelRequestStartEvent(event_id=ctx.run_id, loop_index=current_loop, message_count=len(run.all_messages()))
        )

        partial_text.reset()
        await process_node(node, run)

        base_model = cast(Model, agent.model)
        run_usage = coerce_run_usage(run.usage)
        usage_ledger_key = (
            main_agent_info.agent_id if attempt_index == 0 else f"{main_agent_info.agent_id}:attempt:{attempt_index}"
        )
        ctx.update_usage_snapshot_entry(
            ledger_key=usage_ledger_key,
            agent_id=main_agent_info.agent_id,
            agent_name=main_agent_info.agent_name,
            model_id=base_model.model_name,
            usage=run_usage,
            cost_estimate=CostEstimate(),
            source="main_model_request",
        )
        request_usage_id = f"{ctx.run_id}:main:{attempt_index}:{current_loop}"
        ctx.update_usage_snapshot_entry(
            ledger_key=f"cost:{request_usage_id}",
            agent_id=main_agent_info.agent_id,
            agent_name=main_agent_info.agent_name,
            model_id=base_model.model_name,
            usage=RunUsage(),
            cost_estimate=estimate_latest_model_message_cost(run.new_messages()),
            usage_id=request_usage_id,
            source="main_model_request_cost",
        )

        latest_request_usage = get_latest_request_usage(run.all_messages())
        await emit_lifecycle_event(
            ModelRequestCompleteEvent(
                event_id=ctx.run_id,
                loop_index=current_loop,
                duration_seconds=time.perf_counter() - node_start_time,
                context_tokens=(latest_request_usage.total_tokens if latest_request_usage is not None else 0),
                context_window_size=ctx.model_cfg.context_window or 0,
            )
        )
        snapshot = ctx.build_usage_snapshot()
        await ctx.emit_event(
            UsageSnapshotEvent(
                event_id=f"{snapshot.run_id}:usage_snapshot:model_request_complete:{current_loop}",
                snapshot=snapshot,
                source="model_request_complete",
            )
        )

    async def handle_call_tools_node(
        node: CallToolsNode[AgentDepsT, OutputT],
        run: AgentRun[AgentDepsT, OutputT],
        node_start_time: float,
    ) -> None:
        """Handle call_tools node with lifecycle events.

        ToolCalls always follow a ModelRequest, so we use (loop_index - 1)
        to reference the loop that just completed its model request phase.
        """
        current_loop = tracker.loop_index - 1
        has_tool_calls = _has_tool_call_parts(node.model_response.parts)
        if has_tool_calls:
            await emit_lifecycle_event(ToolCallsStartEvent(event_id=ctx.run_id, loop_index=current_loop))

        await process_node(node, run)

        if has_tool_calls:
            await emit_lifecycle_event(
                ToolCallsCompleteEvent(
                    event_id=ctx.run_id,
                    loop_index=current_loop,
                    duration_seconds=time.perf_counter() - node_start_time,
                )
            )

    async def process_all_nodes(run: AgentRun[AgentDepsT, OutputT], *, attempt_index: int) -> None:
        """Process all nodes in the agent run with lifecycle events."""

        async def process_node(node: Any, current_run: AgentRun[Any, Any]) -> None:
            node_start_time = time.perf_counter()

            if Agent.is_user_prompt_node(node) or Agent.is_end_node(node):
                # Skip user_prompt and end nodes - their info is in AgentExecution events
                return
            if Agent.is_model_request_node(node):
                await handle_model_request_node(node, current_run, node_start_time, attempt_index=attempt_index)
            elif Agent.is_call_tools_node(node):
                await handle_call_tools_node(node, current_run, node_start_time)

        await drive_streamed_run(run, process_node)

    async def run_agent_iteration(
        effective_user_prompt: UserPromptT | None,
        effective_deferred_tool_results: DeferredToolResults | None,
        effective_message_history: Sequence[ModelMessage] | None,
        stream_start_time: float,
        *,
        attempt_index: int,
        is_resume_attempt: bool,
    ) -> None:
        """Run one agent iteration attempt with hooks and lifecycle events."""
        await emit_lifecycle_event(
            AgentExecutionStartEvent(
                event_id=ctx.run_id,
                user_prompt=effective_user_prompt,
                deferred_tool_results=effective_deferred_tool_results,
                message_history_count=len(effective_message_history) if effective_message_history else 0,
                attempt_index=attempt_index,
                is_resume_attempt=is_resume_attempt,
            )
        )

        async with agent.iter(
            effective_user_prompt,
            deps=ctx,
            usage_limits=usage_limits,
            message_history=effective_message_history,
            deferred_tool_results=effective_deferred_tool_results,
        ) as run:
            streamer.run = run
            native_attempt_id = f"{ctx.run_id}:{attempt_index}"
            await input_router.bind(run, native_attempt_id=native_attempt_id)
            try:
                start_ctx = AgentStartContext(
                    runtime=runtime, agent_info=main_agent_info, output_queue=output_queue, run=run
                )
                await run_extension_method(extensions, "on_agent_start", start_ctx, logger=logger)
                if on_agent_start:
                    await on_agent_start(start_ctx)

                await process_all_nodes(run, attempt_index=attempt_index)

                complete_ctx = AgentCompleteContext(
                    runtime=runtime, agent_info=main_agent_info, output_queue=output_queue, run=run
                )
                await run_extension_method(extensions, "on_agent_complete", complete_ctx, logger=logger)
                if on_agent_complete:
                    await on_agent_complete(complete_ctx)

                await ctx.emit_usage_snapshot_event(source="session_end")

                await emit_lifecycle_event(
                    AgentExecutionCompleteEvent(
                        event_id=ctx.run_id,
                        total_loops=tracker.loop_index,
                        total_duration_seconds=time.perf_counter() - stream_start_time,
                        final_message_count=len(run.all_messages()),
                        attempt_index=attempt_index,
                    )
                )
            finally:
                input_router.unbind(native_attempt_id=native_attempt_id)

    failure_event_emitted = False

    def stringify_exception(exc: BaseException) -> str:
        try:
            error_str = str(exc)
        except Exception:
            error_str = repr(exc)
        return error_str or repr(exc)

    def split_resume_prompt_for_tool_call_history(
        history: Sequence[ModelMessage] | None,
        prompt: UserPromptT | None,
    ) -> tuple[Sequence[ModelMessage] | None, UserPromptT | None]:
        if prompt is None or not history or not history_has_unreturned_tool_calls(history):
            return history, prompt
        return close_unreturned_tool_calls(history), prompt

    async def resolve_resume_prompt(
        exc: BaseException,
        attempt_index: int,
        history: Sequence[ModelMessage],
    ) -> UserPromptT:
        if resume_prompt_factory is not None:
            value = resume_prompt_factory(exc, attempt_index, history)
            if inspect.isawaitable(value):
                value = await value
            return cast(UserPromptT, value)
        recovery_policy = ctx.stream_recovery_policy
        if recovery_policy is not None:
            return cast(UserPromptT, recovery_policy.resume_prompt)
        return cast(UserPromptT, DEFAULT_STREAM_RESUME_PROMPT)

    async def emit_execution_failed_event(
        exc: BaseException,
        stream_start_time: float,
        *,
        attempt_index: int = 0,
        recoverable: bool = False,
    ) -> str:
        nonlocal failure_event_emitted
        error_str = stringify_exception(exc)
        await ctx.emit_usage_snapshot_event(source="session_end")
        await emit_lifecycle_event(
            AgentExecutionFailedEvent(
                event_id=ctx.run_id,
                error=error_str,
                error_type=type(exc).__name__,
                total_loops=tracker.loop_index,
                total_duration_seconds=time.perf_counter() - stream_start_time,
                attempt_index=attempt_index,
                recoverable=recoverable,
            )
        )
        failure_event_emitted = True
        return error_str

    async def run_main_attempts(
        effective_user_prompt: UserPromptT | None,
        effective_deferred_tool_results: DeferredToolResults | None,
        stream_start_time: float,
    ) -> None:
        nonlocal attempt_failed_during_model_request

        recovery_policy = configure_stream_recovery_policy()
        retry_controller = StreamRetryController(recovery_policy)
        attempt_index = 0
        current_user_prompt = effective_user_prompt
        current_deferred_tool_results = effective_deferred_tool_results
        current_message_history: Sequence[ModelMessage] | None = message_history

        while True:
            successful_model_requests_before_attempt = successful_model_request_count
            try:
                attempt_failed_during_model_request = False
                attempt_message_history, attempt_user_prompt = split_resume_prompt_for_tool_call_history(
                    current_message_history,
                    current_user_prompt,
                )
                await run_agent_iteration(
                    attempt_user_prompt,
                    current_deferred_tool_results,
                    attempt_message_history,
                    stream_start_time,
                    attempt_index=attempt_index,
                    is_resume_attempt=attempt_index > 0,
                )
                return
            except Exception as e:
                decision = retry_controller.record_failure(
                    e,
                    failed_during_model_request=attempt_failed_during_model_request,
                    successful_model_request=(
                        successful_model_request_count > successful_model_requests_before_attempt
                    ),
                )

                error_str = await emit_execution_failed_event(
                    e,
                    stream_start_time,
                    attempt_index=attempt_index,
                    recoverable=decision.recoverable,
                )
                if not decision.recoverable:
                    raise

                next_attempt_index = attempt_index + 1
                resume_history = close_unreturned_tool_calls(
                    extract_resume_history(streamer.run, current_message_history)
                )
                recovery = recover_retry_message_history(e, resume_history, ctx)
                resume_history = recovery.history
                if recovery.changed:
                    logger.info(
                        "Applied retry recovery before resume attempt=%s reasons=%s",
                        next_attempt_index,
                        ",".join(recovery.reasons),
                    )
                if resume_history:
                    next_user_prompt = await resolve_resume_prompt(e, next_attempt_index, resume_history)
                    next_message_history: Sequence[ModelMessage] | None = resume_history
                else:
                    next_user_prompt = current_user_prompt
                    next_message_history = current_message_history

                await emit_lifecycle_event(
                    AgentExecutionResumeEvent(
                        event_id=ctx.run_id,
                        attempt_index=next_attempt_index,
                        previous_attempt_index=attempt_index,
                        error=error_str,
                        error_type=type(e).__name__,
                        message_history_count=len(next_message_history) if next_message_history else 0,
                        resume_prompt=next_user_prompt,
                    )
                )
                logger.warning(
                    "Resuming stream_agent after error attempt_index=%s next_attempt_index=%s "
                    "recovery_budget=%s failures_in_budget=%s max_attempts=%s error_type=%s error=%s",
                    attempt_index,
                    next_attempt_index,
                    decision.budget,
                    decision.failures_in_budget,
                    decision.max_attempts,
                    type(e).__name__,
                    error_str,
                )
                attempt_index = next_attempt_index
                current_user_prompt = next_user_prompt
                current_deferred_tool_results = None
                current_message_history = next_message_history

    async def prepare_runtime_input() -> tuple[UserPromptT | None, DeferredToolResults | None]:
        effective_user_prompt = user_prompt
        effective_deferred_tool_results = deferred_tool_results
        if user_prompt_factory:
            effective_user_prompt = await user_prompt_factory(runtime)

        ctx.user_prompts = effective_user_prompt
        ctx.previous_assistant_response_reference = _extract_previous_assistant_response_reference(message_history)

        ready_ctx = RuntimeReadyContext(
            runtime=runtime,
            agent_info=main_agent_info,
            output_queue=output_queue,
            user_prompt=effective_user_prompt,
            deferred_tool_results=effective_deferred_tool_results,
        )
        await run_extension_method(extensions, "on_runtime_ready", ready_ctx, logger=logger)
        if on_runtime_ready:
            await on_runtime_ready(ready_ctx)
        effective_user_prompt = ready_ctx.user_prompt
        effective_deferred_tool_results = ready_ctx.deferred_tool_results
        # Persist the canonical prompt actually passed to Agent.iter(). The
        # structured ledger, rather than a text-only steering side channel, is
        # authoritative for compaction and continuation retention.
        ctx.user_prompts = effective_user_prompt
        if (
            effective_deferred_tool_results is None
            and effective_user_prompt is not None
            and not ctx.run_input_ledger.records
        ):
            initial_content = (
                (effective_user_prompt,) if isinstance(effective_user_prompt, str) else tuple(effective_user_prompt)
            )
            initial = PendingMessage.from_content(*initial_content, priority="asap")
            if initial is not None:
                ctx.run_input_ledger.record_initial(initial.messages)
        return effective_user_prompt, effective_deferred_tool_results

    async def run_main() -> None:
        """Run the main agent and push events to output_queue."""
        logger.debug("Main agent task started")

        stream_start_time = time.perf_counter()

        try:
            if runtime_was_entered:
                # The enclosing task owns the runtime's env and agent exit stack.
                # Enter only this run's fresh context in the producer task, so its
                # late cleanup remains task-affine without changing runtime entry
                # ownership when the bounded shutdown wait expires.
                async with AsyncExitStack() as stack:
                    if not ctx._entered:
                        await stack.enter_async_context(ctx)
                    effective_user_prompt, effective_deferred_tool_results = await prepare_runtime_input()
                    await run_main_attempts(effective_user_prompt, effective_deferred_tool_results, stream_start_time)
            else:
                async with runtime:
                    effective_user_prompt, effective_deferred_tool_results = await prepare_runtime_input()
                    await run_main_attempts(effective_user_prompt, effective_deferred_tool_results, stream_start_time)

        except BaseException as e:
            if isinstance(e, asyncio.CancelledError):
                logger.debug("Main agent task cancelled")
            elif isinstance(e, TypeError) and "cannot create weak reference to 'NoneType'" in str(e):
                # anyio CancelScope fails when asyncio.current_task() returns None
                # during httpx stream cleanup after cancellation.  The stream was
                # already being torn down; this error is benign.
                logger.debug("Suppressed anyio CancelScope cleanup error: %s", e)
                return
            elif isinstance(e, ValueError) and "was created in a different" in str(e):
                # Suppress ContextVar cleanup error from pydantic-ai.
                # This occurs when async generator cleanup runs in a different
                # contextvars.Context (see pydantic-ai issue #674).
                # The error is benign - model responses were already received.
                logger.warning("Suppressed ContextVar cleanup error: %s", e)
                return
            else:
                logger.exception("Error in main agent task")
                error_ctx = AgentErrorContext(
                    runtime=runtime,
                    agent_info=main_agent_info,
                    output_queue=output_queue,
                    error=e,
                )
                await run_extension_method(extensions, "on_agent_error", error_ctx, logger=logger)
                if not failure_event_emitted:
                    await emit_execution_failed_event(e, stream_start_time)
            raise
        finally:
            logger.debug("Main agent task finished")
            main_done.set()

    async def poll_subagents() -> None:
        """Poll subagent stream queues and push events to output_queue."""
        logger.debug("Subagent polling task started")
        try:
            while True:
                # Check exit condition: main done and all queues empty
                if main_done.is_set():
                    all_empty = all(q.empty() for q in ctx.agent_stream_queues.values())
                    if all_empty:
                        return

                # Collect events from all subagent queues
                for agent_id, queue in list(ctx.agent_stream_queues.items()):
                    try:
                        event = queue.get_nowait()
                        agent_info = (
                            main_agent_info
                            if agent_id == main_agent_info.agent_id
                            else ctx.agent_stream_info.get(agent_id)
                        )
                        await output_queue.put(
                            StreamEvent(
                                agent_id=agent_id,
                                agent_name=agent_info.agent_name if agent_info else "unknown",
                                event=event,
                            )
                        )
                    except asyncio.QueueEmpty:
                        pass

                await asyncio.sleep(0.001)  # Yield control to avoid busy loop
        finally:
            logger.debug("Subagent polling task finished")
            poll_done.set()

    # Start producer tasks.
    # Use a fresh contextvars.Context copy for main_task to isolate each run
    # from stale ContextVar state left by previous cancelled runs.  When a
    # streaming run is cancelled, pydantic-ai's internal wrap_task may not
    # complete its ContextVar cleanup (e.g. _CURRENT_RUN_CONTEXT.reset(token)
    # in set_current_run_context).  If that orphaned cleanup later runs via
    # GC in a different context, it raises "was created in a different Context".
    # A fresh context copy ensures each run starts clean.
    main_task = asyncio.create_task(run_main(), context=contextvars.copy_context())
    poll_task = asyncio.create_task(poll_subagents())

    streamer: AgentStreamer[AgentDepsT, OutputT] = AgentStreamer(
        _output_queue=output_queue,
        _main_task=main_task,
        _poll_done=poll_done,
        _tasks=[main_task, poll_task],
        _partial_text=partial_text,
        _tool_id_wrapper=ctx.tool_id_wrapper,
        _input_router=input_router,
    )

    deferred_suspension = False
    try:
        yield streamer
    except Exception as e:
        logger.exception("Uncaught exception in stream_agent context")
        streamer.exception = e
        if raise_on_error:
            raise  # Re-raise so caller can handle it
    else:
        if (run := streamer.run) and (result := run.result) and isinstance(result.output, DeferredToolRequests):
            deferred_suspension = True
            result.output = ctx.tool_id_wrapper.wrap_deferred_tool_requests(result.output)
    finally:
        # Cancel all running tasks to initiate clean shutdown.
        # This handles both explicit interrupt() calls and external cancellation (e.g., Ctrl+C).
        for task in streamer._tasks:
            if not task.done():
                task.cancel()

        # When the caller is already cancelling, any await in this cleanup path
        # would immediately raise CancelledError and abandon the internal teardown.
        # Temporarily clear the cancellation count, finish cleanup, then restore it.
        current_task, cleared_cancellations = _suspend_current_task_cancellation()
        try:
            # Wait for tasks to complete and capture any exception.
            # We do NOT re-cancel tasks during this wait: doing so would interrupt
            # pydantic-ai's internal ContextVar cleanup (set_current_run_context's
            # finally block), causing "was created in a different Context" errors.
            # A single cancel() is sufficient; the deadline guards against hangs.
            # `asyncio.wait()` deliberately leaves pending tasks untouched when
            # the deadline expires, unlike wait_for()/gather() cancellation.
            gather_interrupt: BaseException | None = None
            cleanup_deadline = time.perf_counter() + _STREAM_CLEANUP_TIMEOUT_SECONDS
            while any(not t.done() for t in [main_task, poll_task]):
                remaining_seconds = cleanup_deadline - time.perf_counter()
                if remaining_seconds <= 0:
                    logger.warning(
                        "Cleanup deadline exceeded (%.1fs), abandoning remaining tasks",
                        _STREAM_CLEANUP_TIMEOUT_SECONDS,
                    )
                    break
                not_done = [task for task in [main_task, poll_task] if not task.done()]
                try:
                    await asyncio.wait(not_done, timeout=remaining_seconds)
                except BaseException as gather_exc:
                    logger.debug("cleanup wait interrupted: %s", type(gather_exc).__name__)
                    gather_interrupt = gather_exc
                    if isinstance(gather_exc, asyncio.CancelledError):
                        # A fresh cancellation request arrived while we were cleaning up.
                        # Clear it too so we can keep draining the inner tasks.
                        _task, newly_cleared = _suspend_current_task_cancellation()
                        if current_task is None:
                            current_task = _task
                        cleared_cancellations += newly_cleared
                    # Do NOT re-cancel tasks here. The initial cancel() is already
                    # in flight; re-cancelling would interrupt the internal cleanup
                    # of pydantic-ai's wrap_task (ContextVar token reset).

            # Collect results from completed tasks.
            results: list[BaseException | None] = []
            for task in [main_task, poll_task]:
                if task.done():
                    if task.cancelled():
                        results.append(asyncio.CancelledError())
                    else:
                        results.append(task.exception())
                else:
                    logger.warning("Task %s still running after cleanup deadline, will self-cleanup", task)
                    results.append(asyncio.CancelledError())

            # Find first real exception (non-CancelledError)
            exceptions = [
                r for r in results if isinstance(r, BaseException) and not isinstance(r, asyncio.CancelledError)
            ]

            if streamer._interrupted:
                streamer.exception = AgentInterrupted("Agent execution was interrupted")
            elif exceptions:
                streamer.exception = exceptions[0]

            # Re-raise gather interrupt only if cleanup failed (tasks still running)
            # or if the interrupt was not a simple CancelledError (e.g., KeyboardInterrupt).
            if gather_interrupt is not None:
                still_running = any(not t.done() for t in [main_task, poll_task])
                if still_running or not isinstance(gather_interrupt, asyncio.CancelledError):
                    raise gather_interrupt
        finally:
            input_router.close(
                reason="logical run completed",
                reject_unresolved=not deferred_suspension,
            )
            ctx.active_run_registry.unregister(input_registration)
            ctx.input_router = None
            if entered_runtime_here:
                await runtime.__aexit__(None, None, None)
            _restore_task_cancellation(current_task, cleared_cancellations)
