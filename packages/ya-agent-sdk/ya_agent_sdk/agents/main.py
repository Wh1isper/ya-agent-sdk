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
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Generic, cast

import jinja2
from pydantic_ai import Agent, AgentRetries, DeferredToolRequests, DeferredToolResults, UsageLimits, UserError
from pydantic_ai._agent_graph import CallToolsNode, ModelRequestNode
from pydantic_ai.capabilities import AbstractCapability, ProcessHistory
from pydantic_ai.messages import (
    BaseToolCallPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    NativeToolCallPart,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
    UserContent,
)
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.output import OutputSpec
from pydantic_ai.run import AgentRun
from typing_extensions import TypeVar
from ya_agent_environment import Environment

from ya_agent_sdk._logger import get_logger
from ya_agent_sdk.agents.compact import create_cache_friendly_compact_filter, create_compact_filter
from ya_agent_sdk.agents.guards import attach_message_bus_guard
from ya_agent_sdk.agents.lifecycle import AgentErrorContext, BaseLifecycleExtension, run_extension_method
from ya_agent_sdk.agents.models import infer_model
from ya_agent_sdk.agents.retry_recovery import recover_retry_message_history
from ya_agent_sdk.context import (
    AgentContext,
    AgentInfo,
    AgentStreamEvent,
    ModelConfig,
    ModelWrapper,
    ResumableState,
    StreamEvent,
    SubagentWrapper,
    ToolConfig,
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
from ya_agent_sdk.filters.auto_load_files import process_auto_load_files
from ya_agent_sdk.filters.cold_start import cold_start_trim
from ya_agent_sdk.filters.environment_instructions import create_environment_instructions_filter
from ya_agent_sdk.filters.runtime_instructions import inject_runtime_instructions
from ya_agent_sdk.filters.system_prompt import create_system_prompt_filter
from ya_agent_sdk.toolsets.core.base import BaseTool, GlobalHooks, Toolset
from ya_agent_sdk.utils import AgentDepsT, EnvT, add_toolset_instructions

from .capabilities import is_process_history_for

if TYPE_CHECKING:
    from pydantic_ai import ModelSettings
    from pydantic_ai.toolsets import AbstractToolset

    from ya_agent_sdk.subagents import SubagentConfig

logger = get_logger(__name__)

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


def _filter_delegation_toolset(
    toolset: AbstractToolset[Any],
    delegation_tags: frozenset[str],
) -> AbstractToolset[Any]:
    """Return a toolset safe for self forks by removing delegation-tagged tools."""
    if isinstance(toolset, Toolset):
        return toolset.exclude_tags(delegation_tags)
    return toolset


def _resolve_agent_retries(retries: int | AgentRetries | None, output_retries: int | None) -> int | AgentRetries:
    """Resolve SDK retry defaults into Pydantic AI's unified AgentRetries form."""
    default_retries: AgentRetries = {"tools": 1, "output": 3}
    if isinstance(retries, dict):
        resolved: AgentRetries = {**default_retries, **retries}
    elif retries is None:
        resolved = default_retries.copy()
    else:
        resolved = {"tools": retries, "output": retries}

    if output_retries is not None:
        resolved["output"] = output_retries

    return resolved


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


@dataclass
class AgentRuntime(Generic[AgentDepsT, OutputT, EnvT]):
    """Container for agent runtime components with lifecycle management.

    This dataclass holds all the components needed to run an agent,
    providing a clean interface for accessing the environment, context,
    and agent instance. It also acts as an async context manager to
    manage the lifecycle of env, ctx, and agent.

    Type Parameters:
        AgentDepsT: The context type (subclass of AgentContext).
        OutputT: The output type from the agent.
        EnvT: The environment type (subclass of Environment). Defaults to Environment.

    Attributes:
        env: The environment instance managing resources.
        ctx: The agent context for session state.
        agent: The configured pydantic-ai Agent instance.
        core_toolset: The core toolset with BaseTool instances.

    Example:
        runtime = create_agent("openai-chat:gpt-4")
        async with runtime:
            result = await runtime.agent.run("Hello", deps=runtime.ctx)
            print(result.output)

        # Or with external env management:
        async with env:
            runtime = create_agent("openai-chat:gpt-4", env=env)
            async with runtime:  # Only enters ctx and agent
                result = await runtime.agent.run("Hello", deps=runtime.ctx)
    """

    env: EnvT
    ctx: AgentDepsT
    agent: Agent[AgentDepsT, OutputT]
    core_toolset: Toolset[AgentDepsT] | None
    lifecycle_extensions: list[BaseLifecycleExtension[AgentDepsT, EnvT]] = field(default_factory=list)
    _exit_stack: AsyncExitStack | None = field(default=None, repr=False)
    _reentrant_exit_stacks: list[AsyncExitStack] = field(default_factory=list, init=False, repr=False)
    _enter_count: int = field(default=0, init=False, repr=False)
    _enter_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    async def __aenter__(self) -> AgentRuntime[AgentDepsT, OutputT, EnvT]:
        """Enter the runtime, managing env/ctx/agent lifecycles.

        Uses reference counting to support re-entrant usage. Only the first
        entry actually sets up resources; subsequent entries just increment
        the counter. This is critical when the runtime is entered by an
        outer lifecycle manager (e.g. TUI app) and re-entered by an inner
        caller (e.g. stream_agent's run_main) potentially in a different
        asyncio task. Without reference counting, the inner entry would
        overwrite the exit stack, orphaning the original resources and
        causing anyio cancel scope errors on cleanup (since MCP sessions
        create anyio task groups bound to the entering task).

        Only enters components that are not already entered (checked via _entered flag).
        Uses AsyncExitStack to track what was entered, ensuring proper cleanup.

        A lock ensures that if two tasks concurrently enter the runtime, the
        second one waits until the first has fully initialized resources before
        returning.
        """
        async with self._enter_lock:
            self._enter_count += 1
            if self._enter_count > 1:
                stack = AsyncExitStack()
                await stack.__aenter__()
                try:
                    if not self.ctx._entered:
                        await stack.enter_async_context(self.ctx)
                except BaseException:
                    self._enter_count -= 1
                    await stack.__aexit__(*sys.exc_info())
                    raise
                self._reentrant_exit_stacks.append(stack)
                return self

            # Build exit stack locally first, then commit on success.
            # If any step fails, we rollback the count and close the partial stack.
            stack = AsyncExitStack()
            await stack.__aenter__()
            try:
                # Enter in order: env -> ctx -> agent
                # Only enter if not already entered
                if not self.env._entered:
                    await stack.enter_async_context(self.env)
                if not self.ctx._entered:
                    await stack.enter_async_context(self.ctx)
                # Agent uses reference counting, safe to enter multiple times
                await stack.enter_async_context(self.agent)
            except BaseException:
                # Rollback: close whatever was partially entered and reset count
                self._enter_count -= 1
                await stack.__aexit__(*sys.exc_info())
                raise

            # Commit - all resources entered successfully
            self._exit_stack = stack
            return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        """Exit the runtime, cleaning up only what we entered.

        Only the last exit (counter reaches 0) actually tears down resources.
        This ensures that the outermost context manager (which entered the
        runtime first) performs cleanup in the same asyncio task where
        resources were created, preventing cross-task cancel scope errors.

        The lock serializes exit with entry so a concurrent __aenter__
        cannot start re-initializing resources while teardown is in progress.
        """
        async with self._enter_lock:
            if self._enter_count <= 0:
                logger.warning("AgentRuntime.__aexit__ called with enter_count=%d (unbalanced)", self._enter_count)
                return None
            self._enter_count -= 1
            if self._enter_count > 0:
                if self._reentrant_exit_stacks:
                    stack = self._reentrant_exit_stacks.pop()
                    return await stack.__aexit__(exc_type, exc_val, exc_tb)
                return None
            while self._reentrant_exit_stacks:
                stack = self._reentrant_exit_stacks.pop()
                await stack.__aexit__(exc_type, exc_val, exc_tb)
            if self._exit_stack:
                result = await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)
                self._exit_stack = None
                return result
            return None


# =============================================================================
# System Prompt Loading
# =============================================================================


def _model_uses_context_headers(model: Model | KnownModelName | str | None) -> bool:
    if not isinstance(model, str):
        return False
    return model.startswith(("oauth@codex:", "openai-responses-rs:", "openai-responses-ws:"))


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
    model: Model | KnownModelName | str | None,
    *,
    # --- Model Configuration ---
    model_settings: ModelSettings | None = None,
    model_wrapper: ModelWrapper | None = None,
    subagent_wrapper: SubagentWrapper | None = None,
    output_type: OutputSpec[OutputT] = str,  # type: ignore[assignment]
    # --- Context ---
    context_type: type[AgentDepsT] = AgentContext,  # type: ignore[assignment]
    model_cfg: ModelConfig | None = None,
    # --- Environment ---
    env: EnvT | type[EnvT] = LocalEnvironment,  # type: ignore[assignment]
    env_kwargs: dict[str, Any] | None = None,
    tool_config: ToolConfig | None = None,
    extra_context_kwargs: dict[str, Any] | None = None,
    state: ResumableState | None = None,
    need_user_approve_tools: Sequence[str] | None = None,
    need_user_approve_mcps: Sequence[str] | None = None,
    # --- Toolset ---
    tools: Sequence[type[BaseTool]] | None = None,
    toolsets: Sequence[AbstractToolset[Any]] | None = None,
    pre_hooks: dict[str, Any] | None = None,
    post_hooks: dict[str, Any] | None = None,
    global_hooks: GlobalHooks | None = None,
    toolset_max_retries: int = 3,
    toolset_timeout: float | None = None,
    skip_unavailable_tools: bool = True,
    # --- Compact Filter ---
    compact_model: str | Model | None = None,
    compact_model_settings: ModelSettings | None = None,
    compact_model_cfg: ModelConfig | None = None,
    use_cache_friendly_compact_filter: bool = True,
    # --- Subagent ---
    subagent_configs: Sequence[SubagentConfig] | None = None,
    include_builtin_subagents: bool = False,
    unified_subagents: bool = False,
    unified_subagent_tool_name: str = "delegate",
    hide_unified_subagent_tool: bool = False,
    inherit_hooks: bool = True,
    # --- Capabilities ---
    pre_capabilities: Sequence[AbstractCapability[AgentDepsT]] | None = None,
    capabilities: Sequence[AbstractCapability[AgentDepsT]] | None = None,
    inherit_pre_capabilities: bool = True,
    inherit_capabilities: bool = True,
    # --- Agent ---
    agent_tools: Sequence[Any] | None = None,
    agent_name: str = "main",
    system_prompt: str | None = None,
    system_prompt_template_vars: dict[str, Any] | None = None,
    retries: int | AgentRetries | None = None,
    output_retries: int | None = None,
    defer_model_check: bool = False,
    end_strategy: str = "exhaustive",
    lifecycle_extensions: Sequence[BaseLifecycleExtension[AgentDepsT, EnvT]] | None = None,
) -> AgentRuntime[AgentDepsT, OutputT, EnvT]:
    """Create and configure an agent runtime.

    This function creates an AgentRuntime containing Environment, AgentContext,
    and Agent. The runtime should be used as an async context manager to manage
    the lifecycle of these components.

    Args:
        model: Model string (e.g., "openai-chat:gpt-4") or Model instance.

        model_settings: Optional model settings for inference configuration.
        model_wrapper: Optional wrapper for model instrumentation (observability, caching).
        output_type: Expected output type for the agent. Defaults to str.

        env: Environment instance or class. Defaults to LocalEnvironment.
        env_kwargs: Keyword arguments for Environment instantiation.

        context_type: AgentContext subclass to use. Defaults to AgentContext.
        model_cfg: ModelConfig for context window and capability settings.
        tool_config: ToolConfig for API keys and tool-specific settings.
        extra_context_kwargs: Additional kwargs passed to context_type constructor.
        state: ResumableState to restore session from. Defaults to None.
        need_user_approve_tools: Tools requiring user approval before execution.
        need_user_approve_mcps: MCP servers requiring user approval for all tools.

        tools: Sequence of BaseTool classes to include in the toolset.
        toolsets: Additional AbstractToolset instances to include.

        subagent_configs: Sequence of SubagentConfig for custom subagents.
        include_builtin_subagents: If True, include builtin subagents from presets/.
        unified_subagents: If True, create a single unified tool that can call any
            subagent by name. If False (default), create separate tools for each subagent.
        unified_subagent_tool_name: Tool name for unified subagents. Defaults to "delegate".
        hide_unified_subagent_tool: Hide the unified subagent tool from model-visible tools and instructions while keeping it callable by code.
        pre_hooks: Dict mapping tool names to pre-hook functions.
        post_hooks: Dict mapping tool names to post-hook functions.
        global_hooks: GlobalHooks instance for all tools.
        toolset_max_retries: Max retries for tool execution. Defaults to 3.
        toolset_timeout: Default timeout for tool execution.
        skip_unavailable_tools: Skip tools where is_available() returns False.

        compact_model: Model for legacy compact filter. Falls back to AgentSettings.
        compact_model_settings: Model settings for legacy compact filter.
        compact_model_cfg: Optional compact-specific ModelConfig override. When
            None, compact uses the runtime context model_cfg.
        use_cache_friendly_compact_filter: Select the cache-friendly compact filter by default.
            Set to False to use the legacy compact agent implementation.

        pre_capabilities: Pydantic AI capabilities that run before SDK history
            capabilities. Use this for ProcessHistory capabilities that must see
            the raw message history before compact, handoff, file inspection reminder,
            and runtime instruction filters.
        capabilities: Pydantic AI capabilities to attach after SDK history capabilities.
            Capabilities bundle tools, lifecycle hooks, instructions, and model settings
            into reusable composable units. See pydantic-ai capabilities documentation.
        inherit_pre_capabilities: If True (default), subagents inherit the parent's
            pre_capabilities unless overridden by their own config.pre_capabilities.
        inherit_capabilities: If True (default), subagents inherit the parent's
            capabilities unless overridden by their own config.capabilities.
            If False, subagents get no capabilities unless explicitly set in config.

        agent_tools: Additional tools to pass directly to Agent (pydantic-ai Tool objects).
        agent_name: Name of the agent for logging.
        system_prompt: Custom system prompt or Jinja2 template string. If None, loads from
            prompts/main.md. When used with system_prompt_template_vars, the string is
            rendered as a Jinja2 template, supporting conditionals and default values.
        system_prompt_template_vars: Variables for Jinja2 template rendering. Works with
            both custom system_prompt strings and the default template file.
        retries: Retry budget for tools and structured output. Int sets both budgets;
            AgentRetries can set tools and output independently. Defaults to tools=1, output=3.
        output_retries: Deprecated compatibility override for output retries.
        defer_model_check: Defer model validation. Defaults to False.
        end_strategy: Strategy for ending agent run. Defaults to "exhaustive".
    Returns:
        AgentRuntime containing env, ctx, and agent. Use as async context manager.

    Example:
        Basic usage::

            runtime = create_agent("openai-chat:gpt-4")
            async with runtime:
                result = await runtime.agent.run("Hello", deps=runtime.ctx)
                print(result.output)

        With custom tools and configuration::

            runtime = create_agent(
                "anthropic:claude-3-5-sonnet",
                tools=[ReadFileTool, WriteFileTool],
                model_cfg=ModelConfig(context_window=200000),
                global_hooks=GlobalHooks(pre=my_pre_hook),
            )
            async with runtime:
                result = await runtime.agent.run("Read config.json", deps=runtime.ctx)

        With external environment management::

            async with SandboxEnvironment(
                mounts=[VirtualMount(Path("."), Path("/workspace"))],
                image="python:3.11",
            ) as sandbox_env:
                runtime = create_agent("openai-chat:gpt-4", env=sandbox_env)

        With templated system prompt::

            runtime = create_agent(
                "openai-chat:gpt-4",
                system_prompt="You are a {{ role }}. {{ extra_instructions | default('') }}",
                system_prompt_template_vars={"role": "helpful assistant"},
            )
                async with runtime:  # Only enters ctx and agent
                    result = await runtime.agent.run("Run tests", deps=runtime.ctx)
    """
    # --- Environment Setup ---
    actual_env = env if isinstance(env, Environment) else env(**(env_kwargs or {}))
    logger.debug("Environment created: %s", type(actual_env).__name__)

    # --- Build Configs ---
    effective_model_cfg = model_cfg or ModelConfig()
    effective_tool_config = tool_config or ToolConfig()

    # --- Context Setup ---
    ctx = context_type(
        env=actual_env,
        model_cfg=effective_model_cfg,
        tool_config=effective_tool_config,
        model_wrapper=model_wrapper,
        subagent_wrapper=subagent_wrapper,
        need_user_approve_tools=list(need_user_approve_tools) if need_user_approve_tools else [],
        need_user_approve_mcps=list(need_user_approve_mcps) if need_user_approve_mcps else [],
        **(extra_context_kwargs or {}),
    ).with_state(state)
    logger.debug("Context created: %s (run_id=%s)", type(ctx).__name__, ctx.run_id)

    # --- Capabilities ---
    # Combine user capabilities, context history capabilities, and built-in ProcessHistory capabilities.
    # Runtime instructions run after compact so restored histories receive fresh runtime context.
    context_history_capabilities = [
        capability
        for capability in ctx.get_history_capabilities()
        if not is_process_history_for(capability, inject_runtime_instructions)
    ]
    compact_filter = (
        create_cache_friendly_compact_filter(
            model_cfg=compact_model_cfg,
        )
        if use_cache_friendly_compact_filter
        else create_compact_filter(
            model=compact_model,
            model_settings=compact_model_settings,
            model_cfg=compact_model_cfg,
            main_model=model,
            main_model_settings=model_settings,
        )
    )

    user_pre_capabilities = list(pre_capabilities or [])
    user_capabilities = list(capabilities or [])
    sdk_history_capabilities: list[AbstractCapability[AgentDepsT]] = [
        *cast(list[AbstractCapability[AgentDepsT]], context_history_capabilities),
        ProcessHistory(compact_filter),
        ProcessHistory(cold_start_trim),
        ProcessHistory(create_environment_instructions_filter(actual_env)),
        ProcessHistory(process_auto_load_files),
        ProcessHistory(inject_runtime_instructions),
    ]
    internal_subagent_capabilities: list[AbstractCapability[AgentDepsT]] = [
        *sdk_history_capabilities,
    ]
    all_capabilities: list[AbstractCapability[AgentDepsT]] = [
        *user_pre_capabilities,
        *sdk_history_capabilities,
        *user_capabilities,
    ]

    # --- Toolset Setup ---
    all_toolsets: list[AbstractToolset[Any]] = []
    core_toolset: Toolset[AgentDepsT] | None = None

    # Create Toolset from BaseTool classes if provided
    tools = tools or []
    logger.debug("Creating core toolset with %d tools", len(tools))
    core_toolset = Toolset(
        tools=tools,
        pre_hooks=pre_hooks,
        post_hooks=post_hooks,
        global_hooks=global_hooks,
        max_retries=toolset_max_retries,
        timeout=toolset_timeout,
        skip_unavailable=skip_unavailable_tools,
        toolset_id="core",
    )

    # Add subagent tools if requested
    if subagent_configs or include_builtin_subagents:
        from ya_agent_sdk.subagents import get_builtin_subagent_configs

        all_subagent_configs = list(subagent_configs) if subagent_configs else []
        if include_builtin_subagents:
            all_subagent_configs.extend(get_builtin_subagent_configs().values())

        if all_subagent_configs:
            logger.debug("Adding %d subagent configs to toolset", len(all_subagent_configs))
            # Resolve user capabilities for subagents. SDK history capabilities are passed
            # separately so subagent config capabilities can override parent user capabilities
            # while preserving the internal ProcessHistory pipeline.
            subagent_pre_capabilities: list[AbstractCapability[Any]] | None = (
                list(user_pre_capabilities) if inherit_pre_capabilities else None
            )
            subagent_capabilities: list[AbstractCapability[Any]] | None = (
                list(user_capabilities) if inherit_capabilities else None
            )
            core_toolset = core_toolset._with_subagents(
                all_subagent_configs,
                model=model,
                model_settings=model_settings,
                model_cfg=effective_model_cfg,
                unified=unified_subagents,
                unified_tool_name=unified_subagent_tool_name,
                hidden=hide_unified_subagent_tool,
                inherit_hooks=inherit_hooks,
                pre_capabilities=subagent_pre_capabilities,
                capabilities=subagent_capabilities,
                sdk_capabilities=cast(list[AbstractCapability[Any]], internal_subagent_capabilities),
            )

    # Auto-detect context management tools from registered tools
    ctx.context_manage_tool_names = [t.name for t in tools if t.is_context_manage_tool]

    all_toolsets.append(core_toolset)

    # Add user-provided toolsets
    if toolsets:
        all_toolsets.extend(toolsets)

    # Add environment toolsets (includes both env._toolsets and resource toolsets)
    all_toolsets.extend(actual_env.get_toolsets())

    # --- System Prompt ---
    effective_system_prompt = _load_system_prompt(system_prompt, system_prompt_template_vars)

    # --- Create Model with Wrapper ---
    model_extra_headers = ctx.get_model_extra_headers() if _model_uses_context_headers(model) else None
    base_model = infer_model(model, extra_headers=model_extra_headers) if isinstance(model, str) else model
    effective_model: Model | None = base_model
    if base_model is not None and ctx.model_wrapper is not None:
        wrapper_metadata = ctx.get_wrapper_metadata()
        wrapped = ctx.model_wrapper(base_model, agent_name, wrapper_metadata)
        if inspect.isawaitable(wrapped):
            raise TypeError(
                "Async model_wrapper cannot be used in create_agent (sync context). "
                "Use a sync wrapper or wrap the model manually after agent creation."
            )
        effective_model = wrapped

    # --- Create Agent ---
    logger.debug("Creating agent with model=%s, output_type=%s", model, output_type)
    agent_retries = _resolve_agent_retries(retries, output_retries)

    delegation_tags = frozenset({"delegation"})
    if core_toolset.has_tags(delegation_tags):
        self_fork_toolsets = [_filter_delegation_toolset(toolset, delegation_tags) for toolset in all_toolsets]
        self_fork_base_model = (
            infer_model(model, extra_headers=model_extra_headers) if isinstance(model, str) else base_model
        )
        self_fork_agent: Agent[AgentContext, str] = add_toolset_instructions(
            Agent(
                model=self_fork_base_model,
                system_prompt=effective_system_prompt,
                model_settings=model_settings,
                deps_type=AgentContext,
                output_type=str,
                tools=agent_tools or (),
                toolsets=self_fork_toolsets if self_fork_toolsets else None,
                capabilities=[
                    *cast(list[AbstractCapability[Any]], all_capabilities),
                    ProcessHistory(create_system_prompt_filter(system_prompt=effective_system_prompt)),
                ],
                retries=agent_retries,
                defer_model_check=defer_model_check,
                end_strategy=end_strategy,  # type: ignore[arg-type]
                name="self",
            ),
            self_fork_toolsets,
        )
        attach_message_bus_guard(self_fork_agent)
        ctx.self_fork_agent = self_fork_agent

    agent: Agent[AgentDepsT, OutputT] = add_toolset_instructions(
        Agent(
            model=effective_model,
            system_prompt=effective_system_prompt,
            model_settings=model_settings,
            deps_type=context_type,
            output_type=output_type,
            tools=agent_tools or (),
            toolsets=all_toolsets if all_toolsets else None,
            capabilities=[
                *all_capabilities,
                ProcessHistory(create_system_prompt_filter(system_prompt=effective_system_prompt)),
            ],
            retries=agent_retries,
            defer_model_check=defer_model_check,
            end_strategy=end_strategy,  # type: ignore[arg-type]
            name=agent_name,
        ),
        all_toolsets,
    )

    # Attach message bus guard for pending message handling
    attach_message_bus_guard(agent)

    logger.debug(
        "Agent created: toolsets=%d, capabilities=%d",
        len(all_toolsets) if all_toolsets else 0,
        len(all_capabilities),
    )
    runtime_extensions = list(lifecycle_extensions or [])
    ctx.lifecycle_extensions = runtime_extensions
    return AgentRuntime[AgentDepsT, OutputT, EnvT](
        env=actual_env,
        ctx=ctx,
        agent=agent,
        core_toolset=core_toolset,
        lifecycle_extensions=runtime_extensions,
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


_DEFAULT_RESUME_PROMPT = """
The previous streaming model request failed before the agent finished.
Continue the task from the available conversation history. Avoid repeating completed work.
""".strip()


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
    resume_prompt: UserPromptT | None = None,
    resume_prompt_factory: ResumePromptFactory | None = None,
    # Lifecycle events
    emit_lifecycle_events: bool = True,
) -> AsyncIterator[AgentStreamer[AgentDepsT, OutputT]]:
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
        resume_max_attempts: Maximum total attempts when resume_on_error is enabled.
            If None, uses ctx.model_cfg.stream_resume_max_attempts.
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

    # Extract agent from runtime
    agent = runtime.agent
    extensions = list(runtime.lifecycle_extensions)

    # Create a fresh context for this run to isolate per-run state.
    # This prevents ContextVar tokens and other run-specific state from
    # leaking between consecutive runs, which causes "was created in a
    # different Context" errors when pydantic-ai's cleanup runs in the
    # wrong contextvars.Context (see pydantic-ai issue #674).
    fresh_ctx = runtime.ctx.prepare_new_run()
    runtime.ctx = fresh_ctx
    ctx = fresh_ctx
    ctx.lifecycle_extensions = extensions

    # Enable streaming for emit_event.
    # Must use object.__setattr__ because Pydantic v2 silently ignores
    # normal attribute assignment on private attrs of model_copy() instances.
    object.__setattr__(ctx, "_stream_queue_enabled", True)

    output_queue: asyncio.Queue[StreamEvent] = asyncio.Queue()
    main_done = asyncio.Event()
    poll_done = asyncio.Event()
    partial_text = PartialTextAccumulator()

    logger.debug(
        "Starting stream_agent with user_prompt=%s",
        user_prompt[:100] if isinstance(user_prompt, str) else type(user_prompt),
    )

    # Build main agent info
    main_agent_info = AgentInfo(agent_id="main", agent_name=agent.name or "main")

    async def process_node(
        node: ModelRequestNode[AgentDepsT, OutputT] | CallToolsNode[AgentDepsT, OutputT],
        run: AgentRun[AgentDepsT, OutputT],
    ) -> None:
        """Process a single node with hooks."""
        # PRE NODE HOOK
        logger.debug("Processing node: %s", type(node).__name__)
        node_ctx = NodeHookContext(agent_info=main_agent_info, node=node, run=run, output_queue=output_queue)
        await run_extension_method(extensions, "on_before_node", node_ctx, logger=logger)
        if pre_node_hook:
            await pre_node_hook(node_ctx)
        try:
            async with node.stream(run.ctx) as request_stream:
                async for event in request_stream:
                    # PRE EVENT HOOK
                    event_ctx = EventHookContext(
                        agent_info=main_agent_info, event=event, node=node, run=run, output_queue=output_queue
                    )
                    await run_extension_method(extensions, "on_before_event", event_ctx, logger=logger)
                    if pre_event_hook:
                        await pre_event_hook(event_ctx)

                    wrapped_event = ctx.tool_id_wrapper.wrap_event(event)
                    partial_text.observe(wrapped_event)
                    await output_queue.put(
                        StreamEvent(
                            agent_id=main_agent_info.agent_id,
                            agent_name=main_agent_info.agent_name,
                            event=wrapped_event,
                        )
                    )

                    # POST EVENT HOOK
                    await run_extension_method(extensions, "on_after_event", event_ctx, logger=logger)
                    if post_event_hook:
                        await post_event_hook(event_ctx)
        except TypeError as e:
            # anyio CancelScope fails when asyncio.current_task() returns None during
            # httpx stream cleanup after cancellation.  Benign -- stream already closing.
            if "cannot create weak reference to 'NoneType'" in str(e):
                logger.debug("Suppressed anyio CancelScope cleanup error from streaming: %s", e)
            else:
                raise
        except ValueError as e:
            # Suppress ContextVar cleanup errors from pydantic-ai's streaming internals.
            # When pydantic-ai's _streaming_handler runs in a child task (wrap_task) with a copied
            # context, set_current_run_context's __exit__ may call ContextVar.reset(token) in a
            # Context object different from where the token was created.  This is harmless -- all
            # stream events have already been emitted -- but the ValueError propagates through
            # pydantic-ai's on_model_request_error (which re-raises by default).
            if "was created in a different Context" in str(e):
                logger.debug("Suppressed ContextVar cleanup error from pydantic-ai streaming: %s", e)
            else:
                raise

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
        ctx.update_usage_snapshot_entry(
            agent_id=main_agent_info.agent_id,
            agent_name=main_agent_info.agent_name,
            model_id=base_model.model_name,
            usage=run.usage,
            source="main_model_request",
        )

        await emit_lifecycle_event(
            ModelRequestCompleteEvent(
                event_id=ctx.run_id,
                loop_index=current_loop,
                duration_seconds=time.perf_counter() - node_start_time,
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

    async def process_all_nodes(run: AgentRun[AgentDepsT, OutputT]) -> None:
        """Process all nodes in the agent run with lifecycle events."""
        async for node in run:
            node_start_time = time.perf_counter()

            if Agent.is_user_prompt_node(node) or Agent.is_end_node(node):
                # Skip user_prompt and end nodes - their info is in AgentExecution events
                continue
            elif Agent.is_model_request_node(node):
                await handle_model_request_node(node, run, node_start_time)
            elif Agent.is_call_tools_node(node):
                await handle_call_tools_node(node, run, node_start_time)

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

            start_ctx = AgentStartContext(
                runtime=runtime, agent_info=main_agent_info, output_queue=output_queue, run=run
            )
            await run_extension_method(extensions, "on_agent_start", start_ctx, logger=logger)
            if on_agent_start:
                await on_agent_start(start_ctx)

            await process_all_nodes(run)

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

    failure_event_emitted = False

    def stringify_exception(exc: BaseException) -> str:
        try:
            error_str = str(exc)
        except Exception:
            error_str = repr(exc)
        return error_str or repr(exc)

    def extract_resume_history(
        run: AgentRun[AgentDepsT, OutputT] | None,
        fallback_history: Sequence[ModelMessage] | None,
    ) -> list[ModelMessage]:
        if run is not None:
            try:
                messages = list(run.all_messages())
            except Exception:
                logger.debug("Failed to extract run messages for stream resume", exc_info=True)
            else:
                if messages:
                    return messages
        return list(fallback_history or [])

    def history_has_unreturned_tool_calls(history: Sequence[ModelMessage]) -> bool:
        unreturned_tool_call_ids: set[str] = set()
        for message in history:
            if isinstance(message, ModelResponse):
                for part in message.parts:
                    if isinstance(part, BaseToolCallPart):
                        unreturned_tool_call_ids.add(part.tool_call_id)
            elif isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, ToolReturnPart | RetryPromptPart):
                        unreturned_tool_call_ids.discard(part.tool_call_id)
        return bool(unreturned_tool_call_ids)

    def split_resume_prompt_for_tool_call_history(
        history: Sequence[ModelMessage] | None,
        prompt: UserPromptT | None,
    ) -> tuple[Sequence[ModelMessage] | None, UserPromptT | None]:
        if prompt is None or not history or not history_has_unreturned_tool_calls(history):
            return history, prompt
        return close_unreturned_tool_calls(history, "new user prompt requested before tool results"), prompt

    def close_unreturned_tool_calls(history: Sequence[ModelMessage], error_message: str) -> list[ModelMessage]:
        unreturned_tool_calls: dict[str, str] = {}
        for message in history:
            if isinstance(message, ModelResponse):
                for part in message.parts:
                    if isinstance(part, BaseToolCallPart):
                        unreturned_tool_calls[part.tool_call_id] = part.tool_name
            elif isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, ToolReturnPart | RetryPromptPart):
                        unreturned_tool_calls.pop(part.tool_call_id, None)

        normalized_history = list(history)
        if not unreturned_tool_calls:
            return normalized_history

        normalized_history.append(
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name=tool_name,
                        tool_call_id=tool_call_id,
                        content=(
                            "Tool execution was interrupted by a stream error before a result was available. "
                            f"stream_error={error_message}"
                        ),
                        outcome="failed",
                    )
                    for tool_call_id, tool_name in unreturned_tool_calls.items()
                ]
            )
        )
        return normalized_history

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
        if resume_prompt is not None:
            return resume_prompt
        if ctx.model_cfg.stream_resume_prompt is not None:
            return ctx.model_cfg.stream_resume_prompt
        return _DEFAULT_RESUME_PROMPT

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
        effective_resume_on_error = ctx.model_cfg.stream_resume_on_error if resume_on_error is None else resume_on_error
        effective_resume_max_attempts = (
            ctx.model_cfg.stream_resume_max_attempts if resume_max_attempts is None else resume_max_attempts
        )
        max_attempts = max(1, effective_resume_max_attempts) if effective_resume_on_error else 1
        attempt_index = 0
        current_user_prompt = effective_user_prompt
        current_deferred_tool_results = effective_deferred_tool_results
        current_message_history: Sequence[ModelMessage] | None = message_history

        while True:
            try:
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
                recoverable = attempt_index + 1 < max_attempts
                error_str = await emit_execution_failed_event(
                    e,
                    stream_start_time,
                    attempt_index=attempt_index,
                    recoverable=recoverable,
                )
                if not recoverable:
                    raise

                next_attempt_index = attempt_index + 1
                resume_history = close_unreturned_tool_calls(
                    extract_resume_history(streamer.run, current_message_history),
                    error_str,
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
                    "Resuming stream_agent after error attempt_index=%s next_attempt_index=%s max_attempts=%s error_type=%s error=%s",
                    attempt_index,
                    next_attempt_index,
                    max_attempts,
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
        return effective_user_prompt, effective_deferred_tool_results

    async def run_main() -> None:
        """Run the main agent and push events to output_queue."""
        logger.debug("Main agent task started")

        stream_start_time = time.perf_counter()

        try:
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
                            else ctx.agent_registry.get(agent_id)
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
    )

    try:
        yield streamer
    except Exception as e:
        logger.exception("Uncaught exception in stream_agent context")
        streamer.exception = e
        if raise_on_error:
            raise  # Re-raise so caller can handle it
    else:
        if (run := streamer.run) and (result := run.result) and isinstance(result.output, DeferredToolRequests):
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
            gather_interrupt: BaseException | None = None
            cleanup_deadline = time.perf_counter() + 5.0
            while any(not t.done() for t in [main_task, poll_task]):
                if time.perf_counter() > cleanup_deadline:
                    logger.warning("Cleanup deadline exceeded (5s), abandoning remaining tasks")
                    break
                not_done = [t for t in [main_task, poll_task] if not t.done()]
                try:
                    await asyncio.gather(*not_done, return_exceptions=True)
                except BaseException as gather_exc:
                    logger.debug("gather interrupted during cleanup: %s", type(gather_exc).__name__)
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
            _restore_task_cancellation(current_task, cleared_cancellations)
