"""Capability-first interactive agent example for ya-agent-sdk.

The example demonstrates:

- explicit capability composition;
- streamed output;
- Pydantic AI deferred approvals and structured questions;
- portable named subagents and self forks through one execution service; and
- message history plus resumable context persistence.

Run from the repository root after configuring ``examples/.env``::

    uv run python examples/general.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import cast

from dotenv import load_dotenv
from pydantic_ai import (
    AgentSpec,
    DeferredToolRequests,
    DeferredToolResults,
    ModelSettings,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    prices,
)
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelMessagesTypeAdapter,
    OutputToolCallEvent,
    OutputToolResultEvent,
    PartDeltaEvent,
    TextPartDelta,
    ToolCallPart,
)
from ya_agent_sdk.agents.main import create_agent, stream_agent
from ya_agent_sdk.capabilities import (
    DocumentConversionCapability,
    FilesystemCapability,
    MediaReadCapability,
    NoteCapability,
    RuntimeFoundationCapability,
    ShellCapability,
    TaskCapability,
    ToolApprovalCapability,
    ToolObservationCapability,
    ToolSupersessionCapability,
    ToolTimeoutCapability,
    ToolVisibilityCapability,
    UserInteractionCapability,
    WebContentCapability,
    WebSearchCapability,
    build_default_capability_catalog,
)
from ya_agent_sdk.context import ModelConfig, ModelFeature, ResumableState, StreamEvent
from ya_agent_sdk.interactions import (
    DeferredApprovalResolution,
    DeferredCallResolution,
    DeferredInteractionResolution,
    DeferredInteractionResolver,
)
from ya_agent_sdk.presets import ANTHROPIC_DEFAULT
from ya_agent_sdk.subagents import (
    DelegationCapability,
    InMemorySubagentExecutionStore,
    InProcessSubagentDriver,
    SelfForkPolicy,
    SubagentExecutionMode,
    SubagentExecutionService,
    SubagentPlanResolver,
    SubagentRegistry,
    SubagentSpec,
)
from ya_agent_sdk.toolsets.core.interaction import (
    AskUserQuestionTool,
    UserQuestionAnswers,
    format_user_question_answers,
    parse_ask_user_question_args,
    parse_user_question_answer,
)

load_dotenv()

MODEL = "anthropic:claude-4-5-sonnet-by-all"
MODEL_SETTINGS = cast(ModelSettings, ANTHROPIC_DEFAULT)
PROMPT_FILE = Path(__file__).parent / "prompts" / "general.md"
SESSION_DIR = Path(__file__).parent / ".session"
MESSAGE_HISTORY_FILE = SESSION_DIR / "message_history.json"
STATE_FILE = SESSION_DIR / "context_state.json"
MAX_TOOL_CONTENT_LENGTH = 200


def load_system_prompt() -> str:
    """Load the example's system prompt."""
    return PROMPT_FILE.read_text(encoding="utf-8")


def load_message_history() -> list[ModelMessage] | None:
    """Load canonical Pydantic AI messages from the previous completed turn."""
    if not MESSAGE_HISTORY_FILE.exists():
        return None
    try:
        return ModelMessagesTypeAdapter.validate_json(MESSAGE_HISTORY_FILE.read_bytes())
    except ValueError as exc:
        print(f"Warning: failed to load message history: {exc}")
        return None


def load_state() -> ResumableState | None:
    """Load SDK-owned resumable context state."""
    if not STATE_FILE.exists():
        return None
    try:
        return ResumableState.model_validate_json(STATE_FILE.read_text(encoding="utf-8"))
    except ValueError as exc:
        print(f"Warning: failed to load context state: {exc}")
        return None


def save_session(messages_json: bytes, state: ResumableState) -> None:
    """Persist one completed turn."""
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    MESSAGE_HISTORY_FILE.write_bytes(messages_json)
    STATE_FILE.write_text(state.model_dump_json(indent=2), encoding="utf-8")


def get_user_input(prompt: str = "You: ") -> str:
    """Read terminal input and treat interruption as an empty response."""
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def truncate(value: object, max_length: int = MAX_TOOL_CONTENT_LENGTH) -> str:
    """Return a bounded display representation."""
    text = str(value)
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def print_stream_event(stream_event: StreamEvent) -> None:
    """Render text and tool events from the SDK stream."""
    event = stream_event.event
    if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
        print(event.part.content, end="", flush=True)
    elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
        print(event.delta.content_delta, end="", flush=True)
    elif isinstance(event, PartEndEvent) and isinstance(event.part, TextPart):
        print()
    elif isinstance(event, FunctionToolCallEvent | OutputToolCallEvent):
        print(f"\n[ToolCall] {event.part.tool_name}({truncate(event.part.args)})")
    elif isinstance(event, FunctionToolResultEvent):
        print(f"[ToolResult] {event.part.tool_name}: {truncate(event.content)}")
    elif isinstance(event, OutputToolResultEvent):
        print(f"[ToolResult] {event.part.tool_name}: {truncate(event.part.content)}")


def format_approval(call: ToolCallPart) -> str:
    """Format one native Pydantic AI approval request."""
    arguments = call.args if isinstance(call.args, str) else json.dumps(call.args, ensure_ascii=False, indent=2)
    return f"Tool: {call.tool_name}\nArgs: {truncate(arguments, 500)}\nID: {call.tool_call_id}"


def collect_deferred_results(requests: DeferredToolRequests) -> DeferredToolResults:
    """Resolve every approval or supported external call explicitly."""
    resolutions: list[DeferredInteractionResolution] = []

    for call in requests.approvals:
        print(f"\n{format_approval(call)}")
        response = get_user_input("Approve? [y/N or rejection reason]: ")
        approved = response.lower() in {"y", "yes", "approve"}
        resolutions.append(
            DeferredApprovalResolution(
                tool_call_id=call.tool_call_id,
                approved=approved,
                reason=None if approved else response or "User rejected the tool call.",
            )
        )

    for call in requests.calls:
        if call.tool_name != AskUserQuestionTool.name:
            raise RuntimeError(f"Unsupported deferred tool: {call.tool_name}")
        question_request = parse_ask_user_question_args(call.args)
        answers = {
            question.question: parse_user_question_answer(
                question,
                get_user_input(f"{question.question}: "),
            )
            for question in question_request.questions
        }
        result = format_user_question_answers(
            UserQuestionAnswers(
                questions=question_request.questions,
                answers=answers,
            )
        )
        resolutions.append(
            DeferredCallResolution(
                tool_call_id=call.tool_call_id,
                result=result,
            )
        )

    return DeferredInteractionResolver().resolve(requests, resolutions)


def build_delegation_capability() -> DelegationCapability:
    """Build one process-local portable delegation service for this example."""
    catalog = build_default_capability_catalog()
    resolver = SubagentPlanResolver(catalog, default_model=MODEL)
    child_capabilities = [
        {"RuntimeFoundationCapability": {}},
        {"FilesystemCapability": {"writable": False}},
        {"WebSearchCapability": {}},
        {"WebContentCapability": {}},
        {"NoteCapability": {}},
        {"ToolObservationCapability": {}},
        {"ToolTimeoutCapability": {}},
    ]
    researcher = resolver.resolve(
        SubagentSpec(
            route="researcher",
            execution_modes=(
                SubagentExecutionMode.foreground,
                SubagentExecutionMode.background,
            ),
            agent=AgentSpec.from_dict({
                "name": "researcher",
                "description": "Research a bounded question and return sourced findings",
                "model": MODEL,
                "model_settings": dict(MODEL_SETTINGS),
                "instructions": (
                    "Investigate only the delegated question. Return concise findings, "
                    "sources, uncertainties, and no unsupported conclusions."
                ),
                "capabilities": child_capabilities,
            }),
        )
    )
    self_fork = resolver.resolve_self(
        SelfForkPolicy(
            agent=AgentSpec.from_dict({
                "name": "self",
                "description": "Fork the parent context for a bounded independent task",
                "model": MODEL,
                "model_settings": dict(MODEL_SETTINGS),
                "instructions": load_system_prompt(),
                "capabilities": child_capabilities,
            }),
            execution_modes=(
                SubagentExecutionMode.foreground,
                SubagentExecutionMode.background,
            ),
        )
    )
    registry = SubagentRegistry([researcher, self_fork])
    service = SubagentExecutionService(
        registry,
        InMemorySubagentExecutionStore(),
        InProcessSubagentDriver(
            custom_capability_types=catalog.custom_capability_types,
        ),
    )
    return DelegationCapability(registry=registry, service=service)


async def main() -> None:
    """Run one interactive turn, including any deferred continuations."""
    message_history = load_message_history()
    state = load_state()
    user_prompt = get_user_input()
    if not user_prompt:
        print("No input provided, exiting.")
        return

    runtime = create_agent(
        model=MODEL,
        model_settings=MODEL_SETTINGS,
        system_prompt=load_system_prompt(),
        capabilities=[
            RuntimeFoundationCapability(),
            MediaReadCapability(),
            DocumentConversionCapability(),
            FilesystemCapability(),
            ShellCapability(),
            WebSearchCapability(),
            WebContentCapability(),
            TaskCapability(),
            NoteCapability(),
            UserInteractionCapability(),
            build_delegation_capability(),
            ToolSupersessionCapability(),
            ToolVisibilityCapability(),
            ToolApprovalCapability(tools=frozenset({"shell_exec"})),
            ToolObservationCapability(),
            ToolTimeoutCapability(),
        ],
        model_cfg=ModelConfig(
            context_window=200_000,
            capabilities={ModelFeature.vision},
        ),
        state=state,
        output_type=[str, DeferredToolRequests],
    )

    deferred_results: DeferredToolResults | None = None
    async with runtime:
        while True:
            async with stream_agent(
                runtime,
                user_prompt=user_prompt,
                message_history=message_history,
                deferred_tool_results=deferred_results,
            ) as stream:
                async for event in stream:
                    print_stream_event(event)
                stream.raise_if_exception()
                run = stream.run

            if run is None or run.result is None:
                raise RuntimeError("Agent run completed without a result")
            message_history = run.all_messages()
            output = run.result.output
            if isinstance(output, DeferredToolRequests):
                deferred_results = collect_deferred_results(output)
                user_prompt = None
                continue

            print(f"\nUsage: {run.result.usage}")
            save_session(run.all_messages_json(), runtime.ctx.export_state())
            break


if __name__ == "__main__":
    # Keep price refresh alive for the application, not for each agent runtime.
    with prices.update_in_background():
        asyncio.run(main())
