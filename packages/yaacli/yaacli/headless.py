from __future__ import annotations

import asyncio
import json
import sys
import uuid
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from pydantic_ai import AgentRun, DeferredToolRequests, DeferredToolResults, ToolDenied, UsageLimits
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, RetryPromptPart
from ya_agent_sdk.agents.main import AgentInterrupted, stream_agent
from ya_agent_sdk.context import PROJECT_GUIDANCE_TAG, USER_RULES_TAG, ResumableState
from ya_agent_sdk.utils import get_latest_request_usage
from ya_agent_stream_protocol.agui import AguiReplayConfig, validate_display_events
from ya_agent_stream_protocol.sdk import AguiAdapterConfig, AguiEventAdapter

from yaacli.config import ConfigManager, YaacliConfig
from yaacli.display_replay import MAX_DISPLAY_REPLAY_LOAD_BYTES, BoundedDisplayReplay
from yaacli.errors import safe_exception_str
from yaacli.hooks import emit_context_update
from yaacli.logging import get_logger
from yaacli.model_profiles import ResolvedModelProfile, get_model_profile, get_startup_model_profile
from yaacli.runtime import create_tui_runtime
from yaacli.sessions import read_head_artifacts, restore_resumable_state_safely, save_session_turn

logger = get_logger(__name__)

_DEFAULT_MAX_TURNS_PER_SESSION = 20
_DEFAULT_MAX_SESSIONS = 100
YAACLI_AGUI_ADAPTER_CONFIG = AguiAdapterConfig(run_event_prefix="yaacli", stream_metadata_prefix="yaacli")
YAACLI_AGUI_REPLAY_CONFIG = AguiReplayConfig(
    agent_id_field="yaacliAgentId",
    main_agent_id="main",
    drop_subagent_detail_events=True,
)


def _positive_int_config(value: object, default: int) -> int:
    return value if isinstance(value, int) and value > 0 else default


def _optional_positive_int_config(value: object) -> int | None:
    return value if isinstance(value, int) and value > 0 else None


def _deny_deferred_tool_requests(deferred_requests: DeferredToolRequests, *, reason: str) -> DeferredToolResults:
    results = DeferredToolResults()
    for request in deferred_requests.approvals:
        results.approvals[request.tool_call_id] = ToolDenied(message=reason)
    for request in deferred_requests.calls:
        results.calls[request.tool_call_id] = RetryPromptPart(
            content=reason,
            tool_name=request.tool_name,
            tool_call_id=request.tool_call_id,
        )
    return results


def _completed_run_request_count(run: AgentRun[Any, Any]) -> int:
    """Return a conservative model-request count for one completed stream run."""
    requests = run.usage.requests
    return requests if isinstance(requests, int) and requests > 0 else 1


@dataclass(slots=True)
class HeadlessRunResult:
    session_id: str
    output_text: str | None
    display_messages: list[dict[str, Any]]


class HeadlessEventSink:
    def __init__(self, output_stream: TextIO | None = None) -> None:
        self.replay = BoundedDisplayReplay(config=YAACLI_AGUI_REPLAY_CONFIG)
        self._output_stream = output_stream if output_stream is not None else sys.stdout

    def emit(self, event: dict[str, Any]) -> None:
        self.replay.append(event)
        self._output_stream.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._output_stream.flush()

    def emit_many(self, events: list[dict[str, Any]]) -> None:
        for event in events:
            self.emit(event)


def _load_guidance_files(config_manager: ConfigManager, working_dir: Path) -> tuple[str | None, str | None]:
    project_guidance = None
    user_rules = None

    agents_path = working_dir / "AGENTS.md"
    if agents_path.exists() and agents_path.is_file():
        try:
            content = agents_path.read_text(encoding="utf-8")
            if content.strip():
                project_guidance = (
                    f"<{PROJECT_GUIDANCE_TAG} name={agents_path.name}>\n{content}\n</{PROJECT_GUIDANCE_TAG}>"
                )
        except Exception as exc:
            logger.warning("Failed to read %s: %s", agents_path, exc)

    rules_path = config_manager.config_dir / "RULES.md"
    if rules_path.exists() and rules_path.is_file():
        try:
            content = rules_path.read_text(encoding="utf-8")
            if content.strip():
                user_rules = (
                    f"<{USER_RULES_TAG} location={rules_path.absolute().as_posix()}>\n{content}\n</{USER_RULES_TAG}>"
                )
        except Exception as exc:
            logger.warning("Failed to read %s: %s", rules_path, exc)

    return project_guidance, user_rules


def _build_user_prompt(config_manager: ConfigManager, working_dir: Path, prompt: str) -> str | list[Any]:
    project_guidance, user_rules = _load_guidance_files(config_manager, working_dir)
    if not project_guidance and not user_rules:
        return prompt
    parts: list[Any] = [prompt]
    if project_guidance:
        parts.append(project_guidance)
    if user_rules:
        parts.append(user_rules)
    return parts


def _load_session_artifacts(
    config_manager: ConfigManager,
    session_id: str | None,
) -> tuple[str, list[ModelMessage] | None, ResumableState | None, list[dict[str, Any]]]:
    if session_id is None:
        return uuid.uuid4().hex[:12], None, None, []

    artifacts = read_head_artifacts(
        config_manager,
        session_id,
        max_display_messages_bytes=MAX_DISPLAY_REPLAY_LOAD_BYTES,
    )
    message_history = ModelMessagesTypeAdapter.validate_json(artifacts.message_history_json)
    state = (
        ResumableState.model_validate_json(artifacts.context_state_json)
        if artifacts.context_state_json is not None
        else None
    )
    display_messages: list[dict[str, Any]] = []
    if artifacts.display_messages_json is not None:
        try:
            display_messages = validate_display_events(json.loads(artifacts.display_messages_json))
        except Exception as exc:
            logger.warning(
                "Session %s has invalid display replay; continuing from message history: %s",
                artifacts.session_id,
                safe_exception_str(exc),
            )
    return artifacts.session_id, message_history, state, display_messages


def _save_session_artifacts(
    *,
    config: YaacliConfig,
    config_manager: ConfigManager,
    session_id: str,
    working_dir: Path,
    message_history: list[ModelMessage],
    state: ResumableState,
    display_messages: list[dict[str, Any]],
    output_text: str | None,
    model_profile_id: str | None,
    model_label: str | None,
    model: str | None,
) -> None:
    save_session_turn(
        config_manager=config_manager,
        session_id=session_id,
        working_dir=working_dir,
        model_profile_id=model_profile_id,
        model_label=model_label,
        model=model,
        message_history_json=ModelMessagesTypeAdapter.dump_json(message_history, indent=2),
        context_state_json=state.model_dump_json(indent=2),
        display_messages=display_messages,
        output_text=output_text,
        save_reason="headless_success",
        max_turns=_positive_int_config(
            getattr(getattr(config, "session", None), "max_turns_per_session", None), _DEFAULT_MAX_TURNS_PER_SESSION
        ),
        max_sessions=_positive_int_config(
            getattr(getattr(config, "session", None), "max_sessions", None), _DEFAULT_MAX_SESSIONS
        ),
        max_session_age_days=_optional_positive_int_config(
            getattr(getattr(config, "session", None), "max_session_age_days", None)
        ),
    )


async def run_headless_prompt(
    *,
    config: YaacliConfig,
    config_manager: ConfigManager,
    prompt: str,
    working_dir: Path,
    session_id: str | None = None,
    model_profile_id: str | None = None,
    worker: bool = False,
) -> HeadlessRunResult:
    """Run a single prompt, reserving stdout exclusively for NDJSON events."""
    ndjson_stream = sys.stdout
    with redirect_stdout(sys.stderr):
        return await _run_headless_prompt(
            config=config,
            config_manager=config_manager,
            prompt=prompt,
            working_dir=working_dir,
            session_id=session_id,
            model_profile_id=model_profile_id,
            worker=worker,
            ndjson_stream=ndjson_stream,
        )


async def _run_headless_prompt(
    *,
    config: YaacliConfig,
    config_manager: ConfigManager,
    prompt: str,
    working_dir: Path,
    session_id: str | None,
    model_profile_id: str | None,
    worker: bool,
    ndjson_stream: TextIO,
) -> HeadlessRunResult:
    mcp_config = config_manager.load_mcp_config()
    resolved_session_id, message_history, restored_state, restored_display_messages = _load_session_artifacts(
        config_manager,
        session_id,
    )

    effective_model_profile: ResolvedModelProfile | None
    if model_profile_id is not None:
        effective_model_profile = get_model_profile(config, model_profile_id)
        if effective_model_profile is None:
            raise ValueError(f"Unknown model profile: {model_profile_id}")
    else:
        effective_model_profile = get_startup_model_profile(config, config_manager.config_dir)

    runtime = create_tui_runtime(
        config=config,
        mcp_config=mcp_config,
        working_dir=working_dir,
        config_dir=config_manager.config_dir,
        model_profile=effective_model_profile,
        enable_async_subagents=False,
        enable_delegate_subagents=not worker,
    )
    run_id = uuid.uuid4().hex[:12]
    adapter = AguiEventAdapter(session_id=resolved_session_id, run_id=run_id, config=YAACLI_AGUI_ADAPTER_CONFIG)
    sink = HeadlessEventSink(ndjson_stream)
    output_text: str | None = None

    try:
        async with runtime:
            runtime.ctx.injected_context_tags = (
                *runtime.ctx.injected_context_tags,
                PROJECT_GUIDANCE_TAG,
                USER_RULES_TAG,
            )
            if restored_state is not None:
                restore_resumable_state_safely(restored_state, runtime.ctx)
                if runtime.ctx.usage_snapshot_entries:
                    runtime.ctx.usage_snapshot_entries.clear()

            sink.replay.extend_snapshot(restored_display_messages)
            sink.emit(adapter.build_run_started_event())

            user_prompt = _build_user_prompt(config_manager, working_dir, prompt)
            deferred_tool_results: DeferredToolResults | None = None
            cumulative_model_requests = 0
            max_model_requests = config.general.max_requests
            while True:
                remaining_model_requests = max_model_requests - cumulative_model_requests
                if remaining_model_requests <= 0:
                    raise RuntimeError(
                        "Headless deferred continuation exhausted the cumulative "
                        f"model request limit of {max_model_requests}."
                    )

                async with stream_agent(
                    runtime,  # type: ignore[arg-type]
                    user_prompt=user_prompt,
                    message_history=message_history,
                    deferred_tool_results=deferred_tool_results,
                    usage_limits=UsageLimits(request_limit=remaining_model_requests),
                    post_node_hook=emit_context_update,
                    resume_on_error=config.general.agent_stream_resume_on_error,
                    resume_max_attempts=config.general.agent_stream_resume_max_attempts,
                    resume_prompt=config.general.agent_stream_resume_prompt,
                ) as streamer:
                    async for stream_event in streamer:
                        sink.emit_many(adapter.adapt_stream_event(stream_event))

                    streamer.raise_if_exception()
                    if streamer.run is None:
                        raise RuntimeError("Stream agent completed without run context.")

                    result = streamer.run.result
                    output = result.output if result is not None else None
                    message_history = streamer.recoverable_messages()
                    latest_usage = get_latest_request_usage(message_history)
                    if latest_usage is not None:
                        runtime.ctx.build_usage_snapshot()

                    if isinstance(output, DeferredToolRequests):
                        if not output.approvals and not output.calls:
                            raise RuntimeError("Agent returned an empty DeferredToolRequests payload.")

                        denial_reason = "Headless mode denies HITL requests by default."
                        sink.emit(
                            adapter.build_run_custom_event(
                                "hitl_auto_denied",
                                {
                                    "approval_count": len(output.approvals),
                                    "approvals": [item.tool_call_id for item in output.approvals],
                                    "call_count": len(output.calls),
                                    "calls": [item.tool_call_id for item in output.calls],
                                    "reason": denial_reason,
                                },
                            )
                        )
                        deferred_tool_results = _deny_deferred_tool_requests(output, reason=denial_reason)
                        # Every stream starts with fresh pydantic-ai usage. Carry its model
                        # requests forward so repeated HITL continuations share one budget.
                        cumulative_model_requests += _completed_run_request_count(streamer.run)
                        user_prompt = None
                        continue

                    output_text = str(output) if output is not None else None
                    break

            finished_event = adapter.build_run_finished_event(result={"output_text": output_text})
            persisted_replay = BoundedDisplayReplay(config=YAACLI_AGUI_REPLAY_CONFIG)
            persisted_replay.extend_snapshot(sink.replay.snapshot())
            persisted_replay.append(finished_event)
            resumable_state = runtime.ctx.export_state(include_usage_ledger=False)

        _save_session_artifacts(
            config=config,
            config_manager=config_manager,
            session_id=resolved_session_id,
            working_dir=working_dir,
            message_history=message_history,
            state=resumable_state,
            display_messages=persisted_replay.snapshot(),
            output_text=output_text,
            model_profile_id=effective_model_profile.id if effective_model_profile is not None else None,
            model_label=effective_model_profile.label if effective_model_profile is not None else None,
            model=(
                effective_model_profile.model
                if effective_model_profile is not None
                else config.general.model
                if isinstance(config, YaacliConfig)
                else None
            ),
        )
        sink.emit(finished_event)
    except asyncio.CancelledError:
        sink.emit(adapter.build_run_custom_event("run_cancelled", {"reason": "cancelled"}))
        raise
    except (AgentInterrupted, KeyboardInterrupt):
        sink.emit(adapter.build_run_custom_event("run_cancelled", {"reason": "interrupted"}))
        raise
    except Exception as exc:
        sink.emit(adapter.build_run_error_event(message=safe_exception_str(exc), code=type(exc).__name__))
        raise

    return HeadlessRunResult(
        session_id=resolved_session_id,
        output_text=output_text,
        display_messages=sink.replay.snapshot(),
    )
