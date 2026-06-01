from __future__ import annotations

import json
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic_ai import DeferredToolRequests, DeferredToolResults, ToolDenied, UsageLimits
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, RetryPromptPart
from ya_agent_sdk.agents.main import AgentInterrupted, stream_agent
from ya_agent_sdk.context import PROJECT_GUIDANCE_TAG, USER_RULES_TAG, ResumableState
from ya_agent_sdk.utils import get_latest_request_usage

from yaacli.agui import DisplayEventAdapter, DisplayReplayBuffer, validate_display_events
from yaacli.config import ConfigManager, YaacliConfig
from yaacli.hooks import emit_context_update
from yaacli.logging import get_logger
from yaacli.model_profiles import get_model_profile
from yaacli.runtime import create_tui_runtime
from yaacli.sessions import get_head_artifact_paths, save_session_turn

logger = get_logger(__name__)

_DEFAULT_MAX_TURNS_PER_SESSION = 20
_DEFAULT_MAX_SESSIONS = 100


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


@dataclass(slots=True)
class HeadlessRunResult:
    session_id: str
    output_text: str | None
    display_messages: list[dict[str, Any]]


class HeadlessEventSink:
    def __init__(self) -> None:
        self.replay = DisplayReplayBuffer()

    def emit(self, event: dict[str, Any]) -> None:
        self.replay.append(event)
        sys.stdout.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
        sys.stdout.flush()

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

    paths = get_head_artifact_paths(config_manager, session_id)
    history_file = paths.message_history_file
    state_file = paths.context_state_file
    display_file = paths.display_messages_file

    message_history = (
        ModelMessagesTypeAdapter.validate_json(history_file.read_bytes())
        if history_file is not None and history_file.exists()
        else None
    )
    state = (
        ResumableState.model_validate_json(state_file.read_text())
        if state_file is not None and state_file.exists()
        else None
    )
    display_messages = (
        validate_display_events(json.loads(display_file.read_text()))
        if display_file is not None and display_file.exists()
        else []
    )
    return paths.session_id, message_history, state, display_messages


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
) -> None:
    save_session_turn(
        config_manager=config_manager,
        session_id=session_id,
        working_dir=working_dir,
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
    """Run a single prompt and stream display events to stdout as NDJSON."""
    mcp_config = config_manager.load_mcp_config()
    resolved_session_id, message_history, restored_state, restored_display_messages = _load_session_artifacts(
        config_manager,
        session_id,
    )

    model_profile = None
    if model_profile_id:
        model_profile = get_model_profile(config, model_profile_id)
        if model_profile is None:
            raise ValueError(f"Unknown model profile: {model_profile_id}")

    runtime = create_tui_runtime(
        config=config,
        mcp_config=mcp_config,
        working_dir=working_dir,
        config_dir=config_manager.config_dir,
        model_profile=model_profile,
        enable_async_subagents=False,
        enable_delegate_subagents=not worker,
    )
    async with runtime:
        runtime.ctx.injected_context_tags = (
            *runtime.ctx.injected_context_tags,
            PROJECT_GUIDANCE_TAG,
            USER_RULES_TAG,
        )
        if restored_state is not None:
            restored_state.restore(runtime.ctx)
            if runtime.ctx.usage_snapshot_entries:
                runtime.ctx.usage_snapshot_entries.clear()

        run_id = uuid.uuid4().hex[:12]
        adapter = DisplayEventAdapter(session_id=resolved_session_id, run_id=run_id)
        sink = HeadlessEventSink()
        sink.replay.extend_snapshot(restored_display_messages)
        sink.emit(adapter.build_run_started_event(input_text=prompt))

        output_text: str | None = None
        try:
            user_prompt = _build_user_prompt(config_manager, working_dir, prompt)
            deferred_tool_results: DeferredToolResults | None = None
            while True:
                async with stream_agent(
                    runtime,  # type: ignore[arg-type]
                    user_prompt=user_prompt,
                    message_history=message_history,
                    deferred_tool_results=deferred_tool_results,
                    usage_limits=UsageLimits(request_limit=config.general.max_requests),
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
                        denial_reason = "Headless mode denies HITL requests by default."
                        sink.emit(
                            adapter.build_system_event(
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
                        user_prompt = None
                        continue

                    output_text = str(output) if output is not None else None
                    break

            sink.emit(adapter.build_run_finished_event(result={"output_text": output_text}))
        except (AgentInterrupted, KeyboardInterrupt):
            sink.emit(adapter.build_system_event("run_cancelled", {"reason": "interrupted"}))
            raise
        except Exception as exc:
            sink.emit(adapter.build_run_error_event(message=str(exc) or repr(exc), code=type(exc).__name__))
            raise

        if message_history:
            _save_session_artifacts(
                config=config,
                config_manager=config_manager,
                session_id=resolved_session_id,
                working_dir=working_dir,
                message_history=message_history,
                state=runtime.ctx.export_state(include_usage_ledger=False),
                display_messages=sink.replay.snapshot(),
                output_text=output_text,
            )

        return HeadlessRunResult(
            session_id=resolved_session_id,
            output_text=output_text,
            display_messages=sink.replay.snapshot(),
        )
