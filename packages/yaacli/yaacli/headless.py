"""Durable NDJSON frontend for one YAACLI prompt."""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO, cast

from pydantic import JsonValue
from pydantic_ai import (
    AgentSpec,
    DeferredToolRequests,
    DeferredToolResults,
    ToolDenied,
)
from pydantic_ai.messages import RetryPromptPart
from ya_agent_sdk.context import (
    PROJECT_GUIDANCE_TAG,
    USER_RULES_TAG,
    StreamEvent,
)
from ya_agent_sdk.subagents import SubagentExecutionMode, SubagentExecutionRecord
from ya_agent_sdk.toolsets.skills import SHARED_SKILLS_DIR_NAME, SkillToolset
from ya_agent_stream_protocol.agui import AguiReplayConfig
from ya_agent_stream_protocol.sdk import AguiAdapterConfig, AguiEventAdapter

from yaacli.app.commands import format_skill_invocation, parse_skill_invocation
from yaacli.config import ConfigManager, YaacliConfig
from yaacli.display_replay import BoundedDisplayReplay
from yaacli.durable.application import SessionApplicationService
from yaacli.durable.executor import LocalExecutionWorker, LocalRuntimeSpec
from yaacli.durable.sqlite import SQLiteSessionStore
from yaacli.errors import safe_exception_str
from yaacli.logging import get_logger
from yaacli.model_profiles import (
    ResolvedModelProfile,
    get_model_profile,
    get_startup_model_profile,
)
from yaacli.runtime import (
    build_runtime_agent_spec,
    compile_child_plan_manifest,
    compile_runtime_sources,
    create_tui_runtime,
)

logger = get_logger(__name__)

YAACLI_AGUI_ADAPTER_CONFIG = AguiAdapterConfig(
    run_event_prefix="yaacli",
    stream_metadata_prefix="yaacli",
)
YAACLI_AGUI_REPLAY_CONFIG = AguiReplayConfig(
    agent_id_field="yaacliAgentId",
    main_agent_id="main",
    drop_subagent_detail_events=True,
)


@dataclass(slots=True)
class HeadlessRunResult:
    session_id: str
    output_text: str | None
    display_messages: list[dict[str, Any]]


class _DenySubagentDeferredResolver:
    async def resolve(
        self,
        record: SubagentExecutionRecord,
        requests: DeferredToolRequests,
    ) -> DeferredToolResults:
        del record
        results = DeferredToolResults()
        for request in requests.approvals:
            results.approvals[request.tool_call_id] = ToolDenied(
                message="Headless mode denies child HITL requests by policy."
            )
        for request in requests.calls:
            results.calls[request.tool_call_id] = RetryPromptPart(
                content="Headless mode denies child deferred calls by policy.",
                tool_name=request.tool_name,
                tool_call_id=request.tool_call_id,
            )
        return results


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


def _load_guidance_files(
    config_manager: ConfigManager,
    working_dir: Path,
) -> tuple[str | None, str | None]:
    project_guidance = _read_guidance(
        working_dir / "AGENTS.md",
        lambda path, content: f"<{PROJECT_GUIDANCE_TAG} name={path.name}>\n{content}\n</{PROJECT_GUIDANCE_TAG}>",
    )
    user_rules = _read_guidance(
        config_manager.config_dir / "RULES.md",
        lambda path, content: (
            f"<{USER_RULES_TAG} location={path.absolute().as_posix()}>\n{content}\n</{USER_RULES_TAG}>"
        ),
    )
    return project_guidance, user_rules


def _read_guidance(
    path: Path,
    render: Any,
) -> str | None:
    if not path.is_file():
        return None
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to read %s: %s", path, safe_exception_str(exc))
        return None
    return render(path, content) if content.strip() else None


def _build_user_prompt(
    config_manager: ConfigManager,
    working_dir: Path,
    prompt: str,
) -> list[JsonValue]:
    project_guidance, user_rules = _load_guidance_files(config_manager, working_dir)
    parts: list[JsonValue] = [prompt]
    if project_guidance:
        parts.append(project_guidance)
    if user_rules:
        parts.append(user_rules)
    return parts


def _resolve_model_profile(
    config: YaacliConfig,
    config_manager: ConfigManager,
    model_profile_id: str | None,
) -> ResolvedModelProfile | None:
    if model_profile_id is None:
        return get_startup_model_profile(config, config_manager.config_dir)
    profile = get_model_profile(config, model_profile_id)
    if profile is None:
        raise ValueError(f"Unknown model profile: {model_profile_id}")
    return profile


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
    """Run one durable prompt, reserving stdout exclusively for NDJSON."""
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
    effective_profile = _resolve_model_profile(
        config,
        config_manager,
        model_profile_id,
    )
    mcp_config = config_manager.load_mcp_config()
    capability_plugins = config_manager.load_capability_plugin_config()
    capability_catalog = capability_plugins.catalog
    sources = compile_runtime_sources(
        config,
        config_dir=config_manager.config_dir,
        include_subagents=not worker,
    )
    child_deferred_resolver = _DenySubagentDeferredResolver()
    sink = HeadlessEventSink(ndjson_stream)
    database_path = config_manager.get_session_database_path()
    subagent_mode = None if worker else SubagentExecutionMode.foreground
    child_manifest = (
        compile_child_plan_manifest(
            config,
            profile=effective_profile,
            sources=sources,
            capability_catalog=capability_catalog,
        )
        if subagent_mode is not None
        else None
    )
    runtime_id = effective_profile.id if effective_profile is not None else "default"
    agent_spec = AgentSpec.model_validate(
        build_runtime_agent_spec(
            config,
            profile=effective_profile,
            capability_plugins=capability_plugins,
        )
    )
    skill_toolset = SkillToolset(
        toolset_id="skills",
        extra_dir_names=[SHARED_SKILLS_DIR_NAME],
    )
    execution_worker: LocalExecutionWorker | None = None
    resolved_session_id = session_id
    run_id = uuid.uuid4().hex[:12]
    adapter: AguiEventAdapter | None = None
    base_display_projection: list[JsonValue] = []

    async def event_sink(event: StreamEvent) -> None:
        if adapter is None:
            raise RuntimeError("Headless event adapter is not initialized")
        sink.emit_many(adapter.adapt_stream_event(event))

    def build_runtime(binding_ref: str):
        return create_tui_runtime(
            config=config,
            mcp_config=mcp_config,
            working_dir=working_dir,
            system_prompt=sources.system_prompt,
            child_plan_manifest=child_manifest,
            config_dir=config_manager.config_dir,
            model_profile=effective_profile,
            subagent_default_mode=subagent_mode,
            enable_user_input=False,
            skill_toolset=skill_toolset,
            durable_binding_ref=binding_ref,
            durable_database_path=database_path,
            subagent_deferred_resolver=child_deferred_resolver,
            agent_spec=agent_spec,
            capability_catalog=capability_catalog,
            agent_name="yaacli_main_v2",
        )

    store = SQLiteSessionStore(
        database_path,
        max_turns_per_session=config.session.max_turns_per_session,
        max_sessions=config.session.max_sessions,
        max_session_age_days=config.session.max_session_age_days,
    )
    try:
        execution_worker = await LocalExecutionWorker.create(
            store=store,
            state_path=database_path,
            active_runtime_id=runtime_id,
            runtime_specs=[
                LocalRuntimeSpec(
                    runtime_id=runtime_id,
                    build=build_runtime,
                    request_limit=config.general.max_requests,
                    hitl_policy="deny",
                )
            ],
            event_sink=event_sink,
            display_projection_provider=lambda: [
                *base_display_projection,
                *cast(list[JsonValue], sink.replay.snapshot()),
            ],
        )
        service = SessionApplicationService(store, execution_worker.coordinator)
        if resolved_session_id is None:
            session = service.create_session(str(working_dir.resolve()))
            resolved_session_id = session.session_id
        else:
            session = service.get_session(resolved_session_id)
            if session.head_revision_id is not None:
                previous = store.get_revision(session.head_revision_id)
                if previous is not None:
                    base_display_projection.extend(previous.display_projection)

        adapter = AguiEventAdapter(
            session_id=resolved_session_id,
            run_id=run_id,
            config=YAACLI_AGUI_ADAPTER_CONFIG,
        )
        sink.emit(adapter.build_run_started_event())

        await skill_toolset.refresh_context(execution_worker.runtime.ctx)
        invocation = parse_skill_invocation(
            prompt,
            execution_worker.runtime.ctx.available_skills,
        )
        effective_prompt = (
            format_skill_invocation(
                invocation,
                execution_worker.runtime.ctx.available_skills,
            )
            if invocation is not None
            else prompt
        )
        content = _build_user_prompt(
            config_manager,
            working_dir,
            effective_prompt,
        )
        revision = await service.run_turn(
            session.session_id,
            content,
            model=effective_profile.model if effective_profile is not None else config.general.model,
            model_profile_id=effective_profile.id if effective_profile is not None else None,
        )
        output = revision.terminal.get("output")
        output_text = str(output) if output is not None else None
        finished_event = adapter.build_run_finished_event(result={"output_text": output_text})
        sink.emit(finished_event)
        return HeadlessRunResult(
            session_id=session.session_id,
            output_text=output_text,
            display_messages=sink.replay.snapshot(),
        )
    except asyncio.CancelledError:
        if adapter is not None:
            sink.emit(
                adapter.build_run_custom_event(
                    "run_cancelled",
                    {"reason": "cancelled"},
                )
            )
        raise
    except KeyboardInterrupt:
        if adapter is not None:
            sink.emit(
                adapter.build_run_custom_event(
                    "run_cancelled",
                    {"reason": "interrupted"},
                )
            )
        raise
    except Exception as exc:
        if adapter is not None:
            sink.emit(
                adapter.build_run_error_event(
                    message=safe_exception_str(exc),
                    code=type(exc).__name__,
                )
            )
        raise
    finally:
        if execution_worker is not None:
            await execution_worker.close()
        store.close()
