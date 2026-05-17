from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import AliasChoices, BaseModel, Field

from ya_claw.json_types import JsonObject, JsonValue
from ya_claw.orm.tables import AgencyFireRecord, RunRecord, SessionMemoryStateRecord, SessionRecord
from ya_claw.workspace.models import WorkspaceBindingSpec
from ya_claw.workspace.runtime_models import SessionWorkspaceState, build_session_workspace_state


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SessionStatus(StrEnum):
    IDLE = "idle"
    QUEUED = RunStatus.QUEUED
    RUNNING = RunStatus.RUNNING
    COMPLETED = RunStatus.COMPLETED
    FAILED = RunStatus.FAILED
    CANCELLED = RunStatus.CANCELLED


class SessionStatusReason(StrEnum):
    IDLE = "idle"
    RUN_QUEUED = "run_queued"
    RUN_RUNNING = "run_running"
    HITL_PENDING = "hitl_pending"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    RUN_CANCELLED = "run_cancelled"


class TriggerType(StrEnum):
    API = "api"
    BRIDGE = "bridge"
    SCHEDULE = "schedule"
    HEARTBEAT = "heartbeat"
    MEMORY = "memory"
    AGENCY = "agency"


class SessionType(StrEnum):
    CONVERSATION = "conversation"
    MEMORY = "memory"
    AGENCY = "agency"


class AgencyFireKind(StrEnum):
    MANUAL = "manual"
    TIMER = "timer"
    MEMORY_COMMITTED = "memory_committed"
    COMPACT = "compact"


class AgencyFireStatus(StrEnum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    STEERED = "steered"
    MERGED = "merged"
    CONSUMED = "consumed"
    SKIPPED = "skipped"
    FAILED = "failed"


class MemoryJobKind(StrEnum):
    EXTRACT = "extract"
    SUMMARY = "summary"


class TerminationReason(StrEnum):
    COMPLETED = "completed"
    ERROR = "error"
    CANCEL = "cancel"
    INTERRUPT = "interrupt"


class TextPart(BaseModel):
    type: Literal["text"]
    text: str
    metadata: dict[str, Any] | None = None


class UrlPart(BaseModel):
    type: Literal["url"]
    url: str
    kind: str
    filename: str | None = None
    storage: Literal["ephemeral", "persistent", "inline"] = "ephemeral"
    metadata: dict[str, Any] | None = None


class FilePart(BaseModel):
    type: Literal["file"]
    path: str
    kind: str
    metadata: dict[str, Any] | None = None


class BinaryPart(BaseModel):
    type: Literal["binary"]
    data: str
    mime_type: str
    kind: str
    filename: str | None = None
    storage: Literal["ephemeral", "persistent", "inline"] = "ephemeral"
    metadata: dict[str, Any] | None = None


class ModePart(BaseModel):
    type: Literal["mode"]
    mode: str
    params: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class CommandPart(BaseModel):
    type: Literal["command"]
    name: str
    params: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


InputPart = Annotated[
    ModePart | CommandPart | TextPart | UrlPart | FilePart | BinaryPart,
    Field(discriminator="type"),
]

ContentPart = Annotated[
    TextPart | UrlPart | FilePart | BinaryPart,
    Field(discriminator="type"),
]


class DispatchMode(StrEnum):
    QUEUE = "queue"
    ASYNC = "async"
    STREAM = "stream"


class UserInteraction(BaseModel):
    tool_call_id: str
    approved: bool
    reason: str | None = None
    user_input: JsonValue = None


class ActiveInteraction(BaseModel):
    interaction_id: str
    run_id: str
    session_id: str
    tool_call_id: str
    tool_name: str | None = None
    kind: str = "approval"
    title: str
    description: str | None = None
    arguments_preview: JsonValue = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: Literal["pending", "approved", "denied"] = "pending"
    sequence_no: int = 1
    total_count: int = 1
    created_at: datetime | None = None
    resolved_at: datetime | None = None


class InteractionRespondRequest(BaseModel):
    approved: bool
    reason: str | None = None
    user_input: JsonValue = None
    client_token: str | None = None


class InteractionRespondResponse(BaseModel):
    session_id: str
    run_id: str
    interaction_id: str
    tool_call_id: str
    status: Literal["pending", "approved", "denied"]
    remaining_interaction_count: int
    current_interaction: ActiveInteraction | None = None


class ToolResult(BaseModel):
    tool_call_id: str
    content: JsonValue
    error: str | None = None


class SessionCreateRequest(BaseModel):
    profile_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    workspace: WorkspaceBindingSpec | None = None
    input_parts: list[InputPart] = Field(default_factory=list)
    dispatch_mode: DispatchMode = DispatchMode.ASYNC
    trigger_type: TriggerType = TriggerType.API


class SessionRunCreateRequest(BaseModel):
    restore_from_run_id: str | None = None
    reset_state: bool = False
    input_parts: list[InputPart] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    workspace: WorkspaceBindingSpec | None = None
    dispatch_mode: DispatchMode = DispatchMode.ASYNC
    trigger_type: TriggerType = TriggerType.API


class SessionForkRequest(BaseModel):
    restore_from_run_id: str | None = None
    profile_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    workspace: WorkspaceBindingSpec | None = None


class RunCreateRequest(BaseModel):
    session_id: str | None = None
    restore_from_run_id: str | None = None
    reset_state: bool = False
    profile_name: str | None = None
    input_parts: list[InputPart] = Field(default_factory=list)
    trigger_type: TriggerType = TriggerType.API
    metadata: dict[str, Any] = Field(default_factory=dict)
    workspace: WorkspaceBindingSpec | None = None
    dispatch_mode: DispatchMode = DispatchMode.ASYNC


class SteerRequest(BaseModel):
    input_parts: list[InputPart] = Field(default_factory=list)


class RunSummary(BaseModel):
    id: str
    session_id: str
    sequence_no: int
    restore_from_run_id: str | None = None
    status: RunStatus
    trigger_type: TriggerType
    profile_name: str | None = None
    input_preview: str | None = None
    input_parts: list[InputPart] | None = None
    output_text: str | None = None
    output_summary: str | None = None
    error_message: str | None = None
    termination_reason: TerminationReason | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    committed_at: datetime | None = None
    message: list[dict[str, Any]] | None = None


class RunDetail(RunSummary):
    metadata: dict[str, Any] = Field(default_factory=dict)
    has_state: bool = False
    has_message: bool = False


class SessionTurn(BaseModel):
    run_id: str
    session_id: str
    sequence_no: int
    restore_from_run_id: str | None = None
    profile_name: str | None = None
    input_preview: str | None = None
    input_parts: list[InputPart] = Field(default_factory=list)
    output_text: str | None = None
    output_summary: str | None = None
    created_at: datetime
    committed_at: datetime | None = None


class SessionTurnsResponse(BaseModel):
    session_id: str
    limit: int
    has_more: bool = False
    next_cursor: str | None = None
    next_before_sequence_no: int | None = None
    turns: list[SessionTurn] = Field(default_factory=list)


class AgencyRiskPolicy(BaseModel):
    max_auto_action_risk: Literal["low", "medium", "high", "extra_high"] = "extra_high"


class AgencyFireSummary(BaseModel):
    id: str
    kind: AgencyFireKind | str
    status: AgencyFireStatus | str
    scheduled_at: datetime
    fired_at: datetime | None = None
    dedupe_key: str
    source_session_id: str | None = None
    source_run_id: str | None = None
    agency_session_id: str | None = None
    run_id: str | None = None
    active_run_id: str | None = None
    run_status: RunStatus | str | None = None
    priority: int = 100
    payload: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime
    consumed_at: datetime | None = None


class AgencyFireListResponse(BaseModel):
    fires: list[AgencyFireSummary] = Field(default_factory=list)


class AgencyConfigResponse(BaseModel):
    enabled: bool = True
    profile_name: str
    timer_interval_seconds: int
    agency_session_id: str
    singleton_scope_key: str
    singleton_source_session_id: str
    risk_policy: AgencyRiskPolicy = Field(default_factory=AgencyRiskPolicy)
    memory_files: dict[str, str] = Field(default_factory=dict)
    next_fire_at: datetime | None = None


class AgencyStatusResponse(BaseModel):
    enabled: bool = True
    agency_session_id: str
    state: Literal["idle", "queued", "running"] = "idle"
    active_run: RunSummary | None = None
    latest_run: RunSummary | None = None
    active_run_id: str | None = None
    latest_run_id: str | None = None
    next_fire_at: datetime | None = None
    pending_fire_count: int = 0
    last_fire: AgencyFireSummary | None = None
    agency_session: SessionSummary


class AgencyTriggerRequest(BaseModel):
    kind: AgencyFireKind = AgencyFireKind.MANUAL
    source_session_id: str | None = None
    source_run_id: str | None = None
    client_token: str | None = None
    prompt: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class AgencyTriggerResponse(BaseModel):
    accepted: bool = True
    agency_session_id: str
    fire: AgencyFireSummary
    run_id: str | None = None
    active_run_id: str | None = None
    delivery: Literal["steered", "submitted", "merged", "pending", "duplicate", "skipped"]


class RunTraceItem(BaseModel):
    sequence_no: int
    type: Literal["tool_call", "tool_response"]
    tool_call_id: str | None = None
    tool_name: str | None = None
    message_id: str | None = None
    role: str | None = None
    content: str | None = None
    truncated: bool = False


class RunTraceResponse(BaseModel):
    run_id: str
    session_id: str
    item_count: int
    max_item_chars: int
    max_total_chars: int
    truncated: bool = False
    trace: list[RunTraceItem] = Field(default_factory=list)


class SessionSummary(BaseModel):
    id: str
    parent_session_id: str | None = None
    profile_name: str | None = None
    session_type: SessionType = SessionType.CONVERSATION
    source_session_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    status: SessionStatus = SessionStatus.IDLE
    status_reason: SessionStatusReason = SessionStatusReason.IDLE
    status_detail: dict[str, Any] = Field(default_factory=dict)
    run_count: int = 0
    head_run_id: str | None = None
    head_success_run_id: str | None = None
    active_run_id: str | None = None
    latest_run: RunSummary | None = None
    memory_state: MemoryStateSummary | None = None
    workspace_state: SessionWorkspaceState | None = None


class SessionDetail(SessionSummary):
    runs: list[RunSummary] = Field(default_factory=list)
    runs_limit: int = 0
    runs_has_more: bool = False
    runs_next_before_sequence_no: int | None = None


class SessionCreateResponse(BaseModel):
    session: SessionSummary
    run: RunDetail | None = None


class MemoryStateSummary(BaseModel):
    source_session_id: str
    memory_session_id: str | None = None
    enabled: bool = True
    last_extracted_sequence_no: int = 0
    turns_since_extract: int = 0
    extract_count: int = 0
    extracts_since_summary: int = 0
    pending_extract: bool = False
    pending_summary: bool = False
    last_extract_run_id: str | None = None
    last_summary_run_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class MemoryActionRequest(BaseModel):
    reason: str = "manual"
    run_ids: list[str] = Field(default_factory=list)


class MemoryActionResponse(BaseModel):
    accepted: bool = True
    source_session_id: str
    run_id: str | None = None
    kind: MemoryJobKind
    reason: str | None = None


class SessionGetResponse(BaseModel):
    session: SessionDetail
    state: JsonObject | None = None
    message: list[dict[str, Any]] | None = None


class RunGetResponse(BaseModel):
    session: SessionSummary
    run: RunDetail
    state: JsonObject | None = None
    message: list[dict[str, Any]] | None = None


class ControlResponse(BaseModel):
    session_id: str
    run_id: str
    status: RunStatus
    accepted: bool = True


class ProfileSubagent(BaseModel):
    name: str
    description: str
    system_prompt: str
    model: str | None = None
    model_settings_preset: str | None = None
    model_settings_override: dict[str, Any] | None = None
    model_config_preset: str | None = None
    model_config_override: dict[str, Any] | None = None


class ProfileMCPServer(BaseModel):
    transport: Literal["streamable_http"] = "streamable_http"
    url: str
    headers: dict[str, str] = Field(default_factory=dict)
    description: str = ""
    required: bool = True


class ProfileUpsertRequest(BaseModel):
    model: str
    model_settings_preset: str | None = None
    model_settings_override: dict[str, Any] | None = None
    model_config_preset: str | None = None
    model_config_override: dict[str, Any] | None = None
    system_prompt: str | None = None
    builtin_toolsets: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("builtin_toolsets", "toolsets"),
    )
    subagents: list[ProfileSubagent] = Field(default_factory=list)
    include_builtin_subagents: bool = False
    unified_subagents: bool = False
    need_user_approve_tools: list[str] = Field(default_factory=list)
    need_user_approve_mcps: list[str] = Field(default_factory=list)
    enabled_mcps: list[str] = Field(default_factory=list)
    disabled_mcps: list[str] = Field(default_factory=list)
    mcp_servers: dict[str, ProfileMCPServer] = Field(default_factory=dict)
    workspace_backend_hint: str | None = None
    enabled: bool = True
    source_type: str | None = None
    source_version: str | None = None
    source_checksum: str | None = None


class ProfileSummary(BaseModel):
    name: str
    model: str
    workspace_backend_hint: str | None = None
    enabled: bool
    source_type: str | None = None
    source_version: str | None = None
    updated_at: datetime


class ProfileDetail(ProfileSummary):
    model_settings_preset: str | None = None
    model_settings_override: dict[str, Any] | None = None
    model_config_preset: str | None = None
    model_config_override: dict[str, Any] | None = None
    system_prompt: str | None = None
    builtin_toolsets: list[str] = Field(default_factory=list)
    toolsets: list[str] = Field(default_factory=list)
    subagents: list[ProfileSubagent] = Field(default_factory=list)
    include_builtin_subagents: bool = False
    unified_subagents: bool = False
    need_user_approve_tools: list[str] = Field(default_factory=list)
    need_user_approve_mcps: list[str] = Field(default_factory=list)
    enabled_mcps: list[str] = Field(default_factory=list)
    disabled_mcps: list[str] = Field(default_factory=list)
    mcp_servers: dict[str, ProfileMCPServer] = Field(default_factory=dict)
    source_checksum: str | None = None
    created_at: datetime


class ProfileSeedRequest(BaseModel):
    prune_missing: bool = False


class ProfileSeedResponse(BaseModel):
    seeded_names: list[str] = Field(default_factory=list)
    seed_file: str
    prune_missing: bool = False


def extract_input_preview(input_parts: list[InputPart]) -> str | None:
    for part in input_parts:
        if isinstance(part, TextPart):
            normalized_text = part.text.strip()
            if normalized_text:
                return normalized_text
    return None


def parse_input_parts(raw_input_parts: list[dict[str, Any]] | None) -> list[InputPart]:
    parsed_parts: list[InputPart] = []
    for raw_part in raw_input_parts or []:
        part_type = raw_part.get("type")
        if part_type == "text":
            parsed_parts.append(TextPart.model_validate(raw_part))
        elif part_type == "url":
            parsed_parts.append(UrlPart.model_validate(raw_part))
        elif part_type == "file":
            parsed_parts.append(FilePart.model_validate(raw_part))
        elif part_type == "binary":
            parsed_parts.append(BinaryPart.model_validate(raw_part))
        elif part_type == "mode":
            parsed_parts.append(ModePart.model_validate(raw_part))
        elif part_type == "command":
            parsed_parts.append(CommandPart.model_validate(raw_part))
        else:
            raise ValueError(f"Unsupported input part type: {part_type!r}")
    return parsed_parts


def public_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return dict(metadata)


def parse_message_events(raw_message_payload: JsonValue) -> list[JsonObject] | None:
    if raw_message_payload is None:
        return None
    if not isinstance(raw_message_payload, list):
        raise TypeError("message payload must be a top-level JSON array of AGUI event objects")
    parsed_events: list[JsonObject] = [event for event in raw_message_payload if isinstance(event, dict)]
    if len(parsed_events) != len(raw_message_payload):
        raise TypeError("message payload must contain only AGUI event objects")
    return parsed_events


def run_summary_from_record(
    record: RunRecord,
    *,
    message: list[dict[str, Any]] | None = None,
    include_input_parts: bool = False,
) -> RunSummary:
    input_parts = parse_input_parts(list(record.input_parts))
    termination_reason = TerminationReason(record.termination_reason) if record.termination_reason else None
    return RunSummary(
        id=record.id,
        session_id=record.session_id,
        sequence_no=record.sequence_no,
        restore_from_run_id=record.restore_from_run_id,
        status=RunStatus(record.status),
        trigger_type=TriggerType(record.trigger_type),
        profile_name=record.profile_name,
        input_preview=extract_input_preview(input_parts),
        input_parts=input_parts if include_input_parts else None,
        output_text=record.output_text,
        output_summary=record.output_summary,
        error_message=record.error_message,
        termination_reason=termination_reason,
        created_at=record.created_at,
        started_at=record.started_at,
        finished_at=record.finished_at,
        committed_at=record.committed_at,
        message=message,
    )


def run_detail_from_record(record: RunRecord, *, has_state: bool = False, has_message: bool = False) -> RunDetail:
    return RunDetail(
        **run_summary_from_record(record, include_input_parts=True).model_dump(),
        metadata=public_metadata(dict(record.run_metadata)),
        has_state=has_state,
        has_message=has_message,
    )


def session_turn_from_record(record: RunRecord) -> SessionTurn:
    input_parts = parse_input_parts(list(record.input_parts))
    return SessionTurn(
        run_id=record.id,
        session_id=record.session_id,
        sequence_no=record.sequence_no,
        restore_from_run_id=record.restore_from_run_id,
        profile_name=record.profile_name,
        input_preview=extract_input_preview(input_parts),
        input_parts=input_parts,
        output_text=record.output_text,
        output_summary=record.output_summary,
        created_at=record.created_at,
        committed_at=record.committed_at,
    )


def active_interactions_from_run_record(record: RunRecord) -> list[dict[str, Any]]:
    if not isinstance(record.run_metadata, dict):
        return []
    interactions = record.run_metadata.get("active_interactions")
    if not isinstance(interactions, list):
        return []
    return [interaction for interaction in interactions if isinstance(interaction, dict)]


def resolve_session_status(latest_run: RunSummary | None) -> SessionStatus:
    if latest_run is None:
        return SessionStatus.IDLE
    return SessionStatus(latest_run.status)


def resolve_session_status_reason(
    latest_run: RunSummary | None,
    *,
    active_interactions: list[dict[str, Any]] | None = None,
) -> SessionStatusReason:
    if latest_run is None:
        return SessionStatusReason.IDLE
    if latest_run.status == RunStatus.QUEUED:
        return SessionStatusReason.RUN_QUEUED
    if latest_run.status == RunStatus.RUNNING:
        if active_interactions:
            return SessionStatusReason.HITL_PENDING
        return SessionStatusReason.RUN_RUNNING
    if latest_run.status == RunStatus.COMPLETED:
        return SessionStatusReason.RUN_COMPLETED
    if latest_run.status == RunStatus.FAILED:
        return SessionStatusReason.RUN_FAILED
    if latest_run.status == RunStatus.CANCELLED:
        return SessionStatusReason.RUN_CANCELLED
    return SessionStatusReason.IDLE


def resolve_session_status_detail(
    latest_run: RunSummary | None,
    *,
    active_interactions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if latest_run is None:
        return {}

    detail: dict[str, Any] = {
        "run_id": latest_run.id,
        "sequence_no": latest_run.sequence_no,
        "trigger_type": latest_run.trigger_type,
    }
    if latest_run.termination_reason is not None:
        detail["termination_reason"] = latest_run.termination_reason
    if latest_run.error_message is not None:
        detail["error_message"] = latest_run.error_message
    if active_interactions:
        detail["active_interactions"] = active_interactions
        detail["active_interaction_count"] = len(active_interactions)
    return detail


def memory_state_summary_from_record(record: SessionMemoryStateRecord) -> MemoryStateSummary:
    return MemoryStateSummary(
        source_session_id=record.source_session_id,
        memory_session_id=record.memory_session_id,
        enabled=bool(record.enabled),
        last_extracted_sequence_no=record.last_extracted_sequence_no,
        turns_since_extract=record.turns_since_extract,
        extract_count=record.extract_count,
        extracts_since_summary=record.extracts_since_summary,
        pending_extract=bool(record.pending_extract),
        pending_summary=bool(record.pending_summary),
        last_extract_run_id=record.last_extract_run_id,
        last_summary_run_id=record.last_summary_run_id,
        metadata=public_memory_metadata(dict(record.memory_metadata or {})),
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def agency_fire_summary_from_record(
    record: AgencyFireRecord,
    *,
    run_status: str | None = None,
) -> AgencyFireSummary:
    return AgencyFireSummary(
        id=record.id,
        kind=record.kind,
        status=record.status,
        scheduled_at=record.scheduled_at,
        fired_at=record.fired_at,
        dedupe_key=record.dedupe_key,
        source_session_id=record.source_session_id,
        source_run_id=record.source_run_id,
        agency_session_id=record.agency_session_id,
        run_id=record.run_id,
        active_run_id=record.active_run_id,
        run_status=run_status,
        priority=record.priority,
        payload=public_metadata(dict(record.payload or {})),
        error_message=record.error_message,
        created_at=record.created_at,
        updated_at=record.updated_at,
        consumed_at=record.consumed_at,
    )


def public_memory_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    pending_value = metadata.get("pending_requests")
    if not isinstance(pending_value, list):
        return {}
    pending_requests = []
    for item in pending_value:
        if not isinstance(item, dict):
            continue
        source_run_ids = item.get("source_run_ids")
        pending_requests.append({
            "kind": item.get("kind") if isinstance(item.get("kind"), str) else None,
            "reason": item.get("reason") if isinstance(item.get("reason"), str) else None,
            "source_sequence_start": item.get("source_sequence_start")
            if isinstance(item.get("source_sequence_start"), int)
            else None,
            "source_sequence_end": item.get("source_sequence_end")
            if isinstance(item.get("source_sequence_end"), int)
            else None,
            "source_run_count": len(source_run_ids) if isinstance(source_run_ids, list) else 0,
            "has_context_handoff": isinstance(item.get("context_handoff"), dict),
        })
    return {"pending_requests": pending_requests} if pending_requests else {}


def session_summary_from_record(
    record: SessionRecord,
    *,
    run_count: int,
    latest_run: RunSummary | None,
    memory_state: MemoryStateSummary | None = None,
    active_interactions: list[dict[str, Any]] | None = None,
    workspace_state: SessionWorkspaceState | None = None,
) -> SessionSummary:
    return SessionSummary(
        id=record.id,
        parent_session_id=record.parent_session_id,
        profile_name=record.profile_name,
        session_type=SessionType(record.session_type),
        source_session_id=record.source_session_id,
        metadata=public_metadata(dict(record.session_metadata)),
        created_at=record.created_at,
        updated_at=record.updated_at,
        status=resolve_session_status(latest_run),
        status_reason=resolve_session_status_reason(latest_run, active_interactions=active_interactions),
        status_detail=resolve_session_status_detail(latest_run, active_interactions=active_interactions),
        run_count=run_count,
        head_run_id=record.head_run_id,
        head_success_run_id=record.head_success_run_id,
        active_run_id=record.active_run_id,
        latest_run=latest_run,
        memory_state=memory_state,
        workspace_state=workspace_state
        if workspace_state is not None
        else build_session_workspace_state(record.session_metadata),
    )
