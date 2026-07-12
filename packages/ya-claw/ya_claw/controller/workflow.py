from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from contextlib import suppress
from datetime import datetime
from typing import Any, Literal, cast
from uuid import uuid4

from fastapi import HTTPException
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import Select, exists, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ya_claw.config import ClawSettings
from ya_claw.controller.models import (
    DispatchMode,
    RunCreateRequest,
    SteerRequest,
    TextPart,
    TriggerType,
    parse_input_parts,
)
from ya_claw.controller.run import RunController
from ya_claw.controller.session_lifecycle import lock_session_reference
from ya_claw.orm.tables import (
    RunRecord,
    SessionRecord,
    WorkflowDefinitionRecord,
    WorkflowEventRecord,
    WorkflowNodeRunRecord,
    WorkflowRunRecord,
    utc_now,
)
from ya_claw.runtime_state import InMemoryRuntimeState
from ya_claw.workspace.models import WorkspaceBindingSpec

WorkflowDefinitionStatus = Literal["draft", "active", "archived"]
WorkflowScope = Literal["global", "session"]
WorkflowRunStatus = Literal["queued", "running", "waiting", "completed", "failed", "cancelled"]
WorkflowTriggerKind = Literal["web", "api", "agent", "schedule", "bridge", "system"]
WorkflowNodeRunStatus = Literal[
    "pending", "ready", "queued", "running", "waiting", "completed", "failed", "cancelled", "skipped"
]
WorkflowNodeMode = Literal["isolate", "continue", "fork", "steer"]

_TERMINAL_WORKFLOW_STATUSES = frozenset({"completed", "failed", "cancelled"})
_TERMINAL_NODE_STATUSES = frozenset({"completed", "failed", "cancelled", "skipped"})
_TERMINAL_RUN_STATUSES = frozenset({"completed", "failed", "cancelled"})
_ALLOWED_NODE_MODES = frozenset({"isolate", "continue", "fork", "steer"})
_TEMPLATE_EXPR_RE = re.compile(r"{{\s*(.*?)\s*}}")
_DEFAULT_SCHEMA_VERSION = "ya-claw.workflow.v1"


class WorkflowActorContext(BaseModel):
    actor_kind: Literal["api", "user", "agent", "schedule", "bridge", "system"] = "api"
    current_session_id: str | None = None
    current_run_id: str | None = None
    profile_name: str | None = None


class WorkflowDefinitionCreateRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    status: WorkflowDefinitionStatus = "active"
    scope: WorkflowScope = "global"
    tags: list[str] = Field(default_factory=list)
    when_to_use: str | None = None
    argument_hint: str | None = None
    input_schema: dict[str, Any] = Field(default_factory=dict)
    definition: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)
    owner_kind: Literal["user", "agent", "api", "system"] = "api"
    owner_session_id: str | None = None
    owner_run_id: str | None = None

    @model_validator(mode="after")
    def validate_definition_payload(self) -> WorkflowDefinitionCreateRequest:
        normalized = normalize_workflow_definition(self.definition)
        self.definition = normalized
        self.name = _clean_optional(self.name) or _string_or_none(normalized.get("name"))
        if self.name is None:
            raise ValueError("name is required")
        self.description = _clean_optional(self.description) or _string_or_none(normalized.get("description"))
        self.when_to_use = _clean_optional(self.when_to_use) or _string_or_none(normalized.get("when_to_use"))
        self.argument_hint = _clean_optional(self.argument_hint) or _string_or_none(normalized.get("argument_hint"))
        self.tags = _normalize_tags(self.tags or normalized.get("tags"))
        if not self.input_schema:
            inputs = normalized.get("inputs")
            self.input_schema = dict(inputs) if isinstance(inputs, dict) else {}
        return self


class WorkflowDefinitionUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    status: WorkflowDefinitionStatus | None = None
    scope: WorkflowScope | None = None
    tags: list[str] | None = None
    when_to_use: str | None = None
    argument_hint: str | None = None
    input_schema: dict[str, Any] | None = None
    definition: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class WorkflowTriggerRequest(BaseModel):
    inputs: dict[str, Any] = Field(default_factory=dict)
    profile_name: str | None = None
    workspace: WorkspaceBindingSpec | None = None
    supervisor_session_id: str | None = None
    supervisor_run_id: str | None = None
    trigger_kind: WorkflowTriggerKind = "api"
    metadata: dict[str, Any] = Field(default_factory=dict)
    inherit_shell_env: bool = True
    shell_env: dict[str, str] = Field(default_factory=dict)


class WorkflowCancelRequest(BaseModel):
    reason: str | None = None


class WorkflowNodeSteerRequest(BaseModel):
    input_parts: list[dict[str, Any]] = Field(default_factory=list)
    prompt: str | None = None

    @model_validator(mode="after")
    def validate_steer_payload(self) -> WorkflowNodeSteerRequest:
        if self.prompt is not None and self.prompt.strip() != "":
            return self
        if self.input_parts:
            return self
        raise ValueError("prompt or input_parts is required")


class WorkflowDefinitionSummary(BaseModel):
    id: str
    name: str
    description: str | None = None
    status: WorkflowDefinitionStatus
    definition_version: int
    schema_version: str
    owner_kind: str
    owner_session_id: str | None = None
    owner_run_id: str | None = None
    scope: WorkflowScope
    tags: list[str] = Field(default_factory=list)
    when_to_use: str | None = None
    argument_hint: str | None = None
    latest_run: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None = None


class WorkflowDefinitionDetail(WorkflowDefinitionSummary):
    input_schema: dict[str, Any] = Field(default_factory=dict)
    definition: dict[str, Any] = Field(default_factory=dict)


class WorkflowDefinitionListResponse(BaseModel):
    workflows: list[WorkflowDefinitionSummary] = Field(default_factory=list)


class WorkflowNodeRunSummary(BaseModel):
    id: str
    workflow_run_id: str
    node_id: str
    attempt_no: int
    status: WorkflowNodeRunStatus
    profile_name: str | None = None
    session_id: str | None = None
    run_id: str | None = None
    input_preview: str | None = None
    output_text: str | None = None
    output_json: dict[str, Any] | None = None
    error_message: str | None = None
    needs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    updated_at: datetime


class WorkflowEventSummary(BaseModel):
    id: str
    workflow_run_id: str
    node_run_id: str | None = None
    source_kind: str
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class WorkflowRunSummary(BaseModel):
    id: str
    workflow_id: str
    workflow_version: int
    workflow_name: str | None = None
    status: WorkflowRunStatus
    trigger_kind: WorkflowTriggerKind
    supervisor_session_id: str | None = None
    supervisor_run_id: str | None = None
    profile_name: str | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    error_message: str | None = None
    current_node_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    updated_at: datetime


class WorkflowRunDetail(WorkflowRunSummary):
    definition: dict[str, Any] = Field(default_factory=dict)
    nodes: list[WorkflowNodeRunSummary] = Field(default_factory=list)
    events: list[WorkflowEventSummary] = Field(default_factory=list)


class WorkflowRunListResponse(BaseModel):
    workflow_runs: list[WorkflowRunSummary] = Field(default_factory=list)


class WorkflowEventListResponse(BaseModel):
    workflow_run_id: str
    events: list[WorkflowEventSummary] = Field(default_factory=list)


class AgentPresetCard(BaseModel):
    name: str
    model: str | None = None
    enabled: bool = True
    builtin_toolsets: list[str] = Field(default_factory=list)


class WorkflowController:
    def __init__(self) -> None:
        self._run_controller = RunController()

    async def list_definitions(
        self,
        db_session: AsyncSession,
        *,
        query: str | None = None,
        tags: list[str] | None = None,
        status: str | None = None,
        scope: str | None = None,
        owner_kind: str | None = None,
        owner_session_id: str | None = None,
        supervisor_session_id: str | None = None,
        trigger_kind: str | None = None,
        created_by_current_session: bool = False,
        supervised_by_current_session: bool = False,
        touched_by_current_session: bool = False,
        only_current_session: bool = False,
        include_archived: bool = False,
        current_session_id: str | None = None,
        limit: int = 100,
        include_latest_run: bool = True,
    ) -> WorkflowDefinitionListResponse:
        normalized_limit = min(max(limit, 1), 500)
        statement: Select[tuple[WorkflowDefinitionRecord]] = select(WorkflowDefinitionRecord)
        if not include_archived:
            statement = statement.where(WorkflowDefinitionRecord.status != "archived")
        if _non_empty(status):
            statement = statement.where(WorkflowDefinitionRecord.status == status)
        if _non_empty(scope):
            statement = statement.where(WorkflowDefinitionRecord.scope == scope)
        if _non_empty(owner_kind):
            statement = statement.where(WorkflowDefinitionRecord.owner_kind == owner_kind)
        if _non_empty(owner_session_id):
            statement = statement.where(WorkflowDefinitionRecord.owner_session_id == owner_session_id)
        if _non_empty(supervisor_session_id):
            statement = statement.where(
                exists().where(
                    WorkflowRunRecord.workflow_id == WorkflowDefinitionRecord.id,
                    WorkflowRunRecord.supervisor_session_id == supervisor_session_id,
                )
            )
        if _non_empty(trigger_kind):
            statement = statement.where(
                exists().where(
                    WorkflowRunRecord.workflow_id == WorkflowDefinitionRecord.id,
                    WorkflowRunRecord.trigger_kind == trigger_kind,
                )
            )
        if _non_empty(query):
            normalized_query = query.strip() if isinstance(query, str) else ""
            pattern = f"%{normalized_query}%"
            statement = statement.where(
                or_(
                    WorkflowDefinitionRecord.name.ilike(pattern),
                    WorkflowDefinitionRecord.description.ilike(pattern),
                    WorkflowDefinitionRecord.when_to_use.ilike(pattern),
                    WorkflowDefinitionRecord.argument_hint.ilike(pattern),
                )
            )
        if tags:
            normalized_tags = _normalize_tags(tags)
            statement = self._apply_definition_current_session_filters(
                statement,
                current_session_id=current_session_id,
                created_by_current_session=created_by_current_session,
                supervised_by_current_session=supervised_by_current_session,
                touched_by_current_session=touched_by_current_session,
                only_current_session=only_current_session,
            )
            result = await db_session.execute(statement.order_by(WorkflowDefinitionRecord.updated_at.desc()))
            records = [
                record for record in result.scalars().all() if set(normalized_tags).issubset(set(record.tags or []))
            ]
            records = self._filter_definition_records_for_current_session(
                records,
                current_session_id=current_session_id,
                created_by_current_session=created_by_current_session,
                supervised_by_current_session=supervised_by_current_session,
                touched_by_current_session=touched_by_current_session,
                only_current_session=only_current_session,
            )[:normalized_limit]
            return WorkflowDefinitionListResponse(
                workflows=[
                    await self._definition_summary_from_record(
                        db_session, record, include_latest_run=include_latest_run
                    )
                    for record in records
                ]
            )
        statement = self._apply_definition_current_session_filters(
            statement,
            current_session_id=current_session_id,
            created_by_current_session=created_by_current_session,
            supervised_by_current_session=supervised_by_current_session,
            touched_by_current_session=touched_by_current_session,
            only_current_session=only_current_session,
        )
        statement = statement.order_by(WorkflowDefinitionRecord.updated_at.desc()).limit(normalized_limit)
        result = await db_session.execute(statement)
        records = list(result.scalars().all())
        return WorkflowDefinitionListResponse(
            workflows=[
                await self._definition_summary_from_record(db_session, record, include_latest_run=include_latest_run)
                for record in records
            ]
        )

    async def get_definition(self, db_session: AsyncSession, workflow_id: str) -> WorkflowDefinitionDetail:
        record = await self._get_definition_record(db_session, workflow_id, include_archived=True)
        return await self._definition_detail_from_record(db_session, record)

    async def create_definition(
        self,
        db_session: AsyncSession,
        request: WorkflowDefinitionCreateRequest,
        *,
        actor: WorkflowActorContext | None = None,
    ) -> WorkflowDefinitionDetail:
        effective_actor = actor or WorkflowActorContext()
        definition = normalize_workflow_definition(request.definition)
        now = utc_now()
        owner_kind = "agent" if effective_actor.actor_kind == "agent" else request.owner_kind
        owner_session_id = (
            effective_actor.current_session_id if effective_actor.actor_kind == "agent" else request.owner_session_id
        )
        owner_run_id = effective_actor.current_run_id if effective_actor.actor_kind == "agent" else request.owner_run_id
        scope = "session" if effective_actor.actor_kind == "agent" and request.scope == "global" else request.scope
        record = WorkflowDefinitionRecord(
            id=uuid4().hex,
            name=(request.name or str(definition.get("name"))).strip(),
            description=request.description,
            status=request.status,
            definition_version=int(definition.get("version") or 1),
            schema_version=str(definition.get("schema") or _DEFAULT_SCHEMA_VERSION),
            owner_kind=owner_kind,
            owner_session_id=owner_session_id,
            owner_run_id=owner_run_id,
            scope=scope,
            tags=_normalize_tags(request.tags or definition.get("tags")),
            when_to_use=request.when_to_use,
            argument_hint=request.argument_hint,
            input_schema=dict(request.input_schema),
            definition=definition,
            workflow_metadata=dict(request.metadata),
            created_at=now,
            updated_at=now,
            archived_at=now if request.status == "archived" else None,
        )
        db_session.add(record)
        await db_session.commit()
        await db_session.refresh(record)
        return await self._definition_detail_from_record(db_session, record)

    async def update_definition(  # noqa: C901
        self,
        db_session: AsyncSession,
        workflow_id: str,
        request: WorkflowDefinitionUpdateRequest,
    ) -> WorkflowDefinitionDetail:
        record = await self._get_definition_record(db_session, workflow_id, include_archived=True)
        definition_changed = False
        if request.definition is not None:
            definition = normalize_workflow_definition(request.definition)
            record.definition = definition
            record.schema_version = str(definition.get("schema") or record.schema_version or _DEFAULT_SCHEMA_VERSION)
            record.definition_version = int(definition.get("version") or record.definition_version + 1)
            if request.name is None and isinstance(definition.get("name"), str):
                record.name = str(definition["name"]).strip()
            if request.description is None:
                record.description = _string_or_none(definition.get("description"))
            if request.input_schema is None:
                inputs = definition.get("inputs")
                record.input_schema = dict(inputs) if isinstance(inputs, dict) else {}
            definition_changed = True
        if request.name is not None:
            normalized_name = request.name.strip()
            if normalized_name == "":
                raise HTTPException(status_code=422, detail="name is required.")
            record.name = normalized_name
        if request.description is not None:
            record.description = _clean_optional(request.description)
        if request.status is not None:
            record.status = request.status
            record.archived_at = utc_now() if request.status == "archived" else None
        if request.scope is not None:
            record.scope = request.scope
        if request.tags is not None:
            record.tags = _normalize_tags(request.tags)
        if request.when_to_use is not None:
            record.when_to_use = _clean_optional(request.when_to_use)
        if request.argument_hint is not None:
            record.argument_hint = _clean_optional(request.argument_hint)
        if request.input_schema is not None:
            record.input_schema = dict(request.input_schema)
        if request.metadata is not None:
            record.workflow_metadata = dict(request.metadata)
        if definition_changed and request.tags is None:
            record.tags = _normalize_tags(record.definition.get("tags") or record.tags)
        record.updated_at = utc_now()
        await db_session.commit()
        await db_session.refresh(record)
        return await self._definition_detail_from_record(db_session, record)

    async def archive_definition(self, db_session: AsyncSession, workflow_id: str) -> WorkflowDefinitionDetail:
        record = await self._get_definition_record(db_session, workflow_id, include_archived=True)
        record.status = "archived"
        record.archived_at = utc_now()
        record.updated_at = utc_now()
        await db_session.commit()
        await db_session.refresh(record)
        return await self._definition_detail_from_record(db_session, record)

    async def trigger(
        self,
        db_session: AsyncSession,
        workflow_id: str,
        request: WorkflowTriggerRequest,
        *,
        actor: WorkflowActorContext | None = None,
    ) -> WorkflowRunDetail:
        record = await self._get_definition_record(db_session, workflow_id)
        if record.status != "active":
            raise HTTPException(status_code=409, detail=f"Workflow '{workflow_id}' is not active.")
        validate_workflow_inputs(record.input_schema, request.inputs)
        effective_actor = actor or WorkflowActorContext()
        trigger_kind = "agent" if effective_actor.actor_kind == "agent" else request.trigger_kind
        supervisor_session_id = request.supervisor_session_id
        supervisor_run_id = request.supervisor_run_id
        profile_name = _clean_optional(request.profile_name)
        if effective_actor.actor_kind == "agent":
            supervisor_session_id = effective_actor.current_session_id
            supervisor_run_id = effective_actor.current_run_id
            profile_name = profile_name or effective_actor.profile_name
        elif profile_name is None:
            profile_name = await self._resolve_supervisor_profile(db_session, supervisor_session_id, supervisor_run_id)
        now = utc_now()
        workflow_metadata = {"workflow_name": record.name, **dict(request.metadata)}
        workflow_metadata.pop("shell_env", None)
        if request.inherit_shell_env:
            shell_env = _normalize_shell_env(request.shell_env)
            if shell_env:
                workflow_metadata["shell_env"] = shell_env
        run_record = WorkflowRunRecord(
            id=uuid4().hex,
            workflow_id=record.id,
            workflow_version=record.definition_version,
            definition_snapshot=dict(record.definition),
            status="queued",
            trigger_kind=trigger_kind,
            supervisor_session_id=supervisor_session_id,
            supervisor_run_id=supervisor_run_id,
            profile_name=profile_name,
            workspace=request.workspace.model_dump(mode="json") if request.workspace is not None else None,
            inputs=dict(request.inputs),
            workflow_metadata=workflow_metadata,
            created_at=now,
            updated_at=now,
        )
        db_session.add(run_record)
        await db_session.flush()
        await self._emit_event(
            db_session,
            run_record,
            event_type="workflow_queued",
            source_kind="workflow",
            message=f"Workflow '{record.name}' queued.",
            payload={"workflow_id": record.id, "workflow_version": record.definition_version},
        )
        await db_session.commit()
        await db_session.refresh(run_record)
        return await self._run_detail_from_record(db_session, run_record, include_events=True)

    async def list_runs(
        self,
        db_session: AsyncSession,
        *,
        workflow_id: str | None = None,
        status: str | None = None,
        trigger_kind: str | None = None,
        supervisor_session_id: str | None = None,
        only_current_session: bool = False,
        only_supervised_by_current_session: bool = False,
        only_touched_by_current_session: bool = False,
        include_completed: bool = True,
        current_session_id: str | None = None,
        limit: int = 100,
    ) -> WorkflowRunListResponse:
        normalized_limit = min(max(limit, 1), 500)
        statement: Select[tuple[WorkflowRunRecord]] = select(WorkflowRunRecord)
        if _non_empty(workflow_id):
            statement = statement.where(WorkflowRunRecord.workflow_id == workflow_id)
        if _non_empty(status):
            statement = statement.where(WorkflowRunRecord.status == status)
        elif not include_completed:
            statement = statement.where(WorkflowRunRecord.status.not_in(_TERMINAL_WORKFLOW_STATUSES))
        if _non_empty(trigger_kind):
            statement = statement.where(WorkflowRunRecord.trigger_kind == trigger_kind)
        if _non_empty(supervisor_session_id):
            statement = statement.where(WorkflowRunRecord.supervisor_session_id == supervisor_session_id)
        statement = self._apply_run_current_session_filters(
            statement,
            current_session_id=current_session_id,
            only_current_session=only_current_session,
            only_supervised_by_current_session=only_supervised_by_current_session,
            only_touched_by_current_session=only_touched_by_current_session,
        )
        statement = statement.order_by(WorkflowRunRecord.created_at.desc()).limit(normalized_limit)
        result = await db_session.execute(statement)
        return WorkflowRunListResponse(
            workflow_runs=[await self._run_summary_from_record(db_session, record) for record in result.scalars().all()]
        )

    async def get_run(self, db_session: AsyncSession, workflow_run_id: str) -> WorkflowRunDetail:
        record = await self._get_run_record(db_session, workflow_run_id)
        return await self._run_detail_from_record(db_session, record, include_events=True)

    async def list_events(
        self,
        db_session: AsyncSession,
        workflow_run_id: str,
        *,
        after_event_id: str | None = None,
        limit: int = 200,
    ) -> WorkflowEventListResponse:
        await self._get_run_record(db_session, workflow_run_id)
        normalized_limit = min(max(limit, 1), 1000)
        statement = (
            select(WorkflowEventRecord)
            .where(WorkflowEventRecord.workflow_run_id == workflow_run_id)
            .order_by(WorkflowEventRecord.created_at.asc(), WorkflowEventRecord.id.asc())
            .limit(normalized_limit)
        )
        if _non_empty(after_event_id):
            cursor = await db_session.get(WorkflowEventRecord, after_event_id)
            if isinstance(cursor, WorkflowEventRecord):
                statement = statement.where(WorkflowEventRecord.created_at > cursor.created_at)
        result = await db_session.execute(statement)
        return WorkflowEventListResponse(
            workflow_run_id=workflow_run_id,
            events=[event_summary_from_record(record) for record in result.scalars().all()],
        )

    async def cancel_run(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        runtime_state: InMemoryRuntimeState,
        workflow_run_id: str,
        request: WorkflowCancelRequest | None = None,
    ) -> WorkflowRunDetail:
        record = await self._get_run_record(db_session, workflow_run_id)
        if record.status in _TERMINAL_WORKFLOW_STATUSES:
            return await self._run_detail_from_record(db_session, record, include_events=True)
        statement = select(WorkflowNodeRunRecord).where(
            WorkflowNodeRunRecord.workflow_run_id == record.id,
            WorkflowNodeRunRecord.status.in_(("queued", "running", "waiting")),
        )
        result = await db_session.execute(statement)
        for node in result.scalars().all():
            if isinstance(node.run_id, str):
                with suppress(HTTPException):
                    await self._run_controller.cancel(db_session, settings, runtime_state, node.run_id)
            node.status = "cancelled"
            node.finished_at = utc_now()
            node.error_message = request.reason if request is not None else None
        record.status = "cancelled"
        record.error_message = request.reason if request is not None else None
        record.finished_at = utc_now()
        record.current_node_ids = []
        record.updated_at = utc_now()
        await self._emit_event(
            db_session,
            record,
            event_type="workflow_cancelled",
            source_kind="workflow",
            message="Workflow cancelled.",
            payload={"reason": request.reason if request is not None else None},
        )
        await db_session.commit()
        await db_session.refresh(record)
        return await self._run_detail_from_record(db_session, record, include_events=True)

    async def steer_node(
        self,
        db_session: AsyncSession,
        runtime_state: InMemoryRuntimeState,
        workflow_run_id: str,
        node_id: str,
        request: WorkflowNodeSteerRequest,
    ) -> WorkflowRunDetail:
        record = await self._get_run_record(db_session, workflow_run_id)
        statement = select(WorkflowNodeRunRecord).where(
            WorkflowNodeRunRecord.workflow_run_id == workflow_run_id,
            WorkflowNodeRunRecord.node_id == node_id,
        )
        result = await db_session.execute(statement)
        candidates = [node for node in result.scalars().all() if node.status in {"queued", "running", "waiting"}]
        if not candidates:
            raise HTTPException(status_code=409, detail=f"Workflow node '{node_id}' is not active.")
        node = sorted(candidates, key=lambda item: item.attempt_no, reverse=True)[0]
        if not isinstance(node.run_id, str):
            raise HTTPException(status_code=409, detail=f"Workflow node '{node_id}' has no active run.")
        input_parts = (
            [TextPart(type="text", text=request.prompt).model_dump(mode="json")]
            if isinstance(request.prompt, str) and request.prompt.strip() != ""
            else list(request.input_parts)
        )
        await self._run_controller.steer(
            db_session,
            runtime_state,
            node.run_id,
            SteerRequest(input_parts=parse_input_parts(input_parts)),
        )
        await self._emit_event(
            db_session,
            record,
            event_type="node_steered",
            source_kind="steer",
            node_run=node,
            message=f"Node '{node_id}' steered.",
            payload={"node_id": node_id, "run_id": node.run_id},
        )
        await db_session.commit()
        await db_session.refresh(record)
        return await self._run_detail_from_record(db_session, record, include_events=True)

    async def ensure_run_started(self, db_session: AsyncSession, record: WorkflowRunRecord) -> None:
        if record.status == "waiting":
            record.status = "running"
        if record.status != "queued":
            return
        definition = normalize_workflow_definition(record.definition_snapshot)
        validate_workflow_inputs(_input_schema_from_definition(definition), record.inputs)
        node_specs = workflow_node_specs(definition)
        existing_statement = select(WorkflowNodeRunRecord).where(WorkflowNodeRunRecord.workflow_run_id == record.id)
        existing = await db_session.execute(existing_statement)
        if not existing.scalars().first():
            for node_id, spec in node_specs.items():
                node = WorkflowNodeRunRecord(
                    id=uuid4().hex,
                    workflow_run_id=record.id,
                    node_id=node_id,
                    attempt_no=1,
                    status="pending",
                    profile_name=_node_profile(spec, record.profile_name),
                    needs=list(spec.get("needs") or []),
                    node_metadata=dict(spec.get("metadata") or {}),
                )
                db_session.add(node)
        record.status = "running"
        record.started_at = record.started_at or utc_now()
        record.updated_at = utc_now()
        await self._emit_event(
            db_session,
            record,
            event_type="workflow_started",
            source_kind="workflow",
            message="Workflow started.",
            payload={"workflow_id": record.workflow_id},
        )
        await db_session.flush()

    async def execute_once(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        runtime_state: InMemoryRuntimeState,
        run_dispatcher: Any,
        record: WorkflowRunRecord,
    ) -> bool:
        changed = False
        await self.ensure_run_started(db_session, record)
        changed = True
        nodes = await self._list_nodes(db_session, record.id)
        definition = normalize_workflow_definition(record.definition_snapshot)
        node_specs = workflow_node_specs(definition)
        changed = await self._refresh_node_statuses(db_session, record, nodes) or changed
        nodes = await self._list_nodes(db_session, record.id)
        if record.status in _TERMINAL_WORKFLOW_STATUSES:
            await db_session.flush()
            return changed
        nodes = await self._list_nodes(db_session, record.id)
        failed_nodes = [node for node in nodes if node.status == "failed"]
        cancelled_nodes = [node for node in nodes if node.status == "cancelled"]
        if failed_nodes:
            await self._fail_workflow(
                db_session, record, failed_nodes[0].error_message or f"Node '{failed_nodes[0].node_id}' failed."
            )
            await db_session.flush()
            return True
        if cancelled_nodes:
            record.status = "cancelled"
            record.finished_at = utc_now()
            record.current_node_ids = []
            record.updated_at = utc_now()
            await self._emit_event(
                db_session,
                record,
                event_type="workflow_cancelled",
                source_kind="workflow",
                message="Workflow cancelled after node cancellation.",
                payload={"node_id": cancelled_nodes[0].node_id},
            )
            await db_session.flush()
            return True
        ready_changed = await self._mark_ready_nodes(db_session, record, nodes)
        changed = ready_changed or changed
        await db_session.flush()
        nodes = await self._list_nodes(db_session, record.id)
        max_concurrency = _policy_int(definition, "max_concurrency", default=1, minimum=1, maximum=20)
        active_count = len([node for node in nodes if node.status in {"queued", "running", "waiting"}])
        available = max(max_concurrency - active_count, 0)
        ready_nodes = [node for node in nodes if node.status == "ready"][:available]
        for node in ready_nodes:
            await self._start_node(
                db_session, settings, runtime_state, run_dispatcher, record, node, node_specs[node.node_id]
            )
            changed = True
        nodes = await self._list_nodes(db_session, record.id)
        await self._finalize_if_complete(db_session, record, definition, nodes)
        record.current_node_ids = [
            node.node_id for node in nodes if node.status in {"ready", "queued", "running", "waiting"}
        ]
        if record.status not in _TERMINAL_WORKFLOW_STATUSES:
            record.status = "running" if record.current_node_ids else "waiting"
            if record.status == "waiting":
                await self._emit_event(
                    db_session,
                    record,
                    event_type="workflow_waiting",
                    source_kind="workflow",
                    message="Workflow is waiting for runnable nodes.",
                    payload={},
                )
        record.updated_at = utc_now()
        await db_session.flush()
        return changed

    def _apply_definition_current_session_filters(
        self,
        statement: Select[tuple[WorkflowDefinitionRecord]],
        *,
        current_session_id: str | None,
        created_by_current_session: bool,
        supervised_by_current_session: bool,
        touched_by_current_session: bool,
        only_current_session: bool,
    ) -> Select[tuple[WorkflowDefinitionRecord]]:
        if not _non_empty(current_session_id):
            return statement
        session_id = current_session_id.strip() if isinstance(current_session_id, str) else ""
        if created_by_current_session:
            return statement.where(WorkflowDefinitionRecord.owner_session_id == session_id)
        if supervised_by_current_session:
            return statement.where(
                exists().where(
                    WorkflowRunRecord.workflow_id == WorkflowDefinitionRecord.id,
                    WorkflowRunRecord.supervisor_session_id == session_id,
                )
            )
        if touched_by_current_session or only_current_session:
            return statement.where(
                or_(
                    WorkflowDefinitionRecord.owner_session_id == session_id,
                    exists().where(
                        WorkflowRunRecord.workflow_id == WorkflowDefinitionRecord.id,
                        WorkflowRunRecord.supervisor_session_id == session_id,
                    ),
                    exists().where(
                        WorkflowRunRecord.workflow_id == WorkflowDefinitionRecord.id,
                        WorkflowNodeRunRecord.workflow_run_id == WorkflowRunRecord.id,
                        WorkflowNodeRunRecord.session_id == session_id,
                    ),
                )
            )
        return statement

    def _filter_definition_records_for_current_session(
        self,
        records: list[WorkflowDefinitionRecord],
        *,
        current_session_id: str | None,
        created_by_current_session: bool,
        supervised_by_current_session: bool,
        touched_by_current_session: bool,
        only_current_session: bool,
    ) -> list[WorkflowDefinitionRecord]:
        _ = supervised_by_current_session, touched_by_current_session, only_current_session
        if created_by_current_session and _non_empty(current_session_id):
            return [record for record in records if record.owner_session_id == current_session_id]
        return records

    def _apply_run_current_session_filters(
        self,
        statement: Select[tuple[WorkflowRunRecord]],
        *,
        current_session_id: str | None,
        only_current_session: bool,
        only_supervised_by_current_session: bool,
        only_touched_by_current_session: bool,
    ) -> Select[tuple[WorkflowRunRecord]]:
        if not _non_empty(current_session_id):
            return statement
        session_id = current_session_id.strip() if isinstance(current_session_id, str) else ""
        if only_supervised_by_current_session:
            return statement.where(WorkflowRunRecord.supervisor_session_id == session_id)
        if only_current_session or only_touched_by_current_session:
            return statement.where(
                or_(
                    WorkflowRunRecord.supervisor_session_id == session_id,
                    exists().where(
                        WorkflowNodeRunRecord.workflow_run_id == WorkflowRunRecord.id,
                        WorkflowNodeRunRecord.session_id == session_id,
                    ),
                )
            )
        return statement

    async def _get_definition_record(
        self,
        db_session: AsyncSession,
        workflow_id: str,
        *,
        include_archived: bool = False,
    ) -> WorkflowDefinitionRecord:
        record = await db_session.get(WorkflowDefinitionRecord, workflow_id)
        if not isinstance(record, WorkflowDefinitionRecord) or (record.status == "archived" and not include_archived):
            raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' was not found.")
        return record

    async def _get_run_record(self, db_session: AsyncSession, workflow_run_id: str) -> WorkflowRunRecord:
        record = await db_session.get(WorkflowRunRecord, workflow_run_id)
        if not isinstance(record, WorkflowRunRecord):
            raise HTTPException(status_code=404, detail=f"Workflow run '{workflow_run_id}' was not found.")
        return record

    async def _definition_summary_from_record(
        self,
        db_session: AsyncSession,
        record: WorkflowDefinitionRecord,
        *,
        include_latest_run: bool,
    ) -> WorkflowDefinitionSummary:
        latest_run = None
        if include_latest_run:
            statement = (
                select(WorkflowRunRecord)
                .where(WorkflowRunRecord.workflow_id == record.id)
                .order_by(WorkflowRunRecord.created_at.desc())
                .limit(1)
            )
            result = await db_session.execute(statement)
            run_record = result.scalar_one_or_none()
            if isinstance(run_record, WorkflowRunRecord):
                latest_run = (await self._run_summary_from_record(db_session, run_record)).model_dump(mode="json")
        return definition_summary_from_record(record, latest_run=latest_run)

    async def _definition_detail_from_record(
        self,
        db_session: AsyncSession,
        record: WorkflowDefinitionRecord,
    ) -> WorkflowDefinitionDetail:
        summary = await self._definition_summary_from_record(db_session, record, include_latest_run=True)
        return WorkflowDefinitionDetail(
            **summary.model_dump(),
            input_schema=dict(record.input_schema or {}),
            definition=dict(record.definition or {}),
        )

    async def _run_summary_from_record(self, db_session: AsyncSession, record: WorkflowRunRecord) -> WorkflowRunSummary:
        workflow_name = _string_or_none((record.workflow_metadata or {}).get("workflow_name"))
        if workflow_name is None:
            definition = await db_session.get(WorkflowDefinitionRecord, record.workflow_id)
            workflow_name = definition.name if isinstance(definition, WorkflowDefinitionRecord) else None
        return WorkflowRunSummary(
            id=record.id,
            workflow_id=record.workflow_id,
            workflow_version=record.workflow_version,
            workflow_name=workflow_name,
            status=cast(WorkflowRunStatus, record.status),
            trigger_kind=cast(WorkflowTriggerKind, record.trigger_kind),
            supervisor_session_id=record.supervisor_session_id,
            supervisor_run_id=record.supervisor_run_id,
            profile_name=record.profile_name,
            inputs=dict(record.inputs or {}),
            result=record.result,
            error_message=record.error_message,
            current_node_ids=list(record.current_node_ids or []),
            metadata=dict(record.workflow_metadata or {}),
            created_at=record.created_at,
            started_at=record.started_at,
            finished_at=record.finished_at,
            updated_at=record.updated_at,
        )

    async def _run_detail_from_record(
        self,
        db_session: AsyncSession,
        record: WorkflowRunRecord,
        *,
        include_events: bool,
    ) -> WorkflowRunDetail:
        summary = await self._run_summary_from_record(db_session, record)
        nodes = await self._list_nodes(db_session, record.id)
        events: list[WorkflowEventSummary] = []
        if include_events:
            statement = (
                select(WorkflowEventRecord)
                .where(WorkflowEventRecord.workflow_run_id == record.id)
                .order_by(WorkflowEventRecord.created_at.asc(), WorkflowEventRecord.id.asc())
                .limit(200)
            )
            result = await db_session.execute(statement)
            events = [event_summary_from_record(event) for event in result.scalars().all()]
        return WorkflowRunDetail(
            **summary.model_dump(),
            definition=dict(record.definition_snapshot or {}),
            nodes=[node_summary_from_record(node) for node in nodes],
            events=events,
        )

    async def _list_nodes(self, db_session: AsyncSession, workflow_run_id: str) -> list[WorkflowNodeRunRecord]:
        statement = (
            select(WorkflowNodeRunRecord)
            .where(WorkflowNodeRunRecord.workflow_run_id == workflow_run_id)
            .order_by(WorkflowNodeRunRecord.node_id.asc(), WorkflowNodeRunRecord.attempt_no.asc())
        )
        result = await db_session.execute(statement)
        return list(result.scalars().all())

    async def _emit_event(
        self,
        db_session: AsyncSession,
        run_record: WorkflowRunRecord,
        *,
        event_type: str,
        source_kind: str,
        message: str,
        payload: dict[str, Any],
        node_run: WorkflowNodeRunRecord | None = None,
    ) -> WorkflowEventRecord:
        full_payload = {
            "workflow_run_id": run_record.id,
            "workflow_id": run_record.workflow_id,
            "message": message,
            **payload,
        }
        if node_run is not None:
            full_payload.update({"node_id": node_run.node_id, "node_run_id": node_run.id})
            if node_run.session_id is not None:
                full_payload["session_id"] = node_run.session_id
            if node_run.run_id is not None:
                full_payload["run_id"] = node_run.run_id
        event = WorkflowEventRecord(
            id=uuid4().hex,
            workflow_run_id=run_record.id,
            node_run_id=node_run.id if node_run is not None else None,
            source_kind=source_kind,
            event_type=event_type,
            payload=full_payload,
            created_at=utc_now(),
        )
        db_session.add(event)
        return event

    async def _refresh_node_statuses(
        self,
        db_session: AsyncSession,
        record: WorkflowRunRecord,
        nodes: list[WorkflowNodeRunRecord],
    ) -> bool:
        changed = False
        definition = normalize_workflow_definition(record.definition_snapshot)
        specs = workflow_node_specs(definition)
        for node in nodes:
            if node.status not in {"queued", "running", "waiting"} or not isinstance(node.run_id, str):
                continue
            run_record = await db_session.get(RunRecord, node.run_id)
            if not isinstance(run_record, RunRecord):
                continue
            if run_record.status == "running" and node.status != "running":
                node.status = "running"
                node.started_at = node.started_at or run_record.started_at or utc_now()
                node.updated_at = utc_now()
                await self._emit_event(
                    db_session,
                    record,
                    event_type="node_running",
                    source_kind="run",
                    node_run=node,
                    message=f"Node '{node.node_id}' is running.",
                    payload={"run_id": run_record.id},
                )
                changed = True
            if run_record.status not in _TERMINAL_RUN_STATUSES:
                continue
            node.finished_at = run_record.finished_at or run_record.committed_at or utc_now()
            node.updated_at = utc_now()
            if run_record.status == "completed":
                node.status = "completed"
                node.output_text = run_record.output_text
                node.output_json = _parse_json_object(run_record.output_text)
                await self._emit_event(
                    db_session,
                    record,
                    event_type="node_completed",
                    source_kind="run",
                    node_run=node,
                    message=f"Node '{node.node_id}' completed.",
                    payload={"run_id": run_record.id, "output_preview": _truncate(run_record.output_text, 1000)},
                )
                await self._emit_event(
                    db_session,
                    record,
                    event_type="node_output_available",
                    source_kind="node",
                    node_run=node,
                    message=f"Node '{node.node_id}' output is available.",
                    payload={"run_id": run_record.id},
                )
            elif run_record.status == "failed" and await self._retry_node_if_available(
                db_session, record, node, specs.get(node.node_id, {})
            ):
                pass
            else:
                node.status = "cancelled" if run_record.status == "cancelled" else "failed"
                node.error_message = run_record.error_message or run_record.termination_reason
                await self._emit_event(
                    db_session,
                    record,
                    event_type="node_failed" if node.status == "failed" else "node_cancelled",
                    source_kind="run",
                    node_run=node,
                    message=f"Node '{node.node_id}' {node.status}.",
                    payload={"run_id": run_record.id, "error_message": node.error_message},
                )
            changed = True
        return changed

    async def _retry_node_if_available(
        self,
        db_session: AsyncSession,
        record: WorkflowRunRecord,
        node: WorkflowNodeRunRecord,
        spec: Mapping[str, Any],
    ) -> bool:
        retry_count = int(spec.get("retry_count") or _policy_int(record.definition_snapshot, "retry_count", 0, 0, 10))
        if node.attempt_no > retry_count:
            return False
        node.attempt_no += 1
        node.status = "ready"
        node.session_id = None
        node.run_id = None
        node.error_message = None
        node.started_at = None
        node.finished_at = None
        node.updated_at = utc_now()
        await self._emit_event(
            db_session,
            record,
            event_type="node_ready",
            source_kind="workflow",
            node_run=node,
            message=f"Node '{node.node_id}' retry attempt {node.attempt_no} is ready.",
            payload={"attempt_no": node.attempt_no},
        )
        return True

    async def _mark_ready_nodes(
        self,
        db_session: AsyncSession,
        record: WorkflowRunRecord,
        nodes: list[WorkflowNodeRunRecord],
    ) -> bool:
        changed = False
        completed = {node.node_id for node in nodes if node.status == "completed"}
        for node in nodes:
            if node.status != "pending":
                continue
            if all(dep in completed for dep in list(node.needs or [])):
                node.status = "ready"
                node.updated_at = utc_now()
                await self._emit_event(
                    db_session,
                    record,
                    event_type="node_ready",
                    source_kind="workflow",
                    node_run=node,
                    message=f"Node '{node.node_id}' is ready.",
                    payload={"needs": list(node.needs or [])},
                )
                changed = True
        return changed

    async def _start_node(
        self,
        db_session: AsyncSession,
        settings: ClawSettings,
        runtime_state: InMemoryRuntimeState,
        run_dispatcher: Any,
        record: WorkflowRunRecord,
        node: WorkflowNodeRunRecord,
        spec: Mapping[str, Any],
    ) -> None:
        profile_name = _node_profile(spec, record.profile_name) or settings.default_profile
        mode = _node_mode(spec)
        input_parts = render_node_input_parts(record, node, spec, await self._node_context(db_session, record.id))
        workspace = (
            WorkspaceBindingSpec.model_validate(record.workspace) if isinstance(record.workspace, dict) else None
        )
        metadata: dict[str, Any] = {
            "source": "workflow",
            "workflow_id": record.workflow_id,
            "workflow_run_id": record.id,
            "workflow_node_id": node.node_id,
            "workflow_node_run_id": node.id,
            "workflow_node_mode": mode,
        }
        workflow_shell_env = _normalize_shell_env((record.workflow_metadata or {}).get("shell_env"))
        if workflow_shell_env:
            metadata["shell_env"] = workflow_shell_env
        session_id, restore_from_run_id, reset_state = await self._resolve_node_session_plan(
            db_session, record, node, mode
        )
        if mode == "steer" and isinstance(session_id, str):
            session = await db_session.get(SessionRecord, session_id)
            if isinstance(session, SessionRecord) and isinstance(session.active_run_id, str):
                await self._run_controller.steer(
                    db_session,
                    runtime_state,
                    session.active_run_id,
                    SteerRequest(input_parts=parse_input_parts(input_parts)),
                )
                node.profile_name = profile_name
                node.session_id = session.id
                node.run_id = session.active_run_id
                node.input_parts = input_parts
                node.status = "waiting"
                node.started_at = utc_now()
                node.updated_at = utc_now()
                await self._emit_event(
                    db_session,
                    record,
                    event_type="node_steered",
                    source_kind="workflow",
                    node_run=node,
                    message=f"Node '{node.node_id}' steered an active run.",
                    payload={"session_id": session.id, "run_id": session.active_run_id, "mode": mode},
                )
                return
        run = await self._run_controller.create(
            db_session,
            settings,
            runtime_state,
            RunCreateRequest(
                session_id=session_id,
                restore_from_run_id=restore_from_run_id,
                reset_state=reset_state,
                profile_name=profile_name,
                input_parts=parse_input_parts(input_parts),
                trigger_type=TriggerType.WORKFLOW,
                metadata=metadata,
                workspace=workspace,
                dispatch_mode=DispatchMode.ASYNC,
            ),
        )
        if mode in {"continue", "steer"} and not isinstance(
            (record.workflow_metadata or {}).get("shared_session_id"), str
        ):
            updated_metadata = dict(record.workflow_metadata or {})
            updated_metadata["shared_session_id"] = run.session_id
            record.workflow_metadata = updated_metadata
        node.profile_name = profile_name
        node.session_id = run.session_id
        node.run_id = run.id
        node.input_parts = input_parts
        node.status = "queued"
        node.started_at = utc_now()
        node.updated_at = utc_now()
        dispatch_result = run_dispatcher.dispatch(run.id, DispatchMode.ASYNC)
        if not getattr(dispatch_result, "submitted", False):
            node.status = "queued"
            node.error_message = f"Dispatch pending: {getattr(dispatch_result, 'reason', None)}"
        await self._emit_event(
            db_session,
            record,
            event_type="node_queued",
            source_kind="workflow",
            node_run=node,
            message=f"Node '{node.node_id}' queued.",
            payload={"session_id": run.session_id, "run_id": run.id, "mode": mode},
        )

    async def _resolve_node_session_plan(
        self,
        db_session: AsyncSession,
        record: WorkflowRunRecord,
        node: WorkflowNodeRunRecord,
        mode: str,
    ) -> tuple[str | None, str | None, bool]:
        if mode == "continue":
            shared_session_id = _string_or_none((record.workflow_metadata or {}).get("shared_session_id"))
            if shared_session_id is not None:
                return shared_session_id, None, False
            return None, None, True
        if mode == "fork":
            parent_session_id = await self._resolve_fork_parent_session_id(db_session, record, node)
            if parent_session_id is not None:
                parent_session = await lock_session_reference(db_session, parent_session_id)
                if parent_session is None:
                    raise RuntimeError(f"Workflow fork parent session '{parent_session_id}' was not found")
                fork_session = SessionRecord(
                    id=uuid4().hex,
                    parent_session_id=parent_session_id,
                    profile_name=node.profile_name or record.profile_name,
                    session_metadata={"source": "workflow", "workflow_run_id": record.id, "node_id": node.node_id},
                )
                db_session.add(fork_session)
                await db_session.flush()
                return fork_session.id, None, True
        if mode == "steer":
            shared_session_id = _string_or_none((record.workflow_metadata or {}).get("shared_session_id"))
            if shared_session_id is not None:
                session = await db_session.get(SessionRecord, shared_session_id)
                if isinstance(session, SessionRecord) and isinstance(session.active_run_id, str):
                    return session.id, None, False
                return shared_session_id, None, False
        return None, None, True

    async def _resolve_fork_parent_session_id(
        self,
        db_session: AsyncSession,
        record: WorkflowRunRecord,
        node: WorkflowNodeRunRecord,
    ) -> str | None:
        for dep_id in list(node.needs or []):
            result = await db_session.execute(
                select(WorkflowNodeRunRecord)
                .where(
                    WorkflowNodeRunRecord.workflow_run_id == record.id,
                    WorkflowNodeRunRecord.node_id == dep_id,
                    WorkflowNodeRunRecord.status == "completed",
                )
                .order_by(WorkflowNodeRunRecord.attempt_no.desc())
                .limit(1)
            )
            dep_node = result.scalar_one_or_none()
            if isinstance(dep_node, WorkflowNodeRunRecord) and isinstance(dep_node.session_id, str):
                return dep_node.session_id
        return record.supervisor_session_id

    async def _node_context(self, db_session: AsyncSession, workflow_run_id: str) -> dict[str, Any]:
        nodes = await self._list_nodes(db_session, workflow_run_id)
        return {
            "nodes": {
                node.node_id: {
                    "output_text": node.output_text,
                    "output_json": node.output_json,
                    "session_id": node.session_id,
                    "run_id": node.run_id,
                    "status": node.status,
                }
                for node in nodes
            }
        }

    async def _finalize_if_complete(
        self,
        db_session: AsyncSession,
        record: WorkflowRunRecord,
        definition: Mapping[str, Any],
        nodes: list[WorkflowNodeRunRecord],
    ) -> None:
        if not nodes or any(node.status not in _TERMINAL_NODE_STATUSES for node in nodes):
            return
        if any(node.status == "failed" for node in nodes):
            await self._fail_workflow(db_session, record, "One or more workflow nodes failed.")
            return
        if any(node.status == "cancelled" for node in nodes):
            record.status = "cancelled"
            record.finished_at = utc_now()
            record.updated_at = utc_now()
            return
        record.result = project_workflow_result(definition, nodes)
        record.status = "completed"
        record.finished_at = utc_now()
        record.current_node_ids = []
        record.updated_at = utc_now()
        await self._emit_event(
            db_session,
            record,
            event_type="workflow_completed",
            source_kind="workflow",
            message="Workflow completed.",
            payload={"result": record.result},
        )

    async def _fail_workflow(self, db_session: AsyncSession, record: WorkflowRunRecord, error_message: str) -> None:
        record.status = "failed"
        record.error_message = error_message
        record.finished_at = utc_now()
        record.current_node_ids = []
        record.updated_at = utc_now()
        await self._emit_event(
            db_session,
            record,
            event_type="workflow_failed",
            source_kind="workflow",
            message="Workflow failed.",
            payload={"error_message": error_message},
        )

    async def _resolve_supervisor_profile(
        self,
        db_session: AsyncSession,
        supervisor_session_id: str | None,
        supervisor_run_id: str | None,
    ) -> str | None:
        if _non_empty(supervisor_run_id):
            run = await db_session.get(RunRecord, supervisor_run_id)
            if isinstance(run, RunRecord) and _non_empty(run.profile_name):
                return run.profile_name
        if _non_empty(supervisor_session_id):
            session = await db_session.get(SessionRecord, supervisor_session_id)
            if isinstance(session, SessionRecord) and _non_empty(session.profile_name):
                return session.profile_name
        return None


def _normalize_shell_env(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): str(item) for key, item in value.items() if isinstance(key, str) and isinstance(item, str)}


def definition_summary_from_record(
    record: WorkflowDefinitionRecord,
    *,
    latest_run: dict[str, Any] | None = None,
) -> WorkflowDefinitionSummary:
    return WorkflowDefinitionSummary(
        id=record.id,
        name=record.name,
        description=record.description,
        status=cast(WorkflowDefinitionStatus, record.status),
        definition_version=record.definition_version,
        schema_version=record.schema_version,
        owner_kind=record.owner_kind,
        owner_session_id=record.owner_session_id,
        owner_run_id=record.owner_run_id,
        scope=cast(WorkflowScope, record.scope),
        tags=list(record.tags or []),
        when_to_use=record.when_to_use,
        argument_hint=record.argument_hint,
        latest_run=latest_run,
        metadata=dict(record.workflow_metadata or {}),
        created_at=record.created_at,
        updated_at=record.updated_at,
        archived_at=record.archived_at,
    )


def node_summary_from_record(record: WorkflowNodeRunRecord) -> WorkflowNodeRunSummary:
    return WorkflowNodeRunSummary(
        id=record.id,
        workflow_run_id=record.workflow_run_id,
        node_id=record.node_id,
        attempt_no=record.attempt_no,
        status=cast(WorkflowNodeRunStatus, record.status),
        profile_name=record.profile_name,
        session_id=record.session_id,
        run_id=record.run_id,
        input_preview=_prompt_from_input_parts(record.input_parts),
        output_text=record.output_text,
        output_json=record.output_json,
        error_message=record.error_message,
        needs=list(record.needs or []),
        metadata=dict(record.node_metadata or {}),
        started_at=record.started_at,
        finished_at=record.finished_at,
        updated_at=record.updated_at,
    )


def event_summary_from_record(record: WorkflowEventRecord) -> WorkflowEventSummary:
    return WorkflowEventSummary(
        id=record.id,
        workflow_run_id=record.workflow_run_id,
        node_run_id=record.node_run_id,
        source_kind=record.source_kind,
        event_type=record.event_type,
        payload=dict(record.payload or {}),
        created_at=record.created_at,
    )


def normalize_workflow_definition(definition: Mapping[str, Any]) -> dict[str, Any]:  # noqa: C901
    if not isinstance(definition, Mapping):
        raise HTTPException(status_code=422, detail="definition must be an object.")
    normalized = dict(definition)
    schema = str(normalized.get("schema") or _DEFAULT_SCHEMA_VERSION)
    if schema != _DEFAULT_SCHEMA_VERSION:
        raise HTTPException(status_code=422, detail=f"Unsupported workflow schema '{schema}'.")
    normalized["schema"] = schema
    nodes = normalized.get("nodes")
    if not isinstance(nodes, Mapping) or not nodes:
        raise HTTPException(status_code=422, detail="definition.nodes must contain at least one node.")
    node_specs: dict[str, dict[str, Any]] = {}
    for raw_node_id, raw_spec in nodes.items():
        node_id = str(raw_node_id).strip()
        if node_id == "":
            raise HTTPException(status_code=422, detail="workflow node id must not be empty.")
        if not isinstance(raw_spec, Mapping):
            raise HTTPException(status_code=422, detail=f"workflow node '{node_id}' must be an object.")
        spec = dict(raw_spec)
        needs = spec.get("needs") or []
        if not isinstance(needs, list) or any(not isinstance(item, str) or item.strip() == "" for item in needs):
            raise HTTPException(status_code=422, detail=f"workflow node '{node_id}' has invalid needs.")
        spec["needs"] = [item.strip() for item in needs]
        mode = str(spec.get("mode") or "isolate")
        if mode not in _ALLOWED_NODE_MODES:
            raise HTTPException(status_code=422, detail=f"workflow node '{node_id}' has invalid mode '{mode}'.")
        spec["mode"] = mode
        if "prompt" not in spec and "input_parts" not in spec:
            raise HTTPException(status_code=422, detail=f"workflow node '{node_id}' requires prompt or input_parts.")
        node_specs[node_id] = spec
    node_ids = set(node_specs)
    for node_id, spec in node_specs.items():
        missing = [dep for dep in spec["needs"] if dep not in node_ids]
        if missing:
            raise HTTPException(
                status_code=422, detail=f"workflow node '{node_id}' references missing nodes {missing}."
            )
    _validate_acyclic(node_specs)
    normalized["nodes"] = node_specs
    policy = normalized.get("policy")
    normalized["policy"] = dict(policy) if isinstance(policy, Mapping) else {}
    if not isinstance(normalized.get("version"), int):
        normalized["version"] = int(normalized.get("version") or 1)
    return normalized


def workflow_node_specs(definition: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    normalized = normalize_workflow_definition(definition)
    return cast(dict[str, dict[str, Any]], normalized["nodes"])


def validate_workflow_inputs(input_schema: Mapping[str, Any], inputs: Mapping[str, Any]) -> None:
    schema_type = input_schema.get("type")
    if schema_type is not None and schema_type != "object":
        raise HTTPException(status_code=422, detail="workflow input schema must be an object schema.")
    required = input_schema.get("required")
    if isinstance(required, list):
        missing = [item for item in required if isinstance(item, str) and item not in inputs]
        if missing:
            raise HTTPException(status_code=422, detail=f"workflow inputs missing required fields: {missing}.")
    properties = input_schema.get("properties")
    if isinstance(properties, Mapping):
        for key, spec in properties.items():
            if key not in inputs or not isinstance(spec, Mapping):
                continue
            expected_type = spec.get("type")
            if isinstance(expected_type, str) and not _json_type_matches(inputs[key], expected_type):
                raise HTTPException(status_code=422, detail=f"workflow input '{key}' must be {expected_type}.")


def render_node_input_parts(
    record: WorkflowRunRecord,
    node: WorkflowNodeRunRecord,
    spec: Mapping[str, Any],
    node_context: Mapping[str, Any],
) -> list[dict[str, Any]]:
    context = {"inputs": dict(record.inputs or {}), **dict(node_context), "node": {"id": node.node_id}}
    if isinstance(spec.get("input_parts"), list):
        rendered = _render_json_compatible(spec["input_parts"], context)
        return [dict(item) for item in rendered if isinstance(item, dict)]
    prompt = str(spec.get("prompt") or "")
    rendered_prompt = render_template(prompt, context)
    return [TextPart(type="text", text=rendered_prompt).model_dump(mode="json")]


def render_template(template: str, context: Mapping[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        expr = match.group(1).strip()
        value = _eval_template_expr(expr, context)
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    return _TEMPLATE_EXPR_RE.sub(replace, template)


def project_workflow_result(definition: Mapping[str, Any], nodes: list[WorkflowNodeRunRecord]) -> dict[str, Any]:
    result_spec = definition.get("result")
    node_map = {node.node_id: node for node in nodes}
    if isinstance(result_spec, Mapping):
        from_node = result_spec.get("from_node")
        if isinstance(from_node, str) and from_node in node_map:
            node = node_map[from_node]
            return {
                "from_node": from_node,
                "output_text": node.output_text,
                "output_json": node.output_json,
                "session_id": node.session_id,
                "run_id": node.run_id,
            }
    completed = [node for node in nodes if node.status == "completed"]
    if completed:
        node = completed[-1]
        return {
            "from_node": node.node_id,
            "output_text": node.output_text,
            "output_json": node.output_json,
            "session_id": node.session_id,
            "run_id": node.run_id,
        }
    return {}


def _validate_acyclic(node_specs: Mapping[str, Mapping[str, Any]]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visited:
            return
        if node_id in visiting:
            raise HTTPException(status_code=422, detail=f"workflow contains a cycle at node '{node_id}'.")
        visiting.add(node_id)
        for dep in list(node_specs[node_id].get("needs") or []):
            visit(dep)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in node_specs:
        visit(node_id)


def _eval_template_expr(expr: str, context: Mapping[str, Any]) -> Any:
    raw_expr, default_value = _split_default_filter(expr)
    value: Any = context
    for part in raw_expr.split("."):
        part = part.strip()
        if part == "":
            continue
        value = value.get(part) if isinstance(value, Mapping) else getattr(value, part, None)
        if value is None:
            break
    if (value is None or value == "") and default_value is not None:
        return default_value
    return value


def _split_default_filter(expr: str) -> tuple[str, str | None]:
    if "|" not in expr:
        return expr, None
    raw_expr, _, raw_filter = expr.partition("|")
    filter_text = raw_filter.strip()
    if not filter_text.startswith("default(") or not filter_text.endswith(")"):
        return raw_expr.strip(), None
    raw_value = filter_text[len("default(") : -1].strip()
    if (raw_value.startswith('"') and raw_value.endswith('"')) or (
        raw_value.startswith("'") and raw_value.endswith("'")
    ):
        return raw_expr.strip(), raw_value[1:-1]
    return raw_expr.strip(), raw_value


def _render_json_compatible(value: Any, context: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        return render_template(value, context)
    if isinstance(value, list):
        return [_render_json_compatible(item, context) for item in value]
    if isinstance(value, dict):
        return {key: _render_json_compatible(item, context) for key, item in value.items()}
    return value


def _json_type_matches(value: Any, expected_type: str) -> bool:
    return (
        (expected_type == "string" and isinstance(value, str))
        or (expected_type == "number" and isinstance(value, int | float))
        or (expected_type == "integer" and isinstance(value, int) and not isinstance(value, bool))
        or (expected_type == "boolean" and isinstance(value, bool))
        or (expected_type == "array" and isinstance(value, list))
        or (expected_type == "object" and isinstance(value, dict))
        or (expected_type == "null" and value is None)
    )


def _node_profile(spec: Mapping[str, Any], workflow_profile: str | None) -> str | None:
    raw_profile = spec.get("profile")
    if not isinstance(raw_profile, str) or raw_profile.strip() == "" or raw_profile.strip() == "Self":
        return workflow_profile
    return raw_profile.strip()


def _node_mode(spec: Mapping[str, Any]) -> WorkflowNodeMode:
    mode = str(spec.get("mode") or "isolate")
    return cast(WorkflowNodeMode, mode if mode in _ALLOWED_NODE_MODES else "isolate")


def _policy_int(
    definition: Mapping[str, Any],
    key: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    policy = definition.get("policy") if isinstance(definition, Mapping) else None
    value = policy.get(key) if isinstance(policy, Mapping) else None
    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError):
        parsed = default
    return min(max(parsed, minimum), maximum)


def _input_schema_from_definition(definition: Mapping[str, Any]) -> dict[str, Any]:
    inputs = definition.get("inputs")
    return dict(inputs) if isinstance(inputs, Mapping) else {}


def _prompt_from_input_parts(input_parts: list[dict[str, Any]]) -> str | None:
    for part in parse_input_parts(input_parts):
        if isinstance(part, TextPart):
            return part.text
    return None


def _normalize_tags(value: Any) -> list[str]:
    if isinstance(value, str):
        raw_items: Iterable[Any] = value.split(",")
    elif isinstance(value, Iterable):
        raw_items = value
    else:
        return []
    tags: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        if isinstance(item, str):
            for raw_tag in item.split(","):
                tag = raw_tag.strip()
                if tag and tag not in seen:
                    seen.add(tag)
                    tags.append(tag)
    return tags


def _parse_json_object(value: str | None) -> dict[str, Any] | None:
    if not isinstance(value, str) or value.strip() == "":
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _truncate(value: str | None, limit: int) -> str | None:
    if value is None:
        return None
    return value if len(value) <= limit else value[:limit]


def _string_or_none(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    return str(value)


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _non_empty(value: str | None) -> bool:
    return isinstance(value, str) and value.strip() != ""
