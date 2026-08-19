from __future__ import annotations

from fastapi import HTTPException
from pydantic_ai import AgentSpec
from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession
from ya_agent_sdk.subagents import SubagentSpec

from ya_claw.config import ClawSettings
from ya_claw.controller.models import (
    ProfileDetail,
    ProfileSeedResponse,
    ProfileSummary,
    ProfileUpsertRequest,
)
from ya_claw.execution.profile import ProfileResolver
from ya_claw.mcp import normalize_profile_mcp_servers
from ya_claw.orm.tables import ProfileRecord
from ya_claw.profile_spec import ClawProfileHostConfig


class ProfileController:
    async def exists(self, db_session: AsyncSession, profile_name: str) -> bool:
        record = await db_session.get(ProfileRecord, profile_name)
        return isinstance(record, ProfileRecord)

    async def list(self, db_session: AsyncSession) -> list[ProfileSummary]:
        statement: Select[tuple[ProfileRecord]] = select(ProfileRecord).order_by(ProfileRecord.name.asc())
        result = await db_session.execute(statement)
        return [profile_summary_from_record(record) for record in result.scalars().all()]

    async def get(self, db_session: AsyncSession, profile_name: str) -> ProfileDetail:
        record = await db_session.get(ProfileRecord, profile_name)
        if not isinstance(record, ProfileRecord):
            raise HTTPException(
                status_code=404,
                detail=f"Profile '{profile_name}' was not found.",
            )
        return profile_detail_from_record(record)

    async def upsert(
        self,
        db_session: AsyncSession,
        profile_name: str,
        request: ProfileUpsertRequest,
    ) -> ProfileDetail:
        normalized_name = profile_name.strip()
        if not normalized_name:
            raise HTTPException(status_code=422, detail="Profile name cannot be empty.")
        if request.agent.name is not None and request.agent.name != normalized_name:
            raise HTTPException(
                status_code=422,
                detail=(f"Native AgentSpec name {request.agent.name!r} must match profile name {normalized_name!r}."),
            )
        record = await db_session.get(ProfileRecord, normalized_name)
        if not isinstance(record, ProfileRecord):
            record = ProfileRecord(
                name=normalized_name,
                agent_spec=request.agent.model_dump(mode="json", by_alias=True),
            )
            db_session.add(record)

        host = request.host.model_copy(update={"mcp_servers": normalize_profile_mcp_servers(request.host.mcp_servers)})
        record.agent_spec = request.agent.model_dump(mode="json", by_alias=True)
        record.host_config = host.model_dump(mode="json")
        record.subagent_specs = [spec.model_dump(mode="json", by_alias=True) for spec in request.subagents]
        record.enabled = request.enabled
        record.source_type = request.source_type or "api"
        record.source_version = request.source_version
        record.source_checksum = request.source_checksum
        await db_session.commit()
        await db_session.refresh(record)
        return profile_detail_from_record(record)

    async def delete(self, db_session: AsyncSession, profile_name: str) -> None:
        record = await db_session.get(ProfileRecord, profile_name)
        if not isinstance(record, ProfileRecord):
            raise HTTPException(
                status_code=404,
                detail=f"Profile '{profile_name}' was not found.",
            )
        await db_session.delete(record)
        await db_session.commit()

    async def seed(
        self,
        *,
        settings: ClawSettings,
        resolver: ProfileResolver,
        prune_missing: bool,
    ) -> ProfileSeedResponse:
        seed_file = settings.resolved_profile_seed_file
        if seed_file is None or not seed_file.exists():
            raise HTTPException(
                status_code=404,
                detail="Profile seed file is not configured or does not exist.",
            )
        seeded_names = await resolver.seed_profiles(prune_missing=prune_missing)
        return ProfileSeedResponse(
            seeded_names=seeded_names,
            seed_file=str(seed_file),
            prune_missing=prune_missing,
        )


def profile_summary_from_record(record: ProfileRecord) -> ProfileSummary:
    agent = AgentSpec.model_validate(record.agent_spec)
    host = ClawProfileHostConfig.model_validate(record.host_config)
    model = agent.model
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"Profile {record.name!r} has no model")
    return ProfileSummary(
        name=record.name,
        model=model,
        workspace_backend_hint=host.workspace_backend_hint,
        enabled=record.enabled,
        source_type=record.source_type,
        source_version=record.source_version,
        updated_at=record.updated_at,
    )


def profile_detail_from_record(record: ProfileRecord) -> ProfileDetail:
    agent = AgentSpec.model_validate(record.agent_spec)
    host = ClawProfileHostConfig.model_validate(record.host_config)
    return ProfileDetail(
        **profile_summary_from_record(record).model_dump(),
        agent=agent,
        host=host,
        subagents=[SubagentSpec.model_validate(item) for item in record.subagent_specs or []],
        source_checksum=record.source_checksum,
        created_at=record.created_at,
    )
