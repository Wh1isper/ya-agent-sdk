from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path

from loguru import logger
from ya_agent_sdk.environment.virtual_path import VirtualPath, normalize_virtual_path

from ya_claw.workspace.provider import WorkspaceBinding

HEARTBEAT_GUIDANCE_FILENAME = "HEARTBEAT.md"
HEARTBEAT_GUIDANCE_TAG = "heartbeat-guidance"


@dataclass(slots=True)
class HeartbeatGuidance:
    host_path: Path
    virtual_path: VirtualPath
    content: str


def load_heartbeat_guidance(binding: WorkspaceBinding) -> HeartbeatGuidance | None:
    guidance_path = binding.host_path / HEARTBEAT_GUIDANCE_FILENAME
    if not guidance_path.is_file():
        return None

    try:
        content = guidance_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to read heartbeat guidance file path={} error={}", guidance_path, exc)
        return None
    except UnicodeDecodeError as exc:
        logger.warning("Failed to decode heartbeat guidance file path={} error={}", guidance_path, exc)
        return None

    if content.strip() == "":
        return None

    return HeartbeatGuidance(
        host_path=guidance_path,
        virtual_path=normalize_virtual_path(binding.virtual_path / HEARTBEAT_GUIDANCE_FILENAME),
        content=content,
    )


def format_heartbeat_guidance(guidance: HeartbeatGuidance) -> str:
    path = escape(guidance.virtual_path.as_posix(), quote=True)
    return f'<{HEARTBEAT_GUIDANCE_TAG} path="{path}">\n{guidance.content}\n</{HEARTBEAT_GUIDANCE_TAG}>'
