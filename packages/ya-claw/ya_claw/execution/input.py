from __future__ import annotations

import base64
import mimetypes
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic_ai import BinaryContent, ImageUrl, VideoUrl
from pydantic_ai.messages import AudioUrl, DocumentUrl
from ya_agent_environment import FileOperator

from ya_claw.controller.models import (
    BinaryPart,
    CommandPart,
    ContentPart,
    FilePart,
    InputPart,
    ModePart,
    TextPart,
    UrlPart,
    extract_input_preview,
)

UserPromptPart = str | BinaryContent | ImageUrl | VideoUrl | AudioUrl | DocumentUrl


@dataclass(slots=True)
class InputMappingResult:
    user_prompt: list[UserPromptPart]
    mode_parts: list[ModePart]
    command_parts: list[CommandPart]
    content_parts: list[ContentPart]
    input_preview: str | None


@dataclass(slots=True)
class SplitInputPartsResult:
    mode_parts: list[ModePart]
    command_parts: list[CommandPart]
    content_parts: list[ContentPart]


def split_input_parts(input_parts: Sequence[InputPart]) -> SplitInputPartsResult:
    mode_parts: list[ModePart] = []
    command_parts: list[CommandPart] = []
    content_parts: list[ContentPart] = []
    stage = "mode"

    for part in input_parts:
        if isinstance(part, ModePart):
            if stage != "mode":
                raise ValueError("ModePart must appear before command and content parts.")
            mode_parts.append(part)
            continue

        if isinstance(part, CommandPart):
            if stage == "content":
                raise ValueError("CommandPart must appear before content parts.")
            stage = "command"
            command_parts.append(part)
            continue

        stage = "content"
        content_parts.append(part)

    return SplitInputPartsResult(
        mode_parts=mode_parts,
        command_parts=command_parts,
        content_parts=content_parts,
    )


async def map_input_parts(
    input_parts: Sequence[InputPart],
    *,
    file_operator: FileOperator | None = None,
) -> InputMappingResult:
    split_result = split_input_parts(input_parts)
    user_prompt: list[UserPromptPart] = []

    for part in split_result.content_parts:
        mapped_parts = await _map_content_part(part, file_operator=file_operator)
        user_prompt.extend(mapped_parts)

    return InputMappingResult(
        user_prompt=user_prompt,
        mode_parts=split_result.mode_parts,
        command_parts=split_result.command_parts,
        content_parts=split_result.content_parts,
        input_preview=extract_input_preview(list(input_parts)),
    )


async def _map_content_part(
    part: ContentPart,
    *,
    file_operator: FileOperator | None,
) -> list[UserPromptPart]:
    if isinstance(part, TextPart):
        return [part.text]

    if isinstance(part, UrlPart):
        return [_map_url_part(part)]

    if isinstance(part, BinaryPart):
        return [_map_binary_part(part)]

    if isinstance(part, FilePart):
        if file_operator is None:
            return [f"Attached file: {part.path}"]
        data = await file_operator.read_bytes(part.path)
        media_type = mimetypes.guess_type(part.path)[0] or _default_media_type(part.kind)
        return [_map_file_bytes(part.kind, data, media_type, part.path)]

    raise TypeError(f"Unsupported content part: {type(part)!r}")


def _map_url_part(part: UrlPart) -> UserPromptPart:
    if part.kind == "image":
        return ImageUrl(url=part.url)
    if part.kind == "video":
        return VideoUrl(url=part.url)
    if part.kind == "audio":
        return AudioUrl(url=part.url)
    return DocumentUrl(url=part.url)


def _map_binary_part(part: BinaryPart) -> UserPromptPart:
    data = base64.b64decode(part.data)
    return _map_file_bytes(part.kind, data, part.mime_type, part.filename)


def _map_file_bytes(kind: str, data: bytes, media_type: str, file_name: str | None) -> UserPromptPart:
    if kind in {"image", "video", "audio", "document"}:
        return BinaryContent(data=data, media_type=media_type)
    label = file_name or "attachment"
    return f"Attached file: {label}"


def _default_media_type(kind: str) -> str:
    if kind == "image":
        return "image/png"
    if kind == "video":
        return "video/mp4"
    if kind == "audio":
        return "audio/mpeg"
    return "application/octet-stream"
