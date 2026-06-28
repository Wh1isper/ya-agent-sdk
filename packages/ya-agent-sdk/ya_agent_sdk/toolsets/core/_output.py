"""Shared helpers for keeping tool outputs within model-safe limits."""

from __future__ import annotations

import json
import uuid
from typing import Any

from ya_agent_environment import FileOperator

from ya_agent_sdk._logger import get_logger

logger = get_logger(__name__)

DEFAULT_OUTPUT_TRUNCATE_LIMIT = 20_000


def dump_tool_output(value: Any) -> str:
    """Serialize a tool result for size checks and tmp spill files."""
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str, ensure_ascii=False)


def tool_output_size(value: Any) -> int:
    """Return the serialized character count of a tool result."""
    return len(dump_tool_output(value))


def output_too_large_message(
    *,
    size: int,
    output_path: str | None,
    noun: str = "output",
) -> str:
    """Build consistent guidance for oversized tool outputs."""
    if output_path is not None:
        return f"Output too large ({size} chars). Full {noun} saved to `output_file_path`. Use `view` to inspect it."
    return f"Output too large ({size} chars). Failed to save full {noun}; showing a bounded preview."


async def write_tmp_output(
    file_operator: FileOperator | None,
    *,
    prefix: str,
    content: str | bytes,
    extension: str = "json",
) -> str | None:
    """Write large output to the environment tmp area when available."""
    if file_operator is None:
        return None

    filename = f"{prefix}-{uuid.uuid4().hex[:12]}.{extension.lstrip('.')}"
    try:
        return await file_operator.write_tmp_file(filename, content)
    except Exception:
        logger.warning("Failed to write %s output to temp file", prefix, exc_info=True)
        return None


def truncate_text(text: str, max_chars: int, *, suffix: str) -> str:
    """Truncate a text field to a hard character budget including suffix."""
    if len(text) <= max_chars:
        return text
    if max_chars <= 0:
        return suffix[: max(0, max_chars)]
    if max_chars <= len(suffix):
        return suffix[:max_chars]
    return text[: max_chars - len(suffix)] + suffix


def fit_text_fields_to_limit(
    result: dict[str, Any],
    *,
    text_fields: tuple[str, ...],
    limit: int = DEFAULT_OUTPUT_TRUNCATE_LIMIT,
    suffix: str,
) -> dict[str, Any]:
    """Shrink selected text fields until the serialized dict fits the limit."""
    if tool_output_size(result) <= limit:
        return result

    preview = dict(result)
    originals = {field: value for field in text_fields if isinstance((value := preview.get(field)), str)}
    if not originals:
        return preview

    for field in originals:
        preview[field] = ""

    base_size = tool_output_size(preview)
    available = max(0, limit - base_size - 1)
    per_field = available // len(originals) if originals else 0

    while True:
        for field, value in originals.items():
            preview[field] = truncate_text(value, per_field, suffix=suffix)
        if tool_output_size(preview) <= limit or per_field <= 0:
            break
        per_field = max(0, int(per_field * 0.8) - 1)

    if tool_output_size(preview) <= limit:
        return preview

    marker = suffix.lstrip("\n") or "... (omitted)"
    for field in originals:
        preview[field] = marker
    return preview


def append_guidance(value: str | None, guidance: str) -> str:
    """Append a guidance sentence to an existing note/hint field."""
    if not value:
        return guidance
    if value.endswith((".", "!", "?")):
        return f"{value} {guidance}"
    return f"{value}. {guidance}"
