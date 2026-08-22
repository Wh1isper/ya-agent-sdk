"""Bounded model-facing projections for subagent execution records."""

from __future__ import annotations

import json
from typing import Any

from ya_agent_sdk.subagents.spec import SubagentExecutionRecord

DEFAULT_EXECUTION_PAGE_SIZE = 20
MAX_EXECUTION_PAGE_SIZE = 100
DEFAULT_OUTPUT_PAGE_CHARS = 4_000
MAX_OUTPUT_PAGE_CHARS = 16_000
_MAX_ERROR_CHARS = 1_000

_MODEL_RECORD_FIELDS = {
    "execution_id",
    "route",
    "mode",
    "state",
    "input_state",
    "delivery_state",
    "resumed_from",
    "segment_index",
}


def model_record_payload(
    record: SubagentExecutionRecord,
    *,
    include_output: bool = False,
    output_offset: int = 0,
    output_limit: int = DEFAULT_OUTPUT_PAGE_CHARS,
) -> dict[str, Any]:
    """Project one record without internal runtime, descriptor, or state identities."""
    payload = record.model_dump(mode="json", include=_MODEL_RECORD_FIELDS)
    payload["has_deferred"] = record.deferred is not None
    payload["error"] = _bounded_error(record.error)
    if include_output:
        payload["output"] = output_page(
            record.output,
            offset=output_offset,
            limit=output_limit,
        )
    return payload


def output_page(
    output: Any,
    *,
    offset: int,
    limit: int,
) -> dict[str, Any] | None:
    """Return one character-bounded page of a text or JSON result."""
    if output is None:
        return None
    if offset < 0:
        raise ValueError("Output offset must be non-negative")
    if limit < 1 or limit > MAX_OUTPUT_PAGE_CHARS:
        raise ValueError(f"Output limit must be between 1 and {MAX_OUTPUT_PAGE_CHARS}")
    if isinstance(output, str):
        content = output
        content_type = "text"
    else:
        content = json.dumps(output, ensure_ascii=False, sort_keys=True)
        content_type = "json"
    total_chars = len(content)
    end = min(offset + limit, total_chars)
    next_offset = end if end < total_chars else None
    return {
        "content": content[offset:end],
        "content_type": content_type,
        "offset": offset,
        "total_chars": total_chars,
        "truncated": offset > 0 or end < total_chars,
        "next_offset": next_offset,
    }


def model_record_json(
    record: SubagentExecutionRecord,
    *,
    include_output: bool = False,
    output_offset: int = 0,
    output_limit: int = DEFAULT_OUTPUT_PAGE_CHARS,
) -> str:
    """Serialize one bounded model-facing execution projection."""
    return json.dumps(
        model_record_payload(
            record,
            include_output=include_output,
            output_offset=output_offset,
            output_limit=output_limit,
        ),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def _bounded_error(error: str | None) -> str | None:
    if error is None or len(error) <= _MAX_ERROR_CHARS:
        return error
    return error[:_MAX_ERROR_CHARS] + "... [truncated]"
