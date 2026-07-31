"""Public configuration for CodeAct execution."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class CodeActConfig:
    """Configure the restricted Python CodeAct capability.

    CodeAct is opt-in at agent construction. Resource limits are enforced at
    their documented boundaries. ``timeout_seconds`` is an execution deadline:
    reaching it requests cancellation, while active in-process host calls are
    still drained before execution ownership is released.
    """

    inline: bool = True
    programs: bool = True
    max_source_bytes: int = 256 * 1024
    max_output_bytes: int = 10 * 1024 * 1024
    max_tool_calls: int = 128
    max_concurrency: int = 16
    timeout_seconds: float = 300.0
    """Deadline before CodeAct requests cancellation.

    Completion may occur later because active in-process tools are drained
    before CodeAct releases their host ownership.
    """
    max_memory_bytes: int = 100 * 1024 * 1024
    max_recursion_depth: int = 1000
    trace_preview_bytes: int = 4096

    def __post_init__(self) -> None:
        if not self.inline and not self.programs:
            raise ValueError("CodeActConfig must enable inline execution, programs, or both")
        for field_name in (
            "max_source_bytes",
            "max_output_bytes",
            "max_tool_calls",
            "max_concurrency",
            "max_memory_bytes",
            "max_recursion_depth",
            "trace_preview_bytes",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be greater than zero")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite")
