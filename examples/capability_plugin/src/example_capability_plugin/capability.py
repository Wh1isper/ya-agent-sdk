"""A small host-neutral custom capability distributed through an entry point."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import FunctionToolset


@dataclass
class TextMetricsCapability(AbstractCapability[Any]):
    """Contribute instructions and one deterministic text-analysis tool."""

    max_characters: int = 20_000

    @classmethod
    def get_serialization_name(cls) -> str:
        return "example.text_metrics"

    def get_instructions(self) -> str:
        return "Use text_metrics when exact character, word, or line counts are useful."

    def get_toolset(self) -> FunctionToolset[Any]:
        toolset: FunctionToolset[Any] = FunctionToolset()
        max_characters = self.max_characters

        @toolset.tool_plain
        def text_metrics(text: str) -> dict[str, int]:
            """Count characters, words, and lines in bounded text."""
            bounded = text[:max_characters]
            return {
                "characters": len(bounded),
                "words": len(bounded.split()),
                "lines": len(bounded.splitlines()),
            }

        return toolset
