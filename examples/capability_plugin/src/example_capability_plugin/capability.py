"""A small host-neutral custom capability distributed through an entry point."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import FunctionToolset


class TextMetricsResult(TypedDict):
    characters: int
    words: int
    lines: int
    max_characters: int
    truncated: bool


@dataclass
class TextMetricsCapability(AbstractCapability[Any]):
    """Contribute instructions and one deterministic text-analysis tool."""

    max_characters: int = 20_000

    def __post_init__(self) -> None:
        if self.max_characters < 1:
            raise ValueError("max_characters must be positive")

    @classmethod
    def get_serialization_name(cls) -> str:
        return "example.text_metrics"

    def get_instructions(self) -> str:
        return "Use text_metrics for character, word, or line counts; it reports when input was truncated."

    def get_toolset(self) -> FunctionToolset[Any]:
        toolset: FunctionToolset[Any] = FunctionToolset()
        max_characters = self.max_characters

        @toolset.tool_plain
        def text_metrics(text: str) -> TextMetricsResult:
            """Count characters, words, and lines in bounded text."""
            bounded = text[:max_characters]
            return {
                "characters": len(bounded),
                "words": len(bounded.split()),
                "lines": len(bounded.splitlines()),
                "max_characters": max_characters,
                "truncated": len(text) > max_characters,
            }

        return toolset
