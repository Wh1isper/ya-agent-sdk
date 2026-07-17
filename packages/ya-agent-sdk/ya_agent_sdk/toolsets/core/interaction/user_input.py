"""Structured user-input tools.

The tool in this module uses Pydantic AI's deferred-tool control flow so the
application hosting an agent can collect answers interactively or persist the
request and resume it later.
"""

from __future__ import annotations

import json
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_ai import CallDeferred, RunContext

from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.toolsets.core.base import BaseTool

ASK_USER_QUESTION_KIND = "ask_user_question"


class UserQuestionOption(BaseModel):
    """One selectable answer to a clarifying question."""

    label: Annotated[str, Field(min_length=1, description="Short option label returned as the answer.")]
    description: Annotated[str, Field(min_length=1, description="Explanation of this option.")]


class UserQuestion(BaseModel):
    """A structured clarifying question for the user."""

    model_config = ConfigDict(populate_by_name=True)

    question: Annotated[str, Field(min_length=1, description="Full question text shown to the user.")]
    header: Annotated[str, Field(min_length=1, max_length=12, description="Short question label.")]
    options: Annotated[
        list[UserQuestionOption],
        Field(min_length=2, max_length=4, description="Two to four suggested answers."),
    ]
    multi_select: Annotated[
        bool,
        Field(alias="multiSelect", description="Whether the user may select more than one option."),
    ] = False


class AskUserQuestionRequest(BaseModel):
    """Validated input for an ``ask_user_question`` call."""

    questions: Annotated[
        list[UserQuestion],
        Field(min_length=1, max_length=4, description="One to four clarifying questions."),
    ]

    @model_validator(mode="after")
    def questions_must_be_unique(self) -> AskUserQuestionRequest:
        texts = [item.question for item in self.questions]
        if len(texts) != len(set(texts)):
            raise ValueError("question text must be unique because answers are keyed by question")
        return self


def _normalize_user_question_answer(question: str, answer: str | list[str]) -> str | list[str]:
    if isinstance(answer, str):
        normalized_answer = answer.strip()
        if not normalized_answer:
            raise ValueError(f"answer for {question!r} must not be empty")
        return normalized_answer

    if not isinstance(answer, list):
        raise TypeError(f"answer for {question!r} must be a string or list of strings")
    if not answer:
        raise ValueError(f"answer list for {question!r} must not be empty")

    normalized_items: list[str] = []
    for item in answer:
        if not isinstance(item, str):
            raise TypeError(f"answer list for {question!r} must contain only strings")
        normalized_item = item.strip()
        if not normalized_item:
            raise ValueError(f"answer list for {question!r} must not contain empty items")
        normalized_items.append(normalized_item)
    return normalized_items


class UserQuestionAnswers(BaseModel):
    """Answers supplied by a user-input host."""

    questions: list[UserQuestion]
    answers: dict[str, str | list[str]] = Field(default_factory=dict)
    response: str | None = None

    @model_validator(mode="after")
    def answers_must_cover_questions(self) -> UserQuestionAnswers:
        has_general_response = False
        if self.response is not None:
            if not isinstance(self.response, str):
                raise TypeError("response must be a string")
            normalized_response = self.response.strip()
            if normalized_response:
                self.response = normalized_response
                has_general_response = True
            else:
                self.response = None

        expected = {item.question for item in self.questions}
        actual = set(self.answers)
        missing = expected.difference(actual)
        extra = actual.difference(expected)
        if extra or (missing and not has_general_response):
            details = []
            if missing and not has_general_response:
                details.append(f"missing answers for: {', '.join(sorted(missing))}")
            if extra:
                details.append(f"unexpected answers for: {', '.join(sorted(extra))}")
            raise ValueError("; ".join(details))

        self.answers = {
            question: _normalize_user_question_answer(question, answer) for question, answer in self.answers.items()
        }
        return self


def parse_user_question_answer(question: UserQuestion, value: str) -> str | list[str]:
    """Map terminal-style numeric input to option labels, preserving free text."""
    normalized = value.strip()
    if not normalized:
        raise ValueError("an answer is required")

    parts = [part.strip() for part in normalized.split(",")]
    try:
        indices = [int(part) - 1 for part in parts]
    except ValueError:
        return normalized

    if not all(0 <= index < len(question.options) for index in indices):
        return normalized

    labels = [question.options[index].label for index in indices]
    if question.multi_select:
        return labels
    if len(labels) != 1:
        raise ValueError("select exactly one option")
    return labels[0]


def format_user_question_answers(answers: UserQuestionAnswers) -> dict[str, object]:
    """Build the structured deferred tool result consumed by the model."""
    return answers.model_dump(mode="json", by_alias=True, exclude_none=True)


def parse_ask_user_question_args(args: object) -> AskUserQuestionRequest:
    """Parse a Pydantic AI tool-call argument payload."""
    if isinstance(args, str):
        return AskUserQuestionRequest.model_validate_json(args)
    if isinstance(args, dict):
        return AskUserQuestionRequest.model_validate(args)
    raise TypeError(f"Unsupported ask_user_question arguments: {type(args).__name__}")


class AskUserQuestionTool(BaseTool):
    """Ask the host application to collect structured clarifying answers."""

    name = "ask_user_question"
    description = (
        "Ask the user one to four clarifying questions when their answers materially affect the result. "
        "Each question provides two to four options and may allow multiple selections. "
        "The host pauses execution and returns the user's selections or free-text answers."
    )

    def is_available(self, ctx: RunContext[AgentContext]) -> bool:
        """Expose host-facing interaction only to the main agent."""
        return ctx.deps.agent_id == "main"

    async def call(
        self,
        ctx: RunContext[AgentContext],
        questions: Annotated[
            list[UserQuestion],
            Field(min_length=1, max_length=4, description="One to four clarifying questions."),
        ],
    ) -> None:
        request = AskUserQuestionRequest(questions=questions)
        raise CallDeferred(
            metadata={
                "kind": ASK_USER_QUESTION_KIND,
                "questions": json.loads(request.model_dump_json(by_alias=True))["questions"],
            }
        )
