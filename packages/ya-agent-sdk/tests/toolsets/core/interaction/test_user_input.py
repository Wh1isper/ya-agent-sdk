"""Tests for the structured user-input tool."""

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError
from pydantic_ai import CallDeferred, RunContext
from ya_agent_sdk.context import AgentContext
from ya_agent_sdk.toolsets.core.interaction import (
    ASK_USER_QUESTION_KIND,
    AskUserQuestionRequest,
    AskUserQuestionTool,
    UserQuestion,
    UserQuestionAnswers,
    UserQuestionOption,
    format_user_question_answers,
    parse_ask_user_question_args,
    parse_user_question_answer,
)


def _question(*, multi_select: bool = False) -> UserQuestion:
    return UserQuestion(
        question="Which output format should I use?",
        header="Format",
        options=[
            UserQuestionOption(label="Summary", description="Brief overview"),
            UserQuestionOption(label="Detailed", description="Full explanation"),
        ],
        multiSelect=multi_select,
    )


def test_question_schema_accepts_claude_style_multi_select_alias() -> None:
    question = _question(multi_select=True)

    assert question.multi_select is True
    assert question.model_dump(by_alias=True)["multiSelect"] is True


def test_question_schema_rejects_invalid_option_count() -> None:
    with pytest.raises(ValidationError):
        UserQuestion(
            question="Choose one",
            header="Choice",
            options=[UserQuestionOption(label="Only", description="Only option")],
        )


def test_request_rejects_duplicate_question_text() -> None:
    question = _question()

    with pytest.raises(ValidationError, match="question text must be unique"):
        AskUserQuestionRequest(questions=[question, question])


def test_parse_user_question_answer_maps_numeric_choices() -> None:
    assert parse_user_question_answer(_question(), "2") == "Detailed"
    assert parse_user_question_answer(_question(multi_select=True), "1, 2") == ["Summary", "Detailed"]


def test_parse_user_question_answer_preserves_free_text() -> None:
    assert parse_user_question_answer(_question(), "Use Markdown") == "Use Markdown"


def test_parse_user_question_answer_rejects_multiple_single_select_choices() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        parse_user_question_answer(_question(), "1,2")


def test_answers_require_every_question_without_general_response() -> None:
    with pytest.raises(ValidationError, match="missing answers"):
        UserQuestionAnswers(questions=[_question()])


def test_blank_general_response_does_not_replace_question_answers() -> None:
    with pytest.raises(ValidationError, match="missing answers"):
        UserQuestionAnswers(questions=[_question()], response="  \n")


def test_general_response_is_stripped_and_replaces_question_answers() -> None:
    value = UserQuestionAnswers(questions=[_question()], response="  Use either format.  ")

    assert value.response == "Use either format."
    assert value.answers == {}


def test_general_response_does_not_allow_blank_supplied_answers() -> None:
    question = _question()

    with pytest.raises(ValidationError, match="must not be empty"):
        UserQuestionAnswers(
            questions=[question],
            answers={question.question: "  "},
            response="Use either format.",
        )


@pytest.mark.parametrize("response", [None, "Use either format."])
def test_answers_reject_unexpected_question_keys(response: str | None) -> None:
    question = _question()

    with pytest.raises(ValidationError, match="unexpected answers"):
        UserQuestionAnswers(
            questions=[question],
            answers={question.question: "Summary", "Unasked question": "Other"},
            response=response,
        )


def test_answers_reject_blank_string_values() -> None:
    question = _question()

    with pytest.raises(ValidationError, match="must not be empty"):
        UserQuestionAnswers(questions=[question], answers={question.question: " \t "})


@pytest.mark.parametrize("answer", [[], ["Summary", "  "]])
def test_answers_reject_empty_list_values_or_items(answer: list[str]) -> None:
    question = _question()

    with pytest.raises(ValidationError, match=r"must not be empty|must not contain empty items"):
        UserQuestionAnswers(questions=[question], answers={question.question: answer})


def test_list_answers_are_stripped() -> None:
    question = _question(multi_select=True)
    value = UserQuestionAnswers(
        questions=[question],
        answers={question.question: [" Summary ", "\tDetailed\n"]},
        response="   ",
    )

    assert value.response is None
    assert value.answers == {question.question: ["Summary", "Detailed"]}


def test_answer_format_round_trips() -> None:
    question = _question()
    value = UserQuestionAnswers(
        questions=[question],
        answers={question.question: " Summary "},
    )

    serialized = format_user_question_answers(value)
    parsed = parse_ask_user_question_args({"questions": [question.model_dump(by_alias=True)]})

    assert serialized["answers"] == {"Which output format should I use?": "Summary"}
    assert parsed.questions == [question]


def test_ask_user_question_tool_is_available_only_to_main_agent() -> None:
    main_ctx = AgentContext()
    subagent_ctx = main_ctx.create_subagent_context("worker", agent_id="worker-1")
    run_ctx = MagicMock(spec=RunContext)
    tool = AskUserQuestionTool()

    run_ctx.deps = main_ctx
    assert tool.is_available(run_ctx) is True

    run_ctx.deps = subagent_ctx
    assert tool.is_available(run_ctx) is False


@pytest.mark.asyncio
async def test_ask_user_question_tool_defers_with_structured_metadata() -> None:
    run_ctx = MagicMock(spec=RunContext)
    run_ctx.deps = AgentContext()

    with pytest.raises(CallDeferred) as exc_info:
        await AskUserQuestionTool().call(run_ctx, [_question()])

    assert exc_info.value.metadata is not None
    assert exc_info.value.metadata["kind"] == ASK_USER_QUESTION_KIND
    assert exc_info.value.metadata["questions"][0]["multiSelect"] is False
