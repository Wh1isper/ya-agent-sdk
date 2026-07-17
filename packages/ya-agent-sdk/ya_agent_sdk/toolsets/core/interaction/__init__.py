"""Tools for structured interaction with the user."""

from ya_agent_sdk.toolsets.core.base import BaseTool
from ya_agent_sdk.toolsets.core.interaction.user_input import (
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

tools: list[type[BaseTool]] = [AskUserQuestionTool]

__all__ = [
    "ASK_USER_QUESTION_KIND",
    "AskUserQuestionRequest",
    "AskUserQuestionTool",
    "UserQuestion",
    "UserQuestionAnswers",
    "UserQuestionOption",
    "format_user_question_answers",
    "parse_ask_user_question_args",
    "parse_user_question_answer",
    "tools",
]
