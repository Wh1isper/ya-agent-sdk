# Structured User Input

`ask_user_question` lets an agent pause for one to four structured clarifying questions. It uses Pydantic AI deferred-tool control flow, so the host remains responsible for presenting the questions, collecting answers, and resuming the run.

## Opt-In Contract

The SDK does not register this tool by default. Only enable it in a host that supports `DeferredToolRequests` and can return matching `DeferredToolResults`.

```python
from pydantic_ai import DeferredToolRequests
from ya_agent_sdk.agents.main import create_agent
from ya_agent_sdk.capabilities import (
    RuntimeFoundationCapability,
    UserInteractionCapability,
)

runtime = create_agent(
    "anthropic:claude-sonnet-4",
    capabilities=[
        RuntimeFoundationCapability(),
        UserInteractionCapability(),
    ],
    output_type=[str, DeferredToolRequests],
)
```

`ask_user_question` carries `main_agent_only` metadata. The final `ToolVisibilityCapability` rejects it in child execution contexts, and the tool's own availability check additionally requires a root context (`agent_id="main"` with no `parent_run_id`). Do not grant `UserInteractionCapability` to a child unless that host explicitly implements a nested deferred-continuation protocol.

## Question Schema

Each tool call contains `questions` with these limits:

- one to four questions per call;
- a non-empty `question` used as the answer-map key;
- a `header` of at most 12 characters;
- two to four options, each with a `label` and `description`;
- `multiSelect: false` for one answer or `multiSelect: true` for multiple answers; and
- unique question text within a call.

A host may accept option labels, numeric selections, or free text. `parse_user_question_answer()` converts terminal-style input such as `"2"` or `"1, 3"` into option labels while preserving non-numeric text.

## Deferred Host Flow

The tool raises `CallDeferred` with metadata shaped like:

```json
{
  "kind": "ask_user_question",
  "questions": [
    {
      "question": "Which output format should I use?",
      "header": "Format",
      "options": [
        {"label": "Summary", "description": "Brief overview"},
        {"label": "Detailed", "description": "Full explanation"}
      ],
      "multiSelect": false
    }
  ]
}
```

Pydantic AI returns the pending call in `DeferredToolRequests.calls`. The metadata is available under `DeferredToolRequests.metadata[tool_call_id]`. Resume with a result under the same tool-call ID:

```python
from pydantic_ai import DeferredToolRequests
from ya_agent_sdk.toolsets.core.interaction import (
    AskUserQuestionTool,
    UserQuestionAnswers,
    format_user_question_answers,
    parse_ask_user_question_args,
    parse_user_question_answer,
)

async with runtime:
    result = await runtime.agent.run(
        "Prepare a report, asking me about material choices first.",
        deps=runtime.ctx,
    )

    while isinstance(result.output, DeferredToolRequests):
        call_results: dict[str, object] = {}
        for call in result.output.calls:
            if call.tool_name != AskUserQuestionTool.name:
                raise RuntimeError(f"Unsupported deferred tool: {call.tool_name}")

            request = parse_ask_user_question_args(call.args)
            answers = {
                question.question: parse_user_question_answer(
                    question,
                    input(f"{question.question}: "),
                )
                for question in request.questions
            }
            call_results[call.tool_call_id] = format_user_question_answers(
                UserQuestionAnswers(questions=request.questions, answers=answers)
            )

        deferred_results = result.output.build_results(calls=call_results)
        result = await runtime.agent.run(
            deps=runtime.ctx,
            message_history=result.all_messages(),
            deferred_tool_results=deferred_results,
        )

    print(result.output)
```

A structured result contains the original questions plus an `answers` mapping keyed by exact question text. Every string or list item is stripped and must remain non-empty; lists must contain at least one item; unknown question keys are rejected; and, without a general response, every question key is required. A host that cannot collect per-question answers may instead provide a non-blank `UserQuestionAnswers.response` as a general response.

## Host Requirements

A compatible host must:

1. configure the agent output type to include `DeferredToolRequests`;
2. preserve the pending run's message history;
3. collect every requested answer without silently choosing defaults;
4. put each structured response in `DeferredToolResults.calls` under the original tool-call ID;
5. resume until the model returns final output or another deferred batch; and
6. define cancellation, persistence, timeout, and cumulative request-limit behavior.

Do not expose the tool in headless or remote runtimes unless they implement an equivalent deferred continuation protocol.

## API Reference

| API                              | Purpose                                                |
| -------------------------------- | ------------------------------------------------------ |
| `AskUserQuestionTool`            | Optional deferred tool named `ask_user_question`       |
| `UserQuestion`                   | Validated question and option schema                   |
| `UserQuestionOption`             | One suggested option                                   |
| `AskUserQuestionRequest`         | Validates a complete tool-call payload                 |
| `UserQuestionAnswers`            | Validates structured host answers                      |
| `parse_ask_user_question_args()` | Parses dict or JSON-string tool arguments              |
| `parse_user_question_answer()`   | Converts numeric terminal input or preserves free text |
| `format_user_question_answers()` | Produces a JSON-compatible deferred result             |
| `ASK_USER_QUESTION_KIND`         | Metadata discriminator, `ask_user_question`            |
