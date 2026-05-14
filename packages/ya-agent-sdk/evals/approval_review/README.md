# Approval Review Eval

This eval checks the generic `approval_review` policy surface across built-in tool styles and MCP-style calls. It covers bounded reads, bounded writes, broad destructive operations, local shell execution, network execution, credential reads, MCP reads, and MCP destructive mutations.

Run the deterministic eval from the repository root:

```bash
uv run python packages/ya-agent-sdk/evals/approval_review/run_eval.py
```

The deterministic reviewer is local and repeatable, so it is suitable for CI and regression checks.

Run a live model-backed eval by passing a reviewer model:

```bash
uv run python packages/ya-agent-sdk/evals/approval_review/run_eval.py \
  --model gateway@openai-responses:gpt-5.4-mini
```

Results are written as JSONL to `approval_review_eval_results.jsonl` by default. Each row records the expected outcome, actual outcome, risk level, rationale, categories, scopes, and pass/fail status.
