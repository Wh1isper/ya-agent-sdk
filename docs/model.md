# Model Configuration

This SDK builds on pydantic-ai. There are two ways to configure models:

1. **Native pydantic-ai model strings** - Direct connection to provider APIs
2. **Gateway mode** - Route requests through a unified gateway

## Quick Start

```python
from pai_agent_sdk.agents.models import infer_model

# Option 1: Native pydantic-ai format (direct provider connection)
model = infer_model("openai:gpt-4o")
model = infer_model("anthropic:claude-3-5-sonnet-20241022")

# Option 2: Gateway format (via gateway proxy)
model = infer_model("mygateway@openai:gpt-4o")
model = infer_model("mygateway@anthropic:claude-3-5-sonnet-20241022")
```

## Native pydantic-ai Models

Model strings without `@` are passed directly to pydantic-ai, supporting all built-in providers.

See official docs: [pydantic-ai Models](https://ai.pydantic.dev/models/)

**Common formats:**

| Provider  | Format              | Example                                |
| --------- | ------------------- | -------------------------------------- |
| OpenAI    | `openai:<model>`    | `openai:gpt-4o`                        |
| Anthropic | `anthropic:<model>` | `anthropic:claude-3-5-sonnet-20241022` |
| Google    | `gemini:<model>`    | `gemini:gemini-1.5-pro`                |
| Groq      | `groq:<model>`      | `groq:llama-3.1-70b-versatile`         |

## Gateway Mode

Use `gateway_name@provider:model` format to route requests through a unified gateway. Useful for:

- Centralized API key management across multiple providers
- Internal proxy/load balancing scenarios
- Sticky routing requirements

### Environment Variables

Gateway mode requires two environment variables (using `mygateway` as example):

```bash
# API Key (required)
MYGATEWAY_API_KEY=your-api-key

# Gateway Base URL (required)
MYGATEWAY_BASE_URL=https://your-gateway.example.com/v1
```

Naming convention: `{GATEWAY_NAME}_API_KEY` and `{GATEWAY_NAME}_BASE_URL`

### Supported Providers

| Provider Name                                 | Model String Format                            |
| --------------------------------------------- | ---------------------------------------------- |
| `openai` / `openai-chat` / `openai-responses` | `gateway@openai:gpt-4o`                        |
| `anthropic`                                   | `gateway@anthropic:claude-3-5-sonnet-20241022` |
| `gemini` / `google-vertex`                    | `gateway@gemini:gemini-1.5-pro`                |
| `groq`                                        | `gateway@groq:llama-3.1-70b-versatile`         |
| `bedrock` / `converse`                        | `gateway@bedrock:anthropic.claude-3-sonnet`    |

### Sticky Routing

For session affinity scenarios, pass `extra_headers`:

```python
model = infer_model(
    "mygateway@gemini:gemini-1.5-pro",
    extra_headers={"x-session-id": "unique-session-id"}
)
```

**Note**: `extra_headers` only applies to Gateway mode, primarily for providers like `gemini` and `bedrock` that require header injection via http_client.

## Integration with pydantic-ai Agent

```python
from pydantic_ai import Agent
from pai_agent_sdk.agents.models import infer_model

agent = Agent(
    model=infer_model("mygateway@openai:gpt-4o"),
    system_prompt="You are a helpful assistant."
)
```

## References

- [pydantic-ai Models Documentation](https://ai.pydantic.dev/models/)
- [pydantic-ai Provider Configuration](https://ai.pydantic.dev/models/#model-configuration)
