# Model Configuration

This SDK builds on pydantic-ai. There are two ways to configure models:

1. **Native pydantic-ai model strings** - Direct connection to provider APIs
2. **Gateway mode** - Route requests through a unified gateway

## Quick Start

```python
from ya_agent_sdk.agents.models import infer_model

# Option 1: Native pydantic-ai format (direct provider connection)
model = infer_model("openai-chat:gpt-4o")
model = infer_model("anthropic:claude-3-5-sonnet-20241022")

# Option 2: Gateway format (via gateway proxy)
model = infer_model("mygateway@openai-chat:gpt-4o")
model = infer_model("mygateway@anthropic:claude-3-5-sonnet-20241022")
```

## Native pydantic-ai Models

Model strings without `@` are passed directly to pydantic-ai, supporting all built-in providers.

See official docs: [pydantic-ai Models](https://ai.pydantic.dev/models/)

**Common formats:**

| Provider                   | Format                                                        | Example                                |
| -------------------------- | ------------------------------------------------------------- | -------------------------------------- |
| OpenAI                     | `openai-chat:<model>` / `openai-responses:<model>`            | `openai-responses:gpt-5.6-sol`         |
| OpenAI Responses WebSocket | `openai-responses-ws:<model>` / `openai-responses-rs:<model>` | `openai-responses-ws:gpt-5.5`          |
| Anthropic                  | `anthropic:<model>`                                           | `anthropic:claude-3-5-sonnet-20241022` |
| Google                     | `google:<model>`                                              | `google:gemini-2.5-pro`                |
| Google Cloud               | `google-cloud:<model>`                                        | `google-cloud:gemini-2.5-pro`          |

## Model Request Auto Retry

By default, `ya-agent-sdk` uses Pydantic AI's tenacity-based HTTP retry transport for transient model provider failures. This applies to SDK-created HTTP model clients, gateway model clients, OAuth Codex HTTP fallback, and pre-stream Responses WebSocket connect/create failures.

Retries are intended for transport-level or clearly transient failures only: network/request errors, stream errors before a usable stream is returned, and HTTP statuses `408,409,425,429,500,502,503,504`. `Retry-After` headers are respected for rate limits.

Configure with environment variables:

| Variable                                              | Default                           | Description                                           |
| ----------------------------------------------------- | --------------------------------- | ----------------------------------------------------- |
| `YA_AGENT_MODEL_REQUEST_RETRY_ENABLED`                | `true`                            | Enable retry transport and WebSocket pre-stream retry |
| `YA_AGENT_MODEL_REQUEST_RETRY_ATTEMPTS`               | `5`                               | Total attempts including the first try                |
| `YA_AGENT_MODEL_REQUEST_RETRY_BACKOFF_MULTIPLIER`     | `1`                               | Exponential backoff multiplier                        |
| `YA_AGENT_MODEL_REQUEST_RETRY_MAX_WAIT_SECONDS`       | `30`                              | Max exponential backoff wait                          |
| `YA_AGENT_MODEL_REQUEST_RETRY_AFTER_MAX_WAIT_SECONDS` | `300`                             | Max wait honored from `Retry-After`                   |
| `YA_AGENT_MODEL_REQUEST_RETRY_STATUS_CODES`           | `408,409,425,429,500,502,503,504` | Comma-separated retryable HTTP statuses               |

## OpenAI Responses WebSocket

`ya-agent-sdk` directly includes the generic OpenAI Responses WebSocket transport. The `openai-responses-ws:<model>` and `openai-responses-rs:<model>` aliases prefer WebSocket for streaming calls and fall back to HTTP in `auto` mode when a recoverable error happens before the first stream event.

```python
model = infer_model("openai-responses-ws:gpt-5.5")
```

Transport mode is controlled by `YA_AGENT_OPENAI_RESPONSES_WEBSOCKET_MODE`:

| Value       | Behavior                                                                              |
| ----------- | ------------------------------------------------------------------------------------- |
| `auto`      | Prefer WebSocket and temporarily fall back to HTTP on recoverable pre-stream failures |
| `websocket` | Force WebSocket and surface WebSocket errors                                          |
| `http`      | Use the standard HTTP Responses transport                                             |

The OAuth Codex provider reuses this SDK transport and only supplies Codex-specific bearer/account headers, beta header, and payload normalization.

## OpenAI Responses Presets

OpenAI Responses presets configure reasoning effort, reasoning summaries, storage, max output tokens, optional priority service tier, and GPT-5.6 reasoning mode.

- Existing effort presets use the API's default `standard` reasoning mode.
- `openai_responses_pro_low`, `openai_responses_pro_medium`, `openai_responses_pro_high`, `openai_responses_pro_xhigh`, and `openai_responses_pro_max` pair GPT-5.6 `pro` mode with an explicit effort.
- `openai_responses_pro` is the medium-effort convenience preset.
- `openai_responses_standard` is an alias for `openai_responses_default`.
- `openai_responses_gpt5_6_pro` and `openai_responses_gpt56_pro` are aliases for `openai_responses_pro`.
- `openai_responses_max` uses GPT-5.6 Sol's `max` reasoning effort with detailed reasoning summaries.
- `openai_responses_gpt5_6_sol`, `openai_responses_gpt56_sol`, and `openai_responses_sol` are aliases for `openai_responses_max`.
- `openai_responses_terra` maps to the balanced `openai_responses_medium` preset.
- `openai_responses_luna` maps to the lower-latency `openai_responses_low` preset.
- `openai_responses_max_fast` combines `max` effort with the priority service tier.
- `openai_responses_xhigh` remains available for GPT-5.5 and providers that expose `xhigh` rather than `max`.

Example:

```python
from ya_agent_sdk.agents import create_agent

runtime = create_agent(
    "openai-responses:gpt-5.6",
    model_settings="openai_responses_pro",
)
```

Pro-effort presets send one authoritative, complete `reasoning` object through `extra_body` until Pydantic AI exposes a model setting for `reasoning.mode`. Select the matching pro-effort preset rather than overriding `openai_reasoning_effort` separately. This preserves `mode`, `effort`, and `summary` for both HTTP and WebSocket Responses transports.

Use `gpt5_350k` for subscription-backed Codex access with a 350K context window. Use the other GPT-5 `model_cfg` presets when they match the provider's documented context window. The OpenAI preview announcement describes `ultra` as a product mode that leverages subagents, but does not publish a stable Responses API payload field; configure it with inline `model_settings` only after your provider documents the exact field.

## Google Vertex AI Configuration

Google Cloud Vertex AI requires additional configuration via environment variables. pydantic-ai automatically reads these variables when using the `google-cloud:` prefix.

### Environment Variables

| Variable                         | Description                          | Required                  |
| -------------------------------- | ------------------------------------ | ------------------------- |
| `GOOGLE_API_KEY`                 | API key for Vertex AI authentication | One of API key or ADC     |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to service account JSON file    | For service account auth  |
| `GOOGLE_CLOUD_PROJECT`           | GCP project ID                       | Recommended for Vertex AI |
| `GOOGLE_CLOUD_LOCATION`          | GCP region (default: `us-central1`)  | Optional                  |

### Authentication Methods

**Method 1: API Key**

```bash
export GOOGLE_API_KEY=your-api-key
```

```python
model = infer_model("google-cloud:gemini-2.5-pro")
```

**Method 2: Application Default Credentials (ADC)**

```bash
# Login with gcloud CLI
gcloud auth application-default login

# Set project and location
export GOOGLE_CLOUD_PROJECT=my-project-id
export GOOGLE_CLOUD_LOCATION=us-central1
```

```python
model = infer_model("google-cloud:gemini-2.5-pro")
```

**Method 3: Service Account**

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
export GOOGLE_CLOUD_PROJECT=my-project-id
export GOOGLE_CLOUD_LOCATION=us-central1
```

```python
model = infer_model("google-cloud:gemini-2.5-pro")
```

### Available Regions

Common regions for Vertex AI:

- `us-central1` (default, most model support)
- `us-east1`, `us-east4`, `us-east5`, `us-west1`, `us-west4`
- `europe-west1`, `europe-west2`, `europe-west3`, `europe-west4`
- `asia-east1`, `asia-northeast1`, `asia-southeast1`
- `global` (higher availability, fewer models)

For full list, see [Vertex AI Locations](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations).

### References

- [pydantic-ai Google Models](https://ai.pydantic.dev/models/google/)
- [Vertex AI Authentication](https://cloud.google.com/vertex-ai/docs/authentication)

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
| `openai` / `openai-chat` / `openai-responses` | `gateway@openai-chat:gpt-4o`                   |
| `anthropic`                                   | `gateway@anthropic:claude-3-5-sonnet-20241022` |
| `gemini` / `google-cloud`                     | `gateway@gemini:gemini-1.5-pro`                |
| `bedrock` / `converse`                        | `gateway@bedrock:anthropic.claude-3-sonnet`    |

### Sticky Routing

For session affinity scenarios, pass `extra_headers`:

```python
model = infer_model("mygateway@gemini:gemini-1.5-pro", extra_headers={"x-session-id": "unique-session-id"})
```

**Note**: `extra_headers` only applies to Gateway mode, primarily for providers like `gemini` and `bedrock` that require header injection via http_client.

## Anthropic Adaptive Thinking Presets

When configuring Anthropic models through ya-agent-sdk presets:

- `anthropic` resolves to adaptive thinking by default.
- `anthropic_adaptive_xhigh` is available for Claude Opus 4.7.
- `anthropic_adaptive_high`, `anthropic_adaptive_medium`, and `anthropic_adaptive_low` remain the general-purpose effort presets.
- `anthropic_off` disables thinking.

Claude Opus 4.7 uses adaptive thinking as the primary thinking mode. The `xhigh` effort level is intended for long-horizon coding and agentic workloads.

## Integration with pydantic-ai Agent

```python
from pydantic_ai import Agent
from ya_agent_sdk.agents.models import infer_model

agent = Agent(model=infer_model("mygateway@openai-chat:gpt-4o"), system_prompt="You are a helpful assistant.")
```

## References

- [pydantic-ai Models Documentation](https://ai.pydantic.dev/models/)
- [pydantic-ai Provider Configuration](https://ai.pydantic.dev/models/#model-configuration)
