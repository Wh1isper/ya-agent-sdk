# ya-agent-stream-protocol

Shared stream protocol adapters between `ya-agent-sdk` and applications.

The Python import package is `ya_agent_stream_protocol`.

The package owns the shared protocol boundary between `ya-agent-sdk` and applications:

- AGUI event adaptation
- compact replay buffers
- message artifact validation
- SSE framing helpers

YAACLI, YA Claw, and future applications configure their own event namespaces through
`AguiAdapterConfig` instead of embedding application-specific names in this package.

SDK lifecycle projections use canonical snake-case names such as
`ya_agent.subagent_start`, `ya_agent.subagent_complete`, and
`ya_agent.model_request_complete`. The adapter is also a fail-closed public-data
boundary: only explicitly allowlisted lifecycle DTOs are emitted, and unknown or future
SDK events are dropped rather than generically serialized. Subagent projections omit
parent logical-run IDs, native enqueue projections expose only a message count,
model-request projections omit internal event/run IDs, file changes omit replacement
content, and execution start/resume events omit prompt and deferred-result content. Full
injected messages, durable input/enqueue identities, idempotency keys, raw lifecycle
event IDs/timestamps, resumable state, and other host-owned data must remain outside
AGUI replay and SSE.
