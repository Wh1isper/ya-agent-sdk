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
