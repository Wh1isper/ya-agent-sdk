# ya-environment-relay

Provider-neutral environment protocol package for YA agent environments.

`ya-environment-relay` currently hosts the YA Environment Protocol spec and future Python adapters. The protocol defines how external execution capabilities are exposed to an agent runtime as Environment components.

Capability families:

- file operations
- stateless shell execution
- background processes
- stateful shell sessions
- custom tools
- resources
- artifacts
- computer use

The protocol string is `ya-environment-protocol.v1`.

`ya-envd` is the official daemon implementation target. It is expected to be a Rust binary that can speak the protocol over stdio, local sockets, named pipes, WebSocket, or future transports.

See [`spec/`](spec/) for the protocol documents.
