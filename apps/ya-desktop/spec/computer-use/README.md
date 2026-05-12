# Computer Use Spec

This folder defines YA Desktop's first-party Host Computer Use system. The design takes inspiration from existing macOS automation products, but the runtime, protocol, permission model, and product integration are YA-owned.

Host Computer Use lets an agent see and operate the user's desktop under explicit user control. The feature belongs in YA Desktop because the desktop app can own OS permissions, trusted local process lifecycle, native notifications, pause/takeover controls, and high-trust UX. Claw remains the runtime authority for sessions, runs, profiles, tool calls, approvals, trace, and artifact persistence.

## Section Map

| Section | Document                                                                 | Topic                                                                            |
| ------- | ------------------------------------------------------------------------ | -------------------------------------------------------------------------------- |
| 01      | [01-product-and-boundary.md](01-product-and-boundary.md)                 | product goal, supported modes, Claw/Desktop boundary                             |
| 02      | [02-native-provider-architecture.md](02-native-provider-architecture.md) | Desktop host computer provider, bridge process, macOS implementation layers      |
| 03      | [03-tool-protocol.md](03-tool-protocol.md)                               | provider-neutral computer tool schema, snapshots, actions, results               |
| 04      | [04-permissions-and-safety.md](04-permissions-and-safety.md)             | macOS permissions, trust model, HITL policy, takeover semantics                  |
| 05      | [05-claw-runtime-integration.md](05-claw-runtime-integration.md)         | profiles, capabilities, tool proxy, run trace, artifacts, remote RPC mode        |
| 06      | [06-desktop-ux.md](06-desktop-ux.md)                                     | Spaces, Chats, Inbox, Settings, live monitor, timeline rendering                 |
| 07      | [07-implementation-plan.md](07-implementation-plan.md)                   | milestones, package layout, test strategy, open decisions                        |
| 08      | [08-relay-based-computer-use.md](08-relay-based-computer-use.md)         | Computer Use as a Desktop Environment Relay capability over ya-environment-relay |

## Design Principles

- Desktop is the permission host for real user devices.
- Claw is the execution and trace authority.
- The computer provider API is provider-neutral and supports macOS host, future Windows host, Linux host, and sandboxed desktop backends.
- The first provider targets macOS through native Rust/Swift system APIs inside YA Desktop.
- Semantic accessibility actions are preferred before coordinate actions.
- Screenshots, UI snapshots, actions, and approvals are first-class run artifacts.
- User control is always visible through pause, takeover, release, and stop actions.
- Remote Claw can request host computer actions through a Desktop-controlled RPC bridge after explicit user enablement.
