# 03. Tool Protocol

## Goal

The computer use tool protocol should give agents a stable, provider-neutral surface while preserving rich provider metadata for Desktop UX, run trace, and debugging.

The protocol has two levels:

- `computer_see` creates a snapshot of the current desktop state.
- `computer_act` executes typed actions against a snapshot element, app/window target, or coordinate.

Additional convenience tools can wrap common action types, but `see + act` should be the canonical provider contract.

## Canonical Tools

```text
computer_see
computer_act
computer_wait
computer_status
```

Convenience tools:

```text
computer_click
computer_type
computer_hotkey
computer_scroll
computer_drag
computer_focus_app
computer_select_menu
computer_close_window
```

Convenience tools compile into `computer_act` internally.

## Snapshot Request

```ts
type ComputerSeeRequest = {
  target?: ComputerTarget;
  include_screenshot?: boolean;
  include_accessibility_tree?: boolean;
  include_text?: boolean;
  include_cursor?: boolean;
  max_depth?: number;
  max_nodes?: number;
  redaction_policy?: "default" | "strict" | "off";
};
```

Targets:

```ts
type ComputerTarget =
  | { kind: "display"; display_id?: string }
  | { kind: "app"; app_name?: string; bundle_id?: string }
  | { kind: "window"; window_id?: string; title?: string; app_name?: string }
  | { kind: "region"; rect: Rect };
```

## Snapshot Result

```ts
type ComputerSnapshot = {
  snapshot_id: string;
  provider_id: string;
  created_at: string;
  target: ComputerTarget;
  screenshot?: ScreenshotArtifact;
  accessibility_tree?: UIElementNode[];
  text_blocks?: TextBlock[];
  cursor?: CursorState;
  active_app?: AppSummary;
  active_window?: WindowSummary;
  warnings: string[];
};
```

The snapshot should be compact enough for model context. Large screenshots and full trees should live as artifacts, with a short summary and references in the tool result.

## Element References

```ts
type ElementRef = {
  snapshot_id: string;
  element_id: string;
};
```

Element references are valid within a snapshot. Providers may attempt recovery against newer UI trees when the exact element disappeared.

## Action Request

```ts
type ComputerAction = {
  action_id?: string;
  kind: ComputerActionKind;
  target?: ComputerActionTarget;
  input?: ComputerActionInput;
  options?: ComputerActionOptions;
};

type ComputerActionKind =
  | "click"
  | "double_click"
  | "right_click"
  | "drag"
  | "scroll"
  | "type_text"
  | "press_key"
  | "hotkey"
  | "focus_app"
  | "focus_window"
  | "select_menu"
  | "set_value"
  | "wait"
  | "open_app"
  | "close_window";
```

Targets:

```ts
type ComputerActionTarget =
  | { kind: "element"; ref: ElementRef }
  | { kind: "coordinate"; x: number; y: number; display_id?: string }
  | { kind: "app"; app_name?: string; bundle_id?: string }
  | { kind: "window"; window_id?: string; title?: string; app_name?: string }
  | { kind: "menu_item"; app_name?: string; path: string[] };
```

Inputs:

```ts
type ComputerActionInput = {
  text?: string;
  keys?: string[];
  delta_x?: number;
  delta_y?: number;
  drag_to?: { x: number; y: number; display_id?: string };
  seconds?: number;
  value?: string;
};
```

Options:

```ts
type ComputerActionOptions = {
  require_fresh_snapshot?: boolean;
  preferred_strategy?: "semantic" | "coordinate" | "auto";
  wait_after_ms?: number;
  human_like?: boolean;
  dry_run?: boolean;
};
```

## Action Result

```ts
type ComputerActionResult = {
  action_id: string;
  status: "succeeded" | "failed" | "blocked" | "requires_approval";
  provider_id: string;
  started_at: string;
  completed_at?: string;
  execution?: NativeActionExecution;
  before_snapshot_id?: string;
  after_snapshot?: ComputerSnapshot;
  artifacts: ComputerArtifactRef[];
  error?: ComputerActionError;
  policy?: PolicyDecision;
};
```

Errors should be typed:

```ts
type ComputerActionError = {
  code:
    | "permission_missing"
    | "target_not_found"
    | "target_stale"
    | "app_unavailable"
    | "policy_blocked"
    | "user_paused"
    | "provider_error";
  message: string;
  recoverable: boolean;
  details?: Record<string, unknown>;
};
```

## Model-Facing Text

Tool results should include concise model-readable text. Example:

```text
Snapshot snap_123 captured Safari window "GitHub". Screenshot artifact art_456 is 1280x832 at scale 2. Accessibility tree includes 83 nodes. Key elements: B1 "Sign in" button, T1 search field, L1 repository link.
```

Action result example:

```text
Clicked B1 "Sign in" using accessibility press. A new snapshot snap_124 shows the login form with T2 "Username or email address" focused.
```

## Trace Projection

Run trace should project computer calls into a stable compact shape:

```ts
type ComputerTraceEntry = {
  tool_call_id: string;
  run_id: string;
  provider_id: string;
  kind: "see" | "act" | "wait" | "status";
  target_summary?: string;
  action_summary?: string;
  status: "succeeded" | "failed" | "blocked" | "requires_approval";
  snapshot_id?: string;
  screenshot_artifact_id?: string;
  app_name?: string;
  window_title?: string;
  policy_decision?: string;
  created_at: string;
};
```

## Artifact Types

```ts
type ComputerArtifactRef = {
  artifact_id: string;
  kind:
    | "screenshot"
    | "accessibility_tree"
    | "text_snapshot"
    | "action_record"
    | "redaction_report";
  mime_type: string;
  uri: string;
  metadata: Record<string, unknown>;
};
```

Artifacts should be stored through the Claw run-store when an action is part of a run. Local provider temporary files should be cleaned after the artifact is transferred or indexed.

## Approval Classification

The provider can classify actions before execution:

```ts
type ComputerActionRisk =
  | "low"
  | "medium"
  | "high"
  | "critical";

type ComputerActionCategory =
  | "screen_read"
  | "click"
  | "text_entry"
  | "keyboard_shortcut"
  | "file_dialog"
  | "app_launch"
  | "system_settings"
  | "credential_field"
  | "destructive_action"
  | "external_communication";
```

Claw profile policy and Desktop local policy both participate in approval decisions. Desktop policy can apply a device-level block before the request reaches OS APIs.
