
CREATE TABLE IF NOT EXISTS schema_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    workspace_ref TEXT NOT NULL,
    status TEXT NOT NULL,
    head_revision_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    tombstoned_at TEXT
);
CREATE INDEX IF NOT EXISTS sessions_updated_idx ON sessions(updated_at DESC);

CREATE TABLE IF NOT EXISTS logical_runs (
    logical_run_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    execution_id TEXT NOT NULL UNIQUE,
    expected_head_revision_id TEXT,
    model TEXT,
    model_profile_id TEXT,
    idempotency_key TEXT NOT NULL,
    status TEXT NOT NULL,
    input_open INTEGER NOT NULL DEFAULT 1,
    cancellation_reason TEXT,
    pending_action_batch_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(session_id, idempotency_key)
);
CREATE INDEX IF NOT EXISTS logical_runs_session_idx
    ON logical_runs(session_id, created_at DESC);

CREATE TABLE IF NOT EXISTS executions (
    execution_id TEXT PRIMARY KEY,
    logical_run_id TEXT NOT NULL UNIQUE REFERENCES logical_runs(logical_run_id),
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS execution_checkpoints (
    execution_id TEXT PRIMARY KEY REFERENCES executions(execution_id),
    logical_run_id TEXT NOT NULL UNIQUE REFERENCES logical_runs(logical_run_id),
    segment_index INTEGER NOT NULL CHECK(segment_index >= 0),
    segment_status TEXT NOT NULL CHECK(segment_status IN ('completed', 'suspended')),
    payload_json TEXT NOT NULL,
    deferred_requests_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_inputs (
    input_id TEXT PRIMARY KEY,
    logical_run_id TEXT NOT NULL REFERENCES logical_runs(logical_run_id),
    order_index INTEGER NOT NULL,
    idempotency_key TEXT NOT NULL,
    origin TEXT NOT NULL,
    priority TEXT NOT NULL,
    content_json TEXT NOT NULL,
    state TEXT NOT NULL,
    native_enqueue_id TEXT,
    rejection_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(logical_run_id, order_index),
    UNIQUE(logical_run_id, idempotency_key)
);
CREATE INDEX IF NOT EXISTS run_inputs_drain_idx
    ON run_inputs(logical_run_id, state, priority, order_index);

CREATE TABLE IF NOT EXISTS revisions (
    revision_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    logical_run_id TEXT NOT NULL REFERENCES logical_runs(logical_run_id),
    commit_kind TEXT NOT NULL,
    parent_revision_id TEXT,
    message_history_json TEXT NOT NULL,
    resumable_state_json TEXT NOT NULL,
    input_ledger_json TEXT NOT NULL,
    display_projection_json TEXT NOT NULL,
    usage_json TEXT NOT NULL,
    terminal_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(logical_run_id, commit_kind)
);
CREATE INDEX IF NOT EXISTS revisions_session_idx
    ON revisions(session_id, created_at DESC);

CREATE TABLE IF NOT EXISTS session_events (
    event_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    logical_run_id TEXT,
    sequence INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(session_id, sequence)
);
CREATE INDEX IF NOT EXISTS session_events_read_idx
    ON session_events(session_id, sequence);

CREATE TABLE IF NOT EXISTS action_batches (
    batch_id TEXT PRIMARY KEY,
    logical_run_id TEXT NOT NULL REFERENCES logical_runs(logical_run_id),
    state TEXT NOT NULL,
    deadline_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS action_items (
    action_item_id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL REFERENCES action_batches(batch_id),
    tool_call_id TEXT NOT NULL,
    decision_kind TEXT NOT NULL,
    request_json TEXT NOT NULL,
    state TEXT NOT NULL,
    decision_id TEXT UNIQUE,
    decision_json TEXT,
    actor TEXT,
    created_at TEXT NOT NULL,
    decided_at TEXT,
    consumed_at TEXT,
    UNIQUE(batch_id, tool_call_id)
);
CREATE INDEX IF NOT EXISTS action_items_batch_idx
    ON action_items(batch_id, created_at, action_item_id);

CREATE TABLE IF NOT EXISTS subagent_plan_descriptors (
    descriptor_id TEXT PRIMARY KEY,
    fingerprint TEXT NOT NULL,
    descriptor_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS subagent_executions (
    execution_id TEXT PRIMARY KEY,
    owner_scope_id TEXT NOT NULL REFERENCES sessions(session_id),
    idempotency_key TEXT NOT NULL,
    record_json TEXT NOT NULL,
    input_open INTEGER NOT NULL CHECK(input_open IN (0, 1)),
    cancel_requested INTEGER NOT NULL DEFAULT 0 CHECK(cancel_requested IN (0, 1)),
    cancellation_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(owner_scope_id, idempotency_key),
    CHECK(cancel_requested = 0 OR input_open = 0)
);
CREATE INDEX IF NOT EXISTS subagent_executions_scope_idx
    ON subagent_executions(owner_scope_id, created_at, execution_id);
CREATE TABLE IF NOT EXISTS subagent_inputs (
    input_id TEXT PRIMARY KEY,
    execution_id TEXT NOT NULL REFERENCES subagent_executions(execution_id) ON DELETE CASCADE,
    order_index INTEGER NOT NULL,
    idempotency_key TEXT NOT NULL,
    origin TEXT NOT NULL CHECK(origin IN ('user', 'feature')),
    priority TEXT NOT NULL CHECK(priority IN ('asap', 'when_idle')),
    content_json TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('accepted', 'enqueued', 'applied', 'rejected')),
    native_enqueue_id TEXT,
    rejection_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(execution_id, idempotency_key),
    UNIQUE(execution_id, order_index)
);
CREATE INDEX IF NOT EXISTS subagent_inputs_execution_idx
    ON subagent_inputs(execution_id, state, order_index);

INSERT INTO schema_metadata(key, value) VALUES('schema_version', '5');
