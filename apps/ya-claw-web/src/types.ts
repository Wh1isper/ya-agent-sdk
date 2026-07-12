export type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue }

export type HealthStatus = {
  status: string
  database: string
  runtime_state: string
}

export type ClawInfo = {
  name: string
  environment: string
  version: string
  service_version: string
  service_commit?: string | null
  service_revision: string
  service_build?: string | null
  service_image?: string | null
  public_base_url: string
  instance_id: string
  auth: 'bearer'
  surfaces: string[]
  workspace_provider_backend: string
  storage_model: string
  features: {
    session_events: boolean
    run_events: boolean
    notifications: boolean
    profiles: boolean
    schedules?: boolean
    heartbeat?: boolean
    workflows?: boolean
  }
}

export type RuntimeCheckStatus =
  | 'ready'
  | 'warning'
  | 'error'
  | 'checking'
  | 'skipped'

export type WorkspaceRuntimeStatusValue =
  | 'ready'
  | 'degraded'
  | 'unavailable'
  | 'checking'

export type RuntimeCheck = {
  id: string
  status: RuntimeCheckStatus
  message: string
  details: Record<string, unknown>
}

export type WorkspacePathStatus = {
  service_path?: string | null
  docker_host_path?: string | null
  virtual_path?: string | null
  exists: boolean
  writable: boolean
}

export type WorkspaceRuntimeCapabilities = {
  file_browse: boolean
  shell: boolean
  sandbox_prepare: boolean
  sandbox_stop: boolean
}

export type DockerRuntimeStatus = {
  daemon: {
    status: RuntimeCheckStatus
    server_version?: string | null
    error_message?: string | null
  }
  image: {
    ref: string
    present: boolean
    digest?: string | null
    error_message?: string | null
  }
  workspace_user: {
    uid?: number | null
    gid?: number | null
    exec_user?: string | null
  }
  container_cache: {
    enabled: boolean
    cache_dir?: string | null
  }
  retention_policy?: string | null
  idle_ttl_seconds?: number | null
}

export type WorkspaceRuntimeStatus = {
  backend: 'local' | 'docker' | 'remote' | 'cloud' | 'unknown'
  status: WorkspaceRuntimeStatusValue
  execution_location: string
  workspace: WorkspacePathStatus
  capabilities: WorkspaceRuntimeCapabilities
  checks: RuntimeCheck[]
  docker?: DockerRuntimeStatus | null
  updated_at: string
}

export type WorkspaceMountView = {
  id?: string | null
  name?: string | null
  host_path: string
  docker_host_path?: string | null
  virtual_path: string
  mode: string
}

export type WorkspaceBindingView = {
  provider: string
  backend_hint?: string | null
  host_path: string
  docker_host_path?: string | null
  virtual_path: string
  cwd: string
  readable_paths: string[]
  writable_paths: string[]
  fingerprint: string
  generation?: number | null
  sandbox_scope?: string | null
  mounts: WorkspaceMountView[]
  metadata: Record<string, unknown>
}

export type SessionSandboxState = {
  backend?: string | null
  scope?: string | null
  status: 'created' | 'mounted' | 'preparing' | 'ready' | 'failed' | 'stopped'
  ready_state: 'not_started' | 'starting' | 'ready' | 'failed'
  container_ref?: string | null
  container_id?: string | null
  verified_container_id?: string | null
  image?: string | null
  image_digest?: string | null
  work_dir?: string | null
  retention_policy?: string | null
  idle_ttl_seconds?: number | null
  ttl_seconds_remaining?: number | null
  expires_at?: string | null
  last_used_at?: string | null
  last_started_at?: string | null
  error_message?: string | null
  updated_at: string
}

export type SessionWorkspaceState = {
  binding?: WorkspaceBindingView | null
  sandbox_state?: SessionSandboxState | null
}

export type WorkspaceResolveResponse = {
  binding: WorkspaceBindingView
  sandbox_state?: SessionSandboxState | null
}

export type FireRunStatus =
  | 'queued'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled'

export type WorkflowRunStatus =
  | 'queued'
  | 'running'
  | 'waiting'
  | 'completed'
  | 'failed'
  | 'cancelled'

export type WorkflowDefinitionStatus = 'draft' | 'active' | 'archived'
export type WorkflowScope = 'global' | 'session'
export type WorkflowTriggerKind =
  | 'web'
  | 'api'
  | 'agent'
  | 'schedule'
  | 'bridge'
  | 'system'

export type WorkflowNodeRunStatus =
  | 'pending'
  | 'ready'
  | 'queued'
  | 'running'
  | 'waiting'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | 'skipped'

export type WorkflowDefinitionSummary = {
  id: string
  name: string
  description?: string | null
  status: WorkflowDefinitionStatus
  definition_version: number
  schema_version: string
  owner_kind: string
  owner_session_id?: string | null
  owner_run_id?: string | null
  scope: WorkflowScope
  tags: string[]
  when_to_use?: string | null
  argument_hint?: string | null
  latest_run?: WorkflowRunSummary | null
  metadata: Record<string, unknown>
  created_at: string
  updated_at: string
  archived_at?: string | null
}

export type WorkflowDefinitionDetail = WorkflowDefinitionSummary & {
  input_schema: Record<string, unknown>
  definition: Record<string, unknown>
}

export type WorkflowDefinitionListResponse = {
  workflows: WorkflowDefinitionSummary[]
}

export type WorkflowNodeRunSummary = {
  id: string
  workflow_run_id: string
  node_id: string
  attempt_no: number
  status: WorkflowNodeRunStatus
  profile_name?: string | null
  session_id?: string | null
  run_id?: string | null
  input_preview?: string | null
  output_text?: string | null
  output_json?: Record<string, unknown> | null
  error_message?: string | null
  needs: string[]
  metadata: Record<string, unknown>
  started_at?: string | null
  finished_at?: string | null
  updated_at: string
}

export type WorkflowEventSummary = {
  id: string
  workflow_run_id: string
  node_run_id?: string | null
  source_kind: string
  event_type: string
  payload: Record<string, unknown>
  created_at: string
}

export type WorkflowRunSummary = {
  id: string
  workflow_id: string
  workflow_version: number
  workflow_name?: string | null
  status: WorkflowRunStatus
  trigger_kind: WorkflowTriggerKind
  supervisor_session_id?: string | null
  supervisor_run_id?: string | null
  profile_name?: string | null
  inputs: Record<string, unknown>
  result?: Record<string, unknown> | null
  error_message?: string | null
  current_node_ids: string[]
  metadata: Record<string, unknown>
  created_at: string
  started_at?: string | null
  finished_at?: string | null
  updated_at: string
}

export type WorkflowRunDetail = WorkflowRunSummary & {
  definition: Record<string, unknown>
  nodes: WorkflowNodeRunSummary[]
  events: WorkflowEventSummary[]
}

export type WorkflowRunListResponse = {
  workflow_runs: WorkflowRunSummary[]
}

export type WorkflowEventListResponse = {
  workflow_run_id: string
  events: WorkflowEventSummary[]
}

export type WorkflowDefinitionCreateRequest = {
  name?: string | null
  description?: string | null
  status?: WorkflowDefinitionStatus
  scope?: WorkflowScope
  tags?: string[]
  when_to_use?: string | null
  argument_hint?: string | null
  input_schema?: Record<string, unknown>
  definition: Record<string, unknown>
  metadata?: Record<string, unknown>
  owner_kind?: 'user' | 'agent' | 'api' | 'system'
  owner_session_id?: string | null
  owner_run_id?: string | null
}

export type WorkflowDefinitionUpdateRequest = Partial<{
  name: string
  description: string | null
  status: WorkflowDefinitionStatus
  scope: WorkflowScope
  tags: string[]
  when_to_use: string | null
  argument_hint: string | null
  input_schema: Record<string, unknown>
  definition: Record<string, unknown>
  metadata: Record<string, unknown>
}>

export type WorkflowTriggerRequest = {
  inputs?: Record<string, unknown>
  profile_name?: string | null
  supervisor_session_id?: string | null
  supervisor_run_id?: string | null
  trigger_kind?: WorkflowTriggerKind
  metadata?: Record<string, unknown>
}

export type WorkflowListFilters = {
  query?: string | null
  tags?: string[] | null
  status?: WorkflowDefinitionStatus | 'all'
  scope?: WorkflowScope | 'all'
  ownerKind?: string | null
  onlyCurrentSession?: boolean
  includeArchived?: boolean
  currentSessionId?: string | null
  limit?: number
}

export type WorkflowRunListFilters = {
  workflowId?: string | null
  status?: WorkflowRunStatus | 'all'
  triggerKind?: WorkflowTriggerKind | 'all'
  onlyCurrentSession?: boolean
  onlySupervisedByCurrentSession?: boolean
  onlyTouchedByCurrentSession?: boolean
  includeCompleted?: boolean
  currentSessionId?: string | null
  limit?: number
}

export type ScheduleFireSummary = {
  id: string
  schedule_id: string
  scheduled_at: string
  fired_at?: string | null
  status: 'pending' | 'submitted' | 'steered' | 'skipped' | 'failed'
  target_session_id?: string | null
  source_session_id?: string | null
  created_session_id?: string | null
  run_id?: string | null
  active_run_id?: string | null
  workflow_run_id?: string | null
  run_status?: FireRunStatus | null
  input_preview?: string | null
  error_message?: string | null
  created_at: string
  updated_at: string
}

export type ScheduleTriggerSummary =
  | {
      kind: 'cron'
      cron?: string | null
      timezone: string
      next_fire_at?: string | null
    }
  | {
      kind: 'once'
      run_at?: string | null
      timezone: string
      next_fire_at?: string | null
    }

export type ScheduleSummary = {
  id: string
  name: string
  description?: string | null
  enabled: boolean
  status: 'active' | 'paused' | 'completed' | 'deleted'
  prompt: string
  trigger: ScheduleTriggerSummary
  cron: {
    expr?: string | null
    timezone: string
    next_fire_at?: string | null
  }
  mode: {
    continue_current_session: boolean
    start_from_current_session: boolean
    steer_when_running: boolean
  }
  execution_mode:
    | 'continue_session'
    | 'fork_session'
    | 'isolate_session'
    | 'workflow'
  workflow_id?: string | null
  workflow_inputs_template?: Record<string, unknown> | null
  last_workflow_run_id?: string | null
  owner_kind: string
  owner_session_id?: string | null
  owner_run_id?: string | null
  profile_name?: string | null
  target_session_id?: string | null
  source_session_id?: string | null
  last_fire?: ScheduleFireSummary | null
  fire_count: number
  failure_count: number
  metadata: Record<string, unknown>
  created_at: string
  updated_at: string
}

export type ScheduleListResponse = {
  schedules: ScheduleSummary[]
}

export type ScheduleListFilters = {
  includeDeleted?: boolean
  includeWorkflow?: boolean
  workflowId?: string | null
  executionMode?:
    | 'continue_session'
    | 'fork_session'
    | 'isolate_session'
    | 'workflow'
    | 'all'
  ownerSessionId?: string | null
  scheduleId?: string | null
  includeRecentRuns?: boolean
  limit?: number
}

export type ScheduleFireListResponse = {
  fires: ScheduleFireSummary[]
}

export type ScheduleCreateRequest = {
  name: string
  description?: string | null
  prompt: string
  trigger_kind?: 'cron' | 'once'
  cron?: string | null
  run_at?: string | null
  timezone: string
  enabled: boolean
  continue_current_session: boolean
  start_from_current_session: boolean
  steer_when_running: boolean
  owner_kind?: 'api' | 'user' | 'agent'
  owner_session_id?: string | null
  owner_run_id?: string | null
  profile_name?: string | null
  metadata?: Record<string, unknown>
  workflow_id?: string | null
  workflow_inputs_template?: Record<string, unknown> | null
}

export type ScheduleUpdateRequest = Partial<{
  name: string
  description: string | null
  prompt: string
  trigger_kind: 'cron' | 'once'
  cron: string | null
  run_at: string | null
  timezone: string
  enabled: boolean
  continue_current_session: boolean
  start_from_current_session: boolean
  steer_when_running: boolean
  metadata: Record<string, unknown>
  workflow_id: string | null
  workflow_inputs_template: Record<string, unknown> | null
}>

export type BridgeEventStatus =
  | 'received'
  | 'queued'
  | 'submitted'
  | 'steered'
  | 'duplicate'
  | 'failed'

export type BridgeConversationSummary = {
  id: string
  adapter: 'lark'
  tenant_key: string
  external_chat_id: string
  session_id: string
  profile_name?: string | null
  metadata: Record<string, unknown>
  active_run_id?: string | null
  event_count: number
  latest_event_status?: BridgeEventStatus | null
  created_at: string
  updated_at: string
  last_event_at?: string | null
}

export type BridgeConversationListResponse = {
  conversations: BridgeConversationSummary[]
}

export type BridgeEventSummary = {
  id: string
  adapter: 'lark'
  tenant_key: string
  event_id: string
  external_message_id?: string | null
  external_chat_id?: string | null
  conversation_id?: string | null
  session_id?: string | null
  run_id?: string | null
  run_status?: FireRunStatus | null
  event_type: string
  status: BridgeEventStatus
  error_message?: string | null
  raw_event: Record<string, unknown>
  normalized_event: Record<string, unknown>
  created_at: string
  updated_at: string
}

export type BridgeEventListResponse = {
  events: BridgeEventSummary[]
}

export type HeartbeatFireSummary = {
  id: string
  scheduled_at: string
  fired_at?: string | null
  status: 'pending' | 'submitted' | 'skipped' | 'failed'
  session_id?: string | null
  run_id?: string | null
  run_status?: FireRunStatus | null
  error_message?: string | null
  metadata: Record<string, unknown>
  created_at: string
  updated_at: string
}

export type HeartbeatConfig = {
  enabled: boolean
  interval_seconds: number
  profile_name: string
  profile_source: 'heartbeat' | 'default'
  prompt: string
  prompt_source: 'heartbeat_setting'
  on_active: string
  guidance_file: {
    path: string
    exists: boolean
  }
  next_fire_at?: string | null
}

export type HeartbeatStatus = {
  enabled: boolean
  next_fire_at?: string | null
  last_fire?: HeartbeatFireSummary | null
}

export type HeartbeatFireListResponse = {
  fires: HeartbeatFireSummary[]
}

export type InputPart =
  | { type: 'text'; text: string; metadata?: Record<string, unknown> | null }
  | {
      type: 'url'
      url: string
      kind: string
      filename?: string | null
      storage?: 'ephemeral' | 'persistent' | 'inline'
      metadata?: Record<string, unknown> | null
    }
  | {
      type: 'file'
      path: string
      kind: string
      metadata?: Record<string, unknown> | null
    }
  | {
      type: 'binary'
      data: string
      mime_type: string
      kind: string
      filename?: string | null
      storage?: 'ephemeral' | 'persistent' | 'inline'
      metadata?: Record<string, unknown> | null
    }
  | {
      type: 'mode'
      mode: string
      params?: Record<string, unknown> | null
      metadata?: Record<string, unknown> | null
    }
  | {
      type: 'command'
      name: string
      params?: Record<string, unknown> | null
      metadata?: Record<string, unknown> | null
    }

export type RunStatus =
  | 'queued'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled'
export type SessionStatus = 'idle' | RunStatus

export type AguiEvent = Record<string, unknown> & {
  type?: string
  timestamp?: number
  messageId?: string
  message_id?: string
  role?: string
  name?: string
  delta?: string
  toolCallId?: string
  tool_call_id?: string
  toolCallName?: string
  tool_call_name?: string
  parentMessageId?: string
  parent_message_id?: string
  content?: unknown
  value?: unknown
  result?: unknown
  message?: string
  code?: string
}

export type RunSummary = {
  id: string
  session_id: string
  sequence_no: number
  restore_from_run_id?: string | null
  status: RunStatus
  trigger_type: string
  profile_name?: string | null
  input_preview?: string | null
  input_parts?: InputPart[] | null
  output_text?: string | null
  error_message?: string | null
  termination_reason?: string | null
  created_at: string
  started_at?: string | null
  finished_at?: string | null
  committed_at?: string | null
  message?: AguiEvent[] | null
}

export type RunDetail = RunSummary & {
  metadata: Record<string, unknown>
  has_state: boolean
  has_message: boolean
}

export type MemoryStateSummary = {
  source_session_id: string
  memory_session_id?: string | null
  enabled: boolean
  last_extracted_sequence_no: number
  turns_since_extract: number
  extract_count: number
  extracts_since_summary: number
  pending_extract: boolean
  pending_summary: boolean
  last_extract_run_id?: string | null
  last_summary_run_id?: string | null
  metadata: Record<string, unknown>
  created_at?: string | null
  updated_at?: string | null
}

export type AgencyRiskPolicy = {
  max_auto_action_risk: 'low' | 'medium' | 'high' | 'extra_high'
}

export type AgencyFireSummary = {
  id: string
  kind:
    | 'message_observed'
    | 'run_output_observed'
    | 'memory_session_completed'
    | 'heartbeat'
    | string
  status:
    | 'pending'
    | 'submitted'
    | 'steered'
    | 'merged'
    | 'consumed'
    | 'failed'
    | string
  scheduled_at: string
  fired_at?: string | null
  dedupe_key: string
  source_session_id?: string | null
  source_run_id?: string | null
  agency_session_id?: string | null
  run_id?: string | null
  active_run_id?: string | null
  run_status?: FireRunStatus | null
  priority: number
  payload: Record<string, unknown>
  error_message?: string | null
  created_at: string
  updated_at: string
  consumed_at?: string | null
}

export type AgencyFireListResponse = {
  fires: AgencyFireSummary[]
}

export type AgencySourceSessionSubmitResponse = {
  source_session_id: string
  delivery: 'submitted' | 'queued' | 'merged' | 'steered' | string
  run_id: string
  status: RunStatus | string
  session_submit: SessionSubmitResponse
}

export type AgencyConfigResponse = {
  enabled: boolean
  profile_name: string
  timer_interval_seconds: number
  agency_session_id: string
  singleton_scope_key: string
  singleton_source_session_id: string
  risk_policy: AgencyRiskPolicy
  memory_files: Record<string, string>
  next_fire_at?: string | null
}

export type AgencyStatusResponse = {
  enabled: boolean
  agency_session_id: string
  state: 'idle' | 'queued' | 'running'
  active_run?: RunSummary | null
  latest_run?: RunSummary | null
  active_run_id?: string | null
  latest_run_id?: string | null
  next_fire_at?: string | null
  pending_fire_count: number
  last_fire?: AgencyFireSummary | null
  agency_session: SessionSummary
}

export type AgencyClearResponse = {
  accepted: boolean
  cleared_session_id?: string | null
  new_agency_session_id: string
  archived_run_ids: string[]
  deleted_fire_count: number
  cleared_at: string
  agency_session: SessionSummary
}

export type SessionSummary = {
  id: string
  parent_session_id?: string | null
  profile_name?: string | null
  session_type: 'conversation' | 'memory' | 'agency' | 'async_task'
  source_session_id?: string | null
  metadata: Record<string, unknown>
  created_at: string
  updated_at: string
  status: SessionStatus
  status_reason?: string
  status_detail?: Record<string, unknown>
  run_count: number
  head_run_id?: string | null
  head_success_run_id?: string | null
  active_run_id?: string | null
  latest_run?: RunSummary | null
  memory_state?: MemoryStateSummary | null
  workspace_state?: SessionWorkspaceState | null
}

export type SessionDetail = SessionSummary & {
  runs: RunSummary[]
  runs_limit: number
  runs_has_more: boolean
  runs_next_before_sequence_no?: number | null
}

export type SessionListResponse = {
  sessions: SessionSummary[]
  total: number
  limit: number
  has_more: boolean
  next_before_updated_at?: string | null
  next_before_id?: string | null
}

export type SessionGetResponse = {
  session: SessionDetail
  state?: Record<string, unknown> | null
  message?: AguiEvent[] | null
}

export type SessionCreateResponse = {
  session: SessionSummary
  run?: RunDetail | null
}

export type SessionSubmitRequest = {
  restore_from_run_id?: string | null
  reset_state?: boolean
  input_parts: InputPart[]
  metadata?: Record<string, unknown>
  trigger_type?: string
}

export type SessionSubmitResponse = {
  session_id: string
  run_id: string
  delivery: 'submitted' | 'queued' | 'merged' | 'steered'
  status: RunStatus | string
  run?: RunDetail | null
}

export type SessionRunCreateRequest = {
  restore_from_run_id?: string | null
  reset_state?: boolean
  input_parts: InputPart[]
  metadata?: Record<string, unknown>
  trigger_type?: string
}

export type RunGetResponse = {
  session: SessionSummary
  run: RunDetail
  state?: Record<string, unknown> | null
  message?: AguiEvent[] | null
}

export type RunTraceItem = {
  sequence_no: number
  type: 'tool_call' | 'tool_response'
  tool_call_id?: string | null
  tool_name?: string | null
  message_id?: string | null
  role?: string | null
  content?: string | null
  truncated: boolean
}

export type RunTraceResponse = {
  run_id: string
  session_id: string
  item_count: number
  max_item_chars: number
  max_total_chars: number
  truncated: boolean
  trace: RunTraceItem[]
}

export type ProfileSubagent = {
  name: string
  description: string
  system_prompt: string
  model?: string | null
  model_settings_preset?: string | null
  model_settings_override?: Record<string, unknown> | null
  model_config_preset?: string | null
  model_config_override?: Record<string, unknown> | null
}

export type ProfileMCPServer = {
  transport: 'streamable_http'
  url: string
  headers: Record<string, string>
  description: string
  required: boolean
}

export type ProfileShellReviewConfig = {
  enabled?: boolean
  model?: string | null
  model_settings?: string | Record<string, unknown> | null
  on_needs_approval?: 'defer' | 'deny'
  risk_threshold?: 'low' | 'medium' | 'high' | 'extra_high'
  system_prompt?: string | null
}

export type ProfileSummary = {
  name: string
  model: string
  workspace_backend_hint?: string | null
  enabled: boolean
  source_type?: string | null
  source_version?: string | null
  updated_at: string
}

export type ProfileDetail = ProfileSummary & {
  model_settings_preset?: string | null
  model_settings_override?: Record<string, unknown> | null
  model_config_preset?: string | null
  model_config_override?: Record<string, unknown> | null
  system_prompt?: string | null
  builtin_toolsets: string[]
  toolsets: string[]
  subagents: ProfileSubagent[]
  include_builtin_subagents: boolean
  unified_subagents: boolean
  need_user_approve_tools: string[]
  need_user_approve_mcps: string[]
  enabled_mcps: string[]
  disabled_mcps: string[]
  mcp_servers: Record<string, ProfileMCPServer>
  source_checksum?: string | null
  created_at: string
}

export type ProfileUpsertRequest = {
  model: string
  model_settings_preset?: string | null
  model_settings_override?: Record<string, unknown> | null
  model_config_preset?: string | null
  model_config_override?: Record<string, unknown> | null
  system_prompt?: string | null
  builtin_toolsets: string[]
  subagents: ProfileSubagent[]
  include_builtin_subagents: boolean
  unified_subagents: boolean
  need_user_approve_tools: string[]
  need_user_approve_mcps: string[]
  enabled_mcps: string[]
  disabled_mcps: string[]
  mcp_servers: Record<string, ProfileMCPServer>
  workspace_backend_hint?: string | null
  enabled: boolean
  source_type?: string | null
  source_version?: string | null
  source_checksum?: string | null
}

export type ProfileSeedResponse = {
  seeded_names: string[]
  seed_file: string
  prune_missing: boolean
}

export type NotificationEvent = {
  id: string
  type: string
  created_at: string
  payload: Record<string, unknown>
}
