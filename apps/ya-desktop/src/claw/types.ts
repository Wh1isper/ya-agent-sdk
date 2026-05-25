export type JsonObject = Record<string, unknown>

export type DesktopClawConnection = {
  id: string
  kind: 'local_embedded'
  name: string
  baseUrl: string
  apiToken?: string | null
  dataDir?: string | null
  workspaceDir?: string | null
}

export type ClawHealth = {
  status: string
  database?: string
  runtime_state?: string
  runtimeState?: string
}

export type ClawInfo = {
  name: string
  environment: string
  version: string
  serviceVersion?: string
  serviceCommit?: string | null
  serviceRevision?: string
  publicBaseUrl?: string
  instanceId?: string
  auth?: string
  surfaces?: string[]
  workspaceProviderBackend?: string
  storageModel?: string
  features?: JsonObject
  workspaceMountModes?: string[]
  sandboxRetentionPolicies?: string[]
  limits?: Record<string, number>
}

export type ClawRunStatus =
  | 'queued'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | 'interrupted'
  | string

export type ClawSessionStatus =
  | 'idle'
  | 'queued'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | 'interrupted'
  | string

export type ClawInputPart = JsonObject & {
  type?: string
  text?: string
}

export type ClawTextInputPart = {
  type: 'text'
  text: string
  metadata?: JsonObject | null
}

export type ClawWorkspaceMount = {
  id: string
  name?: string
  host_path: string
  virtual_path: string
  mode: 'rw' | 'ro'
  docker_host_path?: string
  metadata?: JsonObject
}

export type ClawWorkspaceBinding = {
  mounts: ClawWorkspaceMount[]
  default_mount_id: string
  cwd: string
  metadata?: JsonObject
}

export type ClawSessionStreamInput = {
  profile_name?: string | null
  metadata?: JsonObject
  workspace?: ClawWorkspaceBinding | JsonObject | null
  input_parts: ClawInputPart[]
}

export type ClawSessionRunStreamInput = {
  restore_from_run_id?: string | null
  reset_state?: boolean
  metadata?: JsonObject
  workspace?: ClawWorkspaceBinding | JsonObject | null
  input_parts: ClawInputPart[]
}

export type ClawStreamEvent = {
  id: string
  event: string
  data: string
  payload: JsonObject
}

export type ClawStreamHandlers = {
  onOpen?: () => void
  onEvent?: (event: ClawStreamEvent) => void
  onClose?: () => void
}

export type ClawNotificationHandlers = {
  onOpen?: () => void
  onEvent?: (event: ClawNotificationEvent) => void | Promise<void>
  onClose?: () => void
}

export type ClawNotificationEvent = {
  id: string
  type: string
  created_at?: string
  createdAt?: string
  payload: JsonObject
}

export type ClawProfileSummary = {
  name: string
  model: string
  workspace_backend_hint?: string | null
  workspaceBackendHint?: string | null
  enabled: boolean
  source_type?: string | null
  sourceType?: string | null
  source_version?: string | null
  sourceVersion?: string | null
  updated_at?: string
  updatedAt?: string
}

export type ClawRunSummary = {
  id: string
  session_id?: string
  sessionId?: string
  sequence_no?: number
  sequenceNo?: number
  restore_from_run_id?: string | null
  restoreFromRunId?: string | null
  status: ClawRunStatus
  trigger_type?: string
  triggerType?: string
  profile_name?: string | null
  profileName?: string | null
  input_preview?: string | null
  inputPreview?: string | null
  input_parts?: ClawInputPart[] | null
  inputParts?: ClawInputPart[] | null
  output_text?: string | null
  outputText?: string | null
  output_summary?: string | null
  outputSummary?: string | null
  error_message?: string | null
  errorMessage?: string | null
  termination_reason?: string | null
  terminationReason?: string | null
  created_at?: string
  createdAt?: string
  started_at?: string | null
  startedAt?: string | null
  finished_at?: string | null
  finishedAt?: string | null
  committed_at?: string | null
  committedAt?: string | null
  message?: JsonObject[] | null
}

export type ClawMemoryState = {
  enabled?: boolean
  turns_since_extract?: number
  turnsSinceExtract?: number
  pending_extract?: boolean
  pendingExtract?: boolean
  pending_summary?: boolean
  pendingSummary?: boolean
}

export type ClawWorkspaceState = {
  sandbox_status?: string | null
  sandboxStatus?: string | null
  workspace?: JsonObject | null
}

export type ClawActiveInteraction = {
  interaction_id?: string
  interactionId?: string
  run_id?: string
  runId?: string
  session_id?: string
  sessionId?: string
  tool_call_id?: string
  toolCallId?: string
  tool_name?: string | null
  toolName?: string | null
  kind?: string
  title?: string
  description?: string | null
  arguments_preview?: unknown
  argumentsPreview?: unknown
  metadata?: JsonObject
  status?: 'pending' | 'approved' | 'denied' | string
  sequence_no?: number
  sequenceNo?: number
  total_count?: number
  totalCount?: number
  created_at?: string | null
  createdAt?: string | null
  resolved_at?: string | null
  resolvedAt?: string | null
}

export type ClawInteractionRespondRequest = {
  approved: boolean
  reason?: string | null
  user_input?: unknown
  client_token?: string | null
}

export type ClawInteractionRespondResponse = {
  session_id?: string
  sessionId?: string
  run_id?: string
  runId?: string
  interaction_id?: string
  interactionId?: string
  tool_call_id?: string
  toolCallId?: string
  status: 'pending' | 'approved' | 'denied' | string
  remaining_interaction_count?: number
  remainingInteractionCount?: number
  current_interaction?: ClawActiveInteraction | null
  currentInteraction?: ClawActiveInteraction | null
}

export type ClawSessionSummary = {
  id: string
  parent_session_id?: string | null
  parentSessionId?: string | null
  profile_name?: string | null
  profileName?: string | null
  session_type?: string
  sessionType?: string
  source_session_id?: string | null
  sourceSessionId?: string | null
  metadata?: JsonObject
  created_at?: string
  createdAt?: string
  updated_at?: string
  updatedAt?: string
  status: ClawSessionStatus
  status_reason?: string
  statusReason?: string
  status_detail?: JsonObject & {
    active_interactions?: ClawActiveInteraction[]
    active_interaction_count?: number
  }
  statusDetail?: JsonObject & {
    activeInteractions?: ClawActiveInteraction[]
    activeInteractionCount?: number
  }
  run_count?: number
  runCount?: number
  head_run_id?: string | null
  headRunId?: string | null
  head_success_run_id?: string | null
  headSuccessRunId?: string | null
  active_run_id?: string | null
  activeRunId?: string | null
  latest_run?: ClawRunSummary | null
  latestRun?: ClawRunSummary | null
  memory_state?: ClawMemoryState | null
  memoryState?: ClawMemoryState | null
  workspace_state?: ClawWorkspaceState | null
  workspaceState?: ClawWorkspaceState | null
}

export type ClawSessionDetail = ClawSessionSummary & {
  runs?: ClawRunSummary[]
  runs_limit?: number
  runsLimit?: number
  runs_has_more?: boolean
  runsHasMore?: boolean
  runs_next_before_sequence_no?: number | null
  runsNextBeforeSequenceNo?: number | null
}

export type ClawSessionGetResponse = {
  session: ClawSessionDetail
  state?: JsonObject | null
  message?: JsonObject[] | null
}

export type ClawSessionTurn = {
  run_id?: string
  runId?: string
  session_id?: string
  sessionId?: string
  sequence_no?: number
  sequenceNo?: number
  restore_from_run_id?: string | null
  restoreFromRunId?: string | null
  profile_name?: string | null
  profileName?: string | null
  input_preview?: string | null
  inputPreview?: string | null
  input_parts?: ClawInputPart[]
  inputParts?: ClawInputPart[]
  output_text?: string | null
  outputText?: string | null
  output_summary?: string | null
  outputSummary?: string | null
  created_at?: string
  createdAt?: string
  committed_at?: string | null
  committedAt?: string | null
}

export type ClawSessionTurnsResponse = {
  session_id?: string
  sessionId?: string
  limit: number
  has_more?: boolean
  hasMore?: boolean
  next_cursor?: string | null
  nextCursor?: string | null
  next_before_sequence_no?: number | null
  nextBeforeSequenceNo?: number | null
  turns: ClawSessionTurn[]
}

export type ClawRunTraceItem = {
  sequence_no?: number
  sequenceNo?: number
  type: 'tool_call' | 'tool_response' | string
  tool_call_id?: string | null
  toolCallId?: string | null
  tool_name?: string | null
  toolName?: string | null
  message_id?: string | null
  messageId?: string | null
  role?: string | null
  content?: string | null
  truncated?: boolean
}

export type ClawAgencyRiskPolicy = {
  max_auto_action_risk?: string
  maxAutoActionRisk?: string
}

export type ClawAgencyConfig = {
  enabled: boolean
  profile_name?: string
  profileName?: string
  timer_interval_seconds?: number
  timerIntervalSeconds?: number
  agency_session_id?: string
  agencySessionId?: string
  singleton_scope_key?: string
  singletonScopeKey?: string
  singleton_source_session_id?: string
  singletonSourceSessionId?: string
  risk_policy?: ClawAgencyRiskPolicy
  riskPolicy?: ClawAgencyRiskPolicy
  memory_files?: Record<string, string>
  memoryFiles?: Record<string, string>
  next_fire_at?: string | null
  nextFireAt?: string | null
}

export type ClawAgencyFire = {
  id: string
  kind: string
  status: string
  scheduled_at?: string
  scheduledAt?: string
  fired_at?: string | null
  firedAt?: string | null
  dedupe_key?: string
  dedupeKey?: string
  source_session_id?: string | null
  sourceSessionId?: string | null
  source_run_id?: string | null
  sourceRunId?: string | null
  agency_session_id?: string | null
  agencySessionId?: string | null
  run_id?: string | null
  runId?: string | null
  active_run_id?: string | null
  activeRunId?: string | null
  run_status?: string | null
  runStatus?: string | null
  priority?: number
  payload?: JsonObject
  error_message?: string | null
  errorMessage?: string | null
  created_at?: string
  createdAt?: string
  updated_at?: string
  updatedAt?: string
  consumed_at?: string | null
  consumedAt?: string | null
}

export type ClawAgencyFireListResponse = {
  fires: ClawAgencyFire[]
}

export type ClawAgencyStatus = {
  enabled: boolean
  agency_session_id?: string
  agencySessionId?: string
  state: 'idle' | 'queued' | 'running' | string
  active_run?: ClawRunSummary | null
  activeRun?: ClawRunSummary | null
  latest_run?: ClawRunSummary | null
  latestRun?: ClawRunSummary | null
  active_run_id?: string | null
  activeRunId?: string | null
  latest_run_id?: string | null
  latestRunId?: string | null
  next_fire_at?: string | null
  nextFireAt?: string | null
  pending_fire_count?: number
  pendingFireCount?: number
  last_fire?: ClawAgencyFire | null
  lastFire?: ClawAgencyFire | null
  agency_session?: ClawSessionSummary
  agencySession?: ClawSessionSummary
}

export type ClawAgencyClearResponse = {
  accepted: boolean
  cleared_session_id?: string | null
  clearedSessionId?: string | null
  new_agency_session_id?: string
  newAgencySessionId?: string
  archived_run_ids?: string[]
  archivedRunIds?: string[]
  deleted_fire_count?: number
  deletedFireCount?: number
  cleared_at?: string
  clearedAt?: string
  agency_session?: ClawSessionSummary
  agencySession?: ClawSessionSummary
}

export type ClawRunTraceResponse = {
  run_id?: string
  runId?: string
  session_id?: string
  sessionId?: string
  item_count?: number
  itemCount?: number
  max_item_chars?: number
  maxItemChars?: number
  max_total_chars?: number
  maxTotalChars?: number
  truncated?: boolean
  trace: ClawRunTraceItem[]
}
