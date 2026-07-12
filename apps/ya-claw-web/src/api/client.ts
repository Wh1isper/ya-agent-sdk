import type {
  AgencyClearResponse,
  AgencyConfigResponse,
  AgencyFireListResponse,
  AgencyStatusResponse,
  BridgeConversationListResponse,
  BridgeEventListResponse,
  BridgeEventStatus,
  ClawInfo,
  HealthStatus,
  HeartbeatConfig,
  HeartbeatFireListResponse,
  HeartbeatFireSummary,
  HeartbeatStatus,
  InputPart,
  ProfileDetail,
  ProfileSeedResponse,
  ProfileSummary,
  ProfileUpsertRequest,
  RunDetail,
  RunGetResponse,
  RunTraceResponse,
  ScheduleCreateRequest,
  ScheduleFireListResponse,
  ScheduleFireSummary,
  ScheduleListFilters,
  ScheduleListResponse,
  ScheduleSummary,
  ScheduleUpdateRequest,
  SessionCreateResponse,
  SessionGetResponse,
  SessionListResponse,
  SessionSandboxState,
  SessionSubmitRequest,
  SessionSubmitResponse,
  SessionSummary,
  SessionWorkspaceState,
  WorkflowDefinitionCreateRequest,
  WorkflowDefinitionDetail,
  WorkflowDefinitionListResponse,
  WorkflowDefinitionUpdateRequest,
  WorkflowEventListResponse,
  WorkflowListFilters,
  WorkflowRunDetail,
  WorkflowRunListFilters,
  WorkflowRunListResponse,
  WorkflowTriggerRequest,
  WorkspaceResolveResponse,
  WorkspaceRuntimeStatus,
} from '../types'
import type {
  WorkspaceFileContentResponse,
  WorkspaceFileListResponse,
} from '../features/workspace/types'

export class ApiError extends Error {
  status: number
  detail: unknown

  constructor(message: string, status: number, detail: unknown) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

export type ApiRequestOptions = {
  signal?: AbortSignal
}

export type ApiClientConfig = {
  baseUrl: string
  apiToken: string
  connectionScope?: string
  onUnauthorized?: (connectionScope: string) => void
}

function normalizeBaseUrl(baseUrl: string) {
  return baseUrl.replace(/\/$/, '')
}

function parseResponseBody(rawBody: string): unknown {
  if (!rawBody) return null
  try {
    return JSON.parse(rawBody) as unknown
  } catch {
    return rawBody
  }
}

function messageForStatus(status: number) {
  if (status === 401) return 'The API token is invalid or expired'
  if (status === 403) return 'This token cannot access the requested resource'
  if (status === 404) return 'The requested YA Claw resource was not found'
  if (status === 409) return 'The runtime state changed; refresh and try again'
  if (status >= 500) return 'The YA Claw runtime returned an internal error'
  return `Request failed with ${status}`
}

export class ClawApiClient {
  private readonly baseUrl: string
  private readonly apiToken: string
  private readonly connectionScope: string
  private readonly onUnauthorized?: (connectionScope: string) => void

  constructor(config: ApiClientConfig) {
    this.baseUrl = normalizeBaseUrl(config.baseUrl)
    this.apiToken = config.apiToken
    this.connectionScope = config.connectionScope ?? 'unscoped'
    this.onUnauthorized = config.onUnauthorized
  }

  async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const headers = new Headers(init.headers)
    if (!headers.has('Content-Type') && init.body !== undefined) {
      headers.set('Content-Type', 'application/json')
    }
    if (this.apiToken.trim()) {
      headers.set('Authorization', `Bearer ${this.apiToken.trim()}`)
    }

    let response: Response
    try {
      response = await fetch(`${this.baseUrl}${path}`, {
        ...init,
        headers,
      })
    } catch (error) {
      if (
        init.signal?.aborted ||
        (typeof error === 'object' &&
          error !== null &&
          'name' in error &&
          error.name === 'AbortError')
      ) {
        throw error
      }
      throw new ApiError(
        'Unable to reach the YA Claw runtime',
        0,
        error instanceof Error ? error.message : error,
      )
    }

    const rawBody = response.status === 204 ? '' : await response.text()
    const parsedBody = parseResponseBody(rawBody)

    if (!response.ok) {
      if (response.status === 401) {
        this.onUnauthorized?.(this.connectionScope)
      }
      throw new ApiError(
        messageForStatus(response.status),
        response.status,
        parsedBody,
      )
    }

    if (!rawBody) {
      return undefined as T
    }

    return parsedBody as T
  }

  health(options: ApiRequestOptions = {}) {
    return this.request<HealthStatus>('/healthz', options)
  }

  clawInfo(options: ApiRequestOptions = {}) {
    return this.request<ClawInfo>('/api/v1/claw/info', options)
  }

  getWorkspaceRuntime(options: ApiRequestOptions = {}) {
    return this.request<WorkspaceRuntimeStatus>(
      '/api/v1/workspace/runtime',
      options,
    )
  }

  resolveWorkspace(metadata: Record<string, unknown>) {
    return this.request<WorkspaceResolveResponse>('/api/v1/workspace:resolve', {
      method: 'POST',
      body: JSON.stringify({ metadata }),
    })
  }

  getSessionWorkspace(sessionId: string, options: ApiRequestOptions = {}) {
    return this.request<SessionWorkspaceState>(
      `/api/v1/sessions/${encodeURIComponent(sessionId)}/workspace`,
      options,
    )
  }

  listWorkspaceFiles(
    sessionId: string,
    options: {
      path?: string | null
      includeHidden?: boolean
      limit?: number
      cursor?: string
      offset?: number
      signal?: AbortSignal
    } = {},
  ) {
    const params = new URLSearchParams()
    if (options.path) params.set('path', options.path)
    params.set('include_hidden', String(options.includeHidden ?? false))
    params.set('limit', String(options.limit ?? 500))
    if (options.cursor !== undefined) params.set('cursor', options.cursor)
    if (options.offset !== undefined)
      params.set('offset', String(options.offset))
    return this.request<WorkspaceFileListResponse>(
      `/api/v1/sessions/${encodeURIComponent(sessionId)}/workspace/files?${params.toString()}`,
      { signal: options.signal },
    )
  }

  getWorkspaceFile(
    sessionId: string,
    path: string,
    options: ApiRequestOptions = {},
  ) {
    const params = new URLSearchParams({ path })
    return this.request<WorkspaceFileContentResponse>(
      `/api/v1/sessions/${encodeURIComponent(sessionId)}/workspace/file?${params.toString()}`,
      options,
    )
  }

  async downloadWorkspaceFile(
    sessionId: string,
    path: string,
    options: ApiRequestOptions = {},
  ) {
    const params = new URLSearchParams({ path })
    const response = await fetch(
      `${this.baseUrl}/api/v1/sessions/${encodeURIComponent(sessionId)}/workspace/file:download?${params.toString()}`,
      {
        headers: { Authorization: `Bearer ${this.apiToken.trim()}` },
        signal: options.signal,
      },
    )
    if (!response.ok) {
      const rawBody = await response.text()
      if (response.status === 401) {
        this.onUnauthorized?.(this.connectionScope)
      }
      throw new ApiError(
        messageForStatus(response.status),
        response.status,
        parseResponseBody(rawBody),
      )
    }
    return response.blob()
  }

  getSessionSandbox(sessionId: string) {
    return this.request<SessionSandboxState>(
      `/api/v1/sessions/${encodeURIComponent(sessionId)}/sandbox`,
    )
  }

  prepareSessionSandbox(sessionId: string) {
    return this.request<SessionSandboxState>(
      `/api/v1/sessions/${encodeURIComponent(sessionId)}/sandbox:prepare`,
      { method: 'POST' },
    )
  }

  stopSessionSandbox(sessionId: string) {
    return this.request<SessionSandboxState>(
      `/api/v1/sessions/${encodeURIComponent(sessionId)}/sandbox:stop`,
      { method: 'POST' },
    )
  }

  listBridgeConversations(options: ApiRequestOptions = {}) {
    return this.request<BridgeConversationListResponse>(
      '/api/v1/bridges/conversations',
      options,
    )
  }

  listBridgeEvents(
    filters: {
      conversationId?: string | null
      status?: BridgeEventStatus | 'all'
    } = {},
    options: ApiRequestOptions = {},
  ) {
    const params = new URLSearchParams()
    if (filters.conversationId) {
      params.set('conversation_id', filters.conversationId)
    }
    if (filters.status && filters.status !== 'all') {
      params.set('status', filters.status)
    }
    const query = params.toString()
    return this.request<BridgeEventListResponse>(
      `/api/v1/bridges/events${query ? `?${query}` : ''}`,
      options,
    )
  }

  listSessions(options: ApiRequestOptions = {}) {
    return this.request<SessionSummary[]>('/api/v1/sessions', options)
  }

  listSessionsPage(
    options: {
      limit?: number
      beforeUpdatedAt?: string | null
      beforeId?: string | null
      signal?: AbortSignal
    } = {},
  ) {
    const params = new URLSearchParams()
    params.set('limit', String(options.limit ?? 50))
    params.set('include_latest_output', 'false')
    if (options.beforeUpdatedAt) {
      params.set('before_updated_at', options.beforeUpdatedAt)
    }
    if (options.beforeId) params.set('before_id', options.beforeId)
    return this.request<SessionListResponse>(
      `/api/v1/sessions/page?${params.toString()}`,
      { signal: options.signal },
    )
  }

  getSession(
    sessionId: string,
    options: {
      runsLimit?: number
      beforeSequenceNo?: number | null
      includeMessage?: boolean
      includeInputParts?: boolean
      includeHeadPayload?: boolean
      signal?: AbortSignal
    } = {},
  ) {
    const params = new URLSearchParams()
    params.set('runs_limit', String(options.runsLimit ?? 20))
    params.set('include_message', String(options.includeMessage ?? true))
    params.set('include_input_parts', String(options.includeInputParts ?? true))
    params.set(
      'include_head_payload',
      String(options.includeHeadPayload ?? true),
    )
    if (typeof options.beforeSequenceNo === 'number') {
      params.set('before_sequence_no', String(options.beforeSequenceNo))
    }
    return this.request<SessionGetResponse>(
      `/api/v1/sessions/${encodeURIComponent(sessionId)}?${params.toString()}`,
      { signal: options.signal },
    )
  }

  createSession(payload: {
    profile_name?: string | null
    input_parts: InputPart[]
    metadata?: Record<string, unknown>
  }) {
    return this.request<SessionCreateResponse>('/api/v1/sessions', {
      method: 'POST',
      body: JSON.stringify(payload),
    })
  }

  submitSessionInput(sessionId: string, payload: SessionSubmitRequest) {
    return this.request<SessionSubmitResponse>(
      `/api/v1/sessions/${encodeURIComponent(sessionId)}/submit`,
      {
        method: 'POST',
        body: JSON.stringify(payload),
      },
    )
  }

  getAgencyConfig(options: ApiRequestOptions = {}) {
    return this.request<AgencyConfigResponse>('/api/v1/agency/config', options)
  }

  getAgencyStatus(options: ApiRequestOptions = {}) {
    return this.request<AgencyStatusResponse>('/api/v1/agency/status', options)
  }

  listAgencyFires(options: ApiRequestOptions = {}) {
    return this.request<AgencyFireListResponse>('/api/v1/agency/fires', options)
  }

  clearAgency() {
    return this.request<AgencyClearResponse>('/api/v1/agency:clear', {
      method: 'POST',
    })
  }

  getRun(runId: string, options: ApiRequestOptions = {}) {
    return this.request<RunGetResponse>(
      `/api/v1/runs/${encodeURIComponent(runId)}?include_state=false&include_message=true`,
      options,
    )
  }

  getRunTrace(runId: string, options: ApiRequestOptions = {}) {
    return this.request<RunTraceResponse>(
      `/api/v1/runs/${encodeURIComponent(runId)}/trace`,
      options,
    )
  }

  interruptRun(runId: string) {
    return this.request<RunDetail>(
      `/api/v1/runs/${encodeURIComponent(runId)}/interrupt`,
      {
        method: 'POST',
      },
    )
  }

  cancelRun(runId: string) {
    return this.request<RunDetail>(
      `/api/v1/runs/${encodeURIComponent(runId)}/cancel`,
      {
        method: 'POST',
      },
    )
  }

  listProfiles(options: ApiRequestOptions = {}) {
    return this.request<ProfileSummary[]>('/api/v1/profiles', options)
  }

  getProfile(profileName: string, options: ApiRequestOptions = {}) {
    return this.request<ProfileDetail>(
      `/api/v1/profiles/${encodeURIComponent(profileName)}`,
      options,
    )
  }

  upsertProfile(profileName: string, payload: ProfileUpsertRequest) {
    return this.request<ProfileDetail>(
      `/api/v1/profiles/${encodeURIComponent(profileName)}`,
      {
        method: 'PUT',
        body: JSON.stringify(payload),
      },
    )
  }

  deleteProfile(profileName: string) {
    return this.request<void>(
      `/api/v1/profiles/${encodeURIComponent(profileName)}`,
      {
        method: 'DELETE',
      },
    )
  }

  seedProfiles(pruneMissing: boolean) {
    return this.request<ProfileSeedResponse>('/api/v1/profiles/seed', {
      method: 'POST',
      body: JSON.stringify({ prune_missing: pruneMissing }),
    })
  }

  listWorkflows(
    filters: WorkflowListFilters = {},
    options: ApiRequestOptions = {},
  ) {
    const params = new URLSearchParams()
    if (filters.query?.trim()) params.set('query', filters.query.trim())
    if (filters.tags?.length) {
      for (const tag of filters.tags) params.append('tags', tag)
    }
    if (filters.status && filters.status !== 'all') {
      params.set('status', filters.status)
    }
    if (filters.scope && filters.scope !== 'all')
      params.set('scope', filters.scope)
    if (filters.ownerKind?.trim())
      params.set('owner_kind', filters.ownerKind.trim())
    if (filters.onlyCurrentSession) params.set('only_current_session', 'true')
    if (filters.includeArchived) params.set('include_archived', 'true')
    if (filters.currentSessionId?.trim()) {
      params.set('current_session_id', filters.currentSessionId.trim())
    }
    params.set('limit', String(filters.limit ?? 100))
    const query = params.toString()
    return this.request<WorkflowDefinitionListResponse>(
      `/api/v1/workflows${query ? `?${query}` : ''}`,
      options,
    )
  }

  getWorkflow(workflowId: string, options: ApiRequestOptions = {}) {
    return this.request<WorkflowDefinitionDetail>(
      `/api/v1/workflows/${encodeURIComponent(workflowId)}`,
      options,
    )
  }

  createWorkflow(payload: WorkflowDefinitionCreateRequest) {
    return this.request<WorkflowDefinitionDetail>('/api/v1/workflows', {
      method: 'POST',
      body: JSON.stringify(payload),
    })
  }

  updateWorkflow(workflowId: string, payload: WorkflowDefinitionUpdateRequest) {
    return this.request<WorkflowDefinitionDetail>(
      `/api/v1/workflows/${encodeURIComponent(workflowId)}`,
      {
        method: 'PATCH',
        body: JSON.stringify(payload),
      },
    )
  }

  archiveWorkflow(workflowId: string) {
    return this.request<WorkflowDefinitionDetail>(
      `/api/v1/workflows/${encodeURIComponent(workflowId)}:archive`,
      { method: 'POST' },
    )
  }

  triggerWorkflow(workflowId: string, payload: WorkflowTriggerRequest) {
    return this.request<WorkflowRunDetail>(
      `/api/v1/workflows/${encodeURIComponent(workflowId)}:trigger`,
      {
        method: 'POST',
        body: JSON.stringify(payload),
      },
    )
  }

  listWorkflowRuns(
    filters: WorkflowRunListFilters = {},
    options: ApiRequestOptions = {},
  ) {
    const params = new URLSearchParams()
    if (filters.workflowId?.trim())
      params.set('workflow_id', filters.workflowId.trim())
    if (filters.status && filters.status !== 'all')
      params.set('status', filters.status)
    if (filters.triggerKind && filters.triggerKind !== 'all') {
      params.set('trigger_kind', filters.triggerKind)
    }
    if (filters.onlyCurrentSession) params.set('only_current_session', 'true')
    if (filters.onlySupervisedByCurrentSession) {
      params.set('only_supervised_by_current_session', 'true')
    }
    if (filters.onlyTouchedByCurrentSession) {
      params.set('only_touched_by_current_session', 'true')
    }
    if (filters.includeCompleted === false)
      params.set('include_completed', 'false')
    if (filters.currentSessionId?.trim()) {
      params.set('current_session_id', filters.currentSessionId.trim())
    }
    params.set('limit', String(filters.limit ?? 100))
    const query = params.toString()
    return this.request<WorkflowRunListResponse>(
      `/api/v1/workflow-runs${query ? `?${query}` : ''}`,
      options,
    )
  }

  getWorkflowRun(workflowRunId: string, options: ApiRequestOptions = {}) {
    return this.request<WorkflowRunDetail>(
      `/api/v1/workflow-runs/${encodeURIComponent(workflowRunId)}`,
      options,
    )
  }

  listWorkflowEvents(workflowRunId: string, options: ApiRequestOptions = {}) {
    return this.request<WorkflowEventListResponse>(
      `/api/v1/workflow-runs/${encodeURIComponent(workflowRunId)}/events`,
      options,
    )
  }

  cancelWorkflowRun(workflowRunId: string, reason?: string | null) {
    return this.request<WorkflowRunDetail>(
      `/api/v1/workflow-runs/${encodeURIComponent(workflowRunId)}/cancel`,
      {
        method: 'POST',
        body: JSON.stringify({ reason: reason ?? null }),
      },
    )
  }

  steerWorkflowNode(
    workflowRunId: string,
    nodeId: string,
    payload: { prompt?: string | null; input_parts?: InputPart[] },
  ) {
    return this.request<WorkflowRunDetail>(
      `/api/v1/workflow-runs/${encodeURIComponent(workflowRunId)}/nodes/${encodeURIComponent(nodeId)}/steer`,
      {
        method: 'POST',
        body: JSON.stringify(payload),
      },
    )
  }

  listSchedules(
    filters: ScheduleListFilters = {},
    options: ApiRequestOptions = {},
  ) {
    const params = new URLSearchParams()
    if (filters.includeDeleted) params.set('include_deleted', 'true')
    if (filters.includeWorkflow === false)
      params.set('include_workflow', 'false')
    if (filters.workflowId?.trim())
      params.set('workflow_id', filters.workflowId.trim())
    if (filters.executionMode && filters.executionMode !== 'all') {
      params.set('execution_mode', filters.executionMode)
    }
    if (filters.ownerSessionId?.trim()) {
      params.set('owner_session_id', filters.ownerSessionId.trim())
    }
    if (filters.scheduleId?.trim())
      params.set('schedule_id', filters.scheduleId.trim())
    if (filters.includeRecentRuns === false)
      params.set('include_recent_runs', 'false')
    params.set('limit', String(filters.limit ?? 100))
    const query = params.toString()
    return this.request<ScheduleListResponse>(
      `/api/v1/schedules${query ? `?${query}` : ''}`,
      options,
    )
  }

  getSchedule(scheduleId: string, options: ApiRequestOptions = {}) {
    return this.request<ScheduleSummary>(
      `/api/v1/schedules/${encodeURIComponent(scheduleId)}`,
      options,
    )
  }

  createSchedule(payload: ScheduleCreateRequest) {
    return this.request<ScheduleSummary>('/api/v1/schedules', {
      method: 'POST',
      body: JSON.stringify(payload),
    })
  }

  updateSchedule(scheduleId: string, payload: ScheduleUpdateRequest) {
    return this.request<ScheduleSummary>(
      `/api/v1/schedules/${encodeURIComponent(scheduleId)}`,
      {
        method: 'PATCH',
        body: JSON.stringify(payload),
      },
    )
  }

  deleteSchedule(scheduleId: string) {
    return this.request<ScheduleSummary>(
      `/api/v1/schedules/${encodeURIComponent(scheduleId)}`,
      { method: 'DELETE' },
    )
  }

  triggerSchedule(scheduleId: string, promptOverride?: string | null) {
    return this.request<ScheduleFireSummary>(
      `/api/v1/schedules/${encodeURIComponent(scheduleId)}:trigger`,
      {
        method: 'POST',
        body: JSON.stringify({ prompt_override: promptOverride ?? null }),
      },
    )
  }

  listScheduleFires(scheduleId: string, options: ApiRequestOptions = {}) {
    return this.request<ScheduleFireListResponse>(
      `/api/v1/schedules/${encodeURIComponent(scheduleId)}/fires`,
      options,
    )
  }

  getHeartbeatConfig(options: ApiRequestOptions = {}) {
    return this.request<HeartbeatConfig>('/api/v1/heartbeat/config', options)
  }

  getHeartbeatStatus(options: ApiRequestOptions = {}) {
    return this.request<HeartbeatStatus>('/api/v1/heartbeat/status', options)
  }

  listHeartbeatFires(options: ApiRequestOptions = {}) {
    return this.request<HeartbeatFireListResponse>(
      '/api/v1/heartbeat/fires',
      options,
    )
  }

  triggerHeartbeat() {
    return this.request<HeartbeatFireSummary>('/api/v1/heartbeat:trigger', {
      method: 'POST',
    })
  }
}
