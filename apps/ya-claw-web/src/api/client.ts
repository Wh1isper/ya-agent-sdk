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

export type ApiClientConfig = {
  baseUrl: string
  apiToken: string
}

function normalizeBaseUrl(baseUrl: string) {
  return baseUrl.replace(/\/$/, '')
}

export class ClawApiClient {
  private readonly baseUrl: string
  private readonly apiToken: string

  constructor(config: ApiClientConfig) {
    this.baseUrl = normalizeBaseUrl(config.baseUrl)
    this.apiToken = config.apiToken
  }

  async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const headers = new Headers(init.headers)
    if (!headers.has('Content-Type') && init.body !== undefined) {
      headers.set('Content-Type', 'application/json')
    }
    if (this.apiToken.trim()) {
      headers.set('Authorization', `Bearer ${this.apiToken.trim()}`)
    }

    const response = await fetch(`${this.baseUrl}${path}`, {
      ...init,
      headers,
    })

    if (!response.ok) {
      let detail: unknown = null
      try {
        detail = await response.json()
      } catch {
        detail = await response.text()
      }
      throw new ApiError(
        `Request failed with ${response.status}`,
        response.status,
        detail,
      )
    }

    if (response.status === 204) {
      return undefined as T
    }

    return (await response.json()) as T
  }

  health() {
    return this.request<HealthStatus>('/healthz')
  }

  clawInfo() {
    return this.request<ClawInfo>('/api/v1/claw/info')
  }

  getWorkspaceRuntime() {
    return this.request<WorkspaceRuntimeStatus>('/api/v1/workspace/runtime')
  }

  resolveWorkspace(metadata: Record<string, unknown>) {
    return this.request<WorkspaceResolveResponse>('/api/v1/workspace:resolve', {
      method: 'POST',
      body: JSON.stringify({ metadata }),
    })
  }

  getSessionWorkspace(sessionId: string) {
    return this.request<SessionWorkspaceState>(
      `/api/v1/sessions/${sessionId}/workspace`,
    )
  }

  getSessionSandbox(sessionId: string) {
    return this.request<SessionSandboxState>(
      `/api/v1/sessions/${sessionId}/sandbox`,
    )
  }

  prepareSessionSandbox(sessionId: string) {
    return this.request<SessionSandboxState>(
      `/api/v1/sessions/${sessionId}/sandbox:prepare`,
      { method: 'POST' },
    )
  }

  stopSessionSandbox(sessionId: string) {
    return this.request<SessionSandboxState>(
      `/api/v1/sessions/${sessionId}/sandbox:stop`,
      { method: 'POST' },
    )
  }

  listBridgeConversations() {
    return this.request<BridgeConversationListResponse>(
      '/api/v1/bridges/conversations',
    )
  }

  listBridgeEvents(
    filters: {
      conversationId?: string | null
      status?: BridgeEventStatus | 'all'
    } = {},
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
    )
  }

  listSessions() {
    return this.request<SessionSummary[]>('/api/v1/sessions')
  }

  getSession(
    sessionId: string,
    options: {
      runsLimit?: number
      beforeSequenceNo?: number | null
      includeMessage?: boolean
      includeInputParts?: boolean
    } = {},
  ) {
    const params = new URLSearchParams()
    params.set('runs_limit', String(options.runsLimit ?? 20))
    params.set('include_message', String(options.includeMessage ?? true))
    params.set('include_input_parts', String(options.includeInputParts ?? true))
    if (typeof options.beforeSequenceNo === 'number') {
      params.set('before_sequence_no', String(options.beforeSequenceNo))
    }
    return this.request<SessionGetResponse>(
      `/api/v1/sessions/${sessionId}?${params.toString()}`,
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
      `/api/v1/sessions/${sessionId}/submit`,
      {
        method: 'POST',
        body: JSON.stringify(payload),
      },
    )
  }

  getAgencyConfig() {
    return this.request<AgencyConfigResponse>('/api/v1/agency/config')
  }

  getAgencyStatus() {
    return this.request<AgencyStatusResponse>('/api/v1/agency/status')
  }

  listAgencyFires() {
    return this.request<AgencyFireListResponse>('/api/v1/agency/fires')
  }

  clearAgency() {
    return this.request<AgencyClearResponse>('/api/v1/agency:clear', {
      method: 'POST',
    })
  }

  getRun(runId: string) {
    return this.request<RunGetResponse>(
      `/api/v1/runs/${runId}?include_message=true`,
    )
  }

  getRunTrace(runId: string) {
    return this.request<RunTraceResponse>(`/api/v1/runs/${runId}/trace`)
  }

  interruptRun(runId: string) {
    return this.request<RunDetail>(`/api/v1/runs/${runId}/interrupt`, {
      method: 'POST',
    })
  }

  cancelRun(runId: string) {
    return this.request<RunDetail>(`/api/v1/runs/${runId}/cancel`, {
      method: 'POST',
    })
  }

  listProfiles() {
    return this.request<ProfileSummary[]>('/api/v1/profiles')
  }

  getProfile(profileName: string) {
    return this.request<ProfileDetail>(
      `/api/v1/profiles/${encodeURIComponent(profileName)}`,
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

  listWorkflows(filters: WorkflowListFilters = {}) {
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
    )
  }

  getWorkflow(workflowId: string) {
    return this.request<WorkflowDefinitionDetail>(
      `/api/v1/workflows/${encodeURIComponent(workflowId)}`,
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

  listWorkflowRuns(filters: WorkflowRunListFilters = {}) {
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
    )
  }

  getWorkflowRun(workflowRunId: string) {
    return this.request<WorkflowRunDetail>(
      `/api/v1/workflow-runs/${encodeURIComponent(workflowRunId)}`,
    )
  }

  listWorkflowEvents(workflowRunId: string) {
    return this.request<WorkflowEventListResponse>(
      `/api/v1/workflow-runs/${encodeURIComponent(workflowRunId)}/events`,
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

  listSchedules(filters: ScheduleListFilters = {}) {
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
    )
  }

  getSchedule(scheduleId: string) {
    return this.request<ScheduleSummary>(
      `/api/v1/schedules/${encodeURIComponent(scheduleId)}`,
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

  listScheduleFires(scheduleId: string) {
    return this.request<ScheduleFireListResponse>(
      `/api/v1/schedules/${encodeURIComponent(scheduleId)}/fires`,
    )
  }

  getHeartbeatConfig() {
    return this.request<HeartbeatConfig>('/api/v1/heartbeat/config')
  }

  getHeartbeatStatus() {
    return this.request<HeartbeatStatus>('/api/v1/heartbeat/status')
  }

  listHeartbeatFires() {
    return this.request<HeartbeatFireListResponse>('/api/v1/heartbeat/fires')
  }

  triggerHeartbeat() {
    return this.request<HeartbeatFireSummary>('/api/v1/heartbeat:trigger', {
      method: 'POST',
    })
  }
}
