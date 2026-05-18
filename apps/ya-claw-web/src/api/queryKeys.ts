export const queryKeys = {
  health: ['health'] as const,
  clawInfo: ['claw-info'] as const,
  workspaceRuntime: ['workspace-runtime'] as const,
  sessionWorkspace: (sessionId: string) =>
    ['session-workspace', sessionId] as const,
  sessionSandbox: (sessionId: string) =>
    ['session-sandbox', sessionId] as const,
  bridgeConversations: ['bridge-conversations'] as const,
  bridgeEvents: (conversationId?: string | null, status?: string | null) =>
    ['bridge-events', conversationId ?? 'all', status ?? 'all'] as const,
  sessions: ['sessions'] as const,
  session: (sessionId: string) => ['session', sessionId] as const,
  agencyConfig: ['agency-config'] as const,
  agencyStatus: ['agency-status'] as const,
  agencyFires: ['agency-fires'] as const,
  sessionHistoryBase: (sessionId: string) =>
    ['session-history', sessionId] as const,
  sessionHistory: (sessionId: string, runsLimit: number) =>
    ['session-history', sessionId, runsLimit] as const,
  run: (runId: string) => ['run', runId] as const,
  runTrace: (runId: string) => ['run-trace', runId] as const,
  profiles: ['profiles'] as const,
  profile: (profileName: string) => ['profile', profileName] as const,
  schedules: (includeDeleted = false) => ['schedules', includeDeleted] as const,
  schedule: (scheduleId: string) => ['schedule', scheduleId] as const,
  scheduleFires: (scheduleId: string) =>
    ['schedule-fires', scheduleId] as const,
  heartbeatConfig: ['heartbeat-config'] as const,
  heartbeatStatus: ['heartbeat-status'] as const,
  heartbeatFires: ['heartbeat-fires'] as const,
}
