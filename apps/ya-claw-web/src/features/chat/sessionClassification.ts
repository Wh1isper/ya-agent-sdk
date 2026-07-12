import type { SessionSummary } from '../../types'

export type SessionSource =
  | 'web'
  | 'bridge'
  | 'schedule'
  | 'workflow'
  | 'heartbeat'
  | 'agency'
  | 'memory'
  | 'api'

/** @deprecated Prefer SessionSource. */
export type SessionChannel = SessionSource

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function hasWebChatMetadata(
  metadata: Record<string, unknown> | null | undefined,
) {
  const web = metadata?.web
  return isRecord(web) && web.surface === 'chat'
}

function normalizedMetadataSource(
  metadata: Record<string, unknown> | null | undefined,
) {
  for (const value of [metadata?.source, metadata?.trigger_type]) {
    if (typeof value === 'string' && value.trim()) {
      return value.trim().toLowerCase().replace(/-/g, '_')
    }
  }
  return null
}

function sourceFromMarker(marker: string | null): SessionSource | null {
  if (!marker) return null
  if (marker === 'web' || marker === 'web_chat' || marker === 'chat') {
    return 'web'
  }
  if (marker === 'bridge') return 'bridge'
  if (marker === 'schedule' || marker === 'scheduled') return 'schedule'
  if (marker === 'workflow') return 'workflow'
  if (marker === 'heartbeat') return 'heartbeat'
  if (
    marker === 'agency' ||
    marker === 'agency_handoff' ||
    marker === 'proactive' ||
    marker === 'async_task'
  ) {
    return 'agency'
  }
  if (marker === 'memory' || marker === 'system') return 'memory'
  if (marker === 'api' || marker === 'manual') return 'api'
  return null
}

/**
 * Classify a session by durable session metadata first, then by its latest run.
 * This keeps web and connected-channel sessions recognizable even though those
 * sessions can contain runs submitted through the generic API trigger.
 */
export function sessionSource(session: SessionSummary): SessionSource {
  const metadata = session.metadata

  if (isRecord(metadata.bridge)) return 'bridge'
  if (hasWebChatMetadata(metadata)) return 'web'

  if (session.session_type === 'memory') return 'memory'
  if (
    session.session_type === 'agency' ||
    session.session_type === 'async_task' ||
    isRecord(metadata.agency) ||
    isRecord(metadata.agency_handoff) ||
    isRecord(metadata.proactive)
  ) {
    return 'agency'
  }

  const metadataSource = sourceFromMarker(normalizedMetadataSource(metadata))
  if (metadataSource) return metadataSource

  if (typeof metadata.schedule_id === 'string' || isRecord(metadata.schedule)) {
    return 'schedule'
  }
  if (typeof metadata.workflow_id === 'string' || isRecord(metadata.workflow)) {
    return 'workflow'
  }
  if (isRecord(metadata.heartbeat)) return 'heartbeat'

  return sourceFromMarker(session.latest_run?.trigger_type ?? null) ?? 'api'
}

/** Kept for callers that still present the source as a channel badge. */
export function sessionChannel(session: SessionSummary): SessionSource {
  return sessionSource(session)
}

export function isBridgeSession(session: SessionSummary) {
  return sessionSource(session) === 'bridge'
}

export function isWebChatSession(session: SessionSummary) {
  return sessionSource(session) === 'web'
}

export function channelLabel(source: SessionSource) {
  const labels: Record<SessionSource, string> = {
    web: 'Web chat',
    bridge: 'Connected channel',
    schedule: 'Schedule',
    workflow: 'Workflow',
    heartbeat: 'Heartbeat',
    agency: 'Agency / proactive',
    memory: 'Memory / system',
    api: 'API',
  }
  return labels[source]
}

export function sessionTitle(session: SessionSummary) {
  const latestPreview = session.latest_run?.input_preview?.trim()
  if (latestPreview) return latestPreview
  const metadataTitle = session.metadata.title
  if (typeof metadataTitle === 'string' && metadataTitle.trim()) {
    return metadataTitle.trim()
  }
  return 'Empty session'
}
