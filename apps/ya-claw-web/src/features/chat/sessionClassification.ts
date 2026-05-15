import type { SessionSummary } from '../../types'

export type SessionChannel = 'web' | 'bridge' | 'api'

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function hasBridgeMetadata(
  metadata: Record<string, unknown> | null | undefined,
) {
  return isRecord(metadata?.bridge)
}

function hasWebChatMetadata(
  metadata: Record<string, unknown> | null | undefined,
) {
  const web = metadata?.web
  return isRecord(web) && web.surface === 'chat'
}

export function sessionChannel(session: SessionSummary): SessionChannel {
  if (hasBridgeMetadata(session.metadata)) return 'bridge'
  if (hasWebChatMetadata(session.metadata)) return 'web'
  if (session.latest_run?.trigger_type === 'bridge') return 'bridge'
  return 'api'
}

export function isBridgeSession(session: SessionSummary) {
  return sessionChannel(session) === 'bridge'
}

export function isWebChatSession(session: SessionSummary) {
  return sessionChannel(session) === 'web'
}

export function channelLabel(channel: SessionChannel) {
  if (channel === 'bridge') return 'Bridge'
  if (channel === 'web') return 'Web chat'
  return 'API'
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
