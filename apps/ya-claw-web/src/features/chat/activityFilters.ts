import { apiTimestamp } from '../../lib/date'
import type { SessionSummary } from '../../types'
import {
  channelLabel,
  sessionChannel,
  sessionTitle,
  type SessionSource,
} from './sessionClassification'

export type ActivityFilters = {
  search: string
  status: SessionSummary['status'] | 'all'
  source: SessionSource | 'all'
  profile: string
  time: 'all' | '24h' | '7d' | '30d'
}

export function filterActivitySessions(
  sessions: SessionSummary[],
  filters: ActivityFilters,
  now = Date.now(),
) {
  const needle = filters.search.trim().toLowerCase()
  const cutoffHours =
    filters.time === '24h'
      ? 24
      : filters.time === '7d'
        ? 24 * 7
        : filters.time === '30d'
          ? 24 * 30
          : null
  const cutoff = cutoffHours ? now - cutoffHours * 60 * 60 * 1000 : null

  return sessions.filter((session) => {
    if (filters.status !== 'all' && session.status !== filters.status) {
      return false
    }
    if (
      filters.source !== 'all' &&
      sessionChannel(session) !== filters.source
    ) {
      return false
    }
    if (
      filters.profile !== 'all' &&
      (session.profile_name?.trim() || 'default') !== filters.profile
    ) {
      return false
    }
    if (cutoff && apiTimestamp(session.updated_at) < cutoff) return false
    if (!needle) return true
    return [
      session.id,
      session.profile_name ?? 'default',
      sessionTitle(session),
      sessionChannel(session),
      channelLabel(sessionChannel(session)),
      session.status,
    ]
      .join(' ')
      .toLowerCase()
      .includes(needle)
  })
}
