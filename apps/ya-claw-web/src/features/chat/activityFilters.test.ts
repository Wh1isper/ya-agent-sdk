import { describe, expect, it } from 'vitest'

import type { SessionSummary } from '../../types'
import { filterActivitySessions } from './activityFilters'

function session(
  overrides: Partial<SessionSummary> & Pick<SessionSummary, 'id'>,
): SessionSummary {
  return {
    profile_name: 'default',
    session_type: 'conversation',
    metadata: {},
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-01-10T00:00:00Z',
    status: 'idle',
    run_count: 0,
    ...overrides,
  }
}

const all = {
  search: '',
  status: 'all' as const,
  source: 'all' as const,
  profile: 'all',
  time: 'all' as const,
}

describe('activity filters', () => {
  const rows = [
    session({
      id: 'web-session',
      profile_name: 'research',
      status: 'completed',
      metadata: { web: { surface: 'chat' }, title: 'Quarterly report' },
    }),
    session({
      id: 'bridge-session',
      status: 'failed',
      updated_at: '2025-12-01T00:00:00Z',
      metadata: { bridge: { provider: 'slack' } },
    }),
    session({ id: 'schedule-session', metadata: { source: 'schedule' } }),
    session({ id: 'workflow-session', metadata: { workflow_id: 'flow-1' } }),
    session({ id: 'heartbeat-session', metadata: { source: 'heartbeat' } }),
    session({ id: 'agency-session', session_type: 'agency' }),
    session({ id: 'memory-session', session_type: 'memory' }),
    session({ id: 'api-session' }),
  ]

  it('combines structured status, source, and profile filters', () => {
    expect(
      filterActivitySessions(rows, {
        ...all,
        status: 'completed',
        source: 'web',
        profile: 'research',
      }).map((row) => row.id),
    ).toEqual(['web-session'])
  })

  it('keeps text search alongside structured filters', () => {
    expect(
      filterActivitySessions(rows, { ...all, search: 'quarterly' }).map(
        (row) => row.id,
      ),
    ).toEqual(['web-session'])
    expect(
      filterActivitySessions(rows, { ...all, search: 'bridge' }).map(
        (row) => row.id,
      ),
    ).toEqual(['bridge-session'])
  })

  it.each([
    ['web', 'web-session'],
    ['bridge', 'bridge-session'],
    ['schedule', 'schedule-session'],
    ['workflow', 'workflow-session'],
    ['heartbeat', 'heartbeat-session'],
    ['agency', 'agency-session'],
    ['memory', 'memory-session'],
    ['api', 'api-session'],
  ] as const)('filters the %s source independently', (source, expectedId) => {
    expect(
      filterActivitySessions(rows, { ...all, source }).map((row) => row.id),
    ).toEqual([expectedId])
  })

  it('filters by updated time using a stable reference time', () => {
    expect(
      filterActivitySessions(
        rows,
        { ...all, source: 'web', time: '7d' },
        new Date('2026-01-11T00:00:00Z').getTime(),
      ).map((row) => row.id),
    ).toEqual(['web-session'])
  })
})
