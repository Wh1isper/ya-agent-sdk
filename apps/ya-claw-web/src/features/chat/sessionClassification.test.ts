import { describe, expect, it } from 'vitest'

import type { RunSummary, SessionSummary } from '../../types'
import {
  channelLabel,
  sessionSource,
  type SessionSource,
} from './sessionClassification'

function latestRun(triggerType: string): RunSummary {
  return {
    id: `run-${triggerType}`,
    session_id: `session-${triggerType}`,
    sequence_no: 1,
    status: 'completed',
    trigger_type: triggerType,
    created_at: '2026-01-01T00:00:00Z',
  }
}

function session(overrides: Partial<SessionSummary> = {}): SessionSummary {
  return {
    id: 'session-1',
    profile_name: 'default',
    session_type: 'conversation',
    metadata: {},
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-01-01T00:00:00Z',
    status: 'completed',
    run_count: 1,
    ...overrides,
  }
}

describe('session source classification', () => {
  it.each<[SessionSource, Partial<SessionSummary>]>([
    ['web', { metadata: { web: { surface: 'chat' } } }],
    ['bridge', { metadata: { bridge: { provider: 'lark' } } }],
    ['schedule', { latest_run: latestRun('schedule') }],
    ['workflow', { latest_run: latestRun('workflow') }],
    ['heartbeat', { latest_run: latestRun('heartbeat') }],
    ['agency', { latest_run: latestRun('agency_handoff') }],
    ['memory', { session_type: 'memory' }],
    ['api', { latest_run: latestRun('api') }],
  ])('classifies %s sessions', (expected, overrides) => {
    expect(sessionSource(session(overrides))).toBe(expected)
  })

  it('groups proactive and system trigger aliases into human sources', () => {
    expect(sessionSource(session({ latest_run: latestRun('proactive') }))).toBe(
      'agency',
    )
    expect(sessionSource(session({ latest_run: latestRun('system') }))).toBe(
      'memory',
    )
  })

  it('uses durable metadata markers when a latest run is absent', () => {
    expect(
      sessionSource(
        session({ metadata: { source: 'schedule' }, run_count: 0 }),
      ),
    ).toBe('schedule')
    expect(
      sessionSource(
        session({ metadata: { workflow_id: 'workflow-1' }, run_count: 0 }),
      ),
    ).toBe('workflow')
    expect(sessionSource(session({ session_type: 'agency' }))).toBe('agency')
  })

  it('keeps web and bridge metadata authoritative over generic API runs', () => {
    expect(
      sessionSource(
        session({
          metadata: { web: { surface: 'chat' } },
          latest_run: latestRun('api'),
        }),
      ),
    ).toBe('web')
    expect(
      sessionSource(
        session({
          metadata: { bridge: { provider: 'lark' } },
          latest_run: latestRun('api'),
        }),
      ),
    ).toBe('bridge')
  })

  it('provides human labels for every source', () => {
    expect(channelLabel('bridge')).toBe('Connected channel')
    expect(channelLabel('agency')).toBe('Agency / proactive')
    expect(channelLabel('memory')).toBe('Memory / system')
  })
})
