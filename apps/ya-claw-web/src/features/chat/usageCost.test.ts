import { describe, expect, it } from 'vitest'

import type { AguiEvent, RunSummary, UsageSnapshot } from '../../types'
import {
  latestUsageSnapshotFromEvents,
  parseCostEstimate,
  presentCostEstimate,
  selectRunUsageSnapshot,
  USAGE_SNAPSHOT_EVENT_NAME,
} from './usageCost'

function cost(total: string, pricedRequests = 1, unpricedRequests = 0) {
  return {
    currency: 'USD',
    input_amount: total,
    output_amount: '0',
    total_amount: total,
    priced_requests: pricedRequests,
    unpriced_requests: unpricedRequests,
    basis: 'api_list_price',
    source: 'genai_prices',
  } as const
}

function snapshot(runId: string, total: string): UsageSnapshot {
  return {
    run_id: runId,
    total_usage: {},
    total_cost_estimate: cost(total),
    entries: [],
    agent_usages: {},
    model_usages: {},
    model_cost_estimates: {},
  }
}

function event(value: UsageSnapshot, outerRunId?: string): AguiEvent {
  return {
    type: 'CUSTOM',
    name: USAGE_SNAPSHOT_EVENT_NAME,
    value: { run_id: outerRunId, payload: value },
  }
}

function run(usageSnapshot?: UsageSnapshot): RunSummary {
  return {
    id: 'run-a',
    session_id: 'session-a',
    sequence_no: 1,
    status: 'completed',
    trigger_type: 'api',
    created_at: '2026-07-19T00:00:00Z',
    usage_snapshot: usageSnapshot,
  }
}

describe('usage cost parsing and selection', () => {
  it('accepts Decimal strings and rejects incomplete estimates', () => {
    expect(parseCostEstimate(cost('0.003'))).toMatchObject({
      total_amount: '0.003',
      priced_requests: 1,
    })
    expect(
      parseCostEstimate({ ...cost('0.003'), source: 'unknown' }),
    ).toBeNull()
  })

  it('uses the transport run ID when the SDK execution ID differs', () => {
    const events = [
      event(snapshot('sdk-segment-1', '0.001'), 'run-a'),
      event(snapshot('sdk-other', '9'), 'run-b'),
      event(snapshot('sdk-segment-2', '0.003'), 'run-a'),
    ]

    expect(
      latestUsageSnapshotFromEvents(events, 'run-a')?.total_cost_estimate
        ?.total_amount,
    ).toBe('0.003')
  })

  it('prefers live, then structured API, then raw replay snapshots', () => {
    const replay = event(snapshot('run-a', '0.001'))
    const structured = snapshot('run-a', '0.002')
    const live = event(snapshot('run-a', '0.003'))

    expect(
      selectRunUsageSnapshot({
        run: run(structured),
        liveEvents: [live],
        replayEvents: [replay],
      })?.total_cost_estimate?.total_amount,
    ).toBe('0.003')
    expect(
      selectRunUsageSnapshot({
        run: run(structured),
        replayEvents: [replay],
      })?.total_cost_estimate?.total_amount,
    ).toBe('0.002')
    expect(
      selectRunUsageSnapshot({
        run: run(),
        replayEvents: [replay],
      })?.total_cost_estimate?.total_amount,
    ).toBe('0.001')
  })

  it('distinguishes complete, partial, and unavailable pricing coverage', () => {
    expect(presentCostEstimate(cost('0.003'))).toMatchObject({
      availability: 'complete',
      amount: '~$0.003',
    })
    expect(presentCostEstimate(cost('0.003', 1, 1))).toMatchObject({
      availability: 'partial',
      detail: '1/2 requests priced',
    })
    expect(presentCostEstimate(cost('0', 0, 2))).toEqual({
      availability: 'unavailable',
      amount: '—',
      detail: '0/2 requests priced',
    })
    expect(presentCostEstimate(null).availability).toBe('unavailable')
  })
})
