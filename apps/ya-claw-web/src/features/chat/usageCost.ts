import type {
  AguiEvent,
  CostEstimate,
  RunSummary,
  UsageSnapshot,
} from '../../types'

export const USAGE_SNAPSHOT_EVENT_NAME = 'ya_agent.usage_snapshot'

type UnknownRecord = Record<string, unknown>

export type CostAvailability = 'complete' | 'partial' | 'unavailable'

export type CostPresentation = {
  availability: CostAvailability
  amount: string
  detail: string
}

export function parseCostEstimate(value: unknown): CostEstimate | null {
  const record = asRecord(value)
  if (!record) return null

  const inputAmount = decimalString(record.input_amount)
  const outputAmount = decimalString(record.output_amount)
  const totalAmount = decimalString(record.total_amount)
  const pricedRequests = nonNegativeInteger(record.priced_requests)
  const unpricedRequests = nonNegativeInteger(record.unpriced_requests)
  if (
    record.currency !== 'USD' ||
    record.basis !== 'api_list_price' ||
    record.source !== 'genai_prices' ||
    inputAmount === null ||
    outputAmount === null ||
    totalAmount === null ||
    pricedRequests === null ||
    unpricedRequests === null
  ) {
    return null
  }

  return {
    currency: 'USD',
    input_amount: inputAmount,
    output_amount: outputAmount,
    total_amount: totalAmount,
    priced_requests: pricedRequests,
    unpriced_requests: unpricedRequests,
    basis: 'api_list_price',
    source: 'genai_prices',
  }
}

export function parseUsageSnapshot(value: unknown): UsageSnapshot | null {
  const record = asRecord(value)
  if (!record || typeof record.run_id !== 'string' || !record.run_id) {
    return null
  }

  const totalCost =
    record.total_cost_estimate == null
      ? null
      : parseCostEstimate(record.total_cost_estimate)
  if (record.total_cost_estimate != null && !totalCost) return null

  return {
    run_id: record.run_id,
    total_usage: asRecord(record.total_usage) ?? {},
    total_cost_estimate: totalCost,
    entries: [],
    agent_usages: {},
    model_usages: {},
    model_cost_estimates: parseCostEstimateMap(record.model_cost_estimates),
  }
}

export function usageSnapshotFromEvent(event: AguiEvent): UsageSnapshot | null {
  if (event.type !== 'CUSTOM' || event.name !== USAGE_SNAPSHOT_EVENT_NAME) {
    return null
  }
  const value = asRecord(event.value)
  return parseUsageSnapshot(value?.payload)
}

export function usageSnapshotEventRunId(event: AguiEvent): string | null {
  const value = asRecord(event.value)
  return typeof value?.run_id === 'string' && value.run_id
    ? value.run_id
    : (usageSnapshotFromEvent(event)?.run_id ?? null)
}

export function latestUsageSnapshotFromEvents(
  events: AguiEvent[] | null | undefined,
  runId?: string | null,
): UsageSnapshot | null {
  for (let index = (events?.length ?? 0) - 1; index >= 0; index -= 1) {
    const event = events?.[index] ?? {}
    const snapshot = usageSnapshotFromEvent(event)
    if (snapshot && (!runId || usageSnapshotEventRunId(event) === runId)) {
      return snapshot
    }
  }
  return null
}

export function selectRunUsageSnapshot({
  run,
  liveEvents,
  replayEvents,
}: {
  run: RunSummary | null | undefined
  liveEvents?: AguiEvent[] | null
  replayEvents?: AguiEvent[] | null
}): UsageSnapshot | null {
  const runId = run?.id ?? null
  return (
    latestUsageSnapshotFromEvents(liveEvents, runId) ??
    parseUsageSnapshot(run?.usage_snapshot) ??
    latestUsageSnapshotFromEvents(replayEvents ?? run?.message, runId)
  )
}

export function presentCostEstimate(
  estimate: CostEstimate | null | undefined,
): CostPresentation {
  if (!estimate) {
    return {
      availability: 'unavailable',
      amount: '—',
      detail: 'Pricing unavailable',
    }
  }

  const totalRequests = estimate.priced_requests + estimate.unpriced_requests
  if (estimate.unpriced_requests > 0 && estimate.priced_requests === 0) {
    return {
      availability: 'unavailable',
      amount: '—',
      detail: `0/${totalRequests} requests priced`,
    }
  }

  const availability = estimate.unpriced_requests > 0 ? 'partial' : 'complete'
  return {
    availability,
    amount: formatUsdEstimate(estimate.total_amount),
    detail:
      availability === 'partial'
        ? `${estimate.priced_requests}/${totalRequests} requests priced`
        : totalRequests === 1
          ? '1 request priced'
          : `${totalRequests} requests priced`,
  }
}

export function formatUsdEstimate(amount: string): string {
  const numericAmount = Number(amount)
  if (!Number.isFinite(numericAmount) || numericAmount < 0) return '—'
  const fractionDigits =
    numericAmount === 0 || numericAmount >= 1
      ? 2
      : numericAmount >= 0.01
        ? 4
        : 6
  return `~${new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: fractionDigits,
  }).format(numericAmount)}`
}

function parseCostEstimateMap(value: unknown): Record<string, CostEstimate> {
  const record = asRecord(value)
  if (!record) return {}
  const estimates: Record<string, CostEstimate> = {}
  for (const [modelId, candidate] of Object.entries(record)) {
    const estimate = parseCostEstimate(candidate)
    if (estimate) estimates[modelId] = estimate
  }
  return estimates
}

function asRecord(value: unknown): UnknownRecord | null {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as UnknownRecord)
    : null
}

function decimalString(value: unknown): string | null {
  if (typeof value !== 'string' && typeof value !== 'number') return null
  const normalized = String(value)
  if (!/^(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$/.test(normalized)) {
    return null
  }
  const numericValue = Number(normalized)
  return Number.isFinite(numericValue) && numericValue >= 0 ? normalized : null
}

function nonNegativeInteger(value: unknown): number | null {
  return typeof value === 'number' && Number.isInteger(value) && value >= 0
    ? value
    : null
}
