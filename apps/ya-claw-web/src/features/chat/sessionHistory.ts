import type { AguiEvent, RunSummary, SessionGetResponse } from '../../types'
import { buildTimelineFromRuns, reduceAguiEvent } from './agui/eventReducer'
import type { AguiTimelineState } from './agui/types'

export type SessionHistoryState = {
  runs: RunSummary[]
  timeline: AguiTimelineState
  events: AguiEvent[]
  hasMore: boolean
  loadedRunCount: number
  totalRunCount: number
  oldestSequenceNo: number | null
}

export function mergeSessionHistoryPages(
  pages: SessionGetResponse[] | undefined,
  liveEvents: AguiEvent[] = [],
): SessionHistoryState {
  const latestPage = pages?.[0]
  const orderedRuns = orderRunsForTimeline(dedupeRuns(pages))
  const baseTimeline = buildTimelineFromRuns(orderedRuns, {
    includeRuntimeEvents: false,
  })
  const timeline = liveEvents.reduce(
    (state, event) =>
      reduceAguiEvent(state, event, { includeRuntimeEvents: false }),
    baseTimeline,
  )
  return {
    runs: orderedRuns,
    timeline,
    events: [...orderedRuns.flatMap((run) => run.message ?? []), ...liveEvents],
    hasMore: Boolean(pages?.[pages.length - 1]?.session.runs_has_more),
    loadedRunCount: orderedRuns.length,
    totalRunCount: latestPage?.session.run_count ?? orderedRuns.length,
    oldestSequenceNo: orderedRuns[0]?.sequence_no ?? null,
  }
}

export function latestRunFromHistory(
  runs: RunSummary[],
  fallbackRunId: string | null,
) {
  return (
    runs.find((run) => run.id === fallbackRunId) ??
    [...runs].sort((left, right) => right.sequence_no - left.sequence_no)[0] ??
    null
  )
}

function dedupeRuns(pages: SessionGetResponse[] | undefined) {
  const byId = new Map<string, RunSummary>()
  for (const page of pages ?? []) {
    for (const run of page.session.runs) {
      byId.set(run.id, { ...byId.get(run.id), ...run })
    }
  }
  return [...byId.values()]
}

function orderRunsForTimeline(runs: RunSummary[]) {
  return [...runs].sort((left, right) => {
    if (left.sequence_no !== right.sequence_no) {
      return left.sequence_no - right.sequence_no
    }
    return left.id.localeCompare(right.id)
  })
}
