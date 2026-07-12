import { parseApiDate } from '../../lib/date'
import { formatShortId } from '../../lib/utils'
import type { RunSummary, SessionSummary } from '../../types'

export function dedupeRuns(runs: RunSummary[]) {
  const byId = new Map<string, RunSummary>()
  for (const run of runs) byId.set(run.id, { ...byId.get(run.id), ...run })
  return [...byId.values()]
}

export function orderRuns(runs: RunSummary[]) {
  return [...runs].sort(
    (left, right) =>
      left.sequence_no - right.sequence_no || left.id.localeCompare(right.id),
  )
}

export function selectedRunIdFromFire(fire: {
  status: string
  run_id?: string | null
  active_run_id?: string | null
}) {
  if (
    (fire.status === 'steered' || fire.status === 'merged') &&
    fire.active_run_id
  ) {
    return fire.active_run_id
  }
  return fire.run_id ?? fire.active_run_id ?? null
}

export function sessionLabel(session: SessionSummary) {
  return `${formatShortId(session.id)} · ${session.profile_name ?? 'default'}`
}

export function formatDate(value?: string | null) {
  if (!value) return 'none'
  const date = parseApiDate(value)
  if (Number.isNaN(date.getTime())) return 'unknown'
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(date)
}

export function formatDuration(seconds?: number | null) {
  if (seconds == null) return 'pending'
  if (seconds === 0) return '0s'
  if (seconds % 3600 === 0) return `${seconds / 3600}h`
  if (seconds % 60 === 0) return `${seconds / 60}m`
  return `${seconds}s`
}

export function fireMatchesSearch(
  fire: {
    id: string
    kind: string
    status: string
    source_session_id?: string | null
    source_run_id?: string | null
    run_id?: string | null
    active_run_id?: string | null
    error_message?: string | null
  },
  search: string,
) {
  const needle = search.trim().toLowerCase()
  if (!needle) return true
  return [
    fire.id,
    fire.kind,
    fire.status,
    fire.source_session_id ?? '',
    fire.source_run_id ?? '',
    fire.run_id ?? '',
    fire.active_run_id ?? '',
    fire.error_message ?? '',
  ]
    .join(' ')
    .toLowerCase()
    .includes(needle)
}
