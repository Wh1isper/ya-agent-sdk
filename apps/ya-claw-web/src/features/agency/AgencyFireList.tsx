import { Activity, Search } from 'lucide-react'

import { EmptyState } from '../../components/EmptyState'
import { StatusBadge } from '../../components/StatusBadge'
import { cn, formatShortId } from '../../lib/utils'
import type { AgencyFireSummary } from '../../types'
import { fireMatchesSearch, formatDate, selectedRunIdFromFire } from './utils'

export function AgencyFireList({
  fires,
  loading,
  search,
  selectedRunId,
  onSearchChange,
  onSelectRun,
}: {
  fires: AgencyFireSummary[]
  loading: boolean
  search: string
  selectedRunId: string | null
  onSearchChange: (value: string) => void
  onSelectRun: (runId: string | null) => void
}) {
  const filteredFires = fires.filter((fire) => fireMatchesSearch(fire, search))

  return (
    <aside className="flex h-full min-h-0 flex-col overflow-hidden border-b border-r border-slate-200 bg-white lg:border-b-0">
      <div className="border-b border-slate-200 p-4">
        <div className="flex items-center justify-between gap-3">
          <div>
            <p className="text-sm font-semibold text-slate-950">Agency fires</p>
            <p className="mt-0.5 text-xs text-slate-500">
              Timer, memory, compact, and manual wakeups
            </p>
          </div>
          <span className="rounded-full border border-slate-200 bg-slate-50 px-2 py-1 text-xs font-medium text-slate-500">
            {fires.length}
          </span>
        </div>
        <div className="relative mt-4">
          <Search className="pointer-events-none absolute left-3 top-2.5 h-4 w-4 text-slate-400" />
          <input
            className="w-full rounded-xl border border-slate-200 bg-slate-50 py-2 pl-9 pr-3 text-sm outline-none ring-blue-600 transition focus:bg-white focus:ring-2"
            value={search}
            onChange={(event) => onSearchChange(event.target.value)}
            placeholder="Search fires"
          />
        </div>
      </div>
      <div className="scrollbar-thin min-h-0 flex-1 overscroll-contain overflow-auto p-3">
        {loading ? <FireSkeleton /> : null}
        {!loading && filteredFires.length === 0 ? (
          <EmptyState
            icon={Activity}
            title={search.trim() ? 'No matching fires' : 'No fires'}
            description={
              search.trim()
                ? 'Try a fire id, kind, status, source, or run id.'
                : 'Timer, memory, and manual fires appear here.'
            }
            className="min-h-64 bg-slate-50"
          />
        ) : null}
        <div className="space-y-2">
          {filteredFires.map((fire) => {
            const runId = selectedRunIdFromFire(fire)
            return (
              <FireRow
                key={fire.id}
                fire={fire}
                selected={Boolean(runId && runId === selectedRunId)}
                onSelectRun={onSelectRun}
              />
            )
          })}
        </div>
      </div>
    </aside>
  )
}

function FireRow({
  fire,
  selected,
  onSelectRun,
}: {
  fire: AgencyFireSummary
  selected: boolean
  onSelectRun: (runId: string | null) => void
}) {
  const runId = selectedRunIdFromFire(fire)
  return (
    <button
      type="button"
      className={cn(
        'group w-full rounded-2xl border p-3 text-left transition disabled:cursor-not-allowed disabled:opacity-60',
        selected
          ? 'border-blue-200 bg-blue-50 shadow-sm ring-1 ring-blue-100'
          : 'border-slate-200 bg-white hover:border-blue-200 hover:bg-blue-50/40',
      )}
      onClick={() => onSelectRun(runId)}
      disabled={!runId}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="flex min-w-0 items-center gap-2">
            <p className="mono truncate text-xs text-slate-500">
              {formatShortId(fire.id, 12)}
            </p>
            <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[11px] font-medium text-slate-600">
              {fire.kind}
            </span>
          </div>
          <p className="mt-1 truncate text-sm font-semibold leading-5 text-slate-900">
            {fire.source_session_id
              ? `Source ${formatShortId(fire.source_session_id, 12)}`
              : 'Global fire'}
          </p>
        </div>
        <StatusBadge status={fire.status} />
      </div>
      <div className="mt-3 flex items-center justify-between gap-2 text-xs text-slate-500">
        <span className="truncate">{formatDate(fire.created_at)}</span>
        <div className="flex shrink-0 items-center gap-2">
          <span>p{fire.priority}</span>
          {runId ? (
            <span className="mono rounded-full bg-slate-100 px-2 py-0.5 text-[11px] text-slate-500">
              {formatShortId(runId, 10)}
            </span>
          ) : null}
        </div>
      </div>
      {fire.error_message ? (
        <p className="mt-2 line-clamp-2 text-xs leading-5 text-rose-600">
          {fire.error_message}
        </p>
      ) : null}
    </button>
  )
}

function FireSkeleton() {
  return (
    <div className="space-y-2">
      {Array.from({ length: 5 }).map((_, index) => (
        <div
          key={index}
          className="rounded-2xl border border-slate-200 bg-white p-3"
        >
          <div className="h-3 w-24 animate-pulse rounded bg-slate-100" />
          <div className="mt-3 h-4 w-full animate-pulse rounded bg-slate-100" />
          <div className="mt-2 h-4 w-2/3 animate-pulse rounded bg-slate-100" />
        </div>
      ))}
    </div>
  )
}
