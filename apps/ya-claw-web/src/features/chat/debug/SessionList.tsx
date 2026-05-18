import { MessageSquare, Search } from 'lucide-react'

import { EmptyState } from '../../../components/EmptyState'
import { StatusBadge } from '../../../components/StatusBadge'
import { cn, formatShortId } from '../../../lib/utils'
import type { SessionSandboxState, SessionSummary } from '../../../types'
import {
  channelLabel,
  sessionChannel,
  sessionTitle,
} from '../sessionClassification'
import { sandboxLabel, sandboxTone, ttlLabel } from '../../workspaceDisplay'

export function SessionList({
  sessions,
  selectedSessionId,
  search,
  loading,
  onSearchChange,
  onSelect,
}: {
  sessions: SessionSummary[]
  selectedSessionId: string | null
  search: string
  loading: boolean
  onSearchChange: (value: string) => void
  onSelect: (session: SessionSummary) => void
}) {
  return (
    <aside className="flex h-full min-h-0 flex-col overflow-hidden border-b border-r border-slate-200 bg-white lg:border-b-0">
      <div className="border-b border-slate-200 p-4">
        <div className="relative">
          <Search className="pointer-events-none absolute left-3 top-2.5 h-4 w-4 text-slate-400" />
          <input
            className="w-full rounded-xl border border-slate-200 bg-slate-50 py-2 pl-9 pr-3 text-sm outline-none ring-blue-600 transition focus:bg-white focus:ring-2"
            value={search}
            onChange={(event) => onSearchChange(event.target.value)}
            placeholder="Search sessions"
          />
        </div>
      </div>
      <div className="scrollbar-thin min-h-0 flex-1 overscroll-contain overflow-auto p-3">
        {loading ? <SessionSkeleton /> : null}
        {!loading && sessions.length === 0 ? (
          <EmptyState
            icon={MessageSquare}
            title={search.trim() ? 'No matching sessions' : 'No sessions'}
            description={
              search.trim()
                ? 'Try a session id, profile, status, or prompt keyword.'
                : 'Use New debug run and send the first message to create a session.'
            }
            className="min-h-64 bg-slate-50"
          />
        ) : null}
        <div className="space-y-2">
          {sessions.map((session) => {
            const isActive = selectedSessionId === session.id
            return (
              <button
                type="button"
                key={session.id}
                className={cn(
                  'group w-full rounded-2xl border p-3 text-left transition',
                  isActive
                    ? 'border-blue-200 bg-blue-50 shadow-sm ring-1 ring-blue-100'
                    : 'border-slate-200 bg-white hover:border-blue-200 hover:bg-blue-50/40',
                )}
                onClick={() => onSelect(session)}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <p className="mono text-xs text-slate-500">
                        {formatShortId(session.id, 12)}
                      </p>
                      <SessionChannelPill session={session} />
                      {session.active_run_id ? (
                        <span className="rounded-full bg-amber-100 px-2 py-0.5 text-[11px] font-medium text-amber-700">
                          active
                        </span>
                      ) : null}
                    </div>
                    <p className="mt-1 line-clamp-2 text-sm font-semibold leading-5 text-slate-900">
                      {sessionTitle(session)}
                    </p>
                  </div>
                  <StatusBadge status={session.status} />
                </div>
                <div className="mt-3 flex items-center justify-between gap-2 text-xs text-slate-500">
                  <span className="truncate">
                    {session.profile_name ?? 'default'}
                  </span>
                  <div className="flex shrink-0 items-center gap-2">
                    <span>{session.run_count} runs</span>
                    {session.memory_state ? (
                      <span className="rounded-full bg-violet-50 px-2 py-0.5 font-medium text-violet-700">
                        {session.memory_state.extract_count} extracts
                      </span>
                    ) : null}
                    <SessionSandboxPill
                      sandbox={session.workspace_state?.sandbox_state ?? null}
                    />
                  </div>
                </div>
              </button>
            )
          })}
        </div>
      </div>
    </aside>
  )
}

export function SessionChannelPill({ session }: { session: SessionSummary }) {
  const channel = sessionChannel(session)
  return (
    <span
      className={cn(
        'rounded-full px-2 py-0.5 text-[11px] font-medium',
        channel === 'bridge' && 'bg-indigo-50 text-indigo-700',
        channel === 'web' && 'bg-emerald-50 text-emerald-700',
        channel === 'api' && 'bg-slate-100 text-slate-500',
      )}
    >
      {channelLabel(channel)}
    </span>
  )
}

export function SessionSkeleton() {
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

export function SessionSandboxPill({
  sandbox,
}: {
  sandbox: SessionSandboxState | null
}) {
  const tone = sandboxTone(sandbox)
  return (
    <span
      className={cn(
        'rounded-full px-2 py-0.5 text-[11px] font-medium capitalize',
        tone === 'success' && 'bg-emerald-50 text-emerald-700',
        tone === 'warning' && 'bg-amber-50 text-amber-700',
        tone === 'error' && 'bg-rose-50 text-rose-700',
        tone === 'info' && 'bg-blue-50 text-blue-700',
        tone === 'muted' && 'bg-slate-100 text-slate-500',
      )}
      title={sandbox?.container_ref ?? undefined}
    >
      {sandboxLabel(sandbox)}
      {sandbox?.ttl_seconds_remaining != null
        ? ` · ${ttlLabel(sandbox.ttl_seconds_remaining)}`
        : ''}
    </span>
  )
}
