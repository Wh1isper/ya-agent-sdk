import { Activity, BrainCircuit, Power, RefreshCcw, Send } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'

import {
  useAgencyMutations,
  useSessionAgencyQuery,
  useSessionsQuery,
} from '../../api/hooks'
import { EmptyState } from '../../components/EmptyState'
import { StatusBadge } from '../../components/StatusBadge'
import { cn } from '../../lib/utils'
import { useLayoutStore } from '../../stores/layoutStore'
import type { AgencyGetResponse, SessionSummary } from '../../types'

const buttonClass =
  'inline-flex items-center justify-center gap-2 rounded-xl px-3 py-2 text-sm font-semibold shadow-sm transition disabled:cursor-not-allowed disabled:opacity-60'

export function AgencyPage() {
  const sessions = useSessionsQuery()
  const selectedSessionId = useLayoutStore((state) => state.selectedSessionId)
  const selectSession = useLayoutStore((state) => state.selectSession)
  const [search, setSearch] = useState('')
  const conversationSessions = useMemo(
    () =>
      (sessions.data ?? []).filter(
        (session) => session.session_type === 'conversation',
      ),
    [sessions.data],
  )
  const filteredSessions = useMemo(() => {
    const needle = search.trim().toLowerCase()
    if (!needle) return conversationSessions
    return conversationSessions.filter((session) =>
      [
        session.id,
        session.profile_name,
        session.status,
        session.agency_state?.last_agency_reason,
        session.agency_state?.last_agency_run_id,
      ]
        .filter(Boolean)
        .some((value) => String(value).toLowerCase().includes(needle)),
    )
  }, [conversationSessions, search])

  useEffect(() => {
    if (!selectedSessionId && conversationSessions[0]) {
      selectSession(conversationSessions[0].id)
    }
  }, [conversationSessions, selectSession, selectedSessionId])

  const selectedSession = useMemo(
    () =>
      conversationSessions.find(
        (session) => session.id === selectedSessionId,
      ) ?? null,
    [conversationSessions, selectedSessionId],
  )

  return (
    <div className="flex h-full min-h-0 bg-slate-100">
      <aside className="flex w-96 shrink-0 flex-col border-r border-slate-200 bg-white">
        <div className="border-b border-slate-200 p-4">
          <div className="flex items-center justify-between gap-3">
            <div>
              <p className="text-sm font-medium text-blue-600">Automation</p>
              <h1 className="mt-1 text-xl font-semibold tracking-tight text-slate-950">
                Agency Sessions
              </h1>
            </div>
            <BrainCircuit className="h-5 w-5 text-blue-600" aria-hidden />
          </div>
          <input
            className="mt-4 w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm outline-none ring-blue-600 transition placeholder:text-slate-400 focus:bg-white focus:ring-2"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search source sessions"
          />
          <p className="mt-2 text-xs text-slate-400">
            Showing {filteredSessions.length} of {conversationSessions.length}
          </p>
        </div>
        <div className="scrollbar-thin min-h-0 flex-1 overflow-auto p-3">
          {sessions.isLoading ? <AgencyListSkeleton /> : null}
          {!sessions.isLoading && conversationSessions.length === 0 ? (
            <EmptyState
              title="No conversation sessions"
              description="Agency attaches to conversation sessions and works in the same workspace."
            />
          ) : null}
          <div className="space-y-2">
            {filteredSessions.map((session) => (
              <AgencySessionListItem
                key={session.id}
                session={session}
                active={session.id === selectedSessionId}
                onClick={() => selectSession(session.id)}
              />
            ))}
          </div>
        </div>
      </aside>
      <main className="min-w-0 flex-1 overflow-auto p-6">
        <AgencyDetail session={selectedSession} />
      </main>
    </div>
  )
}

function AgencySessionListItem({
  session,
  active,
  onClick,
}: {
  session: SessionSummary
  active: boolean
  onClick: () => void
}) {
  const agency = session.agency_state
  return (
    <button
      type="button"
      className={cn(
        'w-full rounded-2xl border p-3 text-left transition',
        active
          ? 'border-blue-200 bg-blue-50 shadow-sm'
          : 'border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50',
      )}
      onClick={onClick}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="truncate mono text-sm font-semibold text-slate-900">
            {session.id}
          </p>
          <p className="mt-1 text-xs text-slate-500">
            {session.profile_name ?? 'default profile'}
          </p>
        </div>
        <StatusBadge status={agency?.enabled ? 'enabled' : 'disabled'} />
      </div>
      <div className="mt-3 grid grid-cols-3 gap-2 text-xs">
        <Metric label="Episodes" value={agency?.episode_count ?? 0} />
        <Metric label="Pending" value={agency?.pending_signal_count ?? 0} />
        <Metric label="Runs" value={session.run_count} />
      </div>
      <p className="mt-2 truncate text-xs text-slate-400">
        Last reason: {agency?.last_agency_reason ?? 'none'}
      </p>
    </button>
  )
}

function AgencyDetail({ session }: { session: SessionSummary | null }) {
  const agency = useSessionAgencyQuery(session?.id ?? null)
  const mutations = useAgencyMutations(session?.id ?? null)
  const [promptOverride, setPromptOverride] = useState('')

  if (!session) {
    return (
      <EmptyState
        title="Select a source session"
        description="Choose a conversation session to inspect and steer its paired agency session."
      />
    )
  }

  const data = agency.data
  const state = data?.state ?? session.agency_state ?? null
  const agencySession = data?.agency_session ?? null
  const latestAgencyRun = agencySession?.latest_run ?? null

  const sendManualSignal = async () => {
    await mutations.signal.mutateAsync({
      reason: 'manual',
      client_token: `web-${Date.now()}`,
      prompt_override: promptOverride.trim() || null,
      metadata: promptOverride.trim() ? { prompt: promptOverride.trim() } : {},
    })
    setPromptOverride('')
  }

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-sm font-medium text-blue-600">Source session</p>
            <h2 className="mt-1 mono text-xl font-semibold tracking-tight text-slate-950">
              {session.id}
            </h2>
            <p className="mt-2 text-sm text-slate-500">
              Paired agency keeps intentions, receives timed wake signals, and
              advances bounded workspace work.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <StatusBadge status={state?.enabled ? 'enabled' : 'disabled'} />
            <StatusBadge status={latestAgencyRun?.status ?? 'idle'} />
          </div>
        </div>
        <div className="mt-6 grid gap-3 md:grid-cols-4">
          <MetricCard label="Episodes" value={state?.episode_count ?? 0} />
          <MetricCard
            label="Pending signals"
            value={state?.pending_signal_count ?? 0}
          />
          <MetricCard
            label="Observed turns"
            value={state?.last_observed_sequence_no ?? 0}
          />
          <MetricCard
            label="Agency runs"
            value={agencySession?.run_count ?? 0}
          />
        </div>
      </section>

      <section className="grid gap-6 xl:grid-cols-[1fr_22rem]">
        <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
          <div className="flex items-center justify-between gap-3">
            <div>
              <h3 className="text-base font-semibold text-slate-950">
                Manual signal
              </h3>
              <p className="mt-1 text-sm text-slate-500">
                Wake agency now. New signals steer an active agency run.
              </p>
            </div>
            <Send className="h-5 w-5 text-slate-400" aria-hidden />
          </div>
          <textarea
            className="mt-4 min-h-28 w-full rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm outline-none ring-blue-600 transition placeholder:text-slate-400 focus:bg-white focus:ring-2"
            value={promptOverride}
            onChange={(event) => setPromptOverride(event.target.value)}
            placeholder="Optional instruction for this agency episode"
          />
          <div className="mt-4 flex flex-wrap gap-2">
            <button
              type="button"
              className={cn(
                buttonClass,
                'bg-blue-600 text-white hover:bg-blue-700',
              )}
              onClick={sendManualSignal}
              disabled={mutations.signal.isPending}
            >
              <Send className="h-4 w-4" aria-hidden />
              Signal now
            </button>
            <button
              type="button"
              className={cn(
                buttonClass,
                'border border-slate-200 bg-white text-slate-700 hover:bg-slate-50',
              )}
              onClick={() =>
                mutations.compact.mutate({
                  reason: 'compact',
                  client_token: `compact-${Date.now()}`,
                })
              }
              disabled={mutations.compact.isPending}
            >
              <RefreshCcw className="h-4 w-4" aria-hidden />
              Compact
            </button>
            <button
              type="button"
              className={cn(
                buttonClass,
                'border border-slate-200 bg-white text-slate-700 hover:bg-slate-50',
              )}
              onClick={() =>
                mutations.update.mutate({ enabled: !(state?.enabled ?? false) })
              }
              disabled={mutations.update.isPending}
            >
              <Power className="h-4 w-4" aria-hidden />
              {state?.enabled ? 'Disable' : 'Enable'}
            </button>
          </div>
        </div>

        <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
          <h3 className="text-base font-semibold text-slate-950">
            Agency session
          </h3>
          <dl className="mt-4 space-y-3 text-sm">
            <InfoRow
              label="Session"
              value={agencySession?.id ?? state?.agency_session_id ?? 'pending'}
              mono
            />
            <InfoRow
              label="Active run"
              value={agencySession?.active_run_id ?? 'none'}
              mono
            />
            <InfoRow
              label="Last run"
              value={state?.last_agency_run_id ?? 'none'}
              mono
            />
            <InfoRow
              label="Last action"
              value={formatDate(state?.last_action_at)}
            />
            <InfoRow
              label="Cooldown until"
              value={formatDate(state?.cooldown_until)}
            />
          </dl>
        </div>
      </section>

      <section className="grid gap-6 xl:grid-cols-2">
        <SignalList data={data} loading={agency.isLoading} />
        <AgencyRunList agencySession={agencySession} />
      </section>
    </div>
  )
}

function SignalList({
  data,
  loading,
}: {
  data?: AgencyGetResponse
  loading: boolean
}) {
  const signals = data?.signals ?? []
  return (
    <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
      <div className="flex items-center justify-between gap-3">
        <h3 className="text-base font-semibold text-slate-950">
          Recent signals
        </h3>
        <Activity className="h-5 w-5 text-slate-400" aria-hidden />
      </div>
      <div className="mt-4 space-y-2">
        {loading ? (
          <p className="text-sm text-slate-500">Loading signals...</p>
        ) : null}
        {!loading && signals.length === 0 ? (
          <EmptyState
            title="No signals"
            description="Signals appear after timed wake or manual agency requests."
          />
        ) : null}
        {signals.map((signal) => (
          <div
            key={signal.id}
            className="rounded-2xl border border-slate-200 bg-slate-50 p-3"
          >
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <p className="truncate mono text-xs font-semibold text-slate-900">
                  {signal.id}
                </p>
                <p className="mt-1 text-xs text-slate-500">{signal.reason}</p>
              </div>
              <StatusBadge status={signal.status} />
            </div>
            <p className="mt-2 text-xs text-slate-400">
              Created {formatDate(signal.created_at)}
            </p>
          </div>
        ))}
      </div>
    </div>
  )
}

function AgencyRunList({
  agencySession,
}: {
  agencySession: SessionSummary | null
}) {
  const latest = agencySession?.latest_run
  return (
    <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
      <h3 className="text-base font-semibold text-slate-950">
        Latest agency run
      </h3>
      {!latest ? (
        <EmptyState
          title="No agency runs"
          description="A timed wake or manual signal will create the first agency episode."
        />
      ) : (
        <div className="mt-4 rounded-2xl border border-slate-200 bg-slate-50 p-4">
          <div className="flex items-start justify-between gap-3">
            <div>
              <p className="mono text-sm font-semibold text-slate-900">
                {latest.id}
              </p>
              <p className="mt-1 text-xs text-slate-500">
                Sequence {latest.sequence_no}
              </p>
            </div>
            <StatusBadge status={latest.status} />
          </div>
          <p className="mt-3 whitespace-pre-wrap text-sm text-slate-600">
            {latest.output_summary ?? latest.input_preview ?? 'No summary yet'}
          </p>
          <p className="mt-3 text-xs text-slate-400">
            Updated{' '}
            {formatDate(
              latest.finished_at ?? latest.started_at ?? latest.created_at,
            )}
          </p>
        </div>
      )}
    </div>
  )
}

function Metric({ label, value }: { label: string; value: number }) {
  return (
    <span className="rounded-xl border border-slate-200 bg-slate-50 px-2 py-1 text-slate-600">
      <span className="font-semibold text-slate-900">{value}</span> {label}
    </span>
  )
}

function MetricCard({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
      <p className="text-xs font-medium uppercase tracking-wide text-slate-400">
        {label}
      </p>
      <p className="mt-2 text-2xl font-semibold text-slate-950">{value}</p>
    </div>
  )
}

function InfoRow({
  label,
  value,
  mono = false,
}: {
  label: string
  value: string
  mono?: boolean
}) {
  return (
    <div className="flex items-start justify-between gap-3">
      <dt className="text-slate-500">{label}</dt>
      <dd
        className={cn(
          'min-w-0 truncate text-right text-slate-800',
          mono ? 'mono' : '',
        )}
      >
        {value}
      </dd>
    </div>
  )
}

function AgencyListSkeleton() {
  return (
    <div className="space-y-2">
      {Array.from({ length: 4 }).map((_, index) => (
        <div
          key={index}
          className="h-32 animate-pulse rounded-2xl bg-slate-100"
        />
      ))}
    </div>
  )
}

function formatDate(value?: string | null) {
  if (!value) return 'none'
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(new Date(value))
}
