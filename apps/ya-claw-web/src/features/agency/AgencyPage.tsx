import { Activity, Bot, BrainCircuit, Send } from 'lucide-react'
import { useMemo, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

import {
  useAgencyConfigQuery,
  useAgencyFiresQuery,
  useAgencyMutations,
  useAgencyStatusQuery,
  useSessionHistoryQuery,
  useSessionsQuery,
} from '../../api/hooks'
import { EmptyState } from '../../components/EmptyState'
import { StatusBadge } from '../../components/StatusBadge'
import { cn, formatShortId } from '../../lib/utils'
import type { AgencyFireSummary, RunSummary, SessionSummary } from '../../types'
import { buildTimelineFromRuns } from '../chat/agui/eventReducer'
import type { TimelineBlock } from '../chat/agui/types'

const buttonClass =
  'inline-flex items-center justify-center gap-2 rounded-xl px-3 py-2 text-sm font-semibold shadow-sm transition disabled:cursor-not-allowed disabled:opacity-60'

export function AgencyPage() {
  const config = useAgencyConfigQuery()
  const status = useAgencyStatusQuery()
  const fires = useAgencyFiresQuery()
  const sessions = useSessionsQuery()
  const mutations = useAgencyMutations()
  const [sourceSessionId, setSourceSessionId] = useState('')
  const [prompt, setPrompt] = useState('')

  const agencySessionId =
    status.data?.agency_session_id ?? config.data?.agency_session_id ?? null
  const history = useSessionHistoryQuery(agencySessionId, { runsLimit: 10 })
  const historyPages = history.data?.pages
  const agencyRuns = useMemo(
    () =>
      orderRuns(
        dedupeRuns(historyPages?.flatMap((page) => page.session.runs) ?? []),
      ),
    [historyPages],
  )
  const timeline = useMemo(
    () => buildTimelineFromRuns(agencyRuns, { includeRuntimeEvents: false }),
    [agencyRuns],
  )
  const conversationSessions = useMemo(
    () =>
      (sessions.data ?? []).filter(
        (session) => session.session_type === 'conversation',
      ),
    [sessions.data],
  )
  const selectedSourceSessionId = sourceSessionId

  const sendManualFire = async () => {
    await mutations.trigger.mutateAsync({
      kind: 'manual',
      source_session_id: selectedSourceSessionId || null,
      client_token: `web-${Date.now()}`,
      prompt: prompt.trim() || null,
    })
    setPrompt('')
  }

  return (
    <div className="h-full min-h-0 overflow-auto bg-slate-100 p-6">
      <div className="mx-auto grid max-w-7xl gap-6 xl:grid-cols-[minmax(0,1fr)_24rem]">
        <main className="min-w-0 space-y-6">
          <AgencyHero config={config.data} status={status.data} />
          <ManualTriggerCard
            sessions={conversationSessions}
            selectedSourceSessionId={selectedSourceSessionId}
            setSourceSessionId={setSourceSessionId}
            prompt={prompt}
            setPrompt={setPrompt}
            onSubmit={sendManualFire}
            pending={mutations.trigger.isPending}
          />
          <AgencyEpisodeFlow
            runs={agencyRuns}
            timelineBlocks={timeline.blocks}
            loading={history.isLoading}
            hasMore={Boolean(history.hasNextPage)}
            loadingMore={history.isFetchingNextPage}
            onLoadMore={() => history.fetchNextPage()}
          />
        </main>
        <aside className="min-w-0 space-y-6">
          <AgencyDetails config={config.data} status={status.data} />
          <FireHistory
            fires={fires.data?.fires ?? []}
            loading={fires.isLoading}
          />
        </aside>
      </div>
    </div>
  )
}

function AgencyHero({
  config,
  status,
}: {
  config: ReturnType<typeof useAgencyConfigQuery>['data']
  status: ReturnType<typeof useAgencyStatusQuery>['data']
}) {
  const enabled = config?.enabled ?? status?.enabled ?? false
  return (
    <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-sm font-medium text-blue-600">
            Runtime capability
          </p>
          <h1 className="mt-1 text-2xl font-semibold tracking-tight text-slate-950">
            Agency
          </h1>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-500">
            One Claw instance owns one agency session. Timer, memory commits,
            and manual fires wake it; active work receives new fires through
            steer.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <BrainCircuit className="h-5 w-5 text-blue-600" aria-hidden />
          <StatusBadge status={enabled ? 'active' : 'disabled'} />
          <StatusBadge status={status?.state ?? 'idle'} />
        </div>
      </div>
      <div className="mt-6 grid gap-3 md:grid-cols-4">
        <MetricCard label="State" value={status?.state ?? 'idle'} />
        <MetricCard
          label="Pending fires"
          value={String(status?.pending_fire_count ?? 0)}
        />
        <MetricCard label="Profile" value={config?.profile_name ?? 'default'} />
        <MetricCard
          label="Next timer"
          value={formatDate(status?.next_fire_at ?? config?.next_fire_at)}
        />
      </div>
    </section>
  )
}

function ManualTriggerCard({
  sessions,
  selectedSourceSessionId,
  setSourceSessionId,
  prompt,
  setPrompt,
  onSubmit,
  pending,
}: {
  sessions: SessionSummary[]
  selectedSourceSessionId: string
  setSourceSessionId: (value: string) => void
  prompt: string
  setPrompt: (value: string) => void
  onSubmit: () => Promise<void>
  pending: boolean
}) {
  return (
    <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
      <div className="flex items-center justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold text-slate-950">
            Manual fire
          </h2>
          <p className="mt-1 text-sm text-slate-500">
            Send a bounded instruction into the singleton agency session.
          </p>
        </div>
        <Send className="h-5 w-5 text-slate-400" aria-hidden />
      </div>
      <label className="mt-4 block text-xs font-semibold uppercase tracking-wide text-slate-400">
        Source conversation
      </label>
      <select
        className="mt-2 w-full rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm outline-none ring-blue-600 transition focus:bg-white focus:ring-2"
        value={selectedSourceSessionId}
        onChange={(event) => setSourceSessionId(event.target.value)}
      >
        <option value="">Global fire</option>
        {sessions.map((session) => (
          <option key={session.id} value={session.id}>
            {sessionLabel(session)}
          </option>
        ))}
      </select>
      <textarea
        className="mt-4 min-h-28 w-full rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm outline-none ring-blue-600 transition placeholder:text-slate-400 focus:bg-white focus:ring-2"
        value={prompt}
        onChange={(event) => setPrompt(event.target.value)}
        placeholder="Optional agency instruction"
      />
      <button
        type="button"
        className={cn(
          buttonClass,
          'mt-4 bg-blue-600 text-white hover:bg-blue-700',
        )}
        onClick={() => void onSubmit()}
        disabled={pending}
      >
        <Send className="h-4 w-4" aria-hidden />
        Fire agency
      </button>
    </section>
  )
}

function AgencyDetails({
  config,
  status,
}: {
  config: ReturnType<typeof useAgencyConfigQuery>['data']
  status: ReturnType<typeof useAgencyStatusQuery>['data']
}) {
  return (
    <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
      <h2 className="text-base font-semibold text-slate-950">
        Singleton session
      </h2>
      <dl className="mt-4 space-y-3 text-sm">
        <InfoRow
          label="Session"
          value={
            config?.agency_session_id ?? status?.agency_session_id ?? 'pending'
          }
          mono
        />
        <InfoRow
          label="Scope"
          value={config?.singleton_scope_key ?? 'agency:global'}
          mono
        />
        <InfoRow
          label="Source key"
          value={config?.singleton_source_session_id ?? 'pending'}
          mono
        />
        <InfoRow
          label="Active run"
          value={status?.active_run_id ?? 'none'}
          mono
        />
        <InfoRow
          label="Latest run"
          value={status?.latest_run_id ?? 'none'}
          mono
        />
        <InfoRow
          label="Timer interval"
          value={formatDuration(config?.timer_interval_seconds)}
        />
        <InfoRow
          label="Risk threshold"
          value={config?.risk_policy.max_auto_action_risk ?? 'pending'}
        />
        <InfoRow
          label="Agency index"
          value={config?.memory_files.index ?? 'AGENCY.md'}
          mono
        />
        <InfoRow
          label="Action log"
          value={config?.memory_files.action_log ?? 'agency/ACTION_LOG.md'}
          mono
        />
      </dl>
    </section>
  )
}

function FireHistory({
  fires,
  loading,
}: {
  fires: AgencyFireSummary[]
  loading: boolean
}) {
  return (
    <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-base font-semibold text-slate-950">Fire history</h2>
        <Activity className="h-5 w-5 text-slate-400" aria-hidden />
      </div>
      <div className="mt-4 space-y-2">
        {loading ? (
          <p className="text-sm text-slate-500">Loading fires...</p>
        ) : null}
        {!loading && fires.length === 0 ? (
          <EmptyState
            title="No fires"
            description="Timer, memory, and manual fires appear here."
          />
        ) : null}
        {fires.map((fire) => (
          <FireRow key={fire.id} fire={fire} />
        ))}
      </div>
    </section>
  )
}

function FireRow({ fire }: { fire: AgencyFireSummary }) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-slate-50 p-3">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="mono truncate text-xs font-semibold text-slate-900">
            {fire.id}
          </p>
          <p className="mt-1 text-xs text-slate-500">
            {fire.kind} · {fire.source_session_id ?? 'global'}
          </p>
        </div>
        <StatusBadge status={fire.status} />
      </div>
      <p className="mt-2 text-xs text-slate-400">
        Created {formatDate(fire.created_at)}
      </p>
      {fire.run_id ? (
        <p className="mt-1 mono truncate text-xs text-slate-400">
          Run {fire.run_id}
        </p>
      ) : null}
      {fire.error_message ? (
        <p className="mt-2 text-xs text-rose-600">{fire.error_message}</p>
      ) : null}
    </div>
  )
}

function AgencyEpisodeFlow({
  runs,
  timelineBlocks,
  loading,
  hasMore,
  loadingMore,
  onLoadMore,
}: {
  runs: RunSummary[]
  timelineBlocks: TimelineBlock[]
  loading: boolean
  hasMore: boolean
  loadingMore: boolean
  onLoadMore: () => Promise<unknown>
}) {
  return (
    <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
      <div className="flex items-center justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold text-slate-950">
            Agency session flow
          </h2>
          <p className="mt-1 text-sm text-slate-500">
            The singleton agency session rendered as chat-like episodes.
          </p>
        </div>
        <Bot className="h-5 w-5 text-slate-400" aria-hidden />
      </div>
      {hasMore ? (
        <button
          type="button"
          className="mt-4 rounded-full border border-slate-200 bg-white px-4 py-2 text-xs font-medium text-slate-600 hover:bg-slate-50 disabled:opacity-60"
          disabled={loadingMore}
          onClick={() => void onLoadMore()}
        >
          {loadingMore ? 'Loading older episodes...' : 'Load older episodes'}
        </button>
      ) : null}
      <div className="mt-5 space-y-5">
        {loading ? (
          <p className="text-sm text-slate-500">Loading agency session...</p>
        ) : null}
        {!loading && runs.length === 0 ? (
          <EmptyState
            title="No agency episodes"
            description="A timer, memory commit, or manual fire will create the first episode."
          />
        ) : null}
        {runs.map((run) => (
          <EpisodeCard key={run.id} run={run} />
        ))}
        {timelineBlocks.length > 0 ? (
          <div className="space-y-4 border-t border-slate-100 pt-5">
            {timelineBlocks.map((block) => (
              <TimelineBlockView key={block.id} block={block} />
            ))}
          </div>
        ) : null}
      </div>
    </section>
  )
}

function EpisodeCard({ run }: { run: RunSummary }) {
  const agency = getAgencyMetadata(run)
  const fireIds = stringList(agency.fire_ids)
  const triggerKinds = stringList(agency.trigger_kinds)
  return (
    <article className="rounded-3xl border border-slate-200 bg-slate-50 p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-sm font-semibold text-slate-950">
            Episode {run.sequence_no}
          </p>
          <p className="mono mt-1 text-xs text-slate-500">{run.id}</p>
        </div>
        <StatusBadge status={run.status} />
      </div>
      <div className="mt-3 flex flex-wrap gap-2 text-xs">
        {triggerKinds.map((kind) => (
          <span
            key={kind}
            className="rounded-full bg-blue-50 px-2 py-1 font-medium text-blue-700"
          >
            {kind}
          </span>
        ))}
        {fireIds.map((fireId) => (
          <span
            key={fireId}
            className="mono rounded-full bg-white px-2 py-1 text-slate-500"
          >
            {formatShortId(fireId)}
          </span>
        ))}
      </div>
      <p className="mt-3 whitespace-pre-wrap text-sm leading-6 text-slate-700">
        {run.output_summary ??
          run.output_text ??
          run.input_preview ??
          'Episode queued.'}
      </p>
      <p className="mt-3 text-xs text-slate-400">
        Updated{' '}
        {formatDate(run.finished_at ?? run.started_at ?? run.created_at)}
      </p>
    </article>
  )
}

function TimelineBlockView({ block }: { block: TimelineBlock }) {
  if (block.kind === 'user_input') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[82%] rounded-3xl bg-blue-600 px-4 py-3 text-sm leading-7 text-white shadow-sm">
          {block.parts.map((part, index) =>
            part.type === 'text' ? (
              <p key={index} className="whitespace-pre-wrap">
                {part.text}
              </p>
            ) : (
              <pre
                key={index}
                className="scrollbar-thin max-h-48 overflow-auto rounded-2xl bg-blue-700/60 p-3 text-xs"
              >
                {JSON.stringify(part, null, 2)}
              </pre>
            ),
          )}
        </div>
      </div>
    )
  }
  if (block.kind === 'assistant_message') {
    return (
      <div className="flex items-start gap-3">
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-slate-900 text-white">
          <Bot className="h-4 w-4" />
        </div>
        <div className="min-w-0 flex-1 rounded-3xl border border-slate-200 bg-white px-4 py-3 text-sm leading-7 text-slate-900 shadow-sm">
          <MarkdownMessage content={block.content} />
        </div>
      </div>
    )
  }
  if (block.kind === 'tool_call') {
    return (
      <details className="rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
        <summary className="cursor-pointer font-medium">
          Tool call · {block.name ?? 'tool'} · {block.status}
        </summary>
        <pre className="scrollbar-thin mt-3 max-h-60 overflow-auto rounded-xl bg-white/70 p-3 text-xs leading-5 text-amber-950">
          {JSON.stringify({ args: block.args, result: block.result }, null, 2)}
        </pre>
      </details>
    )
  }
  return null
}

function MarkdownMessage({ content }: { content: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        p: ({ children }) => <p className="mb-3 last:mb-0">{children}</p>,
        ul: ({ children }) => (
          <ul className="mb-3 list-disc pl-5 last:mb-0">{children}</ul>
        ),
        ol: ({ children }) => (
          <ol className="mb-3 list-decimal pl-5 last:mb-0">{children}</ol>
        ),
        code: ({ children }) => (
          <code className="rounded bg-slate-100 px-1 py-0.5 text-[0.9em]">
            {children}
          </code>
        ),
      }}
    >
      {content}
    </ReactMarkdown>
  )
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
      <p className="text-xs font-medium uppercase tracking-wide text-slate-400">
        {label}
      </p>
      <p className="mt-2 truncate text-lg font-semibold text-slate-950">
        {value}
      </p>
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

function getAgencyMetadata(
  run: RunSummary & { metadata?: Record<string, unknown> },
) {
  const agency = run.metadata?.agency
  if (agency && typeof agency === 'object') {
    return agency as Record<string, unknown>
  }
  const fireParts = (run.input_parts ?? []).filter(
    (
      part,
    ): part is Extract<RunSummary['input_parts'], unknown[]>[number] & {
      type: 'command'
      name: 'agency_fire'
      params?: Record<string, unknown> | null
    } => part.type === 'command' && part.name === 'agency_fire',
  )
  return {
    fire_ids: stringList(fireParts.map((part) => part.params?.fire_id)),
    trigger_kinds: stringList(fireParts.map((part) => part.params?.kind)),
  }
}

function stringList(value: unknown) {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === 'string')
    : []
}

function dedupeRuns(runs: RunSummary[]) {
  const byId = new Map<string, RunSummary>()
  for (const run of runs) byId.set(run.id, { ...byId.get(run.id), ...run })
  return [...byId.values()]
}

function orderRuns(runs: RunSummary[]) {
  return [...runs].sort(
    (left, right) =>
      left.sequence_no - right.sequence_no || left.id.localeCompare(right.id),
  )
}

function sessionLabel(session: SessionSummary) {
  return `${formatShortId(session.id)} · ${session.profile_name ?? 'default'}`
}

function formatDate(value?: string | null) {
  if (!value) return 'none'
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(new Date(value))
}

function formatDuration(seconds?: number | null) {
  if (!seconds) return 'pending'
  if (seconds % 3600 === 0) return `${seconds / 3600}h`
  if (seconds % 60 === 0) return `${seconds / 60}m`
  return `${seconds}s`
}
