import {
  Activity,
  Bot,
  BrainCircuit,
  ChevronRight,
  Hash,
  MessageSquare,
  Send,
  TerminalSquare,
  Wrench,
} from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

import {
  useAgencyConfigQuery,
  useAgencyFiresQuery,
  useAgencyMutations,
  useAgencyStatusQuery,
  useRunQuery,
  useRunTraceQuery,
  useSessionHistoryQuery,
  useSessionsQuery,
} from '../../api/hooks'
import { EmptyState } from '../../components/EmptyState'
import { JsonView } from '../../components/JsonView'
import { StatusBadge } from '../../components/StatusBadge'
import { cn, formatShortId, safeJsonStringify } from '../../lib/utils'
import type {
  AgencyFireSummary,
  AguiEvent,
  InputPart,
  RunGetResponse,
  RunSummary,
  RunTraceItem,
  RunTraceResponse,
  SessionSummary,
} from '../../types'
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
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null)

  const agencySessionId =
    status.data?.agency_session_id ?? config.data?.agency_session_id ?? null
  const history = useSessionHistoryQuery(agencySessionId, { runsLimit: 20 })
  const historyPages = history.data?.pages
  const agencyRuns = useMemo(
    () =>
      orderRuns(
        dedupeRuns(historyPages?.flatMap((page) => page.session.runs) ?? []),
      ),
    [historyPages],
  )
  const selectedRunQuery = useRunQuery(selectedRunId)
  const selectedRunTrace = useRunTraceQuery(selectedRunId)
  const selectedRunDetail =
    selectedRunQuery.data?.run.id === selectedRunId
      ? selectedRunQuery.data
      : null
  const selectedTrace =
    selectedRunTrace.data?.run_id === selectedRunId
      ? selectedRunTrace.data
      : null
  const selectedRun = useMemo(() => {
    const detailRun = selectedRunDetail?.run
    if (detailRun) {
      return {
        ...detailRun,
        message: selectedRunDetail.message ?? detailRun.message ?? null,
      }
    }
    return agencyRuns.find((run) => run.id === selectedRunId) ?? null
  }, [agencyRuns, selectedRunId, selectedRunDetail])

  useEffect(() => {
    if (!selectedRunId && agencyRuns.length > 0) {
      setSelectedRunId(agencyRuns[agencyRuns.length - 1].id)
    }
  }, [agencyRuns, selectedRunId])

  useEffect(() => {
    const latestRunId = status.data?.latest_run_id ?? status.data?.active_run_id
    if (latestRunId && !agencyRuns.some((run) => run.id === latestRunId)) {
      void history.refetch()
    }
  }, [
    agencyRuns,
    history,
    status.data?.active_run_id,
    status.data?.latest_run_id,
  ])

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
          <AgencyRunExplorer
            runs={agencyRuns}
            selectedRunId={selectedRunId}
            selectedRun={selectedRun}
            selectedRunDetail={selectedRunDetail}
            selectedRunTrace={selectedTrace}
            loading={
              history.isLoading ||
              (Boolean(selectedRunId) &&
                selectedRunQuery.isFetching &&
                !selectedRun)
            }
            traceLoading={
              Boolean(selectedRunId) &&
              selectedRunTrace.isFetching &&
              !selectedTrace
            }
            hasMore={Boolean(history.hasNextPage)}
            loadingMore={history.isFetchingNextPage}
            onLoadMore={() => history.fetchNextPage()}
            onSelectRun={setSelectedRunId}
          />
        </main>
        <aside className="min-w-0 space-y-6">
          <AgencyDetails config={config.data} status={status.data} />
          <FireHistory
            fires={fires.data?.fires ?? []}
            loading={fires.isLoading}
            selectedRunId={selectedRunId}
            onSelectRun={setSelectedRunId}
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
      <div className="mt-6 grid gap-3 md:grid-cols-2 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_minmax(0,1fr)_minmax(14rem,1.4fr)]">
        <MetricCard label="State" value={status?.state ?? 'idle'} />
        <MetricCard
          label="Pending fires"
          value={String(status?.pending_fire_count ?? 0)}
        />
        <MetricCard label="Profile" value={config?.profile_name ?? 'default'} />
        <MetricCard
          label="Next timer"
          value={formatDate(status?.next_fire_at ?? config?.next_fire_at)}
          valueClassName="text-base leading-6"
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
  selectedRunId,
  onSelectRun,
}: {
  fires: AgencyFireSummary[]
  loading: boolean
  selectedRunId: string | null
  onSelectRun: (runId: string | null) => void
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
        {fires.map((fire) => {
          const selectableRunId = fire.run_id ?? fire.active_run_id ?? null
          return (
            <FireRow
              key={fire.id}
              fire={fire}
              selected={Boolean(
                selectableRunId && selectableRunId === selectedRunId,
              )}
              onSelectRun={onSelectRun}
            />
          )
        })}
      </div>
    </section>
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
  return (
    <button
      type="button"
      className={cn(
        'w-full rounded-2xl border p-3 text-left transition',
        selected
          ? 'border-blue-200 bg-blue-50 ring-1 ring-blue-100'
          : 'border-slate-200 bg-slate-50 hover:border-blue-200 hover:bg-blue-50/50',
      )}
      onClick={() => onSelectRun(fire.run_id ?? fire.active_run_id ?? null)}
      disabled={!fire.run_id && !fire.active_run_id}
    >
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
    </button>
  )
}

function AgencyRunExplorer({
  runs,
  selectedRunId,
  selectedRun,
  selectedRunDetail,
  selectedRunTrace,
  loading,
  traceLoading,
  hasMore,
  loadingMore,
  onLoadMore,
  onSelectRun,
}: {
  runs: RunSummary[]
  selectedRunId: string | null
  selectedRun: RunSummary | null
  selectedRunDetail: RunGetResponse | null
  selectedRunTrace: RunTraceResponse | null
  loading: boolean
  traceLoading: boolean
  hasMore: boolean
  loadingMore: boolean
  onLoadMore: () => Promise<unknown>
  onSelectRun: (runId: string | null) => void
}) {
  const selectedTimeline = useMemo(
    () =>
      buildTimelineFromRuns(selectedRun ? [selectedRun] : [], {
        includeRuntimeEvents: false,
      }).blocks,
    [selectedRun],
  )

  return (
    <section className="rounded-3xl border border-slate-200 bg-white shadow-sm">
      <div className="flex flex-col gap-4 border-b border-slate-200 p-6 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <h2 className="text-base font-semibold text-slate-950">
            Agency run inspector
          </h2>
          <p className="mt-1 text-sm text-slate-500">
            Inspect each Agency session run with input, output, tool trace, and
            raw event replay.
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-2 text-xs text-slate-500">
          <Bot className="h-4 w-4 text-slate-400" aria-hidden />
          <span>{runs.length} loaded runs</span>
        </div>
      </div>

      <RunStrip
        runs={runs}
        selectedRunId={selectedRunId}
        hasMore={hasMore}
        loadingMore={loadingMore}
        onLoadMore={onLoadMore}
        onSelectRun={onSelectRun}
      />

      <div className="p-6">
        {loading ? <RunInspectorSkeleton /> : null}
        {!loading && runs.length === 0 ? (
          <EmptyState
            title="No agency runs"
            description="A timer, memory commit, or manual fire will create the first Agency run."
          />
        ) : null}
        {!loading && selectedRun ? (
          <div className="grid min-w-0 gap-6 xl:grid-cols-[minmax(0,1fr)_24rem]">
            <main className="min-w-0 space-y-4">
              <RunOverview run={selectedRun} />
              <RunTimeline
                run={selectedRun}
                blocks={selectedTimeline}
                artifactsPruned={Boolean(
                  selectedRun.status !== 'queued' &&
                    selectedRun.status !== 'running' &&
                    selectedRunDetail?.run.has_message === false &&
                    (selectedRun.message ?? []).length === 0,
                )}
              />
            </main>
            <RunDebugPanel
              run={selectedRun}
              detail={selectedRunDetail}
              trace={selectedRunTrace}
              traceLoading={traceLoading}
            />
          </div>
        ) : null}
      </div>
    </section>
  )
}

function RunStrip({
  runs,
  selectedRunId,
  hasMore,
  loadingMore,
  onLoadMore,
  onSelectRun,
}: {
  runs: RunSummary[]
  selectedRunId: string | null
  hasMore: boolean
  loadingMore: boolean
  onLoadMore: () => Promise<unknown>
  onSelectRun: (runId: string | null) => void
}) {
  return (
    <div className="flex items-center gap-3 overflow-hidden border-b border-slate-200 bg-slate-50 px-4 py-3">
      <div className="shrink-0">
        <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
          Runs
        </p>
        <p className="text-[11px] text-slate-500">{runs.length} loaded</p>
      </div>
      <div className="scrollbar-thin flex min-w-0 flex-1 gap-2 overflow-x-auto py-1">
        {hasMore ? (
          <button
            type="button"
            className="inline-flex shrink-0 items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 transition hover:border-blue-200 hover:bg-blue-50 hover:text-blue-700 disabled:opacity-60"
            onClick={() => void onLoadMore()}
            disabled={loadingMore}
          >
            {loadingMore ? 'Loading...' : 'Older'}
          </button>
        ) : null}
        {runs.length === 0 ? (
          <span className="rounded-full border border-dashed border-slate-200 px-3 py-1.5 text-xs text-slate-400">
            No runs yet
          </span>
        ) : null}
        {runs.map((run) => (
          <button
            type="button"
            key={run.id}
            className={cn(
              'inline-flex shrink-0 items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-medium transition',
              selectedRunId === run.id
                ? 'border-blue-200 bg-blue-50 text-blue-700 shadow-sm ring-1 ring-blue-100'
                : 'border-slate-200 bg-white text-slate-600 hover:border-blue-200 hover:bg-blue-50/60',
            )}
            onClick={() => onSelectRun(run.id)}
          >
            <Hash className="h-3 w-3" />
            {run.sequence_no}
            <span className="capitalize">{run.status}</span>
          </button>
        ))}
      </div>
    </div>
  )
}

function RunOverview({ run }: { run: RunSummary }) {
  const agency = getAgencyMetadata(run)
  const fireIds = stringList(agency.fire_ids)
  const triggerKinds = stringList(agency.trigger_kinds)
  return (
    <article className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-sm font-semibold text-slate-950">
            Run {run.sequence_no}
          </p>
          <p className="mono mt-1 break-all text-xs text-slate-500">{run.id}</p>
        </div>
        <StatusBadge status={run.status} />
      </div>
      <div className="mt-3 grid gap-3 text-xs text-slate-500 sm:grid-cols-2 lg:grid-cols-4">
        <RunFact label="Trigger" value={run.trigger_type} />
        <RunFact label="Profile" value={run.profile_name ?? 'default'} />
        <RunFact label="Created" value={formatDate(run.created_at)} />
        <RunFact
          label="Finished"
          value={formatDate(run.finished_at ?? run.started_at)}
        />
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
      {(run.output_summary ?? run.output_text ?? run.error_message) ? (
        <p
          className={cn(
            'mt-3 whitespace-pre-wrap text-sm leading-6',
            run.error_message ? 'text-rose-700' : 'text-slate-700',
          )}
        >
          {run.output_summary ?? run.output_text ?? run.error_message}
        </p>
      ) : null}
    </article>
  )
}

function RunFact({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0 rounded-xl bg-white px-3 py-2">
      <p className="text-[11px] font-semibold uppercase tracking-wide text-slate-400">
        {label}
      </p>
      <p className="mt-1 break-words text-slate-700">{value}</p>
    </div>
  )
}

function RunTimeline({
  run,
  blocks,
  artifactsPruned,
}: {
  run: RunSummary
  blocks: TimelineBlock[]
  artifactsPruned: boolean
}) {
  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-slate-950">
            Runtime replay
          </h3>
          <p className="mt-1 text-xs text-slate-500">
            {blocks.length} blocks · {(run.message ?? []).length} raw events
          </p>
        </div>
        <TerminalSquare className="h-4 w-4 text-slate-400" aria-hidden />
      </div>
      <div className="mt-4 space-y-4">
        {blocks.length === 0 && artifactsPruned ? (
          <div className="rounded-xl border border-amber-200 bg-amber-50 p-3 text-sm leading-6 text-amber-900">
            Run replay artifacts have been pruned. Database metadata, input
            parts, status, and summaries are still available.
          </div>
        ) : null}
        {blocks.length === 0 && !artifactsPruned ? (
          <RunFallback run={run} />
        ) : null}
        {blocks.map((block) => (
          <TimelineBlockView key={block.id} block={block} />
        ))}
      </div>
    </section>
  )
}

function RunFallback({ run }: { run: RunSummary }) {
  return (
    <div className="space-y-3">
      {run.input_parts?.length ? (
        <Card icon={MessageSquare} title="Input parts" accent="blue">
          <div className="space-y-2">
            {run.input_parts.map((part, index) => (
              <InputPartView key={index} part={part} />
            ))}
          </div>
        </Card>
      ) : null}
      {run.output_text || run.output_summary ? (
        <Card icon={Bot} title="Output" accent="emerald">
          <MarkdownMessage
            content={run.output_text ?? run.output_summary ?? ''}
          />
        </Card>
      ) : null}
      {!run.input_parts?.length && !run.output_text && !run.output_summary ? (
        <EmptyState
          title="No replay yet"
          description="Queued or newly created runs may not have replay events yet."
        />
      ) : null}
    </div>
  )
}

function RunDebugPanel({
  run,
  detail,
  trace,
  traceLoading,
}: {
  run: RunSummary
  detail: RunGetResponse | null
  trace: RunTraceResponse | null
  traceLoading: boolean
}) {
  const events = detail?.message ?? run.message ?? []
  return (
    <aside className="min-w-0 space-y-4">
      <PanelCard
        title="Run facts"
        subtitle={`${formatShortId(run.id, 12)} · ${run.status}`}
      >
        <dl className="space-y-2 text-xs">
          <InfoRow label="Sequence" value={String(run.sequence_no)} />
          <InfoRow
            label="Restore"
            value={run.restore_from_run_id ?? 'none'}
            mono
          />
          <InfoRow label="Started" value={formatDate(run.started_at)} />
          <InfoRow label="Committed" value={formatDate(run.committed_at)} />
          <InfoRow
            label="Termination"
            value={run.termination_reason ?? 'pending'}
          />
        </dl>
      </PanelCard>

      <PanelCard
        title="Tool trace"
        subtitle={`${trace?.item_count ?? 0} items`}
      >
        {traceLoading ? (
          <p className="text-sm text-slate-500">Loading trace...</p>
        ) : null}
        {!traceLoading && (!trace || trace.trace.length === 0) ? (
          <p className="text-sm text-slate-500">No tool trace for this run.</p>
        ) : null}
        {trace?.truncated ? (
          <p className="mb-2 rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-800">
            Trace is truncated.
          </p>
        ) : null}
        <div className="space-y-2">
          {trace?.trace.map((item) => (
            <TraceRow key={`${item.sequence_no}-${item.type}`} item={item} />
          ))}
        </div>
      </PanelCard>

      <PanelCard title="Raw events" subtitle={`${events.length} events`}>
        {events.length === 0 ? (
          <p className="text-sm text-slate-500">No raw events available.</p>
        ) : null}
        <div className="max-h-[32rem] space-y-2 overflow-auto pr-1">
          {events.map((event, index) => (
            <EventRow
              key={`${index}-${event.type ?? 'event'}`}
              event={event}
              index={index}
            />
          ))}
        </div>
      </PanelCard>

      {detail?.run.metadata ? (
        <PanelCard title="Metadata" subtitle="run.metadata">
          <JsonView value={detail.run.metadata} height="260px" />
        </PanelCard>
      ) : null}
    </aside>
  )
}

function PanelCard({
  title,
  subtitle,
  children,
}: {
  title: string
  subtitle?: string
  children: React.ReactNode
}) {
  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="mb-3 flex items-start justify-between gap-3">
        <h3 className="text-sm font-semibold text-slate-950">{title}</h3>
        {subtitle ? (
          <span className="mono shrink-0 rounded-full bg-slate-100 px-2 py-1 text-[11px] text-slate-500">
            {subtitle}
          </span>
        ) : null}
      </div>
      {children}
    </section>
  )
}

function TraceRow({ item }: { item: RunTraceItem }) {
  return (
    <details className="rounded-xl border border-slate-200 bg-slate-50 p-3">
      <summary className="cursor-pointer text-xs font-medium text-slate-700">
        <span className="mono text-slate-400">#{item.sequence_no}</span>{' '}
        {item.type} · {item.tool_name ?? item.role ?? 'message'}
      </summary>
      <pre className="scrollbar-thin mt-3 max-h-56 overflow-auto rounded-lg bg-white p-3 text-xs leading-5 text-slate-700">
        {item.content ?? ''}
      </pre>
    </details>
  )
}

function EventRow({ event, index }: { event: AguiEvent; index: number }) {
  const [expanded, setExpanded] = useState(false)
  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50">
      <button
        type="button"
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs hover:bg-slate-100"
        onClick={() => setExpanded((value) => !value)}
      >
        <ChevronRight
          className={cn(
            'h-3.5 w-3.5 shrink-0 text-slate-400 transition',
            expanded && 'rotate-90',
          )}
        />
        <span className="mono w-8 shrink-0 text-[11px] text-slate-400">
          {index + 1}
        </span>
        <span className="mono min-w-0 flex-1 truncate text-slate-700">
          {event.type ?? 'event'}
          {event.name ? (
            <span className="text-slate-400"> · {event.name}</span>
          ) : null}
        </span>
      </button>
      {expanded ? (
        <pre className="scrollbar-thin max-h-80 overflow-auto border-t border-slate-200 p-3 text-[11px] leading-5 text-slate-700">
          {safeJsonStringify(event)}
        </pre>
      ) : null}
    </div>
  )
}

function RunInspectorSkeleton() {
  return (
    <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_24rem]">
      <div className="space-y-4">
        {Array.from({ length: 3 }).map((_, index) => (
          <div
            key={index}
            className="h-36 animate-pulse rounded-2xl border border-slate-200 bg-slate-50"
          />
        ))}
      </div>
      <div className="space-y-4">
        {Array.from({ length: 3 }).map((_, index) => (
          <div
            key={index}
            className="h-28 animate-pulse rounded-2xl border border-slate-200 bg-slate-50"
          />
        ))}
      </div>
    </div>
  )
}

function TimelineBlockView({ block }: { block: TimelineBlock }) {
  if (block.kind === 'user_input') {
    return (
      <Card icon={MessageSquare} title="Input" accent="blue">
        <div className="space-y-2">
          {block.parts.map((part, index) => (
            <InputPartView key={index} part={part} />
          ))}
        </div>
      </Card>
    )
  }
  if (block.kind === 'assistant_message') {
    return (
      <Card
        icon={Bot}
        title={block.name ? `Assistant · ${block.name}` : 'Assistant'}
        accent="emerald"
      >
        <MarkdownMessage content={block.content} />
      </Card>
    )
  }
  if (block.kind === 'tool_call') {
    return (
      <Card
        icon={Wrench}
        title={block.name ?? 'Tool call'}
        accent={block.status === 'failed' ? 'rose' : 'amber'}
      >
        <div className="space-y-3">
          <StatusBadge status={block.status} />
          {block.args ? (
            <CodeBlock label="Arguments" value={block.args} />
          ) : null}
          {block.result ? (
            <CodeBlock label="Result" value={block.result} />
          ) : null}
        </div>
      </Card>
    )
  }
  if (block.kind === 'reasoning') {
    return (
      <Card icon={Activity} title="Reasoning" accent="violet">
        <div className="whitespace-pre-wrap text-sm leading-7 text-slate-700">
          {block.content}
        </div>
      </Card>
    )
  }
  if (block.kind === 'runtime_event') {
    return (
      <Card icon={TerminalSquare} title={block.title} accent="slate">
        <JsonView value={block.payload} height="180px" />
      </Card>
    )
  }
  return (
    <Card icon={MessageSquare} title="Event" accent="slate">
      <JsonView value={block} height="180px" />
    </Card>
  )
}

function Card({
  icon: Icon,
  title,
  accent,
  children,
}: {
  icon: typeof Bot
  title: string
  accent: 'blue' | 'emerald' | 'amber' | 'rose' | 'violet' | 'slate'
  children: React.ReactNode
}) {
  const accentClass = {
    blue: 'bg-blue-50 text-blue-600',
    emerald: 'bg-emerald-50 text-emerald-600',
    amber: 'bg-amber-50 text-amber-600',
    rose: 'bg-rose-50 text-rose-600',
    violet: 'bg-violet-50 text-violet-600',
    slate: 'bg-slate-100 text-slate-600',
  }[accent]

  return (
    <article className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="mb-3 flex items-center gap-2">
        <span
          className={cn(
            'inline-flex h-8 w-8 items-center justify-center rounded-xl',
            accentClass,
          )}
        >
          <Icon className="h-4 w-4" />
        </span>
        <h3 className="text-sm font-semibold text-slate-900">{title}</h3>
      </div>
      {children}
    </article>
  )
}

function InputPartView({ part }: { part: InputPart }) {
  if (part.type === 'text') {
    return (
      <div className="whitespace-pre-wrap rounded-xl bg-blue-50 p-3 text-sm leading-7 text-slate-800">
        {part.text}
      </div>
    )
  }
  return <JsonView value={part} height="180px" />
}

function CodeBlock({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="mb-1 text-xs font-medium uppercase tracking-wide text-slate-400">
        {label}
      </p>
      <pre className="scrollbar-thin max-h-60 overflow-auto rounded-xl border border-slate-200 bg-slate-50 p-3 text-xs leading-5 text-slate-700">
        {formatMaybeJson(value)}
      </pre>
    </div>
  )
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

function MetricCard({
  label,
  value,
  valueClassName,
}: {
  label: string
  value: string
  valueClassName?: string
}) {
  return (
    <div className="min-w-0 rounded-2xl border border-slate-200 bg-slate-50 p-4">
      <p className="text-xs font-medium uppercase tracking-wide text-slate-400">
        {label}
      </p>
      <p
        className={cn(
          'mt-2 break-words text-lg font-semibold text-slate-950',
          valueClassName,
        )}
        title={value}
      >
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
      <dt className="shrink-0 text-slate-500">{label}</dt>
      <dd
        className={cn(
          'min-w-0 break-words text-right text-slate-800',
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

function formatMaybeJson(value: string) {
  try {
    return JSON.stringify(JSON.parse(value), null, 2)
  } catch {
    return value
  }
}

function formatDate(value?: string | null) {
  if (!value) return 'none'
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(new Date(value))
}

function formatDuration(seconds?: number | null) {
  if (seconds == null) return 'pending'
  if (seconds === 0) return '0s'
  if (seconds % 3600 === 0) return `${seconds / 3600}h`
  if (seconds % 60 === 0) return `${seconds / 60}m`
  return `${seconds}s`
}
