import { BrainCircuit, RefreshCcw } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { Group, Panel } from 'react-resizable-panels'

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
import type { StreamStatus } from '../../lib/status'
import { formatShortId } from '../../lib/utils'
import type { RunSummary } from '../../types'
import { buildTimelineFromRuns } from '../chat/agui/eventReducer'
import { LivePill } from '../chat/debug/LivePill'
import { ResizeHandle } from '../chat/debug/ResizeHandle'
import { RunStrip } from '../chat/debug/RunControls'
import { TimelinePanel } from '../chat/debug/TimelinePanel'
import { isTerminalAguiEvent } from '../chat/eventUtils'
import { mergeSessionHistoryPages } from '../chat/sessionHistory'
import { useRunEventStream } from '../chat/useRunEventStream'
import { AgencyFireList } from './AgencyFireList'
import { AgencyInspectorPanel } from './AgencyInspectorPanel'
import { AgencyConfigBar, AgencyStatusBar } from './AgencyStatusBars'
import { ManualFireComposer } from './ManualFireComposer'
import { dedupeRuns, orderRuns } from './utils'

export function AgencyPage() {
  const config = useAgencyConfigQuery()
  const status = useAgencyStatusQuery()
  const fires = useAgencyFiresQuery()
  const sessions = useSessionsQuery()
  const mutations = useAgencyMutations()
  const [sourceSessionId, setSourceSessionId] = useState('')
  const [prompt, setPrompt] = useState('')
  const [fireSearch, setFireSearch] = useState('')
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null)

  const agencySessionId =
    status.data?.agency_session_id ?? config.data?.agency_session_id ?? null
  const sessionHistory = useSessionHistoryQuery(agencySessionId, {
    runsLimit: 6,
  })
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

  const historyPages = sessionHistory.data?.pages
  const agencyRuns = useMemo(
    () =>
      orderRuns(
        dedupeRuns(historyPages?.flatMap((page) => page.session.runs) ?? []),
      ),
    [historyPages],
  )

  useEffect(() => {
    const preferredRunId =
      status.data?.active_run_id ?? status.data?.latest_run_id ?? null
    if (preferredRunId && !selectedRunId) {
      setSelectedRunId(preferredRunId)
      return
    }
    if (!selectedRunId && agencyRuns.length > 0) {
      setSelectedRunId(agencyRuns[agencyRuns.length - 1].id)
    }
  }, [
    agencyRuns,
    selectedRunId,
    status.data?.active_run_id,
    status.data?.latest_run_id,
  ])

  useEffect(() => {
    const latestRunId = status.data?.latest_run_id ?? status.data?.active_run_id
    if (latestRunId && !agencyRuns.some((run) => run.id === latestRunId)) {
      void sessionHistory.refetch()
    }
  }, [
    agencyRuns,
    sessionHistory,
    status.data?.active_run_id,
    status.data?.latest_run_id,
  ])

  const selectedRun = useMemo(() => {
    const detailRun = selectedRunDetail?.run
    if (detailRun) {
      return {
        ...detailRun,
        message: selectedRunDetail.message ?? detailRun.message ?? null,
      }
    }
    return agencyRuns.find((run) => run.id === selectedRunId) ?? null
  }, [agencyRuns, selectedRunDetail, selectedRunId])

  const live = useRunEventStream(
    selectedRunId,
    selectedRun?.status ?? null,
    agencySessionId,
  )
  const liveEvents = useMemo(
    () => (selectedRunId ? live.events : []),
    [live.events, selectedRunId],
  )
  const streamStatus: StreamStatus = selectedRunId ? live.status : 'idle'
  const replayEvents = useMemo(
    () => selectedRunDetail?.message ?? selectedRun?.message ?? [],
    [selectedRun, selectedRunDetail],
  )
  const hasCommittedTerminalEvent = useMemo(
    () => replayEvents.some((event) => isTerminalAguiEvent(event)),
    [replayEvents],
  )
  const effectiveLiveEvents = useMemo(
    () => (hasCommittedTerminalEvent ? [] : liveEvents),
    [hasCommittedTerminalEvent, liveEvents],
  )

  const history = useMemo(
    () => mergeSessionHistoryPages(historyPages, effectiveLiveEvents),
    [effectiveLiveEvents, historyPages],
  )
  const runs = history.runs.length ? history.runs : agencyRuns
  const timeline = history.timeline.blocks.length
    ? history.timeline
    : buildTimelineFromRuns(selectedRun ? [selectedRun] : [], {
        includeRuntimeEvents: false,
      })
  const runEvents = history.events.length
    ? history.events
    : [...replayEvents, ...effectiveLiveEvents]
  const contentLoading =
    Boolean(agencySessionId && sessionHistory.isLoading) ||
    Boolean(selectedRunId && selectedRunQuery.isLoading)
  const selectedRunArtifactsPruned = Boolean(
    selectedRun &&
      selectedRun.status !== 'queued' &&
      selectedRun.status !== 'running' &&
      selectedRunDetail?.run.has_message === false &&
      replayEvents.length === 0,
  )
  const conversationSessions = useMemo(
    () =>
      (sessions.data ?? []).filter(
        (session) => session.session_type === 'conversation',
      ),
    [sessions.data],
  )
  const activeAgencyRun = resolveActiveRun(status.data?.active_run, selectedRun)

  async function sendManualFire() {
    if (mutations.trigger.isPending) return
    await mutations.trigger.mutateAsync({
      kind: 'manual',
      source_session_id: sourceSessionId || null,
      client_token: `web-${Date.now()}`,
      prompt: prompt.trim() || null,
    })
    setPrompt('')
  }

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden bg-slate-100">
      <div className="flex shrink-0 flex-col gap-3 border-b border-slate-200 bg-white px-3 py-3 sm:h-16 sm:flex-row sm:items-center sm:justify-between sm:px-5 sm:py-0">
        <div className="flex min-w-0 items-center gap-2 text-sm text-slate-600 sm:gap-3">
          <span className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-3 py-1.5 font-medium text-slate-700">
            <BrainCircuit className="h-3.5 w-3.5" />
            Agency
          </span>
          {selectedRun ? (
            <span className="mono truncate text-xs text-slate-500">
              Run {selectedRun.sequence_no} ·{' '}
              {formatShortId(selectedRun.id, 12)}
            </span>
          ) : (
            <span className="text-xs text-slate-500">
              Inspect singleton agency fires, runs, and replay
            </span>
          )}
        </div>
        <div className="flex shrink-0 items-center gap-2 overflow-x-auto text-xs text-slate-500">
          <LivePill
            status={streamStatus}
            eventCount={effectiveLiveEvents.length}
          />
          <button
            type="button"
            className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 font-medium text-slate-700 shadow-sm transition hover:bg-slate-50"
            onClick={() => {
              void Promise.all([
                config.refetch(),
                status.refetch(),
                fires.refetch(),
                sessionHistory.refetch(),
                selectedRunId ? selectedRunQuery.refetch() : Promise.resolve(),
                selectedRunId ? selectedRunTrace.refetch() : Promise.resolve(),
              ])
            }}
          >
            <RefreshCcw className="h-3.5 w-3.5" />
            Refresh
          </button>
        </div>
      </div>

      <Group orientation="horizontal" className="hidden min-h-0 flex-1 lg:flex">
        <Panel defaultSize="26%" minSize="260px" maxSize="36%">
          <AgencyFireList
            fires={fires.data?.fires ?? []}
            loading={fires.isLoading}
            search={fireSearch}
            selectedRunId={selectedRunId}
            onSearchChange={setFireSearch}
            onSelectRun={setSelectedRunId}
          />
        </Panel>
        <ResizeHandle />
        <Panel defaultSize="74%" minSize="64%">
          <Group orientation="horizontal" className="h-full min-h-0">
            <Panel defaultSize="68%" minSize="44%">
              <div className="flex h-full min-h-0 flex-col overflow-hidden">
                <RunStrip
                  runs={runs}
                  selectedRunId={selectedRunId}
                  history={history}
                  loadingOlder={sessionHistory.isFetchingNextPage}
                  onLoadOlder={() => void sessionHistory.fetchNextPage()}
                  onSelectRun={setSelectedRunId}
                />
                <AgencyStatusBar config={config.data} status={status.data} />
                <AgencyConfigBar config={config.data} />
                <TimelinePanel
                  timeline={timeline}
                  loading={contentLoading}
                  artifactsPruned={selectedRunArtifactsPruned}
                  history={history}
                  loadingOlder={sessionHistory.isFetchingNextPage}
                  onLoadOlder={() => sessionHistory.fetchNextPage()}
                />
                <ManualFireComposer
                  sessions={conversationSessions}
                  selectedSourceSessionId={sourceSessionId}
                  prompt={prompt}
                  activeRun={activeAgencyRun}
                  pending={mutations.trigger.isPending}
                  onSourceSessionChange={setSourceSessionId}
                  onPromptChange={setPrompt}
                  onSubmit={sendManualFire}
                />
              </div>
            </Panel>
            <ResizeHandle />
            <Panel defaultSize="32%" minSize="280px">
              <AgencyInspectorPanel
                config={config.data}
                status={status.data}
                fires={fires.data?.fires ?? []}
                run={selectedRun}
                detail={selectedRunDetail}
                trace={selectedTrace}
                traceLoading={
                  Boolean(selectedRunId) &&
                  selectedRunTrace.isFetching &&
                  !selectedTrace
                }
                events={runEvents}
                streamStatus={streamStatus}
                liveEventCount={effectiveLiveEvents.length}
                loading={contentLoading}
                artifactsPruned={selectedRunArtifactsPruned}
              />
            </Panel>
          </Group>
        </Panel>
      </Group>

      <div className="grid min-h-0 flex-1 grid-rows-[14rem_minmax(0,1fr)] overflow-hidden lg:hidden">
        <AgencyFireList
          fires={fires.data?.fires ?? []}
          loading={fires.isLoading}
          search={fireSearch}
          selectedRunId={selectedRunId}
          onSearchChange={setFireSearch}
          onSelectRun={setSelectedRunId}
        />
        <div className="min-h-0 overflow-hidden">
          <div className="flex h-full min-h-0 flex-col overflow-hidden">
            <RunStrip
              runs={runs}
              selectedRunId={selectedRunId}
              history={history}
              loadingOlder={sessionHistory.isFetchingNextPage}
              onLoadOlder={() => void sessionHistory.fetchNextPage()}
              onSelectRun={setSelectedRunId}
            />
            <AgencyStatusBar config={config.data} status={status.data} />
            <AgencyConfigBar config={config.data} />
            <TimelinePanel
              timeline={timeline}
              loading={contentLoading}
              artifactsPruned={selectedRunArtifactsPruned}
              history={history}
              loadingOlder={sessionHistory.isFetchingNextPage}
              onLoadOlder={() => sessionHistory.fetchNextPage()}
            />
            <div className="min-h-[22rem] shrink-0 border-t border-slate-200">
              <AgencyInspectorPanel
                config={config.data}
                status={status.data}
                fires={fires.data?.fires ?? []}
                run={selectedRun}
                detail={selectedRunDetail}
                trace={selectedTrace}
                traceLoading={
                  Boolean(selectedRunId) &&
                  selectedRunTrace.isFetching &&
                  !selectedTrace
                }
                events={runEvents}
                streamStatus={streamStatus}
                liveEventCount={effectiveLiveEvents.length}
                loading={contentLoading}
                artifactsPruned={selectedRunArtifactsPruned}
              />
            </div>
            <ManualFireComposer
              sessions={conversationSessions}
              selectedSourceSessionId={sourceSessionId}
              prompt={prompt}
              activeRun={activeAgencyRun}
              pending={mutations.trigger.isPending}
              onSourceSessionChange={setSourceSessionId}
              onPromptChange={setPrompt}
              onSubmit={sendManualFire}
            />
          </div>
        </div>
      </div>
    </div>
  )
}

function resolveActiveRun(
  activeRun: RunSummary | null | undefined,
  selectedRun: RunSummary | null,
) {
  if (activeRun) return activeRun
  if (selectedRun?.status === 'running' || selectedRun?.status === 'queued') {
    return selectedRun
  }
  return null
}
