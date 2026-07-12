import { BrainCircuit, RefreshCcw, RotateCcw } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { Group, Panel } from 'react-resizable-panels'

import { navigateApp } from '../../app/navigation'
import {
  useAgencyConfigQuery,
  useAgencyFiresQuery,
  useAgencyMutations,
  useAgencyStatusQuery,
  useRunQuery,
  useRunTraceQuery,
  useSessionHistoryQuery,
} from '../../api/hooks'
import { QueryError, QuerySkeleton } from '../../components/ui'
import { ConfirmDialog } from '../../components/ui/ConfirmDialog'
import type { StreamStatus } from '../../lib/status'
import { buildAgencyPath } from '../../lib/urlState'
import { formatShortId } from '../../lib/utils'
import { useLayoutStore } from '../../stores/layoutStore'
import {
  buildTimelineFromRuns,
  reduceAguiEvent,
} from '../chat/agui/eventReducer'
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
import { dedupeRuns, orderRuns } from './utils'

export function AgencyPage() {
  const config = useAgencyConfigQuery()
  const status = useAgencyStatusQuery()
  const fires = useAgencyFiresQuery()
  const mutations = useAgencyMutations()
  const routeSessionId = useLayoutStore(
    (state) => state.selectedAgencySessionId,
  )
  const routeRunId = useLayoutStore((state) =>
    state.route === 'agency' ? state.selectedRunId : null,
  )
  const [fireSearch, setFireSearch] = useState('')
  const [selectedRunId, setSelectedRunId] = useState<string | null>(routeRunId)

  const currentAgencySessionId =
    status.data?.agency_session_id ?? config.data?.agency_session_id ?? null
  const agencySessionId = routeSessionId ?? currentAgencySessionId
  const viewingHistoricalSession = Boolean(
    routeSessionId && routeSessionId !== currentAgencySessionId,
  )
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
    setSelectedRunId(routeRunId)
  }, [agencySessionId, routeRunId])

  useEffect(() => {
    const preferredRunId = viewingHistoricalSession
      ? null
      : (status.data?.active_run_id ?? status.data?.latest_run_id ?? null)
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
    viewingHistoricalSession,
  ])

  const attemptedLatestRunRef = useRef<string | null>(null)
  const refetchSessionHistory = sessionHistory.refetch
  useEffect(() => {
    const latestRunId = viewingHistoricalSession
      ? null
      : (status.data?.latest_run_id ?? status.data?.active_run_id ?? null)
    if (!latestRunId || agencyRuns.some((run) => run.id === latestRunId)) {
      attemptedLatestRunRef.current = null
      return
    }
    if (attemptedLatestRunRef.current === latestRunId) return
    attemptedLatestRunRef.current = latestRunId
    void refetchSessionHistory()
  }, [
    agencyRuns,
    refetchSessionHistory,
    status.data?.active_run_id,
    status.data?.latest_run_id,
    viewingHistoricalSession,
  ])

  useEffect(() => {
    const runSessionId = selectedRunDetail?.run.session_id
    if (!selectedRunId || !runSessionId || runSessionId === agencySessionId) {
      return
    }
    const path = buildAgencyPath(runSessionId, selectedRunId)
    if (window.location.pathname !== path) navigateApp(path, true)
  }, [agencySessionId, selectedRunDetail, selectedRunId])

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

  const selectedRunSessionId =
    selectedRunDetail?.run.session_id ?? agencySessionId
  const live = useRunEventStream(
    selectedRunId,
    selectedRun?.status ?? null,
    selectedRunSessionId,
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
  const selectedRunTimeline = useMemo(() => {
    const baseTimeline = buildTimelineFromRuns(
      selectedRun ? [selectedRun] : [],
      {
        includeRuntimeEvents: false,
      },
    )
    return effectiveLiveEvents.reduce(
      (state, event) =>
        reduceAguiEvent(state, event, { includeRuntimeEvents: false }),
      baseTimeline,
    )
  }, [effectiveLiveEvents, selectedRun])
  const timeline = selectedRunId ? selectedRunTimeline : history.timeline
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
  const foundationQueries = [config, status, fires]
  const queryStates = [
    ...foundationQueries,
    sessionHistory,
    selectedRunQuery,
    selectedRunTrace,
  ]
  const fatalFoundationQuery = foundationQueries.find(
    (query) => query.isError && query.data === undefined,
  )
  const nonBlockingFailedQuery = queryStates.find(
    (query) => query.isError && query !== fatalFoundationQuery,
  )
  const foundationLoading = foundationQueries.some(
    (query) => query.isLoading && query.data === undefined,
  )

  async function refetchAgencyData() {
    const requests: Array<Promise<unknown>> = [
      config.refetch(),
      status.refetch(),
      fires.refetch(),
    ]
    if (agencySessionId) requests.push(sessionHistory.refetch())
    if (selectedRunId) {
      requests.push(selectedRunQuery.refetch(), selectedRunTrace.refetch())
    }
    await Promise.all(requests)
  }

  function selectAgencyRun(
    runId: string | null,
    sessionId: string | null = agencySessionId,
  ) {
    setSelectedRunId(runId)
    if (!sessionId) return
    const path = buildAgencyPath(sessionId, runId)
    if (window.location.pathname !== path) navigateApp(path)
  }

  function selectAgencyFireRun(sessionId: string | null, runId: string | null) {
    selectAgencyRun(runId, sessionId ?? agencySessionId)
  }

  async function clearAgency() {
    if (mutations.clear.isPending) return
    const response = await mutations.clear.mutateAsync()
    setSelectedRunId(null)
    navigateApp('/automation/agency', true)
    if (response.new_agency_session_id) {
      await sessionHistory.refetch()
    }
  }

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden bg-slate-100">
      <div className="flex shrink-0 flex-col gap-3 border-b border-slate-200 bg-white px-3 py-3 sm:h-16 sm:flex-row sm:items-center sm:justify-between sm:px-5 sm:py-0">
        <div className="flex min-w-0 items-center gap-2 text-sm text-slate-600 sm:gap-3">
          <h1 className="inline-flex shrink-0 items-center gap-2 text-base font-semibold text-slate-950">
            <BrainCircuit className="h-4 w-4" />
            Proactive agent
          </h1>
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
          <ConfirmDialog
            title="Clear proactive agent state?"
            description="This clears the current singleton agency state. YA Claw will start a fresh agency session on the next proactive run."
            confirmLabel="Clear agency"
            danger
            pending={mutations.clear.isPending}
            onConfirm={clearAgency}
            trigger={
              <button
                type="button"
                className="inline-flex items-center gap-2 rounded-xl border border-rose-200 bg-white px-3 py-2 font-medium text-rose-700 shadow-sm transition hover:bg-rose-50 disabled:cursor-not-allowed disabled:opacity-60"
                disabled={mutations.clear.isPending}
                title="Clear agency state and start fresh on the next agency run."
              >
                <RotateCcw className="h-3.5 w-3.5" />
                {mutations.clear.isPending ? 'Clearing' : 'Clear agency'}
              </button>
            }
          />
          <button
            type="button"
            className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 font-medium text-slate-700 shadow-sm transition hover:bg-slate-50"
            onClick={() => void refetchAgencyData()}
          >
            <RefreshCcw className="h-3.5 w-3.5" />
            Refresh
          </button>
        </div>
      </div>

      {viewingHistoricalSession ? (
        <div
          className="flex shrink-0 flex-wrap items-center justify-between gap-3 border-b border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900 sm:px-5"
          role="status"
        >
          <span>
            Viewing historical proactive session{' '}
            <span className="mono font-semibold">
              {formatShortId(agencySessionId, 12)}
            </span>
          </span>
          <button
            type="button"
            className="font-semibold underline underline-offset-2"
            onClick={() => navigateApp('/automation/agency', true)}
          >
            Return to current session
          </button>
        </div>
      ) : null}

      {fatalFoundationQuery ? (
        <div className="shrink-0 p-3 sm:p-4">
          <QueryError
            title="Agency data could not be loaded"
            error={fatalFoundationQuery.error}
            onRetry={() => void refetchAgencyData()}
          />
        </div>
      ) : null}
      {!fatalFoundationQuery && nonBlockingFailedQuery ? (
        <div className="shrink-0 p-3 sm:px-4 sm:py-2">
          <QueryError
            compact
            title="Some agency details could not be loaded"
            error={nonBlockingFailedQuery.error}
            onRetry={() => void refetchAgencyData()}
          />
        </div>
      ) : null}
      {foundationLoading ? (
        <div className="shrink-0 p-3 sm:p-4">
          <QuerySkeleton rows={2} />
        </div>
      ) : null}

      {!fatalFoundationQuery && !foundationLoading ? (
        <>
          <div
            className="hidden min-h-0 flex-1 lg:block"
            data-testid="agency-desktop-layout"
          >
            <Group orientation="horizontal" className="h-full min-h-0">
              <Panel defaultSize="26%" minSize="260px" maxSize="36%">
                <AgencyFireList
                  fires={fires.data?.fires ?? []}
                  loading={fires.isLoading}
                  search={fireSearch}
                  selectedRunId={selectedRunId}
                  onSearchChange={setFireSearch}
                  onSelectRun={selectAgencyFireRun}
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
                        onSelectRun={selectAgencyRun}
                      />
                      <AgencyStatusBar
                        config={config.data}
                        status={status.data}
                      />
                      <AgencyConfigBar config={config.data} />
                      <TimelinePanel
                        timeline={timeline}
                        loading={contentLoading}
                        artifactsPruned={selectedRunArtifactsPruned}
                        history={history}
                        loadingOlder={sessionHistory.isFetchingNextPage}
                        onLoadOlder={() => sessionHistory.fetchNextPage()}
                        historyLoadingDisabled={Boolean(selectedRunId)}
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
                    />
                  </Panel>
                </Group>
              </Panel>
            </Group>
          </div>

          <div
            className="flex min-h-0 flex-1 flex-col overflow-hidden lg:hidden"
            data-testid="agency-mobile-layout"
          >
            <div className="h-[min(14rem,38%)] min-h-40 shrink-0">
              <AgencyFireList
                fires={fires.data?.fires ?? []}
                loading={fires.isLoading}
                search={fireSearch}
                selectedRunId={selectedRunId}
                onSearchChange={setFireSearch}
                onSelectRun={selectAgencyFireRun}
              />
            </div>
            <div className="scrollbar-thin min-h-0 flex-1 overflow-y-auto overscroll-contain">
              <RunStrip
                runs={runs}
                selectedRunId={selectedRunId}
                history={history}
                loadingOlder={sessionHistory.isFetchingNextPage}
                onLoadOlder={() => void sessionHistory.fetchNextPage()}
                onSelectRun={selectAgencyRun}
              />
              <AgencyStatusBar config={config.data} status={status.data} />
              <AgencyConfigBar config={config.data} />
              <div className="h-80 min-h-64">
                <TimelinePanel
                  timeline={timeline}
                  loading={contentLoading}
                  artifactsPruned={selectedRunArtifactsPruned}
                  history={history}
                  loadingOlder={sessionHistory.isFetchingNextPage}
                  onLoadOlder={() => sessionHistory.fetchNextPage()}
                  historyLoadingDisabled={Boolean(selectedRunId)}
                />
              </div>
              <div className="h-[28rem] min-h-[22rem] border-t border-slate-200">
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
                />
              </div>
            </div>
          </div>
        </>
      ) : null}
    </div>
  )
}
