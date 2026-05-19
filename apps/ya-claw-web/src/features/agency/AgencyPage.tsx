import { BrainCircuit, RefreshCcw, RotateCcw } from 'lucide-react'
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
} from '../../api/hooks'
import type { StreamStatus } from '../../lib/status'
import { formatShortId } from '../../lib/utils'
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
  async function clearAgency() {
    if (mutations.clear.isPending) return
    const confirmed = window.confirm(
      'Clear agency state and start a fresh singleton session on the next agency run?',
    )
    if (!confirmed) return
    const response = await mutations.clear.mutateAsync()
    setSelectedRunId(null)
    if (response.new_agency_session_id) {
      await sessionHistory.refetch()
    }
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
            className="inline-flex items-center gap-2 rounded-xl border border-rose-200 bg-white px-3 py-2 font-medium text-rose-700 shadow-sm transition hover:bg-rose-50 disabled:cursor-not-allowed disabled:opacity-60"
            onClick={() => void clearAgency()}
            disabled={mutations.clear.isPending}
            title="Clear agency state and start fresh on the next agency run."
          >
            <RotateCcw className="h-3.5 w-3.5" />
            {mutations.clear.isPending ? 'Clearing' : 'Clear agency'}
          </button>
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
              historyLoadingDisabled={Boolean(selectedRunId)}
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
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
