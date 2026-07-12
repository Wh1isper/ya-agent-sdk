import { Link, useRouterState } from '@tanstack/react-router'
import { ExternalLink, Plus, RefreshCcw } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { Group, Panel } from 'react-resizable-panels'

import {
  useRunQuery,
  useSessionHistoryQuery,
  useSessionQuery,
  useSessionWorkspaceQuery,
  useSessionsQuery,
  useWorkspaceRuntimeQuery,
} from '../../api/hooks'
import { QueryError } from '../../components/ui'
import type { StreamStatus } from '../../lib/status'
import {
  buildChatPath,
  parseUrlSelection,
  replaceBrowserPath,
} from '../../lib/urlState'
import { formatShortId } from '../../lib/utils'
import { useLayoutStore } from '../../stores/layoutStore'
import { buildTimelineFromRuns, reduceAguiEvent } from './agui/eventReducer'
import { isTerminalAguiEvent } from './eventUtils'
import { filterActivitySessions, type ActivityFilters } from './activityFilters'
import { validateActivityRunSelection } from './activityRunSelection'
import { mergeSessionHistoryPages } from './sessionHistory'
import { Composer } from './debug/Composer'
import { LivePill } from './debug/LivePill'
import {
  MemoryStatusBar,
  RunControlBar,
  RunStrip,
  WorkspaceStatusBar,
} from './debug/RunControls'
import { ResizeHandle } from './debug/ResizeHandle'
import { SessionList } from './debug/SessionList'
import { TimelinePanel } from './debug/TimelinePanel'
import { useRunEventStream } from './useRunEventStream'

export function DebugPage() {
  const pathname = useRouterState({
    select: (state) => state.location.pathname,
  })
  const isActivityIndex = pathname === '/activity'
  const routeSelection = useMemo(() => parseUrlSelection(pathname), [pathname])
  const selectedSessionId = isActivityIndex
    ? null
    : routeSelection.selectedSessionId
  const selectedRunId = selectedSessionId ? routeSelection.selectedRunId : null
  const selectSession = useLayoutStore((state) => state.selectSession)
  const selectRun = useLayoutStore((state) => state.selectRun)
  const [sessionSearch, setSessionSearch] = useState('')
  const [statusFilter, setStatusFilter] =
    useState<ActivityFilters['status']>('all')
  const [sourceFilter, setSourceFilter] =
    useState<ActivityFilters['source']>('all')
  const [profileFilter, setProfileFilter] = useState('all')
  const [timeFilter, setTimeFilter] = useState<ActivityFilters['time']>('all')
  const [isComposingNew, setIsComposingNew] = useState(false)
  const autoSelectedSessionRef = useRef(false)
  const sessions = useSessionsQuery()
  const {
    fetchNextPage: fetchNextSessionPage,
    hasNextPage: hasNextSessionPage,
    isFetchingNextPage: isFetchingNextSessionPage,
  } = sessions
  const sessionFilterActive =
    Boolean(sessionSearch.trim()) ||
    statusFilter !== 'all' ||
    sourceFilter !== 'all' ||
    profileFilter !== 'all' ||
    timeFilter !== 'all'
  const workspaceRuntime = useWorkspaceRuntimeQuery()
  const selectedSessionWorkspace = useSessionWorkspaceQuery(selectedSessionId)
  const selectedSession = useSessionQuery(selectedSessionId, {
    runsLimit: 1,
    includeMessage: false,
    includeInputParts: false,
    includeHeadPayload: false,
  })
  const activeSessionData = selectedSessionId ? selectedSession.data : undefined
  const resolvedRunId =
    selectedRunId ??
    activeSessionData?.session.active_run_id ??
    activeSessionData?.session.head_run_id ??
    null
  const sessionHistory = useSessionHistoryQuery(selectedSessionId, {
    runsLimit: 3,
  })
  const selectedRun = useRunQuery(resolvedRunId)
  const { validatedRunId, ownershipMismatch: runOwnershipMismatch } =
    validateActivityRunSelection(
      resolvedRunId,
      selectedSessionId,
      selectedRun.data?.run,
    )
  const activeRunData = validatedRunId ? selectedRun.data : undefined
  const live = useRunEventStream(
    validatedRunId,
    activeRunData?.run.status ?? null,
    selectedSessionId,
  )
  const liveEvents = useMemo(
    () => (validatedRunId ? live.events : []),
    [live.events, validatedRunId],
  )
  const streamStatus: StreamStatus = validatedRunId ? live.status : 'idle'
  const contentLoading =
    Boolean(selectedSessionId && selectedSession.isLoading) ||
    Boolean(resolvedRunId && selectedRun.isLoading) ||
    Boolean(selectedSessionId && sessionHistory.isLoading)
  const detailQueries = [selectedSession, sessionHistory, selectedRun]
  const failedDetailQuery = detailQueries.find((query) => query.isError)
  const workspaceError =
    workspaceRuntime.error ?? selectedSessionWorkspace.error

  useEffect(() => {
    if (
      sessionFilterActive &&
      hasNextSessionPage &&
      !isFetchingNextSessionPage
    ) {
      void fetchNextSessionPage()
    }
  }, [
    fetchNextSessionPage,
    hasNextSessionPage,
    isFetchingNextSessionPage,
    sessionFilterActive,
  ])

  useEffect(() => {
    const firstSessionId = sessions.data?.[0]?.id
    if (
      isActivityIndex &&
      !selectedSessionId &&
      !isComposingNew &&
      firstSessionId &&
      !autoSelectedSessionRef.current
    ) {
      autoSelectedSessionRef.current = true
      useLayoutStore.setState({
        route: 'debug',
        selectedSessionId: firstSessionId,
        selectedRunId: null,
        selectedDebugSessionId: firstSessionId,
        selectedDebugRunId: null,
      })
      replaceBrowserPath(buildChatPath(firstSessionId, null, 'debug'))
    }
  }, [isActivityIndex, isComposingNew, selectedSessionId, sessions.data])

  useEffect(() => {
    if (selectedSessionId) setIsComposingNew(false)
  }, [selectedSessionId])

  useEffect(() => {
    if (!selectedSessionId || selectedRunId) return
    const nextRunId =
      activeSessionData?.session.active_run_id ??
      activeSessionData?.session.head_run_id ??
      null
    if (nextRunId) {
      useLayoutStore.setState({
        route: 'debug',
        selectedSessionId,
        selectedRunId: nextRunId,
        selectedDebugSessionId: selectedSessionId,
        selectedDebugRunId: nextRunId,
      })
      replaceBrowserPath(buildChatPath(selectedSessionId, nextRunId, 'debug'))
    }
  }, [activeSessionData, selectedRunId, selectedSessionId])

  const profileOptions = useMemo(
    () =>
      Array.from(
        new Set(
          (sessions.data ?? []).map(
            (session) => session.profile_name?.trim() || 'default',
          ),
        ),
      ).sort(),
    [sessions.data],
  )
  const filteredSessions = useMemo(
    () =>
      filterActivitySessions(sessions.data ?? [], {
        search: sessionSearch,
        status: statusFilter,
        source: sourceFilter,
        profile: profileFilter,
        time: timeFilter,
      }),
    [
      profileFilter,
      sessionSearch,
      sessions.data,
      sourceFilter,
      statusFilter,
      timeFilter,
    ],
  )

  const historyPages = sessionHistory.data?.pages
  const historyRuns = useMemo(
    () => mergeSessionHistoryPages(historyPages).runs,
    [historyPages],
  )
  const activeRun = useMemo(
    () =>
      activeRunData?.run ??
      historyRuns.find((item) => item.id === validatedRunId) ??
      activeSessionData?.session.runs.find(
        (item) => item.id === validatedRunId,
      ) ??
      null,
    [activeRunData, activeSessionData, historyRuns, validatedRunId],
  )
  const replayEvents = useMemo(
    () =>
      activeRunData?.message ??
      activeRun?.message ??
      activeSessionData?.message ??
      [],
    [activeRun, activeRunData, activeSessionData],
  )
  const selectedRunArtifactsPruned = Boolean(
    activeRun &&
    activeRun.status !== 'queued' &&
    activeRun.status !== 'running' &&
    !activeRunData?.run.has_message &&
    replayEvents.length === 0,
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
  const selectedRunTimeline = useMemo(() => {
    const baseTimeline = buildTimelineFromRuns(activeRun ? [activeRun] : [], {
      includeRuntimeEvents: false,
    })
    return effectiveLiveEvents.reduce(
      (state, event) =>
        reduceAguiEvent(state, event, { includeRuntimeEvents: false }),
      baseTimeline,
    )
  }, [activeRun, effectiveLiveEvents])
  const timeline = validatedRunId ? selectedRunTimeline : history.timeline
  const runs = history.runs.length
    ? history.runs
    : (activeSessionData?.session.runs ?? [])

  async function refetchActivityDetails() {
    const requests: Array<Promise<unknown>> = [workspaceRuntime.refetch()]
    if (selectedSessionId) {
      requests.push(
        selectedSessionWorkspace.refetch(),
        selectedSession.refetch(),
        sessionHistory.refetch(),
      )
    }
    if (resolvedRunId) requests.push(selectedRun.refetch())
    await Promise.all(requests)
  }

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden bg-slate-100">
      <div className="flex shrink-0 flex-col gap-3 border-b border-slate-200 bg-white px-3 py-3 md:h-16 md:flex-row md:items-center md:justify-between md:px-5 md:py-0">
        <div className="flex min-w-0 items-center gap-2 text-sm text-slate-600 sm:gap-3">
          <h1 className="shrink-0 text-base font-semibold text-slate-950">
            Activity
          </h1>
          <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1.5 font-medium text-slate-700">
            {sessions.data?.length ?? 0} sessions
          </span>
          {activeRun ? (
            <span className="mono truncate text-xs text-slate-500">
              Run {activeRun.sequence_no} · {formatShortId(activeRun.id, 12)}
            </span>
          ) : (
            <span className="text-xs text-slate-500">
              Select a session to inspect runtime details
            </span>
          )}
        </div>
        <div className="flex shrink-0 items-center gap-2 text-xs text-slate-500">
          <LivePill
            status={streamStatus}
            eventCount={effectiveLiveEvents.length}
          />
          {selectedSessionId ? (
            <Link
              to={
                validatedRunId
                  ? '/conversations/sessions/$sessionId/runs/$runId'
                  : '/conversations/sessions/$sessionId'
              }
              params={
                validatedRunId
                  ? { sessionId: selectedSessionId, runId: validatedRunId }
                  : { sessionId: selectedSessionId }
              }
              className="inline-flex h-10 w-10 items-center justify-center gap-2 rounded-xl border border-slate-200 bg-white font-medium text-slate-700 shadow-sm transition hover:bg-slate-50 md:h-auto md:w-auto md:px-3 md:py-2"
              aria-label="Open conversation detail"
            >
              <ExternalLink className="h-3.5 w-3.5" />
              <span className="hidden md:inline">Conversation detail</span>
            </Link>
          ) : null}
          <button
            type="button"
            className="inline-flex h-10 w-10 items-center justify-center gap-2 rounded-xl border border-slate-200 bg-white font-medium text-slate-700 shadow-sm transition hover:bg-slate-50 md:h-auto md:w-auto md:px-3 md:py-2"
            aria-label="New diagnostic session"
            onClick={() => {
              autoSelectedSessionRef.current = true
              setIsComposingNew(true)
              selectSession(null)
              selectRun(null)
            }}
          >
            <Plus className="h-3.5 w-3.5" />
            <span className="hidden md:inline">New diagnostic session</span>
          </button>
          <button
            type="button"
            className="inline-flex h-10 w-10 items-center justify-center gap-2 rounded-xl border border-slate-200 bg-white font-medium text-slate-700 shadow-sm transition hover:bg-slate-50 md:h-auto md:w-auto md:px-3 md:py-2"
            aria-label="Refresh activity"
            onClick={() => sessions.refetch()}
          >
            <RefreshCcw className="h-3.5 w-3.5" />
            <span className="hidden md:inline">Refresh</span>
          </button>
        </div>
      </div>

      {failedDetailQuery ? (
        <div className="shrink-0 p-3 sm:p-4">
          <QueryError
            compact
            title="Some activity details could not be loaded"
            error={failedDetailQuery.error}
            onRetry={() => void refetchActivityDetails()}
          />
        </div>
      ) : null}

      {runOwnershipMismatch ? (
        <div className="shrink-0 p-3 sm:p-4" role="alert">
          <div className="rounded-2xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-900">
            <p className="font-semibold">Run does not belong to this session</p>
            <p className="mt-1 text-rose-700">
              Live events and run controls are disabled to keep session data
              isolated.
            </p>
            <button
              type="button"
              className="mt-3 rounded-xl border border-rose-200 bg-white px-3 py-2 text-xs font-medium text-rose-700 transition hover:bg-rose-100"
              onClick={() => selectRun(null)}
            >
              Return to selected session
            </button>
          </div>
        </div>
      ) : null}

      <>
        <div
          className="hidden min-h-0 flex-1 lg:block"
          data-testid="activity-desktop-layout"
        >
          <Group orientation="horizontal" className="h-full min-h-0">
            <Panel defaultSize="26%" minSize="260px" maxSize="36%">
              <SessionList
                sessions={filteredSessions}
                selectedSessionId={selectedSessionId}
                ariaLabel="Desktop activity sessions"
                search={sessionSearch}
                loading={sessions.isLoading}
                loadingMore={sessions.isFetchingNextPage}
                hasMore={Boolean(sessions.hasNextPage)}
                error={sessions.error}
                onRetry={() => void sessions.refetch()}
                onLoadMore={() => sessions.fetchNextPage()}
                onSearchChange={setSessionSearch}
                filters={{
                  status: statusFilter,
                  source: sourceFilter,
                  profile: profileFilter,
                  time: timeFilter,
                }}
                profileOptions={profileOptions}
                onFilterChange={(filter, value) => {
                  if (filter === 'status')
                    setStatusFilter(value as ActivityFilters['status'])
                  if (filter === 'source')
                    setSourceFilter(value as ActivityFilters['source'])
                  if (filter === 'profile') setProfileFilter(value)
                  if (filter === 'time')
                    setTimeFilter(value as ActivityFilters['time'])
                }}
                onClearFilters={() => {
                  setSessionSearch('')
                  setStatusFilter('all')
                  setSourceFilter('all')
                  setProfileFilter('all')
                  setTimeFilter('all')
                }}
                onSelect={(session) => {
                  selectSession(session.id)
                  selectRun(
                    session.active_run_id ??
                      session.head_run_id ??
                      session.latest_run?.id ??
                      null,
                  )
                }}
              />
            </Panel>
            <ResizeHandle />
            <Panel defaultSize="74%" minSize="64%">
              <Group orientation="horizontal" className="h-full min-h-0">
                <Panel defaultSize="68%" minSize="44%">
                  <div className="flex h-full min-h-0 flex-col overflow-hidden">
                    <RunStrip
                      runs={runs}
                      selectedRunId={validatedRunId}
                      history={history}
                      loadingOlder={sessionHistory.isFetchingNextPage}
                      onLoadOlder={() => void sessionHistory.fetchNextPage()}
                      onSelectRun={selectRun}
                    />
                    <RunControlBar
                      sessionId={selectedSessionId}
                      run={activeRunData?.run ?? null}
                      onSelectRun={selectRun}
                    />
                    <WorkspaceStatusBar
                      runtime={workspaceRuntime.data ?? null}
                      sessionId={selectedSessionId}
                      state={
                        selectedSessionWorkspace.data ??
                        activeSessionData?.session.workspace_state ??
                        null
                      }
                      error={workspaceError}
                      onRetry={() => {
                        void Promise.all([
                          workspaceRuntime.refetch(),
                          selectedSessionWorkspace.refetch(),
                        ])
                      }}
                    />
                    <MemoryStatusBar
                      session={activeSessionData?.session ?? null}
                    />
                    <TimelinePanel
                      timeline={timeline}
                      loading={contentLoading}
                      artifactsPruned={selectedRunArtifactsPruned}
                      history={history}
                      loadingOlder={sessionHistory.isFetchingNextPage}
                      onLoadOlder={() => sessionHistory.fetchNextPage()}
                      historyLoadingDisabled={Boolean(validatedRunId)}
                    />
                    <Composer
                      selectedSessionId={selectedSessionId}
                      selectedProfile={
                        activeSessionData?.session.profile_name ?? null
                      }
                      activeRun={
                        activeSessionData?.session.active_run_id
                          ? activeRun
                          : null
                      }
                    />
                  </div>
                </Panel>
              </Group>
            </Panel>
          </Group>
        </div>

        <div
          className="grid min-h-0 flex-1 grid-rows-[minmax(10rem,38%)_minmax(0,1fr)] overflow-hidden lg:hidden"
          data-testid="activity-mobile-layout"
        >
          <SessionList
            sessions={filteredSessions}
            selectedSessionId={selectedSessionId}
            ariaLabel="Mobile activity sessions"
            search={sessionSearch}
            loading={sessions.isLoading}
            loadingMore={sessions.isFetchingNextPage}
            hasMore={Boolean(sessions.hasNextPage)}
            error={sessions.error}
            onRetry={() => void sessions.refetch()}
            onLoadMore={() => sessions.fetchNextPage()}
            onSearchChange={setSessionSearch}
            filters={{
              status: statusFilter,
              source: sourceFilter,
              profile: profileFilter,
              time: timeFilter,
            }}
            profileOptions={profileOptions}
            onFilterChange={(filter, value) => {
              if (filter === 'status')
                setStatusFilter(value as ActivityFilters['status'])
              if (filter === 'source')
                setSourceFilter(value as ActivityFilters['source'])
              if (filter === 'profile') setProfileFilter(value)
              if (filter === 'time')
                setTimeFilter(value as ActivityFilters['time'])
            }}
            onClearFilters={() => {
              setSessionSearch('')
              setStatusFilter('all')
              setSourceFilter('all')
              setProfileFilter('all')
              setTimeFilter('all')
            }}
            onSelect={(session) => {
              selectSession(session.id)
              selectRun(
                session.active_run_id ??
                  session.head_run_id ??
                  session.latest_run?.id ??
                  null,
              )
            }}
          />
          <div className="scrollbar-thin min-h-0 overflow-y-auto overscroll-contain">
            <div className="flex min-h-full flex-col">
              <RunStrip
                runs={runs}
                selectedRunId={validatedRunId}
                history={history}
                loadingOlder={sessionHistory.isFetchingNextPage}
                onLoadOlder={() => void sessionHistory.fetchNextPage()}
                onSelectRun={selectRun}
              />
              <RunControlBar
                sessionId={selectedSessionId}
                run={activeRunData?.run ?? null}
                onSelectRun={selectRun}
              />
              <WorkspaceStatusBar
                runtime={workspaceRuntime.data ?? null}
                sessionId={selectedSessionId}
                state={
                  selectedSessionWorkspace.data ??
                  activeSessionData?.session.workspace_state ??
                  null
                }
                error={workspaceError}
                onRetry={() => {
                  void Promise.all([
                    workspaceRuntime.refetch(),
                    selectedSessionWorkspace.refetch(),
                  ])
                }}
              />
              <MemoryStatusBar session={activeSessionData?.session ?? null} />
              <div className="h-80 min-h-64">
                <TimelinePanel
                  timeline={timeline}
                  loading={contentLoading}
                  artifactsPruned={selectedRunArtifactsPruned}
                  history={history}
                  loadingOlder={sessionHistory.isFetchingNextPage}
                  onLoadOlder={() => sessionHistory.fetchNextPage()}
                  historyLoadingDisabled={Boolean(validatedRunId)}
                />
              </div>
              <Composer
                selectedSessionId={selectedSessionId}
                selectedProfile={
                  activeSessionData?.session.profile_name ?? null
                }
                activeRun={
                  activeSessionData?.session.active_run_id ? activeRun : null
                }
              />
            </div>
          </div>
        </div>
      </>
    </div>
  )
}
