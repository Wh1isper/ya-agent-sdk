import { Plus, RefreshCcw } from 'lucide-react'
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
import type { StreamStatus } from '../../lib/status'
import { formatShortId } from '../../lib/utils'
import { useLayoutStore } from '../../stores/layoutStore'
import { buildTimelineFromRuns } from './agui/eventReducer'
import { isTerminalAguiEvent } from './eventUtils'
import {
  channelLabel,
  sessionChannel,
  sessionTitle,
} from './sessionClassification'
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
  const selectedSessionId = useLayoutStore(
    (state) => state.selectedDebugSessionId,
  )
  const selectedRunId = useLayoutStore((state) => state.selectedDebugRunId)
  const selectSession = useLayoutStore((state) => state.selectSession)
  const selectRun = useLayoutStore((state) => state.selectRun)
  const [sessionSearch, setSessionSearch] = useState('')
  const [isComposingNew, setIsComposingNew] = useState(false)
  const autoSelectedSessionRef = useRef(false)
  const sessions = useSessionsQuery()
  const workspaceRuntime = useWorkspaceRuntimeQuery()
  const selectedSessionWorkspace = useSessionWorkspaceQuery(selectedSessionId)
  const selectedSession = useSessionQuery(selectedSessionId)
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
  const activeRunData = resolvedRunId ? selectedRun.data : undefined
  const live = useRunEventStream(
    resolvedRunId,
    activeRunData?.run.status ?? null,
    selectedSessionId,
  )
  const liveEvents = useMemo(
    () => (resolvedRunId ? live.events : []),
    [live.events, resolvedRunId],
  )
  const streamStatus: StreamStatus = resolvedRunId ? live.status : 'idle'
  const contentLoading =
    Boolean(selectedSessionId && selectedSession.isLoading) ||
    Boolean(resolvedRunId && selectedRun.isLoading) ||
    Boolean(selectedSessionId && sessionHistory.isLoading)

  useEffect(() => {
    const firstSessionId = sessions.data?.[0]?.id
    if (
      !selectedSessionId &&
      !isComposingNew &&
      firstSessionId &&
      !autoSelectedSessionRef.current
    ) {
      autoSelectedSessionRef.current = true
      selectSession(firstSessionId)
    }
  }, [isComposingNew, selectSession, selectedSessionId, sessions.data])

  useEffect(() => {
    if (selectedSessionId) setIsComposingNew(false)
  }, [selectedSessionId])

  useEffect(() => {
    if (!selectedSessionId || selectedRunId) return
    const nextRunId =
      activeSessionData?.session.active_run_id ??
      activeSessionData?.session.head_run_id ??
      null
    if (nextRunId) selectRun(nextRunId)
  }, [activeSessionData, selectRun, selectedRunId, selectedSessionId])

  const filteredSessions = useMemo(() => {
    const needle = sessionSearch.trim().toLowerCase()
    const rows = sessions.data ?? []
    if (!needle) return rows
    return rows.filter((session) => {
      const haystack = [
        session.id,
        session.profile_name ?? '',
        sessionTitle(session),
        channelLabel(sessionChannel(session)),
        session.status,
      ]
        .join(' ')
        .toLowerCase()
      return haystack.includes(needle)
    })
  }, [sessionSearch, sessions.data])

  const historyPages = sessionHistory.data?.pages
  const historyRuns = useMemo(
    () => mergeSessionHistoryPages(historyPages).runs,
    [historyPages],
  )
  const activeRun = useMemo(
    () =>
      activeRunData?.run ??
      historyRuns.find((item) => item.id === resolvedRunId) ??
      activeSessionData?.session.runs.find(
        (item) => item.id === resolvedRunId,
      ) ??
      null,
    [activeRunData, activeSessionData, historyRuns, resolvedRunId],
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
  const timeline = history.timeline.blocks.length
    ? history.timeline
    : buildTimelineFromRuns(activeRun ? [activeRun] : [], {
        includeRuntimeEvents: false,
      })
  const runs = history.runs.length
    ? history.runs
    : (activeSessionData?.session.runs ?? [])

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden bg-slate-100">
      <div className="flex shrink-0 flex-col gap-3 border-b border-slate-200 bg-white px-3 py-3 sm:h-16 sm:flex-row sm:items-center sm:justify-between sm:px-5 sm:py-0">
        <div className="flex min-w-0 items-center gap-2 text-sm text-slate-600 sm:gap-3">
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
        <div className="flex shrink-0 items-center gap-2 overflow-x-auto text-xs text-slate-500">
          <LivePill
            status={streamStatus}
            eventCount={effectiveLiveEvents.length}
          />
          <button
            type="button"
            className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 font-medium text-slate-700 shadow-sm transition hover:bg-slate-50"
            onClick={() => {
              autoSelectedSessionRef.current = true
              setIsComposingNew(true)
              selectSession(null)
              selectRun(null)
            }}
          >
            <Plus className="h-3.5 w-3.5" />
            New debug run
          </button>
          <button
            type="button"
            className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 font-medium text-slate-700 shadow-sm transition hover:bg-slate-50"
            onClick={() => sessions.refetch()}
          >
            <RefreshCcw className="h-3.5 w-3.5" />
            Refresh
          </button>
        </div>
      </div>

      <Group orientation="horizontal" className="hidden min-h-0 flex-1 lg:flex">
        <Panel defaultSize="26%" minSize="260px" maxSize="36%">
          <SessionList
            sessions={filteredSessions}
            selectedSessionId={selectedSessionId}
            search={sessionSearch}
            loading={sessions.isLoading}
            onSearchChange={setSessionSearch}
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
                  selectedRunId={resolvedRunId}
                  history={history}
                  loadingOlder={sessionHistory.isFetchingNextPage}
                  onLoadOlder={() => void sessionHistory.fetchNextPage()}
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
                />
                <MemoryStatusBar session={activeSessionData?.session ?? null} />
                <RunControlBar
                  sessionId={selectedSessionId}
                  run={activeRunData?.run ?? null}
                  onSelectRun={selectRun}
                />
                <TimelinePanel
                  timeline={timeline}
                  loading={contentLoading}
                  artifactsPruned={selectedRunArtifactsPruned}
                  history={history}
                  loadingOlder={sessionHistory.isFetchingNextPage}
                  onLoadOlder={() => sessionHistory.fetchNextPage()}
                />
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
            </Panel>
          </Group>
        </Panel>
      </Group>

      <div className="grid min-h-0 flex-1 grid-rows-[14rem_minmax(0,1fr)] overflow-hidden lg:hidden">
        <SessionList
          sessions={filteredSessions}
          selectedSessionId={selectedSessionId}
          search={sessionSearch}
          loading={sessions.isLoading}
          onSearchChange={setSessionSearch}
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
        <div className="min-h-0 overflow-hidden">
          <div className="flex h-full min-h-0 flex-col overflow-hidden">
            <RunStrip
              runs={runs}
              selectedRunId={resolvedRunId}
              history={history}
              loadingOlder={sessionHistory.isFetchingNextPage}
              onLoadOlder={() => void sessionHistory.fetchNextPage()}
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
            />
            <MemoryStatusBar session={activeSessionData?.session ?? null} />
            <RunControlBar
              sessionId={selectedSessionId}
              run={activeRunData?.run ?? null}
              onSelectRun={selectRun}
            />
            <TimelinePanel
              timeline={timeline}
              loading={contentLoading}
              artifactsPruned={selectedRunArtifactsPruned}
              history={history}
              loadingOlder={sessionHistory.isFetchingNextPage}
              onLoadOlder={() => sessionHistory.fetchNextPage()}
            />
            <Composer
              selectedSessionId={selectedSessionId}
              selectedProfile={activeSessionData?.session.profile_name ?? null}
              activeRun={
                activeSessionData?.session.active_run_id ? activeRun : null
              }
            />
          </div>
        </div>
      </div>
    </div>
  )
}
