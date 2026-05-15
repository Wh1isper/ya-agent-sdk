import { useVirtualizer } from '@tanstack/react-virtual'
import {
  Activity,
  ArchiveX,
  Bot,
  CheckCircle2,
  ChevronRight,
  Clock3,
  FilePenLine,
  Files,
  Hash,
  MessageSquare,
  PauseCircle,
  PlayCircle,
  Plus,
  RefreshCcw,
  Search,
  Send,
  Square,
  TerminalSquare,
  User,
  Wrench,
  XCircle,
} from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { Group, Panel, Separator } from 'react-resizable-panels'
import { toast } from 'sonner'

import {
  useCreateSessionMutation,
  useCreateSessionRunMutation,
  useProfilesQuery,
  useRunControlMutations,
  useRunQuery,
  useSessionHistoryQuery,
  useSessionQuery,
  useSessionSandboxMutations,
  useSessionWorkspaceQuery,
  useSessionsQuery,
  useWorkspaceRuntimeQuery,
} from '../../api/hooks'
import { EmptyState } from '../../components/EmptyState'
import { JsonView } from '../../components/JsonView'
import { StatusBadge } from '../../components/StatusBadge'
import {
  darkPillClass,
  getStreamStatusTone,
  lightPillClass,
  type StreamStatus,
  toneDotClass,
} from '../../lib/status'
import { cn, formatShortId, safeJsonStringify } from '../../lib/utils'
import { useLayoutStore } from '../../stores/layoutStore'
import type {
  AguiEvent,
  InputPart,
  RunSummary,
  SessionSandboxState,
  SessionSummary,
  SessionWorkspaceState,
  WorkspaceRuntimeStatus,
} from '../../types'
import { buildTimelineFromRuns } from './agui/eventReducer'
import type { AguiTimelineState, TimelineBlock } from './agui/types'
import type { SessionHistoryState } from './sessionHistory'
import {
  eventKey,
  eventNameLabel,
  eventTimestampLabel,
  eventTone,
  eventTypeLabel,
  isTerminalAguiEvent,
} from './eventUtils'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useRunEventStream } from './useRunEventStream'
import { sandboxLabel, sandboxTone, ttlLabel } from '../workspaceDisplay'
import {
  channelLabel,
  sessionChannel,
  sessionTitle,
} from './sessionClassification'
import { mergeSessionHistoryPages } from './sessionHistory'

const DEBUG_METADATA = { web: { surface: 'debug' } }

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
  const runEvents = history.events

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
            <ResizeHandle />
            <Panel defaultSize="32%" minSize="260px">
              <EventDevToolsPanel
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

function SessionList({
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

function SessionChannelPill({ session }: { session: SessionSummary }) {
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

function SessionSkeleton() {
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

function RunStrip({
  runs,
  selectedRunId,
  history,
  loadingOlder,
  onLoadOlder,
  onSelectRun,
}: {
  runs: RunSummary[]
  selectedRunId: string | null
  history: SessionHistoryState
  loadingOlder: boolean
  onLoadOlder: () => void
  onSelectRun: (runId: string | null) => void
}) {
  return (
    <div className="flex h-16 shrink-0 items-center gap-3 overflow-hidden border-b border-slate-200 bg-white px-3 sm:px-4">
      <div className="shrink-0">
        <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
          Runs
        </p>
        <p className="text-[11px] text-slate-500">
          {history.loadedRunCount}/{history.totalRunCount || runs.length} loaded
        </p>
      </div>
      <div className="scrollbar-thin flex min-w-0 flex-1 gap-2 overflow-x-auto py-2">
        {history.hasMore ? (
          <button
            type="button"
            className="inline-flex shrink-0 items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 transition hover:border-blue-200 hover:bg-blue-50 hover:text-blue-700 disabled:opacity-60"
            onClick={onLoadOlder}
            disabled={loadingOlder}
          >
            {loadingOlder ? 'Loading...' : 'Older'}
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

function WorkspaceStatusBar({
  runtime,
  sessionId,
  state,
}: {
  runtime: WorkspaceRuntimeStatus | null
  sessionId: string | null
  state: SessionWorkspaceState | null
}) {
  const sandbox = state?.sandbox_state ?? null
  const controls = useSessionSandboxMutations(sessionId)
  const canPrepare = Boolean(
    sessionId &&
      runtime?.capabilities.sandbox_prepare &&
      sandbox?.ready_state !== 'ready',
  )
  const canStop = Boolean(
    sessionId && runtime?.capabilities.sandbox_stop && sandbox?.container_id,
  )

  return (
    <div className="flex shrink-0 flex-col gap-2 border-b border-slate-200 bg-blue-50/60 px-3 py-2 text-xs text-blue-950 sm:flex-row sm:items-center sm:justify-between sm:gap-3 sm:px-4">
      <div className="flex min-w-0 items-center gap-2 font-medium">
        <TerminalSquare className="h-3.5 w-3.5" />
        <span>Workspace</span>
        <span className="rounded-full bg-white/80 px-2 py-0.5 text-blue-700">
          {runtime?.backend ?? sandbox?.backend ?? 'unknown'}
        </span>
        <span className="truncate text-blue-700">
          {state?.binding?.cwd ??
            sandbox?.work_dir ??
            runtime?.workspace.virtual_path ??
            'workspace'}
        </span>
      </div>
      <div className="flex shrink-0 flex-wrap items-center justify-end gap-2">
        <SessionSandboxPill sandbox={sandbox} />
        {sandbox?.container_id ? (
          <span className="mono rounded-full bg-white/80 px-2 py-0.5 text-blue-700">
            {formatShortId(sandbox.container_id, 12)}
          </span>
        ) : null}
        {canPrepare ? (
          <button
            type="button"
            className="rounded-full border border-blue-200 bg-white px-2 py-0.5 font-medium text-blue-700 transition hover:bg-blue-50 disabled:opacity-60"
            onClick={() => controls.prepare.mutate()}
            disabled={controls.prepare.isPending}
          >
            Prepare
          </button>
        ) : null}
        {canStop ? (
          <button
            type="button"
            className="rounded-full border border-slate-200 bg-white px-2 py-0.5 font-medium text-slate-700 transition hover:bg-slate-50 disabled:opacity-60"
            onClick={() => controls.stop.mutate()}
            disabled={controls.stop.isPending}
          >
            Stop
          </button>
        ) : null}
      </div>
    </div>
  )
}

function SessionSandboxPill({
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

function MemoryStatusBar({ session }: { session: SessionSummary | null }) {
  const memory = session?.memory_state
  if (!session || !memory) return null

  return (
    <div className="flex shrink-0 flex-col gap-2 border-b border-slate-200 bg-violet-50/60 px-3 py-2 text-xs text-violet-900 sm:flex-row sm:items-center sm:justify-between sm:gap-3 sm:px-4">
      <div className="flex items-center gap-2 font-medium">
        <ArchiveX className="h-3.5 w-3.5" />
        <span>Memory</span>
      </div>
      <div className="flex flex-wrap items-center justify-end gap-2">
        <span>{memory.extract_count} extracts</span>
        <span>{memory.turns_since_extract} turns since extract</span>
        <span>{memory.extracts_since_summary} extracts since summary</span>
        {memory.pending_extract ? (
          <span className="rounded-full bg-amber-100 px-2 py-0.5 text-amber-700">
            extract pending
          </span>
        ) : null}
        {memory.pending_summary ? (
          <span className="rounded-full bg-amber-100 px-2 py-0.5 text-amber-700">
            summary pending
          </span>
        ) : null}
      </div>
    </div>
  )
}

function RunControlBar({
  sessionId,
  run,
  onSelectRun,
}: {
  sessionId: string | null
  run: RunSummary | null
  onSelectRun: (runId: string | null) => void
}) {
  const runControls = useRunControlMutations(run?.id ?? null)
  const createRun = useCreateSessionRunMutation(sessionId)
  if (!run) return null

  const isActive = run.status === 'queued' || run.status === 'running'
  const canRecover =
    run.status === 'failed' &&
    Boolean(sessionId) &&
    Boolean(run.input_parts?.length)

  async function recover(mode: 'retry' | 'reset_and_retry') {
    if (!run || !sessionId || !run.input_parts?.length) return
    try {
      const createdRun = await createRun.mutateAsync({
        input_parts: run.input_parts,
        reset_state: mode === 'reset_and_retry',
        metadata: {
          recovery: {
            mode,
            source_run_id: run.id,
            source_sequence_no: run.sequence_no,
            reason: 'web_ui',
          },
        },
      })
      onSelectRun(createdRun.id)
      toast.success(
        mode === 'retry' ? 'Retry submitted' : 'Reset and retry submitted',
      )
    } catch (error) {
      toast.error(
        error instanceof Error
          ? error.message
          : 'Failed to submit recovery run',
      )
    }
  }

  if (!isActive && !canRecover) return null

  return (
    <div className="flex flex-col gap-3 border-b border-slate-200 bg-white px-3 py-3 sm:flex-row sm:items-center sm:justify-between sm:px-4">
      <div className="flex items-center gap-2 text-sm text-slate-600">
        <StatusBadge status={run.status} />
        <span className="mono text-xs">{formatShortId(run.id, 12)}</span>
        {canRecover ? (
          <span className="text-xs text-rose-600">Run failed</span>
        ) : null}
      </div>
      <div className="flex items-center gap-2">
        {isActive ? (
          <>
            <button
              type="button"
              className="inline-flex items-center gap-2 rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-xs font-medium text-amber-700 transition hover:bg-amber-100 disabled:opacity-60"
              onClick={() => runControls.interrupt.mutate()}
              disabled={runControls.interrupt.isPending}
            >
              <PauseCircle className="h-3.5 w-3.5" />
              Interrupt
            </button>
            <button
              type="button"
              className="inline-flex items-center gap-2 rounded-xl border border-rose-200 bg-rose-50 px-3 py-2 text-xs font-medium text-rose-700 transition hover:bg-rose-100 disabled:opacity-60"
              onClick={() => runControls.cancel.mutate()}
              disabled={runControls.cancel.isPending}
            >
              <Square className="h-3.5 w-3.5" />
              Cancel
            </button>
          </>
        ) : null}
        {canRecover ? (
          <>
            <button
              type="button"
              className="inline-flex items-center gap-2 rounded-xl border border-blue-200 bg-blue-50 px-3 py-2 text-xs font-medium text-blue-700 transition hover:bg-blue-100 disabled:opacity-60"
              onClick={() => void recover('retry')}
              disabled={createRun.isPending}
            >
              <RefreshCcw className="h-3.5 w-3.5" />
              Retry
            </button>
            <button
              type="button"
              className="inline-flex items-center gap-2 rounded-xl border border-rose-200 bg-rose-50 px-3 py-2 text-xs font-medium text-rose-700 transition hover:bg-rose-100 disabled:opacity-60"
              onClick={() => void recover('reset_and_retry')}
              disabled={createRun.isPending}
            >
              <ArchiveX className="h-3.5 w-3.5" />
              Reset and retry
            </button>
          </>
        ) : null}
      </div>
    </div>
  )
}

function EventDevToolsPanel({
  events,
  streamStatus,
  liveEventCount,
  loading,
  artifactsPruned,
}: {
  events: AguiEvent[]
  streamStatus: StreamStatus
  liveEventCount: number
  loading: boolean
  artifactsPruned: boolean
}) {
  const parentRef = useRef<HTMLDivElement | null>(null)
  const virtualizer = useVirtualizer({
    count: events.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 38,
    overscan: 8,
  })

  return (
    <aside className="flex h-full min-h-0 flex-col overflow-hidden border-l border-slate-200 bg-slate-950 text-slate-100">
      <div className="flex h-12 shrink-0 items-center justify-between border-b border-slate-800 px-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
            Event stream
          </p>
          <p className="mono text-[11px] text-slate-500">
            {events.length} events · {liveEventCount} live
          </p>
        </div>
        <span
          className={cn(
            'rounded-full border px-2 py-1 text-[11px] font-medium capitalize',
            darkPillClass(getStreamStatusTone(streamStatus)),
          )}
        >
          {streamStatus}
        </span>
      </div>
      <div
        ref={parentRef}
        className="scrollbar-thin min-h-0 flex-1 overscroll-contain overflow-auto p-2"
      >
        {loading ? (
          <div className="space-y-2 p-2">
            {Array.from({ length: 6 }).map((_, index) => (
              <div
                key={index}
                className="h-9 animate-pulse rounded bg-slate-900"
              />
            ))}
          </div>
        ) : null}
        {!loading && events.length === 0 ? (
          artifactsPruned ? (
            <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-3 text-xs leading-5 text-amber-200">
              Run replay artifacts have been pruned from disk. Database
              metadata, input parts, status, and summaries are still available.
            </div>
          ) : (
            <div className="rounded-xl border border-slate-800 bg-slate-900 p-3 text-xs text-slate-400">
              Select a run to inspect raw AGUI events.
            </div>
          )
        ) : null}
        {events.length > 0 ? (
          <div
            className="relative"
            style={{ height: `${virtualizer.getTotalSize()}px` }}
          >
            {virtualizer.getVirtualItems().map((item) => {
              const event = events[item.index]
              return (
                <div
                  key={`${item.index}:${eventKey(event)}`}
                  data-index={item.index}
                  ref={virtualizer.measureElement}
                  className="absolute left-0 top-0 w-full pb-1"
                  style={{ transform: `translateY(${item.start}px)` }}
                >
                  <EventRow event={event} index={item.index} />
                </div>
              )
            })}
          </div>
        ) : null}
      </div>
    </aside>
  )
}

function EventRow({ event, index }: { event: AguiEvent; index: number }) {
  const [expanded, setExpanded] = useState(false)
  const type = eventTypeLabel(event)
  const name = eventNameLabel(event)
  const timestamp = eventTimestampLabel(event)
  const tone = eventTone(event)

  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/80">
      <button
        type="button"
        className="flex w-full items-center gap-2 px-2 py-1.5 text-left text-xs hover:bg-slate-800/70"
        onClick={() => setExpanded((value) => !value)}
      >
        <ChevronRight
          className={cn(
            'h-3.5 w-3.5 shrink-0 text-slate-500 transition',
            expanded && 'rotate-90',
          )}
        />
        <span className="mono w-8 shrink-0 text-[11px] text-slate-500">
          {index + 1}
        </span>
        <span
          className={cn('h-2 w-2 shrink-0 rounded-full', toneDotClass(tone))}
        />
        <span className="mono min-w-0 flex-1 truncate text-[11px] text-slate-200">
          {type}
          {name ? <span className="text-slate-500"> · {name}</span> : null}
        </span>
        {timestamp ? (
          <span className="mono shrink-0 text-[10px] text-slate-500">
            {timestamp}
          </span>
        ) : null}
      </button>
      {expanded ? (
        <pre className="scrollbar-thin max-h-80 overflow-auto border-t border-slate-800 p-2 text-[11px] leading-5 text-slate-300">
          {safeJsonStringify(event)}
        </pre>
      ) : null}
    </div>
  )
}

function TimelinePanel({
  timeline,
  loading,
  artifactsPruned,
  history,
  loadingOlder,
  onLoadOlder,
}: {
  timeline: AguiTimelineState
  loading: boolean
  artifactsPruned: boolean
  history: SessionHistoryState
  loadingOlder: boolean
  onLoadOlder: () => Promise<unknown>
}) {
  const scrollRef = useRef<HTMLElement | null>(null)
  const bottomRef = useRef<HTMLDivElement | null>(null)
  const stickToBottomRef = useRef(true)
  const previousScrollHeightRef = useRef<number | null>(null)
  const toolCallCount = timeline.blocks.filter(
    (block) => block.kind === 'tool_call',
  ).length
  const assistantCount = timeline.blocks.filter(
    (block) => block.kind === 'assistant_message',
  ).length
  useEffect(() => {
    const element = scrollRef.current
    if (!element) return
    const previousHeight = previousScrollHeightRef.current
    if (previousHeight != null) {
      element.scrollTop = element.scrollHeight - previousHeight
      previousScrollHeightRef.current = null
      return
    }
    if (!stickToBottomRef.current) return
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [timeline.blocks.length])

  async function loadOlder() {
    const element = scrollRef.current
    if (!element || loadingOlder || !history.hasMore) return
    previousScrollHeightRef.current = element.scrollHeight
    stickToBottomRef.current = false
    await onLoadOlder()
  }

  return (
    <section
      ref={scrollRef}
      className="scrollbar-thin min-h-0 flex-1 overscroll-contain overflow-auto bg-slate-50 p-3 sm:p-5"
      onScroll={() => {
        const element = scrollRef.current
        if (!element) return
        const distanceFromBottom =
          element.scrollHeight - element.scrollTop - element.clientHeight
        stickToBottomRef.current = distanceFromBottom < 160
        if (element.scrollTop < 96 && history.hasMore && !loadingOlder) {
          void loadOlder()
        }
      }}
    >
      <div className="mx-auto mb-4 flex max-w-4xl flex-col gap-3 rounded-2xl border border-slate-200 bg-white/80 px-4 py-3 shadow-sm backdrop-blur sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="text-sm font-semibold text-slate-900">Runtime replay</p>
          <p className="mt-0.5 text-xs text-slate-500">
            {timeline.blocks.length} blocks · {assistantCount} assistant
            messages · {toolCallCount} tool calls
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-500">
            {history.loadedRunCount}/
            {history.totalRunCount || history.loadedRunCount} runs loaded
          </span>
          <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-500">
            Auto-scroll
          </span>
        </div>
      </div>
      {loading ? <TimelineSkeleton /> : null}
      {!loading && timeline.blocks.length === 0 ? (
        artifactsPruned ? (
          <PrunedArtifactsNotice />
        ) : (
          <EmptyState
            icon={MessageSquare}
            title="No replay yet"
            description="Select a run with committed AGUI messages or start a new debug turn."
            className="mx-auto max-w-4xl bg-white"
          />
        )
      ) : null}
      <div className="mx-auto max-w-4xl space-y-4">
        {!loading && timeline.blocks.length > 0 ? (
          <DebugHistoryBoundary
            history={history}
            loadingOlder={loadingOlder}
            onLoadOlder={() => void loadOlder()}
          />
        ) : null}
        {timeline.blocks.map((block) => (
          <TimelineCard key={block.id} block={block} />
        ))}
        <div ref={bottomRef} />
      </div>
    </section>
  )
}

function DebugHistoryBoundary({
  history,
  loadingOlder,
  onLoadOlder,
}: {
  history: SessionHistoryState
  loadingOlder: boolean
  onLoadOlder: () => void
}) {
  if (history.hasMore) {
    return (
      <button
        type="button"
        className="mx-auto flex items-center justify-center rounded-full border border-slate-200 bg-white px-4 py-2 text-xs font-medium text-slate-600 shadow-sm transition hover:border-blue-200 hover:bg-blue-50 hover:text-blue-700 disabled:opacity-60"
        onClick={onLoadOlder}
        disabled={loadingOlder}
      >
        {loadingOlder
          ? 'Loading older runs...'
          : `Load older runs · ${history.loadedRunCount}/${history.totalRunCount}`}
      </button>
    )
  }
  return (
    <div className="mx-auto w-fit rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-500">
      Beginning of session
    </div>
  )
}

function PrunedArtifactsNotice() {
  return (
    <div className="mx-auto max-w-4xl">
      <Card icon={ArchiveX} title="Replay artifacts pruned" accent="amber">
        <div className="space-y-2 text-sm leading-6 text-slate-700">
          <p>
            The raw AGUI replay for this run has been pruned from disk to reduce
            storage usage.
          </p>
          <p className="text-slate-500">
            YA Claw still keeps the run database row, input parts, status,
            output text, and compact summary when available.
          </p>
        </div>
      </Card>
    </div>
  )
}

function TimelineSkeleton() {
  return (
    <div className="mx-auto max-w-4xl space-y-4">
      {Array.from({ length: 3 }).map((_, index) => (
        <div
          key={index}
          className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm"
        >
          <div className="h-4 w-32 animate-pulse rounded bg-slate-100" />
          <div className="mt-4 h-16 animate-pulse rounded bg-slate-100" />
        </div>
      ))}
    </div>
  )
}

function TimelineCard({ block }: { block: TimelineBlock }) {
  if (block.kind === 'user_input') {
    return (
      <Card icon={User} title="User input" accent="blue">
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
  if (block.kind === 'reasoning') {
    return (
      <Card icon={Activity} title="Reasoning" accent="violet" subtle>
        <div className="whitespace-pre-wrap text-sm leading-7 text-slate-700">
          {block.content}
        </div>
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
  if (block.kind === 'task_board') {
    return (
      <Card icon={CheckCircle2} title="Task board" accent="blue">
        <div className="grid gap-2">
          {block.tasks.map((task) => (
            <div
              key={task.id}
              className="rounded-xl border border-slate-200 bg-slate-50 p-3"
            >
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="text-sm font-medium text-slate-900">
                    {task.subject}
                  </p>
                  {task.active_form ? (
                    <p className="mt-1 text-xs text-slate-500">
                      {task.active_form}
                    </p>
                  ) : null}
                </div>
                <StatusBadge status={task.status} />
              </div>
            </div>
          ))}
          {block.tasks.length === 0 ? (
            <p className="text-sm text-slate-500">No tasks in snapshot.</p>
          ) : null}
        </div>
      </Card>
    )
  }
  if (block.kind === 'context_meter') {
    const percent =
      block.contextWindowSize > 0
        ? Math.min(
            100,
            Math.round((block.totalTokens / block.contextWindowSize) * 100),
          )
        : 0
    return (
      <Card icon={Clock3} title="Context" accent="amber" compact>
        <div className="flex items-center gap-3">
          <div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-100">
            <div
              className="h-full rounded-full bg-amber-500"
              style={{ width: `${percent}%` }}
            />
          </div>
          <span className="mono text-xs text-slate-600">
            {block.totalTokens} / {block.contextWindowSize}
          </span>
        </div>
      </Card>
    )
  }
  if (block.kind === 'subagent') {
    return (
      <Card
        icon={Bot}
        title={`Subagent · ${block.agentName}`}
        accent={block.status === 'failed' ? 'rose' : 'violet'}
      >
        <StatusBadge status={block.status} />
        {block.promptPreview ? (
          <p className="mt-3 text-sm text-slate-600">{block.promptPreview}</p>
        ) : null}
        {block.resultPreview ? (
          <p className="mt-3 text-sm text-slate-800">{block.resultPreview}</p>
        ) : null}
        {block.error ? (
          <p className="mt-3 text-sm text-rose-700">{block.error}</p>
        ) : null}
      </Card>
    )
  }
  if (block.kind === 'file_change') {
    return (
      <Card
        icon={Files}
        title={
          block.toolName ? `File changes · ${block.toolName}` : 'File changes'
        }
        accent="emerald"
      >
        <JsonView value={block.changes} height="260px" />
      </Card>
    )
  }
  if (block.kind === 'note_snapshot') {
    return (
      <Card icon={FilePenLine} title="Notes" accent="blue">
        <JsonView value={block.entries} height="220px" />
      </Card>
    )
  }
  if (block.kind === 'usage') {
    return (
      <Card icon={Activity} title="Usage" accent="violet" compact>
        <JsonView value={block.payload} height="180px" />
      </Card>
    )
  }
  if (block.kind === 'runtime_event') {
    return (
      <Card
        icon={TerminalSquare}
        title={block.title}
        accent={accentFromRuntimeStatus(block.status)}
        compact
      >
        <JsonView value={block.payload} height="180px" />
      </Card>
    )
  }
  return (
    <Card icon={MessageSquare} title={block.name} accent="slate" compact>
      <JsonView value={block.payload} height="180px" />
    </Card>
  )
}

function MarkdownMessage({ content }: { content: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        a: ({ className, ...props }) => (
          <a
            className={cn(
              'font-medium text-blue-600 underline decoration-blue-300 underline-offset-2 hover:text-blue-700',
              className,
            )}
            target="_blank"
            rel="noreferrer"
            {...props}
          />
        ),
        blockquote: ({ className, ...props }) => (
          <blockquote
            className={cn(
              'my-4 border-l-4 border-slate-200 pl-4 text-slate-600',
              className,
            )}
            {...props}
          />
        ),
        code: ({ className, children, ...props }) => (
          <code
            className={cn(
              'rounded bg-slate-100 px-1.5 py-0.5 font-mono text-[0.9em] text-slate-800',
              className,
            )}
            {...props}
          >
            {children}
          </code>
        ),
        h1: ({ className, ...props }) => (
          <h1
            className={cn(
              'mb-3 mt-5 text-xl font-semibold text-slate-950',
              className,
            )}
            {...props}
          />
        ),
        h2: ({ className, ...props }) => (
          <h2
            className={cn(
              'mb-3 mt-5 text-lg font-semibold text-slate-950',
              className,
            )}
            {...props}
          />
        ),
        h3: ({ className, ...props }) => (
          <h3
            className={cn(
              'mb-2 mt-4 text-base font-semibold text-slate-950',
              className,
            )}
            {...props}
          />
        ),
        li: ({ className, ...props }) => (
          <li className={cn('pl-1', className)} {...props} />
        ),
        ol: ({ className, ...props }) => (
          <ol
            className={cn('my-3 list-decimal space-y-1 pl-6', className)}
            {...props}
          />
        ),
        p: ({ className, ...props }) => (
          <p
            className={cn('my-3 leading-7 first:mt-0 last:mb-0', className)}
            {...props}
          />
        ),
        pre: ({ className, ...props }) => (
          <pre
            className={cn(
              'scrollbar-thin my-4 max-w-full overflow-auto rounded-xl border border-slate-200 bg-slate-950 p-3 text-xs leading-5 text-slate-100',
              className,
            )}
            {...props}
          />
        ),
        table: ({ className, ...props }) => (
          <div className="scrollbar-thin my-4 overflow-auto">
            <table
              className={cn(
                'w-full border-collapse text-left text-sm',
                className,
              )}
              {...props}
            />
          </div>
        ),
        td: ({ className, ...props }) => (
          <td
            className={cn(
              'border border-slate-200 px-3 py-2 align-top',
              className,
            )}
            {...props}
          />
        ),
        th: ({ className, ...props }) => (
          <th
            className={cn(
              'border border-slate-200 bg-slate-50 px-3 py-2 font-semibold',
              className,
            )}
            {...props}
          />
        ),
        ul: ({ className, ...props }) => (
          <ul
            className={cn('my-3 list-disc space-y-1 pl-6', className)}
            {...props}
          />
        ),
      }}
    >
      {content}
    </ReactMarkdown>
  )
}

function Card({
  icon: Icon,
  title,
  accent,
  subtle,
  compact,
  children,
}: {
  icon: typeof Bot
  title: string
  accent: 'blue' | 'emerald' | 'amber' | 'rose' | 'violet' | 'slate'
  subtle?: boolean
  compact?: boolean
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
    <article
      className={cn(
        'rounded-2xl border border-slate-200 bg-white shadow-sm',
        subtle && 'bg-white/70',
        compact ? 'p-3' : 'p-4',
      )}
    >
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
  if (part.type === 'text')
    return (
      <div className="whitespace-pre-wrap rounded-xl bg-blue-50 p-3 text-sm leading-7 text-slate-800">
        {part.text}
      </div>
    )
  return <JsonView value={part} height="160px" />
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

function formatMaybeJson(value: string) {
  try {
    return JSON.stringify(JSON.parse(value), null, 2)
  } catch {
    return value
  }
}

function Composer({
  selectedSessionId,
  selectedProfile,
  activeRun,
}: {
  selectedSessionId: string | null
  selectedProfile: string | null
  activeRun: RunSummary | null
}) {
  const [text, setText] = useState('')
  const createSession = useCreateSessionMutation()
  const createRun = useCreateSessionRunMutation(selectedSessionId)
  const runControls = useRunControlMutations(activeRun?.id ?? null)
  const profiles = useProfilesQuery()
  const profileOptions = profiles.data ?? []
  const defaultProfileName = profileOptions[0]?.name ?? ''
  const [profileName, setProfileName] = useState(
    selectedProfile ?? defaultProfileName,
  )
  const selectSession = useLayoutStore((store) => store.selectSession)
  const selectRun = useLayoutStore((store) => store.selectRun)
  const canSteer = activeRun?.status === 'running'
  const queuedLocked = activeRun?.status === 'queued'

  useEffect(() => {
    setProfileName(selectedProfile ?? defaultProfileName)
  }, [defaultProfileName, selectedProfile])

  const isPending =
    createSession.isPending ||
    createRun.isPending ||
    runControls.steer.isPending
  const canSend = text.trim().length > 0 && !isPending && !queuedLocked

  async function send() {
    const normalizedText = text.trim()
    if (!normalizedText) return
    const inputParts: InputPart[] = [{ type: 'text', text: normalizedText }]
    try {
      if (canSteer && activeRun) {
        await runControls.steer.mutateAsync(inputParts)
      } else if (selectedSessionId) {
        const run = await createRun.mutateAsync({
          input_parts: inputParts,
          metadata: DEBUG_METADATA,
        })
        selectRun(run.id)
      } else {
        const response = await createSession.mutateAsync({
          profile_name: profileName.trim() || null,
          input_parts: inputParts,
          metadata: DEBUG_METADATA,
        })
        selectSession(response.session.id)
        selectRun(
          response.run?.id ??
            response.session.active_run_id ??
            response.session.head_run_id ??
            null,
        )
      }
      setText('')
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : 'Failed to send message',
      )
    }
  }

  return (
    <div className="border-t border-slate-200 bg-white p-3 sm:p-4">
      <div className="mx-auto max-w-4xl">
        {activeRun ? (
          <div
            className={cn(
              'mb-3 rounded-2xl border px-4 py-3 text-sm',
              canSteer
                ? 'border-blue-200 bg-blue-50 text-blue-800'
                : 'border-amber-200 bg-amber-50 text-amber-800',
            )}
          >
            {canSteer
              ? 'Active run is streaming. New input will steer the current run.'
              : 'This session is queued. Interrupt or cancel before sending a new turn.'}
          </div>
        ) : null}
        <div className="rounded-3xl border border-slate-200 bg-white p-3 shadow-sm ring-1 ring-slate-100 transition focus-within:border-blue-200 focus-within:ring-blue-100">
          <textarea
            className="max-h-48 min-h-24 w-full resize-none rounded-2xl border-0 p-2 text-sm leading-6 text-slate-900 outline-none placeholder:text-slate-400 disabled:bg-white disabled:text-slate-400"
            value={text}
            onChange={(event) => setText(event.target.value)}
            placeholder={
              canSteer
                ? 'Steer the active run...'
                : queuedLocked
                  ? 'Run queued. Controls are available above the replay.'
                  : 'Send a debug prompt to YA Claw...'
            }
            disabled={queuedLocked}
            onKeyDown={(event) => {
              if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
                void send()
              }
            }}
          />
          <div className="flex flex-col gap-3 border-t border-slate-100 pt-3 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex min-w-0 flex-1 items-center gap-2">
              {profileOptions.length > 0 ? (
                <select
                  className="max-w-52 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-700 outline-none ring-blue-600 focus:ring-2 disabled:text-slate-400"
                  value={profileName}
                  onChange={(event) => setProfileName(event.target.value)}
                  disabled={Boolean(selectedSessionId) || Boolean(activeRun)}
                >
                  {profileOptions.map((profile) => (
                    <option key={profile.name} value={profile.name}>
                      {profile.name}
                    </option>
                  ))}
                </select>
              ) : (
                <span className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-500">
                  No profiles
                </span>
              )}
              <span className="hidden text-xs text-slate-400 lg:inline">
                Cmd/Ctrl + Enter to send
              </span>
            </div>
            <button
              type="button"
              className={cn(
                'inline-flex items-center gap-2 rounded-xl px-4 py-2 text-sm font-semibold text-white shadow-sm transition disabled:bg-slate-300',
                canSteer
                  ? 'bg-amber-600 hover:bg-amber-700'
                  : 'bg-blue-600 hover:bg-blue-700',
              )}
              disabled={!canSend}
              onClick={() => void send()}
            >
              {canSteer ? (
                <Wrench className="h-4 w-4" />
              ) : selectedSessionId ? (
                <Send className="h-4 w-4" />
              ) : (
                <Plus className="h-4 w-4" />
              )}
              {isPending
                ? canSteer
                  ? 'Steering'
                  : 'Sending'
                : canSteer
                  ? 'Steer run'
                  : selectedSessionId
                    ? 'Send'
                    : 'New debug run'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

function LivePill({
  status,
  eventCount,
}: {
  status: StreamStatus
  eventCount: number
}) {
  const icon =
    status === 'streaming'
      ? PlayCircle
      : status === 'error'
        ? XCircle
        : status === 'closed'
          ? CheckCircle2
          : Clock3
  const Icon = icon
  return (
    <span
      className={cn(
        'inline-flex items-center gap-2 rounded-full border px-3 py-1.5 font-medium capitalize',
        lightPillClass(getStreamStatusTone(status)),
      )}
      aria-live="polite"
    >
      <Icon className="h-3.5 w-3.5" aria-hidden />
      {status} · {eventCount} live
    </span>
  )
}

function ResizeHandle() {
  return (
    <Separator className="group relative w-1 shrink-0 bg-slate-100 transition hover:bg-blue-100">
      <div className="absolute inset-y-0 left-1/2 w-px -translate-x-1/2 bg-slate-200 group-hover:bg-blue-300" />
    </Separator>
  )
}

function accentFromRuntimeStatus(
  status: 'info' | 'running' | 'success' | 'warning' | 'error',
) {
  if (status === 'running') return 'amber'
  if (status === 'success') return 'emerald'
  if (status === 'warning') return 'amber'
  if (status === 'error') return 'rose'
  return 'slate'
}
