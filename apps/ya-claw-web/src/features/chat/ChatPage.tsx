import { Link, useRouterState } from '@tanstack/react-router'
import {
  Bot,
  MessageSquare,
  PanelLeft,
  Plus,
  RefreshCcw,
  Send,
} from 'lucide-react'
import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react'
import { toast } from 'sonner'

import { ApiError } from '../../api/client'
import {
  useCreateSessionMutation,
  useProfilesQuery,
  useRunQuery,
  useSubmitSessionInputMutation,
  useSessionHistoryQuery,
  useSessionQuery,
  useSessionWorkspaceQuery,
  useSessionsQuery,
} from '../../api/hooks'
import { EmptyState } from '../../components/EmptyState'
import { StatusBadge } from '../../components/StatusBadge'
import { Sheet, SheetContent, SheetHeader } from '../../components/ui/Sheet'
import { QueryError } from '../../components/ui/QueryState'
import {
  buildChatPath,
  parseUrlSelection,
  pushBrowserPath,
  replaceBrowserPath,
} from '../../lib/urlState'
import { cn, formatShortId } from '../../lib/utils'
import { useLayoutStore } from '../../stores/layoutStore'
import type {
  InputPart,
  RunSummary,
  SessionDetail,
  SessionSummary,
  SessionWorkspaceState,
} from '../../types'
import type { AguiTimelineState, TimelineBlock } from './agui/types'
import { isTerminalAguiEvent } from './eventUtils'
import { remarkNormalizeRouteHeadings } from './markdownHeadings'
import { useRunEventStream } from './useRunEventStream'
import { isSubmissionTargetActive, useSessionDraft } from './sessionDraft'
import { sessionTitle } from './sessionClassification'
import {
  mergeSessionHistoryPages,
  type SessionHistoryState,
} from './sessionHistory'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

const WEB_CHAT_METADATA = { web: { surface: 'chat' } }

type ConversationTab =
  | 'conversation'
  | 'runs'
  | 'tools'
  | 'memory'
  | 'workspace'
  | 'advanced'

export function ChatPage() {
  const pathname = useRouterState({
    select: (state) => state.location.pathname,
  })
  const isConversationIndex = pathname === '/conversations'
  const isNewConversationPath = pathname === '/conversations/new'
  const routeSelection = useMemo(() => parseUrlSelection(pathname), [pathname])
  const selectedSessionId =
    !isConversationIndex && !isNewConversationPath
      ? routeSelection.selectedSessionId
      : null
  const selectedRunId = selectedSessionId ? routeSelection.selectedRunId : null
  const advancedMode = useLayoutStore((state) => state.advancedMode)
  const selectSession = useLayoutStore((state) => state.selectSession)
  const selectRun = useLayoutStore((state) => state.selectRun)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [isComposingNew, setIsComposingNew] = useState(isNewConversationPath)
  const [detailTab, setDetailTab] = useState<ConversationTab>('conversation')
  const sessions = useSessionsQuery()
  const conversationSessions = useMemo(
    () => sessions.data ?? [],
    [sessions.data],
  )
  const composingNew =
    isNewConversationPath || (isComposingNew && isConversationIndex)
  const effectiveSessionId =
    !composingNew && selectedSessionId ? selectedSessionId : null
  const effectiveRunId =
    !composingNew && selectedSessionId ? selectedRunId : null
  const selectedSession = useSessionQuery(effectiveSessionId, {
    runsLimit: 1,
    includeMessage: false,
    includeInputParts: false,
    includeHeadPayload: false,
  })
  const sessionWorkspace = useSessionWorkspaceQuery(effectiveSessionId)
  const activeSessionData = effectiveSessionId
    ? selectedSession.data
    : undefined
  const resolvedRunId =
    effectiveRunId ??
    activeSessionData?.session.active_run_id ??
    activeSessionData?.session.head_run_id ??
    null
  const sessionHistory = useSessionHistoryQuery(effectiveSessionId, {
    runsLimit: 3,
  })
  const historyPages = useMemo(
    () =>
      sessionHistory.data?.pages ??
      (activeSessionData ? [activeSessionData] : undefined),
    [activeSessionData, sessionHistory.data?.pages],
  )
  const historyRuns = useMemo(
    () => mergeSessionHistoryPages(historyPages).runs,
    [historyPages],
  )
  const resolvedRunIsInHistory = historyRuns.some(
    (run) => run.id === resolvedRunId,
  )
  const selectedRun = useRunQuery(resolvedRunId, {
    enabled: sessionHistory.isFetched && !resolvedRunIsInHistory,
  })
  const activeRunData =
    resolvedRunId && selectedRun.data?.session.id === effectiveSessionId
      ? selectedRun.data
      : undefined
  const runSessionMismatch = Boolean(
    resolvedRunId &&
    selectedRun.data &&
    selectedRun.data.session.id !== effectiveSessionId,
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
  const live = useRunEventStream(
    resolvedRunId,
    activeRun?.status ?? null,
    effectiveSessionId,
  )
  const selectedRunReplayEvents = useMemo(
    () =>
      activeRunData?.message ??
      activeRun?.message ??
      activeSessionData?.message ??
      [],
    [activeRun, activeRunData, activeSessionData],
  )
  const liveEvents = useMemo(
    () => (resolvedRunId ? live.events : []),
    [live.events, resolvedRunId],
  )
  const hasCommittedTerminalEvent = useMemo(
    () => selectedRunReplayEvents.some((event) => isTerminalAguiEvent(event)),
    [selectedRunReplayEvents],
  )
  const effectiveLiveEvents = useMemo(
    () => (hasCommittedTerminalEvent ? [] : liveEvents),
    [hasCommittedTerminalEvent, liveEvents],
  )
  const history = useMemo(
    () => mergeSessionHistoryPages(historyPages, effectiveLiveEvents),
    [effectiveLiveEvents, historyPages],
  )
  const timeline = history.timeline
  const currentSession = activeSessionData?.session ?? null
  const activeRunForComposer = currentSession?.active_run_id ? activeRun : null

  useEffect(() => {
    if (isNewConversationPath) {
      setIsComposingNew(true)
      return
    }
    setIsComposingNew(false)
  }, [isNewConversationPath])

  useEffect(() => {
    if (!advancedMode && detailTab === 'advanced') {
      setDetailTab('conversation')
    }
  }, [advancedMode, detailTab])

  useEffect(() => {
    const session = conversationSessions[0]
    if (
      isConversationIndex &&
      !effectiveSessionId &&
      !composingNew &&
      session?.id
    ) {
      const runId =
        session.active_run_id ??
        session.head_run_id ??
        session.latest_run?.id ??
        null
      useLayoutStore.setState({
        route: 'chat',
        selectedSessionId: session.id,
        selectedRunId: runId,
        selectedChatSessionId: session.id,
        selectedChatRunId: runId,
      })
      replaceBrowserPath(buildChatPath(session.id, runId))
    }
  }, [
    composingNew,
    conversationSessions,
    effectiveSessionId,
    isConversationIndex,
  ])

  useEffect(() => {
    if (!effectiveSessionId || effectiveRunId) return
    const nextRunId =
      activeSessionData?.session.active_run_id ??
      activeSessionData?.session.head_run_id ??
      null
    if (nextRunId) {
      useLayoutStore.setState({
        route: 'chat',
        selectedSessionId: effectiveSessionId,
        selectedRunId: nextRunId,
        selectedChatSessionId: effectiveSessionId,
        selectedChatRunId: nextRunId,
      })
      replaceBrowserPath(buildChatPath(effectiveSessionId, nextRunId))
    }
  }, [activeSessionData, effectiveRunId, effectiveSessionId])

  function startNewChat() {
    setIsComposingNew(true)
    setDetailTab('conversation')
    useLayoutStore.setState({
      route: 'chat',
      selectedSessionId: null,
      selectedRunId: null,
      selectedChatSessionId: null,
      selectedChatRunId: null,
    })
    pushBrowserPath('/conversations/new')
    setSidebarOpen(false)
  }

  function selectChat(session: SessionSummary) {
    setIsComposingNew(false)
    setDetailTab('conversation')
    selectSession(session.id)
    selectRun(
      session.active_run_id ??
        session.head_run_id ??
        session.latest_run?.id ??
        null,
    )
    setSidebarOpen(false)
  }

  if (
    sessions.isError &&
    conversationSessions.length === 0 &&
    isConversationIndex
  ) {
    return (
      <div className="h-full overflow-auto p-4 sm:p-6">
        <h1 className="sr-only">Conversations</h1>
        <QueryError
          title="Could not load conversations"
          error={sessions.error}
          onRetry={() => void sessions.refetch()}
        />
      </div>
    )
  }

  const conversationError =
    effectiveSessionId && !activeSessionData ? selectedSession.error : null
  const conversationNotFound =
    selectedSession.error instanceof ApiError &&
    selectedSession.error.status === 404
  if (conversationError) {
    return (
      <div className="h-full overflow-auto p-4 sm:p-6">
        <h1 className="sr-only">Conversations</h1>
        <QueryError
          title={
            conversationNotFound
              ? 'Conversation not found'
              : 'Could not load this conversation'
          }
          error={conversationError}
          onRetry={() => void selectedSession.refetch()}
        />
      </div>
    )
  }

  return (
    <div className="flex h-full min-h-0 overflow-hidden bg-white">
      <ChatSidebar
        sessions={conversationSessions}
        selectedSessionId={effectiveSessionId}
        loading={sessions.isLoading}
        loadingMore={sessions.isFetchingNextPage}
        hasMore={Boolean(sessions.hasNextPage)}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onNewChat={startNewChat}
        onSelect={selectChat}
        onRefresh={() => sessions.refetch()}
        onLoadMore={() => sessions.fetchNextPage()}
      />

      <section className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden bg-slate-50">
        <header className="flex h-16 shrink-0 items-center justify-between border-b border-slate-200 bg-white px-3 sm:px-5">
          <div className="flex min-w-0 items-center gap-2 sm:gap-3">
            <button
              type="button"
              className="inline-flex h-10 w-10 items-center justify-center rounded-xl border border-slate-200 bg-white text-slate-600 shadow-sm lg:hidden"
              onClick={() => setSidebarOpen(true)}
              aria-label="Open chats"
            >
              <PanelLeft className="h-4 w-4" />
            </button>
            <div className="min-w-0">
              <h1 className="truncate text-sm font-semibold text-slate-950 sm:text-base">
                {currentSession ? sessionTitle(currentSession) : 'New chat'}
              </h1>
              <p className="truncate text-xs text-slate-500">
                {currentSession
                  ? `${advancedMode ? `${formatShortId(currentSession.id, 12)} · ` : ''}${currentSession.run_count} turns`
                  : 'New conversation draft'}
              </p>
            </div>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            {currentSession ? (
              <StatusBadge status={currentSession.status} />
            ) : null}
            <button
              type="button"
              className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-700 shadow-sm transition hover:bg-slate-50"
              onClick={startNewChat}
            >
              <Plus className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">New chat</span>
            </button>
          </div>
        </header>

        {currentSession ? (
          <ConversationTabs
            value={detailTab}
            advancedMode={advancedMode}
            onChange={setDetailTab}
          />
        ) : null}
        {sessions.isError && effectiveSessionId ? (
          <div className="shrink-0 p-3 pb-0 sm:px-4">
            <QueryError
              compact
              title="Conversation list could not be refreshed"
              error={sessions.error}
              onRetry={() => void sessions.refetch()}
            />
          </div>
        ) : null}
        {sessionHistory.isError || selectedRun.isError ? (
          <div className="shrink-0 p-3 pb-0 sm:px-4">
            <QueryError
              compact
              title="Some conversation history could not be loaded"
              error={sessionHistory.error ?? selectedRun.error}
              onRetry={() => {
                void Promise.all([
                  sessionHistory.refetch(),
                  selectedRun.refetch(),
                ])
              }}
            />
          </div>
        ) : null}
        {runSessionMismatch ? (
          <div className="shrink-0 px-3 pt-3 sm:px-4" role="alert">
            <div className="rounded-xl border border-rose-200 bg-rose-50 p-3 text-sm text-rose-900">
              <p className="font-semibold">
                Run does not belong to this conversation
              </p>
              <button
                type="button"
                className="mt-2 rounded-lg border border-rose-200 bg-white px-3 py-2 text-xs font-medium text-rose-700"
                onClick={() => selectRun(null)}
              >
                Return to conversation
              </button>
            </div>
          </div>
        ) : null}
        {detailTab === 'conversation' || !currentSession ? (
          <div
            className="flex min-h-0 flex-1 flex-col"
            role={currentSession ? 'tabpanel' : undefined}
            id={currentSession ? 'conversation-panel-conversation' : undefined}
            aria-labelledby={
              currentSession ? 'conversation-tab-conversation' : undefined
            }
            tabIndex={currentSession ? 0 : undefined}
          >
            <ChatTranscript
              timeline={timeline}
              loading={
                selectedSession.isLoading ||
                selectedRun.isLoading ||
                sessionHistory.isLoading
              }
              hasSession={Boolean(currentSession)}
              history={history}
              loadingOlder={sessionHistory.isFetchingNextPage}
              onLoadOlder={() => sessionHistory.fetchNextPage()}
            />
            <ChatComposer
              selectedSessionId={effectiveSessionId}
              selectedProfile={currentSession?.profile_name ?? null}
              activeRun={activeRunForComposer}
              onSessionCreated={() => setIsComposingNew(false)}
            />
          </div>
        ) : (
          <ConversationDetailPanel
            tab={detailTab}
            advancedMode={advancedMode}
            session={currentSession}
            runs={runsForConversation(historyRuns, currentSession.runs)}
            timeline={timeline}
            workspace={sessionWorkspace.data ?? null}
            workspaceError={sessionWorkspace.error}
            onRetryWorkspace={() => void sessionWorkspace.refetch()}
            replayEvents={selectedRunReplayEvents}
            liveEvents={effectiveLiveEvents}
          />
        )}
      </section>
    </div>
  )
}

function ConversationTabs({
  value,
  advancedMode,
  onChange,
}: {
  value: ConversationTab
  advancedMode: boolean
  onChange: (value: ConversationTab) => void
}) {
  const tabs: Array<{ value: ConversationTab; label: string }> = [
    { value: 'conversation', label: 'Conversation' },
    { value: 'runs', label: 'Runs' },
    { value: 'tools', label: 'Tools' },
    { value: 'memory', label: 'Memory' },
    { value: 'workspace', label: 'Workspace' },
    ...(advancedMode
      ? ([{ value: 'advanced', label: 'Advanced' }] as const)
      : []),
  ]
  function handleKeyDown(event: KeyboardEvent<HTMLButtonElement>) {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
      return
    }
    event.preventDefault()
    const tabList = event.currentTarget.closest('[role="tablist"]')
    const triggers = Array.from(
      tabList?.querySelectorAll<HTMLButtonElement>('[role="tab"]') ?? [],
    )
    const currentIndex = triggers.indexOf(event.currentTarget)
    if (currentIndex < 0 || triggers.length === 0) return
    const nextIndex =
      event.key === 'Home'
        ? 0
        : event.key === 'End'
          ? triggers.length - 1
          : (currentIndex +
              (event.key === 'ArrowRight' ? 1 : -1) +
              triggers.length) %
            triggers.length
    triggers[nextIndex]?.focus()
    triggers[nextIndex]?.click()
  }

  return (
    <div className="scrollbar-none shrink-0 overflow-x-auto border-b border-[var(--border)] bg-[var(--surface)] px-1 sm:px-5">
      <div
        className="flex min-w-max gap-px sm:gap-1"
        role="tablist"
        aria-label="Conversation details"
      >
        {tabs.map((tab) => (
          <button
            key={tab.value}
            type="button"
            role="tab"
            id={`conversation-tab-${tab.value}`}
            aria-controls={`conversation-panel-${tab.value}`}
            aria-selected={value === tab.value}
            tabIndex={value === tab.value ? 0 : -1}
            className={cn(
              'border-b-2 px-1.5 py-3 text-xs font-medium transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--focus)] sm:px-3 sm:text-sm',
              value === tab.value
                ? 'border-[var(--primary)] text-[var(--primary)]'
                : 'border-transparent text-[var(--muted-foreground)] hover:text-[var(--foreground)]',
            )}
            onClick={() => onChange(tab.value)}
            onKeyDown={handleKeyDown}
          >
            {tab.label}
          </button>
        ))}
      </div>
    </div>
  )
}

function ConversationDetailPanel({
  tab,
  advancedMode,
  session,
  runs,
  timeline,
  workspace,
  workspaceError,
  onRetryWorkspace,
  replayEvents,
  liveEvents,
}: {
  tab: Exclude<ConversationTab, 'conversation'>
  advancedMode: boolean
  session: SessionDetail
  runs: RunSummary[]
  timeline: AguiTimelineState
  workspace: SessionWorkspaceState | null
  workspaceError: unknown
  onRetryWorkspace: () => void
  replayEvents: unknown[]
  liveEvents: unknown[]
}) {
  const toolBlocks = timeline.blocks.filter(
    (block) => block.kind === 'tool_call',
  )
  return (
    <div
      className="scrollbar-thin min-h-0 flex-1 overflow-auto p-4 sm:p-6"
      role="tabpanel"
      id={`conversation-panel-${tab}`}
      aria-labelledby={`conversation-tab-${tab}`}
      tabIndex={0}
    >
      <div className="mx-auto max-w-4xl">
        {tab === 'runs' ? (
          <div className="space-y-3">
            {runs.map((run) => (
              <article
                key={run.id}
                className="rounded-xl border border-[var(--border)] bg-[var(--surface)] p-4"
              >
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <p className="text-sm font-semibold">
                      Run {run.sequence_no}
                    </p>
                    {advancedMode ? (
                      <p className="mono mt-1 text-xs text-[var(--subtle-foreground)]">
                        {run.id}
                      </p>
                    ) : null}
                  </div>
                  <StatusBadge status={run.status} />
                </div>
                <p className="mt-3 text-sm text-[var(--muted-foreground)]">
                  {run.input_preview ??
                    run.output_text ??
                    'No preview available'}
                </p>
                {run.error_message ? (
                  <p className="mt-3 rounded-lg bg-rose-50 p-3 text-sm text-rose-700">
                    {run.error_message}
                  </p>
                ) : null}
              </article>
            ))}
          </div>
        ) : null}
        {tab === 'tools' ? (
          <div className="space-y-3">
            {toolBlocks.length ? (
              toolBlocks.map((block) => (
                <ChatBlock key={block.id} block={block} />
              ))
            ) : (
              <EmptyState
                headingLevel={2}
                title="No tool activity"
                description="Tool calls and responses for this conversation will appear here."
              />
            )}
          </div>
        ) : null}
        {tab === 'memory' ? (
          <div className="rounded-xl border border-[var(--border)] bg-[var(--surface)] p-5">
            <h2 className="text-base font-semibold">Conversation memory</h2>
            {session.memory_state ? (
              <dl className="mt-5 grid gap-4 sm:grid-cols-2">
                <DetailItem
                  label="Enabled"
                  value={session.memory_state.enabled ? 'Yes' : 'No'}
                />
                <DetailItem
                  label="Extracts"
                  value={String(session.memory_state.extract_count)}
                />
                <DetailItem
                  label="Turns since extract"
                  value={String(session.memory_state.turns_since_extract)}
                />
                <DetailItem
                  label="Pending extract"
                  value={session.memory_state.pending_extract ? 'Yes' : 'No'}
                />
                <DetailItem
                  label="Pending summary"
                  value={session.memory_state.pending_summary ? 'Yes' : 'No'}
                />
                {advancedMode ? (
                  <DetailItem
                    label="Memory session"
                    value={
                      session.memory_state.memory_session_id ?? 'Not created'
                    }
                    mono
                  />
                ) : null}
              </dl>
            ) : (
              <p className="mt-3 text-sm text-[var(--muted-foreground)]">
                Memory has not been initialized for this conversation.
              </p>
            )}
          </div>
        ) : null}
        {tab === 'workspace' ? (
          <div className="rounded-xl border border-[var(--border)] bg-[var(--surface)] p-5">
            <h2 className="text-base font-semibold">Execution workspace</h2>
            {workspaceError ? (
              <div className="mt-4">
                <QueryError
                  compact
                  title="Workspace details could not be loaded"
                  error={workspaceError}
                  onRetry={onRetryWorkspace}
                />
              </div>
            ) : (
              <>
                <dl className="mt-5 grid gap-4 sm:grid-cols-2">
                  <DetailItem
                    label="Provider"
                    value={workspace?.binding?.provider ?? 'Unknown'}
                  />
                  <DetailItem
                    label="Working directory"
                    value={workspace?.binding?.cwd ?? 'Unknown'}
                    mono
                  />
                  <DetailItem
                    label="Virtual path"
                    value={workspace?.binding?.virtual_path ?? 'Unknown'}
                    mono
                  />
                  <DetailItem
                    label="Sandbox"
                    value={workspace?.sandbox_state?.status ?? 'Not required'}
                  />
                </dl>
                <p className="mt-5 text-sm text-[var(--muted-foreground)]">
                  File browsing, memory documents, and run artifacts are
                  available in the Workspace view.
                </p>
              </>
            )}
          </div>
        ) : null}
        {tab === 'advanced' ? (
          <div className="space-y-4">
            <div className="rounded-xl border border-[var(--border)] bg-slate-950 p-4 text-slate-100">
              <p className="text-sm font-semibold">Raw runtime events</p>
              <pre className="scrollbar-thin mt-3 max-h-[60dvh] overflow-auto text-xs leading-5">
                {JSON.stringify(
                  { committed: replayEvents, live: liveEvents },
                  null,
                  2,
                )}
              </pre>
            </div>
          </div>
        ) : null}
      </div>
    </div>
  )
}

function DetailItem({
  label,
  value,
  mono,
}: {
  label: string
  value: string
  mono?: boolean
}) {
  return (
    <div>
      <dt className="text-xs font-medium uppercase tracking-wide text-[var(--subtle-foreground)]">
        {label}
      </dt>
      <dd className={cn('mt-1 break-all text-sm', mono && 'mono')}>{value}</dd>
    </div>
  )
}

function runsForConversation(primary: RunSummary[], fallback: RunSummary[]) {
  const byId = new Map<string, RunSummary>()
  for (const run of [...primary, ...fallback]) byId.set(run.id, run)
  return [...byId.values()].sort((a, b) => b.sequence_no - a.sequence_no)
}

function ChatSidebar({
  sessions,
  selectedSessionId,
  loading,
  loadingMore,
  hasMore,
  open,
  onClose,
  onNewChat,
  onSelect,
  onRefresh,
  onLoadMore,
}: {
  sessions: SessionSummary[]
  selectedSessionId: string | null
  loading: boolean
  loadingMore: boolean
  hasMore: boolean
  open: boolean
  onClose: () => void
  onNewChat: () => void
  onSelect: (session: SessionSummary) => void
  onRefresh: () => void
  onLoadMore: () => Promise<unknown>
}) {
  const contentProps = {
    sessions,
    selectedSessionId,
    loading,
    loadingMore,
    hasMore,
    onNewChat,
    onSelect,
    onRefresh,
    onLoadMore,
  }

  useEffect(() => {
    if (typeof window.matchMedia !== 'function') return
    const desktop = window.matchMedia('(min-width: 1024px)')
    const closeOnDesktop = () => {
      if (desktop.matches && open) onClose()
    }
    closeOnDesktop()
    desktop.addEventListener('change', closeOnDesktop)
    return () => desktop.removeEventListener('change', closeOnDesktop)
  }, [onClose, open])

  return (
    <>
      <Sheet
        open={open}
        onOpenChange={(nextOpen) => {
          if (!nextOpen) onClose()
        }}
      >
        <SheetContent side="left" className="w-80 max-w-[88vw] p-0 lg:hidden">
          <ChatSidebarContent {...contentProps} mobile />
        </SheetContent>
      </Sheet>
      <aside
        aria-label="Conversation list"
        className="hidden w-80 shrink-0 flex-col border-r border-slate-200 bg-white lg:flex"
      >
        <ChatSidebarContent {...contentProps} />
      </aside>
    </>
  )
}

function ChatSidebarContent({
  sessions,
  selectedSessionId,
  loading,
  loadingMore,
  hasMore,
  mobile = false,
  onNewChat,
  onSelect,
  onRefresh,
  onLoadMore,
}: {
  sessions: SessionSummary[]
  selectedSessionId: string | null
  loading: boolean
  loadingMore: boolean
  hasMore: boolean
  mobile?: boolean
  onNewChat: () => void
  onSelect: (session: SessionSummary) => void
  onRefresh: () => void
  onLoadMore: () => Promise<unknown>
}) {
  return (
    <>
      {mobile ? (
        <SheetHeader
          title="Conversations"
          description="All runtime sources"
          className="min-h-16 py-3.5"
        />
      ) : (
        <div className="flex h-16 shrink-0 items-center border-b border-slate-200 px-4">
          <div>
            <p className="text-sm font-semibold text-slate-950">
              Conversations
            </p>
            <p className="text-xs text-slate-500">All runtime sources</p>
          </div>
        </div>
      )}
      <div className="flex gap-2 border-b border-slate-200 p-3">
        <button
          type="button"
          className="inline-flex flex-1 items-center justify-center gap-2 rounded-xl bg-blue-600 px-3 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-700"
          onClick={onNewChat}
        >
          <Plus className="h-4 w-4" />
          New chat
        </button>
        <button
          type="button"
          className="inline-flex h-10 w-10 items-center justify-center rounded-xl border border-slate-200 bg-white text-slate-600 shadow-sm transition hover:bg-slate-50"
          onClick={onRefresh}
          aria-label="Refresh chats"
        >
          <RefreshCcw className="h-4 w-4" />
        </button>
      </div>
      <div className="scrollbar-thin min-h-0 flex-1 overscroll-contain overflow-auto p-3">
        {loading ? <ChatListSkeleton /> : null}
        {!loading && sessions.length === 0 ? (
          <EmptyState
            icon={MessageSquare}
            title="No conversations yet"
            description="Start a conversation here or connect an integration to receive channel work."
            className="min-h-64 bg-slate-50"
          />
        ) : null}
        <div className="space-y-2">
          {sessions.map((session) => {
            const active = selectedSessionId === session.id
            return (
              <button
                type="button"
                key={session.id}
                className={cn(
                  'w-full rounded-2xl border p-3 text-left transition',
                  active
                    ? 'border-blue-200 bg-blue-50 shadow-sm ring-1 ring-blue-100'
                    : 'border-slate-200 bg-white hover:border-blue-200 hover:bg-blue-50/40',
                )}
                onClick={() => onSelect(session)}
              >
                <p className="line-clamp-2 text-sm font-semibold leading-5 text-slate-900">
                  {sessionTitle(session)}
                </p>
                <div className="mt-3 flex items-center justify-between gap-2 text-xs text-slate-500">
                  <span>{session.run_count} turns</span>
                  <StatusBadge status={session.status} />
                </div>
              </button>
            )
          })}
          {hasMore ? (
            <button
              type="button"
              className="mt-2 w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-50 disabled:cursor-wait disabled:opacity-60"
              disabled={loadingMore}
              onClick={() => void onLoadMore()}
            >
              {loadingMore ? 'Loading…' : 'Load older conversations'}
            </button>
          ) : null}
        </div>
      </div>
    </>
  )
}

function ChatTranscript({
  timeline,
  loading,
  hasSession,
  history,
  loadingOlder,
  onLoadOlder,
}: {
  timeline: AguiTimelineState
  loading: boolean
  hasSession: boolean
  history: SessionHistoryState
  loadingOlder: boolean
  onLoadOlder: () => Promise<unknown>
}) {
  const scrollRef = useRef<HTMLElement | null>(null)
  const bottomRef = useRef<HTMLDivElement | null>(null)
  const stickToBottomRef = useRef(true)
  const previousScrollHeightRef = useRef<number | null>(null)
  const previousSessionTotalRef = useRef(history.totalRunCount)
  const blockCount = timeline.blocks.length

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
    const bottom = bottomRef.current
    if (bottom && typeof bottom.scrollIntoView === 'function') {
      bottom.scrollIntoView({ behavior: 'smooth', block: 'end' })
    }
  }, [blockCount])

  useEffect(() => {
    const previousTotal = previousSessionTotalRef.current
    previousSessionTotalRef.current = history.totalRunCount
    if (previousTotal !== history.totalRunCount) {
      stickToBottomRef.current = true
    }
  }, [history.totalRunCount])

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
      aria-label="Conversation messages"
      className="scrollbar-thin min-h-0 flex-1 overscroll-contain overflow-auto px-3 py-5 sm:px-6"
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
      <div className="mx-auto flex max-w-3xl flex-col gap-5">
        {loading ? <ChatSkeleton /> : null}
        {!loading && timeline.blocks.length === 0 ? (
          <EmptyState
            icon={Bot}
            title={hasSession ? 'No messages yet' : 'Start a web chat'}
            headingLevel={2}
            description="Send a message below and YA Claw will create a dedicated web chat session."
            className="min-h-80 bg-white"
          />
        ) : null}
        {!loading && timeline.blocks.length > 0 ? (
          <HistoryBoundary
            history={history}
            loadingOlder={loadingOlder}
            onLoadOlder={() => void loadOlder()}
          />
        ) : null}
        {timeline.blocks.map((block) => (
          <ChatBlock key={block.id} block={block} />
        ))}
        <div ref={bottomRef} />
      </div>
    </section>
  )
}

function HistoryBoundary({
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
        className="mx-auto inline-flex items-center justify-center rounded-full border border-slate-200 bg-white px-4 py-2 text-xs font-medium text-slate-600 shadow-sm transition hover:border-blue-200 hover:bg-blue-50 hover:text-blue-700 disabled:opacity-60"
        onClick={onLoadOlder}
        disabled={loadingOlder}
      >
        {loadingOlder
          ? 'Loading older messages...'
          : `Load older messages · ${history.loadedRunCount}/${history.totalRunCount}`}
      </button>
    )
  }
  return (
    <div className="mx-auto rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-500">
      Beginning of this chat
    </div>
  )
}

function ChatBlock({ block }: { block: TimelineBlock }) {
  if (block.kind === 'user_input') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[88%] rounded-3xl bg-blue-600 px-4 py-3 text-sm leading-7 text-white shadow-sm sm:max-w-[78%]">
          <div className="space-y-2">
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
      </div>
    )
  }

  if (block.kind === 'assistant_message') {
    return (
      <div className="flex items-start gap-3">
        <Avatar icon={Bot} tone="assistant" />
        <div className="min-w-0 flex-1 rounded-3xl border border-slate-200 bg-white px-4 py-3 text-sm leading-7 text-slate-900 shadow-sm">
          <MarkdownMessage content={block.content} />
        </div>
      </div>
    )
  }

  if (block.kind === 'tool_call') {
    return (
      <div className="flex items-start gap-3">
        <Avatar icon={Bot} tone="tool" />
        <details className="min-w-0 flex-1 rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
          <summary className="cursor-pointer font-medium">
            Tool call · {block.name ?? 'tool'} · {block.status}
          </summary>
          <pre className="scrollbar-thin mt-3 max-h-60 overflow-auto rounded-xl bg-white/70 p-3 text-xs leading-5 text-amber-950">
            {JSON.stringify(
              { args: block.args, result: block.result },
              null,
              2,
            )}
          </pre>
        </details>
      </div>
    )
  }

  if (block.kind === 'reasoning') return null
  if (block.kind === 'context_meter') return null
  if (block.kind === 'usage') return null

  return (
    <div className="flex items-start gap-3">
      <Avatar icon={Bot} tone="tool" />
      <details className="min-w-0 flex-1 rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-700">
        <summary className="cursor-pointer font-medium">Runtime detail</summary>
        <pre className="scrollbar-thin mt-3 max-h-60 overflow-auto rounded-xl bg-slate-50 p-3 text-xs leading-5">
          {JSON.stringify(block, null, 2)}
        </pre>
      </details>
    </div>
  )
}

function ChatComposer({
  selectedSessionId,
  selectedProfile,
  activeRun,
  onSessionCreated,
}: {
  selectedSessionId: string | null
  selectedProfile: string | null
  activeRun: RunSummary | null
  onSessionCreated: () => void
}) {
  const {
    text,
    revision,
    setText,
    clearIfUnchanged: clearDraftIfUnchanged,
  } = useSessionDraft(selectedSessionId)
  const sendingRef = useRef(false)
  const createSession = useCreateSessionMutation()
  const submitInput = useSubmitSessionInputMutation(selectedSessionId)
  const profiles = useProfilesQuery()
  const profileOptions = profiles.data ?? []
  const defaultProfileName = profileOptions[0]?.name ?? ''
  const [profileName, setProfileName] = useState(
    selectedProfile ?? defaultProfileName,
  )
  const selectSession = useLayoutStore((store) => store.selectSession)
  const selectRun = useLayoutStore((store) => store.selectRun)
  const canAppend = activeRun?.status === 'queued'
  const canSteer = activeRun?.status === 'running'

  useEffect(() => {
    setProfileName(selectedProfile ?? defaultProfileName)
  }, [defaultProfileName, selectedProfile])

  const isPending = createSession.isPending || submitInput.isPending
  const profilesReady = Boolean(selectedSessionId) || !profiles.isLoading
  const canSend =
    text.trim().length > 0 && !isPending && profilesReady && !profiles.isError

  async function send() {
    if (!canSend || sendingRef.current) return
    sendingRef.current = true
    const submittedDraft = { text, revision }
    const normalizedText = submittedDraft.text.trim()
    const inputParts: InputPart[] = [{ type: 'text', text: normalizedText }]
    const targetSessionId = selectedSessionId
    try {
      if (targetSessionId) {
        const response = await submitInput.mutateAsync({
          input_parts: inputParts,
          metadata: WEB_CHAT_METADATA,
        })
        const current = useLayoutStore.getState()
        if (
          isSubmissionTargetActive(
            current.route,
            current.selectedSessionId,
            targetSessionId,
            'chat',
          )
        ) {
          selectRun(response.run_id)
        }
      } else {
        const response = await createSession.mutateAsync({
          profile_name: profileName.trim() || null,
          input_parts: inputParts,
          metadata: WEB_CHAT_METADATA,
        })
        const current = useLayoutStore.getState()
        if (
          isSubmissionTargetActive(
            current.route,
            current.selectedSessionId,
            targetSessionId,
            'chat',
          )
        ) {
          onSessionCreated()
          selectSession(response.session.id)
          selectRun(
            response.run?.id ??
              response.session.active_run_id ??
              response.session.head_run_id ??
              null,
          )
        }
      }
      clearDraftIfUnchanged(submittedDraft)
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : 'Failed to send message',
      )
    } finally {
      sendingRef.current = false
    }
  }

  return (
    <footer className="shrink-0 border-t border-slate-200 bg-white p-3 sm:p-4">
      <div className="mx-auto max-w-3xl">
        {profiles.isError ? (
          <div className="mb-3">
            <QueryError
              compact
              title="Could not load agent profiles"
              error={profiles.error}
              onRetry={() => void profiles.refetch()}
            />
          </div>
        ) : null}
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
              : 'This session is queued. New input will be appended to the queued run.'}
          </div>
        ) : null}
        <div className="rounded-3xl border border-slate-200 bg-white p-2 shadow-sm ring-1 ring-slate-100 transition focus-within:border-blue-200 focus-within:ring-blue-100">
          <textarea
            aria-label="Message"
            className="max-h-40 min-h-20 w-full resize-none rounded-2xl border-0 px-3 py-2 text-sm leading-6 text-slate-900 outline-none placeholder:text-slate-400 disabled:bg-white disabled:text-slate-400"
            value={text}
            onChange={(event) => setText(event.target.value)}
            placeholder={
              canSteer
                ? 'Steer the active response...'
                : canAppend
                  ? 'Append to the queued run...'
                  : 'Message YA Claw...'
            }
            onKeyDown={(event) => {
              if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
                event.preventDefault()
                void send()
              }
            }}
          />
          <div className="flex items-center justify-between gap-2 border-t border-slate-100 px-1 pt-2">
            <div className="flex min-w-0 flex-1 items-center gap-2">
              {profiles.isLoading && profiles.data === undefined ? (
                <span
                  className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-500"
                  role="status"
                >
                  Loading profiles…
                </span>
              ) : profileOptions.length > 0 ? (
                <select
                  aria-label="Agent profile"
                  className="max-w-40 rounded-xl border border-slate-200 bg-slate-50 px-2 py-2 text-xs text-slate-700 outline-none ring-blue-600 focus:ring-2 disabled:text-slate-400 sm:max-w-52 sm:px-3"
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
              ) : !profiles.isError ? (
                <span className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-500">
                  No profiles
                  <Link className="font-semibold text-blue-700" to="/agents">
                    Create one
                  </Link>
                </span>
              ) : null}
              <span className="hidden text-xs text-slate-400 md:inline">
                Cmd/Ctrl + Enter to send
              </span>
            </div>
            <button
              type="button"
              className="inline-flex h-10 items-center justify-center gap-2 rounded-xl bg-blue-600 px-4 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-700 disabled:bg-slate-300"
              disabled={!canSend}
              onClick={() => void send()}
            >
              <Send className="h-4 w-4" />
              <span className="hidden sm:inline">
                {isPending
                  ? 'Sending'
                  : canSteer
                    ? 'Steer'
                    : canAppend
                      ? 'Append'
                      : 'Send'}
              </span>
            </button>
          </div>
        </div>
      </div>
    </footer>
  )
}

export function MarkdownMessage({ content }: { content: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkNormalizeRouteHeadings]}
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
          <h2
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
              'mb-3 mt-5 text-xl font-semibold text-slate-950',
              className,
            )}
            {...props}
          />
        ),
        h3: ({ className, ...props }) => (
          <h3
            className={cn(
              'mb-3 mt-5 text-lg font-semibold text-slate-950',
              className,
            )}
            {...props}
          />
        ),
        h4: ({ className, ...props }) => (
          <h4
            className={cn(
              'mb-2 mt-4 text-base font-semibold text-slate-950',
              className,
            )}
            {...props}
          />
        ),
        h5: ({ className, ...props }) => (
          <h5
            className={cn(
              'mb-2 mt-4 text-sm font-semibold text-slate-950',
              className,
            )}
            {...props}
          />
        ),
        h6: ({ className, ...props }) => (
          <h6
            className={cn(
              'mb-2 mt-4 text-xs font-semibold uppercase tracking-wide text-slate-800',
              className,
            )}
            {...props}
          />
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

function Avatar({
  icon: Icon,
  tone,
}: {
  icon: typeof Bot
  tone: 'assistant' | 'tool'
}) {
  return (
    <span
      className={cn(
        'mt-1 hidden h-8 w-8 shrink-0 items-center justify-center rounded-2xl sm:inline-flex',
        tone === 'assistant' && 'bg-slate-900 text-white',
        tone === 'tool' && 'bg-amber-100 text-amber-700',
      )}
    >
      <Icon className="h-4 w-4" />
    </span>
  )
}

function ChatListSkeleton() {
  return (
    <div className="space-y-2">
      {Array.from({ length: 5 }).map((_, index) => (
        <div
          key={index}
          className="rounded-2xl border border-slate-200 bg-white p-3"
        >
          <div className="h-4 w-full animate-pulse rounded bg-slate-100" />
          <div className="mt-2 h-4 w-2/3 animate-pulse rounded bg-slate-100" />
          <div className="mt-3 h-3 w-24 animate-pulse rounded bg-slate-100" />
        </div>
      ))}
    </div>
  )
}

function ChatSkeleton() {
  return (
    <div className="space-y-5">
      {Array.from({ length: 3 }).map((_, index) => (
        <div
          key={index}
          className={cn(
            'h-24 animate-pulse rounded-3xl bg-slate-100',
            index % 2 === 0 ? 'ml-auto w-2/3' : 'w-4/5',
          )}
        />
      ))}
    </div>
  )
}
