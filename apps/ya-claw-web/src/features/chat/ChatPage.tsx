import {
  Bot,
  ChevronLeft,
  MessageSquare,
  PanelLeft,
  Plus,
  RefreshCcw,
  Send,
} from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { toast } from 'sonner'

import {
  useCreateSessionMutation,
  useProfilesQuery,
  useRunQuery,
  useSubmitSessionInputMutation,
  useSessionHistoryQuery,
  useSessionQuery,
  useSessionsQuery,
} from '../../api/hooks'
import { EmptyState } from '../../components/EmptyState'
import { StatusBadge } from '../../components/StatusBadge'
import { cn, formatShortId } from '../../lib/utils'
import { useLayoutStore } from '../../stores/layoutStore'
import type { InputPart, RunSummary, SessionSummary } from '../../types'
import type { AguiTimelineState, TimelineBlock } from './agui/types'
import { isTerminalAguiEvent } from './eventUtils'
import { useRunEventStream } from './useRunEventStream'
import { isWebChatSession, sessionTitle } from './sessionClassification'
import {
  mergeSessionHistoryPages,
  type SessionHistoryState,
} from './sessionHistory'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

const WEB_CHAT_METADATA = { web: { surface: 'chat' } }

export function ChatPage() {
  const selectedSessionId = useLayoutStore(
    (state) => state.selectedChatSessionId,
  )
  const selectedRunId = useLayoutStore((state) => state.selectedChatRunId)
  const selectSession = useLayoutStore((state) => state.selectSession)
  const selectRun = useLayoutStore((state) => state.selectRun)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const sessions = useSessionsQuery()
  const webSessions = useMemo(
    () => (sessions.data ?? []).filter(isWebChatSession),
    [sessions.data],
  )
  const selectedSessionBelongsToChat = useMemo(
    () => webSessions.some((session) => session.id === selectedSessionId),
    [selectedSessionId, webSessions],
  )
  const effectiveSessionId = selectedSessionBelongsToChat
    ? selectedSessionId
    : null
  const effectiveRunId = selectedSessionBelongsToChat ? selectedRunId : null
  const selectedSession = useSessionQuery(effectiveSessionId)
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
  const selectedRun = useRunQuery(resolvedRunId)
  const activeRunData = resolvedRunId ? selectedRun.data : undefined
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
  const live = useRunEventStream(
    resolvedRunId,
    activeRunData?.run.status ?? null,
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
    if (!effectiveSessionId && webSessions[0]?.id) {
      selectSession(webSessions[0].id)
      selectRun(
        webSessions[0].active_run_id ??
          webSessions[0].head_run_id ??
          webSessions[0].latest_run?.id ??
          null,
      )
    }
  }, [effectiveSessionId, selectRun, selectSession, webSessions])

  useEffect(() => {
    if (!effectiveSessionId || effectiveRunId) return
    const nextRunId =
      activeSessionData?.session.active_run_id ??
      activeSessionData?.session.head_run_id ??
      null
    if (nextRunId) selectRun(nextRunId)
  }, [activeSessionData, effectiveRunId, effectiveSessionId, selectRun])

  function startNewChat() {
    selectSession(null)
    selectRun(null)
    setSidebarOpen(false)
  }

  function selectChat(session: SessionSummary) {
    selectSession(session.id)
    selectRun(
      session.active_run_id ??
        session.head_run_id ??
        session.latest_run?.id ??
        null,
    )
    setSidebarOpen(false)
  }

  return (
    <div className="flex h-full min-h-0 overflow-hidden bg-white">
      <ChatSidebar
        sessions={webSessions}
        selectedSessionId={effectiveSessionId}
        loading={sessions.isLoading}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onNewChat={startNewChat}
        onSelect={selectChat}
        onRefresh={() => sessions.refetch()}
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
              <p className="truncate text-sm font-semibold text-slate-950 sm:text-base">
                {currentSession ? sessionTitle(currentSession) : 'New chat'}
              </p>
              <p className="mono truncate text-xs text-slate-500">
                {currentSession
                  ? `${formatShortId(currentSession.id, 12)} · ${currentSession.run_count} turns`
                  : 'Web chat session'}
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
        />
      </section>
    </div>
  )
}

function ChatSidebar({
  sessions,
  selectedSessionId,
  loading,
  open,
  onClose,
  onNewChat,
  onSelect,
  onRefresh,
}: {
  sessions: SessionSummary[]
  selectedSessionId: string | null
  loading: boolean
  open: boolean
  onClose: () => void
  onNewChat: () => void
  onSelect: (session: SessionSummary) => void
  onRefresh: () => void
}) {
  return (
    <>
      <div
        className={cn(
          'fixed inset-0 z-30 bg-slate-950/30 transition lg:hidden',
          open ? 'opacity-100' : 'pointer-events-none opacity-0',
        )}
        onClick={onClose}
      />
      <aside
        className={cn(
          'fixed inset-y-0 left-0 z-40 flex w-80 max-w-[88vw] flex-col border-r border-slate-200 bg-white shadow-xl transition lg:static lg:z-auto lg:w-80 lg:translate-x-0 lg:shadow-none',
          open ? 'translate-x-0' : '-translate-x-full',
        )}
      >
        <div className="flex h-16 shrink-0 items-center justify-between border-b border-slate-200 px-4">
          <div>
            <p className="text-sm font-semibold text-slate-950">Chats</p>
            <p className="text-xs text-slate-500">Pure web conversations</p>
          </div>
          <button
            type="button"
            className="inline-flex h-9 w-9 items-center justify-center rounded-xl border border-slate-200 bg-white text-slate-600 shadow-sm lg:hidden"
            onClick={onClose}
            aria-label="Close chats"
          >
            <ChevronLeft className="h-4 w-4" />
          </button>
        </div>
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
              title="No web chats"
              description="Start a new chat from the web UI. Bridge sessions stay in Debug and Bridges."
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
          </div>
        </div>
      </aside>
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
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
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
    <main
      ref={scrollRef}
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
    </main>
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
}: {
  selectedSessionId: string | null
  selectedProfile: string | null
  activeRun: RunSummary | null
}) {
  const [text, setText] = useState('')
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
  const canSend = text.trim().length > 0 && !isPending

  async function send() {
    const normalizedText = text.trim()
    if (!normalizedText) return
    const inputParts: InputPart[] = [{ type: 'text', text: normalizedText }]
    try {
      if (selectedSessionId) {
        const response = await submitInput.mutateAsync({
          input_parts: inputParts,
          metadata: WEB_CHAT_METADATA,
        })
        selectRun(response.run_id)
      } else {
        const response = await createSession.mutateAsync({
          profile_name: profileName.trim() || null,
          input_parts: inputParts,
          metadata: WEB_CHAT_METADATA,
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
    <footer className="shrink-0 border-t border-slate-200 bg-white p-3 sm:p-4">
      <div className="mx-auto max-w-3xl">
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
                void send()
              }
            }}
          />
          <div className="flex items-center justify-between gap-2 border-t border-slate-100 px-1 pt-2">
            <div className="flex min-w-0 flex-1 items-center gap-2">
              {profileOptions.length > 0 ? (
                <select
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
              ) : (
                <span className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-500">
                  No profiles
                </span>
              )}
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
