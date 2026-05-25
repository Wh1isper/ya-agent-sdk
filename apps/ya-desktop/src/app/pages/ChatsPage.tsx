import { Loader2, Send, Square } from 'lucide-react'
import { useEffect, useMemo, useRef, useState, type FormEvent } from 'react'

import {
  isRunErrorEvent,
  isRunFinishedEvent,
  streamErrorMessage,
  streamRunId,
  streamSessionId,
  streamTextDelta,
  useActiveClawConnection,
  useCancelClawSession,
  useClawRunTraces,
  useClawSession,
  useClawSessions,
  useClawSessionTurns,
  useCreateClawSessionRunStream,
  type ClawStreamEvent,
} from '../../claw'
import { ComposerFrame, HomeStreamPreview, LiveSessionList } from '../ui'
import {
  RunDetailsDisclosure,
  SessionTurnsPanel,
} from '../chat/ConversationPanels'
import type { DesktopSpace, HomeStreamStatus } from '../types'
import {
  desktopSpaceMetadataFromWorkspace,
  sessionTitle,
  submitFormOnEnter,
  workspaceBindingFromSpace,
  workspaceFromSession,
} from '../utils'

export function ChatsPage({
  selectedSessionId,
  selectedSpace,
  onClearSession,
  onOpenSession,
}: {
  selectedSessionId: string | null
  selectedSpace: DesktopSpace
  onClearSession: () => void
  onOpenSession: (sessionId: string) => void
}) {
  const activeConnectionQuery = useActiveClawConnection()
  const connection = activeConnectionQuery.data?.connection ?? null
  const sessionsQuery = useClawSessions(connection)
  const createRunStream = useCreateClawSessionRunStream(connection)
  const cancelSession = useCancelClawSession(connection)
  const abortControllerRef = useRef<AbortController | null>(null)
  const [prompt, setPrompt] = useState('')
  const [liveStreamSessionId, setLiveStreamSessionId] = useState<string | null>(
    null,
  )
  const [streamStatus, setStreamStatus] = useState<HomeStreamStatus>('idle')
  const [streamOutput, setStreamOutput] = useState('')
  const [streamError, setStreamError] = useState<string | null>(null)
  const [streamEventCount, setStreamEventCount] = useState(0)
  const [lastRunLabel, setLastRunLabel] = useState<string | null>(null)
  const sessions = useMemo(() => sessionsQuery.data ?? [], [sessionsQuery.data])
  const selectedSessionExists = selectedSessionId
    ? sessions.some((session) => session.id === selectedSessionId)
    : false
  const effectiveSessionId = selectedSessionExists ? selectedSessionId : null
  const sessionQuery = useClawSession(connection, effectiveSessionId)
  const turnsQuery = useClawSessionTurns(connection, effectiveSessionId)
  const selectedSession =
    sessionQuery.data?.session ??
    sessions.find((session) => session.id === effectiveSessionId) ??
    null
  const runs =
    sessionQuery.data?.session.runs ??
    (selectedSession?.latest_run ? [selectedSession.latest_run] : [])
  const [runDetailsOpen, setRunDetailsOpen] = useState(false)
  const traceQueries = useClawRunTraces(connection, runDetailsOpen ? runs : [])
  const trimmedPrompt = prompt.trim()
  const streamingActive =
    streamStatus === 'connecting' || streamStatus === 'streaming'
  const activeRunId =
    selectedSession?.active_run_id ?? selectedSession?.activeRunId
  const canContinue = Boolean(
    connection && effectiveSessionId && trimmedPrompt && !streamingActive,
  )
  const canCancel = Boolean(connection && effectiveSessionId && activeRunId)
  const scopedLiveOutput =
    liveStreamSessionId === effectiveSessionId ? streamOutput : ''
  const scopedLiveStatus =
    liveStreamSessionId === effectiveSessionId ? streamStatus : 'idle'

  useEffect(() => () => abortControllerRef.current?.abort(), [])

  useEffect(() => {
    if (!selectedSessionId || sessionsQuery.isLoading || selectedSessionExists)
      return
    onClearSession()
  }, [
    onClearSession,
    selectedSessionExists,
    selectedSessionId,
    sessionsQuery.isLoading,
  ])

  async function handleContinue(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!effectiveSessionId || !trimmedPrompt || streamingActive) return
    const abortController = new AbortController()
    abortControllerRef.current?.abort()
    abortControllerRef.current = abortController
    setLiveStreamSessionId(effectiveSessionId)
    setStreamStatus('connecting')
    setStreamOutput('')
    setStreamError(null)
    setStreamEventCount(0)
    setLastRunLabel(null)

    try {
      const sessionWorkspace = workspaceFromSession(
        sessionQuery.data?.session ?? null,
      )
      const fallbackWorkspace = workspaceBindingFromSpace(selectedSpace)
      const workspace = sessionWorkspace ?? fallbackWorkspace
      const spaceMetadata = desktopSpaceMetadataFromWorkspace(
        workspace,
        selectedSpace,
      )

      await createRunStream.mutateAsync({
        sessionId: effectiveSessionId,
        input: {
          workspace,
          metadata: {
            desktop: {
              source: 'chat_continue',
              space_id: spaceMetadata.spaceId,
              space_name: spaceMetadata.spaceName,
            },
          },
          input_parts: [{ type: 'text', text: trimmedPrompt }],
        },
        signal: abortController.signal,
        handlers: {
          onOpen: () => setStreamStatus('streaming'),
          onEvent: handleStreamEvent,
          onClose: () => {
            setStreamStatus((status) =>
              status === 'failed' ? status : 'completed',
            )
          },
        },
      })
      setPrompt('')
      setStreamStatus((status) => (status === 'failed' ? status : 'completed'))
    } catch (error) {
      if (abortController.signal.aborted) {
        if (abortControllerRef.current === abortController)
          setStreamStatus('idle')
        return
      }
      setStreamStatus('failed')
      setStreamError(error instanceof Error ? error.message : String(error))
    } finally {
      if (abortControllerRef.current === abortController)
        abortControllerRef.current = null
    }
  }

  function handleStreamEvent(event: ClawStreamEvent) {
    setStreamEventCount((count) => count + 1)
    const runId = streamRunId(event)
    if (runId) setLastRunLabel(runId.slice(0, 8))
    const sessionId = streamSessionId(event)
    if (sessionId) setLiveStreamSessionId(sessionId)
    const delta = streamTextDelta(event)
    if (delta) setStreamOutput((output) => `${output}${delta}`)
    if (isRunErrorEvent(event)) {
      setStreamStatus('failed')
      setStreamError(streamErrorMessage(event))
      return
    }
    if (isRunFinishedEvent(event)) {
      setStreamStatus((status) => (status === 'failed' ? status : 'completed'))
      return
    }
    setStreamStatus((status) =>
      status === 'failed' || status === 'completed' ? status : 'streaming',
    )
  }

  async function handleCancelActiveRun() {
    if (!effectiveSessionId || !canCancel) return
    await cancelSession.mutateAsync(effectiveSessionId)
  }

  return (
    <div className="grid h-full min-h-[680px] grid-cols-1 bg-white xl:grid-cols-[300px_1fr]">
      <aside className="hidden min-h-0 border-r border-black/[0.08] bg-[#fbfbfa] p-3 xl:flex xl:flex-col">
        <div className="px-2 py-2">
          <h2 className="text-sm font-semibold text-[#171717]">Chats</h2>
          <p className="mt-1 text-xs text-[#8a8a8a]">
            Pick up where work paused.
          </p>
        </div>
        <div className="min-h-0 flex-1 overflow-auto pt-2">
          <LiveSessionList
            compact
            connectionReady={Boolean(connection)}
            emptyDetail="Start from Home after Local Claw is running."
            emptyTitle="No chats yet"
            error={sessionsQuery.error}
            loading={sessionsQuery.isLoading}
            onSelectSession={onOpenSession}
            selectedSessionId={effectiveSessionId}
            sessions={sessions}
          />
        </div>
      </aside>
      <section className="flex min-h-0 flex-col">
        {!effectiveSessionId && (
          <div className="max-h-[42vh] overflow-auto border-b border-black/[0.08] bg-[#fbfbfa] p-3 xl:hidden">
            <LiveSessionList
              compact
              connectionReady={Boolean(connection)}
              emptyDetail="Start from Home after Local Claw is running."
              emptyTitle="No chats yet"
              error={sessionsQuery.error}
              loading={sessionsQuery.isLoading}
              onSelectSession={onOpenSession}
              selectedSessionId={effectiveSessionId}
              sessions={sessions}
            />
          </div>
        )}
        <div className="flex h-16 shrink-0 items-center justify-between border-b border-black/[0.08] px-5">
          <div className="min-w-0">
            <h2 className="truncate text-sm font-semibold text-[#171717]">
              {selectedSession
                ? sessionTitle(selectedSession)
                : connection
                  ? 'Select a chat'
                  : 'Local Claw is offline'}
            </h2>
            <p className="mt-1 truncate text-xs text-[#8a8a8a]">
              {selectedSession
                ? `${selectedSession.run_count ?? selectedSession.runCount ?? 0} runs · ${selectedSession.profile_name ?? selectedSession.profileName ?? 'default'} profile`
                : (activeConnectionQuery.data?.status.message ??
                  'Start Local Claw from Settings.')}
            </p>
          </div>
          {canCancel && (
            <button
              className="inline-flex h-9 items-center gap-2 rounded-xl border border-amber-200 bg-amber-50 px-3 text-xs font-medium text-amber-800 transition hover:bg-amber-100"
              type="button"
              onClick={handleCancelActiveRun}
            >
              <Square className="h-3.5 w-3.5" />
              Cancel run
            </button>
          )}
        </div>
        <div className="min-h-0 flex-1 overflow-auto bg-white px-5 py-6">
          <div className="mx-auto max-w-3xl">
            <SessionTurnsPanel
              error={sessionQuery.error ?? turnsQuery.error}
              liveOutput={scopedLiveOutput}
              liveStatus={scopedLiveStatus}
              loading={sessionQuery.isLoading || turnsQuery.isLoading}
              replayMessage={sessionQuery.data?.message ?? null}
              selectedSession={selectedSession}
              sessionDetail={sessionQuery.data?.session ?? null}
              turns={turnsQuery.data?.turns ?? []}
            />
            {selectedSession && (
              <RunDetailsDisclosure
                loading={traceQueries.some((query) => query.isLoading)}
                error={traceQueries.find((query) => query.error)?.error ?? null}
                onToggleOpen={setRunDetailsOpen}
                open={runDetailsOpen}
                runs={runs}
                traces={traceQueries.flatMap((query) =>
                  query.data ? [query.data] : [],
                )}
              />
            )}
          </div>
        </div>
        <div className="shrink-0 border-t border-black/[0.08] bg-white p-4">
          <form className="mx-auto max-w-3xl" onSubmit={handleContinue}>
            <ComposerFrame compact>
              <div className="flex items-end gap-3">
                <textarea
                  value={prompt}
                  onChange={(event) => setPrompt(event.target.value)}
                  aria-label="Message YA in this chat"
                  className="max-h-32 min-h-12 flex-1 resize-none bg-transparent text-sm leading-6 text-[#171717] outline-none placeholder:text-[#9a9a9a]"
                  onKeyDown={submitFormOnEnter}
                  placeholder="Message YA..."
                />
                <button
                  className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-[#171717] text-white transition hover:bg-[#2f2f2f] disabled:cursor-not-allowed disabled:bg-[#d8d8d4]"
                  disabled={!canContinue}
                  type="submit"
                  aria-label="Send message"
                >
                  {streamingActive ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Send className="h-4 w-4" />
                  )}
                </button>
              </div>
              <HomeStreamPreview
                eventCount={streamEventCount}
                error={streamError}
                output={scopedLiveOutput}
                runLabel={lastRunLabel}
                status={scopedLiveStatus}
              />
            </ComposerFrame>
          </form>
        </div>
      </section>
    </div>
  )
}
