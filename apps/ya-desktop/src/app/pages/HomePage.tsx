import { Folder, Loader2, Send, Sparkles, TerminalSquare } from 'lucide-react'
import { useEffect, useMemo, useRef, useState, type FormEvent } from 'react'

import { isRunErrorEvent, isRunFinishedEvent, streamErrorMessage, streamRunId, streamSessionId, streamTextDelta, useActiveClawConnection, useClawHealth, useClawInfo, useClawProfiles, useClawSessions, useCreateClawSessionStream, type ClawStreamEvent } from '../../claw'
import type { DesktopSpace, HomeStreamStatus } from '../types'
import { ComposerFrame, HomeStreamPreview, InfoPill, LiveSessionList, SelectPill } from '../ui'
import { enabledProfiles, profileNameOrDefault, spaceDetail, submitFormOnEnter, workspaceBindingFromSpace } from '../utils'

export function HomePage({
  selectedSpace,
  onOpenSession,
}: {
  selectedSpace: DesktopSpace
  onOpenSession: (sessionId: string) => void
}) {
  const activeConnectionQuery = useActiveClawConnection()
  const connection = activeConnectionQuery.data?.connection ?? null
  const healthQuery = useClawHealth(connection)
  const infoQuery = useClawInfo(connection)
  const profilesQuery = useClawProfiles(connection)
  const sessionsQuery = useClawSessions(connection)
  const createSessionStream = useCreateClawSessionStream(connection)
  const abortControllerRef = useRef<AbortController | null>(null)
  const [prompt, setPrompt] = useState('')
  const [streamStatus, setStreamStatus] = useState<HomeStreamStatus>('idle')
  const [streamOutput, setStreamOutput] = useState('')
  const [streamError, setStreamError] = useState<string | null>(null)
  const [streamEventCount, setStreamEventCount] = useState(0)
  const [lastRunLabel, setLastRunLabel] = useState<string | null>(null)
  const [createdSessionId, setCreatedSessionId] = useState<string | null>(null)
  const [selectedProfileName, setSelectedProfileName] = useState('default')
  const profiles = useMemo(
    () => enabledProfiles(profilesQuery.data ?? []),
    [profilesQuery.data],
  )
  const effectiveProfileName = profileNameOrDefault(
    selectedProfileName,
    profiles,
  )
  const trimmedPrompt = prompt.trim()
  const streamingActive =
    streamStatus === 'connecting' || streamStatus === 'streaming'
  const canStart = Boolean(connection && trimmedPrompt && !streamingActive)
  const recentSessions = sessionsQuery.data?.slice(0, 5) ?? []
  const selectedWorkspace = workspaceBindingFromSpace(selectedSpace)
  const runtimeDetail = connection
    ? `${infoQuery.data?.serviceVersion ?? infoQuery.data?.version ?? 'Claw'} · ${healthQuery.data?.status ?? 'checking'}`
    : (activeConnectionQuery.data?.status.message ?? 'Local Claw is stopped')

  useEffect(() => () => abortControllerRef.current?.abort(), [])

  async function handleStart(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!connection || !trimmedPrompt || streamingActive) return

    const abortController = new AbortController()
    abortControllerRef.current?.abort()
    abortControllerRef.current = abortController
    setStreamStatus('connecting')
    setStreamOutput('')
    setStreamError(null)
    setStreamEventCount(0)
    setLastRunLabel(null)
    setCreatedSessionId(null)

    try {
      await createSessionStream.mutateAsync({
        input: {
          profile_name: effectiveProfileName,
          workspace: selectedWorkspace,
          metadata: {
            title: trimmedPrompt.slice(0, 120),
            desktop: {
              source: 'home_command',
              space_id: selectedSpace.id,
              space_name: selectedSpace.name,
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
    if (sessionId) setCreatedSessionId(sessionId)
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

  return (
    <div className="mx-auto flex min-h-full w-full max-w-5xl flex-col px-5 py-10 lg:px-8">
      <section className="flex flex-1 flex-col justify-center pb-8">
        <div className="mx-auto w-full max-w-3xl text-center">
          <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-2xl bg-[#171717] text-white shadow-sm">
            <Sparkles className="h-5 w-5" />
          </div>
          <h2 className="mt-6 text-4xl font-semibold tracking-[-0.04em] text-[#171717] md:text-5xl">
            What should YA do next?
          </h2>
          <p className="mx-auto mt-3 max-w-xl text-sm leading-6 text-[#6b6b6b]">
            Start with one request. YA keeps the workspace, runtime, and
            approvals in the background until they need your attention.
          </p>
        </div>
        <form className="mx-auto mt-8 w-full max-w-3xl" onSubmit={handleStart}>
          <ComposerFrame>
            <textarea
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              aria-label="Start a new YA request"
              className="max-h-44 min-h-24 w-full resize-none bg-transparent px-1 text-base leading-7 text-[#171717] outline-none placeholder:text-[#9a9a9a]"
              onKeyDown={submitFormOnEnter}
              placeholder="Ask YA to write, debug, refactor, research, or ship something..."
            />
            <div className="mt-3 flex flex-wrap items-center justify-between gap-3 border-t border-black/[0.06] pt-3">
              <div className="flex min-w-0 flex-wrap items-center gap-2 text-xs text-[#6b6b6b]">
                <SelectPill
                  label="Profile"
                  value={effectiveProfileName}
                  onChange={setSelectedProfileName}
                  options={
                    profiles.length
                      ? profiles.map((profile) => ({
                          label: `${profile.name} · ${profile.model}`,
                          value: profile.name,
                        }))
                      : [{ label: 'default', value: 'default' }]
                  }
                />
                <InfoPill icon={Folder} text={spaceDetail(selectedSpace)} />
                <InfoPill icon={TerminalSquare} text={runtimeDetail} />
              </div>
              <button
                className="inline-flex h-10 items-center gap-2 rounded-xl bg-[#171717] px-4 text-sm font-medium text-white transition hover:bg-[#2f2f2f] disabled:cursor-not-allowed disabled:bg-[#d8d8d4]"
                disabled={!canStart}
                type="submit"
              >
                {streamingActive ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Send className="h-4 w-4" />
                )}
                {streamingActive ? 'Running' : 'Start'}
              </button>
            </div>
            <HomeStreamPreview
              eventCount={streamEventCount}
              error={streamError}
              output={streamOutput}
              onOpenSession={
                createdSessionId
                  ? () => onOpenSession(createdSessionId)
                  : undefined
              }
              runLabel={lastRunLabel}
              status={streamStatus}
            />
          </ComposerFrame>
        </form>
      </section>
      <section className="mx-auto w-full max-w-3xl pb-10">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-[#171717]">Recent chats</h3>
          <span className="text-xs text-[#8a8a8a]">
            {connection ? 'Live' : 'Offline'}
          </span>
        </div>
        <div className="mt-3 space-y-2">
          <LiveSessionList
            compact
            connectionReady={Boolean(connection)}
            emptyDetail="Your conversations will appear here after the first run."
            emptyTitle="No recent chats"
            error={sessionsQuery.error}
            loading={sessionsQuery.isLoading}
            onSelectSession={onOpenSession}
            sessions={recentSessions}
          />
        </div>
      </section>
    </div>
  )
}
