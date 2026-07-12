import { fetchEventSource } from '@microsoft/fetch-event-source'
import { useQueryClient } from '@tanstack/react-query'
import { useEffect, useState } from 'react'

import { queryKeys } from '../../api/queryKeys'
import { useConnectionStore } from '../../stores/connectionStore'
import type { AguiEvent, RunSummary } from '../../types'
import type { StreamStatus } from '../../lib/status'
import { isTerminalAguiEvent } from './eventUtils'

const maxBufferedEvents = 1_000
const eventBatchIntervalMs = 32

export function useRunEventStream(
  runId: string | null,
  status: RunSummary['status'] | null,
  sessionId: string | null,
): { status: StreamStatus; events: AguiEvent[] } {
  const baseUrl = useConnectionStore((state) => state.baseUrl)
  const apiToken = useConnectionStore((state) => state.apiToken)
  const connectionScope = useConnectionStore((state) => state.connectionScope)
  const invalidateConnection = useConnectionStore(
    (state) => state.invalidateConnection,
  )
  const queryClient = useQueryClient()
  const [streamState, setStreamState] = useState<{
    connectionScope: string
    runId: string | null
    status: StreamStatus
    events: AguiEvent[]
  }>({ connectionScope, runId, status: 'idle', events: [] })

  useEffect(() => {
    const ownsStream = (state: typeof streamState) =>
      state.connectionScope === connectionScope && state.runId === runId
    const setOwnedStatus = (nextStatus: StreamStatus) => {
      setStreamState((previous) =>
        ownsStream(previous) ? { ...previous, status: nextStatus } : previous,
      )
    }
    setStreamState((previous) =>
      ownsStream(previous)
        ? previous
        : { connectionScope, runId, status: 'idle', events: [] },
    )
    if (!runId || (status !== 'running' && status !== 'queued')) {
      setOwnedStatus(runId ? 'closed' : 'idle')
      return
    }
    if (!apiToken.trim()) {
      setOwnedStatus('idle')
      return
    }

    const controller = new AbortController()
    let pendingEvents: AguiEvent[] = []
    let flushTimer: ReturnType<typeof setTimeout> | null = null
    const flushEvents = () => {
      if (flushTimer) clearTimeout(flushTimer)
      flushTimer = null
      if (pendingEvents.length === 0 || controller.signal.aborted) return
      const batch = pendingEvents
      pendingEvents = []
      setStreamState((previous) =>
        ownsStream(previous)
          ? {
              ...previous,
              events: [...previous.events, ...batch].slice(-maxBufferedEvents),
            }
          : previous,
      )
    }
    const queueEvent = (event: AguiEvent) => {
      pendingEvents.push(event)
      if (!flushTimer) {
        flushTimer = setTimeout(flushEvents, eventBatchIntervalMs)
      }
    }
    setOwnedStatus('connecting')

    const streamPromise = fetchEventSource(
      `${baseUrl.replace(/\/$/, '')}/api/v1/runs/${encodeURIComponent(runId)}/events`,
      {
        signal: controller.signal,
        headers: { Authorization: `Bearer ${apiToken.trim()}` },
        openWhenHidden: true,
        async onopen(response) {
          if (!response.ok) {
            setOwnedStatus('error')
            if (response.status === 401) {
              invalidateConnection(
                'Your API token is invalid or expired.',
                connectionScope,
              )
            }
            throw new Error(`run event stream failed with ${response.status}`)
          }
          setOwnedStatus('streaming')
        },
        onmessage(message) {
          if (!message.data) return
          let event: AguiEvent
          try {
            event = JSON.parse(message.data) as AguiEvent
          } catch (error) {
            console.warn('Ignored malformed run event', error)
            return
          }
          queueEvent(event)
          if (isTerminalAguiEvent(event)) {
            flushEvents()
            void Promise.all([
              queryClient.invalidateQueries({ queryKey: queryKeys.sessions }),
              sessionId
                ? queryClient.invalidateQueries({
                    queryKey: queryKeys.session(sessionId),
                  })
                : Promise.resolve(),
              queryClient.invalidateQueries({ queryKey: queryKeys.run(runId) }),
            ])
            setOwnedStatus('closed')
          }
        },
        onclose() {
          setOwnedStatus('closed')
        },
        onerror(error) {
          if (!controller.signal.aborted) setOwnedStatus('error')
          throw error
        },
      },
    )
    void streamPromise.catch((error: unknown) => {
      if (
        controller.signal.aborted ||
        (error instanceof Error && error.name === 'AbortError')
      ) {
        return
      }
      setOwnedStatus('error')
    })

    return () => {
      controller.abort()
      if (flushTimer) clearTimeout(flushTimer)
      pendingEvents = []
    }
  }, [
    apiToken,
    baseUrl,
    connectionScope,
    invalidateConnection,
    queryClient,
    runId,
    sessionId,
    status,
  ])

  const ownsCurrentStream =
    streamState.connectionScope === connectionScope &&
    streamState.runId === runId
  return {
    status: ownsCurrentStream ? streamState.status : 'idle',
    events: ownsCurrentStream ? streamState.events : [],
  }
}
