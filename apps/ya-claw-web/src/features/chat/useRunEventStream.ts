import { fetchEventSource } from '@microsoft/fetch-event-source'
import { useQueryClient } from '@tanstack/react-query'
import { useEffect, useState } from 'react'

import { queryKeys } from '../../api/queryKeys'
import { useConnectionStore } from '../../stores/connectionStore'
import type { AguiEvent, RunSummary } from '../../types'
import type { StreamStatus } from '../../lib/status'
import { isTerminalAguiEvent } from './eventUtils'

const maxBufferedEvents = 1_000

export function useRunEventStream(
  runId: string | null,
  status: RunSummary['status'] | null,
  sessionId: string | null,
): { status: StreamStatus; events: AguiEvent[] } {
  const baseUrl = useConnectionStore((state) => state.baseUrl)
  const apiToken = useConnectionStore((state) => state.apiToken)
  const queryClient = useQueryClient()
  const [streamStatus, setStreamStatus] = useState<StreamStatus>('idle')
  const [events, setEvents] = useState<AguiEvent[]>([])

  useEffect(() => {
    setEvents([])
  }, [runId])

  useEffect(() => {
    if (!runId || (status !== 'running' && status !== 'queued')) {
      setStreamStatus(runId ? 'closed' : 'idle')
      return
    }
    if (!apiToken.trim()) {
      setStreamStatus('idle')
      return
    }

    const controller = new AbortController()
    setStreamStatus('connecting')

    void fetchEventSource(
      `${baseUrl.replace(/\/$/, '')}/api/v1/runs/${encodeURIComponent(runId)}/events`,
      {
        signal: controller.signal,
        headers: { Authorization: `Bearer ${apiToken.trim()}` },
        openWhenHidden: true,
        async onopen(response) {
          if (!response.ok) {
            setStreamStatus('error')
            throw new Error(`run event stream failed with ${response.status}`)
          }
          setStreamStatus('streaming')
        },
        onmessage(message) {
          if (!message.data) return
          const event = JSON.parse(message.data) as AguiEvent
          setEvents((previous) =>
            [...previous, event].slice(-maxBufferedEvents),
          )
          if (isTerminalAguiEvent(event)) {
            void Promise.all([
              queryClient.invalidateQueries({ queryKey: queryKeys.sessions }),
              sessionId
                ? queryClient.invalidateQueries({
                    queryKey: queryKeys.session(sessionId),
                  })
                : Promise.resolve(),
              queryClient.invalidateQueries({ queryKey: queryKeys.run(runId) }),
            ])
            setStreamStatus('closed')
          }
        },
        onclose() {
          setStreamStatus('closed')
        },
        onerror(error) {
          if (!controller.signal.aborted) setStreamStatus('error')
          throw error
        },
      },
    )

    return () => {
      controller.abort()
    }
  }, [apiToken, baseUrl, queryClient, runId, sessionId, status])

  return { status: streamStatus, events }
}
