import { fetchEventSource } from '@microsoft/fetch-event-source'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, render } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useConnectionStore } from '../../stores/connectionStore'
import type { AguiEvent, RunSummary } from '../../types'
import type { StreamStatus } from '../../lib/status'
import { useRunEventStream } from './useRunEventStream'

vi.mock('@microsoft/fetch-event-source', () => ({
  fetchEventSource: vi.fn(() => new Promise<void>(() => undefined)),
}))

type StreamObservation = {
  runId: string | null
  status: StreamStatus
  events: AguiEvent[]
}

const observations: StreamObservation[] = []

function Probe({
  runId,
  status,
}: {
  runId: string | null
  status: RunSummary['status'] | null
}) {
  const stream = useRunEventStream(runId, status, 'session-1')
  observations.push({ runId, status: stream.status, events: stream.events })
  return null
}

function streamMessage(messageId: string, delta: string) {
  return {
    data: JSON.stringify({
      type: 'TEXT_MESSAGE_CONTENT',
      message_id: messageId,
      delta,
    }),
    event: '',
    id: '',
    retry: undefined,
  }
}

function Wrapper({ children }: { children: ReactNode }) {
  return (
    <QueryClientProvider
      client={
        new QueryClient({ defaultOptions: { queries: { retry: false } } })
      }
    >
      {children}
    </QueryClientProvider>
  )
}

describe('useRunEventStream ownership', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.clearAllMocks()
    observations.length = 0
    useConnectionStore.setState({
      baseUrl: 'https://claw.example',
      apiToken: 'test-token',
      connectionScope: 'stream-owner-test',
      connectionIssue: null,
    })
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('isolates events, callbacks, timers, and status by run and connection scope', async () => {
    const view = render(<Probe runId="run-a" status="running" />, {
      wrapper: Wrapper,
    })
    expect(fetchEventSource).toHaveBeenCalledOnce()

    const runAOptions = vi.mocked(fetchEventSource).mock.calls[0]?.[1]
    await act(async () => {
      await runAOptions?.onopen?.(new Response(null, { status: 200 }))
      runAOptions?.onmessage?.(streamMessage('message-a', 'event from run A'))
      vi.advanceTimersByTime(32)
    })

    const latestRunA = observations[observations.length - 1]
    expect(latestRunA).toMatchObject({
      runId: 'run-a',
      status: 'streaming',
    })
    expect(latestRunA?.events).toHaveLength(1)

    runAOptions?.onmessage?.(
      streamMessage('pending-message-a', 'pending event from run A'),
    )
    observations.length = 0
    view.rerender(<Probe runId="run-b" status="running" />)

    expect(observations[0]).toEqual({
      runId: 'run-b',
      status: 'idle',
      events: [],
    })
    expect(fetchEventSource).toHaveBeenCalledTimes(2)
    const runBOptions = vi.mocked(fetchEventSource).mock.calls[1]?.[1]
    await act(async () => {
      await runBOptions?.onopen?.(new Response(null, { status: 200 }))
      runBOptions?.onmessage?.(streamMessage('message-b', 'event from run B'))
      vi.advanceTimersByTime(32)
    })
    expect(observations[observations.length - 1]).toMatchObject({
      runId: 'run-b',
      status: 'streaming',
      events: [{ message_id: 'message-b' }],
    })

    await act(async () => {
      await runAOptions?.onopen?.(new Response(null, { status: 200 }))
      runAOptions?.onmessage?.(
        streamMessage('stale-message-a', 'late event from run A'),
      )
      runAOptions?.onclose?.()
      expect(() =>
        runAOptions?.onerror?.(new Error('late run A stream error')),
      ).toThrow('late run A stream error')
      vi.advanceTimersByTime(64)
    })
    expect(observations[observations.length - 1]).toMatchObject({
      runId: 'run-b',
      status: 'streaming',
      events: [{ message_id: 'message-b' }],
    })

    runBOptions?.onmessage?.(
      streamMessage('pending-message-b', 'pending old-scope event'),
    )
    observations.length = 0
    act(() => {
      useConnectionStore.setState({ connectionScope: 'stream-owner-test-2' })
    })

    expect(observations[0]).toEqual({
      runId: 'run-b',
      status: 'idle',
      events: [],
    })
    expect(fetchEventSource).toHaveBeenCalledTimes(3)
    const newScopeOptions = vi.mocked(fetchEventSource).mock.calls[2]?.[1]
    await act(async () => {
      await newScopeOptions?.onopen?.(new Response(null, { status: 200 }))
      newScopeOptions?.onmessage?.(
        streamMessage('message-new-scope', 'event from new scope'),
      )
      vi.advanceTimersByTime(32)
    })

    await act(async () => {
      await runBOptions?.onopen?.(new Response(null, { status: 200 }))
      runBOptions?.onmessage?.(
        streamMessage('stale-message-b', 'late old-scope event'),
      )
      runBOptions?.onclose?.()
      expect(() =>
        runBOptions?.onerror?.(new Error('late old-scope stream error')),
      ).toThrow('late old-scope stream error')
      vi.advanceTimersByTime(64)
    })
    expect(observations[observations.length - 1]).toMatchObject({
      runId: 'run-b',
      status: 'streaming',
      events: [{ message_id: 'message-new-scope' }],
    })
  })
})
