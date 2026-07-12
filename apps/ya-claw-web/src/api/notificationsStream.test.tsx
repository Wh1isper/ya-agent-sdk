import { fetchEventSource } from '@microsoft/fetch-event-source'
import {
  type InfiniteData,
  QueryClient,
  QueryClientProvider,
} from '@tanstack/react-query'
import { act, renderHook, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useConnectionStore } from '../stores/connectionStore'
import type { SessionListResponse, SessionSummary } from '../types'
import { useNotificationStream } from './notificationsStream'
import { queryKeys } from './queryKeys'

vi.mock('@microsoft/fetch-event-source', () => ({
  fetchEventSource: vi.fn(),
}))

function createTestContext() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  const Wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  )
  return { queryClient, Wrapper }
}

function createWrapper() {
  return createTestContext().Wrapper
}

function sessionSummary(id: string, updatedAt: string): SessionSummary {
  return {
    id,
    session_type: 'conversation',
    metadata: {},
    created_at: updatedAt,
    updated_at: updatedAt,
    status: 'idle',
    run_count: 1,
    latest_run: {
      id: `run-${id}`,
      session_id: id,
      sequence_no: 1,
      status: 'completed',
      trigger_type: 'user',
      created_at: updatedAt,
    },
  }
}

describe('useNotificationStream lifecycle', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    useConnectionStore.setState({
      baseUrl: 'http://claw.local',
      apiToken: 'token',
      connectionScope: 'scope-one',
      connectionIssue: null,
    })
  })

  it('clears the replay cursor and aborts the previous stream on scope changes', async () => {
    vi.mocked(fetchEventSource).mockImplementation(
      () => new Promise<void>(() => undefined),
    )
    renderHook(() => useNotificationStream(), { wrapper: createWrapper() })

    await waitFor(() => expect(fetchEventSource).toHaveBeenCalledTimes(1))
    const firstOptions = vi.mocked(fetchEventSource).mock.calls[0]![1]
    act(() => {
      firstOptions.onmessage?.({ data: '', event: '', id: 'cursor-one' })
    })

    act(() => {
      useConnectionStore.setState({ connectionScope: 'scope-two' })
    })

    await waitFor(() => expect(fetchEventSource).toHaveBeenCalledTimes(2))
    expect(firstOptions.signal?.aborted).toBe(true)
    const secondHeaders = vi.mocked(fetchEventSource).mock.calls[1]![1]
      .headers as Record<string, string>
    expect(secondHeaders['Last-Event-ID']).toBeUndefined()
  })

  it('keeps transient stream failures retryable', async () => {
    vi.mocked(fetchEventSource).mockImplementation(
      () => new Promise<void>(() => undefined),
    )
    const { result } = renderHook(() => useNotificationStream(), {
      wrapper: createWrapper(),
    })

    await waitFor(() => expect(fetchEventSource).toHaveBeenCalledTimes(1))
    const options = vi.mocked(fetchEventSource).mock.calls[0]![1]
    await act(async () => {
      await options.onopen?.(new Response(null, { status: 200 }))
    })
    expect(result.current).toBe('connected')

    let retryDelay: number | void | null = undefined
    act(() => {
      retryDelay = options.onerror?.(new Error('temporary network failure'))
    })
    expect(retryDelay).toBe(2_000)
    expect(result.current).toBe('connecting')
  })

  it('patches and reorders cached infinite session pages', async () => {
    vi.mocked(fetchEventSource).mockImplementation(
      () => new Promise<void>(() => undefined),
    )
    const { queryClient, Wrapper } = createTestContext()
    queryClient.setQueryData<InfiniteData<SessionListResponse>>(
      queryKeys.sessions,
      {
        pages: [
          {
            sessions: [
              sessionSummary('newest', '2026-07-12T10:00:00Z'),
              sessionSummary('middle', '2026-07-12T09:00:00Z'),
            ],
            total: 3,
            limit: 2,
            has_more: true,
            next_before_updated_at: '2026-07-12T09:00:00Z',
            next_before_id: 'middle',
          },
          {
            sessions: [sessionSummary('oldest', '2026-07-12T08:00:00Z')],
            total: 3,
            limit: 2,
            has_more: false,
          },
        ],
        pageParams: [{}, { beforeUpdatedAt: '2026-07-12T09:00:00Z' }],
      },
    )
    renderHook(() => useNotificationStream(), { wrapper: Wrapper })

    await waitFor(() => expect(fetchEventSource).toHaveBeenCalledTimes(1))
    const options = vi.mocked(fetchEventSource).mock.calls[0]![1]
    act(() => {
      options.onmessage?.({
        data: JSON.stringify({
          type: 'session.updated',
          payload: {
            session_id: 'oldest',
            latest_run_id: 'run-oldest',
            status: 'failed',
            error_message: 'boom',
            termination_reason: 'error',
            updated_at: '2026-07-12T11:00:00Z',
          },
        }),
        event: '',
        id: 'event-1',
      })
    })

    const cached = queryClient.getQueryData<InfiniteData<SessionListResponse>>(
      queryKeys.sessions,
    )
    expect(
      cached?.pages.flatMap((page) => page.sessions).map(({ id }) => id),
    ).toEqual(['oldest', 'newest', 'middle'])
    expect(cached?.pages[0]?.sessions[0]).toMatchObject({
      id: 'oldest',
      status: 'idle',
      updated_at: '2026-07-12T11:00:00Z',
      latest_run: {
        status: 'failed',
        error_message: 'boom',
        termination_reason: 'error',
      },
    })
  })

  it('catches non-abort stream failures and exposes an error state', async () => {
    vi.mocked(fetchEventSource).mockRejectedValueOnce(new Error('network down'))

    const { result } = renderHook(() => useNotificationStream(), {
      wrapper: createWrapper(),
    })

    await waitFor(() => expect(result.current).toBe('error'))
  })
})
