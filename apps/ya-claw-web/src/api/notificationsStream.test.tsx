import { fetchEventSource } from '@microsoft/fetch-event-source'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, renderHook, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useConnectionStore } from '../stores/connectionStore'
import { useNotificationStream } from './notificationsStream'

vi.mock('@microsoft/fetch-event-source', () => ({
  fetchEventSource: vi.fn(),
}))

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    )
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

  it('catches non-abort stream failures and exposes an error state', async () => {
    vi.mocked(fetchEventSource).mockRejectedValueOnce(new Error('network down'))

    const { result } = renderHook(() => useNotificationStream(), {
      wrapper: createWrapper(),
    })

    await waitFor(() => expect(result.current).toBe('error'))
  })
})
