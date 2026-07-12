import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, renderHook, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type {
  WorkspaceFileEntry,
  WorkspaceFileListResponse,
} from '../features/workspace/types'
import { useConnectionStore } from '../stores/connectionStore'
import { useWorkspaceFilesQuery } from './hooks'

function entry(name: string, path = `/workspace/${name}`): WorkspaceFileEntry {
  return {
    name,
    path,
    kind: 'file',
    size_bytes: 1,
    modified_at: null,
    hidden: false,
  }
}

function page(
  items: WorkspaceFileEntry[],
  nextCursor: string | null,
  options: { sessionId?: string; path?: string; offset?: number } = {},
): WorkspaceFileListResponse {
  return {
    session_id: options.sessionId ?? 'session-1',
    path: options.path ?? '/workspace',
    items,
    limit: 500,
    offset: options.offset ?? 0,
    has_more: nextCursor !== null,
    next_cursor: nextCursor,
    next_offset:
      nextCursor === null ? null : (options.offset ?? 0) + items.length,
    truncated: nextCursor !== null,
  }
}

function jsonResponse(body: WorkspaceFileListResponse, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function wrapperFor(queryClient: QueryClient) {
  return function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    )
  }
}

function createQueryClient() {
  return new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: Infinity } },
  })
}

describe('useWorkspaceFilesQuery cursor pagination', () => {
  beforeEach(() => {
    useConnectionStore.getState().setConnection({
      baseUrl: 'http://claw.local',
      apiToken: 'token',
    })
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('automatically loads three cursor pages and deduplicates canonical paths', async () => {
    const requestedCursors: Array<string | null> = []
    vi.spyOn(globalThis, 'fetch').mockImplementation((input) => {
      const url = new URL(String(input))
      const cursor = url.searchParams.get('cursor')
      requestedCursors.push(cursor)
      expect(url.searchParams.has('offset')).toBe(false)
      if (cursor === null) {
        return Promise.resolve(jsonResponse(page([entry('a.txt')], 'cursor-1')))
      }
      if (cursor === 'cursor-1') {
        return Promise.resolve(
          jsonResponse(
            page(
              [entry('a-renamed.txt', '/workspace/a.txt'), entry('b.txt')],
              'cursor-2',
              { offset: 1 },
            ),
          ),
        )
      }
      return Promise.resolve(
        jsonResponse(page([entry('c.txt')], null, { offset: 3 })),
      )
    })
    const queryClient = createQueryClient()

    const { result, unmount } = renderHook(
      () =>
        useWorkspaceFilesQuery('session-1', '/workspace', {
          autoLoadAll: true,
        }),
      { wrapper: wrapperFor(queryClient) },
    )

    await waitFor(() => expect(requestedCursors).toHaveLength(3))
    await waitFor(() => expect(result.current.hasNextPage).toBe(false))
    expect(requestedCursors).toEqual([null, 'cursor-1', 'cursor-2'])
    expect(result.current.data?.items.map((item) => item.path)).toEqual([
      '/workspace/a.txt',
      '/workspace/b.txt',
      '/workspace/c.txt',
    ])
    expect(result.current.data?.items[0]?.name).toBe('a-renamed.txt')

    unmount()
    queryClient.clear()
  })

  it('stops after page two fails and continues only after explicit retry', async () => {
    let cursorOneAttempts = 0
    const requestedCursors: Array<string | null> = []
    vi.spyOn(globalThis, 'fetch').mockImplementation((input) => {
      const cursor = new URL(String(input)).searchParams.get('cursor')
      requestedCursors.push(cursor)
      if (cursor === null) {
        return Promise.resolve(jsonResponse(page([entry('a.txt')], 'cursor-1')))
      }
      if (cursor === 'cursor-1' && cursorOneAttempts++ === 0) {
        return Promise.resolve(
          new Response(JSON.stringify({ detail: 'page failed' }), {
            status: 503,
            headers: { 'Content-Type': 'application/json' },
          }),
        )
      }
      if (cursor === 'cursor-1') {
        return Promise.resolve(
          jsonResponse(page([entry('b.txt')], 'cursor-2', { offset: 1 })),
        )
      }
      return Promise.resolve(
        jsonResponse(page([entry('c.txt')], null, { offset: 2 })),
      )
    })
    const queryClient = createQueryClient()

    const { result, unmount } = renderHook(
      () =>
        useWorkspaceFilesQuery('session-1', '/workspace', {
          autoLoadAll: true,
        }),
      { wrapper: wrapperFor(queryClient) },
    )

    await waitFor(() => expect(result.current.isFetchNextPageError).toBe(true))
    expect(requestedCursors).toEqual([null, 'cursor-1'])
    await new Promise((resolve) => setTimeout(resolve, 30))
    expect(requestedCursors).toEqual([null, 'cursor-1'])

    await act(async () => {
      await result.current.fetchNextPage()
    })

    await waitFor(() => expect(result.current.hasNextPage).toBe(false))
    expect(requestedCursors).toEqual([null, 'cursor-1', 'cursor-1', 'cursor-2'])
    expect(result.current.data?.items.map((item) => item.name)).toEqual([
      'a.txt',
      'b.txt',
      'c.txt',
    ])

    unmount()
    queryClient.clear()
  })

  it('aborts an old continuation when session and path switch', async () => {
    let oldContinuationStarted = false
    let oldContinuationAborted = false
    vi.spyOn(globalThis, 'fetch').mockImplementation((input, init) => {
      const url = new URL(String(input))
      const cursor = url.searchParams.get('cursor')
      if (url.pathname.includes('session-1') && cursor === null) {
        return Promise.resolve(jsonResponse(page([entry('a.txt')], 'cursor-1')))
      }
      if (url.pathname.includes('session-1') && cursor === 'cursor-1') {
        oldContinuationStarted = true
        return new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener(
            'abort',
            () => {
              oldContinuationAborted = true
              reject(new DOMException('Aborted', 'AbortError'))
            },
            { once: true },
          )
        })
      }
      return Promise.resolve(
        jsonResponse(
          page([entry('new.txt', '/workspace/other/new.txt')], null, {
            sessionId: 'session-2',
            path: '/workspace/other',
          }),
        ),
      )
    })
    const queryClient = createQueryClient()

    const { result, rerender, unmount } = renderHook(
      ({ sessionId, path }) =>
        useWorkspaceFilesQuery(sessionId, path, { autoLoadAll: true }),
      {
        initialProps: { sessionId: 'session-1', path: '/workspace' },
        wrapper: wrapperFor(queryClient),
      },
    )

    await waitFor(() => expect(oldContinuationStarted).toBe(true))
    rerender({ sessionId: 'session-2', path: '/workspace/other' })

    await waitFor(() => expect(oldContinuationAborted).toBe(true))
    await waitFor(() =>
      expect(result.current.data?.session_id).toBe('session-2'),
    )
    expect(result.current.data?.items.map((item) => item.name)).toEqual([
      'new.txt',
    ])

    unmount()
    queryClient.clear()
  })

  it('stops on an empty page that still advertises a continuation', async () => {
    const fetchMock = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValue(jsonResponse(page([], 'stuck-cursor')))
    const queryClient = createQueryClient()

    const { result, unmount } = renderHook(
      () =>
        useWorkspaceFilesQuery('session-1', '/workspace', {
          autoLoadAll: true,
        }),
      { wrapper: wrapperFor(queryClient) },
    )

    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(result.current.hasNextPage).toBe(false)
    await new Promise((resolve) => setTimeout(resolve, 30))
    expect(fetchMock).toHaveBeenCalledTimes(1)

    unmount()
    queryClient.clear()
  })

  it('stops when a server repeats the requested cursor', async () => {
    const requestedCursors: Array<string | null> = []
    vi.spyOn(globalThis, 'fetch').mockImplementation((input) => {
      const cursor = new URL(String(input)).searchParams.get('cursor')
      requestedCursors.push(cursor)
      return Promise.resolve(
        jsonResponse(
          cursor === null
            ? page([entry('a.txt')], 'cursor-1')
            : page([entry('b.txt')], 'cursor-1', { offset: 1 }),
        ),
      )
    })
    const queryClient = createQueryClient()

    const { result, unmount } = renderHook(
      () =>
        useWorkspaceFilesQuery('session-1', '/workspace', {
          autoLoadAll: true,
        }),
      { wrapper: wrapperFor(queryClient) },
    )

    await waitFor(() => expect(requestedCursors).toHaveLength(2))
    await waitFor(() => expect(result.current.hasNextPage).toBe(false))
    await new Promise((resolve) => setTimeout(resolve, 30))
    expect(requestedCursors).toEqual([null, 'cursor-1'])

    unmount()
    queryClient.clear()
  })

  it('stops when a legacy server repeats the current offset', async () => {
    const requestedOffsets: Array<string | null> = []
    vi.spyOn(globalThis, 'fetch').mockImplementation((input) => {
      const offset = new URL(String(input)).searchParams.get('offset')
      requestedOffsets.push(offset)
      const currentOffset = offset === null ? 0 : Number(offset)
      return Promise.resolve(
        jsonResponse({
          ...page([entry(`file-${currentOffset}.txt`)], null, {
            offset: currentOffset,
          }),
          has_more: true,
          next_offset: 1,
          truncated: true,
        }),
      )
    })
    const queryClient = createQueryClient()

    const { result, unmount } = renderHook(
      () =>
        useWorkspaceFilesQuery('session-1', '/workspace', {
          autoLoadAll: true,
        }),
      { wrapper: wrapperFor(queryClient) },
    )

    await waitFor(() => expect(requestedOffsets).toHaveLength(2))
    await waitFor(() => expect(result.current.hasNextPage).toBe(false))
    await new Promise((resolve) => setTimeout(resolve, 30))
    expect(requestedOffsets).toEqual([null, '1'])

    unmount()
    queryClient.clear()
  })

  it('bounds auto-loading when every page advertises a unique cursor', async () => {
    const requestedCursors: Array<string | null> = []
    vi.spyOn(globalThis, 'fetch').mockImplementation((input) => {
      const cursor = new URL(String(input)).searchParams.get('cursor')
      requestedCursors.push(cursor)
      const pageNumber = requestedCursors.length
      return Promise.resolve(
        jsonResponse(
          page([entry(`file-${pageNumber}.txt`)], `cursor-${pageNumber}`, {
            offset: pageNumber - 1,
          }),
        ),
      )
    })
    const queryClient = createQueryClient()

    const { result, unmount } = renderHook(
      () =>
        useWorkspaceFilesQuery('session-1', '/workspace', {
          autoLoadAll: true,
        }),
      { wrapper: wrapperFor(queryClient) },
    )

    await waitFor(() => expect(requestedCursors).toHaveLength(50), {
      timeout: 3_000,
    })
    await waitFor(() => expect(result.current.hasNextPage).toBe(false))
    await new Promise((resolve) => setTimeout(resolve, 30))
    expect(requestedCursors).toHaveLength(50)
    expect(result.current.data?.items).toHaveLength(50)

    unmount()
    queryClient.clear()
  })
})
