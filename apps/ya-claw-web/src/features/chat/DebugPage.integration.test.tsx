import { fetchEventSource } from '@microsoft/fetch-event-source'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import * as hooks from '../../api/hooks'
import { useConnectionStore } from '../../stores/connectionStore'
import { useLayoutStore } from '../../stores/layoutStore'
import { DebugPage } from './DebugPage'

vi.mock('@microsoft/fetch-event-source', () => ({
  fetchEventSource: vi.fn(),
}))

vi.mock('@tanstack/react-router', () => ({
  useRouterState: ({
    select,
  }: {
    select: (state: { location: { pathname: string } }) => unknown
  }) => select({ location: { pathname: window.location.pathname } }),
  Link: ({ children }: { children: ReactNode }) => <a href="#">{children}</a>,
}))

vi.mock('../../api/hooks', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../../api/hooks')>()),
  useRunQuery: vi.fn(),
  useSessionHistoryQuery: vi.fn(),
  useSessionQuery: vi.fn(),
  useSessionWorkspaceQuery: vi.fn(),
  useSessionsQuery: vi.fn(),
  useWorkspaceRuntimeQuery: vi.fn(),
}))

const refetch = vi.fn(async () => undefined)

function queryResult(data: unknown) {
  return {
    data,
    error: null,
    isError: false,
    isLoading: false,
    refetch,
  }
}

function failedQuery(error: Error, data?: unknown) {
  return {
    data,
    error,
    isError: true,
    isLoading: false,
    refetch,
  }
}

function renderDebugPage() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={queryClient}>
      <DebugPage />
    </QueryClientProvider>,
  )
}

describe('DebugPage cross-session run isolation', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    window.history.replaceState(
      null,
      '',
      '/activity/sessions/session-a/runs/run-b',
    )
    useConnectionStore.setState({
      baseUrl: 'https://claw.example',
      apiToken: 'test-token',
      connectionScope: 'activity-isolation-test',
      connectionIssue: null,
    })
    useLayoutStore.setState({
      route: 'debug',
      selectedSessionId: 'session-a',
      selectedRunId: 'run-b',
      selectedDebugSessionId: 'session-a',
      selectedDebugRunId: 'run-b',
    })

    vi.mocked(hooks.useSessionsQuery).mockReturnValue(
      queryResult([]) as unknown as ReturnType<typeof hooks.useSessionsQuery>,
    )
    vi.mocked(hooks.useWorkspaceRuntimeQuery).mockReturnValue(
      queryResult(null) as unknown as ReturnType<
        typeof hooks.useWorkspaceRuntimeQuery
      >,
    )
    vi.mocked(hooks.useSessionWorkspaceQuery).mockReturnValue(
      queryResult(null) as unknown as ReturnType<
        typeof hooks.useSessionWorkspaceQuery
      >,
    )
    vi.mocked(hooks.useSessionQuery).mockReturnValue(
      queryResult({
        session: {
          id: 'session-a',
          session_type: 'conversation',
          metadata: {},
          created_at: '2026-07-11T00:00:00Z',
          updated_at: '2026-07-11T00:00:00Z',
          status: 'idle',
          run_count: 0,
          runs: [],
        },
        state: null,
        message: [],
      }) as unknown as ReturnType<typeof hooks.useSessionQuery>,
    )
    vi.mocked(hooks.useSessionHistoryQuery).mockReturnValue({
      ...queryResult({ pages: [] }),
      isFetchingNextPage: false,
      fetchNextPage: vi.fn(async () => undefined),
    } as unknown as ReturnType<typeof hooks.useSessionHistoryQuery>)
    vi.mocked(hooks.useRunQuery).mockReturnValue(
      queryResult({
        run: {
          id: 'run-b',
          session_id: 'session-b',
          sequence_no: 7,
          status: 'running',
          trigger_type: 'api',
          created_at: '2026-07-11T00:00:00Z',
          message: [],
        },
        message: [],
      }) as unknown as ReturnType<typeof hooks.useRunQuery>,
    )
  })

  it('blocks another session run before starting SSE or rendering run details', () => {
    renderDebugPage()

    expect(hooks.useSessionQuery).toHaveBeenCalledWith('session-a')
    expect(hooks.useRunQuery).toHaveBeenCalledWith('run-b')
    expect(screen.getByRole('alert')).toHaveTextContent(
      'Run does not belong to this session',
    )
    expect(screen.getByRole('alert')).toHaveTextContent(
      'Live events and run controls are disabled',
    )
    expect(fetchEventSource).not.toHaveBeenCalled()
    expect(screen.queryByText('Interrupt')).not.toBeInTheDocument()
    expect(screen.queryByText('run-b')).not.toBeInTheDocument()
    expect(screen.getByTestId('activity-desktop-layout')).toBeInTheDocument()
    expect(screen.getByTestId('activity-mobile-layout')).toBeInTheDocument()
  })

  it('keeps both layouts mounted when workspace status fails', () => {
    window.history.replaceState(null, '', '/activity/sessions/session-a')
    useLayoutStore.setState({
      selectedRunId: null,
      selectedDebugRunId: null,
    })
    vi.mocked(hooks.useWorkspaceRuntimeQuery).mockReturnValue(
      failedQuery(
        new Error('Workspace runtime unavailable'),
      ) as unknown as ReturnType<typeof hooks.useWorkspaceRuntimeQuery>,
    )
    vi.mocked(hooks.useRunQuery).mockReturnValue(
      queryResult(undefined) as unknown as ReturnType<typeof hooks.useRunQuery>,
    )

    renderDebugPage()

    expect(screen.getByTestId('activity-desktop-layout')).toBeInTheDocument()
    expect(screen.getByTestId('activity-mobile-layout')).toBeInTheDocument()
    expect(screen.getAllByText('Workspace status unavailable')).toHaveLength(2)
    expect(screen.getAllByRole('button', { name: 'Try again' })).toHaveLength(2)
  })

  it('builds a default-run URL from the route session instead of stale store state', async () => {
    const run = {
      id: 'run-b',
      session_id: 'session-b',
      sequence_no: 1,
      status: 'completed',
      trigger_type: 'api',
      created_at: '2026-07-11T00:00:00Z',
      message: [],
    }
    const detail = {
      session: {
        id: 'session-b',
        session_type: 'conversation',
        metadata: { title: 'Session B' },
        created_at: '2026-07-11T00:00:00Z',
        updated_at: '2026-07-11T00:00:00Z',
        status: 'idle',
        run_count: 1,
        head_run_id: 'run-b',
        head_success_run_id: 'run-b',
        active_run_id: null,
        latest_run: run,
        runs: [run],
      },
      state: null,
      message: [],
    }
    window.history.replaceState(null, '', '/activity/sessions/session-b')
    useLayoutStore.setState({
      route: 'debug',
      selectedSessionId: 'session-a',
      selectedRunId: null,
      selectedDebugSessionId: 'session-a',
      selectedDebugRunId: null,
    })
    vi.mocked(hooks.useSessionQuery).mockReturnValue(
      queryResult(detail) as unknown as ReturnType<
        typeof hooks.useSessionQuery
      >,
    )
    vi.mocked(hooks.useSessionHistoryQuery).mockReturnValue({
      ...queryResult({ pages: [detail] }),
      isFetchingNextPage: false,
      fetchNextPage: vi.fn(async () => undefined),
    } as unknown as ReturnType<typeof hooks.useSessionHistoryQuery>)
    vi.mocked(hooks.useRunQuery).mockReturnValue(
      queryResult({
        session: detail.session,
        run,
        state: null,
        message: [],
      }) as unknown as ReturnType<typeof hooks.useRunQuery>,
    )

    renderDebugPage()

    await waitFor(() => {
      expect(window.location.pathname).toBe(
        '/activity/sessions/session-b/runs/run-b',
      )
    })
    expect(useLayoutStore.getState()).toMatchObject({
      selectedSessionId: 'session-b',
      selectedRunId: 'run-b',
      selectedDebugSessionId: 'session-b',
      selectedDebugRunId: 'run-b',
    })
  })

  it('uses the current route immediately when an existing view changes sessions', () => {
    window.history.replaceState(null, '', '/activity/sessions/session-a')
    vi.mocked(hooks.useSessionQuery).mockImplementation(
      (sessionId) =>
        queryResult(
          sessionId
            ? {
                session: {
                  id: sessionId,
                  session_type: 'conversation',
                  metadata: { title: `Session ${sessionId}` },
                  created_at: '2026-07-11T00:00:00Z',
                  updated_at: '2026-07-11T00:00:00Z',
                  status: 'idle',
                  run_count: 0,
                  runs: [],
                },
                state: null,
                message: [],
              }
            : undefined,
        ) as unknown as ReturnType<typeof hooks.useSessionQuery>,
    )
    vi.mocked(hooks.useSessionHistoryQuery).mockImplementation(
      (sessionId) =>
        ({
          ...queryResult(sessionId ? { pages: [] } : undefined),
          isFetchingNextPage: false,
          fetchNextPage: vi.fn(async () => undefined),
        }) as unknown as ReturnType<typeof hooks.useSessionHistoryQuery>,
    )
    vi.mocked(hooks.useRunQuery).mockReturnValue(
      queryResult(undefined) as unknown as ReturnType<typeof hooks.useRunQuery>,
    )

    const view = renderDebugPage()
    expect(hooks.useSessionQuery).toHaveBeenLastCalledWith('session-a')

    window.history.pushState(null, '', '/activity/sessions/session-b')
    view.rerender(
      <QueryClientProvider client={new QueryClient()}>
        <DebugPage />
      </QueryClientProvider>,
    )

    expect(hooks.useSessionQuery).toHaveBeenLastCalledWith('session-b')
    expect(hooks.useSessionWorkspaceQuery).toHaveBeenLastCalledWith('session-b')
    expect(hooks.useSessionHistoryQuery).toHaveBeenLastCalledWith('session-b', {
      runsLimit: 3,
    })
  })

  it('keeps navigation and timelines mounted when history or run detail fails', () => {
    vi.mocked(hooks.useSessionHistoryQuery).mockReturnValue({
      ...failedQuery(new Error('History unavailable')),
      isFetchingNextPage: false,
      fetchNextPage: vi.fn(async () => undefined),
    } as unknown as ReturnType<typeof hooks.useSessionHistoryQuery>)
    vi.mocked(hooks.useRunQuery).mockReturnValue(
      failedQuery(new Error('Run detail unavailable')) as unknown as ReturnType<
        typeof hooks.useRunQuery
      >,
    )

    renderDebugPage()

    expect(
      screen.getByRole('heading', {
        name: 'Some activity details could not be loaded',
      }),
    ).toBeVisible()
    expect(screen.getByTestId('activity-desktop-layout')).toBeInTheDocument()
    expect(screen.getByTestId('activity-mobile-layout')).toBeInTheDocument()
  })
})
