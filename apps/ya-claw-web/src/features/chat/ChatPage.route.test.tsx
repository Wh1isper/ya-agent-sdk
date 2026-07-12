import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import type { ReactNode } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import * as hooks from '../../api/hooks'
import { useLayoutStore } from '../../stores/layoutStore'
import { formatA11yViolations, getA11yViolations } from '../../test/a11y'
import { ChatPage } from './ChatPage'

vi.mock('@tanstack/react-router', () => ({
  useRouterState: ({
    select,
  }: {
    select: (state: { location: { pathname: string } }) => unknown
  }) => select({ location: { pathname: window.location.pathname } }),
  Link: ({ children, ...props }: { children: ReactNode }) => (
    <a href="#" {...props}>
      {children}
    </a>
  ),
}))

vi.mock('../../api/hooks', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../../api/hooks')>()),
  useCreateSessionMutation: vi.fn(),
  useProfilesQuery: vi.fn(),
  useRunQuery: vi.fn(),
  useSessionHistoryQuery: vi.fn(),
  useSessionQuery: vi.fn(),
  useSessionWorkspaceQuery: vi.fn(),
  useSessionsQuery: vi.fn(),
  useSubmitSessionInputMutation: vi.fn(),
}))

vi.mock('./useRunEventStream', () => ({
  useRunEventStream: () => ({
    events: [],
    status: 'idle',
    error: null,
  }),
}))

const refetch = vi.fn(async () => undefined)

function queryResult(data: unknown) {
  return {
    data,
    error: null,
    isError: false,
    isLoading: false,
    isFetching: false,
    refetch,
  }
}

function renderChatPage() {
  return render(
    <main>
      <ChatPage />
    </main>,
  )
}

function sessionDetail(id: string, runId: string | null = null) {
  const run = runId
    ? {
        id: runId,
        session_id: id,
        sequence_no: 1,
        status: 'completed',
        trigger_type: 'api',
        created_at: '2026-07-11T00:00:00Z',
      }
    : null
  return {
    session: {
      id,
      profile_name: 'default',
      session_type: 'conversation',
      metadata: { title: `Conversation ${id}` },
      created_at: '2026-07-11T00:00:00Z',
      updated_at: '2026-07-11T00:00:00Z',
      status: 'idle',
      run_count: run ? 1 : 0,
      head_run_id: runId,
      head_success_run_id: runId,
      active_run_id: null,
      latest_run: run,
      runs: run ? [run] : [],
      runs_limit: 3,
      runs_has_more: false,
      runs_next_before_sequence_no: null,
    },
    state: null,
    message: [],
  }
}

describe('ChatPage route authority', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    window.history.replaceState(null, '', '/conversations/sessions/session-a')
    useLayoutStore.setState({
      route: 'chat',
      selectedSessionId: 'session-a',
      selectedRunId: null,
      selectedChatSessionId: 'session-a',
      selectedChatRunId: null,
    })

    vi.mocked(hooks.useSessionsQuery).mockReturnValue(
      queryResult([
        sessionDetail('session-a').session,
        sessionDetail('session-b').session,
      ]) as unknown as ReturnType<typeof hooks.useSessionsQuery>,
    )
    vi.mocked(hooks.useSessionQuery).mockImplementation(
      (sessionId) =>
        queryResult(
          sessionId ? sessionDetail(sessionId) : undefined,
        ) as unknown as ReturnType<typeof hooks.useSessionQuery>,
    )
    vi.mocked(hooks.useSessionHistoryQuery).mockImplementation(
      (sessionId) =>
        ({
          ...queryResult(
            sessionId ? { pages: [sessionDetail(sessionId)] } : undefined,
          ),
          isFetchingNextPage: false,
          fetchNextPage: vi.fn(async () => undefined),
        }) as unknown as ReturnType<typeof hooks.useSessionHistoryQuery>,
    )
    vi.mocked(hooks.useSessionWorkspaceQuery).mockImplementation(
      (sessionId) =>
        queryResult(
          sessionId ? { binding: null, sandbox_state: null } : undefined,
        ) as unknown as ReturnType<typeof hooks.useSessionWorkspaceQuery>,
    )
    vi.mocked(hooks.useRunQuery).mockReturnValue(
      queryResult(undefined) as unknown as ReturnType<typeof hooks.useRunQuery>,
    )
    vi.mocked(hooks.useProfilesQuery).mockReturnValue(
      queryResult([]) as unknown as ReturnType<typeof hooks.useProfilesQuery>,
    )
    vi.mocked(hooks.useCreateSessionMutation).mockReturnValue({
      isPending: false,
      mutateAsync: vi.fn(),
    } as unknown as ReturnType<typeof hooks.useCreateSessionMutation>)
    vi.mocked(hooks.useSubmitSessionInputMutation).mockReturnValue({
      isPending: false,
      mutateAsync: vi.fn(),
    } as unknown as ReturnType<typeof hooks.useSubmitSessionInputMutation>)
  })

  it.each(['Tools', 'Memory', 'Workspace'] as const)(
    'keeps the populated %s tab heading hierarchy accessible',
    async (tabName) => {
      const user = userEvent.setup()
      renderChatPage()

      await user.click(screen.getByRole('tab', { name: tabName }))

      const violations = await getA11yViolations()
      expect(violations, formatA11yViolations(violations)).toEqual([])
    },
  )

  it('builds a default-run URL from the route session instead of stale store state', async () => {
    const detail = sessionDetail('session-b', 'run-b')
    window.history.replaceState(null, '', '/conversations/sessions/session-b')
    useLayoutStore.setState({
      route: 'chat',
      selectedSessionId: 'session-a',
      selectedRunId: null,
      selectedChatSessionId: 'session-a',
      selectedChatRunId: null,
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

    renderChatPage()

    await waitFor(() => {
      expect(window.location.pathname).toBe(
        '/conversations/sessions/session-b/runs/run-b',
      )
    })
    expect(useLayoutStore.getState()).toMatchObject({
      selectedSessionId: 'session-b',
      selectedRunId: 'run-b',
      selectedChatSessionId: 'session-b',
      selectedChatRunId: 'run-b',
    })
  })

  it('uses a new detail URL before the mirrored layout selection updates', () => {
    const view = renderChatPage()

    expect(
      screen.getByRole('heading', { level: 1, name: 'Conversation session-a' }),
    ).toBeVisible()
    expect(hooks.useSessionQuery).toHaveBeenLastCalledWith(
      'session-a',
      expect.objectContaining({ runsLimit: 1, includeHeadPayload: false }),
    )

    window.history.pushState(null, '', '/conversations/sessions/session-b')
    view.rerender(
      <main>
        <ChatPage />
      </main>,
    )

    expect(hooks.useSessionQuery).toHaveBeenLastCalledWith(
      'session-b',
      expect.objectContaining({ runsLimit: 1, includeHeadPayload: false }),
    )
    expect(hooks.useSessionWorkspaceQuery).toHaveBeenLastCalledWith('session-b')
    expect(hooks.useSessionHistoryQuery).toHaveBeenLastCalledWith('session-b', {
      runsLimit: 3,
    })
    expect(
      screen.getByRole('heading', { level: 1, name: 'Conversation session-b' }),
    ).toBeVisible()
    expect(
      screen.queryByRole('heading', {
        level: 1,
        name: 'Conversation session-a',
      }),
    ).not.toBeInTheDocument()
  })
})
