import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import {
  createMemoryHistory,
  createRootRoute,
  createRoute,
  createRouter,
  Outlet,
  RouterProvider,
} from '@tanstack/react-router'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { HttpResponse, http } from 'msw'
import { beforeEach, describe, expect, it } from 'vitest'

import { queryKeys } from '../../api/queryKeys'
import { useConnectionStore } from '../../stores/connectionStore'
import { TEST_API_TOKEN } from '../../test/fixtures'
import { apiServer } from '../../test/server'
import type { ScheduleSummary } from '../../types'
import { SchedulesPage } from './SchedulesPage'

const scheduleFixture: ScheduleSummary = {
  id: 'schedule-1',
  name: 'Morning sync',
  description: null,
  enabled: true,
  status: 'active',
  prompt: 'Summarize the workspace.',
  trigger: {
    kind: 'cron',
    cron: '0 9 * * *',
    timezone: 'Asia/Shanghai',
    next_fire_at: '2026-07-12T01:00:00Z',
  },
  cron: {
    expr: '0 9 * * *',
    timezone: 'Asia/Shanghai',
    next_fire_at: '2026-07-12T01:00:00Z',
  },
  mode: {
    continue_current_session: false,
    start_from_current_session: false,
    steer_when_running: false,
  },
  execution_mode: 'isolate_session',
  owner_kind: 'user',
  fire_count: 0,
  failure_count: 0,
  metadata: {},
  created_at: '2026-07-11T00:00:00Z',
  updated_at: '2026-07-11T00:00:00Z',
}

async function renderDirectSchedule(
  getSchedule: () => ScheduleSummary,
  onDetailRequest: () => void,
) {
  apiServer.use(
    http.get('*/api/v1/schedules', () => HttpResponse.json({ schedules: [] })),
    http.get('*/api/v1/schedules/:scheduleId/fires', () =>
      HttpResponse.json({ fires: [] }),
    ),
    http.get('*/api/v1/schedules/:scheduleId', ({ params }) => {
      onDetailRequest()
      if (params.scheduleId !== scheduleFixture.id) {
        return HttpResponse.json(
          { detail: 'Schedule not found' },
          { status: 404 },
        )
      }
      return HttpResponse.json(getSchedule())
    }),
  )

  const rootRoute = createRootRoute({ component: Outlet })
  const detailRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: '/automation/schedules/$scheduleId',
    component: SchedulesPage,
  })
  const router = createRouter({
    routeTree: rootRoute.addChildren([detailRoute]),
    history: createMemoryHistory({
      initialEntries: [`/automation/schedules/${scheduleFixture.id}`],
    }),
  })
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  })
  await router.load()

  const rendered = render(
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>,
  )

  return {
    ...rendered,
    queryClient,
    user: userEvent.setup(),
  }
}

describe('SchedulesPage direct-detail refresh', () => {
  beforeEach(() => {
    useConnectionStore.setState({
      baseUrl: window.location.origin,
      apiToken: TEST_API_TOKEN,
      connectionScope: 'schedule-detail-refresh-test',
      connectionIssue: null,
    })
  })

  it('adopts a newer version returned by the real detail query while clean', async () => {
    let currentSchedule = scheduleFixture
    let detailRequestCount = 0
    const { queryClient } = await renderDirectSchedule(
      () => currentSchedule,
      () => {
        detailRequestCount += 1
      },
    )

    expect(await screen.findByLabelText('Name')).toHaveValue('Morning sync')
    currentSchedule = {
      ...scheduleFixture,
      name: 'Morning sync from server',
      updated_at: '2026-07-11T01:00:00Z',
    }

    await queryClient.refetchQueries({
      queryKey: queryKeys.schedule(scheduleFixture.id),
      exact: true,
    })

    await waitFor(() => {
      expect(screen.getByLabelText('Name')).toHaveValue(
        'Morning sync from server',
      )
    })
    expect(detailRequestCount).toBeGreaterThanOrEqual(2)
    expect(
      screen.queryByText(/A newer server version is available/),
    ).not.toBeInTheDocument()
  })

  it('preserves dirty edits and reports a conflict for a newer detail version', async () => {
    let currentSchedule = scheduleFixture
    let detailRequestCount = 0
    const { queryClient, user } = await renderDirectSchedule(
      () => currentSchedule,
      () => {
        detailRequestCount += 1
      },
    )

    const nameInput = await screen.findByLabelText('Name')
    await user.clear(nameInput)
    await user.type(nameInput, 'Unsaved local name')
    currentSchedule = {
      ...scheduleFixture,
      name: 'Morning sync from server',
      updated_at: '2026-07-11T01:00:00Z',
    }

    await queryClient.refetchQueries({
      queryKey: queryKeys.schedule(scheduleFixture.id),
      exact: true,
    })

    expect(screen.getByLabelText('Name')).toHaveValue('Unsaved local name')
    expect(
      await screen.findByText(
        'A newer server version is available. Your unsaved changes are preserved.',
      ),
    ).toBeVisible()
    expect(detailRequestCount).toBeGreaterThanOrEqual(2)
  })
})
