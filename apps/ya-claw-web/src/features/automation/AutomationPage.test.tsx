import { render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import * as hooks from '../../api/hooks'
import { AutomationPage } from './AutomationPage'

vi.mock('@tanstack/react-router', () => ({
  Link: ({
    to,
    params,
    children,
    ...props
  }: {
    to: string
    params?: Record<string, string>
    children: ReactNode
  }) => {
    const href = Object.entries(params ?? {}).reduce(
      (path, [key, value]) =>
        path.replace(`$${key}`, encodeURIComponent(value)),
      to,
    )
    return (
      <a href={href} {...props}>
        {children}
      </a>
    )
  },
}))

vi.mock('../../api/hooks', () => ({
  useAgencyFiresQuery: vi.fn(),
  useAgencyStatusQuery: vi.fn(),
  useHeartbeatFiresQuery: vi.fn(),
  useHeartbeatStatusQuery: vi.fn(),
  useSchedulesQuery: vi.fn(),
  useWorkflowRunsQuery: vi.fn(),
  useWorkflowsQuery: vi.fn(),
}))

function query<T>(data: T) {
  return {
    data,
    isLoading: false,
    isError: false,
    error: null,
    refetch: vi.fn(),
  }
}

describe('AutomationPage unified history', () => {
  beforeEach(() => {
    vi.mocked(hooks.useSchedulesQuery).mockReturnValue(
      query({
        schedules: [
          {
            id: 'schedule-1',
            name: 'Daily report',
            enabled: true,
            failure_count: 0,
            trigger: {
              kind: 'cron',
              timezone: 'UTC',
              next_fire_at: '2026-07-12T09:00:00Z',
            },
            last_fire: {
              id: 'fire-1',
              status: 'submitted',
              run_status: 'completed',
              created_at: '2026-07-11T09:00:00Z',
              target_session_id: 'schedule-session',
              run_id: 'schedule-run',
            },
          },
        ],
      }) as never,
    )
    vi.mocked(hooks.useWorkflowsQuery).mockReturnValue(
      query({ workflows: [{ id: 'workflow-1' }] }) as never,
    )
    vi.mocked(hooks.useWorkflowRunsQuery).mockReturnValue(
      query({
        workflow_runs: [
          {
            id: 'workflow-run-1',
            workflow_name: 'Research pipeline',
            status: 'running',
            updated_at: '2026-07-11T10:00:00Z',
            supervisor_session_id: 'workflow-session',
            supervisor_run_id: 'workflow-supervisor-run',
          },
        ],
      }) as never,
    )
    vi.mocked(hooks.useHeartbeatStatusQuery).mockReturnValue(
      query({ enabled: true, next_fire_at: null, last_fire: null }) as never,
    )
    vi.mocked(hooks.useHeartbeatFiresQuery).mockReturnValue(
      query({
        fires: [
          {
            id: 'heartbeat-1',
            status: 'submitted',
            run_status: 'completed',
            created_at: '2026-07-11T11:00:00Z',
            session_id: 'heartbeat-session',
            run_id: 'heartbeat-run',
          },
        ],
      }) as never,
    )
    vi.mocked(hooks.useAgencyStatusQuery).mockReturnValue(
      query({ enabled: true }) as never,
    )
    vi.mocked(hooks.useAgencyFiresQuery).mockReturnValue(
      query({
        fires: [
          {
            id: 'agency-1',
            status: 'consumed',
            run_status: 'completed',
            updated_at: '2026-07-11T12:00:00Z',
            agency_session_id: 'agency-session',
            run_id: 'agency-run',
          },
        ],
      }) as never,
    )
  })

  it('keeps healthy automation sections usable when a secondary query fails', () => {
    vi.mocked(hooks.useAgencyFiresQuery).mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true,
      error: new Error('Agency history unavailable'),
      refetch: vi.fn(),
    } as never)

    render(<AutomationPage />)

    expect(
      screen.getByRole('heading', {
        level: 1,
        name: 'Work that continues without supervision',
      }),
    ).toBeVisible()
    expect(
      screen.getByText('Some automation overview data is unavailable'),
    ).toBeVisible()
    expect(screen.getByRole('link', { name: /Schedules/ })).toBeVisible()
    expect(screen.getByRole('link', { name: /Workflows/ })).toBeVisible()
  })

  it('requests the backend maximum and warns when a list reaches the cap', () => {
    vi.mocked(hooks.useWorkflowsQuery).mockReturnValue(
      query({
        workflows: Array.from({ length: 500 }, (_, index) => ({
          id: `workflow-${index}`,
        })),
      }) as never,
    )

    render(<AutomationPage />)

    expect(hooks.useSchedulesQuery).toHaveBeenCalledWith(
      expect.objectContaining({ limit: 500 }),
    )
    expect(hooks.useWorkflowsQuery).toHaveBeenCalledWith({ limit: 500 })
    expect(hooks.useWorkflowRunsQuery).toHaveBeenCalledWith({ limit: 500 })
    expect(screen.getByRole('status')).toHaveTextContent(
      /show up to 500 items per list/i,
    )
  })

  it('combines schedule, workflow, heartbeat, and proactive history with Activity links', () => {
    render(<AutomationPage />)

    function activityLink(title: string) {
      return screen
        .getAllByText(title)
        .map((element) => element.closest('a'))
        .find((link) => link?.getAttribute('href')?.startsWith('/activity'))
    }

    expect(activityLink('Daily report')).toHaveAttribute(
      'href',
      '/activity/sessions/schedule-session/runs/schedule-run',
    )
    expect(activityLink('Research pipeline')).toHaveAttribute(
      'href',
      '/activity/sessions/workflow-session/runs/workflow-supervisor-run',
    )
    expect(activityLink('Heartbeat pulse')).toHaveAttribute(
      'href',
      '/activity/sessions/heartbeat-session/runs/heartbeat-run',
    )
    expect(activityLink('Proactive follow-up')).toHaveAttribute(
      'href',
      '/activity/sessions/agency-session/runs/agency-run',
    )
  })
})
