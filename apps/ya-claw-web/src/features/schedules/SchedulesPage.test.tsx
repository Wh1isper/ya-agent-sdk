import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import type { ReactNode } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import * as hooks from '../../api/hooks'
import type { ScheduleSummary } from '../../types'
import { SchedulesPage } from './SchedulesPage'

vi.mock('@tanstack/react-router', () => ({
  Link: ({ children, ...props }: { children: ReactNode }) => (
    <a {...props}>{children}</a>
  ),
  useBlocker: () => ({
    status: 'idle',
    proceed: vi.fn(),
    reset: vi.fn(),
  }),
  useRouterState: ({ select }: { select: (state: unknown) => unknown }) =>
    select({ location: { pathname: window.location.pathname } }),
}))

vi.mock('../../api/hooks', () => ({
  useCreateScheduleMutation: vi.fn(),
  useDeleteScheduleMutation: vi.fn(),
  useScheduleFiresQuery: vi.fn(),
  useScheduleQuery: vi.fn(),
  useSchedulesQuery: vi.fn(),
  useTriggerScheduleMutation: vi.fn(),
  useUpdateScheduleMutation: vi.fn(),
}))

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

describe('SchedulesPage', () => {
  const createSchedule = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
    window.history.replaceState(null, '', '/automation/schedules')
    createSchedule.mockReset()
    vi.mocked(hooks.useSchedulesQuery).mockReturnValue({
      data: { schedules: [] },
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    } as never)
    vi.mocked(hooks.useScheduleFiresQuery).mockReturnValue({
      data: { fires: [] },
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    } as never)
    vi.mocked(hooks.useScheduleQuery).mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    } as never)
    vi.mocked(hooks.useCreateScheduleMutation).mockReturnValue({
      mutateAsync: createSchedule,
      isPending: false,
    } as never)
    vi.mocked(hooks.useUpdateScheduleMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false,
    } as never)
    vi.mocked(hooks.useDeleteScheduleMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false,
    } as never)
    vi.mocked(hooks.useTriggerScheduleMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false,
    } as never)
  })

  it('keeps an invalid direct detail URL and shows its resource error', () => {
    window.history.replaceState(
      null,
      '',
      '/automation/schedules/missing-schedule',
    )
    vi.mocked(hooks.useScheduleQuery).mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true,
      error: new Error('Schedule not found'),
      refetch: vi.fn(),
    } as never)

    render(<SchedulesPage />)

    expect(screen.getByText('Could not load this schedule')).toBeVisible()
    expect(hooks.useScheduleQuery).toHaveBeenCalledWith('missing-schedule')
    expect(window.location.pathname).toBe(
      '/automation/schedules/missing-schedule',
    )
  })

  it('shows a DST-gap error and does not save the schedule', async () => {
    const user = userEvent.setup()
    render(<SchedulesPage />)

    await user.click(screen.getByRole('button', { name: 'New' }))
    await user.type(screen.getByLabelText('Name'), 'DST gap')
    fireEvent.change(screen.getByLabelText('Trigger'), {
      target: { value: 'once' },
    })
    fireEvent.change(await screen.findByLabelText(/^Run at/), {
      target: { value: '2026-03-08T02:30' },
    })
    fireEvent.change(screen.getByLabelText(/^Run timezone/), {
      target: { value: 'America/New_York' },
    })
    await user.type(screen.getByLabelText('Prompt'), 'Run this once')
    await user.click(screen.getByRole('button', { name: 'Save' }))

    expect(await screen.findByRole('alert')).toHaveTextContent(
      /2026-03-08 02:30:00 does not exist in America\/New_York/i,
    )
    await waitFor(() => expect(createSchedule).not.toHaveBeenCalled())
  })

  it('shows a visible error when a one-time run date is missing', async () => {
    const user = userEvent.setup()
    render(<SchedulesPage />)

    await user.click(screen.getByRole('button', { name: 'New' }))
    await user.type(screen.getByLabelText('Name'), 'Missing run date')
    fireEvent.change(screen.getByLabelText('Trigger'), {
      target: { value: 'once' },
    })
    await user.type(screen.getByLabelText('Prompt'), 'Run this once')
    await user.click(screen.getByRole('button', { name: 'Save' }))

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'Choose a run date and time.',
    )
    expect(createSchedule).not.toHaveBeenCalled()
  })

  it('disables schedule fields while a save is pending', async () => {
    const user = userEvent.setup()
    vi.mocked(hooks.useCreateScheduleMutation).mockReturnValue({
      mutateAsync: createSchedule,
      isPending: true,
    } as never)
    render(<SchedulesPage />)

    await user.click(screen.getByRole('button', { name: 'New' }))

    expect(screen.getByLabelText('Name')).toBeDisabled()
    expect(screen.getByLabelText('Prompt')).toBeDisabled()
  })

  it('shows a visible error when schedule creation is rejected', async () => {
    const user = userEvent.setup()
    createSchedule.mockRejectedValue(new Error('Schedule service unavailable'))
    render(<SchedulesPage />)

    await user.click(screen.getByRole('button', { name: 'New' }))
    await user.type(screen.getByLabelText('Name'), 'Failed schedule')
    await user.type(screen.getByLabelText('Prompt'), 'Try this later')
    await user.click(screen.getByRole('button', { name: 'Save' }))

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'Schedule service unavailable',
    )
  })

  it('warns that deleting a dirty schedule discards its edits', async () => {
    const user = userEvent.setup()
    vi.mocked(hooks.useSchedulesQuery).mockReturnValue({
      data: { schedules: [scheduleFixture] },
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    } as never)
    render(<SchedulesPage />)

    await user.click(
      await screen.findByRole('button', { name: /Morning sync/ }),
    )
    await user.type(screen.getByLabelText('Description'), 'Unsaved detail')
    await user.click(screen.getByRole('button', { name: 'Delete' }))

    expect(screen.getByRole('dialog')).toHaveTextContent(
      /unsaved edits will be permanently discarded/i,
    )
    expect(
      screen.getByRole('button', { name: 'Discard edits and delete' }),
    ).toBeVisible()
  })

  it('preserves dirty edits when the same schedule refreshes', async () => {
    const user = userEvent.setup()
    vi.mocked(hooks.useSchedulesQuery).mockReturnValue({
      data: { schedules: [scheduleFixture] },
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    } as never)
    const { rerender } = render(<SchedulesPage />)

    await user.click(
      await screen.findByRole('button', { name: /Morning sync/ }),
    )
    const name = screen.getByLabelText('Name')
    await user.clear(name)
    await user.type(name, 'Unsaved local name')

    vi.mocked(hooks.useSchedulesQuery).mockReturnValue({
      data: {
        schedules: [
          {
            ...scheduleFixture,
            name: 'Server refresh name',
            updated_at: '2026-07-11T01:00:00Z',
          },
        ],
      },
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    } as never)
    rerender(<SchedulesPage />)

    expect(screen.getByLabelText('Name')).toHaveValue('Unsaved local name')
    expect(
      screen.getByText(
        'A newer server version is available. Your unsaved changes are preserved.',
      ),
    ).toBeVisible()
  })

  it('adopts a newer same-schedule server version when the form is clean', async () => {
    const user = userEvent.setup()
    vi.mocked(hooks.useSchedulesQuery).mockReturnValue({
      data: { schedules: [scheduleFixture] },
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    } as never)
    const { rerender } = render(<SchedulesPage />)

    await user.click(
      await screen.findByRole('button', { name: /Morning sync/ }),
    )
    expect(screen.getByLabelText('Name')).toHaveValue('Morning sync')

    vi.mocked(hooks.useSchedulesQuery).mockReturnValue({
      data: {
        schedules: [
          {
            ...scheduleFixture,
            name: 'Server refresh name',
            updated_at: '2026-07-11T01:00:00Z',
          },
        ],
      },
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    } as never)
    rerender(<SchedulesPage />)

    expect(screen.getByLabelText('Name')).toHaveValue('Server refresh name')
    expect(
      screen.queryByText(/A newer server version is available/),
    ).not.toBeInTheDocument()
  })

  it('does not navigate after a pending schedule create resolves off-page', async () => {
    const user = userEvent.setup()
    let resolveCreate: ((value: ScheduleSummary) => void) | undefined
    const pendingCreate = new Promise<ScheduleSummary>((resolve) => {
      resolveCreate = resolve
    })
    createSchedule.mockReturnValue(pendingCreate)
    const view = render(<SchedulesPage />)

    await user.click(screen.getByRole('button', { name: 'New' }))
    await user.type(screen.getByLabelText('Name'), scheduleFixture.name)
    await user.type(screen.getByLabelText('Prompt'), scheduleFixture.prompt)
    await user.click(screen.getByRole('button', { name: 'Save' }))
    await waitFor(() => expect(createSchedule).toHaveBeenCalledOnce())

    window.history.replaceState(null, '', '/settings')
    view.unmount()
    await act(async () => {
      resolveCreate?.(scheduleFixture)
      await pendingCreate
    })

    expect(window.location.pathname).toBe('/settings')
  })

  it('selects a newly created schedule instead of creating it twice', async () => {
    const user = userEvent.setup()
    const created = { ...scheduleFixture, id: 'schedule-created' }
    createSchedule.mockResolvedValue(created)
    render(<SchedulesPage />)

    await user.click(screen.getByRole('button', { name: 'New' }))
    await user.type(screen.getByLabelText('Name'), created.name)
    await user.type(screen.getByLabelText('Prompt'), created.prompt)
    await user.click(screen.getByRole('button', { name: 'Save' }))

    await waitFor(() => expect(createSchedule).toHaveBeenCalledOnce())
    expect(window.location.pathname).toBe(
      '/automation/schedules/schedule-created',
    )
    expect(hooks.useScheduleQuery).toHaveBeenCalledWith('schedule-created')
    expect(
      screen.queryByRole('heading', { name: 'Create schedule' }),
    ).not.toBeInTheDocument()
  })
})
