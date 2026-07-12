import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type {
  ScheduleCreateRequest,
  ScheduleSummary,
  WorkflowDefinitionDetail,
  WorkflowDefinitionListResponse,
  WorkflowRunListResponse,
} from '../../types'
import * as hooks from '../../api/hooks'
import { WorkflowsPage } from './WorkflowsPage'

const { useBlocker, idleBlocker } = vi.hoisted(() => {
  const idleBlocker = {
    status: 'idle' as const,
    proceed: undefined,
    reset: undefined,
  }
  return { useBlocker: vi.fn(() => idleBlocker), idleBlocker }
})

vi.mock('@tanstack/react-router', () => ({ useBlocker }))

vi.mock('../../api/hooks', () => ({
  useArchiveWorkflowMutation: vi.fn(),
  useCreateScheduleMutation: vi.fn(),
  useCreateWorkflowMutation: vi.fn(),
  useDeleteScheduleMutation: vi.fn(),
  useScheduleFiresQuery: vi.fn(),
  useSchedulesQuery: vi.fn(),
  useTriggerScheduleMutation: vi.fn(),
  useTriggerWorkflowMutation: vi.fn(),
  useUpdateScheduleMutation: vi.fn(),
  useUpdateWorkflowMutation: vi.fn(),
  useWorkflowEventsQuery: vi.fn(),
  useWorkflowQuery: vi.fn(),
  useWorkflowRunMutations: vi.fn(),
  useWorkflowRunQuery: vi.fn(),
  useWorkflowRunsQuery: vi.fn(),
  useWorkflowsQuery: vi.fn(),
}))

const workflow: WorkflowDefinitionDetail = {
  id: 'workflow-1',
  name: 'Daily research workflow',
  description: 'Research and summarize the market',
  status: 'active',
  definition_version: 1,
  schema_version: 'ya-claw.workflow.v1',
  owner_kind: 'user',
  owner_session_id: null,
  owner_run_id: null,
  scope: 'global',
  tags: ['daily', 'research'],
  when_to_use: 'Use for daily research',
  argument_hint: 'topic',
  latest_run: null,
  metadata: {},
  created_at: '2026-01-01T00:00:00Z',
  updated_at: '2026-01-01T00:00:00Z',
  archived_at: null,
  input_schema: { type: 'object' },
  definition: {
    schema: 'ya-claw.workflow.v1',
    nodes: {
      draft: { profile: 'Self', prompt: 'Draft research' },
    },
    result: { from_node: 'draft' },
  },
}

const workflowSchedule: ScheduleSummary = {
  id: 'schedule-1',
  name: 'Weekday research',
  description: 'Run the workflow on weekdays',
  enabled: true,
  status: 'active',
  prompt: '',
  trigger: {
    kind: 'cron',
    cron: '0 9 * * 1-5',
    timezone: 'UTC',
    next_fire_at: '2026-01-02T09:00:00Z',
  },
  cron: {
    expr: '0 9 * * 1-5',
    timezone: 'UTC',
    next_fire_at: '2026-01-02T09:00:00Z',
  },
  mode: {
    continue_current_session: false,
    start_from_current_session: false,
    steer_when_running: false,
  },
  execution_mode: 'workflow',
  workflow_id: workflow.id,
  workflow_inputs_template: { topic: 'market' },
  last_workflow_run_id: 'workflow-run-1',
  owner_kind: 'user',
  owner_session_id: null,
  owner_run_id: null,
  profile_name: null,
  target_session_id: null,
  source_session_id: null,
  last_fire: null,
  fire_count: 1,
  failure_count: 0,
  metadata: {},
  created_at: '2026-01-01T00:00:00Z',
  updated_at: '2026-01-01T00:00:00Z',
}

function setupHookMocks(
  options: {
    schedules?: ScheduleSummary[]
    createSchedule?: (
      payload: ScheduleCreateRequest,
    ) => Promise<ScheduleSummary>
    archiveWorkflow?: (workflowId: string) => Promise<WorkflowDefinitionDetail>
    createWorkflow?: () => Promise<WorkflowDefinitionDetail>
    updateWorkflow?: () => Promise<WorkflowDefinitionDetail>
    deleteSchedule?: (scheduleId: string) => Promise<ScheduleSummary>
    archivePending?: boolean
    deletePending?: boolean
    updatePending?: boolean
  } = {},
) {
  vi.mocked(hooks.useWorkflowsQuery).mockReturnValue({
    data: { workflows: [workflow] } satisfies WorkflowDefinitionListResponse,
    isLoading: false,
  } as unknown as ReturnType<typeof hooks.useWorkflowsQuery>)
  vi.mocked(hooks.useWorkflowQuery).mockImplementation(
    (workflowId) =>
      ({
        data: workflowId === workflow.id ? workflow : undefined,
      }) as unknown as ReturnType<typeof hooks.useWorkflowQuery>,
  )
  vi.mocked(hooks.useSchedulesQuery).mockImplementation(
    (filters = {}) =>
      ({
        data: {
          schedules:
            filters.workflowId === workflow.id
              ? (options.schedules ?? [workflowSchedule])
              : [],
        },
        isLoading: false,
      }) as unknown as ReturnType<typeof hooks.useSchedulesQuery>,
  )
  vi.mocked(hooks.useWorkflowRunsQuery).mockReturnValue({
    data: { workflow_runs: [] } satisfies WorkflowRunListResponse,
  } as unknown as ReturnType<typeof hooks.useWorkflowRunsQuery>)
  vi.mocked(hooks.useWorkflowRunQuery).mockReturnValue({
    data: undefined,
  } as unknown as ReturnType<typeof hooks.useWorkflowRunQuery>)
  vi.mocked(hooks.useWorkflowEventsQuery).mockReturnValue({
    data: { workflow_run_id: '', events: [] },
  } as unknown as ReturnType<typeof hooks.useWorkflowEventsQuery>)
  vi.mocked(hooks.useScheduleFiresQuery).mockReturnValue({
    data: { fires: [] },
  } as unknown as ReturnType<typeof hooks.useScheduleFiresQuery>)
  vi.mocked(hooks.useCreateScheduleMutation).mockReturnValue({
    mutateAsync: options.createSchedule ?? vi.fn(async () => workflowSchedule),
    isPending: false,
  } as unknown as ReturnType<typeof hooks.useCreateScheduleMutation>)
  vi.mocked(hooks.useUpdateScheduleMutation).mockReturnValue({
    mutateAsync: vi.fn(async () => workflowSchedule),
    isPending: false,
  } as unknown as ReturnType<typeof hooks.useUpdateScheduleMutation>)
  vi.mocked(hooks.useDeleteScheduleMutation).mockReturnValue({
    mutateAsync: options.deleteSchedule ?? vi.fn(async () => workflowSchedule),
    isPending: options.deletePending ?? false,
  } as unknown as ReturnType<typeof hooks.useDeleteScheduleMutation>)
  vi.mocked(hooks.useTriggerScheduleMutation).mockReturnValue({
    mutateAsync: vi.fn(),
    isPending: false,
  } as unknown as ReturnType<typeof hooks.useTriggerScheduleMutation>)
  vi.mocked(hooks.useCreateWorkflowMutation).mockReturnValue({
    mutateAsync: options.createWorkflow ?? vi.fn(async () => workflow),
    isPending: false,
  } as unknown as ReturnType<typeof hooks.useCreateWorkflowMutation>)
  vi.mocked(hooks.useUpdateWorkflowMutation).mockReturnValue({
    mutateAsync: options.updateWorkflow ?? vi.fn(async () => workflow),
    isPending: options.updatePending ?? false,
  } as unknown as ReturnType<typeof hooks.useUpdateWorkflowMutation>)
  vi.mocked(hooks.useArchiveWorkflowMutation).mockReturnValue({
    mutateAsync: options.archiveWorkflow ?? vi.fn(async () => workflow),
    isPending: options.archivePending ?? false,
  } as unknown as ReturnType<typeof hooks.useArchiveWorkflowMutation>)
  vi.mocked(hooks.useTriggerWorkflowMutation).mockReturnValue({
    mutateAsync: vi.fn(async () => ({ id: 'workflow-run-1' })),
    isPending: false,
  } as unknown as ReturnType<typeof hooks.useTriggerWorkflowMutation>)
  vi.mocked(hooks.useWorkflowRunMutations).mockReturnValue({
    cancel: { mutate: vi.fn() },
    steerNode: { mutate: vi.fn() },
  } as unknown as ReturnType<typeof hooks.useWorkflowRunMutations>)
}

describe('WorkflowsPage workflow schedules', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    useBlocker.mockReturnValue(idleBlocker)
    setupHookMocks()
  })

  it('loads workflow-backed schedules from the selected workflow detail', async () => {
    render(<WorkflowsPage />)

    await waitFor(() => {
      expect(
        screen.getAllByText('Daily research workflow').length,
      ).toBeGreaterThan(0)
    })
    expect(await screen.findByText('Workflow recurrence')).toBeInTheDocument()
    expect(screen.getByText('Weekday research')).toBeInTheDocument()
    expect(screen.getByText('Last run workflow')).toBeInTheDocument()

    await waitFor(() => {
      expect(hooks.useSchedulesQuery).toHaveBeenCalledWith({
        workflowId: workflow.id,
        executionMode: 'workflow',
        includeWorkflow: true,
        includeDeleted: true,
        limit: 500,
      })
    })
  })

  it('preserves dirty edits when a newer workflow arrives in the background', async () => {
    const user = userEvent.setup()
    const { rerender } = render(<WorkflowsPage />)
    const name = (await screen.findAllByLabelText('Name'))[0]

    fireEvent.change(name, { target: { value: 'Local workflow name' } })

    const remoteWorkflow = {
      ...workflow,
      name: 'Remote workflow name',
      updated_at: '2026-02-01T00:00:00Z',
    }
    vi.mocked(hooks.useWorkflowQuery).mockImplementation(
      (workflowId) =>
        ({
          data: workflowId === workflow.id ? remoteWorkflow : undefined,
        }) as unknown as ReturnType<typeof hooks.useWorkflowQuery>,
    )
    rerender(<WorkflowsPage />)

    expect(screen.getAllByLabelText('Name')[0]).toHaveValue(
      'Local workflow name',
    )
    expect(
      screen.getByText(/A newer server version is available/),
    ).toBeVisible()

    const staleWorkflow = {
      ...workflow,
      name: 'Stale workflow name',
      updated_at: '2026-01-15T00:00:00Z',
    }
    vi.mocked(hooks.useWorkflowQuery).mockImplementation(
      (workflowId) =>
        ({
          data: workflowId === workflow.id ? staleWorkflow : undefined,
        }) as unknown as ReturnType<typeof hooks.useWorkflowQuery>,
    )
    rerender(<WorkflowsPage />)
    expect(screen.getAllByLabelText('Name')[0]).toHaveValue(
      'Local workflow name',
    )

    await user.click(
      screen.getByRole('button', { name: 'Load server version' }),
    )
    expect(
      screen.getByRole('dialog', {
        name: /Discard unsaved workflow changes and load the server version/,
      }),
    ).toBeVisible()
    expect(screen.getAllByLabelText('Name')[0]).toHaveValue(
      'Local workflow name',
    )

    await user.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(screen.getAllByLabelText('Name')[0]).toHaveValue(
      'Local workflow name',
    )

    await user.click(
      screen.getByRole('button', { name: 'Load server version' }),
    )
    await user.click(
      screen.getByRole('button', { name: 'Discard changes and load' }),
    )
    expect(screen.getAllByLabelText('Name')[0]).toHaveValue(
      'Remote workflow name',
    )
  }, 10_000)

  it('disables workflow fields while a save is pending', async () => {
    setupHookMocks({ updatePending: true })
    render(<WorkflowsPage />)

    expect((await screen.findAllByLabelText('Name'))[0]).toBeDisabled()
  })

  it('does not navigate after a pending workflow create resolves off-page', async () => {
    const user = userEvent.setup()
    let resolveCreate: ((value: WorkflowDefinitionDetail) => void) | undefined
    const pendingCreate = new Promise<WorkflowDefinitionDetail>((resolve) => {
      resolveCreate = resolve
    })
    const createWorkflow = vi.fn(() => pendingCreate)
    setupHookMocks({ createWorkflow })
    window.history.replaceState(null, '', '/automation/workflows')
    const view = render(<WorkflowsPage />)

    await user.click(screen.getByRole('button', { name: 'New' }))
    await user.click(screen.getByRole('button', { name: 'Save' }))
    await waitFor(() => expect(createWorkflow).toHaveBeenCalledOnce())

    window.history.replaceState(null, '', '/settings')
    view.unmount()
    await act(async () => {
      resolveCreate?.(workflow)
      await pendingCreate
    })

    expect(window.location.pathname).toBe('/settings')
  })

  it('does not report its own successful save as a remote conflict', async () => {
    const savedWorkflow = {
      ...workflow,
      name: 'Saved workflow name',
      updated_at: '2026-03-01T00:00:00Z',
    }
    setupHookMocks({
      updateWorkflow: vi.fn(async () => savedWorkflow),
    })
    const { rerender } = render(<WorkflowsPage />)
    const name = (await screen.findAllByLabelText('Name'))[0]

    fireEvent.change(name, { target: { value: savedWorkflow.name } })
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))
    await waitFor(() => expect(name).toHaveValue(savedWorkflow.name))
    fireEvent.change(name, {
      target: { value: 'Saved workflow name with local edit' },
    })

    vi.mocked(hooks.useWorkflowQuery).mockImplementation(
      (workflowId) =>
        ({
          data: workflowId === workflow.id ? savedWorkflow : undefined,
        }) as unknown as ReturnType<typeof hooks.useWorkflowQuery>,
    )
    rerender(<WorkflowsPage />)

    expect(name).toHaveValue('Saved workflow name with local edit')
    expect(
      screen.queryByText(/A newer server version is available/),
    ).not.toBeInTheDocument()
  }, 10_000)

  it('resolves blocked workflow navigation through the discard dialog', async () => {
    const user = userEvent.setup()
    const proceed = vi.fn()
    const reset = vi.fn()
    useBlocker.mockReturnValue({
      status: 'blocked',
      proceed,
      reset,
    } as never)

    render(<WorkflowsPage />)

    const dialog = screen.getByRole('dialog', {
      name: 'Discard unsaved workflow changes?',
    })
    expect(dialog).toBeVisible()
    await user.click(within(dialog).getByRole('button', { name: 'Stay here' }))
    expect(reset).toHaveBeenCalled()
    expect(proceed).not.toHaveBeenCalled()

    await user.click(
      within(dialog).getByRole('button', { name: 'Discard and leave' }),
    )
    expect(proceed).toHaveBeenCalled()
  })

  it('enables SPA and beforeunload protection after a workflow edit', async () => {
    const user = userEvent.setup()
    render(<WorkflowsPage />)
    const name = (await screen.findAllByLabelText('Name'))[0]

    await user.type(name, ' changed')

    await waitFor(() =>
      expect(useBlocker).toHaveBeenCalledWith(
        expect.objectContaining({
          disabled: false,
          enableBeforeUnload: true,
          withResolver: true,
        }),
      ),
    )
  })

  it('protects unsaved trigger inputs with the route blocker', async () => {
    render(<WorkflowsPage />)
    await screen.findByText('Workflow recurrence')

    fireEvent.change(screen.getByLabelText('Supervisor session'), {
      target: { value: 'session-with-unsaved-trigger' },
    })

    await waitFor(() =>
      expect(useBlocker).toHaveBeenCalledWith(
        expect.objectContaining({
          disabled: false,
          enableBeforeUnload: true,
          withResolver: true,
        }),
      ),
    )
  })

  it('preserves a dirty trigger when a newer workflow arrives', async () => {
    const user = userEvent.setup()
    const { rerender } = render(<WorkflowsPage />)
    await screen.findByText('Workflow recurrence')

    const supervisor = screen.getByLabelText('Supervisor session')
    fireEvent.change(supervisor, {
      target: { value: 'session-with-unsaved-trigger' },
    })

    const remoteWorkflow = {
      ...workflow,
      name: 'Remote workflow name',
      updated_at: '2026-02-01T00:00:00Z',
    }
    vi.mocked(hooks.useWorkflowQuery).mockImplementation(
      (workflowId) =>
        ({
          data: workflowId === workflow.id ? remoteWorkflow : undefined,
        }) as unknown as ReturnType<typeof hooks.useWorkflowQuery>,
    )
    rerender(<WorkflowsPage />)

    expect(supervisor).toHaveValue('session-with-unsaved-trigger')
    expect(
      screen.getByText(/A newer server version is available/),
    ).toBeVisible()

    await user.click(
      screen.getByRole('button', { name: 'Load server version' }),
    )
    await user.click(
      screen.getByRole('button', { name: 'Discard changes and load' }),
    )
    expect(supervisor).toHaveValue('')
  })

  it('adopts a newer embedded schedule when the editor is clean', async () => {
    const user = userEvent.setup()
    const { rerender } = render(<WorkflowsPage />)
    await screen.findByText('Workflow recurrence')
    await user.click(screen.getByText('Weekday research'))

    const scheduleEditor = screen
      .getByText('Edit workflow schedule')
      .closest('.rounded-2xl') as HTMLElement
    const name = within(scheduleEditor).getByLabelText('Name')
    expect(name).toHaveValue('Weekday research')

    const remoteSchedule = {
      ...workflowSchedule,
      name: 'Remote weekday research',
      updated_at: '2026-02-01T00:00:00Z',
    }
    vi.mocked(hooks.useSchedulesQuery).mockImplementation(
      (filters = {}) =>
        ({
          data: {
            schedules:
              filters.workflowId === workflow.id ? [remoteSchedule] : [],
          },
          isLoading: false,
        }) as unknown as ReturnType<typeof hooks.useSchedulesQuery>,
    )
    rerender(<WorkflowsPage />)

    await waitFor(() => expect(name).toHaveValue('Remote weekday research'))
  })

  it('preserves dirty embedded schedule edits when a newer version arrives', async () => {
    const user = userEvent.setup()
    const { rerender } = render(<WorkflowsPage />)
    await screen.findByText('Workflow recurrence')
    await user.click(screen.getByText('Weekday research'))

    const scheduleEditor = screen
      .getByText('Edit workflow schedule')
      .closest('.rounded-2xl') as HTMLElement
    const name = within(scheduleEditor).getByLabelText('Name')
    await user.clear(name)
    await user.type(name, 'Local schedule name')

    const remoteSchedule = {
      ...workflowSchedule,
      name: 'Remote schedule name',
      updated_at: '2026-02-01T00:00:00Z',
    }
    vi.mocked(hooks.useSchedulesQuery).mockImplementation(
      (filters = {}) =>
        ({
          data: {
            schedules:
              filters.workflowId === workflow.id ? [remoteSchedule] : [],
          },
          isLoading: false,
        }) as unknown as ReturnType<typeof hooks.useSchedulesQuery>,
    )
    rerender(<WorkflowsPage />)

    expect(name).toHaveValue('Local schedule name')
    expect(
      screen.getByText(/A newer server version of this schedule is available/),
    ).toBeVisible()

    await user.click(
      screen.getByRole('button', { name: 'Load schedule server version' }),
    )
    await user.click(
      screen.getByRole('button', { name: 'Discard changes and load' }),
    )
    expect(name).toHaveValue('Remote schedule name')
  })

  it('confirms before replacing dirty embedded schedule edits', async () => {
    const user = userEvent.setup()
    render(<WorkflowsPage />)
    await screen.findByText('Workflow recurrence')
    await user.click(screen.getByText('Weekday research'))

    const scheduleEditor = screen
      .getByText('Edit workflow schedule')
      .closest('.rounded-2xl') as HTMLElement
    const scheduleForm = within(scheduleEditor)
    const name = scheduleForm.getByLabelText('Name')
    await user.clear(name)
    await user.type(name, 'Unsaved schedule name')

    await user.click(screen.getByRole('button', { name: /New schedule/i }))

    const dialog = screen.getByRole('dialog', {
      name: 'Discard unsaved schedule changes?',
    })
    expect(dialog).toBeVisible()
    expect(name).toHaveValue('Unsaved schedule name')
    await waitFor(() =>
      expect(useBlocker).toHaveBeenCalledWith(
        expect.objectContaining({
          disabled: false,
          enableBeforeUnload: true,
        }),
      ),
    )

    await user.click(
      within(dialog).getByRole('button', { name: 'Discard changes' }),
    )
    expect(screen.getByText('Create workflow schedule')).toBeVisible()
    expect(
      within(
        screen
          .getByText('Create workflow schedule')
          .closest('.rounded-2xl') as HTMLElement,
      ).getByLabelText('Name'),
    ).toHaveValue('Workflow schedule')
  })

  it('prevents saving a structurally invalid workflow definition', async () => {
    const updateWorkflow = vi.fn(async () => workflow)
    vi.mocked(hooks.useUpdateWorkflowMutation).mockReturnValue({
      mutateAsync: updateWorkflow,
      isPending: false,
    } as unknown as ReturnType<typeof hooks.useUpdateWorkflowMutation>)

    render(<WorkflowsPage />)
    await screen.findByText('1 workflow step ready')

    fireEvent.change(screen.getByLabelText('Prompt'), {
      target: { value: '' },
    })

    expect(
      await screen.findByText(/Save is unavailable until/i),
    ).toBeInTheDocument()
    const saveButton = screen.getByRole('button', { name: /^Save$/i })
    expect(saveButton).toBeDisabled()
    fireEvent.click(saveButton)
    expect(updateWorkflow).not.toHaveBeenCalled()
  })

  it('confirms workflow archive with its target and consequence', async () => {
    const user = userEvent.setup()
    const archiveWorkflow = vi.fn(async () => workflow)
    setupHookMocks({ archiveWorkflow })
    render(<WorkflowsPage />)
    await screen.findByText('Workflow recurrence')

    await user.click(screen.getByRole('button', { name: /^Archive$/i }))
    expect(
      screen.getByRole('heading', {
        name: 'Archive Daily research workflow?',
      }),
    ).toBeInTheDocument()
    expect(
      screen.getByText(/no longer be available for new runs or schedules/i),
    ).toBeInTheDocument()
    expect(archiveWorkflow).not.toHaveBeenCalled()

    await user.click(screen.getByRole('button', { name: 'Archive workflow' }))
    await waitFor(() =>
      expect(archiveWorkflow).toHaveBeenCalledWith(workflow.id),
    )
  })

  it('requires explicit discard confirmation before archiving a dirty workflow', async () => {
    const user = userEvent.setup()
    const archivedWorkflow: WorkflowDefinitionDetail = {
      ...workflow,
      status: 'archived',
      archived_at: '2026-01-02T00:00:00Z',
    }
    const archiveWorkflow = vi.fn(async () => archivedWorkflow)
    setupHookMocks({ archiveWorkflow })
    render(<WorkflowsPage />)
    await screen.findByText('Workflow recurrence')

    const nameInput = screen.getAllByLabelText('Name')[0]
    fireEvent.change(nameInput, { target: { value: 'Unsaved workflow name' } })
    await user.click(screen.getByRole('button', { name: /^Archive$/i }))

    expect(
      screen.getByText(/unsaved changes will be permanently discarded/i),
    ).toBeInTheDocument()
    expect(archiveWorkflow).not.toHaveBeenCalled()

    await user.click(
      screen.getByRole('button', { name: 'Discard changes and archive' }),
    )

    await waitFor(() =>
      expect(archiveWorkflow).toHaveBeenCalledWith(workflow.id),
    )
    expect(nameInput).toHaveValue(workflow.name)
  })

  it('disables archive confirmation while the mutation is pending', async () => {
    const user = userEvent.setup()
    setupHookMocks({ archivePending: true })
    render(<WorkflowsPage />)
    await screen.findByText('Workflow recurrence')

    await user.click(screen.getByRole('button', { name: /^Archive$/i }))

    expect(
      screen.getByRole('button', { name: 'Archive workflow' }),
    ).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Cancel' })).toBeDisabled()
  })

  it('keeps a failed workflow archive confirmation open and shows the error', async () => {
    const user = userEvent.setup()
    setupHookMocks({
      archiveWorkflow: vi.fn(async () => {
        throw new Error('Archive request failed')
      }),
    })
    render(<WorkflowsPage />)
    await screen.findByText('Workflow recurrence')

    await user.click(screen.getByRole('button', { name: /^Archive$/i }))
    await user.click(screen.getByRole('button', { name: 'Archive workflow' }))

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'Archive request failed',
    )
    expect(
      screen.getByRole('heading', {
        name: 'Archive Daily research workflow?',
      }),
    ).toBeInTheDocument()
  })

  it('confirms deletion of a workflow-backed schedule', async () => {
    const user = userEvent.setup()
    const deleteSchedule = vi.fn(async () => workflowSchedule)
    setupHookMocks({ deleteSchedule })
    render(<WorkflowsPage />)
    await screen.findByText('Workflow recurrence')

    await user.click(screen.getByText('Weekday research'))
    await user.click(screen.getByRole('button', { name: /^Delete$/i }))
    expect(
      screen.getByRole('heading', { name: 'Delete Weekday research?' }),
    ).toBeInTheDocument()
    expect(screen.getByText(/will stop firing/i)).toBeInTheDocument()
    expect(deleteSchedule).not.toHaveBeenCalled()

    await user.click(screen.getByRole('button', { name: 'Delete schedule' }))
    await waitFor(() =>
      expect(deleteSchedule).toHaveBeenCalledWith(workflowSchedule.id),
    )
  })

  it('warns that deleting a dirty workflow schedule discards its edits', async () => {
    const user = userEvent.setup()
    setupHookMocks()
    render(<WorkflowsPage />)
    await screen.findByText('Workflow recurrence')

    await user.click(screen.getByText('Weekday research'))
    const scheduleEditor = screen
      .getByText('Edit workflow schedule')
      .closest('.rounded-2xl')
    expect(scheduleEditor).toBeTruthy()
    await user.type(
      within(scheduleEditor as HTMLElement).getByLabelText('Description'),
      'Unsaved detail',
    )
    await user.click(
      within(scheduleEditor as HTMLElement).getByRole('button', {
        name: /^Delete$/i,
      }),
    )

    expect(screen.getByRole('dialog')).toHaveTextContent(
      /unsaved edits in this embedded schedule will be permanently discarded/i,
    )
    expect(
      screen.getByRole('button', { name: 'Discard edits and delete' }),
    ).toBeVisible()
  })

  it('creates workflow-backed schedules from the Workflows detail panel', async () => {
    const user = userEvent.setup()
    const createSchedule = vi.fn(async (payload: ScheduleCreateRequest) => ({
      ...workflowSchedule,
      id: 'schedule-created',
      name: payload.name,
      description: payload.description,
      workflow_inputs_template: payload.workflow_inputs_template ?? {},
      metadata: payload.metadata ?? {},
    }))
    setupHookMocks({ schedules: [], createSchedule })

    render(<WorkflowsPage />)
    await screen.findByText('Workflow recurrence')

    await user.click(screen.getByRole('button', { name: /New schedule/i }))
    const scheduleEditorTitle = screen.getByText('Create workflow schedule')
    const scheduleEditor = scheduleEditorTitle.closest('.rounded-2xl')
    expect(scheduleEditor).toBeTruthy()
    const scheduleForm = within(scheduleEditor as HTMLElement)

    fireEvent.change(scheduleForm.getByLabelText('Name'), {
      target: { value: 'Morning workflow' },
    })
    fireEvent.change(scheduleForm.getByLabelText('Cron'), {
      target: { value: '30 8 * * 1-5' },
    })
    fireEvent.change(scheduleForm.getByLabelText(/Workflow inputs template/i), {
      target: { value: '{"topic":"{{ schedule.name }}"}' },
    })
    fireEvent.change(scheduleForm.getByLabelText('Metadata'), {
      target: { value: '{"source":"web-test"}' },
    })

    await user.click(
      scheduleForm.getByRole('button', { name: /Save schedule/i }),
    )

    await waitFor(() => expect(createSchedule).toHaveBeenCalledTimes(1))
    expect(createSchedule).toHaveBeenCalledWith({
      name: 'Morning workflow',
      description: null,
      prompt: '',
      trigger_kind: 'cron',
      cron: '30 8 * * 1-5',
      run_at: null,
      timezone: expect.any(String),
      enabled: true,
      continue_current_session: false,
      start_from_current_session: false,
      steer_when_running: false,
      owner_kind: 'user',
      workflow_id: workflow.id,
      workflow_inputs_template: { topic: '{{ schedule.name }}' },
      metadata: { source: 'web-test' },
    })

    expect(screen.getByText('No workflow schedules yet.')).toBeInTheDocument()
  })
})
