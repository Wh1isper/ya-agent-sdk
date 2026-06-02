import { render, screen, waitFor, within } from '@testing-library/react'
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
    mutate: vi.fn(),
  } as unknown as ReturnType<typeof hooks.useDeleteScheduleMutation>)
  vi.mocked(hooks.useTriggerScheduleMutation).mockReturnValue({
    mutate: vi.fn(),
  } as unknown as ReturnType<typeof hooks.useTriggerScheduleMutation>)
  vi.mocked(hooks.useCreateWorkflowMutation).mockReturnValue({
    mutateAsync: vi.fn(async () => workflow),
    isPending: false,
  } as unknown as ReturnType<typeof hooks.useCreateWorkflowMutation>)
  vi.mocked(hooks.useUpdateWorkflowMutation).mockReturnValue({
    mutateAsync: vi.fn(async () => workflow),
    isPending: false,
  } as unknown as ReturnType<typeof hooks.useUpdateWorkflowMutation>)
  vi.mocked(hooks.useArchiveWorkflowMutation).mockReturnValue({
    mutate: vi.fn(),
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
        limit: 100,
      })
    })
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

    await user.clear(scheduleForm.getByLabelText('Name'))
    await user.type(scheduleForm.getByLabelText('Name'), 'Morning workflow')
    await user.clear(scheduleForm.getByLabelText('Cron'))
    await user.type(scheduleForm.getByLabelText('Cron'), '30 8 * * 1-5')

    const inputsTemplate = scheduleForm.getByLabelText(
      /Workflow inputs template/i,
    )
    await user.clear(inputsTemplate)
    await user.click(inputsTemplate)
    await user.paste('{"topic":"{{ schedule.name }}"}')

    const metadata = scheduleForm.getByLabelText('Metadata')
    await user.clear(metadata)
    await user.click(metadata)
    await user.paste('{"source":"web-test"}')

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
