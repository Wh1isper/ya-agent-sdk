import {
  Archive,
  CalendarClock,
  GitBranch,
  Play,
  Plus,
  Save,
  Send,
  Trash2,
  Workflow,
  XCircle,
} from 'lucide-react'
import { type ReactNode, useEffect, useMemo, useState } from 'react'

import {
  useArchiveWorkflowMutation,
  useCreateScheduleMutation,
  useCreateWorkflowMutation,
  useDeleteScheduleMutation,
  useScheduleFiresQuery,
  useSchedulesQuery,
  useTriggerScheduleMutation,
  useTriggerWorkflowMutation,
  useUpdateScheduleMutation,
  useUpdateWorkflowMutation,
  useWorkflowEventsQuery,
  useWorkflowQuery,
  useWorkflowRunMutations,
  useWorkflowRunQuery,
  useWorkflowRunsQuery,
  useWorkflowsQuery,
} from '../../api/hooks'
import { EmptyState } from '../../components/EmptyState'
import { JsonView } from '../../components/JsonView'
import { StatusBadge } from '../../components/StatusBadge'
import {
  cn,
  formatShortId,
  joinCsv,
  parseJsonObject,
  safeJsonStringify,
  splitCsv,
} from '../../lib/utils'
import {
  describeBrowserDateTime,
  describeScheduledAndLocalDateTime,
  getBrowserTimeZone,
  getSupportedTimeZones,
  toZonedDatetimeLocalValue,
  zonedDatetimeLocalToIso,
} from '../../lib/timezone'
import type {
  ScheduleCreateRequest,
  ScheduleSummary,
  WorkflowDefinitionDetail,
  WorkflowDefinitionStatus,
  WorkflowDefinitionSummary,
  WorkflowEventSummary,
  WorkflowRunDetail,
  WorkflowRunSummary,
  WorkflowScope,
  WorkflowTriggerKind,
} from '../../types'

type WorkflowFormValues = {
  name: string
  description: string
  status: WorkflowDefinitionStatus
  scope: WorkflowScope
  tags: string
  when_to_use: string
  argument_hint: string
  input_schema: string
  definition: string
  metadata: string
}

type TriggerFormValues = {
  inputs: string
  profile_name: string
  supervisor_session_id: string
  supervisor_run_id: string
  trigger_kind: WorkflowTriggerKind
  metadata: string
}

type WorkflowScheduleFormValues = {
  schedule_id: string
  name: string
  description: string
  trigger_kind: 'cron' | 'once'
  cron: string
  run_at: string
  timezone: string
  enabled: boolean
  workflow_inputs_template: string
  metadata: string
}

const blankDefinition = {
  schema: 'ya-claw.workflow.v1',
  name: 'New workflow',
  version: 1,
  inputs: {
    type: 'object',
    properties: {
      topic: { type: 'string' },
    },
  },
  policy: { max_concurrency: 1 },
  nodes: {
    draft: {
      profile: 'Self',
      mode: 'isolate',
      prompt: 'Work on {{ inputs.topic | default("the requested task") }}.',
    },
  },
  result: { from_node: 'draft' },
}

const inputClass =
  'mt-2 w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm outline-none ring-blue-600 transition focus:bg-white focus:ring-2'
const textareaClass =
  'mt-2 w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm outline-none ring-blue-600 transition focus:bg-white focus:ring-2'
const cardClass = 'rounded-2xl border border-slate-200 bg-white p-4 shadow-sm'

function blankWorkflowForm(): WorkflowFormValues {
  return {
    name: 'New workflow',
    description: '',
    status: 'active',
    scope: 'global',
    tags: '',
    when_to_use: '',
    argument_hint: '',
    input_schema: safeJsonStringify(blankDefinition.inputs),
    definition: safeJsonStringify(blankDefinition),
    metadata: '{}',
  }
}

function workflowToForm(
  workflow: WorkflowDefinitionDetail,
): WorkflowFormValues {
  return {
    name: workflow.name,
    description: workflow.description ?? '',
    status: workflow.status,
    scope: workflow.scope,
    tags: joinCsv(workflow.tags),
    when_to_use: workflow.when_to_use ?? '',
    argument_hint: workflow.argument_hint ?? '',
    input_schema: safeJsonStringify(workflow.input_schema),
    definition: safeJsonStringify(workflow.definition),
    metadata: safeJsonStringify(workflow.metadata),
  }
}

function blankTriggerForm(): TriggerFormValues {
  return {
    inputs: '{}',
    profile_name: '',
    supervisor_session_id: '',
    supervisor_run_id: '',
    trigger_kind: 'web',
    metadata: '{}',
  }
}

function blankWorkflowScheduleForm(): WorkflowScheduleFormValues {
  return {
    schedule_id: '',
    name: 'Workflow schedule',
    description: '',
    trigger_kind: 'cron',
    cron: '0 9 * * *',
    run_at: '',
    timezone: getBrowserTimeZone(),
    enabled: true,
    workflow_inputs_template: '{}',
    metadata: '{}',
  }
}

function workflowScheduleToForm(
  schedule: ScheduleSummary,
): WorkflowScheduleFormValues {
  return {
    schedule_id: schedule.id,
    name: schedule.name,
    description: schedule.description ?? '',
    trigger_kind: schedule.trigger.kind,
    cron: schedule.trigger.kind === 'cron' ? (schedule.trigger.cron ?? '') : '',
    run_at:
      schedule.trigger.kind === 'once'
        ? toZonedDatetimeLocalValue(
            schedule.trigger.run_at,
            schedule.trigger.timezone,
          )
        : '',
    timezone: schedule.trigger.timezone,
    enabled: schedule.enabled,
    workflow_inputs_template: safeJsonStringify(
      schedule.workflow_inputs_template ?? {},
    ),
    metadata: safeJsonStringify(schedule.metadata ?? {}),
  }
}

export function WorkflowsPage() {
  const [selectedWorkflowId, setSelectedWorkflowId] = useState<string | null>(
    null,
  )
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null)
  const [creating, setCreating] = useState(false)
  const [query, setQuery] = useState('')
  const [tagText, setTagText] = useState('')
  const [status, setStatus] = useState<WorkflowDefinitionStatus | 'all'>('all')
  const [scope, setScope] = useState<WorkflowScope | 'all'>('all')
  const [includeArchived, setIncludeArchived] = useState(false)
  const [onlyCurrentSession, setOnlyCurrentSession] = useState(false)
  const [currentSessionId, setCurrentSessionId] = useState('')

  const workflowFilters = useMemo(
    () => ({
      query,
      tags: splitCsv(tagText),
      status,
      scope,
      includeArchived,
      onlyCurrentSession,
      currentSessionId,
      limit: 100,
    }),
    [
      currentSessionId,
      includeArchived,
      onlyCurrentSession,
      query,
      scope,
      status,
      tagText,
    ],
  )
  const workflows = useWorkflowsQuery(workflowFilters)
  const workflowRows = useMemo(
    () => workflows.data?.workflows ?? [],
    [workflows.data?.workflows],
  )
  const selectedWorkflow = useWorkflowQuery(
    creating ? null : selectedWorkflowId,
  )
  const workflowSchedules = useSchedulesQuery({
    workflowId: selectedWorkflowId,
    executionMode: 'workflow',
    includeWorkflow: true,
    includeDeleted: true,
    limit: 100,
  })
  const workflowScheduleRows = useMemo(
    () => workflowSchedules.data?.schedules ?? [],
    [workflowSchedules.data?.schedules],
  )

  const runFilters = useMemo(
    () => ({
      workflowId: selectedWorkflowId,
      includeCompleted: true,
      limit: 100,
    }),
    [selectedWorkflowId],
  )
  const runs = useWorkflowRunsQuery(runFilters)
  const runRows = useMemo(
    () => runs.data?.workflow_runs ?? [],
    [runs.data?.workflow_runs],
  )

  useEffect(() => {
    if (!selectedWorkflowId && workflowRows[0] && !creating) {
      setSelectedWorkflowId(workflowRows[0].id)
    }
  }, [creating, selectedWorkflowId, workflowRows])

  useEffect(() => {
    if (selectedRunId && runRows.some((run) => run.id === selectedRunId)) return
    setSelectedRunId(runRows[0]?.id ?? null)
  }, [runRows, selectedRunId])

  return (
    <div className="flex h-full min-h-0 bg-slate-100">
      <aside className="flex w-96 shrink-0 flex-col border-r border-slate-200 bg-white">
        <div className="border-b border-slate-200 p-4">
          <div className="flex items-center justify-between gap-2">
            <div>
              <p className="text-sm font-medium text-blue-600">Orchestration</p>
              <h1 className="mt-1 text-xl font-semibold tracking-tight text-slate-950">
                Workflows
              </h1>
            </div>
            <button
              type="button"
              className="inline-flex items-center gap-2 rounded-xl bg-blue-600 px-3 py-2 text-xs font-semibold text-white shadow-sm transition hover:bg-blue-700"
              onClick={() => {
                setCreating(true)
                setSelectedWorkflowId(null)
                setSelectedRunId(null)
              }}
            >
              <Plus className="h-3.5 w-3.5" />
              New
            </button>
          </div>
          <div className="mt-4 space-y-2">
            <input
              className={inputClass.replace('mt-2 ', '')}
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search workflows"
            />
            <input
              className={inputClass.replace('mt-2 ', '')}
              value={tagText}
              onChange={(event) => setTagText(event.target.value)}
              placeholder="Tags, comma separated"
            />
            <div className="grid grid-cols-2 gap-2">
              <select
                className={selectClass()}
                value={status}
                onChange={(event) =>
                  setStatus(
                    event.target.value as WorkflowDefinitionStatus | 'all',
                  )
                }
              >
                <option value="all">All status</option>
                <option value="active">Active</option>
                <option value="draft">Draft</option>
                <option value="archived">Archived</option>
              </select>
              <select
                className={selectClass()}
                value={scope}
                onChange={(event) =>
                  setScope(event.target.value as WorkflowScope | 'all')
                }
              >
                <option value="all">All scope</option>
                <option value="global">Global</option>
                <option value="session">Session</option>
              </select>
            </div>
            <label className="flex items-center justify-between rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600">
              Include archived
              <input
                type="checkbox"
                checked={includeArchived}
                onChange={(event) => setIncludeArchived(event.target.checked)}
              />
            </label>
            <label className="flex items-center justify-between rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600">
              Only current session
              <input
                type="checkbox"
                checked={onlyCurrentSession}
                onChange={(event) =>
                  setOnlyCurrentSession(event.target.checked)
                }
              />
            </label>
            {onlyCurrentSession ? (
              <input
                className={inputClass.replace('mt-2 ', '')}
                value={currentSessionId}
                onChange={(event) => setCurrentSessionId(event.target.value)}
                placeholder="Current session ID"
              />
            ) : null}
            <p className="text-xs text-slate-400">
              Showing {workflowRows.length} workflows
            </p>
          </div>
        </div>
        <div className="scrollbar-thin min-h-0 flex-1 overflow-auto p-3">
          {workflows.isLoading ? <ListSkeleton /> : null}
          {!workflows.isLoading && workflowRows.length === 0 ? (
            <EmptyState
              title="No workflows"
              description="Create a reusable DAG workflow for agent orchestration."
            />
          ) : null}
          <div className="space-y-2">
            {workflowRows.map((workflow) => (
              <WorkflowListItem
                key={workflow.id}
                workflow={workflow}
                scheduleCount={
                  selectedWorkflowId === workflow.id
                    ? workflowScheduleRows.filter(
                        (schedule) => schedule.status !== 'deleted',
                      ).length
                    : undefined
                }
                active={!creating && selectedWorkflowId === workflow.id}
                onClick={() => {
                  setCreating(false)
                  setSelectedWorkflowId(workflow.id)
                }}
              />
            ))}
          </div>
        </div>
      </aside>

      <main className="grid min-w-0 flex-1 grid-cols-[minmax(0,1fr)_28rem] overflow-hidden">
        <section className="scrollbar-thin min-w-0 overflow-auto p-6">
          <WorkflowEditor
            workflow={creating ? null : (selectedWorkflow.data ?? null)}
            schedules={workflowScheduleRows}
            creating={creating}
            onCreated={(workflowId) => {
              setCreating(false)
              setSelectedWorkflowId(workflowId)
            }}
          />
        </section>
        <aside className="flex min-h-0 flex-col border-l border-slate-200 bg-white">
          <RunInspector
            workflow={creating ? null : (selectedWorkflow.data ?? null)}
            runs={runRows}
            selectedRunId={selectedRunId}
            setSelectedRunId={setSelectedRunId}
          />
        </aside>
      </main>
    </div>
  )
}

function WorkflowListItem({
  workflow,
  scheduleCount,
  active,
  onClick,
}: {
  workflow: WorkflowDefinitionSummary
  scheduleCount?: number
  active: boolean
  onClick: () => void
}) {
  return (
    <button
      type="button"
      className={cn(
        'w-full rounded-2xl border p-3 text-left transition',
        active
          ? 'border-blue-200 bg-blue-50 shadow-sm'
          : 'border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50',
      )}
      onClick={onClick}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="truncate text-sm font-semibold text-slate-900">
            {workflow.name}
          </p>
          <p className="mt-1 truncate mono text-xs text-slate-500">
            v{workflow.definition_version} · {workflow.scope} ·{' '}
            {formatShortId(workflow.id)}
          </p>
        </div>
        <StatusBadge status={workflow.status} />
      </div>
      <p className="mt-2 line-clamp-2 text-xs text-slate-500">
        {workflow.description || workflow.when_to_use || 'No description'}
      </p>
      <div className="mt-3 flex flex-wrap gap-1">
        {workflow.tags.slice(0, 4).map((tag) => (
          <span
            key={tag}
            className="rounded-full bg-slate-100 px-2 py-0.5 text-[10px] font-medium text-slate-500"
          >
            {tag}
          </span>
        ))}
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-slate-400">
        {workflow.latest_run ? (
          <span>
            Latest run: {workflow.latest_run.status} ·{' '}
            {formatShortId(workflow.latest_run.id)}
          </span>
        ) : null}
        {typeof scheduleCount === 'number' ? (
          <span className="inline-flex items-center gap-1 rounded-full bg-blue-50 px-2 py-0.5 font-medium text-blue-600">
            <CalendarClock className="h-3 w-3" />
            {scheduleCount} schedules
          </span>
        ) : null}
      </div>
    </button>
  )
}

function WorkflowEditor({
  workflow,
  schedules,
  creating,
  onCreated,
}: {
  workflow: WorkflowDefinitionDetail | null
  schedules: ScheduleSummary[]
  creating: boolean
  onCreated: (workflowId: string) => void
}) {
  const [form, setForm] = useState<WorkflowFormValues>(blankWorkflowForm)
  const [trigger, setTrigger] = useState<TriggerFormValues>(blankTriggerForm)
  const [error, setError] = useState<string | null>(null)
  const createWorkflow = useCreateWorkflowMutation()
  const updateWorkflow = useUpdateWorkflowMutation()
  const archiveWorkflow = useArchiveWorkflowMutation()
  const triggerWorkflow = useTriggerWorkflowMutation()

  useEffect(() => {
    setError(null)
    setTrigger(blankTriggerForm())
    setForm(workflow ? workflowToForm(workflow) : blankWorkflowForm())
  }, [workflow, creating])

  if (!creating && !workflow) {
    return (
      <EmptyState
        title="Select a workflow"
        description="Choose a workflow from the list or create a new one."
      />
    )
  }

  const save = async () => {
    setError(null)
    try {
      const payload = {
        name: form.name.trim(),
        description: form.description.trim() || null,
        status: form.status,
        scope: form.scope,
        tags: splitCsv(form.tags),
        when_to_use: form.when_to_use.trim() || null,
        argument_hint: form.argument_hint.trim() || null,
        input_schema: parseJsonObject(form.input_schema) ?? {},
        definition: parseJsonObject(form.definition) ?? {},
        metadata: parseJsonObject(form.metadata) ?? {},
      }
      if (creating || !workflow) {
        const created = await createWorkflow.mutateAsync(payload)
        onCreated(created.id)
        return
      }
      await updateWorkflow.mutateAsync({ workflowId: workflow.id, payload })
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught))
    }
  }

  const start = async () => {
    if (!workflow) return
    setError(null)
    try {
      await triggerWorkflow.mutateAsync({
        workflowId: workflow.id,
        payload: {
          inputs: parseJsonObject(trigger.inputs) ?? {},
          profile_name: trigger.profile_name.trim() || null,
          supervisor_session_id: trigger.supervisor_session_id.trim() || null,
          supervisor_run_id: trigger.supervisor_run_id.trim() || null,
          trigger_kind: trigger.trigger_kind,
          metadata: parseJsonObject(trigger.metadata) ?? {},
        },
      })
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught))
    }
  }

  return (
    <div className="space-y-4">
      <div className={cardClass}>
        <div className="flex items-start justify-between gap-3">
          <div>
            <p className="text-sm font-medium text-blue-600">Definition</p>
            <h2 className="mt-1 text-xl font-semibold tracking-tight text-slate-950">
              {creating
                ? 'New workflow'
                : (workflow?.name ?? 'Select a workflow')}
            </h2>
            {workflow ? (
              <p className="mt-1 mono text-xs text-slate-500">
                {workflow.id} · updated {formatDate(workflow.updated_at)}
              </p>
            ) : null}
          </div>
          <div className="flex gap-2">
            {workflow && workflow.status !== 'archived' ? (
              <button
                type="button"
                className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-700 shadow-sm hover:bg-slate-50"
                onClick={() => archiveWorkflow.mutate(workflow.id)}
              >
                <Archive className="h-3.5 w-3.5" />
                Archive
              </button>
            ) : null}
            <button
              type="button"
              className="inline-flex items-center gap-2 rounded-xl bg-blue-600 px-3 py-2 text-xs font-semibold text-white shadow-sm hover:bg-blue-700"
              onClick={save}
              disabled={createWorkflow.isPending || updateWorkflow.isPending}
            >
              <Save className="h-3.5 w-3.5" />
              Save
            </button>
          </div>
        </div>
        {error ? (
          <div className="mt-4 rounded-xl border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700">
            {error}
          </div>
        ) : null}
        <div className="mt-4 grid grid-cols-2 gap-4">
          <label className="text-sm font-medium text-slate-700">
            Name
            <input
              className={inputClass}
              value={form.name}
              onChange={(event) =>
                setForm({ ...form, name: event.target.value })
              }
            />
          </label>
          <label className="text-sm font-medium text-slate-700">
            Tags
            <input
              className={inputClass}
              value={form.tags}
              onChange={(event) =>
                setForm({ ...form, tags: event.target.value })
              }
              placeholder="research, daily"
            />
          </label>
          <label className="text-sm font-medium text-slate-700">
            Status
            <select
              className={inputClass}
              value={form.status}
              onChange={(event) =>
                setForm({
                  ...form,
                  status: event.target.value as WorkflowDefinitionStatus,
                })
              }
            >
              <option value="active">Active</option>
              <option value="draft">Draft</option>
              <option value="archived">Archived</option>
            </select>
          </label>
          <label className="text-sm font-medium text-slate-700">
            Scope
            <select
              className={inputClass}
              value={form.scope}
              onChange={(event) =>
                setForm({ ...form, scope: event.target.value as WorkflowScope })
              }
            >
              <option value="global">Global</option>
              <option value="session">Session</option>
            </select>
          </label>
        </div>
        <label className="mt-4 block text-sm font-medium text-slate-700">
          Description
          <textarea
            className={textareaClass}
            rows={3}
            value={form.description}
            onChange={(event) =>
              setForm({ ...form, description: event.target.value })
            }
          />
        </label>
        <div className="mt-4 grid grid-cols-2 gap-4">
          <label className="text-sm font-medium text-slate-700">
            When to use
            <textarea
              className={textareaClass}
              rows={3}
              value={form.when_to_use}
              onChange={(event) =>
                setForm({ ...form, when_to_use: event.target.value })
              }
            />
          </label>
          <label className="text-sm font-medium text-slate-700">
            Argument hint
            <textarea
              className={textareaClass}
              rows={3}
              value={form.argument_hint}
              onChange={(event) =>
                setForm({ ...form, argument_hint: event.target.value })
              }
            />
          </label>
        </div>
      </div>

      {workflow ? (
        <WorkflowSchedulesPanel workflow={workflow} schedules={schedules} />
      ) : null}

      <div className={cardClass}>
        <div className="flex items-center gap-2">
          <Workflow className="h-4 w-4 text-blue-600" />
          <h3 className="font-semibold text-slate-900">Workflow JSON</h3>
        </div>
        <label className="mt-4 block text-sm font-medium text-slate-700">
          Input schema
          <textarea
            className={`${textareaClass} mono`}
            rows={8}
            value={form.input_schema}
            onChange={(event) =>
              setForm({ ...form, input_schema: event.target.value })
            }
          />
        </label>
        <label className="mt-4 block text-sm font-medium text-slate-700">
          Definition
          <textarea
            className={`${textareaClass} mono`}
            rows={18}
            value={form.definition}
            onChange={(event) =>
              setForm({ ...form, definition: event.target.value })
            }
          />
        </label>
        <label className="mt-4 block text-sm font-medium text-slate-700">
          Metadata
          <textarea
            className={`${textareaClass} mono`}
            rows={5}
            value={form.metadata}
            onChange={(event) =>
              setForm({ ...form, metadata: event.target.value })
            }
          />
        </label>
      </div>

      {workflow ? (
        <div className={cardClass}>
          <div className="flex items-center justify-between gap-3">
            <div>
              <p className="text-sm font-medium text-blue-600">Run</p>
              <h3 className="font-semibold text-slate-900">Trigger workflow</h3>
            </div>
            <button
              type="button"
              className="inline-flex items-center gap-2 rounded-xl bg-emerald-600 px-3 py-2 text-xs font-semibold text-white shadow-sm hover:bg-emerald-700"
              onClick={start}
              disabled={
                triggerWorkflow.isPending || workflow.status !== 'active'
              }
            >
              <Play className="h-3.5 w-3.5" />
              Start
            </button>
          </div>
          <div className="mt-4 grid grid-cols-2 gap-4">
            <label className="text-sm font-medium text-slate-700">
              Trigger kind
              <select
                className={inputClass}
                value={trigger.trigger_kind}
                onChange={(event) =>
                  setTrigger({
                    ...trigger,
                    trigger_kind: event.target.value as WorkflowTriggerKind,
                  })
                }
              >
                <option value="web">Web</option>
                <option value="api">API</option>
                <option value="agent">Agent</option>
                <option value="schedule">Schedule</option>
                <option value="system">System</option>
              </select>
            </label>
            <label className="text-sm font-medium text-slate-700">
              Profile
              <input
                className={inputClass}
                value={trigger.profile_name}
                onChange={(event) =>
                  setTrigger({ ...trigger, profile_name: event.target.value })
                }
                placeholder="default"
              />
            </label>
            <label className="text-sm font-medium text-slate-700">
              Supervisor session
              <input
                className={inputClass}
                value={trigger.supervisor_session_id}
                onChange={(event) =>
                  setTrigger({
                    ...trigger,
                    supervisor_session_id: event.target.value,
                  })
                }
              />
            </label>
            <label className="text-sm font-medium text-slate-700">
              Supervisor run
              <input
                className={inputClass}
                value={trigger.supervisor_run_id}
                onChange={(event) =>
                  setTrigger({
                    ...trigger,
                    supervisor_run_id: event.target.value,
                  })
                }
              />
            </label>
          </div>
          <label className="mt-4 block text-sm font-medium text-slate-700">
            Inputs
            <textarea
              className={`${textareaClass} mono`}
              rows={6}
              value={trigger.inputs}
              onChange={(event) =>
                setTrigger({ ...trigger, inputs: event.target.value })
              }
            />
          </label>
          <label className="mt-4 block text-sm font-medium text-slate-700">
            Metadata
            <textarea
              className={`${textareaClass} mono`}
              rows={4}
              value={trigger.metadata}
              onChange={(event) =>
                setTrigger({ ...trigger, metadata: event.target.value })
              }
            />
          </label>
        </div>
      ) : null}
    </div>
  )
}

function WorkflowSchedulesPanel({
  workflow,
  schedules,
}: {
  workflow: WorkflowDefinitionDetail
  schedules: ScheduleSummary[]
}) {
  const createSchedule = useCreateScheduleMutation()
  const updateSchedule = useUpdateScheduleMutation()
  const deleteSchedule = useDeleteScheduleMutation()
  const triggerSchedule = useTriggerScheduleMutation()
  const [form, setForm] = useState<WorkflowScheduleFormValues>(
    blankWorkflowScheduleForm,
  )
  const [error, setError] = useState<string | null>(null)
  const activeSchedules = schedules.filter(
    (schedule) => schedule.status !== 'deleted',
  )
  const selectedSchedule = schedules.find(
    (schedule) => schedule.id === form.schedule_id,
  )
  const supportedTimeZones = useMemo(() => getSupportedTimeZones(), [])
  const fires = useScheduleFiresQuery(selectedSchedule?.id ?? null)

  useEffect(() => {
    setError(null)
    setForm(blankWorkflowScheduleForm())
  }, [workflow.id])

  const selectSchedule = (schedule: ScheduleSummary) => {
    setError(null)
    setForm(workflowScheduleToForm(schedule))
  }

  const resetForm = () => {
    setError(null)
    setForm(blankWorkflowScheduleForm())
  }

  const save = async () => {
    setError(null)
    try {
      const payload: ScheduleCreateRequest = {
        name: form.name.trim(),
        description: form.description.trim() || null,
        prompt: '',
        trigger_kind: form.trigger_kind,
        cron: form.trigger_kind === 'cron' ? form.cron : null,
        run_at:
          form.trigger_kind === 'once'
            ? zonedDatetimeLocalToIso(form.run_at, form.timezone)
            : null,
        timezone: form.timezone,
        enabled: form.enabled,
        continue_current_session: false,
        start_from_current_session: false,
        steer_when_running: false,
        owner_kind: 'user',
        workflow_id: workflow.id,
        workflow_inputs_template:
          parseJsonObject(form.workflow_inputs_template) ?? {},
        metadata: parseJsonObject(form.metadata) ?? {},
      }
      if (form.schedule_id) {
        await updateSchedule.mutateAsync({
          scheduleId: form.schedule_id,
          payload,
        })
      } else {
        const created = await createSchedule.mutateAsync(payload)
        setForm(workflowScheduleToForm(created))
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught))
    }
  }

  return (
    <div className={cardClass}>
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="text-sm font-medium text-blue-600">Schedules</p>
          <h3 className="font-semibold text-slate-900">Workflow recurrence</h3>
        </div>
        <button
          type="button"
          className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-700 shadow-sm hover:bg-slate-50"
          onClick={resetForm}
        >
          <Plus className="h-3.5 w-3.5" />
          New schedule
        </button>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-[18rem_minmax(0,1fr)]">
        <div className="space-y-2">
          {activeSchedules.length === 0 ? (
            <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50 p-4 text-sm text-slate-500">
              No workflow schedules yet.
            </div>
          ) : null}
          {activeSchedules.map((schedule) => (
            <button
              key={schedule.id}
              type="button"
              className={cn(
                'w-full rounded-2xl border p-3 text-left transition',
                form.schedule_id === schedule.id
                  ? 'border-blue-200 bg-blue-50'
                  : 'border-slate-200 bg-white hover:bg-slate-50',
              )}
              onClick={() => selectSchedule(schedule)}
            >
              <div className="flex items-start justify-between gap-2">
                <p className="truncate text-sm font-semibold text-slate-900">
                  {schedule.name}
                </p>
                <StatusBadge status={schedule.status} />
              </div>
              <p className="mt-1 text-xs text-slate-500">
                {formatWorkflowScheduleTrigger(schedule)}
              </p>
              <p className="mt-1 text-xs text-slate-400">
                Next:{' '}
                {describeScheduledAndLocalDateTime(
                  schedule.trigger.next_fire_at,
                  schedule.trigger.timezone,
                )}
              </p>
              {schedule.last_workflow_run_id ? (
                <p className="mt-1 mono text-xs text-slate-400">
                  Last run {formatShortId(schedule.last_workflow_run_id)}
                </p>
              ) : null}
            </button>
          ))}
        </div>

        <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
          <div className="flex items-start justify-between gap-3">
            <div>
              <p className="text-sm font-semibold text-slate-900">
                {form.schedule_id
                  ? 'Edit workflow schedule'
                  : 'Create workflow schedule'}
              </p>
              <p className="mt-1 text-xs text-slate-500">
                Recurrence creates workflow runs with trigger_kind=schedule.
              </p>
            </div>
            <div className="flex gap-2">
              {selectedSchedule && selectedSchedule.status !== 'deleted' ? (
                <button
                  type="button"
                  className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50"
                  onClick={() =>
                    triggerSchedule.mutate({ scheduleId: selectedSchedule.id })
                  }
                >
                  <Play className="h-3.5 w-3.5" />
                  Trigger
                </button>
              ) : null}
              {selectedSchedule && selectedSchedule.status !== 'deleted' ? (
                <button
                  type="button"
                  className="inline-flex items-center gap-2 rounded-xl border border-rose-200 bg-white px-3 py-2 text-xs font-semibold text-rose-700 hover:bg-rose-50"
                  onClick={() => deleteSchedule.mutate(selectedSchedule.id)}
                >
                  <Trash2 className="h-3.5 w-3.5" />
                  Delete
                </button>
              ) : null}
            </div>
          </div>

          {error ? (
            <div className="mt-3 rounded-xl border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700">
              {error}
            </div>
          ) : null}

          <div className="mt-4 grid grid-cols-2 gap-4">
            <WorkflowField label="Name">
              <input
                className={inputClass}
                value={form.name}
                onChange={(event) =>
                  setForm({ ...form, name: event.target.value })
                }
              />
            </WorkflowField>
            <WorkflowField label="Trigger">
              <select
                className={inputClass}
                value={form.trigger_kind}
                onChange={(event) =>
                  setForm({
                    ...form,
                    trigger_kind: event.target.value as 'cron' | 'once',
                  })
                }
              >
                <option value="cron">Recurring cron</option>
                <option value="once">One-time</option>
              </select>
            </WorkflowField>
            {form.trigger_kind === 'cron' ? (
              <WorkflowField label="Cron">
                <input
                  className={`${inputClass} mono`}
                  value={form.cron}
                  onChange={(event) =>
                    setForm({ ...form, cron: event.target.value })
                  }
                />
              </WorkflowField>
            ) : (
              <WorkflowField label="Run at">
                <input
                  type="datetime-local"
                  className={inputClass}
                  value={form.run_at}
                  onChange={(event) =>
                    setForm({ ...form, run_at: event.target.value })
                  }
                />
              </WorkflowField>
            )}
            <WorkflowField label="Timezone">
              {supportedTimeZones.length > 0 ? (
                <select
                  className={inputClass}
                  value={form.timezone}
                  onChange={(event) =>
                    setForm({ ...form, timezone: event.target.value })
                  }
                >
                  {supportedTimeZones.map((timeZone) => (
                    <option key={timeZone} value={timeZone}>
                      {timeZone}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  className={inputClass}
                  value={form.timezone}
                  onChange={(event) =>
                    setForm({ ...form, timezone: event.target.value })
                  }
                />
              )}
            </WorkflowField>
            <WorkflowField label="Description">
              <input
                className={inputClass}
                value={form.description}
                onChange={(event) =>
                  setForm({ ...form, description: event.target.value })
                }
              />
            </WorkflowField>
            <WorkflowField label="Enabled">
              <label className="mt-2 flex h-10 items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 text-sm text-slate-700">
                <input
                  type="checkbox"
                  checked={form.enabled}
                  onChange={(event) =>
                    setForm({ ...form, enabled: event.target.checked })
                  }
                />
                Enabled
              </label>
            </WorkflowField>
          </div>
          <WorkflowField
            label="Workflow inputs template"
            hint={
              'JSON object rendered with schedule context, for example {"topic": "{{ schedule.name }}"}.'
            }
          >
            <textarea
              className={`${textareaClass} mono`}
              rows={5}
              value={form.workflow_inputs_template}
              onChange={(event) =>
                setForm({
                  ...form,
                  workflow_inputs_template: event.target.value,
                })
              }
            />
          </WorkflowField>
          <WorkflowField label="Metadata">
            <textarea
              className={`${textareaClass} mono`}
              rows={3}
              value={form.metadata}
              onChange={(event) =>
                setForm({ ...form, metadata: event.target.value })
              }
            />
          </WorkflowField>
          <div className="mt-4 flex justify-end">
            <button
              type="button"
              className="inline-flex items-center gap-2 rounded-xl bg-blue-600 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-blue-700"
              onClick={save}
              disabled={createSchedule.isPending || updateSchedule.isPending}
            >
              <Save className="h-4 w-4" />
              Save schedule
            </button>
          </div>

          {selectedSchedule ? (
            <div className="mt-5 border-t border-slate-200 pt-4">
              <h4 className="text-sm font-semibold text-slate-900">
                Recent fires
              </h4>
              <div className="mt-3 space-y-2">
                {(fires.data?.fires ?? []).map((fire) => (
                  <div
                    key={fire.id}
                    className="rounded-xl border border-slate-200 bg-white p-3 text-sm"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="mono text-xs text-slate-500">
                        {formatShortId(fire.id)}
                      </span>
                      <StatusBadge
                        status={mapWorkflowScheduleFireStatus(
                          fire.status,
                          fire.run_status,
                        )}
                      />
                    </div>
                    <p className="mt-1 text-xs text-slate-500">
                      Workflow run{' '}
                      {fire.workflow_run_id
                        ? formatShortId(fire.workflow_run_id)
                        : 'none'}{' '}
                      · {describeBrowserDateTime(fire.created_at)}
                    </p>
                    {fire.error_message ? (
                      <p className="mt-1 text-xs text-rose-600">
                        {fire.error_message}
                      </p>
                    ) : null}
                  </div>
                ))}
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  )
}

function RunInspector({
  workflow,
  runs,
  selectedRunId,
  setSelectedRunId,
}: {
  workflow: WorkflowDefinitionDetail | null
  runs: WorkflowRunSummary[]
  selectedRunId: string | null
  setSelectedRunId: (runId: string | null) => void
}) {
  const selectedRun = useWorkflowRunQuery(selectedRunId)
  const events = useWorkflowEventsQuery(selectedRunId)
  const runDetail = selectedRun.data ?? null

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="border-b border-slate-200 p-4">
        <div className="flex items-center justify-between gap-2">
          <div>
            <p className="text-sm font-medium text-blue-600">Runs</p>
            <h2 className="mt-1 text-lg font-semibold tracking-tight text-slate-950">
              {workflow ? workflow.name : 'Workflow runs'}
            </h2>
          </div>
          <StatusBadge status={`${runs.length} runs`} />
        </div>
      </div>
      <div className="scrollbar-thin max-h-56 overflow-auto border-b border-slate-200 p-3">
        {runs.length === 0 ? (
          <EmptyState
            title="No runs"
            description="Start this workflow to create a run."
          />
        ) : null}
        <div className="space-y-2">
          {runs.map((run) => (
            <button
              key={run.id}
              type="button"
              className={cn(
                'w-full rounded-xl border p-3 text-left transition',
                selectedRunId === run.id
                  ? 'border-blue-200 bg-blue-50'
                  : 'border-slate-200 bg-white hover:bg-slate-50',
              )}
              onClick={() => setSelectedRunId(run.id)}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="mono text-xs font-semibold text-slate-700">
                  {formatShortId(run.id)}
                </span>
                <StatusBadge status={run.status} />
              </div>
              <p className="mt-2 text-xs text-slate-500">
                {run.trigger_kind} · {formatDate(run.created_at)}
              </p>
            </button>
          ))}
        </div>
      </div>
      <div className="scrollbar-thin min-h-0 flex-1 overflow-auto p-4">
        <WorkflowRunDetailPanel
          run={runDetail}
          events={events.data?.events ?? []}
        />
      </div>
    </div>
  )
}

function WorkflowRunDetailPanel({
  run,
  events,
}: {
  run: WorkflowRunDetail | null
  events: WorkflowEventSummary[]
}) {
  const [nodeId, setNodeId] = useState('')
  const [prompt, setPrompt] = useState('')
  const [reason, setReason] = useState('')
  const mutations = useWorkflowRunMutations(run?.id ?? null)

  useEffect(() => {
    setNodeId(
      run?.nodes.find((node) => isActiveNodeStatus(node.status))?.node_id ?? '',
    )
    setPrompt('')
  }, [run])

  if (!run) {
    return (
      <EmptyState
        title="No run selected"
        description="Select a workflow run to inspect nodes and events."
      />
    )
  }

  return (
    <div className="space-y-4">
      <div className="rounded-2xl border border-slate-200 bg-slate-50 p-3">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <p className="mono truncate text-xs font-semibold text-slate-700">
              {run.id}
            </p>
            <p className="mt-1 text-xs text-slate-500">
              {run.trigger_kind} · {formatDate(run.created_at)}
            </p>
          </div>
          <StatusBadge status={run.status} />
        </div>
        {run.error_message ? (
          <p className="mt-2 text-xs text-rose-600">{run.error_message}</p>
        ) : null}
      </div>

      {run.status !== 'completed' &&
      run.status !== 'failed' &&
      run.status !== 'cancelled' ? (
        <div className="rounded-2xl border border-slate-200 bg-white p-3">
          <div className="flex items-center gap-2">
            <XCircle className="h-4 w-4 text-rose-600" />
            <p className="text-sm font-semibold text-slate-900">Cancel run</p>
          </div>
          <input
            className={inputClass}
            value={reason}
            onChange={(event) => setReason(event.target.value)}
            placeholder="Reason"
          />
          <button
            type="button"
            className="mt-3 inline-flex items-center gap-2 rounded-xl border border-rose-200 bg-rose-50 px-3 py-2 text-xs font-semibold text-rose-700 hover:bg-rose-100"
            onClick={() => mutations.cancel.mutate(reason || null)}
          >
            Cancel
          </button>
        </div>
      ) : null}

      <section>
        <div className="mb-2 flex items-center gap-2">
          <GitBranch className="h-4 w-4 text-blue-600" />
          <h3 className="text-sm font-semibold text-slate-900">Nodes</h3>
        </div>
        <div className="space-y-2">
          {run.nodes.map((node) => (
            <div
              key={node.id}
              className="rounded-2xl border border-slate-200 bg-white p-3"
            >
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0">
                  <p className="truncate text-sm font-semibold text-slate-900">
                    {node.node_id}
                  </p>
                  <p className="mt-1 mono text-xs text-slate-500">
                    attempt {node.attempt_no} · {formatShortId(node.run_id)}
                  </p>
                </div>
                <StatusBadge status={node.status} />
              </div>
              {node.needs.length ? (
                <p className="mt-2 text-xs text-slate-500">
                  Needs: {node.needs.join(', ')}
                </p>
              ) : null}
              {node.input_preview ? (
                <p className="mt-2 line-clamp-3 text-xs text-slate-500">
                  {node.input_preview}
                </p>
              ) : null}
              {node.output_text ? (
                <p className="mt-2 line-clamp-3 text-xs text-slate-600">
                  {node.output_text}
                </p>
              ) : null}
              {isActiveNodeStatus(node.status) ? (
                <button
                  type="button"
                  className="mt-3 inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-1.5 text-xs font-semibold text-slate-700 hover:bg-slate-50"
                  onClick={() => setNodeId(node.node_id)}
                >
                  <Send className="h-3.5 w-3.5" />
                  Steer
                </button>
              ) : null}
            </div>
          ))}
        </div>
      </section>

      {nodeId ? (
        <section className="rounded-2xl border border-slate-200 bg-white p-3">
          <p className="text-sm font-semibold text-slate-900">
            Steer node {nodeId}
          </p>
          <textarea
            className={textareaClass}
            rows={4}
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            placeholder="Additional instruction"
          />
          <button
            type="button"
            className="mt-3 inline-flex items-center gap-2 rounded-xl bg-blue-600 px-3 py-2 text-xs font-semibold text-white hover:bg-blue-700"
            onClick={() => mutations.steerNode.mutate({ nodeId, prompt })}
            disabled={!prompt.trim()}
          >
            <Send className="h-3.5 w-3.5" />
            Send steer
          </button>
        </section>
      ) : null}

      <section>
        <h3 className="mb-2 text-sm font-semibold text-slate-900">Result</h3>
        <JsonView value={run.result ?? {}} height="220px" />
      </section>

      <section>
        <h3 className="mb-2 text-sm font-semibold text-slate-900">Events</h3>
        <div className="space-y-2">
          {events.map((event) => (
            <div
              key={event.id}
              className="rounded-xl border border-slate-200 bg-white p-3"
            >
              <div className="flex items-center justify-between gap-2">
                <p className="text-xs font-semibold text-slate-800">
                  {event.event_type}
                </p>
                <span className="text-xs text-slate-400">
                  {formatDate(event.created_at)}
                </span>
              </div>
              <p className="mt-1 text-xs text-slate-500">{event.source_kind}</p>
              <JsonView value={event.payload} height="140px" />
            </div>
          ))}
        </div>
      </section>

      <section>
        <h3 className="mb-2 text-sm font-semibold text-slate-900">Inputs</h3>
        <JsonView value={run.inputs} height="180px" />
      </section>
    </div>
  )
}

function WorkflowField({
  label,
  hint,
  children,
}: {
  label: string
  hint?: string
  children: ReactNode
}) {
  return (
    <label className="mt-4 block text-sm font-medium text-slate-700">
      {label}
      {children}
      {hint ? (
        <span className="mt-1 block text-xs font-normal text-slate-400">
          {hint}
        </span>
      ) : null}
    </label>
  )
}

function ListSkeleton() {
  return (
    <div className="space-y-2">
      {Array.from({ length: 5 }).map((_, index) => (
        <div
          key={index}
          className="rounded-2xl border border-slate-200 bg-white p-3"
        >
          <div className="h-4 w-32 animate-pulse rounded bg-slate-100" />
          <div className="mt-3 h-3 w-full animate-pulse rounded bg-slate-100" />
          <div className="mt-3 h-3 w-20 animate-pulse rounded bg-slate-100" />
        </div>
      ))}
    </div>
  )
}

function selectClass() {
  return 'rounded-xl border border-slate-200 bg-slate-50 px-2 py-2 text-xs text-slate-700 outline-none ring-blue-600 focus:ring-2'
}

function formatDate(value: string | null | undefined) {
  if (!value) return 'none'
  return new Intl.DateTimeFormat(undefined, {
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(new Date(value))
}

function formatWorkflowScheduleTrigger(schedule: ScheduleSummary) {
  if (schedule.trigger.kind === 'once') {
    return `once · ${describeBrowserDateTime(schedule.trigger.run_at)}`
  }
  return `${schedule.trigger.cron ?? schedule.cron.expr ?? 'cron'} · ${schedule.trigger.timezone}`
}

function mapWorkflowScheduleFireStatus(
  status: string,
  runStatus?: string | null,
) {
  if (runStatus === 'failed') return 'failed'
  if (runStatus === 'cancelled') return 'cancelled'
  if (runStatus === 'completed') return 'completed'
  if (runStatus === 'queued' || runStatus === 'running') return 'running'
  if (status === 'failed') return 'failed'
  if (status === 'pending' || status === 'submitted' || status === 'steered') {
    return 'running'
  }
  return 'completed'
}

function isActiveNodeStatus(status: string) {
  return status === 'queued' || status === 'running' || status === 'waiting'
}
