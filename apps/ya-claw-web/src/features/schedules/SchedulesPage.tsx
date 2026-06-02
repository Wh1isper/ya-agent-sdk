import { EyeOff, Play, Plus, RefreshCcw, Save, Trash2 } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { useForm } from 'react-hook-form'

import {
  useCreateScheduleMutation,
  useDeleteScheduleMutation,
  useScheduleFiresQuery,
  useSchedulesQuery,
  useTriggerScheduleMutation,
  useUpdateScheduleMutation,
} from '../../api/hooks'
import { EmptyState } from '../../components/EmptyState'
import { StatusBadge } from '../../components/StatusBadge'
import {
  describeScheduledAndLocalDateTime,
  describeBrowserDateTime,
  formatDateTimeInTimeZone,
  getBrowserTimeZone,
  getSupportedTimeZones,
  toZonedDatetimeLocalValue,
  zonedDatetimeLocalToIso,
} from '../../lib/timezone'
import { cn } from '../../lib/utils'
import type { ScheduleCreateRequest, ScheduleSummary } from '../../types'

type ScheduleFormValues = {
  name: string
  description: string
  prompt: string
  trigger_kind: 'cron' | 'once'
  cron: string
  run_at: string
  timezone: string
  enabled: boolean
  continue_current_session: boolean
  start_from_current_session: boolean
  steer_when_running: boolean
}

type ScheduleStatusFilter = ScheduleSummary['status'] | 'all'
type ScheduleEnabledFilter = 'all' | 'enabled' | 'disabled'
type ScheduleTriggerFilter = ScheduleSummary['trigger']['kind'] | 'all'

function createBlankSchedule(): ScheduleFormValues {
  return {
    name: '',
    description: '',
    prompt: '',
    trigger_kind: 'cron',
    cron: '0 9 * * *',
    run_at: '',
    timezone: getBrowserTimeZone(),
    enabled: true,
    continue_current_session: false,
    start_from_current_session: false,
    steer_when_running: false,
  }
}

const inputClass =
  'mt-2 w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm outline-none ring-blue-600 transition focus:bg-white focus:ring-2'
const textareaClass =
  'w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm outline-none ring-blue-600 transition focus:bg-white focus:ring-2'
const checkClass =
  'inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-slate-700'

export function SchedulesPage() {
  const [showHidden, setShowHidden] = useState(false)
  const schedules = useSchedulesQuery({
    includeDeleted: showHidden,
    includeWorkflow: false,
  })
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [search, setSearch] = useState('')
  const [statusFilter, setStatusFilter] = useState<ScheduleStatusFilter>('all')
  const [enabledFilter, setEnabledFilter] =
    useState<ScheduleEnabledFilter>('all')
  const [triggerFilter, setTriggerFilter] =
    useState<ScheduleTriggerFilter>('all')
  const scheduleRows = useMemo(
    () => schedules.data?.schedules ?? [],
    [schedules.data?.schedules],
  )
  const hiddenScheduleCount = useMemo(
    () =>
      scheduleRows.filter((schedule) => schedule.status === 'deleted').length,
    [scheduleRows],
  )
  const filteredSchedules = useMemo(() => {
    const needle = search.trim().toLowerCase()
    return scheduleRows.filter((schedule) => {
      if (statusFilter !== 'all' && schedule.status !== statusFilter) {
        return false
      }
      if (enabledFilter === 'enabled' && !schedule.enabled) return false
      if (enabledFilter === 'disabled' && schedule.enabled) return false
      if (triggerFilter !== 'all' && schedule.trigger.kind !== triggerFilter) {
        return false
      }
      if (!needle) return true
      return [
        schedule.id,
        schedule.name,
        schedule.description,
        schedule.prompt,
        schedule.status,
        schedule.enabled ? 'enabled' : 'disabled',
        schedule.trigger.kind,
        schedule.trigger.timezone,
        schedule.trigger.kind === 'cron' ? schedule.trigger.cron : null,
        schedule.trigger.kind === 'once' ? schedule.trigger.run_at : null,
        schedule.profile_name,
        schedule.owner_session_id,
        schedule.target_session_id,
        schedule.source_session_id,
      ]
        .filter(Boolean)
        .some((value) => String(value).toLowerCase().includes(needle))
    })
  }, [enabledFilter, scheduleRows, search, statusFilter, triggerFilter])
  const selectedSchedule = useMemo(
    () => scheduleRows.find((schedule) => schedule.id === selectedId) ?? null,
    [scheduleRows, selectedId],
  )

  useEffect(() => {
    if (!showHidden && statusFilter === 'deleted') {
      setStatusFilter('all')
    }
  }, [showHidden, statusFilter])

  useEffect(() => {
    if (!selectedId && scheduleRows[0]) {
      setSelectedId(scheduleRows[0].id)
      return
    }
    if (
      selectedId &&
      selectedId !== '__new__' &&
      !scheduleRows.some((schedule) => schedule.id === selectedId)
    ) {
      setSelectedId(scheduleRows[0]?.id ?? null)
    }
  }, [scheduleRows, selectedId])

  return (
    <div className="flex h-full min-h-0 bg-slate-100">
      <aside className="flex w-96 shrink-0 flex-col border-r border-slate-200 bg-white">
        <div className="border-b border-slate-200 p-4">
          <div className="flex items-center justify-between gap-2">
            <div>
              <p className="text-sm font-medium text-blue-600">Automation</p>
              <h1 className="mt-1 text-xl font-semibold tracking-tight text-slate-950">
                Schedules
              </h1>
            </div>
            <button
              type="button"
              className="inline-flex items-center gap-2 rounded-xl bg-blue-600 px-3 py-2 text-xs font-semibold text-white shadow-sm transition hover:bg-blue-700"
              onClick={() => setSelectedId('__new__')}
            >
              <Plus className="h-3.5 w-3.5" />
              New
            </button>
          </div>
          <div className="mt-4 space-y-2">
            <input
              className="w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm outline-none ring-blue-600 transition placeholder:text-slate-400 focus:bg-white focus:ring-2"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search schedules"
            />
            <div className="grid grid-cols-3 gap-2">
              <select
                className="rounded-xl border border-slate-200 bg-slate-50 px-2 py-2 text-xs text-slate-700 outline-none ring-blue-600 focus:ring-2"
                value={statusFilter}
                onChange={(event) =>
                  setStatusFilter(event.target.value as ScheduleStatusFilter)
                }
              >
                <option value="all">All status</option>
                <option value="active">Active</option>
                <option value="paused">Paused</option>
                <option value="completed">Completed</option>
                {showHidden ? <option value="deleted">Deleted</option> : null}
              </select>
              <select
                className="rounded-xl border border-slate-200 bg-slate-50 px-2 py-2 text-xs text-slate-700 outline-none ring-blue-600 focus:ring-2"
                value={enabledFilter}
                onChange={(event) =>
                  setEnabledFilter(event.target.value as ScheduleEnabledFilter)
                }
              >
                <option value="all">All state</option>
                <option value="enabled">Enabled</option>
                <option value="disabled">Disabled</option>
              </select>
              <select
                className="rounded-xl border border-slate-200 bg-slate-50 px-2 py-2 text-xs text-slate-700 outline-none ring-blue-600 focus:ring-2"
                value={triggerFilter}
                onChange={(event) =>
                  setTriggerFilter(event.target.value as ScheduleTriggerFilter)
                }
              >
                <option value="all">All trigger</option>
                <option value="cron">Cron</option>
                <option value="once">Once</option>
              </select>
            </div>
            <label className="flex items-center justify-between gap-2 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600">
              <span className="inline-flex items-center gap-2">
                <EyeOff className="h-3.5 w-3.5" />
                Show hidden
              </span>
              <input
                type="checkbox"
                checked={showHidden}
                onChange={(event) => setShowHidden(event.target.checked)}
              />
            </label>
            <p className="text-xs text-slate-400">
              Showing {filteredSchedules.length} of {scheduleRows.length}
              {showHidden ? ` · ${hiddenScheduleCount} hidden` : ''}
            </p>
          </div>
        </div>
        <div className="scrollbar-thin min-h-0 flex-1 overflow-auto p-3">
          {schedules.isLoading ? <ScheduleListSkeleton /> : null}
          {!schedules.isLoading && scheduleRows.length === 0 ? (
            <EmptyState
              title="No schedules"
              description="Create a schedule to run agent work later or on a recurrence."
            />
          ) : null}
          {!schedules.isLoading &&
          scheduleRows.length > 0 &&
          filteredSchedules.length === 0 ? (
            <EmptyState
              title="No matching schedules"
              description="Adjust the search or filters to find a schedule."
            />
          ) : null}
          <div className="space-y-2">
            {filteredSchedules.map((schedule) => (
              <ScheduleListItem
                key={schedule.id}
                schedule={schedule}
                active={selectedId === schedule.id}
                onClick={() => setSelectedId(schedule.id)}
              />
            ))}
          </div>
        </div>
      </aside>
      <main className="min-w-0 flex-1 overflow-auto p-6">
        <ScheduleEditor
          schedule={selectedId === '__new__' ? null : selectedSchedule}
          creating={selectedId === '__new__'}
        />
      </main>
    </div>
  )
}

function ScheduleListItem({
  schedule,
  active,
  onClick,
}: {
  schedule: ScheduleSummary
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
            {schedule.name}
          </p>
          <p className="mt-1 truncate mono text-xs text-slate-500">
            {formatTrigger(schedule)}
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-1.5">
          {isAutoHiddenSchedule(schedule) ? (
            <span className="rounded-full bg-slate-100 px-2 py-1 text-[10px] font-semibold uppercase tracking-wide text-slate-500">
              Hidden
            </span>
          ) : null}
          <StatusBadge status={schedule.status} />
        </div>
      </div>
      <p className="mt-2 line-clamp-2 text-xs text-slate-500">
        {schedule.prompt}
      </p>
      <p className="mt-2 line-clamp-2 text-xs text-slate-400">
        Next:{' '}
        {describeScheduledAndLocalDateTime(
          schedule.trigger.next_fire_at,
          schedule.trigger.timezone,
        )}
      </p>
    </button>
  )
}

function ScheduleEditor({
  schedule,
  creating,
}: {
  schedule: ScheduleSummary | null
  creating: boolean
}) {
  const createSchedule = useCreateScheduleMutation()
  const updateSchedule = useUpdateScheduleMutation()
  const deleteSchedule = useDeleteScheduleMutation()
  const triggerSchedule = useTriggerScheduleMutation()
  const isDeleted = schedule?.status === 'deleted'
  const fires = useScheduleFiresQuery(schedule?.id ?? null)
  const form = useForm<ScheduleFormValues>({
    defaultValues: createBlankSchedule(),
  })
  const triggerKind = form.watch('trigger_kind')
  const timezone = form.watch('timezone')
  const supportedTimeZones = useMemo(() => getSupportedTimeZones(), [])

  useEffect(() => {
    if (schedule) {
      form.reset({
        name: schedule.name,
        description: schedule.description ?? '',
        prompt: schedule.prompt,
        trigger_kind: schedule.trigger.kind,
        cron:
          schedule.trigger.kind === 'cron' ? (schedule.trigger.cron ?? '') : '',
        run_at:
          schedule.trigger.kind === 'once'
            ? toZonedDatetimeLocalValue(
                schedule.trigger.run_at,
                schedule.trigger.timezone,
              )
            : '',
        timezone: schedule.trigger.timezone,
        enabled: schedule.enabled,
        continue_current_session: schedule.mode.continue_current_session,
        start_from_current_session: schedule.mode.start_from_current_session,
        steer_when_running: schedule.mode.steer_when_running,
      })
    } else {
      form.reset(createBlankSchedule())
    }
  }, [form, schedule])

  const onSubmit = form.handleSubmit(async (values) => {
    const payload: ScheduleCreateRequest = {
      name: values.name,
      description: values.description || null,
      prompt: values.prompt,
      trigger_kind: values.trigger_kind,
      cron: values.trigger_kind === 'cron' ? values.cron : null,
      run_at:
        values.trigger_kind === 'once'
          ? zonedDatetimeLocalToIso(values.run_at, values.timezone)
          : null,
      timezone: values.timezone,
      enabled: values.enabled,
      continue_current_session: values.continue_current_session,
      start_from_current_session: values.start_from_current_session,
      steer_when_running: values.steer_when_running,
      owner_kind: 'user',
    }
    if (creating) {
      await createSchedule.mutateAsync(payload)
    } else if (schedule) {
      await updateSchedule.mutateAsync({ scheduleId: schedule.id, payload })
    }
  })

  if (!creating && !schedule) {
    return (
      <EmptyState
        title="Select a schedule"
        description="Choose a schedule or create a new one."
      />
    )
  }

  return (
    <div className="mx-auto max-w-5xl space-y-6">
      <div className="flex items-center justify-between gap-4">
        <div>
          <p className="text-sm font-medium text-blue-600">
            {creating ? 'New schedule' : 'Schedule'}
          </p>
          <h2 className="mt-1 text-2xl font-semibold tracking-tight text-slate-950">
            {creating ? 'Create schedule' : schedule?.name}
          </h2>
        </div>
        <div className="flex gap-2">
          {schedule && schedule.status !== 'deleted' ? (
            <button
              type="button"
              className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 shadow-sm transition hover:bg-slate-50"
              onClick={() =>
                triggerSchedule.mutate({ scheduleId: schedule.id })
              }
            >
              <Play className="h-4 w-4" />
              Trigger
            </button>
          ) : null}
          {schedule && schedule.status !== 'deleted' ? (
            <button
              type="button"
              className="inline-flex items-center gap-2 rounded-xl border border-rose-200 bg-white px-3 py-2 text-sm font-medium text-rose-700 shadow-sm transition hover:bg-rose-50"
              onClick={() => deleteSchedule.mutate(schedule.id)}
            >
              <Trash2 className="h-4 w-4" />
              Delete
            </button>
          ) : null}
        </div>
      </div>

      {schedule && isAutoHiddenSchedule(schedule) ? (
        <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4 text-sm text-slate-600">
          This one-time schedule was hidden automatically after it expired.
        </div>
      ) : null}

      <form
        className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"
        onSubmit={onSubmit}
      >
        <fieldset disabled={isDeleted} className="contents">
          <div className="grid grid-cols-2 gap-4">
            <Field label="Name">
              <input
                className={inputClass}
                {...form.register('name', { required: true })}
              />
            </Field>
            <Field label="Trigger">
              <select className={inputClass} {...form.register('trigger_kind')}>
                <option value="cron">Recurring cron</option>
                <option value="once">One-time</option>
              </select>
            </Field>
            {triggerKind === 'cron' ? (
              <Field
                label="Cron"
                hint={`Evaluated in ${timezone || getBrowserTimeZone()}. Example: 0 9 * * * runs at 09:00 in that timezone.`}
              >
                <input
                  className={`${inputClass} mono`}
                  {...form.register('cron', {
                    required: triggerKind === 'cron',
                  })}
                />
              </Field>
            ) : (
              <Field
                label="Run at"
                hint={`Interpreted as wall-clock time in ${timezone || getBrowserTimeZone()}.`}
              >
                <input
                  type="datetime-local"
                  className={inputClass}
                  {...form.register('run_at', {
                    required: triggerKind === 'once',
                  })}
                />
              </Field>
            )}
            <Field
              label={triggerKind === 'cron' ? 'Cron timezone' : 'Run timezone'}
              hint="Changing this timezone keeps the same wall-clock input and updates the stored fire time."
            >
              {supportedTimeZones.length > 0 ? (
                <select
                  className={inputClass}
                  {...form.register('timezone', { required: true })}
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
                  {...form.register('timezone', { required: true })}
                />
              )}
            </Field>
            <Field label="Description">
              <input className={inputClass} {...form.register('description')} />
            </Field>
          </div>
          <Field label="Prompt">
            <textarea
              className={`${textareaClass} mt-2 min-h-40`}
              {...form.register('prompt', { required: true })}
            />
          </Field>
          <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
            <label className={checkClass}>
              <input type="checkbox" {...form.register('enabled')} /> Enabled
            </label>
            <label className={checkClass}>
              <input
                type="checkbox"
                {...form.register('continue_current_session')}
              />{' '}
              Continue current session
            </label>
            <label className={checkClass}>
              <input
                type="checkbox"
                {...form.register('start_from_current_session')}
              />{' '}
              Start from current session
            </label>
            <label className={checkClass}>
              <input type="checkbox" {...form.register('steer_when_running')} />{' '}
              Steer when running
            </label>
          </div>
          <div className="mt-5 flex justify-end">
            <button
              type="submit"
              disabled={isDeleted}
              className="inline-flex items-center gap-2 rounded-xl bg-blue-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-slate-300"
            >
              <Save className="h-4 w-4" />
              Save
            </button>
          </div>
        </fieldset>
      </form>

      {schedule ? (
        <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-950">
              Recent fires
            </h3>
            <RefreshCcw className="h-4 w-4 text-slate-400" />
          </div>
          <div className="mt-4 space-y-2">
            {(fires.data?.fires ?? []).map((fire) => (
              <div
                key={fire.id}
                className="rounded-xl border border-slate-100 p-3 text-sm"
              >
                <div className="flex items-center justify-between">
                  <span className="mono text-xs text-slate-500">
                    {fire.id.slice(0, 10)}
                  </span>
                  <StatusBadge
                    status={mapFireStatus(fire.status, fire.run_status)}
                  />
                </div>
                <p className="mt-2 text-slate-600">{fire.input_preview}</p>
                <p className="mt-1 text-xs text-slate-400">
                  Run{' '}
                  {fire.run_id?.slice(0, 10) ??
                    fire.workflow_run_id?.slice(0, 10) ??
                    'none'}{' '}
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
        </section>
      ) : null}
    </div>
  )
}

function Field({
  label,
  hint,
  children,
}: {
  label: string
  hint?: string
  children: React.ReactNode
}) {
  return (
    <label className="block text-sm font-medium text-slate-700">
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

function ScheduleListSkeleton() {
  return (
    <div className="space-y-2">
      {Array.from({ length: 4 }).map((_, index) => (
        <div
          key={index}
          className="h-24 animate-pulse rounded-2xl bg-slate-100"
        />
      ))}
    </div>
  )
}

function isAutoHiddenSchedule(schedule: ScheduleSummary) {
  return schedule.status === 'deleted' && schedule.metadata.auto_hidden === true
}

function formatTrigger(schedule: ScheduleSummary) {
  if (schedule.trigger.kind === 'once') {
    return `once · ${formatDateTimeInTimeZone(
      schedule.trigger.run_at,
      schedule.trigger.timezone,
    )} ${schedule.trigger.timezone}`
  }
  return `${schedule.trigger.cron ?? schedule.cron.expr ?? 'cron'} · runs in ${schedule.trigger.timezone}`
}

function mapFireStatus(status: string, runStatus?: string | null) {
  if (runStatus === 'failed') return 'failed'
  if (runStatus === 'cancelled') return 'cancelled'
  if (runStatus === 'completed') return 'completed'
  if (runStatus === 'queued' || runStatus === 'running') return 'running'
  if (status === 'failed') return 'failed'
  if (status === 'pending' || status === 'submitted' || status === 'steered')
    return 'running'
  return 'completed'
}
