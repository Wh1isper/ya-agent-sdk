import { Link, useBlocker, useRouterState } from '@tanstack/react-router'
import {
  ChevronLeft,
  EyeOff,
  Play,
  Plus,
  RefreshCcw,
  Save,
  Trash2,
} from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { useForm } from 'react-hook-form'

import {
  useCreateScheduleMutation,
  useDeleteScheduleMutation,
  useScheduleFiresQuery,
  useScheduleQuery,
  useSchedulesQuery,
  useTriggerScheduleMutation,
  useUpdateScheduleMutation,
} from '../../api/hooks'
import { EmptyState } from '../../components/EmptyState'
import { StatusBadge } from '../../components/StatusBadge'
import { ConfirmDialog, QueryError, QuerySkeleton } from '../../components/ui'
import {
  describeScheduledAndLocalDateTime,
  describeBrowserDateTime,
  formatDateTimeInTimeZone,
  getBrowserTimeZone,
  getSupportedTimeZones,
  toZonedDatetimeLocalValue,
  zonedDatetimeLocalToIso,
} from '../../lib/timezone'
import { hasRegisteredAppNavigation, navigateApp } from '../../app/navigation'
import { isNewerApiTimestamp } from '../../lib/date'
import { safeDecodePathSegment } from '../../lib/urlState'
import { cn } from '../../lib/utils'
import type {
  ScheduleCreateRequest,
  ScheduleFireSummary,
  ScheduleSummary,
} from '../../types'
import {
  AUTOMATION_LIST_LIMIT,
  mayHaveMoreAutomationRows,
} from '../automation/listLimit'
import {
  buildSimpleCron,
  describeSimpleRecurrence,
  nextSimpleOccurrences,
  parseSimpleCron,
  type SimpleFrequency,
} from './scheduleRecurrence'

type ScheduleFormValues = {
  name: string
  description: string
  prompt: string
  trigger_kind: 'cron' | 'once'
  frequency: SimpleFrequency
  time: string
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
    frequency: 'daily',
    time: '09:00',
    cron: '0 9 * * *',
    run_at: '',
    timezone: getBrowserTimeZone(),
    enabled: true,
    continue_current_session: false,
    start_from_current_session: false,
    steer_when_running: false,
  }
}

function scheduleToFormValues(schedule: ScheduleSummary): ScheduleFormValues {
  const parsedRecurrence = parseSimpleCron(
    schedule.trigger.kind === 'cron' ? (schedule.trigger.cron ?? '') : '',
  )
  return {
    name: schedule.name,
    description: schedule.description ?? '',
    prompt: schedule.prompt,
    trigger_kind: schedule.trigger.kind,
    frequency: parsedRecurrence.frequency,
    time: parsedRecurrence.time,
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
    continue_current_session: schedule.mode.continue_current_session,
    start_from_current_session: schedule.mode.start_from_current_session,
    steer_when_running: schedule.mode.steer_when_running,
  }
}

const inputClass =
  'mt-2 w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm outline-none ring-blue-600 transition focus:bg-white focus:ring-2'
const textareaClass =
  'w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm outline-none ring-blue-600 transition focus:bg-white focus:ring-2'
const checkClass =
  'inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-slate-700'

function scheduleIdFromPath(pathname: string): string | null {
  const prefix = '/automation/schedules/'
  if (!pathname.startsWith(prefix)) return null
  const segment = pathname.slice(prefix.length)
  if (!segment || segment.includes('/')) return null
  return segment === 'new' ? '__new__' : safeDecodePathSegment(segment)
}

export function SchedulesPage() {
  const pathname = useRouterState({
    select: (state) => state.location.pathname,
  })
  const routeSelectedId = scheduleIdFromPath(pathname)
  const routeDetailId =
    routeSelectedId && routeSelectedId !== '__new__' ? routeSelectedId : null
  const routeSchedule = useScheduleQuery(routeDetailId)
  const [showHidden, setShowHidden] = useState(false)
  const schedules = useSchedulesQuery({
    includeDeleted: showHidden,
    includeWorkflow: false,
    limit: AUTOMATION_LIST_LIMIT,
  })
  const [selectedId, setSelectedId] = useState<string | null>(routeSelectedId)
  const mobileDetailOpen = routeSelectedId !== null
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
    () =>
      (routeDetailId && routeSchedule.data?.id === routeDetailId
        ? routeSchedule.data
        : null) ??
      scheduleRows.find((schedule) => schedule.id === selectedId) ??
      null,
    [routeDetailId, routeSchedule.data, scheduleRows, selectedId],
  )

  useEffect(() => {
    if (!showHidden && statusFilter === 'deleted') {
      setStatusFilter('all')
    }
  }, [showHidden, statusFilter])

  function applySelection(nextId: string) {
    if (!hasRegisteredAppNavigation()) setSelectedId(nextId)
    navigateApp(
      nextId === '__new__'
        ? '/automation/schedules/new'
        : `/automation/schedules/${encodeURIComponent(nextId)}`,
    )
  }

  function requestSelection(nextId: string) {
    if (nextId === selectedId && mobileDetailOpen) return
    applySelection(nextId)
  }

  useEffect(() => {
    if (routeSelectedId !== null) {
      if (routeSelectedId !== selectedId) setSelectedId(routeSelectedId)
      return
    }
    if (selectedId === '__new__') {
      setSelectedId(scheduleRows[0]?.id ?? null)
    }
  }, [routeSelectedId, scheduleRows, selectedId])

  useEffect(() => {
    if (!selectedId && scheduleRows[0]) {
      setSelectedId(scheduleRows[0].id)
      return
    }
    if (
      routeSelectedId === null &&
      selectedId &&
      selectedId !== '__new__' &&
      !scheduleRows.some((schedule) => schedule.id === selectedId)
    ) {
      setSelectedId(scheduleRows[0]?.id ?? null)
    }
  }, [routeSelectedId, scheduleRows, selectedId])

  return (
    <div className="flex h-full min-h-0 flex-col overflow-auto bg-slate-100 lg:flex-row lg:overflow-hidden">
      <h1 className="sr-only">Schedules</h1>
      <aside
        aria-label="Schedule list"
        className={cn(
          'max-h-none w-full shrink-0 flex-col border-b border-slate-200 bg-white lg:flex lg:max-h-none lg:w-96 lg:border-b-0 lg:border-r',
          mobileDetailOpen ? 'hidden' : 'flex',
        )}
      >
        <div className="border-b border-slate-200 p-4">
          <div className="flex items-center justify-between gap-2">
            <div>
              <p className="text-sm font-medium text-blue-600">Automation</p>
              <h2 className="mt-1 text-xl font-semibold tracking-tight text-slate-950">
                Schedules
              </h2>
            </div>
            <button
              type="button"
              className="inline-flex items-center gap-2 rounded-xl bg-blue-600 px-3 py-2 text-xs font-semibold text-white shadow-sm transition hover:bg-blue-700"
              onClick={() => requestSelection('__new__')}
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
              aria-label="Search schedules"
            />
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
              <select
                className="rounded-xl border border-slate-200 bg-slate-50 px-2 py-2 text-xs text-slate-700 outline-none ring-blue-600 focus:ring-2"
                value={statusFilter}
                aria-label="Filter schedules by status"
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
                aria-label="Filter schedules by enabled state"
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
                aria-label="Filter schedules by trigger type"
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
            {mayHaveMoreAutomationRows(scheduleRows.length) ? (
              <p className="text-xs font-medium text-amber-700" role="status">
                Showing the first {AUTOMATION_LIST_LIMIT} schedules. Narrow the
                filters if the schedule you need is not listed.
              </p>
            ) : null}
          </div>
        </div>
        <div className="scrollbar-thin min-h-0 flex-1 overflow-auto p-3">
          {schedules.isLoading ? <ScheduleListSkeleton /> : null}
          {!schedules.isLoading && schedules.isError ? (
            <QueryError
              title="Schedules could not be loaded"
              error={schedules.error}
              onRetry={() => void schedules.refetch()}
            />
          ) : null}
          {!schedules.isLoading &&
          !schedules.isError &&
          scheduleRows.length === 0 ? (
            <EmptyState
              title="No schedules"
              description="Create a schedule to run agent work later or on a recurrence."
              headingLevel={2}
              action={
                <button
                  type="button"
                  className="inline-flex items-center gap-2 rounded-xl bg-blue-600 px-3 py-2 text-sm font-semibold text-white"
                  onClick={() => requestSelection('__new__')}
                >
                  <Plus className="h-4 w-4" /> Create schedule
                </button>
              }
            />
          ) : null}
          {!schedules.isLoading &&
          !schedules.isError &&
          scheduleRows.length > 0 &&
          filteredSchedules.length === 0 ? (
            <EmptyState
              title="No matching schedules"
              description="Adjust the search or filters to find a schedule."
              action={
                <button
                  type="button"
                  className="text-sm font-semibold text-blue-700"
                  onClick={() => {
                    setSearch('')
                    setStatusFilter('all')
                    setEnabledFilter('all')
                    setTriggerFilter('all')
                  }}
                >
                  Clear filters
                </button>
              }
            />
          ) : null}
          <div className="space-y-2">
            {!schedules.isError
              ? filteredSchedules.map((schedule) => (
                  <ScheduleListItem
                    key={schedule.id}
                    schedule={schedule}
                    active={selectedId === schedule.id}
                    onClick={() => requestSelection(schedule.id)}
                  />
                ))
              : null}
          </div>
        </div>
      </aside>
      <section
        aria-label="Schedule editor"
        className={cn(
          'min-h-0 w-full min-w-0 flex-1 overflow-auto p-4 lg:block lg:p-6',
          mobileDetailOpen ? 'block' : 'hidden',
        )}
      >
        <button
          type="button"
          className="mb-4 inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 lg:hidden"
          onClick={() => navigateApp('/automation/schedules', true)}
        >
          <ChevronLeft className="h-4 w-4" aria-hidden />
          Back to schedules
        </button>
        {routeDetailId && routeSchedule.isError ? (
          <QueryError
            title="Could not load this schedule"
            error={routeSchedule.error}
            onRetry={() => void routeSchedule.refetch()}
          />
        ) : routeDetailId && !selectedSchedule ? (
          <QuerySkeleton rows={4} />
        ) : (
          <ScheduleEditor
            schedule={selectedId === '__new__' ? null : selectedSchedule}
            creating={selectedId === '__new__'}
            onCreated={applySelection}
          />
        )}
      </section>
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
  onCreated,
}: {
  schedule: ScheduleSummary | null
  creating: boolean
  onCreated: (scheduleId: string) => void
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
  const [remoteUpdateAvailable, setRemoteUpdateAvailable] = useState(false)
  const [remoteReloadConfirmOpen, setRemoteReloadConfirmOpen] = useState(false)
  const [operationError, setOperationError] = useState<string | null>(null)
  const loadedEditorKeyRef = useRef<string | null>(null)
  const loadedVersionRef = useRef<string | null>(null)
  const pendingRemoteScheduleRef = useRef<ScheduleSummary | null>(null)
  const operationGenerationRef = useRef(0)
  const isDirty = form.formState.isDirty
  const isSaving = createSchedule.isPending || updateSchedule.isPending
  const navigationBlocker = useBlocker({
    shouldBlockFn: () => isDirty,
    enableBeforeUnload: isDirty,
    disabled: !isDirty,
    withResolver: true,
  })
  const triggerKind = form.watch('trigger_kind')
  const frequency = form.watch('frequency')
  const time = form.watch('time')
  const timezone = form.watch('timezone')
  const cron = form.watch('cron')
  const runAt = form.watch('run_at')
  const continueCurrentSession = form.watch('continue_current_session')
  const startFromCurrentSession = form.watch('start_from_current_session')
  const steerWhenRunning = form.watch('steer_when_running')
  const supportedTimeZones = useMemo(() => getSupportedTimeZones(), [])

  useEffect(() => {
    operationGenerationRef.current += 1
  }, [creating, schedule?.id])

  useEffect(
    () => () => {
      operationGenerationRef.current += 1
    },
    [],
  )

  useEffect(() => {
    const editorKey = creating ? '__new__' : (schedule?.id ?? null)
    if (!editorKey) return
    const version = creating ? null : (schedule?.updated_at ?? null)
    const changingSchedule = loadedEditorKeyRef.current !== editorKey
    const versionChanged = loadedVersionRef.current !== version
    if (!changingSchedule && !versionChanged) return
    const latestKnownVersion =
      pendingRemoteScheduleRef.current?.updated_at ?? loadedVersionRef.current
    if (
      !changingSchedule &&
      versionChanged &&
      !isNewerApiTimestamp(version, latestKnownVersion)
    ) {
      return
    }
    if (!changingSchedule && isDirty && schedule) {
      pendingRemoteScheduleRef.current = schedule
      setRemoteUpdateAvailable(true)
      return
    }

    loadedEditorKeyRef.current = editorKey
    loadedVersionRef.current = version
    pendingRemoteScheduleRef.current = null
    setRemoteUpdateAvailable(false)
    form.reset(
      schedule ? scheduleToFormValues(schedule) : createBlankSchedule(),
    )
  }, [creating, form, isDirty, schedule])

  useEffect(() => {
    if (triggerKind !== 'cron' || frequency === 'custom') return
    const nextCron = buildSimpleCron(frequency, time)
    if (nextCron && nextCron !== form.getValues('cron')) {
      form.setValue('cron', nextCron, { shouldDirty: true })
    }
  }, [form, frequency, time, triggerKind])

  const futureOccurrences = useMemo(
    () =>
      triggerKind === 'cron'
        ? nextSimpleOccurrences(cron, timezone || getBrowserTimeZone())
        : null,
    [cron, timezone, triggerKind],
  )

  const onSubmit = form.handleSubmit(async (values) => {
    const operationGeneration = ++operationGenerationRef.current
    setOperationError(null)
    try {
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
        const created = await createSchedule.mutateAsync(payload)
        if (operationGenerationRef.current !== operationGeneration) return
        loadedEditorKeyRef.current = created.id
        loadedVersionRef.current = created.updated_at
        pendingRemoteScheduleRef.current = null
        setRemoteUpdateAvailable(false)
        form.reset(scheduleToFormValues(created))
        onCreated(created.id)
      } else if (schedule) {
        const updated = await updateSchedule.mutateAsync({
          scheduleId: schedule.id,
          payload,
        })
        if (operationGenerationRef.current !== operationGeneration) return
        loadedEditorKeyRef.current = updated.id
        loadedVersionRef.current = updated.updated_at
        pendingRemoteScheduleRef.current = null
        setRemoteUpdateAvailable(false)
        form.reset(scheduleToFormValues(updated))
      }
    } catch (error) {
      if (operationGenerationRef.current === operationGeneration) {
        setOperationError(
          error instanceof Error
            ? error.message
            : 'Could not save the schedule.',
        )
      }
    }
  })

  async function triggerNow() {
    if (!schedule) return
    setOperationError(null)
    try {
      await triggerSchedule.mutateAsync({ scheduleId: schedule.id })
    } catch (error) {
      setOperationError(
        error instanceof Error
          ? error.message
          : 'Could not trigger the schedule.',
      )
    }
  }

  if (!creating && !schedule) {
    return (
      <EmptyState
        title="Select a schedule"
        description="Choose a schedule or create a new one."
      />
    )
  }

  return (
    <>
      <div className="mx-auto max-w-5xl space-y-6">
        <div className="flex min-w-0 flex-col items-start justify-between gap-4 sm:flex-row">
          <div className="min-w-0">
            <p className="text-sm font-medium text-blue-600">
              {creating ? 'New schedule' : 'Schedule'}
            </p>
            <h2 className="mt-1 break-words text-2xl font-semibold tracking-tight text-slate-950">
              {creating ? 'Create schedule' : schedule?.name}
            </h2>
          </div>
          <div className="flex w-full flex-wrap gap-2 sm:w-auto">
            {schedule && schedule.status !== 'deleted' ? (
              <button
                type="button"
                className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 shadow-sm transition hover:bg-slate-50"
                disabled={triggerSchedule.isPending}
                onClick={() => void triggerNow()}
              >
                <Play className="h-4 w-4" />
                {triggerSchedule.isPending ? 'Triggering…' : 'Trigger'}
              </button>
            ) : null}
            {schedule && schedule.status !== 'deleted' ? (
              <ConfirmDialog
                title={`Delete ${schedule.name}?`}
                description={
                  isDirty
                    ? 'Your unsaved edits will be permanently discarded. This schedule will stop firing and move to the hidden schedule history.'
                    : 'This schedule will stop firing and move to the hidden schedule history.'
                }
                confirmLabel={
                  isDirty ? 'Discard edits and delete' : 'Delete schedule'
                }
                danger
                pending={deleteSchedule.isPending}
                onConfirm={async () => {
                  await deleteSchedule.mutateAsync(schedule.id)
                  form.reset(form.getValues())
                }}
                trigger={
                  <button
                    type="button"
                    className="inline-flex items-center gap-2 rounded-xl border border-rose-200 bg-white px-3 py-2 text-sm font-medium text-rose-700 shadow-sm transition hover:bg-rose-50"
                  >
                    <Trash2 className="h-4 w-4" />
                    Delete
                  </button>
                }
              />
            ) : null}
          </div>
        </div>

        {remoteUpdateAvailable && pendingRemoteScheduleRef.current ? (
          <div
            className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900"
            role="status"
          >
            <span>
              A newer server version is available. Your unsaved changes are
              preserved.
            </span>
            <button
              type="button"
              className="font-semibold underline underline-offset-2"
              onClick={() => setRemoteReloadConfirmOpen(true)}
            >
              Load server version
            </button>
          </div>
        ) : null}
        <ConfirmDialog
          open={remoteReloadConfirmOpen}
          onOpenChange={setRemoteReloadConfirmOpen}
          title="Discard unsaved schedule changes and load the server version?"
          description="Loading the server version permanently replaces the edits in this form."
          confirmLabel="Discard changes and load"
          danger
          onConfirm={() => {
            const candidate = pendingRemoteScheduleRef.current
            if (!candidate) return
            form.reset(scheduleToFormValues(candidate))
            loadedEditorKeyRef.current = candidate.id
            loadedVersionRef.current = candidate.updated_at
            pendingRemoteScheduleRef.current = null
            setRemoteUpdateAvailable(false)
          }}
        />

        {operationError ? (
          <div
            className="rounded-xl border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700"
            role="alert"
          >
            {operationError}
          </div>
        ) : null}

        {schedule && isAutoHiddenSchedule(schedule) ? (
          <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4 text-sm text-slate-600">
            This one-time schedule was hidden automatically after it expired.
          </div>
        ) : null}

        <form
          className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"
          onSubmit={onSubmit}
        >
          <fieldset disabled={isDeleted || isSaving} className="contents">
            <div className="mb-5 rounded-xl border border-blue-100 bg-blue-50 p-4">
              <p className="text-xs font-semibold uppercase tracking-wide text-blue-600">
                Live schedule
              </p>
              <p className="mt-1 text-sm font-semibold text-blue-950">
                {triggerKind === 'cron'
                  ? describeSimpleRecurrence(frequency, time, timezone)
                  : describeScheduleInput(triggerKind, cron, runAt, timezone)}
              </p>
              {triggerKind === 'cron' ? (
                <div className="mt-3">
                  <p className="text-xs font-semibold text-blue-900">
                    Next three runs
                  </p>
                  {futureOccurrences ? (
                    <ol className="mt-1 grid gap-1 text-xs text-blue-800 sm:grid-cols-3">
                      {futureOccurrences.map((occurrence) => (
                        <li key={occurrence.toISOString()}>
                          {formatDateTimeInTimeZone(
                            occurrence.toISOString(),
                            timezone,
                          )}
                        </li>
                      ))}
                    </ol>
                  ) : (
                    <p className="mt-1 text-xs text-blue-800">
                      Preview is unavailable for this custom cron. The runtime
                      will evaluate it in {timezone}; save it to confirm the
                      next fire.
                    </p>
                  )}
                </div>
              ) : null}
            </div>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <Field label="Name" error={form.formState.errors.name?.message}>
                <input
                  className={inputClass}
                  aria-invalid={Boolean(form.formState.errors.name)}
                  {...form.register('name', {
                    required: 'Enter a schedule name.',
                  })}
                />
              </Field>
              <Field label="Trigger">
                <select
                  className={inputClass}
                  {...form.register('trigger_kind')}
                >
                  <option value="cron">Recurring cron</option>
                  <option value="once">One-time</option>
                </select>
              </Field>
              {triggerKind === 'cron' ? (
                <>
                  <Field label="Frequency">
                    <select
                      className={inputClass}
                      {...form.register('frequency')}
                    >
                      <option value="hourly">Every hour</option>
                      <option value="daily">Every day</option>
                      <option value="weekdays">Weekdays</option>
                      <option value="weekly">Every Monday</option>
                      <option value="custom">Custom cron</option>
                    </select>
                  </Field>
                  {frequency !== 'hourly' && frequency !== 'custom' ? (
                    <Field
                      label="Time"
                      hint="Wall-clock time in the selected timezone."
                      error={form.formState.errors.time?.message}
                    >
                      <input
                        type="time"
                        className={inputClass}
                        aria-invalid={Boolean(form.formState.errors.time)}
                        {...form.register('time', {
                          required: 'Choose a time.',
                        })}
                      />
                    </Field>
                  ) : null}
                </>
              ) : (
                <Field
                  label="Run at"
                  hint={`Interpreted as wall-clock time in ${timezone || getBrowserTimeZone()}.`}
                  error={form.formState.errors.run_at?.message}
                >
                  <input
                    type="datetime-local"
                    className={inputClass}
                    aria-invalid={Boolean(form.formState.errors.run_at)}
                    {...form.register('run_at', {
                      required:
                        triggerKind === 'once'
                          ? 'Choose a run date and time.'
                          : false,
                      validate: (value) => {
                        if (triggerKind !== 'once' || !value) return true
                        try {
                          zonedDatetimeLocalToIso(
                            value,
                            form.getValues('timezone'),
                          )
                          return true
                        } catch (error) {
                          return error instanceof Error
                            ? error.message
                            : 'This local time is invalid in the selected timezone.'
                        }
                      },
                    })}
                  />
                </Field>
              )}
              <Field
                label={
                  triggerKind === 'cron' ? 'Cron timezone' : 'Run timezone'
                }
                hint="Changing this timezone keeps the same wall-clock input and updates the stored fire time."
                error={form.formState.errors.timezone?.message}
              >
                {supportedTimeZones.length > 0 ? (
                  <select
                    className={inputClass}
                    aria-invalid={Boolean(form.formState.errors.timezone)}
                    {...form.register('timezone', {
                      required: 'Choose a timezone.',
                    })}
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
                    aria-invalid={Boolean(form.formState.errors.timezone)}
                    {...form.register('timezone', {
                      required: 'Choose a timezone.',
                    })}
                  />
                )}
              </Field>
              <Field label="Description">
                <input
                  className={inputClass}
                  {...form.register('description')}
                />
              </Field>
            </div>
            {triggerKind === 'cron' ? (
              <details className="mt-4 rounded-xl border border-slate-200 bg-slate-50 p-3">
                <summary className="cursor-pointer text-sm font-semibold text-slate-700">
                  Advanced · raw cron
                </summary>
                <Field
                  label="Cron expression"
                  hint={`Evaluated in ${timezone || getBrowserTimeZone()}. Custom expressions use runtime validation and may not have a browser preview.`}
                  error={form.formState.errors.cron?.message}
                >
                  <input
                    className={`${inputClass} mono`}
                    aria-invalid={Boolean(form.formState.errors.cron)}
                    {...form.register('cron', {
                      required: 'Enter a cron expression.',
                    })}
                    onChange={(event) => {
                      form.register('cron').onChange(event)
                      const parsed = parseSimpleCron(event.target.value)
                      form.setValue('frequency', parsed.frequency)
                      if (parsed.frequency !== 'custom') {
                        form.setValue('time', parsed.time)
                      }
                    }}
                  />
                </Field>
              </details>
            ) : null}
            <Field label="Prompt" error={form.formState.errors.prompt?.message}>
              <textarea
                className={`${textareaClass} mt-2 min-h-40`}
                aria-invalid={Boolean(form.formState.errors.prompt)}
                {...form.register('prompt', {
                  required: 'Enter the prompt to run.',
                })}
              />
            </Field>
            <div className="mt-4 rounded-xl border border-slate-200 bg-slate-50 p-3 text-sm text-slate-600">
              {describeTargetBehavior(
                continueCurrentSession,
                startFromCurrentSession,
                steerWhenRunning,
              )}
            </div>
            <div className="mt-4 grid grid-cols-1 gap-3 text-sm sm:grid-cols-2">
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
                <input
                  type="checkbox"
                  {...form.register('steer_when_running')}
                />{' '}
                Steer when running
              </label>
            </div>
            <div className="mt-5 flex justify-end">
              <button
                type="submit"
                disabled={isDeleted || isSaving}
                className="inline-flex items-center gap-2 rounded-xl bg-blue-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-slate-300"
              >
                <Save className="h-4 w-4" />
                {isSaving ? 'Saving…' : 'Save'}
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
              <button
                type="button"
                aria-label="Refresh recent fires"
                className="rounded-lg p-2 text-slate-500 hover:bg-slate-100"
                onClick={() => void fires.refetch()}
              >
                <RefreshCcw className="h-4 w-4" />
              </button>
            </div>
            <div className="mt-4 space-y-2">
              {fires.isLoading ? (
                <div className="h-20 animate-pulse rounded-xl bg-slate-100" />
              ) : null}
              {fires.isError ? (
                <QueryError
                  title="Recent fires could not be loaded"
                  error={fires.error}
                  onRetry={() => void fires.refetch()}
                />
              ) : null}
              {!fires.isLoading &&
              !fires.isError &&
              (fires.data?.fires.length ?? 0) === 0 ? (
                <EmptyState
                  title="No runs yet"
                  description="Trigger this schedule now or wait for its next run."
                  action={
                    schedule.status !== 'deleted' ? (
                      <button
                        type="button"
                        className="inline-flex items-center gap-2 rounded-xl bg-blue-600 px-3 py-2 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:bg-slate-300"
                        disabled={triggerSchedule.isPending}
                        onClick={() => void triggerNow()}
                      >
                        <Play className="h-4 w-4" />{' '}
                        {triggerSchedule.isPending ? 'Starting…' : 'Run now'}
                      </button>
                    ) : null
                  }
                  className="min-h-40"
                />
              ) : null}
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
                  <ScheduleFireActivityLink fire={fire} />
                </div>
              ))}
            </div>
          </section>
        ) : null}
      </div>
      <ConfirmDialog
        open={navigationBlocker.status === 'blocked'}
        onOpenChange={(open) => {
          if (!open && navigationBlocker.status === 'blocked') {
            navigationBlocker.reset()
          }
        }}
        title="Discard unsaved schedule changes?"
        description="Your edits will be lost if you leave this page."
        confirmLabel="Discard and leave"
        danger
        onConfirm={() => {
          if (navigationBlocker.status === 'blocked') {
            navigationBlocker.proceed()
          }
        }}
      />
    </>
  )
}

function ScheduleFireActivityLink({ fire }: { fire: ScheduleFireSummary }) {
  const sessionId =
    fire.target_session_id ??
    fire.created_session_id ??
    fire.source_session_id ??
    null
  const runId = fire.run_id ?? fire.active_run_id ?? null
  if (!sessionId) return null
  return (
    <Link
      to={
        runId
          ? '/activity/sessions/$sessionId/runs/$runId'
          : '/activity/sessions/$sessionId'
      }
      params={runId ? { sessionId, runId } : { sessionId }}
      className="mt-3 inline-flex rounded-lg border border-slate-200 px-2.5 py-1.5 text-xs font-semibold text-blue-700 hover:bg-blue-50"
    >
      Open in Activity
    </Link>
  )
}

function Field({
  label,
  hint,
  error,
  children,
}: {
  label: string
  hint?: string
  error?: string
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
      {error ? (
        <span
          className="mt-1 block text-xs font-normal text-rose-600"
          role="alert"
        >
          {error}
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

function describeScheduleInput(
  triggerKind: 'cron' | 'once',
  cron: string,
  runAt: string,
  timezone: string,
) {
  const zone = timezone || getBrowserTimeZone()
  if (triggerKind === 'once') {
    return runAt
      ? `Once at ${runAt.replace('T', ' ')} in ${zone}`
      : `Choose a date and time in ${zone}`
  }
  const descriptions: Record<string, string> = {
    '0 9 * * *': 'Every day at 09:00',
    '0 9 * * 1-5': 'Every weekday at 09:00',
    '0 * * * *': 'At the start of every hour',
    '0 9 * * 1': 'Every Monday at 09:00',
  }
  return `${descriptions[cron.trim()] ?? `Cron ${cron || 'not set'}`} in ${zone}`
}

function describeTargetBehavior(
  continueCurrentSession: boolean,
  startFromCurrentSession: boolean,
  steerWhenRunning: boolean,
) {
  const target = continueCurrentSession
    ? 'The run continues in the current session.'
    : startFromCurrentSession
      ? 'The run starts a new session from the current session context.'
      : 'The run starts in its configured automation session.'
  const running = steerWhenRunning
    ? ' If the target is already running, this prompt is sent as steering to that active run.'
    : ' If the target is already running, it is not steered; normal runtime concurrency rules apply.'
  return target + running
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
