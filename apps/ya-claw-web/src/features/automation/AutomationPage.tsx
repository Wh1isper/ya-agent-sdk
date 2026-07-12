import { Link } from '@tanstack/react-router'
import {
  Activity,
  AlertTriangle,
  BrainCircuit,
  CalendarClock,
  ChevronRight,
  Clock3,
  GitBranch,
  HeartPulse,
  Plus,
} from 'lucide-react'

import {
  useAgencyFiresQuery,
  useAgencyStatusQuery,
  useHeartbeatFiresQuery,
  useHeartbeatStatusQuery,
  useSchedulesQuery,
  useWorkflowRunsQuery,
  useWorkflowsQuery,
} from '../../api/hooks'
import { EmptyState } from '../../components/EmptyState'
import { StatusBadge } from '../../components/StatusBadge'
import { QueryError } from '../../components/ui'
import { apiTimestamp, parseApiDate } from '../../lib/date'
import type { ScheduleSummary } from '../../types'
import { AUTOMATION_LIST_LIMIT, mayHaveMoreAutomationRows } from './listLimit'

type AutomationActivityItem = {
  id: string
  kind: string
  title: string
  status: string
  timestamp: string
  sessionId: string | null
  runId: string | null
  fallbackPath:
    | '/automation/schedules'
    | '/automation/workflows'
    | '/automation/heartbeat'
    | '/automation/agency'
}

export function AutomationPage() {
  const schedules = useSchedulesQuery({
    includeWorkflow: false,
    limit: AUTOMATION_LIST_LIMIT,
  })
  const workflows = useWorkflowsQuery({ limit: AUTOMATION_LIST_LIMIT })
  const workflowRuns = useWorkflowRunsQuery({ limit: AUTOMATION_LIST_LIMIT })
  const heartbeat = useHeartbeatStatusQuery()
  const heartbeatFires = useHeartbeatFiresQuery()
  const agency = useAgencyStatusQuery()
  const agencyFires = useAgencyFiresQuery()
  const queries = [
    schedules,
    workflows,
    workflowRuns,
    heartbeat,
    heartbeatFires,
    agency,
    agencyFires,
  ]

  const failedQuery = queries.find((query) => query.isError)
  const allDataLoading = queries.every(
    (query) => query.isLoading && query.data === undefined,
  )

  if (allDataLoading) {
    return <AutomationSkeleton />
  }

  const scheduleRows = schedules.data?.schedules ?? []
  const workflowRows = workflows.data?.workflows ?? []
  const workflowRunRows = workflowRuns.data?.workflow_runs ?? []
  const listMayBeTruncated = [
    scheduleRows.length,
    workflowRows.length,
    workflowRunRows.length,
  ].some(mayHaveMoreAutomationRows)
  const enabledSchedules = scheduleRows.filter((schedule) => schedule.enabled)
  const failureCount = scheduleRows.reduce(
    (total, schedule) => total + schedule.failure_count,
    0,
  )
  const nextRuns = enabledSchedules
    .filter((schedule) => schedule.trigger.next_fire_at)
    .sort(
      (left, right) =>
        apiTimestamp(left.trigger.next_fire_at ?? 0) -
        apiTimestamp(right.trigger.next_fire_at ?? 0),
    )
    .slice(0, 3)
  const recentActivity: AutomationActivityItem[] = [
    ...scheduleRows.flatMap((schedule) => {
      const fire = schedule.last_fire
      if (!fire) return []
      return [
        {
          id: `schedule-${fire.id}`,
          kind: 'Schedule',
          title: schedule.name,
          status: fire.run_status ?? fire.status,
          timestamp: fire.created_at,
          sessionId:
            fire.target_session_id ??
            fire.created_session_id ??
            fire.source_session_id ??
            null,
          runId: fire.run_id ?? fire.active_run_id ?? null,
          fallbackPath: '/automation/schedules' as const,
        },
      ]
    }),
    ...workflowRunRows.map((run) => ({
      id: `workflow-${run.id}`,
      kind: 'Workflow',
      title: run.workflow_name ?? `Workflow run ${run.id.slice(0, 8)}`,
      status: run.status,
      timestamp: run.updated_at,
      sessionId: run.supervisor_session_id ?? null,
      runId: run.supervisor_run_id ?? null,
      fallbackPath: '/automation/workflows' as const,
    })),
    ...(heartbeatFires.data?.fires ?? []).map((fire) => ({
      id: `heartbeat-${fire.id}`,
      kind: 'Heartbeat',
      title: 'Heartbeat pulse',
      status: fire.run_status ?? fire.status,
      timestamp: fire.created_at,
      sessionId: fire.session_id ?? null,
      runId: fire.run_id ?? null,
      fallbackPath: '/automation/heartbeat' as const,
    })),
    ...(agencyFires.data?.fires ?? []).map((fire) => ({
      id: `agency-${fire.id}`,
      kind: 'Proactive agent',
      title: 'Proactive follow-up',
      status: fire.run_status ?? fire.status,
      timestamp: fire.updated_at,
      sessionId: fire.agency_session_id ?? fire.source_session_id ?? null,
      runId: fire.run_id ?? fire.active_run_id ?? null,
      fallbackPath: '/automation/agency' as const,
    })),
  ]
    .sort(
      (left, right) =>
        apiTimestamp(right.timestamp) - apiTimestamp(left.timestamp),
    )
    .slice(0, 8)
  const activeAutomation =
    enabledSchedules.length +
    (heartbeat.data?.enabled ? 1 : 0) +
    (agency.data?.enabled ? 1 : 0)

  return (
    <div className="mx-auto w-full max-w-6xl space-y-6 p-4 sm:p-6 lg:p-8">
      <section className="grid gap-4 lg:grid-cols-[1fr_18rem]">
        <div>
          <p className="text-sm font-medium text-[var(--primary)]">
            Automation
          </p>
          <h1 className="mt-1 text-3xl font-semibold tracking-tight">
            Work that continues without supervision
          </h1>
          <p className="mt-3 max-w-2xl text-sm leading-6 text-[var(--muted-foreground)]">
            Schedule prompts, run reusable workflows, and monitor YA Claw's
            proactive background agents from one place.
          </p>
        </div>
        <div className="rounded-xl border border-[var(--border)] bg-[var(--surface)] p-4">
          <p className="text-xs font-semibold uppercase tracking-wide text-[var(--subtle-foreground)]">
            Active automation
          </p>
          <p className="mt-2 text-3xl font-semibold">{activeAutomation}</p>
          <p className="mt-1 text-sm text-[var(--muted-foreground)]">
            {enabledSchedules.length} schedules · heartbeat{' '}
            {heartbeat.isError && heartbeat.data === undefined
              ? 'unavailable'
              : heartbeat.data?.enabled
                ? 'enabled'
                : 'disabled'}
          </p>
        </div>
      </section>

      {listMayBeTruncated ? (
        <div
          className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900"
          role="status"
        >
          Automation summaries show up to {AUTOMATION_LIST_LIMIT} items per
          list. Open the relevant page and narrow its filters if an older item
          is not shown.
        </div>
      ) : null}

      {failedQuery ? (
        <QueryError
          compact
          title="Some automation overview data is unavailable"
          error={failedQuery.error}
          onRetry={() => {
            void Promise.all(queries.map((query) => query.refetch()))
          }}
        />
      ) : null}

      <section
        className="grid grid-cols-2 gap-3 lg:grid-cols-4"
        aria-label="Automation health"
      >
        <Metric
          icon={CalendarClock}
          label="Enabled schedules"
          value={enabledSchedules.length}
        />
        <Metric
          icon={AlertTriangle}
          label="Recorded failures"
          value={failureCount}
          warning={failureCount > 0}
        />
        <Metric icon={Clock3} label="Upcoming runs" value={nextRuns.length} />
        <Metric
          icon={Activity}
          label="Recent activity"
          value={recentActivity.length}
        />
      </section>

      <div className="grid gap-4 md:grid-cols-2">
        <AutomationCard
          to="/automation/schedules"
          icon={CalendarClock}
          title="Schedules"
          description="Run prompts or workflows once, on a cadence, or in a chosen timezone."
          detail={
            schedules.isError && schedules.data === undefined
              ? 'Schedule data unavailable'
              : `${scheduleRows.length} schedules · ${enabledSchedules.length} enabled`
          }
        />
        <AutomationCard
          to="/automation/workflows"
          icon={GitBranch}
          title="Workflows"
          description="Build and observe durable multi-step agent work."
          detail={
            workflows.isError && workflows.data === undefined
              ? 'Workflow data unavailable'
              : `${workflowRows.length} workflow definitions`
          }
        />
        <AutomationCard
          to="/automation/agency"
          icon={BrainCircuit}
          title="Proactive agent"
          description="Inspect autonomous wake-ups, decisions, and memory-driven follow-up."
          detail={
            agency.isError && agency.data === undefined
              ? 'Status unavailable'
              : agency.data?.enabled
                ? 'Enabled'
                : 'Disabled'
          }
        />
        <AutomationCard
          to="/automation/heartbeat"
          icon={HeartPulse}
          title="Heartbeat"
          description="Review periodic runtime guidance and recent background pulses."
          detail={
            heartbeat.isError && heartbeat.data === undefined
              ? 'Status unavailable'
              : heartbeat.data?.enabled
                ? 'Enabled'
                : 'Disabled'
          }
        />
      </div>

      <section className="grid gap-4 lg:grid-cols-2">
        <OverviewList title="Next runs">
          {nextRuns.length ? (
            nextRuns.map((schedule) => (
              <ScheduleOverviewRow
                key={schedule.id}
                schedule={schedule}
                mode="next"
              />
            ))
          ) : (
            <EmptyState
              title="Nothing scheduled next"
              description="Enable a schedule to see its next run here."
              action={
                <Link
                  to="/automation/schedules"
                  className="inline-flex items-center gap-2 rounded-lg bg-[var(--primary)] px-3 py-2 text-sm font-semibold text-white"
                >
                  <Plus className="h-4 w-4" /> Create schedule
                </Link>
              }
              className="min-h-52 border-0 bg-transparent"
            />
          )}
        </OverviewList>
        <OverviewList title="Recent activity">
          {recentActivity.length ? (
            recentActivity.map((item) => (
              <AutomationActivityRow key={item.id} item={item} />
            ))
          ) : (
            <EmptyState
              title="No schedule activity yet"
              description="Recent fires and their outcomes will appear here."
              action={
                <Link
                  to="/automation/schedules"
                  className="text-sm font-semibold text-[var(--primary)]"
                >
                  Review schedules
                </Link>
              }
              className="min-h-52 border-0 bg-transparent"
            />
          )}
        </OverviewList>
      </section>
    </div>
  )
}

function Metric({
  icon: Icon,
  label,
  value,
  warning = false,
}: {
  icon: typeof CalendarClock
  label: string
  value: number
  warning?: boolean
}) {
  return (
    <div className="rounded-xl border border-[var(--border)] bg-[var(--surface)] p-4">
      <Icon
        className={`h-4 w-4 ${warning ? 'text-amber-600' : 'text-[var(--primary)]'}`}
        aria-hidden
      />
      <p className="mt-3 text-2xl font-semibold">{value}</p>
      <p className="text-xs text-[var(--muted-foreground)]">{label}</p>
    </div>
  )
}

function OverviewList({
  title,
  children,
}: {
  title: string
  children: React.ReactNode
}) {
  return (
    <div className="rounded-xl border border-[var(--border)] bg-[var(--surface)] p-4">
      <h2 className="text-sm font-semibold">{title}</h2>
      <div className="mt-3 space-y-2">{children}</div>
    </div>
  )
}

function ScheduleOverviewRow({
  schedule,
  mode,
}: {
  schedule: ScheduleSummary
  mode: 'next' | 'recent'
}) {
  const fireStatus = schedule.last_fire
    ? (schedule.last_fire.run_status ?? schedule.last_fire.status)
    : null
  const timestamp =
    mode === 'next'
      ? schedule.trigger.next_fire_at
      : schedule.last_fire?.created_at
  return (
    <Link
      to="/automation/schedules/$scheduleId"
      params={{ scheduleId: schedule.id }}
      className="flex items-center justify-between gap-3 rounded-lg border border-[var(--border)] p-3 transition hover:bg-[var(--subtle)]"
    >
      <div className="min-w-0">
        <p className="truncate text-sm font-semibold">{schedule.name}</p>
        <p className="mt-1 text-xs text-[var(--muted-foreground)]">
          {formatOverviewDate(timestamp)} · {schedule.trigger.timezone}
        </p>
      </div>
      {mode === 'recent' && fireStatus ? (
        <StatusBadge status={fireStatus} />
      ) : (
        <ChevronRight className="h-4 w-4 text-[var(--subtle-foreground)]" />
      )}
    </Link>
  )
}

function AutomationActivityRow({ item }: { item: AutomationActivityItem }) {
  const content = (
    <>
      <div className="min-w-0">
        <p className="truncate text-sm font-semibold">{item.title}</p>
        <p className="mt-1 text-xs text-[var(--muted-foreground)]">
          {item.kind} · {formatOverviewDate(item.timestamp)}
        </p>
      </div>
      <StatusBadge status={item.status} />
    </>
  )
  const className =
    'flex items-center justify-between gap-3 rounded-lg border border-[var(--border)] p-3 transition hover:bg-[var(--subtle)]'

  if (item.sessionId && item.runId) {
    return (
      <Link
        to="/activity/sessions/$sessionId/runs/$runId"
        params={{ sessionId: item.sessionId, runId: item.runId }}
        className={className}
      >
        {content}
      </Link>
    )
  }
  if (item.sessionId) {
    return (
      <Link
        to="/activity/sessions/$sessionId"
        params={{ sessionId: item.sessionId }}
        className={className}
      >
        {content}
      </Link>
    )
  }
  return (
    <Link to={item.fallbackPath} className={className}>
      {content}
    </Link>
  )
}

function formatOverviewDate(value?: string | null) {
  if (!value) return 'Not scheduled'
  const date = parseApiDate(value)
  if (Number.isNaN(date.getTime())) return 'Time unavailable'
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(date)
}

function AutomationSkeleton() {
  return (
    <div
      className="mx-auto w-full max-w-6xl space-y-6 p-4 sm:p-6 lg:p-8"
      aria-label="Loading automation overview"
    >
      <div className="h-32 animate-pulse rounded-xl bg-slate-100" />
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {Array.from({ length: 4 }).map((_, index) => (
          <div
            key={index}
            className="h-28 animate-pulse rounded-xl bg-slate-100"
          />
        ))}
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        {Array.from({ length: 4 }).map((_, index) => (
          <div
            key={index}
            className="h-44 animate-pulse rounded-xl bg-slate-100"
          />
        ))}
      </div>
    </div>
  )
}

function AutomationCard({
  to,
  icon: Icon,
  title,
  description,
  detail,
}: {
  to: string
  icon: typeof CalendarClock
  title: string
  description: string
  detail: string
}) {
  return (
    <Link
      to={to}
      className="group flex min-h-44 flex-col rounded-xl border border-[var(--border)] bg-[var(--surface)] p-5 transition hover:border-[var(--primary)] hover:shadow-[var(--shadow-sm)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--focus)]"
    >
      <span className="flex h-10 w-10 items-center justify-center rounded-lg bg-[var(--primary-subtle)] text-[var(--primary)]">
        <Icon className="h-5 w-5" aria-hidden />
      </span>
      <h2 className="mt-4 text-lg font-semibold">{title}</h2>
      <p className="mt-1 text-sm leading-6 text-[var(--muted-foreground)]">
        {description}
      </p>
      <span className="mt-auto flex items-center justify-between pt-4 text-xs font-medium text-[var(--subtle-foreground)]">
        {detail}
        <ChevronRight className="h-4 w-4 transition group-hover:translate-x-0.5" />
      </span>
    </Link>
  )
}
