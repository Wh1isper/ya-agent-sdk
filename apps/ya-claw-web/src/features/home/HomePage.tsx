import { Link } from '@tanstack/react-router'
import type { ReactNode } from 'react'
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  CalendarClock,
  MessageSquarePlus,
  Workflow,
} from 'lucide-react'

import {
  useHeartbeatStatusQuery,
  useSchedulesQuery,
  useSessionsQuery,
  useWorkspaceRuntimeQuery,
} from '../../api/hooks'
import { EmptyState } from '../../components/EmptyState'
import { StatusBadge } from '../../components/StatusBadge'
import { QueryError, QuerySkeleton } from '../../components/ui/QueryState'
import { apiTimestamp } from '../../lib/date'
import type { SessionSummary } from '../../types'
import {
  channelLabel,
  sessionChannel,
  sessionTitle,
} from '../chat/sessionClassification'

export function HomePage() {
  const sessions = useSessionsQuery()
  const schedules = useSchedulesQuery({ includeWorkflow: true })
  const heartbeat = useHeartbeatStatusQuery()
  const workspace = useWorkspaceRuntimeQuery()

  const queries = [sessions, schedules, heartbeat, workspace]
  const failedQuery = queries.find((query) => query.isError)
  const allDataLoading = queries.every(
    (query) => query.isLoading && query.data === undefined,
  )

  if (allDataLoading) {
    return (
      <HomeFrame>
        <QuerySkeleton rows={5} />
      </HomeFrame>
    )
  }

  const rows = sessions.data ?? []
  const active = rows.filter((session) =>
    ['queued', 'running'].includes(session.status),
  )
  const attention = rows.filter((session) =>
    ['failed', 'cancelled'].includes(session.latest_run?.status ?? ''),
  )
  const upcoming = (schedules.data?.schedules ?? [])
    .filter((schedule) => schedule.enabled && schedule.trigger.next_fire_at)
    .slice(0, 4)

  return (
    <div className="mx-auto w-full max-w-7xl space-y-7 p-4 sm:p-6 lg:p-8">
      <section className="flex flex-col justify-between gap-5 border-b border-[var(--border)] pb-7 md:flex-row md:items-end">
        <div>
          <p className="text-sm font-medium text-[var(--primary)]">Home</p>
          <h1 className="mt-1 text-3xl font-semibold tracking-tight sm:text-4xl">
            What should YA Claw work on?
          </h1>
          <p className="mt-3 max-w-2xl text-sm leading-6 text-[var(--muted-foreground)]">
            Start a conversation, continue active work, or review automation
            that needs attention.
          </p>
        </div>
        <Link
          to="/conversations/new"
          className="inline-flex h-11 shrink-0 items-center justify-center gap-2 rounded-lg bg-[var(--primary)] px-4 text-sm font-semibold text-white shadow-sm transition hover:bg-[var(--primary-hover)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--focus)]"
        >
          <MessageSquarePlus className="h-4 w-4" />
          New conversation
        </Link>
      </section>

      {failedQuery ? (
        <QueryError
          compact
          title="Some workspace overview data is unavailable"
          error={failedQuery.error}
          onRetry={() => {
            void Promise.all(queries.map((query) => query.refetch()))
          }}
        />
      ) : null}

      {attention.length > 0 ? (
        <section className="rounded-xl border border-amber-200 bg-amber-50 p-4">
          <div className="flex items-start gap-3">
            <AlertTriangle className="mt-0.5 h-5 w-5 text-amber-700" />
            <div className="min-w-0">
              <h2 className="font-semibold text-amber-950">Needs attention</h2>
              <p className="mt-1 text-sm text-amber-800">
                {`${attention.length} recent conversation${attention.length === 1 ? '' : 's'} ended without completing (latest ${rows.length} checked).`}
              </p>
              <Link
                to="/activity"
                className="mt-3 inline-flex items-center gap-1 text-sm font-semibold text-amber-900 underline underline-offset-2"
              >
                Review activity <ArrowRight className="h-3.5 w-3.5" />
              </Link>
            </div>
          </div>
        </section>
      ) : null}

      <section className="grid grid-cols-2 gap-3 sm:gap-4 xl:grid-cols-4">
        <Metric
          icon={Activity}
          label="Active work"
          value={String(active.length)}
          detail={`${sessions.total} total · latest ${rows.length} checked`}
        />
        <Metric
          icon={Workflow}
          label="Workspace"
          value={
            workspace.data?.status ??
            (workspace.isError ? 'Unavailable' : 'Checking')
          }
          detail={
            workspace.data?.backend ??
            (workspace.isError ? 'Retry workspace status' : 'Runtime backend')
          }
        />
        <Metric
          icon={CalendarClock}
          label="Schedules"
          value={
            schedules.isError && schedules.data === undefined
              ? 'Unavailable'
              : String(
                  schedules.data?.schedules.filter((item) => item.enabled)
                    .length ?? 0,
                )
          }
          detail={
            schedules.isError && schedules.data === undefined
              ? 'Retry schedule status'
              : `${schedules.data?.schedules.length ?? 0} configured`
          }
        />
        <Metric
          icon={Activity}
          label="Heartbeat"
          value={
            heartbeat.isError && heartbeat.data === undefined
              ? 'Unavailable'
              : heartbeat.data?.enabled
                ? 'Enabled'
                : 'Disabled'
          }
          detail={
            heartbeat.isError && heartbeat.data === undefined
              ? 'Retry heartbeat status'
              : heartbeat.data?.next_fire_at
                ? `Next ${formatRelative(heartbeat.data.next_fire_at)}`
                : 'Not scheduled'
          }
        />
      </section>

      {active.length > 0 ? (
        <section className="rounded-xl border border-[var(--border)] bg-[var(--surface)]">
          <div className="flex items-center justify-between gap-3 border-b border-[var(--border)] px-5 py-4">
            <div>
              <h2 className="font-semibold">Active work</h2>
              <p className="mt-0.5 text-sm text-[var(--muted-foreground)]">
                Running and queued conversations
              </p>
            </div>
            <Link
              to="/activity"
              className="text-sm font-semibold text-[var(--primary)]"
            >
              Inspect all
            </Link>
          </div>
          <div className="divide-y divide-[var(--border)]">
            {active.slice(0, 5).map((session) => (
              <div
                key={session.id}
                className="flex flex-col gap-3 px-5 py-4 sm:flex-row sm:items-center sm:justify-between"
              >
                <div className="min-w-0">
                  <p className="truncate text-sm font-semibold">
                    {sessionTitle(session)}
                  </p>
                  <p className="mt-1 text-xs text-[var(--subtle-foreground)]">
                    {channelLabel(sessionChannel(session))} ·{' '}
                    {session.profile_name ?? 'Default agent'}
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <StatusBadge status={session.status} />
                  <Link
                    to="/conversations/sessions/$sessionId"
                    params={{ sessionId: session.id }}
                    className="rounded-lg border border-[var(--border)] px-3 py-2 text-sm font-semibold"
                  >
                    Open
                  </Link>
                  <Link
                    to="/activity/sessions/$sessionId"
                    params={{ sessionId: session.id }}
                    className="rounded-lg bg-[var(--primary)] px-3 py-2 text-sm font-semibold text-white"
                  >
                    Inspect
                  </Link>
                </div>
              </div>
            ))}
          </div>
        </section>
      ) : null}

      <section className="grid gap-6 xl:grid-cols-[1.35fr_0.65fr]">
        <div className="min-w-0 rounded-xl border border-[var(--border)] bg-[var(--surface)]">
          <div className="flex min-w-0 items-center justify-between gap-3 border-b border-[var(--border)] px-5 py-4">
            <div className="min-w-0">
              <h2 className="font-semibold">Recent conversations</h2>
              <p className="mt-0.5 text-sm text-[var(--muted-foreground)]">
                Continue where you left off
              </p>
            </div>
            <Link
              to="/conversations"
              className="shrink-0 text-sm font-semibold text-[var(--primary)]"
            >
              View all
            </Link>
          </div>
          <div className="divide-y divide-[var(--border)]">
            {rows.slice(0, 8).map((session) => (
              <RecentConversationLink key={session.id} session={session} />
            ))}
            {!sessions.isLoading && rows.length === 0 ? (
              <EmptyState
                title="No conversations yet"
                description="Start the first conversation and YA Claw will keep its work here."
                action={
                  <Link
                    to="/conversations/new"
                    className="rounded-lg bg-[var(--primary)] px-3 py-2 text-sm font-semibold text-white"
                  >
                    Start a conversation
                  </Link>
                }
                className="m-4"
              />
            ) : null}
          </div>
        </div>

        <div className="min-w-0 rounded-xl border border-[var(--border)] bg-[var(--surface)]">
          <div className="border-b border-[var(--border)] px-5 py-4">
            <h2 className="font-semibold">Upcoming automation</h2>
            <p className="mt-0.5 text-sm text-[var(--muted-foreground)]">
              Next scheduled work
            </p>
          </div>
          <div className="space-y-2 p-3">
            {upcoming.map((schedule) => (
              <Link
                key={schedule.id}
                to="/automation/schedules/$scheduleId"
                params={{ scheduleId: schedule.id }}
                className="block rounded-lg p-3 transition hover:bg-[var(--subtle)]"
              >
                <p className="text-sm font-semibold">{schedule.name}</p>
                <p className="mt-1 text-xs text-[var(--subtle-foreground)]">
                  {formatRelative(schedule.trigger.next_fire_at ?? '')} ·{' '}
                  {schedule.trigger.timezone}
                </p>
              </Link>
            ))}
            {!schedules.isLoading && upcoming.length === 0 ? (
              <EmptyState
                title="No upcoming work"
                description="Create a schedule when you want YA Claw to continue later."
                action={
                  <Link
                    to="/automation/schedules"
                    className="text-sm font-semibold text-[var(--primary)]"
                  >
                    Create schedule
                  </Link>
                }
                className="min-h-56 border-0"
              />
            ) : null}
          </div>
        </div>
      </section>
    </div>
  )
}

function RecentConversationLink({ session }: { session: SessionSummary }) {
  const channel = sessionChannel(session)
  const content = (
    <>
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-semibold">
          {sessionTitle(session)}
        </p>
        <p className="mt-1 truncate text-xs text-[var(--subtle-foreground)]">
          {channelLabel(channel)} · {session.profile_name ?? 'Default agent'} ·{' '}
          {formatRelative(session.updated_at)}
        </p>
      </div>
      <StatusBadge status={session.status} />
    </>
  )
  const className =
    'flex min-w-0 items-center justify-between gap-4 px-5 py-4 transition hover:bg-[var(--subtle)]'

  return (
    <Link
      to="/conversations/sessions/$sessionId"
      params={{ sessionId: session.id }}
      className={className}
    >
      {content}
    </Link>
  )
}

function HomeFrame({ children }: { children: ReactNode }) {
  return (
    <div className="mx-auto w-full max-w-7xl space-y-7 p-4 sm:p-6 lg:p-8">
      <section className="border-b border-[var(--border)] pb-7">
        <p className="text-sm font-medium text-[var(--primary)]">Home</p>
        <h1 className="mt-1 text-3xl font-semibold tracking-tight sm:text-4xl">
          What should YA Claw work on?
        </h1>
        <p className="mt-3 max-w-2xl text-sm leading-6 text-[var(--muted-foreground)]">
          Start a conversation, continue active work, or review automation that
          needs attention.
        </p>
      </section>
      {children}
    </div>
  )
}

function Metric({
  icon: Icon,
  label,
  value,
  detail,
}: {
  icon: typeof Activity
  label: string
  value: string
  detail: string
}) {
  return (
    <div className="rounded-xl border border-[var(--border)] bg-[var(--surface)] p-4">
      <Icon className="h-5 w-5 text-[var(--primary)]" />
      <p className="mt-4 text-xs font-medium uppercase tracking-wide text-[var(--subtle-foreground)]">
        {label}
      </p>
      <p className="mt-1 truncate text-2xl font-semibold capitalize">{value}</p>
      <p className="mt-1 text-xs text-[var(--muted-foreground)]">{detail}</p>
    </div>
  )
}

function formatRelative(value: string) {
  const timestamp = apiTimestamp(value)
  if (!Number.isFinite(timestamp)) return 'Not scheduled'
  const deltaMinutes = Math.round((timestamp - Date.now()) / 60_000)
  const formatter = new Intl.RelativeTimeFormat(undefined, { numeric: 'auto' })
  if (Math.abs(deltaMinutes) < 60)
    return formatter.format(deltaMinutes, 'minute')
  const hours = Math.round(deltaMinutes / 60)
  if (Math.abs(hours) < 24) return formatter.format(hours, 'hour')
  return formatter.format(Math.round(hours / 24), 'day')
}
