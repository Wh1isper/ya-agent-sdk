import {
  Activity,
  ArchiveX,
  CalendarClock,
  CheckCircle2,
  Database,
  HeartPulse,
  MessageSquare,
  PlayCircle,
  Server,
  Workflow,
  type LucideIcon,
} from 'lucide-react'

import {
  useClawInfoQuery,
  useHealthQuery,
  useHeartbeatStatusQuery,
  useProfilesQuery,
  useSchedulesQuery,
  useSessionsQuery,
  useWorkspaceRuntimeQuery,
} from '../../api/hooks'
import { StatusBadge } from '../../components/StatusBadge'
import { cn, formatShortId } from '../../lib/utils'
import { useLayoutStore } from '../../stores/layoutStore'
import type { SessionSandboxState, SessionSummary } from '../../types'
import {
  runtimeHeroTone,
  sandboxLabel,
  sandboxTone,
  ttlLabel,
} from '../workspaceDisplay'

export function OverviewPage() {
  const health = useHealthQuery()
  const info = useClawInfoQuery()
  const sessions = useSessionsQuery()
  const profiles = useProfilesQuery()
  const schedules = useSchedulesQuery()
  const heartbeat = useHeartbeatStatusQuery()
  const workspaceRuntime = useWorkspaceRuntimeQuery()
  const setRoute = useLayoutStore((state) => state.setRoute)
  const selectSession = useLayoutStore((state) => state.selectSession)
  const rows = sessions.data ?? []
  const activeRuns = rows.filter(
    (session) => session.status === 'queued' || session.status === 'running',
  )
  const completedSessions = rows.filter(
    (session) => session.status === 'completed',
  )
  const memoryExtractCount = rows.reduce(
    (count, session) => count + (session.memory_state?.extract_count ?? 0),
    0,
  )
  const readySandboxCount = rows.filter(
    (session) =>
      session.workspace_state?.sandbox_state?.ready_state === 'ready',
  ).length
  const enabledSchedules =
    schedules.data?.schedules.filter((item) => item.enabled).length ?? 0

  return (
    <div className="space-y-6 p-6">
      <section className="overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm">
        <div className="grid gap-0 lg:grid-cols-[1.4fr_0.8fr]">
          <div className="bg-gradient-to-br from-slate-950 via-slate-900 to-blue-950 p-6 text-white">
            <p className="text-sm font-medium text-blue-200">Runtime console</p>
            <h2 className="mt-2 text-3xl font-semibold tracking-tight">
              Operate agents from one focused workspace
            </h2>
            <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-300">
              Monitor health, jump into active sessions, and confirm background
              automation status from the first screen.
            </p>
            <div className="mt-6 flex flex-wrap gap-2">
              <HeroPill
                icon={Server}
                label="Service"
                value={health.data?.status ?? 'checking'}
                tone={health.data?.status === 'ok' ? 'success' : 'warning'}
              />
              <HeroPill
                icon={Database}
                label="Storage"
                value={
                  info.data?.storage_model ?? health.data?.database ?? 'unknown'
                }
                tone="info"
              />
              <HeroPill
                icon={HeartPulse}
                label="Heartbeat"
                value={heartbeat.data?.enabled ? 'enabled' : 'disabled'}
                tone={heartbeat.data?.enabled ? 'success' : 'muted'}
              />
              <HeroPill
                icon={Server}
                label="Workspace"
                value={workspaceRuntime.data?.status ?? 'checking'}
                tone={runtimeHeroTone(workspaceRuntime.data?.status)}
              />
            </div>
          </div>
          <div className="grid gap-3 bg-white p-6">
            <QuickAction
              icon={MessageSquare}
              title="Open chat runtime"
              description="Review session replay, live events, and run controls."
              onClick={() => setRoute('chat')}
            />
            <QuickAction
              icon={CalendarClock}
              title="Review automation"
              description={`${enabledSchedules} enabled schedules · heartbeat ${heartbeat.data?.enabled ? 'enabled' : 'disabled'}`}
              onClick={() => setRoute('schedules')}
            />
            <QuickAction
              icon={Activity}
              title="Tune profiles"
              description={`${profiles.data?.length ?? 0} agent profiles available`}
              onClick={() => setRoute('profiles')}
            />
          </div>
        </div>
      </section>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          icon={Workflow}
          label="Active runs"
          value={String(activeRuns.length)}
          detail={`${rows.length} total sessions`}
          accent="amber"
        />
        <MetricCard
          icon={CheckCircle2}
          label="Completed"
          value={String(completedSessions.length)}
          detail="Successful session state"
          accent="emerald"
        />
        <MetricCard
          icon={ArchiveX}
          label="Memory extracts"
          value={String(memoryExtractCount)}
          detail="Workspace-native memory updates"
          accent="violet"
        />
        <MetricCard
          icon={Server}
          label="Ready sandboxes"
          value={String(readySandboxCount)}
          detail={`${rows.length} session workspace states`}
          accent="blue"
        />
      </div>

      <section className="grid gap-4 xl:grid-cols-[1.2fr_0.8fr]">
        <div className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-center justify-between gap-3">
            <div>
              <h2 className="text-sm font-semibold text-slate-950">
                Recent sessions
              </h2>
              <p className="mt-1 text-sm text-slate-500">
                Latest conversation entry points with run and memory state.
              </p>
            </div>
            <button
              type="button"
              className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-700 shadow-sm transition hover:bg-slate-50"
              onClick={() => setRoute('chat')}
            >
              <PlayCircle className="h-3.5 w-3.5" />
              Open chat
            </button>
          </div>
          <div className="mt-4 space-y-2">
            {sessions.isLoading ? <SessionRowsSkeleton /> : null}
            {!sessions.isLoading && rows.length === 0 ? (
              <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50 p-8 text-center">
                <p className="text-sm font-semibold text-slate-900">
                  No sessions yet
                </p>
                <p className="mt-2 text-sm text-slate-500">
                  Start from Chat Runtime to create the first session.
                </p>
              </div>
            ) : null}
            {rows.slice(0, 8).map((session) => (
              <SessionRow
                key={session.id}
                session={session}
                onOpen={() => selectSession(session.id)}
              />
            ))}
          </div>
        </div>

        <div className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
          <h2 className="text-sm font-semibold text-slate-950">
            Runtime details
          </h2>
          <p className="mt-1 text-sm text-slate-500">
            Deployment metadata and workspace backend.
          </p>
          <dl className="mt-5 grid gap-4 text-sm">
            <Detail
              label="Environment"
              value={info.data?.environment ?? 'unknown'}
            />
            <Detail
              label="Version"
              value={
                info.data?.service_version ?? info.data?.version ?? 'unknown'
              }
            />
            <Detail
              label="Revision"
              value={shortRevision(info.data?.service_revision)}
              title={info.data?.service_revision ?? undefined}
            />
            <Detail label="Build" value={info.data?.service_build ?? 'local'} />
            <Detail label="Image" value={info.data?.service_image ?? 'local'} />
            <Detail
              label="Workspace"
              value={info.data?.workspace_provider_backend ?? 'unknown'}
            />
            <Detail
              label="Workspace status"
              value={workspaceRuntime.data?.status ?? 'checking'}
            />
            <Detail
              label="Execution location"
              value={workspaceRuntime.data?.execution_location ?? 'unknown'}
            />
            <Detail
              label="Service path"
              value={workspaceRuntime.data?.workspace.service_path ?? 'unknown'}
            />
            <Detail
              label="Virtual path"
              value={workspaceRuntime.data?.workspace.virtual_path ?? 'unknown'}
            />
            {workspaceRuntime.data?.docker ? (
              <>
                <Detail
                  label="Docker image"
                  value={workspaceRuntime.data.docker.image.ref}
                />
                <Detail
                  label="Idle TTL"
                  value={ttlLabel(
                    workspaceRuntime.data.docker.idle_ttl_seconds,
                  )}
                />
              </>
            ) : null}
            <Detail
              label="Base URL"
              value={info.data?.public_base_url ?? 'unknown'}
            />
          </dl>
        </div>
      </section>
    </div>
  )
}

const accentClasses: Record<string, string> = {
  blue: 'bg-blue-50 text-blue-600 ring-blue-100',
  emerald: 'bg-emerald-50 text-emerald-600 ring-emerald-100',
  amber: 'bg-amber-50 text-amber-600 ring-amber-100',
  violet: 'bg-violet-50 text-violet-600 ring-violet-100',
}

function MetricCard({
  icon: Icon,
  label,
  value,
  detail,
  accent,
}: {
  icon: LucideIcon
  label: string
  value: string
  detail: string
  accent: string
}) {
  return (
    <div className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex items-start justify-between gap-3">
        <div
          className={cn(
            'inline-flex rounded-2xl p-2.5 ring-1',
            accentClasses[accent] ?? accentClasses.blue,
          )}
        >
          <Icon className="h-5 w-5" />
        </div>
        <span className="text-xs font-medium text-slate-400">Live</span>
      </div>
      <p className="mt-5 text-sm font-medium text-slate-500">{label}</p>
      <p className="mt-1 text-3xl font-semibold capitalize tracking-tight text-slate-950">
        {value}
      </p>
      <p className="mt-2 text-xs text-slate-500">{detail}</p>
    </div>
  )
}

function HeroPill({
  icon: Icon,
  label,
  value,
  tone,
}: {
  icon: LucideIcon
  label: string
  value: string
  tone: 'success' | 'warning' | 'info' | 'muted'
}) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-medium capitalize',
        tone === 'success' &&
          'border-emerald-400/30 bg-emerald-400/10 text-emerald-100',
        tone === 'warning' &&
          'border-amber-400/30 bg-amber-400/10 text-amber-100',
        tone === 'info' && 'border-blue-400/30 bg-blue-400/10 text-blue-100',
        tone === 'muted' &&
          'border-slate-400/30 bg-slate-400/10 text-slate-200',
      )}
    >
      <Icon className="h-3.5 w-3.5" />
      {label}: {value}
    </span>
  )
}

function QuickAction({
  icon: Icon,
  title,
  description,
  onClick,
}: {
  icon: LucideIcon
  title: string
  description: string
  onClick: () => void
}) {
  return (
    <button
      type="button"
      className="group flex items-center gap-3 rounded-2xl border border-slate-200 bg-white p-4 text-left shadow-sm transition hover:border-blue-200 hover:bg-blue-50/60"
      onClick={onClick}
    >
      <span className="flex h-10 w-10 items-center justify-center rounded-2xl bg-slate-100 text-slate-600 transition group-hover:bg-blue-600 group-hover:text-white">
        <Icon className="h-4 w-4" />
      </span>
      <span className="min-w-0">
        <span className="block text-sm font-semibold text-slate-900">
          {title}
        </span>
        <span className="mt-1 block text-xs leading-5 text-slate-500">
          {description}
        </span>
      </span>
    </button>
  )
}

function SessionRow({
  session,
  onOpen,
}: {
  session: SessionSummary
  onOpen: () => void
}) {
  return (
    <button
      type="button"
      className="flex w-full items-center justify-between gap-4 rounded-2xl border border-slate-100 p-3 text-left transition hover:border-blue-200 hover:bg-blue-50/50"
      onClick={onOpen}
    >
      <div className="min-w-0">
        <div className="flex items-center gap-2">
          <p className="mono text-xs text-slate-500">
            {formatShortId(session.id, 12)}
          </p>
          <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-500">
            {session.run_count} runs
          </span>
        </div>
        <p className="mt-1 line-clamp-1 text-sm font-medium text-slate-900">
          {session.latest_run?.input_preview ?? 'No input yet'}
        </p>
        <p className="mt-1 text-xs text-slate-500">
          {session.profile_name ?? 'default'} · {session.session_type}
        </p>
      </div>
      <div className="flex shrink-0 items-center gap-2">
        {session.memory_state ? (
          <span className="rounded-full bg-violet-50 px-2 py-1 text-xs font-medium text-violet-700">
            {session.memory_state.extract_count} extracts
          </span>
        ) : null}
        <SandboxBadge
          sandbox={session.workspace_state?.sandbox_state ?? null}
        />
        <StatusBadge status={session.status} />
      </div>
    </button>
  )
}

function SandboxBadge({ sandbox }: { sandbox: SessionSandboxState | null }) {
  const tone = sandboxTone(sandbox)
  return (
    <span
      className={cn(
        'rounded-full px-2 py-1 text-xs font-medium capitalize',
        tone === 'success' && 'bg-emerald-50 text-emerald-700',
        tone === 'warning' && 'bg-amber-50 text-amber-700',
        tone === 'error' && 'bg-rose-50 text-rose-700',
        tone === 'info' && 'bg-blue-50 text-blue-700',
        tone === 'muted' && 'bg-slate-100 text-slate-500',
      )}
      title={sandbox?.container_ref ?? undefined}
    >
      {sandboxLabel(sandbox)}
      {sandbox?.ttl_seconds_remaining != null
        ? ` · ${ttlLabel(sandbox.ttl_seconds_remaining)}`
        : ''}
    </span>
  )
}

function SessionRowsSkeleton() {
  return (
    <div className="space-y-2">
      {Array.from({ length: 5 }).map((_, index) => (
        <div
          key={index}
          className="rounded-2xl border border-slate-100 bg-white p-3"
        >
          <div className="h-3 w-28 animate-pulse rounded bg-slate-100" />
          <div className="mt-3 h-4 w-full animate-pulse rounded bg-slate-100" />
          <div className="mt-2 h-3 w-32 animate-pulse rounded bg-slate-100" />
        </div>
      ))}
    </div>
  )
}

function Detail({
  label,
  value,
  title,
}: {
  label: string
  value: string
  title?: string
}) {
  return (
    <div className="rounded-2xl border border-slate-100 bg-slate-50 p-3">
      <dt className="text-xs font-medium uppercase tracking-wide text-slate-400">
        {label}
      </dt>
      <dd className="mt-1 break-all text-slate-800" title={title}>
        {value}
      </dd>
    </div>
  )
}

function shortRevision(value: string | null | undefined) {
  return value ? value.slice(0, 12) : 'unknown'
}
