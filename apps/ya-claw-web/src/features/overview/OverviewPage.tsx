import {
  Activity,
  ArchiveX,
  CalendarClock,
  Database,
  HeartPulse,
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
} from '../../api/hooks'
import { StatusBadge } from '../../components/StatusBadge'

export function OverviewPage() {
  const health = useHealthQuery()
  const info = useClawInfoQuery()
  const sessions = useSessionsQuery()
  const profiles = useProfilesQuery()
  const schedules = useSchedulesQuery()
  const heartbeat = useHeartbeatStatusQuery()
  const activeRuns = (sessions.data ?? []).filter(
    (session) => session.status === 'queued' || session.status === 'running',
  )
  const memoryExtractCount = (sessions.data ?? []).reduce(
    (count, session) => count + (session.memory_state?.extract_count ?? 0),
    0,
  )

  return (
    <div className="space-y-6 p-6">
      <div>
        <p className="text-sm font-medium text-blue-600">Runtime console</p>
        <h1 className="mt-1 text-2xl font-semibold tracking-tight text-slate-950">
          YA Claw Overview
        </h1>
        <p className="mt-2 text-sm text-slate-500">
          Monitor runtime health, active sessions, and profile state.
        </p>
      </div>

      <div className="grid grid-cols-4 gap-4">
        <MetricCard
          icon={Server}
          label="Service"
          value={health.data?.status ?? 'unknown'}
          accent="blue"
        />
        <MetricCard
          icon={Database}
          label="Storage"
          value={info.data?.storage_model ?? health.data?.database ?? 'unknown'}
          accent="emerald"
        />
        <MetricCard
          icon={Workflow}
          label="Active runs"
          value={String(activeRuns.length)}
          accent="amber"
        />
        <MetricCard
          icon={Activity}
          label="Profiles"
          value={String(profiles.data?.length ?? 0)}
          accent="violet"
        />
        <MetricCard
          icon={ArchiveX}
          label="Memory extracts"
          value={String(memoryExtractCount)}
          accent="violet"
        />
        <MetricCard
          icon={CalendarClock}
          label="Schedules"
          value={String(
            schedules.data?.schedules.filter((item) => item.enabled).length ??
              0,
          )}
          accent="blue"
        />
        <MetricCard
          icon={HeartPulse}
          label="Heartbeat"
          value={heartbeat.data?.enabled ? 'enabled' : 'disabled'}
          accent="emerald"
        />
      </div>

      <section className="grid grid-cols-2 gap-4">
        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <h2 className="text-sm font-semibold text-slate-950">
            Recent Sessions
          </h2>
          <div className="mt-4 space-y-3">
            {(sessions.data ?? []).slice(0, 8).map((session) => (
              <div
                key={session.id}
                className="flex items-center justify-between rounded-xl border border-slate-100 p-3"
              >
                <div>
                  <p className="mono text-xs text-slate-500">
                    {session.id.slice(0, 10)}
                  </p>
                  <p className="mt-1 text-sm font-medium text-slate-900">
                    {session.latest_run?.input_preview ?? 'No input yet'}
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  {session.memory_state ? (
                    <span className="rounded-full bg-violet-50 px-2 py-1 text-xs font-medium text-violet-700">
                      {session.memory_state.extract_count} extracts
                    </span>
                  ) : null}
                  <StatusBadge status={session.status} />
                </div>
              </div>
            ))}
            {sessions.data?.length === 0 ? (
              <p className="text-sm text-slate-500">No sessions yet.</p>
            ) : null}
          </div>
        </div>

        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <h2 className="text-sm font-semibold text-slate-950">
            Runtime Details
          </h2>
          <dl className="mt-4 grid grid-cols-2 gap-4 text-sm">
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
  blue: 'bg-blue-50 text-blue-600',
  emerald: 'bg-emerald-50 text-emerald-600',
  amber: 'bg-amber-50 text-amber-600',
  violet: 'bg-violet-50 text-violet-600',
}

function MetricCard({
  icon: Icon,
  label,
  value,
  accent,
}: {
  icon: LucideIcon
  label: string
  value: string
  accent: string
}) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div
        className={`inline-flex rounded-xl p-2 ${accentClasses[accent] ?? accentClasses.blue}`}
      >
        <Icon className="h-5 w-5" />
      </div>
      <p className="mt-4 text-sm text-slate-500">{label}</p>
      <p className="mt-1 text-2xl font-semibold capitalize text-slate-950">
        {value}
      </p>
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
    <div>
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
