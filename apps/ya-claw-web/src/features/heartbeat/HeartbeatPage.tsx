import { Activity, FileText, Play, RadioTower } from 'lucide-react'

import {
  useHeartbeatConfigQuery,
  useHeartbeatFiresQuery,
  useHeartbeatStatusQuery,
  useTriggerHeartbeatMutation,
} from '../../api/hooks'
import { EmptyState } from '../../components/EmptyState'
import { StatusBadge } from '../../components/StatusBadge'
import { QueryError, QuerySkeleton } from '../../components/ui'
import { parseApiDate } from '../../lib/date'

export function HeartbeatPage() {
  const config = useHeartbeatConfigQuery()
  const status = useHeartbeatStatusQuery()
  const fires = useHeartbeatFiresQuery()
  const triggerHeartbeat = useTriggerHeartbeatMutation()
  const queries = [config, status, fires]

  if (queries.some((query) => query.isError)) {
    const failed = queries.find((query) => query.isError)
    return (
      <div className="p-4 sm:p-6">
        <QueryError
          title="Heartbeat could not be loaded"
          error={failed?.error}
          onRetry={() =>
            void Promise.all(queries.map((query) => query.refetch()))
          }
        />
      </div>
    )
  }

  if (queries.some((query) => query.isLoading && query.data === undefined)) {
    return (
      <div className="p-4 sm:p-6">
        <QuerySkeleton rows={5} />
      </div>
    )
  }

  const heartbeatEnabled = config.data?.enabled === true

  return (
    <div className="space-y-6 overflow-auto p-4 sm:p-6">
      <div className="flex flex-col items-start justify-between gap-4 sm:flex-row sm:items-center">
        <div>
          <p className="text-sm font-medium text-blue-600">Runtime timer</p>
          <h1 className="mt-1 text-2xl font-semibold tracking-tight text-slate-950">
            Heartbeat
          </h1>
          <p className="mt-2 text-sm text-slate-500">
            View heartbeat settings, guidance file status, and recent heartbeat
            runs.
          </p>
        </div>
        <div className="text-right">
          <button
            type="button"
            className="inline-flex items-center gap-2 rounded-xl bg-blue-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-slate-300"
            disabled={!heartbeatEnabled || triggerHeartbeat.isPending}
            title={
              heartbeatEnabled
                ? 'Trigger a heartbeat now'
                : 'Heartbeat is disabled in runtime configuration'
            }
            onClick={() => triggerHeartbeat.mutate()}
          >
            <Play className="h-4 w-4" />
            {triggerHeartbeat.isPending ? 'Triggering' : 'Trigger'}
          </button>
          {!heartbeatEnabled ? (
            <p className="mt-2 max-w-xs text-xs text-slate-500">
              Enable heartbeat in runtime configuration before triggering a
              fire.
            </p>
          ) : null}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 sm:gap-4 xl:grid-cols-4">
        <Metric
          icon={RadioTower}
          label="Enabled"
          value={config.data?.enabled ? 'yes' : 'no'}
        />
        <Metric
          icon={Activity}
          label="Interval"
          value={`${config.data?.interval_seconds ?? 0}s`}
        />
        <Metric
          icon={FileText}
          label="Guidance"
          value={config.data?.guidance_file.exists ? 'found' : 'missing'}
        />
        <Metric
          icon={Activity}
          label="Next fire"
          value={formatDate(status.data?.next_fire_at)}
        />
      </div>

      <section className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <h2 className="text-sm font-semibold text-slate-950">
            Effective config
          </h2>
          <dl className="mt-4 grid grid-cols-1 gap-4 text-sm sm:grid-cols-2">
            <Detail
              label="Profile"
              value={`${config.data?.profile_name ?? 'unknown'} (${formatProfileSource(config.data?.profile_source)})`}
            />
            <Detail
              label="Dispatcher"
              value={config.data?.enabled ? 'enabled' : 'disabled'}
            />
            <Detail
              label="Guidance file"
              value={`${config.data?.guidance_file.exists ? 'found' : 'missing'} · ${config.data?.guidance_file.path ?? 'unknown'}`}
              wide
            />
            <Detail
              label="Prompt"
              value={`${config.data?.prompt ?? ''} (${formatPromptSource(config.data?.prompt_source)})`}
              wide
            />
          </dl>
        </div>
        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <h2 className="text-sm font-semibold text-slate-950">Last fire</h2>
          {status.data?.last_fire ? (
            <div className="mt-4 rounded-xl border border-slate-100 p-3 text-sm">
              <div className="flex items-center justify-between">
                <span className="mono text-xs text-slate-500">
                  {status.data.last_fire.id.slice(0, 10)}
                </span>
                <StatusBadge
                  status={mapFireStatus(
                    status.data.last_fire.status,
                    status.data.last_fire.run_status,
                  )}
                />
              </div>
              <p className="mt-2 text-slate-600">
                Run {status.data.last_fire.run_id?.slice(0, 10) ?? 'none'}
              </p>
              <p className="mt-1 text-xs text-slate-400">
                {formatDate(status.data.last_fire.created_at)}
              </p>
              {status.data.last_fire.error_message ? (
                <p className="mt-1 text-xs text-rose-600">
                  {status.data.last_fire.error_message}
                </p>
              ) : null}
            </div>
          ) : (
            <EmptyState
              title="No heartbeat fires"
              description={
                heartbeatEnabled
                  ? 'Heartbeat has not fired yet. Trigger it now to verify the runtime timer.'
                  : 'Heartbeat is disabled. Enable it in runtime configuration to start firing.'
              }
              action={
                heartbeatEnabled ? (
                  <button
                    type="button"
                    className="inline-flex items-center gap-2 rounded-xl bg-blue-600 px-3 py-2 text-sm font-semibold text-white disabled:bg-slate-300"
                    disabled={triggerHeartbeat.isPending}
                    onClick={() => triggerHeartbeat.mutate()}
                  >
                    <Play className="h-4 w-4" /> Trigger now
                  </button>
                ) : null
              }
            />
          )}
        </div>
      </section>

      <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <h2 className="text-sm font-semibold text-slate-950">Recent fires</h2>
        <div className="mt-4 space-y-2">
          {(fires.data?.fires ?? []).map((fire) => (
            <div
              key={fire.id}
              className="grid grid-cols-1 gap-2 rounded-xl border border-slate-100 p-3 text-sm sm:grid-cols-2 xl:grid-cols-[1fr_120px_160px_1fr] xl:items-center xl:gap-3"
            >
              <span className="mono text-xs text-slate-500">
                {fire.id.slice(0, 12)}
              </span>
              <StatusBadge
                status={mapFireStatus(fire.status, fire.run_status)}
              />
              <span className="text-xs text-slate-500">
                {formatDate(fire.created_at)}
              </span>
              <span className="truncate text-xs text-slate-500">
                Run {fire.run_id ?? 'none'}
              </span>
            </div>
          ))}
          {fires.data?.fires.length === 0 ? (
            <EmptyState
              title="No fires"
              description={
                heartbeatEnabled
                  ? 'No heartbeat history yet. Trigger a fire to verify the automation.'
                  : 'Heartbeat is disabled in runtime configuration.'
              }
              action={
                heartbeatEnabled ? (
                  <button
                    type="button"
                    className="inline-flex items-center gap-2 rounded-xl bg-blue-600 px-3 py-2 text-sm font-semibold text-white disabled:bg-slate-300"
                    disabled={triggerHeartbeat.isPending}
                    onClick={() => triggerHeartbeat.mutate()}
                  >
                    <Play className="h-4 w-4" /> Trigger now
                  </button>
                ) : null
              }
            />
          ) : null}
        </div>
      </section>
    </div>
  )
}

function Metric({
  icon: Icon,
  label,
  value,
}: {
  icon: typeof Activity
  label: string
  value: string
}) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="inline-flex rounded-xl bg-blue-50 p-2 text-blue-600">
        <Icon className="h-5 w-5" />
      </div>
      <p className="mt-4 text-sm text-slate-500">{label}</p>
      <p className="mt-1 truncate text-2xl font-semibold text-slate-950">
        {value}
      </p>
    </div>
  )
}

function Detail({
  label,
  value,
  wide,
}: {
  label: string
  value: string
  wide?: boolean
}) {
  return (
    <div className={wide ? 'sm:col-span-2' : ''}>
      <dt className="text-xs font-medium uppercase tracking-wide text-slate-400">
        {label}
      </dt>
      <dd className="mt-1 break-all text-slate-800">{value}</dd>
    </div>
  )
}

function formatDate(value?: string | null) {
  if (!value) return 'not scheduled'
  return parseApiDate(value).toLocaleString()
}

function mapFireStatus(status: string, runStatus?: string | null) {
  if (runStatus === 'failed') return 'failed'
  if (runStatus === 'cancelled') return 'cancelled'
  if (runStatus === 'completed') return 'completed'
  if (runStatus === 'queued' || runStatus === 'running') return 'running'
  if (status === 'failed') return 'failed'
  if (status === 'pending' || status === 'submitted') return 'running'
  return 'completed'
}

function formatProfileSource(source?: string) {
  if (source === 'heartbeat') return 'YA_CLAW_HEARTBEAT_PROFILE'
  return 'YA_CLAW_DEFAULT_PROFILE'
}

function formatPromptSource(source?: string) {
  if (source === 'heartbeat_setting') return 'YA_CLAW_HEARTBEAT_PROMPT'
  return source ?? 'unknown'
}
