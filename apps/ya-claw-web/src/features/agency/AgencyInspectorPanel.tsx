import { useMemo } from 'react'

import { JsonView } from '../../components/JsonView'
import { StatusBadge } from '../../components/StatusBadge'
import { formatShortId } from '../../lib/utils'
import type {
  AgencyConfigResponse,
  AgencyFireSummary,
  AgencyStatusResponse,
  RunGetResponse,
  RunSummary,
  RunTraceItem,
  RunTraceResponse,
} from '../../types'
import { formatDate } from './utils'

export function AgencyInspectorPanel({
  config,
  status,
  fires,
  run,
  detail,
  trace,
  traceLoading,
}: {
  config?: AgencyConfigResponse
  status?: AgencyStatusResponse
  fires: AgencyFireSummary[]
  run: RunSummary | null
  detail: RunGetResponse | null
  trace: RunTraceResponse | null
  traceLoading: boolean
}) {
  const selectedRunFires = useMemo(() => {
    if (!run) return []
    return fires.filter(
      (fire) => fire.run_id === run.id || fire.active_run_id === run.id,
    )
  }, [fires, run])

  return (
    <aside className="flex h-full min-h-0 flex-col overflow-hidden border-l border-slate-200 bg-white">
      <div className="flex h-12 shrink-0 items-center justify-between border-b border-slate-200 px-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
            Agency inspector
          </p>
          <p className="mono text-[11px] text-slate-500">
            {run ? `run ${formatShortId(run.id, 12)}` : 'no run selected'}
          </p>
        </div>
        {run ? <StatusBadge status={run.status} /> : null}
      </div>
      <div className="scrollbar-thin min-h-0 flex-1 overflow-auto p-3">
        <div className="space-y-3">
          <PanelCard
            title="Run facts"
            subtitle={run ? `#${run.sequence_no}` : undefined}
          >
            {run ? (
              <dl className="space-y-2 text-xs">
                <InfoRow label="Run" value={run.id} mono />
                <InfoRow label="Trigger" value={run.trigger_type} />
                <InfoRow
                  label="Profile"
                  value={run.profile_name ?? 'default'}
                />
                <InfoRow label="Created" value={formatDate(run.created_at)} />
                <InfoRow label="Started" value={formatDate(run.started_at)} />
                <InfoRow
                  label="Committed"
                  value={formatDate(run.committed_at)}
                />
                <InfoRow
                  label="Termination"
                  value={run.termination_reason ?? 'pending'}
                />
              </dl>
            ) : (
              <p className="text-sm text-slate-500">Select a fire or run.</p>
            )}
          </PanelCard>

          <PanelCard
            title="Fires for run"
            subtitle={`${selectedRunFires.length} fires`}
          >
            {selectedRunFires.length === 0 ? (
              <p className="text-sm text-slate-500">
                No fire is linked to the selected run.
              </p>
            ) : null}
            <div className="space-y-2">
              {selectedRunFires.map((fire) => (
                <div
                  key={fire.id}
                  className="rounded-xl border border-slate-200 bg-slate-50 p-3"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <p className="mono truncate text-xs font-medium text-slate-700">
                        {fire.id}
                      </p>
                      <p className="mt-1 text-xs text-slate-500">
                        {fire.kind} · {formatDate(fire.created_at)}
                      </p>
                    </div>
                    <StatusBadge status={fire.status} />
                  </div>
                </div>
              ))}
            </div>
          </PanelCard>

          <PanelCard
            title="Tool trace"
            subtitle={`${trace?.item_count ?? 0} items`}
          >
            {traceLoading ? (
              <p className="text-sm text-slate-500">Loading trace...</p>
            ) : null}
            {!traceLoading && (!trace || trace.trace.length === 0) ? (
              <p className="text-sm text-slate-500">
                No tool trace for this run.
              </p>
            ) : null}
            {trace?.truncated ? (
              <p className="mb-2 rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-800">
                Trace is truncated.
              </p>
            ) : null}
            <div className="space-y-2">
              {trace?.trace.map((item) => (
                <TraceRow
                  key={`${item.sequence_no}-${item.type}`}
                  item={item}
                />
              ))}
            </div>
          </PanelCard>

          <PanelCard
            title="Singleton config"
            subtitle={config?.profile_name ?? 'default'}
          >
            <dl className="space-y-2 text-xs">
              <InfoRow
                label="Session"
                value={
                  config?.agency_session_id ??
                  status?.agency_session_id ??
                  'pending'
                }
                mono
              />
              <InfoRow
                label="Scope"
                value={config?.singleton_scope_key ?? 'agency:global'}
                mono
              />
              <InfoRow
                label="Source key"
                value={config?.singleton_source_session_id ?? 'pending'}
                mono
              />
              <InfoRow
                label="Active run"
                value={status?.active_run_id ?? 'none'}
                mono
              />
              <InfoRow
                label="Latest run"
                value={status?.latest_run_id ?? 'none'}
                mono
              />
              <InfoRow
                label="Action log"
                value={
                  config?.memory_files.action_log ?? 'agency/ACTION_LOG.md'
                }
                mono
              />
            </dl>
          </PanelCard>

          {detail?.run.metadata ? (
            <PanelCard title="Metadata" subtitle="run.metadata">
              <JsonView value={detail.run.metadata} height="260px" />
            </PanelCard>
          ) : null}
        </div>
      </div>
    </aside>
  )
}

function PanelCard({
  title,
  subtitle,
  children,
}: {
  title: string
  subtitle?: string
  children: React.ReactNode
}) {
  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="mb-3 flex items-start justify-between gap-3">
        <h3 className="text-sm font-semibold text-slate-950">{title}</h3>
        {subtitle ? (
          <span className="mono shrink-0 rounded-full bg-slate-100 px-2 py-1 text-[11px] text-slate-500">
            {subtitle}
          </span>
        ) : null}
      </div>
      {children}
    </section>
  )
}

function InfoRow({
  label,
  value,
  mono = false,
}: {
  label: string
  value: string
  mono?: boolean
}) {
  return (
    <div className="flex items-start justify-between gap-3">
      <dt className="shrink-0 text-slate-500">{label}</dt>
      <dd
        className={
          mono
            ? 'mono min-w-0 break-words text-right text-slate-800'
            : 'min-w-0 break-words text-right text-slate-800'
        }
      >
        {value}
      </dd>
    </div>
  )
}

function TraceRow({ item }: { item: RunTraceItem }) {
  return (
    <details className="rounded-xl border border-slate-200 bg-slate-50 p-3">
      <summary className="cursor-pointer text-xs font-medium text-slate-700">
        <span className="mono text-slate-400">#{item.sequence_no}</span>{' '}
        {item.type} · {item.tool_name ?? item.role ?? 'message'}
      </summary>
      <pre className="scrollbar-thin mt-3 max-h-56 overflow-auto rounded-lg bg-white p-3 text-xs leading-5 text-slate-700">
        {item.content ?? ''}
      </pre>
    </details>
  )
}
