import {
  ArchiveX,
  BrainCircuit,
  Clock3,
  FileText,
  Gauge,
  Hash,
  TerminalSquare,
} from 'lucide-react'

import { StatusBadge } from '../../components/StatusBadge'
import { cn, formatShortId } from '../../lib/utils'
import type { AgencyConfigResponse, AgencyStatusResponse } from '../../types'
import { formatDate, formatDuration } from './utils'

export function AgencyStatusBar({
  config,
  status,
}: {
  config?: AgencyConfigResponse
  status?: AgencyStatusResponse
}) {
  const enabled = config?.enabled ?? status?.enabled ?? false
  const sessionId = status?.agency_session_id ?? config?.agency_session_id
  return (
    <div className="flex shrink-0 flex-col gap-2 border-b border-slate-200 bg-blue-50/60 px-3 py-2 text-xs text-blue-950 sm:flex-row sm:items-center sm:justify-between sm:gap-3 sm:px-4">
      <div className="flex min-w-0 items-center gap-2 font-medium">
        <BrainCircuit className="h-3.5 w-3.5" />
        <span>Agency</span>
        <StatusBadge status={enabled ? 'active' : 'disabled'} />
        <StatusBadge status={status?.state ?? 'idle'} />
        <span className="truncate text-blue-700">
          {sessionId ? formatShortId(sessionId, 12) : 'session pending'}
        </span>
      </div>
      <div className="flex shrink-0 flex-wrap items-center justify-end gap-2">
        <Pill
          icon={Hash}
          label={`${status?.pending_fire_count ?? 0} pending`}
        />
        <Pill icon={Gauge} label={config?.profile_name ?? 'default'} />
        <Pill
          icon={Clock3}
          label={`next ${formatDate(status?.next_fire_at ?? config?.next_fire_at)}`}
        />
      </div>
    </div>
  )
}

export function AgencyConfigBar({ config }: { config?: AgencyConfigResponse }) {
  if (!config) return null
  return (
    <div className="flex shrink-0 flex-col gap-2 border-b border-slate-200 bg-violet-50/60 px-3 py-2 text-xs text-violet-900 sm:flex-row sm:items-center sm:justify-between sm:gap-3 sm:px-4">
      <div className="flex min-w-0 items-center gap-2 font-medium">
        <ArchiveX className="h-3.5 w-3.5" />
        <span>Singleton session</span>
        <span className="mono truncate rounded-full bg-white/80 px-2 py-0.5 text-violet-700">
          {config.singleton_scope_key}
        </span>
      </div>
      <div className="flex shrink-0 flex-wrap items-center justify-end gap-2">
        <Pill
          icon={Clock3}
          label={formatDuration(config.timer_interval_seconds)}
        />
        <Pill
          icon={Gauge}
          label={`risk ${config.risk_policy.max_auto_action_risk}`}
        />
        <Pill
          icon={FileText}
          label={config.memory_files.index ?? 'AGENCY.md'}
          mono
        />
      </div>
    </div>
  )
}

function Pill({
  icon: Icon,
  label,
  mono,
}: {
  icon: typeof TerminalSquare
  label: string
  mono?: boolean
}) {
  return (
    <span
      className={cn(
        'inline-flex min-w-0 items-center gap-1 rounded-full bg-white/80 px-2 py-0.5 font-medium',
        mono && 'mono',
      )}
      title={label}
    >
      <Icon className="h-3 w-3 shrink-0" />
      <span className="truncate">{label}</span>
    </span>
  )
}
