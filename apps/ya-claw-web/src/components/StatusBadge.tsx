import { cn } from '../lib/utils'
import type { RunStatus, SessionStatus } from '../types'

const statusClass: Record<string, string> = {
  idle: 'border-slate-200 bg-slate-50 text-slate-600',
  queued: 'border-blue-200 bg-blue-50 text-blue-700',
  running: 'border-amber-200 bg-amber-50 text-amber-700',
  completed: 'border-emerald-200 bg-emerald-50 text-emerald-700',
  failed: 'border-rose-200 bg-rose-50 text-rose-700',
  cancelled: 'border-slate-200 bg-slate-100 text-slate-600',
  active: 'border-emerald-200 bg-emerald-50 text-emerald-700',
  paused: 'border-slate-200 bg-slate-100 text-slate-600',
  deleted: 'border-slate-200 bg-slate-100 text-slate-600',
  enabled: 'border-emerald-200 bg-emerald-50 text-emerald-700',
  disabled: 'border-slate-200 bg-slate-100 text-slate-600',
}

export function StatusBadge({
  status,
  className,
}: {
  status: RunStatus | SessionStatus | string
  className?: string
}) {
  return (
    <span
      className={cn(
        'inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-medium capitalize',
        statusClass[status] ?? 'border-slate-200 bg-slate-50 text-slate-600',
        className,
      )}
    >
      {status.replace(/_/g, ' ')}
    </span>
  )
}
