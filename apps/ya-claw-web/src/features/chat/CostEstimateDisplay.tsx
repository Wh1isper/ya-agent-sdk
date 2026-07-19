import { CircleDollarSign } from 'lucide-react'

import { cn } from '../../lib/utils'
import type { CostEstimate, UsageSnapshot } from '../../types'
import { presentCostEstimate } from './usageCost'

export function CostEstimateDisplay({
  estimate,
  snapshot,
  compact = false,
  className,
}: {
  estimate?: CostEstimate | null
  snapshot?: UsageSnapshot | null
  compact?: boolean
  className?: string
}) {
  const presentation = presentCostEstimate(
    estimate ?? snapshot?.total_cost_estimate,
  )
  const tone =
    presentation.availability === 'complete'
      ? 'border-emerald-200 bg-emerald-50 text-emerald-800'
      : presentation.availability === 'partial'
        ? 'border-amber-200 bg-amber-50 text-amber-800'
        : 'border-slate-200 bg-slate-50 text-slate-600'

  if (compact) {
    return (
      <div
        className={cn(
          'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs',
          tone,
          className,
        )}
        title={`Estimated API list price · ${presentation.detail}`}
      >
        <CircleDollarSign className="h-3.5 w-3.5" aria-hidden="true" />
        <span className="font-medium">Estimated API list price</span>
        <span className="mono font-semibold">{presentation.amount}</span>
        {presentation.availability !== 'complete' ? (
          <span className="capitalize">{presentation.availability}</span>
        ) : null}
      </div>
    )
  }

  return (
    <div
      className={cn(
        'flex flex-wrap items-center justify-between gap-3 rounded-xl border px-3 py-2.5',
        tone,
        className,
      )}
    >
      <div className="flex min-w-0 items-center gap-2">
        <CircleDollarSign className="h-4 w-4 shrink-0" aria-hidden="true" />
        <div className="min-w-0">
          <p className="text-xs font-semibold">Estimated API list price</p>
          <p className="truncate text-[11px] opacity-80">
            {presentation.detail}
          </p>
        </div>
      </div>
      <div className="text-right">
        <p className="mono text-sm font-semibold">{presentation.amount}</p>
        <p className="text-[10px] font-medium uppercase tracking-wide opacity-75">
          {presentation.availability}
        </p>
      </div>
    </div>
  )
}
