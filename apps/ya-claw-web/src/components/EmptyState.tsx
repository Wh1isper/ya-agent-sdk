import type { LucideIcon } from 'lucide-react'

import { cn } from '../lib/utils'

export function EmptyState({
  title,
  description,
  icon: Icon,
  action,
  className,
  headingLevel = 3,
}: {
  title: string
  description?: string
  icon?: LucideIcon
  action?: React.ReactNode
  className?: string
  headingLevel?: 2 | 3
}) {
  const Heading = headingLevel === 2 ? 'h2' : 'h3'
  return (
    <div
      className={cn(
        'flex h-full min-h-48 flex-col items-center justify-center rounded-2xl border border-dashed border-slate-200 bg-white/70 p-8 text-center',
        className,
      )}
    >
      {Icon ? (
        <span className="mb-4 flex h-11 w-11 items-center justify-center rounded-2xl bg-slate-100 text-slate-500">
          <Icon className="h-5 w-5" />
        </span>
      ) : null}
      <Heading className="text-sm font-semibold text-slate-900">
        {title}
      </Heading>
      {description ? (
        <p className="mt-2 max-w-sm text-sm leading-6 text-slate-500">
          {description}
        </p>
      ) : null}
      {action ? <div className="mt-4">{action}</div> : null}
    </div>
  )
}
