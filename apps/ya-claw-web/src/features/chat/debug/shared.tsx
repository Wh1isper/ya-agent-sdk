import { Bot, BrainCircuit } from 'lucide-react'

import { JsonView } from '../../../components/JsonView'
import type { InputPart } from '../../../types'
import { cn } from '../../../lib/utils'

export function Card({
  icon: Icon,
  title,
  accent,
  subtle,
  compact,
  children,
}: {
  icon: typeof Bot
  title: string
  accent: 'blue' | 'emerald' | 'amber' | 'rose' | 'violet' | 'slate'
  subtle?: boolean
  compact?: boolean
  children: React.ReactNode
}) {
  const accentClass = {
    blue: 'bg-blue-50 text-blue-600',
    emerald: 'bg-emerald-50 text-emerald-600',
    amber: 'bg-amber-50 text-amber-600',
    rose: 'bg-rose-50 text-rose-600',
    violet: 'bg-violet-50 text-violet-600',
    slate: 'bg-slate-100 text-slate-600',
  }[accent]

  return (
    <article
      className={cn(
        'rounded-2xl border border-slate-200 bg-white shadow-sm',
        subtle && 'bg-white/70',
        compact ? 'p-3' : 'p-4',
      )}
    >
      <div className="mb-3 flex items-center gap-2">
        <span
          className={cn(
            'inline-flex h-8 w-8 items-center justify-center rounded-xl',
            accentClass,
          )}
        >
          <Icon className="h-4 w-4" />
        </span>
        <h3 className="text-sm font-semibold text-slate-900">{title}</h3>
      </div>
      {children}
    </article>
  )
}

export function InputPartView({ part }: { part: InputPart }) {
  if (part.type === 'text') {
    const isAgencyHandoff = part.metadata?.source === 'agency_handoff'
    return (
      <div
        className={cn(
          'whitespace-pre-wrap rounded-xl p-3 text-sm leading-7 text-slate-800',
          isAgencyHandoff
            ? 'border border-violet-200 bg-violet-50'
            : 'bg-blue-50',
        )}
      >
        {isAgencyHandoff ? (
          <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-violet-700">
            <BrainCircuit className="h-3.5 w-3.5" />
            Agency handoff
          </div>
        ) : null}
        {part.text}
      </div>
    )
  }
  return <JsonView value={part} height="160px" />
}

export function CodeBlock({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="mb-1 text-xs font-medium uppercase tracking-wide text-slate-400">
        {label}
      </p>
      <pre className="scrollbar-thin max-h-60 overflow-auto rounded-xl border border-slate-200 bg-slate-50 p-3 text-xs leading-5 text-slate-700">
        {formatMaybeJson(value)}
      </pre>
    </div>
  )
}

function formatMaybeJson(value: string) {
  try {
    return JSON.stringify(JSON.parse(value), null, 2)
  } catch {
    return value
  }
}
