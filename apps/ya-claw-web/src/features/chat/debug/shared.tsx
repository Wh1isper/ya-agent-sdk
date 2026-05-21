import { Bot, BrainCircuit } from 'lucide-react'
import type { ReactNode } from 'react'

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
  children: ReactNode
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
    const handoffKind =
      typeof part.metadata?.handoff_kind === 'string'
        ? part.metadata.handoff_kind
        : 'reminder'
    const handoffHint = getAgencyHandoffHint(handoffKind)
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
          <div className="mb-2 rounded-lg border border-violet-200 bg-white/70 px-2.5 py-2 text-xs text-violet-800">
            <div className="mb-1 flex items-center gap-2 font-semibold uppercase tracking-wide">
              <BrainCircuit className="h-3.5 w-3.5" />
              Agency {handoffKind.replace(/_/g, ' ')}
            </div>
            <div className="leading-5 text-violet-700">{handoffHint}</div>
          </div>
        ) : null}
        {part.text}
      </div>
    )
  }
  if (part.type === 'command' && part.name === 'agency_fire') {
    const payload = recordValue(part.params?.payload)
    const outputText = stringValue(payload?.output_text)
    const memory = recordValue(payload?.memory)
    const finalOutput = outputText ?? stringValue(memory?.output_text)

    if (finalOutput) {
      return (
        <div className="space-y-3 rounded-xl border border-violet-200 bg-violet-50 p-3 text-sm leading-7 text-slate-800">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-violet-700">
            <BrainCircuit className="h-3.5 w-3.5" />
            Agency fire · {stringValue(part.params?.kind) ?? 'observed output'}
          </div>
          <pre className="scrollbar-thin max-h-80 overflow-auto whitespace-pre-wrap rounded-lg bg-white p-3 text-sm leading-6 text-slate-800">
            {finalOutput}
          </pre>
        </div>
      )
    }
  }
  return <JsonView value={part} height="160px" />
}

function recordValue(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null
}

function stringValue(value: unknown) {
  return typeof value === 'string' && value.trim() ? value : null
}

function getAgencyHandoffHint(kind: string) {
  const hints: Record<string, string> = {
    context: 'Use this background context when it improves the next answer.',
    exchange: 'Use this cross-session context when it improves local judgment.',
    reminder: 'Use this timely nudge when it helps the current session.',
    task: 'Consider whether this should become a task, follow-up, or owner handoff.',
    risk: 'Review this before taking a sensitive or irreversible action.',
    async_result: 'Integrate this completed background work when useful.',
    decision: 'Align with this decision context or ask for confirmation.',
    conflict: 'Reconcile this conflicting context before acting.',
  }
  return hints[kind] ?? hints.reminder
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
