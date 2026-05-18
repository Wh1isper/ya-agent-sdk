import { Send, Wrench } from 'lucide-react'

import { cn } from '../../lib/utils'
import type { RunSummary, SessionSummary } from '../../types'
import { sessionLabel } from './utils'

export function ManualFireComposer({
  sessions,
  selectedSourceSessionId,
  prompt,
  activeRun,
  pending,
  onSourceSessionChange,
  onPromptChange,
  onSubmit,
}: {
  sessions: SessionSummary[]
  selectedSourceSessionId: string
  prompt: string
  activeRun: RunSummary | null
  pending: boolean
  onSourceSessionChange: (value: string) => void
  onPromptChange: (value: string) => void
  onSubmit: () => Promise<void>
}) {
  const active = activeRun?.status === 'running'
  const queued = activeRun?.status === 'queued'
  return (
    <div className="border-t border-slate-200 bg-white p-3 sm:p-4">
      <div className="mx-auto max-w-4xl">
        {activeRun ? (
          <div
            className={cn(
              'mb-3 rounded-2xl border px-4 py-3 text-sm',
              active
                ? 'border-blue-200 bg-blue-50 text-blue-800'
                : 'border-amber-200 bg-amber-50 text-amber-800',
            )}
          >
            {active
              ? 'Agency is running. This fire will steer the active run.'
              : queued
                ? 'Agency run is queued. This fire will be merged or kept pending by the backend.'
                : 'Manual fires wake the singleton agency session.'}
          </div>
        ) : null}
        <div className="rounded-3xl border border-slate-200 bg-white p-3 shadow-sm ring-1 ring-slate-100 transition focus-within:border-blue-200 focus-within:ring-blue-100">
          <textarea
            className="max-h-48 min-h-24 w-full resize-none rounded-2xl border-0 p-2 text-sm leading-6 text-slate-900 outline-none placeholder:text-slate-400 disabled:bg-white disabled:text-slate-400"
            value={prompt}
            onChange={(event) => onPromptChange(event.target.value)}
            placeholder="Optional agency instruction..."
            onKeyDown={(event) => {
              if (
                !pending &&
                (event.metaKey || event.ctrlKey) &&
                event.key === 'Enter'
              ) {
                event.preventDefault()
                void onSubmit()
              }
            }}
          />
          <div className="flex flex-col gap-3 border-t border-slate-100 pt-3 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex min-w-0 flex-1 items-center gap-2">
              <select
                className="max-w-72 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-700 outline-none ring-blue-600 focus:ring-2"
                value={selectedSourceSessionId}
                onChange={(event) => onSourceSessionChange(event.target.value)}
              >
                <option value="">Global fire</option>
                {sessions.map((session) => (
                  <option key={session.id} value={session.id}>
                    {sessionLabel(session)}
                  </option>
                ))}
              </select>
              <span className="hidden text-xs text-slate-400 lg:inline">
                Cmd/Ctrl + Enter to fire
              </span>
            </div>
            <button
              type="button"
              className={cn(
                'inline-flex items-center gap-2 rounded-xl px-4 py-2 text-sm font-semibold text-white shadow-sm transition disabled:bg-slate-300',
                active
                  ? 'bg-amber-600 hover:bg-amber-700'
                  : 'bg-blue-600 hover:bg-blue-700',
              )}
              disabled={pending}
              onClick={() => void onSubmit()}
            >
              {active ? (
                <Wrench className="h-4 w-4" />
              ) : (
                <Send className="h-4 w-4" />
              )}
              {pending ? 'Firing' : active ? 'Steer agency' : 'Fire agency'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
