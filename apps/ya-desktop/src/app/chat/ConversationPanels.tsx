import { Bot } from 'lucide-react'

import type { ClawRunSummary, ClawRunTraceResponse, ClawSessionDetail, ClawSessionSummary, ClawSessionTurn, JsonObject } from '../../claw'
import { cn } from '../../lib'
import type { HomeStreamStatus } from '../types'
import { collectCommittedReplayText, formatDate, inputPartsPreview } from '../utils'
import { EmptyState, RunRow } from '../ui'

export function SessionTurnsPanel({
  loading,
  error,
  liveOutput,
  liveStatus,
  turns,
  selectedSession,
  replayMessage,
  sessionDetail,
}: {
  loading: boolean
  error: Error | null
  liveOutput: string
  liveStatus: HomeStreamStatus
  replayMessage: JsonObject[] | null
  turns: ClawSessionTurn[]
  selectedSession: ClawSessionSummary | null
  sessionDetail: ClawSessionDetail | null
}) {
  if (!selectedSession) {
    return (
      <div className="flex min-h-[420px] items-center justify-center text-center">
        <div>
          <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-2xl bg-[#171717] text-white">
            <Bot className="h-5 w-5" />
          </div>
          <h3 className="mt-4 text-lg font-semibold text-[#171717]">
            Conversation surface
          </h3>
          <p className="mx-auto mt-2 max-w-sm text-sm leading-6 text-[#6b6b6b]">
            Select a chat or start from Home to focus on one thread of work.
          </p>
        </div>
      </div>
    )
  }

  if (loading)
    return (
      <EmptyState title="Loading turns" detail="Reading completed turns." />
    )
  if (error)
    return <EmptyState title="Could not load turns" detail={error.message} />

  const replayText = collectCommittedReplayText(sessionDetail, replayMessage)
  const hasLiveOutput = liveStatus !== 'idle' && liveOutput.length > 0
  if (turns.length === 0 && !replayText && !hasLiveOutput) {
    return (
      <EmptyState
        title="No completed turns"
        detail="Messages will appear after a successful run."
      />
    )
  }

  return (
    <div className="space-y-6">
      {replayText && (
        <TranscriptBubble
          label="Committed replay"
          role="assistant"
          text={replayText}
        />
      )}
      {turns.map((turn) => (
        <div key={turn.run_id ?? turn.runId} className="space-y-5">
          <TranscriptBubble
            label={`Turn ${turn.sequence_no ?? turn.sequenceNo ?? '—'} · ${formatDate(turn.created_at ?? turn.createdAt)}`}
            role="user"
            text={
              turn.input_preview ??
              turn.inputPreview ??
              inputPartsPreview(turn.input_parts ?? turn.inputParts)
            }
          />
          <TranscriptBubble
            label={formatDate(
              turn.committed_at ??
                turn.committedAt ??
                turn.created_at ??
                turn.createdAt,
            )}
            role="assistant"
            text={
              turn.output_text ??
              turn.outputText ??
              turn.output_summary ??
              turn.outputSummary ??
              'No output summary.'
            }
          />
        </div>
      ))}
      {hasLiveOutput && (
        <TranscriptBubble
          label="Streaming now"
          role="assistant"
          text={liveOutput}
        />
      )}
    </div>
  )
}

export function TranscriptBubble({
  label,
  role,
  text,
}: {
  label: string
  role: 'assistant' | 'user'
  text: string
}) {
  return (
    <div className={cn('flex', role === 'user' && 'justify-end')}>
      <article
        className={cn(
          'max-w-[88%] rounded-3xl px-4 py-3',
          role === 'assistant'
            ? 'bg-white text-[#171717]'
            : 'bg-[#f2f2ef] text-[#171717]',
        )}
      >
        <p className="mb-1 text-[11px] font-medium uppercase tracking-[0.14em] text-[#9a9a9a]">
          {label}
        </p>
        <p className="whitespace-pre-wrap text-sm leading-7">{text}</p>
      </article>
    </div>
  )
}

export function RunDetailsDisclosure({
  loading,
  error,
  onToggleOpen,
  open,
  runs,
  traces,
}: {
  loading: boolean
  error: Error | null
  onToggleOpen: (open: boolean) => void
  open: boolean
  runs: ClawRunSummary[]
  traces: ClawRunTraceResponse[]
}) {
  const traceItems = traces.flatMap((trace) => trace.trace ?? []).slice(0, 6)
  return (
    <details
      className="mt-8 rounded-2xl border border-black/[0.08] bg-[#fbfbfa] p-4 text-sm text-[#5f5f5f]"
      open={open}
      onToggle={(event) => onToggleOpen(event.currentTarget.open)}
    >
      <summary className="cursor-pointer font-medium text-[#171717]">
        Run details
      </summary>
      <div className="mt-4 grid gap-3 md:grid-cols-2">
        <div>
          <h4 className="mb-2 text-xs font-semibold uppercase tracking-[0.14em] text-[#8a8a8a]">
            Timeline
          </h4>
          <div className="space-y-2">
            {runs.slice(0, 5).map((run) => (
              <RunRow key={run.id} run={run} />
            ))}
            {runs.length === 0 && (
              <p className="text-xs text-[#8a8a8a]">No runs loaded.</p>
            )}
          </div>
        </div>
        <div>
          <h4 className="mb-2 text-xs font-semibold uppercase tracking-[0.14em] text-[#8a8a8a]">
            Trace
          </h4>
          {loading ? (
            <p className="text-xs text-[#8a8a8a]">Loading tool calls.</p>
          ) : error ? (
            <p className="text-xs text-amber-700">{error.message}</p>
          ) : traceItems.length === 0 ? (
            <p className="text-xs text-[#8a8a8a]">No tool calls loaded.</p>
          ) : (
            <div className="space-y-2">
              {traceItems.map((item, index) => (
                <div
                  key={`${item.tool_call_id ?? item.toolCallId ?? index}-${index}`}
                  className="rounded-xl bg-white p-3"
                >
                  <p className="text-xs font-medium text-[#171717]">
                    {item.type === 'tool_call' ? 'Tool call' : 'Tool response'}{' '}
                    ·{' '}
                    {item.tool_name ?? item.toolName ?? item.role ?? 'runtime'}
                  </p>
                  <p className="mt-1 line-clamp-3 text-xs leading-5 text-[#6b6b6b]">
                    {item.content ?? 'No trace content.'}
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </details>
  )
}
