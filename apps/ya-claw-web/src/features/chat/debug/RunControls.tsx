import {
  ArchiveX,
  Hash,
  PauseCircle,
  RefreshCcw,
  Square,
  TerminalSquare,
} from 'lucide-react'
import { toast } from 'sonner'

import {
  useRunControlMutations,
  useSessionSandboxMutations,
  useSubmitSessionInputMutation,
} from '../../../api/hooks'
import { StatusBadge } from '../../../components/StatusBadge'
import { cn, formatShortId } from '../../../lib/utils'
import type {
  RunSummary,
  SessionSummary,
  SessionWorkspaceState,
  WorkspaceRuntimeStatus,
} from '../../../types'
import type { SessionHistoryState } from '../sessionHistory'
import { SessionSandboxPill } from './SessionList'

export function RunStrip({
  runs,
  selectedRunId,
  history,
  loadingOlder,
  onLoadOlder,
  onSelectRun,
}: {
  runs: RunSummary[]
  selectedRunId: string | null
  history: SessionHistoryState
  loadingOlder: boolean
  onLoadOlder: () => void
  onSelectRun: (runId: string | null) => void
}) {
  return (
    <div className="flex h-16 shrink-0 items-center gap-3 overflow-hidden border-b border-slate-200 bg-white px-3 sm:px-4">
      <div className="shrink-0">
        <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
          Runs
        </p>
        <p className="text-[11px] text-slate-500">
          {history.loadedRunCount}/{history.totalRunCount || runs.length} loaded
        </p>
      </div>
      <div className="scrollbar-thin flex min-w-0 flex-1 gap-2 overflow-x-auto py-2">
        {history.hasMore ? (
          <button
            type="button"
            className="inline-flex shrink-0 items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 transition hover:border-blue-200 hover:bg-blue-50 hover:text-blue-700 disabled:opacity-60"
            onClick={onLoadOlder}
            disabled={loadingOlder}
          >
            {loadingOlder ? 'Loading...' : 'Older'}
          </button>
        ) : null}
        {runs.length === 0 ? (
          <span className="rounded-full border border-dashed border-slate-200 px-3 py-1.5 text-xs text-slate-400">
            No runs yet
          </span>
        ) : null}
        {runs.map((run) => (
          <button
            type="button"
            key={run.id}
            className={cn(
              'inline-flex shrink-0 items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-medium transition',
              selectedRunId === run.id
                ? 'border-blue-200 bg-blue-50 text-blue-700 shadow-sm ring-1 ring-blue-100'
                : 'border-slate-200 bg-white text-slate-600 hover:border-blue-200 hover:bg-blue-50/60',
            )}
            onClick={() => onSelectRun(run.id)}
          >
            <Hash className="h-3 w-3" />
            {run.sequence_no}
            <span className="capitalize">{run.status}</span>
          </button>
        ))}
      </div>
    </div>
  )
}

export function WorkspaceStatusBar({
  runtime,
  sessionId,
  state,
}: {
  runtime: WorkspaceRuntimeStatus | null
  sessionId: string | null
  state: SessionWorkspaceState | null
}) {
  const sandbox = state?.sandbox_state ?? null
  const controls = useSessionSandboxMutations(sessionId)
  const canPrepare = Boolean(
    sessionId &&
      runtime?.capabilities.sandbox_prepare &&
      sandbox?.ready_state !== 'ready',
  )
  const canStop = Boolean(
    sessionId && runtime?.capabilities.sandbox_stop && sandbox?.container_id,
  )

  return (
    <div className="flex shrink-0 flex-col gap-2 border-b border-slate-200 bg-blue-50/60 px-3 py-2 text-xs text-blue-950 sm:flex-row sm:items-center sm:justify-between sm:gap-3 sm:px-4">
      <div className="flex min-w-0 items-center gap-2 font-medium">
        <TerminalSquare className="h-3.5 w-3.5" />
        <span>Workspace</span>
        <span className="rounded-full bg-white/80 px-2 py-0.5 text-blue-700">
          {runtime?.backend ?? sandbox?.backend ?? 'unknown'}
        </span>
        <span className="truncate text-blue-700">
          {state?.binding?.cwd ??
            sandbox?.work_dir ??
            runtime?.workspace.virtual_path ??
            'workspace'}
        </span>
      </div>
      <div className="flex shrink-0 flex-wrap items-center justify-end gap-2">
        <SessionSandboxPill sandbox={sandbox} />
        {sandbox?.container_id ? (
          <span className="mono rounded-full bg-white/80 px-2 py-0.5 text-blue-700">
            {formatShortId(sandbox.container_id, 12)}
          </span>
        ) : null}
        {canPrepare ? (
          <button
            type="button"
            className="rounded-full border border-blue-200 bg-white px-2 py-0.5 font-medium text-blue-700 transition hover:bg-blue-50 disabled:opacity-60"
            onClick={() => controls.prepare.mutate()}
            disabled={controls.prepare.isPending}
          >
            Prepare
          </button>
        ) : null}
        {canStop ? (
          <button
            type="button"
            className="rounded-full border border-slate-200 bg-white px-2 py-0.5 font-medium text-slate-700 transition hover:bg-slate-50 disabled:opacity-60"
            onClick={() => controls.stop.mutate()}
            disabled={controls.stop.isPending}
          >
            Stop
          </button>
        ) : null}
      </div>
    </div>
  )
}

export function MemoryStatusBar({
  session,
}: {
  session: SessionSummary | null
}) {
  const memory = session?.memory_state
  if (!session || !memory) return null

  return (
    <div className="flex shrink-0 flex-col gap-2 border-b border-slate-200 bg-violet-50/60 px-3 py-2 text-xs text-violet-900 sm:flex-row sm:items-center sm:justify-between sm:gap-3 sm:px-4">
      <div className="flex items-center gap-2 font-medium">
        <ArchiveX className="h-3.5 w-3.5" />
        <span>Memory</span>
      </div>
      <div className="flex flex-wrap items-center justify-end gap-2">
        <span>{memory.extract_count} extracts</span>
        <span>{memory.turns_since_extract} turns since extract</span>
        <span>{memory.extracts_since_summary} extracts since summary</span>
        {memory.pending_extract ? (
          <span className="rounded-full bg-amber-100 px-2 py-0.5 text-amber-700">
            extract pending
          </span>
        ) : null}
        {memory.pending_summary ? (
          <span className="rounded-full bg-amber-100 px-2 py-0.5 text-amber-700">
            summary pending
          </span>
        ) : null}
      </div>
    </div>
  )
}

export function RunControlBar({
  sessionId,
  run,
  onSelectRun,
}: {
  sessionId: string | null
  run: RunSummary | null
  onSelectRun: (runId: string | null) => void
}) {
  const runControls = useRunControlMutations(run?.id ?? null)
  const submitInput = useSubmitSessionInputMutation(sessionId)
  if (!run) return null

  const isActive = run.status === 'queued' || run.status === 'running'
  const canRecover =
    run.status === 'failed' &&
    Boolean(sessionId) &&
    Boolean(run.input_parts?.length)

  async function recover(mode: 'retry' | 'reset_and_retry') {
    if (!run || !sessionId || !run.input_parts?.length) return
    try {
      const response = await submitInput.mutateAsync({
        input_parts: run.input_parts,
        reset_state: mode === 'reset_and_retry',
        metadata: {
          recovery: {
            mode,
            source_run_id: run.id,
            source_sequence_no: run.sequence_no,
            reason: 'web_ui',
          },
        },
      })
      onSelectRun(response.run_id)
      toast.success(
        mode === 'retry' ? 'Retry submitted' : 'Reset and retry submitted',
      )
    } catch (error) {
      toast.error(
        error instanceof Error
          ? error.message
          : 'Failed to submit recovery run',
      )
    }
  }

  if (!isActive && !canRecover) return null

  return (
    <div className="flex flex-col gap-3 border-b border-slate-200 bg-white px-3 py-3 sm:flex-row sm:items-center sm:justify-between sm:px-4">
      <div className="flex items-center gap-2 text-sm text-slate-600">
        <StatusBadge status={run.status} />
        <span className="mono text-xs">{formatShortId(run.id, 12)}</span>
        {canRecover ? (
          <span className="text-xs text-rose-600">Run failed</span>
        ) : null}
      </div>
      <div className="flex items-center gap-2">
        {isActive ? (
          <>
            <button
              type="button"
              className="inline-flex items-center gap-2 rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-xs font-medium text-amber-700 transition hover:bg-amber-100 disabled:opacity-60"
              onClick={() => runControls.interrupt.mutate()}
              disabled={runControls.interrupt.isPending}
            >
              <PauseCircle className="h-3.5 w-3.5" />
              Interrupt
            </button>
            <button
              type="button"
              className="inline-flex items-center gap-2 rounded-xl border border-rose-200 bg-rose-50 px-3 py-2 text-xs font-medium text-rose-700 transition hover:bg-rose-100 disabled:opacity-60"
              onClick={() => runControls.cancel.mutate()}
              disabled={runControls.cancel.isPending}
            >
              <Square className="h-3.5 w-3.5" />
              Cancel
            </button>
          </>
        ) : null}
        {canRecover ? (
          <>
            <button
              type="button"
              className="inline-flex items-center gap-2 rounded-xl border border-blue-200 bg-blue-50 px-3 py-2 text-xs font-medium text-blue-700 transition hover:bg-blue-100 disabled:opacity-60"
              onClick={() => void recover('retry')}
              disabled={submitInput.isPending}
            >
              <RefreshCcw className="h-3.5 w-3.5" />
              Retry
            </button>
            <button
              type="button"
              className="inline-flex items-center gap-2 rounded-xl border border-rose-200 bg-rose-50 px-3 py-2 text-xs font-medium text-rose-700 transition hover:bg-rose-100 disabled:opacity-60"
              onClick={() => void recover('reset_and_retry')}
              disabled={submitInput.isPending}
            >
              <ArchiveX className="h-3.5 w-3.5" />
              Reset and retry
            </button>
          </>
        ) : null}
      </div>
    </div>
  )
}
