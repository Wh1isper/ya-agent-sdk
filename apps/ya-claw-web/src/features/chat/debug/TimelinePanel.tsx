import {
  Activity,
  ArchiveX,
  Bot,
  CheckCircle2,
  Clock3,
  FilePenLine,
  Files,
  MessageSquare,
  Send,
  TerminalSquare,
  User,
  Wrench,
} from 'lucide-react'
import { useEffect, useRef } from 'react'

import { EmptyState } from '../../../components/EmptyState'
import { JsonView } from '../../../components/JsonView'
import { StatusBadge } from '../../../components/StatusBadge'
import type { AguiTimelineState, TimelineBlock } from '../agui/types'
import type { SessionHistoryState } from '../sessionHistory'
import { Card, InputPartView, CodeBlock } from './shared'
import { MarkdownMessage } from './MarkdownMessage'
import { accentFromRuntimeStatus } from './runtimeStatus'

export function TimelinePanel({
  timeline,
  loading,
  artifactsPruned,
  history,
  loadingOlder,
  onLoadOlder,
  historyLoadingDisabled = false,
}: {
  timeline: AguiTimelineState
  loading: boolean
  artifactsPruned: boolean
  history: SessionHistoryState
  loadingOlder: boolean
  onLoadOlder: () => Promise<unknown>
  historyLoadingDisabled?: boolean
}) {
  const scrollRef = useRef<HTMLElement | null>(null)
  const bottomRef = useRef<HTMLDivElement | null>(null)
  const stickToBottomRef = useRef(true)
  const previousScrollHeightRef = useRef<number | null>(null)
  const toolCallCount = timeline.blocks.filter(
    (block) => block.kind === 'tool_call',
  ).length
  const assistantCount = timeline.blocks.filter(
    (block) => block.kind === 'assistant_message',
  ).length
  useEffect(() => {
    const element = scrollRef.current
    if (!element) return
    const previousHeight = previousScrollHeightRef.current
    if (previousHeight != null) {
      element.scrollTop = element.scrollHeight - previousHeight
      previousScrollHeightRef.current = null
      return
    }
    if (!stickToBottomRef.current) return
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [timeline.blocks.length])

  async function loadOlder() {
    const element = scrollRef.current
    if (!element || loadingOlder || !history.hasMore || historyLoadingDisabled)
      return
    previousScrollHeightRef.current = element.scrollHeight
    stickToBottomRef.current = false
    await onLoadOlder()
  }

  return (
    <section
      ref={scrollRef}
      className="scrollbar-thin min-h-0 flex-1 overscroll-contain overflow-auto bg-slate-50 p-3 sm:p-5"
      onScroll={() => {
        const element = scrollRef.current
        if (!element) return
        const distanceFromBottom =
          element.scrollHeight - element.scrollTop - element.clientHeight
        stickToBottomRef.current = distanceFromBottom < 160
      }}
    >
      <div className="mx-auto mb-4 flex max-w-4xl flex-col gap-3 rounded-2xl border border-slate-200 bg-white/80 px-4 py-3 shadow-sm backdrop-blur sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="text-sm font-semibold text-slate-900">Runtime replay</p>
          <p className="mt-0.5 text-xs text-slate-500">
            {timeline.blocks.length} blocks · {assistantCount} assistant
            messages · {toolCallCount} tool calls
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-500">
            {history.loadedRunCount}/
            {history.totalRunCount || history.loadedRunCount} runs loaded
          </span>
          <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-500">
            Auto-scroll
          </span>
        </div>
      </div>
      {loading ? <TimelineSkeleton /> : null}
      {!loading && timeline.blocks.length === 0 ? (
        artifactsPruned ? (
          <PrunedArtifactsNotice />
        ) : (
          <EmptyState
            icon={MessageSquare}
            title="No replay yet"
            description="Select a run with committed AGUI messages or start a new debug turn."
            className="mx-auto max-w-4xl bg-white"
          />
        )
      ) : null}
      <div className="mx-auto max-w-4xl space-y-4">
        {!loading && timeline.blocks.length > 0 && !historyLoadingDisabled ? (
          <DebugHistoryBoundary
            history={history}
            loadingOlder={loadingOlder}
            onLoadOlder={() => void loadOlder()}
          />
        ) : null}
        {timeline.blocks.map((block) => (
          <TimelineCard key={block.id} block={block} />
        ))}
        <div ref={bottomRef} />
      </div>
    </section>
  )
}

export function DebugHistoryBoundary({
  history,
  loadingOlder,
  onLoadOlder,
}: {
  history: SessionHistoryState
  loadingOlder: boolean
  onLoadOlder: () => void
}) {
  if (history.hasMore) {
    return (
      <button
        type="button"
        className="mx-auto flex items-center justify-center rounded-full border border-slate-200 bg-white px-4 py-2 text-xs font-medium text-slate-600 shadow-sm transition hover:border-blue-200 hover:bg-blue-50 hover:text-blue-700 disabled:opacity-60"
        onClick={onLoadOlder}
        disabled={loadingOlder}
      >
        {loadingOlder
          ? 'Loading older runs...'
          : `Load older runs · ${history.loadedRunCount}/${history.totalRunCount}`}
      </button>
    )
  }
  return (
    <div className="mx-auto w-fit rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-500">
      Beginning of session
    </div>
  )
}

export function PrunedArtifactsNotice() {
  return (
    <div className="mx-auto max-w-4xl">
      <Card icon={ArchiveX} title="Replay artifacts pruned" accent="amber">
        <div className="space-y-2 text-sm leading-6 text-slate-700">
          <p>
            The raw AGUI replay for this run has been pruned from disk to reduce
            storage usage.
          </p>
          <p className="text-slate-500">
            YA Claw still keeps the run database row, input parts, status,
            output text, and compact summary when available.
          </p>
        </div>
      </Card>
    </div>
  )
}

export function TimelineSkeleton() {
  return (
    <div className="mx-auto max-w-4xl space-y-4">
      {Array.from({ length: 3 }).map((_, index) => (
        <div
          key={index}
          className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm"
        >
          <div className="h-4 w-32 animate-pulse rounded bg-slate-100" />
          <div className="mt-4 h-16 animate-pulse rounded bg-slate-100" />
        </div>
      ))}
    </div>
  )
}

export function TimelineCard({ block }: { block: TimelineBlock }) {
  if (block.kind === 'user_input') {
    return (
      <Card icon={User} title="User input" accent="blue">
        <div className="space-y-2">
          {block.parts.map((part, index) => (
            <InputPartView key={index} part={part} />
          ))}
        </div>
      </Card>
    )
  }
  if (block.kind === 'assistant_message') {
    return (
      <Card
        icon={Bot}
        title={block.name ? `Assistant · ${block.name}` : 'Assistant'}
        accent="emerald"
      >
        <MarkdownMessage content={block.content} />
      </Card>
    )
  }
  if (block.kind === 'reasoning') {
    return (
      <Card icon={Activity} title="Reasoning" accent="violet" subtle>
        <div className="whitespace-pre-wrap text-sm leading-7 text-slate-700">
          {block.content}
        </div>
      </Card>
    )
  }
  if (block.kind === 'tool_call') {
    return (
      <Card
        icon={Wrench}
        title={block.name ?? 'Tool call'}
        accent={block.status === 'failed' ? 'rose' : 'amber'}
      >
        <div className="space-y-3">
          <StatusBadge status={block.status} />
          {block.args ? (
            <CodeBlock label="Arguments" value={block.args} />
          ) : null}
          {block.result ? (
            <CodeBlock label="Result" value={block.result} />
          ) : null}
        </div>
      </Card>
    )
  }
  if (block.kind === 'task_board') {
    return (
      <Card icon={CheckCircle2} title="Task board" accent="blue">
        <div className="grid gap-2">
          {block.tasks.map((task) => (
            <div
              key={task.id}
              className="rounded-xl border border-slate-200 bg-slate-50 p-3"
            >
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="text-sm font-medium text-slate-900">
                    {task.subject}
                  </p>
                  {task.active_form ? (
                    <p className="mt-1 text-xs text-slate-500">
                      {task.active_form}
                    </p>
                  ) : null}
                </div>
                <StatusBadge status={task.status} />
              </div>
            </div>
          ))}
          {block.tasks.length === 0 ? (
            <p className="text-sm text-slate-500">No tasks in snapshot.</p>
          ) : null}
        </div>
      </Card>
    )
  }
  if (block.kind === 'context_meter') {
    const percent =
      block.contextWindowSize > 0
        ? Math.min(
            100,
            Math.round((block.totalTokens / block.contextWindowSize) * 100),
          )
        : 0
    return (
      <Card icon={Clock3} title="Context" accent="amber" compact>
        <div className="flex items-center gap-3">
          <div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-100">
            <div
              className="h-full rounded-full bg-amber-500"
              style={{ width: `${percent}%` }}
            />
          </div>
          <span className="mono text-xs text-slate-600">
            {block.totalTokens} / {block.contextWindowSize}
          </span>
        </div>
      </Card>
    )
  }
  if (block.kind === 'subagent') {
    return (
      <Card
        icon={Bot}
        title={`Subagent · ${block.agentName}`}
        accent={block.status === 'failed' ? 'rose' : 'violet'}
      >
        <StatusBadge status={block.status} />
        {block.promptPreview ? (
          <p className="mt-3 text-sm text-slate-600">{block.promptPreview}</p>
        ) : null}
        {block.resultPreview ? (
          <p className="mt-3 text-sm text-slate-800">{block.resultPreview}</p>
        ) : null}
        {block.error ? (
          <p className="mt-3 text-sm text-rose-700">{block.error}</p>
        ) : null}
      </Card>
    )
  }
  if (block.kind === 'file_change') {
    return (
      <Card
        icon={Files}
        title={
          block.toolName ? `File changes · ${block.toolName}` : 'File changes'
        }
        accent="emerald"
      >
        <JsonView value={block.changes} height="260px" />
      </Card>
    )
  }
  if (block.kind === 'note_snapshot') {
    return (
      <Card icon={FilePenLine} title="Notes" accent="blue">
        <JsonView value={block.entries} height="220px" />
      </Card>
    )
  }
  if (block.kind === 'steering') {
    return (
      <Card
        icon={Send}
        title={block.title}
        accent={block.status === 'injected' ? 'emerald' : 'blue'}
      >
        <div className="space-y-3">
          <div className="flex flex-wrap items-center gap-2 text-xs font-medium text-slate-500">
            <span className="rounded-full bg-slate-100 px-2 py-0.5">
              {block.status}
            </span>
            {block.delivery ? (
              <span className="rounded-full bg-slate-100 px-2 py-0.5">
                {block.delivery}
              </span>
            ) : null}
          </div>
          {block.inputParts.length > 0 ? (
            <div className="space-y-2">
              {block.inputParts.map((part, index) => (
                <InputPartView key={index} part={part} />
              ))}
            </div>
          ) : null}
          {typeof block.prompt === 'string' ? (
            <CodeBlock
              label={
                block.status === 'injected'
                  ? 'Injected prompt'
                  : 'Delivered prompt'
              }
              value={block.prompt}
            />
          ) : block.prompt !== undefined ? (
            <JsonView value={block.prompt} height="180px" />
          ) : null}
        </div>
      </Card>
    )
  }
  if (block.kind === 'usage') {
    return (
      <Card icon={Activity} title="Usage" accent="violet" compact>
        <JsonView value={block.payload} height="180px" />
      </Card>
    )
  }
  if (block.kind === 'runtime_event') {
    return (
      <Card
        icon={TerminalSquare}
        title={block.title}
        accent={accentFromRuntimeStatus(block.status)}
        compact
      >
        <JsonView value={block.payload} height="180px" />
      </Card>
    )
  }
  return (
    <Card icon={MessageSquare} title={block.name} accent="slate" compact>
      <JsonView value={block.payload} height="180px" />
    </Card>
  )
}
