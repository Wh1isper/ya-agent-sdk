import { useVirtualizer } from '@tanstack/react-virtual'
import { ChevronRight } from 'lucide-react'
import { useRef, useState } from 'react'

import {
  darkPillClass,
  getStreamStatusTone,
  type StreamStatus,
  toneDotClass,
} from '../../../lib/status'
import { cn, safeJsonStringify } from '../../../lib/utils'
import type { AguiEvent } from '../../../types'
import {
  eventKey,
  eventNameLabel,
  eventTimestampLabel,
  eventTone,
  eventTypeLabel,
} from '../eventUtils'

export function EventDevToolsPanel({
  events,
  streamStatus,
  liveEventCount,
  loading,
  artifactsPruned,
}: {
  events: AguiEvent[]
  streamStatus: StreamStatus
  liveEventCount: number
  loading: boolean
  artifactsPruned: boolean
}) {
  const parentRef = useRef<HTMLDivElement | null>(null)
  const virtualizer = useVirtualizer({
    count: events.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 38,
    overscan: 8,
  })

  return (
    <aside className="flex h-full min-h-0 flex-col overflow-hidden border-l border-slate-200 bg-slate-950 text-slate-100">
      <div className="flex h-12 shrink-0 items-center justify-between border-b border-slate-800 px-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
            Event stream
          </p>
          <p className="mono text-[11px] text-slate-500">
            {events.length} events · {liveEventCount} live
          </p>
        </div>
        <span
          className={cn(
            'rounded-full border px-2 py-1 text-[11px] font-medium capitalize',
            darkPillClass(getStreamStatusTone(streamStatus)),
          )}
        >
          {streamStatus}
        </span>
      </div>
      <div
        ref={parentRef}
        className="scrollbar-thin min-h-0 flex-1 overscroll-contain overflow-auto p-2"
      >
        {loading ? (
          <div className="space-y-2 p-2">
            {Array.from({ length: 6 }).map((_, index) => (
              <div
                key={index}
                className="h-9 animate-pulse rounded bg-slate-900"
              />
            ))}
          </div>
        ) : null}
        {!loading && events.length === 0 ? (
          artifactsPruned ? (
            <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-3 text-xs leading-5 text-amber-200">
              Run replay artifacts have been pruned from disk. Database
              metadata, input parts, status, and summaries are still available.
            </div>
          ) : (
            <div className="rounded-xl border border-slate-800 bg-slate-900 p-3 text-xs text-slate-400">
              Select a run to inspect raw AGUI events.
            </div>
          )
        ) : null}
        {events.length > 0 ? (
          <div
            className="relative"
            style={{ height: `${virtualizer.getTotalSize()}px` }}
          >
            {virtualizer.getVirtualItems().map((item) => {
              const event = events[item.index]
              return (
                <div
                  key={`${item.index}:${eventKey(event)}`}
                  data-index={item.index}
                  ref={virtualizer.measureElement}
                  className="absolute left-0 top-0 w-full pb-1"
                  style={{ transform: `translateY(${item.start}px)` }}
                >
                  <EventRow event={event} index={item.index} />
                </div>
              )
            })}
          </div>
        ) : null}
      </div>
    </aside>
  )
}

export function EventRow({
  event,
  index,
}: {
  event: AguiEvent
  index: number
}) {
  const [expanded, setExpanded] = useState(false)
  const type = eventTypeLabel(event)
  const name = eventNameLabel(event)
  const timestamp = eventTimestampLabel(event)
  const tone = eventTone(event)

  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/80">
      <button
        type="button"
        className="flex w-full items-center gap-2 px-2 py-1.5 text-left text-xs hover:bg-slate-800/70"
        onClick={() => setExpanded((value) => !value)}
      >
        <ChevronRight
          className={cn(
            'h-3.5 w-3.5 shrink-0 text-slate-500 transition',
            expanded && 'rotate-90',
          )}
        />
        <span className="mono w-8 shrink-0 text-[11px] text-slate-500">
          {index + 1}
        </span>
        <span
          className={cn('h-2 w-2 shrink-0 rounded-full', toneDotClass(tone))}
        />
        <span className="mono min-w-0 flex-1 truncate text-[11px] text-slate-200">
          {type}
          {name ? <span className="text-slate-500"> · {name}</span> : null}
        </span>
        {timestamp ? (
          <span className="mono shrink-0 text-[10px] text-slate-500">
            {timestamp}
          </span>
        ) : null}
      </button>
      {expanded ? (
        <pre className="scrollbar-thin max-h-80 overflow-auto border-t border-slate-800 p-2 text-[11px] leading-5 text-slate-300">
          {safeJsonStringify(event)}
        </pre>
      ) : null}
    </div>
  )
}
