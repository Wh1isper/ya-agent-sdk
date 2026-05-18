import { CheckCircle2, Clock3, PlayCircle, XCircle } from 'lucide-react'

import {
  getStreamStatusTone,
  lightPillClass,
  type StreamStatus,
} from '../../../lib/status'
import { cn } from '../../../lib/utils'

export function LivePill({
  status,
  eventCount,
}: {
  status: StreamStatus
  eventCount: number
}) {
  const icon =
    status === 'streaming'
      ? PlayCircle
      : status === 'error'
        ? XCircle
        : status === 'closed'
          ? CheckCircle2
          : Clock3
  const Icon = icon
  return (
    <span
      className={cn(
        'inline-flex items-center gap-2 rounded-full border px-3 py-1.5 font-medium capitalize',
        lightPillClass(getStreamStatusTone(status)),
      )}
      aria-live="polite"
    >
      <Icon className="h-3.5 w-3.5" aria-hidden />
      {status} · {eventCount} live
    </span>
  )
}
