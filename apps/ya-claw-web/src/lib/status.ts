import type { AguiEvent, RunStatus, SessionStatus } from '../types'

export type Tone =
  | 'neutral'
  | 'info'
  | 'running'
  | 'success'
  | 'warning'
  | 'error'
export type StreamStatus =
  | 'idle'
  | 'connecting'
  | 'streaming'
  | 'closed'
  | 'error'

export function getBackendTone(options: {
  isError: boolean
  status?: string | null
}): 'ok' | 'pending' | 'error' {
  if (options.isError) return 'error'
  if (options.status === 'ok') return 'ok'
  return 'pending'
}

export function getNotificationTone(
  status: string,
): 'ok' | 'pending' | 'error' {
  if (status === 'connected') return 'ok'
  if (status === 'error') return 'error'
  return 'pending'
}

export function getRunStatusTone(
  status: RunStatus | SessionStatus | string,
): Tone {
  if (status === 'completed' || status === 'enabled') return 'success'
  if (status === 'running') return 'running'
  if (status === 'queued') return 'info'
  if (status === 'failed') return 'error'
  if (status === 'cancelled' || status === 'disabled') return 'neutral'
  return 'neutral'
}

export function getStreamStatusTone(status: StreamStatus): Tone {
  if (status === 'streaming') return 'success'
  if (status === 'connecting') return 'warning'
  if (status === 'error') return 'error'
  return 'neutral'
}

export function getEventTone(event: AguiEvent, label: string): Tone {
  const normalized = label.toLowerCase()
  if (normalized.includes('error') || normalized.includes('failed'))
    return 'error'
  if (normalized.includes('interrupt') || normalized.includes('cancel'))
    return 'warning'
  if (normalized.includes('finished') || normalized.includes('complete')) {
    return 'success'
  }
  if (normalized.includes('start') || normalized.includes('running')) {
    return 'running'
  }
  if (typeof event.type === 'string' && event.type.includes('TOOL'))
    return 'info'
  return 'neutral'
}

export function toneDotClass(tone: Tone) {
  return {
    neutral: 'bg-slate-500',
    info: 'bg-blue-400',
    running: 'bg-blue-400',
    success: 'bg-emerald-400',
    warning: 'bg-amber-400',
    error: 'bg-rose-400',
  }[tone]
}

export function darkPillClass(tone: Tone) {
  return {
    neutral: 'border-slate-700 bg-slate-900 text-slate-400',
    info: 'border-blue-500/40 bg-blue-500/10 text-blue-300',
    running: 'border-blue-500/40 bg-blue-500/10 text-blue-300',
    success: 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300',
    warning: 'border-amber-500/40 bg-amber-500/10 text-amber-300',
    error: 'border-rose-500/40 bg-rose-500/10 text-rose-300',
  }[tone]
}

export function lightPillClass(tone: Tone) {
  return {
    neutral: 'border-slate-200 bg-white text-slate-600',
    info: 'border-blue-200 bg-blue-50 text-blue-700',
    running: 'border-amber-200 bg-amber-50 text-amber-700',
    success: 'border-emerald-200 bg-emerald-50 text-emerald-700',
    warning: 'border-amber-200 bg-amber-50 text-amber-700',
    error: 'border-rose-200 bg-rose-50 text-rose-700',
  }[tone]
}
