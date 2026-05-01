import type { AguiEvent } from '../../types'
import { getEventTone } from '../../lib/status'

export function isTerminalAguiEvent(event: AguiEvent) {
  const eventType = typeof event.type === 'string' ? event.type : ''
  return eventType === 'RUN_FINISHED' || eventType === 'RUN_ERROR'
}

export function eventKey(event: AguiEvent) {
  return [
    typeof event.type === 'string' ? event.type : 'event',
    typeof event.name === 'string' ? event.name : '',
    typeof event.timestamp === 'number' || typeof event.timestamp === 'string'
      ? String(event.timestamp)
      : '',
    typeof event.messageId === 'string' ? event.messageId : '',
    typeof event.toolCallId === 'string' ? event.toolCallId : '',
  ].join(':')
}

export function eventTypeLabel(event: AguiEvent) {
  return typeof event.type === 'string' && event.type.trim()
    ? event.type
    : 'UNKNOWN'
}

export function eventNameLabel(event: AguiEvent) {
  if (typeof event.name === 'string' && event.name.trim()) return event.name
  const value = event.value
  if (value && typeof value === 'object') {
    const payload = value as Record<string, unknown>
    const nestedName = payload.name
    if (typeof nestedName === 'string' && nestedName.trim()) return nestedName
  }
  if (typeof event.toolCallName === 'string' && event.toolCallName.trim()) {
    return event.toolCallName
  }
  if (typeof event.tool_call_name === 'string' && event.tool_call_name.trim()) {
    return event.tool_call_name
  }
  return ''
}

export function eventTimestampLabel(event: AguiEvent) {
  const timestamp = event.timestamp as unknown
  if (typeof timestamp === 'number') {
    return new Date(timestamp).toLocaleTimeString()
  }
  if (typeof timestamp === 'string' && timestamp.trim()) {
    const parsed = Date.parse(timestamp)
    if (Number.isFinite(parsed)) return new Date(parsed).toLocaleTimeString()
    return timestamp
  }
  return ''
}

export function eventTone(event: AguiEvent) {
  return getEventTone(
    event,
    `${eventTypeLabel(event)} ${eventNameLabel(event)}`,
  )
}
