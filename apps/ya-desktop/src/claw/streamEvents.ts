import type { EventSourceMessage } from '@microsoft/fetch-event-source'

import type { ClawStreamEvent, JsonObject } from './types'

export function parseStreamMessage(message: EventSourceMessage): ClawStreamEvent {
  return {
    id: message.id,
    event: message.event || 'message',
    data: message.data,
    payload: parseJsonObject(message.data),
  }
}

export function streamTextDelta(event: ClawStreamEvent) {
  const payload = event.payload
  const delta = payload.delta
  if (typeof delta === 'string') return delta
  const content = payload.content
  if (typeof content === 'string' && event.event === 'TEXT_MESSAGE_CHUNK') {
    return content
  }
  const value = payload.value
  if (
    isRecord(value) &&
    value.name === 'ya_agent.final_result' &&
    isRecord(value.payload) &&
    typeof value.payload.output_text === 'string'
  ) {
    return value.payload.output_text
  }
  return ''
}

export function streamRunId(event: ClawStreamEvent) {
  const runId = event.payload.runId ?? event.payload.run_id
  if (typeof runId === 'string') return runId
  const value = event.payload.value
  if (isRecord(value)) {
    const nestedRunId = value.run_id ?? value.runId
    if (typeof nestedRunId === 'string') return nestedRunId
  }
  return null
}

export function streamSessionId(event: ClawStreamEvent) {
  const sessionId =
    event.payload.sessionId ??
    event.payload.session_id ??
    event.payload.threadId ??
    event.payload.thread_id
  if (typeof sessionId === 'string') return sessionId
  const value = event.payload.value
  if (isRecord(value)) {
    const nestedSessionId =
      value.session_id ?? value.sessionId ?? value.thread_id ?? value.threadId
    if (typeof nestedSessionId === 'string') return nestedSessionId
  }
  return null
}

export function isRunErrorEvent(event: ClawStreamEvent) {
  return event.event === 'RUN_ERROR' || event.payload.type === 'RUN_ERROR'
}

export function isRunFinishedEvent(event: ClawStreamEvent) {
  return event.event === 'RUN_FINISHED' || event.payload.type === 'RUN_FINISHED'
}

export function isRunStartedEvent(event: ClawStreamEvent) {
  return event.event === 'RUN_STARTED' || event.payload.type === 'RUN_STARTED'
}

export function streamErrorMessage(event: ClawStreamEvent) {
  const message = event.payload.message
  if (typeof message === 'string') return message
  const value = event.payload.value
  if (isRecord(value)) {
    if (typeof value.message === 'string') return value.message
    if (typeof value.error === 'string') return value.error
    if (
      isRecord(value.payload) &&
      typeof value.payload.message === 'string'
    ) {
      return value.payload.message
    }
  }
  return 'The streamed run returned an error event.'
}

export function collectTextFromReplay(events: JsonObject[] | null | undefined) {
  return (events ?? [])
    .map((payload, index) =>
      streamTextDelta({
        id: String(index + 1),
        event: typeof payload.type === 'string' ? payload.type : 'message',
        data: JSON.stringify(payload),
        payload,
      }),
    )
    .join('')
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value))
}

function parseJsonObject(value: string): JsonObject {
  try {
    const parsedValue: unknown = JSON.parse(value)
    if (isJsonObject(parsedValue)) return parsedValue
    return { value: parsedValue }
  } catch {
    return { value }
  }
}

function isJsonObject(value: unknown): value is JsonObject {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value))
}
