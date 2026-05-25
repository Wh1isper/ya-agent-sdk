import {
  EventStreamContentType,
  fetchEventSource,
} from '@microsoft/fetch-event-source'

import type {
  ClawAgencyClearResponse,
  ClawAgencyConfig,
  ClawAgencyFireListResponse,
  ClawAgencyStatus,
  ClawHealth,
  ClawInteractionRespondRequest,
  ClawInteractionRespondResponse,
  ClawInfo,
  ClawNotificationEvent,
  ClawNotificationHandlers,
  ClawProfileSummary,
  ClawRunSummary,
  ClawRunTraceResponse,
  ClawSessionGetResponse,
  ClawSessionRunStreamInput,
  ClawSessionStreamInput,
  ClawSessionSummary,
  ClawSessionTurnsResponse,
  ClawStreamHandlers,
  DesktopClawConnection,
} from './types'
import { parseStreamMessage } from './streamEvents'

export class ClawClientError extends Error {
  readonly status: number
  readonly detail: unknown

  constructor(message: string, status: number, detail: unknown) {
    super(message)
    this.name = 'ClawClientError'
    this.status = status
    this.detail = detail
  }
}

export class ClawHttpClient {
  readonly connection: DesktopClawConnection

  constructor(connection: DesktopClawConnection) {
    this.connection = connection
  }

  health() {
    return this.fetchJson<ClawHealth>('/healthz')
  }

  info() {
    return this.fetchJson<ClawInfo>('/api/v1/claw/info')
  }

  listProfiles() {
    return this.fetchJson<ClawProfileSummary[]>('/api/v1/profiles')
  }

  listSessions() {
    return this.fetchJson<ClawSessionSummary[]>('/api/v1/sessions')
  }

  getAgencyConfig() {
    return this.fetchJson<ClawAgencyConfig>('/api/v1/agency/config')
  }

  getAgencyStatus() {
    return this.fetchJson<ClawAgencyStatus>('/api/v1/agency/status')
  }

  listAgencyFires() {
    return this.fetchJson<ClawAgencyFireListResponse>('/api/v1/agency/fires')
  }

  clearAgency() {
    return this.fetchJson<ClawAgencyClearResponse>('/api/v1/agency:clear', {
      method: 'POST',
    })
  }

  createSessionStream(
    input: ClawSessionStreamInput,
    handlers: ClawStreamHandlers = {},
    signal?: AbortSignal,
  ) {
    return this.fetchStream('/api/v1/sessions:stream', input, handlers, signal)
  }

  createSessionRunStream(
    sessionId: string,
    input: ClawSessionRunStreamInput,
    handlers: ClawStreamHandlers = {},
    signal?: AbortSignal,
  ) {
    return this.fetchStream(
      `/api/v1/sessions/${encodeURIComponent(sessionId)}/runs:stream`,
      input,
      handlers,
      signal,
    )
  }

  getSession(sessionId: string) {
    return this.fetchJson<ClawSessionGetResponse>(
      `/api/v1/sessions/${encodeURIComponent(sessionId)}?runs_limit=20&include_message=true&include_input_parts=true`,
    )
  }

  listSessionTurns(sessionId: string, limit = 20) {
    const normalizedLimit = Number.isFinite(limit)
      ? Math.max(1, Math.min(100, Math.trunc(limit)))
      : 20
    const searchParams = new URLSearchParams({
      limit: String(normalizedLimit),
    })
    return this.fetchJson<ClawSessionTurnsResponse>(
      `/api/v1/sessions/${encodeURIComponent(sessionId)}/turns?${searchParams.toString()}`,
    )
  }

  getRun(runId: string) {
    return this.fetchJson<{ session: ClawSessionSummary; run: ClawRunSummary }>(
      `/api/v1/runs/${encodeURIComponent(runId)}?include_state=false&include_message=false`,
    )
  }

  getRunTrace(runId: string, maxItemChars = 2000, maxTotalChars = 8000) {
    const searchParams = new URLSearchParams({
      max_item_chars: String(maxItemChars),
      max_total_chars: String(maxTotalChars),
    })
    return this.fetchJson<ClawRunTraceResponse>(
      `/api/v1/runs/${encodeURIComponent(runId)}/trace?${searchParams.toString()}`,
    )
  }

  cancelSession(sessionId: string) {
    return this.fetchJson<ClawRunSummary>(
      `/api/v1/sessions/${encodeURIComponent(sessionId)}/cancel`,
      { method: 'POST' },
    )
  }

  respondInteraction(
    runId: string,
    interactionId: string,
    input: ClawInteractionRespondRequest,
  ) {
    return this.fetchJson<ClawInteractionRespondResponse>(
      `/api/v1/runs/${encodeURIComponent(runId)}/interactions/${encodeURIComponent(interactionId)}:respond`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(input),
      },
    )
  }

  interruptSession(sessionId: string) {
    return this.fetchJson<ClawRunSummary>(
      `/api/v1/sessions/${encodeURIComponent(sessionId)}/interrupt`,
      { method: 'POST' },
    )
  }

  streamNotifications(
    handlers: ClawNotificationHandlers = {},
    signal?: AbortSignal,
    lastEventId?: string | null,
  ) {
    const headers: Record<string, string> = {
      Accept: EventStreamContentType,
      ...(this.connection.apiToken
        ? { Authorization: `Bearer ${this.connection.apiToken}` }
        : {}),
    }
    if (lastEventId) headers['Last-Event-ID'] = lastEventId

    return fetchEventSource(
      buildUrl(this.connection.baseUrl, '/api/v1/claw/notifications'),
      {
        method: 'GET',
        headers,
        openWhenHidden: true,
        signal,
        async onopen(response) {
          if (!response.ok) {
            const detail = await readResponseDetail(response)
            throw new ClawClientError(
              `Claw notifications failed with HTTP ${response.status}`,
              response.status,
              detail,
            )
          }
          handlers.onOpen?.()
        },
        onmessage(message) {
          void handlers.onEvent?.(parseNotificationMessage(message))
        },
        onclose() {
          handlers.onClose?.()
        },
        onerror(error) {
          throw error
        },
      },
    )
  }

  private fetchStream(
    path: string,
    input: unknown,
    handlers: ClawStreamHandlers = {},
    signal?: AbortSignal,
  ) {
    return fetchEventSource(buildUrl(this.connection.baseUrl, path), {
      method: 'POST',
      headers: {
        Accept: EventStreamContentType,
        'Content-Type': 'application/json',
        ...(this.connection.apiToken
          ? { Authorization: `Bearer ${this.connection.apiToken}` }
          : {}),
      },
      body: JSON.stringify(input),
      openWhenHidden: true,
      signal,
      async onopen(response) {
        if (!response.ok) {
          const detail = await readResponseDetail(response)
          throw new ClawClientError(
            `Claw stream failed with HTTP ${response.status}`,
            response.status,
            detail,
          )
        }

        const contentType = response.headers.get('content-type') ?? ''
        if (!contentType.startsWith(EventStreamContentType)) {
          throw new ClawClientError(
            `Claw stream returned unsupported content type ${contentType || 'unknown'}`,
            response.status,
            contentType,
          )
        }

        handlers.onOpen?.()
      },
      onmessage(message) {
        handlers.onEvent?.(parseStreamMessage(message))
      },
      onclose() {
        handlers.onClose?.()
      },
      onerror(error) {
        throw error
      },
    })
  }

  private async fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await fetch(buildUrl(this.connection.baseUrl, path), {
      ...init,
      headers: {
        Accept: 'application/json',
        ...(this.connection.apiToken
          ? { Authorization: `Bearer ${this.connection.apiToken}` }
          : {}),
        ...init?.headers,
      },
    })

    if (!response.ok) {
      const detail = await readResponseDetail(response)
      throw new ClawClientError(
        `Claw request failed with HTTP ${response.status}`,
        response.status,
        detail,
      )
    }

    return (await response.json()) as T
  }
}

export function createClawClient(connection: DesktopClawConnection) {
  return new ClawHttpClient(connection)
}

function parseNotificationMessage(message: {
  id: string
  event: string
  data: string
}): ClawNotificationEvent {
  try {
    const parsedValue: unknown = JSON.parse(message.data)
    if (isRecord(parsedValue)) {
      return {
        id:
          message.id ||
          (typeof parsedValue.id === 'string' ? parsedValue.id : ''),
        type:
          typeof parsedValue.type === 'string'
            ? parsedValue.type
            : message.event,
        created_at:
          typeof parsedValue.created_at === 'string'
            ? parsedValue.created_at
            : undefined,
        payload: isRecord(parsedValue.payload) ? parsedValue.payload : {},
      }
    }
  } catch {
    // Keep reconnects alive for malformed notification payloads.
  }
  return { id: message.id, type: message.event || 'message', payload: {} }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value))
}

function buildUrl(baseUrl: string, path: string) {
  const normalizedBaseUrl = baseUrl.endsWith('/')
    ? baseUrl.slice(0, -1)
    : baseUrl
  const normalizedPath = path.startsWith('/') ? path : `/${path}`
  return `${normalizedBaseUrl}${normalizedPath}`
}

async function readResponseDetail(response: Response) {
  const contentType = response.headers.get('content-type') ?? ''
  try {
    if (contentType.includes('application/json')) return await response.json()
    return await response.text()
  } catch {
    return null
  }
}
