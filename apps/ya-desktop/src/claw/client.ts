import {
  EventStreamContentType,
  fetchEventSource,
  type EventSourceMessage,
} from '@microsoft/fetch-event-source'

import type {
  ClawHealth,
  ClawInfo,
  ClawRunTraceResponse,
  ClawSessionGetResponse,
  ClawSessionStreamInput,
  ClawSessionSummary,
  ClawSessionTurnsResponse,
  ClawStreamHandlers,
  DesktopClawConnection,
  JsonObject,
} from './types'

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

  listSessions() {
    return this.fetchJson<ClawSessionSummary[]>('/api/v1/sessions')
  }

  createSessionStream(
    input: ClawSessionStreamInput,
    handlers: ClawStreamHandlers = {},
    signal?: AbortSignal,
  ) {
    return fetchEventSource(
      buildUrl(this.connection.baseUrl, '/api/v1/sessions:stream'),
      {
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
      },
    )
  }

  getSession(sessionId: string) {
    return this.fetchJson<ClawSessionGetResponse>(
      `/api/v1/sessions/${encodeURIComponent(sessionId)}?runs_limit=20&include_input_parts=true`,
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

  getRunTrace(runId: string, maxItemChars = 2000, maxTotalChars = 8000) {
    const searchParams = new URLSearchParams({
      max_item_chars: String(maxItemChars),
      max_total_chars: String(maxTotalChars),
    })
    return this.fetchJson<ClawRunTraceResponse>(
      `/api/v1/runs/${encodeURIComponent(runId)}/trace?${searchParams.toString()}`,
    )
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

function buildUrl(baseUrl: string, path: string) {
  const normalizedBaseUrl = baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl
  const normalizedPath = path.startsWith('/') ? path : `/${path}`
  return `${normalizedBaseUrl}${normalizedPath}`
}

function parseStreamMessage(message: EventSourceMessage) {
  return {
    id: message.id,
    event: message.event || 'message',
    data: message.data,
    payload: parseJsonObject(message.data),
  }
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

async function readResponseDetail(response: Response) {
  const contentType = response.headers.get('content-type') ?? ''
  try {
    if (contentType.includes('application/json')) return await response.json()
    return await response.text()
  } catch {
    return null
  }
}
