import type {
  ClawHealth,
  ClawInfo,
  ClawRunTraceResponse,
  ClawSessionGetResponse,
  ClawSessionSummary,
  ClawSessionTurnsResponse,
  DesktopClawConnection,
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

  getSession(sessionId: string) {
    return this.fetchJson<ClawSessionGetResponse>(
      `/api/v1/sessions/${encodeURIComponent(sessionId)}?runs_limit=20&include_input_parts=true`,
    )
  }

  listSessionTurns(sessionId: string, limit = 20) {
    return this.fetchJson<ClawSessionTurnsResponse>(
      `/api/v1/sessions/${encodeURIComponent(sessionId)}/turns?limit=${limit}`,
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

async function readResponseDetail(response: Response) {
  const contentType = response.headers.get('content-type') ?? ''
  try {
    if (contentType.includes('application/json')) return await response.json()
    return await response.text()
  } catch {
    return null
  }
}
