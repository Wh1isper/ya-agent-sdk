import { HttpResponse, http, type HttpHandler } from 'msw'

import {
  clawInfoFixture,
  healthFixture,
  heartbeatStatusFixture,
  schedulesFixture,
  sessionsFixture,
  TEST_API_TOKEN,
  workspaceRuntimeFixture,
} from './fixtures'

export type HandledApiRequest = {
  method: string
  pathname: string
  authorization: string | null
}

export const handledApiRequests: HandledApiRequest[] = []

export function resetHandledApiRequests() {
  handledApiRequests.length = 0
}

function record(request: Request) {
  const url = new URL(request.url)
  handledApiRequests.push({
    method: request.method,
    pathname: url.pathname,
    authorization: request.headers.get('Authorization'),
  })
}

function rejectUnlessAuthorized(request: Request) {
  record(request)
  if (request.headers.get('Authorization') === `Bearer ${TEST_API_TOKEN}`) {
    return null
  }
  return HttpResponse.json(
    { detail: 'The test runtime requires a valid bearer token.' },
    { status: 401 },
  )
}

function authorizedJson(
  request: Request,
  value: Parameters<typeof HttpResponse.json>[0],
) {
  return rejectUnlessAuthorized(request) ?? HttpResponse.json(value)
}

/**
 * Contract-shaped handlers shared by integration tests. Keep endpoint behavior
 * here instead of mocking hooks so tests exercise the production API client,
 * connection gate, query client, and router together.
 */
export const apiHandlers: HttpHandler[] = [
  http.get('*/healthz', ({ request }) =>
    authorizedJson(request, healthFixture),
  ),
  http.get('*/api/v1/claw/info', ({ request }) =>
    authorizedJson(request, clawInfoFixture),
  ),
  http.get('*/api/v1/workspace/runtime', ({ request }) =>
    authorizedJson(request, workspaceRuntimeFixture),
  ),
  http.get('*/api/v1/sessions', ({ request }) =>
    authorizedJson(request, sessionsFixture),
  ),
  http.get('*/api/v1/profiles', ({ request }) => authorizedJson(request, [])),
  http.get('*/api/v1/bridges/conversations', ({ request }) =>
    authorizedJson(request, { conversations: [] }),
  ),
  http.get('*/api/v1/bridges/events', ({ request }) =>
    authorizedJson(request, { events: [] }),
  ),
  http.get('*/api/v1/schedules', ({ request }) =>
    authorizedJson(request, schedulesFixture),
  ),
  http.get('*/api/v1/heartbeat/status', ({ request }) =>
    authorizedJson(request, heartbeatStatusFixture),
  ),
  http.get('*/api/v1/heartbeat/fires', ({ request }) =>
    authorizedJson(request, { fires: [] }),
  ),
  http.get('*/api/v1/workflows', ({ request }) =>
    authorizedJson(request, { workflows: [] }),
  ),
  http.get('*/api/v1/workflow-runs', ({ request }) =>
    authorizedJson(request, { workflow_runs: [] }),
  ),
  http.get('*/api/v1/agency/config', ({ request }) =>
    authorizedJson(request, {
      enabled: false,
      profile_name: 'default',
      timer_interval_seconds: 3600,
      agency_session_id: null,
      singleton_scope_key: 'agency:global',
      singleton_source_session_id: null,
      risk_policy: { max_auto_action_risk: 'extra_high' },
      memory_files: {
        index: 'AGENCY.md',
        action_log: 'agency/ACTION_LOG.md',
      },
      next_fire_at: null,
    }),
  ),
  http.get('*/api/v1/agency/status', ({ request }) =>
    authorizedJson(request, {
      enabled: false,
      agency_session_id: null,
      state: 'idle',
      active_run: null,
      latest_run: null,
      active_run_id: null,
      latest_run_id: null,
      next_fire_at: null,
      pending_fire_count: 0,
      last_fire: null,
      agency_session: null,
    }),
  ),
  http.get('*/api/v1/agency/fires', ({ request }) =>
    authorizedJson(request, { fires: [] }),
  ),
  http.get('*/api/v1/claw/notifications', ({ request }) => {
    const unauthorized = rejectUnlessAuthorized(request)
    if (unauthorized) return unauthorized

    const encoder = new TextEncoder()
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode(': test stream connected\n\n'))
      },
    })
    return new HttpResponse(body, {
      headers: {
        'Cache-Control': 'no-cache',
        Connection: 'keep-alive',
        'Content-Type': 'text/event-stream',
      },
    })
  }),
]
