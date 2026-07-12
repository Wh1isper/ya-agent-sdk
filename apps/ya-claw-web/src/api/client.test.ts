import { afterEach, describe, expect, it, vi } from 'vitest'

import { ApiError, ClawApiClient } from './client'

function mockJsonResponse(body: unknown, status = 200) {
  return Promise.resolve(
    new Response(JSON.stringify(body), {
      status,
      headers: { 'Content-Type': 'application/json' },
    }),
  )
}

describe('ClawApiClient schedule and workflow query serialization', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('serializes workflow-backed schedule filters for the schedules endpoint', async () => {
    const fetchMock = vi
      .spyOn(globalThis, 'fetch')
      .mockImplementation(() => mockJsonResponse({ schedules: [] }))
    const api = new ClawApiClient({
      baseUrl: 'http://claw.local/',
      apiToken: 'token',
    })

    await api.listSchedules({
      includeDeleted: true,
      includeWorkflow: true,
      workflowId: ' workflow-1 ',
      executionMode: 'workflow',
      ownerSessionId: ' session-1 ',
      scheduleId: ' schedule-1 ',
      includeRecentRuns: false,
      limit: 25,
    })

    const url = new URL(String(fetchMock.mock.calls[0]?.[0]))
    expect(url.pathname).toBe('/api/v1/schedules')
    expect(url.searchParams.get('include_deleted')).toBe('true')
    expect(url.searchParams.get('workflow_id')).toBe('workflow-1')
    expect(url.searchParams.get('execution_mode')).toBe('workflow')
    expect(url.searchParams.get('owner_session_id')).toBe('session-1')
    expect(url.searchParams.get('schedule_id')).toBe('schedule-1')
    expect(url.searchParams.get('include_recent_runs')).toBe('false')
    expect(url.searchParams.get('limit')).toBe('25')
    expect(url.searchParams.has('include_workflow')).toBe(false)
  })

  it('serializes prompt-only schedule filtering and repeated workflow tags', async () => {
    const fetchMock = vi
      .spyOn(globalThis, 'fetch')
      .mockImplementation((input) => {
        const url = String(input)
        if (url.includes('/api/v1/workflows')) {
          return mockJsonResponse({ workflows: [] })
        }
        return mockJsonResponse({ schedules: [] })
      })
    const api = new ClawApiClient({
      baseUrl: 'http://claw.local',
      apiToken: '',
    })

    await api.listSchedules({ includeWorkflow: false })
    await api.listWorkflows({ tags: ['daily', 'research'], limit: 10 })

    const schedulesUrl = new URL(String(fetchMock.mock.calls[0]?.[0]))
    expect(schedulesUrl.searchParams.get('include_workflow')).toBe('false')
    expect(schedulesUrl.searchParams.get('limit')).toBe('100')

    const workflowsUrl = new URL(String(fetchMock.mock.calls[1]?.[0]))
    expect(workflowsUrl.searchParams.getAll('tags')).toEqual([
      'daily',
      'research',
    ])
    expect(workflowsUrl.searchParams.get('limit')).toBe('10')
  })

  it('serializes workspace directory pagination without exposing host paths', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(() =>
      mockJsonResponse({
        session_id: 'session-1',
        path: '/workspace/reports',
        items: [],
        limit: 25,
        offset: 50,
        has_more: false,
        next_cursor: null,
        next_offset: null,
        truncated: false,
      }),
    )
    const api = new ClawApiClient({
      baseUrl: 'http://claw.local',
      apiToken: 'token',
    })

    await api.listWorkspaceFiles('session/1', {
      path: '/workspace/reports',
      includeHidden: true,
      limit: 25,
      cursor: 'cursor-token',
    })

    const url = new URL(String(fetchMock.mock.calls[0]?.[0]))
    expect(url.pathname).toBe('/api/v1/sessions/session%2F1/workspace/files')
    expect(url.searchParams.get('path')).toBe('/workspace/reports')
    expect(url.searchParams.get('include_hidden')).toBe('true')
    expect(url.searchParams.get('limit')).toBe('25')
    expect(url.searchParams.get('cursor')).toBe('cursor-token')
    expect(url.searchParams.has('offset')).toBe(false)
  })

  it('preserves text error details and reports a useful status message', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response('token rejected', { status: 401 }),
    )
    const onUnauthorized = vi.fn()
    const api = new ClawApiClient({
      baseUrl: 'http://claw.local',
      apiToken: 'bad-token',
      connectionScope: 'scope-1',
      onUnauthorized,
    })

    await expect(api.clawInfo()).rejects.toMatchObject({
      status: 401,
      detail: 'token rejected',
      message: 'The API token is invalid or expired',
    } satisfies Partial<ApiError>)
    expect(onUnauthorized).toHaveBeenCalledWith('scope-1')
  })

  it('does not invalidate the connection for ordinary 403 responses', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response('forbidden', { status: 403 }),
    )
    const onUnauthorized = vi.fn()
    const api = new ClawApiClient({
      baseUrl: 'http://claw.local',
      apiToken: 'limited-token',
      connectionScope: 'scope-1',
      onUnauthorized,
    })

    await expect(api.clawInfo()).rejects.toMatchObject({ status: 403 })
    expect(onUnauthorized).not.toHaveBeenCalled()
  })

  it('uses the same 401-only invalidation rule for downloads', async () => {
    const fetchMock = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(new Response('forbidden', { status: 403 }))
      .mockResolvedValueOnce(new Response('expired', { status: 401 }))
    const onUnauthorized = vi.fn()
    const api = new ClawApiClient({
      baseUrl: 'http://claw.local',
      apiToken: 'token',
      connectionScope: 'download-scope',
      onUnauthorized,
    })

    await expect(
      api.downloadWorkspaceFile('session', 'report.txt'),
    ).rejects.toMatchObject({ status: 403 })
    expect(onUnauthorized).not.toHaveBeenCalled()

    await expect(
      api.downloadWorkspaceFile('session', 'report.txt'),
    ).rejects.toMatchObject({ status: 401 })
    expect(onUnauthorized).toHaveBeenCalledOnce()
    expect(onUnauthorized).toHaveBeenCalledWith('download-scope')
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it('serializes lightweight session pagination cursors', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(() =>
      mockJsonResponse({
        sessions: [],
        total: 1000,
        limit: 50,
        has_more: false,
      }),
    )
    const api = new ClawApiClient({
      baseUrl: 'http://claw.local',
      apiToken: 'token',
    })

    await api.listSessionsPage({
      limit: 25,
      beforeUpdatedAt: '2026-07-12T07:00:00Z',
      beforeId: 'session-25',
    })

    const url = new URL(String(fetchMock.mock.calls[0]?.[0]))
    expect(url.pathname).toBe('/api/v1/sessions/page')
    expect(url.searchParams.get('limit')).toBe('25')
    expect(url.searchParams.get('before_updated_at')).toBe(
      '2026-07-12T07:00:00Z',
    )
    expect(url.searchParams.get('before_id')).toBe('session-25')
    expect(url.searchParams.get('include_latest_output')).toBe('false')
  })

  it('forwards query cancellation signals so abort stops the fetch', async () => {
    const controller = new AbortController()
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(
      (_input, init) =>
        new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener('abort', () => {
            reject(new DOMException('Aborted', 'AbortError'))
          })
        }),
    )
    const api = new ClawApiClient({
      baseUrl: 'http://claw.local',
      apiToken: 'token',
    })

    const request = api.listSessions({ signal: controller.signal })
    controller.abort()

    await expect(request).rejects.toMatchObject({ name: 'AbortError' })
    expect(fetchMock.mock.calls[0]?.[1]?.signal).toBe(controller.signal)
  })

  it('encodes session and run identifiers as path segments', async () => {
    const fetchMock = vi
      .spyOn(globalThis, 'fetch')
      .mockImplementation(() => mockJsonResponse({}))
    const api = new ClawApiClient({
      baseUrl: 'http://claw.local',
      apiToken: 'token',
    })

    await api.getSession('session/with?reserved')
    await api.getRun('run/#reserved')

    expect(String(fetchMock.mock.calls[0]?.[0])).toContain(
      '/sessions/session%2Fwith%3Freserved?',
    )
    expect(String(fetchMock.mock.calls[1]?.[0])).toContain(
      '/runs/run%2F%23reserved?',
    )
  })

  it('supports empty successful responses', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(null, { status: 204 }),
    )
    const api = new ClawApiClient({
      baseUrl: 'http://claw.local',
      apiToken: 'token',
    })

    await expect(api.deleteProfile('default')).resolves.toBeUndefined()
  })
})
