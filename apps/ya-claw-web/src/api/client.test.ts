import { afterEach, describe, expect, it, vi } from 'vitest'

import { ClawApiClient } from './client'

function mockJsonResponse(body: unknown) {
  return Promise.resolve({
    ok: true,
    status: 200,
    json: () => Promise.resolve(body),
  } as Response)
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
})
