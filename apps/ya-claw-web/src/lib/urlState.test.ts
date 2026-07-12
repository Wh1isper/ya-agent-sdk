import { describe, expect, it } from 'vitest'

import {
  buildAgencyPath,
  buildChatPath,
  buildProfilePath,
  buildRoutePath,
  parseUrlSelection,
} from './urlState'

describe('top-level route URL contract', () => {
  it.each([
    ['automation', '/automation'],
    ['heartbeat', '/automation/heartbeat'],
    ['workspace', '/workspace'],
  ] as const)('round-trips the %s route', (route, path) => {
    expect(buildRoutePath(route)).toBe(path)
    expect(parseUrlSelection(path)).toEqual({
      route,
      selectedSessionId: null,
      selectedRunId: null,
      selectedProfileName: null,
    })
  })
})

describe('URL entity selection', () => {
  it.each(['chat', 'debug'] as const)(
    'round-trips reserved session and run characters for %s routes',
    (route) => {
      const sessionId = 'session/with spaces?and#a%percent'
      const runId = 'run/with spaces?and#b%percent'
      const path = buildChatPath(sessionId, runId, route)

      expect(path).not.toContain(sessionId)
      expect(parseUrlSelection(path)).toMatchObject({
        route,
        selectedSessionId: sessionId,
        selectedRunId: runId,
      })
    },
  )

  it('does not accept a run without a selected session path', () => {
    expect(parseUrlSelection('/activity/runs/orphan-run')).toMatchObject({
      route: 'debug',
      selectedSessionId: null,
      selectedRunId: null,
    })
  })

  it('treats malformed encoded entity segments as unselected', () => {
    expect(
      parseUrlSelection('/conversations/sessions/%E0%A4%A/runs/run-a'),
    ).toMatchObject({
      selectedSessionId: null,
      selectedRunId: null,
    })
  })

  it('keeps the new agent editor distinct from agents named new', () => {
    expect(buildProfilePath('__new__')).toBe('/agents/new')
    expect(buildProfilePath('new')).toBe('/agents/by-name/new')
    expect(parseUrlSelection('/agents/new')).toMatchObject({
      route: 'profiles',
      selectedProfileName: '__new__',
    })
    expect(parseUrlSelection('/agents/by-name/new')).toMatchObject({
      route: 'profiles',
      selectedProfileName: 'new',
    })
  })

  it('builds and parses canonical nested agency session and run routes', () => {
    const sessionId = 'agency/session?#%'
    const runId = 'agency/run?#%'
    const sessionPath = buildAgencyPath(sessionId)
    const runPath = buildAgencyPath(sessionId, runId)

    expect(sessionPath).toBe(
      '/automation/agency/sessions/agency%2Fsession%3F%23%25',
    )
    expect(runPath).toBe(
      '/automation/agency/sessions/agency%2Fsession%3F%23%25/runs/agency%2Frun%3F%23%25',
    )
    expect(parseUrlSelection(sessionPath)).toMatchObject({
      route: 'agency',
      selectedSessionId: sessionId,
      selectedRunId: null,
    })
    expect(parseUrlSelection(runPath)).toMatchObject({
      route: 'agency',
      selectedSessionId: sessionId,
      selectedRunId: runId,
    })
  })
})
