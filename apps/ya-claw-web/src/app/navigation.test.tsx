import { act, render } from '@testing-library/react'
import { createMemoryHistory } from '@tanstack/react-router'
import { describe, expect, it } from 'vitest'

import App from '../App'
import { navigateApp } from './navigation'
import { createAppRouter } from './router'

describe('canonical application routes', () => {
  it.each([
    ['/agents/new', '/agents/new'],
    ['/agents/by-name/default', '/agents/by-name/$profileName'],
    ['/automation/schedules/schedule-1', '/automation/schedules/$scheduleId'],
    ['/automation/workflows/workflow-1', '/automation/workflows/$workflowId'],
    ['/integrations/setup', '/integrations/setup'],
    [
      '/integrations/conversations/conversation-1',
      '/integrations/conversations/$conversationId',
    ],
    [
      '/automation/agency/sessions/agency-session',
      '/automation/agency/sessions/$sessionId',
    ],
    [
      '/automation/agency/sessions/agency-session/runs/agency-run',
      '/automation/agency/sessions/$sessionId/runs/$runId',
    ],
  ] as const)(
    'matches direct URL %s through route %s',
    async (path, routeId) => {
      const router = createAppRouter(
        createMemoryHistory({ initialEntries: [path] }),
      )

      await router.load()

      expect(router.state.location.pathname).toBe(path)
      expect(router.state.matches.map((match) => match.routeId)).toContain(
        routeId,
      )
    },
  )

  it('does not resurrect a replaced detail entry on browser Back or Forward', () => {
    const history = createMemoryHistory({ initialEntries: ['/agents'] })
    history.push('/agents/by-name/default')
    history.replace('/agents')

    history.back()
    expect(history.location.pathname).toBe('/agents')
    history.forward()
    expect(history.location.pathname).toBe('/agents')
  })
})

describe('app navigation registration lifecycle', () => {
  it('targets the active router and restores or clears registrations on teardown', () => {
    window.history.replaceState(null, '', '/')
    const firstRouter = createAppRouter(
      createMemoryHistory({ initialEntries: ['/'] }),
    )
    const secondRouter = createAppRouter(
      createMemoryHistory({ initialEntries: ['/'] }),
    )
    const switchedRouter = createAppRouter(
      createMemoryHistory({ initialEntries: ['/'] }),
    )

    const firstApp = render(<App appRouter={firstRouter} />)
    const secondApp = render(<App appRouter={secondRouter} />)

    act(() => navigateApp('/activity'))
    expect(secondRouter.history.location.pathname).toBe('/activity')
    expect(firstRouter.history.location.pathname).toBe('/')

    secondApp.unmount()
    act(() => navigateApp('/agents'))
    expect(firstRouter.history.location.pathname).toBe('/agents')
    expect(secondRouter.history.location.pathname).toBe('/activity')

    firstApp.rerender(<App appRouter={switchedRouter} />)
    act(() => navigateApp('/settings'))
    expect(switchedRouter.history.location.pathname).toBe('/settings')
    expect(firstRouter.history.location.pathname).toBe('/agents')

    firstApp.unmount()
    act(() => navigateApp('/workspace'))
    expect(window.location.pathname).toBe('/workspace')
    expect(switchedRouter.history.location.pathname).toBe('/settings')
  })
})
