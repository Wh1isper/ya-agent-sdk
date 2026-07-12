import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { registerAppNavigate } from '../app/navigation'
import { useLayoutStore } from './layoutStore'

describe('layoutStore route synchronization', () => {
  let disposeNavigate: (() => void) | undefined

  beforeEach(() => {
    localStorage.clear()
    window.history.replaceState(null, '', '/agents/by-name/agent-a')
    useLayoutStore.setState({
      route: 'profiles',
      selectedSessionId: null,
      selectedRunId: null,
      selectedChatSessionId: null,
      selectedChatRunId: null,
      selectedDebugSessionId: null,
      selectedDebugRunId: null,
      selectedAgencySessionId: null,
      selectedProfileName: 'agent-a',
      inspectorTab: 'summary',
      advancedMode: false,
      railCollapsed: false,
    })
  })

  afterEach(() => {
    disposeNavigate?.()
    disposeNavigate = undefined
  })

  it('does not change the editable profile until navigation commits', () => {
    let requestedPath: string | null = null
    disposeNavigate = registerAppNavigate((path) => {
      requestedPath = path
    })

    useLayoutStore.getState().selectProfile('agent-b')

    expect(requestedPath).toBe('/agents/by-name/agent-b')
    expect(window.location.pathname).toBe('/agents/by-name/agent-a')
    expect(useLayoutStore.getState().selectedProfileName).toBe('agent-a')

    if (!requestedPath) throw new Error('Navigation was not requested')
    window.history.pushState(null, '', requestedPath)
    useLayoutStore.getState().syncFromUrl()

    expect(window.location.pathname).toBe('/agents/by-name/agent-b')
    expect(useLayoutStore.getState().selectedProfileName).toBe('agent-b')
  })

  it('requests replacement navigation when returning from profile detail', () => {
    let requestedNavigation: { path: string; replace?: boolean } | null = null
    disposeNavigate = registerAppNavigate((path, replace) => {
      requestedNavigation = { path, replace }
    })

    useLayoutStore.getState().selectProfile(null, { replace: true })

    expect(requestedNavigation).toEqual({ path: '/agents', replace: true })
  })

  it('resets connection-scoped selections and the entity URL only', () => {
    let requestedNavigation: { path: string; replace?: boolean } | null = null
    disposeNavigate = registerAppNavigate((path, replace) => {
      requestedNavigation = { path, replace }
    })
    useLayoutStore.setState({
      route: 'debug',
      selectedSessionId: 'session-a',
      selectedRunId: 'run-a',
      selectedChatSessionId: 'chat-session',
      selectedChatRunId: 'chat-run',
      selectedDebugSessionId: 'session-a',
      selectedDebugRunId: 'run-a',
      selectedAgencySessionId: 'agency-session',
      selectedProfileName: 'agent-a',
      inspectorTab: 'events',
      advancedMode: true,
      railCollapsed: true,
    })

    useLayoutStore.getState().resetConnectionSelection()

    expect(requestedNavigation).toEqual({ path: '/', replace: true })
    expect(useLayoutStore.getState()).toMatchObject({
      route: 'overview',
      selectedSessionId: null,
      selectedRunId: null,
      selectedChatSessionId: null,
      selectedChatRunId: null,
      selectedDebugSessionId: null,
      selectedDebugRunId: null,
      selectedAgencySessionId: null,
      selectedProfileName: null,
      inspectorTab: 'events',
      advancedMode: true,
      railCollapsed: true,
    })
    expect(localStorage.getItem('ya-claw-layout')).not.toContain('session-a')
    expect(localStorage.getItem('ya-claw-layout')).not.toContain('agent-a')
  })
})
