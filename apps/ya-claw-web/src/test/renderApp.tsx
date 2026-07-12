import { render, type RenderResult } from '@testing-library/react'
import userEvent, { type UserEvent } from '@testing-library/user-event'
import { createMemoryHistory } from '@tanstack/react-router'

import App from '../App'
import { createAppRouter, type AppRouter } from '../app/router'
import { parseUrlSelection } from '../lib/urlState'
import { useConnectionStore } from '../stores/connectionStore'
import { useLayoutStore, type LayoutState } from '../stores/layoutStore'
import { TEST_API_TOKEN } from './fixtures'
import { setTestViewport } from './viewport'

export type RenderAppOptions = {
  route?: string
  connected?: boolean
  baseUrl?: string
  viewport?: { width: number; height?: number }
  layoutState?: Partial<LayoutState>
}

export type RenderedApp = RenderResult & {
  user: UserEvent
  router: AppRouter
}

/**
 * Render the real provider/gate/router stack from a deterministic test state.
 *
 * Every render owns a fresh memory history and router. Loading that router
 * before mounting ensures assertions cannot observe a previous test's matches
 * while TanStack Router is still resolving a lazy route.
 */
export async function renderApp({
  route = '/',
  connected = false,
  baseUrl = window.location.origin,
  viewport = { width: 1024, height: 900 },
  layoutState = {},
}: RenderAppOptions = {}): Promise<RenderedApp> {
  localStorage.clear()
  setTestViewport(viewport.width, viewport.height)
  useConnectionStore.setState({
    baseUrl,
    apiToken: connected ? TEST_API_TOKEN : '',
    connectionScope: 'integration-test-scope',
    connectionIssue: null,
  })
  const selection = parseUrlSelection(route)
  useLayoutStore.setState({
    ...selection,
    selectedChatSessionId:
      selection.route === 'chat' ? selection.selectedSessionId : null,
    selectedChatRunId:
      selection.route === 'chat' ? selection.selectedRunId : null,
    selectedDebugSessionId:
      selection.route === 'debug' ? selection.selectedSessionId : null,
    selectedDebugRunId:
      selection.route === 'debug' ? selection.selectedRunId : null,
    selectedAgencySessionId:
      selection.route === 'agency' ? selection.selectedSessionId : null,
    inspectorTab: 'summary',
    advancedMode: false,
    railCollapsed: false,
    ...layoutState,
  })

  const router = createAppRouter(
    createMemoryHistory({ initialEntries: [route] }),
  )
  window.history.replaceState(null, '', route)
  await router.load()

  return {
    user: userEvent.setup(),
    router,
    ...render(<App appRouter={router} />),
  }
}
