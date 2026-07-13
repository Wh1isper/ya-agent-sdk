import { act, screen, waitFor, within } from '@testing-library/react'
import { HttpResponse, http } from 'msw'
import { describe, expect, it } from 'vitest'

import { formatA11yViolations, getA11yViolations } from '../test/a11y'
import { TEST_API_TOKEN } from '../test/fixtures'
import { handledApiRequests } from '../test/handlers'
import { apiServer } from '../test/server'
import { useLayoutStore } from '../stores/layoutStore'
import { renderApp } from '../test/renderApp'
import {
  findHorizontalOverflowSignals,
  REFERENCE_VIEWPORTS,
} from '../test/viewport'

const NAVIGATION_DESTINATIONS = [
  '/',
  '/conversations',
  '/activity',
  '/automation',
  '/workspace',
  '/agents',
  '/integrations',
  '/settings',
] as const

describe('App MSW integration safety net', () => {
  it('crosses ConnectionGate into AppShell through the real API client', async () => {
    const { user } = await renderApp()

    expect(
      screen.getByRole('heading', { name: 'Connect to your runtime' }),
    ).toBeInTheDocument()

    await user.type(screen.getByLabelText('API token'), TEST_API_TOKEN)
    await user.click(screen.getByRole('button', { name: 'Test and connect' }))

    expect(
      await screen.findByRole('heading', {
        name: 'What should YA Claw work on?',
      }),
    ).toBeInTheDocument()
    const primaryHeadings = screen.getAllByRole('heading', { level: 1 })
    expect(primaryHeadings).toHaveLength(1)
    expect(primaryHeadings[0]).toHaveTextContent('What should YA Claw work on?')

    await waitFor(() => {
      expect(handledApiRequests.map((request) => request.pathname)).toEqual(
        expect.arrayContaining([
          '/healthz',
          '/api/v1/claw/info',
          '/api/v1/workspace/runtime',
          '/api/v1/sessions/page',
          '/api/v1/schedules',
          '/api/v1/heartbeat/status',
        ]),
      )
    })
    expect(
      handledApiRequests.every(
        (request) => request.authorization === `Bearer ${TEST_API_TOKEN}`,
      ),
    ).toBe(true)
  })

  it('persists a connection when the user explicitly opts in', async () => {
    const { user } = await renderApp()

    await user.type(screen.getByLabelText('API token'), TEST_API_TOKEN)
    await user.click(
      screen.getByLabelText('Remember this connection on this device'),
    )
    await user.click(screen.getByRole('button', { name: 'Test and connect' }))

    expect(
      await screen.findByRole('heading', {
        name: 'What should YA Claw work on?',
      }),
    ).toBeInTheDocument()
    const persisted = JSON.parse(
      localStorage.getItem('ya-claw-connection') ?? '{}',
    ) as { state?: { apiToken?: string; rememberConnection?: boolean } }
    expect(persisted.state).toMatchObject({
      apiToken: TEST_API_TOKEN,
      rememberConnection: true,
    })
  })

  it('falls back to the legacy session list when the page endpoint is unavailable', async () => {
    let legacyRequests = 0
    apiServer.use(
      http.get('*/api/v1/sessions/page', () =>
        HttpResponse.json({ detail: 'Not found' }, { status: 404 }),
      ),
      http.get('*/api/v1/sessions', () => {
        legacyRequests += 1
        return HttpResponse.json([])
      }),
    )

    await renderApp({ connected: true })

    expect(
      await screen.findByRole('heading', {
        level: 1,
        name: 'What should YA Claw work on?',
      }),
    ).toBeVisible()
    await waitFor(() => expect(legacyRequests).toBe(1))
  })

  it('preserves a requested deep link after connecting', async () => {
    const { user, router } = await renderApp({ route: '/agents/new' })

    await user.type(screen.getByLabelText('API token'), TEST_API_TOKEN)
    await user.click(screen.getByRole('button', { name: 'Test and connect' }))

    await waitFor(() => {
      expect(router.state.location.pathname).toBe('/agents/new')
    })
    expect(await screen.findByLabelText('Name')).toBeVisible()
  })

  it('does not block navigation to a newly saved agent', async () => {
    const savedProfile = {
      name: 'saved-agent',
      model: 'openai:gpt-4.1-mini',
      enabled: true,
      updated_at: '2026-07-13T00:00:00Z',
      created_at: '2026-07-13T00:00:00Z',
      builtin_toolsets: ['session'],
      toolsets: [],
      subagents: [],
      include_builtin_subagents: true,
      unified_subagents: true,
      need_user_approve_tools: [],
      need_user_approve_mcps: [],
      enabled_mcps: [],
      disabled_mcps: [],
      mcp_servers: {},
    }
    apiServer.use(
      http.put('*/api/v1/profiles/:profileName', ({ params }) => {
        expect(params.profileName).toBe(savedProfile.name)
        return HttpResponse.json(savedProfile)
      }),
      http.get('*/api/v1/profiles', () => HttpResponse.json([savedProfile])),
      http.get('*/api/v1/profiles/:profileName', () =>
        HttpResponse.json(savedProfile),
      ),
    )

    const { router, user } = await renderApp({
      connected: true,
      route: '/agents/new',
    })

    await user.type(screen.getByLabelText('Name'), savedProfile.name)
    await user.click(screen.getByRole('button', { name: 'Save agent' }))

    await waitFor(() => {
      expect(router.state.location.pathname).toBe(
        `/agents/by-name/${savedProfile.name}`,
      )
    })
    expect(
      screen.queryByRole('dialog', {
        name: 'Discard unsaved agent changes?',
      }),
    ).not.toBeInTheDocument()
  })

  it('blocks manual navigation away from a dirty new agent', async () => {
    const { router, user } = await renderApp({
      connected: true,
      route: '/agents/new',
    })

    const name = await screen.findByLabelText('Name')
    await user.type(name, 'draft-agent')
    await user.click(screen.getAllByRole('link', { name: /Settings/ })[0])

    expect(
      await screen.findByRole('dialog', {
        name: 'Discard unsaved agent changes?',
      }),
    ).toBeVisible()
    expect(router.state.location.pathname).toBe('/agents/new')
    expect(name).toHaveValue('draft-agent')
  })

  it('keeps long recent conversation text constrained on narrow Home layouts', async () => {
    const longTitle =
      'conversation-with-an-extremely-long-unbroken-title-that-must-not-expand-the-page'
    const longProfileName =
      'agent-with-an-extremely-long-unbroken-profile-name-that-must-not-expand-the-page'
    apiServer.use(
      http.get('*/api/v1/sessions/page', () =>
        HttpResponse.json({
          sessions: [
            {
              id: 'long-session',
              profile_name: longProfileName,
              session_type: 'conversation',
              metadata: { title: longTitle },
              created_at: '2026-07-13T00:00:00Z',
              updated_at: '2026-07-13T00:00:00Z',
              status: 'idle',
              run_count: 0,
              head_run_id: null,
              head_success_run_id: null,
              active_run_id: null,
              latest_run: null,
            },
          ],
          total: 1,
          limit: 50,
          has_more: false,
          next_before_updated_at: null,
          next_before_id: null,
        }),
      ),
    )

    await renderApp({ connected: true, viewport: { width: 320 } })

    const conversation = await screen.findByRole('link', {
      name: new RegExp(longTitle),
    })
    const text = conversation.querySelectorAll('p')
    const card = conversation.parentElement?.parentElement

    expect(card).toHaveClass('min-w-0')
    expect(conversation).toHaveClass('min-w-0')
    expect(text[0]).toHaveClass('truncate')
    expect(text[1]).toHaveClass('truncate')
    expect(conversation.querySelector('div')).toHaveClass('min-w-0', 'flex-1')
  })

  it('keeps an invalid credential in the connection gate', async () => {
    apiServer.use(
      http.get('*/api/v1/claw/info', () =>
        HttpResponse.json(
          { detail: 'invalid test credential' },
          { status: 401 },
        ),
      ),
    )
    const { user } = await renderApp()

    await user.type(screen.getByLabelText('API token'), TEST_API_TOKEN)
    await user.click(screen.getByRole('button', { name: 'Test and connect' }))

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'The API token is invalid or expired',
    )
    expect(
      screen.getByRole('heading', { name: 'Connect to your runtime' }),
    ).toBeVisible()
    expect(
      screen.queryByRole('heading', { name: 'What should YA Claw work on?' }),
    ).not.toBeInTheDocument()
  })

  it('keeps Home usable when a secondary overview query fails', async () => {
    apiServer.use(
      http.get('*/api/v1/heartbeat/status', () =>
        HttpResponse.json(
          { detail: 'Heartbeat status unavailable' },
          { status: 404 },
        ),
      ),
    )

    await renderApp({ connected: true })

    expect(
      await screen.findByRole('heading', {
        level: 1,
        name: 'What should YA Claw work on?',
      }),
    ).toBeVisible()
    expect(
      await screen.findByText('Some workspace overview data is unavailable'),
    ).toBeVisible()
    expect(screen.getByRole('link', { name: 'New conversation' })).toBeVisible()
  })

  it('keeps the Conversations page heading when its list query fails', async () => {
    apiServer.use(
      http.get('*/api/v1/sessions/page', () =>
        HttpResponse.json({ detail: 'unavailable' }, { status: 503 }),
      ),
    )

    await renderApp({ connected: true, route: '/conversations' })

    expect(
      await screen.findByRole(
        'heading',
        {
          level: 2,
          name: 'Could not load conversations',
        },
        { timeout: 5_000 },
      ),
    ).toBeInTheDocument()
    expect(
      screen.getByRole('heading', { level: 1, name: 'Conversations' }),
    ).toBeInTheDocument()
  })

  it('returns to reauthentication when an authenticated query receives 401', async () => {
    apiServer.use(
      http.get('*/api/v1/sessions/page', () =>
        HttpResponse.json({ detail: 'expired' }, { status: 401 }),
      ),
    )

    await renderApp({ connected: true })

    expect(
      await screen.findByRole('heading', { name: 'Connect to your runtime' }),
    ).toBeVisible()
    expect(screen.getByRole('alert')).toHaveTextContent(
      'Your API token is invalid or expired',
    )
  })

  it('keeps a detail deep link authoritative when the session list is stale', async () => {
    const deepLinkedSessionId = 'deep-linked-session'
    const firstCachedSessionId = 'first-cached-session'
    const requestedSessionIds: string[] = []
    apiServer.use(
      http.get('*/api/v1/sessions/page', () =>
        HttpResponse.json([
          {
            id: firstCachedSessionId,
            session_type: 'conversation',
            metadata: { title: 'First cached conversation' },
            profile_name: 'default',
            created_at: '2026-07-11T00:00:00Z',
            updated_at: '2026-07-11T00:00:00Z',
            status: 'idle',
            run_count: 0,
            head_run_id: null,
            active_run_id: null,
            latest_run: null,
          },
        ]),
      ),
      http.get('*/api/v1/sessions/:sessionId/workspace', () =>
        HttpResponse.json({ binding: null, sandbox_state: null }),
      ),
      http.get('*/api/v1/sessions/:sessionId', ({ params }) => {
        const sessionId = String(params.sessionId)
        requestedSessionIds.push(sessionId)
        return HttpResponse.json({
          session: {
            id: sessionId,
            profile_name: 'default',
            session_type: 'conversation',
            metadata: { title: 'Deep linked conversation' },
            created_at: '2026-07-11T00:00:00Z',
            updated_at: '2026-07-11T00:00:00Z',
            status: 'idle',
            run_count: 0,
            head_run_id: null,
            head_success_run_id: null,
            active_run_id: null,
            latest_run: null,
            runs: [],
            runs_limit: 3,
            runs_has_more: false,
            runs_next_before_sequence_no: null,
          },
          state: null,
          message: [],
        })
      }),
    )

    const { router } = await renderApp({
      connected: true,
      route: `/conversations/sessions/${deepLinkedSessionId}`,
    })

    expect(
      await screen.findByRole('heading', {
        level: 1,
        name: 'Deep linked conversation',
      }),
    ).toBeVisible()
    expect(requestedSessionIds).toContain(deepLinkedSessionId)
    expect(requestedSessionIds).not.toContain(firstCachedSessionId)
    expect(router.state.location.pathname).toBe(
      `/conversations/sessions/${deepLinkedSessionId}`,
    )
  })

  it('keeps a conversation usable when optional list, history, and workspace queries fail', async () => {
    const sessionId = 'partially-available-conversation'
    apiServer.use(
      http.get('*/api/v1/sessions/page', () =>
        HttpResponse.json({ detail: 'List unavailable' }, { status: 503 }),
      ),
      http.get('*/api/v1/sessions/:sessionId/workspace', () =>
        HttpResponse.json({ detail: 'Workspace unavailable' }, { status: 503 }),
      ),
      http.get('*/api/v1/sessions/:sessionId', ({ request }) => {
        const url = new URL(request.url)
        if (url.searchParams.get('runs_limit') === '3') {
          return HttpResponse.json(
            { detail: 'History unavailable' },
            { status: 503 },
          )
        }
        return HttpResponse.json({
          session: {
            id: sessionId,
            profile_name: 'default',
            session_type: 'conversation',
            metadata: { title: 'Partially available conversation' },
            created_at: '2026-07-11T00:00:00Z',
            updated_at: '2026-07-11T00:00:00Z',
            status: 'idle',
            run_count: 0,
            head_run_id: null,
            head_success_run_id: null,
            active_run_id: null,
            latest_run: null,
            runs: [],
            runs_limit: 20,
            runs_has_more: false,
            runs_next_before_sequence_no: null,
          },
          state: null,
          message: [],
        })
      }),
    )

    const { router, user } = await renderApp({
      connected: true,
      route: `/conversations/sessions/${sessionId}`,
    })

    expect(
      await screen.findByRole('heading', {
        level: 1,
        name: 'Partially available conversation',
      }),
    ).toBeVisible()
    expect(
      await screen.findByRole(
        'heading',
        { name: 'Conversation list could not be refreshed' },
        { timeout: 5_000 },
      ),
    ).toBeVisible()
    expect(
      await screen.findByRole(
        'heading',
        { name: 'Some conversation history could not be loaded' },
        { timeout: 5_000 },
      ),
    ).toBeVisible()
    expect(screen.getByRole('textbox', { name: 'Message' })).toBeVisible()

    await user.click(screen.getByRole('tab', { name: 'Workspace' }))

    expect(
      screen.getByRole('heading', {
        name: 'Workspace details could not be loaded',
      }),
    ).toBeVisible()
    expect(router.state.location.pathname).toBe(
      `/conversations/sessions/${sessionId}`,
    )
  })

  it('preserves an invalid conversation deep link and shows not found', async () => {
    const missingSessionId = 'missing-conversation'
    apiServer.use(
      http.get('*/api/v1/sessions/page', () =>
        HttpResponse.json([
          {
            id: 'existing-session',
            session_type: 'conversation',
            metadata: { title: 'Existing conversation' },
            profile_name: 'default',
            created_at: '2026-07-11T00:00:00Z',
            updated_at: '2026-07-11T00:00:00Z',
            status: 'idle',
            run_count: 0,
            head_run_id: null,
            active_run_id: null,
            latest_run: null,
          },
        ]),
      ),
      http.get('*/api/v1/sessions/:sessionId/workspace', () =>
        HttpResponse.json({ detail: 'Session not found' }, { status: 404 }),
      ),
      http.get('*/api/v1/sessions/:sessionId', () =>
        HttpResponse.json({ detail: 'Session not found' }, { status: 404 }),
      ),
    )

    const { router } = await renderApp({
      connected: true,
      route: `/conversations/sessions/${missingSessionId}`,
    })

    expect(
      await screen.findByRole('heading', {
        level: 2,
        name: 'Conversation not found',
      }),
    ).toBeVisible()
    expect(router.state.location.pathname).toBe(
      `/conversations/sessions/${missingSessionId}`,
    )
  })

  it('preserves explicit new-conversation intent when history exists', async () => {
    apiServer.use(
      http.get('*/api/v1/sessions/page', () =>
        HttpResponse.json([
          {
            id: 'bridge-session-1',
            session_type: 'conversation',
            metadata: { bridge: { adapter: 'lark' } },
            profile_name: 'default',
            created_at: '2026-07-11T00:00:00Z',
            updated_at: '2026-07-11T00:00:00Z',
            status: 'idle',
            run_count: 1,
            head_run_id: 'bridge-run-1',
            active_run_id: null,
            latest_run: {
              id: 'bridge-run-1',
              session_id: 'bridge-session-1',
              sequence_no: 1,
              status: 'completed',
              trigger_type: 'bridge',
              input_preview: 'Existing channel conversation',
              created_at: '2026-07-11T00:00:00Z',
            },
          },
        ]),
      ),
    )

    const { router } = await renderApp({
      connected: true,
      route: '/conversations/new',
    })

    expect(
      await screen.findByRole('heading', { level: 1, name: 'New chat' }),
    ).toBeVisible()
    await waitFor(() => {
      expect(screen.getByText('Existing channel conversation')).toBeVisible()
      expect(router.state.location.pathname).toBe('/conversations/new')
    })
  })

  it.each([
    ['/chat', '/conversations'],
    ['/chat/sessions/chat%20session', '/conversations/sessions/chat%20session'],
    [
      '/chat/sessions/chat%20session/runs/chat%23run',
      '/conversations/sessions/chat%20session/runs/chat%23run',
    ],
    ['/debug', '/activity'],
    ['/debug/sessions/debug%20session', '/activity/sessions/debug%20session'],
    [
      '/debug/sessions/debug%20session/runs/debug%23run',
      '/activity/sessions/debug%20session/runs/debug%23run',
    ],
    ['/schedules', '/automation/schedules'],
    ['/workflows', '/automation/workflows'],
    ['/agency', '/automation/agency'],
    [
      '/agency/sessions/agency%20session',
      '/automation/agency/sessions/agency%20session',
    ],
    [
      '/agency/sessions/agency%20session/runs/agency%23run',
      '/automation/agency/sessions/agency%20session/runs/agency%23run',
    ],
    ['/automation/background', '/automation/agency'],
    ['/heartbeat', '/automation/heartbeat'],
    ['/profiles', '/agents'],
    ['/profiles/__new__', '/agents/new'],
    ['/profiles/agent%20one', '/agents/by-name/agent%20one'],
    ['/agents/agent%20one', '/agents/by-name/agent%20one'],
    ['/bridges', '/integrations'],
  ] as const)(
    'redirects legacy URL %s to %s',
    async (legacyUrl, canonicalUrl) => {
      const { router } = await renderApp({ route: legacyUrl })

      await waitFor(() => {
        expect(router.state.location.href).toBe(canonicalUrl)
      })
    },
  )

  it.each(['desktop', 'mobile'] as const)(
    'navigates a %s Agency fire to its owning session and run',
    async (layout) => {
      const ownerSessionId = 'owner-agency-session'
      const ownerRunId = 'owner-agency-run'
      apiServer.use(
        http.get('*/api/v1/agency/fires', () =>
          HttpResponse.json({
            fires: [
              {
                id: 'owner-fire',
                kind: 'message_observed',
                status: 'consumed',
                source_session_id: 'source-session',
                source_run_id: 'source-run',
                agency_session_id: ownerSessionId,
                run_id: ownerRunId,
                active_run_id: null,
                run_status: 'completed',
                priority: 1,
                payload: {},
                error_message: null,
                created_at: '2026-07-11T00:00:00Z',
                updated_at: '2026-07-11T00:01:00Z',
                consumed_at: '2026-07-11T00:01:00Z',
              },
            ],
          }),
        ),
        http.get('*/api/v1/sessions/:sessionId', () =>
          HttpResponse.json({ detail: 'Session not found' }, { status: 404 }),
        ),
        http.get('*/api/v1/runs/:runId', () =>
          HttpResponse.json({ detail: 'Run not found' }, { status: 404 }),
        ),
        http.get('*/api/v1/runs/:runId/trace', () =>
          HttpResponse.json({ detail: 'Trace not found' }, { status: 404 }),
        ),
      )
      const { router, user } = await renderApp({
        connected: true,
        route: '/automation/agency',
      })
      const fireButton = await within(
        await screen.findByTestId(`agency-${layout}-layout`),
      ).findByRole('button', { name: /owner-fire/i })

      await user.click(fireButton)

      await waitFor(() => {
        expect(router.state.location.pathname).toBe(
          `/automation/agency/sessions/${ownerSessionId}/runs/${ownerRunId}`,
        )
      })
    },
  )

  it('canonicalizes an Agency run URL to the run-owned session', async () => {
    const actualSessionId = 'actual-agency-session'
    const runId = 'agency-run'
    const run = {
      id: runId,
      session_id: actualSessionId,
      sequence_no: 1,
      status: 'completed',
      trigger_type: 'agency',
      profile_name: 'default',
      input_preview: 'Review pending work',
      output_text: 'No action required.',
      created_at: '2026-07-11T00:00:00Z',
      finished_at: '2026-07-11T00:01:00Z',
      metadata: {},
      has_state: false,
      has_message: true,
    }
    const session = {
      id: actualSessionId,
      profile_name: 'default',
      session_type: 'agency',
      source_session_id: null,
      metadata: {},
      created_at: '2026-07-11T00:00:00Z',
      updated_at: '2026-07-11T00:01:00Z',
      status: 'idle',
      run_count: 1,
      head_run_id: runId,
      head_success_run_id: runId,
      active_run_id: null,
      latest_run: run,
    }
    const requestedSessionIds: string[] = []
    apiServer.use(
      http.get('*/api/v1/sessions/:sessionId', ({ params }) => {
        const requestedSessionId = String(params.sessionId)
        requestedSessionIds.push(requestedSessionId)
        const isActualSession = requestedSessionId === actualSessionId
        return HttpResponse.json({
          session: {
            ...session,
            id: requestedSessionId,
            run_count: isActualSession ? 1 : 0,
            head_run_id: isActualSession ? runId : null,
            head_success_run_id: isActualSession ? runId : null,
            latest_run: isActualSession ? run : null,
            runs: isActualSession ? [run] : [],
            runs_limit: 6,
            runs_has_more: false,
            runs_next_before_sequence_no: null,
          },
          state: null,
          message: [],
        })
      }),
      http.get('*/api/v1/runs/:runId', () =>
        HttpResponse.json({ session, run, state: null, message: [] }),
      ),
      http.get('*/api/v1/runs/:runId/trace', () =>
        HttpResponse.json({
          run_id: runId,
          session_id: actualSessionId,
          item_count: 0,
          max_item_chars: 16_000,
          max_total_chars: 128_000,
          truncated: false,
          trace: [],
        }),
      ),
    )

    const { router } = await renderApp({
      connected: true,
      route: `/automation/agency/sessions/wrong-session/runs/${runId}`,
    })

    expect(
      await screen.findByRole('heading', { level: 1, name: 'Proactive agent' }),
    ).toBeVisible()
    await waitFor(() => {
      expect(router.state.location.pathname).toBe(
        `/automation/agency/sessions/${actualSessionId}/runs/${runId}`,
      )
      expect(requestedSessionIds).toContain('wrong-session')
      expect(requestedSessionIds).toContain(actualSessionId)
    })
  })

  it('keeps the Agency workspace usable when an optional run detail fails', async () => {
    apiServer.use(
      http.get('*/api/v1/sessions/:sessionId', ({ params }) =>
        HttpResponse.json({
          session: {
            id: params.sessionId,
            profile_name: 'default',
            session_type: 'agency',
            source_session_id: null,
            metadata: {},
            created_at: '2026-07-11T00:00:00Z',
            updated_at: '2026-07-11T00:00:00Z',
            status: 'idle',
            run_count: 0,
            head_run_id: null,
            head_success_run_id: null,
            active_run_id: null,
            latest_run: null,
            runs: [],
            runs_limit: 6,
            runs_has_more: false,
            runs_next_before_sequence_no: null,
          },
          state: null,
          message: [],
        }),
      ),
      http.get('*/api/v1/runs/:runId', () =>
        HttpResponse.json(
          { detail: 'Run details unavailable' },
          { status: 404 },
        ),
      ),
      http.get('*/api/v1/runs/:runId/trace', () =>
        HttpResponse.json({ detail: 'Trace unavailable' }, { status: 404 }),
      ),
    )

    await renderApp({
      connected: true,
      route: '/automation/agency/sessions/agency-session/runs/missing-run',
    })

    expect(
      await screen.findByText('Some agency details could not be loaded'),
    ).toBeVisible()
    expect(screen.getByTestId('agency-desktop-layout')).toBeInTheDocument()
    expect(screen.getByTestId('agency-mobile-layout')).toBeInTheDocument()
  })

  it.each([
    ['/conversations/new', 'New chat'],
    ['/activity', 'Activity'],
    ['/automation', 'Work that continues without supervision'],
    ['/automation/agency', 'Proactive agent'],
    ['/automation/workflows', 'Workflows'],
    ['/automation/schedules', 'Schedules'],
    ['/workspace', 'Files, memory, and artifacts'],
    ['/agents', 'Agents'],
    ['/integrations', 'Integrations'],
  ] as const)(
    'gives %s exactly one route-owned primary heading',
    async (route, title) => {
      await renderApp({ connected: true, route })

      expect(
        await screen.findByRole('heading', { level: 1, name: title }),
      ).toBeVisible()
      expect(screen.getAllByRole('heading', { level: 1 })).toHaveLength(1)
    },
  )

  it.each([
    ['/automation/workflows', 'Workflows', 320],
    ['/automation/workflows', 'Workflows', 390],
    ['/automation/workflows/new', 'Workflows', 320],
    ['/automation/workflows/new', 'Workflows', 390],
    ['/automation/schedules', 'Schedules', 320],
    ['/automation/schedules', 'Schedules', 390],
    ['/automation/schedules/new', 'Schedules', 320],
    ['/automation/schedules/new', 'Schedules', 390],
    ['/agents', 'Agents', 320],
    ['/agents', 'Agents', 390],
    ['/agents/new', 'Agents', 320],
    ['/agents/new', 'Agents', 390],
    ['/integrations', 'Integrations', 320],
    ['/integrations', 'Integrations', 390],
    ['/integrations/setup', 'Integrations', 320],
    ['/integrations/setup', 'Integrations', 390],
  ] as const)(
    'keeps one route-level primary heading on the mobile route %s (%s) at %ipx',
    async (route, title, width) => {
      await renderApp({
        connected: true,
        route,
        viewport: { width },
      })

      expect(
        await screen.findByRole('heading', { level: 1, name: title }),
      ).toBeInTheDocument()
      expect(screen.getAllByRole('heading', { level: 1 })).toHaveLength(1)
    },
  )

  it.each([
    ['/activity', 'activity'],
    ['/automation/agency', 'agency'],
  ] as const)(
    'separates desktop and mobile inspector layouts for %s',
    async (route, layoutName) => {
      await renderApp({ connected: true, route })

      expect(
        await screen.findByTestId(`${layoutName}-desktop-layout`),
      ).toHaveClass('hidden', 'lg:block')
      expect(screen.getByTestId(`${layoutName}-mobile-layout`)).toHaveClass(
        'lg:hidden',
      )
    },
  )

  it.each([
    ['/', 'What should YA Claw work on?'],
    ['/conversations/new', 'New chat'],
    ['/activity', 'Activity'],
    ['/automation/agency', 'Proactive agent'],
    ['/workspace', 'Files, memory, and artifacts'],
    ['/agents', 'Agents'],
    ['/integrations', 'Integrations'],
    ['/settings', 'Settings & runtime'],
  ] as const)(
    'has no automatically detectable axe violations on %s',
    async (route, title) => {
      await renderApp({ connected: true, route })
      await screen.findByRole('heading', { level: 1, name: title })

      const violations = await getA11yViolations()
      expect(violations, formatA11yViolations(violations)).toEqual([])
    },
  )

  it('moves focus and updates the document title after SPA navigation', async () => {
    const { router, user } = await renderApp({ connected: true })
    await screen.findByRole('heading', {
      name: 'What should YA Claw work on?',
    })

    await user.click(screen.getAllByRole('link', { name: /Automation/ })[0])

    await waitFor(() => {
      expect(router.state.location.pathname).toBe('/automation')
      expect(document.activeElement).toBe(
        document.getElementById('main-content'),
      )
      expect(document.title).toBe('Automation · YA Claw')
    })
  })

  it('does not force main focus across detail-only pathname changes', async () => {
    const { router } = await renderApp({
      connected: true,
      route: '/automation/workflows',
    })
    const search = await screen.findByLabelText('Search workflows')
    await waitFor(() =>
      expect(document.activeElement).toBe(
        document.getElementById('main-content'),
      ),
    )
    search.focus()
    expect(document.activeElement).toBe(search)

    await act(async () => {
      await router.navigate({
        to: '/automation/workflows/$workflowId',
        params: { workflowId: 'new' },
      })
    })

    await waitFor(() => {
      expect(router.state.location.pathname).toBe('/automation/workflows/new')
      expect(document.activeElement).not.toBe(
        document.getElementById('main-content'),
      )
      expect(document.title).toBe('Workflows · YA Claw')
    })
  })

  it('keeps route focus on main after navigating from the mobile dialog', async () => {
    const { router, user } = await renderApp({
      connected: true,
      viewport: { width: 390 },
    })
    await screen.findByRole('heading', {
      name: 'What should YA Claw work on?',
    })

    await user.click(screen.getByRole('button', { name: 'Open navigation' }))
    const dialog = await screen.findByRole('dialog', { name: 'Navigate' })
    await user.click(within(dialog).getByRole('link', { name: /Automation/ }))

    await waitFor(() => {
      expect(router.state.location.pathname).toBe('/automation')
      expect(document.activeElement).toBe(
        document.getElementById('main-content'),
      )
    })
  })

  it.each(REFERENCE_VIEWPORTS)(
    'keeps navigation reachable and emits no DOM overflow signal at %ipx',
    async (width) => {
      const { user } = await renderApp({
        connected: true,
        viewport: { width },
      })
      await screen.findByRole('heading', {
        name: 'What should YA Claw work on?',
      })
      await screen.findByRole('heading', { name: 'No conversations yet' })

      const navigationHrefs = Array.from(
        document.querySelectorAll<HTMLAnchorElement>('nav a[href]'),
        (link) => new URL(link.href).pathname,
      )
      for (const destination of NAVIGATION_DESTINATIONS) {
        expect(navigationHrefs).toContain(destination)
      }

      await user.click(screen.getByRole('button', { name: 'Open navigation' }))
      const dialog = await screen.findByRole('dialog', { name: 'Navigate' })
      for (const label of [
        'Home',
        'Conversations',
        'Activity',
        'Automation',
        'Workspace',
        'Agents',
        'Integrations',
        'Settings',
      ]) {
        expect(
          within(dialog).getByRole('link', { name: new RegExp(label) }),
        ).toBeInTheDocument()
      }
      expect(
        within(dialog).getByRole('region', { name: 'Runtime status' }),
      ).toBeVisible()
      expect(
        within(dialog).getByRole('button', { name: /Disconnect runtime/ }),
      ).toBeVisible()

      expect(findHorizontalOverflowSignals()).toEqual([])
    },
  )

  it('guards global disconnect when the Settings connection draft is dirty', async () => {
    const { user } = await renderApp({
      connected: true,
      route: '/settings',
      viewport: { width: 390 },
    })
    await screen.findByRole('heading', { name: 'Settings & runtime' })
    await user.type(screen.getByLabelText('Backend URL'), '/changed')

    await user.click(screen.getByRole('button', { name: 'Open navigation' }))
    let navigation = await screen.findByRole('dialog', { name: 'Navigate' })
    await user.click(
      within(navigation).getByRole('button', { name: /Disconnect runtime/ }),
    )

    let confirmation = await screen.findByRole('dialog', {
      name: 'Discard connection changes and disconnect?',
    })
    expect(
      screen.queryByText('Connect to your runtime'),
    ).not.toBeInTheDocument()
    await user.click(
      within(confirmation).getByRole('button', { name: 'Keep editing' }),
    )

    navigation = await screen.findByRole('dialog', { name: 'Navigate' })
    await user.click(
      within(navigation).getByRole('button', { name: /Disconnect runtime/ }),
    )
    confirmation = await screen.findByRole('dialog', {
      name: 'Discard connection changes and disconnect?',
    })
    await user.click(
      within(confirmation).getByRole('button', {
        name: 'Discard and disconnect',
      }),
    )

    expect(
      await screen.findByRole('heading', { name: 'Connect to your runtime' }),
    ).toBeVisible()
  })

  it('disconnects from More and clears entity selection without UI preferences', async () => {
    const { user, router } = await renderApp({
      connected: true,
      viewport: { width: 375 },
      layoutState: {
        advancedMode: true,
        railCollapsed: true,
        selectedSessionId: 'session-a',
        selectedRunId: 'run-a',
        selectedDebugSessionId: 'session-a',
        selectedDebugRunId: 'run-a',
      },
    })

    await screen.findByRole('button', { name: 'Open navigation' })
    await user.click(screen.getByRole('button', { name: 'Open navigation' }))
    const dialog = await screen.findByRole('dialog', { name: 'Navigate' })
    await user.click(
      within(dialog).getByRole('button', { name: /Disconnect runtime/ }),
    )

    expect(
      await screen.findByRole('heading', { name: 'Connect to your runtime' }),
    ).toBeVisible()
    await waitFor(() => {
      expect(router.history.location.pathname).toBe('/')
      expect(useLayoutStore.getState()).toMatchObject({
        route: 'overview',
        selectedSessionId: null,
        selectedRunId: null,
        selectedDebugSessionId: null,
        selectedDebugRunId: null,
        advancedMode: true,
        railCollapsed: true,
      })
    })
  })
})
