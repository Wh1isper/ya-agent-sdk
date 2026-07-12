import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import type { ReactNode } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import * as hooks from '../../api/hooks'
import type { BridgeConversationSummary, BridgeEventSummary } from '../../types'
import { BridgesPage } from './BridgesPage'

vi.mock('@tanstack/react-router', () => ({
  useRouterState: ({
    select,
  }: {
    select: (state: { location: { pathname: string } }) => unknown
  }) => select({ location: { pathname: window.location.pathname } }),
  Link: ({
    to,
    params,
    children,
    ...props
  }: {
    to: string
    params?: Record<string, string>
    children: ReactNode
  }) => {
    const href = Object.entries(params ?? {}).reduce(
      (path, [key, value]) =>
        path.replace(`$${key}`, encodeURIComponent(value)),
      to,
    )
    return (
      <a href={href} {...props}>
        {children}
      </a>
    )
  },
}))

vi.mock('../../api/hooks', () => ({
  useBridgeConversationsQuery: vi.fn(),
  useBridgeEventsQuery: vi.fn(),
}))

const refetchConversations = vi.fn()
const refetchEvents = vi.fn()

function mockEvents(overrides: Record<string, unknown> = {}) {
  vi.mocked(hooks.useBridgeEventsQuery).mockReturnValue({
    data: { events: [] },
    isLoading: false,
    isError: false,
    error: null,
    refetch: refetchEvents,
    ...overrides,
  } as unknown as ReturnType<typeof hooks.useBridgeEventsQuery>)
}

describe('Integrations page', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    window.history.replaceState(null, '', '/integrations')
    mockEvents()
  })

  it('provides actionable connection setup when no channel is configured', async () => {
    const user = userEvent.setup()
    vi.mocked(hooks.useBridgeConversationsQuery).mockReturnValue({
      data: { conversations: [] },
      isLoading: false,
      isError: false,
      error: null,
      refetch: refetchConversations,
    } as unknown as ReturnType<typeof hooks.useBridgeConversationsQuery>)

    const view = render(<BridgesPage />)

    expect(
      screen.getByRole('heading', { level: 1, name: 'Integrations' }),
    ).toBeInTheDocument()
    expect(
      screen.getByRole('heading', { name: 'Connect your first channel' }),
    ).toBeInTheDocument()
    expect(screen.queryByText('Configure credentials')).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'View setup steps' }))
    view.rerender(<BridgesPage />)
    expect(screen.getByText('Configure credentials')).toBeVisible()
    expect(screen.getByText('Register delivery')).toBeVisible()

    await user.click(screen.getByRole('button', { name: 'Check connection' }))
    expect(refetchConversations).toHaveBeenCalledOnce()
  })

  it('links bridge sessions and runs to explicit Activity detail routes', async () => {
    vi.mocked(hooks.useBridgeConversationsQuery).mockReturnValue({
      data: {
        conversations: [
          {
            id: 'conversation-1',
            adapter: 'lark',
            tenant_key: 'tenant-1',
            external_chat_id: 'chat-1',
            session_id: 'session/bridge 1',
            profile_name: 'support',
            metadata: {},
            active_run_id: 'active/run 1',
            event_count: 1,
            latest_event_status: 'submitted',
            created_at: '2026-01-01T00:00:00Z',
            updated_at: '2026-01-01T00:00:00Z',
            last_event_at: '2026-01-01T00:00:00Z',
          },
        ],
      },
      isLoading: false,
      isError: false,
      error: null,
      refetch: refetchConversations,
    } as unknown as ReturnType<typeof hooks.useBridgeConversationsQuery>)
    mockEvents({
      data: {
        events: [
          {
            id: 'event-1',
            adapter: 'lark',
            tenant_key: 'tenant-1',
            event_id: 'external-event-1',
            external_message_id: 'message-1',
            external_chat_id: 'chat-1',
            conversation_id: 'conversation-1',
            session_id: 'event/session 1',
            run_id: 'event/run 1',
            run_status: 'completed',
            event_type: 'message',
            status: 'submitted',
            raw_event: {},
            normalized_event: {},
            created_at: '2026-01-01T00:00:00Z',
            updated_at: '2026-01-01T00:00:00Z',
          },
        ],
      },
    })

    window.history.replaceState(
      null,
      '',
      '/integrations/conversations/conversation-1',
    )
    render(<BridgesPage />)

    expect(
      await screen.findByRole('link', { name: 'Open active run' }),
    ).toHaveAttribute(
      'href',
      '/activity/sessions/session%2Fbridge%201/runs/active%2Frun%201',
    )
    expect(screen.getAllByRole('link', { name: 'Open session' })).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          href: expect.stringContaining(
            '/activity/sessions/session%2Fbridge%201',
          ),
        }),
        expect.objectContaining({
          href: expect.stringContaining(
            '/activity/sessions/event%2Fsession%201',
          ),
        }),
      ]),
    )
    expect(screen.getByRole('link', { name: 'Open run' })).toHaveAttribute(
      'href',
      '/activity/sessions/event%2Fsession%201/runs/event%2Frun%201',
    )
  })

  it('keeps the list unselected and shows a selection prompt', () => {
    vi.mocked(hooks.useBridgeConversationsQuery).mockReturnValue({
      data: {
        conversations: [
          {
            id: 'conversation-1',
            adapter: 'lark',
            tenant_key: 'tenant-1',
            external_chat_id: 'chat-1',
            session_id: 'session-1',
            profile_name: 'support',
            metadata: {},
            active_run_id: null,
            event_count: 0,
            latest_event_status: null,
            created_at: '2026-01-01T00:00:00Z',
            updated_at: '2026-01-01T00:00:00Z',
            last_event_at: null,
          },
        ],
      },
      isLoading: false,
      isError: false,
      error: null,
      refetch: refetchConversations,
    } as unknown as ReturnType<typeof hooks.useBridgeConversationsQuery>)

    render(<BridgesPage />)

    expect(
      screen.getByRole('heading', { name: 'Select a bridge conversation' }),
    ).toBeVisible()
    expect(
      screen.queryByRole('link', { name: 'Open session' }),
    ).not.toBeInTheDocument()
  })

  it('preserves an invalid deep link and shows not found after the list loads', async () => {
    const user = userEvent.setup()
    window.history.replaceState(
      null,
      '',
      '/integrations/conversations/missing-conversation',
    )
    vi.mocked(hooks.useBridgeConversationsQuery).mockReturnValue({
      data: undefined,
      isLoading: true,
      isError: false,
      error: null,
      refetch: refetchConversations,
    } as unknown as ReturnType<typeof hooks.useBridgeConversationsQuery>)

    const view = render(<BridgesPage />)

    expect(
      screen.queryByRole('heading', {
        name: 'Integration conversation not found',
      }),
    ).not.toBeInTheDocument()
    expect(window.location.pathname).toBe(
      '/integrations/conversations/missing-conversation',
    )

    vi.mocked(hooks.useBridgeConversationsQuery).mockReturnValue({
      data: { conversations: [] },
      isLoading: false,
      isError: false,
      error: null,
      refetch: refetchConversations,
    } as unknown as ReturnType<typeof hooks.useBridgeConversationsQuery>)
    view.rerender(<BridgesPage />)

    expect(
      screen.getByRole('heading', {
        name: 'Integration conversation not found',
      }),
    ).toBeVisible()
    expect(screen.getByRole('alert')).toHaveTextContent('missing-conversation')
    expect(window.location.pathname).toBe(
      '/integrations/conversations/missing-conversation',
    )

    await user.click(
      screen.getByRole('button', { name: 'Return to integration list' }),
    )
    view.rerender(<BridgesPage />)
    expect(window.location.pathname).toBe('/integrations')
  })

  it('hides placeholder events and metrics when the conversation or status changes', async () => {
    const user = userEvent.setup()
    let secondConversationReady = false
    window.history.replaceState(
      null,
      '',
      '/integrations/conversations/conversation-1',
    )

    const conversation = (
      id: string,
      externalChatId: string,
    ): BridgeConversationSummary => ({
      id,
      adapter: 'lark',
      tenant_key: 'tenant-1',
      external_chat_id: externalChatId,
      session_id: `session-${id}`,
      profile_name: 'support',
      metadata: {},
      active_run_id: null,
      event_count: 1,
      latest_event_status: 'submitted',
      created_at: '2026-01-01T00:00:00Z',
      updated_at: '2026-01-01T00:00:00Z',
      last_event_at: '2026-01-01T00:00:00Z',
    })
    const bridgeEvent = (
      id: string,
      conversationId: string,
      messageId: string,
    ): BridgeEventSummary => ({
      id,
      adapter: 'lark',
      tenant_key: 'tenant-1',
      event_id: id,
      external_message_id: messageId,
      external_chat_id: `chat-${conversationId}`,
      conversation_id: conversationId,
      session_id: `session-${conversationId}`,
      run_id: `run-${id}`,
      run_status: 'completed',
      event_type: 'message',
      status: 'submitted',
      raw_event: {},
      normalized_event: {},
      created_at: '2026-01-01T00:00:00Z',
      updated_at: '2026-01-01T00:00:00Z',
    })
    const firstEvent = bridgeEvent(
      'event-1',
      'conversation-1',
      'message-from-conversation-1',
    )
    const secondEvent = bridgeEvent(
      'event-2',
      'conversation-2',
      'message-from-conversation-2',
    )

    vi.mocked(hooks.useBridgeConversationsQuery).mockReturnValue({
      data: {
        conversations: [
          conversation('conversation-1', 'chat-1'),
          conversation('conversation-2', 'chat-2'),
        ],
      },
      isLoading: false,
      isError: false,
      error: null,
      refetch: refetchConversations,
    } as unknown as ReturnType<typeof hooks.useBridgeConversationsQuery>)
    vi.mocked(hooks.useBridgeEventsQuery).mockImplementation((filters) => {
      const changingConversation =
        filters.conversationId === 'conversation-2' &&
        filters.status === 'all' &&
        !secondConversationReady
      const changingStatus = filters.status === 'failed'
      return {
        data: {
          events:
            filters.conversationId === 'conversation-1' || changingConversation
              ? [firstEvent]
              : [secondEvent],
        },
        isLoading: false,
        isPlaceholderData: changingConversation || changingStatus,
        isError: false,
        error: null,
        refetch: refetchEvents,
      } as unknown as ReturnType<typeof hooks.useBridgeEventsQuery>
    })

    const view = render(<BridgesPage />)

    expect(screen.getByText('message-from-conversation-1')).toBeVisible()
    expect(screen.getByText('Delivered').parentElement).toHaveTextContent(
      'Delivered1',
    )

    window.history.pushState(
      null,
      '',
      '/integrations/conversations/conversation-2',
    )
    view.rerender(<BridgesPage />)

    expect(hooks.useBridgeEventsQuery).toHaveBeenLastCalledWith({
      conversationId: 'conversation-2',
      status: 'all',
    })
    expect(
      screen.queryByText('message-from-conversation-1'),
    ).not.toBeInTheDocument()
    expect(
      screen.queryByText('message-from-conversation-2'),
    ).not.toBeInTheDocument()
    expect(screen.queryByText('Delivered')).not.toBeInTheDocument()
    expect(
      screen.getByRole('status', { name: 'Loading bridge metrics' }),
    ).toBeVisible()
    expect(
      screen.getByRole('status', { name: 'Loading bridge events' }),
    ).toBeVisible()

    secondConversationReady = true
    view.rerender(<BridgesPage />)
    expect(screen.getByText('message-from-conversation-2')).toBeVisible()
    expect(screen.getByText('Delivered').parentElement).toHaveTextContent(
      'Delivered1',
    )

    await user.selectOptions(
      screen.getByRole('combobox', {
        name: 'Filter bridge events by status',
      }),
      'failed',
    )

    expect(hooks.useBridgeEventsQuery).toHaveBeenLastCalledWith({
      conversationId: 'conversation-2',
      status: 'failed',
    })
    expect(
      screen.queryByText('message-from-conversation-2'),
    ).not.toBeInTheDocument()
    expect(screen.queryByText('Delivered')).not.toBeInTheDocument()
    expect(
      screen.getByRole('status', { name: 'Loading bridge metrics' }),
    ).toBeVisible()
    expect(
      screen.getByRole('status', { name: 'Loading bridge events' }),
    ).toBeVisible()
  })

  it('shows query errors with an explicit retry action', async () => {
    const user = userEvent.setup()
    vi.mocked(hooks.useBridgeConversationsQuery).mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true,
      error: new Error('adapter service unavailable'),
      refetch: refetchConversations,
    } as unknown as ReturnType<typeof hooks.useBridgeConversationsQuery>)

    render(<BridgesPage />)

    expect(screen.getByText('Could not load integrations')).toBeVisible()
    expect(screen.getByText('adapter service unavailable')).toBeVisible()
    expect(screen.getByText('Technical details')).toBeVisible()
    await user.click(screen.getByRole('button', { name: 'Try again' }))
    expect(refetchConversations).toHaveBeenCalledOnce()
  })
})
