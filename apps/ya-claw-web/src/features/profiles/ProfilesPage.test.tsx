import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import * as hooks from '../../api/hooks'
import { useLayoutStore } from '../../stores/layoutStore'
import type { ProfileDetail } from '../../types'
import { ProfilesPage } from './ProfilesPage'

const seedMutateAsync = vi.fn()
const upsertMutateAsync = vi.fn()

const storedProfile: ProfileDetail = {
  schema_version: 2,
  name: 'server-agent',
  model: 'openai:gpt-5',
  enabled: true,
  updated_at: '2026-01-02T00:00:00Z',
  created_at: '2026-01-01T00:00:00Z',
  source_type: 'api',
  agent: {
    model: 'openai:gpt-5',
    name: 'server-agent',
    instructions: 'Use evidence.',
    capabilities: ['FilesystemCapability'],
  },
  host: {
    tool_groups: ['session'],
    need_user_approve_tools: [],
    need_user_approve_mcps: [],
    enabled_mcps: [],
    disabled_mcps: [],
    mcp_servers: {},
  },
  subagents: [],
}

const { useBlocker, idleBlocker } = vi.hoisted(() => {
  const idleBlocker = {
    status: 'idle' as const,
    current: undefined,
    next: undefined,
    action: undefined,
    proceed: undefined,
    reset: undefined,
  }
  return { useBlocker: vi.fn(() => idleBlocker), idleBlocker }
})

vi.mock('@tanstack/react-router', () => ({ useBlocker }))
vi.mock('../../api/hooks', () => ({
  useDeleteProfileMutation: vi.fn(),
  useProfileQuery: vi.fn(),
  useProfilesQuery: vi.fn(),
  useSeedProfilesMutation: vi.fn(),
  useUpsertProfileMutation: vi.fn(),
}))

function setupHooks() {
  vi.mocked(hooks.useProfilesQuery).mockReturnValue({
    data: [],
    isLoading: false,
  } as unknown as ReturnType<typeof hooks.useProfilesQuery>)
  vi.mocked(hooks.useProfileQuery).mockReturnValue({
    data: undefined,
    isLoading: false,
    refetch: vi.fn(),
  } as unknown as ReturnType<typeof hooks.useProfileQuery>)
  vi.mocked(hooks.useSeedProfilesMutation).mockReturnValue({
    mutateAsync: seedMutateAsync,
    isPending: false,
  } as unknown as ReturnType<typeof hooks.useSeedProfilesMutation>)
  vi.mocked(hooks.useUpsertProfileMutation).mockReturnValue({
    mutateAsync: upsertMutateAsync,
    isPending: false,
  } as unknown as ReturnType<typeof hooks.useUpsertProfileMutation>)
  vi.mocked(hooks.useDeleteProfileMutation).mockReturnValue({
    mutateAsync: vi.fn(),
    isPending: false,
  } as unknown as ReturnType<typeof hooks.useDeleteProfileMutation>)
}

function setupExistingProfile(profile: ProfileDetail = storedProfile) {
  useLayoutStore.setState({ selectedProfileName: profile.name })
  vi.mocked(hooks.useProfilesQuery).mockReturnValue({
    data: [profile],
    isLoading: false,
  } as unknown as ReturnType<typeof hooks.useProfilesQuery>)
  vi.mocked(hooks.useProfileQuery).mockReturnValue({
    data: profile,
    isLoading: false,
    refetch: vi.fn(),
  } as unknown as ReturnType<typeof hooks.useProfileQuery>)
}

describe('ProfilesPage native profile editor', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    useBlocker.mockReturnValue(idleBlocker)
    setupHooks()
    useLayoutStore.setState({
      selectedProfileName: '__new__',
      route: 'profiles',
    })
  })

  it('edits the three versioned profile boundaries directly', () => {
    render(<ProfilesPage />)

    expect(
      screen.getByRole('heading', { name: 'New native profile' }),
    ).toBeVisible()
    expect(screen.getByLabelText('AgentSpec JSON')).toBeVisible()
    expect(screen.getByLabelText('Host policy JSON')).toBeVisible()
    expect(screen.getByLabelText('SubagentSpec list JSON')).toBeVisible()
    expect(screen.queryByText('Unified subagents')).not.toBeInTheDocument()
    expect(screen.queryByText('Builtin toolsets')).not.toBeInTheDocument()
  })

  it('submits a native version 2 payload', async () => {
    const user = userEvent.setup()
    upsertMutateAsync.mockResolvedValue({
      ...storedProfile,
      name: 'support-agent',
      agent: { ...storedProfile.agent, name: 'support-agent' },
    })
    render(<ProfilesPage />)

    await user.type(screen.getByLabelText('Name'), 'support-agent')
    fireEvent.change(screen.getByLabelText('AgentSpec JSON'), {
      target: {
        value: JSON.stringify({
          model: 'test',
          instructions: 'Help the user.',
          capabilities: ['FilesystemCapability'],
        }),
      },
    })
    await user.click(screen.getByRole('button', { name: 'Save agent' }))

    await waitFor(() => expect(upsertMutateAsync).toHaveBeenCalledOnce())
    expect(upsertMutateAsync).toHaveBeenCalledWith(
      expect.objectContaining({
        name: 'support-agent',
        payload: expect.objectContaining({
          schema_version: 2,
          agent: expect.objectContaining({
            model: 'test',
            name: 'support-agent',
          }),
          host: expect.objectContaining({ tool_groups: ['session'] }),
          subagents: [],
        }),
      }),
    )
  })

  it('preserves dirty native JSON when a remote profile arrives', () => {
    setupExistingProfile()
    const { rerender } = render(<ProfilesPage />)
    const agent = screen.getByLabelText('AgentSpec JSON')
    fireEvent.change(agent, { target: { value: '{"model":"local"}' } })

    setupExistingProfile({
      ...storedProfile,
      model: 'remote',
      updated_at: '2026-02-01T00:00:00Z',
      agent: { model: 'remote', name: 'server-agent' },
    })
    rerender(<ProfilesPage />)

    expect(screen.getByLabelText('AgentSpec JSON')).toHaveValue(
      '{"model":"local"}',
    )
    expect(
      screen.getByText(/A newer server version is available/),
    ).toBeVisible()
  })

  it('disables the native document fields while save is pending', () => {
    vi.mocked(hooks.useUpsertProfileMutation).mockReturnValue({
      mutateAsync: upsertMutateAsync,
      isPending: true,
    } as unknown as ReturnType<typeof hooks.useUpsertProfileMutation>)
    render(<ProfilesPage />)

    expect(screen.getByLabelText('Name')).toBeDisabled()
    expect(screen.getByLabelText('AgentSpec JSON')).toBeDisabled()
  })

  it('confirms destructive seed pruning', async () => {
    const user = userEvent.setup()
    seedMutateAsync.mockResolvedValue({ seeded_names: [] })
    render(<ProfilesPage />)

    await user.click(
      screen.getByRole('checkbox', { name: 'Prune missing seeded profiles' }),
    )
    await user.click(screen.getByRole('button', { name: 'Seed profiles' }))
    expect(
      screen.getByRole('dialog', {
        name: 'Seed profiles and prune missing profiles?',
      }),
    ).toBeVisible()
    await user.click(screen.getByRole('button', { name: 'Seed and prune' }))
    expect(seedMutateAsync).toHaveBeenCalledWith(true)
  })

  it('resolves blocked SPA navigation through the shared dialog', async () => {
    const user = userEvent.setup()
    const proceed = vi.fn()
    const reset = vi.fn()
    useBlocker.mockReturnValue({
      status: 'blocked',
      current: {},
      next: {},
      action: 'PUSH',
      proceed,
      reset,
    } as never)
    render(<ProfilesPage />)

    await user.click(screen.getByRole('button', { name: 'Discard and leave' }))
    expect(proceed).toHaveBeenCalledOnce()
  })
})
