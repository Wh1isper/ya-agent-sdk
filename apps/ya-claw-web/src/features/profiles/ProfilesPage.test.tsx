import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import * as hooks from '../../api/hooks'
import { useLayoutStore } from '../../stores/layoutStore'
import type { ProfileDetail } from '../../types'
import { ProfilesPage } from './ProfilesPage'

const seedMutateAsync = vi.fn()
const upsertMutateAsync = vi.fn()

const reloadedProfile: ProfileDetail = {
  name: 'server-agent',
  model: 'openai:gpt-5',
  enabled: true,
  updated_at: '2026-01-02T00:00:00Z',
  created_at: '2026-01-01T00:00:00Z',
  builtin_toolsets: [],
  toolsets: [],
  subagents: [],
  include_builtin_subagents: true,
  unified_subagents: false,
  need_user_approve_tools: [],
  need_user_approve_mcps: [],
  enabled_mcps: [],
  disabled_mcps: [],
  mcp_servers: {},
}
const cachedProfile: ProfileDetail = {
  ...reloadedProfile,
  model: 'openai:gpt-4.1',
  updated_at: '2026-01-01T00:00:00Z',
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

function setupExistingProfile(refetch: ReturnType<typeof vi.fn>) {
  useLayoutStore.setState({ selectedProfileName: cachedProfile.name })
  vi.mocked(hooks.useProfilesQuery).mockReturnValue({
    data: [cachedProfile],
    isLoading: false,
  } as unknown as ReturnType<typeof hooks.useProfilesQuery>)
  vi.mocked(hooks.useProfileQuery).mockReturnValue({
    data: cachedProfile,
    isLoading: false,
    refetch,
  } as unknown as ReturnType<typeof hooks.useProfileQuery>)
}

describe('ProfilesPage progressive disclosure', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    useBlocker.mockReturnValue(idleBlocker)
    setupHooks()
    useLayoutStore.setState({
      selectedProfileName: '__new__',
      route: 'profiles',
    })
  })

  it('starts with the essential sections open and advanced groups collapsed', async () => {
    const user = userEvent.setup()
    render(<ProfilesPage />)

    expect(screen.getByRole('button', { name: /Basics/ })).toHaveAttribute(
      'aria-expanded',
      'true',
    )
    expect(screen.getByRole('button', { name: /Behavior/ })).toHaveAttribute(
      'aria-expanded',
      'true',
    )
    expect(
      screen.getByRole('button', { name: /Tools & permissions/ }),
    ).toHaveAttribute('aria-expanded', 'false')
    expect(screen.queryByLabelText('Builtin toolsets')).not.toBeInTheDocument()

    const toolsSection = screen.getByRole('button', {
      name: /Tools & permissions/,
    })
    await user.click(toolsSection)
    expect(toolsSection).toHaveAttribute('aria-expanded', 'true')
    expect(screen.getByPlaceholderText('core, web, document')).toBeVisible()

    for (const name of [
      'MCP servers',
      'Subagents',
      'Runtime & safety',
      'Advanced JSON/source metadata',
    ]) {
      expect(
        screen.getByRole('button', { name: new RegExp(name) }),
      ).toHaveAttribute('aria-expanded', 'false')
    }
  })

  it('preserves dirty edits when a newer profile arrives in the background', async () => {
    const user = userEvent.setup()
    setupExistingProfile(vi.fn())
    const { rerender } = render(<ProfilesPage />)

    const model = screen.getByLabelText('Model')
    await user.clear(model)
    await user.type(model, 'local:model')

    const remoteProfile = {
      ...cachedProfile,
      model: 'remote:model',
      updated_at: '2026-02-01T00:00:00Z',
    }
    vi.mocked(hooks.useProfileQuery).mockReturnValue({
      data: remoteProfile,
      isLoading: false,
      refetch: vi.fn(),
    } as unknown as ReturnType<typeof hooks.useProfileQuery>)
    rerender(<ProfilesPage />)

    expect(screen.getByLabelText('Model')).toHaveValue('local:model')
    expect(
      screen.getByText(/A newer server version is available/),
    ).toBeVisible()

    vi.mocked(hooks.useProfileQuery).mockReturnValue({
      data: {
        ...cachedProfile,
        model: 'stale:model',
        updated_at: '2026-01-15T00:00:00Z',
      },
      isLoading: false,
      refetch: vi.fn(),
    } as unknown as ReturnType<typeof hooks.useProfileQuery>)
    rerender(<ProfilesPage />)

    await user.click(
      screen.getByRole('button', { name: 'Review latest version' }),
    )
    await user.click(
      screen.getByRole('button', { name: 'Discard changes and reload' }),
    )
    expect(screen.getByLabelText('Model')).toHaveValue('remote:model')
  })

  it('disables agent fields while a save is pending', () => {
    vi.mocked(hooks.useUpsertProfileMutation).mockReturnValue({
      mutateAsync: upsertMutateAsync,
      isPending: true,
    } as unknown as ReturnType<typeof hooks.useUpsertProfileMutation>)

    render(<ProfilesPage />)

    expect(screen.getByLabelText('Name')).toBeDisabled()
    expect(screen.getByLabelText('Model')).toBeDisabled()
  })

  it('does not navigate after a pending agent create resolves off-page', async () => {
    const user = userEvent.setup()
    let resolveUpsert: ((value: ProfileDetail) => void) | undefined
    const pendingUpsert = new Promise<ProfileDetail>((resolve) => {
      resolveUpsert = resolve
    })
    upsertMutateAsync.mockReturnValue(pendingUpsert)
    window.history.replaceState(null, '', '/agents/new')
    const view = render(<ProfilesPage />)

    await user.type(screen.getByLabelText('Name'), 'support-agent')
    await user.click(screen.getByRole('button', { name: 'Save agent' }))
    await waitFor(() => expect(upsertMutateAsync).toHaveBeenCalledOnce())

    window.history.replaceState(null, '', '/settings')
    view.unmount()
    await act(async () => {
      resolveUpsert?.({ ...reloadedProfile, name: 'support-agent' })
      await pendingUpsert
    })

    expect(window.location.pathname).toBe('/settings')
  })

  it('does not offer reload for a new agent', () => {
    render(<ProfilesPage />)

    expect(
      screen.getByRole('heading', { name: 'New agent' }),
    ).toBeInTheDocument()
    expect(
      screen.queryByRole('button', { name: 'Reload' }),
    ).not.toBeInTheDocument()
  })

  it('warns that deleting a dirty agent discards its edits', async () => {
    const user = userEvent.setup()
    setupExistingProfile(vi.fn())
    render(<ProfilesPage />)

    await user.type(screen.getByLabelText('System prompt'), 'Unsaved guidance')
    await user.click(screen.getByRole('button', { name: 'Delete' }))

    expect(screen.getByRole('dialog')).toHaveTextContent(
      /unsaved edits will be permanently discarded/i,
    )
    expect(
      screen.getByRole('button', { name: 'Discard edits and delete' }),
    ).toBeVisible()
  })

  it('reports required name and model validation as inline form errors', async () => {
    const user = userEvent.setup()
    render(<ProfilesPage />)

    await user.clear(screen.getByLabelText('Model'))
    await user.click(screen.getByRole('button', { name: 'Save agent' }))

    expect(screen.getByText('Profile name is required')).toHaveAttribute(
      'role',
      'alert',
    )
    expect(screen.getByText('Model is required')).toHaveAttribute(
      'role',
      'alert',
    )
    expect(screen.getByLabelText('Name')).toHaveAttribute(
      'aria-invalid',
      'true',
    )
    expect(screen.getByLabelText('Model')).toHaveAttribute(
      'aria-invalid',
      'true',
    )
    expect(upsertMutateAsync).not.toHaveBeenCalled()
  })

  it('confirms destructive seed pruning but seeds normally without confirmation', async () => {
    const user = userEvent.setup()
    seedMutateAsync.mockResolvedValue({ seeded_names: [] })
    render(<ProfilesPage />)

    await user.click(screen.getByRole('button', { name: 'Seed profiles' }))
    expect(seedMutateAsync).toHaveBeenCalledWith(false)
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()

    await user.click(
      screen.getByRole('checkbox', { name: 'Prune missing seeded profiles' }),
    )
    await user.click(screen.getByRole('button', { name: 'Seed profiles' }))

    expect(
      screen.getByRole('dialog', {
        name: 'Seed profiles and prune missing profiles?',
      }),
    ).toBeVisible()
    expect(seedMutateAsync).not.toHaveBeenCalledWith(true)

    await user.click(screen.getByRole('button', { name: 'Seed and prune' }))
    expect(seedMutateAsync).toHaveBeenCalledWith(true)
  })

  it('keeps a failed prune dialog open and displays the mutation error', async () => {
    const user = userEvent.setup()
    seedMutateAsync.mockRejectedValue(new Error('Seed source is unavailable'))
    render(<ProfilesPage />)

    await user.click(
      screen.getByRole('checkbox', { name: 'Prune missing seeded profiles' }),
    )
    await user.click(screen.getByRole('button', { name: 'Seed profiles' }))
    await user.click(screen.getByRole('button', { name: 'Seed and prune' }))

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'Seed source is unavailable',
    )
    expect(screen.getByRole('dialog')).toBeVisible()
  })

  it('confirms dirty reload, awaits refetch, and resets from the refetch result', async () => {
    const user = userEvent.setup()
    let resolveRefetch: ((result: unknown) => void) | undefined
    const refetch = vi.fn(
      () =>
        new Promise((resolve) => {
          resolveRefetch = resolve
        }),
    )
    setupExistingProfile(refetch)
    render(<ProfilesPage />)

    const modelInput = screen.getByLabelText('Model')
    await user.clear(modelInput)
    await user.type(modelInput, 'local-model')
    await user.click(screen.getByRole('button', { name: 'Reload' }))

    expect(
      screen.getByRole('dialog', {
        name: 'Discard unsaved agent changes and reload?',
      }),
    ).toBeVisible()
    expect(refetch).not.toHaveBeenCalled()

    await user.click(
      screen.getByRole('button', { name: 'Discard changes and reload' }),
    )
    expect(refetch).toHaveBeenCalledOnce()
    expect(modelInput).toHaveValue('local-model')

    await act(async () => {
      resolveRefetch?.({ data: reloadedProfile, error: null })
    })
    await vi.waitFor(() =>
      expect(screen.getByLabelText('Model')).toHaveValue('openai:gpt-5'),
    )
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
  })

  it('keeps the dirty reload confirmation open when refetch fails', async () => {
    const user = userEvent.setup()
    const refetch = vi.fn().mockResolvedValue({
      data: undefined,
      error: new Error('Agent reload failed'),
    })
    setupExistingProfile(refetch)
    render(<ProfilesPage />)

    const modelInput = screen.getByLabelText('Model')
    await user.clear(modelInput)
    await user.type(modelInput, 'local-model')
    await user.click(screen.getByRole('button', { name: 'Reload' }))
    await user.click(
      screen.getByRole('button', { name: 'Discard changes and reload' }),
    )

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'Agent reload failed',
    )
    expect(screen.getByRole('dialog')).toBeVisible()
    expect(modelInput).toHaveValue('local-model')
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

    expect(
      screen.getByRole('dialog', {
        name: 'Discard unsaved agent changes?',
      }),
    ).toBeVisible()
    await user.click(screen.getByRole('button', { name: 'Discard and leave' }))
    expect(proceed).toHaveBeenCalledOnce()
    expect(reset).not.toHaveBeenCalled()
  })

  it('enables the TanStack SPA and beforeunload guard after an edit', async () => {
    const user = userEvent.setup()
    render(<ProfilesPage />)

    await user.type(screen.getByLabelText('Name'), 'support-agent')

    expect(useBlocker).toHaveBeenLastCalledWith(
      expect.objectContaining({
        disabled: false,
        enableBeforeUnload: true,
        withResolver: true,
      }),
    )
  })
})
