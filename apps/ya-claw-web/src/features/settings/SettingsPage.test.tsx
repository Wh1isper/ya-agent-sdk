import { act, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const { useBlocker } = vi.hoisted(() => ({ useBlocker: vi.fn() }))

vi.mock('@tanstack/react-router', () => ({ useBlocker }))

import * as connection from '../../api/connection'
import * as hooks from '../../api/hooks'
import { useConnectionStore } from '../../stores/connectionStore'
import { SettingsPage } from './SettingsPage'

vi.mock('../../api/hooks', () => ({
  useClawInfoQuery: vi.fn(),
  useHealthQuery: vi.fn(),
  useWorkspaceRuntimeQuery: vi.fn(),
}))
vi.mock('../../api/connection', () => ({
  validateConnection: vi.fn(),
  getConnectionErrorMessage: vi.fn((error: unknown) =>
    error instanceof Error ? error.message : 'Connection failed',
  ),
}))

const healthRefetch = vi.fn()
const infoRefetch = vi.fn()
const runtimeRefetch = vi.fn()

function setupHealthyRuntime() {
  vi.mocked(hooks.useHealthQuery).mockReturnValue({
    data: { status: 'healthy', database: 'ready', runtime_state: 'running' },
    isLoading: false,
    isError: false,
    isFetching: false,
    error: null,
    refetch: healthRefetch,
  } as unknown as ReturnType<typeof hooks.useHealthQuery>)
  vi.mocked(hooks.useClawInfoQuery).mockReturnValue({
    data: {
      name: 'YA Claw',
      environment: 'production',
      version: '2.4.0',
      service_version: '2.4.1',
      service_revision: 'rev-42',
      instance_id: 'instance-a',
      auth: 'bearer',
      surfaces: ['web', 'api'],
      workspace_provider_backend: 'docker',
      storage_model: 'postgres',
      public_base_url: 'https://claw.example.test',
      features: {
        session_events: true,
        run_events: true,
        notifications: true,
        profiles: true,
      },
    },
    isLoading: false,
    isError: false,
    isFetching: false,
    error: null,
    refetch: infoRefetch,
  } as unknown as ReturnType<typeof hooks.useClawInfoQuery>)
  vi.mocked(hooks.useWorkspaceRuntimeQuery).mockReturnValue({
    data: {
      backend: 'docker',
      status: 'ready',
      execution_location: 'local daemon',
      workspace: {
        service_path: '/workspace',
        docker_host_path: '/srv/workspaces',
        virtual_path: '/work',
        exists: true,
        writable: true,
      },
      capabilities: {
        file_browse: true,
        shell: true,
        sandbox_prepare: true,
        sandbox_stop: false,
      },
      checks: [
        {
          id: 'docker_daemon',
          status: 'ready',
          message: 'Docker daemon is available',
          details: { socket: 'available' },
        },
      ],
      docker: null,
      updated_at: '2026-07-11T15:00:00Z',
    },
    isLoading: false,
    isError: false,
    isFetching: false,
    error: null,
    refetch: runtimeRefetch,
  } as unknown as ReturnType<typeof hooks.useWorkspaceRuntimeQuery>)
}

describe('SettingsPage runtime and connection diagnostics', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    useBlocker.mockReturnValue({
      status: 'idle',
      proceed: vi.fn(),
      reset: vi.fn(),
    })
    setupHealthyRuntime()
    useConnectionStore.setState({
      baseUrl: 'https://claw.example.test',
      apiToken: 'session-secret',
      connectionIssue: null,
      connectionScope: 'test-scope',
    })
  })

  it('consolidates server identity, health, workspace capabilities and checks', () => {
    render(<SettingsPage />)

    expect(screen.getByText('Backend reachable')).toBeVisible()
    expect(screen.getByText('2.4.0')).toBeVisible()
    expect(screen.getByText('2.4.1')).toBeVisible()
    expect(screen.getByText('Docker Daemon')).toBeVisible()
    expect(screen.getByText('Docker daemon is available')).toBeVisible()
    expect(screen.getByText('File Browse')).toBeVisible()
    expect(screen.getByText(/token stays in browser memory/i)).toBeVisible()
    expect(screen.queryByText('session-secret')).not.toBeInTheDocument()
  })

  it('validates connection drafts before saving them', async () => {
    const user = userEvent.setup()
    render(<SettingsPage />)

    await user.clear(screen.getByLabelText('Backend URL'))
    await user.click(screen.getByRole('button', { name: 'Test and save' }))

    expect(screen.getByRole('alert')).toHaveTextContent(
      'Backend URL is required',
    )
    expect(connection.validateConnection).not.toHaveBeenCalled()
  })

  it('protects unsaved connection drafts and confirms before logout', async () => {
    const user = userEvent.setup()
    render(<SettingsPage />)

    await user.type(screen.getByLabelText('Backend URL'), '/changed')

    expect(useBlocker).toHaveBeenCalledWith(
      expect.objectContaining({
        disabled: false,
        enableBeforeUnload: true,
        withResolver: true,
      }),
    )
    await user.click(screen.getByRole('button', { name: 'Logout' }))
    expect(
      screen.getByRole('dialog', {
        name: 'Discard connection changes and disconnect?',
      }),
    ).toBeVisible()
    expect(useConnectionStore.getState().apiToken).toBe('session-secret')

    await user.click(
      screen.getByRole('button', { name: 'Discard and disconnect' }),
    )
    expect(useConnectionStore.getState().apiToken).toBe('')
  })

  it('does not restore credentials when deferred validation finishes after logout', async () => {
    const user = userEvent.setup()
    let resolveValidation: (() => void) | undefined
    const validation = new Promise<
      Awaited<ReturnType<typeof connection.validateConnection>>
    >((resolve) => {
      resolveValidation = () =>
        resolve({} as Awaited<ReturnType<typeof connection.validateConnection>>)
    })
    vi.mocked(connection.validateConnection).mockReturnValue(validation)
    render(<SettingsPage />)

    const tokenInput = screen.getByLabelText('API Token')
    await user.clear(tokenInput)
    await user.type(tokenInput, 'pending-secret')
    await user.click(screen.getByRole('button', { name: 'Test and save' }))

    expect(connection.validateConnection).toHaveBeenCalledWith({
      baseUrl: 'https://claw.example.test',
      apiToken: 'pending-secret',
    })
    expect(screen.getByLabelText('Backend URL')).toBeDisabled()
    expect(tokenInput).toBeDisabled()
    await user.click(screen.getByRole('button', { name: 'Logout' }))
    await user.click(
      screen.getByRole('button', { name: 'Discard and disconnect' }),
    )

    expect(useConnectionStore.getState().apiToken).toBe('')
    expect(tokenInput).toHaveValue('')

    await act(async () => {
      resolveValidation?.()
      await validation
    })

    expect(useConnectionStore.getState().apiToken).toBe('')
    expect(localStorage.getItem('ya-claw-connection')).not.toContain(
      'pending-secret',
    )
  })

  it('shows technical query errors with a retry action', async () => {
    const user = userEvent.setup()
    vi.mocked(hooks.useWorkspaceRuntimeQuery).mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true,
      isFetching: false,
      error: new Error('Workspace probe failed'),
      refetch: runtimeRefetch,
    } as unknown as ReturnType<typeof hooks.useWorkspaceRuntimeQuery>)

    render(<SettingsPage />)

    expect(screen.getByText('Workspace probe failed')).toBeVisible()
    expect(screen.getAllByText('Technical details')).not.toHaveLength(0)
    await user.click(screen.getByRole('button', { name: 'Try again' }))
    expect(runtimeRefetch).toHaveBeenCalledOnce()
  })
})
