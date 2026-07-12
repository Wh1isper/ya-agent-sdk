import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import type { ReactNode } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import * as hooks from '../../api/hooks'
import type { WorkspaceFileEntry, WorkspaceFileListResponse } from './types'
import { WorkspacePage } from './WorkspacePage'

vi.mock('@tanstack/react-router', () => ({
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
  useApiClient: vi.fn(),
  useSessionQuery: vi.fn(),
  useSessionsQuery: vi.fn(),
  useSessionSandboxMutations: vi.fn(),
  useSessionWorkspaceQuery: vi.fn(),
  useWorkspaceFileQuery: vi.fn(),
  useWorkspaceFilesQuery: vi.fn(),
  useWorkspaceRuntimeQuery: vi.fn(),
}))

const rootEntries: WorkspaceFileEntry[] = [
  fileEntry('memory', 'directory'),
  fileEntry('outputs', 'directory'),
  fileEntry('report.md'),
  fileEntry('data.csv'),
]
const memoryEntries: WorkspaceFileEntry[] = [
  fileEntry('MEMORY.md', 'file', '/workspace/memory/MEMORY.md'),
  fileEntry('CHANGELOG.md', 'file', '/workspace/memory/CHANGELOG.md'),
  fileEntry('20260711-event.md', 'file', '/workspace/memory/20260711-event.md'),
  fileEntry('scratch.md', 'file', '/workspace/memory/scratch.md'),
]
const stopSandbox = vi.fn(async () => undefined)
const prepareSandbox = vi.fn(async () => undefined)

function fileEntry(
  name: string,
  kind: WorkspaceFileEntry['kind'] = 'file',
  path = `/workspace/${name}`,
): WorkspaceFileEntry {
  return {
    name,
    path,
    kind,
    size_bytes: kind === 'file' ? 128 : null,
    modified_at: '2026-07-11T00:00:00Z',
    hidden: false,
  }
}

function fileList(
  path: string,
  items: WorkspaceFileEntry[],
): WorkspaceFileListResponse {
  return {
    session_id: 'session-1',
    path,
    items,
    limit: 500,
    offset: 0,
    has_more: false,
    next_cursor: null,
    next_offset: null,
    truncated: false,
  }
}

function setupHooks(options: { sandboxReady?: boolean } = {}) {
  const sandboxReady = options.sandboxReady ?? true
  vi.mocked(hooks.useWorkspaceRuntimeQuery).mockReturnValue({
    data: {
      backend: 'docker',
      status: 'ready',
      execution_location: 'docker',
      workspace: { exists: true, writable: true, virtual_path: '/workspace' },
      capabilities: {
        file_browse: true,
        shell: true,
        sandbox_prepare: true,
        sandbox_stop: true,
      },
      checks: [],
      updated_at: '2026-07-11T00:00:00Z',
    },
    isLoading: false,
    isError: false,
    refetch: vi.fn(),
  } as unknown as ReturnType<typeof hooks.useWorkspaceRuntimeQuery>)
  vi.mocked(hooks.useSessionQuery).mockReturnValue({
    data: {
      session: {
        id: 'session-1',
        session_type: 'conversation',
        metadata: {},
        created_at: '2026-07-11T00:00:00Z',
        updated_at: '2026-07-11T00:00:00Z',
        status: 'idle',
        run_count: 1,
        head_run_id: 'run-1',
        runs: [
          {
            id: 'run-1',
            session_id: 'session-1',
            sequence_no: 1,
            status: 'completed',
            trigger_type: 'api',
            created_at: '2026-07-11T00:00:00Z',
            message: [{ type: 'RUN_FINISHED', timestamp: 1 }],
          },
        ],
      },
      state: null,
      message: [],
    },
    isLoading: false,
    isError: false,
    error: null,
    refetch: vi.fn(),
  } as unknown as ReturnType<typeof hooks.useSessionQuery>)
  vi.mocked(hooks.useSessionsQuery).mockReturnValue({
    data: [
      {
        id: 'session-1',
        session_type: 'conversation',
        metadata: {},
        created_at: '2026-07-11T00:00:00Z',
        updated_at: '2026-07-11T00:00:00Z',
        status: 'idle',
        run_count: 1,
        head_run_id: 'run-1',
        latest_run: { id: 'run-1', input_preview: 'Research request' },
      },
    ],
    isLoading: false,
    isError: false,
    refetch: vi.fn(),
  } as unknown as ReturnType<typeof hooks.useSessionsQuery>)
  vi.mocked(hooks.useWorkspaceFilesQuery).mockImplementation(
    (_sessionId, path) => {
      const memory = path === '/workspace/memory'
      return {
        data: memory
          ? fileList('/workspace/memory', memoryEntries)
          : fileList('/workspace', rootEntries),
        isLoading: false,
        isError: false,
        isFetchingNextPage: false,
        hasNextPage: false,
        error: null,
        fetchNextPage: vi.fn(),
        refetch: vi.fn(),
      } as unknown as ReturnType<typeof hooks.useWorkspaceFilesQuery>
    },
  )
  vi.mocked(hooks.useWorkspaceFileQuery).mockImplementation(
    (_sessionId, path) =>
      ({
        data: path
          ? {
              session_id: 'session-1',
              path,
              content: path.endsWith('MEMORY.md')
                ? '# Durable memory'
                : 'file body',
              encoding: 'utf-8',
              size_bytes: 16,
            }
          : undefined,
        isLoading: false,
        isError: false,
        error: null,
        refetch: vi.fn(),
      }) as unknown as ReturnType<typeof hooks.useWorkspaceFileQuery>,
  )
  vi.mocked(hooks.useSessionWorkspaceQuery).mockReturnValue({
    data: {
      binding: { cwd: '/workspace' },
      sandbox_state: sandboxReady
        ? {
            status: 'ready',
            ready_state: 'ready',
            container_id: 'container-1',
            backend: 'docker',
            updated_at: '2026-07-11T00:00:00Z',
          }
        : {
            status: 'stopped',
            ready_state: 'not_started',
            backend: 'docker',
            updated_at: '2026-07-11T00:00:00Z',
          },
    },
    isError: false,
  } as unknown as ReturnType<typeof hooks.useSessionWorkspaceQuery>)
  vi.mocked(hooks.useSessionSandboxMutations).mockReturnValue({
    prepare: { mutateAsync: prepareSandbox, isPending: false },
    stop: { mutateAsync: stopSandbox, isPending: false },
  } as unknown as ReturnType<typeof hooks.useSessionSandboxMutations>)
  vi.mocked(hooks.useApiClient).mockReturnValue({
    downloadWorkspaceFile: vi.fn(),
  } as unknown as ReturnType<typeof hooks.useApiClient>)
}

describe('WorkspacePage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    setupHooks()
  })

  it('provides read-only Files, Memory, and Artifacts sections backed by workspace data', async () => {
    const user = userEvent.setup()
    render(<WorkspacePage />)

    expect(screen.getByText('Read-only')).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: 'Files' })).toHaveAttribute(
      'aria-selected',
      'true',
    )

    await user.click(screen.getByRole('tab', { name: 'Memory' }))
    expect(await screen.findByText('memory/MEMORY.md')).toBeInTheDocument()
    expect(screen.getByText('memory/CHANGELOG.md')).toBeInTheDocument()
    expect(screen.getByText('Event files index')).toBeInTheDocument()
    expect(await screen.findByText('20260711-event.md')).toBeInTheDocument()
    expect(screen.queryByText('scratch.md')).not.toBeInTheDocument()

    await user.click(screen.getByText('memory/MEMORY.md'))
    expect(await screen.findByText('# Durable memory')).toBeInTheDocument()

    await user.click(screen.getByRole('tab', { name: 'Artifacts' }))
    expect(screen.getByText('Generated output folders')).toBeInTheDocument()
    expect(screen.getByText('Reports & documents')).toBeInTheDocument()
    expect(screen.getByText('Data & exports')).toBeInTheDocument()
    expect(
      screen.getByText(/does not report file-to-run provenance/i),
    ).toBeInTheDocument()
    expect(
      screen.getByRole('link', { name: /open related activity/i }),
    ).toHaveAttribute('href', '/activity/sessions/session-1/runs/run-1')
    expect(screen.getByRole('link', { name: /run 1/i })).toHaveAttribute(
      'href',
      '/activity/sessions/session-1/runs/run-1',
    )
  })

  it('provides linked tabpanels and keyboard roving between workspace tabs', async () => {
    const user = userEvent.setup()
    render(<WorkspacePage />)

    const filesTab = screen.getByRole('tab', { name: 'Files' })
    const memoryTab = screen.getByRole('tab', { name: 'Memory' })
    expect(filesTab).toHaveAttribute('aria-selected', 'true')
    expect(filesTab).toHaveAttribute('aria-controls')

    await user.click(filesTab)
    await user.keyboard('{ArrowRight}')

    expect(memoryTab).toHaveFocus()
    expect(memoryTab).toHaveAttribute('aria-selected', 'true')
    expect(filesTab).toHaveAttribute('tabindex', '-1')
    const panel = screen.getByRole('tabpanel')
    expect(memoryTab).toHaveAttribute('aria-controls', panel.id)
    expect(panel).toHaveAttribute('aria-labelledby', memoryTab.id)
  })

  it('offers another stable directory page instead of ignoring truncation', async () => {
    const fetchNextPage = vi.fn(async () => undefined)
    vi.mocked(hooks.useWorkspaceFilesQuery).mockImplementation(
      (_sessionId, path) =>
        ({
          data:
            path === '/workspace/memory'
              ? fileList('/workspace/memory', memoryEntries)
              : {
                  ...fileList('/workspace', rootEntries),
                  has_more: true,
                  next_offset: rootEntries.length,
                  truncated: true,
                },
          isLoading: false,
          isError: false,
          isFetchingNextPage: false,
          hasNextPage: path !== '/workspace/memory',
          error: null,
          fetchNextPage,
          refetch: vi.fn(),
        }) as unknown as ReturnType<typeof hooks.useWorkspaceFilesQuery>,
    )
    const user = userEvent.setup()
    render(<WorkspacePage />)

    await user.click(await screen.findByRole('button', { name: 'Load more' }))
    expect(fetchNextPage).toHaveBeenCalledOnce()
  })

  it('requests complete automatic pagination for memory and artifact indexes', async () => {
    const user = userEvent.setup()
    render(<WorkspacePage />)

    await user.click(screen.getByRole('tab', { name: 'Memory' }))
    await waitFor(() =>
      expect(hooks.useWorkspaceFilesQuery).toHaveBeenCalledWith(
        'session-1',
        '/workspace/memory',
        { autoLoadAll: true },
      ),
    )

    await user.click(screen.getByRole('tab', { name: 'Artifacts' }))
    await waitFor(() =>
      expect(hooks.useWorkspaceFilesQuery).toHaveBeenCalledWith(
        'session-1',
        null,
        { autoLoadAll: true },
      ),
    )
  })

  it('uses the existing sandbox lifecycle controls', async () => {
    const user = userEvent.setup()
    render(<WorkspacePage />)

    await waitFor(() =>
      expect(screen.getAllByText('ready').length).toBeGreaterThan(0),
    )
    await user.click(screen.getByRole('button', { name: 'Stop' }))
    expect(stopSandbox).toHaveBeenCalledOnce()
  })

  it('offers prepare when the selected sandbox is stopped', async () => {
    setupHooks({ sandboxReady: false })
    const user = userEvent.setup()
    render(<WorkspacePage />)

    await user.click(await screen.findByRole('button', { name: 'Prepare' }))
    expect(prepareSandbox).toHaveBeenCalledOnce()
  })
})
