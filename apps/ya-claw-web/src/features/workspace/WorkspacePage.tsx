import { Link } from '@tanstack/react-router'
import {
  Archive,
  ChevronLeft,
  Database,
  Download,
  ExternalLink,
  File,
  FileClock,
  Folder,
  FolderTree,
  HardDrive,
  LockKeyhole,
  Play,
  RefreshCcw,
  Square,
} from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { toast } from 'sonner'

import { ApiError } from '../../api/client'
import {
  useApiClient,
  useSessionQuery,
  useSessionsQuery,
  useSessionSandboxMutations,
  useSessionWorkspaceQuery,
  useWorkspaceFileQuery,
  useWorkspaceFilesQuery,
  useWorkspaceRuntimeQuery,
} from '../../api/hooks'
import { EmptyState } from '../../components/EmptyState'
import { StatusBadge } from '../../components/StatusBadge'
import { QueryError, QuerySkeleton } from '../../components/ui/QueryState'
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from '../../components/ui/Tabs'
import { buildChatPath } from '../../lib/urlState'
import { cn } from '../../lib/utils'
import type { RunSummary } from '../../types'
import type { WorkspaceFileEntry } from './types'
import {
  groupWorkspaceArtifacts,
  isMemoryEventFile,
  joinVirtualPath,
  type WorkspaceSection,
} from './workspaceView'

const sections: Array<{
  id: WorkspaceSection
  label: string
  icon: typeof FolderTree
}> = [
  { id: 'files', label: 'Files', icon: FolderTree },
  { id: 'memory', label: 'Memory', icon: Database },
  { id: 'artifacts', label: 'Artifacts', icon: Archive },
]

export function WorkspacePage() {
  const runtime = useWorkspaceRuntimeQuery()
  const sessions = useSessionsQuery()
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [section, setSection] = useState<WorkspaceSection>('files')
  const [path, setPath] = useState<string | null>(null)
  const [rootPath, setRootPath] = useState<string | null>(null)
  const [selectedFile, setSelectedFile] = useState<string | null>(null)
  const files = useWorkspaceFilesQuery(sessionId, path, {
    autoLoadAll: section === 'artifacts',
  })
  const memoryPath = rootPath ? joinVirtualPath(rootPath, 'memory') : null
  const memoryFiles = useWorkspaceFilesQuery(
    section === 'memory' && memoryPath ? sessionId : null,
    memoryPath,
    { autoLoadAll: section === 'memory' },
  )
  const file = useWorkspaceFileQuery(sessionId, selectedFile)
  const artifactSession = useSessionQuery(
    section === 'artifacts' ? sessionId : null,
  )
  const sessionWorkspace = useSessionWorkspaceQuery(sessionId)
  const sandboxControls = useSessionSandboxMutations(sessionId)
  const api = useApiClient()

  const selectedSession = useMemo(
    () => (sessions.data ?? []).find((session) => session.id === sessionId),
    [sessionId, sessions.data],
  )
  const sandbox =
    sessionWorkspace.data?.sandbox_state ??
    selectedSession?.workspace_state?.sandbox_state ??
    null
  const memoryEntries = memoryFiles.data?.items ?? []
  const memoryIndex = memoryEntries.find(
    (entry) => entry.kind === 'file' && entry.name === 'MEMORY.md',
  )
  const memoryChangelog = memoryEntries.find(
    (entry) => entry.kind === 'file' && entry.name === 'CHANGELOG.md',
  )
  const memoryEvents = memoryEntries.filter(isMemoryEventFile)
  const artifactGroups = useMemo(
    () => groupWorkspaceArtifacts(files.data?.items ?? []),
    [files.data?.items],
  )
  const relatedRunId =
    selectedSession?.active_run_id ??
    selectedSession?.head_run_id ??
    selectedSession?.latest_run?.id ??
    null
  const activityHref = sessionId
    ? buildChatPath(sessionId, relatedRunId, 'debug')
    : '/activity'

  useEffect(() => {
    if (!sessionId && sessions.data?.[0]) setSessionId(sessions.data[0].id)
  }, [sessionId, sessions.data])

  useEffect(() => {
    setPath(null)
    setRootPath(null)
    setSelectedFile(null)
  }, [sessionId])

  useEffect(() => {
    if (
      !path &&
      files.data?.path &&
      files.data.session_id === sessionId &&
      rootPath !== files.data.path
    ) {
      setRootPath(files.data.path)
    }
  }, [files.data?.path, files.data?.session_id, path, rootPath, sessionId])

  async function downloadSelected() {
    if (!sessionId || !selectedFile) return
    try {
      const blob = await api.downloadWorkspaceFile(sessionId, selectedFile)
      const url = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = selectedFile.split('/').pop() ?? 'download'
      anchor.click()
      URL.revokeObjectURL(url)
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Download failed')
    }
  }

  async function runSandboxAction(action: 'prepare' | 'stop') {
    try {
      if (action === 'prepare') await sandboxControls.prepare.mutateAsync()
      else await sandboxControls.stop.mutateAsync()
      toast.success(
        action === 'prepare' ? 'Sandbox prepared' : 'Sandbox stopped',
      )
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : `Could not ${action} sandbox`,
      )
    }
  }

  if (runtime.isError || sessions.isError) {
    return (
      <div className="p-4 sm:p-6 lg:p-8">
        <QueryError
          title="Workspace is unavailable"
          error={runtime.error ?? sessions.error}
          onRetry={() => {
            void Promise.all([runtime.refetch(), sessions.refetch()])
          }}
        />
      </div>
    )
  }

  if (
    (runtime.isLoading && runtime.data === undefined) ||
    (sessions.isLoading && sessions.data === undefined)
  ) {
    return (
      <div className="p-4 sm:p-6 lg:p-8">
        <QuerySkeleton rows={5} />
      </div>
    )
  }

  if (sessionId && sessionWorkspace.isError) {
    return (
      <div className="p-4 sm:p-6 lg:p-8">
        <QueryError
          title="Could not resolve this conversation workspace"
          error={sessionWorkspace.error}
          onRetry={() => void sessionWorkspace.refetch()}
        />
      </div>
    )
  }

  const canPrepare = Boolean(
    sessionId &&
    runtime.data?.capabilities.sandbox_prepare &&
    sandbox?.ready_state !== 'ready' &&
    sandbox?.ready_state !== 'starting',
  )
  const canStop = Boolean(
    sessionId &&
    runtime.data?.capabilities.sandbox_stop &&
    sandbox?.status !== 'stopped' &&
    (sandbox?.container_id || sandbox?.container_ref),
  )

  return (
    <div className="flex min-h-full flex-col gap-5 p-4 sm:p-6 lg:p-8">
      <section className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-sm font-medium text-[var(--primary)]">Workspace</p>
          <div className="mt-1 flex flex-wrap items-center gap-3">
            <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
              Files, memory, and artifacts
            </h1>
            {runtime.data ? <StatusBadge status={runtime.data.status} /> : null}
            <span className="inline-flex items-center gap-1 rounded-full bg-[var(--subtle)] px-2.5 py-1 text-xs font-medium text-[var(--muted-foreground)]">
              <LockKeyhole className="h-3.5 w-3.5" />
              Read-only
            </span>
          </div>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-[var(--muted-foreground)]">
            Inspect workspace files without editing them. Sandbox lifecycle
            controls affect execution only; this browser remains read-only.
          </p>
        </div>
        <div className="min-w-56">
          <label className="text-sm font-medium">
            Conversation
            <select
              className="mt-1.5 h-10 w-full rounded-lg border border-[var(--border)] bg-[var(--surface)] px-3 text-sm outline-none focus:ring-2 focus:ring-[var(--focus)]"
              value={sessionId ?? ''}
              onChange={(event) => setSessionId(event.target.value || null)}
            >
              {(sessions.data ?? []).map((session) => (
                <option key={session.id} value={session.id}>
                  {session.latest_run?.input_preview || session.id.slice(0, 12)}
                </option>
              ))}
            </select>
          </label>
          {sessions.hasNextPage ? (
            <button
              type="button"
              className="mt-2 text-xs font-semibold text-[var(--primary)] disabled:opacity-60"
              disabled={sessions.isFetchingNextPage}
              onClick={() => void sessions.fetchNextPage()}
            >
              {sessions.isFetchingNextPage
                ? 'Loading conversations…'
                : 'Load older conversations'}
            </button>
          ) : null}
        </div>
      </section>

      {!sessions.isLoading && (sessions.data ?? []).length === 0 ? (
        <EmptyState
          icon={FolderTree}
          title="No conversation workspace yet"
          headingLevel={2}
          description="Start a conversation first. Its files, memory, and artifacts will appear here."
        />
      ) : (
        <>
          <SandboxPanel
            backend={runtime.data?.backend ?? sandbox?.backend ?? 'unknown'}
            cwd={
              sessionWorkspace.data?.binding?.cwd ??
              sandbox?.work_dir ??
              rootPath ??
              'Resolving workspace…'
            }
            status={sandbox?.status ?? 'not prepared'}
            error={sandbox?.error_message ?? null}
            isWorkspaceError={sessionWorkspace.isError}
            canPrepare={canPrepare}
            canStop={canStop}
            preparing={sandboxControls.prepare.isPending}
            stopping={sandboxControls.stop.isPending}
            onPrepare={() => void runSandboxAction('prepare')}
            onStop={() => void runSandboxAction('stop')}
          />

          <Tabs
            value={section}
            orientation="horizontal"
            onValueChange={(value) => {
              const nextSection = value as WorkspaceSection
              setSection(nextSection)
              setSelectedFile(null)
              if (nextSection !== 'files') setPath(null)
            }}
          >
            <TabsList
              className="flex h-auto w-full gap-1 overflow-x-auto rounded-xl border border-[var(--border)] bg-[var(--surface)] p-1"
              aria-label="Workspace sections"
            >
              {sections.map((item) => {
                const Icon = item.icon
                return (
                  <TabsTrigger
                    key={item.id}
                    value={item.id}
                    className="h-auto min-w-28 flex-1 gap-2 rounded-lg px-3 py-2 data-[state=active]:bg-[var(--primary)] data-[state=active]:text-white"
                  >
                    <Icon className="h-4 w-4" />
                    {item.label}
                  </TabsTrigger>
                )
              })}
            </TabsList>

            <TabsContent value="files" className="mt-5">
              <section className="grid min-h-[36rem] flex-1 overflow-hidden rounded-xl border border-[var(--border)] bg-[var(--surface)] lg:grid-cols-[22rem_minmax(0,1fr)]">
                <FilesIndex
                  files={files}
                  path={path}
                  rootPath={rootPath}
                  selectedFile={selectedFile}
                  onPathChange={setPath}
                  onFileSelect={setSelectedFile}
                />
                <FilePreview
                  selectedFile={selectedFile}
                  file={file}
                  onDownload={() => void downloadSelected()}
                />
              </section>
            </TabsContent>

            <TabsContent value="memory" className="mt-5">
              <section className="grid min-h-[36rem] flex-1 overflow-hidden rounded-xl border border-[var(--border)] bg-[var(--surface)] lg:grid-cols-[22rem_minmax(0,1fr)]">
                <MemoryIndex
                  loading={memoryFiles.isLoading}
                  loadingMore={memoryFiles.isFetchingNextPage}
                  hasMore={memoryFiles.hasNextPage}
                  error={memoryFiles.error}
                  index={memoryIndex}
                  changelog={memoryChangelog}
                  events={memoryEvents}
                  selectedFile={selectedFile}
                  onFileSelect={setSelectedFile}
                  onLoadMore={() => void memoryFiles.fetchNextPage()}
                  onRetry={() =>
                    void (memoryFiles.isFetchNextPageError
                      ? memoryFiles.fetchNextPage()
                      : memoryFiles.refetch())
                  }
                />
                <FilePreview
                  selectedFile={selectedFile}
                  file={file}
                  onDownload={() => void downloadSelected()}
                />
              </section>
            </TabsContent>

            <TabsContent value="artifacts" className="mt-5">
              <section className="grid min-h-[36rem] flex-1 overflow-hidden rounded-xl border border-[var(--border)] bg-[var(--surface)] lg:grid-cols-[22rem_minmax(0,1fr)]">
                <ArtifactsIndex
                  loading={files.isLoading}
                  loadingMore={files.isFetchingNextPage}
                  hasMore={files.hasNextPage}
                  error={files.error}
                  groups={artifactGroups}
                  runs={artifactSession.data?.session.runs ?? []}
                  runsLoading={artifactSession.isLoading}
                  runsError={artifactSession.error}
                  selectedFile={selectedFile}
                  activityHref={activityHref}
                  onFileSelect={setSelectedFile}
                  onLoadMore={() => void files.fetchNextPage()}
                  onOpenDirectory={(entry) => {
                    setSection('files')
                    setPath(entry.path)
                    setSelectedFile(null)
                  }}
                  onRetry={() =>
                    void Promise.all([
                      files.isFetchNextPageError
                        ? files.fetchNextPage()
                        : files.refetch(),
                      artifactSession.refetch(),
                    ])
                  }
                />
                <FilePreview
                  selectedFile={selectedFile}
                  file={file}
                  onDownload={() => void downloadSelected()}
                />
              </section>
            </TabsContent>
          </Tabs>
        </>
      )}
    </div>
  )
}

function SandboxPanel({
  backend,
  cwd,
  status,
  error,
  isWorkspaceError,
  canPrepare,
  canStop,
  preparing,
  stopping,
  onPrepare,
  onStop,
}: {
  backend: string
  cwd: string
  status: string
  error: string | null
  isWorkspaceError: boolean
  canPrepare: boolean
  canStop: boolean
  preparing: boolean
  stopping: boolean
  onPrepare: () => void
  onStop: () => void
}) {
  return (
    <section className="flex flex-col gap-3 rounded-xl border border-[var(--border)] bg-[var(--surface)] p-4 sm:flex-row sm:items-center sm:justify-between">
      <div className="min-w-0">
        <div className="flex flex-wrap items-center gap-2">
          <HardDrive className="h-4 w-4 text-[var(--primary)]" />
          <h2 className="text-sm font-semibold">Sandbox</h2>
          <StatusBadge status={status} />
          <span className="rounded-full bg-[var(--subtle)] px-2 py-0.5 text-xs text-[var(--muted-foreground)]">
            {backend}
          </span>
        </div>
        <p className="mono mt-1 truncate text-xs text-[var(--muted-foreground)]">
          {cwd}
        </p>
        {error || isWorkspaceError ? (
          <p className="mt-1 text-xs text-rose-700" role="alert">
            {error ?? 'Could not refresh this sandbox state.'}
          </p>
        ) : null}
      </div>
      <div className="flex shrink-0 flex-wrap gap-2">
        {canPrepare ? (
          <button
            type="button"
            className="inline-flex h-9 items-center gap-2 rounded-lg bg-[var(--primary)] px-3 text-sm font-medium text-white disabled:opacity-60"
            onClick={onPrepare}
            disabled={preparing || stopping}
          >
            <Play className="h-4 w-4" />
            {preparing ? 'Preparing…' : 'Prepare'}
          </button>
        ) : null}
        {canStop ? (
          <button
            type="button"
            className="inline-flex h-9 items-center gap-2 rounded-lg border border-[var(--border)] px-3 text-sm font-medium disabled:opacity-60"
            onClick={onStop}
            disabled={preparing || stopping}
          >
            <Square className="h-3.5 w-3.5" />
            {stopping ? 'Stopping…' : 'Stop'}
          </button>
        ) : null}
      </div>
    </section>
  )
}

function FilesIndex({
  files,
  path,
  rootPath,
  selectedFile,
  onPathChange,
  onFileSelect,
}: {
  files: ReturnType<typeof useWorkspaceFilesQuery>
  path: string | null
  rootPath: string | null
  selectedFile: string | null
  onPathChange: (path: string | null) => void
  onFileSelect: (path: string | null) => void
}) {
  return (
    <aside className="flex min-h-72 flex-col border-b border-[var(--border)] lg:border-b-0 lg:border-r">
      <div className="flex items-center gap-2 border-b border-[var(--border)] p-3">
        <button
          type="button"
          className="inline-flex h-9 w-9 items-center justify-center rounded-lg border border-[var(--border)] disabled:opacity-40"
          onClick={() => {
            onPathChange(parentPath(path, rootPath))
            onFileSelect(null)
          }}
          disabled={!path || path === rootPath}
          aria-label="Open parent directory"
        >
          <ChevronLeft className="h-4 w-4" />
        </button>
        <p className="mono min-w-0 flex-1 truncate text-xs text-[var(--muted-foreground)]">
          {files.data?.path ?? path ?? 'Workspace root'}
        </p>
        <button
          type="button"
          className="inline-flex h-9 w-9 items-center justify-center rounded-lg border border-[var(--border)]"
          onClick={() => void files.refetch()}
          aria-label="Refresh files"
        >
          <RefreshCcw className="h-4 w-4" />
        </button>
      </div>
      <EntryListState
        loading={files.isLoading}
        error={files.error}
        empty={!files.isLoading && files.data?.items.length === 0}
        onRetry={() => void files.refetch()}
      >
        {(files.data?.items ?? []).map((entry) => (
          <WorkspaceEntryButton
            key={entry.path}
            entry={entry}
            selected={selectedFile === entry.path}
            onClick={() => {
              if (entry.kind === 'directory') {
                onPathChange(entry.path)
                onFileSelect(null)
              } else {
                onFileSelect(entry.path)
              }
            }}
          />
        ))}
        <LoadMoreEntries
          hasMore={files.hasNextPage}
          loading={files.isFetchingNextPage}
          onLoadMore={() => void files.fetchNextPage()}
        />
      </EntryListState>
    </aside>
  )
}

function MemoryIndex({
  loading,
  loadingMore,
  hasMore,
  error,
  index,
  changelog,
  events,
  selectedFile,
  onFileSelect,
  onLoadMore,
  onRetry,
}: {
  loading: boolean
  loadingMore: boolean
  hasMore: boolean
  error: Error | null
  index?: WorkspaceFileEntry
  changelog?: WorkspaceFileEntry
  events: WorkspaceFileEntry[]
  selectedFile: string | null
  onFileSelect: (path: string) => void
  onLoadMore: () => void
  onRetry: () => void
}) {
  return (
    <aside className="flex min-h-72 flex-col border-b border-[var(--border)] lg:border-b-0 lg:border-r">
      <div className="border-b border-[var(--border)] p-4">
        <h2 className="font-semibold">Workspace memory</h2>
        <p className="mt-1 text-xs leading-5 text-[var(--muted-foreground)]">
          The durable brief, its change history, and protocol-named event notes.
          Contents are shown exactly as stored.
        </p>
      </div>
      <div className="scrollbar-thin min-h-0 flex-1 overflow-auto p-3">
        {loading ? (
          <p className="p-3 text-sm text-[var(--muted-foreground)]">
            Loading memory…
          </p>
        ) : null}
        {error ? (
          <QueryError
            compact
            title="Could not load workspace memory"
            error={error}
            onRetry={onRetry}
          />
        ) : null}
        {!error && !loading ? (
          <>
            <MemoryFileCard
              title="memory/MEMORY.md"
              description="Compact durable brief loaded as workspace memory context."
              entry={index}
              selected={selectedFile === index?.path}
              onSelect={onFileSelect}
            />
            <MemoryFileCard
              title="memory/CHANGELOG.md"
              description="History of updates to workspace memory files."
              entry={changelog}
              selected={selectedFile === changelog?.path}
              onSelect={onFileSelect}
            />
            <div className="mt-4 flex items-center justify-between gap-2 px-1">
              <h3 className="text-xs font-semibold uppercase tracking-wide text-[var(--muted-foreground)]">
                Event files index
              </h3>
              <span className="text-xs text-[var(--subtle-foreground)]">
                {events.length}
              </span>
            </div>
            <p className="px-1 py-2 text-xs leading-5 text-[var(--muted-foreground)]">
              Files matching memory/YYYYMMDD-event.md. This is a filename index,
              not inferred run provenance.
            </p>
            {events.map((entry) => (
              <WorkspaceEntryButton
                key={entry.path}
                entry={entry}
                selected={selectedFile === entry.path}
                icon={FileClock}
                onClick={() => onFileSelect(entry.path)}
              />
            ))}
            {events.length === 0 && !hasMore && !loadingMore ? (
              <p className="rounded-lg border border-dashed border-[var(--border)] p-3 text-center text-xs text-[var(--muted-foreground)]">
                No protocol-named event files yet.
              </p>
            ) : null}
            <LoadMoreEntries
              hasMore={hasMore}
              loading={loadingMore}
              onLoadMore={onLoadMore}
              automatic
            />
          </>
        ) : null}
      </div>
    </aside>
  )
}

function MemoryFileCard({
  title,
  description,
  entry,
  selected,
  onSelect,
}: {
  title: string
  description: string
  entry?: WorkspaceFileEntry
  selected: boolean
  onSelect: (path: string) => void
}) {
  return (
    <button
      type="button"
      className={cn(
        'mb-2 w-full rounded-lg border p-3 text-left transition',
        selected
          ? 'border-[var(--primary)] bg-[var(--primary-subtle)]'
          : 'border-[var(--border)] hover:bg-[var(--subtle)]',
        !entry && 'cursor-not-allowed opacity-60',
      )}
      disabled={!entry}
      onClick={() => entry && onSelect(entry.path)}
    >
      <span className="mono block text-xs font-semibold">{title}</span>
      <span className="mt-1 block text-xs leading-5 text-[var(--muted-foreground)]">
        {entry ? description : `${description} Not present.`}
      </span>
    </button>
  )
}

function ArtifactsIndex({
  loading,
  loadingMore,
  hasMore,
  error,
  groups,
  runs,
  runsLoading,
  runsError,
  selectedFile,
  activityHref,
  onFileSelect,
  onLoadMore,
  onOpenDirectory,
  onRetry,
}: {
  loading: boolean
  loadingMore: boolean
  hasMore: boolean
  error: Error | null
  groups: ReturnType<typeof groupWorkspaceArtifacts>
  runs: RunSummary[]
  runsLoading: boolean
  runsError: Error | null
  selectedFile: string | null
  activityHref: string
  onFileSelect: (path: string) => void
  onLoadMore: () => void
  onOpenDirectory: (entry: WorkspaceFileEntry) => void
  onRetry: () => void
}) {
  return (
    <aside className="flex min-h-72 flex-col border-b border-[var(--border)] lg:border-b-0 lg:border-r">
      <div className="border-b border-[var(--border)] p-4">
        <h2 className="font-semibold">Artifact index</h2>
        <p className="mt-1 text-xs leading-5 text-[var(--muted-foreground)]">
          Root entries grouped by conventional output folder names, then file
          extension. Memory is excluded; unknown files remain visible under
          Other.
        </p>
        <div className="mt-3 rounded-lg bg-[var(--subtle)] p-3">
          <p className="text-xs font-semibold">Related activity</p>
          <p className="mt-1 text-xs leading-5 text-[var(--muted-foreground)]">
            Workspace output groups are based on names and extensions because
            the files API does not report file-to-run provenance. Committed run
            records below provide truthful Activity links without claiming a
            particular run created a file.
          </p>
          <Link
            to={activityHref}
            className="mt-2 inline-flex items-center gap-1 text-xs font-semibold text-[var(--primary)] hover:underline"
          >
            Open related activity <ExternalLink className="h-3.5 w-3.5" />
          </Link>
        </div>
        <div className="mt-3">
          <p className="text-xs font-semibold">Run artifacts</p>
          {runsLoading ? (
            <p className="mt-2 text-xs text-[var(--muted-foreground)]">
              Loading committed runs…
            </p>
          ) : null}
          {runsError ? (
            <p className="mt-2 text-xs text-rose-700">
              Run artifacts could not be loaded.
            </p>
          ) : null}
          {!runsLoading && !runsError && runs.length === 0 ? (
            <p className="mt-2 text-xs text-[var(--muted-foreground)]">
              No committed run artifacts yet.
            </p>
          ) : null}
          <div className="mt-2 space-y-2">
            {runs.map((run) => (
              <Link
                key={run.id}
                to="/activity/sessions/$sessionId/runs/$runId"
                params={{ sessionId: run.session_id, runId: run.id }}
                className="flex items-center justify-between gap-2 rounded-lg border border-[var(--border)] bg-[var(--surface)] p-2 text-xs hover:border-[var(--primary)]"
              >
                <span className="min-w-0">
                  <span className="block truncate font-semibold">
                    Run {run.sequence_no}
                  </span>
                  <span className="text-[var(--muted-foreground)]">
                    Run record ·{' '}
                    {run.message?.length
                      ? 'Replay events available'
                      : 'Summary available'}
                  </span>
                </span>
                <StatusBadge status={run.status} />
              </Link>
            ))}
          </div>
        </div>
      </div>
      <EntryListState
        loading={loading}
        error={error}
        empty={!loading && groups.length === 0}
        onRetry={onRetry}
        emptyText="No workspace root entries to group yet."
      >
        {groups.map((group) => (
          <div key={group.id} className="mb-4">
            <div className="px-3 pb-1">
              <h3 className="text-xs font-semibold uppercase tracking-wide text-[var(--muted-foreground)]">
                {group.label}
              </h3>
              <p className="mt-0.5 text-[11px] leading-4 text-[var(--subtle-foreground)]">
                {group.description}
              </p>
            </div>
            {group.items.map((entry) => (
              <WorkspaceEntryButton
                key={entry.path}
                entry={entry}
                selected={selectedFile === entry.path}
                onClick={() =>
                  entry.kind === 'directory'
                    ? onOpenDirectory(entry)
                    : onFileSelect(entry.path)
                }
              />
            ))}
          </div>
        ))}
        <LoadMoreEntries
          hasMore={hasMore}
          loading={loadingMore}
          onLoadMore={onLoadMore}
          automatic
        />
      </EntryListState>
    </aside>
  )
}

function LoadMoreEntries({
  hasMore,
  loading,
  onLoadMore,
  automatic = false,
}: {
  hasMore: boolean
  loading: boolean
  onLoadMore: () => void
  automatic?: boolean
}) {
  if (!hasMore && !loading) return null
  return (
    <div className="p-2 text-center">
      <button
        type="button"
        className="w-full rounded-lg border border-[var(--border)] px-3 py-2 text-xs font-semibold text-[var(--primary)] disabled:cursor-wait disabled:opacity-60"
        disabled={loading}
        onClick={onLoadMore}
      >
        {loading
          ? automatic
            ? 'Loading remaining entries…'
            : 'Loading more…'
          : 'Load more'}
      </button>
    </div>
  )
}

function EntryListState({
  loading,
  error,
  empty,
  emptyText = 'This directory is empty.',
  onRetry,
  children,
}: {
  loading: boolean
  error: Error | null
  empty: boolean
  emptyText?: string
  onRetry: () => void
  children: React.ReactNode
}) {
  return (
    <div className="scrollbar-thin min-h-0 flex-1 overflow-auto p-2">
      {loading ? (
        <p className="p-3 text-sm text-[var(--muted-foreground)]">
          Loading files…
        </p>
      ) : null}
      {error ? (
        <QueryError
          compact
          title="Could not load workspace files"
          error={error}
          onRetry={onRetry}
        />
      ) : null}
      {!loading && !error ? children : null}
      {!error && empty ? (
        <p className="p-6 text-center text-sm text-[var(--muted-foreground)]">
          {emptyText}
        </p>
      ) : null}
    </div>
  )
}

function WorkspaceEntryButton({
  entry,
  selected,
  onClick,
  icon,
}: {
  entry: WorkspaceFileEntry
  selected: boolean
  onClick: () => void
  icon?: typeof File
}) {
  const Icon = icon ?? (entry.kind === 'directory' ? Folder : File)
  return (
    <button
      type="button"
      className={cn(
        'flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-sm transition hover:bg-[var(--subtle)]',
        selected && 'bg-[var(--primary-subtle)] text-[var(--primary)]',
      )}
      onClick={onClick}
    >
      <Icon className="h-4 w-4 shrink-0" />
      <span className="min-w-0 flex-1 truncate">{entry.name}</span>
      {entry.size_bytes != null ? (
        <span className="text-xs text-[var(--subtle-foreground)]">
          {formatBytes(entry.size_bytes)}
        </span>
      ) : null}
    </button>
  )
}

function FilePreview({
  selectedFile,
  file,
  onDownload,
}: {
  selectedFile: string | null
  file: ReturnType<typeof useWorkspaceFileQuery>
  onDownload: () => void
}) {
  return (
    <div className="flex min-h-[28rem] min-w-0 flex-col">
      {selectedFile ? (
        <>
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[var(--border)] p-3">
            <div className="min-w-0">
              <p className="mono truncate text-xs">{selectedFile}</p>
              <p className="mt-1 text-[11px] text-[var(--muted-foreground)]">
                Read-only preview
              </p>
            </div>
            <button
              type="button"
              className="inline-flex h-9 items-center gap-2 rounded-lg border border-[var(--border)] px-3 text-sm font-medium"
              onClick={onDownload}
            >
              <Download className="h-4 w-4" />
              Download
            </button>
          </div>
          <div className="scrollbar-thin min-h-0 flex-1 overflow-auto bg-slate-950 p-4 text-slate-100">
            {file.isLoading ? <p className="text-sm">Loading file…</p> : null}
            {file.isError &&
            file.error instanceof ApiError &&
            file.error.status === 415 ? (
              <div className="text-sm text-amber-200" role="status">
                This file cannot be previewed as UTF-8 text. Download it
                instead.
              </div>
            ) : file.isError ? (
              <QueryError
                compact
                title="This file could not be previewed"
                error={file.error}
                onRetry={() => void file.refetch()}
              />
            ) : null}
            {!file.isError && file.data ? (
              <pre className="whitespace-pre-wrap break-words text-xs leading-6">
                {file.data.content}
              </pre>
            ) : null}
          </div>
        </>
      ) : (
        <EmptyState
          icon={File}
          title="Select a file"
          description="Choose a file to inspect its stored contents. This workspace view never edits files."
          className="m-4 min-h-[24rem] border-0 bg-[var(--subtle)]"
        />
      )}
    </div>
  )
}

function parentPath(path: string | null, root: string | null) {
  if (!path || !root || path === root) return root
  const parent = path.slice(0, path.lastIndexOf('/')) || '/'
  return parent.length >= root.length ? parent : root
}

function formatBytes(value: number) {
  if (value < 1024) return `${value} B`
  if (value < 1024 * 1024) return `${Math.round(value / 1024)} KB`
  return `${(value / (1024 * 1024)).toFixed(1)} MB`
}
