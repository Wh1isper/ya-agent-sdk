import {
  Activity,
  HardDrive,
  LogOut,
  RefreshCcw,
  Save,
  Server,
  ShieldCheck,
  Wrench,
} from 'lucide-react'
import { useBlocker } from '@tanstack/react-router'
import { useEffect, useRef, useState, type ReactNode } from 'react'
import { toast } from 'sonner'

import {
  getConnectionErrorMessage,
  validateConnection,
} from '../../api/connection'
import {
  useClawInfoQuery,
  useHealthQuery,
  useWorkspaceRuntimeQuery,
} from '../../api/hooks'
import { StatusBadge } from '../../components/StatusBadge'
import { ConfirmDialog, QueryError } from '../../components/ui'
import { useConnectionStore } from '../../stores/connectionStore'
import type { RuntimeCheck, WorkspaceRuntimeStatus } from '../../types'

export function SettingsPage() {
  const baseUrl = useConnectionStore((state) => state.baseUrl)
  const apiToken = useConnectionStore((state) => state.apiToken)
  const setConnection = useConnectionStore((state) => state.setConnection)
  const setConnectionDraftDirty = useConnectionStore(
    (state) => state.setConnectionDraftDirty,
  )
  const logout = useConnectionStore((state) => state.logout)
  const health = useHealthQuery()
  const clawInfo = useClawInfoQuery()
  const workspaceRuntime = useWorkspaceRuntimeQuery()
  const [draftBaseUrl, setDraftBaseUrl] = useState(baseUrl)
  const [draftToken, setDraftToken] = useState(apiToken)
  const [isValidating, setIsValidating] = useState(false)
  const [connectionError, setConnectionError] = useState<string | null>(null)
  const [logoutConfirmOpen, setLogoutConfirmOpen] = useState(false)
  const validationGeneration = useRef(0)
  const isMounted = useRef(true)
  const isDirty = draftBaseUrl !== baseUrl || draftToken !== apiToken
  const navigationBlocker = useBlocker({
    shouldBlockFn: () => isDirty,
    enableBeforeUnload: isDirty,
    disabled: !isDirty,
    withResolver: true,
  })

  useEffect(() => {
    setConnectionDraftDirty(isDirty)
  }, [isDirty, setConnectionDraftDirty])

  useEffect(() => {
    isMounted.current = true
    return () => {
      isMounted.current = false
      validationGeneration.current += 1
      setConnectionDraftDirty(false)
    }
  }, [setConnectionDraftDirty])

  function isCurrentValidation(generation: number, connectionScope: string) {
    return (
      isMounted.current &&
      validationGeneration.current === generation &&
      useConnectionStore.getState().connectionScope === connectionScope
    )
  }

  function disconnect() {
    validationGeneration.current += 1
    logout()
    setDraftBaseUrl(useConnectionStore.getState().baseUrl)
    setDraftToken('')
    setConnectionError(null)
    setIsValidating(false)
    setLogoutConfirmOpen(false)
  }

  async function saveConnection() {
    const normalizedBaseUrl = draftBaseUrl.trim().replace(/\/+$/, '')
    const normalizedToken = draftToken.trim()
    if (!normalizedBaseUrl) {
      setConnectionError('Backend URL is required')
      return
    }
    if (!normalizedToken) {
      setConnectionError('API token is required')
      return
    }

    const generation = validationGeneration.current + 1
    validationGeneration.current = generation
    const connectionScope = useConnectionStore.getState().connectionScope
    setConnectionError(null)
    setIsValidating(true)
    try {
      await validateConnection({
        baseUrl: normalizedBaseUrl,
        apiToken: normalizedToken,
      })
      if (!isCurrentValidation(generation, connectionScope)) return

      setConnection({ baseUrl: normalizedBaseUrl, apiToken: normalizedToken })
      setDraftBaseUrl(normalizedBaseUrl)
      setDraftToken(normalizedToken)
      toast.success('Connection verified and saved')
    } catch (error) {
      if (!isCurrentValidation(generation, connectionScope)) return

      const message = getConnectionErrorMessage(error)
      setConnectionError(message)
      toast.error(message)
    } finally {
      if (isMounted.current && validationGeneration.current === generation) {
        setIsValidating(false)
      }
    }
  }

  return (
    <div className="min-h-full bg-slate-100 p-4 sm:p-6 lg:p-8">
      <div className="mx-auto max-w-6xl space-y-6">
        <ConfirmDialog
          open={navigationBlocker.status === 'blocked'}
          onOpenChange={(open) => {
            if (!open && navigationBlocker.status === 'blocked') {
              navigationBlocker.reset()
            }
          }}
          title="Discard unsaved connection changes?"
          description="The backend URL or API token has changed but has not been tested and saved."
          confirmLabel="Discard and leave"
          cancelLabel="Stay here"
          danger
          onConfirm={() => {
            if (navigationBlocker.status === 'blocked') {
              navigationBlocker.proceed()
            }
          }}
        />
        <ConfirmDialog
          open={logoutConfirmOpen}
          onOpenChange={setLogoutConfirmOpen}
          title="Discard connection changes and disconnect?"
          description="Your unsaved backend URL or API token edits will be lost and the active session credential will be cleared."
          confirmLabel="Discard and disconnect"
          cancelLabel="Keep editing"
          danger
          onConfirm={disconnect}
        />
        <header>
          <p className="text-sm font-medium text-blue-600">Administration</p>
          <h1 className="mt-1 text-2xl font-semibold tracking-tight text-slate-950 sm:text-3xl">
            Settings & runtime
          </h1>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">
            Manage this browser connection and inspect the server, storage, and
            workspace execution environment from one place.
          </p>
        </header>

        <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
          <SectionHeading
            icon={<ShieldCheck className="h-5 w-5" />}
            title="Connection"
            description="The console validates the URL and credentials against the server before replacing the active connection."
          />

          {connectionError ? (
            <div
              className="mt-5 rounded-xl border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700"
              role="alert"
            >
              {connectionError}
            </div>
          ) : null}

          <div className="mt-6 grid gap-5 lg:grid-cols-[minmax(0,1fr)_22rem]">
            <div className="space-y-5">
              <label className="block">
                <span className="text-sm font-medium text-slate-700">
                  Backend URL
                </span>
                <input
                  className="mt-2 w-full rounded-xl border border-slate-200 px-3 py-2 text-sm outline-none ring-blue-600 transition focus:ring-2 disabled:cursor-wait disabled:bg-slate-50 disabled:text-slate-500"
                  value={draftBaseUrl}
                  onChange={(event) => setDraftBaseUrl(event.target.value)}
                  disabled={isValidating}
                  placeholder={
                    typeof window !== 'undefined'
                      ? window.location.origin
                      : 'http://127.0.0.1:9042'
                  }
                  autoComplete="url"
                />
              </label>

              <label className="block">
                <span className="text-sm font-medium text-slate-700">
                  API Token
                </span>
                <input
                  className="mt-2 w-full rounded-xl border border-slate-200 px-3 py-2 text-sm outline-none ring-blue-600 transition focus:ring-2 disabled:cursor-wait disabled:bg-slate-50 disabled:text-slate-500"
                  value={draftToken}
                  onChange={(event) => setDraftToken(event.target.value)}
                  disabled={isValidating}
                  type="password"
                  placeholder="YA_CLAW_API_TOKEN"
                  autoComplete="current-password"
                />
              </label>

              <div className="flex flex-wrap items-center gap-3">
                <button
                  type="button"
                  className="inline-flex items-center gap-2 rounded-xl bg-blue-600 px-4 py-2 text-sm font-medium text-white shadow-sm transition hover:bg-blue-700 disabled:cursor-wait disabled:opacity-60"
                  onClick={() => void saveConnection()}
                  disabled={isValidating}
                >
                  <Save className="h-4 w-4" />
                  {isValidating ? 'Testing connection…' : 'Test and save'}
                </button>
                <button
                  type="button"
                  className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-700 shadow-sm transition hover:bg-slate-50"
                  onClick={() => {
                    if (isDirty) {
                      setLogoutConfirmOpen(true)
                      return
                    }
                    disconnect()
                  }}
                >
                  <LogOut className="h-4 w-4" />
                  Logout
                </button>
              </div>
            </div>

            <div className="rounded-xl border border-blue-100 bg-blue-50/70 p-4 text-sm leading-6 text-slate-700">
              <p className="font-semibold text-slate-900">Security & storage</p>
              <ul className="mt-2 list-disc space-y-2 pl-5">
                <li>
                  The token is sent only as a Bearer authorization header to the
                  configured backend.
                </li>
                <li>
                  The token stays in browser memory for this session. It is not
                  written to persisted browser storage.
                </li>
                <li>
                  Only the backend URL is persisted locally so the console can
                  restore the endpoint after a reload.
                </li>
              </ul>
            </div>
          </div>
        </section>

        <div className="grid gap-6 xl:grid-cols-2">
          <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
            <SectionHeading
              icon={<Activity className="h-5 w-5" />}
              title="Server reachability"
              description="Live health polling refreshes every 15 seconds."
              action={
                <RetryButton
                  label="Refresh health"
                  fetching={health.isFetching}
                  onClick={() => void health.refetch()}
                />
              }
            />
            <div className="mt-5">
              {health.isLoading && health.data === undefined ? (
                <LoadingState label="Checking server health…" />
              ) : health.isError ? (
                <QueryError
                  compact
                  title="Server health is unavailable"
                  error={health.error}
                  onRetry={() => void health.refetch()}
                />
              ) : health.data ? (
                <div className="space-y-4">
                  <div className="flex items-center justify-between gap-3 rounded-xl border border-slate-200 p-4">
                    <div>
                      <p className="text-sm font-semibold text-slate-900">
                        Backend reachable
                      </p>
                      <p className="mt-1 text-xs text-slate-500">{baseUrl}</p>
                    </div>
                    <StatusBadge status={health.data.status} />
                  </div>
                  <DefinitionGrid
                    rows={[
                      ['Database', health.data.database],
                      ['Runtime state', health.data.runtime_state],
                    ]}
                  />
                  <TechnicalDetails value={health.data} />
                </div>
              ) : (
                <EmptyState message="The health endpoint returned no status data." />
              )}
            </div>
          </section>

          <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
            <SectionHeading
              icon={<Server className="h-5 w-5" />}
              title="Runtime identity"
              description="Build and deployment identity reported by YA Claw."
              action={
                <RetryButton
                  label="Refresh identity"
                  fetching={clawInfo.isFetching}
                  onClick={() => void clawInfo.refetch()}
                />
              }
            />
            <div className="mt-5">
              {clawInfo.isLoading && clawInfo.data === undefined ? (
                <LoadingState label="Loading runtime identity…" />
              ) : clawInfo.isError ? (
                <QueryError
                  compact
                  title="Runtime identity is unavailable"
                  error={clawInfo.error}
                  onRetry={() => void clawInfo.refetch()}
                />
              ) : clawInfo.data ? (
                <div className="space-y-4">
                  <div className="flex flex-wrap items-center gap-2">
                    <p className="text-lg font-semibold text-slate-950">
                      {clawInfo.data.name || 'Unnamed runtime'}
                    </p>
                    <StatusBadge
                      status={clawInfo.data.environment || 'unknown'}
                    />
                  </div>
                  <DefinitionGrid
                    rows={[
                      ['Application version', clawInfo.data.version],
                      ['Service version', clawInfo.data.service_version],
                      ['Revision', clawInfo.data.service_revision],
                      ['Instance ID', clawInfo.data.instance_id],
                      [
                        'Workspace provider',
                        clawInfo.data.workspace_provider_backend,
                      ],
                      ['Storage model', clawInfo.data.storage_model],
                      ['Authentication', clawInfo.data.auth],
                    ]}
                  />
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                      Enabled surfaces
                    </p>
                    {clawInfo.data.surfaces.length ? (
                      <div className="mt-2 flex flex-wrap gap-2">
                        {clawInfo.data.surfaces.map((surface) => (
                          <span
                            key={surface}
                            className="rounded-full bg-slate-100 px-2.5 py-1 text-xs font-medium text-slate-700"
                          >
                            {surface}
                          </span>
                        ))}
                      </div>
                    ) : (
                      <p className="mt-2 text-sm text-slate-500">
                        No surfaces were reported.
                      </p>
                    )}
                  </div>
                  <TechnicalDetails value={clawInfo.data} />
                </div>
              ) : (
                <EmptyState message="The server returned no runtime identity." />
              )}
            </div>
          </section>
        </div>

        <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
          <SectionHeading
            icon={<HardDrive className="h-5 w-5" />}
            title="Workspace runtime"
            description="Execution backend, filesystem readiness, capabilities, and diagnostic checks."
            action={
              <RetryButton
                label="Refresh workspace runtime"
                fetching={workspaceRuntime.isFetching}
                onClick={() => void workspaceRuntime.refetch()}
              />
            }
          />
          <div className="mt-5">
            {workspaceRuntime.isLoading &&
            workspaceRuntime.data === undefined ? (
              <LoadingState label="Inspecting workspace runtime…" />
            ) : workspaceRuntime.isError ? (
              <QueryError
                compact
                title="Workspace runtime is unavailable"
                error={workspaceRuntime.error}
                onRetry={() => void workspaceRuntime.refetch()}
              />
            ) : workspaceRuntime.data ? (
              <WorkspaceRuntimeDetails runtime={workspaceRuntime.data} />
            ) : (
              <EmptyState message="No workspace runtime is configured or the server returned no runtime data." />
            )}
          </div>
        </section>
      </div>
    </div>
  )
}

function WorkspaceRuntimeDetails({
  runtime,
}: {
  runtime: WorkspaceRuntimeStatus
}) {
  const capabilities = Object.entries(runtime.capabilities)
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center gap-3">
        <StatusBadge status={runtime.status} />
        <span className="text-sm font-semibold capitalize text-slate-900">
          {runtime.backend} backend
        </span>
        <span className="text-sm text-slate-500">
          {runtime.execution_location || 'Execution location not reported'}
        </span>
      </div>

      <DefinitionGrid
        rows={[
          ['Service path', runtime.workspace.service_path || 'Not reported'],
          [
            'Docker host path',
            runtime.workspace.docker_host_path || 'Not applicable',
          ],
          ['Virtual path', runtime.workspace.virtual_path || 'Not reported'],
          ['Workspace exists', runtime.workspace.exists ? 'Yes' : 'No'],
          ['Workspace writable', runtime.workspace.writable ? 'Yes' : 'No'],
          ['Last checked', runtime.updated_at || 'Not reported'],
        ]}
      />

      <div>
        <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
          Capabilities
        </p>
        {capabilities.length ? (
          <div className="mt-2 grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
            {capabilities.map(([name, enabled]) => (
              <div
                key={name}
                className="flex items-center justify-between rounded-xl border border-slate-200 px-3 py-2"
              >
                <span className="text-sm text-slate-700">{humanize(name)}</span>
                <StatusBadge status={enabled ? 'enabled' : 'disabled'} />
              </div>
            ))}
          </div>
        ) : (
          <EmptyState message="No workspace capabilities were reported." />
        )}
      </div>

      <div>
        <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
          Runtime checks
        </p>
        {runtime.checks.length ? (
          <div className="mt-2 grid gap-3 lg:grid-cols-2">
            {runtime.checks.map((check) => (
              <RuntimeCheckRow key={check.id} check={check} />
            ))}
          </div>
        ) : (
          <div className="mt-2">
            <EmptyState message="The runtime did not report any diagnostic checks." />
          </div>
        )}
      </div>

      {runtime.docker ? (
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            Docker diagnostics
          </p>
          <div className="mt-2">
            <DefinitionGrid
              rows={[
                ['Daemon', runtime.docker.daemon.status],
                [
                  'Server version',
                  runtime.docker.daemon.server_version || 'Not reported',
                ],
                ['Image', runtime.docker.image.ref],
                ['Image present', runtime.docker.image.present ? 'Yes' : 'No'],
                [
                  'Retention policy',
                  runtime.docker.retention_policy || 'Not reported',
                ],
                [
                  'Idle TTL',
                  runtime.docker.idle_ttl_seconds == null
                    ? 'Not reported'
                    : `${runtime.docker.idle_ttl_seconds} seconds`,
                ],
              ]}
            />
          </div>
        </div>
      ) : null}

      <TechnicalDetails value={runtime} />
    </div>
  )
}

function RuntimeCheckRow({ check }: { check: RuntimeCheck }) {
  return (
    <div className="rounded-xl border border-slate-200 p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-sm font-semibold text-slate-900">
            {humanize(check.id)}
          </p>
          <p className="mt-1 text-sm leading-5 text-slate-600">
            {check.message || 'No check message was provided.'}
          </p>
        </div>
        <StatusBadge status={check.status} />
      </div>
      {Object.keys(check.details).length ? (
        <TechnicalDetails label="Check details" value={check.details} />
      ) : null}
    </div>
  )
}

function SectionHeading({
  icon,
  title,
  description,
  action,
}: {
  icon: ReactNode
  title: string
  description: string
  action?: ReactNode
}) {
  return (
    <div className="flex flex-wrap items-start justify-between gap-3">
      <div className="flex gap-3">
        <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-blue-50 text-blue-700">
          {icon}
        </span>
        <div>
          <h2 className="text-lg font-semibold text-slate-950">{title}</h2>
          <p className="mt-1 max-w-2xl text-sm leading-6 text-slate-500">
            {description}
          </p>
        </div>
      </div>
      {action}
    </div>
  )
}

function RetryButton({
  label,
  fetching,
  onClick,
}: {
  label: string
  fetching?: boolean
  onClick: () => void
}) {
  return (
    <button
      type="button"
      className="inline-flex items-center gap-2 rounded-lg border border-slate-200 px-3 py-2 text-xs font-semibold text-slate-700 transition hover:bg-slate-50 disabled:opacity-60"
      aria-label={label}
      disabled={fetching}
      onClick={onClick}
    >
      <RefreshCcw className={`h-3.5 w-3.5 ${fetching ? 'animate-spin' : ''}`} />
      {fetching ? 'Refreshing…' : 'Refresh'}
    </button>
  )
}

function DefinitionGrid({ rows }: { rows: Array<[string, string]> }) {
  return (
    <dl className="grid gap-x-6 gap-y-3 sm:grid-cols-2">
      {rows.map(([label, value]) => (
        <div key={label} className="min-w-0 border-b border-slate-100 pb-2">
          <dt className="text-xs font-medium text-slate-500">{label}</dt>
          <dd className="mt-1 break-words text-sm font-medium text-slate-900">
            {value || 'Not reported'}
          </dd>
        </div>
      ))}
    </dl>
  )
}

function TechnicalDetails({
  value,
  label = 'Technical details',
}: {
  value: unknown
  label?: string
}) {
  return (
    <details className="mt-3 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
      <summary className="cursor-pointer text-xs font-semibold text-slate-700">
        {label}
      </summary>
      <pre className="mt-2 max-h-64 overflow-auto whitespace-pre-wrap break-words text-xs leading-5 text-slate-600">
        {JSON.stringify(value, null, 2)}
      </pre>
    </details>
  )
}

function LoadingState({ label }: { label: string }) {
  return (
    <div
      className="flex min-h-36 items-center justify-center rounded-xl border border-slate-200 bg-slate-50 text-sm text-slate-500"
      role="status"
    >
      <Wrench className="mr-2 h-4 w-4 animate-pulse" />
      {label}
    </div>
  )
}

function EmptyState({ message }: { message: string }) {
  return (
    <div className="rounded-xl border border-dashed border-slate-300 bg-slate-50 p-4 text-sm text-slate-500">
      {message}
    </div>
  )
}

function humanize(value: string) {
  return value
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (letter: string) => letter.toUpperCase())
}
