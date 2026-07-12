import { fetchEventSource } from '@microsoft/fetch-event-source'
import { type InfiniteData, useQueryClient } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'

import { useConnectionStore } from '../stores/connectionStore'
import type {
  NotificationEvent,
  RunStatus,
  SessionGetResponse,
  SessionListResponse,
  SessionSandboxState,
  SessionSummary,
  SessionWorkspaceState,
} from '../types'
import { queryKeys } from './queryKeys'

export type NotificationStatus = 'idle' | 'connecting' | 'connected' | 'error'

class FatalNotificationStreamError extends Error {}

export function useNotificationStream() {
  const baseUrl = useConnectionStore((state) => state.baseUrl)
  const apiToken = useConnectionStore((state) => state.apiToken)
  const connectionScope = useConnectionStore((state) => state.connectionScope)
  const invalidateConnection = useConnectionStore(
    (state) => state.invalidateConnection,
  )
  const queryClient = useQueryClient()
  const [status, setStatus] = useState<NotificationStatus>('idle')
  const replayCursorRef = useRef<{
    connectionScope: string
    lastEventId: string | null
  }>({ connectionScope, lastEventId: null })

  useEffect(() => {
    if (replayCursorRef.current.connectionScope !== connectionScope) {
      replayCursorRef.current = { connectionScope, lastEventId: null }
    }
    if (!apiToken.trim()) {
      setStatus('idle')
      return
    }

    const controller = new AbortController()
    setStatus('connecting')

    const streamPromise = fetchEventSource(
      `${baseUrl.replace(/\/$/, '')}/api/v1/claw/notifications`,
      {
        signal: controller.signal,
        headers: {
          Authorization: `Bearer ${apiToken.trim()}`,
          ...(replayCursorRef.current.lastEventId
            ? { 'Last-Event-ID': replayCursorRef.current.lastEventId }
            : {}),
        },
        openWhenHidden: true,
        async onopen(response) {
          if (!response.ok) {
            if (response.status === 401) {
              setStatus('error')
              invalidateConnection(
                'Your API token is invalid or expired.',
                connectionScope,
              )
              throw new FatalNotificationStreamError(
                'notification stream authentication failed',
              )
            }
            if (response.status >= 400 && response.status < 500) {
              setStatus('error')
              throw new FatalNotificationStreamError(
                `notification stream failed with ${response.status}`,
              )
            }
            setStatus('connecting')
            return
          }
          setStatus('connected')
        },
        onmessage(message) {
          if (
            message.id &&
            replayCursorRef.current.connectionScope === connectionScope
          ) {
            replayCursorRef.current.lastEventId = message.id
          }
          if (!message.data) return
          try {
            const event = JSON.parse(message.data) as NotificationEvent
            invalidateForNotification(queryClient, event)
          } catch (error) {
            console.warn('Ignored malformed YA Claw notification', error)
          }
        },
        onclose() {
          if (!controller.signal.aborted) {
            setStatus('connecting')
            throw new Error('notification stream closed')
          }
        },
        onerror(error) {
          if (error instanceof FatalNotificationStreamError) throw error
          if (!controller.signal.aborted) setStatus('connecting')
          return 2_000
        },
      },
    )
    void streamPromise.catch((error: unknown) => {
      if (
        controller.signal.aborted ||
        (error instanceof Error && error.name === 'AbortError')
      ) {
        return
      }
      setStatus('error')
    })

    return () => {
      controller.abort()
    }
  }, [apiToken, baseUrl, connectionScope, invalidateConnection, queryClient])

  return status
}

function stringPayloadField(
  payload: Record<string, unknown>,
  ...names: string[]
) {
  for (const name of names) {
    const value = payload[name]
    if (typeof value === 'string' && value.trim()) return value
  }
  return null
}

function nullableStringPayloadField(
  payload: Record<string, unknown>,
  name: string,
): string | null | undefined {
  if (!(name in payload)) return undefined
  const value = payload[name]
  return value === null || typeof value === 'string' ? value : undefined
}

function objectPayloadField(
  payload: Record<string, unknown>,
  name: string,
): Record<string, unknown> | undefined {
  const value = payload[name]
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined
}

function runStatusFromNotification(event: NotificationEvent) {
  const status = stringPayloadField(event.payload, 'status')
  return isRunStatus(status) ? status : null
}

function sessionStatusFromRunStatus(status: RunStatus) {
  return status === 'queued' || status === 'running' ? status : 'idle'
}

function isRunStatus(value: string | null): value is RunStatus {
  return (
    value === 'queued' ||
    value === 'running' ||
    value === 'completed' ||
    value === 'failed' ||
    value === 'cancelled'
  )
}

function isSessionSandboxState(value: unknown): value is SessionSandboxState {
  if (!value || typeof value !== 'object') return false
  const candidate = value as Record<string, unknown>
  return typeof candidate.status === 'string'
}

function patchSessionPages(
  queryClient: ReturnType<typeof useQueryClient>,
  update: (session: SessionSummary) => SessionSummary,
) {
  queryClient.setQueryData<InfiniteData<SessionListResponse>>(
    queryKeys.sessions,
    (previous) => {
      if (!previous) return previous
      const updatedSessions = previous.pages
        .flatMap((page) => page.sessions.map(update))
        .sort(
          (left, right) =>
            right.updated_at.localeCompare(left.updated_at) ||
            right.id.localeCompare(left.id),
        )
      let offset = 0
      return {
        ...previous,
        pages: previous.pages.map((page) => {
          const sessions = updatedSessions.slice(
            offset,
            offset + page.sessions.length,
          )
          offset += page.sessions.length
          return { ...page, sessions }
        }),
      }
    },
  )
}

function patchSessionWorkspaceFromNotification(
  queryClient: ReturnType<typeof useQueryClient>,
  event: NotificationEvent,
  sessionId: string | null,
) {
  if (!sessionId || event.type !== 'workspace.sandbox.updated') return
  const sandboxState = event.payload.sandbox_state
  if (!isSessionSandboxState(sandboxState)) return

  const applyWorkspaceState = (
    workspaceState: SessionWorkspaceState | null | undefined,
  ): SessionWorkspaceState => ({
    binding: workspaceState?.binding ?? null,
    sandbox_state: sandboxState,
  })

  patchSessionPages(queryClient, (session) =>
    session.id === sessionId
      ? {
          ...session,
          workspace_state: applyWorkspaceState(session.workspace_state),
        }
      : session,
  )
  queryClient.setQueriesData<SessionGetResponse>(
    { queryKey: queryKeys.session(sessionId) },
    (previous) =>
      previous
        ? {
            ...previous,
            session: {
              ...previous.session,
              workspace_state: applyWorkspaceState(
                previous.session.workspace_state,
              ),
            },
          }
        : previous,
  )
  queryClient.setQueryData<SessionWorkspaceState>(
    queryKeys.sessionWorkspace(sessionId),
    (previous) => applyWorkspaceState(previous),
  )
  queryClient.setQueryData<SessionSandboxState>(
    queryKeys.sessionSandbox(sessionId),
    sandboxState,
  )
}

function patchSessionStatusFromNotification(
  queryClient: ReturnType<typeof useQueryClient>,
  event: NotificationEvent,
  sessionId: string | null,
  runId: string | null,
) {
  if (!sessionId) return
  const runStatus = event.type.startsWith('run.')
    ? runStatusFromNotification(event)
    : null
  if (!runStatus) return
  const sessionStatus = sessionStatusFromRunStatus(runStatus)

  patchSessionPages(queryClient, (session) =>
    session.id === sessionId ? { ...session, status: sessionStatus } : session,
  )
  queryClient.setQueriesData<SessionGetResponse>(
    { queryKey: queryKeys.session(sessionId) },
    (previous) =>
      previous
        ? {
            ...previous,
            session: { ...previous.session, status: sessionStatus },
          }
        : previous,
  )
  if (runId) {
    patchSessionPages(queryClient, (session) => {
      if (session.id !== sessionId || session.latest_run?.id !== runId) {
        return session
      }
      return {
        ...session,
        latest_run: { ...session.latest_run, status: runStatus },
      }
    })
  }
}

function patchSessionSummaryFromNotification(
  queryClient: ReturnType<typeof useQueryClient>,
  event: NotificationEvent,
  sessionId: string | null,
) {
  if (!sessionId || event.type !== 'session.updated') return
  const payload = event.payload
  const rawStatus = stringPayloadField(payload, 'status')
  const status = isRunStatus(rawStatus)
    ? sessionStatusFromRunStatus(rawStatus)
    : undefined
  const statusReason = nullableStringPayloadField(payload, 'status_reason')
  const statusDetail = objectPayloadField(payload, 'status_detail')
  const profileName = nullableStringPayloadField(payload, 'profile_name')
  const headRunId = nullableStringPayloadField(payload, 'head_run_id')
  const activeRunId = nullableStringPayloadField(payload, 'active_run_id')
  const latestRunId = nullableStringPayloadField(payload, 'latest_run_id')
  const updatedAt = nullableStringPayloadField(payload, 'updated_at')
  const terminationReason = nullableStringPayloadField(
    payload,
    'termination_reason',
  )
  const errorMessage = nullableStringPayloadField(payload, 'error_message')

  const updateSession = <T extends SessionSummary>(session: T): T => {
    if (session.id !== sessionId) return session
    const latestRun =
      session.latest_run && latestRunId === session.latest_run.id
        ? {
            ...session.latest_run,
            ...(rawStatus && isRunStatus(rawStatus)
              ? { status: rawStatus }
              : {}),
            ...(terminationReason !== undefined
              ? { termination_reason: terminationReason }
              : {}),
            ...(errorMessage !== undefined
              ? { error_message: errorMessage }
              : {}),
          }
        : session.latest_run
    return {
      ...session,
      ...(status ? { status } : {}),
      ...(typeof statusReason === 'string'
        ? { status_reason: statusReason }
        : {}),
      ...(statusDetail ? { status_detail: statusDetail } : {}),
      ...(profileName !== undefined ? { profile_name: profileName } : {}),
      ...(headRunId !== undefined ? { head_run_id: headRunId } : {}),
      ...(activeRunId !== undefined ? { active_run_id: activeRunId } : {}),
      ...(updatedAt ? { updated_at: updatedAt } : {}),
      latest_run: latestRun,
    } as T
  }

  patchSessionPages(queryClient, updateSession)
  queryClient.setQueriesData<SessionGetResponse>(
    { queryKey: queryKeys.session(sessionId) },
    (previous) =>
      previous
        ? { ...previous, session: updateSession(previous.session) }
        : previous,
  )
}

function invalidateForNotification(
  queryClient: ReturnType<typeof useQueryClient>,
  event: NotificationEvent,
) {
  const sessionId = stringPayloadField(event.payload, 'session_id')
  const sourceSessionId = stringPayloadField(event.payload, 'source_session_id')
  const runId = stringPayloadField(event.payload, 'run_id', 'id')
  const profileName = stringPayloadField(event.payload, 'profile_name', 'name')

  if (
    event.type.startsWith('session.') ||
    event.type.startsWith('run.') ||
    event.type.startsWith('workspace.') ||
    event.type === 'agency.source_session.submitted'
  ) {
    patchSessionStatusFromNotification(queryClient, event, sessionId, runId)
    patchSessionSummaryFromNotification(queryClient, event, sessionId)
    patchSessionWorkspaceFromNotification(queryClient, event, sessionId)
    if (event.type === 'session.created' || event.type === 'run.created') {
      void queryClient.resetQueries({ queryKey: queryKeys.sessions })
    }
    if (sessionId) {
      void queryClient.invalidateQueries({
        queryKey: queryKeys.session(sessionId),
      })
      void queryClient.invalidateQueries({
        queryKey: queryKeys.sessionHistoryBase(sessionId),
      })
      void queryClient.invalidateQueries({
        queryKey: queryKeys.sessionWorkspace(sessionId),
      })
      void queryClient.invalidateQueries({
        queryKey: queryKeys.sessionSandbox(sessionId),
      })
    }
    if (sourceSessionId && sourceSessionId !== sessionId) {
      void queryClient.invalidateQueries({
        queryKey: queryKeys.session(sourceSessionId),
      })
      void queryClient.invalidateQueries({
        queryKey: queryKeys.sessionHistoryBase(sourceSessionId),
      })
    }
    if (event.type.startsWith('workspace.')) {
      void queryClient.invalidateQueries({
        queryKey: queryKeys.workspaceRuntime,
      })
    }
    if (runId) {
      void queryClient.invalidateQueries({ queryKey: queryKeys.run(runId) })
      void queryClient.invalidateQueries({
        queryKey: queryKeys.runTrace(runId),
      })
    }
  }

  if (event.type.startsWith('agency.')) {
    void queryClient.invalidateQueries({ queryKey: queryKeys.agencyConfig })
    void queryClient.invalidateQueries({ queryKey: queryKeys.agencyStatus })
    void queryClient.invalidateQueries({ queryKey: queryKeys.agencyFires })
  }

  if (event.type.startsWith('profile.') || event.type === 'profiles.seeded') {
    void queryClient.invalidateQueries({ queryKey: queryKeys.profiles })
    if (profileName)
      void queryClient.invalidateQueries({
        queryKey: queryKeys.profile(profileName),
      })
  }
}
