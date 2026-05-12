import { fetchEventSource } from '@microsoft/fetch-event-source'
import { useQueryClient } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'

import { useConnectionStore } from '../stores/connectionStore'
import type {
  NotificationEvent,
  RunStatus,
  SessionGetResponse,
  SessionSummary,
} from '../types'
import { queryKeys } from './queryKeys'

export type NotificationStatus = 'idle' | 'connecting' | 'connected' | 'error'

export function useNotificationStream() {
  const baseUrl = useConnectionStore((state) => state.baseUrl)
  const apiToken = useConnectionStore((state) => state.apiToken)
  const queryClient = useQueryClient()
  const [status, setStatus] = useState<NotificationStatus>('idle')
  const lastEventIdRef = useRef<string | null>(null)

  useEffect(() => {
    if (!apiToken.trim()) {
      setStatus('idle')
      return
    }

    const controller = new AbortController()
    setStatus('connecting')

    void fetchEventSource(
      `${baseUrl.replace(/\/$/, '')}/api/v1/claw/notifications`,
      {
        signal: controller.signal,
        headers: {
          Authorization: `Bearer ${apiToken.trim()}`,
          ...(lastEventIdRef.current
            ? { 'Last-Event-ID': lastEventIdRef.current }
            : {}),
        },
        openWhenHidden: true,
        async onopen(response) {
          if (!response.ok) {
            setStatus('error')
            throw new Error(
              `notification stream failed with ${response.status}`,
            )
          }
          setStatus('connected')
        },
        onmessage(message) {
          if (message.id) {
            lastEventIdRef.current = message.id
          }
          if (!message.data) return
          const event = JSON.parse(message.data) as NotificationEvent
          invalidateForNotification(queryClient, event)
        },
        onerror(error) {
          setStatus('error')
          throw error
        },
      },
    )

    return () => {
      controller.abort()
    }
  }, [apiToken, baseUrl, queryClient])

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

  queryClient.setQueryData<SessionSummary[]>(queryKeys.sessions, (previous) =>
    previous?.map((session) =>
      session.id === sessionId
        ? { ...session, status: sessionStatus }
        : session,
    ),
  )
  queryClient.setQueryData<SessionGetResponse>(
    queryKeys.session(sessionId),
    (previous) =>
      previous
        ? {
            ...previous,
            session: { ...previous.session, status: sessionStatus },
          }
        : previous,
  )
  if (runId) {
    queryClient.setQueryData<SessionSummary[]>(queryKeys.sessions, (previous) =>
      previous?.map((session) => {
        if (session.id !== sessionId || session.latest_run?.id !== runId) {
          return session
        }
        return {
          ...session,
          latest_run: { ...session.latest_run, status: runStatus },
        }
      }),
    )
  }
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
    event.type.startsWith('workspace.')
  ) {
    patchSessionStatusFromNotification(queryClient, event, sessionId, runId)
    void queryClient.invalidateQueries({ queryKey: queryKeys.sessions })
    if (sessionId) {
      void queryClient.invalidateQueries({
        queryKey: queryKeys.session(sessionId),
      })
      void queryClient.invalidateQueries({
        queryKey: queryKeys.sessionWorkspace(sessionId),
      })
      void queryClient.invalidateQueries({
        queryKey: queryKeys.sessionSandbox(sessionId),
      })
    }
    if (sourceSessionId && sourceSessionId !== sessionId)
      void queryClient.invalidateQueries({
        queryKey: queryKeys.session(sourceSessionId),
      })
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

  if (event.type.startsWith('profile.') || event.type === 'profiles.seeded') {
    void queryClient.invalidateQueries({ queryKey: queryKeys.profiles })
    if (profileName)
      void queryClient.invalidateQueries({
        queryKey: queryKeys.profile(profileName),
      })
  }
}
