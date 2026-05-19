import { useEffect, useRef } from 'react'
import {
  type QueryClient,
  useMutation,
  useQueries,
  useQuery,
  useQueryClient,
} from '@tanstack/react-query'

import { createClawClient } from './client'
import { getActiveClawConnection } from './connection'
import type {
  ClawNotificationEvent,
  ClawSessionRunStreamInput,
  ClawRunSummary,
  ClawSessionStreamInput,
  ClawStreamHandlers,
  DesktopClawConnection,
} from './types'

export const clawQueryKeys = {
  activeConnection: ['claw', 'active-connection'] as const,
  health: (connection?: DesktopClawConnection | null) =>
    ['claw', connectionScope(connection), 'health'] as const,
  info: (connection?: DesktopClawConnection | null) =>
    ['claw', connectionScope(connection), 'info'] as const,
  profiles: (connection?: DesktopClawConnection | null) =>
    ['claw', connectionScope(connection), 'profiles'] as const,
  sessions: (connection?: DesktopClawConnection | null) =>
    ['claw', connectionScope(connection), 'sessions'] as const,
  session: (
    connection: DesktopClawConnection | null | undefined,
    sessionId: string | null,
  ) => ['claw', connectionScope(connection), 'sessions', sessionId] as const,
  turns: (
    connection: DesktopClawConnection | null | undefined,
    sessionId: string | null,
  ) =>
    [
      'claw',
      connectionScope(connection),
      'sessions',
      sessionId,
      'turns',
    ] as const,
  trace: (
    connection: DesktopClawConnection | null | undefined,
    runId: string | null,
  ) => ['claw', connectionScope(connection), 'runs', runId, 'trace'] as const,
  agencyConfig: (connection?: DesktopClawConnection | null) =>
    ['claw', connectionScope(connection), 'agency', 'config'] as const,
  agencyStatus: (connection?: DesktopClawConnection | null) =>
    ['claw', connectionScope(connection), 'agency', 'status'] as const,
  agencyFires: (connection?: DesktopClawConnection | null) =>
    ['claw', connectionScope(connection), 'agency', 'fires'] as const,
}

export function useActiveClawConnection() {
  return useQuery({
    queryKey: clawQueryKeys.activeConnection,
    queryFn: getActiveClawConnection,
    refetchInterval: 5_000,
  })
}

export function useClawHealth(connection?: DesktopClawConnection | null) {
  return useQuery({
    queryKey: clawQueryKeys.health(connection),
    queryFn: () => createClawClient(requiredConnection(connection)).health(),
    enabled: Boolean(connection),
    refetchInterval: connection ? 10_000 : false,
  })
}

export function useClawInfo(connection?: DesktopClawConnection | null) {
  return useQuery({
    queryKey: clawQueryKeys.info(connection),
    queryFn: () => createClawClient(requiredConnection(connection)).info(),
    enabled: Boolean(connection),
  })
}

export function useClawProfiles(connection?: DesktopClawConnection | null) {
  return useQuery({
    queryKey: clawQueryKeys.profiles(connection),
    queryFn: () => createClawClient(requiredConnection(connection)).listProfiles(),
    enabled: Boolean(connection),
  })
}

export function useClawSessions(connection?: DesktopClawConnection | null) {
  return useQuery({
    queryKey: clawQueryKeys.sessions(connection),
    queryFn: () =>
      createClawClient(requiredConnection(connection)).listSessions(),
    enabled: Boolean(connection),
    refetchInterval: connection ? 10_000 : false,
  })
}

export function useCreateClawSessionStream(
  connection?: DesktopClawConnection | null,
) {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async ({
      input,
      handlers,
      signal,
    }: {
      input: ClawSessionStreamInput
      handlers?: ClawStreamHandlers
      signal?: AbortSignal
    }) => {
      await createClawClient(requiredConnection(connection)).createSessionStream(
        input,
        handlers,
        signal,
      )
    },
    onSettled: async () => {
      await queryClient.invalidateQueries({
        queryKey: clawQueryKeys.sessions(connection),
      })
    },
  })
}

export function useCreateClawSessionRunStream(
  connection?: DesktopClawConnection | null,
) {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async ({
      sessionId,
      input,
      handlers,
      signal,
    }: {
      sessionId: string
      input: ClawSessionRunStreamInput
      handlers?: ClawStreamHandlers
      signal?: AbortSignal
    }) => {
      await createClawClient(requiredConnection(connection)).createSessionRunStream(
        sessionId,
        input,
        handlers,
        signal,
      )
    },
    onSettled: async (_data, _error, variables) => {
      await Promise.all([
        queryClient.invalidateQueries({
          queryKey: clawQueryKeys.sessions(connection),
        }),
        queryClient.invalidateQueries({
          queryKey: clawQueryKeys.session(connection, variables?.sessionId ?? null),
        }),
        queryClient.invalidateQueries({
          queryKey: clawQueryKeys.turns(connection, variables?.sessionId ?? null),
        }),
      ])
    },
  })
}

export function useClawAgencyConfig(connection?: DesktopClawConnection | null) {
  return useQuery({
    queryKey: clawQueryKeys.agencyConfig(connection),
    queryFn: () => createClawClient(requiredConnection(connection)).getAgencyConfig(),
    enabled: Boolean(connection),
  })
}

export function useClawAgencyStatus(connection?: DesktopClawConnection | null) {
  return useQuery({
    queryKey: clawQueryKeys.agencyStatus(connection),
    queryFn: () => createClawClient(requiredConnection(connection)).getAgencyStatus(),
    enabled: Boolean(connection),
    refetchInterval: connection ? 10_000 : false,
  })
}

export function useClawAgencyFires(connection?: DesktopClawConnection | null) {
  return useQuery({
    queryKey: clawQueryKeys.agencyFires(connection),
    queryFn: () => createClawClient(requiredConnection(connection)).listAgencyFires(),
    enabled: Boolean(connection),
    refetchInterval: connection ? 10_000 : false,
  })
}

export function useClearClawAgency(connection?: DesktopClawConnection | null) {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: () => createClawClient(requiredConnection(connection)).clearAgency(),
    onSettled: async () => {
      await Promise.all([
        queryClient.invalidateQueries({
          queryKey: clawQueryKeys.agencyConfig(connection),
        }),
        queryClient.invalidateQueries({
          queryKey: clawQueryKeys.agencyStatus(connection),
        }),
        queryClient.invalidateQueries({
          queryKey: clawQueryKeys.agencyFires(connection),
        }),
        queryClient.invalidateQueries({
          queryKey: clawQueryKeys.sessions(connection),
        }),
      ])
    },
  })
}

export function useCancelClawSession(connection?: DesktopClawConnection | null) {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (sessionId: string) =>
      createClawClient(requiredConnection(connection)).cancelSession(sessionId),
    onSettled: async (_data, _error, sessionId) => {
      await Promise.all([
        queryClient.invalidateQueries({
          queryKey: clawQueryKeys.sessions(connection),
        }),
        queryClient.invalidateQueries({
          queryKey: clawQueryKeys.session(connection, sessionId ?? null),
        }),
      ])
    },
  })
}

export function useClawNotifications(connection?: DesktopClawConnection | null) {
  const queryClient = useQueryClient()
  const lastEventIdRef = useRef<string | null>(null)

  useEffect(() => {
    if (!connection) return
    const activeConnection = connection
    const abortController = new AbortController()

    async function connect() {
      while (!abortController.signal.aborted) {
        try {
          const client = createClawClient(activeConnection)
          await client.streamNotifications(
            {
              onEvent: async (event) => {
                lastEventIdRef.current = event.id || lastEventIdRef.current
                await applyNotificationEvent(queryClient, activeConnection, event)
              },
            },
            abortController.signal,
            lastEventIdRef.current,
          )
        } catch {
          if (abortController.signal.aborted) return
        }
        await delay(1_500, abortController.signal)
      }
    }

    void connect()
    return () => abortController.abort()
  }, [connection, queryClient])
}

export function useClawSession(
  connection: DesktopClawConnection | null | undefined,
  sessionId: string | null,
) {
  return useQuery({
    queryKey: clawQueryKeys.session(connection, sessionId),
    queryFn: () =>
      createClawClient(requiredConnection(connection)).getSession(
        requiredId(sessionId),
      ),
    enabled: Boolean(connection && sessionId),
  })
}

export function useClawSessionTurns(
  connection: DesktopClawConnection | null | undefined,
  sessionId: string | null,
) {
  return useQuery({
    queryKey: clawQueryKeys.turns(connection, sessionId),
    queryFn: () =>
      createClawClient(requiredConnection(connection)).listSessionTurns(
        requiredId(sessionId),
      ),
    enabled: Boolean(connection && sessionId),
  })
}

export function useClawRunTraces(
  connection: DesktopClawConnection | null | undefined,
  runs: ClawRunSummary[],
) {
  return useQueries({
    queries: runs.slice(0, 3).map((run) => ({
      queryKey: clawQueryKeys.trace(connection, run.id),
      queryFn: () =>
        createClawClient(requiredConnection(connection)).getRunTrace(run.id),
      enabled: Boolean(connection && run.id),
    })),
  })
}

async function applyNotificationEvent(
  queryClient: QueryClient,
  connection: DesktopClawConnection,
  event: ClawNotificationEvent,
) {
  if (
    event.type === 'session.created' ||
    event.type === 'session.updated' ||
    event.type === 'session.deleted' ||
    event.type === 'run.created' ||
    event.type === 'run.updated' ||
    event.type === 'run.hitl.responded' ||
    event.type === 'interaction.requested' ||
    event.type === 'interaction.updated' ||
    event.type === 'interaction.resolved' ||
    event.type === 'interaction.expired'
  ) {
    await queryClient.invalidateQueries({
      queryKey: clawQueryKeys.sessions(connection),
    })
    const sessionId = event.payload.session_id ?? event.payload.sessionId
    if (typeof sessionId === 'string') {
      await Promise.all([
        queryClient.invalidateQueries({
          queryKey: clawQueryKeys.session(connection, sessionId),
        }),
        queryClient.invalidateQueries({
          queryKey: clawQueryKeys.turns(connection, sessionId),
        }),
      ])
    }
  }

  if (
    event.type === 'agency.config.updated' ||
    event.type === 'agency.fire.updated' ||
    event.type === 'agency.cleared'
  ) {
    await Promise.all([
      queryClient.invalidateQueries({
        queryKey: clawQueryKeys.agencyConfig(connection),
      }),
      queryClient.invalidateQueries({
        queryKey: clawQueryKeys.agencyStatus(connection),
      }),
      queryClient.invalidateQueries({
        queryKey: clawQueryKeys.agencyFires(connection),
      }),
    ])
  }

  if (
    event.type === 'profile.created' ||
    event.type === 'profile.updated' ||
    event.type === 'profile.deleted' ||
    event.type === 'profiles.seeded'
  ) {
    await queryClient.invalidateQueries({
      queryKey: clawQueryKeys.profiles(connection),
    })
  }
}

function delay(milliseconds: number, signal: AbortSignal) {
  return new Promise<void>((resolve) => {
    const timeoutId = window.setTimeout(resolve, milliseconds)
    signal.addEventListener(
      'abort',
      () => {
        window.clearTimeout(timeoutId)
        resolve()
      },
      { once: true },
    )
  })
}

function connectionScope(connection?: DesktopClawConnection | null) {
  if (!connection) return undefined
  return `${connection.id}:${connection.baseUrl}`
}

function requiredConnection(
  connection?: DesktopClawConnection | null,
): DesktopClawConnection {
  if (!connection) throw new Error('Active Claw connection is unavailable.')
  return connection
}

function requiredId(id: string | null): string {
  if (!id) throw new Error('Claw resource id is unavailable.')
  return id
}
