import { useQueries, useQuery } from '@tanstack/react-query'

import { createClawClient } from './client'
import { getActiveClawConnection } from './connection'
import type { ClawRunSummary, DesktopClawConnection } from './types'

export const clawQueryKeys = {
  activeConnection: ['claw', 'active-connection'] as const,
  health: (connection?: DesktopClawConnection | null) =>
    ['claw', connectionScope(connection), 'health'] as const,
  info: (connection?: DesktopClawConnection | null) =>
    ['claw', connectionScope(connection), 'info'] as const,
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

export function useClawSessions(connection?: DesktopClawConnection | null) {
  return useQuery({
    queryKey: clawQueryKeys.sessions(connection),
    queryFn: () =>
      createClawClient(requiredConnection(connection)).listSessions(),
    enabled: Boolean(connection),
    refetchInterval: connection ? 10_000 : false,
  })
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
