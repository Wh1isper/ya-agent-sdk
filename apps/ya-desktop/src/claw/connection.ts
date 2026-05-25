import { getLocalClawStatus, type LocalClawStatus } from '../runtime'
import type { DesktopClawConnection } from './types'

export type ActiveClawConnectionState = {
  status: LocalClawStatus
  connection: DesktopClawConnection | null
}

export async function getActiveClawConnection(): Promise<ActiveClawConnectionState> {
  const status = await getLocalClawStatus()
  return {
    status,
    connection: connectionFromLocalStatus(status),
  }
}

export function connectionFromLocalStatus(
  status: LocalClawStatus,
): DesktopClawConnection | null {
  if (!status.running || !status.baseUrl) return null

  return {
    id: 'local',
    kind: 'local_embedded',
    name: 'Local Claw',
    baseUrl: status.baseUrl,
    apiToken: status.apiToken,
    dataDir: status.dataDir,
    workspaceDir: status.workspaceDir,
  }
}
