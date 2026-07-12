import type { ClawInfo, HealthStatus, WorkspaceRuntimeStatus } from '../types'
import { ApiError, ClawApiClient } from './client'

export type ConnectionValidationResult = {
  health: HealthStatus
  info: ClawInfo
  workspace: WorkspaceRuntimeStatus | null
}

export function validateRuntimeUrl(value: string) {
  const normalized = value.trim().replace(/\/+$/, '')
  let url: URL
  try {
    url = new URL(normalized)
  } catch {
    throw new Error('Enter a valid runtime URL.')
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    throw new Error('Runtime URL must use HTTP or HTTPS.')
  }
  return normalized
}

export async function validateConnection(
  connection: { baseUrl: string; apiToken: string },
  options: { signal?: AbortSignal } = {},
): Promise<ConnectionValidationResult> {
  const baseUrl = validateRuntimeUrl(connection.baseUrl)
  const apiToken = connection.apiToken.trim()
  if (!apiToken) throw new Error('API token is required.')

  const api = new ClawApiClient({ baseUrl, apiToken })
  const health = await api.health(options)
  assertHealthResponse(health)
  const info = await api.clawInfo(options)
  assertClawInfoResponse(info)
  const workspace = await api.getWorkspaceRuntime(options)
  assertWorkspaceRuntimeResponse(workspace)
  return { health, info, workspace }
}

function assertHealthResponse(value: HealthStatus) {
  if (!value || typeof value !== 'object' || typeof value.status !== 'string') {
    throw new Error('This runtime returned an incompatible health response.')
  }
}

function assertClawInfoResponse(value: ClawInfo) {
  if (
    !value ||
    typeof value !== 'object' ||
    typeof value.name !== 'string' ||
    typeof value.version !== 'string'
  ) {
    throw new Error('This runtime is not compatible with this console.')
  }
}

function assertWorkspaceRuntimeResponse(value: WorkspaceRuntimeStatus) {
  if (
    !value ||
    typeof value !== 'object' ||
    typeof value.backend !== 'string' ||
    typeof value.status !== 'string'
  ) {
    throw new Error('This runtime returned an incompatible workspace response.')
  }
}

export function getConnectionErrorMessage(error: unknown) {
  if (error instanceof ApiError) {
    if (error.status === 0) return 'Cannot reach this YA Claw runtime.'
    if (error.status === 401) return 'The API token is invalid or expired.'
    if (error.status === 403) return 'This API token cannot access the runtime.'
    if (error.status === 404) {
      return 'The YA Claw API was not found at this URL.'
    }
    if (error.status >= 500) {
      return 'The runtime returned an internal error. Try again shortly.'
    }
    return error.message
  }
  return error instanceof Error
    ? error.message
    : 'Connection validation failed.'
}
