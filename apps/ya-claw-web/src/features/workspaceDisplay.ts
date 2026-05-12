import type { SessionSandboxState, WorkspaceRuntimeStatusValue } from '../types'

export type UiTone = 'success' | 'warning' | 'info' | 'muted' | 'error'

export function runtimeHeroTone(
  status: WorkspaceRuntimeStatusValue | string | null | undefined,
): 'success' | 'warning' | 'info' | 'muted' {
  if (status === 'ready') return 'success'
  if (status === 'degraded' || status === 'checking') return 'warning'
  if (status === 'unavailable') return 'warning'
  return 'muted'
}

export function sandboxTone(
  sandbox: SessionSandboxState | null | undefined,
): UiTone {
  if (!sandbox) return 'muted'
  if (sandbox.ready_state === 'ready') return 'success'
  if (sandbox.ready_state === 'starting') return 'warning'
  if (sandbox.ready_state === 'failed') return 'error'
  if (sandbox.status === 'stopped') return 'muted'
  return 'info'
}

export function sandboxLabel(sandbox: SessionSandboxState | null | undefined) {
  if (!sandbox) return 'workspace'
  if (sandbox.ready_state === 'ready') return 'sandbox ready'
  if (sandbox.ready_state === 'starting') return 'starting'
  if (sandbox.ready_state === 'failed') return 'failed'
  if (sandbox.status === 'stopped') return 'stopped'
  return sandbox.status
}

export function ttlLabel(value: number | null | undefined) {
  if (typeof value !== 'number') return 'unknown'
  if (value <= 0) return 'expired'
  if (value < 60) return `${value}s`
  if (value < 3600) return `${Math.floor(value / 60)}m`
  return `${Math.floor(value / 3600)}h ${Math.floor((value % 3600) / 60)}m`
}
