import { invoke } from '@tauri-apps/api/core'

export type LocalClawStatus = {
  running: boolean
  baseUrl?: string | null
  pid?: number | null
  dataDir?: string | null
  workspaceDir?: string | null
  sqlitePath?: string | null
  logFile?: string | null
  lockFile?: string | null
  message: string
}

export function getLocalClawStatus() {
  return invoke<LocalClawStatus>('get_local_claw_status')
}

export function startLocalClaw() {
  return invoke<LocalClawStatus>('start_local_claw')
}

export function stopLocalClaw() {
  return invoke<LocalClawStatus>('stop_local_claw')
}

export function restartLocalClaw() {
  return invoke<LocalClawStatus>('restart_local_claw')
}
