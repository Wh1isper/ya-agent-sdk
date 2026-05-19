import { invoke } from '@tauri-apps/api/core'

export type LocalClawEnvVar = {
  key: string
  value: string
}

export type LocalClawLaunchConfig = {
  agencyEnabled: boolean
  memoryEnabled: boolean
  presetName?: string | null
  env: LocalClawEnvVar[]
  configFile?: string | null
}

export type LocalClawStatus = {
  running: boolean
  baseUrl?: string | null
  pid?: number | null
  dataDir?: string | null
  workspaceDir?: string | null
  sqlitePath?: string | null
  logFile?: string | null
  lockFile?: string | null
  apiToken?: string | null
  message: string
}

export function getLocalClawStatus() {
  return invoke<LocalClawStatus>('get_local_claw_status')
}

export function startLocalClaw() {
  return invoke<LocalClawStatus>('start_local_claw')
}

export function getLocalClawLaunchConfig() {
  return invoke<LocalClawLaunchConfig>('get_local_claw_launch_config')
}

export function updateLocalClawLaunchConfig(config: LocalClawLaunchConfig) {
  return invoke<LocalClawLaunchConfig>('update_local_claw_launch_config', { config })
}

export function resetLocalClawLaunchConfig() {
  return invoke<LocalClawLaunchConfig>('reset_local_claw_launch_config')
}

export function importLocalClawLaunchPreset(raw: string) {
  return invoke<LocalClawLaunchConfig>('import_local_claw_launch_preset', { raw })
}

export function stopLocalClaw() {
  return invoke<LocalClawStatus>('stop_local_claw')
}

export function restartLocalClaw() {
  return invoke<LocalClawStatus>('restart_local_claw')
}
