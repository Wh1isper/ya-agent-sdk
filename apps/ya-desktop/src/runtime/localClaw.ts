import { invoke } from '@tauri-apps/api/core'

export type LocalClawEnvVar = {
  key: string
  value: string
}

export type LocalClawLaunchConfig = {
  agencyEnabled: boolean
  memoryEnabled: boolean
  shellReviewEnabled: boolean
  shellReviewModel: string
  shellReviewModelSettings: string
  shellReviewRiskThreshold: 'low' | 'medium' | 'high' | 'extra_high'
  shellReviewUnattendedRiskThreshold: 'low' | 'medium' | 'high' | 'extra_high'
  shellReviewAction: 'defer' | 'deny'
  shellSandboxEnabled: boolean
  shellSandboxBackend:
    | 'auto'
    | 'linux_bwrap_seccomp'
    | 'macos_seatbelt'
    | 'windows_restricted_token'
    | 'raw_host'
  shellSandboxNetwork: 'blocked' | 'restricted' | 'proxy' | 'full'
  shellSandboxAllowRawHost: boolean
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
  profileSeedFile?: string | null
  relayProtocol: 'ya-environment-relay.v1'
  message: string
}

export type DesktopWorkspaceStatus = {
  workspaceRoot: string
  profileSeedFile: string
  relayProtocol: 'ya-environment-relay.v1'
  shellReviewEnabled: boolean
  shellReviewRiskThreshold: string
  shellReviewUnattendedRiskThreshold: string
  shellReviewAction: string
  shellSandboxEnabled: boolean
  shellSandboxBackend: string
  shellSandboxNetwork: string
  shellSandboxAllowRawHost: boolean
  message: string
}

export type DesktopOnboardingResult = {
  config: LocalClawLaunchConfig
  workspaceStatus: DesktopWorkspaceStatus
  apiTokenConfigured: boolean
  message: string
}

export function getLocalClawStatus() {
  return invoke<LocalClawStatus>('get_local_claw_status')
}

export function getDesktopWorkspaceStatus() {
  return invoke<DesktopWorkspaceStatus>('get_desktop_workspace_status')
}

export function runDesktopOnboarding(config?: LocalClawLaunchConfig | null) {
  return invoke<DesktopOnboardingResult>('run_desktop_onboarding', {
    config: config ?? null,
  })
}

export function startLocalClaw() {
  return invoke<LocalClawStatus>('start_local_claw')
}

export function getLocalClawLaunchConfig() {
  return invoke<LocalClawLaunchConfig>('get_local_claw_launch_config')
}

export function updateLocalClawLaunchConfig(config: LocalClawLaunchConfig) {
  return invoke<LocalClawLaunchConfig>('update_local_claw_launch_config', {
    config,
  })
}

export function resetLocalClawLaunchConfig() {
  return invoke<LocalClawLaunchConfig>('reset_local_claw_launch_config')
}

export function importLocalClawLaunchPreset(raw: string) {
  return invoke<LocalClawLaunchConfig>('import_local_claw_launch_preset', {
    raw,
  })
}

export function stopLocalClaw() {
  return invoke<LocalClawStatus>('stop_local_claw')
}

export function restartLocalClaw() {
  return invoke<LocalClawStatus>('restart_local_claw')
}
