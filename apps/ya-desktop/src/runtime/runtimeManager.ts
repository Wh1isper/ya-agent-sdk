import { invoke } from '@tauri-apps/api/core'

import type { LocalClawStatus } from './localClaw'

export type ActiveClawRuntime = {
  entrypoint: string
  runtimeDir: string
  version: string
  packageSpec: string
  pythonVersion: string
  uvPath: string
  installedAt: number
  contract: string
}

export type InstalledClawRuntime = {
  id: string
  runtimeDir: string
  version?: string | null
  active: boolean
  failed: boolean
  logFile?: string | null
}

export type RuntimeUpdateState = {
  lastCheckedAt?: number | null
  nextCheckAfter?: number | null
  checkInProgress: boolean
  updateReady: boolean
  candidate?: ActiveClawRuntime | null
  lastError?: string | null
  lastLogFile?: string | null
  autoUpdateEnabled: boolean
}

export type RuntimeManagerStatus = {
  active?: ActiveClawRuntime | null
  runtimes: InstalledClawRuntime[]
  uvPath?: string | null
  clawDir: string
  logsDir: string
  updateState: RuntimeUpdateState
  message: string
}

export type RuntimeInstallResult = {
  runtime: ActiveClawRuntime
  logFile: string
  message: string
}

export type RuntimeActionResult = {
  success: boolean
  message: string
}

export type RuntimeUpdateCheckResult = {
  updateReady: boolean
  candidate?: ActiveClawRuntime | null
  logFile?: string | null
  message: string
}

export function getRuntimeManagerStatus() {
  return invoke<RuntimeManagerStatus>('get_runtime_manager_status')
}

export function installLatestClawRuntime() {
  return invoke<RuntimeInstallResult>('install_latest_claw_runtime')
}

export function updateClawRuntime() {
  return invoke<RuntimeInstallResult>('update_claw_runtime')
}

export function repairClawRuntime(version?: string) {
  return invoke<RuntimeInstallResult>('repair_claw_runtime', { version })
}

export function removeClawRuntime(version: string) {
  return invoke<RuntimeActionResult>('remove_claw_runtime', { version })
}

export function checkClawRuntimeUpdate() {
  return invoke<RuntimeUpdateCheckResult>('check_claw_runtime_update')
}

export function applyReadyClawRuntimeUpdate() {
  return invoke<LocalClawStatus>('apply_ready_claw_runtime_update')
}

export function getRuntimeInstallLog(version?: string) {
  return invoke<string>('get_runtime_install_log', { version })
}
