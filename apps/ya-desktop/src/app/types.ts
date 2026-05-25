import type { LucideIcon } from 'lucide-react'

export type AppRoute =
  | 'home'
  | 'chats'
  | 'board'
  | 'spaces'
  | 'inbox'
  | 'settings'

export type DesktopLayoutPreferences = {
  leftSidebarCollapsed: boolean
  detailPanelOpen: boolean
}

export type HomeStreamStatus =
  | 'idle'
  | 'connecting'
  | 'streaming'
  | 'completed'
  | 'failed'

export type DesktopSpaceKind = 'embedded' | 'local_folder' | 'remote' | 'cloud'

export type DesktopExecutionLocation =
  | 'this_device'
  | 'remote_claw'
  | 'cloud_workspace'

export type DesktopTrustLevel = 'trusted' | 'ask_before_write' | 'read_only'

export type DesktopShellSafetyMode =
  | 'review_then_run'
  | 'read_only_shell'
  | 'disabled'

export type DesktopShellSafetyPolicy = {
  mode: DesktopShellSafetyMode
  reviewRiskThreshold: 'low' | 'medium' | 'high' | 'extra_high'
  unattendedRiskThreshold: 'low' | 'medium' | 'high' | 'extra_high'
  approvalPolicy: 'defer' | 'deny'
  networkPolicy: 'inherit' | 'restricted' | 'blocked'
  maxRuntimeSeconds: number
  auditEnabled: boolean
}

export type DesktopRelayStatus = {
  enabled: boolean
  connectionId?: string | null
  protocol: 'ya-environment-relay.v1'
  capabilities: Array<
    'fileops' | 'shell' | 'tools' | 'resources' | 'artifacts' | 'computer'
  >
}

export type DesktopSpace = {
  id: string
  name: string
  path: string
  runtime: string
  trust: string
  default: boolean
  kind: DesktopSpaceKind
  executionLocation: DesktopExecutionLocation
  trustLevel: DesktopTrustLevel
  shellSafety: DesktopShellSafetyPolicy
  relay: DesktopRelayStatus
}

export type NavItem = {
  route: AppRoute
  label: string
  helper: string
  icon: LucideIcon
}
