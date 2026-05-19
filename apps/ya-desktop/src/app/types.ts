import type { LucideIcon } from 'lucide-react'

export type AppRoute = 'home' | 'chats' | 'board' | 'spaces' | 'agency' | 'inbox' | 'settings'

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

export type DesktopSpace = {
  id: string
  name: string
  path: string
  runtime: string
  trust: string
  default: boolean
}

export type NavItem = {
  route: AppRoute
  label: string
  helper: string
  icon: LucideIcon
}
