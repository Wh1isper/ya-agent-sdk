import {
  BriefcaseBusiness,
  Home,
  BrainCircuit,
  Inbox,
  LayoutDashboard,
  MessageSquareText,
} from 'lucide-react'

import type { DesktopLayoutPreferences, DesktopSpace, NavItem } from './types'

export const defaultSpaceId = 'local-workspace'

export const defaultDesktopSpaces: DesktopSpace[] = [
  {
    id: defaultSpaceId,
    name: 'Local workspace',
    path: '',
    runtime: 'Local Claw',
    trust: 'Trusted',
    default: true,
  },
]

export const spacesStorageKey = 'ya-desktop.spaces.v1'
export const layoutPreferencesStorageKey = 'ya-desktop.layout-preferences.v2'

export const defaultLayoutPreferences: DesktopLayoutPreferences = {
  leftSidebarCollapsed: false,
  detailPanelOpen: false,
}

export const navItems: NavItem[] = [
  { route: 'home', label: 'Home', helper: 'Start work', icon: Home },
  {
    route: 'chats',
    label: 'Chats',
    helper: 'Conversations',
    icon: MessageSquareText,
  },
  {
    route: 'board',
    label: 'Board',
    helper: 'Work status',
    icon: LayoutDashboard,
  },
  {
    route: 'spaces',
    label: 'Spaces',
    helper: 'Workspaces',
    icon: BriefcaseBusiness,
  },
  {
    route: 'agency',
    label: 'Agency',
    helper: 'Proactive work',
    icon: BrainCircuit,
  },
  { route: 'inbox', label: 'Inbox', helper: 'Decisions', icon: Inbox },
]
