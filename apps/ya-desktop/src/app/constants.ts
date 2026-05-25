import {
  BriefcaseBusiness,
  Home,
  Inbox,
  LayoutDashboard,
  MessageSquareText,
} from 'lucide-react'

import type {
  DesktopLayoutPreferences,
  DesktopShellSafetyPolicy,
  DesktopSpace,
  NavItem,
} from './types'

export const defaultSpaceId = 'local-workspace'

export const defaultShellSafetyPolicy: DesktopShellSafetyPolicy = {
  mode: 'review_then_run',
  reviewRiskThreshold: 'extra_high',
  unattendedRiskThreshold: 'extra_high',
  approvalPolicy: 'defer',
  networkPolicy: 'restricted',
  maxRuntimeSeconds: 180,
  auditEnabled: true,
}

export const defaultDesktopSpaces: DesktopSpace[] = [
  {
    id: defaultSpaceId,
    name: 'Desktop workspace',
    path: '',
    runtime: 'Local Claw',
    trust: 'Trusted · Shell review',
    default: true,
    kind: 'embedded',
    executionLocation: 'this_device',
    trustLevel: 'trusted',
    shellSafety: defaultShellSafetyPolicy,
    relay: {
      enabled: false,
      connectionId: null,
      protocol: 'ya-environment-relay.v1',
      capabilities: ['fileops', 'shell', 'artifacts'],
    },
  },
]

export const spacesStorageKey = 'ya-desktop.spaces.v2'
export const legacySpacesStorageKey = 'ya-desktop.spaces.v1'
export const layoutPreferencesStorageKey = 'ya-desktop.layout-preferences.v2'
export const onboardingStorageKey = 'ya-desktop.onboarding.v1'

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
  { route: 'inbox', label: 'Inbox', helper: 'Decisions', icon: Inbox },
]
