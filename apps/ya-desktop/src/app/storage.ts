import {
  defaultDesktopSpaces,
  defaultLayoutPreferences,
  defaultShellSafetyPolicy,
  layoutPreferencesStorageKey,
  legacySpacesStorageKey,
  onboardingStorageKey,
  spacesStorageKey,
} from './constants'
import type {
  DesktopExecutionLocation,
  DesktopLayoutPreferences,
  DesktopShellSafetyMode,
  DesktopShellSafetyPolicy,
  DesktopSpace,
  DesktopSpaceKind,
  DesktopTrustLevel,
} from './types'

export function readSpaces(): DesktopSpace[] {
  if (typeof window === 'undefined') return defaultDesktopSpaces
  try {
    const rawValue =
      window.localStorage.getItem(spacesStorageKey) ??
      window.localStorage.getItem(legacySpacesStorageKey)
    if (!rawValue) return defaultDesktopSpaces
    const parsedValue = JSON.parse(rawValue) as Partial<DesktopSpace>[]
    if (!Array.isArray(parsedValue) || parsedValue.length === 0)
      return defaultDesktopSpaces
    return parsedValue.map(normalizeSpace)
  } catch {
    return defaultDesktopSpaces
  }
}

export function writeSpaces(spaces: DesktopSpace[]) {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.setItem(
      spacesStorageKey,
      JSON.stringify(spaces.map(normalizeSpace)),
    )
  } catch {
    // Keep local workspace selection usable in restricted storage contexts.
  }
}

export function readLayoutPreferences(): DesktopLayoutPreferences {
  if (typeof window === 'undefined') return defaultLayoutPreferences
  try {
    const rawValue = window.localStorage.getItem(layoutPreferencesStorageKey)
    if (!rawValue) return defaultLayoutPreferences
    const parsedValue = JSON.parse(
      rawValue,
    ) as Partial<DesktopLayoutPreferences> & { rightPanelCollapsed?: boolean }
    return {
      leftSidebarCollapsed:
        typeof parsedValue.leftSidebarCollapsed === 'boolean'
          ? parsedValue.leftSidebarCollapsed
          : defaultLayoutPreferences.leftSidebarCollapsed,
      detailPanelOpen:
        typeof parsedValue.detailPanelOpen === 'boolean'
          ? parsedValue.detailPanelOpen
          : typeof parsedValue.rightPanelCollapsed === 'boolean'
            ? !parsedValue.rightPanelCollapsed
            : defaultLayoutPreferences.detailPanelOpen,
    }
  } catch {
    return defaultLayoutPreferences
  }
}

export function writeLayoutPreferences(preferences: DesktopLayoutPreferences) {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.setItem(
      layoutPreferencesStorageKey,
      JSON.stringify(preferences),
    )
  } catch {
    // Keep the prototype usable in restricted storage contexts.
  }
}

export function readOnboardingCompleted() {
  if (typeof window === 'undefined') return true
  try {
    const rawValue = window.localStorage.getItem(onboardingStorageKey)
    if (!rawValue) return false
    const parsedValue = JSON.parse(rawValue) as { completed?: unknown }
    return parsedValue.completed === true
  } catch {
    return false
  }
}

export function writeOnboardingCompleted() {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.setItem(
      onboardingStorageKey,
      JSON.stringify({
        completed: true,
        completedAt: new Date().toISOString(),
      }),
    )
  } catch {
    // Keep startup usable in restricted storage contexts.
  }
}

function normalizeSpace(
  space: Partial<DesktopSpace>,
  index: number,
): DesktopSpace {
  const path = typeof space.path === 'string' ? space.path : ''
  const kind = normalizeKind(space.kind, path)
  const shellSafety = normalizeShellSafety(space.shellSafety)
  const trustLevel = normalizeTrustLevel(
    space.trustLevel,
    space.trust,
    shellSafety.mode,
  )
  const executionLocation = normalizeExecutionLocation(
    space.executionLocation,
    kind,
  )
  return {
    id: typeof space.id === 'string' ? space.id : `space-${index}`,
    name: typeof space.name === 'string' ? space.name : 'Workspace',
    path,
    runtime: typeof space.runtime === 'string' ? space.runtime : 'Local Claw',
    trust:
      typeof space.trust === 'string'
        ? space.trust
        : trustLabel(trustLevel, shellSafety.mode),
    default: Boolean(space.default),
    kind,
    executionLocation,
    trustLevel,
    shellSafety,
    relay: {
      enabled: Boolean(space.relay?.enabled),
      connectionId:
        typeof space.relay?.connectionId === 'string'
          ? space.relay.connectionId
          : null,
      protocol: 'ya-environment-relay.v1',
      capabilities: normalizeRelayCapabilities(space.relay?.capabilities),
    },
  }
}

function normalizeShellSafety(
  policy?: Partial<DesktopShellSafetyPolicy>,
): DesktopShellSafetyPolicy {
  return {
    mode: normalizeShellSafetyMode(policy?.mode),
    reviewRiskThreshold: normalizeRisk(
      policy?.reviewRiskThreshold,
      defaultShellSafetyPolicy.reviewRiskThreshold,
    ),
    unattendedRiskThreshold: normalizeRisk(
      policy?.unattendedRiskThreshold,
      defaultShellSafetyPolicy.unattendedRiskThreshold,
    ),
    approvalPolicy: policy?.approvalPolicy === 'deny' ? 'deny' : 'defer',
    networkPolicy:
      policy?.networkPolicy === 'restricted' ||
      policy?.networkPolicy === 'blocked'
        ? policy.networkPolicy
        : 'inherit',
    maxRuntimeSeconds:
      typeof policy?.maxRuntimeSeconds === 'number' &&
      policy.maxRuntimeSeconds > 0
        ? Math.min(Math.floor(policy.maxRuntimeSeconds), 3600)
        : defaultShellSafetyPolicy.maxRuntimeSeconds,
    auditEnabled:
      typeof policy?.auditEnabled === 'boolean'
        ? policy.auditEnabled
        : defaultShellSafetyPolicy.auditEnabled,
  }
}

function normalizeKind(kind: unknown, path: string): DesktopSpaceKind {
  if (
    kind === 'remote' ||
    kind === 'cloud' ||
    kind === 'local_folder' ||
    kind === 'embedded'
  )
    return kind
  return path.trim() ? 'local_folder' : 'embedded'
}

function normalizeExecutionLocation(
  value: unknown,
  kind: DesktopSpaceKind,
): DesktopExecutionLocation {
  if (
    value === 'remote_claw' ||
    value === 'cloud_workspace' ||
    value === 'this_device'
  )
    return value
  if (kind === 'remote') return 'remote_claw'
  if (kind === 'cloud') return 'cloud_workspace'
  return 'this_device'
}

function normalizeTrustLevel(
  value: unknown,
  legacyTrust: unknown,
  mode: DesktopShellSafetyMode,
): DesktopTrustLevel {
  if (
    value === 'trusted' ||
    value === 'ask_before_write' ||
    value === 'read_only'
  )
    return value
  if (
    typeof legacyTrust === 'string' &&
    legacyTrust.toLowerCase().includes('read')
  )
    return 'read_only'
  if (mode === 'read_only_shell') return 'read_only'
  return 'trusted'
}

function normalizeShellSafetyMode(value: unknown): DesktopShellSafetyMode {
  if (
    value === 'disabled' ||
    value === 'read_only_shell' ||
    value === 'review_then_run'
  )
    return value
  return defaultShellSafetyPolicy.mode
}

function normalizeRisk(
  value: unknown,
  fallback: DesktopShellSafetyPolicy['reviewRiskThreshold'],
) {
  if (
    value === 'low' ||
    value === 'medium' ||
    value === 'high' ||
    value === 'extra_high'
  )
    return value
  return fallback
}

function normalizeRelayCapabilities(
  value: unknown,
): DesktopSpace['relay']['capabilities'] {
  const allowed = new Set([
    'fileops',
    'shell',
    'tools',
    'resources',
    'artifacts',
    'computer',
  ])
  if (!Array.isArray(value)) return ['fileops', 'shell', 'artifacts']
  return value.filter(
    (item): item is DesktopSpace['relay']['capabilities'][number] =>
      typeof item === 'string' && allowed.has(item),
  )
}

function trustLabel(
  trustLevel: DesktopTrustLevel,
  shellMode: DesktopShellSafetyMode,
) {
  if (trustLevel === 'read_only') return 'Read-only'
  if (shellMode === 'disabled') return 'Shell disabled'
  if (shellMode === 'read_only_shell') return 'Read-only shell'
  return 'Trusted · Shell review'
}
