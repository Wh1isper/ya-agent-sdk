import {
  defaultDesktopSpaces,
  defaultLayoutPreferences,
  layoutPreferencesStorageKey,
  spacesStorageKey,
} from './constants'
import type { DesktopLayoutPreferences, DesktopSpace } from './types'

export function readSpaces(): DesktopSpace[] {
  if (typeof window === 'undefined') return defaultDesktopSpaces
  try {
    const rawValue = window.localStorage.getItem(spacesStorageKey)
    if (!rawValue) return defaultDesktopSpaces
    const parsedValue = JSON.parse(rawValue) as DesktopSpace[]
    if (!Array.isArray(parsedValue) || parsedValue.length === 0)
      return defaultDesktopSpaces
    return parsedValue.map((space, index) => ({
      id: typeof space.id === 'string' ? space.id : `space-${index}`,
      name: typeof space.name === 'string' ? space.name : 'Workspace',
      path: typeof space.path === 'string' ? space.path : '',
      runtime: typeof space.runtime === 'string' ? space.runtime : 'Local Claw',
      trust: typeof space.trust === 'string' ? space.trust : 'Trusted',
      default: Boolean(space.default),
    }))
  } catch {
    return defaultDesktopSpaces
  }
}

export function writeSpaces(spaces: DesktopSpace[]) {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.setItem(spacesStorageKey, JSON.stringify(spaces))
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
