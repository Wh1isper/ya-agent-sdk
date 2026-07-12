import { navigateApp } from '../app/navigation'
import type { AppRoute } from '../stores/layoutStore'

const routePaths: Record<AppRoute, string> = {
  overview: '/',
  chat: '/conversations',
  debug: '/activity',
  automation: '/automation',
  agency: '/automation/agency',
  schedules: '/automation/schedules',
  workflows: '/automation/workflows',
  bridges: '/integrations',
  heartbeat: '/automation/heartbeat',
  workspace: '/workspace',
  profiles: '/agents',
  settings: '/settings',
}

export type UrlSelection = {
  route: AppRoute
  selectedSessionId: string | null
  selectedRunId: string | null
  selectedProfileName: string | null
}

export function parseUrlSelection(
  pathname = window.location.pathname,
): UrlSelection {
  const segments = pathname.split('/').filter(Boolean)
  if (
    segments[0] === 'chat' ||
    segments[0] === 'debug' ||
    segments[0] === 'conversations' ||
    segments[0] === 'activity'
  ) {
    const isDebug = segments[0] === 'debug' || segments[0] === 'activity'
    const selectedSessionId =
      segments[1] === 'sessions' ? safeDecodePathSegment(segments[2]) : null
    return {
      route: isDebug ? 'debug' : 'chat',
      selectedSessionId,
      selectedRunId:
        selectedSessionId && segments[3] === 'runs'
          ? safeDecodePathSegment(segments[4])
          : null,
      selectedProfileName: null,
    }
  }
  if (
    segments[0] === 'agency' ||
    (segments[0] === 'automation' && segments[1] === 'agency')
  ) {
    const sessionsIndex = segments[0] === 'agency' ? 1 : 2
    const selectedSessionId =
      segments[sessionsIndex] === 'sessions'
        ? safeDecodePathSegment(segments[sessionsIndex + 1])
        : null
    return {
      route: 'agency',
      selectedSessionId,
      selectedRunId:
        selectedSessionId && segments[sessionsIndex + 2] === 'runs'
          ? safeDecodePathSegment(segments[sessionsIndex + 3])
          : null,
      selectedProfileName: null,
    }
  }
  if (segments[0] === 'automation') {
    const route: AppRoute =
      segments[1] === 'schedules'
        ? 'schedules'
        : segments[1] === 'workflows'
          ? 'workflows'
          : segments[1] === 'heartbeat'
            ? 'heartbeat'
            : segments[1] === 'background'
              ? 'agency'
              : 'automation'
    return {
      route,
      selectedSessionId: null,
      selectedRunId: null,
      selectedProfileName: null,
    }
  }
  if (segments[0] === 'profiles' || segments[0] === 'agents') {
    return {
      route: 'profiles',
      selectedSessionId: null,
      selectedRunId: null,
      selectedProfileName:
        segments[1] === 'new'
          ? '__new__'
          : safeDecodePathSegment(
              segments[1] === 'by-name' ? segments[2] : segments[1],
            ),
    }
  }
  const route = routeFromSegment(segments[0])
  return {
    route,
    selectedSessionId: null,
    selectedRunId: null,
    selectedProfileName: null,
  }
}

export function buildRoutePath(route: AppRoute) {
  return routePaths[route]
}

export function buildChatPath(
  sessionId: string | null,
  runId?: string | null,
  route: 'chat' | 'debug' = 'chat',
) {
  const prefix = route === 'debug' ? '/activity' : '/conversations'
  if (!sessionId) return prefix
  const encodedSession = encodeURIComponent(sessionId)
  if (!runId) return `${prefix}/sessions/${encodedSession}`
  return `${prefix}/sessions/${encodedSession}/runs/${encodeURIComponent(runId)}`
}

export function buildAgencyPath(
  sessionId: string | null,
  runId?: string | null,
) {
  if (!sessionId) return '/automation/agency'
  const sessionPath = `/automation/agency/sessions/${encodeURIComponent(sessionId)}`
  return runId
    ? `${sessionPath}/runs/${encodeURIComponent(runId)}`
    : sessionPath
}

export function buildProfilePath(profileName: string | null) {
  if (profileName === '__new__') return '/agents/new'
  return profileName
    ? `/agents/by-name/${encodeURIComponent(profileName)}`
    : '/agents'
}

export function replaceBrowserPath(path: string, force = false) {
  if (!force && window.location.pathname === path) return
  navigateApp(path, true)
}

export function pushBrowserPath(path: string) {
  if (window.location.pathname === path) return
  navigateApp(path)
}

export function safeDecodePathSegment(segment: string | undefined) {
  if (!segment) return null
  try {
    return decodeURIComponent(segment)
  } catch {
    return null
  }
}

function routeFromSegment(segment: string | undefined): AppRoute {
  if (segment === 'debug' || segment === 'activity') return 'debug'
  if (segment === 'agency') return 'agency'
  if (segment === 'schedules') return 'schedules'
  if (segment === 'workflows') return 'workflows'
  if (segment === 'bridges' || segment === 'integrations') return 'bridges'
  if (segment === 'heartbeat') return 'heartbeat'
  if (segment === 'workspace') return 'workspace'
  if (segment === 'profiles' || segment === 'agents') return 'profiles'
  if (segment === 'automation') return 'automation'
  if (segment === 'settings') return 'settings'
  if (segment === 'chat' || segment === 'conversations') return 'chat'
  return 'overview'
}
