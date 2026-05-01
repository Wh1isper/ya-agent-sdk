import type { AppRoute } from '../stores/layoutStore'

const routePaths: Record<AppRoute, string> = {
  overview: '/',
  chat: '/chat',
  schedules: '/schedules',
  bridges: '/bridges',
  heartbeat: '/heartbeat',
  profiles: '/profiles',
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
  if (segments[0] === 'chat') {
    return {
      route: 'chat',
      selectedSessionId:
        segments[1] === 'sessions' ? (segments[2] ?? null) : null,
      selectedRunId: segments[3] === 'runs' ? (segments[4] ?? null) : null,
      selectedProfileName: null,
    }
  }
  if (segments[0] === 'profiles') {
    return {
      route: 'profiles',
      selectedSessionId: null,
      selectedRunId: null,
      selectedProfileName: segments[1] ? decodeURIComponent(segments[1]) : null,
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

export function buildChatPath(sessionId: string | null, runId?: string | null) {
  if (!sessionId) return '/chat'
  const encodedSession = encodeURIComponent(sessionId)
  if (!runId) return `/chat/sessions/${encodedSession}`
  return `/chat/sessions/${encodedSession}/runs/${encodeURIComponent(runId)}`
}

export function buildProfilePath(profileName: string | null) {
  return profileName
    ? `/profiles/${encodeURIComponent(profileName)}`
    : '/profiles'
}

export function replaceBrowserPath(path: string) {
  if (window.location.pathname === path) return
  window.history.replaceState(null, '', path)
}

export function pushBrowserPath(path: string) {
  if (window.location.pathname === path) return
  window.history.pushState(null, '', path)
}

function routeFromSegment(segment: string | undefined): AppRoute {
  if (segment === 'schedules') return 'schedules'
  if (segment === 'bridges') return 'bridges'
  if (segment === 'heartbeat') return 'heartbeat'
  if (segment === 'profiles') return 'profiles'
  if (segment === 'settings') return 'settings'
  if (segment === 'chat') return 'chat'
  return 'overview'
}
