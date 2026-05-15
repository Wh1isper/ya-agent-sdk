import {
  Bot,
  Bug,
  CalendarClock,
  Circle,
  HeartPulse,
  Home,
  LogOut,
  MessageSquareMore,
  Settings,
  SlidersHorizontal,
  Wifi,
  WifiOff,
} from 'lucide-react'
import { lazy, Suspense, useEffect } from 'react'

import { useHealthQuery } from '../api/hooks'
import { useNotificationStream } from '../api/notificationsStream'
import { getBackendTone, getNotificationTone } from '../lib/status'
import { cn } from '../lib/utils'
import { useConnectionStore } from '../stores/connectionStore'
import { type AppRoute, useLayoutStore } from '../stores/layoutStore'

const BridgesPage = lazy(() =>
  import('../features/bridges/BridgesPage').then((module) => ({
    default: module.BridgesPage,
  })),
)
const ChatPage = lazy(() =>
  import('../features/chat/ChatPage').then((module) => ({
    default: module.ChatPage,
  })),
)
const DebugPage = lazy(() =>
  import('../features/chat/DebugPage').then((module) => ({
    default: module.DebugPage,
  })),
)
const HeartbeatPage = lazy(() =>
  import('../features/heartbeat/HeartbeatPage').then((module) => ({
    default: module.HeartbeatPage,
  })),
)
const OverviewPage = lazy(() =>
  import('../features/overview/OverviewPage').then((module) => ({
    default: module.OverviewPage,
  })),
)
const ProfilesPage = lazy(() =>
  import('../features/profiles/ProfilesPage').then((module) => ({
    default: module.ProfilesPage,
  })),
)
const SchedulesPage = lazy(() =>
  import('../features/schedules/SchedulesPage').then((module) => ({
    default: module.SchedulesPage,
  })),
)
const SettingsPage = lazy(() =>
  import('../features/settings/SettingsPage').then((module) => ({
    default: module.SettingsPage,
  })),
)

const navItems: Array<{
  route: AppRoute
  label: string
  helper: string
  icon: typeof Home
}> = [
  {
    route: 'overview',
    label: 'Overview',
    helper: 'Health and activity',
    icon: Home,
  },
  { route: 'chat', label: 'Chat', helper: 'Web conversations', icon: Bot },
  { route: 'debug', label: 'Debug', helper: 'Sessions and runs', icon: Bug },
  {
    route: 'schedules',
    label: 'Schedules',
    helper: 'Recurring work',
    icon: CalendarClock,
  },
  {
    route: 'bridges',
    label: 'Bridges',
    helper: 'External events',
    icon: MessageSquareMore,
  },
  {
    route: 'heartbeat',
    label: 'Heartbeat',
    helper: 'Background pulse',
    icon: HeartPulse,
  },
  {
    route: 'profiles',
    label: 'Profiles',
    helper: 'Agent runtime setup',
    icon: SlidersHorizontal,
  },
  {
    route: 'settings',
    label: 'Settings',
    helper: 'Connection details',
    icon: Settings,
  },
]

const routeCopy: Record<AppRoute, { eyebrow: string; title: string }> = {
  overview: { eyebrow: 'Runtime', title: 'Overview' },
  chat: { eyebrow: 'Web', title: 'Chat' },
  debug: { eyebrow: 'AGUI', title: 'Debug Runtime' },
  schedules: { eyebrow: 'Automation', title: 'Schedules' },
  bridges: { eyebrow: 'Integrations', title: 'Bridges' },
  heartbeat: { eyebrow: 'Automation', title: 'Heartbeat' },
  profiles: { eyebrow: 'Configuration', title: 'Profiles' },
  settings: { eyebrow: 'Workspace', title: 'Settings' },
}

export function AppShell() {
  const route = useLayoutStore((state) => state.route)
  const setRoute = useLayoutStore((state) => state.setRoute)
  const syncFromUrl = useLayoutStore((state) => state.syncFromUrl)
  const baseUrl = useConnectionStore((state) => state.baseUrl)
  const apiToken = useConnectionStore((state) => state.apiToken)
  const logout = useConnectionStore((state) => state.logout)
  const health = useHealthQuery()
  const notificationStatus = useNotificationStream()
  const activeRoute = routeCopy[route]
  const backendState = health.data?.status === 'ok' ? 'online' : 'checking'
  const backendTone = getBackendTone({
    isError: health.isError,
    status: health.data?.status,
  })

  useEffect(() => {
    window.addEventListener('popstate', syncFromUrl)
    return () => window.removeEventListener('popstate', syncFromUrl)
  }, [syncFromUrl])

  return (
    <div className="flex min-h-dvh bg-slate-100 text-slate-950">
      <aside className="hidden w-72 shrink-0 flex-col border-r border-slate-200/80 bg-white/95 shadow-sm backdrop-blur lg:flex">
        <div className="border-b border-slate-200 p-5">
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-gradient-to-br from-blue-600 to-indigo-600 text-sm font-semibold text-white shadow-sm">
              YA
            </div>
            <div className="min-w-0">
              <p className="font-semibold tracking-tight">YA Claw</p>
              <p className="text-xs text-slate-500">Runtime Console</p>
            </div>
          </div>
          <div className="mt-4 rounded-2xl border border-slate-200 bg-slate-50 p-3">
            <div className="flex items-center justify-between gap-3">
              <div className="flex min-w-0 items-center gap-2">
                <StatusDot status={backendTone} />
                <span className="text-sm font-medium capitalize text-slate-800">
                  Backend {health.isError ? 'unavailable' : backendState}
                </span>
              </div>
              {health.data?.status === 'ok' ? (
                <Wifi className="h-4 w-4 text-emerald-500" aria-hidden />
              ) : (
                <WifiOff className="h-4 w-4 text-slate-400" aria-hidden />
              )}
            </div>
            <p className="mono mt-2 truncate text-[11px] text-slate-500">
              {baseUrl}
            </p>
          </div>
        </div>
        <NavList route={route} setRoute={setRoute} />
        <div className="border-t border-slate-200 p-4 text-xs text-slate-500">
          <div className="rounded-2xl border border-slate-200 bg-slate-50 p-3">
            <p className="font-medium text-slate-700">Connection</p>
            <p className="mt-1">
              Token {apiToken.trim() ? 'configured' : 'missing'}
            </p>
            <p className="mt-1 capitalize">
              Notifications {notificationStatus}
            </p>
          </div>
          <button
            type="button"
            className="mt-3 inline-flex w-full items-center justify-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 shadow-sm transition hover:bg-slate-50"
            onClick={logout}
          >
            <LogOut className="h-4 w-4" aria-hidden />
            Logout
          </button>
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-16 shrink-0 items-center justify-between border-b border-slate-200/80 bg-white/85 px-3 backdrop-blur sm:px-6">
          <div className="min-w-0">
            <p className="text-xs font-semibold uppercase tracking-wide text-blue-600">
              {activeRoute.eyebrow}
            </p>
            <h1 className="truncate text-lg font-semibold tracking-tight text-slate-950">
              {activeRoute.title}
            </h1>
          </div>
          <div className="hidden items-center gap-3 text-xs text-slate-500 sm:flex">
            <span className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1.5 font-medium text-slate-700 shadow-sm">
              <StatusDot status={backendTone} />
              Backend{' '}
              {health.data?.status ??
                (health.isError ? 'unavailable' : 'checking')}
            </span>
            <span
              className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1.5 font-medium text-slate-700 shadow-sm capitalize"
              aria-live="polite"
            >
              <StatusDot status={getNotificationTone(notificationStatus)} />
              Notifications {notificationStatus}
            </span>
          </div>
        </header>

        <main
          className={cn(
            'min-h-0 flex-1 pb-16 lg:pb-0',
            route === 'chat' || route === 'debug'
              ? 'overflow-hidden'
              : 'overflow-auto',
          )}
        >
          <Suspense fallback={<PageLoading />}>{renderRoute(route)}</Suspense>
        </main>
        <MobileNav route={route} setRoute={setRoute} />
      </div>
    </div>
  )
}

function NavList({
  route,
  setRoute,
}: {
  route: AppRoute
  setRoute: (route: AppRoute) => void
}) {
  return (
    <nav className="flex-1 space-y-1.5 p-3" aria-label="Primary navigation">
      {navItems.map((item) => {
        const Icon = item.icon
        const active = route === item.route
        return (
          <button
            key={item.route}
            type="button"
            aria-current={active ? 'page' : undefined}
            className={cn(
              'group flex w-full items-center gap-3 rounded-2xl px-3 py-3 text-left transition',
              active
                ? 'bg-blue-50 text-blue-700 shadow-sm ring-1 ring-blue-100'
                : 'text-slate-600 hover:bg-slate-50 hover:text-slate-950',
            )}
            onClick={() => setRoute(item.route)}
          >
            <span
              className={cn(
                'flex h-9 w-9 items-center justify-center rounded-xl transition',
                active
                  ? 'bg-blue-600 text-white shadow-sm'
                  : 'bg-slate-100 text-slate-500 group-hover:bg-white group-hover:text-slate-700',
              )}
            >
              <Icon className="h-4 w-4" aria-hidden />
            </span>
            <span className="min-w-0 flex-1">
              <span className="block text-sm font-semibold">{item.label}</span>
              <span
                className={cn(
                  'mt-0.5 block text-xs',
                  active ? 'text-blue-500' : 'text-slate-400',
                )}
              >
                {item.helper}
              </span>
            </span>
            {active ? (
              <Circle
                className="h-2 w-2 fill-current text-blue-600"
                aria-hidden
              />
            ) : null}
          </button>
        )
      })}
    </nav>
  )
}

function MobileNav({
  route,
  setRoute,
}: {
  route: AppRoute
  setRoute: (route: AppRoute) => void
}) {
  return (
    <nav className="fixed inset-x-0 bottom-0 z-20 grid grid-cols-4 border-t border-slate-200 bg-white/95 px-2 py-2 shadow-lg backdrop-blur lg:hidden">
      {navItems.slice(0, 4).map((item) => {
        const Icon = item.icon
        const active = route === item.route
        return (
          <button
            key={item.route}
            type="button"
            className={cn(
              'flex flex-col items-center gap-1 rounded-2xl px-2 py-2 text-[11px] font-medium transition',
              active ? 'bg-blue-50 text-blue-700' : 'text-slate-500',
            )}
            onClick={() => setRoute(item.route)}
          >
            <Icon className="h-4 w-4" aria-hidden />
            <span>{item.label}</span>
          </button>
        )
      })}
    </nav>
  )
}

function renderRoute(route: AppRoute) {
  if (route === 'overview') return <OverviewPage />
  if (route === 'chat') return <ChatPage />
  if (route === 'debug') return <DebugPage />
  if (route === 'schedules') return <SchedulesPage />
  if (route === 'bridges') return <BridgesPage />
  if (route === 'heartbeat') return <HeartbeatPage />
  if (route === 'profiles') return <ProfilesPage />
  return <SettingsPage />
}

function PageLoading() {
  return (
    <div className="flex h-full min-h-64 items-center justify-center p-6">
      <div className="rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm font-medium text-slate-500 shadow-sm">
        Loading workspace…
      </div>
    </div>
  )
}

function StatusDot({ status }: { status: 'ok' | 'pending' | 'error' }) {
  return (
    <span
      className={cn(
        'h-2.5 w-2.5 rounded-full ring-2 ring-white',
        status === 'ok' && 'bg-emerald-500',
        status === 'pending' && 'bg-amber-500',
        status === 'error' && 'bg-rose-500',
      )}
    />
  )
}
