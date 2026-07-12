import * as Dialog from '@radix-ui/react-dialog'
import * as DropdownMenu from '@radix-ui/react-dropdown-menu'
import { Link, Outlet, useRouterState } from '@tanstack/react-router'
import {
  Activity,
  Bot,
  Check,
  ChevronsLeft,
  ChevronsRight,
  Command,
  FolderTree,
  Home,
  LogOut,
  Menu,
  MessageSquare,
  MoreHorizontal,
  Plug,
  Search,
  Settings,
  ShieldCheck,
  Wifi,
  Workflow,
  X,
} from 'lucide-react'
import { Suspense, useEffect, useMemo, useRef, useState } from 'react'

import { useHealthQuery } from '../api/hooks'
import { useNotificationStream } from '../api/notificationsStream'
import { ConfirmDialog } from '../components/ui'
import { getBackendTone, getNotificationTone } from '../lib/status'
import { parseUrlSelection } from '../lib/urlState'
import { cn } from '../lib/utils'
import { useConnectionStore } from '../stores/connectionStore'
import { useLayoutStore } from '../stores/layoutStore'
import { ConnectionGate } from './ConnectionGate'

const primaryNav = [
  { to: '/', label: 'Home', helper: 'Your command center', icon: Home },
  {
    to: '/conversations',
    label: 'Conversations',
    helper: 'Work with agents',
    icon: MessageSquare,
  },
  {
    to: '/activity',
    label: 'Activity',
    helper: 'Runs and diagnostics',
    icon: Activity,
  },
  {
    to: '/automation',
    label: 'Automation',
    helper: 'Schedules and workflows',
    icon: Workflow,
  },
  {
    to: '/workspace',
    label: 'Workspace',
    helper: 'Files, memory, artifacts',
    icon: FolderTree,
  },
] as const

const secondaryNav = [
  {
    to: '/agents',
    label: 'Agents',
    helper: 'Models, tools, behavior',
    icon: Bot,
  },
  {
    to: '/integrations',
    label: 'Integrations',
    helper: 'Channels and delivery',
    icon: Plug,
  },
  {
    to: '/settings',
    label: 'Settings',
    helper: 'Connection and runtime',
    icon: Settings,
  },
] as const

const pageCopy = [
  { prefix: '/conversations', eyebrow: 'Work', title: 'Conversations' },
  { prefix: '/activity', eyebrow: 'Operate', title: 'Activity' },
  {
    prefix: '/automation/schedules',
    eyebrow: 'Automation',
    title: 'Schedules',
  },
  {
    prefix: '/automation/workflows',
    eyebrow: 'Automation',
    title: 'Workflows',
  },
  {
    prefix: '/automation/agency',
    eyebrow: 'Automation',
    title: 'Proactive agent',
  },
  {
    prefix: '/automation/heartbeat',
    eyebrow: 'Automation',
    title: 'Heartbeat',
  },
  { prefix: '/automation', eyebrow: 'Operate', title: 'Automation' },
  { prefix: '/workspace', eyebrow: 'Work', title: 'Workspace' },
  { prefix: '/agents', eyebrow: 'Configure', title: 'Agents' },
  {
    prefix: '/integrations',
    eyebrow: 'Configure',
    title: 'Integrations',
  },
  { prefix: '/settings', eyebrow: 'Configure', title: 'Settings' },
] as const

export function AuthenticatedAppShell() {
  return (
    <ConnectionGate>
      <AppShell />
    </ConnectionGate>
  )
}

export function AppShell() {
  const pathname = useRouterState({
    select: (state) => state.location.pathname,
  })
  const baseUrl = useConnectionStore((state) => state.baseUrl)
  const connectionDraftDirty = useConnectionStore(
    (state) => state.connectionDraftDirty,
  )
  const logout = useConnectionStore((state) => state.logout)
  const health = useHealthQuery()
  const notificationStatus = useNotificationStream()
  const [moreOpen, setMoreOpen] = useState(false)
  const [commandOpen, setCommandOpen] = useState(false)
  const [disconnectConfirmOpen, setDisconnectConfirmOpen] = useState(false)
  const focusMainAfterMobileNavigation = useRef(false)
  const railCollapsed = useLayoutStore((state) => state.railCollapsed)
  const setRailCollapsed = useLayoutStore((state) => state.setRailCollapsed)
  const advancedMode = useLayoutStore((state) => state.advancedMode)
  const setAdvancedMode = useLayoutStore((state) => state.setAdvancedMode)
  const activePage = useMemo(() => getPageCopy(pathname), [pathname])
  const backendTone = getBackendTone({
    isError: health.isError,
    status: health.data?.status,
  })

  const requestDisconnect = () => {
    if (useConnectionStore.getState().connectionDraftDirty) {
      setDisconnectConfirmOpen(true)
      return
    }
    logout()
  }

  useEffect(() => {
    function openCommandPalette(event: KeyboardEvent) {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault()
        setCommandOpen(true)
      }
    }
    window.addEventListener('keydown', openCommandPalette)
    return () => window.removeEventListener('keydown', openCommandPalette)
  }, [])

  useEffect(() => {
    document.title = `${activePage.title} · YA Claw`
    const frame = window.requestAnimationFrame(() => {
      document.getElementById('main-content')?.focus({ preventScroll: true })
    })
    return () => window.cancelAnimationFrame(frame)
  }, [activePage.title])

  useEffect(() => {
    const next = parseUrlSelection(pathname)
    useLayoutStore.setState((state) => ({
      route: next.route,
      selectedSessionId: next.selectedSessionId,
      selectedRunId: next.selectedRunId,
      selectedChatSessionId:
        next.route === 'chat'
          ? next.selectedSessionId
          : state.selectedChatSessionId,
      selectedChatRunId:
        next.route === 'chat' ? next.selectedRunId : state.selectedChatRunId,
      selectedDebugSessionId:
        next.route === 'debug'
          ? next.selectedSessionId
          : state.selectedDebugSessionId,
      selectedDebugRunId:
        next.route === 'debug' ? next.selectedRunId : state.selectedDebugRunId,
      selectedAgencySessionId:
        next.route === 'agency'
          ? next.selectedSessionId
          : state.selectedAgencySessionId,
      selectedProfileName:
        next.route === 'profiles'
          ? next.selectedProfileName
          : state.selectedProfileName,
    }))
  }, [pathname])

  return (
    <div className="flex h-dvh overflow-hidden bg-[var(--canvas)] text-[var(--foreground)]">
      <a
        href="#main-content"
        className="fixed left-3 top-3 z-[100] -translate-y-20 rounded-lg bg-[var(--primary)] px-3 py-2 text-sm font-semibold text-white shadow-[var(--shadow-md)] transition focus:translate-y-0 focus:outline-none"
      >
        Skip to main content
      </a>
      <aside
        aria-label="Application navigation"
        className={cn(
          'hidden shrink-0 flex-col border-r border-[var(--border)] bg-[var(--surface)] transition-[width] lg:flex',
          railCollapsed ? 'w-[4.75rem]' : 'w-60',
        )}
      >
        <div
          className={cn(
            'flex h-16 items-center gap-3 border-b border-[var(--border)] px-4',
            railCollapsed && 'justify-center px-2',
          )}
        >
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-[var(--primary)] text-xs font-bold text-white shadow-sm">
            YA
          </div>
          {!railCollapsed ? (
            <div className="min-w-0">
              <p className="font-semibold tracking-tight">YA Claw</p>
              <p className="text-xs text-[var(--subtle-foreground)]">
                Agent workspace
              </p>
            </div>
          ) : null}
        </div>

        <nav
          className="scrollbar-thin min-h-0 flex-1 overflow-auto p-3"
          aria-label="Primary navigation"
        >
          {!railCollapsed ? (
            <p className="px-2 pb-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--subtle-foreground)]">
              Workspace
            </p>
          ) : null}
          <div className="space-y-1">
            {primaryNav.map((item) => (
              <NavLink
                key={item.to}
                item={item}
                pathname={pathname}
                collapsed={railCollapsed}
              />
            ))}
          </div>

          {!railCollapsed ? (
            <p className="mt-6 px-2 pb-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--subtle-foreground)]">
              Manage
            </p>
          ) : (
            <div className="my-3 border-t border-[var(--border)]" />
          )}
          <div className="space-y-1">
            {secondaryNav.map((item) => (
              <NavLink
                key={item.to}
                item={item}
                pathname={pathname}
                collapsed={railCollapsed}
              />
            ))}
          </div>
        </nav>

        <div className="space-y-1 border-t border-[var(--border)] p-3">
          <button
            type="button"
            className="inline-flex h-9 w-full items-center justify-center gap-2 rounded-lg text-sm font-medium text-[var(--muted-foreground)] transition hover:bg-[var(--subtle)] hover:text-[var(--foreground)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--focus)]"
            onClick={() => setRailCollapsed(!railCollapsed)}
            aria-label={
              railCollapsed ? 'Expand navigation' : 'Collapse navigation'
            }
          >
            {railCollapsed ? (
              <ChevronsRight className="h-4 w-4" aria-hidden />
            ) : (
              <ChevronsLeft className="h-4 w-4" aria-hidden />
            )}
            {!railCollapsed ? 'Collapse rail' : null}
          </button>
          <button
            type="button"
            className="inline-flex h-9 w-full items-center justify-center gap-2 rounded-lg text-sm font-medium text-[var(--muted-foreground)] transition hover:bg-[var(--subtle)] hover:text-[var(--foreground)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--focus)]"
            onClick={requestDisconnect}
            aria-label="Disconnect runtime"
          >
            <LogOut className="h-4 w-4" aria-hidden />
            {!railCollapsed ? 'Disconnect' : null}
          </button>
        </div>
      </aside>

      <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
        <header className="flex h-16 shrink-0 items-center justify-between border-b border-[var(--border)] bg-[color:var(--surface-translucent)] px-4 backdrop-blur sm:px-6">
          <div className="min-w-0">
            <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--primary)]">
              {activePage.eyebrow}
            </p>
            <p className="truncate text-sm font-semibold tracking-tight text-[var(--foreground)]">
              {activePage.title}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              className="hidden h-10 min-w-52 items-center justify-between gap-3 rounded-lg border border-[var(--border)] bg-[var(--surface)] px-3 text-sm text-[var(--muted-foreground)] shadow-[var(--shadow-sm)] transition hover:border-[var(--border-strong)] md:inline-flex"
              onClick={() => setCommandOpen(true)}
              aria-label="Open command palette"
            >
              <span className="inline-flex items-center gap-2">
                <Search className="h-4 w-4" aria-hidden />
                Search pages and actions
              </span>
              <kbd className="rounded border border-[var(--border)] bg-[var(--subtle)] px-1.5 py-0.5 text-[10px]">
                Ctrl K
              </kbd>
            </button>
            <RuntimeStatusMenu
              baseUrl={baseUrl}
              backendTone={backendTone}
              healthLabel={
                health.data?.status ??
                (health.isError ? 'Runtime unavailable' : 'Checking runtime')
              }
              notificationStatus={notificationStatus}
            />
            <button
              type="button"
              className="hidden h-10 w-10 items-center justify-center rounded-lg border border-[var(--border)] bg-[var(--surface)] text-[var(--muted-foreground)] sm:inline-flex"
              onClick={() => setCommandOpen(true)}
              aria-label="Open quick navigation"
            >
              <Command className="h-4 w-4" aria-hidden />
            </button>
            <Dialog.Root open={moreOpen} onOpenChange={setMoreOpen}>
              <Dialog.Trigger asChild>
                <button
                  type="button"
                  className="inline-flex h-10 w-10 items-center justify-center rounded-lg border border-[var(--border)] bg-[var(--surface)] text-[var(--muted-foreground)] lg:hidden"
                  aria-label="Open navigation"
                >
                  <Menu className="h-5 w-5" />
                </button>
              </Dialog.Trigger>
              <MobileNavigation
                pathname={pathname}
                advancedMode={advancedMode}
                baseUrl={baseUrl}
                backendTone={backendTone}
                healthLabel={
                  health.data?.status ??
                  (health.isError ? 'Runtime unavailable' : 'Checking runtime')
                }
                notificationStatus={notificationStatus}
                onAdvancedModeChange={setAdvancedMode}
                onCommand={() => {
                  setMoreOpen(false)
                  setCommandOpen(true)
                }}
                disconnectRequiresConfirmation={connectionDraftDirty}
                onDisconnect={logout}
                onNavigate={() => {
                  focusMainAfterMobileNavigation.current = true
                  setMoreOpen(false)
                }}
                shouldFocusMainAfterClose={() => {
                  if (!focusMainAfterMobileNavigation.current) return false
                  focusMainAfterMobileNavigation.current = false
                  return true
                }}
              />
            </Dialog.Root>
          </div>
        </header>

        <main
          id="main-content"
          tabIndex={-1}
          className={cn(
            'min-h-0 flex-1 pb-[calc(4.25rem+env(safe-area-inset-bottom))] lg:pb-0',
            isImmersivePath(pathname)
              ? 'overflow-hidden'
              : 'overflow-auto overscroll-contain',
          )}
        >
          <Suspense fallback={<PageLoading />}>
            <Outlet />
          </Suspense>
        </main>
        <MobileNav pathname={pathname} onMore={() => setMoreOpen(true)} />
      </div>
      <ConfirmDialog
        open={disconnectConfirmOpen}
        onOpenChange={setDisconnectConfirmOpen}
        title="Discard connection changes and disconnect?"
        description="Your unsaved backend URL or API token edits will be lost and the active session credential will be cleared."
        confirmLabel="Discard and disconnect"
        cancelLabel="Keep editing"
        danger
        onConfirm={logout}
      />
      <CommandPalette
        open={commandOpen}
        onOpenChange={setCommandOpen}
        advancedMode={advancedMode}
        onAdvancedModeChange={setAdvancedMode}
      />
    </div>
  )
}

type NavItem = (typeof primaryNav)[number] | (typeof secondaryNav)[number]

function NavLink({
  item,
  pathname,
  collapsed,
}: {
  item: NavItem
  pathname: string
  collapsed: boolean
}) {
  const Icon = item.icon
  const active = isActivePath(pathname, item.to)
  return (
    <Link
      to={item.to}
      aria-current={active ? 'page' : undefined}
      title={collapsed ? item.label : undefined}
      className={cn(
        'group flex min-h-11 items-center gap-3 rounded-lg px-2.5 py-2 text-left transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--focus)]',
        collapsed && 'justify-center px-1',
        active
          ? 'bg-[var(--primary-subtle)] text-[var(--primary)]'
          : 'text-[var(--muted-foreground)] hover:bg-[var(--subtle)] hover:text-[var(--foreground)]',
      )}
    >
      <span
        className={cn(
          'flex h-8 w-8 shrink-0 items-center justify-center rounded-md',
          active ? 'bg-[var(--primary)] text-white' : 'bg-[var(--subtle)]',
        )}
      >
        <Icon className="h-4 w-4" aria-hidden />
      </span>
      {!collapsed ? (
        <span className="min-w-0 flex-1">
          <span className="block text-sm font-semibold">{item.label}</span>
          <span
            className={cn(
              'mt-0.5 block truncate text-xs',
              active
                ? 'text-[var(--primary-muted)]'
                : 'text-[var(--subtle-foreground)]',
            )}
          >
            {item.helper}
          </span>
        </span>
      ) : null}
    </Link>
  )
}

function MobileNav({
  pathname,
  onMore,
}: {
  pathname: string
  onMore: () => void
}) {
  const items = primaryNav.slice(0, 3)
  const moreItems = [...primaryNav.slice(3), ...secondaryNav]
  const moreActive = moreItems.some((item) => isActivePath(pathname, item.to))
  return (
    <nav
      className="fixed inset-x-0 bottom-0 z-30 grid grid-cols-4 border-t border-[var(--border)] bg-[color:var(--surface-translucent)] px-2 pb-[env(safe-area-inset-bottom)] pt-1.5 shadow-[var(--shadow-lg)] backdrop-blur lg:hidden"
      aria-label="Mobile navigation"
    >
      {items.map((item) => {
        const Icon = item.icon
        const active = isActivePath(pathname, item.to)
        return (
          <Link
            key={item.to}
            to={item.to}
            aria-current={active ? 'page' : undefined}
            className={cn(
              'flex min-h-14 flex-col items-center justify-center gap-1 rounded-lg text-[11px] font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--focus)]',
              active
                ? 'bg-[var(--primary-subtle)] text-[var(--primary)]'
                : 'text-[var(--muted-foreground)]',
            )}
          >
            <Icon className="h-4 w-4" aria-hidden />
            {item.label}
          </Link>
        )
      })}
      <button
        type="button"
        aria-current={moreActive ? 'page' : undefined}
        className={cn(
          'flex min-h-14 flex-col items-center justify-center gap-1 rounded-lg text-[11px] font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--focus)]',
          moreActive
            ? 'bg-[var(--primary-subtle)] text-[var(--primary)]'
            : 'text-[var(--muted-foreground)]',
        )}
        onClick={onMore}
      >
        <MoreHorizontal className="h-4 w-4" aria-hidden />
        More
      </button>
    </nav>
  )
}

function MobileNavigation({
  pathname,
  advancedMode,
  baseUrl,
  backendTone,
  healthLabel,
  notificationStatus,
  onAdvancedModeChange,
  onCommand,
  disconnectRequiresConfirmation,
  onDisconnect,
  onNavigate,
  shouldFocusMainAfterClose,
}: {
  pathname: string
  advancedMode: boolean
  baseUrl: string
  backendTone: 'ok' | 'pending' | 'error'
  healthLabel: string
  notificationStatus: 'idle' | 'connecting' | 'connected' | 'error'
  onAdvancedModeChange: (enabled: boolean) => void
  onCommand: () => void
  disconnectRequiresConfirmation: boolean
  onDisconnect: () => void
  onNavigate: () => void
  shouldFocusMainAfterClose: () => boolean
}) {
  const [disconnectConfirming, setDisconnectConfirming] = useState(false)

  return (
    <Dialog.Portal>
      <Dialog.Overlay className="fixed inset-0 z-40 bg-slate-950/40 backdrop-blur-[2px] data-[state=closed]:animate-out data-[state=open]:animate-in" />
      <Dialog.Content
        className="fixed inset-x-0 bottom-0 z-50 max-h-[88dvh] overflow-auto rounded-t-2xl border-t border-[var(--border)] bg-[var(--surface)] p-4 pb-[calc(1rem+env(safe-area-inset-bottom))] shadow-[var(--shadow-lg)] focus:outline-none"
        onCloseAutoFocus={(event) => {
          if (shouldFocusMainAfterClose()) {
            event.preventDefault()
            window.requestAnimationFrame(() => {
              document.getElementById('main-content')?.focus({
                preventScroll: true,
              })
            })
          }
        }}
      >
        {disconnectConfirming ? (
          <div className="space-y-5 py-2">
            <div>
              <Dialog.Title className="text-lg font-semibold">
                Discard connection changes and disconnect?
              </Dialog.Title>
              <Dialog.Description className="mt-2 text-sm leading-6 text-[var(--muted-foreground)]">
                Your unsaved backend URL or API token edits will be lost and the
                active session credential will be cleared.
              </Dialog.Description>
            </div>
            <div className="grid gap-2 sm:grid-cols-2">
              <button
                type="button"
                className="rounded-xl border border-[var(--border)] px-4 py-3 text-sm font-semibold"
                onClick={() => setDisconnectConfirming(false)}
              >
                Keep editing
              </button>
              <button
                type="button"
                className="rounded-xl bg-rose-600 px-4 py-3 text-sm font-semibold text-white"
                onClick={onDisconnect}
              >
                Discard and disconnect
              </button>
            </div>
          </div>
        ) : (
          <>
            <div className="flex items-center justify-between">
              <div>
                <Dialog.Title className="text-lg font-semibold">
                  Navigate
                </Dialog.Title>
                <Dialog.Description className="text-sm text-[var(--muted-foreground)]">
                  All YA Claw workspace tools
                </Dialog.Description>
              </div>
              <Dialog.Close
                className="inline-flex h-10 w-10 items-center justify-center rounded-lg border border-[var(--border)]"
                aria-label="Close navigation"
              >
                <X className="h-4 w-4" />
              </Dialog.Close>
            </div>
            <div className="mt-5 grid gap-2 sm:grid-cols-2">
              {[...primaryNav, ...secondaryNav].map((item) => {
                const Icon = item.icon
                return (
                  <Link
                    key={item.to}
                    to={item.to}
                    aria-current={
                      isActivePath(pathname, item.to) ? 'page' : undefined
                    }
                    className={cn(
                      'flex items-center gap-3 rounded-xl border p-3 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--focus)]',
                      isActivePath(pathname, item.to)
                        ? 'border-[var(--primary)] bg-[var(--primary-subtle)]'
                        : 'border-[var(--border)]',
                    )}
                    onClick={onNavigate}
                  >
                    <span className="flex h-10 w-10 items-center justify-center rounded-lg bg-[var(--primary-subtle)] text-[var(--primary)]">
                      <Icon className="h-5 w-5" />
                    </span>
                    <span>
                      <span className="block text-sm font-semibold">
                        {item.label}
                      </span>
                      <span className="block text-xs text-[var(--muted-foreground)]">
                        {item.helper}
                      </span>
                    </span>
                  </Link>
                )
              })}
            </div>
            <section
              className="mt-4 rounded-xl border border-[var(--border)] p-2"
              aria-label="Runtime status"
            >
              <div className="px-3 py-2">
                <p className="text-sm font-semibold">Runtime status</p>
                <p className="mono mt-1 truncate text-xs text-[var(--subtle-foreground)]">
                  {baseUrl}
                </p>
              </div>
              <StatusRow
                icon={Wifi}
                label="Server reachable"
                value={healthLabel}
                tone={backendTone}
              />
              <StatusRow
                icon={Activity}
                label="Live updates"
                value={notificationStatus}
                tone={getNotificationTone(notificationStatus)}
              />
            </section>
            <div className="mt-4 grid gap-2 border-t border-[var(--border)] pt-4 sm:grid-cols-2">
              <button
                type="button"
                className="flex items-center gap-3 rounded-xl border border-[var(--border)] p-3 text-left"
                onClick={onCommand}
              >
                <Search className="h-5 w-5 text-[var(--primary)]" />
                <span>
                  <span className="block text-sm font-semibold">
                    Quick navigation
                  </span>
                  <span className="block text-xs text-[var(--muted-foreground)]">
                    Search pages and actions
                  </span>
                </span>
              </button>
              <button
                type="button"
                role="switch"
                aria-checked={advancedMode}
                className="flex items-center gap-3 rounded-xl border border-[var(--border)] p-3 text-left"
                onClick={() => onAdvancedModeChange(!advancedMode)}
              >
                <span
                  className={cn(
                    'flex h-5 w-5 items-center justify-center rounded border',
                    advancedMode
                      ? 'border-[var(--primary)] bg-[var(--primary)] text-white'
                      : 'border-[var(--border-strong)]',
                  )}
                >
                  {advancedMode ? <Check className="h-3 w-3" /> : null}
                </span>
                <span>
                  <span className="block text-sm font-semibold">
                    Advanced mode
                  </span>
                  <span className="block text-xs text-[var(--muted-foreground)]">
                    Raw events, traces, and IDs
                  </span>
                </span>
              </button>
              <button
                type="button"
                className="flex items-center gap-3 rounded-xl border border-rose-200 p-3 text-left text-rose-700"
                onClick={() => {
                  if (disconnectRequiresConfirmation) {
                    setDisconnectConfirming(true)
                    return
                  }
                  onDisconnect()
                }}
              >
                <LogOut className="h-5 w-5" aria-hidden />
                <span>
                  <span className="block text-sm font-semibold">
                    Disconnect runtime
                  </span>
                  <span className="block text-xs text-rose-600">
                    Clear this session credential
                  </span>
                </span>
              </button>
            </div>
          </>
        )}
      </Dialog.Content>
    </Dialog.Portal>
  )
}

function RuntimeStatusMenu({
  baseUrl,
  backendTone,
  healthLabel,
  notificationStatus,
}: {
  baseUrl: string
  backendTone: 'ok' | 'pending' | 'error'
  healthLabel: string
  notificationStatus: 'idle' | 'connecting' | 'connected' | 'error'
}) {
  const liveTone = getNotificationTone(notificationStatus)
  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <button
          type="button"
          className="inline-flex h-10 items-center gap-2 rounded-lg border border-[var(--border)] bg-[var(--surface)] px-3 text-xs font-semibold text-[var(--muted-foreground)] shadow-[var(--shadow-sm)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--focus)]"
          aria-label="Open runtime connection status"
        >
          <StatusDot status={backendTone} />
          <StatusDot status={liveTone} />
          <span className="hidden sm:inline">Runtime</span>
        </button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content
          align="end"
          sideOffset={8}
          className="z-50 w-80 rounded-xl border border-[var(--border)] bg-[var(--surface)] p-2 shadow-[var(--shadow-lg)]"
        >
          <div className="px-3 py-2">
            <p className="text-sm font-semibold">Connection status</p>
            <p className="mono mt-1 truncate text-xs text-[var(--subtle-foreground)]">
              {baseUrl}
            </p>
          </div>
          <div className="my-1 border-t border-[var(--border)]" />
          <StatusRow
            icon={Wifi}
            label="Server reachable"
            value={healthLabel}
            tone={backendTone}
          />
          <StatusRow
            icon={ShieldCheck}
            label="Authenticated"
            value="Verified for this session"
            tone="ok"
          />
          <StatusRow
            icon={Activity}
            label="Live updates"
            value={notificationStatus}
            tone={liveTone}
          />
          <div className="my-1 border-t border-[var(--border)]" />
          <DropdownMenu.Item asChild>
            <Link
              to="/settings"
              className="flex cursor-pointer items-center justify-between rounded-lg px-3 py-2 text-sm font-medium outline-none hover:bg-[var(--subtle)] focus:bg-[var(--subtle)]"
            >
              Connection settings
              <Settings className="h-4 w-4" aria-hidden />
            </Link>
          </DropdownMenu.Item>
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  )
}

function StatusRow({
  icon: Icon,
  label,
  value,
  tone,
}: {
  icon: typeof Wifi
  label: string
  value: string
  tone: 'ok' | 'pending' | 'error'
}) {
  return (
    <div className="flex items-start gap-3 rounded-lg px-3 py-2">
      <span className="mt-0.5 flex h-7 w-7 items-center justify-center rounded-md bg-[var(--subtle)]">
        <Icon className="h-4 w-4" aria-hidden />
      </span>
      <span className="min-w-0 flex-1">
        <span className="flex items-center gap-2 text-sm font-medium">
          {label}
          <StatusDot status={tone} />
        </span>
        <span className="mt-0.5 block truncate text-xs capitalize text-[var(--subtle-foreground)]">
          {value}
        </span>
      </span>
    </div>
  )
}

function CommandPalette({
  open,
  onOpenChange,
  advancedMode,
  onAdvancedModeChange,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  advancedMode: boolean
  onAdvancedModeChange: (enabled: boolean) => void
}) {
  const [query, setQuery] = useState('')
  const allItems = useMemo(
    () => [
      {
        to: '/conversations/new',
        label: 'New conversation',
        helper: 'Start work with an agent',
        icon: MessageSquare,
      },
      ...primaryNav,
      ...secondaryNav,
    ],
    [],
  )
  const filteredItems = useMemo(() => {
    const needle = query.trim().toLowerCase()
    if (!needle) return allItems
    return allItems.filter((item) =>
      `${item.label} ${item.helper}`.toLowerCase().includes(needle),
    )
  }, [allItems, query])

  useEffect(() => {
    if (!open) setQuery('')
  }, [open])

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-[70] bg-slate-950/45 backdrop-blur-[2px]" />
        <Dialog.Content className="fixed left-1/2 top-[12dvh] z-[80] max-h-[76dvh] w-[min(42rem,calc(100vw-2rem))] -translate-x-1/2 overflow-hidden rounded-2xl border border-[var(--border)] bg-[var(--surface)] shadow-[var(--shadow-lg)] focus:outline-none">
          <Dialog.Title className="sr-only">Quick navigation</Dialog.Title>
          <Dialog.Description className="sr-only">
            Search pages and common YA Claw actions.
          </Dialog.Description>
          <div className="flex items-center gap-3 border-b border-[var(--border)] px-4">
            <Search className="h-5 w-5 text-[var(--subtle-foreground)]" />
            <input
              autoFocus
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              className="h-14 min-w-0 flex-1 bg-transparent text-sm outline-none placeholder:text-[var(--subtle-foreground)]"
              placeholder="Search pages and actions"
              aria-label="Search pages and actions"
            />
            <Dialog.Close
              className="inline-flex h-8 w-8 items-center justify-center rounded-lg border border-[var(--border)]"
              aria-label="Close quick navigation"
            >
              <X className="h-4 w-4" />
            </Dialog.Close>
          </div>
          <div className="scrollbar-thin max-h-[55dvh] overflow-auto p-2">
            <p className="px-3 py-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--subtle-foreground)]">
              Navigate
            </p>
            {filteredItems.map((item) => {
              const Icon = item.icon
              return (
                <Link
                  key={`${item.to}-${item.label}`}
                  to={item.to}
                  className="flex items-center gap-3 rounded-xl px-3 py-2.5 outline-none hover:bg-[var(--subtle)] focus:bg-[var(--subtle)]"
                  onClick={() => onOpenChange(false)}
                >
                  <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-[var(--primary-subtle)] text-[var(--primary)]">
                    <Icon className="h-4 w-4" aria-hidden />
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block text-sm font-semibold">
                      {item.label}
                    </span>
                    <span className="block truncate text-xs text-[var(--subtle-foreground)]">
                      {item.helper}
                    </span>
                  </span>
                </Link>
              )
            })}
            {filteredItems.length === 0 ? (
              <p className="px-3 py-8 text-center text-sm text-[var(--muted-foreground)]">
                No matching page or action.
              </p>
            ) : null}
          </div>
          <div className="flex flex-wrap items-center justify-between gap-3 border-t border-[var(--border)] bg-[var(--subtle)] px-4 py-3">
            <button
              type="button"
              role="switch"
              aria-checked={advancedMode}
              className="inline-flex items-center gap-2 rounded-lg px-2 py-1.5 text-sm font-medium hover:bg-[var(--surface)]"
              onClick={() => onAdvancedModeChange(!advancedMode)}
            >
              <span
                className={cn(
                  'flex h-5 w-5 items-center justify-center rounded border',
                  advancedMode
                    ? 'border-[var(--primary)] bg-[var(--primary)] text-white'
                    : 'border-[var(--border-strong)] bg-[var(--surface)]',
                )}
              >
                {advancedMode ? <Check className="h-3 w-3" /> : null}
              </span>
              Advanced mode
            </button>
            <span className="inline-flex items-center gap-1 text-xs text-[var(--subtle-foreground)]">
              <Command className="h-3.5 w-3.5" /> Ctrl/⌘ K
            </span>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}

function getPageCopy(pathname: string) {
  if (pathname === '/') return { eyebrow: 'Workspace', title: 'Home' }
  return (
    pageCopy.find((item) => pathname.startsWith(item.prefix)) ?? {
      eyebrow: 'YA Claw',
      title: 'Page not found',
    }
  )
}

function isActivePath(pathname: string, to: string) {
  if (to === '/') return pathname === '/'
  return pathname === to || pathname.startsWith(`${to}/`)
}

function isImmersivePath(pathname: string) {
  return (
    pathname.startsWith('/conversations') ||
    pathname.startsWith('/activity') ||
    pathname.startsWith('/automation/agency')
  )
}

export function PageLoading() {
  return (
    <div className="flex h-full min-h-64 items-center justify-center p-6">
      <div
        className="rounded-lg border border-[var(--border)] bg-[var(--surface)] px-4 py-3 text-sm font-medium text-[var(--muted-foreground)] shadow-[var(--shadow-sm)]"
        role="status"
        aria-live="polite"
        aria-atomic="true"
        aria-busy="true"
      >
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
      aria-hidden
    />
  )
}
