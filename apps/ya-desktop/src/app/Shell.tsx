import { Activity, Bell, Folder, MessageSquareText, PanelLeft, PanelLeftClose, PanelRight, Plus, Settings, ShieldCheck, type LucideIcon } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Toaster } from 'sonner'

import { useActiveClawConnection, useClawNotifications, useClawSession, type DesktopClawConnection } from '../claw'
import { cn } from '../lib'
import { defaultSpaceId, navItems } from './constants'
import { AppRouteOutlet } from './routes'
import { readLayoutPreferences, readSpaces, writeLayoutPreferences, writeSpaces } from './storage'
import type { AppRoute, DesktopLayoutPreferences, DesktopSpace } from './types'
import { IconButton, PanelCard } from './ui'
import { inboxItemsFromSessions, sessionTitle, spaceDetail } from './utils'

export function DesktopShell() {
  const [route, setRoute] = useState<AppRoute>('home')
  const [layoutPreferences, setLayoutPreferences] =
    useState<DesktopLayoutPreferences>(readLayoutPreferences)
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(
    null,
  )
  const [spaces, setSpaces] = useState<DesktopSpace[]>(readSpaces)
  const [selectedSpaceId, setSelectedSpaceId] = useState(defaultSpaceId)
  const activeConnectionQuery = useActiveClawConnection()
  const connection = activeConnectionQuery.data?.connection ?? null
  useClawNotifications(connection)

  const selectedSpace =
    spaces.find((space) => space.id === selectedSpaceId) ?? spaces[0]
  const activeNavItem =
    route === 'settings'
      ? { route, label: 'Settings', helper: 'Preferences', icon: Settings }
      : (navItems.find((item) => item.route === route) ?? navItems[0])

  useEffect(() => {
    writeLayoutPreferences(layoutPreferences)
  }, [layoutPreferences])

  useEffect(() => {
    writeSpaces(spaces)
  }, [spaces])

  function openSession(sessionId: string) {
    setSelectedSessionId(sessionId)
    setRoute('chats')
  }

  function clearSelectedSession() {
    setSelectedSessionId(null)
  }

  function openHome() {
    setRoute('home')
  }

  function startNewChat() {
    clearSelectedSession()
    setRoute('home')
  }

  function toggleSidebar() {
    setLayoutPreferences((current) => ({
      ...current,
      leftSidebarCollapsed: !current.leftSidebarCollapsed,
    }))
  }

  function toggleDetailPanel() {
    setLayoutPreferences((current) => ({
      ...current,
      detailPanelOpen: !current.detailPanelOpen,
    }))
  }

  return (
    <div className="h-screen bg-[#f7f7f5] text-[#171717]">
      <div className="flex h-full overflow-hidden">
        <Sidebar
          activeRoute={route}
          collapsed={layoutPreferences.leftSidebarCollapsed}
          connection={connection}
          statusMessage={
            activeConnectionQuery.data?.status.message ?? 'Checking Local Claw'
          }
          onHome={openHome}
          onNewChat={startNewChat}
          onSelectRoute={setRoute}
          onSettings={() => setRoute('settings')}
          onToggle={toggleSidebar}
        />
        <main className="flex min-w-0 flex-1 flex-col bg-white">
          <TopBar
            active={activeNavItem}
            detailPanelOpen={layoutPreferences.detailPanelOpen}
            leftSidebarCollapsed={layoutPreferences.leftSidebarCollapsed}
            onNewChat={startNewChat}
            onToggleDetailPanel={toggleDetailPanel}
            onToggleSidebar={toggleSidebar}
          />
          <div className="min-h-0 flex-1 overflow-auto">
            <AppRouteOutlet
              route={route}
              selectedSessionId={selectedSessionId}
              selectedSpace={selectedSpace}
              spaces={spaces}
              onAddSpace={(space) =>
                setSpaces((current) => [...current, space])
              }
              onClearSession={clearSelectedSession}
              onOpenSession={openSession}
              onSelectSpace={setSelectedSpaceId}
            />
          </div>
        </main>
        {layoutPreferences.detailPanelOpen && (
          <aside className="hidden w-[320px] shrink-0 border-l border-black/[0.08] bg-[#f7f7f5] p-4 xl:block">
            <DetailPanel
              connection={connection}
              selectedSessionId={selectedSessionId}
              selectedSpace={selectedSpace}
            />
          </aside>
        )}
      </div>
      <Toaster richColors />
    </div>
  )
}

function Sidebar({
  activeRoute,
  collapsed,
  connection,
  statusMessage,
  onHome,
  onNewChat,
  onSelectRoute,
  onSettings,
  onToggle,
}: {
  activeRoute: AppRoute
  collapsed: boolean
  connection: DesktopClawConnection | null
  statusMessage: string
  onHome: () => void
  onNewChat: () => void
  onSelectRoute: (route: AppRoute) => void
  onSettings: () => void
  onToggle: () => void
}) {
  return (
    <aside
      className={cn(
        'flex shrink-0 flex-col border-r border-black/[0.08] bg-[#f7f7f5] transition-[width] duration-200',
        collapsed ? 'w-[72px]' : 'w-[284px]',
      )}
    >
      <div className="flex h-16 items-center gap-3 px-3">
        <button
          type="button"
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-[#171717] text-sm font-bold text-white"
          onClick={onHome}
          aria-label="Open Home"
        >
          YA
        </button>
        {!collapsed && (
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-semibold tracking-tight text-[#171717]">
              YA Desktop
            </p>
            <p className="truncate text-xs text-[#6b6b6b]">
              Native Agent Workspace
            </p>
          </div>
        )}
        {!collapsed && (
          <IconButton
            label="Collapse navigation"
            icon={PanelLeftClose}
            onClick={onToggle}
          />
        )}
      </div>
      {collapsed ? (
        <div className="px-3 pb-3">
          <IconButton
            label="Expand navigation"
            icon={PanelLeft}
            onClick={onToggle}
          />
        </div>
      ) : (
        <div className="px-3 pb-3">
          <button
            type="button"
            className="flex w-full items-center justify-center gap-2 rounded-xl border border-black/[0.08] bg-white px-3 py-2.5 text-sm font-medium text-[#171717] shadow-sm transition hover:bg-[#f2f2ef]"
            onClick={onNewChat}
          >
            <Plus className="h-4 w-4" />
            New chat
          </button>
        </div>
      )}
      <nav className="min-h-0 flex-1 space-y-1 overflow-auto px-2 py-2">
        {navItems.map((item) => (
          <SidebarNavItem
            key={item.route}
            active={activeRoute === item.route}
            collapsed={collapsed}
            item={item}
            onClick={() => onSelectRoute(item.route)}
          />
        ))}
      </nav>
      <div className="border-t border-black/[0.08] p-3">
        <RuntimeStatus
          collapsed={collapsed}
          connection={connection}
          statusMessage={statusMessage}
        />
        <SidebarNavItem
          active={activeRoute === 'settings'}
          collapsed={collapsed}
          item={{
            route: 'settings',
            label: 'Settings',
            helper: 'Preferences',
            icon: Settings,
          }}
          onClick={onSettings}
        />
      </div>
    </aside>
  )
}

function SidebarNavItem({
  active,
  collapsed,
  item,
  onClick,
}: {
  active: boolean
  collapsed: boolean
  item: { route: AppRoute; label: string; helper: string; icon: LucideIcon }
  onClick: () => void
}) {
  const Icon = item.icon
  return (
    <button
      type="button"
      aria-current={active ? 'page' : undefined}
      title={collapsed ? item.label : undefined}
      className={cn(
        'group flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left text-sm transition',
        collapsed && 'justify-center px-2',
        active
          ? 'bg-white text-[#171717] shadow-sm ring-1 ring-black/[0.08]'
          : 'text-[#5f5f5f] hover:bg-white hover:text-[#171717]',
      )}
      onClick={onClick}
    >
      <Icon className="h-4 w-4 shrink-0" />
      {!collapsed && (
        <span className="min-w-0 flex-1">
          <span className="block truncate font-medium">{item.label}</span>
          <span className="block truncate text-xs text-[#8a8a8a]">
            {item.helper}
          </span>
        </span>
      )}
    </button>
  )
}

function RuntimeStatus({
  collapsed,
  connection,
  statusMessage,
}: {
  collapsed: boolean
  connection: DesktopClawConnection | null
  statusMessage: string
}) {
  const ready = Boolean(connection)
  return (
    <div
      className={cn(
        'mb-2 rounded-xl px-3 py-2 text-xs',
        ready
          ? 'bg-emerald-50 text-emerald-800'
          : 'bg-white text-[#6b6b6b] ring-1 ring-black/[0.06]',
        collapsed && 'flex justify-center px-2',
      )}
      title={ready ? 'Local ready' : statusMessage}
    >
      <div className="flex items-center gap-2">
        <span
          className={cn(
            'h-2 w-2 rounded-full',
            ready ? 'bg-emerald-500' : 'bg-[#9a9a9a]',
          )}
        />
        {!collapsed && (
          <span className="truncate">
            {ready ? 'Local ready' : statusMessage}
          </span>
        )}
      </div>
    </div>
  )
}

function TopBar({
  active,
  detailPanelOpen,
  leftSidebarCollapsed,
  onNewChat,
  onToggleDetailPanel,
  onToggleSidebar,
}: {
  active: { label: string; helper: string; icon: LucideIcon }
  detailPanelOpen: boolean
  leftSidebarCollapsed: boolean
  onNewChat: () => void
  onToggleDetailPanel: () => void
  onToggleSidebar: () => void
}) {
  return (
    <header className="flex h-16 shrink-0 items-center justify-between border-b border-black/[0.08] bg-white px-4 lg:px-6">
      <div className="flex min-w-0 items-center gap-3">
        <IconButton
          label={
            leftSidebarCollapsed ? 'Expand navigation' : 'Collapse navigation'
          }
          icon={leftSidebarCollapsed ? PanelLeft : PanelLeftClose}
          onClick={onToggleSidebar}
        />
        <div className="min-w-0">
          <h1 className="truncate text-sm font-semibold text-[#171717]">
            {active.label}
          </h1>
          <p className="truncate text-xs text-[#8a8a8a]">{active.helper}</p>
        </div>
      </div>
      <div className="flex items-center gap-2">
        <button
          type="button"
          className={cn(
            'hidden h-10 items-center gap-2 rounded-xl border border-black/[0.08] bg-white px-3 text-sm transition hover:bg-[#f7f7f5] xl:inline-flex',
            detailPanelOpen ? 'text-[#171717]' : 'text-[#5f5f5f]',
          )}
          onClick={onToggleDetailPanel}
        >
          <PanelRight className="h-4 w-4" />
          Details
        </button>
        <button
          type="button"
          className="inline-flex h-10 items-center gap-2 rounded-xl bg-[#171717] px-3 text-sm font-medium text-white transition hover:bg-[#2f2f2f]"
          onClick={onNewChat}
        >
          <Plus className="h-4 w-4" />
          New chat
        </button>
      </div>
    </header>
  )
}

function DetailPanel({
  connection,
  selectedSessionId,
  selectedSpace,
}: {
  connection: DesktopClawConnection | null
  selectedSessionId: string | null
  selectedSpace: DesktopSpace
}) {
  const sessionQuery = useClawSession(connection, selectedSessionId)
  const session = sessionQuery.data?.session ?? null
  const latestRun = session?.latest_run ?? session?.latestRun ?? null
  const pendingCount = session ? inboxItemsFromSessions([session]).length : 0

  return (
    <div className="flex h-full flex-col gap-3">
      <div className="px-1 py-2">
        <h2 className="text-sm font-semibold text-[#171717]">Details</h2>
        <p className="mt-1 text-xs text-[#8a8a8a]">
          Runtime context when you need it.
        </p>
      </div>
      <PanelCard
        title="Active chat"
        detail={session ? sessionTitle(session) : 'No selected chat'}
        icon={MessageSquareText}
      />
      <PanelCard
        title="Latest run"
        detail={
          latestRun
            ? `Run #${latestRun.sequence_no ?? latestRun.sequenceNo ?? '—'} · ${latestRun.status}`
            : 'No run loaded'
        }
        icon={Activity}
      />
      <PanelCard
        title="Active space"
        detail={spaceDetail(selectedSpace)}
        icon={Folder}
      />
      <PanelCard
        title="Trust"
        detail={selectedSpace.trust}
        icon={ShieldCheck}
      />
      <PanelCard
        title="Inbox"
        detail={
          pendingCount
            ? `${pendingCount} pending items`
            : 'No blocking approvals'
        }
        icon={Bell}
      />
    </div>
  )
}
