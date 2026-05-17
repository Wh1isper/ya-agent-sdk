import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import {
  Activity,
  Bell,
  Bot,
  BriefcaseBusiness,
  ChevronRight,
  Command,
  FileCode2,
  Folder,
  GitBranch,
  HardDrive,
  Home,
  Inbox,
  KeyRound,
  LayoutDashboard,
  MessageSquareText,
  PanelLeft,
  PanelLeftClose,
  PanelRight,
  PanelRightClose,
  Plus,
  Search,
  Settings,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  TerminalSquare,
  type LucideIcon,
} from 'lucide-react'
import { useEffect, useState, type ReactNode } from 'react'
import { Toaster } from 'sonner'

import { cn } from '../lib'
import { RuntimeManagerPanel } from '../runtime/RuntimeManagerPanel'

type AppRoute = 'home' | 'chats' | 'board' | 'spaces' | 'inbox' | 'settings'

type DesktopLayoutPreferences = {
  leftSidebarCollapsed: boolean
  rightPanelCollapsed: boolean
}

const defaultLayoutPreferences: DesktopLayoutPreferences = {
  leftSidebarCollapsed: false,
  rightPanelCollapsed: false,
}

const layoutPreferencesStorageKey = 'ya-desktop.layout-preferences.v1'

const queryClient = new QueryClient()

const navItems: Array<{
  route: AppRoute
  label: string
  helper: string
  icon: LucideIcon
}> = [
  { route: 'home', label: 'Home', helper: 'Start and resume', icon: Home },
  { route: 'chats', label: 'Chats', helper: 'Conversations', icon: MessageSquareText },
  { route: 'board', label: 'Board', helper: 'Kanban view', icon: LayoutDashboard },
  { route: 'spaces', label: 'Spaces', helper: 'Workspace folders', icon: BriefcaseBusiness },
  { route: 'inbox', label: 'Inbox', helper: 'Approvals and alerts', icon: Inbox },
]

const conversations = [
  {
    title: 'Ship the YA Desktop shell',
    space: 'ya-mono',
    status: 'Active',
    detail: 'Tauri · React · native workspace',
    tone: 'blue',
  },
  {
    title: 'Review local sandbox trust',
    space: 'ya-mono',
    status: 'Waiting',
    detail: 'Command approval · workspace boundary',
    tone: 'amber',
  },
  {
    title: 'Design product navigation',
    space: 'ya-mono',
    status: 'Done',
    detail: 'Home · Chats · Board · Spaces · Inbox',
    tone: 'emerald',
  },
]

const spaces = [
  {
    name: 'ya-mono',
    path: '~/code/yet-another-agents/ya-mono',
    runtime: 'Local Claw',
    trust: 'Trusted',
  },
  {
    name: 'personal-notes',
    path: '~/Documents/notes',
    runtime: 'Local Claw',
    trust: 'Read-only',
  },
  {
    name: 'team-cloud',
    path: 'cloud://team/main',
    runtime: 'Cloud Claw',
    trust: 'Team',
  },
]

const rightContext = [
  { icon: HardDrive, title: 'Active space', detail: 'ya-mono · local folder' },
  { icon: ShieldCheck, title: 'Trust', detail: 'Trusted local project' },
  { icon: Bell, title: 'Inbox', detail: 'No blocking approvals' },
]

const boardColumns = [
  { title: 'Active', items: conversations.filter((item) => item.status === 'Active') },
  { title: 'Waiting', items: conversations.filter((item) => item.status === 'Waiting') },
  { title: 'Done', items: conversations.filter((item) => item.status === 'Done') },
]

export function App() {
  const [route, setRoute] = useState<AppRoute>('home')
  const [layoutPreferences, setLayoutPreferences] = useState<DesktopLayoutPreferences>(
    readLayoutPreferences,
  )
  const { leftSidebarCollapsed, rightPanelCollapsed } = layoutPreferences
  const active =
    route === 'settings'
      ? { route, label: 'Settings', helper: 'Preferences', icon: Settings }
      : (navItems.find((item) => item.route === route) ?? navItems[0])

  useEffect(() => {
    writeLayoutPreferences(layoutPreferences)
  }, [layoutPreferences])

  const toggleLeftSidebar = () => {
    setLayoutPreferences((current) => ({
      ...current,
      leftSidebarCollapsed: !current.leftSidebarCollapsed,
    }))
  }

  const toggleRightPanel = () => {
    setLayoutPreferences((current) => ({
      ...current,
      rightPanelCollapsed: !current.rightPanelCollapsed,
    }))
  }

  return (
    <QueryClientProvider client={queryClient}>
      <div className="min-h-screen bg-[#f7f7f4] text-[#171717]">
        <div className="pointer-events-none fixed inset-0 bg-[radial-gradient(circle_at_20%_0%,rgba(59,130,246,0.10),transparent_32%),radial-gradient(circle_at_80%_12%,rgba(15,23,42,0.06),transparent_28%),linear-gradient(180deg,#fbfbf8_0%,#f4f3ef_100%)]" />
        <div className="relative flex h-screen p-3">
          <aside
            className={cn(
              'flex shrink-0 flex-col rounded-[28px] border border-black/[0.06] bg-white/80 shadow-[0_24px_80px_rgba(15,23,42,0.08)] backdrop-blur-xl transition-[width] duration-300 ease-out',
              leftSidebarCollapsed ? 'w-[84px]' : 'w-[292px]',
            )}
          >
            <SidebarHeader collapsed={leftSidebarCollapsed} onToggle={toggleLeftSidebar} />
            <nav className="min-h-0 flex-1 space-y-1 overflow-auto px-3 py-2">
              {navItems.map((item) => (
                <NavItem
                  key={item.route}
                  item={item}
                  active={route === item.route}
                  collapsed={leftSidebarCollapsed}
                  onClick={() => setRoute(item.route)}
                />
              ))}
            </nav>
            <SidebarFooter
              active={route === 'settings'}
              collapsed={leftSidebarCollapsed}
              onSettings={() => setRoute('settings')}
            />
          </aside>

          <main className="ml-3 flex min-w-0 flex-1 flex-col overflow-hidden rounded-[28px] border border-black/[0.06] bg-white/65 shadow-[0_24px_80px_rgba(15,23,42,0.08)] backdrop-blur-xl">
            <TopBar
              active={active}
              leftSidebarCollapsed={leftSidebarCollapsed}
              rightPanelCollapsed={rightPanelCollapsed}
              onToggleLeftSidebar={toggleLeftSidebar}
              onToggleRightPanel={toggleRightPanel}
            />
            <div className="min-h-0 flex-1 overflow-auto px-5 py-5 lg:px-8 lg:py-7">
              {renderRoute(route)}
            </div>
          </main>

          {!rightPanelCollapsed && (
            <aside className="ml-3 hidden w-[336px] shrink-0 flex-col rounded-[28px] border border-black/[0.06] bg-white/70 p-4 shadow-[0_24px_80px_rgba(15,23,42,0.08)] backdrop-blur-xl 2xl:flex">
              <RightPanel
                onCollapse={() =>
                  setLayoutPreferences((current) => ({
                    ...current,
                    rightPanelCollapsed: true,
                  }))
                }
              />
            </aside>
          )}
        </div>
        <Toaster richColors />
      </div>
    </QueryClientProvider>
  )
}

function SidebarHeader({ collapsed, onToggle }: { collapsed: boolean; onToggle: () => void }) {
  return (
    <div className="border-b border-black/[0.06] p-4">
      <div className={cn('flex items-center gap-3', collapsed && 'justify-center')}>
        <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-[#111827] text-sm font-black tracking-tight text-white shadow-lg shadow-slate-950/15">
          YA
        </div>
        {!collapsed && (
          <div className="min-w-0 flex-1">
            <p className="font-semibold tracking-tight text-slate-950">YA Desktop</p>
            <p className="mt-0.5 text-xs text-slate-500">Native Agent Workspace</p>
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
        <button
          type="button"
          aria-label="Expand navigation"
          className="mt-4 flex h-11 w-full items-center justify-center rounded-2xl border border-black/[0.06] bg-[#f7f7f4] text-slate-500 transition hover:bg-white hover:text-slate-900 hover:shadow-sm"
          onClick={onToggle}
        >
          <PanelLeft className="h-4 w-4" />
        </button>
      ) : (
        <button className="mt-4 flex w-full items-center gap-2 rounded-2xl border border-black/[0.06] bg-[#f7f7f4] px-3 py-2.5 text-left text-sm text-slate-500 transition hover:bg-white hover:text-slate-900 hover:shadow-sm">
          <Search className="h-4 w-4" />
          <span className="min-w-0 flex-1 truncate">Search chats, spaces, runs</span>
          <span className="rounded-lg border border-black/[0.06] bg-white px-1.5 py-0.5 text-[10px] text-slate-400">
            ⌘K
          </span>
        </button>
      )}
    </div>
  )
}

function NavItem({
  item,
  active,
  collapsed,
  onClick,
}: {
  item: (typeof navItems)[number]
  active: boolean
  collapsed: boolean
  onClick: () => void
}) {
  const Icon = item.icon
  return (
    <button
      type="button"
      aria-current={active ? 'page' : undefined}
      title={collapsed ? item.label : undefined}
      className={cn(
        'group flex w-full items-center gap-3 rounded-2xl px-3 py-3 text-left transition',
        collapsed && 'justify-center px-2',
        active
          ? 'bg-[#111827] text-white shadow-lg shadow-slate-950/10'
          : 'text-slate-600 hover:bg-[#f7f7f4] hover:text-slate-950',
      )}
      onClick={onClick}
    >
      <span
        className={cn(
          'flex h-9 w-9 items-center justify-center rounded-xl transition',
          active
            ? 'bg-white/12 text-white'
            : 'bg-white text-slate-500 shadow-sm ring-1 ring-black/[0.05]',
        )}
      >
        <Icon className="h-4 w-4" />
      </span>
      {!collapsed && (
        <span className="min-w-0 flex-1">
          <span className="block truncate text-sm font-semibold">{item.label}</span>
          <span
            className={cn(
              'mt-0.5 block truncate text-xs',
              active ? 'text-slate-300' : 'text-slate-400',
            )}
          >
            {item.helper}
          </span>
        </span>
      )}
    </button>
  )
}

function SidebarFooter({
  active,
  collapsed,
  onSettings,
}: {
  active: boolean
  collapsed: boolean
  onSettings: () => void
}) {
  return (
    <div className="border-t border-black/[0.06] p-4">
      <div
        className={cn(
          'rounded-2xl border border-emerald-900/10 bg-emerald-50/80 p-3',
          collapsed && 'flex justify-center px-2 py-3',
        )}
        title={collapsed ? 'Local ready' : undefined}
      >
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-emerald-500 shadow-[0_0_0_4px_rgba(16,185,129,0.12)]" />
          {!collapsed && <p className="text-sm font-semibold text-emerald-950">Local ready</p>}
        </div>
        {!collapsed && (
          <p className="mt-1 text-xs leading-5 text-emerald-800/70">
            This Mac · ya-mono · trusted
          </p>
        )}
      </div>
      <button
        type="button"
        title={collapsed ? 'Settings' : undefined}
        className={cn(
          'mt-3 flex w-full items-center gap-3 rounded-2xl px-3 py-3 text-left transition',
          collapsed && 'justify-center px-2',
          active
            ? 'bg-[#111827] text-white shadow-lg shadow-slate-950/10'
            : 'text-slate-600 hover:bg-[#f7f7f4] hover:text-slate-950',
        )}
        onClick={onSettings}
      >
        <span
          className={cn(
            'flex h-9 w-9 items-center justify-center rounded-xl',
            active
              ? 'bg-white/12 text-white'
              : 'bg-white text-slate-500 shadow-sm ring-1 ring-black/[0.05]',
          )}
        >
          <Settings className="h-4 w-4" />
        </span>
        {!collapsed && (
          <span className="min-w-0 flex-1">
            <span className="block text-sm font-semibold">Settings</span>
            <span
              className={cn(
                'mt-0.5 block text-xs',
                active ? 'text-slate-300' : 'text-slate-400',
              )}
            >
              Preferences
            </span>
          </span>
        )}
      </button>
    </div>
  )
}

function TopBar({
  active,
  leftSidebarCollapsed,
  rightPanelCollapsed,
  onToggleLeftSidebar,
  onToggleRightPanel,
}: {
  active: { label: string; icon: LucideIcon }
  leftSidebarCollapsed: boolean
  rightPanelCollapsed: boolean
  onToggleLeftSidebar: () => void
  onToggleRightPanel: () => void
}) {
  const Icon = active.icon
  return (
    <header className="flex h-20 shrink-0 items-center justify-between border-b border-black/[0.06] px-5 lg:px-8">
      <div className="flex items-center gap-3">
        <IconButton
          label={leftSidebarCollapsed ? 'Expand navigation' : 'Collapse navigation'}
          icon={leftSidebarCollapsed ? PanelLeft : PanelLeftClose}
          onClick={onToggleLeftSidebar}
        />
        <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-white text-slate-800 shadow-sm ring-1 ring-black/[0.06]">
          <Icon className="h-5 w-5" />
        </div>
        <div>
          <p className="text-xs font-medium uppercase tracking-[0.18em] text-slate-400">Desktop</p>
          <h1 className="text-lg font-semibold text-slate-950">{active.label}</h1>
        </div>
      </div>
      <div className="flex items-center gap-2">
        <button
          type="button"
          className="hidden items-center gap-2 rounded-2xl border border-black/[0.06] bg-white px-3 py-2 text-sm font-medium text-slate-600 shadow-sm transition hover:text-slate-950 md:inline-flex"
          onClick={onToggleRightPanel}
        >
          {rightPanelCollapsed ? (
            <PanelRight className="h-4 w-4" />
          ) : (
            <PanelRightClose className="h-4 w-4" />
          )}
          {rightPanelCollapsed ? 'Show context' : 'Hide context'}
        </button>
        <button className="inline-flex items-center gap-2 rounded-2xl bg-[#111827] px-4 py-2.5 text-sm font-semibold text-white shadow-lg shadow-slate-950/15 transition hover:bg-slate-800">
          <Plus className="h-4 w-4" />
          New chat
        </button>
      </div>
    </header>
  )
}

function renderRoute(route: AppRoute) {
  switch (route) {
    case 'home':
      return <HomePage />
    case 'chats':
      return <ChatsPage />
    case 'board':
      return <BoardPage />
    case 'spaces':
      return <SpacesPage />
    case 'inbox':
      return <InboxPage />
    case 'settings':
      return <SettingsPage />
  }
}

function HomePage() {
  return (
    <div className="mx-auto max-w-5xl space-y-6 py-3">
      <section className="rounded-[2rem] border border-black/[0.06] bg-white p-7 text-center shadow-sm">
        <div className="mx-auto inline-flex items-center gap-2 rounded-full border border-black/[0.06] bg-[#fbfbf8] px-3 py-1.5 text-xs font-semibold text-slate-600 shadow-sm">
          <Sparkles className="h-3.5 w-3.5 text-blue-500" />
          Home
        </div>
        <h2 className="mx-auto mt-6 max-w-3xl text-5xl font-semibold tracking-[-0.04em] text-slate-950 md:text-6xl">
          What should YA do next?
        </h2>
        <p className="mx-auto mt-4 max-w-2xl text-base leading-7 text-slate-500">
          Start a new conversation from selected text, clipboard, screenshots, active app context, or the current space.
        </p>
        <div className="mx-auto mt-8 max-w-3xl rounded-[2rem] border border-black/[0.06] bg-white p-3 shadow-[0_24px_80px_rgba(15,23,42,0.10)]">
          <div className="flex items-center gap-3 rounded-[1.35rem] bg-[#f7f7f4] px-4 py-4 ring-1 ring-black/[0.04]">
            <Command className="h-5 w-5 text-slate-400" />
            <input
              className="min-w-0 flex-1 bg-transparent text-lg text-slate-950 outline-none placeholder:text-slate-400"
              placeholder="Ask YA to ship, debug, explain, refactor, summarize..."
            />
            <button className="rounded-2xl bg-[#111827] px-4 py-2 text-sm font-semibold text-white shadow-lg shadow-slate-950/15">
              Start
            </button>
          </div>
          <div className="mt-3 grid gap-3 md:grid-cols-3">
            <ContextPill icon={FileCode2} title="Selection" detail="No text captured" />
            <ContextPill icon={Folder} title="Space" detail="ya-mono" />
            <ContextPill icon={TerminalSquare} title="Runtime" detail="Local Claw" />
          </div>
        </div>
      </section>

      <section className="grid gap-5 xl:grid-cols-[1fr_0.8fr]">
        <Card title="Recent chats" action="Open Chats">
          <div className="space-y-3">
            {conversations.slice(0, 2).map((conversation) => (
              <ConversationRow key={conversation.title} {...conversation} />
            ))}
          </div>
        </Card>
        <Card title="Current space" action="Open Spaces">
          <div className="grid gap-3">
            <HeroMetric label="Folder" value="ya-mono" />
            <HeroMetric label="Runtime" value="Local Claw" />
          </div>
        </Card>
      </section>
    </div>
  )
}

function ChatsPage() {
  return (
    <div className="grid min-h-full gap-5 xl:grid-cols-[360px_1fr]">
      <section className="rounded-[2rem] border border-black/[0.06] bg-white p-5 shadow-sm">
        <SectionHeader title="Chats" action="New" />
        <div className="mt-4 space-y-3">
          {conversations.map((conversation) => (
            <ConversationRow key={conversation.title} {...conversation} compact />
          ))}
        </div>
      </section>
      <section className="flex min-h-[620px] flex-col rounded-[2rem] border border-black/[0.06] bg-white shadow-sm">
        <div className="border-b border-black/[0.06] p-5">
          <p className="text-sm font-semibold text-blue-600">Conversation</p>
          <h2 className="mt-1 text-2xl font-semibold tracking-[-0.025em] text-slate-950">
            Ship the YA Desktop shell
          </h2>
        </div>
        <div className="grid flex-1 gap-5 p-5 lg:grid-cols-[1fr_300px]">
          <div className="flex items-center justify-center rounded-[1.6rem] bg-[#fbfbf8] p-8 text-center ring-1 ring-black/[0.04]">
            <div>
              <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-3xl bg-[#111827] text-white shadow-lg shadow-slate-950/15">
                <Bot className="h-7 w-7" />
              </div>
              <h3 className="mt-5 text-xl font-semibold text-slate-950">Conversation surface</h3>
              <p className="mx-auto mt-2 max-w-md text-sm leading-6 text-slate-500">
                Messages, AGUI replay, run timeline, shell output, diffs and artifacts live inside the selected chat.
              </p>
            </div>
          </div>
          <div className="space-y-3">
            <ActionRow icon={Activity} title="Run timeline" detail="No live run" />
            <ActionRow icon={TerminalSquare} title="Shell output" detail="No command running" />
            <ActionRow icon={GitBranch} title="File diff" detail="No diff loaded" />
          </div>
        </div>
      </section>
    </div>
  )
}

function BoardPage() {
  return (
    <div className="grid min-h-full gap-5 xl:grid-cols-3">
      {boardColumns.map((column) => (
        <section
          key={column.title}
          className="rounded-[2rem] border border-black/[0.06] bg-white p-5 shadow-sm"
        >
          <SectionHeader title={column.title} action={`${column.items.length}`} />
          <div className="mt-4 space-y-3">
            {column.items.map((conversation) => (
              <ConversationCard key={conversation.title} {...conversation} />
            ))}
          </div>
        </section>
      ))}
    </div>
  )
}

function InboxPage() {
  return (
    <PanelPage
      eyebrow="Inbox"
      title="Decisions that need a human"
      body="Approvals, alerts, failed background runs, bridge events, and scheduled work that need attention land here."
      cards={[
        ['Command approvals', 'Approve shell commands with preview and risk context', TerminalSquare],
        ['File diffs', 'Review file changes before write operations continue', GitBranch],
        ['Workspace trust', 'Confirm execution location and filesystem access', ShieldCheck],
      ]}
    />
  )
}

function SpacesPage() {
  return (
    <section className="rounded-[2rem] border border-black/[0.06] bg-white p-7 shadow-sm">
      <p className="text-sm font-semibold text-blue-600">Spaces</p>
      <h2 className="mt-2 max-w-3xl text-4xl font-semibold tracking-[-0.035em] text-slate-950">
        Workspace folders and runtime locations
      </h2>
      <p className="mt-4 max-w-3xl text-sm leading-6 text-slate-500">
        A Space combines a folder or cloud workspace, runtime connection, trust level, execution location, and default profile.
      </p>
      <div className="mt-8 grid gap-4 xl:grid-cols-3">
        {spaces.map((space) => (
          <SpaceCard key={space.name} {...space} />
        ))}
      </div>
    </section>
  )
}

function SettingsPage() {
  return (
    <div className="space-y-5">
      <PanelPage
        eyebrow="Settings"
        title="Desktop preferences and advanced runtime"
        body="Hotkeys, notifications, theme, voice, tokens, autostart, diagnostics and advanced runtime controls live here."
        cards={[
          ['Preferences', 'Hotkeys, notifications, voice and appearance', Settings],
          ['Secrets', 'Keychain-backed token storage', KeyRound],
          ['Advanced Runtime', 'Profiles, schedules, bridges, heartbeat, logs', SlidersHorizontal],
        ]}
      />
      <RuntimeManagerPanel />
    </div>
  )
}

function PanelPage({
  eyebrow,
  title,
  body,
  cards,
}: {
  eyebrow: string
  title: string
  body: string
  cards: Array<[string, string, LucideIcon]>
}) {
  return (
    <section className="rounded-[2rem] border border-black/[0.06] bg-white p-7 shadow-sm">
      <p className="text-sm font-semibold text-blue-600">{eyebrow}</p>
      <h2 className="mt-2 max-w-3xl text-4xl font-semibold tracking-[-0.035em] text-slate-950">{title}</h2>
      <p className="mt-4 max-w-3xl text-sm leading-6 text-slate-500">{body}</p>
      <div className="mt-8 grid gap-4 md:grid-cols-3">
        {cards.map(([cardTitle, detail, Icon]) => (
          <div key={cardTitle} className="rounded-3xl border border-black/[0.06] bg-[#f7f7f4] p-5">
            <Icon className="h-5 w-5 text-slate-700" />
            <h3 className="mt-4 text-sm font-semibold text-slate-950">{cardTitle}</h3>
            <p className="mt-2 text-xs leading-5 text-slate-500">{detail}</p>
          </div>
        ))}
      </div>
    </section>
  )
}

function RightPanel({ onCollapse }: { onCollapse: () => void }) {
  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between gap-3">
        <SectionHeader title="Live context" action="Refresh" />
        <IconButton label="Hide context" icon={PanelRightClose} onClick={onCollapse} />
      </div>
      <div className="mt-4 rounded-3xl border border-black/[0.06] bg-white p-4 shadow-sm">
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">Active chat</p>
        <h3 className="mt-2 text-lg font-semibold text-slate-950">No live run</h3>
        <p className="mt-2 text-sm leading-6 text-slate-500">Global SSE notifications will hydrate this panel.</p>
      </div>
      <div className="mt-5 space-y-3">
        {rightContext.map((row) => (
          <ActionRow key={row.title} {...row} />
        ))}
      </div>
    </div>
  )
}

function Card({ title, action, children }: { title: string; action: string; children: ReactNode }) {
  return (
    <section className="rounded-[2rem] border border-black/[0.06] bg-white p-6 shadow-sm">
      <SectionHeader title={title} action={action} />
      <div className="mt-4">{children}</div>
    </section>
  )
}

function ContextPill({ icon: Icon, title, detail }: { icon: LucideIcon; title: string; detail: string }) {
  return (
    <div className="rounded-2xl border border-black/[0.06] bg-[#fbfbf8] p-4 text-left">
      <Icon className="h-4 w-4 text-slate-500" />
      <p className="mt-2 text-sm font-semibold text-slate-950">{title}</p>
      <p className="mt-1 text-xs text-slate-500">{detail}</p>
    </div>
  )
}

function HeroMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-3xl border border-black/[0.06] bg-[#fbfbf8] p-4">
      <p className="text-xs text-slate-500">{label}</p>
      <p className="mt-2 text-lg font-semibold text-slate-950">{value}</p>
    </div>
  )
}

function SectionHeader({ title, action }: { title: string; action: string }) {
  return (
    <div className="flex flex-1 items-center justify-between gap-3">
      <h2 className="text-sm font-semibold text-slate-950">{title}</h2>
      <button className="text-xs font-semibold text-blue-600 hover:text-blue-700">{action}</button>
    </div>
  )
}

function IconButton({
  label,
  icon: Icon,
  onClick,
}: {
  label: string
  icon: LucideIcon
  onClick: () => void
}) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl border border-black/[0.06] bg-white text-slate-500 shadow-sm transition hover:bg-[#f7f7f4] hover:text-slate-950"
      onClick={onClick}
    >
      <Icon className="h-4 w-4" />
    </button>
  )
}

function ActionRow({ icon: Icon, title, detail }: { icon: LucideIcon; title: string; detail: string }) {
  return (
    <button className="flex w-full items-center gap-3 rounded-2xl border border-black/[0.06] bg-white p-3 text-left shadow-sm transition hover:bg-[#fbfbf8]">
      <span className="flex h-10 w-10 items-center justify-center rounded-2xl bg-[#f7f7f4] text-slate-600 ring-1 ring-black/[0.04]">
        <Icon className="h-4 w-4" />
      </span>
      <span className="min-w-0 flex-1">
        <span className="block truncate text-sm font-semibold text-slate-950">{title}</span>
        <span className="mt-0.5 block truncate text-xs text-slate-500">{detail}</span>
      </span>
      <ChevronRight className="h-4 w-4 text-slate-300" />
    </button>
  )
}

function ConversationRow({
  title,
  detail,
  status,
  tone,
  space,
  compact,
}: {
  title: string
  detail: string
  status: string
  tone: string
  space: string
  compact?: boolean
}) {
  return (
    <button className={cn('flex w-full items-center gap-4 rounded-3xl border border-black/[0.06] bg-white text-left shadow-sm transition hover:bg-[#fbfbf8]', compact ? 'p-3' : 'p-4')}>
      <span className={cn('h-3 w-3 rounded-full', statusTone(tone))} />
      <span className="min-w-0 flex-1">
        <span className="block truncate text-sm font-semibold text-slate-950">{title}</span>
        <span className="mt-1 block truncate text-xs text-slate-500">{space} · {detail}</span>
      </span>
      <span className="rounded-full border border-black/[0.06] bg-[#f7f7f4] px-3 py-1 text-xs font-semibold text-slate-600">
        {status}
      </span>
    </button>
  )
}

function ConversationCard({
  title,
  detail,
  status,
  tone,
  space,
}: {
  title: string
  detail: string
  status: string
  tone: string
  space: string
}) {
  return (
    <button className="w-full rounded-3xl border border-black/[0.06] bg-[#fbfbf8] p-4 text-left shadow-sm transition hover:bg-white">
      <div className="flex items-center justify-between gap-3">
        <span className={cn('h-3 w-3 rounded-full', statusTone(tone))} />
        <span className="rounded-full border border-black/[0.06] bg-white px-2.5 py-1 text-[11px] font-semibold text-slate-600">
          {status}
        </span>
      </div>
      <h3 className="mt-4 text-sm font-semibold leading-5 text-slate-950">{title}</h3>
      <p className="mt-2 text-xs leading-5 text-slate-500">{space} · {detail}</p>
    </button>
  )
}

function SpaceCard({
  name,
  path,
  runtime,
  trust,
}: {
  name: string
  path: string
  runtime: string
  trust: string
}) {
  return (
    <button className="rounded-3xl border border-black/[0.06] bg-[#f7f7f4] p-5 text-left transition hover:bg-white hover:shadow-sm">
      <Folder className="h-6 w-6 text-slate-700" />
      <h3 className="mt-4 text-sm font-semibold text-slate-950">{name}</h3>
      <p className="mt-2 truncate text-xs text-slate-500">{path}</p>
      <div className="mt-4 flex flex-wrap gap-2">
        <span className="rounded-full border border-black/[0.06] bg-white px-2.5 py-1 text-[11px] font-semibold text-slate-600">
          {runtime}
        </span>
        <span className="rounded-full border border-black/[0.06] bg-white px-2.5 py-1 text-[11px] font-semibold text-slate-600">
          {trust}
        </span>
      </div>
    </button>
  )
}

function readLayoutPreferences(): DesktopLayoutPreferences {
  if (typeof window === 'undefined') return defaultLayoutPreferences

  try {
    const rawValue = window.localStorage.getItem(layoutPreferencesStorageKey)
    if (!rawValue) return defaultLayoutPreferences

    const parsedValue = JSON.parse(rawValue) as Partial<DesktopLayoutPreferences>
    return {
      leftSidebarCollapsed:
        typeof parsedValue.leftSidebarCollapsed === 'boolean'
          ? parsedValue.leftSidebarCollapsed
          : defaultLayoutPreferences.leftSidebarCollapsed,
      rightPanelCollapsed:
        typeof parsedValue.rightPanelCollapsed === 'boolean'
          ? parsedValue.rightPanelCollapsed
          : defaultLayoutPreferences.rightPanelCollapsed,
    }
  } catch {
    return defaultLayoutPreferences
  }
}

function writeLayoutPreferences(preferences: DesktopLayoutPreferences) {
  if (typeof window === 'undefined') return

  try {
    window.localStorage.setItem(layoutPreferencesStorageKey, JSON.stringify(preferences))
  } catch {
    // Keep the prototype usable in restricted storage contexts.
  }
}

function statusTone(tone: string) {
  if (tone === 'blue') return 'bg-blue-500 shadow-[0_0_0_4px_rgba(59,130,246,.10)]'
  if (tone === 'amber') return 'bg-amber-500 shadow-[0_0_0_4px_rgba(245,158,11,.12)]'
  return 'bg-emerald-500 shadow-[0_0_0_4px_rgba(16,185,129,.12)]'
}
