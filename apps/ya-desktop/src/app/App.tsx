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
import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type ReactNode,
} from 'react'
import { Toaster } from 'sonner'

import {
  collectTextFromReplay,
  isRunErrorEvent,
  isRunFinishedEvent,
  streamErrorMessage,
  streamRunId,
  streamSessionId,
  streamTextDelta,
  useActiveClawConnection,
  useCancelClawSession,
  useClawHealth,
  useClawInfo,
  useClawNotifications,
  useClawProfiles,
  useClawRunTraces,
  useClawSession,
  useClawSessions,
  useClawSessionTurns,
  useCreateClawSessionRunStream,
  useCreateClawSessionStream,
  type ClawRunStatus,
  type ClawRunSummary,
  type ClawProfileSummary,
  type ClawRunTraceResponse,
  type ClawSessionDetail,
  type ClawSessionStatus,
  type ClawSessionSummary,
  type ClawSessionTurn,
  type ClawStreamEvent,
  type ClawWorkspaceBinding,
  type JsonObject,
} from '../claw'
import { cn } from '../lib'
import { RuntimeManagerPanel } from '../runtime/RuntimeManagerPanel'

type AppRoute = 'home' | 'chats' | 'board' | 'spaces' | 'inbox' | 'settings'

type DesktopLayoutPreferences = {
  leftSidebarCollapsed: boolean
  rightPanelCollapsed: boolean
}

type HomeStreamStatus = 'idle' | 'connecting' | 'streaming' | 'completed' | 'failed'

type DesktopSpace = {
  id: string
  name: string
  path: string
  runtime: string
  trust: string
  default: boolean
}

const defaultSpaceId = 'local-workspace'

const defaultDesktopSpaces: DesktopSpace[] = [
  {
    id: defaultSpaceId,
    name: 'Local workspace',
    path: '',
    runtime: 'Local Claw',
    trust: 'Trusted',
    default: true,
  },
]

const spacesStorageKey = 'ya-desktop.spaces.v1'

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
  {
    route: 'chats',
    label: 'Chats',
    helper: 'Conversations',
    icon: MessageSquareText,
  },
  {
    route: 'board',
    label: 'Board',
    helper: 'Kanban view',
    icon: LayoutDashboard,
  },
  {
    route: 'spaces',
    label: 'Spaces',
    helper: 'Workspace folders',
    icon: BriefcaseBusiness,
  },
  {
    route: 'inbox',
    label: 'Inbox',
    helper: 'Approvals and alerts',
    icon: Inbox,
  },
]

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <DesktopShell />
    </QueryClientProvider>
  )
}

function DesktopShell() {
  const [route, setRoute] = useState<AppRoute>('home')
  const [layoutPreferences, setLayoutPreferences] =
    useState<DesktopLayoutPreferences>(readLayoutPreferences)
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null)
  const [spaces, setSpaces] = useState<DesktopSpace[]>(readSpaces)
  const [selectedSpaceId, setSelectedSpaceId] = useState(defaultSpaceId)
  const activeConnectionQuery = useActiveClawConnection()
  const shellConnection = activeConnectionQuery.data?.connection ?? null
  useClawNotifications(shellConnection)
  const selectedSpace =
    spaces.find((space) => space.id === selectedSpaceId) ?? spaces[0]
  const { leftSidebarCollapsed, rightPanelCollapsed } = layoutPreferences
  const openSession = (sessionId: string | null) => {
    setSelectedSessionId(sessionId)
    if (sessionId) setRoute('chats')
  }
  const active =
    route === 'settings'
      ? { route, label: 'Settings', helper: 'Preferences', icon: Settings }
      : (navItems.find((item) => item.route === route) ?? navItems[0])

  useEffect(() => {
    writeLayoutPreferences(layoutPreferences)
  }, [layoutPreferences])

  useEffect(() => {
    writeSpaces(spaces)
  }, [spaces])

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
    <div className="min-h-screen bg-[#f7f7f4] text-[#171717]">
      <div className="pointer-events-none fixed inset-0 bg-[radial-gradient(circle_at_20%_0%,rgba(59,130,246,0.10),transparent_32%),radial-gradient(circle_at_80%_12%,rgba(15,23,42,0.06),transparent_28%),linear-gradient(180deg,#fbfbf8_0%,#f4f3ef_100%)]" />
      <div className="relative flex h-screen p-3">
        <aside
          className={cn(
            'flex shrink-0 flex-col rounded-[28px] border border-black/[0.06] bg-white/80 shadow-[0_24px_80px_rgba(15,23,42,0.08)] backdrop-blur-xl transition-[width] duration-300 ease-out',
            leftSidebarCollapsed ? 'w-[84px]' : 'w-[292px]',
          )}
        >
          <SidebarHeader
            collapsed={leftSidebarCollapsed}
            onToggle={toggleLeftSidebar}
          />
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
            connectionReady={Boolean(shellConnection)}
            statusMessage={
              activeConnectionQuery.data?.status.message ??
              'Checking Local Claw'
            }
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
            {renderRoute(route, {
              selectedSessionId,
              selectedSpace,
              spaces,
              onAddSpace: (space) => setSpaces((current) => [...current, space]),
              onSelectSession: openSession,
              onSelectSpace: setSelectedSpaceId,
            })}
          </div>
        </main>

        {!rightPanelCollapsed && (
          <aside className="ml-3 hidden w-[336px] shrink-0 flex-col rounded-[28px] border border-black/[0.06] bg-white/70 p-4 shadow-[0_24px_80px_rgba(15,23,42,0.08)] backdrop-blur-xl 2xl:flex">
            <RightPanel
              connection={shellConnection}
              selectedSessionId={selectedSessionId}
              selectedSpace={selectedSpace}
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
  )
}

function SidebarHeader({
  collapsed,
  onToggle,
}: {
  collapsed: boolean
  onToggle: () => void
}) {
  return (
    <div className="border-b border-black/[0.06] p-4">
      <div
        className={cn('flex items-center gap-3', collapsed && 'justify-center')}
      >
        <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-[#111827] text-sm font-black tracking-tight text-white shadow-lg shadow-slate-950/15">
          YA
        </div>
        {!collapsed && (
          <div className="min-w-0 flex-1">
            <p className="font-semibold tracking-tight text-slate-950">
              YA Desktop
            </p>
            <p className="mt-0.5 text-xs text-slate-500">
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
          <span className="min-w-0 flex-1 truncate">
            Search chats, spaces, runs
          </span>
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
          <span className="block truncate text-sm font-semibold">
            {item.label}
          </span>
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
  connectionReady,
  statusMessage,
  onSettings,
}: {
  active: boolean
  collapsed: boolean
  connectionReady: boolean
  statusMessage: string
  onSettings: () => void
}) {
  const statusTitle = connectionReady ? 'Local ready' : 'Local offline'
  return (
    <div className="border-t border-black/[0.06] p-4">
      <div
        className={cn(
          connectionReady
            ? 'rounded-2xl border border-emerald-900/10 bg-emerald-50/80 p-3'
            : 'rounded-2xl border border-slate-900/10 bg-slate-50/80 p-3',
          collapsed && 'flex justify-center px-2 py-3',
        )}
        title={collapsed ? statusTitle : undefined}
      >
        <div className="flex items-center gap-2">
          <span
            className={cn(
              'h-2 w-2 rounded-full',
              connectionReady
                ? 'bg-emerald-500 shadow-[0_0_0_4px_rgba(16,185,129,0.12)]'
                : 'bg-slate-400 shadow-[0_0_0_4px_rgba(100,116,139,0.10)]',
            )}
          />
          {!collapsed && (
            <p
              className={cn(
                'text-sm font-semibold',
                connectionReady ? 'text-emerald-950' : 'text-slate-700',
              )}
            >
              {statusTitle}
            </p>
          )}
        </div>
        {!collapsed && (
          <p
            className={cn(
              'mt-1 text-xs leading-5',
              connectionReady ? 'text-emerald-800/70' : 'text-slate-500',
            )}
          >
            {connectionReady ? 'This computer · active runtime' : statusMessage}
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
          label={
            leftSidebarCollapsed ? 'Expand navigation' : 'Collapse navigation'
          }
          icon={leftSidebarCollapsed ? PanelLeft : PanelLeftClose}
          onClick={onToggleLeftSidebar}
        />
        <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-white text-slate-800 shadow-sm ring-1 ring-black/[0.06]">
          <Icon className="h-5 w-5" />
        </div>
        <div>
          <p className="text-xs font-medium uppercase tracking-[0.18em] text-slate-400">
            Desktop
          </p>
          <h1 className="text-lg font-semibold text-slate-950">
            {active.label}
          </h1>
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

function renderRoute(
  route: AppRoute,
  props: {
    selectedSessionId: string | null
    selectedSpace: DesktopSpace
    spaces: DesktopSpace[]
    onAddSpace: (space: DesktopSpace) => void
    onSelectSession: (sessionId: string | null) => void
    onSelectSpace: (spaceId: string) => void
  },
) {
  switch (route) {
    case 'home':
      return <HomePage selectedSpace={props.selectedSpace} />
    case 'chats':
      return (
        <ChatsPage
          selectedSessionId={props.selectedSessionId}
          selectedSpace={props.selectedSpace}
          onSelectSession={props.onSelectSession}
        />
      )
    case 'board':
      return <BoardPage onSelectSession={props.onSelectSession} />
    case 'spaces':
      return (
        <SpacesPage
          selectedSpaceId={props.selectedSpace.id}
          spaces={props.spaces}
          onAddSpace={props.onAddSpace}
          onSelectSpace={props.onSelectSpace}
        />
      )
    case 'inbox':
      return <InboxPage onSelectSession={props.onSelectSession} />
    case 'settings':
      return <SettingsPage />
  }
}

function HomePage({ selectedSpace }: { selectedSpace: DesktopSpace }) {
  const activeConnectionQuery = useActiveClawConnection()
  const connection = activeConnectionQuery.data?.connection ?? null
  const healthQuery = useClawHealth(connection)
  const infoQuery = useClawInfo(connection)
  const profilesQuery = useClawProfiles(connection)
  const sessionsQuery = useClawSessions(connection)
  const createSessionStream = useCreateClawSessionStream(connection)
  const abortControllerRef = useRef<AbortController | null>(null)
  const [prompt, setPrompt] = useState('')
  const [streamStatus, setStreamStatus] = useState<HomeStreamStatus>('idle')
  const [streamOutput, setStreamOutput] = useState('')
  const [streamError, setStreamError] = useState<string | null>(null)
  const [streamEventCount, setStreamEventCount] = useState(0)
  const [lastRunLabel, setLastRunLabel] = useState<string | null>(null)
  const [selectedProfileName, setSelectedProfileName] = useState('default')
  const profiles = useMemo(
    () => enabledProfiles(profilesQuery.data ?? []),
    [profilesQuery.data],
  )
  const effectiveProfileName = profileNameOrDefault(
    selectedProfileName,
    profiles,
  )
  const selectedWorkspace = workspaceBindingFromSpace(selectedSpace)
  const recentSessions = sessionsQuery.data?.slice(0, 3) ?? []
  const runtimeDetail = connection
    ? `${infoQuery.data?.serviceVersion ?? infoQuery.data?.version ?? 'Claw'} · ${healthQuery.data?.status ?? 'checking'}`
    : (activeConnectionQuery.data?.status.message ?? 'Local Claw is stopped')
  const trimmedPrompt = prompt.trim()
  const streamingActive =
    streamStatus === 'connecting' || streamStatus === 'streaming'
  const canStart = Boolean(connection && trimmedPrompt && !streamingActive)

  useEffect(() => {
    return () => abortControllerRef.current?.abort()
  }, [])

  async function handleStart(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!connection || !trimmedPrompt || streamingActive) return

    const abortController = new AbortController()
    abortControllerRef.current?.abort()
    abortControllerRef.current = abortController
    setStreamStatus('connecting')
    setStreamOutput('')
    setStreamError(null)
    setStreamEventCount(0)
    setLastRunLabel(null)

    try {
      await createSessionStream.mutateAsync({
        input: {
          profile_name: effectiveProfileName,
          workspace: selectedWorkspace,
          metadata: {
            title: trimmedPrompt.slice(0, 120),
            desktop: {
              source: 'home_command',
              space_id: selectedSpace.id,
              space_name: selectedSpace.name,
            },
          },
          input_parts: [{ type: 'text', text: trimmedPrompt }],
        },
        signal: abortController.signal,
        handlers: {
          onOpen: () => setStreamStatus('streaming'),
          onEvent: handleStreamEvent,
          onClose: () => {
            setStreamStatus((status) =>
              status === 'failed' ? status : 'completed',
            )
          },
        },
      })
      setPrompt('')
      setStreamStatus((status) => (status === 'failed' ? status : 'completed'))
    } catch (error) {
      if (abortController.signal.aborted) {
        if (abortControllerRef.current === abortController) setStreamStatus('idle')
        return
      }
      setStreamStatus('failed')
      setStreamError(error instanceof Error ? error.message : String(error))
    } finally {
      if (abortControllerRef.current === abortController) {
        abortControllerRef.current = null
      }
    }
  }

  function handleStreamEvent(event: ClawStreamEvent) {
    setStreamEventCount((count) => count + 1)
    const runId = streamRunId(event)
    if (runId) setLastRunLabel(runId.slice(0, 8))

    const delta = streamTextDelta(event)
    if (delta) setStreamOutput((output) => `${output}${delta}`)

    if (isRunErrorEvent(event)) {
      setStreamStatus('failed')
      setStreamError(streamErrorMessage(event))
      return
    }

    if (isRunFinishedEvent(event)) {
      setStreamStatus((status) => (status === 'failed' ? status : 'completed'))
      return
    }

    setStreamStatus((status) =>
      status === 'failed' || status === 'completed' ? status : 'streaming',
    )
  }

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
          Start a new conversation from selected text, clipboard, screenshots,
          active app context, or the current space.
        </p>
        <form
          className="mx-auto mt-8 max-w-3xl rounded-[2rem] border border-black/[0.06] bg-white p-3 shadow-[0_24px_80px_rgba(15,23,42,0.10)]"
          onSubmit={handleStart}
        >
          <div className="flex items-center gap-3 rounded-[1.35rem] bg-[#f7f7f4] px-4 py-4 ring-1 ring-black/[0.04]">
            <Command className="h-5 w-5 text-slate-400" />
            <input
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              className="min-w-0 flex-1 bg-transparent text-lg text-slate-950 outline-none placeholder:text-slate-400"
              placeholder="Ask YA to ship, debug, explain, refactor, summarize..."
            />
            <button
              className="rounded-2xl bg-[#111827] px-4 py-2 text-sm font-semibold text-white shadow-lg shadow-slate-950/15 transition disabled:cursor-not-allowed disabled:bg-slate-300 disabled:shadow-none"
              disabled={!canStart}
              type="submit"
            >
              {streamingActive ? 'Running' : 'Start'}
            </button>
          </div>
          <HomeStreamPreview
            eventCount={streamEventCount}
            error={streamError}
            output={streamOutput}
            runLabel={lastRunLabel}
            status={streamStatus}
          />
          <div className="mt-3 grid gap-3 md:grid-cols-2">
            <label className="rounded-2xl border border-black/[0.06] bg-[#fbfbf8] p-4 text-left">
              <span className="text-xs font-semibold text-slate-500">
                Profile
              </span>
              <select
                className="mt-2 w-full bg-transparent text-sm font-semibold text-slate-950 outline-none"
                value={effectiveProfileName}
                onChange={(event) => setSelectedProfileName(event.target.value)}
              >
                {profiles.length === 0 ? (
                  <option value="default">default</option>
                ) : (
                  profiles.map((profile) => (
                    <option key={profile.name} value={profile.name}>
                      {profile.name} · {profile.model}
                    </option>
                  ))
                )}
              </select>
            </label>
            <ContextPill
              icon={Folder}
              title="Space"
              detail={spaceDetail(selectedSpace)}
            />
          </div>
          <div className="mt-3 grid gap-3 md:grid-cols-3">
            <ContextPill
              icon={FileCode2}
              title="Selection"
              detail="No text captured"
            />
            <ContextPill
              icon={Folder}
              title="Workspace"
              detail={selectedWorkspace ? selectedWorkspace.cwd : 'Local workspace'}
            />
            <ContextPill
              icon={TerminalSquare}
              title="Runtime"
              detail={runtimeDetail}
            />
          </div>
        </form>
      </section>

      <section className="grid gap-5 xl:grid-cols-[1fr_0.8fr]">
        <Card title="Recent chats" action="Open Chats">
          <LiveSessionList
            connectionReady={Boolean(connection)}
            loading={sessionsQuery.isLoading}
            error={sessionsQuery.error}
            sessions={recentSessions}
            emptyTitle="No chats yet"
            emptyDetail="Start a conversation after Local Claw is running."
          />
        </Card>
        <Card title="Current runtime" action="Open Settings">
          <div className="grid gap-3">
            <HeroMetric
              label="Connection"
              value={connection?.name ?? 'Local Claw stopped'}
            />
            <HeroMetric
              label="Health"
              value={
                healthQuery.data?.status ??
                (connection ? 'Checking' : 'Offline')
              }
            />
          </div>
        </Card>
      </section>
    </div>
  )
}

function ChatsPage({
  selectedSessionId,
  selectedSpace,
  onSelectSession,
}: {
  selectedSessionId: string | null
  selectedSpace: DesktopSpace
  onSelectSession: (sessionId: string | null) => void
}) {
  const activeConnectionQuery = useActiveClawConnection()
  const connection = activeConnectionQuery.data?.connection ?? null
  const sessionsQuery = useClawSessions(connection)
  const createRunStream = useCreateClawSessionRunStream(connection)
  const cancelSession = useCancelClawSession(connection)
  const abortControllerRef = useRef<AbortController | null>(null)
  const [prompt, setPrompt] = useState('')
  const [liveStreamSessionId, setLiveStreamSessionId] = useState<string | null>(null)
  const [streamStatus, setStreamStatus] = useState<HomeStreamStatus>('idle')
  const [streamOutput, setStreamOutput] = useState('')
  const [streamError, setStreamError] = useState<string | null>(null)
  const [streamEventCount, setStreamEventCount] = useState(0)
  const [lastRunLabel, setLastRunLabel] = useState<string | null>(null)
  const sessions = useMemo(() => sessionsQuery.data ?? [], [sessionsQuery.data])
  const selectedSessionExists = selectedSessionId
    ? sessions.some((session) => session.id === selectedSessionId)
    : false
  const effectiveSessionId = selectedSessionExists
    ? selectedSessionId
    : (sessions[0]?.id ?? null)
  const sessionQuery = useClawSession(connection, effectiveSessionId)
  const turnsQuery = useClawSessionTurns(connection, effectiveSessionId)
  const selectedSession =
    sessionQuery.data?.session ??
    sessions.find((session) => session.id === effectiveSessionId) ??
    null
  const runs =
    sessionQuery.data?.session.runs ??
    (selectedSession?.latest_run ? [selectedSession.latest_run] : [])
  const traceQueries = useClawRunTraces(connection, runs)
  const trimmedPrompt = prompt.trim()
  const streamingActive =
    streamStatus === 'connecting' || streamStatus === 'streaming'
  const activeRunId = selectedSession?.active_run_id ?? selectedSession?.activeRunId
  const canContinue = Boolean(
    connection && effectiveSessionId && trimmedPrompt && !streamingActive,
  )
  const canCancel = Boolean(connection && effectiveSessionId && activeRunId)
  const scopedLiveOutput =
    liveStreamSessionId === effectiveSessionId ? streamOutput : ''
  const scopedLiveStatus =
    liveStreamSessionId === effectiveSessionId ? streamStatus : 'idle'

  useEffect(() => {
    return () => abortControllerRef.current?.abort()
  }, [])

  useEffect(() => {
    if (!selectedSessionId || sessionsQuery.isLoading || selectedSessionExists) {
      return
    }
    onSelectSession(null)
  }, [onSelectSession, selectedSessionExists, selectedSessionId, sessionsQuery.isLoading])

  async function handleContinue(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!effectiveSessionId || !trimmedPrompt || streamingActive) return
    const abortController = new AbortController()
    abortControllerRef.current?.abort()
    abortControllerRef.current = abortController
    setLiveStreamSessionId(effectiveSessionId)
    setStreamStatus('connecting')
    setStreamOutput('')
    setStreamError(null)
    setStreamEventCount(0)
    setLastRunLabel(null)

    try {
      await createRunStream.mutateAsync({
        sessionId: effectiveSessionId,
        input: {
          workspace: workspaceBindingFromSpace(selectedSpace),
          metadata: {
            desktop: {
              source: 'chat_continue',
              space_id: selectedSpace.id,
              space_name: selectedSpace.name,
            },
          },
          input_parts: [{ type: 'text', text: trimmedPrompt }],
        },
        signal: abortController.signal,
        handlers: {
          onOpen: () => setStreamStatus('streaming'),
          onEvent: handleStreamEvent,
          onClose: () => {
            setStreamStatus((status) =>
              status === 'failed' ? status : 'completed',
            )
          },
        },
      })
      setPrompt('')
      setStreamStatus((status) => (status === 'failed' ? status : 'completed'))
    } catch (error) {
      if (abortController.signal.aborted) {
        if (abortControllerRef.current === abortController) setStreamStatus('idle')
        return
      }
      setStreamStatus('failed')
      setStreamError(error instanceof Error ? error.message : String(error))
    } finally {
      if (abortControllerRef.current === abortController) {
        abortControllerRef.current = null
      }
    }
  }

  function handleStreamEvent(event: ClawStreamEvent) {
    setStreamEventCount((count) => count + 1)
    const runId = streamRunId(event)
    if (runId) setLastRunLabel(runId.slice(0, 8))
    const sessionId = streamSessionId(event)
    if (sessionId) setLiveStreamSessionId(sessionId)
    const delta = streamTextDelta(event)
    if (delta) setStreamOutput((output) => `${output}${delta}`)
    if (isRunErrorEvent(event)) {
      setStreamStatus('failed')
      setStreamError(streamErrorMessage(event))
      return
    }
    if (isRunFinishedEvent(event)) {
      setStreamStatus((status) => (status === 'failed' ? status : 'completed'))
      return
    }
    setStreamStatus((status) =>
      status === 'failed' || status === 'completed' ? status : 'streaming',
    )
  }

  async function handleCancelActiveRun() {
    if (!effectiveSessionId || !canCancel) return
    await cancelSession.mutateAsync(effectiveSessionId)
  }

  return (
    <div className="grid min-h-full gap-5 xl:grid-cols-[360px_1fr]">
      <section className="rounded-[2rem] border border-black/[0.06] bg-white p-5 shadow-sm">
        <SectionHeader title="Chats" action={connection ? 'Live' : 'Offline'} />
        <div className="mt-4 space-y-3">
          <LiveSessionList
            connectionReady={Boolean(connection)}
            loading={sessionsQuery.isLoading}
            error={sessionsQuery.error}
            sessions={sessions}
            selectedSessionId={effectiveSessionId}
            onSelectSession={onSelectSession}
            compact
            emptyTitle="No sessions found"
            emptyDetail="Create a chat once Local Claw is running."
          />
        </div>
      </section>
      <section className="flex min-h-[620px] flex-col rounded-[2rem] border border-black/[0.06] bg-white shadow-sm">
        <div className="border-b border-black/[0.06] p-5">
          <p className="text-sm font-semibold text-blue-600">Conversation</p>
          <h2 className="mt-1 text-2xl font-semibold tracking-[-0.025em] text-slate-950">
            {selectedSession
              ? sessionTitle(selectedSession)
              : connection
                ? 'Select a chat'
                : 'Local Claw is offline'}
          </h2>
          <p className="mt-2 text-xs text-slate-500">
            {selectedSession
              ? `${selectedSession.run_count ?? selectedSession.runCount ?? 0} runs · ${selectedSession.profile_name ?? selectedSession.profileName ?? 'default'} profile`
              : (activeConnectionQuery.data?.status.message ??
                'Start Local Claw from Settings to load chats.')}
          </p>
        </div>
        <div className="grid flex-1 gap-5 p-5 lg:grid-cols-[1fr_320px]">
          <div className="min-h-0 rounded-[1.6rem] bg-[#fbfbf8] p-5 ring-1 ring-black/[0.04]">
            <SessionTurnsPanel
              loading={sessionQuery.isLoading || turnsQuery.isLoading}
              error={sessionQuery.error ?? turnsQuery.error}
              liveOutput={scopedLiveOutput}
              liveStatus={scopedLiveStatus}
              selectedSession={selectedSession}
              replayMessage={sessionQuery.data?.message ?? null}
              sessionDetail={sessionQuery.data?.session ?? null}
              turns={turnsQuery.data?.turns ?? []}
            />
            <form
              className="mt-4 rounded-3xl border border-black/[0.06] bg-white p-3 shadow-sm"
              onSubmit={handleContinue}
            >
              <div className="flex items-center gap-2">
                <input
                  value={prompt}
                  onChange={(event) => setPrompt(event.target.value)}
                  className="min-w-0 flex-1 rounded-2xl bg-[#f7f7f4] px-4 py-3 text-sm text-slate-950 outline-none placeholder:text-slate-400"
                  placeholder="Continue this chat..."
                />
                <button
                  className="rounded-2xl bg-[#111827] px-4 py-3 text-sm font-semibold text-white shadow-lg shadow-slate-950/15 transition disabled:cursor-not-allowed disabled:bg-slate-300 disabled:shadow-none"
                  disabled={!canContinue}
                  type="submit"
                >
                  {streamingActive ? 'Running' : 'Send'}
                </button>
              </div>
              <HomeStreamPreview
                eventCount={streamEventCount}
                error={streamError}
                output={scopedLiveOutput}
                runLabel={lastRunLabel}
                status={scopedLiveStatus}
              />
            </form>
          </div>
          <div className="space-y-3">
            <RunTimeline
              runs={runs}
              canCancel={canCancel && !cancelSession.isPending}
              onCancel={handleCancelActiveRun}
            />
            <TracePreview
              loading={traceQueries.some((query) => query.isLoading)}
              error={traceQueries.find((query) => query.error)?.error ?? null}
              traces={traceQueries.flatMap((query) =>
                query.data ? [query.data] : [],
              )}
            />
          </div>
        </div>
      </section>
    </div>
  )
}

function BoardPage({
  onSelectSession,
}: {
  onSelectSession: (sessionId: string | null) => void
}) {
  const activeConnectionQuery = useActiveClawConnection()
  const connection = activeConnectionQuery.data?.connection ?? null
  const sessionsQuery = useClawSessions(connection)
  const groupedSessions = groupSessionsForBoard(sessionsQuery.data ?? [])

  if (!connection) {
    return <EmptyState title="Local Claw is offline" detail="Start Local Claw to load the live Board." />
  }

  if (sessionsQuery.isLoading) {
    return <EmptyState title="Loading Board" detail="Reading live sessions." />
  }

  if (sessionsQuery.error) {
    return <EmptyState title="Could not load Board" detail={sessionsQuery.error.message} />
  }

  return (
    <div className="grid min-h-full gap-5 xl:grid-cols-4">
      {groupedSessions.map((column) => (
        <section
          key={column.title}
          className="rounded-[2rem] border border-black/[0.06] bg-white p-5 shadow-sm"
        >
          <SectionHeader title={column.title} action={`${column.items.length}`} />
          <div className="mt-4 space-y-3">
            {column.items.length === 0 ? (
              <EmptyState title="No chats" detail="This lane is clear." />
            ) : (
              column.items.map((session) => (
                <SessionRow
                  key={session.id}
                  session={session}
                  compact
                  onClick={() => onSelectSession(session.id)}
                />
              ))
            )}
          </div>
        </section>
      ))}
    </div>
  )
}

function InboxPage({
  onSelectSession,
}: {
  onSelectSession: (sessionId: string | null) => void
}) {
  const activeConnectionQuery = useActiveClawConnection()
  const connection = activeConnectionQuery.data?.connection ?? null
  const sessionsQuery = useClawSessions(connection)
  const inboxItems = inboxItemsFromSessions(sessionsQuery.data ?? [])

  if (!connection) {
    return <EmptyState title="Local Claw is offline" detail="Start Local Claw to load Inbox items." />
  }

  return (
    <section className="rounded-[2rem] border border-black/[0.06] bg-white p-7 shadow-sm">
      <p className="text-sm font-semibold text-blue-600">Inbox</p>
      <h2 className="mt-2 max-w-3xl text-4xl font-semibold tracking-[-0.035em] text-slate-950">
        Decisions that need a human
      </h2>
      <p className="mt-4 max-w-3xl text-sm leading-6 text-slate-500">
        Failed runs, interrupted work, pending approvals, bridge alerts, and
        schedule decisions land here.
      </p>
      <div className="mt-8 space-y-3">
        {sessionsQuery.isLoading ? (
          <EmptyState title="Loading Inbox" detail="Reading live sessions." />
        ) : sessionsQuery.error ? (
          <EmptyState title="Could not load Inbox" detail={sessionsQuery.error.message} />
        ) : inboxItems.length === 0 ? (
          <EmptyState title="Inbox clear" detail="No failed runs or pending approvals." />
        ) : (
          inboxItems.map((item) => (
            <button
              key={`${item.session.id}-${item.title}`}
              className="flex w-full items-start gap-4 rounded-3xl border border-black/[0.06] bg-[#fbfbf8] p-4 text-left shadow-sm transition hover:bg-white"
              onClick={() => onSelectSession(item.session.id)}
              type="button"
            >
              <span className={cn('mt-1 h-3 w-3 rounded-full', statusTone(item.tone))} />
              <span className="min-w-0 flex-1">
                <span className="block text-sm font-semibold text-slate-950">
                  {item.title}
                </span>
                <span className="mt-1 block text-xs leading-5 text-slate-500">
                  {item.detail}
                </span>
              </span>
              <ChevronRight className="mt-1 h-4 w-4 text-slate-300" />
            </button>
          ))
        )}
      </div>
    </section>
  )
}

function SpacesPage({
  selectedSpaceId,
  spaces,
  onAddSpace,
  onSelectSpace,
}: {
  selectedSpaceId: string
  spaces: DesktopSpace[]
  onAddSpace: (space: DesktopSpace) => void
  onSelectSpace: (spaceId: string) => void
}) {
  const [name, setName] = useState('')
  const [path, setPath] = useState('')

  function handleAddSpace(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const normalizedPath = path.trim()
    if (!normalizedPath) return
    const normalizedName = name.trim() || folderName(normalizedPath)
    const space: DesktopSpace = {
      id: `local-${Date.now()}`,
      name: normalizedName,
      path: normalizedPath,
      runtime: 'Local Claw',
      trust: 'Trusted',
      default: false,
    }
    onAddSpace(space)
    onSelectSpace(space.id)
    setName('')
    setPath('')
  }

  return (
    <section className="rounded-[2rem] border border-black/[0.06] bg-white p-7 shadow-sm">
      <p className="text-sm font-semibold text-blue-600">Spaces</p>
      <h2 className="mt-2 max-w-3xl text-4xl font-semibold tracking-[-0.035em] text-slate-950">
        Workspace folders and runtime locations
      </h2>
      <p className="mt-4 max-w-3xl text-sm leading-6 text-slate-500">
        A Space combines a folder or cloud workspace, runtime connection, trust
        level, execution location, and default profile.
      </p>
      <form
        className="mt-6 grid gap-3 rounded-3xl border border-black/[0.06] bg-[#fbfbf8] p-4 md:grid-cols-[1fr_1.4fr_auto]"
        onSubmit={handleAddSpace}
      >
        <input
          value={name}
          onChange={(event) => setName(event.target.value)}
          className="rounded-2xl border border-black/[0.06] bg-white px-4 py-3 text-sm outline-none"
          placeholder="Space name"
        />
        <input
          value={path}
          onChange={(event) => setPath(event.target.value)}
          className="rounded-2xl border border-black/[0.06] bg-white px-4 py-3 text-sm outline-none"
          placeholder="/absolute/path/to/workspace"
        />
        <button className="rounded-2xl bg-[#111827] px-4 py-3 text-sm font-semibold text-white" type="submit">
          Add space
        </button>
      </form>
      <div className="mt-8 grid gap-4 xl:grid-cols-3">
        {spaces.map((space) => (
          <SpaceCard
            key={space.id}
            selected={space.id === selectedSpaceId}
            space={space}
            onClick={() => onSelectSpace(space.id)}
          />
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
          [
            'Preferences',
            'Hotkeys, notifications, voice and appearance',
            Settings,
          ],
          ['Secrets', 'Keychain-backed token storage', KeyRound],
          [
            'Advanced Runtime',
            'Profiles, schedules, bridges, heartbeat, logs',
            SlidersHorizontal,
          ],
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
      <h2 className="mt-2 max-w-3xl text-4xl font-semibold tracking-[-0.035em] text-slate-950">
        {title}
      </h2>
      <p className="mt-4 max-w-3xl text-sm leading-6 text-slate-500">{body}</p>
      <div className="mt-8 grid gap-4 md:grid-cols-3">
        {cards.map(([cardTitle, detail, Icon]) => (
          <div
            key={cardTitle}
            className="rounded-3xl border border-black/[0.06] bg-[#f7f7f4] p-5"
          >
            <Icon className="h-5 w-5 text-slate-700" />
            <h3 className="mt-4 text-sm font-semibold text-slate-950">
              {cardTitle}
            </h3>
            <p className="mt-2 text-xs leading-5 text-slate-500">{detail}</p>
          </div>
        ))}
      </div>
    </section>
  )
}

function RightPanel({
  connection,
  selectedSessionId,
  selectedSpace,
  onCollapse,
}: {
  connection: Parameters<typeof useClawSession>[0]
  selectedSessionId: string | null
  selectedSpace: DesktopSpace
  onCollapse: () => void
}) {
  const sessionQuery = useClawSession(connection, selectedSessionId)
  const session = sessionQuery.data?.session ?? null
  const latestRun = session?.latest_run ?? session?.latestRun ?? null
  const pendingCount = session ? inboxItemsFromSessions([session]).length : 0

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between gap-3">
        <SectionHeader title="Live context" action={connection ? 'Live' : 'Offline'} />
        <IconButton
          label="Hide context"
          icon={PanelRightClose}
          onClick={onCollapse}
        />
      </div>
      <div className="mt-4 rounded-3xl border border-black/[0.06] bg-white p-4 shadow-sm">
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
          Active chat
        </p>
        <h3 className="mt-2 text-lg font-semibold text-slate-950">
          {session ? sessionTitle(session) : 'No selected chat'}
        </h3>
        <p className="mt-2 text-sm leading-6 text-slate-500">
          {latestRun
            ? `Run #${latestRun.sequence_no ?? latestRun.sequenceNo ?? '—'} · ${latestRun.status}`
            : connection
              ? 'Select a chat to inspect run state.'
              : 'Start Local Claw to hydrate this panel.'}
        </p>
      </div>
      <div className="mt-5 space-y-3">
        <ActionRow
          icon={HardDrive}
          title="Active space"
          detail={spaceDetail(selectedSpace)}
        />
        <ActionRow icon={ShieldCheck} title="Trust" detail={selectedSpace.trust} />
        <ActionRow
          icon={Bell}
          title="Inbox"
          detail={pendingCount ? `${pendingCount} pending items` : 'No blocking approvals'}
        />
      </div>
    </div>
  )
}

function LiveSessionList({
  connectionReady,
  loading,
  error,
  sessions,
  selectedSessionId,
  onSelectSession,
  compact,
  emptyTitle,
  emptyDetail,
}: {
  connectionReady: boolean
  loading: boolean
  error: Error | null
  sessions: ClawSessionSummary[]
  selectedSessionId?: string | null
  onSelectSession?: (sessionId: string) => void
  compact?: boolean
  emptyTitle: string
  emptyDetail: string
}) {
  if (!connectionReady) {
    return (
      <EmptyState
        title="Local Claw is offline"
        detail="Open Settings and start Local Claw to load chats."
      />
    )
  }

  if (loading) {
    return (
      <EmptyState
        title="Loading chats"
        detail="Reading sessions from the active Local Claw runtime."
      />
    )
  }

  if (error) {
    return <EmptyState title="Could not load chats" detail={error.message} />
  }

  if (sessions.length === 0) {
    return <EmptyState title={emptyTitle} detail={emptyDetail} />
  }

  return (
    <div className="space-y-3">
      {sessions.map((session) => (
        <SessionRow
          key={session.id}
          session={session}
          compact={compact}
          selected={session.id === selectedSessionId}
          onClick={
            onSelectSession ? () => onSelectSession(session.id) : undefined
          }
        />
      ))}
    </div>
  )
}

function SessionTurnsPanel({
  loading,
  error,
  liveOutput,
  liveStatus,
  turns,
  selectedSession,
  replayMessage,
  sessionDetail,
}: {
  loading: boolean
  error: Error | null
  liveOutput: string
  liveStatus: HomeStreamStatus
  replayMessage: JsonObject[] | null
  turns: ClawSessionTurn[]
  selectedSession: ClawSessionSummary | null
  sessionDetail: ClawSessionDetail | null
}) {
  if (!selectedSession) {
    return (
      <div className="flex h-full min-h-[360px] items-center justify-center text-center">
        <div>
          <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-3xl bg-[#111827] text-white shadow-lg shadow-slate-950/15">
            <Bot className="h-7 w-7" />
          </div>
          <h3 className="mt-5 text-xl font-semibold text-slate-950">
            Conversation surface
          </h3>
          <p className="mx-auto mt-2 max-w-md text-sm leading-6 text-slate-500">
            Select a live Claw session to inspect turns, runs, traces, and
            replay metadata.
          </p>
        </div>
      </div>
    )
  }

  if (loading)
    return (
      <EmptyState
        title="Loading turns"
        detail="Reading successful completed turns."
      />
    )
  if (error)
    return <EmptyState title="Could not load turns" detail={error.message} />

  const replayText = collectCommittedReplayText(sessionDetail, replayMessage)
  const hasLiveOutput = liveStatus !== 'idle' && liveOutput.length > 0
  if (turns.length === 0 && !replayText && !hasLiveOutput) {
    return (
      <EmptyState
        title="No completed turns"
        detail="Runs will appear here after a successful completion."
      />
    )
  }

  return (
    <div className="space-y-4">
      {replayText && (
        <TranscriptBubble
          label="Committed replay"
          role="assistant"
          text={replayText}
        />
      )}
      {turns.map((turn) => (
        <div key={turn.run_id ?? turn.runId} className="space-y-3">
          <TranscriptBubble
            label={`Turn ${turn.sequence_no ?? turn.sequenceNo ?? '—'} · ${formatDate(
              turn.created_at ?? turn.createdAt,
            )}`}
            role="user"
            text={turn.input_preview ?? turn.inputPreview ?? 'Input parts'}
          />
          <TranscriptBubble
            label={formatDate(
              turn.committed_at ??
                turn.committedAt ??
                turn.created_at ??
                turn.createdAt,
            )}
            role="assistant"
            text={
              turn.output_text ??
              turn.outputText ??
              turn.output_summary ??
              turn.outputSummary ??
              'No output summary.'
            }
          />
        </div>
      ))}
      {hasLiveOutput && (
        <TranscriptBubble label="Streaming now" role="assistant" text={liveOutput} />
      )}
    </div>
  )
}

function TranscriptBubble({
  label,
  role,
  text,
}: {
  label: string
  role: 'assistant' | 'user'
  text: string
}) {
  return (
    <div
      className={cn(
        'rounded-3xl border border-black/[0.06] p-4 shadow-sm',
        role === 'assistant' ? 'bg-white' : 'bg-[#eef5ff]',
      )}
    >
      <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">
        {label}
      </p>
      <p className="mt-2 whitespace-pre-wrap text-sm leading-6 text-slate-700">
        {text}
      </p>
    </div>
  )
}

function RunTimeline({
  canCancel,
  onCancel,
  runs,
}: {
  canCancel?: boolean
  onCancel?: () => void
  runs: ClawRunSummary[]
}) {
  if (runs.length === 0) {
    return (
      <ActionRow icon={Activity} title="Run timeline" detail="No runs loaded" />
    )
  }

  return (
    <div className="rounded-3xl border border-black/[0.06] bg-white p-4 shadow-sm">
      <SectionHeader title="Run timeline" action={`${runs.length}`} />
      {canCancel && (
        <button
          className="mt-3 w-full rounded-2xl border border-amber-200 bg-amber-50 px-3 py-2 text-xs font-semibold text-amber-700 transition hover:bg-amber-100"
          type="button"
          onClick={onCancel}
        >
          Cancel active run
        </button>
      )}
      <div className="mt-3 space-y-3">
        {runs.slice(0, 5).map((run) => (
          <div
            key={run.id}
            className="flex items-start gap-3 rounded-2xl bg-[#fbfbf8] p-3"
          >
            <span
              className={cn(
                'mt-1 h-2.5 w-2.5 rounded-full',
                statusTone(statusToneName(run.status)),
              )}
            />
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-semibold text-slate-950">
                Run #{run.sequence_no ?? run.sequenceNo ?? '—'} · {run.status}
              </p>
              <p className="mt-1 truncate text-xs text-slate-500">
                {run.output_summary ??
                  run.outputSummary ??
                  run.error_message ??
                  run.errorMessage ??
                  run.input_preview ??
                  run.inputPreview ??
                  'No summary'}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function TracePreview({
  loading,
  error,
  traces,
}: {
  loading: boolean
  error: Error | null
  traces: ClawRunTraceResponse[]
}) {
  const items = traces.flatMap((trace) => trace.trace ?? []).slice(0, 6)
  if (loading) {
    return (
      <ActionRow
        icon={TerminalSquare}
        title="Run trace"
        detail="Loading tool calls"
      />
    )
  }
  if (error) {
    return (
      <ActionRow
        icon={TerminalSquare}
        title="Run trace"
        detail={error.message}
      />
    )
  }
  if (items.length === 0) {
    return (
      <ActionRow
        icon={TerminalSquare}
        title="Run trace"
        detail="No tool calls loaded"
      />
    )
  }

  return (
    <div className="rounded-3xl border border-black/[0.06] bg-white p-4 shadow-sm">
      <SectionHeader title="Run trace" action={`${items.length}`} />
      <div className="mt-3 space-y-2">
        {items.map((item, index) => (
          <div
            key={`${item.tool_call_id ?? item.toolCallId ?? index}-${index}`}
            className="rounded-2xl bg-[#fbfbf8] p-3"
          >
            <p className="text-xs font-semibold text-slate-900">
              {item.type === 'tool_call' ? 'Tool call' : 'Tool response'} ·{' '}
              {item.tool_name ?? item.toolName ?? item.role ?? 'runtime'}
            </p>
            <p className="mt-1 line-clamp-3 text-xs leading-5 text-slate-500">
              {item.content ?? 'No trace content.'}
            </p>
          </div>
        ))}
      </div>
    </div>
  )
}

function EmptyState({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="rounded-3xl border border-dashed border-black/[0.08] bg-[#fbfbf8] p-5 text-center">
      <p className="text-sm font-semibold text-slate-950">{title}</p>
      <p className="mt-2 text-xs leading-5 text-slate-500">{detail}</p>
    </div>
  )
}

function Card({
  title,
  action,
  children,
}: {
  title: string
  action: string
  children: ReactNode
}) {
  return (
    <section className="rounded-[2rem] border border-black/[0.06] bg-white p-6 shadow-sm">
      <SectionHeader title={title} action={action} />
      <div className="mt-4">{children}</div>
    </section>
  )
}

function HomeStreamPreview({
  eventCount,
  error,
  output,
  runLabel,
  status,
}: {
  eventCount: number
  error: string | null
  output: string
  runLabel: string | null
  status: HomeStreamStatus
}) {
  if (status === 'idle') return null

  const statusLabel = homeStreamStatusLabel(status)
  const previewText =
    error ??
    (output.length > 0
      ? output
      : status === 'connecting'
        ? 'Opening a Claw run stream...'
        : 'Waiting for the first assistant chunk...')

  return (
    <div className="mt-3 rounded-[1.35rem] border border-black/[0.06] bg-[#fbfbf8] p-4 text-left ring-1 ring-black/[0.03]">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2 text-xs font-semibold text-slate-600">
          <span
            className={cn(
              'h-2.5 w-2.5 rounded-full',
              status === 'failed'
                ? statusTone('amber')
                : status === 'completed'
                  ? statusTone('emerald')
                  : statusTone('blue'),
            )}
          />
          {statusLabel}
        </div>
        <p className="text-xs text-slate-400">
          {runLabel ? `Run ${runLabel}` : `${eventCount} stream events`}
        </p>
      </div>
      <p
        className={cn(
          'mt-3 max-h-40 overflow-auto whitespace-pre-wrap text-sm leading-6',
          error ? 'text-amber-700' : 'text-slate-600',
        )}
      >
        {previewText}
      </p>
    </div>
  )
}

function ContextPill({
  icon: Icon,
  title,
  detail,
}: {
  icon: LucideIcon
  title: string
  detail: string
}) {
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
      <button className="text-xs font-semibold text-blue-600 hover:text-blue-700">
        {action}
      </button>
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

function ActionRow({
  icon: Icon,
  title,
  detail,
}: {
  icon: LucideIcon
  title: string
  detail: string
}) {
  return (
    <button
      className="flex w-full items-center gap-3 rounded-2xl border border-black/[0.06] bg-white p-3 text-left shadow-sm transition hover:bg-[#fbfbf8]"
      type="button"
    >
      <span className="flex h-10 w-10 items-center justify-center rounded-2xl bg-[#f7f7f4] text-slate-600 ring-1 ring-black/[0.04]">
        <Icon className="h-4 w-4" />
      </span>
      <span className="min-w-0 flex-1">
        <span className="block truncate text-sm font-semibold text-slate-950">
          {title}
        </span>
        <span className="mt-0.5 block truncate text-xs text-slate-500">
          {detail}
        </span>
      </span>
      <ChevronRight className="h-4 w-4 text-slate-300" />
    </button>
  )
}

function SessionRow({
  session,
  compact,
  selected,
  onClick,
}: {
  session: ClawSessionSummary
  compact?: boolean
  selected?: boolean
  onClick?: () => void
}) {
  const latestRun = session.latest_run ?? session.latestRun
  const status = session.status
  return (
    <button
      type="button"
      className={cn(
        'flex w-full items-center gap-4 rounded-3xl border text-left shadow-sm transition hover:bg-[#fbfbf8]',
        compact ? 'p-3' : 'p-4',
        selected
          ? 'border-slate-950 bg-[#fbfbf8]'
          : 'border-black/[0.06] bg-white',
      )}
      onClick={onClick}
    >
      <span
        className={cn(
          'h-3 w-3 rounded-full',
          statusTone(statusToneName(status)),
        )}
      />
      <span className="min-w-0 flex-1">
        <span className="block truncate text-sm font-semibold text-slate-950">
          {sessionTitle(session)}
        </span>
        <span className="mt-1 block truncate text-xs text-slate-500">
          {session.profile_name ?? session.profileName ?? 'default'} ·{' '}
          {latestRun?.output_summary ??
            latestRun?.outputSummary ??
            latestRun?.input_preview ??
            latestRun?.inputPreview ??
            `${session.run_count ?? session.runCount ?? 0} runs`}
        </span>
      </span>
      <span className="rounded-full border border-black/[0.06] bg-[#f7f7f4] px-3 py-1 text-xs font-semibold text-slate-600">
        {labelForStatus(status)}
      </span>
    </button>
  )
}

function SpaceCard({
  selected,
  space,
  onClick,
}: {
  selected: boolean
  space: DesktopSpace
  onClick: () => void
}) {
  return (
    <button
      className={cn(
        'rounded-3xl border p-5 text-left transition hover:bg-white hover:shadow-sm',
        selected
          ? 'border-slate-950 bg-white shadow-sm'
          : 'border-black/[0.06] bg-[#f7f7f4]',
      )}
      onClick={onClick}
      type="button"
    >
      <Folder className="h-6 w-6 text-slate-700" />
      <h3 className="mt-4 text-sm font-semibold text-slate-950">
        {space.name}
      </h3>
      <p className="mt-2 truncate text-xs text-slate-500">
        {space.path || 'Embedded local workspace'}
      </p>
      <div className="mt-4 flex flex-wrap gap-2">
        <span className="rounded-full border border-black/[0.06] bg-white px-2.5 py-1 text-[11px] font-semibold text-slate-600">
          {space.runtime}
        </span>
        <span className="rounded-full border border-black/[0.06] bg-white px-2.5 py-1 text-[11px] font-semibold text-slate-600">
          {space.trust}
        </span>
        {selected && (
          <span className="rounded-full border border-blue-200 bg-blue-50 px-2.5 py-1 text-[11px] font-semibold text-blue-700">
            Active
          </span>
        )}
      </div>
    </button>
  )
}

function enabledProfiles(profiles: ClawProfileSummary[]) {
  return profiles.filter((profile) => profile.enabled)
}

function profileNameOrDefault(
  selectedProfileName: string,
  profiles: ClawProfileSummary[],
) {
  if (profiles.some((profile) => profile.name === selectedProfileName)) {
    return selectedProfileName
  }
  return profiles[0]?.name ?? 'default'
}

function workspaceBindingFromSpace(
  space: DesktopSpace,
): ClawWorkspaceBinding | null {
  if (!space.path.trim()) return null
  const virtualPath = '/workspace/main'
  return {
    mounts: [
      {
        id: 'main',
        name: space.name,
        host_path: space.path,
        virtual_path: virtualPath,
        mode: 'rw',
        metadata: {
          desktop_space_id: space.id,
        },
      },
    ],
    default_mount_id: 'main',
    cwd: virtualPath,
    metadata: {
      desktop_space_id: space.id,
      desktop_space_name: space.name,
    },
  }
}

function collectCommittedReplayText(
  session: ClawSessionDetail | null,
  replayMessage: JsonObject[] | null,
) {
  const topLevelText = collectTextFromReplay(replayMessage).trim()
  if (topLevelText) return topLevelText
  if (!session?.runs) return ''
  return session.runs
    .map((run) => collectTextFromReplay(run.message).trim())
    .filter(Boolean)
    .join('\n\n')
}

function groupSessionsForBoard(sessions: ClawSessionSummary[]) {
  const waiting = sessions.filter(
    (session) => session.status === 'interrupted' || isHitlPending(session),
  )
  const active = sessions.filter(
    (session) =>
      !isHitlPending(session) &&
      session.status !== 'interrupted' &&
      ['queued', 'running'].includes(session.status),
  )
  const failed = sessions.filter(
    (session) => session.status === 'failed' && !isHitlPending(session),
  )
  const done = sessions.filter((session) =>
    ['completed', 'idle', 'cancelled'].includes(session.status),
  )
  return [
    { title: 'Active', items: active },
    { title: 'Waiting', items: waiting },
    { title: 'Done', items: done },
    { title: 'Failed', items: failed },
  ]
}

function isHitlPending(session: ClawSessionSummary) {
  return (
    session.status_reason === 'hitl_pending' ||
    session.statusReason === 'hitl_pending'
  )
}

function inboxItemsFromSessions(sessions: ClawSessionSummary[]) {
  return sessions.flatMap((session) => {
    const detail = session.status_detail ?? session.statusDetail ?? {}
    const activeInteractionCount = detail.active_interaction_count
    const items: Array<{
      title: string
      detail: string
      tone: string
      session: ClawSessionSummary
    }> = []
    if (
      session.status_reason === 'hitl_pending' ||
      session.statusReason === 'hitl_pending' ||
      (typeof activeInteractionCount === 'number' && activeInteractionCount > 0)
    ) {
      items.push({
        title: `Approval needed · ${sessionTitle(session)}`,
        detail: `${activeInteractionCount || 1} active interactions waiting for a decision.`,
        tone: 'blue',
        session,
      })
    }
    if (session.status === 'failed' || session.status === 'interrupted') {
      const latestRun = session.latest_run ?? session.latestRun
      items.push({
        title: `${labelForStatus(session.status)} · ${sessionTitle(session)}`,
        detail:
          latestRun?.error_message ??
          latestRun?.errorMessage ??
          String(detail.error_message ?? 'Open the chat to recover this run.'),
        tone: 'amber',
        session,
      })
    }
    return items
  })
}

function folderName(path: string) {
  return path.split('/').filter(Boolean).at(-1) ?? 'Workspace'
}

function spaceDetail(space: DesktopSpace) {
  return space.path ? `${space.name} · ${space.path}` : `${space.name} · embedded`
}

function readSpaces(): DesktopSpace[] {
  if (typeof window === 'undefined') return defaultDesktopSpaces
  try {
    const rawValue = window.localStorage.getItem(spacesStorageKey)
    if (!rawValue) return defaultDesktopSpaces
    const parsedValue = JSON.parse(rawValue) as DesktopSpace[]
    if (!Array.isArray(parsedValue) || parsedValue.length === 0) {
      return defaultDesktopSpaces
    }
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

function writeSpaces(spaces: DesktopSpace[]) {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.setItem(spacesStorageKey, JSON.stringify(spaces))
  } catch {
    // Keep local workspace selection usable in restricted storage contexts.
  }
}

function readLayoutPreferences(): DesktopLayoutPreferences {
  if (typeof window === 'undefined') return defaultLayoutPreferences

  try {
    const rawValue = window.localStorage.getItem(layoutPreferencesStorageKey)
    if (!rawValue) return defaultLayoutPreferences

    const parsedValue = JSON.parse(
      rawValue,
    ) as Partial<DesktopLayoutPreferences>
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
    window.localStorage.setItem(
      layoutPreferencesStorageKey,
      JSON.stringify(preferences),
    )
  } catch {
    // Keep the prototype usable in restricted storage contexts.
  }
}

function sessionTitle(session: ClawSessionSummary) {
  const latestRun = session.latest_run ?? session.latestRun
  const metadataTitle = session.metadata?.title
  if (typeof metadataTitle === 'string' && metadataTitle.trim())
    return metadataTitle
  return (
    latestRun?.input_preview ??
    latestRun?.inputPreview ??
    `Session ${session.id.slice(0, 8)}`
  )
}

function labelForStatus(status: ClawSessionStatus | ClawRunStatus) {
  const normalized = String(status)
  return (
    normalized.charAt(0).toUpperCase() +
    normalized.slice(1).replaceAll('_', ' ')
  )
}

function homeStreamStatusLabel(status: HomeStreamStatus) {
  if (status === 'connecting') return 'Connecting to Claw stream'
  if (status === 'streaming') return 'Streaming run output'
  if (status === 'completed') return 'Run completed'
  if (status === 'failed') return 'Run needs attention'
  return 'Ready'
}

function statusToneName(status: ClawSessionStatus | ClawRunStatus) {
  if (status === 'queued' || status === 'running') return 'blue'
  if (status === 'failed' || status === 'interrupted') return 'amber'
  if (status === 'cancelled') return 'slate'
  return 'emerald'
}

function formatDate(value?: string | null) {
  if (!value) return 'No date'
  try {
    return new Intl.DateTimeFormat(undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    }).format(new Date(value))
  } catch {
    return value
  }
}

function statusTone(tone: string) {
  if (tone === 'blue')
    return 'bg-blue-500 shadow-[0_0_0_4px_rgba(59,130,246,.10)]'
  if (tone === 'amber')
    return 'bg-amber-500 shadow-[0_0_0_4px_rgba(245,158,11,.12)]'
  if (tone === 'emerald')
    return 'bg-emerald-500 shadow-[0_0_0_4px_rgba(16,185,129,.12)]'
  return 'bg-slate-400 shadow-[0_0_0_4px_rgba(100,116,139,.10)]'
}
