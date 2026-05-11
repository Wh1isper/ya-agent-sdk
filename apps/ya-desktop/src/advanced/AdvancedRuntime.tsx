import { Activity, CalendarClock, HeartPulse, MessageSquareMore, Settings, SlidersHorizontal } from 'lucide-react'

const panels = [
  { title: 'Profiles', detail: 'AgentProfile, models, tools, subagents, MCP', icon: SlidersHorizontal },
  { title: 'Schedules', detail: 'Recurring and one-shot automation', icon: CalendarClock },
  { title: 'Bridges', detail: 'External events and delivery state', icon: MessageSquareMore },
  { title: 'Heartbeat', detail: 'Background pulse and guidance status', icon: HeartPulse },
  { title: 'Diagnostics', detail: 'Logs, runtime instances, storage, capabilities', icon: Activity },
]

export function AdvancedRuntime({ initialPanel }: { initialPanel?: 'desktop' }) {
  if (initialPanel === 'desktop') {
    return <DesktopSettings />
  }

  return (
    <div className="p-6">
      <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
        <p className="text-sm font-medium text-blue-600">Advanced Runtime</p>
        <h1 className="mt-1 text-2xl font-semibold tracking-tight text-slate-950">
          Operational controls
        </h1>
        <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-500">
          Advanced Runtime supports the Desktop product experience with lower
          level Claw management surfaces. These controls should stay available
          without becoming the main product navigation model.
        </p>
        <div className="mt-6 grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {panels.map((panel) => (
            <Panel key={panel.title} {...panel} />
          ))}
        </div>
      </section>
    </div>
  )
}

function DesktopSettings() {
  return (
    <div className="p-6">
      <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="inline-flex rounded-2xl bg-blue-50 p-3 text-blue-600">
          <Settings className="h-5 w-5" />
        </div>
        <h1 className="mt-4 text-2xl font-semibold tracking-tight text-slate-950">
          Desktop Settings
        </h1>
        <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-500">
          Hotkeys, notifications, appearance, voice, autostart, always-on
          behavior, and diagnostics export preferences belong here.
        </p>
      </section>
    </div>
  )
}

function Panel({
  icon: Icon,
  title,
  detail,
}: {
  icon: typeof Activity
  title: string
  detail: string
}) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
      <Icon className="h-5 w-5 text-blue-600" />
      <h2 className="mt-3 text-sm font-semibold text-slate-950">{title}</h2>
      <p className="mt-1 text-xs leading-5 text-slate-500">{detail}</p>
    </div>
  )
}
