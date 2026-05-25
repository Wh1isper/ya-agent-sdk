import {
  Activity,
  Bot,
  FileText,
  GitBranch,
  HardDrive,
  Inbox,
  PlayCircle,
  ShieldCheck,
  SquareCheckBig,
} from 'lucide-react'

export function WorkspaceHome() {
  return (
    <div className="space-y-6 p-6">
      <section className="overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm">
        <div className="grid gap-0 lg:grid-cols-[1.4fr_0.8fr]">
          <div className="bg-gradient-to-br from-slate-950 via-slate-900 to-blue-950 p-6 text-white">
            <p className="text-sm font-medium text-blue-200">Workspace Home</p>
            <h1 className="mt-2 text-3xl font-semibold tracking-tight">
              Work with agents from the context of this Mac
            </h1>
            <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-300">
              YA Desktop is the OS-native agent workspace: start work from a
              global command center, track tasks, review changes, approve risky
              actions, and keep local runtime control close to the user.
            </p>
            <div className="mt-6 flex flex-wrap gap-2">
              <Pill label="Run location" value="This Mac" />
              <Pill label="Tool execution" value="This Mac" />
              <Pill label="Trust" value="Trusted workspace" />
            </div>
          </div>
          <div className="grid gap-3 bg-white p-6">
            <QuickAction
              icon={PlayCircle}
              title="Start task"
              description="Create a task from this workspace with selected context."
            />
            <QuickAction
              icon={Bot}
              title="Open chat"
              description="Move into the rich work surface for streaming and replay."
            />
            <QuickAction
              icon={Inbox}
              title="Review approvals"
              description="Handle pending command, diff, and workspace decisions."
            />
          </div>
        </div>
      </section>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          icon={SquareCheckBig}
          label="Active tasks"
          value="0"
          detail="Task read model pending"
        />
        <MetricCard
          icon={Activity}
          label="Background runs"
          value="0"
          detail="SSE stream wiring pending"
        />
        <MetricCard
          icon={ShieldCheck}
          label="Approvals"
          value="0"
          detail="HITL inbox pending"
        />
        <MetricCard
          icon={HardDrive}
          label="Workspace"
          value="Local"
          detail="Sandbox status pending"
        />
      </div>

      <section className="grid gap-4 xl:grid-cols-[1.2fr_0.8fr]">
        <div className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
          <h2 className="text-sm font-semibold text-slate-950">Recent work</h2>
          <p className="mt-1 text-sm text-slate-500">
            Tasks and runs will appear here once the Claw client is connected.
          </p>
          <div className="mt-4 rounded-2xl border border-dashed border-slate-200 bg-slate-50 p-8 text-center">
            <SquareCheckBig className="mx-auto h-8 w-8 text-slate-300" />
            <p className="mt-3 text-sm font-semibold text-slate-900">
              No tasks loaded
            </p>
            <p className="mt-2 text-sm text-slate-500">
              The first connected view should hydrate active and recent runs
              from the selected Claw connection.
            </p>
          </div>
        </div>

        <div className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
          <h2 className="text-sm font-semibold text-slate-950">
            Workspace context
          </h2>
          <p className="mt-1 text-sm text-slate-500">
            Desktop should make local context explicit before running tools.
          </p>
          <dl className="mt-5 grid gap-4 text-sm">
            <Detail label="Connection" value="Local Claw" />
            <Detail label="Workspace provider" value="local" />
            <Detail label="Filesystem" value="Path bounded" />
            <Detail label="Shell" value="Sandbox guidance pending" />
            <Detail label="Memory" value="Workspace-native memory pending" />
          </dl>
        </div>
      </section>

      <section className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
        <h2 className="text-sm font-semibold text-slate-950">
          Native context capture
        </h2>
        <div className="mt-4 grid gap-3 md:grid-cols-3">
          <ContextCard icon={FileText} title="Selected text" />
          <ContextCard icon={GitBranch} title="Workspace changes" />
          <ContextCard icon={HardDrive} title="Clipboard and screenshots" />
        </div>
      </section>
    </div>
  )
}

function Pill({ label, value }: { label: string; value: string }) {
  return (
    <span className="inline-flex items-center gap-2 rounded-full border border-white/15 bg-white/10 px-3 py-1.5 text-xs font-medium text-slate-100">
      <span className="text-blue-200">{label}</span>
      <span>{value}</span>
    </span>
  )
}

function QuickAction({
  icon: Icon,
  title,
  description,
}: {
  icon: typeof Bot
  title: string
  description: string
}) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
      <div className="inline-flex rounded-xl bg-blue-50 p-2 text-blue-600">
        <Icon className="h-4 w-4" />
      </div>
      <h2 className="mt-3 text-sm font-semibold text-slate-950">{title}</h2>
      <p className="mt-1 text-xs leading-5 text-slate-500">{description}</p>
    </div>
  )
}

function MetricCard({
  icon: Icon,
  label,
  value,
  detail,
}: {
  icon: typeof Activity
  label: string
  value: string
  detail: string
}) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="inline-flex rounded-xl bg-blue-50 p-2 text-blue-600">
        <Icon className="h-5 w-5" />
      </div>
      <p className="mt-4 text-sm text-slate-500">{label}</p>
      <p className="mt-1 truncate text-xl font-semibold text-slate-950">
        {value}
      </p>
      <p className="mt-1 text-xs text-slate-400">{detail}</p>
    </div>
  )
}

function Detail({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-xs font-medium uppercase tracking-wide text-slate-400">
        {label}
      </dt>
      <dd className="mt-1 text-slate-800">{value}</dd>
    </div>
  )
}

function ContextCard({
  icon: Icon,
  title,
}: {
  icon: typeof FileText
  title: string
}) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
      <Icon className="h-5 w-5 text-blue-600" />
      <p className="mt-3 text-sm font-semibold text-slate-900">{title}</p>
      <p className="mt-1 text-xs leading-5 text-slate-500">
        Captured by Rust Core through Tauri commands.
      </p>
    </div>
  )
}
