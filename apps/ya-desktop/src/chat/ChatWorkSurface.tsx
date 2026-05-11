import { FileCode2, GitCompare, PlayCircle, RotateCcw, Square, TerminalSquare } from 'lucide-react'

export function ChatWorkSurface() {
  return (
    <div className="flex h-screen min-h-0 bg-slate-100">
      <aside className="flex w-80 shrink-0 flex-col border-r border-slate-200 bg-white">
        <div className="border-b border-slate-200 p-4">
          <p className="text-sm font-medium text-blue-600">Chat Work Surface</p>
          <h1 className="mt-1 text-xl font-semibold tracking-tight text-slate-950">
            Agent collaboration
          </h1>
          <p className="mt-2 text-sm text-slate-500">
            Rich chat, run replay, tools, shell output, diffs, and artifacts.
          </p>
        </div>
        <div className="flex-1 p-3">
          <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50 p-6 text-center">
            <p className="text-sm font-semibold text-slate-900">
              No task selected
            </p>
            <p className="mt-2 text-xs leading-5 text-slate-500">
              Open a task or start from Command Center to create a session.
            </p>
          </div>
        </div>
      </aside>
      <main className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-16 items-center justify-between border-b border-slate-200 bg-white px-6">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-blue-600">
              Work surface
            </p>
            <h2 className="text-lg font-semibold tracking-tight text-slate-950">
              New task conversation
            </h2>
          </div>
          <div className="flex items-center gap-2">
            <button className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-700 shadow-sm">
              <RotateCcw className="h-3.5 w-3.5" />
              Rerun
            </button>
            <button className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-700 shadow-sm">
              <Square className="h-3.5 w-3.5" />
              Cancel
            </button>
          </div>
        </header>
        <div className="flex-1 overflow-auto p-6">
          <div className="mx-auto max-w-4xl space-y-4">
            <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
              <div className="inline-flex rounded-2xl bg-blue-50 p-3 text-blue-600">
                <PlayCircle className="h-6 w-6" />
              </div>
              <h3 className="mt-4 text-lg font-semibold text-slate-950">
                Ready for Claw streaming
              </h3>
              <p className="mt-2 text-sm leading-6 text-slate-500">
                Chat is one work surface inside Desktop. It should show live
                assistant output, AGUI replay, tool events, approvals, command
                output, diffs, and artifacts for the selected task.
              </p>
            </section>
            <div className="grid gap-4 md:grid-cols-3">
              <Panel icon={TerminalSquare} title="Shell output" />
              <Panel icon={GitCompare} title="File diffs" />
              <Panel icon={FileCode2} title="Artifacts" />
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}

function Panel({ icon: Icon, title }: { icon: typeof TerminalSquare; title: string }) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
      <Icon className="h-5 w-5 text-blue-600" />
      <h3 className="mt-3 text-sm font-semibold text-slate-950">{title}</h3>
      <p className="mt-1 text-xs leading-5 text-slate-500">
        Connected after the run trace and message replay API are wired.
      </p>
    </div>
  )
}
