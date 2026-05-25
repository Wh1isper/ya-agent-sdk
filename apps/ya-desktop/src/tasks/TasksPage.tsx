import { Clock, GitPullRequest, RotateCcw, SquareCheckBig } from 'lucide-react'

const statuses = [
  { label: 'Active', value: '0', detail: 'Queued and running work' },
  { label: 'Waiting', value: '0', detail: 'Pending approvals' },
  { label: 'Completed', value: '0', detail: 'Successful runs' },
  { label: 'Failed', value: '0', detail: 'Needs review' },
]

export function TasksPage() {
  return (
    <div className="flex h-screen min-h-0 bg-slate-100">
      <aside className="flex w-80 shrink-0 flex-col border-r border-slate-200 bg-white">
        <div className="border-b border-slate-200 p-4">
          <p className="text-sm font-medium text-blue-600">Tasks / Runs</p>
          <h1 className="mt-1 text-xl font-semibold tracking-tight text-slate-950">
            Work queue
          </h1>
          <p className="mt-2 text-sm text-slate-500">
            Product-level tasks backed by Claw sessions and runs.
          </p>
        </div>
        <div className="grid gap-2 p-3">
          {statuses.map((status) => (
            <button
              key={status.label}
              type="button"
              className="rounded-2xl border border-slate-200 bg-white p-3 text-left transition hover:bg-slate-50"
            >
              <div className="flex items-center justify-between">
                <span className="text-sm font-semibold text-slate-900">
                  {status.label}
                </span>
                <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-600">
                  {status.value}
                </span>
              </div>
              <p className="mt-1 text-xs text-slate-500">{status.detail}</p>
            </button>
          ))}
        </div>
      </aside>

      <main className="min-w-0 flex-1 overflow-auto p-6">
        <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
          <div className="flex items-start justify-between gap-4">
            <div>
              <p className="text-sm font-medium text-blue-600">Task model</p>
              <h2 className="mt-1 text-2xl font-semibold tracking-tight text-slate-950">
                Runs become user-facing work
              </h2>
              <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-500">
                Desktop should track work as tasks, with Claw sessions and runs
                as the runtime backing. This page will connect status groups,
                run controls, replay, diffs, artifacts, and approvals.
              </p>
            </div>
            <button className="inline-flex items-center gap-2 rounded-xl bg-blue-600 px-4 py-2 text-sm font-semibold text-white shadow-sm">
              <SquareCheckBig className="h-4 w-4" />
              New task
            </button>
          </div>

          <div className="mt-6 grid gap-4 md:grid-cols-3">
            <Feature
              icon={Clock}
              title="Lifecycle"
              text="queued, running, waiting, completed, failed"
            />
            <Feature
              icon={RotateCcw}
              title="Run controls"
              text="cancel, retry, rerun, pause, resume, handoff"
            />
            <Feature
              icon={GitPullRequest}
              title="Review"
              text="tool timeline, shell output, diffs, artifacts"
            />
          </div>
        </section>
      </main>
    </div>
  )
}

function Feature({
  icon: Icon,
  title,
  text,
}: {
  icon: typeof Clock
  title: string
  text: string
}) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
      <Icon className="h-5 w-5 text-blue-600" />
      <h3 className="mt-3 text-sm font-semibold text-slate-950">{title}</h3>
      <p className="mt-1 text-xs leading-5 text-slate-500">{text}</p>
    </div>
  )
}
