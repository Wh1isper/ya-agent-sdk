import { Bell, FileDiff, ShieldCheck, TerminalSquare } from 'lucide-react'

export function ApprovalsInbox() {
  return (
    <div className="p-6">
      <section className="rounded-3xl border border-slate-200 bg-white p-8 shadow-sm">
        <div className="inline-flex rounded-2xl bg-blue-50 p-3 text-blue-600">
          <ShieldCheck className="h-6 w-6" />
        </div>
        <h1 className="mt-4 text-2xl font-semibold tracking-tight text-slate-950">
          Approvals Inbox
        </h1>
        <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-500">
          Desktop owns the focused HITL experience: native notifications,
          command previews, file diff previews, workspace trust signals, and
          audit metadata for approval decisions.
        </p>
        <div className="mt-6 grid gap-4 md:grid-cols-3">
          <ApprovalType icon={TerminalSquare} title="Commands" />
          <ApprovalType icon={FileDiff} title="File diffs" />
          <ApprovalType icon={Bell} title="Bridge and schedule events" />
        </div>
      </section>
    </div>
  )
}

function ApprovalType({
  icon: Icon,
  title,
}: {
  icon: typeof ShieldCheck
  title: string
}) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
      <Icon className="h-5 w-5 text-blue-600" />
      <h2 className="mt-3 text-sm font-semibold text-slate-950">{title}</h2>
      <p className="mt-1 text-xs leading-5 text-slate-500">
        Fed by Claw notification and approval response APIs.
      </p>
    </div>
  )
}
