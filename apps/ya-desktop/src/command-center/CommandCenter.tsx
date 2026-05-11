import { Camera, Clipboard, Command, FileText, Send } from 'lucide-react'

export function CommandCenter() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-slate-100 p-6">
      <section className="w-full max-w-3xl rounded-3xl border border-slate-200 bg-white p-5 shadow-xl">
        <div className="flex items-center gap-3 border-b border-slate-100 pb-4">
          <div className="rounded-2xl bg-blue-50 p-3 text-blue-600">
            <Command className="h-5 w-5" />
          </div>
          <div>
            <h1 className="text-lg font-semibold tracking-tight text-slate-950">
              Command Center
            </h1>
            <p className="text-sm text-slate-500">
              Global entry for tasks, selected text, clipboard, screenshots, and
              active workspace context.
            </p>
          </div>
        </div>
        <div className="mt-4 flex items-center gap-3 rounded-2xl border border-slate-200 bg-slate-50 p-3">
          <input
            className="min-w-0 flex-1 bg-transparent text-sm outline-none placeholder:text-slate-400"
            placeholder="Ask YA to work in the active workspace"
          />
          <button className="inline-flex items-center gap-2 rounded-xl bg-blue-600 px-3 py-2 text-xs font-semibold text-white shadow-sm">
            <Send className="h-3.5 w-3.5" />
            Start task
          </button>
        </div>
        <div className="mt-4 grid gap-3 md:grid-cols-3">
          <Context icon={FileText} title="Selection" text="No selected text captured" />
          <Context icon={Clipboard} title="Clipboard" text="Clipboard preview pending" />
          <Context icon={Camera} title="Screenshot" text="Screenshot capture pending" />
        </div>
      </section>
    </div>
  )
}

function Context({
  icon: Icon,
  title,
  text,
}: {
  icon: typeof FileText
  title: string
  text: string
}) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
      <Icon className="h-4 w-4 text-blue-600" />
      <p className="mt-2 text-sm font-semibold text-slate-900">{title}</p>
      <p className="mt-1 text-xs text-slate-500">{text}</p>
    </div>
  )
}
