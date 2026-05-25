import { Cloud, HardDrive, KeyRound, Network, Save, Server } from 'lucide-react'

export function ConnectionsPage() {
  return (
    <div className="space-y-6 p-6">
      <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
        <p className="text-sm font-medium text-blue-600">Connections</p>
        <h1 className="mt-1 text-2xl font-semibold tracking-tight text-slate-950">
          Runtime targets
        </h1>
        <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-500">
          Configure local sidecar, remote Claw, cloud Claw, keychain-backed
          tokens, capability discovery, default workspaces, and diagnostics.
        </p>
      </section>

      <div className="grid gap-4 xl:grid-cols-3">
        <ConnectionCard
          icon={HardDrive}
          title="Local Claw"
          detail="Bundled sidecar on this Mac"
        />
        <ConnectionCard
          icon={Server}
          title="Remote Claw"
          detail="Self-hosted HTTPS runtime"
        />
        <ConnectionCard
          icon={Cloud}
          title="Cloud Claw"
          detail="Hosted workspace runtime"
        />
      </div>

      <section className="max-w-3xl rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex items-center gap-2">
          <Network className="h-5 w-5 text-blue-600" />
          <h2 className="text-sm font-semibold text-slate-950">
            Active connection
          </h2>
        </div>
        <div className="mt-6 grid gap-5">
          <label className="block">
            <span className="text-sm font-medium text-slate-700">
              Backend URL
            </span>
            <input
              className="mt-2 w-full rounded-xl border border-slate-200 px-3 py-2 text-sm outline-none ring-blue-600 transition focus:ring-2"
              placeholder="http://127.0.0.1:9042"
            />
          </label>
          <label className="block">
            <span className="text-sm font-medium text-slate-700">
              API Token
            </span>
            <div className="mt-2 flex items-center gap-2 rounded-xl border border-slate-200 px-3 py-2">
              <KeyRound className="h-4 w-4 text-slate-400" />
              <input
                className="min-w-0 flex-1 text-sm outline-none"
                type="password"
                placeholder="YA_CLAW_API_TOKEN"
              />
            </div>
          </label>
          <button className="inline-flex w-fit items-center gap-2 rounded-xl bg-blue-600 px-4 py-2 text-sm font-semibold text-white shadow-sm">
            <Save className="h-4 w-4" />
            Save connection
          </button>
        </div>
      </section>
    </div>
  )
}

function ConnectionCard({
  icon: Icon,
  title,
  detail,
}: {
  icon: typeof HardDrive
  title: string
  detail: string
}) {
  return (
    <div className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="inline-flex rounded-2xl bg-blue-50 p-3 text-blue-600">
        <Icon className="h-5 w-5" />
      </div>
      <h2 className="mt-4 text-sm font-semibold text-slate-950">{title}</h2>
      <p className="mt-1 text-xs leading-5 text-slate-500">{detail}</p>
    </div>
  )
}
