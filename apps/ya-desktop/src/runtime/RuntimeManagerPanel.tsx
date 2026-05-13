import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Activity,
  Download,
  FileText,
  Play,
  RefreshCw,
  RotateCcw,
  Trash2,
} from 'lucide-react'
import { useState } from 'react'
import { toast } from 'sonner'

import { cn } from '../lib'
import {
  applyReadyClawRuntimeUpdate,
  checkClawRuntimeUpdate,
  getLocalClawStatus,
  getRuntimeInstallLog,
  getRuntimeManagerStatus,
  installLatestClawRuntime,
  removeClawRuntime,
  repairClawRuntime,
  restartLocalClaw,
  startLocalClaw,
  stopLocalClaw,
  updateClawRuntime,
  type InstalledClawRuntime,
} from './index'

const runtimeStatusKey = ['runtime-manager-status']
const localClawStatusKey = ['local-claw-status']

export function RuntimeManagerPanel() {
  const queryClient = useQueryClient()
  const [installLog, setInstallLog] = useState('')

  const runtimeStatus = useQuery({
    queryKey: runtimeStatusKey,
    queryFn: getRuntimeManagerStatus,
  })
  const localStatus = useQuery({
    queryKey: localClawStatusKey,
    queryFn: getLocalClawStatus,
  })

  const invalidate = async () => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: runtimeStatusKey }),
      queryClient.invalidateQueries({ queryKey: localClawStatusKey }),
    ])
  }

  const installMutation = useRuntimeMutation(installLatestClawRuntime, invalidate)
  const updateMutation = useRuntimeMutation(updateClawRuntime, invalidate)
  const repairMutation = useRuntimeMutation(() => repairClawRuntime(), invalidate)
  const startMutation = useRuntimeMutation(startLocalClaw, invalidate)
  const stopMutation = useRuntimeMutation(stopLocalClaw, invalidate)
  const restartMutation = useRuntimeMutation(restartLocalClaw, invalidate)
  const removeMutation = useRuntimeMutation(removeClawRuntime, invalidate)
  const checkUpdateMutation = useRuntimeMutation(checkClawRuntimeUpdate, invalidate)
  const applyUpdateMutation = useRuntimeMutation(applyReadyClawRuntimeUpdate, invalidate)
  const logMutation = useMutation({
    mutationFn: getRuntimeInstallLog,
    onSuccess: setInstallLog,
    onError: showError,
  })

  const active = runtimeStatus.data?.active
  const updateState = runtimeStatus.data?.updateState
  const running = localStatus.data?.running ?? false
  const busy =
    installMutation.isPending ||
    updateMutation.isPending ||
    repairMutation.isPending ||
    startMutation.isPending ||
    stopMutation.isPending ||
    restartMutation.isPending ||
    removeMutation.isPending ||
    checkUpdateMutation.isPending ||
    applyUpdateMutation.isPending

  return (
    <section className="rounded-[2rem] border border-black/[0.06] bg-white p-7 shadow-sm">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-sm font-semibold text-blue-600">Runtime Manager</p>
          <h2 className="mt-2 text-3xl font-semibold tracking-[-0.035em] text-slate-950">
            Local Claw runtime
          </h2>
          <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-500">
            Desktop installs Claw with app-managed uv, activates verified runtimes, and launches
            ya-clawd as a local child process.
          </p>
        </div>
        <button
          type="button"
          className="inline-flex items-center gap-2 rounded-2xl border border-black/[0.06] bg-white px-4 py-2.5 text-sm font-semibold text-slate-700 shadow-sm transition hover:text-slate-950"
          onClick={() => void invalidate()}
        >
          <RefreshCw className="h-4 w-4" />
          Refresh
        </button>
      </div>

      {updateState?.updateReady && updateState.candidate && (
        <div className="mt-6 rounded-3xl border border-emerald-200 bg-emerald-50 p-5">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <p className="text-sm font-semibold text-emerald-900">Claw update ready</p>
              <p className="mt-1 text-sm text-emerald-800/75">
                Version {updateState.candidate.version} is verified. Restart local Claw to apply it.
              </p>
            </div>
            <ActionButton
              icon={Play}
              label="Restart to apply"
              busy={applyUpdateMutation.isPending}
              disabled={busy}
              onClick={() => applyUpdateMutation.mutate()}
            />
          </div>
        </div>
      )}

      {updateState?.lastError && (
        <div className="mt-6 rounded-3xl border border-rose-200 bg-rose-50 p-5">
          <p className="text-sm font-semibold text-rose-900">Runtime update check failed</p>
          <p className="mt-1 text-sm text-rose-800/75">{updateState.lastError}</p>
        </div>
      )}

      <div className="mt-6 grid gap-4 xl:grid-cols-4">
        <StatusCard
          label="Active Claw"
          value={active?.version ?? 'Missing'}
          detail={active?.contract ?? 'Install runtime to activate local Claw'}
        />
        <StatusCard
          label="Local daemon"
          value={running ? 'Running' : 'Stopped'}
          detail={localStatus.data?.baseUrl ?? localStatus.data?.message ?? 'Waiting for status'}
        />
        <StatusCard
          label="Auto update"
          value={updateState?.updateReady ? 'Ready' : 'Enabled'}
          detail={formatUpdateStateDetail(updateState)}
        />
        <StatusCard
          label="uv"
          value={runtimeStatus.data?.uvPath ? 'Configured' : 'Resolving'}
          detail={runtimeStatus.data?.uvPath ?? 'Bundled or PATH uv'}
        />
      </div>

      <div className="mt-6 flex flex-wrap gap-2">
        <ActionButton
          icon={Download}
          label="Install latest"
          busy={installMutation.isPending}
          disabled={busy}
          onClick={() => installMutation.mutate()}
        />
        <ActionButton
          icon={RefreshCw}
          label="Update now"
          busy={updateMutation.isPending}
          disabled={busy}
          onClick={() => updateMutation.mutate()}
        />
        <ActionButton
          icon={RefreshCw}
          label="Check auto update"
          busy={checkUpdateMutation.isPending}
          disabled={busy}
          onClick={() => checkUpdateMutation.mutate()}
        />
        <ActionButton
          icon={RotateCcw}
          label="Repair"
          busy={repairMutation.isPending}
          disabled={busy}
          onClick={() => repairMutation.mutate()}
        />
        <ActionButton
          icon={Play}
          label={running ? 'Restart daemon' : 'Start daemon'}
          busy={running ? restartMutation.isPending : startMutation.isPending}
          disabled={busy}
          onClick={() => (running ? restartMutation.mutate() : startMutation.mutate())}
        />
        <ActionButton
          icon={Activity}
          label="Stop daemon"
          busy={stopMutation.isPending}
          disabled={busy || !running}
          onClick={() => stopMutation.mutate()}
        />
        <ActionButton
          icon={FileText}
          label="Load latest log"
          busy={logMutation.isPending}
          disabled={logMutation.isPending}
          onClick={() => logMutation.mutate(undefined)}
        />
      </div>

      <div className="mt-6 rounded-3xl border border-black/[0.06] bg-[#f7f7f4] p-4">
        <div className="flex items-center justify-between gap-3">
          <h3 className="text-sm font-semibold text-slate-950">Installed runtimes</h3>
          <span className="text-xs text-slate-500">
            {runtimeStatus.data?.clawDir ?? 'Runtime directory'}
          </span>
        </div>
        <div className="mt-3 space-y-2">
          {(runtimeStatus.data?.runtimes ?? []).map((runtime) => (
            <RuntimeRow
              key={runtime.id}
              runtime={runtime}
              disabled={busy}
              onLog={() => logMutation.mutate(runtime.id)}
              onRemove={() => removeMutation.mutate(runtime.id)}
            />
          ))}
          {runtimeStatus.data?.runtimes.length === 0 && (
            <p className="rounded-2xl bg-white px-4 py-3 text-sm text-slate-500">
              No managed runtime is installed yet.
            </p>
          )}
        </div>
      </div>

      {installLog && (
        <pre className="mt-6 max-h-80 overflow-auto rounded-3xl bg-slate-950 p-4 text-xs leading-5 text-slate-100">
          {installLog}
        </pre>
      )}
    </section>
  )
}

function useRuntimeMutation<TResult>(
  mutationFn: () => Promise<TResult>,
  onSettled: () => Promise<void>,
): ReturnType<typeof useMutation<TResult, Error, void>>
function useRuntimeMutation<TArgs, TResult>(
  mutationFn: (args: TArgs) => Promise<TResult>,
  onSettled: () => Promise<void>,
): ReturnType<typeof useMutation<TResult, Error, TArgs>>
function useRuntimeMutation<TArgs, TResult>(
  mutationFn: ((args: TArgs) => Promise<TResult>) | (() => Promise<TResult>),
  onSettled: () => Promise<void>,
) {
  return useMutation({
    mutationFn,
    onSuccess: () => toast.success('Runtime operation completed'),
    onError: showError,
    onSettled,
  })
}

function showError(error: unknown) {
  toast.error(error instanceof Error ? error.message : String(error))
}

function formatUpdateStateDetail(updateState: RuntimeManagerStatusUpdateState | undefined) {
  if (!updateState) {
    return 'Waiting for status'
  }
  if (updateState.checkInProgress) {
    return 'Checking latest Claw runtime'
  }
  if (updateState.updateReady && updateState.candidate) {
    return `Ready ${updateState.candidate.version}`
  }
  if (updateState.lastCheckedAt) {
    return `Last checked ${formatUnixTime(updateState.lastCheckedAt)}`
  }
  return 'Checks after startup and every 24h'
}

type RuntimeManagerStatusUpdateState = NonNullable<
  Awaited<ReturnType<typeof getRuntimeManagerStatus>>['updateState']
>

function formatUnixTime(value: number) {
  return new Date(value * 1000).toLocaleString()
}

function StatusCard({ label, value, detail }: { label: string; value: string; detail: string }) {
  return (
    <div className="rounded-3xl border border-black/[0.06] bg-[#f7f7f4] p-5">
      <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">{label}</p>
      <p className="mt-3 text-lg font-semibold text-slate-950">{value}</p>
      <p className="mt-1 truncate text-xs text-slate-500">{detail}</p>
    </div>
  )
}

function ActionButton({
  icon: Icon,
  label,
  busy,
  disabled,
  onClick,
}: {
  icon: typeof Download
  label: string
  busy: boolean
  disabled: boolean
  onClick: () => void
}) {
  return (
    <button
      type="button"
      className={cn(
        'inline-flex items-center gap-2 rounded-2xl px-4 py-2.5 text-sm font-semibold shadow-sm transition',
        disabled
          ? 'cursor-not-allowed bg-slate-100 text-slate-400'
          : 'bg-[#111827] text-white hover:bg-slate-800',
      )}
      disabled={disabled}
      onClick={onClick}
    >
      <Icon className={cn('h-4 w-4', busy && 'animate-spin')} />
      {label}
    </button>
  )
}

function RuntimeRow({
  runtime,
  disabled,
  onLog,
  onRemove,
}: {
  runtime: InstalledClawRuntime
  disabled: boolean
  onLog: () => void
  onRemove: () => void
}) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl bg-white px-4 py-3">
      <div className="min-w-0">
        <div className="flex items-center gap-2">
          <p className="truncate text-sm font-semibold text-slate-950">
            {runtime.version ?? runtime.id}
          </p>
          {runtime.active && (
            <span className="rounded-full bg-emerald-50 px-2 py-0.5 text-xs font-semibold text-emerald-700">
              Active
            </span>
          )}
          {runtime.failed && (
            <span className="rounded-full bg-rose-50 px-2 py-0.5 text-xs font-semibold text-rose-700">
              Failed
            </span>
          )}
        </div>
        <p className="mt-1 truncate text-xs text-slate-500">{runtime.runtimeDir}</p>
      </div>
      <div className="flex gap-2">
        <button
          type="button"
          className="rounded-xl border border-black/[0.06] px-3 py-2 text-xs font-semibold text-slate-600"
          onClick={onLog}
        >
          Log
        </button>
        <button
          type="button"
          className="rounded-xl border border-rose-200 px-3 py-2 text-xs font-semibold text-rose-600 disabled:cursor-not-allowed disabled:border-slate-200 disabled:text-slate-400"
          disabled={disabled || runtime.active}
          onClick={onRemove}
        >
          <Trash2 className="h-3.5 w-3.5" />
        </button>
      </div>
    </div>
  )
}
