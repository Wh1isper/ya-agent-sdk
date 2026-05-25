import { useMutation, useQueryClient } from '@tanstack/react-query'
import { CheckCircle2, ShieldCheck } from 'lucide-react'
import { toast } from 'sonner'

import { runDesktopOnboarding, type LocalClawLaunchConfig } from '../../runtime'
import { cn } from '../../lib'

export function OnboardingDialog({
  open,
  onComplete,
}: {
  open: boolean
  onComplete: () => void
}) {
  const queryClient = useQueryClient()
  const conservativeMutation = useOnboardingMutation(onComplete, queryClient)
  const recommendedMutation = useOnboardingMutation(onComplete, queryClient)
  const busy = conservativeMutation.isPending || recommendedMutation.isPending

  if (!open) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/35 p-6 backdrop-blur-sm">
      <section className="w-full max-w-3xl rounded-[2rem] border border-black/[0.08] bg-white p-7 shadow-2xl">
        <div className="flex items-start gap-4">
          <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-blue-50 text-blue-600">
            <ShieldCheck className="h-6 w-6" />
          </div>
          <div>
            <p className="text-sm font-semibold text-blue-600">
              Desktop onboarding
            </p>
            <h2 className="mt-2 text-3xl font-semibold tracking-[-0.035em] text-slate-950">
              Initialize Local Claw safety settings
            </h2>
            <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-500">
              Desktop can initialize the launch config, profile seed, and local
              API token. The default keeps shell review at extra_high and runs
              shell commands through the Local Claw sandbox.
            </p>
          </div>
        </div>

        <div className="mt-6 grid gap-4 md:grid-cols-2">
          <OnboardingOption
            title="Conservative defaults"
            detail="Shell review at extra_high, sandbox enabled, full network, raw host shell gated."
            selected
          />
          <OnboardingOption
            title="Recommended review preset"
            detail="Shell review at medium, unattended review at high, sandbox remains enabled with full network."
          />
        </div>

        <div className="mt-6 flex flex-wrap justify-end gap-3">
          <button
            type="button"
            className="rounded-2xl border border-black/[0.08] bg-white px-4 py-2.5 text-sm font-semibold text-slate-700 shadow-sm disabled:cursor-not-allowed disabled:text-slate-400"
            disabled={busy}
            onClick={() => conservativeMutation.mutate(null)}
          >
            Initialize conservative defaults
          </button>
          <button
            type="button"
            className={cn(
              'rounded-2xl bg-slate-950 px-4 py-2.5 text-sm font-semibold text-white shadow-sm',
              busy && 'cursor-not-allowed bg-slate-300',
            )}
            disabled={busy}
            onClick={() =>
              recommendedMutation.mutate(recommendedLaunchConfig())
            }
          >
            Use recommended preset
          </button>
        </div>
      </section>
    </div>
  )
}

function OnboardingOption({
  title,
  detail,
  selected = false,
}: {
  title: string
  detail: string
  selected?: boolean
}) {
  return (
    <div
      className={cn(
        'rounded-3xl border p-5',
        selected
          ? 'border-blue-200 bg-blue-50/70'
          : 'border-black/[0.06] bg-[#f7f7f4]',
      )}
    >
      <div className="flex items-center gap-2">
        <CheckCircle2
          className={cn(
            'h-4 w-4',
            selected ? 'text-blue-600' : 'text-slate-400',
          )}
        />
        <p className="text-sm font-semibold text-slate-950">{title}</p>
      </div>
      <p className="mt-2 text-sm leading-6 text-slate-500">{detail}</p>
    </div>
  )
}

function useOnboardingMutation(
  onComplete: () => void,
  queryClient: ReturnType<typeof useQueryClient>,
) {
  return useMutation({
    mutationFn: (config: LocalClawLaunchConfig | null) =>
      runDesktopOnboarding(config),
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({
          queryKey: ['local-claw-launch-config'],
        }),
        queryClient.invalidateQueries({ queryKey: ['local-claw-status'] }),
        queryClient.invalidateQueries({
          queryKey: ['desktop-workspace-status'],
        }),
      ])
      toast.success('Desktop onboarding completed')
      onComplete()
    },
    onError: (error) => {
      toast.error(error instanceof Error ? error.message : String(error))
    },
  })
}

function recommendedLaunchConfig(): LocalClawLaunchConfig {
  return {
    agencyEnabled: true,
    memoryEnabled: true,
    shellReviewEnabled: true,
    shellReviewModel: 'gateway@openai-responses:gpt-5.4-mini',
    shellReviewModelSettings: 'openai_responses_low',
    shellReviewRiskThreshold: 'medium',
    shellReviewUnattendedRiskThreshold: 'high',
    shellReviewAction: 'defer',
    shellSandboxEnabled: true,
    shellSandboxBackend: 'auto',
    shellSandboxNetwork: 'full',
    shellSandboxAllowRawHost: false,
    presetName: 'Desktop recommended',
    env: [],
    configFile: null,
  }
}
