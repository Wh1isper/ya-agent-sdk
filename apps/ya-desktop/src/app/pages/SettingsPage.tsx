import {
  BrainCircuit,
  Database,
  KeyRound,
  Network,
  Settings,
  ShieldCheck,
  SlidersHorizontal,
} from 'lucide-react'

import { RuntimeManagerPanel } from '../../runtime/RuntimeManagerPanel'
import { PageFrame, SettingCard } from '../ui'

export function SettingsPage({
  onRunOnboarding,
}: {
  onRunOnboarding: () => void
}) {
  return (
    <PageFrame
      eyebrow="Settings"
      title="Preferences and runtime"
      body="Keep everyday controls simple and place lower-level runtime tools behind clear sections."
    >
      <div className="grid gap-3 md:grid-cols-5">
        <SettingCard
          icon={Settings}
          title="Preferences"
          detail="Appearance, hotkeys, notifications, voice."
        />
        <SettingCard
          icon={KeyRound}
          title="Secrets"
          detail="Keychain-backed tokens and credentials."
        />
        <SettingCard
          icon={SlidersHorizontal}
          title="Advanced Runtime"
          detail="Profiles, schedules, bridges, logs, diagnostics."
        />
        <SettingCard
          icon={BrainCircuit}
          title="Agency"
          detail="Desktop launches Local Claw with proactive Agency enabled."
        />
        <SettingCard
          icon={Database}
          title="Memory"
          detail="Desktop launches Local Claw with memory extraction enabled."
        />
        <SettingCard
          icon={ShieldCheck}
          title="Shell Safety"
          detail="Desktop seeds Local Claw profiles with shell review enabled by default."
        />
        <SettingCard
          icon={Network}
          title="Environment Relay"
          detail="Prepare local capability grants for central Claw agents through ya-environment-relay.v1."
        />
      </div>
      <div className="mt-5 rounded-[2rem] border border-blue-100 bg-blue-50/60 p-5">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <p className="text-sm font-semibold text-blue-700">Onboarding</p>
            <p className="mt-1 text-sm text-blue-900/70">
              Re-run onboarding to initialize or rewrite Desktop Local Claw
              config files.
            </p>
          </div>
          <button
            type="button"
            className="rounded-2xl bg-slate-950 px-4 py-2.5 text-sm font-semibold text-white shadow-sm"
            onClick={onRunOnboarding}
          >
            Run onboarding again
          </button>
        </div>
      </div>
      <div className="mt-5">
        <RuntimeManagerPanel />
      </div>
    </PageFrame>
  )
}
