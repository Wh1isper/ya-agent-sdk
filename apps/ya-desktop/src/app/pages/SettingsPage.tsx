import { KeyRound, Settings, SlidersHorizontal } from 'lucide-react'

import { RuntimeManagerPanel } from '../../runtime/RuntimeManagerPanel'
import { PageFrame, SettingCard } from '../ui'

export function SettingsPage() {
  return (
    <PageFrame
      eyebrow="Settings"
      title="Preferences and runtime"
      body="Keep everyday controls simple and place lower-level runtime tools behind clear sections."
    >
      <div className="grid gap-3 md:grid-cols-3">
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
      </div>
      <div className="mt-5">
        <RuntimeManagerPanel />
      </div>
    </PageFrame>
  )
}
