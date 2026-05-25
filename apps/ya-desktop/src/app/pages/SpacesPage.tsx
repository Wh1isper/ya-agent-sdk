import {
  FolderOpen,
  HardDrive,
  Network,
  ShieldCheck,
  TerminalSquare,
} from 'lucide-react'
import { useState, type FormEvent } from 'react'

import { defaultShellSafetyPolicy } from '../constants'
import type {
  DesktopShellSafetyMode,
  DesktopSpace,
  DesktopTrustLevel,
} from '../types'
import { Chip, PageFrame, SpaceCard } from '../ui'
import { folderName, shellSafetyLabel, spaceDetail } from '../utils'

export function SpacesPage({
  selectedSpaceId,
  spaces,
  onAddSpace,
  onSelectSpace,
}: {
  selectedSpaceId: string
  spaces: DesktopSpace[]
  onAddSpace: (space: DesktopSpace) => void
  onSelectSpace: (spaceId: string) => void
}) {
  const [name, setName] = useState('')
  const [path, setPath] = useState('')
  const [trustLevel, setTrustLevel] = useState<DesktopTrustLevel>('trusted')
  const [shellMode, setShellMode] =
    useState<DesktopShellSafetyMode>('review_then_run')
  const [relayEnabled, setRelayEnabled] = useState(false)
  const [formError, setFormError] = useState<string | null>(null)

  const selectedSpace =
    spaces.find((space) => space.id === selectedSpaceId) ?? spaces[0]

  function handleAddSpace(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const normalizedPath = path.trim()
    if (!normalizedPath) return
    if (!normalizedPath.startsWith('/')) {
      setFormError('Use an absolute workspace path.')
      return
    }
    if (spaces.some((space) => space.path === normalizedPath)) {
      setFormError('This workspace is already in Spaces.')
      return
    }
    const normalizedName = name.trim() || folderName(normalizedPath)
    const effectiveTrustLevel =
      shellMode === 'read_only_shell' ? 'read_only' : trustLevel
    const space: DesktopSpace = {
      id: `local-${Date.now()}`,
      name: normalizedName,
      path: normalizedPath,
      runtime: 'Local Claw',
      trust: trustLabel(effectiveTrustLevel, shellMode),
      default: false,
      kind: 'local_folder',
      executionLocation: 'this_device',
      trustLevel: effectiveTrustLevel,
      shellSafety: {
        ...defaultShellSafetyPolicy,
        mode: shellMode,
      },
      relay: {
        enabled: relayEnabled,
        connectionId: null,
        protocol: 'ya-environment-relay.v1',
        capabilities: relayEnabled
          ? ['fileops', 'shell', 'tools', 'resources', 'artifacts']
          : ['fileops', 'shell', 'artifacts'],
      },
    }
    onAddSpace(space)
    onSelectSpace(space.id)
    setName('')
    setPath('')
    setTrustLevel('trusted')
    setShellMode('review_then_run')
    setRelayEnabled(false)
    setFormError(null)
  }

  return (
    <PageFrame
      eyebrow="Spaces"
      title="Desktop workspaces"
      body="A Space binds a folder, local execution policy, shell safety, and future relay grants for central Claw agents."
    >
      {selectedSpace && (
        <section className="mb-5 grid gap-3 lg:grid-cols-4">
          <InfoCard
            icon={HardDrive}
            title="Active workspace"
            detail={spaceDetail(selectedSpace)}
          />
          <InfoCard
            icon={TerminalSquare}
            title="Shell safety"
            detail={shellSafetyLabel(selectedSpace.shellSafety)}
          />
          <InfoCard
            icon={ShieldCheck}
            title="Trust"
            detail={selectedSpace.trust}
          />
          <InfoCard
            icon={Network}
            title="Relay"
            detail={
              selectedSpace.relay.enabled
                ? `${selectedSpace.relay.protocol} · ready to grant`
                : `${selectedSpace.relay.protocol} · local only`
            }
          />
        </section>
      )}

      <form
        className="rounded-2xl border border-black/[0.08] bg-[#fbfbfa] p-4"
        onSubmit={handleAddSpace}
      >
        <div className="grid gap-3 md:grid-cols-[1fr_1.4fr]">
          <input
            value={name}
            onChange={(event) => setName(event.target.value)}
            aria-label="Space name"
            className="h-11 rounded-xl border border-black/[0.08] bg-white px-3 text-sm outline-none transition focus:border-[#171717]"
            placeholder="Space name"
          />
          <input
            value={path}
            onChange={(event) => {
              setPath(event.target.value)
              setFormError(null)
            }}
            aria-label="Workspace folder path"
            className="h-11 rounded-xl border border-black/[0.08] bg-white px-3 text-sm outline-none transition focus:border-[#171717]"
            placeholder="/absolute/path/to/workspace"
          />
        </div>
        <div className="mt-3 grid gap-3 lg:grid-cols-[1fr_1fr_auto]">
          <label className="text-xs font-medium text-[#6b6b6b]">
            Trust level
            <select
              className="mt-1 h-11 w-full rounded-xl border border-black/[0.08] bg-white px-3 text-sm text-[#171717] outline-none"
              value={trustLevel}
              onChange={(event) =>
                setTrustLevel(event.target.value as DesktopTrustLevel)
              }
            >
              <option value="trusted">Trusted workspace</option>
              <option value="ask_before_write">
                Ask before write-heavy work
              </option>
              <option value="read_only">Read-only workspace</option>
            </select>
          </label>
          <label className="text-xs font-medium text-[#6b6b6b]">
            Shell mode
            <select
              className="mt-1 h-11 w-full rounded-xl border border-black/[0.08] bg-white px-3 text-sm text-[#171717] outline-none"
              value={shellMode}
              onChange={(event) =>
                setShellMode(event.target.value as DesktopShellSafetyMode)
              }
            >
              <option value="review_then_run">Review risky commands</option>
              <option value="read_only_shell">Read-only shell</option>
              <option value="disabled">Disable shell</option>
            </select>
          </label>
          <label className="flex h-11 items-center gap-2 self-end rounded-xl border border-black/[0.08] bg-white px-3 text-sm text-[#171717]">
            <input
              type="checkbox"
              checked={relayEnabled}
              onChange={(event) => setRelayEnabled(event.target.checked)}
            />
            Relay ready
          </label>
        </div>
        <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap gap-2">
            <Chip>Local execution</Chip>
            <Chip>Shell review default</Chip>
            <Chip>
              {relayEnabled ? 'Relay grant candidate' : 'Desktop only'}
            </Chip>
          </div>
          <button
            className="h-11 rounded-xl bg-[#171717] px-4 text-sm font-medium text-white"
            type="submit"
          >
            Add space
          </button>
        </div>
      </form>
      {formError && <p className="mt-2 text-sm text-amber-700">{formError}</p>}
      <div className="mt-5 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        {spaces.map((space) => (
          <SpaceCard
            key={space.id}
            selected={space.id === selectedSpaceId}
            space={space}
            onClick={() => onSelectSpace(space.id)}
          />
        ))}
      </div>
    </PageFrame>
  )
}

function InfoCard({
  icon: Icon,
  title,
  detail,
}: {
  icon: typeof FolderOpen
  title: string
  detail: string
}) {
  return (
    <div className="rounded-2xl border border-black/[0.08] bg-white p-4">
      <Icon className="h-4 w-4 text-[#6b6b6b]" />
      <p className="mt-3 text-xs font-medium uppercase tracking-[0.14em] text-[#8a8a8a]">
        {title}
      </p>
      <p className="mt-1 text-sm leading-5 text-[#171717]">{detail}</p>
    </div>
  )
}

function trustLabel(
  trustLevel: DesktopTrustLevel,
  shellMode: DesktopShellSafetyMode,
) {
  if (trustLevel === 'read_only') return 'Read-only'
  if (shellMode === 'disabled') return 'Trusted · Shell disabled'
  if (shellMode === 'read_only_shell') return 'Read-only shell'
  if (trustLevel === 'ask_before_write')
    return 'Ask before write · Shell review'
  return 'Trusted · Shell review'
}
