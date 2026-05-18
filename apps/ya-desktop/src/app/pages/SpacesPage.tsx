import { useState, type FormEvent } from 'react'

import type { DesktopSpace } from '../types'
import { PageFrame, SpaceCard } from '../ui'
import { folderName } from '../utils'

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

  const [formError, setFormError] = useState<string | null>(null)

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
    const space: DesktopSpace = {
      id: `local-${Date.now()}`,
      name: normalizedName,
      path: normalizedPath,
      runtime: 'Local Claw',
      trust: 'Trusted',
      default: false,
    }
    onAddSpace(space)
    onSelectSpace(space.id)
    setName('')
    setPath('')
    setFormError(null)
  }

  return (
    <PageFrame
      eyebrow="Spaces"
      title="Workspaces"
      body="A Space keeps folder, trust, runtime location, and execution context together."
    >
      <form
        className="grid gap-3 rounded-2xl border border-black/[0.08] bg-[#fbfbfa] p-3 md:grid-cols-[1fr_1.4fr_auto]"
        onSubmit={handleAddSpace}
      >
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
        <button
          className="h-11 rounded-xl bg-[#171717] px-4 text-sm font-medium text-white"
          type="submit"
        >
          Add space
        </button>
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
