import { describe, expect, it } from 'vitest'

import type { WorkspaceFileEntry } from './types'
import {
  groupWorkspaceArtifacts,
  isMemoryEventFile,
  joinVirtualPath,
} from './workspaceView'

function entry(
  name: string,
  kind: WorkspaceFileEntry['kind'] = 'file',
): WorkspaceFileEntry {
  return {
    name,
    path: `/workspace/${name}`,
    kind,
    size_bytes: kind === 'file' ? 12 : null,
    modified_at: null,
    hidden: false,
  }
}

describe('workspace view helpers', () => {
  it('joins child paths without escaping the virtual root', () => {
    expect(joinVirtualPath('/workspace', 'memory')).toBe('/workspace/memory')
    expect(joinVirtualPath('/', 'memory')).toBe('/memory')
  })

  it('recognizes only protocol-named memory event files', () => {
    expect(isMemoryEventFile(entry('20260711-event.md'))).toBe(true)
    expect(isMemoryEventFile(entry('20260711-notes.md'))).toBe(false)
    expect(isMemoryEventFile(entry('20260711-event.md', 'directory'))).toBe(
      false,
    )
  })

  it('groups artifacts by explainable names and extensions while excluding memory', () => {
    const groups = groupWorkspaceArtifacts([
      entry('memory', 'directory'),
      entry('outputs', 'directory'),
      entry('summary.md'),
      entry('records.csv'),
      entry('chart.png'),
      entry('worker.py'),
      entry('misc.bin'),
    ])

    expect(groups.map((group) => group.id)).toEqual([
      'generated',
      'documents',
      'data',
      'media',
      'code',
      'other',
    ])
    expect(
      groups.flatMap((group) => group.items.map((item) => item.name)),
    ).not.toContain('memory')
  })
})
