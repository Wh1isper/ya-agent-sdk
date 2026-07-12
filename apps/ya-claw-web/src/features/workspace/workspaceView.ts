import type { WorkspaceFileEntry } from './types'

export type WorkspaceSection = 'files' | 'memory' | 'artifacts'

export type ArtifactGroup = {
  id: string
  label: string
  description: string
  items: WorkspaceFileEntry[]
}

const documentExtensions = new Set([
  'md',
  'txt',
  'pdf',
  'html',
  'doc',
  'docx',
  'ppt',
  'pptx',
])
const dataExtensions = new Set([
  'csv',
  'json',
  'jsonl',
  'parquet',
  'tsv',
  'xml',
  'xls',
  'xlsx',
  'yaml',
  'yml',
])
const mediaExtensions = new Set([
  'gif',
  'jpeg',
  'jpg',
  'mp3',
  'mp4',
  'ogg',
  'png',
  'svg',
  'wav',
  'webm',
  'webp',
])
const codeAndArchiveExtensions = new Set([
  'css',
  'go',
  'gz',
  'js',
  'jsx',
  'py',
  'rs',
  'sh',
  'tar',
  'toml',
  'ts',
  'tsx',
  'zip',
])
const generatedDirectoryNames = new Set([
  'artifact',
  'artifacts',
  'export',
  'exports',
  'output',
  'outputs',
  'report',
  'reports',
  'result',
  'results',
])

const groupDefinitions = [
  {
    id: 'generated',
    label: 'Generated output folders',
    description:
      'Folders named output, artifacts, reports, results, or exports.',
  },
  {
    id: 'documents',
    label: 'Reports & documents',
    description:
      'Readable document extensions such as Markdown, text, HTML, and PDF.',
  },
  {
    id: 'data',
    label: 'Data & exports',
    description:
      'Structured data extensions such as CSV, JSON, YAML, and spreadsheets.',
  },
  {
    id: 'media',
    label: 'Media',
    description: 'Image, audio, and video extensions.',
  },
  {
    id: 'code',
    label: 'Code & archives',
    description: 'Source code, scripts, configuration, and archive extensions.',
  },
  {
    id: 'folders',
    label: 'Other folders',
    description:
      'Workspace folders that do not use a conventional output name.',
  },
  {
    id: 'other',
    label: 'Other files',
    description: 'Files whose extension does not match another group.',
  },
] as const

export function joinVirtualPath(root: string, child: string) {
  return root === '/' ? `/${child}` : `${root.replace(/\/$/, '')}/${child}`
}

export function isMemoryEventFile(entry: WorkspaceFileEntry) {
  return entry.kind === 'file' && /^\d{8}-event\.md$/i.test(entry.name)
}

export function groupWorkspaceArtifacts(
  entries: WorkspaceFileEntry[],
): ArtifactGroup[] {
  const grouped = new Map<string, WorkspaceFileEntry[]>()
  for (const entry of entries) {
    if (entry.name.toLowerCase() === 'memory') continue
    const id = artifactGroupId(entry)
    grouped.set(id, [...(grouped.get(id) ?? []), entry])
  }
  return groupDefinitions.flatMap((definition) => {
    const items = grouped.get(definition.id)
    return items?.length ? [{ ...definition, items }] : []
  })
}

function artifactGroupId(entry: WorkspaceFileEntry) {
  const lowerName = entry.name.toLowerCase()
  if (entry.kind === 'directory') {
    return generatedDirectoryNames.has(lowerName) ? 'generated' : 'folders'
  }
  const extension = lowerName.includes('.')
    ? (lowerName.split('.').pop() ?? '')
    : ''
  if (documentExtensions.has(extension)) return 'documents'
  if (dataExtensions.has(extension)) return 'data'
  if (mediaExtensions.has(extension)) return 'media'
  if (codeAndArchiveExtensions.has(extension)) return 'code'
  return 'other'
}
