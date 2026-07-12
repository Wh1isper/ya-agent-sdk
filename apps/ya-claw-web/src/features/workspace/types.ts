export type WorkspaceFileKind = 'file' | 'directory' | 'symlink' | 'other'

export type WorkspaceFileEntry = {
  name: string
  path: string
  kind: WorkspaceFileKind
  size_bytes: number | null
  modified_at: string | null
  hidden: boolean
}

export type WorkspaceFileListResponse = {
  session_id: string
  path: string
  items: WorkspaceFileEntry[]
  limit: number
  offset: number
  has_more: boolean
  next_cursor: string | null
  /** Backwards-compatible continuation for older offset-based clients. */
  next_offset: number | null
  /** Backwards-compatible alias for has_more. */
  truncated: boolean
}

export type WorkspaceFileContentResponse = {
  session_id: string
  path: string
  content: string
  encoding: 'utf-8'
  size_bytes: number
}
