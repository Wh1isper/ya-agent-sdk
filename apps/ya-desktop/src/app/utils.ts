import type { KeyboardEvent } from 'react'

import { collectTextFromReplay, type ClawProfileSummary, type ClawRunStatus, type ClawSessionDetail, type ClawSessionStatus, type ClawSessionSummary, type ClawWorkspaceBinding, type JsonObject } from '../claw'

import type { DesktopSpace, HomeStreamStatus } from './types'

export function enabledProfiles(profiles: ClawProfileSummary[]) {
  return profiles.filter((profile) => profile.enabled)
}

export function profileNameOrDefault(
  selectedProfileName: string,
  profiles: ClawProfileSummary[],
) {
  if (profiles.some((profile) => profile.name === selectedProfileName))
    return selectedProfileName
  return profiles[0]?.name ?? 'default'
}

export function workspaceBindingFromSpace(
  space: DesktopSpace,
): ClawWorkspaceBinding | null {
  if (!space.path.trim()) return null
  const virtualPath = '/workspace/main'
  return {
    mounts: [
      {
        id: 'main',
        name: space.name,
        host_path: space.path,
        virtual_path: virtualPath,
        mode: 'rw',
        metadata: { desktop_space_id: space.id },
      },
    ],
    default_mount_id: 'main',
    cwd: virtualPath,
    metadata: {
      desktop_space_id: space.id,
      desktop_space_name: space.name,
    },
  }
}

export function workspaceFromSession(session: ClawSessionDetail | null) {
  const workspace =
    session?.workspace_state?.workspace ?? session?.workspaceState?.workspace
  if (isWorkspaceBinding(workspace)) return workspace
  return null
}

function isWorkspaceBinding(value: unknown): value is ClawWorkspaceBinding {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false
  const candidate = value as Partial<ClawWorkspaceBinding>
  return Array.isArray(candidate.mounts) && typeof candidate.cwd === 'string'
}

export function desktopSpaceMetadataFromWorkspace(
  workspace: ClawWorkspaceBinding | null,
  fallbackSpace: DesktopSpace,
) {
  const metadata = workspace?.metadata ?? {}
  return {
    spaceId:
      typeof metadata.desktop_space_id === 'string'
        ? metadata.desktop_space_id
        : fallbackSpace.id,
    spaceName:
      typeof metadata.desktop_space_name === 'string'
        ? metadata.desktop_space_name
        : fallbackSpace.name,
  }
}

export function collectCommittedReplayText(
  session: ClawSessionDetail | null,
  replayMessage: JsonObject[] | null,
) {
  const topLevelText = collectTextFromReplay(replayMessage).trim()
  if (topLevelText) return topLevelText
  if (!session?.runs) return ''
  return session.runs
    .map((run) => collectTextFromReplay(run.message).trim())
    .filter(Boolean)
    .join('\n\n')
}

export function submitFormOnEnter(event: KeyboardEvent<HTMLTextAreaElement>) {
  if (
    event.key !== 'Enter' ||
    event.shiftKey ||
    event.nativeEvent.isComposing
  ) {
    return
  }
  event.preventDefault()
  event.currentTarget.form?.requestSubmit()
}

export function inputPartsPreview(parts?: JsonObject[] | null) {
  if (!parts?.length) return 'Input parts'
  return parts
    .map((part) =>
      typeof part.text === 'string' ? part.text : JSON.stringify(part),
    )
    .join('\n')
}

export function groupSessionsForBoard(sessions: ClawSessionSummary[]) {
  const waiting = sessions.filter(
    (session) => session.status === 'interrupted' || isHitlPending(session),
  )
  const active = sessions.filter(
    (session) =>
      !isHitlPending(session) &&
      session.status !== 'interrupted' &&
      ['queued', 'running'].includes(session.status),
  )
  const failed = sessions.filter(
    (session) => session.status === 'failed' && !isHitlPending(session),
  )
  const done = sessions.filter((session) =>
    ['completed', 'idle', 'cancelled'].includes(session.status),
  )
  return [
    { title: 'Active', items: active },
    { title: 'Waiting', items: waiting },
    { title: 'Done', items: done },
    { title: 'Failed', items: failed },
  ]
}

function isHitlPending(session: ClawSessionSummary) {
  return (
    session.status_reason === 'hitl_pending' ||
    session.statusReason === 'hitl_pending'
  )
}

export function inboxItemsFromSessions(sessions: ClawSessionSummary[]) {
  return sessions.flatMap((session) => {
    const detail = session.status_detail ?? session.statusDetail ?? {}
    const activeInteractionCount =
      typeof detail.active_interaction_count === 'number'
        ? detail.active_interaction_count
        : typeof detail.activeInteractionCount === 'number'
          ? detail.activeInteractionCount
          : undefined
    const items: Array<{
      title: string
      detail: string
      tone: string
      session: ClawSessionSummary
    }> = []
    if (
      session.status_reason === 'hitl_pending' ||
      session.statusReason === 'hitl_pending' ||
      (typeof activeInteractionCount === 'number' && activeInteractionCount > 0)
    ) {
      items.push({
        title: `Approval needed · ${sessionTitle(session)}`,
        detail: `${activeInteractionCount || 1} active interactions waiting for a decision.`,
        tone: 'blue',
        session,
      })
    }
    if (session.status === 'failed' || session.status === 'interrupted') {
      const latestRun = session.latest_run ?? session.latestRun
      items.push({
        title: `${labelForStatus(session.status)} · ${sessionTitle(session)}`,
        detail:
          latestRun?.error_message ??
          latestRun?.errorMessage ??
          String(detail.error_message ?? 'Open the chat to recover this run.'),
        tone: 'amber',
        session,
      })
    }
    return items
  })
}

export function folderName(path: string) {
  return path.split('/').filter(Boolean).at(-1) ?? 'Workspace'
}

export function spaceDetail(space: DesktopSpace) {
  return space.path
    ? `${space.name} · ${space.path}`
    : `${space.name} · embedded`
}

export function sessionTitle(session: ClawSessionSummary) {
  const latestRun = session.latest_run ?? session.latestRun
  const metadataTitle = session.metadata?.title
  if (typeof metadataTitle === 'string' && metadataTitle.trim())
    return metadataTitle
  return (
    latestRun?.input_preview ??
    latestRun?.inputPreview ??
    `Session ${session.id.slice(0, 8)}`
  )
}

export function labelForStatus(status: ClawSessionStatus | ClawRunStatus) {
  const normalized = String(status)
  return (
    normalized.charAt(0).toUpperCase() +
    normalized.slice(1).replaceAll('_', ' ')
  )
}

export function homeStreamStatusLabel(status: HomeStreamStatus) {
  if (status === 'connecting') return 'Connecting to Claw'
  if (status === 'streaming') return 'Streaming output'
  if (status === 'completed') return 'Run completed'
  if (status === 'failed') return 'Needs attention'
  return 'Ready'
}

export function statusToneName(status: ClawSessionStatus | ClawRunStatus) {
  if (status === 'queued' || status === 'running') return 'blue'
  if (status === 'failed' || status === 'interrupted') return 'amber'
  if (status === 'cancelled') return 'slate'
  return 'emerald'
}

export function formatDate(value?: string | null) {
  if (!value) return 'No date'
  try {
    return new Intl.DateTimeFormat(undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    }).format(new Date(value))
  } catch {
    return value
  }
}

export function statusTone(tone: string) {
  if (tone === 'blue')
    return 'bg-blue-500 shadow-[0_0_0_4px_rgba(59,130,246,.10)]'
  if (tone === 'amber')
    return 'bg-amber-500 shadow-[0_0_0_4px_rgba(245,158,11,.12)]'
  if (tone === 'emerald')
    return 'bg-emerald-500 shadow-[0_0_0_4px_rgba(16,185,129,.12)]'
  return 'bg-[#9a9a9a] shadow-[0_0_0_4px_rgba(100,116,139,.10)]'
}
