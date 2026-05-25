import type { KeyboardEvent } from 'react'

import {
  collectTextFromReplay,
  type ClawActiveInteraction,
  type ClawProfileSummary,
  type ClawRunStatus,
  type ClawSessionDetail,
  type ClawSessionStatus,
  type ClawSessionSummary,
  type ClawWorkspaceBinding,
  type JsonObject,
} from '../claw'

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
        metadata: {
          desktop_space_id: space.id,
          desktop_trust_level: space.trustLevel,
          desktop_shell_mode: space.shellSafety.mode,
          desktop_relay_enabled: space.relay.enabled,
          desktop_relay_protocol: space.relay.protocol,
        },
      },
    ],
    default_mount_id: 'main',
    cwd: virtualPath,
    metadata: {
      desktop_space_id: space.id,
      desktop_space_name: space.name,
      desktop_execution_location: space.executionLocation,
      desktop_trust_level: space.trustLevel,
      desktop_shell_safety: space.shellSafety,
      desktop_relay: space.relay,
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

export type DesktopInboxItem = {
  id: string
  title: string
  detail: string
  tone: string
  session: ClawSessionSummary
  interaction?: ClawActiveInteraction
  runId?: string
  interactionId?: string
}

export function inboxItemsFromSessions(sessions: ClawSessionSummary[]) {
  return sessions.flatMap((session): DesktopInboxItem[] => {
    const detail = session.status_detail ?? session.statusDetail ?? {}
    const activeInteractions = activeInteractionsFromDetail(detail)
    const items: DesktopInboxItem[] = activeInteractions.map((interaction) => {
      const interactionId =
        interaction.interaction_id ?? interaction.interactionId
      const runId =
        interaction.run_id ??
        interaction.runId ??
        session.active_run_id ??
        session.activeRunId ??
        session.latest_run?.id ??
        session.latestRun?.id
      return {
        id: `${session.id}-${interactionId ?? interaction.tool_call_id ?? interaction.toolCallId ?? 'approval'}`,
        title: `${interaction.title ?? 'Approval needed'} · ${sessionTitle(session)}`,
        detail: interaction.description ?? interactionDetail(interaction),
        tone: 'blue',
        session,
        interaction,
        runId,
        interactionId,
      }
    })
    const activeInteractionCount = activeInteractionCountFromDetail(detail)
    if (
      activeInteractions.length === 0 &&
      (session.status_reason === 'hitl_pending' ||
        session.statusReason === 'hitl_pending' ||
        (typeof activeInteractionCount === 'number' &&
          activeInteractionCount > 0))
    ) {
      items.push({
        id: `${session.id}-approval`,
        title: `Approval needed · ${sessionTitle(session)}`,
        detail: `${activeInteractionCount || 1} active interactions waiting for a decision.`,
        tone: 'blue',
        session,
      })
    }
    if (session.status === 'failed' || session.status === 'interrupted') {
      const latestRun = session.latest_run ?? session.latestRun
      items.push({
        id: `${session.id}-${session.status}`,
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

function activeInteractionsFromDetail(
  detail: JsonObject,
): ClawActiveInteraction[] {
  const snake = detail.active_interactions
  if (Array.isArray(snake)) return snake.filter(isActiveInteraction)
  const camel = detail.activeInteractions
  if (Array.isArray(camel)) return camel.filter(isActiveInteraction)
  return []
}

function activeInteractionCountFromDetail(detail: JsonObject) {
  if (typeof detail.active_interaction_count === 'number')
    return detail.active_interaction_count
  if (typeof detail.activeInteractionCount === 'number')
    return detail.activeInteractionCount
  return undefined
}

function isActiveInteraction(value: unknown): value is ClawActiveInteraction {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value))
}

function interactionDetail(interaction: ClawActiveInteraction) {
  const toolName = interaction.tool_name ?? interaction.toolName ?? 'tool call'
  const sequence = interaction.sequence_no ?? interaction.sequenceNo
  const total = interaction.total_count ?? interaction.totalCount
  const countDetail = sequence && total ? ` · ${sequence}/${total}` : ''
  return `${toolName}${countDetail} is waiting for your decision.`
}

export function folderName(path: string) {
  return path.split('/').filter(Boolean).at(-1) ?? 'Workspace'
}

export function spaceDetail(space: DesktopSpace) {
  return space.path
    ? `${space.name} · ${space.path}`
    : `${space.name} · embedded desktop workspace`
}

export function shellSafetyLabel(policy: DesktopSpace['shellSafety']) {
  const mode = policy.mode.replaceAll('_', ' ')
  const approval = policy.approvalPolicy === 'defer' ? 'approval' : 'deny'
  return `${mode} · ${policy.reviewRiskThreshold}+ ${approval} · unattended ${policy.unattendedRiskThreshold}+`
}

export function relayLabel(space: DesktopSpace) {
  return space.relay.enabled
    ? `${space.relay.protocol} · ${space.relay.capabilities.join(', ')}`
    : `${space.relay.protocol} · local disabled`
}

export function executionLocationLabel(space: DesktopSpace) {
  if (space.executionLocation === 'cloud_workspace') return 'Cloud workspace'
  if (space.executionLocation === 'remote_claw') return 'Remote Claw'
  return 'This device'
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
