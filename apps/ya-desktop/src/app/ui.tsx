import {
  Folder,
  Network,
  ShieldCheck,
  TerminalSquare,
  type LucideIcon,
} from 'lucide-react'
import type { ReactNode } from 'react'

import type { ClawRunSummary, ClawSessionSummary } from '../claw'
import { cn } from '../lib'
import type { DesktopSpace, HomeStreamStatus } from './types'
import {
  executionLocationLabel,
  homeStreamStatusLabel,
  labelForStatus,
  relayLabel,
  sessionTitle,
  shellSafetyLabel,
  statusTone,
  statusToneName,
} from './utils'

export function LiveSessionList({
  connectionReady,
  loading,
  error,
  sessions,
  selectedSessionId,
  onSelectSession,
  compact,
  emptyTitle,
  emptyDetail,
}: {
  connectionReady: boolean
  loading: boolean
  error: Error | null
  sessions: ClawSessionSummary[]
  selectedSessionId?: string | null
  onSelectSession?: (sessionId: string) => void
  compact?: boolean
  emptyTitle: string
  emptyDetail: string
}) {
  if (!connectionReady)
    return (
      <EmptyState
        title="Local Claw is offline"
        detail="Open Settings and start Local Claw to load chats."
      />
    )
  if (loading)
    return (
      <EmptyState
        title="Loading chats"
        detail="Reading sessions from Local Claw."
      />
    )
  if (error)
    return <EmptyState title="Could not load chats" detail={error.message} />
  if (sessions.length === 0)
    return <EmptyState title={emptyTitle} detail={emptyDetail} />

  return (
    <div className="space-y-2">
      {sessions.map((session) => (
        <SessionRow
          key={session.id}
          compact={compact}
          onClick={
            onSelectSession ? () => onSelectSession(session.id) : undefined
          }
          selected={session.id === selectedSessionId}
          session={session}
        />
      ))}
    </div>
  )
}

export function SessionRow({
  session,
  compact,
  selected,
  onClick,
}: {
  session: ClawSessionSummary
  compact?: boolean
  selected?: boolean
  onClick?: () => void
}) {
  const latestRun = session.latest_run ?? session.latestRun
  const status = session.status
  return (
    <button
      type="button"
      className={cn(
        'flex w-full items-center gap-3 rounded-2xl border text-left transition',
        compact ? 'p-3' : 'p-4',
        selected
          ? 'border-[#171717] bg-white'
          : 'border-black/[0.08] bg-white hover:bg-[#fbfbfa]',
      )}
      onClick={onClick}
    >
      <span
        className={cn(
          'h-2.5 w-2.5 shrink-0 rounded-full',
          statusTone(statusToneName(status)),
        )}
      />
      <span className="min-w-0 flex-1">
        <span className="block truncate text-sm font-medium text-[#171717]">
          {sessionTitle(session)}
        </span>
        <span className="mt-1 block truncate text-xs text-[#6b6b6b]">
          {latestRun?.output_summary ??
            latestRun?.outputSummary ??
            latestRun?.input_preview ??
            latestRun?.inputPreview ??
            `${session.run_count ?? session.runCount ?? 0} runs`}
        </span>
      </span>
      <span className="rounded-full bg-[#f2f2ef] px-2.5 py-1 text-[11px] font-medium text-[#6b6b6b]">
        {labelForStatus(status)}
      </span>
    </button>
  )
}

export function HomeStreamPreview({
  eventCount,
  error,
  output,
  onOpenSession,
  runLabel,
  status,
}: {
  eventCount: number
  error: string | null
  output: string
  onOpenSession?: () => void
  runLabel: string | null
  status: HomeStreamStatus
}) {
  if (status === 'idle') return null
  const previewText =
    error ??
    (output.length > 0
      ? output
      : status === 'connecting'
        ? 'Opening a Claw run stream...'
        : 'Waiting for the first assistant chunk...')

  return (
    <div className="mt-3 rounded-2xl bg-[#fbfbfa] p-3 text-left ring-1 ring-black/[0.06]">
      <div className="flex items-center justify-between gap-2 text-xs">
        <div className="flex items-center gap-2 font-medium text-[#5f5f5f]">
          <span
            className={cn(
              'h-2 w-2 rounded-full',
              status === 'failed'
                ? statusTone('amber')
                : status === 'completed'
                  ? statusTone('emerald')
                  : statusTone('blue'),
            )}
          />
          {homeStreamStatusLabel(status)}
        </div>
        <div className="flex items-center gap-3">
          {onOpenSession && (
            <button
              className="font-medium text-[#171717] underline-offset-4 hover:underline"
              type="button"
              onClick={onOpenSession}
            >
              Open chat
            </button>
          )}
          <p className="text-[#9a9a9a]">
            {runLabel ? `Run ${runLabel}` : `${eventCount} events`}
          </p>
        </div>
      </div>
      <p
        className={cn(
          'mt-2 max-h-40 overflow-auto whitespace-pre-wrap text-sm leading-6',
          error ? 'text-amber-700' : 'text-[#5f5f5f]',
        )}
      >
        {previewText}
      </p>
    </div>
  )
}

export function ComposerFrame({
  children,
  compact,
}: {
  children: ReactNode
  compact?: boolean
}) {
  return (
    <div
      className={cn(
        'rounded-[1.6rem] border border-black/[0.08] bg-white shadow-[0_18px_50px_rgba(0,0,0,0.08)]',
        compact ? 'p-3' : 'p-4',
      )}
    >
      {children}
    </div>
  )
}

export function SelectPill({
  label,
  options,
  value,
  onChange,
}: {
  label: string
  options: Array<{ label: string; value: string }>
  value: string
  onChange: (value: string) => void
}) {
  return (
    <label className="inline-flex h-8 items-center gap-2 rounded-full bg-[#f2f2ef] px-3">
      <span className="text-[#8a8a8a]">{label}</span>
      <select
        className="max-w-48 bg-transparent font-medium text-[#171717] outline-none"
        value={value}
        onChange={(event) => onChange(event.target.value)}
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  )
}

export function InfoPill({
  icon: Icon,
  text,
}: {
  icon: LucideIcon
  text: string
}) {
  return (
    <span className="inline-flex h-8 max-w-64 items-center gap-2 rounded-full bg-[#f2f2ef] px-3">
      <Icon className="h-3.5 w-3.5 shrink-0 text-[#8a8a8a]" />
      <span className="truncate">{text}</span>
    </span>
  )
}

export function PageFrame({
  eyebrow,
  title,
  body,
  children,
}: {
  eyebrow: string
  title: string
  body: string
  children: ReactNode
}) {
  return (
    <div className="mx-auto w-full max-w-6xl px-5 py-8 lg:px-8">
      <div className="mb-6 max-w-3xl">
        <p className="text-xs font-semibold uppercase tracking-[0.16em] text-[#8a8a8a]">
          {eyebrow}
        </p>
        <h2 className="mt-2 text-3xl font-semibold tracking-[-0.035em] text-[#171717]">
          {title}
        </h2>
        <p className="mt-3 text-sm leading-6 text-[#6b6b6b]">{body}</p>
      </div>
      {children}
    </div>
  )
}

export function PageEmpty({
  title,
  detail,
}: {
  title: string
  detail: string
}) {
  return (
    <div className="flex min-h-full items-center justify-center px-5 py-10">
      <EmptyState title={title} detail={detail} />
    </div>
  )
}

export function EmptyState({
  title,
  detail,
}: {
  title: string
  detail: string
}) {
  return (
    <div className="rounded-2xl border border-dashed border-black/[0.10] bg-[#fbfbfa] p-5 text-center">
      <p className="text-sm font-semibold text-[#171717]">{title}</p>
      <p className="mt-2 text-xs leading-5 text-[#6b6b6b]">{detail}</p>
    </div>
  )
}

export function SettingCard({
  icon: Icon,
  title,
  detail,
}: {
  icon: LucideIcon
  title: string
  detail: string
}) {
  return (
    <div className="rounded-2xl border border-black/[0.08] bg-[#fbfbfa] p-4">
      <Icon className="h-5 w-5 text-[#5f5f5f]" />
      <h3 className="mt-3 text-sm font-semibold text-[#171717]">{title}</h3>
      <p className="mt-2 text-xs leading-5 text-[#6b6b6b]">{detail}</p>
    </div>
  )
}

export function PanelCard({
  icon: Icon,
  title,
  detail,
}: {
  icon: LucideIcon
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

export function SpaceCard({
  selected,
  space,
  onClick,
}: {
  selected: boolean
  space: DesktopSpace
  onClick: () => void
}) {
  return (
    <button
      className={cn(
        'rounded-2xl border p-4 text-left transition hover:bg-white',
        selected
          ? 'border-[#171717] bg-white'
          : 'border-black/[0.08] bg-[#fbfbfa]',
      )}
      onClick={onClick}
      type="button"
    >
      <div className="flex items-center justify-between gap-3">
        <Folder className="h-5 w-5 text-[#5f5f5f]" />
        <Chip>{executionLocationLabel(space)}</Chip>
      </div>
      <h3 className="mt-3 text-sm font-semibold text-[#171717]">
        {space.name}
      </h3>
      <p className="mt-2 truncate text-xs text-[#6b6b6b]">
        {space.path || 'Embedded local workspace'}
      </p>
      <div className="mt-4 space-y-2 text-xs text-[#6b6b6b]">
        <SpaceMetaRow
          icon={TerminalSquare}
          text={shellSafetyLabel(space.shellSafety)}
        />
        <SpaceMetaRow icon={ShieldCheck} text={space.trust} />
        <SpaceMetaRow icon={Network} text={relayLabel(space)} />
      </div>
      <div className="mt-4 flex flex-wrap gap-2">
        <Chip>{space.runtime}</Chip>
        <Chip>{space.kind.replaceAll('_', ' ')}</Chip>
        {selected && <Chip>Active</Chip>}
      </div>
    </button>
  )
}

export function RunRow({ run }: { run: ClawRunSummary }) {
  return (
    <div className="rounded-xl bg-white p-3">
      <div className="flex items-center gap-2">
        <span
          className={cn(
            'h-2 w-2 rounded-full',
            statusTone(statusToneName(run.status)),
          )}
        />
        <p className="truncate text-xs font-medium text-[#171717]">
          Run #{run.sequence_no ?? run.sequenceNo ?? '—'} · {run.status}
        </p>
      </div>
      <p className="mt-1 truncate text-xs text-[#6b6b6b]">
        {run.output_summary ??
          run.outputSummary ??
          run.error_message ??
          run.errorMessage ??
          run.input_preview ??
          run.inputPreview ??
          'No summary'}
      </p>
    </div>
  )
}

export function Chip({ children }: { children: ReactNode }) {
  return (
    <span className="rounded-full bg-[#f2f2ef] px-2.5 py-1 text-[11px] font-medium text-[#6b6b6b]">
      {children}
    </span>
  )
}

function SpaceMetaRow({
  icon: Icon,
  text,
}: {
  icon: LucideIcon
  text: string
}) {
  return (
    <div className="flex min-w-0 items-center gap-2">
      <Icon className="h-3.5 w-3.5 shrink-0 text-[#8a8a8a]" />
      <span className="truncate">{text}</span>
    </div>
  )
}

export function IconButton({
  label,
  icon: Icon,
  onClick,
}: {
  label: string
  icon: LucideIcon
  onClick: () => void
}) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl text-[#6b6b6b] transition hover:bg-white hover:text-[#171717] hover:shadow-sm"
      onClick={onClick}
    >
      <Icon className="h-4 w-4" />
    </button>
  )
}
