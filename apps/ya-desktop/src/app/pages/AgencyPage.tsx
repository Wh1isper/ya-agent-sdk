import { BrainCircuit, Database, FileText, RefreshCw, Sparkles } from 'lucide-react'
import { useMemo } from 'react'

import {
  useActiveClawConnection,
  useClawAgencyConfig,
  useClawAgencyFires,
  useClawAgencyStatus,
  useClawSessions,
  type ClawAgencyFire,
  type ClawSessionSummary,
} from '../../claw'
import { cn } from '../../lib'
import { EmptyState, PageFrame } from '../ui'
import { labelForStatus, sessionTitle, statusTone, statusToneName } from '../utils'

export function AgencyPage({ onOpenSession }: { onOpenSession: (sessionId: string) => void }) {
  const activeConnectionQuery = useActiveClawConnection()
  const connection = activeConnectionQuery.data?.connection ?? null
  const config = useClawAgencyConfig(connection)
  const status = useClawAgencyStatus(connection)
  const fires = useClawAgencyFires(connection)
  const sessions = useClawSessions(connection)

  const agencySessionId =
    status.data?.agency_session_id ??
    status.data?.agencySessionId ??
    config.data?.agency_session_id ??
    config.data?.agencySessionId ??
    null
  const agencySession = status.data?.agency_session ?? status.data?.agencySession ?? null
  const agencyRuns = agencySession?.run_count ?? agencySession?.runCount ?? 0
  const pendingFires = status.data?.pending_fire_count ?? status.data?.pendingFireCount ?? 0
  const latestRun = status.data?.latest_run ?? status.data?.latestRun ?? null
  const activeRun = status.data?.active_run ?? status.data?.activeRun ?? null
  const state = status.data?.state ?? 'idle'
  const memorySessions = useMemo(
    () => (sessions.data ?? []).filter((session) => sessionType(session) === 'memory'),
    [sessions.data],
  )
  const recentFires = fires.data?.fires ?? []

  async function refreshAll() {
    await Promise.all([
      config.refetch(),
      status.refetch(),
      fires.refetch(),
      sessions.refetch(),
    ])
  }

  return (
    <PageFrame
      eyebrow="Agency"
      title="Proactive work and memory"
      body="Agency watches copied chat inputs and completed memory sessions, then coordinates bounded proactive work through the local Claw runtime."
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap gap-2 text-xs text-[#5f5f5f]">
          <StatusPill label="Agency" value={config.data?.enabled ? 'Enabled' : 'Disabled'} tone={config.data?.enabled ? 'emerald' : 'slate'} />
          <StatusPill label="State" value={labelForStatus(state)} tone={statusToneName(state)} />
          <StatusPill label="Memory" value={`${memorySessions.length} sessions`} tone="blue" />
        </div>
        <button
          type="button"
          className="inline-flex items-center gap-2 rounded-2xl border border-black/[0.08] bg-white px-4 py-2.5 text-sm font-semibold text-[#171717] shadow-sm transition hover:bg-[#f2f2ef]"
          onClick={() => void refreshAll()}
        >
          <RefreshCw className="h-4 w-4" />
          Refresh
        </button>
      </div>

      {!connection ? (
        <div className="mt-5">
          <EmptyState
            title="Local Claw is offline"
            detail="Start Local Claw from Settings to inspect Agency and memory sessions."
          />
        </div>
      ) : (
        <div className="mt-5 grid gap-5 xl:grid-cols-[1.15fr_0.85fr]">
          <section className="space-y-5">
            <div className="grid gap-3 md:grid-cols-3">
              <MetricCard icon={BrainCircuit} label="Agency session" value={agencySessionId ? shortId(agencySessionId) : 'Pending'} detail={`${agencyRuns} runs`} />
              <MetricCard icon={Sparkles} label="Pending fires" value={String(pendingFires)} detail="Copied inputs waiting for delivery" />
              <MetricCard icon={Database} label="Memory sessions" value={String(memorySessions.length)} detail="Background extract and summary work" />
            </div>

            <section className="rounded-[2rem] border border-black/[0.06] bg-white p-6 shadow-sm">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="text-sm font-semibold text-blue-600">Agency runtime</p>
                  <h2 className="mt-2 text-2xl font-semibold tracking-[-0.03em] text-[#171717]">
                    Singleton coordinator
                  </h2>
                </div>
                {agencySessionId && (
                  <button
                    type="button"
                    className="rounded-2xl bg-[#171717] px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-black"
                    onClick={() => onOpenSession(agencySessionId)}
                  >
                    Open session
                  </button>
                )}
              </div>
              <div className="mt-5 grid gap-3 md:grid-cols-2">
                <InfoRow label="Profile" value={config.data?.profile_name ?? config.data?.profileName ?? 'default'} />
                <InfoRow label="Timer" value={`${config.data?.timer_interval_seconds ?? config.data?.timerIntervalSeconds ?? 0}s`} />
                <InfoRow label="Risk" value={config.data?.risk_policy?.max_auto_action_risk ?? config.data?.riskPolicy?.maxAutoActionRisk ?? 'extra_high'} />
                <InfoRow label="Next fire" value={config.data?.next_fire_at ?? config.data?.nextFireAt ?? 'Not scheduled'} />
              </div>
              <div className="mt-5 rounded-3xl bg-[#f7f7f4] p-4">
                <p className="text-sm font-semibold text-[#171717]">Durable Agency files</p>
                <div className="mt-3 grid gap-2 md:grid-cols-2">
                  {Object.entries(config.data?.memory_files ?? config.data?.memoryFiles ?? {}).map(([label, path]) => (
                    <div key={label} className="rounded-2xl bg-white px-3 py-2 text-sm text-[#5f5f5f] ring-1 ring-black/[0.05]">
                      <span className="font-medium text-[#171717]">{label}</span>
                      <span className="mt-1 block truncate font-mono text-xs">{path}</span>
                    </div>
                  ))}
                </div>
              </div>
            </section>

            <section className="rounded-[2rem] border border-black/[0.06] bg-white p-6 shadow-sm">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="text-sm font-semibold text-blue-600">Memory sessions</p>
                  <h2 className="mt-2 text-2xl font-semibold tracking-[-0.03em] text-[#171717]">
                    Extract and summary jobs
                  </h2>
                </div>
                <FileText className="h-5 w-5 text-[#8a8a8a]" />
              </div>
              <div className="mt-4 space-y-2">
                {memorySessions.slice(0, 8).map((session) => (
                  <SessionLink key={session.id} session={session} onOpen={() => onOpenSession(session.id)} />
                ))}
                {memorySessions.length === 0 && (
                  <p className="rounded-2xl bg-[#f7f7f4] px-4 py-3 text-sm text-[#6b6b6b]">
                    Memory is enabled by Desktop launch defaults. Memory sessions appear after chat turns are processed.
                  </p>
                )}
              </div>
            </section>
          </section>

          <aside className="space-y-5">
            <RunCard title="Active run" run={activeRun} onOpenSession={() => agencySessionId && onOpenSession(agencySessionId)} />
            <RunCard title="Latest run" run={latestRun} onOpenSession={() => agencySessionId && onOpenSession(agencySessionId)} />
            <section className="rounded-[2rem] border border-black/[0.06] bg-white p-6 shadow-sm">
              <p className="text-sm font-semibold text-blue-600">Recent fires</p>
              <div className="mt-4 space-y-2">
                {recentFires.slice(0, 12).map((fire) => (
                  <FireRow key={fire.id} fire={fire} />
                ))}
                {recentFires.length === 0 && (
                  <p className="rounded-2xl bg-[#f7f7f4] px-4 py-3 text-sm text-[#6b6b6b]">
                    Agency fires appear after chats or memory jobs produce copied inputs.
                  </p>
                )}
              </div>
            </section>
          </aside>
        </div>
      )}
    </PageFrame>
  )
}

function StatusPill({ label, value, tone }: { label: string; value: string; tone: string }) {
  return (
    <span className="inline-flex items-center gap-2 rounded-full bg-white px-3 py-1.5 ring-1 ring-black/[0.06]">
      <span className={cn('h-2 w-2 rounded-full', statusTone(tone))} />
      <span className="text-[#8a8a8a]">{label}</span>
      <span className="font-medium text-[#171717]">{value}</span>
    </span>
  )
}

function MetricCard({ icon: Icon, label, value, detail }: { icon: typeof BrainCircuit; label: string; value: string; detail: string }) {
  return (
    <div className="rounded-[1.5rem] border border-black/[0.06] bg-white p-5 shadow-sm">
      <Icon className="h-5 w-5 text-blue-600" />
      <p className="mt-4 text-sm text-[#6b6b6b]">{label}</p>
      <p className="mt-1 text-2xl font-semibold tracking-[-0.03em] text-[#171717]">{value}</p>
      <p className="mt-1 text-xs text-[#8a8a8a]">{detail}</p>
    </div>
  )
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl bg-[#f7f7f4] px-4 py-3">
      <p className="text-xs font-medium uppercase tracking-[0.16em] text-[#9a9a9a]">{label}</p>
      <p className="mt-1 truncate text-sm font-medium text-[#171717]">{value}</p>
    </div>
  )
}

function SessionLink({ session, onOpen }: { session: ClawSessionSummary; onOpen: () => void }) {
  return (
    <button
      type="button"
      className="flex w-full items-center gap-3 rounded-2xl bg-[#f7f7f4] px-4 py-3 text-left transition hover:bg-[#f2f2ef]"
      onClick={onOpen}
    >
      <span className={cn('h-2.5 w-2.5 rounded-full', statusTone(statusToneName(session.status)))} />
      <span className="min-w-0 flex-1">
        <span className="block truncate text-sm font-medium text-[#171717]">{sessionTitle(session)}</span>
        <span className="block truncate text-xs text-[#6b6b6b]">{session.id}</span>
      </span>
      <span className="rounded-full bg-white px-2.5 py-1 text-[11px] font-medium text-[#6b6b6b]">
        {labelForStatus(session.status)}
      </span>
    </button>
  )
}

function RunCard({ title, run, onOpenSession }: { title: string; run: { id: string; status: string; output_summary?: string | null; outputSummary?: string | null; input_preview?: string | null; inputPreview?: string | null } | null; onOpenSession: () => void }) {
  return (
    <section className="rounded-[2rem] border border-black/[0.06] bg-white p-6 shadow-sm">
      <p className="text-sm font-semibold text-blue-600">{title}</p>
      {run ? (
        <button type="button" className="mt-4 w-full rounded-2xl bg-[#f7f7f4] p-4 text-left transition hover:bg-[#f2f2ef]" onClick={onOpenSession}>
          <div className="flex items-center justify-between gap-3">
            <span className="font-mono text-xs text-[#6b6b6b]">{shortId(run.id)}</span>
            <span className="rounded-full bg-white px-2.5 py-1 text-[11px] font-medium text-[#6b6b6b]">{labelForStatus(run.status)}</span>
          </div>
          <p className="mt-3 line-clamp-3 text-sm leading-6 text-[#5f5f5f]">
            {run.output_summary ?? run.outputSummary ?? run.input_preview ?? run.inputPreview ?? 'No summary yet'}
          </p>
        </button>
      ) : (
        <p className="mt-4 rounded-2xl bg-[#f7f7f4] px-4 py-3 text-sm text-[#6b6b6b]">No run yet.</p>
      )}
    </section>
  )
}

function FireRow({ fire }: { fire: ClawAgencyFire }) {
  const sourceSessionId = fire.source_session_id ?? fire.sourceSessionId
  const runId = fire.run_id ?? fire.runId ?? fire.active_run_id ?? fire.activeRunId
  return (
    <div className="rounded-2xl bg-[#f7f7f4] px-4 py-3">
      <div className="flex items-center justify-between gap-3">
        <span className="truncate text-sm font-medium text-[#171717]">{fire.kind}</span>
        <span className="rounded-full bg-white px-2.5 py-1 text-[11px] font-medium text-[#6b6b6b]">{labelForStatus(fire.status)}</span>
      </div>
      <div className="mt-2 space-y-1 font-mono text-xs text-[#8a8a8a]">
        <p>{shortId(fire.id)}</p>
        {sourceSessionId && <p>source {shortId(sourceSessionId)}</p>}
        {runId && <p>run {shortId(runId)}</p>}
      </div>
    </div>
  )
}

function sessionType(session: ClawSessionSummary) {
  return session.session_type ?? session.sessionType ?? 'conversation'
}

function shortId(value: string, size = 10) {
  return value.length <= size ? value : value.slice(0, size)
}
