import { Link, useRouterState } from '@tanstack/react-router'
import {
  Activity,
  Bot,
  Cable,
  ChevronLeft,
  CircleCheck,
  MessageSquare,
  RefreshCcw,
  Route,
  Send,
  Settings2,
} from 'lucide-react'
import { useMemo, useState } from 'react'

import {
  useBridgeConversationsQuery,
  useBridgeEventsQuery,
} from '../../api/hooks'
import { EmptyState } from '../../components/EmptyState'
import { JsonView } from '../../components/JsonView'
import { StatusBadge } from '../../components/StatusBadge'
import { QueryError } from '../../components/ui'
import { navigateApp } from '../../app/navigation'
import { parseApiDate } from '../../lib/date'
import { safeDecodePathSegment } from '../../lib/urlState'
import { cn, formatShortId } from '../../lib/utils'
import type {
  BridgeConversationSummary,
  BridgeEventStatus,
  BridgeEventSummary,
} from '../../types'

const statusOptions: Array<BridgeEventStatus | 'all'> = [
  'all',
  'received',
  'queued',
  'submitted',
  'steered',
  'duplicate',
  'failed',
]

type IntegrationRouteSelection =
  | { kind: 'list'; conversationId?: undefined }
  | { kind: 'setup'; conversationId?: undefined }
  | { kind: 'conversation'; conversationId: string }

function integrationSelectionFromPath(
  pathname: string,
): IntegrationRouteSelection {
  if (pathname === '/integrations/setup') return { kind: 'setup' }
  const prefix = '/integrations/conversations/'
  if (!pathname.startsWith(prefix)) return { kind: 'list' }
  const segment = pathname.slice(prefix.length)
  if (!segment || segment.includes('/')) return { kind: 'list' }
  const conversationId = safeDecodePathSegment(segment)
  return conversationId
    ? { kind: 'conversation', conversationId }
    : { kind: 'list' }
}

export function BridgesPage() {
  const pathname = useRouterState({
    select: (state) => state.location.pathname,
  })
  const routeSelection = integrationSelectionFromPath(pathname)
  const conversations = useBridgeConversationsQuery()
  const selectedConversationId =
    routeSelection.kind === 'conversation'
      ? routeSelection.conversationId
      : null
  const [statusFilter, setStatusFilter] = useState<BridgeEventStatus | 'all'>(
    'all',
  )
  const setupOpen = routeSelection.kind === 'setup'
  const mobileDetailOpen = routeSelection.kind !== 'list'

  function openConversation(conversationId: string) {
    navigateApp(
      `/integrations/conversations/${encodeURIComponent(conversationId)}`,
    )
  }

  function openSetup() {
    navigateApp('/integrations/setup')
  }
  const events = useBridgeEventsQuery({
    conversationId: selectedConversationId,
    status: statusFilter,
  })
  // React Query intentionally keeps the previous result while a new filter key
  // loads. Treat that result as transitional so events and derived metrics from
  // the old conversation/filter are never presented as belonging to the new one.
  const eventsAreTransitioning = events.isLoading || events.isPlaceholderData
  const authoritativeEvents = eventsAreTransitioning
    ? []
    : (events.data?.events ?? [])
  const conversationRows = useMemo(
    () => conversations.data?.conversations ?? [],
    [conversations.data?.conversations],
  )
  const selectedConversation = useMemo(
    () =>
      conversationRows.find((item) => item.id === selectedConversationId) ??
      null,
    [conversationRows, selectedConversationId],
  )
  const conversationNotFound =
    routeSelection.kind === 'conversation' &&
    conversations.data !== undefined &&
    !conversations.isLoading &&
    !conversations.isError &&
    selectedConversation === null

  return (
    <div className="flex h-full min-h-0 flex-col overflow-auto bg-slate-100 lg:flex-row lg:overflow-hidden">
      <h1 className="sr-only">Integrations</h1>
      <aside
        aria-label="Integration list"
        className={cn(
          'max-h-none w-full shrink-0 flex-col border-b border-slate-200 bg-white lg:flex lg:max-h-none lg:w-[28rem] lg:border-b-0 lg:border-r',
          mobileDetailOpen ? 'hidden' : 'flex',
        )}
      >
        <div className="border-b border-slate-200 p-4">
          <div className="flex items-start justify-between gap-3">
            <div>
              <p className="text-sm font-medium text-blue-600">
                Connected channels
              </p>
              <h2 className="mt-1 text-xl font-semibold tracking-tight text-slate-950">
                Integrations
              </h2>
              <p className="mt-2 max-w-xs text-xs leading-5 text-slate-500">
                Monitor inbound adapters, their conversation mappings, and
                delivery health.
              </p>
            </div>
            <button
              type="button"
              className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-700 shadow-sm transition hover:bg-slate-50"
              onClick={() => void conversations.refetch()}
            >
              <RefreshCcw className="h-3.5 w-3.5" />
              Refresh
            </button>
          </div>
        </div>
        <div className="scrollbar-thin min-h-0 flex-1 overflow-auto p-3">
          {conversations.isLoading ? <ConversationSkeleton /> : null}
          {conversations.isError ? (
            <QueryError
              compact
              title="Could not load integrations"
              error={conversations.error}
              onRetry={() => void conversations.refetch()}
            />
          ) : null}
          {!conversations.isLoading &&
          !conversations.isError &&
          conversationRows.length === 0 ? (
            <EmptyState
              icon={Cable}
              title="Connect your first channel"
              description="Configure an inbound adapter, send a test message, then refresh to verify its conversation mapping."
              action={
                <button
                  type="button"
                  className="rounded-xl bg-blue-600 px-3 py-2 text-xs font-semibold text-white hover:bg-blue-700"
                  onClick={openSetup}
                >
                  View setup steps
                </button>
              }
            />
          ) : null}
          <div className="space-y-2">
            {!conversations.isError
              ? conversationRows.map((conversation) => (
                  <ConversationListItem
                    key={conversation.id}
                    conversation={conversation}
                    active={selectedConversationId === conversation.id}
                    onClick={() => openConversation(conversation.id)}
                  />
                ))
              : null}
          </div>
        </div>
      </aside>

      <section
        aria-label="Integration details"
        className={cn(
          'min-h-0 w-full min-w-0 flex-1 overflow-auto p-4 lg:block lg:p-6',
          mobileDetailOpen ? 'block' : 'hidden',
        )}
      >
        <div className="mx-auto max-w-6xl space-y-6">
          <button
            type="button"
            className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 lg:hidden"
            onClick={() => navigateApp('/integrations', true)}
          >
            <ChevronLeft className="h-4 w-4" aria-hidden />
            Back to integrations
          </button>
          <IntegrationSetup
            open={setupOpen}
            onOpenChange={(open) => {
              if (!open && routeSelection.kind === 'setup') {
                navigateApp('/integrations', true)
              }
            }}
            onRefresh={() => void conversations.refetch()}
          />
          {conversationNotFound && routeSelection.kind === 'conversation' ? (
            <ConversationNotFound
              conversationId={routeSelection.conversationId}
              onReturn={() => navigateApp('/integrations', true)}
            />
          ) : (
            <>
              {eventsAreTransitioning ? (
                <BridgeMetricsSkeleton />
              ) : (
                <BridgeMetrics
                  conversations={conversationRows}
                  events={authoritativeEvents}
                />
              )}
              <ConversationDetail conversation={selectedConversation} />
              <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <div className="flex flex-col items-start justify-between gap-3 sm:flex-row sm:items-center">
                  <div>
                    <h2 className="text-sm font-semibold text-slate-950">
                      Bridge event delivery
                    </h2>
                    <p className="mt-1 text-xs text-slate-500">
                      Status shows whether each inbound event created a run,
                      queued a run, steered an active run, duplicated, or
                      failed.
                    </p>
                  </div>
                  <select
                    className="w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-700 outline-none ring-blue-600 focus:ring-2 sm:w-auto"
                    value={statusFilter}
                    aria-label="Filter bridge events by status"
                    onChange={(event) =>
                      setStatusFilter(
                        event.target.value as BridgeEventStatus | 'all',
                      )
                    }
                  >
                    {statusOptions.map((status) => (
                      <option key={status} value={status}>
                        {status.replace(/_/g, ' ')}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="mt-4 space-y-2">
                  {eventsAreTransitioning ? <EventSkeleton /> : null}
                  {!eventsAreTransitioning && events.isError ? (
                    <QueryError
                      compact
                      title="Could not load delivery events"
                      error={events.error}
                      onRetry={() => void events.refetch()}
                    />
                  ) : null}
                  {!eventsAreTransitioning &&
                  !events.isError &&
                  authoritativeEvents.length === 0 ? (
                    <EmptyState
                      title="No bridge events"
                      description="Select a conversation or adjust the status filter."
                    />
                  ) : null}
                  {!eventsAreTransitioning && !events.isError
                    ? authoritativeEvents.map((event) => (
                        <BridgeEventRow key={event.id} event={event} />
                      ))
                    : null}
                </div>
              </section>
            </>
          )}
        </div>
      </section>
    </div>
  )
}

function IntegrationSetup({
  open,
  onOpenChange,
  onRefresh,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  onRefresh: () => void
}) {
  return (
    <section className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
      <button
        type="button"
        className="flex w-full items-center gap-3 p-5 text-left hover:bg-slate-50"
        aria-expanded={open}
        aria-controls="integration-setup-steps"
        onClick={() => onOpenChange(!open)}
      >
        <span className="rounded-xl bg-blue-50 p-2 text-blue-600">
          <Settings2 className="h-5 w-5" aria-hidden />
        </span>
        <span className="min-w-0 flex-1">
          <span className="block text-sm font-semibold text-slate-950">
            Connect and configure a channel
          </span>
          <span className="mt-1 block text-xs leading-5 text-slate-500">
            Inbound adapters authenticate the provider, map external chats to YA
            Claw conversations, and submit normalized events.
          </span>
        </span>
        <span
          className={cn('text-slate-400 transition', open && 'rotate-180')}
          aria-hidden
        >
          ⌄
        </span>
      </button>
      {open ? (
        <div
          id="integration-setup-steps"
          className="border-t border-slate-100 p-5"
        >
          <ol className="grid gap-4 text-sm sm:grid-cols-3">
            <SetupStep
              number="1"
              title="Configure credentials"
              description="Add the provider token, tenant key, webhook secret, and adapter-specific options in the YA Claw service configuration."
            />
            <SetupStep
              number="2"
              title="Register delivery"
              description="Point the provider webhook at the bridge endpoint and keep its event/message identifiers stable for deduplication."
            />
            <SetupStep
              number="3"
              title="Verify an event"
              description="Send a test message, refresh this page, and inspect its normalized payload, run mapping, and diagnostic outcome."
            />
          </ol>
          <div className="mt-5 flex flex-wrap items-center gap-3 rounded-xl bg-blue-50 px-4 py-3 text-xs text-blue-800">
            <CircleCheck className="h-4 w-4" aria-hidden />
            <span className="flex-1">
              Setup is healthy when an adapter appears below and its latest
              event is queued, submitted, or steered.
            </span>
            <button
              type="button"
              className="rounded-lg bg-white px-3 py-1.5 font-semibold text-blue-700 shadow-sm hover:bg-blue-100"
              onClick={onRefresh}
            >
              Check connection
            </button>
          </div>
        </div>
      ) : null}
    </section>
  )
}

function SetupStep({
  number,
  title,
  description,
}: {
  number: string
  title: string
  description: string
}) {
  return (
    <li className="flex gap-3">
      <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-slate-900 text-xs font-semibold text-white">
        {number}
      </span>
      <span>
        <strong className="block font-semibold text-slate-900">{title}</strong>
        <span className="mt-1 block text-xs leading-5 text-slate-500">
          {description}
        </span>
      </span>
    </li>
  )
}

function ConversationListItem({
  conversation,
  active,
  onClick,
}: {
  conversation: BridgeConversationSummary
  active: boolean
  onClick: () => void
}) {
  return (
    <button
      type="button"
      className={cn(
        'w-full rounded-2xl border p-3 text-left transition',
        active
          ? 'border-blue-200 bg-blue-50 shadow-sm'
          : 'border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50',
      )}
      onClick={onClick}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="mono text-xs text-slate-500">
            {conversation.adapter} · {conversation.tenant_key}
          </p>
          <p className="mt-1 truncate text-sm font-semibold text-slate-900">
            {conversation.external_chat_id}
          </p>
        </div>
        {conversation.latest_event_status ? (
          <StatusBadge status={conversation.latest_event_status} />
        ) : null}
      </div>
      <div className="mt-3 grid grid-cols-2 gap-2 text-xs text-slate-500">
        <span>Events {conversation.event_count}</span>
        <span>Profile {conversation.profile_name ?? 'default'}</span>
        <span className="mono">
          Session {formatShortId(conversation.session_id)}
        </span>
        <span className="mono">
          Active {formatShortId(conversation.active_run_id)}
        </span>
      </div>
      <p className="mt-2 text-xs text-slate-400">
        Last event {formatDate(conversation.last_event_at)}
      </p>
    </button>
  )
}

function BridgeMetricsSkeleton() {
  return (
    <section
      className="space-y-3"
      role="status"
      aria-label="Loading bridge metrics"
    >
      <span className="sr-only">Loading bridge metrics</span>
      <div className="h-12 animate-pulse rounded-xl bg-slate-200" />
      <div className="grid grid-cols-2 gap-3 sm:gap-4 xl:grid-cols-5">
        {Array.from({ length: 5 }).map((_, index) => (
          <div
            key={index}
            className="h-36 animate-pulse rounded-2xl bg-slate-200"
          />
        ))}
      </div>
    </section>
  )
}

function BridgeMetrics({
  conversations,
  events,
}: {
  conversations: BridgeConversationSummary[]
  events: BridgeEventSummary[]
}) {
  const steered = events.filter((event) => event.status === 'steered').length
  const failed = events.filter((event) => event.status === 'failed').length
  const delivered = events.filter((event) =>
    ['queued', 'submitted', 'steered'].includes(event.status),
  ).length
  const adapters = new Set(conversations.map((item) => item.adapter)).size
  return (
    <section aria-labelledby="integration-status-heading">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div>
          <h2
            id="integration-status-heading"
            className="text-sm font-semibold text-slate-950"
          >
            Connection status
          </h2>
          <p className="mt-1 text-xs text-slate-500">
            Current adapter coverage and delivery outcomes for the selected
            conversation.
          </p>
        </div>
        <StatusBadge
          status={failed > 0 ? 'failed' : adapters > 0 ? 'enabled' : 'disabled'}
        />
      </div>
      <div className="grid grid-cols-2 gap-3 sm:gap-4 xl:grid-cols-5">
        <MetricCard
          icon={Cable}
          label="Adapters"
          value={String(adapters)}
          accent="blue"
        />
        <MetricCard
          icon={MessageSquare}
          label="Conversations"
          value={String(conversations.length)}
          accent="blue"
        />
        <MetricCard
          icon={Send}
          label="Delivered"
          value={String(delivered)}
          accent="emerald"
        />
        <MetricCard
          icon={Route}
          label="Steered"
          value={String(steered)}
          accent="amber"
        />
        <MetricCard
          icon={Bot}
          label="Failed"
          value={String(failed)}
          accent="rose"
        />
      </div>
    </section>
  )
}

function ConversationNotFound({
  conversationId,
  onReturn,
}: {
  conversationId: string
  onReturn: () => void
}) {
  return (
    <div role="alert">
      <EmptyState
        headingLevel={2}
        title="Integration conversation not found"
        description={`No integration conversation matches “${conversationId}”. It may have been removed or the link may be invalid.`}
        action={
          <button
            type="button"
            className="rounded-xl bg-blue-600 px-3 py-2 text-sm font-semibold text-white hover:bg-blue-700"
            onClick={onReturn}
          >
            Return to integration list
          </button>
        }
      />
    </div>
  )
}

function ConversationDetail({
  conversation,
}: {
  conversation: BridgeConversationSummary | null
}) {
  if (!conversation) {
    return (
      <EmptyState
        title="Select a bridge conversation"
        description="Conversation mapping, session ID, active run, and metadata appear here."
      />
    )
  }
  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex flex-col items-start justify-between gap-4 sm:flex-row">
        <div>
          <p className="text-sm font-medium text-blue-600">
            Conversation record
          </p>
          <h2 className="mt-1 text-2xl font-semibold tracking-tight text-slate-950">
            {conversation.external_chat_id}
          </h2>
        </div>
        <div className="flex flex-wrap gap-2">
          <Link
            className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-700 shadow-sm transition hover:bg-slate-50"
            to="/activity/sessions/$sessionId"
            params={{ sessionId: conversation.session_id }}
          >
            Open session
          </Link>
          {conversation.active_run_id ? (
            <Link
              className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-700 shadow-sm transition hover:bg-slate-50"
              to="/activity/sessions/$sessionId/runs/$runId"
              params={{
                sessionId: conversation.session_id,
                runId: conversation.active_run_id,
              }}
            >
              Open active run
            </Link>
          ) : null}
        </div>
      </div>
      <dl className="mt-5 grid grid-cols-1 gap-4 text-sm sm:grid-cols-2 xl:grid-cols-3">
        <Detail label="Adapter" value={conversation.adapter} />
        <Detail label="Tenant" value={conversation.tenant_key} />
        <Detail
          label="Profile"
          value={conversation.profile_name ?? 'default'}
        />
        <Detail label="Session" value={conversation.session_id} mono />
        <Detail
          label="Active run"
          value={conversation.active_run_id ?? 'none'}
          mono
        />
        <Detail
          label="Last event"
          value={formatDate(conversation.last_event_at)}
        />
      </dl>
      <div className="mt-5">
        <p className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-400">
          Metadata
        </p>
        <JsonView value={conversation.metadata} height="180px" />
      </div>
    </section>
  )
}

function BridgeEventRow({ event }: { event: BridgeEventSummary }) {
  const [expanded, setExpanded] = useState(false)
  return (
    <article className="rounded-xl border border-slate-100 p-3 text-sm">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="mono text-xs text-slate-500">
            {formatShortId(event.id, 12)} · {event.event_type}
          </p>
          <p className="mt-1 truncate font-medium text-slate-900">
            {event.external_message_id ?? event.event_id}
          </p>
        </div>
        <StatusBadge status={mapEventStatus(event.status, event.run_status)} />
      </div>
      <div className="mt-3 grid grid-cols-1 gap-2 text-xs text-slate-500 sm:grid-cols-2 xl:grid-cols-4">
        <span className="mono">
          Chat {formatShortId(event.external_chat_id)}
        </span>
        <span className="mono">Session {formatShortId(event.session_id)}</span>
        <span className="mono">Run {formatShortId(event.run_id)}</span>
        <span>{formatDate(event.created_at)}</span>
      </div>
      <div className="mt-3 flex items-start gap-2 rounded-lg bg-slate-50 px-3 py-2 text-xs text-slate-600">
        <Activity className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden />
        <span>
          <strong className="font-semibold text-slate-700">Diagnostic:</strong>{' '}
          {diagnosticForEvent(event)}
        </span>
      </div>
      {event.error_message ? (
        <p
          className="mt-2 rounded-lg bg-rose-50 px-3 py-2 text-xs text-rose-700"
          role="alert"
        >
          {event.error_message}
        </p>
      ) : null}
      <div className="mt-3 flex flex-wrap items-center gap-2">
        {event.session_id ? (
          <Link
            className="rounded-lg border border-slate-200 px-2 py-1 text-xs font-medium text-slate-600 hover:bg-slate-50"
            to="/activity/sessions/$sessionId"
            params={{ sessionId: event.session_id }}
          >
            Open session
          </Link>
        ) : null}
        {event.run_id ? (
          <Link
            className="rounded-lg border border-slate-200 px-2 py-1 text-xs font-medium text-slate-600 hover:bg-slate-50"
            to={
              event.session_id
                ? '/activity/sessions/$sessionId/runs/$runId'
                : '/activity'
            }
            params={
              event.session_id
                ? { sessionId: event.session_id, runId: event.run_id }
                : undefined
            }
          >
            Open run
          </Link>
        ) : null}
        <button
          type="button"
          className="rounded-lg border border-slate-200 px-2 py-1 text-xs font-medium text-slate-600 hover:bg-slate-50"
          onClick={() => setExpanded((value) => !value)}
        >
          {expanded ? 'Hide payload' : 'Show payload'}
        </button>
      </div>
      {expanded ? (
        <div className="mt-3 space-y-3">
          <dl className="grid grid-cols-1 gap-3 rounded-xl border border-slate-100 bg-slate-50 p-3 text-xs sm:grid-cols-3">
            <Detail label="Event ID" value={event.event_id} mono />
            <Detail label="Delivery status" value={event.status} />
            <Detail
              label="Run status"
              value={event.run_status ?? 'not created'}
            />
          </dl>
          <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
            <div>
              <p className="mb-2 text-xs font-medium text-slate-500">
                Normalized event
              </p>
              <JsonView value={event.normalized_event} height="240px" />
            </div>
            <div>
              <p className="mb-2 text-xs font-medium text-slate-500">
                Raw adapter event
              </p>
              <JsonView value={event.raw_event} height="240px" />
            </div>
          </div>
        </div>
      ) : null}
    </article>
  )
}

const metricAccentClasses: Record<string, string> = {
  blue: 'bg-blue-50 text-blue-600',
  emerald: 'bg-emerald-50 text-emerald-600',
  amber: 'bg-amber-50 text-amber-600',
  rose: 'bg-rose-50 text-rose-600',
}

function MetricCard({
  icon: Icon,
  label,
  value,
  accent,
}: {
  icon: typeof MessageSquare
  label: string
  value: string
  accent: string
}) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div
        className={`inline-flex rounded-xl p-2 ${metricAccentClasses[accent] ?? metricAccentClasses.blue}`}
      >
        <Icon className="h-5 w-5" />
      </div>
      <p className="mt-4 text-sm text-slate-500">{label}</p>
      <p className="mt-1 text-2xl font-semibold text-slate-950">{value}</p>
    </div>
  )
}

function Detail({
  label,
  value,
  mono,
}: {
  label: string
  value: string
  mono?: boolean
}) {
  return (
    <div>
      <dt className="text-xs font-medium uppercase tracking-wide text-slate-400">
        {label}
      </dt>
      <dd className={cn('mt-1 break-all text-slate-800', mono && 'mono')}>
        {value}
      </dd>
    </div>
  )
}

function ConversationSkeleton() {
  return (
    <div className="space-y-2">
      {Array.from({ length: 5 }).map((_, index) => (
        <div
          key={index}
          className="h-32 animate-pulse rounded-2xl bg-slate-100"
        />
      ))}
    </div>
  )
}

function EventSkeleton() {
  return (
    <div className="space-y-2" role="status" aria-label="Loading bridge events">
      <span className="sr-only">Loading bridge events</span>
      {Array.from({ length: 4 }).map((_, index) => (
        <div
          key={index}
          className="h-24 animate-pulse rounded-xl bg-slate-100"
        />
      ))}
    </div>
  )
}

function diagnosticForEvent(event: BridgeEventSummary) {
  if (event.error_message) return `Delivery failed: ${event.error_message}`
  if (event.status === 'duplicate') {
    return 'The adapter event ID was already processed, so no duplicate run was created.'
  }
  if (event.status === 'steered') {
    return 'The message was delivered to the active run as steering input.'
  }
  if (event.status === 'queued') {
    return 'The event is waiting for the mapped conversation to accept a run.'
  }
  if (event.status === 'submitted') {
    return event.run_id
      ? `A run was submitted and mapped to ${formatShortId(event.run_id, 12)}.`
      : 'The event was submitted; its run mapping has not been returned yet.'
  }
  if (event.status === 'received') {
    return 'The webhook was accepted and is awaiting normalization and dispatch.'
  }
  return 'Inspect the raw and normalized payloads for adapter-specific details.'
}

function formatDate(value?: string | null) {
  if (!value) return 'none'
  return parseApiDate(value).toLocaleString()
}

function mapEventStatus(status: BridgeEventStatus, runStatus?: string | null) {
  if (status === 'failed') return 'failed'
  if (runStatus === 'completed') return 'completed'
  if (runStatus === 'failed') return 'failed'
  if (runStatus === 'cancelled') return 'cancelled'
  if (status === 'queued' || status === 'submitted' || status === 'steered') {
    return 'running'
  }
  return status
}
