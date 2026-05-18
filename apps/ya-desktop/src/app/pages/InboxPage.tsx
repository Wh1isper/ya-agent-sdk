import { ChevronRight } from 'lucide-react'

import { useActiveClawConnection, useClawSessions } from '../../claw'
import { cn } from '../../lib'
import { EmptyState, PageEmpty, PageFrame } from '../ui'
import { inboxItemsFromSessions, statusTone } from '../utils'

export function InboxPage({
  onOpenSession,
}: {
  onOpenSession: (sessionId: string) => void
}) {
  const activeConnectionQuery = useActiveClawConnection()
  const connection = activeConnectionQuery.data?.connection ?? null
  const sessionsQuery = useClawSessions(connection)
  const inboxItems = inboxItemsFromSessions(sessionsQuery.data ?? [])

  if (!connection)
    return (
      <PageEmpty
        title="Local Claw is offline"
        detail="Start Local Claw to load Inbox items."
      />
    )

  return (
    <PageFrame
      eyebrow="Inbox"
      title="Decisions that need you"
      body="Approvals, failed background work, and recovery paths appear as focused cards."
    >
      {sessionsQuery.isLoading ? (
        <EmptyState title="Loading Inbox" detail="Reading live sessions." />
      ) : sessionsQuery.error ? (
        <EmptyState
          title="Could not load Inbox"
          detail={sessionsQuery.error.message}
        />
      ) : inboxItems.length === 0 ? (
        <EmptyState
          title="Inbox clear"
          detail="No failed runs or pending approvals."
        />
      ) : (
        <div className="space-y-2">
          {inboxItems.map((item) => (
            <button
              key={`${item.session.id}-${item.title}`}
              className="flex w-full items-start gap-4 rounded-2xl border border-black/[0.08] bg-white p-4 text-left transition hover:bg-[#fbfbfa]"
              onClick={() => onOpenSession(item.session.id)}
              type="button"
            >
              <span
                className={cn(
                  'mt-1 h-2.5 w-2.5 rounded-full',
                  statusTone(item.tone),
                )}
              />
              <span className="min-w-0 flex-1">
                <span className="block text-sm font-semibold text-[#171717]">
                  {item.title}
                </span>
                <span className="mt-1 block text-xs leading-5 text-[#6b6b6b]">
                  {item.detail}
                </span>
              </span>
              <ChevronRight className="mt-1 h-4 w-4 text-[#b5b5b0]" />
            </button>
          ))}
        </div>
      )}
    </PageFrame>
  )
}
