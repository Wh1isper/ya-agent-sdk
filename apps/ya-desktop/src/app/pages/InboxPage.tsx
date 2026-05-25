import { Check, ChevronRight, X } from 'lucide-react'
import { toast } from 'sonner'

import {
  useActiveClawConnection,
  useClawSessions,
  useRespondClawInteraction,
} from '../../claw'
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
  const respondInteraction = useRespondClawInteraction(connection)
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
          {inboxItems.map((item) => {
            const canRespond = Boolean(item.runId && item.interactionId)
            return (
              <div
                key={item.id}
                className="rounded-2xl border border-black/[0.08] bg-white p-4 transition hover:bg-[#fbfbfa]"
              >
                <button
                  className="flex w-full items-start gap-4 text-left"
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
                {canRespond && (
                  <div className="mt-3 flex flex-wrap gap-2 pl-6">
                    <button
                      className="inline-flex h-9 items-center gap-2 rounded-xl bg-[#171717] px-3 text-xs font-medium text-white disabled:cursor-not-allowed disabled:bg-[#d8d8d4]"
                      disabled={respondInteraction.isPending}
                      onClick={() => {
                        respondInteraction.mutate(
                          {
                            runId: item.runId!,
                            interactionId: item.interactionId!,
                            sessionId: item.session.id,
                            input: {
                              approved: true,
                              reason: 'Approved from YA Desktop Inbox.',
                            },
                          },
                          {
                            onSuccess: () => toast.success('Approval sent'),
                            onError: (error) =>
                              toast.error(
                                error instanceof Error
                                  ? error.message
                                  : String(error),
                              ),
                          },
                        )
                      }}
                      type="button"
                    >
                      <Check className="h-3.5 w-3.5" />
                      Approve
                    </button>
                    <button
                      className="inline-flex h-9 items-center gap-2 rounded-xl border border-black/[0.08] bg-white px-3 text-xs font-medium text-[#6b6b6b] disabled:cursor-not-allowed disabled:text-[#b5b5b0]"
                      disabled={respondInteraction.isPending}
                      onClick={() => {
                        respondInteraction.mutate(
                          {
                            runId: item.runId!,
                            interactionId: item.interactionId!,
                            sessionId: item.session.id,
                            input: {
                              approved: false,
                              reason: 'Denied from YA Desktop Inbox.',
                            },
                          },
                          {
                            onSuccess: () => toast.success('Denial sent'),
                            onError: (error) =>
                              toast.error(
                                error instanceof Error
                                  ? error.message
                                  : String(error),
                              ),
                          },
                        )
                      }}
                      type="button"
                    >
                      <X className="h-3.5 w-3.5" />
                      Deny
                    </button>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
    </PageFrame>
  )
}
