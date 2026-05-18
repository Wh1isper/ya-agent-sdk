import { useActiveClawConnection, useClawSessions } from '../../claw'
import { EmptyState, PageEmpty, PageFrame, SessionRow } from '../ui'
import { groupSessionsForBoard } from '../utils'

export function BoardPage({
  onOpenSession,
}: {
  onOpenSession: (sessionId: string) => void
}) {
  const activeConnectionQuery = useActiveClawConnection()
  const connection = activeConnectionQuery.data?.connection ?? null
  const sessionsQuery = useClawSessions(connection)
  const groupedSessions = groupSessionsForBoard(sessionsQuery.data ?? [])

  if (!connection)
    return (
      <PageEmpty
        title="Local Claw is offline"
        detail="Start Local Claw to load the live board."
      />
    )
  if (sessionsQuery.isLoading)
    return <PageEmpty title="Loading board" detail="Reading live sessions." />
  if (sessionsQuery.error)
    return (
      <PageEmpty
        title="Could not load board"
        detail={sessionsQuery.error.message}
      />
    )

  return (
    <PageFrame
      eyebrow="Board"
      title="Work status"
      body="A calm overview for active work, waiting decisions, completed chats, and recovery items."
    >
      <div className="grid gap-4 xl:grid-cols-4">
        {groupedSessions.map((column) => (
          <section
            key={column.title}
            className="rounded-2xl border border-black/[0.08] bg-[#fbfbfa] p-3"
          >
            <div className="flex items-center justify-between px-1 pb-2">
              <h3 className="text-sm font-semibold text-[#171717]">
                {column.title}
              </h3>
              <span className="text-xs text-[#8a8a8a]">
                {column.items.length}
              </span>
            </div>
            <div className="space-y-2">
              {column.items.length === 0 ? (
                <EmptyState title="Clear" detail="No chats in this lane." />
              ) : (
                column.items.map((session) => (
                  <SessionRow
                    key={session.id}
                    session={session}
                    compact
                    onClick={() => onOpenSession(session.id)}
                  />
                ))
              )}
            </div>
          </section>
        ))}
      </div>
    </PageFrame>
  )
}
