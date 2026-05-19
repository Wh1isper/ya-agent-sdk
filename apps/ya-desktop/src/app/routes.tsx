import type { DesktopSpace, AppRoute } from './types'
import { BoardPage } from './pages/BoardPage'
import { ChatsPage } from './pages/ChatsPage'
import { AgencyPage } from './pages/AgencyPage'
import { HomePage } from './pages/HomePage'
import { InboxPage } from './pages/InboxPage'
import { SettingsPage } from './pages/SettingsPage'
import { SpacesPage } from './pages/SpacesPage'

export function AppRouteOutlet({
  route,
  selectedSessionId,
  selectedSpace,
  spaces,
  onAddSpace,
  onClearSession,
  onOpenSession,
  onSelectSpace,
}: {
  route: AppRoute
  selectedSessionId: string | null
  selectedSpace: DesktopSpace
  spaces: DesktopSpace[]
  onAddSpace: (space: DesktopSpace) => void
  onClearSession: () => void
  onOpenSession: (sessionId: string) => void
  onSelectSpace: (spaceId: string) => void
}) {
  switch (route) {
    case 'home':
      return (
        <HomePage
          selectedSpace={selectedSpace}
          onOpenSession={onOpenSession}
        />
      )
    case 'chats':
      return (
        <ChatsPage
          selectedSessionId={selectedSessionId}
          selectedSpace={selectedSpace}
          onClearSession={onClearSession}
          onOpenSession={onOpenSession}
        />
      )
    case 'board':
      return <BoardPage onOpenSession={onOpenSession} />
    case 'spaces':
      return (
        <SpacesPage
          selectedSpaceId={selectedSpace.id}
          spaces={spaces}
          onAddSpace={onAddSpace}
          onSelectSpace={onSelectSpace}
        />
      )
    case 'agency':
      return <AgencyPage onOpenSession={onOpenSession} />
    case 'inbox':
      return <InboxPage onOpenSession={onOpenSession} />
    case 'settings':
      return <SettingsPage />
  }
}
