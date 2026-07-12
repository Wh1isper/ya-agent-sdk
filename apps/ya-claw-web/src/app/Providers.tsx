import { QueryClientProvider } from '@tanstack/react-query'
import { useEffect, useMemo, useRef, type ReactNode } from 'react'
import { Toaster } from 'sonner'

import { useConnectionStore } from '../stores/connectionStore'
import { useLayoutStore } from '../stores/layoutStore'
import { createAppQueryClient } from './queryClient'

export function Providers({ children }: { children: ReactNode }) {
  const apiToken = useConnectionStore((state) => state.apiToken)
  const connectionScope = useConnectionStore((state) => state.connectionScope)
  const previousConnection = useRef({
    authenticated: Boolean(apiToken.trim()),
    scope: connectionScope,
  })
  const queryClient = useMemo(createAppQueryClient, [connectionScope])

  useEffect(() => {
    const previous = previousConnection.current
    const authenticated = Boolean(apiToken.trim())
    if (previous.scope !== connectionScope && previous.authenticated) {
      useLayoutStore.getState().resetConnectionSelection()
    }
    previousConnection.current = { authenticated, scope: connectionScope }
  }, [apiToken, connectionScope])

  useEffect(
    () => () => {
      void queryClient.cancelQueries()
      queryClient.clear()
    },
    [queryClient],
  )

  return (
    <QueryClientProvider client={queryClient}>
      {children}
      <Toaster richColors position="bottom-right" />
    </QueryClientProvider>
  )
}
