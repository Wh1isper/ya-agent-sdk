import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

import { DesktopShell } from './Shell'

const queryClient = new QueryClient()

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <DesktopShell />
    </QueryClientProvider>
  )
}
