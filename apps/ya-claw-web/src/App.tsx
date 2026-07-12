import { RouterProvider } from '@tanstack/react-router'
import { useEffect } from 'react'

import { Providers } from './app/Providers'
import { registerAppRouter, router, type AppRouter } from './app/router'
import './styles.css'

type AppProps = {
  appRouter?: AppRouter
}

function App({ appRouter = router }: AppProps) {
  useEffect(() => registerAppRouter(appRouter), [appRouter])

  return (
    <Providers>
      <RouterProvider router={appRouter} />
    </Providers>
  )
}

export default App
