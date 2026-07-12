import { MutationCache, QueryCache, QueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'

import { ApiError } from '../api/client'

function shouldRetry(failureCount: number, error: unknown) {
  if (error instanceof ApiError && [401, 403, 404].includes(error.status)) {
    return false
  }
  return failureCount < 1
}

function reportError(error: unknown) {
  if (error instanceof ApiError && error.status === 401) return
  toast.error(error instanceof Error ? error.message : 'YA Claw request failed')
}

export function createAppQueryClient() {
  return new QueryClient({
    queryCache: new QueryCache({ onError: reportError }),
    mutationCache: new MutationCache({ onError: reportError }),
    defaultOptions: {
      queries: {
        refetchOnWindowFocus: false,
        retry: shouldRetry,
      },
      mutations: {
        retry: false,
      },
    },
  })
}
