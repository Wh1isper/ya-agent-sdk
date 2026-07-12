import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export function getDefaultBaseUrl() {
  if (import.meta.env.VITE_CLAW_BASE_URL) {
    return normalizeBaseUrl(import.meta.env.VITE_CLAW_BASE_URL)
  }
  if (typeof window !== 'undefined') {
    return window.location.origin
  }
  return 'http://127.0.0.1:9042'
}

export function normalizeBaseUrl(baseUrl: string) {
  return baseUrl.trim().replace(/\/+$/, '')
}

function createConnectionScope() {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID()
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`
}

export type ConnectionState = {
  baseUrl: string
  apiToken: string
  connectionScope: string
  connectionIssue: string | null
  connectionDraftDirty: boolean
  setConnectionDraftDirty: (dirty: boolean) => void
  setConnection: (connection: { baseUrl: string; apiToken: string }) => void
  invalidateConnection: (reason: string, connectionScope?: string) => void
  logout: () => void
}

export const useConnectionStore = create<ConnectionState>()(
  persist(
    (set) => ({
      baseUrl: getDefaultBaseUrl(),
      apiToken: '',
      connectionScope: createConnectionScope(),
      connectionIssue: null,
      connectionDraftDirty: false,
      setConnectionDraftDirty: (connectionDraftDirty) =>
        set({ connectionDraftDirty }),
      setConnection: ({ baseUrl, apiToken }) =>
        set({
          baseUrl: normalizeBaseUrl(baseUrl),
          apiToken: apiToken.trim(),
          connectionScope: createConnectionScope(),
          connectionIssue: null,
          connectionDraftDirty: false,
        }),
      invalidateConnection: (reason, connectionScope) =>
        set((state) => {
          if (connectionScope && connectionScope !== state.connectionScope) {
            return state
          }
          return {
            apiToken: '',
            connectionScope: createConnectionScope(),
            connectionIssue: reason,
            connectionDraftDirty: false,
          }
        }),
      logout: () =>
        set({
          apiToken: '',
          connectionScope: createConnectionScope(),
          connectionIssue: null,
          connectionDraftDirty: false,
        }),
    }),
    {
      name: 'ya-claw-connection',
      version: 1,
      partialize: (state) => ({ baseUrl: state.baseUrl }),
      migrate: (persistedState) => {
        const persisted = persistedState as { baseUrl?: unknown } | null
        return typeof persisted?.baseUrl === 'string'
          ? { baseUrl: normalizeBaseUrl(persisted.baseUrl) }
          : {}
      },
      merge: (persistedState, currentState) => {
        const persisted = persistedState as { baseUrl?: unknown } | null
        return {
          ...currentState,
          ...(typeof persisted?.baseUrl === 'string'
            ? { baseUrl: normalizeBaseUrl(persisted.baseUrl) }
            : {}),
        }
      },
    },
  ),
)
